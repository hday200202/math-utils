#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>

#include "DeviceABI.h"

/*
    The HIP backend. The only translation unit hipcc compiles.

    Two things about this card in particular:

    - gfx1100 is RDNA3, and RDNA runs wave32, not the wave64 that almost
      every AMD kernel you will read online assumes. A cross-wave reduction
      written for 64 lanes does not crash here, it just silently sums the
      wrong half. warpSize is used rather than a literal so the mistake
      cannot be made, and the launch is sized against it.

    - bfloat16 needs no conversion instruction. It is the top 16 bits of a
      float32, so widening is a shift into the high half - the same
      __uint_as_float(bits << 16) the CPU kernel does. Nothing is lost that
      the checkpoint ever had.
*/

namespace ai {
namespace gpu {

namespace {

constexpr int BLOCK     = 256;
constexpr int MAX_WAVES = BLOCK / 32;     // wave32; a wave64 device uses fewer

/* ========================================================================= //
    Context

    Device selection and the scratch buffers that hold activations for the
    duration of a call. Allocated once and grown, never per matmul: hipMalloc
    serialises against the device and a token issues several hundred of these.
// ========================================================================= */

struct Context {
    bool        ok = false;
    std::string name;
    std::string error;

    /*
        One lock around every trip to the card.

        The scratch buffers below and the stream the kernels run on are
        shared by the whole process, and the mixture-of-experts block calls
        apply() from eight threads at once - one per chosen expert. Without
        this, thread A stages its activations, thread B overwrites them
        before A's kernel has read them, and the model answers in garbage
        that happens to end its turn. The dense path never saw it because
        it only ever calls from one thread.

        Serialising costs nothing real: the card is one device, every
        dispatch already synchronises on its own result, and the threads
        were there for the CPU experts, which still run in parallel.
    */
    std::mutex  gate;

    size_t         allocated = 0;
    AiGpuCounters  stats{};

    void*  scratchIn      = nullptr;
    size_t scratchInBytes = 0;
    void*  scratchOut      = nullptr;
    size_t scratchOutBytes = 0;

    static Context& instance() {
        static Context ctx;
        return ctx;
    }

    Context() {
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
            error = "no HIP device found";
            return;
        }
        if (hipSetDevice(0) != hipSuccess) {
            error = "hipSetDevice(0) failed";
            return;
        }

        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
            error = "hipGetDeviceProperties failed";
            return;
        }

        name = props.name;
        ok   = true;
    }

    ~Context() {
        if (scratchIn)  (void)hipFree(scratchIn);
        if (scratchOut) (void)hipFree(scratchOut);
    }

    float* growIn(size_t bytes)  { return grow(scratchIn,  scratchInBytes,  bytes); }
    float* growOut(size_t bytes) { return grow(scratchOut, scratchOutBytes, bytes); }

private:
    float* grow(void*& slot, size_t& held, size_t bytes) {
        if (held >= bytes) return static_cast<float*>(slot);

        if (slot) { (void)hipFree(slot); allocated -= held; slot = nullptr; held = 0; }

        void* fresh = nullptr;
        if (hipMalloc(&fresh, bytes) != hipSuccess) {
            error = "hipMalloc failed for " + std::to_string(bytes) + " scratch bytes";
            return nullptr;
        }

        slot = fresh;
        held = bytes;
        allocated += bytes;
        return static_cast<float*>(slot);
    }
};


/* ========================================================================= //
    Reduction
// ========================================================================= */

__device__ __forceinline__ float widenBF16(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

/*
    blockSum(v):
    - reduces one value per thread down to a single value in thread 0.
      Within a wave the shuffle needs no barrier at all - lanes advance in
      lockstep - so only the cross-wave step touches LDS. With BLOCK 256 on
      wave32 that is eight partials, summed serially by one thread, which is
      cheaper than a second shuffle round for eight values
*/
__device__ __forceinline__ float blockSum(float v) {
    __shared__ float partial[MAX_WAVES];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);

    const int wave = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) % warpSize;

    if (lane == 0) partial[wave] = v;
    __syncthreads();

    if (threadIdx.x != 0) return 0.0f;

    const int waves = static_cast<int>(blockDim.x) / warpSize;
    float sum = 0.0f;
    for (int i = 0; i < waves; ++i) sum += partial[i];
    return sum;
}


/* ========================================================================= //
    GEMV

    grid.x is the output column, grid.y the row of A. During generation
    grid.y is 1 and grid.x is the whole weight matrix's row count, which is
    what keeps every compute unit fed on a matrix-vector product.

    The loop is a strided walk so that the 256 threads of a block read 256
    consecutive weights at a time - one coalesced burst per step, which is
    the only access pattern that reaches the card's bandwidth.
// ========================================================================= */

/*
    ...Vec variants read four weights per instruction (a ushort4, 8 bytes)
    against one float4. Fewer, wider loads matter more here than they do on
    the CPU: a memory-bound kernel is limited by outstanding requests, not
    by the arithmetic between them. Requires K % 4 == 0, which also makes
    every row start correctly aligned for both vector types - checked on the
    host, with a scalar kernel for anything else.
*/
__global__ __launch_bounds__(BLOCK)
void gemvBF16Vec(const float* __restrict__ A, const uint16_t* __restrict__ B,
                 float* __restrict__ C, int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;

    const uint16_t* w = B + static_cast<size_t>(j) * K;
    const float*    a = A + static_cast<size_t>(i) * K;

    const int quads = K >> 2;
    float acc = 0.0f;

    for (int q = threadIdx.x; q < quads; q += BLOCK) {
        const ushort4 p = reinterpret_cast<const ushort4*>(w)[q];
        const float4  x = reinterpret_cast<const float4*>(a)[q];
        acc += x.x * widenBF16(p.x) + x.y * widenBF16(p.y)
             + x.z * widenBF16(p.z) + x.w * widenBF16(p.w);
    }

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}

__global__ __launch_bounds__(BLOCK)
void gemvBF16Scalar(const float* __restrict__ A, const uint16_t* __restrict__ B,
                    float* __restrict__ C, int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;

    const uint16_t* w = B + static_cast<size_t>(j) * K;
    const float*    a = A + static_cast<size_t>(i) * K;

    float acc = 0.0f;
    for (int t = threadIdx.x; t < K; t += BLOCK)
        acc += a[t] * widenBF16(w[t]);

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}

/*
    Q8: int8 codes with one float scale per block of 32 along K.

    Per block rather than per row, which costs an eighth of a byte per
    weight and halves the error - a row's worth of dynamic range is set by
    its largest outlier, and at 255 levels that still leaves the quiet
    weights on far fewer steps than a block scale would give them.

    The scale therefore cannot come out of the whole reduction the way it
    could for a row scale. It does not need to: four consecutive codes sit
    in one block, so a thread handling quad q wants scales[q / 8] - and
    since consecutive threads take consecutive quads, every lane of a wave
    reads the SAME scale. It is a broadcast, not a gather.
*/
__global__ __launch_bounds__(BLOCK)
void gemvQ8Vec(const float* __restrict__ A, const int8_t* __restrict__ B,
               const float* __restrict__ scales, float* __restrict__ C,
               int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int blocks = K / 32;

    const int8_t* w = B + static_cast<size_t>(j) * K;
    const float*  s = scales + static_cast<size_t>(j) * blocks;
    const float*  a = A + static_cast<size_t>(i) * K;

    const int quads = K >> 2;
    float acc = 0.0f;

    for (int q = threadIdx.x; q < quads; q += BLOCK) {
        const char4  p = reinterpret_cast<const char4*>(w)[q];
        const float4 x = reinterpret_cast<const float4*>(a)[q];
        acc += (x.x * float(p.x) + x.y * float(p.y)
              + x.z * float(p.z) + x.w * float(p.w)) * s[q >> 3];
    }

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}

__global__ __launch_bounds__(BLOCK)
void gemvQ8Scalar(const float* __restrict__ A, const int8_t* __restrict__ B,
                  const float* __restrict__ scales, float* __restrict__ C,
                  int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int blocks = K / 32;

    const int8_t* w = B + static_cast<size_t>(j) * K;
    const float*  s = scales + static_cast<size_t>(j) * blocks;
    const float*  a = A + static_cast<size_t>(i) * K;

    float acc = 0.0f;
    for (int t = threadIdx.x; t < K; t += BLOCK)
        acc += a[t] * (float(w[t]) * s[t >> 5]);

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}

/*
    Q4: 32 codes per block, one float scale each, two codes per byte.

    One 16-byte load brings in a whole block - uint4 is exactly 32 nibbles.
    Byte i carries weight i and weight i+16, so the two nibbles of a byte
    land sixteen apart rather than side by side; see quantiseQ4 for why the
    layout is shaped for the CPU's benefit rather than this kernel's.

    The scale multiplies the block's partial sum rather than each weight,
    so the arithmetic cost of blocking is one extra multiply per 32 codes.
    Against 16 bytes of memory traffic that is free.
*/
__global__ __launch_bounds__(BLOCK)
void gemvQ4(const float* __restrict__ A, const uint8_t* __restrict__ B,
            const float* __restrict__ scales, float* __restrict__ C,
            int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int blocks = K / 32;

    const uint8_t* w = B + static_cast<size_t>(j) * (K / 2);
    const float*   s = scales + static_cast<size_t>(j) * blocks;
    const float*   a = A + static_cast<size_t>(i) * K;

    float acc = 0.0f;

    for (int b = threadIdx.x; b < blocks; b += BLOCK) {
        const uint4  packed = reinterpret_cast<const uint4*>(w)[b];
        const float* x      = a + b * 32;

        const unsigned words[4] = { packed.x, packed.y, packed.z, packed.w };

        float partial = 0.0f;
        for (int word = 0; word < 4; ++word) {
            unsigned bits = words[word];
            for (int k = 0; k < 4; ++k) {
                const unsigned byte = bits & 0xFFu;
                const int      i    = word * 4 + k;
                partial += x[i]      * float(int(byte & 0xFu) - 8);
                partial += x[i + 16] * float(int(byte >> 4) - 8);
                bits >>= 8;
            }
        }

        acc += partial * s[b];
    }

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}

__global__ __launch_bounds__(BLOCK)
void gemvF32Vec(const float* __restrict__ A, const float* __restrict__ B,
                float* __restrict__ C, int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;

    const float* w = B + static_cast<size_t>(j) * K;
    const float* a = A + static_cast<size_t>(i) * K;

    const int quads = K >> 2;
    float acc = 0.0f;

    for (int q = threadIdx.x; q < quads; q += BLOCK) {
        const float4 wv = reinterpret_cast<const float4*>(w)[q];
        const float4 xv = reinterpret_cast<const float4*>(a)[q];
        acc += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}

__global__ __launch_bounds__(BLOCK)
void gemvF32Scalar(const float* __restrict__ A, const float* __restrict__ B,
                   float* __restrict__ C, int K, int N) {
    const int j = blockIdx.x;
    const int i = blockIdx.y;

    const float* w = B + static_cast<size_t>(j) * K;
    const float* a = A + static_cast<size_t>(i) * K;

    float acc = 0.0f;
    for (int t = threadIdx.x; t < K; t += BLOCK)
        acc += a[t] * w[t];

    const float total = blockSum(acc);
    if (threadIdx.x == 0) C[static_cast<size_t>(i) * N + j] = total;
}


using Clock = std::chrono::steady_clock;

inline double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

/*
    dispatch(...):
    - the shared half of both entry points: stage A in, launch, bring C back.
      Timed either side of a device synchronise so the kernel and the bus are
      measured apart rather than as one lump
*/
template <typename Launch>
void dispatch(const float* A, float* C, size_t M, size_t K, size_t N, Launch launch) {
    Context& ctx = Context::instance();
    if (!ctx.ok) return;

    std::lock_guard<std::mutex> lock(ctx.gate);

    const size_t inBytes  = M * K * sizeof(float);
    const size_t outBytes = M * N * sizeof(float);

    float* dA = ctx.growIn(inBytes);
    float* dC = ctx.growOut(outBytes);
    if (!dA || !dC) return;

    Clock::time_point mark = Clock::now();
    if (hipMemcpy(dA, A, inBytes, hipMemcpyHostToDevice) != hipSuccess) {
        ctx.error = "hipMemcpy host to device failed";
        return;
    }
    ctx.stats.transferSeconds += since(mark);
    ctx.stats.hostToDeviceBytes += inBytes;

    mark = Clock::now();
    launch(dA, dC);
    if (hipDeviceSynchronize() != hipSuccess) {
        ctx.error = "kernel launch failed: ";
        ctx.error += hipGetErrorString(hipGetLastError());
        return;
    }
    ctx.stats.kernelSeconds += since(mark);
    ++ctx.stats.launches;

    mark = Clock::now();
    if (hipMemcpy(C, dC, outBytes, hipMemcpyDeviceToHost) != hipSuccess) {
        ctx.error = "hipMemcpy device to host failed";
        return;
    }
    ctx.stats.transferSeconds += since(mark);
    ctx.stats.deviceToHostBytes += outBytes;
}

} // !namespace


/* ========================================================================= //
    Interface

    The C entry points declared in DeviceABI.h, and the only symbols this
    translation unit exports. Everything above stays C++ and stays private
    to this file; nothing with a mangled name crosses out.

    That is what lets hipcc compile this on Windows - where it targets the
    MSVC ABI - and MinGW g++ link against the result. The bodies are the
    same ones the C++ interface had; only the signatures changed, from
    Buffer& to void* and from std::string to const char*.
// ========================================================================= */

extern "C" {

AI_GPU_ABI int aigpu_available(void) { return Context::instance().ok ? 1 : 0; }

/*
    Borrowed pointers into the context's own strings. The caller copies
    immediately - see Device.hpp - so these only have to outlive the call,
    and a std::string never leaves this translation unit.
*/
AI_GPU_ABI const char* aigpu_device_name(void) {
    return Context::instance().name.c_str();
}

AI_GPU_ABI const char* aigpu_last_error(void) {
    return Context::instance().error.c_str();
}

AI_GPU_ABI uint64_t aigpu_allocated_bytes(void) {
    return static_cast<uint64_t>(Context::instance().allocated);
}

AI_GPU_ABI uint64_t aigpu_total_memory(void) {
    size_t free = 0, total = 0;
    if (!Context::instance().ok || hipMemGetInfo(&free, &total) != hipSuccess) return 0;
    return static_cast<uint64_t>(total);
}

AI_GPU_ABI uint64_t aigpu_free_memory(void) {
    size_t free = 0, total = 0;
    if (!Context::instance().ok || hipMemGetInfo(&free, &total) != hipSuccess) return 0;
    return static_cast<uint64_t>(free);
}

AI_GPU_ABI void* aigpu_upload(const void* host, uint64_t bytes) {
    Context& ctx = Context::instance();
    if (!ctx.ok || bytes == 0) return nullptr;

    std::lock_guard<std::mutex> lock(ctx.gate);

    void* device = nullptr;
    if (hipMalloc(&device, static_cast<size_t>(bytes)) != hipSuccess) {
        ctx.error = "out of VRAM: hipMalloc(" + std::to_string(bytes) + ") failed";
        return nullptr;
    }

    if (hipMemcpy(device, host, static_cast<size_t>(bytes),
                  hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(device);
        ctx.error = "hipMemcpy host to device failed during upload";
        return nullptr;
    }

    ctx.allocated += static_cast<size_t>(bytes);
    return device;
}

/*
    Freed by the side that allocated it. The byte count comes back in so the
    running total needs no lookup table - and because the caller's Buffer
    already knows it.
*/
AI_GPU_ABI void aigpu_release(void* device, uint64_t bytes) {
    if (!device) return;

    Context& ctx = Context::instance();
    std::lock_guard<std::mutex> lock(ctx.gate);
    (void)hipFree(device);
    ctx.allocated -= static_cast<size_t>(bytes);
}

AI_GPU_ABI int aigpu_download(void* host, const void* device, uint64_t bytes) {
    if (!Context::instance().ok || !device) return 0;
    return hipMemcpy(host, device, static_cast<size_t>(bytes),
                     hipMemcpyDeviceToHost) == hipSuccess ? 1 : 0;
}

AI_GPU_ABI void aigpu_matmul_nt_bf16(const float* a, const void* b, float* c,
                                     uint64_t m, uint64_t k, uint64_t n) {
    if (!b) return;

    const size_t M = static_cast<size_t>(m);
    const size_t K = static_cast<size_t>(k);
    const size_t N = static_cast<size_t>(n);

    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M));
    const int  kk = static_cast<int>(K);
    const int  nn = static_cast<int>(N);
    const uint16_t* weights = static_cast<const uint16_t*>(b);

    dispatch(a, c, M, K, N, [&](float* dA, float* dC) {
        if (K % 4 == 0)
            hipLaunchKernelGGL(gemvBF16Vec, grid, dim3(BLOCK),
                               0, 0, dA, weights, dC, kk, nn);
        else
            hipLaunchKernelGGL(gemvBF16Scalar, grid, dim3(BLOCK),
                               0, 0, dA, weights, dC, kk, nn);
    });
}

AI_GPU_ABI void aigpu_matmul_nt_q8(const float* a, const void* b,
                                   const void* scales, float* c,
                                   uint64_t m, uint64_t k, uint64_t n) {
    if (!b || !scales) return;

    const size_t M = static_cast<size_t>(m);
    const size_t K = static_cast<size_t>(k);
    const size_t N = static_cast<size_t>(n);

    if (K % 32 != 0) {
        Context::instance().error = "Q8 block scales need K divisible by 32";
        return;
    }

    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M));
    const int  kk = static_cast<int>(K);
    const int  nn = static_cast<int>(N);
    const int8_t* codes = static_cast<const int8_t*>(b);
    const float*  rows  = static_cast<const float*>(scales);

    dispatch(a, c, M, K, N, [&](float* dA, float* dC) {
        if (K % 4 == 0)
            hipLaunchKernelGGL(gemvQ8Vec, grid, dim3(BLOCK),
                               0, 0, dA, codes, rows, dC, kk, nn);
        else
            hipLaunchKernelGGL(gemvQ8Scalar, grid, dim3(BLOCK),
                               0, 0, dA, codes, rows, dC, kk, nn);
    });
}

AI_GPU_ABI void aigpu_matmul_nt_q4(const float* a, const void* b,
                                   const void* scales, float* c,
                                   uint64_t m, uint64_t k, uint64_t n) {
    if (!b || !scales) return;

    const size_t M = static_cast<size_t>(m);
    const size_t K = static_cast<size_t>(k);
    const size_t N = static_cast<size_t>(n);

    /* the packer refuses a partial block, so the loader keeps any matrix
       whose width is not a multiple of 32 at Q8. Nothing should reach here */
    if (K % 32 != 0) {
        Context::instance().error = "Q4 needs K divisible by 32";
        return;
    }

    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M));
    const int  kk = static_cast<int>(K);
    const int  nn = static_cast<int>(N);
    const uint8_t* codes = static_cast<const uint8_t*>(b);
    const float*   rows  = static_cast<const float*>(scales);

    dispatch(a, c, M, K, N, [&](float* dA, float* dC) {
        hipLaunchKernelGGL(gemvQ4, grid, dim3(BLOCK),
                           0, 0, dA, codes, rows, dC, kk, nn);
    });
}

AI_GPU_ABI void aigpu_matmul_nt(const float* a, const void* b, float* c,
                                uint64_t m, uint64_t k, uint64_t n) {
    if (!b) return;

    const size_t M = static_cast<size_t>(m);
    const size_t K = static_cast<size_t>(k);
    const size_t N = static_cast<size_t>(n);

    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M));
    const int  kk = static_cast<int>(K);
    const int  nn = static_cast<int>(N);
    const float* weights = static_cast<const float*>(b);

    dispatch(a, c, M, K, N, [&](float* dA, float* dC) {
        if (K % 4 == 0)
            hipLaunchKernelGGL(gemvF32Vec, grid, dim3(BLOCK),
                               0, 0, dA, weights, dC, kk, nn);
        else
            hipLaunchKernelGGL(gemvF32Scalar, grid, dim3(BLOCK),
                               0, 0, dA, weights, dC, kk, nn);
    });
}

AI_GPU_ABI void aigpu_synchronize(void) {
    if (Context::instance().ok) (void)hipDeviceSynchronize();
}

AI_GPU_ABI void aigpu_counters(AiGpuCounters* out) {
    if (out) *out = Context::instance().stats;
}

AI_GPU_ABI void aigpu_reset_counters(void) {
    Context::instance().stats = AiGpuCounters{};
}

} // !extern "C"

} // !namespace gpu
} // !namespace ai
