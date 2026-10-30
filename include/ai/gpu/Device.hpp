#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "DeviceABI.h"

/*
    The device boundary.

    Everything below this line is plain C++ - no HIP types, no HIP headers.
    That matters for one practical reason: the rest of the library is header
    only and compiled by g++, and HIP kernels cannot be. Keeping the
    interface free of device types means exactly one translation unit
    (src/gpu/Kernels.hip) goes through hipcc, and the g++ build links its
    object without ever seeing a __global__.

    It is also entirely INLINE. Not one symbol here is defined by the
    backend; every function below is a wrapper the caller's own compiler
    emits, over the C entry points in DeviceABI.h. That is what lets the
    backend be built by a different compiler than everything else - which on
    Windows it must be, since hipcc there targets the MSVC ABI and the rest
    of this project is MinGW g++. See DeviceABI.h for why that is fatal to a
    C++ boundary and harmless to a C one.

    Linux gets the identical arrangement. It does not need it - one GCC
    builds both sides - but two boundaries to keep in step is a worse idea
    than one that happens to be stricter than Linux requires.

    src/gpu/Stub.cpp implements the same C entry points with available() ==
    false, so a CPU-only build needs neither Metal nor ROCm. Metal.mm
    implements the same interface on Apple GPUs without exposing Objective-C
    types to the model.

    The general matmul interface uploads activations and returns results
    synchronously. Its counters distinguish execution from copying. On Metal,
    compatible dense models also use the optional resident decoder in
    DeviceABI.h: one command buffer per token, with intermediates and KV
    history kept on the GPU. The synchronous interface remains the fallback
    for prompt processing, traces and unsupported architectures.

    Buffers are opaque backend handles. They need not be raw device addresses;
    Metal wraps an MTLBuffer and its allocation size. Callers must never do
    pointer arithmetic on Buffer::ptr().

*/

namespace ai {
namespace gpu {

/* ========================================================================= //
    Buffer

    A backend-owned GPU allocation (shared memory on Apple silicon). The destructor is inline and calls the backend's
    C free, so whoever allocated the memory is who frees it - the rule that
    makes it safe for the two sides to have been built by different
    compilers with different allocators.
// ========================================================================= */

class Buffer {
public:
    Buffer() = default;
    Buffer(void* pointer, size_t bytes) : m_ptr(pointer), m_bytes(bytes) {}

    ~Buffer() {
        if (m_ptr) aigpu_release(m_ptr, static_cast<uint64_t>(m_bytes));
    }

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    void*  ptr()   const { return m_ptr; }
    size_t bytes() const { return m_bytes; }
    bool   valid() const { return m_ptr != nullptr; }

private:
    void*  m_ptr   = nullptr;
    size_t m_bytes = 0;
};

using BufferRef = std::shared_ptr<Buffer>;


/* ========================================================================= //
    Device
// ========================================================================= */

/* true when a usable device was found. Safe to call before anything else,
   and safe to call when the stub backend is linked */
inline bool available() { return aigpu_available() != 0; }

/*
    Returned BY VALUE, copied out of the backend's buffer at the call.
    A reference would mean a std::string living on one side of the boundary
    and being read on the other, which is the exact thing that cannot work
    when the two sides were built by different compilers.
*/
inline std::string deviceName() {
    const char* text = aigpu_device_name();
    return text ? std::string(text) : std::string();
}

inline std::string lastError() {
    const char* text = aigpu_last_error();
    return text ? std::string(text) : std::string();
}

inline size_t totalMemory()    { return static_cast<size_t>(aigpu_total_memory()); }
inline size_t freeMemory()     { return static_cast<size_t>(aigpu_free_memory()); }

/* bytes currently held in live Buffers */
inline size_t allocatedBytes() { return static_cast<size_t>(aigpu_allocated_bytes()); }


/*
    upload(host, bytes):
    - allocates in VRAM and copies. Returns null on failure, with the reason
      in lastError().
    - bf16 weights alias the mmap'd checkpoint, so for those this is a
      straight copy of file bytes: no widening, no staging buffer, no
      conversion pass. The dtype the file was written in is the dtype that
      lands on the card
*/
inline BufferRef upload(const void* host, size_t bytes) {
    void* device = aigpu_upload(host, static_cast<uint64_t>(bytes));
    if (!device) return nullptr;
    return std::make_shared<Buffer>(device, bytes);
}

inline bool download(void* host, const Buffer& source, size_t bytes) {
    if (!source.valid()) return false;
    return aigpu_download(host, source.ptr(), static_cast<uint64_t>(bytes)) != 0;
}


/* ========================================================================= //
    Kernels

    C[M,N] = A[M,K] @ B[N,K]^T, with B resident in VRAM as [N,K] row-major.

    The same orientation the CPU kernels use, for the same reason: every
    HuggingFace Linear weight is stored [out, in] already, so no transpose
    is ever materialised.

    M is 1 for every token after the prompt, which makes this a GEMV: one
    workgroup per output column, reading one weight row and reducing across
    the block. Prefill (M > 1) is handled by giving the grid a second
    dimension - correct, and enough to measure, but it re-reads the weight
    row per row of A. A tiled GEMM belongs here eventually.
// ========================================================================= */

inline void matmulNT(const float* A, const Buffer& B, float* C,
                     size_t M, size_t K, size_t N) {
    if (!B.valid()) return;
    aigpu_matmul_nt(A, B.ptr(), C, uint64_t(M), uint64_t(K), uint64_t(N));
}

inline void matmulNTBF16(const float* A, const Buffer& B, float* C,
                         size_t M, size_t K, size_t N) {
    if (!B.valid()) return;
    aigpu_matmul_nt_bf16(A, B.ptr(), C, uint64_t(M), uint64_t(K), uint64_t(N));
}

/*
    Q8: B holds int8 codes, scales one float per block of 32 along K -
    the same granularity ai::quant uses on the CPU, reading the same
    bytes. Requires K divisible by 32.
*/
inline void matmulNTQ8(const float* A, const Buffer& B, const Buffer& scales, float* C,
                       size_t M, size_t K, size_t N) {
    if (!B.valid() || !scales.valid()) return;
    aigpu_matmul_nt_q8(A, B.ptr(), scales.ptr(), C,
                       uint64_t(M), uint64_t(K), uint64_t(N));
}

/*
    Q4: two codes per byte pairing weight j with weight j+16, one float
    scale per block of 32 along K. Requires K divisible by 32 - see
    ai::quant::quantizeRow for the layout and why it is shaped that way.
*/
inline void matmulNTQ4(const float* A, const Buffer& B, const Buffer& scales, float* C,
                       size_t M, size_t K, size_t N) {
    if (!B.valid() || !scales.valid()) return;
    aigpu_matmul_nt_q4(A, B.ptr(), scales.ptr(), C,
                       uint64_t(M), uint64_t(K), uint64_t(N));
}

inline void synchronize() { aigpu_synchronize(); }


/* ========================================================================= //
    Counters

    The point of separating these is that transferSeconds is the tax for
    leaving the activations in host memory. When it dwarfs kernelSeconds,
    the next stage is not a faster kernel.
// ========================================================================= */

using Counters = AiGpuCounters;

/*
    A snapshot, refreshed on every call, rather than a window onto the
    backend's live struct. Holding a reference across work and expecting it
    to move would mean reading the backend's memory directly, which is the
    one thing the boundary exists to prevent. Call it again instead.
*/
inline const Counters& counters() {
    static thread_local Counters snapshot;
    aigpu_counters(&snapshot);
    return snapshot;
}

inline void resetCounters() { aigpu_reset_counters(); }

} // !namespace gpu
} // !namespace ai
