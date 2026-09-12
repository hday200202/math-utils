#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

/*
    The device boundary.

    Everything below this line is plain C++ - no HIP types, no HIP headers.
    That matters for one practical reason: the rest of the library is header
    only and compiled by g++, and HIP kernels cannot be. Keeping the
    interface free of device types means exactly one translation unit
    (src/gpu/Kernels.hip) goes through hipcc, and the g++ build links its
    object without ever seeing a __global__.

    src/gpu/Stub.cpp implements the same interface with available() == false,
    so a build without ROCm compiles and runs unchanged on the CPU path.

    WHAT THIS STAGE DOES

    Weights are uploaded once at load and the matmuls run on the GPU. The
    activations still live in host memory, so every matmul is
    upload-launch-download. That is deliberately visible rather than hidden:
    counters() reports transfer time separately from kernel time, because
    the ratio between those two numbers is the entire argument for moving
    the residual stream onto the device as well.

    A 40-layer model does 7 matmuls per layer plus the head, so a single
    token crosses the bus about 560 times. The kernel time will look
    excellent and the wall clock will not, and the counters say why.
*/

namespace ai {
namespace gpu {

/* ========================================================================= //
    Buffer

    An allocation in VRAM. The destructor is defined in the backend TU
    because freeing it is a HIP call - which is also why this cannot be a
    unique_ptr with a custom deleter without dragging hipFree into scope.
// ========================================================================= */

class Buffer {
public:
    Buffer() = default;
    Buffer(void* pointer, size_t bytes) : m_ptr(pointer), m_bytes(bytes) {}
    ~Buffer();

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
bool available();

const std::string& deviceName();
const std::string& lastError();

size_t totalMemory();
size_t freeMemory();

/* bytes currently held in live Buffers */
size_t allocatedBytes();


/*
    upload(host, bytes):
    - allocates in VRAM and copies. Returns null on failure, with the reason
      in lastError().
    - bf16 weights alias the mmap'd checkpoint, so for those this is a
      straight copy of file bytes: no widening, no staging buffer, no
      conversion pass. The dtype the file was written in is the dtype that
      lands on the card
*/
BufferRef upload(const void* host, size_t bytes);
bool      download(void* host, const Buffer& source, size_t bytes);


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

void matmulNT    (const float* A, const Buffer& B, float* C,
                  size_t M, size_t K, size_t N);

void matmulNTBF16(const float* A, const Buffer& B, float* C,
                  size_t M, size_t K, size_t N);

/*
    Q8: B holds int8 codes, scales one float per block of 32 along K -
    the same granularity ai::quant uses on the CPU, reading the same
    bytes. Requires K divisible by 32.
*/
void matmulNTQ8(const float* A, const Buffer& B, const Buffer& scales, float* C,
                size_t M, size_t K, size_t N);

/*
    Q4: two codes per byte pairing weight j with weight j+16, one float
    scale per block of 32 along K. Requires K divisible by 32 - see
    ai::quant::quantizeRow for the layout and why it is shaped that way.
*/
void matmulNTQ4(const float* A, const Buffer& B, const Buffer& scales, float* C,
                size_t M, size_t K, size_t N);

void synchronize();


/* ========================================================================= //
    Counters

    The point of separating these is that transferSeconds is the tax for
    leaving the activations in host memory. When it dwarfs kernelSeconds,
    the next stage is not a faster kernel.
// ========================================================================= */

struct Counters {
    size_t launches         = 0;
    size_t hostToDeviceBytes = 0;
    size_t deviceToHostBytes = 0;
    double kernelSeconds    = 0.0;
    double transferSeconds  = 0.0;
};

const Counters& counters();
void resetCounters();

} // !namespace gpu
} // !namespace ai
