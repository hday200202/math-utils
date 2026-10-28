#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ThreadPool.hpp"
#include "Quant.hpp"

/*
    The hot loops, kept apart from Tensor because they trade readability for
    speed and because they are where the threading lives.

    The thing to understand about generating one token at a time: it is bound
    by MEMORY BANDWIDTH, not arithmetic. Every weight in the model is read
    exactly once per token and each one is used for a single multiply-add, so
    the processor spends its time waiting on RAM. That is why the kernels
    below are shaped the way they are - every access walks forward through
    contiguous memory, and the work is split so each thread owns a distinct
    slab of the weights rather than sharing cache lines.

    No intrinsics. The inner loops are written so that -O3 -march=native
    vectorises them (AVX-512 with FMA on this machine); four running
    accumulators keep the FMA pipeline fed instead of stalling on a single
    serial dependency.
*/

namespace ai {
namespace kernels {

/* ========================================================================= //
    Dot product
// ========================================================================= */

/*
    dot(a, b, n):
    - LANES independent accumulators, held as an array so the compiler keeps
      them in vector registers.
    - the count matters more than it looks. Floating point addition is not
      associative, so the compiler may only reduce as widely as the code
      explicitly permits: four named accumulators license a 4-wide reduction
      and no more, which on AVX-512 leaves seven eighths of the width idle.
      Measured on this machine, 4 accumulators give 8.1 GFLOP/s and 32 give
      65.3 - and 16 gives 6.9, because one zmm register cannot hide the
      four-cycle FMA latency. 32 is two registers deep, which can.
    - summing in a different order than strictly left to right changes the
      rounding slightly. That is unavoidable in any parallel reduction, and
      the layer-by-layer diff against the reference is what confirms it
      stays within tolerance
*/
constexpr size_t LANES = 32;

inline float dot(const float* a, const float* b, size_t n) {
    float acc[LANES] = {};

    size_t i = 0;
    for (; i + LANES <= n; i += LANES)
        for (size_t j = 0; j < LANES; ++j)
            acc[j] += a[i + j] * b[i + j];

    float sum = 0.0f;
    for (size_t j = 0; j < LANES; ++j) sum += acc[j];
    for (; i < n; ++i) sum += a[i] * b[i];

    return sum;
}


/*
    dotBF16(a, b, n):
    - the same reduction with the weights left in bfloat16 and widened as they
      are read. bfloat16 IS the top 16 bits of a float32, so widening is a
      shift into the high half rather than a conversion - no exponent
      rebiasing, no subnormal handling, no table.
    - the point is not the arithmetic, it is the memory. Generating a token
      reads every weight in the model exactly once, so halving the bytes very
      nearly halves the time. The accuracy given up is the mantissa bits that
      were never there: the checkpoint was bfloat16 on disk, and widening it
      to float32 at load time added zeroes, not information
*/
inline float dotBF16(const float* a, const uint16_t* b, size_t n) {
    float acc[LANES] = {};

    size_t i = 0;
    for (; i + LANES <= n; i += LANES)
        for (size_t j = 0; j < LANES; ++j)
            acc[j] += a[i + j] * std::bit_cast<float>(static_cast<uint32_t>(b[i + j]) << 16);

    float sum = 0.0f;
    for (size_t j = 0; j < LANES; ++j) sum += acc[j];
    for (; i < n; ++i)
        sum += a[i] * std::bit_cast<float>(static_cast<uint32_t>(b[i]) << 16);

    return sum;
}


/* ========================================================================= //
    Matmul against a transposed right operand

    C[M,N] = A[M,K] @ B[N,K]^T, with B stored row-major as [N,K].

    Every weight matrix in a HuggingFace Linear layer is stored [out, in],
    which is exactly this orientation - so no transpose is ever materialised
    and both operands are walked along their rows.

    The split is over N, the output columns, because N is the large dimension
    during generation: M is 1 for every token after the prompt, so splitting
    over rows would leave every thread but one asleep.
// ========================================================================= */

inline void matmulNT(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N, bool threaded = true) {
    /*
        j outermost, i innermost. The other order re-reads the whole weight
        matrix once per row of A, so a 32-token prompt streams every weight
        32 times and prefill runs no faster than decoding those tokens one
        at a time. This way each weight row is loaded once and reused across
        every row of A, which is what makes processing a prompt in one pass
        worth doing at all - A is only M*K and stays in cache.
    */
    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const float* brow = B + j * K;
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dot(A + i * K, brow, K);
        }
    };

    /* below a few thousand output columns the dispatch costs more than the
       work, and these are called several hundred times per token */
    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}


/*
    matmulNTQuant(...):
    - the same shape as matmulNTBF16 with the weights held as quantised
      blocks. This is the kernel that matters once a model is too big to hold
      in bfloat16: it reads between a half and a quarter of the bytes, and at
      batch one that is very nearly the whole cost
*/
inline void matmulNTQuant(const float* A, const quant::QuantTensor& W, float* C,
                          size_t M, size_t K, size_t N, bool threaded = true) {
    const size_t blocks = K / quant::BLOCK;
    const size_t rowQuants = (W.type == quant::Type::Q4) ? K / 2 : K;
    const bool q4 = (W.type == quant::Type::Q4);

    const int8_t* base = W.quants.get();
    const float* allScales = W.scales.get();

    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const int8_t* qrow = base + j * rowQuants;
            const float* srow = allScales + j * blocks;

            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = q4 ? quant::dotQ4(A + i * K, qrow, srow, K)
                                  : quant::dotQ8(A + i * K, qrow, srow, K);
        }
    };

    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}


/*
    matmulNTBias(...):
    - the same, plus a bias row broadcast across the output. Folding it in
      here saves a second pass over C, which at these sizes is another trip
      through memory for no arithmetic
*/
/* matmulNT with the right operand left in bfloat16 */
inline void matmulNTBF16(const float* A, const uint16_t* B, float* C,
                         size_t M, size_t K, size_t N, bool threaded = true) {
    // j outermost so each weight row is streamed once for all M inputs
    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const uint16_t* brow = B + j * K;
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dotBF16(A + i * K, brow, K);
        }
    };

    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}


/*
    matmulNTFP8(A, W, C, ...):
    - the same against FP8 weights with a scale per tile. Each weight row is
      decoded into a scratch row once - every code through the table, times
      its tile's scale - and then dotted against every row of A, so a prompt
      pays for the decode once per weight row rather than once per token.
    - the scratch is per call of the body, so each thread has its own and
      nothing is shared between them
*/
inline void matmulNTFP8(const float* A, const quant::Fp8Tensor& W, float* C,
                        size_t M, size_t K, size_t N, bool threaded = true) {
    const auto body = [&](size_t begin, size_t end) {
        std::vector<float> row(K);
        for (size_t j = begin; j < end; ++j) {
            W.dequantRow(j, row.data());
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dot(A + i * K, row.data(), K);
        }
    };

    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}


inline void matmulNTBias(const float* A, const float* B, const float* bias, float* C,
                         size_t M, size_t K, size_t N, bool threaded = true) {
    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const float* brow = B + j * K;
            const float b = bias[j];
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dot(A + i * K, brow, K) + b;
        }
    };

    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}

} // !namespace kernels
} // !namespace ai
