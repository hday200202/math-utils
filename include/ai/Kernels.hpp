#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
    /* the activations go to int8 once, here, and every weight row reuses
       them - see quant::dotQ8Q8 */
    quant::Activations act;
    act.quantize(A, M, K);

    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j)
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = quant::dotQuantRow(act.row(i), W, j);
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
    - the same against FP8 weights with a scale per tile.
    - generating a token (M = 1) decodes each code straight into the dot
      product: nothing is written, and the one pass over the codes is the
      only memory traffic
    - a prompt (M > 1) decodes each weight row into scratch once and dots it
      against every row of A, so the decode is paid once per weight row
      rather than once per token. The scratch is per call of the body, so
      each thread has its own
*/
inline void matmulNTFP8(const float* A, const quant::Fp8Tensor& W, float* C,
                        size_t M, size_t K, size_t N, bool threaded = true) {
    const auto body = [&](size_t begin, size_t end) {
        if (M == 1) {
            for (size_t j = begin; j < end; ++j) C[j] = W.dotRow(A, j);
            return;
        }
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


/* ========================================================================= //
    Tiled matmul, for prompts

    The kernels above compute each output as one dot product: a row of A
    against a row of B, two loads for every multiply-add. That is the right
    shape for generating a token, where A is one row and the whole job is
    streaming B out of memory once. It is the wrong shape for a prompt of a
    few hundred tokens, where the work is arithmetic and two loads per FMA
    caps it at half of what the core can do - and where A, being too big for
    cache, is fetched again for every row of B.

    So a prompt goes through a register tile instead: MR rows of A against
    NR rows of B at once, MR + NR vector loads feeding MR * NR FMAs. Around
    that, B is taken a block of NB rows by KB columns at a time, decoded to
    float once into a scratch that stays in L2, and every row of A is run
    past it before the next block - so a weight in any format is widened
    once per block rather than once per token.

    The vectors are GCC/Clang vector types rather than intrinsics: the
    compiler picks the instructions for whatever -march says, and the same
    source is AVX-512 on the server and AVX2 on a laptop.
// ========================================================================= */

#if defined(__AVX512F__)
constexpr size_t VL = 16;
constexpr size_t MR = 4, NR = 4;            // 16 accumulators of 32 zmm registers
#elif defined(__AVX__)
constexpr size_t VL = 8;
constexpr size_t MR = 2, NR = 4;            // 8 accumulators of 16 ymm registers
#else
/* SSE only: a wider vector than the hardware has would be split anyway,
   and passing one by value changes the ABI */
constexpr size_t VL = 4;
constexpr size_t MR = 2, NR = 4;
#endif
typedef float VFloat __attribute__((vector_size(VL * sizeof(float))));

constexpr size_t NB = 64;                   // weight rows per block
constexpr size_t KB = 512;                  // columns per block: a multiple of every
                                            // quantisation block, 32 and 128

/* below this many rows of A the plain kernels win: there is too little
   reuse to pay for the decoding into scratch */
constexpr size_t TILE_MIN_ROWS = 8;

inline VFloat loadV(const float* p) {
    VFloat v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/*
    tile<R, Q>(a, b, length, c, ldc):
    - c[r * ldc + q] += a[r] . b[q] over `length` floats, for R rows of A
      and Q rows of scratch. Full tiles are MR x NR; the ragged edge of a
      block uses smaller instantiations of the same code
*/
template <size_t R, size_t Q>
inline void tile(const float* const* a, const float* const* b, size_t length, float* c, size_t ldc) {
    VFloat acc[R][Q] = {};
    size_t k = 0;
    for (; k + VL <= length; k += VL) {
        VFloat bv[Q];
        for (size_t q = 0; q < Q; ++q) bv[q] = loadV(b[q] + k);
        for (size_t r = 0; r < R; ++r) {
            const VFloat av = loadV(a[r] + k);
            for (size_t q = 0; q < Q; ++q) acc[r][q] += av * bv[q];
        }
    }

    for (size_t r = 0; r < R; ++r)
        for (size_t q = 0; q < Q; ++q) {
            float s = 0.0f;
            for (size_t l = 0; l < VL; ++l) s += acc[r][q][l];
            for (size_t t = k; t < length; ++t) s += a[r][t] * b[q][t];
            c[r * ldc + q] += s;
        }
}

/*
    matmulTiled(A, C, M, K, N, decode, threaded):
    - C[M,N] = A[M,K] @ B^T for a B that is only ever seen through
      decode(row, from, length, out), which writes columns [from, from +
      length) of B's row as floats. Every weight format supplies one.
    - split across threads by weight rows, the same static slices as every
      other kernel, so each thread works on the rows the loader placed near
      it. A matrix too narrow to split that way - a router, a gate - is
      split by rows of A instead, since a long prompt through one thread is
      the thing being fixed here
*/
template <typename Decode>
inline void matmulTiled(const float* A, float* C, size_t M, size_t K, size_t N,
                        const Decode& decode, bool threaded = true) {
    const auto run = [&](size_t j0, size_t j1, size_t i0, size_t i1) {
        std::vector<float> scratch(NB * KB);

        for (size_t jb = j0; jb < j1; jb += NB) {
            const size_t nb = std::min(NB, j1 - jb);
            for (size_t i = i0; i < i1; ++i) std::fill(C + i * N + jb, C + i * N + jb + nb, 0.0f);

            for (size_t kb = 0; kb < K; kb += KB) {
                const size_t kl = std::min(KB, K - kb);
                for (size_t jj = 0; jj < nb; ++jj) decode(jb + jj, kb, kl, &scratch[jj * kl]);

                const float* a[MR];
                const float* b[NR];
                size_t i = i0;
                for (; i + MR <= i1; i += MR) {
                    for (size_t r = 0; r < MR; ++r) a[r] = A + (i + r) * K + kb;
                    size_t jj = 0;
                    for (; jj + NR <= nb; jj += NR) {
                        for (size_t q = 0; q < NR; ++q) b[q] = &scratch[(jj + q) * kl];
                        tile<MR, NR>(a, b, kl, C + i * N + jb + jj, N);
                    }
                    for (; jj < nb; ++jj) {
                        b[0] = &scratch[jj * kl];
                        tile<MR, 1>(a, b, kl, C + i * N + jb + jj, N);
                    }
                }
                for (; i < i1; ++i) {
                    a[0] = A + i * K + kb;
                    size_t jj = 0;
                    for (; jj + NR <= nb; jj += NR) {
                        for (size_t q = 0; q < NR; ++q) b[q] = &scratch[(jj + q) * kl];
                        tile<1, NR>(a, b, kl, C + i * N + jb + jj, N);
                    }
                    for (; jj < nb; ++jj) {
                        b[0] = &scratch[jj * kl];
                        tile<1, 1>(a, b, kl, C + i * N + jb + jj, N);
                    }
                }
            }
        }
    };

    if (!threaded) { run(0, N, 0, M); return; }
    if (N >= 512)  { Pool::instance().run(N, [&](size_t b, size_t e) { run(b, e, 0, M); }); return; }
    Pool::instance().run(M, [&](size_t b, size_t e) { run(0, N, b, e); });
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
