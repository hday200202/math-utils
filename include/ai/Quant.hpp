#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifndef AI_QUANT_AVX512
#define AI_QUANT_AVX512 1
#endif

#if defined(__AVX2__)
    #include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#include "ThreadPool.hpp"

/*
    Block-wise symmetric quantisation of the weight matrices.

    Generating a token reads every weight exactly once and uses each for a
    single multiply-add, so the whole thing is bound by how many BYTES have to
    come out of RAM. Storing the weights in fewer bits is therefore close to a
    direct speedup, and it is the only lever left once the kernels are already
    running at the memory bus limit.

    The scheme is the simple one, and it is simple for a reason: a row is cut
    into blocks of 32 consecutive weights, each block gets its own scale, and
    every weight in it is stored as a small signed integer times that scale.
    One scale per 32 weights is enough because neighbouring weights in a row
    are similar in magnitude; one scale for a whole row is not, and a few
    large outliers would crush everything else to zero.

        bf16   2 bytes/weight    the baseline
        Q8     1.125             int8 plus a float scale per 32
        Q4     0.625             int4 packed two per byte, same scale

    Nothing is calibrated and nothing is learned - the scale is just the
    block's largest magnitude divided by the integer range. That is enough to
    keep a model coherent; it is not enough to be indistinguishable from the
    original, and Q4 in particular is a real trade.
*/

namespace ai {
namespace quant {

/* the contraction dimension is cut into blocks of this many weights */
constexpr size_t BLOCK = 32;

enum class Type { None, Q8, Q4 };

inline const char* typeName(Type t) {
    switch (t) {
        case Type::Q8: return "q8";
        case Type::Q4: return "q4";
        default:       return "bf16";
    }
}

inline double bytesPerWeight(Type t) {
    switch (t) {
        case Type::Q8: return 1.0 + 4.0 / double(BLOCK);        // int8 + f32 scale
        case Type::Q4: return 0.5 + 4.0 / double(BLOCK);        // 2 per byte + scale
        default:       return 2.0;                              // bf16
    }
}


/*
    QuantTensor:
    - a 2-D [out, in] matrix with in divisible by BLOCK. quants holds the
      integers row-major, scales one float per block.
    - the two live in separate arrays rather than interleaved so each is
      contiguous and the inner loop reads both forwards
*/
struct QuantTensor {
    Type   type = Type::None;
    size_t outDim = 0, inDim = 0;

    std::shared_ptr<int8_t[]>  quants;      // Q8: one per weight. Q4: two per byte
    std::shared_ptr<float[]>   scales;      // outDim * (inDim / BLOCK)

    bool valid() const { return type != Type::None && quants != nullptr; }

    size_t blocksPerRow() const { return inDim / BLOCK; }

    size_t byteCount() const {
        const size_t weights = outDim * inDim;
        const size_t q = (type == Type::Q4) ? weights / 2 : weights;
        return q + outDim * blocksPerRow() * sizeof(float);
    }
};


/* ========================================================================= //
    Quantising
// ========================================================================= */

inline float widenBF16(uint16_t h) {
    return std::bit_cast<float>(static_cast<uint32_t>(h) << 16);
}

/* the quantiser reads either a bfloat16 checkpoint or floats that were
   already widened, which is what an FP8 one has to become first */
inline float widenSource(uint16_t h) { return widenBF16(h); }
inline float widenSource(float f)    { return f; }


/* ========================================================================= //
    FP8

    Some checkpoints ship their big matrices as 8-bit floats rather than
    bfloat16: E4M3, one sign bit, four of exponent and three of mantissa. The
    range is tiny - the largest value is 448 - so the format is never used
    bare. Every 128x128 tile of the matrix carries a float scale of its own,
    stored alongside as a separate tensor named <weight>_scale_inv, and the
    real weight is the code times the scale of the tile it sits in.

    The name says inverse but the value is not: it is what the code is
    MULTIPLIED by. It is the inverse of the scale that was divided out when
    the checkpoint was written, which is the writer's point of view.
// ========================================================================= */

/*
    fp8E4M3ToFloat(code):
    - exponent bias 7. An exponent of zero is subnormal, the mantissa then
      counting in steps of 2^-9 with no implied leading one. E4M3 spends the
      all-ones pattern on NaN alone and has no infinities, which is how it
      reaches 448 rather than 240
*/
constexpr float fp8E4M3ToFloat(uint8_t code) {
    const bool     negative = (code & 0x80u) != 0;
    const uint32_t exponent = (code >> 3) & 0x0Fu;
    const uint32_t mantissa = code & 0x07u;

    float magnitude = 0.0f;
    if (exponent == 0x0Fu && mantissa == 0x07u) {
        magnitude = std::numeric_limits<float>::quiet_NaN();
    } else if (exponent == 0) {
        magnitude = static_cast<float>(mantissa) / 512.0f;              // m * 2^-9
    } else {
        float power = 1.0f;                                             // 2^(e - 7)
        for (uint32_t e = exponent; e > 7; --e) power *= 2.0f;
        for (uint32_t e = exponent; e < 7; ++e) power *= 0.5f;
        magnitude = power * (1.0f + static_cast<float>(mantissa) / 8.0f);
    }
    return negative ? -magnitude : magnitude;
}

/*
    fp8Decode(code):
    - the same value by bit arithmetic, for the hot loops. A table lookup
      per weight is a gather, which the vectoriser either refuses or turns
      into gather instructions far slower than this:
        - a normal code's exponent and mantissa are 7 contiguous bits, and
          moved up to bit 20 they become a float32's exponent and top three
          mantissa bits; adding 120 to the exponent turns bias 7 into bias 127
        - a subnormal (exponent 0) is just mantissa * 2^-9
      Branch-free, so it vectorises. NaN (0x7F, 0xFF) never occurs in a
      weight and is not handled; the tests check the other 254 codes against
      the table above
*/
inline float fp8Decode(uint8_t code) {
    const uint32_t bits = code & 0x7Fu;
    const float normal = std::bit_cast<float>((bits + (120u << 3)) << 20);
    const float tiny   = static_cast<float>(bits) * (1.0f / 512.0f);
    const float value  = bits < 8 ? tiny : normal;
    return std::bit_cast<float>(std::bit_cast<uint32_t>(value) | (uint32_t(code & 0x80u) << 24));
}

/* all 256 codes decoded once. Fine for anything that is not a hot loop, and
   the reference the faster decode above is tested against */
inline const float* fp8Table() {
    static const auto table = [] {
        std::array<float, 256> t{};
        for (size_t i = 0; i < 256; ++i) t[i] = fp8E4M3ToFloat(static_cast<uint8_t>(i));
        return t;
    }();
    return table.data();
}

/*
    Fp8Tensor:
    - an [out, in] matrix of E4M3 codes, aliasing the checkpoint's bytes, and
      the grid of tile scales that goes with it.
    - the tile shape is derived from the two shapes rather than assumed:
      a [4096, 12288] weight with a [32, 96] scale grid is 128x128 tiles
*/
struct Fp8Tensor {
    std::shared_ptr<uint8_t[]> codes;
    std::shared_ptr<float[]>   scales;
    size_t outDim = 0, inDim = 0;
    size_t scaleRows = 1, scaleCols = 1;
    size_t tileOut = 0, tileIn = 0;

    bool valid() const { return codes != nullptr && scales != nullptr && outDim && inDim; }

    /* one row of real weights: every code times its tile's scale */
    void dequantRow(size_t row, float* out) const { dequantRange(row, 0, inDim, out); }

    /* the same for columns [from, from + length) of the row only */
    void dequantRange(size_t row, size_t from, size_t length, float* out) const {
        const uint8_t* src  = codes.get() + row * inDim;
        const float*   srow = scales.get() + (row / tileOut) * scaleCols;
        const size_t   to   = from + length;

        for (size_t i = from; i < to; ) {
            const size_t c   = i / tileIn;
            const size_t end = std::min((c + 1) * tileIn, to);
            const float  s   = srow[c];
            for (; i < end; ++i) out[i - from] = fp8Decode(src[i]) * s;
        }
    }

    /*
        dotRow(x, row):
        - x . (that row of real weights), without writing the row anywhere.
          Each tile is summed in its own codes' units and multiplied by its
          scale once at the end, so the scale costs one multiply per tile
          rather than one per weight
    */
    float dotRow(const float* x, size_t row) const {
        constexpr size_t LANES = 16;
        const uint8_t* src  = codes.get() + row * inDim;
        const float*   srow = scales.get() + (row / tileOut) * scaleCols;

        float total = 0.0f;
        for (size_t c = 0; c < scaleCols; ++c) {
            const size_t begin = c * tileIn;
            const size_t end   = std::min(begin + tileIn, inDim);

            float acc[LANES] = {};
            size_t i = begin;
            for (; i + LANES <= end; i += LANES)
                for (size_t j = 0; j < LANES; ++j)
                    acc[j] += x[i + j] * fp8Decode(src[i + j]);

            float tile = 0.0f;
            for (size_t j = 0; j < LANES; ++j) tile += acc[j];
            for (; i < end; ++i) tile += x[i] * fp8Decode(src[i]);

            total += tile * srow[c];
        }
        return total;
    }
};

/*
    makeFp8(codes, outDim, inDim, scales, scaleRows, scaleCols):
    - puts the pieces together, working out the tile from the two shapes.
      A checkpoint with no scale tensor at all is plain unscaled FP8, which
      is one tile covering the whole matrix with a scale of one
*/
inline Fp8Tensor makeFp8(std::shared_ptr<uint8_t[]> codes, size_t outDim, size_t inDim,
                         std::shared_ptr<float[]> scales, size_t scaleRows, size_t scaleCols) {
    Fp8Tensor t;
    t.codes  = std::move(codes);
    t.outDim = outDim;
    t.inDim  = inDim;

    if (!scales || scaleRows == 0 || scaleCols == 0) {
        t.scales = std::shared_ptr<float[]>(new float[1]{1.0f});
        scaleRows = scaleCols = 1;
    } else {
        t.scales = std::move(scales);
    }

    t.scaleRows = scaleRows;
    t.scaleCols = scaleCols;
    t.tileOut   = (outDim + scaleRows - 1) / scaleRows;
    t.tileIn    = (inDim  + scaleCols - 1) / scaleCols;
    return t;
}

/*
    quantizeRow(src, out, scales, inDim, type):
    - symmetric, so zero maps to zero exactly and no zero-point is needed.
      The scale is the block's largest magnitude over the integer range, which
      means the largest weight is represented exactly and everything else
      rounds to the nearest step.
    - Q4 packs two weights per byte - weight j and weight j+16 - offset by 8
      so the stored range 0..15 covers -8..7. See the Q4 branch for why the
      scale and the pairing are both chosen the way they are
*/
#if defined(__AVX2__)
// lround uses ties away from zero. Adding 0.5 before truncation is not
// equivalent near half-way values, so compare the exact fractional part.
inline __m256i roundAwayAVX2(__m256 v) {
    const __m256i whole = _mm256_cvttps_epi32(v);
    const __m256 fraction = _mm256_andnot_ps(_mm256_set1_ps(-0.0f),
        _mm256_sub_ps(v, _mm256_cvtepi32_ps(whole)));
    const __m256i step = _mm256_and_si256(_mm256_castps_si256(
        _mm256_cmp_ps(fraction, _mm256_set1_ps(0.5f), _CMP_GE_OQ)), _mm256_set1_epi32(1));
    const __m256i sign = _mm256_srai_epi32(_mm256_castps_si256(v), 31);
    return _mm256_add_epi32(whole, _mm256_sub_epi32(_mm256_xor_si256(step, sign), sign));
}

inline __m256i quantizeBlockAVX2(const float* in, float inv, int lower, int upper) {
    __m256i q[4];
    for (size_t i = 0; i < 4; ++i) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(in + i * 8), _mm256_set1_ps(inv));
        q[i] = _mm256_min_epi32(_mm256_max_epi32(roundAwayAVX2(v),
            _mm256_set1_epi32(lower)), _mm256_set1_epi32(upper));
    }
    const __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]),
                                             _mm256_packs_epi32(q[2], q[3]));
    return _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
#endif

template <typename Src>
inline void quantizeRow(const Src* src, int8_t* out, float* scales,
                        size_t inDim, Type type) {
    const size_t blocks = inDim / BLOCK;

    for (size_t b = 0; b < blocks; ++b) {
        const Src* in = src + b * BLOCK;

        float amax = 0.0f, extreme = 0.0f;
        for (size_t j = 0; j < BLOCK; ++j) {
            const float v = widenSource(in[j]);
            if (std::fabs(v) > amax) { amax = std::fabs(v); extreme = v; }
        }

        if (type == Type::Q8) {
            const float scale = amax / 127.0f;
            const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            scales[b] = scale;

            int8_t* dst = out + b * BLOCK;
#if defined(__AVX2__)
            alignas(32) float values[BLOCK];
            for (size_t j = 0; j < BLOCK; ++j) values[j] = widenSource(in[j]);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), quantizeBlockAVX2(values, inv, -127, 127));
#else
            for (size_t j = 0; j < BLOCK; ++j) {
                const int v = static_cast<int>(std::lround(widenSource(in[j]) * inv));
                dst[j] = static_cast<int8_t>(std::clamp(v, -127, 127));
            }
#endif
        } else {
            /*
                The scale comes from the SIGNED extreme over -8, not from the
                magnitude over 7, and the difference is not cosmetic.

                amax/7 maps the block onto codes -7..+7. That is fifteen of
                the sixteen a nibble can hold - code 0 never occurs - and
                every step is 14% coarser than it needs to be. Dividing the
                signed extreme by -8 instead lands that element exactly on
                code 0 or code 15, whichever side it is on, and uses the
                whole range in between.

                Measured on a real down_proj it is 9.16% reconstruction
                error against 10.37%, for the same bytes. End to end on
                Qwen3-0.6B it was the difference between +4.17% perplexity
                and +30.63% - four bits has so little margin that a 12.5%
                finer step compounds into something far larger over the
                layers.

                The cost is that the opposite end of the block clips at 7/8
                of the extreme. Within 32 contiguous weights both ends are
                rarely extreme, so the finer step wins and keeps winning.
            */
            const float scale = extreme / -8.0f;
            const float inv = (scale != 0.0f) ? 1.0f / scale : 0.0f;
            scales[b] = (scale != 0.0f) ? scale : 1.0f;

            /*
                Byte j carries weight j and weight j+16 - the two HALVES of
                the block, not adjacent weights.

                Pairing j with j+1 is the obvious choice and it costs
                throughput: the low nibbles become the even-indexed weights
                and the high nibbles the odd ones, so a CPU kernel has to
                deinterleave two strided streams before it can multiply them
                by the activations. Splitting the block in half instead
                gives two CONTIGUOUS runs of sixteen, which took decoding
                from 87.8 to 99.7 tok/s on Qwen3-0.6B. A GPU does not care
                either way; this layout is chosen for the weaker consumer.
            */
            constexpr size_t HALF = BLOCK / 2;
            int8_t* dst = out + b * HALF;
#if defined(__AVX2__)
            alignas(32) float values[BLOCK];
            for (size_t j = 0; j < BLOCK; ++j) values[j] = widenSource(in[j]);
            const __m256i codes = _mm256_add_epi8(quantizeBlockAVX2(values, inv, -8, 7), _mm256_set1_epi8(8));
            const __m128i lo = _mm256_castsi256_si128(codes), hi = _mm256_extracti128_si256(codes, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm_or_si128(lo, _mm_slli_epi16(hi, 4)));
#else
            for (size_t j = 0; j < HALF; ++j) {
                const int lo = std::clamp(
                    static_cast<int>(std::lround(widenSource(in[j]) * inv)) + 8, 0, 15);
                const int hi = std::clamp(
                    static_cast<int>(std::lround(widenSource(in[j + HALF]) * inv)) + 8, 0, 15);
                dst[j] = static_cast<int8_t>((hi << 4) | lo);
            }
#endif
        }
    }
}


/*
    quantize(src, outDim, inDim, type):
    - takes a bfloat16 (or float) [out, in] matrix and returns its quantised
      form. Returns an invalid tensor when the row length is not a multiple
      of the block size, so the caller can fall back rather than silently
      truncating
*/
template <typename Src>
inline QuantTensor quantize(const Src* src, size_t outDim, size_t inDim, Type type) {
    QuantTensor out;
    if (type == Type::None || inDim % BLOCK != 0) return out;

    out.type = type;
    out.outDim = outDim;
    out.inDim = inDim;

    const size_t weights = outDim * inDim;
    const size_t stored = (type == Type::Q4) ? weights / 2 : weights;
    const size_t blocks = outDim * (inDim / BLOCK);

    out.quants = std::shared_ptr<int8_t[]>(new int8_t[stored]);
    out.scales = std::shared_ptr<float[]>(new float[blocks]);

    /* rows are independent, and a large model is tens of gigabytes to
       convert - single threaded that dominates the load time */
    const size_t rowStride = (type == Type::Q4) ? inDim / 2 : inDim;
    kernels::Pool::instance().run(outDim, [&](size_t begin, size_t end) {
        for (size_t r = begin; r < end; ++r)
            quantizeRow(src + r * inDim,
                        out.quants.get() + r * rowStride,
                        out.scales.get() + r * (inDim / BLOCK),
                        inDim, type);
    });

    return out;
}


// Decode one FP8 row per worker, then quantize directly into final storage.
// Full-tensor FP32 staging both quadrupled scratch and first-touched it on
// the caller's NUMA node before the worker could place its own output rows.
inline QuantTensor quantizeFp8(const Fp8Tensor& source, Type type) {
    QuantTensor out;
    if (!source.valid() || type == Type::None || source.inDim % BLOCK) return out;
    out.type = type; out.outDim = source.outDim; out.inDim = source.inDim;
    const size_t stride = type == Type::Q4 ? out.inDim / 2 : out.inDim;
    out.quants = std::shared_ptr<int8_t[]>(new int8_t[out.outDim * stride]);
    out.scales = std::shared_ptr<float[]>(new float[out.outDim * out.blocksPerRow()]);
    kernels::Pool::instance().run(out.outDim, [&](size_t begin, size_t end) {
        std::vector<float> row(out.inDim);
        for (size_t r = begin; r < end; ++r) {
            source.dequantRow(r, row.data());
            quantizeRow(row.data(), out.quants.get() + r * stride,
                out.scales.get() + r * out.blocksPerRow(), out.inDim, type);
        }
    });
    return out;
}

/* ========================================================================= //
    Dot products

    Both keep BLOCK running accumulators, one per position within a block, for
    the same reason the float kernel does: floating point addition is not
    associative, so the compiler only reduces as widely as the code permits.
    Here the block size and the useful accumulator count happen to coincide.
// ========================================================================= */

inline float dotQ8(const float* a, const int8_t* q, const float* scales, size_t n) {
    float acc[BLOCK] = {};
    const size_t blocks = n / BLOCK;

    for (size_t b = 0; b < blocks; ++b) {
        const float* ar = a + b * BLOCK;
        const int8_t* qr = q + b * BLOCK;
        const float sc = scales[b];

        for (size_t j = 0; j < BLOCK; ++j)
            acc[j] += ar[j] * (static_cast<float>(qr[j]) * sc);
    }

    float sum = 0.0f;
    for (size_t j = 0; j < BLOCK; ++j) sum += acc[j];
    return sum;
}


inline float dotQ4(const float* a, const int8_t* q, const float* scales, size_t n) {
    float acc[BLOCK] = {};
    const size_t blocks = n / BLOCK;

    for (size_t b = 0; b < blocks; ++b) {
        const float* ar = a + b * BLOCK;
        const uint8_t* qr = reinterpret_cast<const uint8_t*>(q) + b * (BLOCK / 2);
        const float sc = scales[b];

        constexpr size_t HALF = BLOCK / 2;
        for (size_t j = 0; j < HALF; ++j) {
            const uint8_t packed = qr[j];
            acc[j]        += ar[j]        * (static_cast<float>(int(packed & 0x0F) - 8) * sc);
            acc[j + HALF] += ar[j + HALF] * (static_cast<float>(int(packed >> 4)   - 8) * sc);
        }
    }

    float sum = 0.0f;
    for (size_t j = 0; j < BLOCK; ++j) sum += acc[j];
    return sum;
}


/*
    dequantRange(t, row, from, length, out):
    - columns [from, from + length) of one row back to float, from and
      length both whole blocks. The tiled matmul widens a weight a block at
      a time with this, and dequantRow is the whole row
*/
inline void dequantRange(const QuantTensor& t, size_t row, size_t from, size_t length, float* out) {
    const size_t blocks = t.blocksPerRow();
    const float* scales = t.scales.get() + row * blocks;
    const size_t first = from / BLOCK, last = (from + length) / BLOCK;

    if (t.type == Type::Q8) {
        const int8_t* q = t.quants.get() + row * t.inDim;
        for (size_t b = first; b < last; ++b)
            for (size_t j = 0; j < BLOCK; ++j)
                out[(b - first) * BLOCK + j] = static_cast<float>(q[b * BLOCK + j]) * scales[b];
        return;
    }

    constexpr size_t HALF = BLOCK / 2;
    const uint8_t* q = reinterpret_cast<const uint8_t*>(t.quants.get()) + row * (t.inDim / 2);
    for (size_t b = first; b < last; ++b)
        for (size_t j = 0; j < HALF; ++j) {
            const uint8_t packed = q[b * HALF + j];
            out[(b - first) * BLOCK + j]        = static_cast<float>(int(packed & 0x0F) - 8) * scales[b];
            out[(b - first) * BLOCK + j + HALF] = static_cast<float>(int(packed >> 4)   - 8) * scales[b];
        }
}

/*
    dequantRow(t, row, out):
    - one row back to float. Only the embedding lookup needs this; every
      other use goes through a dot product that never materialises the row
*/
inline void dequantRow(const QuantTensor& t, size_t row, float* out) {
    dequantRange(t, row, 0, t.inDim, out);
}


/* ========================================================================= //
    Integer dot products

    Q8 and Q4 weights are integers, and multiplying them by float activations
    means converting every one of them to float first: a convert and a
    multiply-add per weight. Quantise the ACTIVATIONS to int8 as well - in
    the same blocks of 32, each with its own scale - and a block's dot
    product becomes 32 integer multiplies that the CPU does in one or two
    instructions, with the two scales applied once per block.

    The instructions, best first:
        AVX-512 VNNI / AVX-VNNI   vpdpbusd: 32 byte products summed into
                                  eight 32-bit lanes, one instruction
        AVX2                      vpmaddubsw then vpmaddwd: the same in two
        ARM64 Neon                signed dot products (SDOT), or widening
                                  multiplies when dotprod is not targeted
        anything else             a plain loop, as correct and slower
    vpdpbusd and vpmaddubsw multiply UNSIGNED bytes by signed ones, so the
    activation's sign is moved onto the weight first - |a| * (w * sign a) is
    a * w - which is why activations are kept to -127..127 and never -128,
    whose magnitude does not fit in a signed byte.

    Quantising activations to 8 bits costs a little accuracy, around the
    same as Q8 weights do; it is the trade llama.cpp makes for the same
    reason.
// ========================================================================= */

/*
    quantizeActivation(x, n, q, scales):
    - n floats (a multiple of BLOCK) to int8, a scale per block of 32
*/
inline void quantizeActivation(const float* x, size_t n, int8_t* q, float* scales) {
    for (size_t b = 0; b < n / BLOCK; ++b) {
        const float* in = x + b * BLOCK;
#if defined(__aarch64__) && defined(__ARM_NEON)
        float32x4_t maximum = vdupq_n_f32(0.0f);
        for (size_t j = 0; j < BLOCK; j += 4)
            maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(in + j)));
        const float amax = vmaxvq_f32(maximum);
#else
        float amax = 0.0f;
        for (size_t j = 0; j < BLOCK; ++j) amax = std::max(amax, std::fabs(in[j]));
#endif

        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[b] = scale;
#if defined(__AVX2__)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + b * BLOCK), quantizeBlockAVX2(in, inv, -127, 127));
#elif defined(__aarch64__) && defined(__ARM_NEON)
        for (size_t j = 0; j < BLOCK; j += 8) {
            // FCVTAS matches lround: nearest, with half-way ties away from
            // zero. Saturate to the same -127..127 range before narrowing.
            const int32x4_t lo = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(in + j), inv));
            const int32x4_t hi = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(in + j + 4), inv));
            int8x8_t codes = vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
            codes = vmax_s8(codes, vdup_n_s8(-127));
            vst1_s8(q + b * BLOCK + j, codes);
        }
#else
        for (size_t j = 0; j < BLOCK; ++j)
            q[b * BLOCK + j] = static_cast<int8_t>(std::clamp(
                static_cast<int>(std::lround(in[j] * inv)), -127, 127));
#endif
    }
}

/* a quantised activation row: where its codes and its block scales are */
struct ActivationView {
    const int8_t* q = nullptr;
    const float*  s = nullptr;
};

/* the int8 copy of a whole matrix of activations, row by row */
struct Activations {
    std::vector<int8_t> q;
    std::vector<float>  s;
    size_t              width = 0;

    void quantize(const float* x, size_t rows, size_t n) {
        width = n;
        q.resize(rows * n);
        s.resize(rows * (n / BLOCK));
        for (size_t r = 0; r < rows; ++r)
            quantizeActivation(x + r * n, n, q.data() + r * n, s.data() + r * (n / BLOCK));
    }

    ActivationView row(size_t r) const {
        return {q.data() + r * width, s.data() + r * (width / BLOCK)};
    }
};

#if defined(__AVX2__)
/* 32 signed bytes of a times 32 of w, as eight 32-bit partial sums */
inline __m256i dotBytes(__m256i a, __m256i w) {
    const __m256i magnitude = _mm256_sign_epi8(a, a);
    const __m256i signedW   = _mm256_sign_epi8(w, a);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), magnitude, signedW);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), magnitude, signedW);
#else
    /* byte pairs into int16 - at most 2 * 127 * 128, inside int16 - then
       pairs of those into int32 */
    return _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude, signedW), _mm256_set1_epi16(1));
#endif
}

inline float sumLanes(__m256 v) {
    const __m128 half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 quad = _mm_add_ps(half, _mm_movehl_ps(half, half));
    return _mm_cvtss_f32(_mm_add_ss(quad, _mm_movehdup_ps(quad)));
}
#endif

#if AI_QUANT_AVX512 && defined(__AVX512F__) && defined(__AVX512BW__)
inline __m512i dotBytes512(__m512i a, __m512i w) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i magnitude = _mm512_abs_epi8(a);
    const __m512i signedW = _mm512_mask_sub_epi8(w, _mm512_cmpgt_epi8_mask(zero, a), zero, w);
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(zero, magnitude, signedW);
#else
    return _mm512_madd_epi16(_mm512_maddubs_epi16(magnitude, signedW), _mm512_set1_epi16(1));
#endif
}

// Two scale blocks per ZMM, two independent accumulation chains. Unlike
// globally enabling fast-math this changes only the dot-product reduction.
inline float dotQuant512(ActivationView a, const int8_t* weights, const float* ws,
                         size_t n, bool q4) {
    const size_t blocks = n / BLOCK;
    const auto pair = [&](size_t b) {
        __m512i w;
        if (q4) {
            const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + b * 16));
            const __m256i mask = _mm256_set1_epi8(15);
            const __m256i lo = _mm256_and_si256(p, mask);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(p, 4), mask);
            w = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            w = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), w);
            w = _mm512_sub_epi8(w, _mm512_set1_epi8(8));
        } else w = _mm512_loadu_si512(weights + b * BLOCK);
        const __m512i av = _mm512_loadu_si512(a.q + b * BLOCK);
        const __m512 scale = _mm512_mask_blend_ps(0xff00,
            _mm512_set1_ps(a.s[b] * ws[b]), _mm512_set1_ps(a.s[b + 1] * ws[b + 1]));
        return _mm512_mul_ps(_mm512_cvtepi32_ps(dotBytes512(av, w)), scale);
    };
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    size_t b = 0;
    for (; b + 3 < blocks; b += 4) {
        a0 = _mm512_add_ps(a0, pair(b));
        a1 = _mm512_add_ps(a1, pair(b + 2));
    }
    if (b + 1 < blocks) { a0 = _mm512_add_ps(a0, pair(b)); b += 2; }
    float result = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    if (b < blocks) {
        int sum = 0;
        for (size_t j = 0; j < BLOCK; ++j) {
            const uint8_t packed = uint8_t(weights[b * 16 + j % 16]);
            const int w = q4 ? int((packed >> (j < 16 ? 0 : 4)) & 15) - 8 : weights[b * BLOCK + j];
            sum += int(a.q[b * BLOCK + j]) * w;
        }
        result += float(sum) * (a.s[b] * ws[b]);
    }
    return result;
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
// Four partial sums of 16 signed byte products. Keep the widening fallback
// for ARM64 targets that do not enable the optional dot-product extension.
inline int32x4_t dotBytesNEON(int32x4_t acc, int8x16_t a, int8x16_t w) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, w);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(w));
    const int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(w));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
#endif
}
#endif

/* the 32 weights of block b of a Q4 row, unpacked to -8..7 */
inline void unpackQ4(const uint8_t* packed, int8_t* out) {
    constexpr size_t HALF = BLOCK / 2;
    for (size_t j = 0; j < HALF; ++j) {
        out[j]        = static_cast<int8_t>(int(packed[j] & 0x0F) - 8);
        out[j + HALF] = static_cast<int8_t>(int(packed[j] >> 4) - 8);
    }
}

/* a . w for a Q8 weight row, the activation already in int8 */
inline float dotQ8Q8(ActivationView a, const int8_t* w, const float* ws, size_t n) {
    const size_t blocks = n / BLOCK;
#if AI_QUANT_AVX512 && defined(__AVX512F__) && defined(__AVX512BW__)
    (void)blocks;
    return dotQuant512(a, w, ws, n, false);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < blocks; ++b) {
        const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q + b * BLOCK));
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + b * BLOCK));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dotBytes(av, wv)),
                              _mm256_set1_ps(a.s[b] * ws[b]), acc);
    }
    return sumLanes(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < blocks; ++b) {
        const int8_t* av = a.q + b * BLOCK;
        const int8_t* wv = w + b * BLOCK;
        int32x4_t dot = dotBytesNEON(vdupq_n_s32(0), vld1q_s8(av), vld1q_s8(wv));
        dot = dotBytesNEON(dot, vld1q_s8(av + 16), vld1q_s8(wv + 16));
        acc = vfmaq_n_f32(acc, vcvtq_f32_s32(dot), a.s[b] * ws[b]);
    }
    return vaddvq_f32(acc);
#else
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; ++b) {
        int32_t block = 0;
        for (size_t j = 0; j < BLOCK; ++j)
            block += int32_t(a.q[b * BLOCK + j]) * int32_t(w[b * BLOCK + j]);
        sum += static_cast<float>(block) * a.s[b] * ws[b];
    }
    return sum;
#endif
}

/* the same for a Q4 row: weight j of a block in byte j's low nibble,
   weight j+16 in its high nibble */
inline float dotQ4Q8(ActivationView a, const int8_t* w, const float* ws, size_t n) {
    const size_t blocks = n / BLOCK;
    const uint8_t* packed = reinterpret_cast<const uint8_t*>(w);
#if AI_QUANT_AVX512 && defined(__AVX512F__) && defined(__AVX512BW__)
    (void)blocks; (void)packed;
    return dotQuant512(a, w, ws, n, true);
#elif defined(__AVX2__)
    const __m128i low  = _mm_set1_epi8(0x0F);
    const __m256i bias = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < blocks; ++b) {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + b * (BLOCK / 2)));
        const __m128i lo = _mm_and_si128(bytes, low);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), low);
        const __m256i wv = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), bias);
        const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q + b * BLOCK));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dotBytes(av, wv)),
                              _mm256_set1_ps(a.s[b] * ws[b]), acc);
    }
    return sumLanes(acc);
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < blocks; ++b) {
        const uint8x16_t bytes = vld1q_u8(packed + b * (BLOCK / 2));
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, mask)), bias);
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), bias);
        const int8_t* av = a.q + b * BLOCK;
        int32x4_t dot = dotBytesNEON(vdupq_n_s32(0), vld1q_s8(av), lo);
        dot = dotBytesNEON(dot, vld1q_s8(av + 16), hi);
        acc = vfmaq_n_f32(acc, vcvtq_f32_s32(dot), a.s[b] * ws[b]);
    }
    return vaddvq_f32(acc);
#else
    int8_t unpacked[BLOCK];
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; ++b) {
        unpackQ4(packed + b * (BLOCK / 2), unpacked);
        int32_t block = 0;
        for (size_t j = 0; j < BLOCK; ++j)
            block += int32_t(a.q[b * BLOCK + j]) * int32_t(unpacked[j]);
        sum += static_cast<float>(block) * a.s[b] * ws[b];
    }
    return sum;
#endif
}

/* either, by the tensor's type */
inline float dotQuantRow(ActivationView a, const QuantTensor& t, size_t row) {
    const size_t blocks = t.blocksPerRow();
    const bool   q4     = t.type == Type::Q4;
    const int8_t* w     = t.quants.get() + row * (q4 ? t.inDim / 2 : t.inDim);
    const float*  ws    = t.scales.get() + row * blocks;
    return q4 ? dotQ4Q8(a, w, ws, t.inDim) : dotQ8Q8(a, w, ws, t.inDim);
}

} // !namespace quant
} // !namespace ai
