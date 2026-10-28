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

/* all 256 codes decoded once. Decoding is then one load from a table that
   sits in L1, which is cheaper than the arithmetic and cannot get it wrong */
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
    void dequantRow(size_t row, float* out) const {
        const float*   table = fp8Table();
        const uint8_t* src   = codes.get() + row * inDim;
        const float*   srow  = scales.get() + (row / tileOut) * scaleCols;

        for (size_t c = 0; c < scaleCols; ++c) {
            const size_t begin = c * tileIn;
            const size_t end   = std::min(begin + tileIn, inDim);
            const float  s     = srow[c];
            for (size_t i = begin; i < end; ++i) out[i] = table[src[i]] * s;
        }
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
            for (size_t j = 0; j < BLOCK; ++j) {
                const int v = static_cast<int>(std::lround(widenSource(in[j]) * inv));
                dst[j] = static_cast<int8_t>(std::clamp(v, -127, 127));
            }
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
            for (size_t j = 0; j < HALF; ++j) {
                const int lo = std::clamp(
                    static_cast<int>(std::lround(widenSource(in[j]) * inv)) + 8, 0, 15);
                const int hi = std::clamp(
                    static_cast<int>(std::lround(widenSource(in[j + HALF]) * inv)) + 8, 0, 15);
                dst[j] = static_cast<int8_t>((hi << 4) | lo);
            }
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
    dequantRow(t, row, out):
    - one row back to float. Only the embedding lookup needs this; every
      other use goes through a dot product that never materialises the row
*/
inline void dequantRow(const QuantTensor& t, size_t row, float* out) {
    const size_t blocks = t.blocksPerRow();
    const float* scales = t.scales.get() + row * blocks;

    if (t.type == Type::Q8) {
        const int8_t* q = t.quants.get() + row * t.inDim;
        for (size_t b = 0; b < blocks; ++b)
            for (size_t j = 0; j < BLOCK; ++j)
                out[b * BLOCK + j] = static_cast<float>(q[b * BLOCK + j]) * scales[b];
        return;
    }

    constexpr size_t HALF = BLOCK / 2;
    const uint8_t* q = reinterpret_cast<const uint8_t*>(t.quants.get()) + row * (t.inDim / 2);
    for (size_t b = 0; b < blocks; ++b)
        for (size_t j = 0; j < HALF; ++j) {
            const uint8_t packed = q[b * HALF + j];
            out[b * BLOCK + j]        = static_cast<float>(int(packed & 0x0F) - 8) * scales[b];
            out[b * BLOCK + j + HALF] = static_cast<float>(int(packed >> 4)   - 8) * scales[b];
        }
}

} // !namespace quant
} // !namespace ai
