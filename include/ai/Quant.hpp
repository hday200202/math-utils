#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
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

/*
    quantizeRow(src, out, scales, inDim, type):
    - symmetric, so zero maps to zero exactly and no zero-point is needed.
      The scale is the block's largest magnitude over the integer range, which
      means the largest weight is represented exactly and everything else
      rounds to the nearest step.
    - Q4 packs two weights per byte, the even index in the low nibble. The
      stored value is offset by 8 so the range 0..15 covers -8..7
*/
inline void quantizeRow(const uint16_t* src, int8_t* out, float* scales,
                        size_t inDim, Type type) {
    const size_t blocks = inDim / BLOCK;

    for (size_t b = 0; b < blocks; ++b) {
        const uint16_t* in = src + b * BLOCK;

        float amax = 0.0f;
        for (size_t j = 0; j < BLOCK; ++j)
            amax = std::max(amax, std::fabs(widenBF16(in[j])));

        if (type == Type::Q8) {
            const float scale = amax / 127.0f;
            const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            scales[b] = scale;

            int8_t* dst = out + b * BLOCK;
            for (size_t j = 0; j < BLOCK; ++j) {
                const int v = static_cast<int>(std::lround(widenBF16(in[j]) * inv));
                dst[j] = static_cast<int8_t>(std::clamp(v, -127, 127));
            }
        } else {
            const float scale = amax / 7.0f;
            const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            scales[b] = scale;

            int8_t* dst = out + b * (BLOCK / 2);
            for (size_t j = 0; j < BLOCK / 2; ++j) {
                const int lo = std::clamp(
                    static_cast<int>(std::lround(widenBF16(in[2 * j]) * inv)), -8, 7) + 8;
                const int hi = std::clamp(
                    static_cast<int>(std::lround(widenBF16(in[2 * j + 1]) * inv)), -8, 7) + 8;
                dst[j] = static_cast<int8_t>((hi << 4) | lo);
            }
        }
    }
}


/*
    quantize(src, outDim, inDim, type):
    - takes a bfloat16 [out, in] matrix and returns its quantised form.
      Returns an invalid tensor when the row length is not a multiple of the
      block size, so the caller can fall back rather than silently truncating
*/
inline QuantTensor quantize(const uint16_t* src, size_t outDim, size_t inDim, Type type) {
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

        for (size_t j = 0; j < BLOCK / 2; ++j) {
            const uint8_t packed = qr[j];
            acc[2 * j]     += ar[2 * j]     * (static_cast<float>(int(packed & 0x0F) - 8) * sc);
            acc[2 * j + 1] += ar[2 * j + 1] * (static_cast<float>(int(packed >> 4)   - 8) * sc);
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

    const uint8_t* q = reinterpret_cast<const uint8_t*>(t.quants.get()) + row * (t.inDim / 2);
    for (size_t b = 0; b < blocks; ++b)
        for (size_t j = 0; j < BLOCK / 2; ++j) {
            const uint8_t packed = q[b * (BLOCK / 2) + j];
            out[b * BLOCK + 2 * j]     = static_cast<float>(int(packed & 0x0F) - 8) * scales[b];
            out[b * BLOCK + 2 * j + 1] = static_cast<float>(int(packed >> 4)   - 8) * scales[b];
        }
}

} // !namespace quant
} // !namespace ai
