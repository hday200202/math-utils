#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Specialize at pipeline creation: no dtype branch in the inner loops.
constant ushort format [[function_constant(0)]]; // F32, BF16, Q8, Q4
struct Shape { uint m, k, n; };

inline float weight(device const uchar* b, device const float* scales,
                    uint row, uint col, uint k) {
    const ulong at = ulong(row) * k + col;
    if (format == 0) return reinterpret_cast<device const float*>(b)[at];
    if (format == 1) return as_type<float>(uint(reinterpret_cast<device const ushort*>(b)[at]) << 16);
    const float scale = scales[ulong(row) * (k / 32) + col / 32];
    if (format == 2) return float(reinterpret_cast<device const char*>(b)[at]) * scale;
    const uchar p = b[ulong(row) * (k / 2) + (col / 32) * 16 + col % 16];
    return float(int((p >> ((col & 16) ? 4 : 0)) & 15) - 8) * scale;
}

inline float row_product(device const float* a, device const uchar* b,
    device const float* scales, uint k, uint row, uint lane) {
    if (format == 3) {
        // Each lane loads four packed bytes once and consumes BOTH nibble
        // halves. A SIMD group covers eight complete blocks per iteration,
        // fetching each packed byte once; four lanes share a block scale.
        float4 loSum(0), hiSum(0);
        const uint offset = (lane % 4) * 4;
        for (uint block = lane / 4; block < k / 32; block += 8) {
            const uchar4 packed = *reinterpret_cast<device const uchar4*>(b + ulong(row) * (k / 2) + block * 16 + offset);
            const float scale = scales[ulong(row) * (k / 32) + block];
            const float4 lo = float4(int4(packed & uchar4(15)) - 8) * scale;
            const float4 hi = float4(int4(packed >> 4) - 8) * scale;
            const float4 x0 = *reinterpret_cast<device const float4*>(a + block * 32 + offset);
            const float4 x1 = *reinterpret_cast<device const float4*>(a + block * 32 + offset + 16);
            loSum = fma(x0, lo, loSum);
            hiSum = fma(x1, hi, hiSum);
        }
        const float4 sum = loSum + hiSum;
        return simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    }
    float4 acc(0);
    for (uint col = lane * 4; col < k; col += 128) {
        float4 x(0), w(0);
        if (col + 3 < k) {
            x = *reinterpret_cast<device const float4*>(a + col);
            const ulong at = ulong(row) * k + col;
            if (format == 0) w = *reinterpret_cast<device const float4*>(b + at * 4);
            else if (format == 1) {
                const ushort4 v = *reinterpret_cast<device const ushort4*>(b + at * 2);
                w = as_type<float4>(uint4(v) << 16);
            } else if (format == 2) {
                const char4 v = *reinterpret_cast<device const char4*>(b + at);
                w = float4(v) * scales[ulong(row) * (k / 32) + col / 32];
            } else {
                const ulong offset = ulong(row) * (k / 2) + (col / 32) * 16 + col % 16;
                const uchar4 v = *reinterpret_cast<device const uchar4*>(b + offset);
                const uchar4 codes = (v >> ((col & 16) ? 4 : 0)) & uchar4(15);
                w = float4(int4(codes) - 8) * scales[ulong(row) * (k / 32) + col / 32];
            }
        } else {
            for (uint j = 0; j < 4 && col + j < k; ++j) {
                x[j] = a[col + j];
                w[j] = weight(b, scales, row, col + j, k);
            }
        }
        acc = fma(x, w, acc);
    }
    const float sum = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    return sum;
}

// Four independent rows per threadgroup, one SIMD group per row. Float4
// loads amortize addressing; Q4 stays packed until it reaches the registers.
kernel void gemv(device const float* a [[buffer(0)]],
                 device const uchar* b [[buffer(1)]],
                 device const float* scales [[buffer(2)]],
                 device float* c [[buffer(3)]], constant Shape& p [[buffer(4)]],
                 uint2 group [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x * 4 + simd;
    if (row >= p.n) return;
    const float sum = row_product(a + ulong(group.y) * p.k, b, scales, p.k, row, lane);
    if (lane == 0) c[ulong(group.y) * p.n + row] = sum;
}

struct MultiShape { uint k, n0, n1, n2; };
kernel void gemv_multi(device const float* a [[buffer(0)]],
    device const uchar* b0 [[buffer(1)]], device const uchar* b1 [[buffer(2)]], device const uchar* b2 [[buffer(3)]],
    device const float* s0 [[buffer(4)]], device const float* s1 [[buffer(5)]], device const float* s2 [[buffer(6)]],
    device float* o0 [[buffer(7)]], device float* o1 [[buffer(8)]], device float* o2 [[buffer(9)]],
    constant MultiShape& p [[buffer(10)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group * 4 + sg;
    if (row >= p.n0 + p.n1 + p.n2) return;
    device const uchar* b = b0; device const float* scales = s0; device float* out = o0;
    if (row >= p.n0 + p.n1) { row -= p.n0 + p.n1; b = b2; scales = s2; out = o2; }
    else if (row >= p.n0) { row -= p.n0; b = b1; scales = s1; out = o1; }
    const float sum = row_product(a, b, scales, p.k, row, lane);
    if (lane == 0) out[row] = sum;
}

// 32x32 output tiles, 32-wide K tiles. Each SIMD group owns a 16x16
// quadrant. Float accumulators avoid a hidden FP16 conversion of weights.
kernel void gemm(device const float* a [[buffer(0)]],
                 device const uchar* b [[buffer(1)]],
                 device const float* scales [[buffer(2)]],
                 device float* c [[buffer(3)]], constant Shape& p [[buffer(4)]],
                 uint2 group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ta[32 * 32];
    threadgroup float tb[32 * 32];
    const uint m0 = group.y * 32, n0 = group.x * 32;
    const uint sm = (simd / 2) * 16, sn = (simd % 2) * 16;
    simdgroup_float8x8 c00(0), c01(0), c10(0), c11(0);
    for (uint k0 = 0; k0 < p.k; k0 += 32) {
        for (uint t = tid; t < 1024; t += 128) {
            const uint r = t / 32, s = t % 32;
            ta[t] = m0 + r < p.m && k0 + s < p.k ? a[ulong(m0 + r) * p.k + k0 + s] : 0;
            // Transpose B while decoding the tile, for A @ B^T.
            tb[s * 32 + r] = n0 + r < p.n && k0 + s < p.k
                ? weight(b, scales, n0 + r, k0 + s, p.k) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Reduce each K tile separately instead of a K-long FMA chain.
        // This bounds cancellation error without reducing input precision.
        simdgroup_float8x8 t00(0), t01(0), t10(0), t11(0);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, ta + sm * 32 + kk, 32);
            simdgroup_load(a1, ta + (sm + 8) * 32 + kk, 32);
            simdgroup_load(b0, tb + kk * 32 + sn, 32);
            simdgroup_load(b1, tb + kk * 32 + sn + 8, 32);
            simdgroup_multiply_accumulate(t00, a0, b0, t00);
            simdgroup_multiply_accumulate(t01, a0, b1, t01);
            simdgroup_multiply_accumulate(t10, a1, b0, t10);
            simdgroup_multiply_accumulate(t11, a1, b1, t11);
        }
        c00.thread_elements() += t00.thread_elements();
        c01.thread_elements() += t01.thread_elements();
        c10.thread_elements() += t10.thread_elements();
        c11.thread_elements() += t11.thread_elements();
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ta + sm * 32 + sn, 32);
    simdgroup_store(c01, ta + sm * 32 + sn + 8, 32);
    simdgroup_store(c10, ta + (sm + 8) * 32 + sn, 32);
    simdgroup_store(c11, ta + (sm + 8) * 32 + sn + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < 1024; t += 128) {
        const uint r = m0 + t / 32, s = n0 + t % 32;
        if (r < p.m && s < p.n) c[ulong(r) * p.n + s] = ta[t];
    }
}

struct DenseParams {
    uint width, rows, heads, kvHeads, dim, capacity, position, rotary;
    float epsilon;
    uint mode, window;
};

kernel void dense_norm(device float* x [[buffer(0)]], device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]], device const float* residual [[buffer(3)]],
    constant DenseParams& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 4 + sg;
    if (row >= p.rows) return;
    float sum = 0;
    for (uint d = lane; d < p.width; d += 32) {
        const float v = x[ulong(row) * p.width + d] + (p.mode ? residual[d] : 0.0f);
        sum = fma(v, v, sum);
    }
    const float inv = 1.0f / sqrt(simd_sum(sum) / float(p.width) + p.epsilon);
    for (uint d = lane; d < p.width; d += 32) {
        const ulong at = ulong(row) * p.width + d;
        const float v = x[at] + (p.mode ? residual[d] : 0.0f);
        if (p.mode) x[at] = v;
        out[at] = (v * inv) * gamma[d];
    }
}

kernel void dense_scores(device const float* q [[buffer(0)]], device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]], constant DenseParams& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint pos = group.x * 4 + sg, head = group.y;
    const uint first = p.window && p.position + 1 > p.window ? p.position + 1 - p.window : 0;
    if (pos > p.position || pos < first) return;
    const uint kh = head / (p.heads / p.kvHeads);
    float4 sum(0);
    for (uint d = lane * 4; d < p.dim; d += 128) {
        const float4 a = *reinterpret_cast<device const float4*>(q + head * p.dim + d);
        const float4 b = *reinterpret_cast<device const float4*>(keys + (ulong(pos) * p.kvHeads + kh) * p.dim + d);
        sum = fma(a, b, sum);
    }
    const float dot = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (lane == 0) scores[ulong(head) * p.capacity + pos] = dot / sqrt(float(p.dim));
}

kernel void dense_swiglu(device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    constant DenseParams& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.width) {
        const float g = gate[i];
        gate[i] = (g / (1.0f + exp(-g))) * up[i];
    }
}

// Q/K head normalization, rotary position encoding and KV insertion share
// one pass. Every SIMD group owns a head, so its rotated pairs never race.
kernel void dense_qk_cache(device float* q [[buffer(0)]], device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* qGamma [[buffer(3)]],
    device const float* kGamma [[buffer(4)]], device float* keys [[buffer(5)]],
    device float* values [[buffer(6)]], device const float2* trig [[buffer(7)]],
    constant DenseParams& p [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = group * 4 + sg;
    if (head >= p.heads + p.kvHeads) return;
    const bool query = head < p.heads;
    const uint h = query ? head : head - p.heads;
    device float* x = (query ? q : k) + h * p.dim;
    device const float* gamma = query ? qGamma : kGamma;
    const bool normalize = query ? (p.mode & 1) : (p.mode & 2);
    float inv = 1;
    if (normalize) {
        float sum = 0;
        for (uint d = lane; d < p.dim; d += 32) sum = fma(x[d], x[d], sum);
        inv = 1.0f / sqrt(simd_sum(sum) / float(p.dim) + p.epsilon);
    }
    const uint halfDim = p.rotary / 2;
    const ulong offset = (ulong(p.position) * p.kvHeads + h) * p.dim;
    for (uint j = lane; j < halfDim; j += 32) {
        const float a = normalize ? (x[j] * inv) * gamma[j] : x[j];
        const float b = normalize ? (x[j + halfDim] * inv) * gamma[j + halfDim] : x[j + halfDim];
        const float2 cs = trig[j];
        const float lo = a * cs.x - b * cs.y, hi = b * cs.x + a * cs.y;
        x[j] = lo; x[j + halfDim] = hi;
        if (!query) { keys[offset + j] = lo; keys[offset + j + halfDim] = hi; }
    }
    for (uint j = p.rotary + lane; j < p.dim; j += 32) {
        const float value = normalize ? (x[j] * inv) * gamma[j] : x[j];
        x[j] = value;
        if (!query) keys[offset + j] = value;
    }
    if (!query) for (uint j = lane; j < p.dim; j += 32) values[offset + j] = v[h * p.dim + j];
}

kernel void dense_softmax_values(device float* scores [[buffer(0)]], device const float* values [[buffer(1)]],
    device float* out [[buffer(2)]], constant DenseParams& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = group * 4 + sg;
    if (head >= p.heads) return;
    device float* s = scores + ulong(head) * p.capacity;
    const uint first = p.window && p.position + 1 > p.window ? p.position + 1 - p.window : 0;
    float maximum = -INFINITY;
    for (uint t = first + lane; t <= p.position; t += 32) maximum = max(maximum, s[t]);
    maximum = simd_max(maximum);
    float sum = 0;
    for (uint t = first + lane; t <= p.position; t += 32) sum += exp(s[t] - maximum);
    const float inv = 1.0f / simd_sum(sum);
    for (uint t = first + lane; t <= p.position; t += 32) s[t] = exp(s[t] - maximum) * inv;
    simdgroup_barrier(mem_flags::mem_device);
    const uint kh = head / (p.heads / p.kvHeads);
    for (uint d = lane * 4; d < p.dim; d += 128) {
        float4 acc(0);
        for (uint t = first; t <= p.position; ++t) {
            const float4 value = *reinterpret_cast<device const float4*>(values + (ulong(t) * p.kvHeads + kh) * p.dim + d);
            acc = fma(float4(s[t]), value, acc);
        }
        *reinterpret_cast<device float4*>(out + head * p.dim + d) = acc;
    }
}


kernel void dense_bias(device float* out [[buffer(0)]], device const float* bias [[buffer(1)]],
    constant DenseParams& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.width) out[i] += bias[i];
}

// Split the value reduction over 128-token partitions. A SIMD group owns
// one head/partition; the merge rescales local softmaxes by their maxima.
// This bounds the serial value walk at long context without approximating it.
kernel void dense_partial_values(device const float* scores [[buffer(0)]],
    device const float* values [[buffer(1)]], device float* partials [[buffer(2)]],
    constant DenseParams& p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = group.x * 4 + sg;
    if (head >= p.heads) return;
    const uint first = p.window && p.position + 1 > p.window ? p.position + 1 - p.window : 0;
    const uint begin = first + group.y * 128, end = min(begin + 128, p.position + 1);
    device const float* s = scores + ulong(head) * p.capacity;
    threadgroup float probabilities[4 * 128];
    threadgroup float* probs = probabilities + sg * 128;
    float maximum = -INFINITY;
    for (uint t = begin + lane; t < end; t += 32) maximum = max(maximum, s[t]);
    maximum = simd_max(maximum);
    float sum = 0;
    for (uint t = begin + lane; t < end; t += 32) {
        const float weight = exp(s[t] - maximum);
        probs[t - begin] = weight; sum += weight;
    }
    sum = simd_sum(sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint kh = head / (p.heads / p.kvHeads);
    device float* dst = partials + (ulong(head) * p.rows + group.y) * (p.dim + 4);
    for (uint d = lane * 4; d < p.dim; d += 128) {
        float4 acc(0);
        for (uint t = begin; t < end; ++t) {
            const float4 v = *reinterpret_cast<device const float4*>(values + (ulong(t) * p.kvHeads + kh) * p.dim + d);
            acc = fma(float4(probs[t - begin]), v, acc);
        }
        *reinterpret_cast<device float4*>(dst + d) = acc;
    }
    if (lane == 0) { dst[p.dim] = maximum; dst[p.dim + 1] = sum; }
}

kernel void dense_merge_values(device const float* partials [[buffer(0)]],
    device float* out [[buffer(1)]], constant DenseParams& p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = group * 4 + sg;
    if (head >= p.heads) return;
    const uint length = p.window ? min(p.position + 1, p.window) : p.position + 1;
    const uint partitions = (length + 127) / 128;
    device const float* src = partials + ulong(head) * p.rows * (p.dim + 4);
    float maximum = -INFINITY;
    for (uint i = lane; i < partitions; i += 32) maximum = max(maximum, src[ulong(i) * (p.dim + 4) + p.dim]);
    maximum = simd_max(maximum);
    float total = 0;
    for (uint i = lane; i < partitions; i += 32) {
        device const float* row = src + ulong(i) * (p.dim + 4);
        total += exp(row[p.dim] - maximum) * row[p.dim + 1];
    }
    const float inv = 1.0f / simd_sum(total);
    for (uint d = lane * 4; d < p.dim; d += 128) {
        float4 sum(0);
        for (uint i = 0; i < partitions; ++i) {
            device const float* row = src + ulong(i) * (p.dim + 4);
            sum = fma(float4(exp(row[p.dim] - maximum) * inv),
                *reinterpret_cast<device const float4*>(row + d), sum);
        }
        *reinterpret_cast<device float4*>(out + head * p.dim + d) = sum;
    }
}
