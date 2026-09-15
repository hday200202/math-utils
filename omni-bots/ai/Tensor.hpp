#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace ai {

template <typename T> struct Tensor {
    std::shared_ptr<T[]> storage;
    T* ptr = nullptr;
    std::vector<size_t> shape;
    std::vector<size_t> strides;


    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape) {
        this->shape = std::move(shape);
        strides = defaultStrides(this->shape);
        size_t n = numel();
        storage = std::make_shared<T[]>(n);
        ptr = storage.get();
    }


    static std::vector<size_t> defaultStrides(const std::vector<size_t>& shape) {
        size_t n = shape.size();
        if (n == 0) return {};

        std::vector<size_t> out = std::vector<size_t>(n);
        out[n - 1] = 1;

        for (size_t d = n - 1; d-- > 0; )
            out[d] = out[d + 1] * shape[d + 1];

        return out;
    }


    size_t ndim() const { return shape.size(); }
    size_t numel() const { size_t n = 1; for (size_t d : shape) n *= d; return n; }
    bool isContiguous() const { return strides == defaultStrides(shape); }


    size_t offset(const std::vector<size_t>& idx) const {
        assert(idx.size() == ndim());

        size_t off = 0;
        for (size_t d = 0; d < ndim(); ++d) {
            assert(idx[d] < shape[d]);
            off += idx[d] * strides[d];
        }

        return off;
    }


    T* data() { return ptr; }
    const T* data() const { return ptr; }


    T& at(const std::vector<size_t>& idx) { return ptr[offset(idx)]; }
    const T& at(const std::vector<size_t>& idx) const { return ptr[offset(idx)]; }


    template <typename... Idx> T& operator()(Idx... i)
    { return at({static_cast<size_t>(i)...}); }

    template <typename... Idx> const T& operator()(Idx... i) const
    { return at({static_cast<size_t>(i)...}); }


    Tensor permute(const std::vector<size_t>& order) const {
        assert(order.size() == ndim());

        std::vector<size_t> newShape(ndim());
        std::vector<size_t> newStrides(ndim());
        for (size_t d = 0; d < ndim(); ++d) {
            newShape[d] = shape[order[d]];
            newStrides[d] = strides[order[d]];
        }

        Tensor res = *this;
        res.shape = std::move(newShape);
        res.strides = std::move(newStrides);
        return res;
    }


    Tensor transpose() const {
        assert(ndim() >= 2);
        std::vector<size_t> order(ndim());
        for (size_t d = 0; d < ndim(); ++d) order[d] = d;
        std::swap(order[ndim() - 2], order[ndim() - 1]);
        return permute(order);
    }


    Tensor reshape(std::vector<size_t> newShape) const {
        assert(isContiguous() && "reshape: not contiguous, call contiguous() first");
        Tensor res = *this;
        res.shape = std::move(newShape);
        assert(res.numel() == numel() && "reshape: element count changed");
        res.strides = defaultStrides(res.shape);
        return res;
    }


    Tensor slice(size_t dim, size_t begin, size_t end) const {
        assert(dim < ndim());
        assert(begin <= end && end <= shape[dim]);
        Tensor res = *this;
        res.ptr += begin * strides[dim];
        res.shape[dim] = end - begin;
        return res;
    }


    Tensor row(size_t i) const {
        assert(ndim() >= 1 && i < shape[0]);
        Tensor res;
        res.storage = storage;
        res.ptr = ptr + i * strides[0];
        res.shape.assign(shape.begin() + 1, shape.end());
        res.strides.assign(strides.begin() + 1, strides.end());
        return res;
    }


    Tensor contiguous() const {
        if (isContiguous()) return *this;

        Tensor out(shape);
        const size_t n = numel();
        if (n == 0) return out;

        std::vector<size_t> idx(ndim(), 0);
        for (size_t c = 0; c < n; ++c) {
            out.ptr[c] = ptr[offset(idx)];
            for (size_t d = ndim(); d-- > 0; ) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }

        return out;
    }


    Tensor clone() const {
        Tensor src = contiguous();
        Tensor out(src.shape);
        std::copy(src.ptr, src.ptr + src.numel(), out.ptr);
        return out;
    }


    void fill(T value) {
        assert(isContiguous() && "fill: cannot fill a non-contiguous view");
        std::fill(ptr, ptr + numel(), value);
    }
};



template <typename T> inline Tensor<T> fromValues(
    const std::vector<size_t>& shape,
    const std::vector<T>& values
) {
    Tensor<T> out(shape);
    assert(values.size() == out.numel());
    std::copy(values.begin(), values.end(), out.ptr);
    return out;
}


template <typename T> inline Tensor<T> gather(
    const Tensor<T>& table, 
    const std::vector<size_t>& ids
) {
    assert(table.ndim() == 2);
    const size_t d = table.shape[1];
    Tensor<T> out({ids.size(), d});

    for (size_t i = 0; i < ids.size(); ++i) {
        assert(ids[i] < table.shape[0]);
        const T* src = table.ptr + ids[i] * table.strides[0];
        T* dst = out.ptr + i * d;
        for (size_t j = 0; j < d; ++j) dst[j] = src[j * table.strides[1]];
    }

    return out;
}


template <typename T> inline std::vector<Tensor<T>> split(
    const Tensor<T>& x, 
    size_t parts, 
    size_t dim
) {
    assert(dim < x.ndim());
    assert(parts > 0 && x.shape[dim] % parts == 0);

    const size_t width = x.shape[dim] / parts;
    std::vector<Tensor<T>> out;
    out.reserve(parts);
    for (size_t p = 0; p < parts; ++p)
        out.push_back(x.slice(dim, p*width, (p + 1) * width));
    
    return out;
}


template <typename T> inline Tensor<T> matmul(
    const Tensor<T>& a, 
    const Tensor<T>& b
) {
    assert(a.ndim() == b.ndim());
    assert(a.ndim() == 2 || a.ndim() == 3);

    Tensor<T> A = a.contiguous();
    Tensor<T> B = b.contiguous();

    const size_t batch = (A.ndim() == 3) ? A.shape[0] : 1;
    const size_t M = A.shape[A.ndim() - 2];
    const size_t K = A.shape[A.ndim() - 1];
    const size_t N = B.shape[B.ndim() - 1];

    assert(B.shape[B.ndim() - 2] == K && "matmul: inner dimension mismatch");
    if (A.ndim() == 3) assert(B.shape[0] == batch && "matmul: batch mismatch");

    Tensor<T> out = (A.ndim() == 3) ? Tensor<T>({batch, M, N}) : Tensor<T>({M, N});

    for (size_t s = 0; s < batch; ++s) {
        const T* pa = A.ptr + s * M * K;
        const T* pb = B.ptr + s * K * N;
        T* pc = out.ptr + s * M * N;

        for (size_t i = 0; i < M; ++i) {
            T* crow = pc + i * N;
            for (size_t k = 0; k < K; ++k) {
                const T aik = pa[i * K + k];
                const T* brow = pb + k * N;
                for (size_t j = 0; j < N; ++j) crow[j] += aik * brow[j];
            }
        }
    }
    return out;
}


template <typename T> inline Tensor<T> add(
    const Tensor<T>& a, 
    const Tensor<T>& b
) {
    assert(a.shape == b.shape);
    Tensor<T> A = a.contiguous(), B = b.contiguous();
    Tensor<T> out(A.shape);

    for (size_t i = 0; i < out.numel(); ++i) 
        out.ptr[i] = A.ptr[i] + B.ptr[i];
    
    return out;
}


template <typename T> inline Tensor<T> addBias(
    const Tensor<T>& x, 
    const Tensor<T>& bias
) {
    assert(bias.ndim() == 1 && x.shape.back() == bias.shape[0]);
    Tensor<T> X = x.contiguous(), B = bias.contiguous();
    Tensor<T> out(X.shape);

    const size_t n = X.shape.back();
    const size_t rows = X.numel() / n;
    for (size_t r = 0; r < rows; ++r) {
        const T* src = X.ptr + r * n;
        T* dst = out.ptr + r * n;
        for (size_t j = 0; j < n; ++j) 
            dst[j] = src[j] + B.ptr[j];
    }

    return out;
}


template <typename T> inline Tensor<T> gelu(
    const Tensor<T>& x
) {
    Tensor<T> X = x.contiguous();
    Tensor<T> out(X.shape);

    const T k = T(0.7978845608028654);      // sqrt(2/pi)
    const T c = T(0.044715);

    for (size_t i = 0; i < X.numel(); ++i) {
        const T v = X.ptr[i];
        out.ptr[i] = T(0.5) * v * (T(1) + std::tanh(k * (v + c * v * v * v)));
    }

    return out;
}


template <typename T> inline Tensor<T> softmaxLastAxis(
    const Tensor<T>& x
) {
    Tensor<T> X = x.contiguous();
    Tensor<T> out(X.shape);

    const size_t n = X.shape.back();
    const size_t rows = X.numel() / n;

    for (size_t r = 0; r < rows; ++r) {
        const T* src = X.ptr + r * n;
        T* dst = out.ptr + r * n;

        T m = src[0];
        for (size_t j = 1; j < n; ++j) m = std::max(m, src[j]);

        T sum = T(0);
        for (size_t j = 0; j < n; ++j) { dst[j] = std::exp(src[j] - m); sum += dst[j]; }
        for (size_t j = 0; j < n; ++j) dst[j] /= sum;
    }

    return out;
}


template <typename T> inline Tensor<T> layernorm(
    const Tensor<T>& x, 
    const Tensor<T>& weight,
    const Tensor<T>& bias, 
    T eps = T(1e-5)
) {
    assert(weight.ndim() == 1 && bias.ndim() == 1);
    assert(x.shape.back() == weight.shape[0] && x.shape.back() == bias.shape[0]);

    Tensor<T> X = x.contiguous(), W = weight.contiguous(), B = bias.contiguous();
    Tensor<T> out(X.shape);

    const size_t n = X.shape.back();
    const size_t rows = X.numel() / n;
    for (size_t r = 0; r < rows; ++r) {
        const T* src = X.ptr + r * n;
        T* dst = out.ptr + r * n;

        T mean = T(0);
        for (size_t j = 0; j < n; ++j) mean += src[j];
        mean /= T(n);

        T var = T(0);
        for (size_t j = 0; j < n; ++j) { const T d = src[j] - mean; var += d * d; }
        var /= T(n);

        const T inv = T(1) / std::sqrt(var + eps);
        for (size_t j = 0; j < n; ++j) dst[j] = (src[j] - mean) * inv * W.ptr[j] + B.ptr[j];
    }

    return out;
}


template <typename T> inline Tensor<T> causalMask(
    const Tensor<T>& scores, 
    T maskValue = T(-1e10)
) {
    assert(scores.ndim() >= 2);
    Tensor<T> X = scores.contiguous();
    Tensor<T> out = X.clone();

    const size_t s = X.shape.back();
    assert(X.shape[X.ndim() - 2] == s && "causalMask: last two dims must be square");

    const size_t mats = X.numel() / (s * s);
    for (size_t m = 0; m < mats; ++m) {
        T* p = out.ptr + m * s * s;
        for (size_t i = 0; i < s; ++i)
            for (size_t j = i + 1; j < s; ++j)
                p[i * s + j] = maskValue;
    }

    return out;
}


}
