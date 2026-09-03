#pragma once

#include <limits>
#include <cassert>
#include <concepts>

#include "../math/Math.hpp"

#define e_const 2.718281828459

namespace ai {

/* ========================================================================= //
    Activations

    The scalar overloads are constrained to std::floating_point: without
    that, the Vec<T> overloads below would make T = Vec<T> a valid
    deduction for the "scalar" overload too, and every call would be
    ambiguous.
// ========================================================================= */

/*
    sigmoid(x):
    - sigmoid activation on a single number x
*/
template <std::floating_point T> T sigmoid(const T x) {
    if (x >= 0) return T(1) / (T(1) + std::exp(-x));
    else return std::exp(x) / (T(1) + std::exp(x));
}

/*
    sigmoid(<x>):
    - sigmoid activation on a vector x
*/
template <typename T> math::Vec<T> sigmoid(const math::Vec<T>& x) {
    return apply(x, [](T xi){ return sigmoid(xi); });
}


/*
    tanh(x):
    - tanh on a single number x
*/
template <std::floating_point T> T tanh(const T x) {
    T t = std::exp(T(-2)*std::abs(x));
    return (std::signbit(x) ? T(-1) : T(1)) * ((T(1) - t)/(T(1) + t));
}

/*
    tanh(<x>):
    - tanh on a vector x
*/
template <typename T> math::Vec<T> tanh(const math::Vec<T>& x) {
    return apply(x, [](T xi){ return tanh(xi); });
}


/*
    relu(x): "ReLU(x)"
    - ReLU on a single number x
*/
template <std::floating_point T> T relu(const T x) {
    return std::max(T(0), x);
}

/*
    relu(<x>): "ReLU(<x>)"
    - ReLU on a vector x
*/
template <typename T> math::Vec<T> relu(const math::Vec<T>& x) {
    return apply(x, [](T xi){ return relu(xi); });
}


/*
    leakyRelu(x, alpha): "LeakyReLU(x)"
    - leaky ReLU on a single number x; alpha is the small slope used for x <= 0
*/
template <std::floating_point T> T leakyRelu(const T x, T alpha = T(0.01)) {
    return (x > 0) ? x : alpha*x;
}

/*
    leakyRelu(<x>, alpha): "LeakyReLU(<x>)"
    - leaky ReLU on a vector x; alpha is the small slope used for x <= 0
*/
template <typename T> math::Vec<T> leakyRelu(const math::Vec<T>& x, T alpha = T(0.01)) {
    return apply(x, [alpha](T xi){ return leakyRelu(xi, alpha); });
}


/*
    softmax(<x>): "softmax(<x>)"
    - converts a vector of raw scores into a probability distribution
      (all outputs positive, summing to 1); numerically stable via
      max-subtraction before exponentiating
*/
template <typename T> math::Vec<T> softmax(const math::Vec<T>& x) {
    const size_t N = x.size();

    T maxVal = std::numeric_limits<T>::lowest();
    T sumVal = T(0);

    for (size_t i = 0; i < N; ++i)
        if (x[i] > maxVal) maxVal = x[i];

    math::Vec<T> shifted = apply(x, [maxVal](T xi){ return std::exp(xi - maxVal); });

    for (size_t i = 0; i < N; ++i)
        sumVal += shifted[i];

    return shifted / sumVal;
}


/*
    sigmoidPrime(x): "sigmoid'(x)"
    - sigmoid' on a single number x
*/
template <std::floating_point T> T sigmoidPrime(const T x) {
    return sigmoid(x) * (T(1) - sigmoid(x));
}

/*
    sigmoidPrime(<x>): "sigmoid'(<x>)"
    - sigmoid' on a vector x
*/
template <typename T> math::Vec<T> sigmoidPrime(const math::Vec<T>& x) {
    return apply(x, [](T xi){ return sigmoidPrime(xi); });
}


/*
    tanhPrime(x): "tanh'(x)"
    - tanh' on a single number x
*/
template <std::floating_point T> T tanhPrime(const T x) {
    T tanhx = tanh(x);
    return T(1) - (tanhx * tanhx);
}

/*
    tanhPrime(<x>): "tanh'(<x>)"
    - tanh' on a vector x
*/
template <typename T> math::Vec<T> tanhPrime(const math::Vec<T>& x) {
    return apply(x, [](T xi){ return tanhPrime(xi); });
}


/*
    reluPrime(x): "ReLU'(x)"
    - ReLU' on a single number x
*/
template <std::floating_point T> T reluPrime(const T x) {
    return (x > 0) ? T(1) : T(0);
}

/*
    reluPrime(<x>): "ReLU'(<x>)"
    - ReLU' on a vector x
*/
template <typename T> math::Vec<T> reluPrime(const math::Vec<T>& x) {
    return apply(x, [](T xi){ return reluPrime(xi); });
}


/*
    leakyReluPrime(x, alpha): "LeakyReLU'(x)"
    - LeakyReLU' on a single number x; alpha is the same slope leakyRelu used for x <= 0
*/
template <std::floating_point T> T leakyReluPrime(const T x, T alpha = T(0.01)) {
    return (x > 0) ? T(1) : alpha;
}

/*
    leakyReluPrime(<x>, alpha): "LeakyReLU'(<x>)"
    - LeakyReLU' on a vector x; alpha is the same slope leakyRelu used for x <= 0
*/
template <typename T> math::Vec<T> leakyReluPrime(const math::Vec<T>& x, T alpha = T(0.01)) {
    return apply(x, [alpha](T xi){ return leakyReluPrime(xi, alpha); });
}



/* ========================================================================= //
    Element-wise / outer product
// ========================================================================= */

/*
    hadamard(a, b): "a ⊙ b"
    - element-wise (Hadamard) product of two same-length vectors
*/
template <typename T> math::Vec<T> hadamard(const math::Vec<T>& a, const math::Vec<T>& b) {
    assert(a.size() == b.size() && "hadamard: size mismatch");
    math::Vec<T> out(a.size());

    for (size_t i = 0; i < a.size(); ++i)
        out[i] = a[i] * b[i];

    return out;
}


/*
    outerProduct(a, b): "a ⊗ b"
    - outer product of a length-N vector and a length-M vector, producing an
      N x M matrix where result[i][j] = a[i] * b[j]
*/
template <typename T> math::Mat<T> outerProduct(const math::Vec<T>& a, const math::Vec<T>& b) {
    const size_t N = a.size();
    const size_t M = b.size();

    math::Mat<T> out(N, M);

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
            out[i][j] = a[i] * b[j];

    return out;
}



/* ========================================================================= //
    Loss functions
// ========================================================================= */

/*
    meanSquaredError(predicted, actual): "MSE"
    - mean squared error loss between a predicted and an actual vector
*/
template <typename T> T meanSquaredError(const math::Vec<T>& predicted, const math::Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "meanSquaredError: size mismatch");
    return (predicted - actual).magnitudeSq() / T(predicted.size());
}

/*
    meanSquaredErrorPrime(predicted, actual): "MSE'"
    - derivative of mean squared error with respect to predicted; this is
      the gradient that starts backprop from the output layer
*/
template <typename T> math::Vec<T> meanSquaredErrorPrime(const math::Vec<T>& predicted, const math::Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "meanSquaredErrorPrime: size mismatch");
    return (predicted - actual) * (T(2)/T(predicted.size()));
}

/*
    crossEntropy(predicted, actual): "cross-entropy"
    - cross-entropy loss between a predicted probability distribution
      (typically a softmax output) and the actual (typically one-hot) distribution
*/
template <typename T> T crossEntropy(const math::Vec<T>& predicted, const math::Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "crossEntropy: size mismatch");
    T sum = T(0);

    for (size_t i = 0; i < predicted.size(); ++i)
        sum += actual[i] * std::log(predicted[i]);

    return -sum;
}

/*
    crossEntropyPrime(predicted, actual): "cross-entropy'"
    - derivative of cross-entropy with respect to predicted
*/
template <typename T> math::Vec<T> crossEntropyPrime(const math::Vec<T>& predicted, const math::Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "crossEntropyPrime: size mismatch");
    math::Vec<T> out(predicted.size());

    for (size_t i = 0; i < predicted.size(); ++i)
        out[i] = -actual[i] / predicted[i];

    return out;
}

} // !namespace ai
