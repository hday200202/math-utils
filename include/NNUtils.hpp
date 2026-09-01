#pragma once

#include <random>
#include <limits>
#include <concepts>
#include <cassert>

#include "Vector.hpp"
#include "Matrix.hpp"

#define e_const 2.718281828459

namespace math {

/* ========================================================================= //
    Activations
// ========================================================================= */

/*
    sigmoid(x):
    - sigmoid activation on a single number x
*/
template <typename T> T sigmoid(const T x) {
    if (x >= 0) return T(1) / (T(1) + std::exp(-x));
    else return std::exp(x) / (T(1) + std::exp(x));
}


/*
    sigmoid(<x>):
    - sigmoid activation on a vector x
*/
template <size_t N, typename T> Vec<N, T> sigmoid(const Vec<N, T>& x) {
    return apply(x, sigmoid<T>);
}


/*
    tanh(x):
    - tanh on a single number x
*/
template <typename T> T tanh(const T x) {
    T t = std::exp(T(-2)*std::abs(x));
    return (std::signbit(x) ? T(-1) : T(1)) * ((T(1) - t)/(T(1) + t));
}


/*
    tanh(<x>):
    - tanh on a vector x
*/
template <size_t N, typename T> Vec<N, T> tanh(const Vec<N, T>& x) {
    return apply(x, tanh<T>);
}


/*
    relu(x): "ReLU(x)"
    - ReLU on a single number x
*/
template <typename T> T relu(const T x) {
    return std::max(T(0), x);
}


/*
    relu(<x>): "ReLU(<x>)"
    - ReLU on a vector x
*/
template <size_t N, typename T> Vec<N, T> relu(const Vec<N, T>& x) {
    return apply(x, relu<T>);
}


/*
    leakyRelu(x, alpha): "LeakyReLU(x)"
    - leaky ReLU on a single number x; alpha is the small slope used for x <= 0
*/
template <typename T> T leakyRelu(const T x, T alpha = T(0.01)) {
    return (x > 0) ? x : alpha*x;
}


/*
    leakyRelu(<x>, alpha): "LeakyReLU(<x>)"
    - leaky ReLU on a vector x; alpha is the small slope used for x <= 0
*/
template <size_t N, typename T> Vec<N, T> leakyRelu(const Vec<N, T>& x, T alpha = T(0.01)) {
    return apply(x, [alpha](T xi){ return leakyRelu(xi, alpha); });
}


/*
    softmax(<x>): "softmax(<x>)"
    - converts a vector of raw scores into a probability distribution
      (all outputs positive, summing to 1); numerically stable via
      max-subtraction before exponentiating
*/
template <size_t N, typename T> Vec<N, T> softmax(const Vec<N, T>& x) {
    T maxVal = std::numeric_limits<T>::lowest();
    T sumVal = T(0);

    for (size_t i = 0; i < N; ++i)
        if (x[i] > maxVal) maxVal = x[i];

    Vec<N, T> shifted = apply(x, [maxVal](T xi){ return std::exp(xi - maxVal); });

    for (size_t i = 0; i < N; ++i)
        sumVal += shifted[i];

    return shifted / sumVal;
}


/*
    sigmoidPrime(x): "sigmoid'(x)"
    - sigmoid' on a single number x
*/
template <typename T> T sigmoidPrime(const T x) {
    return sigmoid(x) * (T(1) - sigmoid(x));
}


/*
    sigmoidPrime(<x>): "sigmoid'(<x>)"
    - sigmoid' on a vector x
*/
template <size_t N, typename T> Vec<N, T> sigmoidPrime(const Vec<N, T>& x) {
    return apply(x, sigmoidPrime<T>);
}


/*
    tanhPrime(x): "tanh'(x)"
    - tanh' on a single number x
*/
template <typename T> T tanhPrime(const T x) {
    T tanhx = tanh(x);
    return T(1) - (tanhx * tanhx);
}


/*
    tanhPrime(<x>): "tanh'(<x>)"
    - tanh' on a vector x
*/
template <size_t N, typename T> Vec<N, T> tanhPrime(const Vec<N, T>& x) {
    return apply(x, tanhPrime<T>);
}


/*
    reluPrime(x): "ReLU'(x)"
    - ReLU' on a single number x
*/
template <typename T> T reluPrime(const T x) {
    return (x > 0) ? T(1) : T(0);
}


/*
    reluPrime(<x>): "ReLU'(<x>)"
    - ReLU' on a vector x
*/
template <size_t N, typename T> Vec<N, T> reluPrime(const Vec<N, T>& x) {
    return apply(x, reluPrime<T>);
}


/*
    leakyReluPrime(x, alpha): "LeakyReLU'(x)"
    - LeakyReLU' on a single number x; alpha is the same slope leakyRelu used for x <= 0
*/
template <typename T> T leakyReluPrime(const T x, T alpha = T(0.01)) {
    return (x > 0) ? T(1) : alpha;
}


/*
    leakyReluPrime(<x>, alpha): "LeakyReLU'(<x>)"
    - LeakyReLU' on a vector x; alpha is the same slope leakyRelu used for x <= 0
*/
template <size_t N, typename T> Vec<N, T> leakyReluPrime(const Vec<N, T>& x, T alpha = T(0.01)) {
    return apply(x, [alpha](T xi){ return leakyReluPrime(xi, alpha); });
}



/* ========================================================================= //
    Element-wise / outer product
// ========================================================================= */

/*
    hadamard(a, b): "a ⊙ b"
    - element-wise (Hadamard) product of two same-length vectors
*/
template <size_t N, typename T> Vec<N, T> hadamard(const Vec<N, T>& a, const Vec<N, T>& b) {
    Vec<N, T> out;

    for (size_t i = 0; i < N; ++i)
        out[i] = a[i] * b[i];

    return out;
}


/*
    outerProduct(a, b): "a ⊗ b"
    - outer product of a length-N vector and a length-M vector, producing an
      N x M matrix where result[i][j] = a[i] * b[j]
*/
template <size_t N, size_t M, typename T> Mat<N, M, T> outerProduct(const Vec<N, T>& a, const Vec<M, T>& b) {
    Mat<N, M, T> out;

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
template <size_t N, typename T> T meanSquaredError(const Vec<N, T>& predicted, const Vec<N, T>& actual) {
    return (predicted - actual).magnitudeSq() / N;
}

/*
    meanSquaredErrorPrime(predicted, actual): "MSE'"
    - derivative of mean squared error with respect to predicted; this is
      the gradient that starts backprop from the output layer
*/
template <size_t N, typename T> Vec<N, T> meanSquaredErrorPrime(const Vec<N, T>& predicted, const Vec<N, T>& actual) {
    return (predicted - actual) * (T(2)/N);
}

/*
    crossEntropy(predicted, actual): "cross-entropy"
    - cross-entropy loss between a predicted probability distribution
      (typically a softmax output) and the actual (typically one-hot) distribution
*/
template <size_t N, typename T> T crossEntropy(const Vec<N, T>& predicted, const Vec<N, T>& actual) {
    T sum = T(0);

    for (size_t i = 0; i < N; ++i)
        sum += actual[i] * std::log(predicted[i]);

    return -sum;
}

/*
    crossEntropyPrime(predicted, actual): "cross-entropy'"
    - derivative of cross-entropy with respect to predicted
*/
template <size_t N, typename T> Vec<N, T> crossEntropyPrime(const Vec<N, T>& predicted, const Vec<N, T>& actual) {
    Vec<N, T> out;

    for (size_t i = 0; i < N; ++i)
        out[i] = -actual[i] / predicted[i];

    return out;
}



/* ========================================================================= //
    Random initialization
// ========================================================================= */

/*
    randomVec(rng, mean, stddev):
    - builds a length-N vector whose entries are drawn from a normal
      distribution; pass an explicit rng so training runs are seedable
      and reproducible rather than silently random every time
*/
template <size_t N, typename T> Vec<N, T> randomVec(std::mt19937& rng, T mean = T(0), T stddev = T(1)) {
    std::normal_distribution<T> dist(mean, stddev);
    Vec<N, T> out;

    for (size_t i = 0; i < N; ++i)
        out[i] = dist(rng);

    return out;
}

/*
    randomMat(rng, mean, stddev):
    - builds an N x M matrix whose entries are drawn from a normal
      distribution; same reproducibility note as randomVec. Built one row
      at a time since Mat<N,M,T> is stored as N rows of Vec<M,T>
*/
template <size_t N, size_t M, typename T> Mat<N, M, T> randomMat(std::mt19937& rng, T mean = T(0), T stddev = T(1)) {
    Mat<N, M, T> out;

    for (size_t i = 0; i < N; ++i)
        out[i] = randomVec<M, T>(rng, mean, stddev);

    return out;
}

} // !namespace math

/* ========================================================================= //
    Dynamically allocated counterparts (math::dyn)
// ========================================================================= */

namespace math::dyn {

/* ========================================================================= //
    Activations
// ========================================================================= */

/*
    NOTE ON THE SCALAR OVERLOADS
    - unlike the static header, the scalar and vector overloads here share the
      same template parameter list shape, so an unconstrained scalar template
      would happily deduce T = Vec<T> and make every call ambiguous. The
      std::floating_point constraint keeps the scalar forms scalar-only.
    - for the same reason the vector forms pass a lambda to apply() rather than
      naming sigmoid<T> directly, which would still be an overload set
*/

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
template <typename T> Vec<T> sigmoid(const Vec<T>& x) {
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
template <typename T> Vec<T> tanh(const Vec<T>& x) {
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
template <typename T> Vec<T> relu(const Vec<T>& x) {
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
template <typename T> Vec<T> leakyRelu(const Vec<T>& x, T alpha = T(0.01)) {
    return apply(x, [alpha](T xi){ return leakyRelu(xi, alpha); });
}


/*
    softmax(<x>): "softmax(<x>)"
    - converts a vector of raw scores into a probability distribution
      (all outputs positive, summing to 1); numerically stable via
      max-subtraction before exponentiating
*/
template <typename T> Vec<T> softmax(const Vec<T>& x) {
    const size_t N = x.size();

    T maxVal = std::numeric_limits<T>::lowest();
    T sumVal = T(0);

    for (size_t i = 0; i < N; ++i)
        if (x[i] > maxVal) maxVal = x[i];

    Vec<T> shifted = apply(x, [maxVal](T xi){ return std::exp(xi - maxVal); });

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
template <typename T> Vec<T> sigmoidPrime(const Vec<T>& x) {
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
template <typename T> Vec<T> tanhPrime(const Vec<T>& x) {
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
template <typename T> Vec<T> reluPrime(const Vec<T>& x) {
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
template <typename T> Vec<T> leakyReluPrime(const Vec<T>& x, T alpha = T(0.01)) {
    return apply(x, [alpha](T xi){ return leakyReluPrime(xi, alpha); });
}



/* ========================================================================= //
    Element-wise / outer product
// ========================================================================= */

/*
    hadamard(a, b): "a (.) b"
    - element-wise (Hadamard) product of two same-length vectors
*/
template <typename T> Vec<T> hadamard(const Vec<T>& a, const Vec<T>& b) {
    assert(a.size() == b.size() && "hadamard: size mismatch");
    Vec<T> out(a.size());

    for (size_t i = 0; i < a.size(); ++i)
        out[i] = a[i] * b[i];

    return out;
}


/*
    outerProduct(a, b): "a (x) b"
    - outer product of a length-N vector and a length-M vector, producing an
      N x M matrix where result[i][j] = a[i] * b[j]
*/
template <typename T> Mat<T> outerProduct(const Vec<T>& a, const Vec<T>& b) {
    const size_t N = a.size();
    const size_t M = b.size();

    Mat<T> out(N, M);

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
template <typename T> T meanSquaredError(const Vec<T>& predicted, const Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "meanSquaredError: size mismatch");
    return (predicted - actual).magnitudeSq() / T(predicted.size());
}

/*
    meanSquaredErrorPrime(predicted, actual): "MSE'"
    - derivative of mean squared error with respect to predicted; this is
      the gradient that starts backprop from the output layer
*/
template <typename T> Vec<T> meanSquaredErrorPrime(const Vec<T>& predicted, const Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "meanSquaredErrorPrime: size mismatch");
    return (predicted - actual) * (T(2)/T(predicted.size()));
}

/*
    crossEntropy(predicted, actual): "cross-entropy"
    - cross-entropy loss between a predicted probability distribution
      (typically a softmax output) and the actual (typically one-hot) distribution
*/
template <typename T> T crossEntropy(const Vec<T>& predicted, const Vec<T>& actual) {
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
template <typename T> Vec<T> crossEntropyPrime(const Vec<T>& predicted, const Vec<T>& actual) {
    assert(predicted.size() == actual.size() && "crossEntropyPrime: size mismatch");
    Vec<T> out(predicted.size());

    for (size_t i = 0; i < predicted.size(); ++i)
        out[i] = -actual[i] / predicted[i];

    return out;
}



/* ========================================================================= //
    Random initialization
// ========================================================================= */

/*
    randomVec(rng, n, mean, stddev):
    - builds a length-n vector whose entries are drawn from a normal
      distribution; pass an explicit rng so training runs are seedable
      and reproducible rather than silently random every time
*/
template <typename T> Vec<T> randomVec(std::mt19937& rng, size_t n, T mean = T(0), T stddev = T(1)) {
    std::normal_distribution<T> dist(mean, stddev);
    Vec<T> out(n);

    for (size_t i = 0; i < n; ++i)
        out[i] = dist(rng);

    return out;
}

/*
    randomMat(rng, n, m, mean, stddev):
    - builds an n x m matrix whose entries are drawn from a normal
      distribution; same reproducibility note as randomVec. Built one row
      at a time since Mat<T> is stored as n rows of Vec<T>
*/
template <typename T> Mat<T> randomMat(std::mt19937& rng, size_t n, size_t m, T mean = T(0), T stddev = T(1)) {
    Mat<T> out(n, m);

    for (size_t i = 0; i < n; ++i)
        out[i] = randomVec<T>(rng, m, mean, stddev);

    return out;
}

} // !namespace math::dyn
