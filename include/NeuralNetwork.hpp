#pragma once

#include <random>

#include "Vector.hpp"
#include "Matrix.hpp"

namespace math {

/* ========================================================================= //
    Activations
// ========================================================================= */

template <size_t N, typename T> Vec<N, T> sigmoid(const Vec<N, T>& x);
template <size_t N, typename T> Vec<N, T> tanh(const Vec<N, T>& x);
template <size_t N, typename T> Vec<N, T> relu(const Vec<N, T>& x);
template <size_t N, typename T> Vec<N, T> leakyRelu(const Vec<N, T>& x, T alpha = T(0.01));
template <size_t N, typename T> Vec<N, T> softmax(const Vec<N, T>& x);

template <size_t N, typename T> Vec<N, T> sigmoidDerivative(const Vec<N, T>& x);
template <size_t N, typename T> Vec<N, T> tanhDerivative(const Vec<N, T>& x);
template <size_t N, typename T> Vec<N, T> reluDerivative(const Vec<N, T>& x);



/* ========================================================================= //
    Shared infrastructure
// ========================================================================= */

template <size_t N, typename T, typename F> Vec<N, T> apply(const Vec<N, T>& v, F func);



/* ========================================================================= //
    Element-wise / outer product
// ========================================================================= */

template <size_t N, typename T> Vec<N, T> hadamard(const Vec<N, T>& a, const Vec<N, T>& b);

template <size_t N, size_t M, typename T> Mat<N, M, T> outerProduct(const Vec<N, T>& a, const Vec<M, T>& b);



/* ========================================================================= //
    Loss functions
// ========================================================================= */

template <size_t N, typename T> T meanSquaredError(const Vec<N, T>& predicted, const Vec<N, T>& actual);
template <size_t N, typename T> Vec<N, T> meanSquaredErrorDerivative(const Vec<N, T>& predicted, const Vec<N, T>& actual);

template <size_t N, typename T> T crossEntropy(const Vec<N, T>& predicted, const Vec<N, T>& actual);
template <size_t N, typename T> Vec<N, T> crossEntropyDerivative(const Vec<N, T>& predicted, const Vec<N, T>& actual);



/* ========================================================================= //
    Random initialization
// ========================================================================= */

template <size_t N, size_t M, typename T> Mat<N, M, T> randomMat(std::mt19937& rng, T mean = T(0), T stddev = T(1));
template <size_t N, typename T>           Vec<N, T>    randomVec(std::mt19937& rng, T mean = T(0), T stddev = T(1));



/* ========================================================================= //
    Layer
// ========================================================================= */

template <size_t InN, size_t OutN, typename T>
struct Layer {
    Mat<OutN, InN, T> weights;
    Vec<OutN, T> bias;

    Vec<OutN, T> forward(const Vec<InN, T>& input) const;

    Vec<InN, T> backward(const Vec<InN, T>& input, const Vec<OutN, T>& outputGradient,
                          Mat<OutN, InN, T>& weightGradient, Vec<OutN, T>& biasGradient) const;
};

// Network (arbitrary-depth composition of Layers) deliberately left undecided —
// a std::tuple<Layer<N0,N1,T>, Layer<N1,N2,T>, ...> bakes the whole shape into
// the type; a polymorphic base-class-pointer design allows runtime-composed
// depth but gives up compile-time size-checking. Pick one deliberately before
// implementing.

} // !namespace math
