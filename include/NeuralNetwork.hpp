#pragma once

#include <tuple>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "NNUtils.hpp"

namespace math {

/* ========================================================================= //
    Layer
// ========================================================================= */

enum class Activation { Sigmoid, Tanh, Relu, LeakyRelu };
enum class Loss { MeanSquaredError, CrossEntropy };



template <typename T> struct NNConfig {
    Activation  hiddenActivation    = Activation::Relu;
    Activation  outputActivation    = Activation::Sigmoid;
    Loss        loss                = Loss::MeanSquaredError;

    T           learningRate        = T(0.01);
    size_t      epochs              = 100;
    size_t      batchSize           = 1;

    T           leakyReluAlpha      = T(0.01);

    uint32_t    seed                = 0;
    T           initMean            = T(0);
    T           initStddev          = T(1);

    bool        shuffleEachEpoch    = true;
    size_t      logInterval         = 0;
};



/* ========================================================================= //
    Layer
// ========================================================================= */

template <size_t InN, size_t OutN, typename T> struct Layer {
    Mat<OutN, InN, T> weights;
    Vec<OutN, T> bias;

    Vec<OutN, T> forward(const Vec<InN, T>& input) const {
        return weights * input + bias;
    }

    Vec<InN, T> backward(
        const Vec<InN, T>& input,
        const Vec<OutN, T>& outputGradient,
        Mat<OutN, InN, T>& weightGradient,
        Vec<OutN, T>& biasGradient
    ) const {
        weightGradient = outerProduct(outputGradient, input);
        biasGradient = outputGradient;

        return weights.transpose() * outputGradient;
    }
};



/* ========================================================================= //
    Neural Network
// ========================================================================= */

/*
    LayerTuple<T, First, Second, Rest...>:
    - builds the std::tuple<Layer<...>, Layer<...>, ...> type for a chain of
      layer sizes, one Layer per consecutive pair. Recursive pack-peeling:
      base case is exactly two sizes left (one Layer), recursive case peels
      the first size off and tuple_cats its Layer onto the tuple built from
      the remaining sizes (Second carries over as the next layer's input size)
*/
template <typename T, size_t First, size_t Second, size_t... Rest>
struct LayerTuple {
    using type = decltype(std::tuple_cat(
        std::tuple<Layer<First, Second, T>>{},
        typename LayerTuple<T, Second, Rest...>::type{}
    ));
};

template <typename T, size_t First, size_t Second>
struct LayerTuple<T, First, Second> {
    using type = std::tuple<Layer<First, Second, T>>;
};

template <typename T, size_t... LayerSizes>
using LayerTupleT = typename LayerTuple<T, LayerSizes...>::type;



template <typename T, size_t... LayerSizes> struct NeuralNetwork {
    static_assert(sizeof...(LayerSizes) >= 2, "NeuralNetwork needs at least an input and output size");


    LayerTupleT<T, LayerSizes...> layers;
    NNConfig<T> cfg = {};
    std::mt19937 rng;


    void init(const NNConfig<T>& cfg) {
        this->cfg = cfg;
        this->rng.seed(this->cfg.seed);

        std::apply([this](auto&... v) {
            (this->randomizeLayer(
                v, this->rng, 
                this->cfg.initMean, 
                this->cfg.initStddev
            ), ...);
        }, layers);
    }

    
    template <typename VecT> auto forward(
        const VecT& input
    ) const { return forwardRecurse<0>(input); }


    template <typename VecIn, typename VecActual> void train(
        const std::vector<std::pair<VecIn, VecActual>>& dataset
    ) {
        auto data = dataset;

        for (size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            if (cfg.shuffleEachEpoch) shuffleData(data);
            for (auto& [input, actual] : data) trainStep(input, actual);
            if (cfg.logInterval > 0 && epoch % cfg.logInterval == 0) {}
        }
    }

    
    template <typename VecT, typename ActualT> void trainStep(
        const VecT& input,
        const ActualT& actual
    ) { trainStepRecurse<0>(input, actual); }


    /*
        saveWeights(path):
        - writes a small header (layer count + each layer size, for a sanity
          check on load) followed by every layer's weights then bias values,
          in order, as whitespace-separated plain text. Returns false if the
          file couldn't be opened.
    */
    bool saveWeights(const std::string& path) const {
        std::ofstream out(path);
        if (!out) return false;

        out << sizeof...(LayerSizes) << '\n';
        ((out << LayerSizes << ' '), ...);
        out << '\n';

        std::apply([&](const auto&... layer) {
            (saveLayer(layer, out), ...);
        }, layers);

        return static_cast<bool>(out);
    }

    /*
        loadWeights(path):
        - reads back what saveWeights wrote, validating the header against
          this network's actual LayerSizes... before touching any layer.
          Returns false (leaving layers untouched) if the file couldn't be
          opened or the header doesn't match.
    */
    bool loadWeights(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;

        size_t count = 0;
        in >> count;
        if (count != sizeof...(LayerSizes)) return false;

        for (size_t expected : { LayerSizes... }) {
            size_t got = 0;
            in >> got;
            if (!in || got != expected) return false;
        }

        std::apply([&](auto&... layer) {
            (loadLayer(layer, in), ...);
        }, layers);

        return static_cast<bool>(in);
    }


private:
    template <size_t InN, size_t OutN> void randomizeLayer(
        Layer<InN, OutN, T>& layer, 
        std::mt19937& rng, 
        T mean, 
        T stddev
    ) {
        layer.weights   = randomMat<OutN, InN>(rng, mean, stddev);
        layer.bias      = randomVec<OutN, T>(rng, mean, stddev);
    }


    template <size_t N> inline Vec<N, T> activationDispatch(
        const Vec<N, T>& x,
        const Activation act
    ) const {
        switch (act) {
            case Activation::Sigmoid:   return sigmoid(x);
            case Activation::Tanh:      return tanh(x);
            case Activation::Relu:      return relu(x);
            case Activation::LeakyRelu: return leakyRelu(x, cfg.leakyReluAlpha);
            default: break;
        }

        return {};
    }

    // same shape as activationDispatch, but for each activation's derivative
    template <size_t N> inline Vec<N, T> activationPrimeDispatch(
        const Vec<N, T>& x,
        const Activation act
    ) const {
        switch (act) {
            case Activation::Sigmoid:   return sigmoidPrime(x);
            case Activation::Tanh:      return tanhPrime(x);
            case Activation::Relu:      return reluPrime(x);
            case Activation::LeakyRelu: return leakyReluPrime(x, cfg.leakyReluAlpha); // add leakyReluPrime to NNUtils.hpp (same shape as reluPrime) before using LeakyRelu in training
            default: break;
        }

        return {};
    }

    // dispatches to the configured loss function's derivative w.r.t. predicted
    template <size_t N> inline Vec<N, T> lossPrimeDispatch(
        const Vec<N, T>& predicted,
        const Vec<N, T>& actual
    ) const {
        switch (cfg.loss) {
            case Loss::MeanSquaredError: return meanSquaredErrorPrime(predicted, actual);
            case Loss::CrossEntropy:     return crossEntropyPrime(predicted, actual);
            default: break;
        }

        return {};
    }

    template <size_t Index, typename VecT> auto forwardRecurse(
        const VecT& a
    ) const {
        constexpr bool isLast = (Index + 1 == std::tuple_size_v<decltype(layers)>);

        auto& layer = std::get<Index>(layers);
        auto z = layer.forward(a);

        Activation act = isLast ? cfg.outputActivation : cfg.hiddenActivation;
        auto out = activationDispatch(z, act);

        if constexpr (isLast) return out;
        else return forwardRecurse<Index + 1>(out);
    }

    /*
        trainStepRecurse<Index>(prevActivation, actual):
        - recurses forward through layer Index (and everything after it),
          then computes and applies that layer's gradients on the way back
          up the call stack. z/a for this layer live as local variables in
          this stack frame, so no separate cache of every layer's z/a is
          needed - the recursion itself is the cache. Returns dL/d(prevActivation),
          the gradient the caller (the previous layer, or nothing if Index==0)
          needs to continue backprop.
    */
    template <size_t Index, typename VecT, typename ActualT> auto trainStepRecurse(
        const VecT& prevActivation,
        const ActualT& actual
    ) {
        auto& layer = std::get<Index>(layers);
        auto z = layer.forward(prevActivation);

        constexpr bool isLast = (Index + 1 == std::tuple_size_v<decltype(layers)>);
        Activation act = isLast ? cfg.outputActivation : cfg.hiddenActivation;
        auto a = activationDispatch(z, act);

        // shared by both branches below: given dL/da for THIS layer's output,
        // compute this layer's gradients, apply the update, and return dL/dx
        auto backpropStep = [&](const auto& dLda) {
            auto dLdz = hadamard(dLda, activationPrimeDispatch(z, act));

            decltype(layer.weights) weightGradient{};
            decltype(layer.bias) biasGradient{};
            auto dLdx = layer.backward(prevActivation, dLdz, weightGradient, biasGradient);

            layer.weights = layer.weights - weightGradient * cfg.learningRate;
            layer.bias   -= cfg.learningRate * biasGradient;

            return dLdx;
        };

        if constexpr (isLast) return backpropStep(lossPrimeDispatch(a, actual));
        else return backpropStep(trainStepRecurse<Index + 1>(a, actual));
    }


    template <typename VecIn, typename VecActual> void shuffleData(
        std::vector<std::pair<VecIn, VecActual>>& data
    ) { std::shuffle(data.begin(), data.end(), rng); }


    template <size_t InN, size_t OutN> void saveLayer(
        const Layer<InN, OutN, T>& layer,
        std::ofstream& out
    ) const {
        for (size_t i = 0; i < OutN; ++i)
            for (size_t j = 0; j < InN; ++j)
                out << layer.weights[i][j] << ' ';

        for (size_t i = 0; i < OutN; ++i)
            out << layer.bias[i] << ' ';

        out << '\n';
    }

    template <size_t InN, size_t OutN> void loadLayer(
        Layer<InN, OutN, T>& layer,
        std::ifstream& in
    ) {
        for (size_t i = 0; i < OutN; ++i)
            for (size_t j = 0; j < InN; ++j)
                in >> layer.weights[i][j];

        for (size_t i = 0; i < OutN; ++i)
            in >> layer.bias[i];
    }

};

} // !namespace math
