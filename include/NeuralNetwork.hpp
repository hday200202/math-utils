#pragma once

#include <tuple>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>

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

template <typename T, size_t First, size_t Second, size_t... Rest>
struct LayerTuple {
    using type = decltype(std::tuple_cat(
        std::tuple<Layer<First, Second, T>>{},
        typename LayerTuple<T, Second, Rest...>::type{}
    ));
};

template <typename T, size_t First, size_t Second>
struct LayerTuple<T, First, Second> { using type = std::tuple<Layer<First, Second, T>>; };

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


    /*
        Gradients:
        - one weight matrix and one bias vector per layer, which is exactly the
          shape of the layer tuple itself, so the accumulator reuses that type
    */
    using Gradients = LayerTupleT<T, LayerSizes...>;


    template <typename VecIn, typename VecActual> void train(
        const std::vector<std::pair<VecIn, VecActual>>& dataset
    ) {
        auto data = dataset;

        const size_t batch = (cfg.batchSize == 0) ? 1 : cfg.batchSize;

        // allocated once and reused for every batch
        Gradients grads;

        for (size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            if (cfg.shuffleEachEpoch) shuffleData(data);

            for (size_t start = 0; start < data.size(); start += batch) {
                const size_t stop = std::min(start + batch, data.size());

                zeroGradients(grads);
                for (size_t i = start; i < stop; ++i)
                    accumulate(data[i].first, data[i].second, grads);

                // the mean gradient over the batch, not the sum, so that the
                // effective learning rate does not scale with batchSize. The
                // last batch of an epoch may be short, hence stop - start
                applyGradients(grads, cfg.learningRate / T(stop - start));
            }

            if (cfg.logInterval > 0 && epoch % cfg.logInterval == 0)
                std::cout << "epoch " << epoch << "  loss " << averageLoss(dataset) << '\n';
        }
    }


    /*
        accumulate(input, actual, grads):
        - runs one sample forward and back and adds its gradient into grads,
          without touching the weights.
        - const on purpose: every sample in a batch reads the same frozen
          weights, which is exactly the property that makes the batch loop
          safe to split across threads later
    */
    template <typename VecT, typename ActualT> void accumulate(
        const VecT& input,
        const ActualT& actual,
        Gradients& grads
    ) const { accumulateRecurse<0>(input, actual, grads); }


    /*
        applyGradients(grads, scale):
        - the single weight update at the end of a batch. Pass
          learningRate / batchSize to descend on the mean gradient
    */
    void applyGradients(const Gradients& grads, T scale) {
        applyGradientsImpl(grads, scale, std::make_index_sequence<std::tuple_size_v<Gradients>>{});
    }


    /*
        trainStep(input, actual):
        - one sample, applied immediately. Identical to a batch of one
    */
    template <typename VecT, typename ActualT> void trainStep(
        const VecT& input,
        const ActualT& actual
    ) {
        Gradients grads;
        zeroGradients(grads);
        accumulate(input, actual, grads);
        applyGradients(grads, cfg.learningRate);
    }


    /*
        averageLoss(dataset):
        - mean loss across the dataset, for the logInterval readout
    */
    template <typename VecIn, typename VecActual> T averageLoss(
        const std::vector<std::pair<VecIn, VecActual>>& dataset
    ) const {
        if (dataset.empty()) return T(0);

        T total = T(0);
        for (const auto& [input, actual] : dataset)
            total += lossDispatch(forward(input), actual);

        return total / T(dataset.size());
    }


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


    template <size_t N> inline Vec<N, T> activationPrimeDispatch(
        const Vec<N, T>& x,
        const Activation act
    ) const {
        switch (act) {
            case Activation::Sigmoid:   return sigmoidPrime(x);
            case Activation::Tanh:      return tanhPrime(x);
            case Activation::Relu:      return reluPrime(x);
            case Activation::LeakyRelu: return leakyReluPrime(x, cfg.leakyReluAlpha);
            default: break;
        }

        return {};
    }


    template <size_t N> inline T lossDispatch(
        const Vec<N, T>& predicted,
        const Vec<N, T>& actual
    ) const {
        switch (cfg.loss) {
            case Loss::MeanSquaredError: return meanSquaredError(predicted, actual);
            case Loss::CrossEntropy:     return crossEntropy(predicted, actual);
            default: break;
        }

        return T(0);
    }


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


    void zeroGradients(Gradients& grads) const {
        std::apply([](auto&... g) {
            ((g.weights = {}, g.bias = {}), ...);
        }, grads);
    }


    template <size_t... Is> void applyGradientsImpl(
        const Gradients& grads,
        T scale,
        std::index_sequence<Is...>
    ) {
        ((std::get<Is>(layers).weights = std::get<Is>(layers).weights - std::get<Is>(grads).weights * scale,
          std::get<Is>(layers).bias   -= std::get<Is>(grads).bias * scale), ...);
    }


    template <size_t Index, typename VecT, typename ActualT> auto accumulateRecurse(
        const VecT& prevActivation,
        const ActualT& actual,
        Gradients& grads
    ) const {
        const auto& layer = std::get<Index>(layers);
        auto z = layer.forward(prevActivation);

        constexpr bool isLast = (Index + 1 == std::tuple_size_v<decltype(layers)>);
        Activation act = isLast ? cfg.outputActivation : cfg.hiddenActivation;
        auto a = activationDispatch(z, act);

        auto backpropStep = [&](const auto& dLda) {
            auto dLdz = hadamard(dLda, activationPrimeDispatch(z, act));

            decltype(layer.weights) weightGradient{};
            decltype(layer.bias) biasGradient{};
            auto dLdx = layer.backward(prevActivation, dLdz, weightGradient, biasGradient);

            auto& g = std::get<Index>(grads);
            g.weights = g.weights + weightGradient;
            g.bias    = g.bias + biasGradient;

            return dLdx;
        };

        if constexpr (isLast) return backpropStep(lossPrimeDispatch(a, actual));
        else return backpropStep(accumulateRecurse<Index + 1>(a, actual, grads));
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

/* ========================================================================= //
    Dynamically allocated counterparts (math::dyn)
// ========================================================================= */

namespace math::dyn {

/* ========================================================================= //
    Config
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

/*
    Layer<T>:
    - the runtime-sized counterpart to math::Layer<InN, OutN, T>. The shapes
      that used to live in the type now live in the weight matrix: weights is
      outN x inN, exactly as before, so inSize()/outSize() just read it back
*/
template <typename T> struct Layer {
    Mat<T> weights;
    Vec<T> bias;

    Layer() = default;
    Layer(size_t inN, size_t outN) : weights(outN, inN), bias(outN) {}

    size_t inSize() const { return weights.numCols(); }
    size_t outSize() const { return weights.numRows(); }

    Vec<T> forward(const Vec<T>& input) const {
        return weights * input + bias;
    }

    Vec<T> backward(
        const Vec<T>& input,
        const Vec<T>& outputGradient,
        Mat<T>& weightGradient,
        Vec<T>& biasGradient
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
    Gradients<T>:
    - one weight matrix and one bias vector per layer, matching the network's
      shape. Batch training sums every sample's gradient in here against a
      single frozen copy of the weights, and applies the total once at the end
      of the batch rather than after every sample
*/
template <typename T> struct Gradients {
    std::vector<Mat<T>> weights;
    std::vector<Vec<T>> bias;

    void reset(const std::vector<Layer<T>>& layers) {
        weights.clear();
        bias.clear();
        weights.reserve(layers.size());
        bias.reserve(layers.size());

        for (const auto& layer : layers) {
            weights.emplace_back(layer.outSize(), layer.inSize());
            bias.emplace_back(layer.outSize());
        }
    }

    void zero() {
        for (auto& w : weights)
            for (size_t i = 0; i < w.numRows(); ++i)
                for (size_t j = 0; j < w.numCols(); ++j)
                    w[i][j] = T(0);

        for (auto& b : bias)
            for (size_t i = 0; i < b.size(); ++i)
                b[i] = T(0);
    }
};



/*
    NeuralNetwork<T>:
    - the runtime-sized counterpart to math::NeuralNetwork<T, LayerSizes...>.
      The layer sizes are a std::vector rather than a template parameter pack,
      so the tuple and the index recursion both collapse into plain loops.
    - the one real consequence: the static version cached each layer's z and
      activation implicitly, in the recursive call frames, and got the backward
      pass for free as the stack unwound. A loop has no such stack, so trainStep
      caches them explicitly and then walks the layers in reverse
*/
template <typename T> struct NeuralNetwork {
    std::vector<Layer<T>> layers;
    std::vector<size_t> sizes;
    NNConfig<T> cfg = {};
    std::mt19937 rng;


    NeuralNetwork() = default;
    explicit NeuralNetwork(const std::vector<size_t>& layerSizes) : sizes(layerSizes) {}


    void init(const std::vector<size_t>& layerSizes, const NNConfig<T>& cfg) {
        this->sizes = layerSizes;
        init(cfg);
    }


    void init(const NNConfig<T>& cfg) {
        assert(sizes.size() >= 2 && "NeuralNetwork needs at least an input and output size");

        this->cfg = cfg;
        this->rng.seed(this->cfg.seed);

        buildLayers();

        for (auto& layer : layers)
            randomizeLayer(layer, this->rng, this->cfg.initMean, this->cfg.initStddev);
    }


    Vec<T> forward(const Vec<T>& input) const {
        assert(!layers.empty() && "forward: network has no layers, call init() first");
        assert(input.size() == layers.front().inSize() && "forward: input size mismatch");

        Vec<T> a = input;

        for (size_t i = 0; i < layers.size(); ++i) {
            Vec<T> z = layers[i].forward(a);
            a = activationDispatch(z, activationFor(i));
        }

        return a;
    }


    void train(const std::vector<std::pair<Vec<T>, Vec<T>>>& dataset) {
        auto data = dataset;

        const size_t batch = (cfg.batchSize == 0) ? 1 : cfg.batchSize;

        // allocated once and reused for every batch, rather than per step
        Gradients<T> grads;
        grads.reset(layers);

        for (size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            if (cfg.shuffleEachEpoch) shuffleData(data);

            for (size_t start = 0; start < data.size(); start += batch) {
                const size_t stop = std::min(start + batch, data.size());

                grads.zero();
                for (size_t i = start; i < stop; ++i)
                    accumulate(data[i].first, data[i].second, grads);

                // the mean gradient over the batch, not the sum, so that the
                // effective learning rate does not scale with batchSize. The
                // last batch of an epoch may be short, hence stop - start
                applyGradients(grads, cfg.learningRate / T(stop - start));
            }

            if (cfg.logInterval > 0 && epoch % cfg.logInterval == 0)
                std::cout << "epoch " << epoch << "  loss " << averageLoss(dataset) << '\n';
        }
    }


    /*
        accumulate(input, actual, grads):
        - runs one sample forward and back and adds its gradient into grads,
          without touching the weights.
        - const on purpose: every sample in a batch reads the same frozen
          weights, which is exactly the property that makes the batch loop
          safe to split across threads later
    */
    void accumulate(const Vec<T>& input, const Vec<T>& actual, Gradients<T>& grads) const {
        const size_t L = layers.size();
        assert(L > 0 && "accumulate: network has no layers, call init() first");

        // forward pass, caching what the backward pass will need
        std::vector<Vec<T>> preActivations(L);
        std::vector<Vec<T>> activations(L + 1);
        activations[0] = input;

        for (size_t i = 0; i < L; ++i) {
            preActivations[i] = layers[i].forward(activations[i]);
            activations[i + 1] = activationDispatch(preActivations[i], activationFor(i));
        }

        // backward pass, summing into grads instead of applying
        Vec<T> dLda = lossPrimeDispatch(activations[L], actual);

        for (size_t i = L; i-- > 0; ) {
            Vec<T> dLdz = hadamard(dLda, activationPrimeDispatch(preActivations[i], activationFor(i)));

            Mat<T> weightGradient;
            Vec<T> biasGradient;
            Vec<T> dLdx = layers[i].backward(activations[i], dLdz, weightGradient, biasGradient);

            grads.weights[i] = grads.weights[i] + weightGradient;
            grads.bias[i]   += biasGradient;

            dLda = dLdx;
        }
    }


    /*
        applyGradients(grads, scale):
        - the single weight update at the end of a batch. Pass
          learningRate / batchSize to descend on the mean gradient
    */
    void applyGradients(const Gradients<T>& grads, T scale) {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].weights = layers[i].weights - grads.weights[i] * scale;
            layers[i].bias   -= scale * grads.bias[i];
        }
    }


    /*
        trainStep(input, actual):
        - one sample, applied immediately. Identical to a batch of one, kept
          because it reads better at a call site than building a Gradients
    */
    void trainStep(const Vec<T>& input, const Vec<T>& actual) {
        Gradients<T> grads;
        grads.reset(layers);
        accumulate(input, actual, grads);
        applyGradients(grads, cfg.learningRate);
    }


    T averageLoss(const std::vector<std::pair<Vec<T>, Vec<T>>>& dataset) const {
        if (dataset.empty()) return T(0);

        T total = T(0);
        for (const auto& [input, actual] : dataset)
            total += lossDispatch(forward(input), actual);

        return total / T(dataset.size());
    }


    bool saveWeights(const std::string& path) const {
        std::ofstream out(path);
        if (!out) return false;

        // default ostream precision is 6 significant digits, which is not
        // enough to round-trip a float (needs max_digits10, 9 for float)
        out << std::setprecision(std::numeric_limits<T>::max_digits10);

        out << sizes.size() << '\n';
        for (size_t s : sizes) out << s << ' ';
        out << '\n';

        for (const auto& layer : layers) saveLayer(layer, out);

        return static_cast<bool>(out);
    }


    bool loadWeights(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;

        size_t count = 0;
        in >> count;
        if (!in || count < 2) return false;

        std::vector<size_t> fileSizes(count);
        for (size_t i = 0; i < count; ++i) {
            in >> fileSizes[i];
            if (!in || fileSizes[i] == 0) return false;
        }

        // rebuild to match the file rather than requiring the caller to have
        // already guessed the topology, which is the point of being runtime-sized
        sizes = fileSizes;
        buildLayers();

        for (auto& layer : layers) loadLayer(layer, in);

        return static_cast<bool>(in);
    }


private:
    void buildLayers() {
        layers.clear();
        layers.reserve(sizes.size() - 1);

        for (size_t i = 0; i + 1 < sizes.size(); ++i)
            layers.emplace_back(sizes[i], sizes[i + 1]);
    }


    Activation activationFor(size_t index) const {
        return (index + 1 == layers.size()) ? cfg.outputActivation : cfg.hiddenActivation;
    }


    void randomizeLayer(
        Layer<T>& layer,
        std::mt19937& rng,
        T mean,
        T stddev
    ) {
        layer.weights   = randomMat<T>(rng, layer.outSize(), layer.inSize(), mean, stddev);
        layer.bias      = randomVec<T>(rng, layer.outSize(), mean, stddev);
    }


    inline Vec<T> activationDispatch(
        const Vec<T>& x,
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


    inline Vec<T> activationPrimeDispatch(
        const Vec<T>& x,
        const Activation act
    ) const {
        switch (act) {
            case Activation::Sigmoid:   return sigmoidPrime(x);
            case Activation::Tanh:      return tanhPrime(x);
            case Activation::Relu:      return reluPrime(x);
            case Activation::LeakyRelu: return leakyReluPrime(x, cfg.leakyReluAlpha);
            default: break;
        }

        return {};
    }


    inline T lossDispatch(
        const Vec<T>& predicted,
        const Vec<T>& actual
    ) const {
        switch (cfg.loss) {
            case Loss::MeanSquaredError: return meanSquaredError(predicted, actual);
            case Loss::CrossEntropy:     return crossEntropy(predicted, actual);
            default: break;
        }

        return T(0);
    }


    inline Vec<T> lossPrimeDispatch(
        const Vec<T>& predicted,
        const Vec<T>& actual
    ) const {
        switch (cfg.loss) {
            case Loss::MeanSquaredError: return meanSquaredErrorPrime(predicted, actual);
            case Loss::CrossEntropy:     return crossEntropyPrime(predicted, actual);
            default: break;
        }

        return {};
    }


    void shuffleData(
        std::vector<std::pair<Vec<T>, Vec<T>>>& data
    ) { std::shuffle(data.begin(), data.end(), rng); }


    void saveLayer(
        const Layer<T>& layer,
        std::ofstream& out
    ) const {
        for (size_t i = 0; i < layer.outSize(); ++i)
            for (size_t j = 0; j < layer.inSize(); ++j)
                out << layer.weights[i][j] << ' ';

        for (size_t i = 0; i < layer.outSize(); ++i)
            out << layer.bias[i] << ' ';

        out << '\n';
    }

    void loadLayer(
        Layer<T>& layer,
        std::ifstream& in
    ) {
        for (size_t i = 0; i < layer.outSize(); ++i)
            for (size_t j = 0; j < layer.inSize(); ++j)
                in >> layer.weights[i][j];

        for (size_t i = 0; i < layer.outSize(); ++i)
            in >> layer.bias[i];
    }

};

} // !namespace math::dyn
