#pragma once

#include <vector>
#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>

#include "../math/Math.hpp"
#include "../files/File.hpp"

/*
    Everything from outside this namespace is spelled out fully qualified -
    math::Vec<T>/math::Mat<T> for the types, math::random:: for the weight
    initialisers. A `using namespace` would read more cleanly at the call
    sites, but in a header it applies to every translation unit that
    includes this one, so the names it pulls in are not this file's to
    decide. sigmoid/hadamard/etc. need no qualification: ai/Math.hpp puts
    them in this same ai namespace.
*/

namespace ai {

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
    - a single fully-connected layer: weights is outN x inN, bias is length
      outN, so inSize()/outSize() just read the shape back out of weights
      rather than storing it separately
*/
template <typename T> struct Layer {
    math::Mat<T> weights;
    math::Vec<T> bias;

    Layer() = default;
    Layer(size_t inN, size_t outN) : weights(outN, inN), bias(outN) {}

    size_t inSize() const { return weights.numCols(); }
    size_t outSize() const { return weights.numRows(); }

    math::Vec<T> forward(const math::Vec<T>& input) const {
        return weights * input + bias;
    }

    math::Vec<T> backward(
        const math::Vec<T>& input,
        const math::Vec<T>& outputGradient,
        math::Mat<T>& weightGradient,
        math::Vec<T>& biasGradient
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
    std::vector<math::Mat<T>> weights;
    std::vector<math::Vec<T>> bias;

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
    - layer sizes are a runtime std::vector, not a template parameter pack,
      so the network is built and walked with plain loops rather than
      recursion over a tuple.
    - trainStep/accumulate cache each layer's z and activation explicitly
      (preActivations/activations below) rather than relying on recursive
      call frames the way a compile-time-sized version could, since a loop
      has no call stack to reuse for that
*/
template <typename T> struct NeuralNetwork {
    std::vector<Layer<T>> layers;
    std::vector<size_t> sizes;
    NNConfig<T> cfg = {};
    math::random::Random rng{0};   // reseeded from cfg.seed in init()


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


    math::Vec<T> forward(const math::Vec<T>& input) const {
        assert(!layers.empty() && "forward: network has no layers, call init() first");
        assert(input.size() == layers.front().inSize() && "forward: input size mismatch");

        math::Vec<T> a = input;

        for (size_t i = 0; i < layers.size(); ++i) {
            math::Vec<T> z = layers[i].forward(a);
            a = activationDispatch(z, activationFor(i));
        }

        return a;
    }


    void train(const std::vector<std::pair<math::Vec<T>, math::Vec<T>>>& dataset) {
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
    void accumulate(const math::Vec<T>& input, const math::Vec<T>& actual, Gradients<T>& grads) const {
        const size_t L = layers.size();
        assert(L > 0 && "accumulate: network has no layers, call init() first");

        // forward pass, caching what the backward pass will need
        std::vector<math::Vec<T>> preActivations(L);
        std::vector<math::Vec<T>> activations(L + 1);
        activations[0] = input;

        for (size_t i = 0; i < L; ++i) {
            preActivations[i] = layers[i].forward(activations[i]);
            activations[i + 1] = activationDispatch(preActivations[i], activationFor(i));
        }

        // backward pass, summing into grads instead of applying
        math::Vec<T> dLda = lossPrimeDispatch(activations[L], actual);

        for (size_t i = L; i-- > 0; ) {
            math::Vec<T> dLdz = hadamard(dLda, activationPrimeDispatch(preActivations[i], activationFor(i)));

            math::Mat<T> weightGradient;
            math::Vec<T> biasGradient;
            math::Vec<T> dLdx = layers[i].backward(activations[i], dLdz, weightGradient, biasGradient);

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
    void trainStep(const math::Vec<T>& input, const math::Vec<T>& actual) {
        Gradients<T> grads;
        grads.reset(layers);
        accumulate(input, actual, grads);
        applyGradients(grads, cfg.learningRate);
    }


    T averageLoss(const std::vector<std::pair<math::Vec<T>, math::Vec<T>>>& dataset) const {
        if (dataset.empty()) return T(0);

        T total = T(0);
        for (const auto& [input, actual] : dataset)
            total += lossDispatch(forward(input), actual);

        return total / T(dataset.size());
    }


    /*
        saveWeights(path):
        - the whole file is built in memory first and handed to files::File in
          one write. File deals in whole strings rather than a stream, so
          there is nothing to format into incrementally
        - OpenMode::Write creates the file if it is missing and truncates it
          if it is not, which is what overwriting a set of weights should do
    */
    bool saveWeights(const std::filesystem::path& path) const {
        std::ostringstream out;

        // default ostream precision is 6 significant digits, which is not
        // enough to round-trip a float (needs max_digits10, 9 for float)
        out << std::setprecision(std::numeric_limits<T>::max_digits10);

        out << sizes.size() << '\n';
        for (size_t s : sizes) out << s << ' ';
        out << '\n';

        for (const auto& layer : layers) saveLayer(layer, out);

        files::File file;
        if (!file.open(path, files::OpenMode::Write)) return false;

        return file.write(out.str());
    }


    /*
        loadWeights(path):
        - reads the file in one go and parses out of the string. The parsing
          itself is unchanged: >> on a stringstream extracts whitespace
          separated numbers exactly as it did on the ifstream
    */
    bool loadWeights(const std::filesystem::path& path) {
        files::File file;
        if (!file.open(path, files::OpenMode::Read)) return false;

        std::istringstream in(file.text());

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
        math::random::Random& rng,
        T mean,
        T stddev
    ) {
        layer.weights   = math::random::randomMat<T>(rng, layer.outSize(), layer.inSize(), mean, stddev);
        layer.bias      = math::random::randomVec<T>(rng, layer.outSize(), mean, stddev);
    }


    inline math::Vec<T> activationDispatch(
        const math::Vec<T>& x,
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


    inline math::Vec<T> activationPrimeDispatch(
        const math::Vec<T>& x,
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
        const math::Vec<T>& predicted,
        const math::Vec<T>& actual
    ) const {
        switch (cfg.loss) {
            case Loss::MeanSquaredError: return meanSquaredError(predicted, actual);
            case Loss::CrossEntropy:     return crossEntropy(predicted, actual);
            default: break;
        }

        return T(0);
    }


    inline math::Vec<T> lossPrimeDispatch(
        const math::Vec<T>& predicted,
        const math::Vec<T>& actual
    ) const {
        switch (cfg.loss) {
            case Loss::MeanSquaredError: return meanSquaredErrorPrime(predicted, actual);
            case Loss::CrossEntropy:     return crossEntropyPrime(predicted, actual);
            default: break;
        }

        return {};
    }


    void shuffleData(
        std::vector<std::pair<math::Vec<T>, math::Vec<T>>>& data
    ) { std::shuffle(data.begin(), data.end(), rng); }


    void saveLayer(
        const Layer<T>& layer,
        std::ostringstream& out
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
        std::istringstream& in
    ) {
        for (size_t i = 0; i < layer.outSize(); ++i)
            for (size_t j = 0; j < layer.inSize(); ++j)
                in >> layer.weights[i][j];

        for (size_t i = 0; i < layer.outSize(); ++i)
            in >> layer.bias[i];
    }

};

} // !namespace ai
