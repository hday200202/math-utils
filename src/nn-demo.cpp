#include <iostream>
#include <vector>
#include <iomanip>

#include "../include/PlatformUtils.hpp"

#include "../include/NeuralNetwork.hpp"
#include "../include/Printing.hpp"

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

#define TRAIN


#ifdef TRAIN

int main(int argc, char* argv[]) {
    // Classic XOR: not linearly separable, so a working hidden layer is required
    // to solve it - a good sanity check that backprop is actually correct.
    math::NeuralNetwork<float, 2, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1> nn;

    math::NNConfig<float> cfg;
    cfg.seed                = 42;
    cfg.learningRate        = 0.5f;
    cfg.epochs              = 1000;
    cfg.hiddenActivation    = math::Activation::Tanh;
    cfg.outputActivation    = math::Activation::Sigmoid;
    cfg.loss                = math::Loss::MeanSquaredError;
    cfg.shuffleEachEpoch    = true;

    nn.init(cfg);

    std::vector<std::pair<math::Vec<2, float>, math::Vec<1, float>>> dataset = {
        {{0.f, 0.f}, {0.f}},
        {{0.f, 1.f}, {1.f}},
        {{1.f, 0.f}, {1.f}},
        {{1.f, 1.f}, {0.f}}
    };

    auto averageLoss = [&]() {
        float total = 0.f;
        for (auto& [input, actual] : dataset)
            total += math::meanSquaredError(nn.forward(input), actual);
        return total / dataset.size();
    };

    std::cout << "Training a " << "2-8-1" << " network on XOR for " << cfg.epochs << " epochs...\n\n";

    // cfg.epochs is the TOTAL epoch count - split across checkpoints for
    // progress printouts, rather than running cfg.epochs once per checkpoint
    size_t totalEpochs = cfg.epochs;
    size_t numCheckpoints = std::min<size_t>(20, totalEpochs);
    size_t epochsPerCheckpoint = totalEpochs / numCheckpoints;
    nn.cfg.epochs = epochsPerCheckpoint;

    for (size_t i = 0; i < numCheckpoints; ++i) {
        nn.train(dataset);
        std::cout << "  epoch " << std::setw(5) << (i + 1) * epochsPerCheckpoint
                  << "  |  avg loss: " << std::fixed << std::setprecision(6) << averageLoss() << "\n";
    }

    std::cout << "\nFinal predictions:\n" << std::fixed << std::setprecision(4);
    bool allCorrect = true;
    for (auto& [input, actual] : dataset) {
        auto predicted = nn.forward(input);
        bool correct = (predicted[0] >= 0.5f) == (actual[0] >= 0.5f);
        allCorrect &= correct;

        std::cout << "  " << input << "  ->  predicted: " << predicted[0]
                  << "  (actual: " << actual[0] << ")  "
                  << (correct ? COLOR_GREEN : COLOR_RED) << (correct ? "[PASS]" : "[FAIL]") << COLOR_RESET << "\n";
    }

    std::cout << "\n" << (allCorrect ? COLOR_GREEN : COLOR_RED)
               << (allCorrect ? "All predictions correct!" : "Some predictions incorrect.")
               << COLOR_RESET << std::endl;

    std::cout << "Saving weights to \"" << platform::exeDir().string() + "/../../data/xor-weights.csv\"" << std::endl;
    nn.saveWeights(platform::exeDir().string() + "/../../data/xor-weights.csv");

    return 0;
}

#else

int main() {
    math::NeuralNetwork<float, 2, 8, 1> nn;
    nn.init({});

    nn.loadWeights(platform::exeDir().string() + "/../../data/xor-weights.csv");

    auto out = nn.forward(math::Vec<2, float>{0.f, 0.f});
    std::cout << out << std::endl;
    out = nn.forward(math::Vec<2, float>{0.f, 1.f});
    std::cout << out << std::endl;
    out = nn.forward(math::Vec<2, float>{1.f, 0.f});
    std::cout << out << std::endl;
    out = nn.forward(math::Vec<2, float>{1.f, 1.f});
    std::cout << out << std::endl;
}

#endif