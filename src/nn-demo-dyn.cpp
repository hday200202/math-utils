#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "../include/PlatformUtils.hpp"

#include "../include/NeuralNetwork.hpp"
#include "../include/Printing.hpp"

using Vecf = math::dyn::Vec<float>;
using Data = std::vector<std::pair<Vecf, Vecf>>;


static const Data XOR_DATA = {
    {{0.f, 0.f}, {0.f}},
    {{0.f, 1.f}, {1.f}},
    {{1.f, 0.f}, {1.f}},
    {{1.f, 1.f}, {0.f}}
};


static void printOutputs(const math::dyn::NeuralNetwork<float>& nn) {
    for (const auto& [input, actual] : XOR_DATA)
        std::cout << "    " << input << " -> " << nn.forward(input)[0]
                  << "   (want " << actual[0] << ")\n";
}


int main() {
    std::cout << std::fixed << std::setprecision(4);


    /* set up ================================================================ */
    math::dyn::NNConfig<float> cfg;
    cfg.hiddenActivation = math::dyn::Activation::Tanh;
    cfg.outputActivation = math::dyn::Activation::Sigmoid;
    cfg.loss             = math::dyn::Loss::MeanSquaredError;
    cfg.epochs           = 20000;
    cfg.batchSize        = 32;
    cfg.learningRate     = 0.1f;
    cfg.initStddev       = 0.5f;
    cfg.logInterval      = 4000;

    // the topology is a runtime value, not a template parameter
    math::dyn::NeuralNetwork<float> nn({ 2, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 });
    nn.init(cfg);


    /* train ================================================================= */
    std::cout << "training:\n";
    nn.train(XOR_DATA);

    std::cout << "\ntrained:\n";
    printOutputs(nn);


    /* save ================================================================== */
    const std::string path = (platform::exeDir() / "xor-weights-dyn.csv").string();
    nn.saveWeights(path);


    /* run from pre-trained weights ========================================== */
    // loadWeights reads the topology out of the file, so this network is never
    // told its own shape
    math::dyn::NeuralNetwork<float> pretrained;
    pretrained.cfg = cfg;
    pretrained.loadWeights(path);

    std::cout << "\nfrom saved weights:\n";
    printOutputs(pretrained);


    return 0;
}
