#include "../include/ai/AI.hpp"

int main() {
    using namespace ai;
    
    NeuralNetwork<float> nn;
    nn.init({3, 9, 1}, NNConfig<float>{
        .hiddenActivation = Activation::Sigmoid, 
        .outputActivation = Activation::Sigmoid, 
        .epochs = 10000
    });

    std::vector<std::pair<math::Vec<float>, math::Vec<float>>> data = {
        {{0, 0, 0}, {0}},
        {{0, 0, 1}, {1}},
        {{0, 1, 0}, {1}},
        {{0, 1, 1}, {0}},

        {{1, 0, 0}, {1}},
        {{1, 0, 1}, {0}},
        {{1, 1, 0}, {0}},
        {{1, 1, 1}, {0}},
    };

    nn.train(data);

    for (float i = 0.0; i < 2.0; i += 1.0)
        for (float j = 0.0; j < 2.0; j += 1.0)
            for (float k = 0.0; k < 2.0; k += 1.0)
                std::cout << math::Vec3{i, j, k} << " : " << (int)std::round(nn.forward({i, j, k})[0]) << std::endl;
}