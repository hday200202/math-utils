#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../include/ai/GPT2.hpp"

/*
    GPT-2 text generation, CPU, from scratch.

    Nothing here is linked against a machine learning framework. The whole
    stack is this repository: files::json parses the config and the weight
    header, files::File reads the bytes, ai::Tensor holds them, ai::Tokenizer
    turns text into token ids, ai::GPT2 runs the forward pass, and
    math::random::Random drives the sampling.

        make release-gpt2demo
        ./bin/release/gpt2 --prompt "The capital of France is" --tokens 40

    Options:
        --model <dir>     default models/gpt2
        --prompt <text>
        --tokens <n>      how many to generate
        --temp <f>        0 is greedy, higher is more random
        --topk <n>        how many candidates to sample from
        --seed <n>
*/

static void usage() {
    std::cout
        << "usage: gpt2 [--model DIR] [--prompt TEXT] [--tokens N]\n"
        << "            [--temp F] [--topk N] [--seed N]\n";
}

int main(int argc, char** argv) {
    std::string modelDir = "models/gpt2";
    std::string prompt   = "The capital of France is";
    size_t tokens        = 40;
    float  temperature   = 0.8f;
    size_t topK          = 40;
    uint64_t seed        = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasNext = (i + 1 < argc);

        if (arg == "--help" || arg == "-h")        { usage(); return 0; }
        else if (arg == "--model"  && hasNext)     { modelDir = argv[++i]; }
        else if (arg == "--prompt" && hasNext)     { prompt = argv[++i]; }
        else if (arg == "--tokens" && hasNext)     { tokens = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--temp"   && hasNext)     { temperature = std::strtof(argv[++i], nullptr); }
        else if (arg == "--topk"   && hasNext)     { topK = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--seed"   && hasNext)     { seed = std::strtoull(argv[++i], nullptr, 10); }
        else { std::cout << "unknown option: " << arg << "\n"; usage(); return 1; }
    }

    using clock = std::chrono::steady_clock;

    ai::GPT2 model;
    std::cout << "loading " << modelDir << " ... " << std::flush;

    const clock::time_point loadBegin = clock::now();
    if (!model.load(modelDir)) {
        std::cout << "\nfailed: " << model.error << std::endl;
        return 1;
    }
    const double loadSeconds =
        std::chrono::duration<double>(clock::now() - loadBegin).count();

    std::cout << model.cfg.nLayer << " layers, "
              << model.cfg.nEmbd << " wide, "
              << model.cfg.nHead << " heads, "
              << model.cfg.vocabSize << " vocab  ("
              << loadSeconds << "s)" << std::endl << std::endl;

    std::vector<size_t> ids = model.tokenizer.encode(prompt);
    if (ids.empty()) {
        std::cout << "the prompt encoded to no tokens" << std::endl;
        return 1;
    }

    math::random::Random rng{seed};

    /* the loop lives here rather than in GPT2::generate so each token can be
       printed the moment it is produced, which is the only way a run this
       slow feels like anything is happening */
    std::cout << prompt << std::flush;

    const clock::time_point genBegin = clock::now();
    size_t produced = 0;

    for (size_t i = 0; i < tokens && ids.size() < model.cfg.nPositions; ++i) {
        ai::Tensor<float> logits = model.forward(ids, true);
        const size_t next = ai::GPT2::sampleTopK(logits.ptr, model.cfg.vocabSize,
                                                 temperature, topK, rng);
        ids.push_back(next);
        ++produced;

        std::cout << model.tokenizer.decodeOne(next) << std::flush;
    }

    const double genSeconds =
        std::chrono::duration<double>(clock::now() - genBegin).count();

    std::cout << std::endl << std::endl
              << produced << " tokens in " << genSeconds << "s  ("
              << (produced / (genSeconds > 0 ? genSeconds : 1)) << " tok/s, "
              << "no KV cache, single threaded)" << std::endl;

    return 0;
}
