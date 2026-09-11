#include <chrono>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../include/platform/Console.hpp"
#include "../include/platform/Memory.hpp"
#include "../include/ai/LocalModel.hpp"

/*
    Interactive local LLM, CPU, from scratch.

    Point it at any model folder. The architecture is read out of config.json
    rather than compiled in, so gpt2, llama, mistral, qwen2, qwen3 and phi3
    all run through the same binary.

    Nothing links against a machine learning framework. files::json reads the
    config, the weight headers and the tokenizer; files::File reads the bytes;
    ai::Tensor holds them; ai::kernels runs the matmuls across every core;
    ai::LocalModel is the model; math::random::Random samples.

        make release-llm
        ./bin/release/llm models/qwen3-8b

    Commands:
        /reset     forget the conversation
        /raw       stop wrapping input in the chat template
        /chat      go back to the chat template
        /stats     show the last timings
        /quit      exit
*/

static void usage() {
    std::cout
        << "usage: LLM <model-dir> [--tokens N] [--temp F] [--topk N]\n"
        << "           [--seed N] [--ctx N] [--raw] [--force] [--q8|--q4] [--prompt TEXT]\n"
        << "           [--random [EPS]]\n"
        << "                        with no EPS, run the topology with invented weights,\n"
        << "                        reading no payload at all. With EPS in (0, 1], keep\n"
        << "                        the trained weights and offset each one by a uniform\n"
        << "                        draw from +/-EPS. --seed reproduces either exactly\n"
        << "\n"
        << "  --q8 / --q4  quantise the weight matrices as they load, so a model\n"
        << "               too large to hold in bfloat16 still fits. About half\n"
        << "               the size for q8, under a third for q4.\n"
        << "\n"
        << "  <model-dir>  a folder holding config.json, the .safetensors,\n"
        << "               and a tokenizer. The architecture is read from\n"
        << "               config.json, so nothing is compiled in per model.\n";
}

/*
    listModels(root):
    - anything under ./models with a config.json in it, with the architecture
      it declares. Purely a convenience for when the directory is forgotten,
      so the answer is a list rather than a usage string
*/
static void listModels(const std::filesystem::path& root) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return;

    bool any = false;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (!entry.is_directory()) continue;

        const std::filesystem::path config = entry.path() / "config.json";
        if (!std::filesystem::exists(config, ec)) continue;

        const files::json::ParseResult parsed = files::json::parseFile(config);
        std::string type = "?";
        if (parsed.ok) {
            const files::json::Value& v = parsed.value;
            type = v["model_type"].asString(v["text_config"]["model_type"].asString("?"));
        }

        const bool known = ai::specFor(type) != nullptr;
        if (!any) { std::cout << "\navailable models:\n"; any = true; }
        std::cout << "  " << entry.path().generic_string()
                  << "   (" << type << (known ? ")" : ", UNSUPPORTED)") << "\n";
    }
}

static std::string formatBytes(size_t bytes) {
    char buffer[64];
    if (bytes >= (size_t(1) << 30))
        std::snprintf(buffer, sizeof(buffer), "%.1f GB", double(bytes) / double(size_t(1) << 30));
    else
        std::snprintf(buffer, sizeof(buffer), "%.0f MB", double(bytes) / double(size_t(1) << 20));
    return buffer;
}

/*
    drawBar(done, total, stage):
    - redraws in place with a carriage return rather than a newline, so the
      bar stays on one line. Only repaints when the filled width or the
      label actually changes: the read reports every 16MB, which on a 30GB
      model is nearly two thousand updates, and repainting all of them costs
      more than it communicates
*/
static void drawBar(size_t done, size_t total, const std::string& stage) {
    constexpr int WIDTH = 28;

    const double fraction = total ? double(done) / double(total) : 0.0;
    const int filled = static_cast<int>(fraction * WIDTH);

    static int lastFilled = -1;
    static std::string lastStage;
    if (filled == lastFilled && stage == lastStage) return;
    lastFilled = filled;
    lastStage = stage;

    std::string bar(static_cast<size_t>(WIDTH), ' ');
    for (int i = 0; i < filled; ++i) bar[static_cast<size_t>(i)] = '#';

    char line[256];
    std::snprintf(line, sizeof(line), "\r  [%s] %3.0f%%  %s / %s  %-34s",
                  bar.c_str(), fraction * 100.0,
                  formatBytes(done).c_str(), formatBytes(total).c_str(), stage.c_str());

    std::cout << line << std::flush;
}

static void clearBar() {
    std::cout << "\r" << std::string(100, ' ') << "\r" << std::flush;
}

int main(int argc, char** argv) {
    /* the console is on an OEM code page by default, which renders UTF-8
       output one byte at a time through the wrong table */
    platform::useUtf8Console();

    std::string modelDir;
    std::string oneShot;
    size_t   maxTokens = 256;
    float    temperature = 0.7f;
    size_t   topK = 20;
    uint64_t seed = 0;
    size_t   context = 4096;
    bool     random  = false;
    float    epsilon = 0.0f;        // 0 means invent the weights outright
    bool     chat = true;
    bool     chatSetByUser = false;
    bool     force = false;
    ai::quant::Type quantType = ai::quant::Type::None;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasNext = (i + 1 < argc);

        if (arg == "--help" || arg == "-h")     { usage(); return 0; }
        else if (arg == "--model"  && hasNext)  { modelDir = argv[++i]; }
        else if (arg == "--tokens" && hasNext)  { maxTokens = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--temp"   && hasNext)  { temperature = std::strtof(argv[++i], nullptr); }
        else if (arg == "--topk"   && hasNext)  { topK = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--seed"   && hasNext)  { seed = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--ctx"    && hasNext)  { context = std::strtoull(argv[++i], nullptr, 10); }
        else if (arg == "--prompt" && hasNext)  { oneShot = argv[++i]; }
        else if (arg == "--raw")                { chat = false; chatSetByUser = true; }
        else if (arg == "--force")              { force = true; }
        else if (arg == "--random") {
            random = true;
            /*
                The epsilon is optional, so take the next argument only if it
                actually reads as a number in range - otherwise "--random
                --q4" would silently swallow the flag after it
            */
            if (hasNext) {
                char* stop = nullptr;
                const double value = std::strtod(argv[i + 1], &stop);
                if (stop && *stop == '\0' && value > 0.0 && value <= 1.0) {
                    epsilon = static_cast<float>(value);
                    ++i;
                }
            }
        }
        else if (arg == "--q8")                 { quantType = ai::quant::Type::Q8; }
        else if (arg == "--q4")                 { quantType = ai::quant::Type::Q4; }
        else if (!arg.empty() && arg[0] != '-' && modelDir.empty()) { modelDir = arg; }
        else { std::cout << "unknown option: " << arg << "\n"; usage(); return 1; }
    }

    if (modelDir.empty()) {
        usage();
        listModels("models");
        return 1;
    }

    using clock = std::chrono::steady_clock;

    /*
        Check before allocating anything. The read would otherwise "succeed"
        well past physical memory by growing the page file, and then every
        weight access during generation touches disk. That is not a slow
        model, it is an unusable one, and it takes the whole machine with it.
    */
    const ai::ModelWeights::Plan sized = ai::LocalModel::plan(modelDir, quantType);
    const size_t onDisk    = sized.onDisk;
    const size_t resident  = sized.resident;
    /* both random modes work in floats before narrowing back down, so the
       one tensor in flight is twice the width a plain read would need */
    const size_t peak      = random ? sized.syntheticPeak() : sized.peak();
    const size_t available = platform::availableMemory();

    if (!sized.ok && !sized.error.empty())
        std::cout << "could not size the model up front: " << sized.error << std::endl;

    if (random && epsilon > 0.0f)
        std::cout << "--random " << epsilon << ": the trained weights, each offset by a\n"
                  << "          uniform draw from +/-" << epsilon << ".\n";
    else if (random)
        std::cout << "--random: the shapes come from the checkpoint, the numbers do not.\n"
                  << "          this is a topology test, not the model.\n";

    if (onDisk && available) {
        std::cout << "weights " << formatBytes(onDisk) << " on disk";
        if (quantType != ai::quant::Type::None)
            std::cout << " -> " << formatBytes(resident) << " as "
                      << ai::quant::typeName(quantType)
                      << " (peak " << formatBytes(peak) << " loading)";
        std::cout << ", available " << formatBytes(available) << std::endl;

        if (peak > available) {
            std::cout << "\nrefusing to load: this needs about "
                      << formatBytes(peak) << " and only " << formatBytes(available)
                      << " is available.\n"
                      << "loading it would page to disk and make the machine unusable.\n";

            if (quantType == ai::quant::Type::None)
                std::cout << "try --q8 ("
                          << formatBytes(ai::LocalModel::plan(
                                 modelDir, ai::quant::Type::Q8).peak())
                          << " peak) or --q4 ("
                          << formatBytes(ai::LocalModel::plan(
                                 modelDir, ai::quant::Type::Q4).peak())
                          << " peak).\n";
            else if (quantType == ai::quant::Type::Q8)
                std::cout << "try --q4 ("
                          << formatBytes(ai::LocalModel::plan(
                                 modelDir, ai::quant::Type::Q4).peak())
                          << " peak).\n";

            std::cout << "pass --force to try anyway.\n";
            if (!force) return 1;
            std::cout << "--force given, continuing.\n";
        } else if (peak > available - available / 8) {
            std::cout << "warning: this leaves under an eighth of memory free.\n";
        }
    }

    ai::LocalModel model;
    std::cout << "loading " << modelDir << std::endl;

    const clock::time_point loadBegin = clock::now();
    const bool ok = model.load(modelDir, drawBar, quantType,
                               random ? std::optional<uint64_t>(seed) : std::nullopt,
                               epsilon);
    clearBar();

    if (!ok) {
        std::cout << "failed: " << model.error << std::endl;
        return 1;
    }
    const double loadSeconds = std::chrono::duration<double>(clock::now() - loadBegin).count();

    if (context > model.cfg.maxPositions) context = model.cfg.maxPositions;

    /* a base model has no chat tokens, so wrapping its input in a template
       it never saw during training only confuses it */
    if (!chatSetByUser && !model.hasChatTemplate()) chat = false;

    const char* normName = model.spec.norm == ai::NormKind::RMSNorm ? "RMSNorm" : "LayerNorm";
    const char* posName  = model.spec.pos  == ai::PosKind::RoPE     ? "RoPE"    : "learned";
    const char* mlpName  = model.spec.mlp  == ai::MLPKind::SwiGLU   ? "SwiGLU"  : "GELU MLP";

    std::cout << "\n"
              << "  architecture " << model.modelType
                                   << "  (" << normName << ", " << posName
                                   << ", " << mlpName
                                   << (model.spec.qkNorm ? ", QK-norm" : "") << ")\n"
              << "  layers       " << model.cfg.nLayer << "\n"
              << "  hidden       " << model.cfg.hidden << "\n"
              << "  heads        " << model.cfg.nHead << " query / "
                                   << model.cfg.nKVHead << " kv  (head_dim "
                                   << model.cfg.headDim << ")\n"
              << "  vocab        " << model.cfg.vocab << "\n"
              << "  parameters   " << (model.parameterCount() / 1000000) << "M\n"
              << "  shards       " << model.weights.shardCount() << "\n"
              << "  weights      " << ai::quant::typeName(model.quantType)
                                   << ", " << (model.weights.residentBytes() / (1024 * 1024))
                                   << " MB resident\n"
              << "  threads      " << ai::kernels::Pool::instance().size() << "\n"
              << "  chat format  " << model.chatFormatName() << "\n"
              << "  context      " << context << "\n"
              << "  loaded in    " << loadSeconds << "s" << std::endl;

    /*
        How much of the perturbation actually landed. bfloat16 keeps 8
        mantissa bits, so the smallest change it can hold is about 0.4% of a
        weight's own magnitude - an epsilon below that rounds straight back
        to where it started for most weights, and reporting the fraction
        turns a silently-empty experiment into a measurement.
    */
    if (model.weights.perturbedValues) {
        const double fraction = 100.0 * double(model.weights.changedValues)
                                      / double(model.weights.perturbedValues);
        std::cout << "  perturbed    " << model.weights.changedValues << " of "
                  << model.weights.perturbedValues << " weights actually changed ("
                  << std::fixed << std::setprecision(1) << fraction << "%)"
                  << std::defaultfloat << std::endl;

        if (fraction < 50.0)
            std::cout << "               most of the offset was smaller than bfloat16 can\n"
                      << "               hold and rounded away; try a larger --random\n";
    }

    ai::LocalCache cache;
    cache.resize(model.cfg, context);

    math::random::Random rng{seed};
    std::vector<float> logits(model.cfg.vocab);

    double lastPrefill = 0.0, lastDecode = 0.0;
    size_t lastPrompt = 0, lastProduced = 0;

    /*
        run(text): prefill then decode.
        - the prompt goes through in one forward pass, where every token is
          independent and the matmuls are tall. Decoding is one row at a time
          and is bound by how fast the weights can be read from memory
        - the cache carries across turns, so a follow-up only pays for its own
          tokens rather than reprocessing the conversation
    */
    const auto run = [&](const std::string& text) {
        /* the wrapper tokens belong to the start of a sequence, so a
           follow-up turn appended to a live cache must not repeat them */
        std::vector<size_t> ids = model.tokenizer.encode(text, cache.length == 0);
        if (ids.empty()) return;

        if (cache.length + ids.size() + 8 >= context) {
            std::cout << "(context full, resetting)" << std::endl;
            cache.reset();
        }

        const clock::time_point prefillBegin = clock::now();
        model.forward(ids.data(), ids.size(), cache.length, cache, logits.data());
        lastPrefill = std::chrono::duration<double>(clock::now() - prefillBegin).count();
        lastPrompt = ids.size();

        const clock::time_point decodeBegin = clock::now();
        size_t produced = 0;

        for (size_t i = 0; i < maxTokens && cache.length < context; ++i) {
            const size_t next =
                ai::LocalModel::sampleTopK(logits.data(), model.cfg.vocab, temperature, topK, rng);
            if (model.isStop(next)) break;

            // only the first token of a generation may have its leading
            // space stripped; see Tokenizer::decode
            std::cout << model.tokenizer.decode({next}, produced == 0) << std::flush;
            ++produced;

            model.forward(&next, 1, cache.length, cache, logits.data());
        }

        lastDecode = std::chrono::duration<double>(clock::now() - decodeBegin).count();
        lastProduced = produced;
        std::cout << std::endl;
    };

    const auto stats = [&] {
        std::cout << "  prompt " << lastPrompt << " tok in " << lastPrefill << "s ("
                  << (lastPrompt / (lastPrefill > 0 ? lastPrefill : 1)) << " tok/s)"
                  << "   generated " << lastProduced << " tok in " << lastDecode << "s ("
                  << (lastProduced / (lastDecode > 0 ? lastDecode : 1)) << " tok/s)"
                  << "   context " << cache.length << "/" << context << std::endl;
    };

    if (!oneShot.empty()) {
        run(chat ? model.chatPrompt(oneShot) : oneShot);
        stats();
        return 0;
    }

    std::cout << "\ntype a prompt, or /quit to exit, /reset to clear, /raw or /chat "
              << "to switch mode, /stats for timings\n" << std::endl;

    std::string line;
    while (true) {
        std::cout << (chat ? "> " : "raw> ") << std::flush;
        if (!std::getline(std::cin, line)) break;

        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") { cache.reset(); std::cout << "(cleared)" << std::endl; continue; }
        if (line == "/raw")   { chat = false; continue; }
        if (line == "/chat")  { chat = true;  continue; }
        if (line == "/stats") { stats(); continue; }

        std::cout << std::endl;
        run(chat ? model.chatPrompt(line) : line);
        std::cout << std::endl;
        stats();
        std::cout << std::endl;
    }

    return 0;
}
