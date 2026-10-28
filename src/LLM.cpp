#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../include/platform/Console.hpp"
#include "../include/platform/ExeDir.hpp"
#include "../include/platform/Memory.hpp"
#include "../include/ai/Cores.hpp"
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

/*
    An instruct model trained on agentic coding traces does not answer a
    "write me a program" prompt by writing the program. It answers by
    orienting itself: listing the directory, opening package.json, calling
    tools that in this harness do not exist. That is the behaviour it was
    trained to have, applied to a prompt that never mentioned there is no
    repository and nothing to call.

    So say so - but say it in the model's own terms. Describing the
    situation ("there are no tools, no file system, nothing to inspect")
    was measured NOT to work on Qwen3.5-27B: it acknowledged the preamble
    and went on emitting tool calls anyway. Naming the literal syntax it
    must not produce did work, first try. The distinction seems to be
    between a fact about the world, which a strong agentic prior overrides,
    and an instruction about output, which it does not.
*/
static constexpr const char* DEFAULT_SYSTEM =
    "There are no tools. Never emit <tool_call>, <function=, or any "
    "tool-call syntax. You cannot list files, read files, or run commands. "
    "Write your complete answer, with full code, in this one message.";

/*
    Reasoning effort.

    Qwen3.5 and later carry a dial for how much scratchpad to spend before
    answering, and it is implemented as nothing more than a sentence at the
    top of the system turn - these sentences, copied verbatim from the
    model's own chat_template so the wording is the one it was tuned on.

    Three levels, and only three: the template raises on anything else.
    "medium" deliberately says nothing at all - it is the model's own
    baseline, not a middle instruction.

    The dial only means anything when thinking is ON, which is why asking
    for an effort turns thinking on. With the scratchpad closed there is no
    reasoning to calibrate and the sentence would be dead weight.
*/
static const char* effortInstruction(const std::string& level) {
    if (level == "xhigh")
        return "Reasoning effort is set to xhigh. Please think carefully through "
               "the task, validate key assumptions, consider plausible alternatives, "
               "and prioritize correctness, consistency, and clarity in the final answer.";
    if (level == "low")
        return "Reasoning effort is set to low. Keep your thinking brief and focused, "
               "moving directly to the conclusion without unnecessary elaboration.";
    return "";                      // medium carries no instruction
}

static void usage() {
    std::cout
        << "usage: LLM <model-dir> [--tokens N] [--temp F] [--topk N]\n"
        << "           [--seed N] [--ctx N] [--raw] [--force] [--q8|--q4] [--prompt TEXT]\n"
        << "           [--system TEXT] [--no-system] [--effort low|medium|xhigh]\n"
        << "           [--random [EPS]]\n"
        << "                        with no EPS, run the topology with invented weights,\n"
        << "                        reading no payload at all. With EPS in (0, 1], keep\n"
        << "                        the trained weights and offset each one by a uniform\n"
        << "                        draw from +/-EPS. --seed reproduces either exactly\n"
        << "\n"
        << "  --gpu        run the matmuls on the GPU. Uploads whatever the weights\n"
        << "               already are - bfloat16, int8 or int4 - and combines with\n"
        << "               --q8 / --q4 to fit a larger model on the card.\n"
        << "  --vram N     cap the GPU at N gigabytes. A model that does not fit puts\n"
        << "               as many whole layers on the card as will and runs the rest\n"
        << "               on the CPU - slower than all-GPU, but it runs.\n"
        << "  --effort L   let the model think before answering, at low, medium or\n"
        << "               xhigh effort. Off by default: the scratchpad is closed and\n"
        << "               the answer comes straight out. Only Qwen3.5 and later have\n"
        << "               the dial; the reasoning is printed along with the answer.\n"
        << "  --system T   replace the preamble that tells the model there are no tools\n"
        << "               to call and to answer directly. --no-system drops it; a\n"
        << "               model that was not trained to call tools does not need it.\n"
        << "  --q8 / --q4  quantise the weight matrices as they load, so a model\n"
        << "               too large to hold in bfloat16 still fits. About half\n"
        << "               the size for q8, under a third for q4.\n"
        << "  --cores C    which CPUs to run on, one pinned thread each: a list such\n"
        << "               as 0-31 or 0-15,32-47, \"all\", or \"auto\" (the default) for\n"
        << "               every CPU no other running llm has claimed. Claims are\n"
        << "               kept in llm-cores.txt beside this executable, so a second\n"
        << "               instance started with auto stays off the first one's CPUs.\n"
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

    /*
        clamped, because this writes into a fixed-width buffer below and a
        caller that reports more done than total would otherwise run off
        the end of it. That is a heap overflow rather than a cosmetic bug,
        and the caller does not have to be wrong on purpose - a total that
        is refined while the work runs is enough
    */
    double fraction = total ? double(done) / double(total) : 0.0;
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;
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
    bool useGpu = false;
    double vramCapGB = 0.0;
    std::string systemPrompt = DEFAULT_SYSTEM;
    std::string effort;            // empty = do not think at all
    std::string coresSpec = "auto";

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
        else if (arg == "--system" && hasNext)  { systemPrompt = argv[++i]; }
        else if (arg == "--no-system")          { systemPrompt.clear(); }
        else if (arg == "--effort" && hasNext)   { effort = argv[++i]; }
        else if (arg == "--cores"  && hasNext)  { coresSpec = argv[++i]; }
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
        else if (arg == "--gpu")                { useGpu = true; }
        else if (arg == "--vram"   && hasNext)  { useGpu = true; vramCapGB = std::strtod(argv[++i], nullptr); }
        else if (!arg.empty() && arg[0] != '-' && modelDir.empty()) { modelDir = arg; }
        else { std::cout << "unknown option: " << arg << "\n"; usage(); return 1; }
    }

    if (modelDir.empty()) {
        usage();
        listModels("models");
        return 1;
    }

    if (!effort.empty() && effort != "low" && effort != "medium" && effort != "xhigh") {
        std::cout << "unknown --effort \"" << effort << "\": expected low, medium or xhigh\n";
        return 1;
    }

    /* the dial is a sentence at the top of the system turn, ahead of
       whatever else is there, exactly where the official template puts it */
    if (!effort.empty()) {
        const std::string line = effortInstruction(effort);
        if (!line.empty())
            systemPrompt = systemPrompt.empty() ? line : line + "\n\n" + systemPrompt;
    }

    /*
        Decide the CPUs before anything touches the thread pool: it is built
        once, on first use, with whatever it was configured for by then.

        An explicit list is taken as given - overlapping another instance is
        allowed, it is only warned about. auto takes what the others have
        not claimed, and falls back to sharing everything rather than
        refusing to run when nothing is left. Choosing and recording happen
        under one lock inside the registry, so two instances started at
        once cannot both take the same free CPUs.

        The registry lives for the rest of main and removes this process's
        line on the way out. A run that is killed instead leaves a line
        that the next reader sees is dead and drops.
    */
    std::vector<size_t> explicitCpus;
    if (coresSpec != "auto") {
        std::string error;
        if (!ai::cores::parse(coresSpec, explicitCpus, error)) {
            std::cout << "--cores " << coresSpec << ": " << error << "\n";
            return 1;
        }
    }

    ai::cores::Registry registry(platform::exeDir() / "llm-cores.txt");
    std::vector<size_t> cpus;
    std::string coresNote;
    {
        std::string error;
        const bool recorded = registry.claim(
            [&](const std::vector<ai::cores::Claim>& others) {
                std::vector<size_t> held;
                for (const ai::cores::Claim& c : others)
                    held.insert(held.end(), c.cpus.begin(), c.cpus.end());
                std::sort(held.begin(), held.end());

                const auto overlaps = [&](const std::vector<size_t>& mine) {
                    for (size_t c : mine)
                        if (std::binary_search(held.begin(), held.end(), c)) return true;
                    return false;
                };

                if (!explicitCpus.empty()) {
                    if (overlaps(explicitCpus))
                        coresNote = "warning: some of these CPUs are also claimed by another "
                                    "running llm; both will run slower";
                    return explicitCpus;
                }

                std::vector<size_t> free;
                for (size_t c : ai::cores::allowed())
                    if (!std::binary_search(held.begin(), held.end(), c)) free.push_back(c);

                if (free.empty()) {
                    coresNote = "warning: every CPU is claimed by another running llm, so "
                                "this one shares all of them. Give each a --cores list";
                    return ai::cores::allowed();
                }
                if (!held.empty())
                    coresNote = "CPUs " + ai::cores::format(held)
                              + " are claimed by another running llm; using the rest";
                return free;
            },
            modelDir, cpus, error);

        if (!recorded) {
            /* an unwritable folder costs the coordination, not the run */
            std::cout << "note: could not record the CPUs in use (" << error << ")\n";
            if (cpus.empty()) cpus = explicitCpus.empty() ? ai::cores::allowed() : explicitCpus;
        }
    }

    if (!coresNote.empty()) std::cout << coresNote << "\n";
    ai::kernels::Pool::configure(cpus);

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

    /*
        Two memories, and with --gpu they answer different questions.

        System RAM is what the LOAD needs: the checkpoint streams through
        it whether or not a card is involved, and the peak is set by the
        largest tensor in flight plus whatever has been narrowed so far.
        That is the number the refusal below is about.

        VRAM is what the RESIDENT weights need once they have been sent
        across. A model can pass the first test and fail the second, in
        which case it does not fail at all - the layers that do not fit
        stay on the CPU - but it will be a great deal slower, and that is
        worth saying before the 25 seconds of loading rather than after.
    */
    const size_t gpuFree = useGpu ? ai::gpu::freeMemory() : 0;

    if (onDisk && available) {
        std::cout << "weights " << formatBytes(onDisk) << " on disk";
        if (quantType != ai::quant::Type::None)
            std::cout << " -> " << formatBytes(resident) << " as "
                      << ai::quant::typeName(quantType);
        std::cout << std::endl;

        if (gpuFree) {
            std::cout << "  system  " << formatBytes(available) << " available, "
                      << formatBytes(peak) << " peak while loading\n"
                      << "  GPU     " << formatBytes(gpuFree) << " free, "
                      << formatBytes(resident) << " to upload";

            const size_t reserve = size_t(512) << 20;
            const size_t usable  = gpuFree > reserve ? gpuFree - reserve : 0;
            if (resident > usable) {
                std::cout << "\n          about " << formatBytes(resident - usable)
                          << " of that will not fit and stays on the CPU; expect it to\n"
                          << "          run closer to CPU speed. A smaller quant may fit whole.";
            }
            std::cout << std::endl;
        } else if (useGpu) {
            std::cout << "  system  " << formatBytes(available) << " available, "
                      << formatBytes(peak) << " peak while loading\n"
                      << "  GPU     none found - this will run on the CPU" << std::endl;
        } else {
            std::cout << "  system  " << formatBytes(available) << " available, "
                      << formatBytes(peak) << " peak while loading" << std::endl;
        }

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

    /*
        Uploading is a separate step from loading on purpose. The weights
        are read, checked and narrowed exactly as they always were, and
        only then does a pointer per matrix change - so a GPU run and a CPU
        run are the same model in the same form, differing in where the
        matmul happens and nothing else. That is what makes the two
        comparable when they disagree.
    */
    std::string deviceLine = "CPU";
    if (useGpu) {
        ai::OffloadPlan plan;
        if (vramCapGB > 0.0)
            plan.vramBudget = static_cast<size_t>(vramCapGB * 1024.0 * 1024.0 * 1024.0);

        const size_t footprint = model.deviceFootprint();
        std::cout << "uploading " << formatBytes(footprint) << " to the GPU" << std::endl;

        const clock::time_point uploadBegin = clock::now();
        const ai::LocalModel::DeviceReport report = model.toDevice(plan, drawBar);
        clearBar();

        if (!report.ok) {
            std::cout << "GPU unavailable: " << report.error << "\n"
                      << "continuing on the CPU.\n";
        } else {
            const double seconds =
                std::chrono::duration<double>(clock::now() - uploadBegin).count();
            deviceLine = ai::gpu::deviceName() + ", " + formatBytes(report.bytes)
                       + " " + ai::quant::typeName(quantType)
                       + " in " + std::to_string(seconds).substr(0, 4) + "s";

            /*
                Say it plainly when only part of the model went across. A
                partial offload is several times slower than a full one and
                looks identical from the outside, which is exactly the kind
                of thing that gets mistaken for the GPU not working at all.
            */
            if (report.partial) {
                std::cout << "\nonly part of the model fits in "
                          << formatBytes(report.budget) << " of VRAM:\n"
                          << "  " << report.layersOnDevice << " of " << report.layersTotal
                          << " layers on the GPU, " << (report.layersTotal - report.layersOnDevice)
                          << " on the CPU"
                          << (report.headOnDevice ? "" : ", output projection on the CPU") << "\n"
                          << "  expect it to run closer to CPU speed than GPU speed.\n"
                          << "  a smaller --q4 / --q8 would fit more of it.\n\n";

                deviceLine += "  (" + std::to_string(report.layersOnDevice) + "/"
                            + std::to_string(report.layersTotal) + " layers)";
            }
        }
    }

    if (!effort.empty() && !model.usesThinkBlock())
        std::cout << "note: --effort has no effect here. This model's chat template has no\n"
                  << "      <think> block, so there is no scratchpad to spend it in.\n";

    if (context > model.cfg.maxPositions) context = model.cfg.maxPositions;

    /* a base model has no chat tokens, so wrapping its input in a template
       it never saw during training only confuses it */
    if (!chatSetByUser && !model.hasChatTemplate()) chat = false;

    const char* normName = model.spec.norm == ai::NormKind::RMSNorm ? "RMSNorm" : "LayerNorm";
    const char* posName  = model.spec.pos  == ai::PosKind::RoPE     ? "RoPE"    : "learned";
    const char* mlpName  = model.spec.mlp  == ai::MLPKind::SwiGLU   ? "SwiGLU"
                         : model.spec.mlp  == ai::MLPKind::MoE      ? "MoE"
                         :                                            "GELU MLP";

    std::cout << "\n"
              << "  architecture " << model.modelType
                                   << "  (" << normName << ", " << posName
                                   << ", " << mlpName
                                   << (model.spec.qkNorm ? ", QK-norm" : "") << ")\n"
              << "  layers       " << model.cfg.nLayer << "\n";
    if (model.cfg.isMoE())
        std::cout << "  experts      " << model.cfg.expertsPerToken << " of "
                  << model.cfg.numExperts << " per token\n";
    std::cout
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
              << "  threads      " << ai::kernels::Pool::instance().size()
                                   << " on CPUs " << ai::cores::format(cpus)
                                   << (ai::kernels::Pool::unpinned()
                                           ? "  (" + std::to_string(ai::kernels::Pool::unpinned())
                                             + " could not be pinned)"
                                           : std::string()) << "\n"
              << "  device       " << deviceLine << "\n"
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
    const auto run = [&](const std::string& userText) {
        /*
            Rendered here rather than by the caller because both the wrapper
            tokens and the system preamble belong to the START of a
            sequence, and only this scope knows whether the cache is empty -
            including after the reset below, which begins a fresh one and so
            needs them again. Building it outside meant a reset produced a
            sequence missing both.
        */
        const auto build = [&] {
            const bool first = (cache.length == 0);
            const std::string text = chat
                ? model.chatPrompt(userText, !effort.empty(),
                                   first ? systemPrompt : std::string(), first)
                : userText;
            return model.tokenizer.encode(text, first);
        };

        std::vector<size_t> ids = build();
        if (ids.empty()) return;

        if (cache.length + ids.size() + 8 >= context) {
            std::cout << "(context full, resetting)" << std::endl;
            cache.reset();
            ids = build();
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
        run(oneShot);
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
        run(line);
        std::cout << std::endl;
        stats();
        std::cout << std::endl;
    }

    return 0;
}
