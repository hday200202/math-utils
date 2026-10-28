#include <algorithm>
#include <cmath>
#include <cstdio>
#include <bit>
#include <fstream>
#include <iostream>
#include <optional>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "../include/ai/LocalModel.hpp"

namespace fs = std::filesystem;

/* ========================================================================= //
    LocalModel test suite.

    The point of this file is to prove the generalisation did not quietly
    break either architecture. Both GPT-2 and Qwen3 are run through the SAME
    forward pass and diffed, stage by stage, against the numpy references
    that were written against the hand-specialised versions.

    GPT-2 exercises every axis the modern models do not: Conv1D [in,out]
    weights, learned absolute positions, LayerNorm with a bias, a fused
    c_attn, a two-matrix GELU feed forward, biases throughout. An
    abstraction validated only against near-identical models is not
    validated at all.

        py tools/gpt2_oracle.py  --prompt "The capital of France is" --dump out/oracle
        py tools/qwen3_oracle.py --ids 3838,374,279,6722,315,9625,30  --dump out/qwen-oracle
        make release-llm-test && ./bin/release/llm-testing
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

static int failures = 0;

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] "
              << COLOR_RESET << label << std::endl;
    if (!pass) ++failures;
}

static void section(const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
}


/* ========================================================================= //
    The spec registry
// ========================================================================= */

static void testRegistry() {
    section("architecture registry");

    check("gpt2 is known",     ai::specFor("gpt2")     != nullptr);
    check("qwen3 is known",    ai::specFor("qwen3")    != nullptr);
    check("qwen2 is known",    ai::specFor("qwen2")    != nullptr);
    check("llama is known",    ai::specFor("llama")    != nullptr);
    check("mistral is known",  ai::specFor("mistral")  != nullptr);
    check("phi3 is known",     ai::specFor("phi3")     != nullptr);
    check("an unknown type is rejected", ai::specFor("mamba") == nullptr);

    const ai::ArchSpec* gpt2 = ai::specFor("gpt2");
    check("gpt2 uses LayerNorm",  gpt2->norm   == ai::NormKind::LayerNorm);
    check("gpt2 uses learned positions", gpt2->pos == ai::PosKind::Learned);
    check("gpt2 uses a two-matrix MLP",  gpt2->mlp == ai::MLPKind::TwoMatrix);
    check("gpt2 uses Conv1D layout",     gpt2->layout == ai::LinearLayout::InOut);
    check("gpt2 fuses qkv",              !gpt2->names.qkv.empty());

    const ai::ArchSpec* qwen3 = ai::specFor("qwen3");
    check("qwen3 uses RMSNorm",   qwen3->norm == ai::NormKind::RMSNorm);
    check("qwen3 uses RoPE",      qwen3->pos  == ai::PosKind::RoPE);
    check("qwen3 uses SwiGLU",    qwen3->mlp  == ai::MLPKind::SwiGLU);
    check("qwen3 has QK-norm",    qwen3->qkNorm);
    check("qwen3 splits qkv",     qwen3->names.qkv.empty());

    check("phi3 fuses qkv and gate_up", [] {
        const ai::ArchSpec* s = ai::specFor("phi3");
        return !s->names.qkv.empty() && !s->names.gateUp.empty();
    }());

    check("name templates expand the layer index",
          ai::expandName("model.layers.{L}.mlp.up_proj.weight", 7)
          == "model.layers.7.mlp.up_proj.weight");
    check("a template without {L} is unchanged",
          ai::expandName("model.norm.weight", 3) == "model.norm.weight");

    const ai::ArchSpec* glm = ai::specFor("glm5_next");
    check("glm5_next is known", glm != nullptr && ai::specFor("glm5_next_text") != nullptr);
    check("glm5_next's linear layers are Kimi Delta Attention",
          glm->linearMixer == ai::MixerKind::KimiDelta);
    check("glm5_next's other layers are latent attention",
          glm->attentionMixer == ai::MixerKind::LatentAttention);
    check("glm5_next reads past the vision tower", [&] {
        for (const std::string& p : glm->skipPrefixes) if (p == "model.visual.") return true;
        return false;
    }());
}


/* ========================================================================= //
    FP8
    The decode is a table built from the bit layout, so it is checked against
    values worked out by hand rather than against itself. The tile scales are
    checked by putting a different scale in each tile and reading it back.
// ========================================================================= */

/* ========================================================================= //
    CPU selection and the claims file
// ========================================================================= */

static void testCores() {
    section("cores");

    std::vector<size_t> cpus;
    std::string error;
    check("a list with ranges parses",
          ai::cores::parseList("0-3, 8,10-11", cpus, error)
          && cpus == std::vector<size_t>({0, 1, 2, 3, 8, 10, 11}));
    check("repeats and order do not matter",
          ai::cores::parseList("5,1-2,2", cpus, error) && cpus == std::vector<size_t>({1, 2, 5}));
    check("a backwards range is rejected", !ai::cores::parseList("7-3", cpus, error));
    check("junk is rejected", !ai::cores::parseList("0-x", cpus, error));
    check("format folds runs back into ranges",
          ai::cores::format({11, 0, 1, 2, 3, 8, 10}) == "0-3,8,10-11");

    const std::vector<size_t> usable = ai::cores::allowed();
    check("this process may run on at least one CPU (" + ai::cores::format(usable) + ")",
          !usable.empty());
    check("\"all\" is every allowed CPU", ai::cores::parse("all", cpus, error) && cpus == usable);
    check("a CPU that is not there is refused",
          !ai::cores::parse(std::to_string(usable.back() + 100000), cpus, error));

    /* on a thread of its own: threads inherit their creator's affinity on
       Linux, so pinning this one would squeeze every later test onto a core */
    bool pinned = false;
    std::thread([&] { pinned = ai::cores::pinCurrentThread(usable.back()); }).join();
    check("a thread can be pinned to an allowed CPU", pinned);

    check("this process is seen as running",
          ai::cores::processStamp(ai::cores::selfPid()) != 0);

    const fs::path file = fs::temp_directory_path() / "llm-testing-cores.txt";
    std::error_code ec;
    fs::remove(file, ec);
    {
        /* a line left by a process that is long gone, as a killed run leaves */
        std::ofstream(file) << "999999999 12345 0-3 dead-model\n";

        ai::cores::Registry registry(file);
        std::vector<size_t> taken;
        size_t othersSeen = 99;
        const bool ok = registry.claim([&](const std::vector<ai::cores::Claim>& others) {
            othersSeen = others.size();
            return std::vector<size_t>{usable.front()};
        }, "test model", taken, error);

        check("a claim is recorded", ok && taken == std::vector<size_t>{usable.front()});
        check("a dead process's claim is ignored", othersSeen == 0);

        std::ifstream in(file);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        check("the file names this process and drops the dead one",
              text.find(std::to_string(ai::cores::selfPid()) + " ") != std::string::npos
              && text.find("dead-model") == std::string::npos);
    }
    check("the claim is released on the way out", !fs::exists(file));
    check("no lock is left behind", !fs::exists(file.string() + ".lock"));
}


static void testFp8() {
    section("fp8");

    const float* t = ai::quant::fp8Table();
    check("0x38 is 1.0",          t[0x38] == 1.0f);
    check("0x3C is 1.5",          t[0x3C] == 1.5f);
    check("0xB8 is -1.0",         t[0xB8] == -1.0f);
    check("0x7E is 448, the max", t[0x7E] == 448.0f);
    check("0x08 is 2^-6, the smallest normal", t[0x08] == 0.015625f);
    check("0x01 is 2^-9, the smallest subnormal", t[0x01] == 0.001953125f);
    check("0x7F is NaN",          std::isnan(t[0x7F]));

    // a 4x4 matrix of 1.0 in 2x2 tiles, each tile scaled differently
    std::shared_ptr<uint8_t[]> codes(new uint8_t[16]);
    for (size_t i = 0; i < 16; ++i) codes[i] = 0x38;
    std::shared_ptr<float[]> scales(new float[4]{1.0f, 2.0f, 3.0f, 4.0f});
    const ai::quant::Fp8Tensor w = ai::quant::makeFp8(codes, 4, 4, scales, 2, 2);

    float row[4];
    w.dequantRow(3, row);
    check("a row picks up the scale of each tile it crosses",
          row[0] == 3.0f && row[1] == 3.0f && row[2] == 4.0f && row[3] == 4.0f);

    const float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4];
    ai::kernels::matmulNTFP8(x, w, y, 1, 4, 4, false);
    check("the FP8 matmul sums the scaled weights",
          y[0] == 6.0f && y[1] == 6.0f && y[2] == 14.0f && y[3] == 14.0f);

    const ai::quant::Fp8Tensor bare = ai::quant::makeFp8(codes, 4, 4, nullptr, 0, 0);
    bare.dequantRow(0, row);
    check("FP8 without a scale tensor is unscaled", row[0] == 1.0f && row[3] == 1.0f);
}


/* ========================================================================= //
    Oracle diff, shared by both architectures
// ========================================================================= */

struct Reference {
    std::string        name;
    std::vector<float> values;
};

static bool loadManifest(const std::string& dir, std::vector<Reference>& out) {
    const files::json::ParseResult manifest = files::json::parseFile(dir + "/manifest.json");
    if (!manifest.ok || !manifest.value.isArray()) return false;

    for (size_t i = 0; i < manifest.value.size(); ++i) {
        Reference r;
        r.name = manifest.value[i]["name"].asString();

        size_t n = 1;
        for (size_t d : manifest.value[i]["shape"].asSizes()) n *= d;

        std::ifstream in(dir + "/" + r.name + ".bin", std::ios::binary);
        if (!in) return false;

        r.values.resize(n);
        in.read(reinterpret_cast<char*>(r.values.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
        if (!in) return false;

        out.push_back(std::move(r));
    }
    return !out.empty();
}

/*
    diffAgainstOracle:
    - runs the prompt through LocalModel with a trace and compares every
      stage the reference dumped. Tolerance is relative to each tensor's own
      magnitude, because logits sit near 100 while a norm output sits near 1
*/
static void diffAgainstOracle(const ai::LocalModel& model,
                              const std::vector<size_t>& ids,
                              const std::string& oracleDir,
                              double tolerance,
                              size_t expectedArgmax) {
    std::vector<Reference> refs;
    if (!loadManifest(oracleDir, refs)) {
        check("the oracle dump at " + oracleDir + " loads", false);
        return;
    }
    check("the oracle dump loads (" + std::to_string(refs.size()) + " tensors)", true);

    ai::LocalCache cache;
    cache.resize(model.cfg, 256);

    ai::LocalTrace trace;
    std::vector<float> logits(model.cfg.vocab);
    model.forward(ids.data(), ids.size(), 0, cache, logits.data(), &trace);

    size_t compared = 0, mismatched = 0, skipped = 0;
    double worst = 0.0;
    std::string worstName;

    for (const Reference& r : refs) {
        const std::vector<float>* got = trace.find(r.name);
        if (!got) {
            /* attention weights are consumed head by head as they are
               produced and never materialised; anything else missing is a
               real gap in coverage */
            if (r.name.find("attnw") == std::string::npos
             && r.name.find("scores") == std::string::npos
             && r.name.find(".qkv") == std::string::npos) {
                std::cout << "    untraced: " << r.name << std::endl;
                ++mismatched;
            }
            ++skipped;
            continue;
        }

        /* the reference keeps every row of the final norm and of the logits;
           the model only computes the last, since nothing reads the others */
        const float* ref = r.values.data();
        const size_t n = got->size();
        if (r.values.size() != n) {
            if (r.values.size() > n && r.values.size() % n == 0) ref += r.values.size() - n;
            else { std::cout << "    size mismatch: " << r.name << std::endl; ++mismatched; continue; }
        }

        double maxAbsRef = 1.0, maxAbsErr = 0.0;
        for (size_t i = 0; i < n; ++i) {
            maxAbsRef = std::max(maxAbsRef, static_cast<double>(std::fabs(ref[i])));
            maxAbsErr = std::max(maxAbsErr, static_cast<double>(std::fabs((*got)[i] - ref[i])));
        }

        const double relative = maxAbsErr / maxAbsRef;
        if (relative > worst) { worst = relative; worstName = r.name; }
        if (relative > tolerance) {
            std::cout << "    " << r.name << "  relative error " << relative << std::endl;
            ++mismatched;
        }
        ++compared;
    }

    check("stages compared: " + std::to_string(compared)
          + ", untraced: " + std::to_string(skipped), compared > 0);
    check("no stage diverges from the reference", mismatched == 0);

    char note[160];
    std::snprintf(note, sizeof(note), "worst relative error is %.2e (%s)",
                  worst, worstName.c_str());
    check(note, worst < tolerance);

    check("argmax of the logits is token " + std::to_string(expectedArgmax),
          ai::LocalModel::argmax(logits.data(), model.cfg.vocab) == expectedArgmax);
}


/* ========================================================================= //
    KV cache, checked per architecture

    The cache is pure optimisation: decoding one token at a time must land on
    the same numbers as running the sequence in one pass.
// ========================================================================= */

static void testCache(const ai::LocalModel& model, const std::vector<size_t>& ids) {
    std::vector<float> whole(model.cfg.vocab);
    ai::LocalCache a;
    a.resize(model.cfg, 128);
    model.forward(ids.data(), ids.size(), 0, a, whole.data());

    std::vector<float> stepped(model.cfg.vocab);
    ai::LocalCache b;
    b.resize(model.cfg, 128);
    for (size_t i = 0; i < ids.size(); ++i)
        model.forward(&ids[i], 1, i, b, stepped.data());

    double worst = 0.0;
    for (size_t i = 0; i < model.cfg.vocab; ++i)
        worst = std::max(worst, static_cast<double>(std::fabs(whole[i] - stepped[i])));

    char note[128];
    std::snprintf(note, sizeof(note),
                  "one-at-a-time decoding matches a single pass (max diff %.2e)", worst);
    check(note, worst < 5e-3);

    check("both paths pick the same token",
          ai::LocalModel::argmax(whole.data(), model.cfg.vocab)
       == ai::LocalModel::argmax(stepped.data(), model.cfg.vocab));

    check("a split prefill matches a whole one", [&] {
        std::vector<float> split(model.cfg.vocab);
        ai::LocalCache c;
        c.resize(model.cfg, 128);
        const size_t half = ids.size() / 2;
        model.forward(ids.data(), half, 0, c, split.data());
        model.forward(ids.data() + half, ids.size() - half, half, c, split.data());

        for (size_t i = 0; i < model.cfg.vocab; ++i)
            if (std::fabs(whole[i] - split[i]) > 5e-3) return false;
        return true;
    }());
}


/* ========================================================================= //
    GPT-2
// ========================================================================= */

static void testGPT2(const std::string& dir, const std::string& oracleDir) {
    section("gpt2 through LocalModel");

    ai::LocalModel model;
    if (!model.load(dir)) {
        check("gpt2 loads (" + model.error + ")", false);
        return;
    }
    check("gpt2 loads", true);

    check("detected as gpt2", model.modelType == "gpt2");
    check("12 layers, 768 wide", model.cfg.nLayer == 12 && model.cfg.hidden == 768);
    check("head_dim derived as 64", model.cfg.headDim == 64);
    check("multi-head means kv heads equal query heads",
          model.cfg.nKVHead == model.cfg.nHead && model.cfg.group() == 1);
    check("intermediate defaults to 4x hidden", model.cfg.intermediate == 3072);
    check("the vocabulary projection is tied to the embedding", model.cfg.tied);

    /* Conv1D weights are [in,out] on disk and must come back [out,in] */
    check("c_attn was transposed and split into q/k/v", [&] {
        const ai::LocalLayer& L = model.layers[0];
        return L.q.outDim == 768 && L.q.inDim == 768
            && L.k.outDim == 768 && L.v.outDim == 768;
    }());
    check("the fused c_attn bias was split too", [&] {
        const ai::LocalLayer& L = model.layers[0];
        return L.q.hasBias() && L.k.hasBias() && L.v.hasBias();
    }());
    check("c_fc was transposed to [3072,768]", [&] {
        const ai::LocalLayer& L = model.layers[0];
        return L.up.outDim == 3072 && L.up.inDim == 768;
    }());
    check("LayerNorm biases were loaded", model.layers[0].inputNormBias.storage != nullptr);

    check("the tokenizer came from vocab.json + merges.txt",
          model.tokenizer.style == ai::PreTokenizer::GPT2);
    check("gpt2 has no chat template", !model.hasChatTemplate());

    const std::vector<size_t> ids = {464, 3139, 286, 4881, 318};
    check("the prompt encodes to the reference ids",
          model.tokenizer.encode("The capital of France is") == ids);

    diffAgainstOracle(model, ids, oracleDir, 2e-3, 262);
    testCache(model, ids);

    /* GPT-2 had no KV cache before; through LocalModel it gets one, so the
       greedy continuation should match what the reference produced */
    check("greedy continuation matches the reference", [&] {
        ai::LocalCache cache;
        cache.resize(model.cfg, 128);
        std::vector<float> logits(model.cfg.vocab);
        std::vector<size_t> cur = ids;
        model.forward(cur.data(), cur.size(), 0, cache, logits.data());

        std::string out;
        for (int i = 0; i < 8; ++i) {
            const size_t next = ai::LocalModel::argmax(logits.data(), model.cfg.vocab);
            out += model.tokenizer.decode({next});
            model.forward(&next, 1, cache.length, cache, logits.data());
        }
        std::cout << "    gpt2 greedy: The capital of France is" << out << std::endl;
        return out == " the capital of the French Republic, and";
    }());
}


/* ========================================================================= //
    Qwen3
// ========================================================================= */

static void testQwen3(const std::string& dir, const std::string& oracleDir) {
    section("qwen3 through LocalModel");

    ai::LocalModel model;
    if (!model.load(dir)) {
        check("qwen3 loads (" + model.error + ")", false);
        return;
    }
    check("qwen3 loads", true);

    check("detected as qwen3", model.modelType == "qwen3");
    check("28 layers, 1024 wide", model.cfg.nLayer == 28 && model.cfg.hidden == 1024);
    check("head_dim is 128 from the config, not hidden/heads", model.cfg.headDim == 128);
    check("GQA group size is 2", model.cfg.group() == 2);
    check("q_proj widens to 2048", model.layers[0].q.outDim == 2048);
    check("weights stayed bfloat16", model.layers[0].q.isBF16());
    check("QK-norm tensors were loaded", model.layers[0].qNorm.storage != nullptr);
    check("no attention biases", !model.layers[0].q.hasBias());

    check("the tokenizer came from tokenizer.json",
          model.tokenizer.style == ai::PreTokenizer::Qwen);
    check("qwen3 has a chat template", model.hasChatTemplate());

    const std::vector<size_t> ids = {3838, 374, 279, 6722, 315, 9625, 30};
    check("the prompt encodes to the reference ids",
          model.tokenizer.encode("What is the capital of France?") == ids);

    diffAgainstOracle(model, ids, oracleDir, 2e-3, 576);
    testCache(model, ids);

    check("the chat answer mentions Paris", [&] {
        ai::LocalCache cache;
        cache.resize(model.cfg, 512);
        std::vector<float> logits(model.cfg.vocab);

        std::vector<size_t> prompt =
            model.tokenizer.encode(model.chatPrompt("What is the capital of France?"));
        model.forward(prompt.data(), prompt.size(), 0, cache, logits.data());

        std::string answer;
        for (int i = 0; i < 24; ++i) {
            const size_t next = ai::LocalModel::argmax(logits.data(), model.cfg.vocab);
            if (model.isStop(next)) break;
            answer += model.tokenizer.decode({next});
            model.forward(&next, 1, cache.length, cache, logits.data());
        }
        std::cout << "    qwen3 answer: " << answer << std::endl;
        return answer.find("Paris") != std::string::npos;
    }());
}


/* ========================================================================= //
    Quantisation

    Checked against the unquantised model rather than against the numpy
    reference: quantising deliberately changes the numbers, so the question
    is not whether they moved but whether they moved far enough to change
    what the model says.
// ========================================================================= */

static void testQuantRoundTrip() {
    section("quantisation round trip");

    /* a block whose values span a wide range, as a real weight row does */
    const size_t n = ai::quant::BLOCK * 4;
    std::vector<uint16_t> src(n);
    std::vector<float> original(n);

    for (size_t i = 0; i < n; ++i) {
        const float v = std::sin(float(i) * 0.7f) * (1.0f + float(i % 7));
        original[i] = std::bit_cast<float>(std::bit_cast<uint32_t>(v) & 0xFFFF0000u);
        src[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(original[i]) >> 16);
    }

    for (ai::quant::Type t : {ai::quant::Type::Q8, ai::quant::Type::Q4}) {
        ai::quant::QuantTensor q = ai::quant::quantize(src.data(), 1, n, t);
        check(std::string(ai::quant::typeName(t)) + " quantises", q.valid());

        std::vector<float> back(n);
        ai::quant::dequantRow(q, 0, back.data());

        double worst = 0.0, scale = 0.0;
        for (size_t i = 0; i < n; ++i) {
            worst = std::max(worst, (double)std::fabs(back[i] - original[i]));
            scale = std::max(scale, (double)std::fabs(original[i]));
        }

        /* one step of the integer grid is the most any value can be off by,
           so the bound is the block range over the number of levels */
        const double levels = (t == ai::quant::Type::Q8) ? 127.0 : 7.0;
        const double bound = scale / levels;

        char note[160];
        std::snprintf(note, sizeof(note),
                      "%s error stays within one quantisation step (%.4f <= %.4f)",
                      ai::quant::typeName(t), worst, bound * 1.01);
        check(note, worst <= bound * 1.01);
    }

    check("a row length that is not a multiple of the block size is refused",
          !ai::quant::quantize(src.data(), 1, ai::quant::BLOCK + 1, ai::quant::Type::Q8).valid());

    check("q8 is about half of bfloat16", [] {
        return ai::quant::bytesPerWeight(ai::quant::Type::Q8) < 1.2
            && ai::quant::bytesPerWeight(ai::quant::Type::Q8) > 1.0;
    }());
    check("q4 is under a third of bfloat16",
          ai::quant::bytesPerWeight(ai::quant::Type::Q4) < 2.0 / 3.0);
}


static void testQuantModel(const std::string& dir) {
    section("quantised model");

    ai::LocalModel ref;
    if (!ref.load(dir)) { check("the reference model loads", false); return; }

    const std::vector<size_t> ids = {3838, 374, 279, 6722, 315, 9625, 30};

    ai::LocalCache c; c.resize(ref.cfg, 128);
    std::vector<float> refLogits(ref.cfg.vocab);
    ref.forward(ids.data(), ids.size(), 0, c, refLogits.data());

    const size_t refBytes = ref.weights.residentBytes();
    const size_t refTop = ai::LocalModel::argmax(refLogits.data(), ref.cfg.vocab);

    for (ai::quant::Type t : {ai::quant::Type::Q8, ai::quant::Type::Q4}) {
        ai::LocalModel m;
        if (!m.load(dir, {}, t)) {
            check(std::string(ai::quant::typeName(t)) + " loads", false);
            continue;
        }
        check(std::string(ai::quant::typeName(t)) + " loads", true);
        check(std::string(ai::quant::typeName(t)) + " reports its type", m.quantType == t);

        const size_t bytes = m.weights.residentBytes();
        const double ratio = double(refBytes) / double(bytes);

        char note[160];
        std::snprintf(note, sizeof(note), "%s is %.2fx smaller (%zu MB vs %zu MB)",
                      ai::quant::typeName(t), ratio,
                      bytes / (1024 * 1024), refBytes / (1024 * 1024));
        check(note, ratio > (t == ai::quant::Type::Q8 ? 1.6 : 2.8));

        ai::LocalCache qc; qc.resize(m.cfg, 128);
        std::vector<float> logits(m.cfg.vocab);
        m.forward(ids.data(), ids.size(), 0, qc, logits.data());

        /*
            q8 is close enough to demand the same token. q4 is not, and
            pretending otherwise would be a test that lies: on a small model
            it genuinely does pick differently sometimes. What it must not do
            is leave the neighbourhood, so the bar is that its choice is
            still among the reference's most likely few.
        */
        const size_t got = ai::LocalModel::argmax(logits.data(), m.cfg.vocab);

        if (t == ai::quant::Type::Q8) {
            check("q8 picks the same next token as bfloat16", got == refTop);
        } else {
            std::vector<size_t> order(ref.cfg.vocab);
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                              [&](size_t a, size_t b) { return refLogits[a] > refLogits[b]; });

            bool inTop = false;
            for (int i = 0; i < 10; ++i) if (order[static_cast<size_t>(i)] == got) inTop = true;

            char where[160];
            std::snprintf(where, sizeof(where),
                          "q4 stays in the reference top 10 (picked %zu, bfloat16 picked %zu)",
                          got, refTop);
            check(where, inTop);
        }

        /* the cache is still pure optimisation, quantised or not */
        ai::LocalCache step; step.resize(m.cfg, 128);
        std::vector<float> stepped(m.cfg.vocab);
        for (size_t i = 0; i < ids.size(); ++i)
            m.forward(&ids[i], 1, i, step, stepped.data());

        double worst = 0.0;
        for (size_t i = 0; i < m.cfg.vocab; ++i)
            worst = std::max(worst, (double)std::fabs(logits[i] - stepped[i]));
        std::snprintf(note, sizeof(note),
                      "%s decoding one at a time still matches a single pass (%.2e)",
                      ai::quant::typeName(t), worst);
        check(note, worst < 5e-3);
    }
}


/* ========================================================================= //
    Qwen3-Next

    A tiny randomly initialised model rather than the real 151GB checkpoint:
    the architecture is identical, and the two pieces worth doubting - the
    per-key-head de-interleaving of the fused linear-attention projections,
    and the routing of a token to its handful of experts - are exercised just
    as thoroughly at eight experts as at five hundred.
// ========================================================================= */

static void testQwen3Next(const std::string& dir, const std::string& oracleDir) {
    section("qwen3_next through LocalModel");

    if (!fs::exists(dir + "/config.json")) {
        std::cout << "    skipped: no model at " << dir
                  << " (py tools/qwen3next_oracle.py)" << std::endl;
        return;
    }

    ai::LocalModel model;
    if (!model.load(dir)) {
        check("qwen3_next loads (" + model.error + ")", false);
        return;
    }
    check("qwen3_next loads", true);

    check("detected as qwen3_next", model.modelType == "qwen3_next");

    /* the layer kinds come from a stride rather than a list, so check the
       stride was read the way transformers reads it: counting from one */
    check("full attention every second layer, counting from one",
          model.cfg.mixers.size() == 4
       && model.cfg.mixers[0] == ai::MixerKind::GatedDeltaNet
       && model.cfg.mixers[1] == ai::MixerKind::Attention
       && model.cfg.mixers[2] == ai::MixerKind::GatedDeltaNet
       && model.cfg.mixers[3] == ai::MixerKind::Attention);

    check("8 experts, 3 per token",
          model.cfg.numExperts == 8 && model.cfg.expertsPerToken == 3);
    check("the experts were all loaded",
          model.layers[0].experts.size() == 8
       && model.layers[0].experts[7].down.valid());
    check("the router and shared expert were loaded",
          model.layers[0].router.valid()
       && model.layers[0].shared.up.valid()
       && model.layers[0].sharedScale.outDim == 1);

    check("the linear projections are fused",
          model.layers[0].linQKV.valid() && !model.layers[0].linZ.valid());
    check("in_proj_qkvz is 2*keyDim + 2*valueDim wide",
          model.layers[0].linQKV.outDim
       == 2 * model.cfg.linKeyWidth() + 2 * model.cfg.linValueWidth());
    check("in_proj_ba is two per value head",
          model.layers[0].linA.outDim == 2 * model.cfg.linValueHeads);

    check("a quarter of each head rotates", model.cfg.rotaryDims() == 4);

    const std::vector<size_t> ids = {5, 17, 42, 7, 300, 12};
    diffAgainstOracle(model, ids, oracleDir, 5e-3, 429);
    testCache(model, ids);
}


/* ========================================================================= //
    Synthetic weights

    Running a topology with invented numbers is only useful if the numbers
    are plausible: draw them too large and the residual stream saturates,
    every logit comes back NaN, and the run says nothing about the
    architecture. So the checks here are that it stays finite, that a seed
    reproduces itself exactly, and that a different seed is a different
    model - not that the output means anything, because it cannot.
// ========================================================================= */

/* ========================================================================= //
    Perturbed weights

    The trained weights, each offset by a uniform draw from +/-epsilon.

    The load for this goes down a different road than a plain one - read,
    widen to float, adjust, narrow back - so the first thing to establish is
    that the road itself changes nothing. An epsilon too small for bfloat16
    to represent has to come back bit-identical to a plain load; if it does
    not, every later measurement is measuring the plumbing.
// ========================================================================= */

static void testPerturbed(const std::string& dir) {
    section("perturbed weights: " + dir);

    if (!fs::exists(dir + "/config.json")) {
        std::cout << "    skipped: no model at " << dir << std::endl;
        return;
    }

    const std::vector<size_t> ids = {5, 17, 42, 7, 300, 12};

    struct Run {
        std::vector<float> logits;
        size_t changed = 0;
        size_t offered = 0;
        bool   ok = false;
    };

    auto runWith = [&](float epsilon, uint64_t seed) {
        Run r;
        ai::LocalModel m;
        const std::optional<uint64_t> s =
            (epsilon > 0.0f) ? std::optional<uint64_t>(seed) : std::nullopt;

        if (!m.load(dir, {}, ai::quant::Type::None, s, epsilon)) return r;

        ai::LocalCache cache;
        cache.resize(m.cfg, 64);
        r.logits.assign(m.cfg.vocab, 0.0f);
        m.forward(ids.data(), ids.size(), 0, cache, r.logits.data());
        r.changed = m.weights.changedValues;
        r.offered = m.weights.perturbedValues;
        r.ok = true;
        return r;
    };

    const Run plain = runWith(0.0f, 0);
    if (!plain.ok) { check("the model loads unperturbed", false); return; }
    check("the model loads unperturbed", true);

    /*
        The perturbing load takes a different road than a plain one - read,
        widen to float, adjust, narrow back - so before measuring anything,
        establish that the road itself changes nothing. An adjustment that
        does nothing has to give the original model bit for bit.

        Doing this with a very small epsilon instead would NOT work, and the
        reason is worth knowing: about 0.004% of the weights in these files
        are exactly zero, and bfloat16 represents magnitudes down to 1e-38,
        so adding any nonzero offset at all to a zero weight changes it. The
        floor on a perturbation is not bfloat16's precision - it is whether
        the weight was zero to begin with.
    */
    check("the adjust path is lossless when the adjustment does nothing", [&] {
        ai::LocalModel m;
        m.weights.perturb = [](const std::string&, const ai::TensorInfo&, float*) {};
        if (!m.load(dir)) return false;

        ai::LocalCache cache;
        cache.resize(m.cfg, 64);
        std::vector<float> through(m.cfg.vocab, 0.0f);
        m.forward(ids.data(), ids.size(), 0, cache, through.data());

        check("  (and it did take the adjust path)", m.weights.perturbedValues > 0);
        return through == plain.logits;
    }());

    /* a barely-there offset should reach almost nothing but the zeros */
    const Run tiny = runWith(1e-12f, 1);
    if (!tiny.ok) { check("a tiny perturbation loads", false); return; }

    const double reached = 100.0 * double(tiny.changed) / double(tiny.offered);
    std::cout << "    eps 1e-12 changed " << reached
              << "% of weights (the exact zeros)" << std::endl;
    check("an offset far below every weight's own scale reaches only the zeros",
          reached > 0.0 && reached < 0.5);

    /* an offset comparable to the weights themselves must land everywhere */
    const Run big = runWith(0.05f, 1);
    if (!big.ok) { check("a large perturbation loads", false); return; }

    const double landed = 100.0 * double(big.changed) / double(big.offered);
    std::cout << "    eps 0.05 changed " << landed << "% of weights" << std::endl;
    check("an offset the size of the weights lands on nearly all of them",
          landed > 95.0);
    check("and changes what the model predicts", big.logits != plain.logits);

    /*
        The offset has to be SYMMETRIC. A draw from [0, eps] instead of
        [-eps, +eps] still changes every weight and still degrades the model,
        so nothing above would notice - but it is a systematic shift of the
        whole checkpoint rather than noise on it, and that is a different
        experiment answering a different question. Measured directly: the
        mean change across a large matrix must sit near zero, not near
        eps/2.
    */
    check("the offset is symmetric, not a systematic shift", [&] {
        const float eps = 0.01f;

        ai::LocalModel a, b;
        if (!a.load(dir)) return false;
        if (!b.load(dir, {}, ai::quant::Type::None, std::optional<uint64_t>(1), eps))
            return false;

        /* the embedding table, because layer 0 is not guaranteed to have a
           q_proj - on qwen3_next it is a linear-attention layer and has none */
        const ai::Tensor<uint16_t>& x = a.embed.bf16;
        const ai::Tensor<uint16_t>& y = b.embed.bf16;
        if (!x.storage || !y.storage || x.numel() != y.numel()) return false;

        double total = 0.0;
        const size_t n = x.numel();
        for (size_t i = 0; i < n; ++i)
            total += double(ai::quant::widenBF16(y.ptr[i]))
                   - double(ai::quant::widenBF16(x.ptr[i]));

        const double mean = total / double(n);
        std::cout << "    mean change " << mean << " over " << n
                  << " weights (one-sided would be " << eps / 2 << ")" << std::endl;

        return std::fabs(mean) < double(eps) / 10.0;
    }());

    /* and it has to be reproducible, or a sweep means nothing */
    const Run again = runWith(0.05f, 1);
    check("the same seed and epsilon reproduce exactly",
          again.ok && again.logits == big.logits);

    const Run other = runWith(0.05f, 2);
    check("a different seed perturbs differently",
          other.ok && other.logits != big.logits);
}


static void testSynthetic(const std::string& dir) {
    section("synthetic weights: " + dir);

    if (!fs::exists(dir + "/config.json")) {
        std::cout << "    skipped: no model at " << dir << std::endl;
        return;
    }

    const std::vector<size_t> ids = {5, 17, 42, 7, 300, 12};

    auto logitsFor = [&](uint64_t seed, std::vector<float>& out) {
        ai::LocalModel m;
        if (!m.load(dir, {}, ai::quant::Type::None, std::optional<uint64_t>(seed)))
            return false;

        ai::LocalCache cache;
        cache.resize(m.cfg, 64);
        out.assign(m.cfg.vocab, 0.0f);
        m.forward(ids.data(), ids.size(), 0, cache, out.data());
        return true;
    };

    std::vector<float> a, b, c;
    if (!logitsFor(1, a)) { check("a synthetic model loads", false); return; }
    check("a synthetic model loads with no weights read", true);

    /*
        The one decision in the policy that the logits cannot pin down. A
        norm weight of twice the identity only doubles them - RMSNorm
        renormalises away the rest - so no plausible bound on "an untrained
        model's logits are small" would reject it.

        So read the weight the loader ACTUALLY produced and check it is
        scale-preserving. Recomputing the identity here instead would only
        re-test this test's own arithmetic: it would agree with a broken
        loader as readily as a working one.
    */
    check("the loaded norm weight is the identity for this convention", [&] {
        ai::LocalModel m;
        if (!m.load(dir, {}, ai::quant::Type::None, std::optional<uint64_t>(1)))
            return false;
        if (!m.finalNorm.storage) return false;

        const size_t n = m.cfg.hidden;
        std::vector<float> x(n), out(n);
        for (size_t i = 0; i < n; ++i) x[i] = static_cast<float>(i % 7) - 3.0f;

        ai::rmsNormRow(x.data(), m.finalNorm.ptr, out.data(), n,
                       m.cfg.normEps, m.spec.normWeightOffset);

        double energy = 0.0;
        for (size_t i = 0; i < n; ++i) energy += double(out[i]) * double(out[i]);
        const double rms = std::sqrt(energy / double(n));

        std::cout << "    norm weight " << m.finalNorm.ptr[0]
                  << " + offset " << m.spec.normWeightOffset
                  << " -> rms " << rms << std::endl;
        return std::fabs(rms - 1.0) < 1e-4;
    }());

    check("every logit is finite", [&] {
        for (float v : a) if (!std::isfinite(v)) return false;
        return true;
    }());

    /* an untrained model has no opinion, so the logits should sit close
       together - a wide spread would mean the scale is wrong */
    /* the bound is tight on purpose. An untrained model's logits sit near
       zero; 50 would have passed an initialisation drawn fifty times too
       wide, which measured 26 and is exactly the mistake worth catching */
    check("the logits are small, as an untrained model's must be", [&] {
        float worst = 0.0f;
        for (float v : a) worst = std::max(worst, std::fabs(v));
        std::cout << "    largest untrained logit: " << worst << std::endl;
        return worst < 5.0f;
    }());

    /*
        Each tensor's stream is seeded from its NAME. Seed them all alike and
        every pair of same-shaped matrices comes out literally equal - a
        model where q_proj and k_proj are the same matrix - and the logits
        give no hint of it, so it is checked for directly.
    */
    check("two tensors of the same shape are not the same tensor", [&] {
        ai::LocalModel m;
        if (!m.load(dir, {}, ai::quant::Type::None, std::optional<uint64_t>(1)))
            return false;

        size_t compared = 0;
        for (size_t i = 0; i < m.weights.raw.size() && compared < 50; ++i)
            for (size_t j = i + 1; j < m.weights.raw.size() && compared < 50; ++j) {
                const ai::Tensor<uint16_t>& x = m.weights.raw[i].second;
                const ai::Tensor<uint16_t>& y = m.weights.raw[j].second;
                if (x.shape != y.shape || x.numel() < 64) continue;

                ++compared;
                if (std::equal(x.ptr, x.ptr + 64, y.ptr)) {
                    std::cout << "    identical: " << m.weights.raw[i].first
                              << " and " << m.weights.raw[j].first << std::endl;
                    return false;
                }
            }

        std::cout << "    compared " << compared << " same-shaped pairs" << std::endl;
        return compared > 0;
    }());

    if (!logitsFor(1, b)) { check("the same seed reloads", false); return; }
    check("the same seed gives bit-identical logits", a == b);

    if (!logitsFor(2, c)) { check("another seed loads", false); return; }
    check("a different seed gives a different model", a != c);
}


int main(int argc, char** argv) {
    const std::string gpt2Dir   = argc > 1 ? argv[1] : "models/gpt2";
    const std::string qwenDir   = argc > 2 ? argv[2] : "models/qwen3-0.6b";
    const std::string gpt2Oracle = argc > 3 ? argv[3] : "out/oracle";
    const std::string qwenOracle = argc > 4 ? argv[4] : "out/qwen-oracle";

    testRegistry();
    testCores();
    testFp8();
    testGPT2(gpt2Dir, gpt2Oracle);
    testQwen3(qwenDir, qwenOracle);
    testQwen3Next(argc > 5 ? argv[5] : "models/tiny-qwen3next",
                  argc > 6 ? argv[6] : "out/qwen3next-oracle");
    /* both norm conventions: qwen3 stores a scale that must start at one,
       qwen3_next a deviation from one that must start at zero. A single
       model cannot tell the two apart, so both are exercised */
    testSynthetic(argc > 5 ? argv[5] : "models/tiny-qwen3next");
    testSynthetic(qwenDir);
    testPerturbed(argc > 5 ? argv[5] : "models/tiny-qwen3next");
    testPerturbed(qwenDir);
    testQuantRoundTrip();
    testQuantModel(qwenDir);

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    return failures;
}
