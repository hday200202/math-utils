#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/ai/LocalModel.hpp"

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


int main(int argc, char** argv) {
    const std::string gpt2Dir   = argc > 1 ? argv[1] : "models/gpt2";
    const std::string qwenDir   = argc > 2 ? argv[2] : "models/qwen3-0.6b";
    const std::string gpt2Oracle = argc > 3 ? argv[3] : "out/oracle";
    const std::string qwenOracle = argc > 4 ? argv[4] : "out/qwen-oracle";

    testRegistry();
    testGPT2(gpt2Dir, gpt2Oracle);
    testQwen3(qwenDir, qwenOracle);

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    return failures;
}
