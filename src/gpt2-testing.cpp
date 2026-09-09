#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/ai/GPT2.hpp"

/* ========================================================================= //
    GPT-2 test suite.

    The important section is the last one: every intermediate of the C++
    forward pass is compared against the same intermediate dumped by the
    numpy reference (tools/gpt2_oracle.py). A transformer that is subtly
    wrong still produces fluent-looking text, so agreeing with the reference
    layer by layer is the only real evidence that it is right.

        py tools/gpt2_oracle.py --prompt "The capital of France is" --dump out/oracle
        make debug-gpt2 && ./bin/debug/gpt2-testing
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

static int failures = 0;

static std::string modelDir  = "models/gpt2";
static std::string oracleDir = "out/oracle";

/* the prompt the oracle was dumped for -- both sides must agree on it */
static const char* PROMPT = "The capital of France is";
static const std::vector<size_t> PROMPT_IDS = {464, 3139, 286, 4881, 318};

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] "
              << COLOR_RESET << label << std::endl;
    if (!pass) ++failures;
}

static void section(const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
}


/* ========================================================================= //
    Tokenizer
// ========================================================================= */

static void testTokenizer(const ai::GPT2& model) {
    section("tokenizer");

    const ai::Tokenizer& tok = model.tokenizer;

    check("the vocabulary is 50257 entries", tok.vocabSize() == 50257);
    check("the merge table loaded", tok.ranks.size() > 40000);

    check("the prompt encodes to the reference ids", tok.encode(PROMPT) == PROMPT_IDS);
    check("the prompt round-trips", tok.decode(tok.encode(PROMPT)) == PROMPT);

    /* a leading space belongs to the token after it, so these are different
       tokens for the same letters -- getting this wrong shifts every id */
    check("a leading space changes the token", tok.encode("the") != tok.encode(" the"));
    check("' the' is one token", tok.encode(" the").size() == 1);

    check("an empty string encodes to nothing", tok.encode("").empty());

    const char* samples[] = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "  leading and trailing spaces  ",
        "numbers 1234 and symbols @#$%",
        "line\nbreaks\tand\ttabs",
        "don't can't won't it's",
        "CamelCase and snake_case and kebab-case",
    };
    for (const char* s : samples)
        check(std::string("round-trips: ") + s, tok.decode(tok.encode(s)) == std::string(s));

    check("every id decodes to something", [&] {
        for (size_t id : tok.encode("The capital of France is Paris."))
            if (tok.decodeOne(id).empty()) return false;
        return true;
    }());
}


/* ========================================================================= //
    Weight loading
// ========================================================================= */

static void testWeights(const ai::GPT2& model) {
    section("weights");

    check("config says 12 layers", model.cfg.nLayer == 12);
    check("config says 12 heads", model.cfg.nHead == 12);
    check("config says d_model 768", model.cfg.nEmbd == 768);
    check("head width is 64", model.cfg.headDim() == 64);

    check("161 header entries, 160 tensors", model.file.count() == 160);

    check("wte is [50257,768]", model.wte.shape == std::vector<size_t>({50257, 768}));
    check("wpe is [1024,768]", model.wpe.shape == std::vector<size_t>({1024, 768}));
    check("ln_f.weight is [768]", model.lnfWeight.shape == std::vector<size_t>({768}));

    check("all 12 blocks loaded", model.blocks.size() == 12);
    check("c_attn is [768,2304] (Conv1D, in-major)",
          model.blocks[0].attnWeight.shape == std::vector<size_t>({768, 2304}));
    check("c_fc is [768,3072]",
          model.blocks[0].fcWeight.shape == std::vector<size_t>({768, 3072}));
    check("mlp c_proj is [3072,768]",
          model.blocks[0].mlpProjWeight.shape == std::vector<size_t>({3072, 768}));

    /*
        An F32 tensor is returned as a view aliasing the one file buffer
        whenever the bytes happen to be 4-aligned, and copied when they are
        not. This file's payload starts at byte 14291, which is not aligned,
        so every tensor here takes the copy path -- no allocation alignment
        can change that, the offset itself is odd. Assert whichever path
        applies rather than assuming.
    */
    const uint8_t* blobBegin = reinterpret_cast<const uint8_t*>(model.file.blob->data());
    const bool payloadAligned =
        reinterpret_cast<uintptr_t>(blobBegin + model.file.payloadBegin) % alignof(float) == 0;
    std::cout << "    payload " << (payloadAligned ? "IS" : "is NOT")
              << " float-aligned, so tensors are "
              << (payloadAligned ? "aliased" : "copied") << std::endl;

    check("an aliased tensor points inside the file buffer, a copied one does not", [&] {
        const uint8_t* wteBegin = reinterpret_cast<const uint8_t*>(model.wte.ptr);
        const bool inside = wteBegin >= blobBegin
                         && wteBegin < blobBegin + model.file.blob->size();
        return inside == payloadAligned;
    }());

    /* a default-constructed Tensor is rank 0, and numel() of a rank-0 tensor
       is 1 by convention -- so an absent tensor has to be detected by its
       null storage, never by numel() == 0 */
    check("a missing tensor comes back with null storage",
          model.file.tensor("nope").storage == nullptr);
    check("a missing tensor still reports numel 1 (rank-0 convention)",
          model.file.tensor("nope").numel() == 1);
    check("all tensors are contiguous", model.wte.isContiguous() && model.wpe.isContiguous());
}


/* ========================================================================= //
    Layer-by-layer diff against the numpy oracle
// ========================================================================= */

struct Reference {
    std::string         name;
    std::vector<size_t> shape;
    std::vector<float>  values;
};

static bool loadManifest(std::vector<Reference>& out) {
    const files::json::ParseResult manifest =
        files::json::parseFile(oracleDir + "/manifest.json");
    if (!manifest.ok || !manifest.value.isArray()) return false;

    for (size_t i = 0; i < manifest.value.size(); ++i) {
        Reference r;
        r.name  = manifest.value[i]["name"].asString();
        r.shape = manifest.value[i]["shape"].asSizes();

        size_t n = 1;
        for (size_t d : r.shape) n *= d;

        std::ifstream in(oracleDir + "/" + r.name + ".bin", std::ios::binary);
        if (!in) return false;

        r.values.resize(n);
        in.read(reinterpret_cast<char*>(r.values.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
        if (!in) return false;

        out.push_back(std::move(r));
    }

    return !out.empty();
}

static void testAgainstOracle(const ai::GPT2& model) {
    section("layer-by-layer diff against the numpy oracle");

    std::vector<Reference> refs;
    if (!loadManifest(refs)) {
        check("the oracle dump loads "
              "(run: py tools/gpt2_oracle.py --prompt \"" + std::string(PROMPT)
              + "\" --dump " + oracleDir + ")", false);
        return;
    }
    check("the oracle dump loads (" + std::to_string(refs.size()) + " tensors)", true);

    ai::Trace trace;
    ai::Tensor<float> logits = model.forward(PROMPT_IDS, false, &trace);

    check("the trace captured every stage the oracle dumped",
          trace.items.size() == refs.size());

    /*
        the reference computes in float64 in places, so an exact match is not
        the bar. Relative error against the magnitude of the tensor itself is,
        and it has to hold for EVERY stage -- a single wrong layer would
        otherwise be diluted by the eleven correct ones after it
    */
    size_t compared = 0;
    size_t mismatched = 0;
    double worst = 0.0;
    std::string worstName;

    for (const Reference& r : refs) {
        const ai::Tensor<float>* got = trace.find(r.name);
        if (!got) {
            std::cout << "    missing from trace: " << r.name << std::endl;
            ++mismatched;
            continue;
        }

        if (got->shape != r.shape) {
            std::cout << "    shape mismatch: " << r.name << std::endl;
            ++mismatched;
            continue;
        }

        double maxAbsRef = 1.0;
        double maxAbsErr = 0.0;
        for (size_t i = 0; i < r.values.size(); ++i) {
            maxAbsRef = std::max(maxAbsRef, static_cast<double>(std::fabs(r.values[i])));
            maxAbsErr = std::max(maxAbsErr,
                                 static_cast<double>(std::fabs(got->ptr[i] - r.values[i])));
        }

        const double relative = maxAbsErr / maxAbsRef;
        if (relative > worst) { worst = relative; worstName = r.name; }
        if (relative > 1e-4) {
            std::cout << "    " << r.name << "  relative error " << relative << std::endl;
            ++mismatched;
        }

        ++compared;
    }

    check("every stage compared (" + std::to_string(compared) + ")", compared == refs.size());
    check("no stage diverges from the reference", mismatched == 0);

    char note[128];
    std::snprintf(note, sizeof(note), "worst relative error is %.2e (%s)", worst, worstName.c_str());
    check(note, worst < 1e-4);

    // the end-to-end answer, independent of any intermediate
    check("argmax of the final logits is token 262 (' the')",
          ai::GPT2::argmax(logits.ptr + 4 * model.cfg.vocabSize, model.cfg.vocabSize) == 262);

    /* lastOnly must give bit-identical results to the full pass, since it is
       purely an optimisation that skips rows nothing reads */
    check("lastOnly matches the last row of the full pass", [&] {
        ai::Tensor<float> last = model.forward(PROMPT_IDS, true);
        if (last.shape != std::vector<size_t>({1, model.cfg.vocabSize})) return false;
        const float* full = logits.ptr + 4 * model.cfg.vocabSize;
        for (size_t i = 0; i < model.cfg.vocabSize; ++i)
            if (std::fabs(last.ptr[i] - full[i]) > 1e-3f) return false;
        return true;
    }());
}


/* ========================================================================= //
    Generation
// ========================================================================= */

static void testGeneration(const ai::GPT2& model) {
    section("generation");

    /* temperature 0 is greedy, so this is deterministic and must reproduce
       what the reference produced for the same prompt */
    const std::string greedy = model.generate(PROMPT, 8, 0.0f, 1, 0);
    std::cout << "    greedy: " << greedy << std::endl;

    check("greedy generation extends the prompt", greedy.rfind(PROMPT, 0) == 0);
    check("greedy generation matches the reference continuation",
          greedy == "The capital of France is the capital of the French Republic, and");

    check("greedy is deterministic", model.generate(PROMPT, 5, 0.0f, 1, 0)
                                  == model.generate(PROMPT, 5, 0.0f, 1, 0));

    check("the same seed gives the same sample", model.generate(PROMPT, 5, 0.8f, 40, 7)
                                              == model.generate(PROMPT, 5, 0.8f, 40, 7));

    const std::string sampled = model.generate(PROMPT, 12, 0.8f, 40, 42);
    std::cout << "    sampled: " << sampled << std::endl;
    check("sampled generation extends the prompt", sampled.rfind(PROMPT, 0) == 0);
    check("sampled generation produces new text", sampled.size() > std::string(PROMPT).size());
}


int main(int argc, char** argv) {
    if (argc > 1) modelDir = argv[1];
    if (argc > 2) oracleDir = argv[2];

    ai::GPT2 model;
    std::cout << "loading " << modelDir << " ..." << std::endl;
    if (!model.load(modelDir)) {
        std::cout << COLOR_RED << "[FAIL] " << COLOR_RESET
                  << "model failed to load: " << model.error << std::endl;
        return 1;
    }
    std::cout << "loaded." << std::endl;

    testTokenizer(model);
    testWeights(model);
    testAgainstOracle(model);
    testGeneration(model);

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    return failures;
}
