#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "BPE.hpp"
#include "Safetensors.hpp"
#include "Tensor.hpp"
#include "../files/JSON.hpp"
#include "../math/Random.hpp"

/*
    GPT-2 inference.

    The architecture is one idea repeated: a RESIDUAL STREAM. x is a
    fixed-width bus, one vector per token, that every sub-block reads from and
    adds back into. Nothing ever replaces it:

        x = x + attention(layernorm(x))
        x = x + feedforward(layernorm(x))

    twelve times over, then a final norm and a projection to vocabulary
    logits. Everything else here is detail about what those two functions are.

    Forward only. There is no backward pass, no optimiser and no autograd,
    because running someone else's trained weights needs none of it.

    Two things about GPT-2 specifically that do not generalise to newer models:
    - its linear layers are stored as Conv1D, so the weights are [in, out] and
      the product is x @ W, not the x @ W.T that a normal Linear would need.
      This is the single most common porting bug in the model
    - the output projection is TIED to the token embedding table, so there is
      no lm_head tensor; the same [vocab, d_model] matrix is reused
*/

namespace ai {

/* ========================================================================= //
    Config
// ========================================================================= */

struct GPT2Config {
    size_t nLayer      = 12;
    size_t nHead       = 12;
    size_t nEmbd       = 768;
    size_t vocabSize   = 50257;
    size_t nPositions  = 1024;
    float  layerNormEpsilon = 1e-5f;

    size_t headDim() const { return nEmbd / nHead; }

    bool loadFrom(const files::json::Value& c) {
        if (!c.isObject()) return false;
        nLayer     = c["n_layer"].asSize(nLayer);
        nHead      = c["n_head"].asSize(nHead);
        nEmbd      = c["n_embd"].asSize(nEmbd);
        vocabSize  = c["vocab_size"].asSize(vocabSize);
        nPositions = c["n_positions"].asSize(nPositions);
        layerNormEpsilon = static_cast<float>(c["layer_norm_epsilon"].asDouble(layerNormEpsilon));
        return nLayer > 0 && nHead > 0 && nEmbd > 0 && nEmbd % nHead == 0;
    }
};


/* ========================================================================= //
    Local math helpers

    These live here rather than in Tensor because they exist for the shapes
    this model happens to need.
// ========================================================================= */

/*
    scale(x, k):
    - attention divides the scores by sqrt(headDim) to keep the softmax out of
      its saturated region as the head width grows. Applied to Q before the
      product rather than to the scores after, which is the same thing for one
      fewer pass over the larger matrix
*/
inline Tensor<float> scale(const Tensor<float>& x, float k) {
    Tensor<float> src = x.contiguous();
    Tensor<float> out(src.shape);
    for (size_t i = 0; i < src.numel(); ++i) out.ptr[i] = src.ptr[i] * k;
    return out;
}

/*
    matmulNT(a, b):
    - C[M,N] = A[M,K] @ B[N,K]^T, with B stored row-major as [N,K].
    - the point is to avoid materialising the transpose. The vocabulary
      projection multiplies by a [50257, 768] table; transposing it through
      contiguous() would allocate and copy 154MB on every single forward.
    - both operands are walked along their rows here, so this is also the
      cache-friendliest of the three possible orientations
*/
inline Tensor<float> matmulNT(const Tensor<float>& a, const Tensor<float>& b) {
    Tensor<float> A = a.contiguous();
    Tensor<float> B = b.contiguous();

    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.shape[1] == B.shape[1] && "matmulNT: inner dimension mismatch");

    const size_t M = A.shape[0];
    const size_t K = A.shape[1];
    const size_t N = B.shape[0];

    Tensor<float> out({M, N});

    for (size_t i = 0; i < M; ++i) {
        const float* arow = A.ptr + i * K;
        float* crow = out.ptr + i * N;

        for (size_t j = 0; j < N; ++j) {
            const float* brow = B.ptr + j * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) sum += arow[k] * brow[k];
            crow[j] = sum;
        }
    }

    return out;
}


/* ========================================================================= //
    Trace

    Captures every intermediate the numpy reference dumps, under the same
    names, so the C++ can be diffed stage by stage. A transformer that is
    subtly wrong does not crash, it produces fluent garbage - comparing
    layer by layer against a known-good reference is the only practical way
    to find that.
// ========================================================================= */

struct Trace {
    std::vector<std::pair<std::string, Tensor<float>>> items;

    void add(const std::string& name, const Tensor<float>& t) {
        items.emplace_back(name, t.clone());
    }

    const Tensor<float>* find(const std::string& name) const {
        for (const auto& i : items) if (i.first == name) return &i.second;
        return nullptr;
    }
};


/* ========================================================================= //
    Weights
// ========================================================================= */

struct GPT2Block {
    Tensor<float> ln1Weight, ln1Bias;
    Tensor<float> attnWeight, attnBias;         // c_attn: [nEmbd, 3*nEmbd]
    Tensor<float> attnProjWeight, attnProjBias; // c_proj: [nEmbd, nEmbd]
    Tensor<float> ln2Weight, ln2Bias;
    Tensor<float> fcWeight, fcBias;             // c_fc:   [nEmbd, 4*nEmbd]
    Tensor<float> mlpProjWeight, mlpProjBias;   // c_proj: [4*nEmbd, nEmbd]
};


/* ========================================================================= //
    Model
// ========================================================================= */

struct GPT2 {
    GPT2Config             cfg;
    Safetensors            file;
    Tokenizer              tokenizer;

    Tensor<float>          wte;                 // [vocab, nEmbd], also the lm_head
    Tensor<float>          wpe;                 // [nPositions, nEmbd]
    std::vector<GPT2Block> blocks;
    Tensor<float>          lnfWeight, lnfBias;

    std::string            error;


    /*
        load(directory):
        - expects the HuggingFace layout: config.json, model.safetensors,
          vocab.json, merges.txt. The topology is read from config.json rather
          than inferred, because the file states it outright
    */
    bool load(const fs::path& directory) {
        error.clear();

        const files::json::ParseResult config =
            files::json::parseFile(directory / "config.json");
        if (!config.ok) { error = "config.json: " + config.error; return false; }
        if (!cfg.loadFrom(config.value)) { error = "config.json: bad or missing fields"; return false; }

        if (!file.load(directory / "model.safetensors")) { error = file.error; return false; }

        if (!tokenizer.load(directory / "vocab.json", directory / "merges.txt")) {
            error = "tokenizer: " + tokenizer.error;
            return false;
        }

        wte = file.tensor("wte.weight");
        wpe = file.tensor("wpe.weight");
        lnfWeight = file.tensor("ln_f.weight");
        lnfBias   = file.tensor("ln_f.bias");

        if (!wte.storage || !wpe.storage) { error = "missing wte/wpe"; return false; }

        blocks.resize(cfg.nLayer);
        for (size_t i = 0; i < cfg.nLayer; ++i) {
            const std::string p = "h." + std::to_string(i);
            GPT2Block& b = blocks[i];

            b.ln1Weight      = file.tensor(p + ".ln_1.weight");
            b.ln1Bias        = file.tensor(p + ".ln_1.bias");
            b.attnWeight     = file.tensor(p + ".attn.c_attn.weight");
            b.attnBias       = file.tensor(p + ".attn.c_attn.bias");
            b.attnProjWeight = file.tensor(p + ".attn.c_proj.weight");
            b.attnProjBias   = file.tensor(p + ".attn.c_proj.bias");
            b.ln2Weight      = file.tensor(p + ".ln_2.weight");
            b.ln2Bias        = file.tensor(p + ".ln_2.bias");
            b.fcWeight       = file.tensor(p + ".mlp.c_fc.weight");
            b.fcBias         = file.tensor(p + ".mlp.c_fc.bias");
            b.mlpProjWeight  = file.tensor(p + ".mlp.c_proj.weight");
            b.mlpProjBias    = file.tensor(p + ".mlp.c_proj.bias");

            if (!b.attnWeight.storage || !b.fcWeight.storage) {
                error = "missing weights for block " + std::to_string(i);
                return false;
            }
        }

        return true;
    }


    /*
        attention(h, block, seq):
        - h is [seq, nEmbd] and already normalised.
        - one projection produces Q, K and V stacked into [seq, 3*nEmbd]; they
          are split apart as views, no copy.
        - each is then split across heads: [seq, nEmbd] reshaped to
          [seq, nHead, headDim] and permuted to [nHead, seq, headDim], so the
          batched matmul treats every head as an independent problem. Reshape
          BEFORE permute - the other order would need a reshape of
          non-contiguous data, which reads the wrong elements
    */
    Tensor<float> attention(const Tensor<float>& h, const GPT2Block& b, size_t seq,
                            const std::string& prefix, Trace* trace) const {
        const size_t nHead = cfg.nHead;
        const size_t dHead = cfg.headDim();

        Tensor<float> qkv = addBias(matmul(h, b.attnWeight), b.attnBias);
        if (trace) trace->add(prefix + ".qkv", qkv);
        std::vector<Tensor<float>> parts = split(qkv, 3, 1);

        Tensor<float> q = parts[0].contiguous().reshape({seq, nHead, dHead}).permute({1, 0, 2}).contiguous();
        Tensor<float> k = parts[1].contiguous().reshape({seq, nHead, dHead}).permute({1, 0, 2}).contiguous();
        Tensor<float> v = parts[2].contiguous().reshape({seq, nHead, dHead}).permute({1, 0, 2}).contiguous();

        q = scale(q, 1.0f / std::sqrt(static_cast<float>(dHead)));

        Tensor<float> scores = matmul(q, k.transpose());        // [nHead, seq, seq]
        scores = causalMask(scores);
        if (trace) trace->add(prefix + ".scores", scores);

        Tensor<float> weights = softmaxLastAxis(scores);
        if (trace) trace->add(prefix + ".attnw", weights);

        Tensor<float> out = matmul(weights, v);                 // [nHead, seq, dHead]
        out = out.permute({1, 0, 2}).contiguous().reshape({seq, cfg.nEmbd});

        out = addBias(matmul(out, b.attnProjWeight), b.attnProjBias);
        if (trace) trace->add(prefix + ".attnout", out);
        return out;
    }


    Tensor<float> feedForward(const Tensor<float>& h, const GPT2Block& b,
                              const std::string& prefix, Trace* trace) const {
        Tensor<float> fc = addBias(matmul(h, b.fcWeight), b.fcBias);
        if (trace) trace->add(prefix + ".fc", fc);

        Tensor<float> up = gelu(fc);
        if (trace) trace->add(prefix + ".gelu", up);

        Tensor<float> out = addBias(matmul(up, b.mlpProjWeight), b.mlpProjBias);
        if (trace) trace->add(prefix + ".mlpout", out);
        return out;
    }


    /*
        forward(ids, lastOnly):
        - returns logits, [seq, vocab] normally or [1, vocab] when lastOnly is
          set. Generation only ever reads the final row, and the vocabulary
          projection is the single largest matmul in the model, so skipping
          the other rows is most of the cost of a token
    */
    Tensor<float> forward(const std::vector<size_t>& ids, bool lastOnly = false,
                          Trace* trace = nullptr) const {
        const size_t seq = ids.size();
        assert(seq > 0 && seq <= cfg.nPositions);

        Tensor<float> x = gather(wte, ids);

        // learned absolute positions, simply added in. Newer models rotate Q
        // and K by position instead, which generalises past the trained length
        Tensor<float> pos = wpe.slice(0, 0, seq).contiguous();
        x = add(x, pos);
        if (trace) trace->add("embed", x);

        for (size_t i = 0; i < cfg.nLayer; ++i) {
            const GPT2Block& b = blocks[i];
            const std::string p = "h." + std::to_string(i);

            Tensor<float> h = layernorm(x, b.ln1Weight, b.ln1Bias, cfg.layerNormEpsilon);
            if (trace) trace->add(p + ".ln1", h);

            x = add(x, attention(h, b, seq, p, trace));
            if (trace) trace->add(p + ".res1", x);

            h = layernorm(x, b.ln2Weight, b.ln2Bias, cfg.layerNormEpsilon);
            if (trace) trace->add(p + ".ln2", h);

            x = add(x, feedForward(h, b, p, trace));
            if (trace) trace->add(p + ".res2", x);
        }

        x = layernorm(x, lnfWeight, lnfBias, cfg.layerNormEpsilon);
        if (trace) trace->add("lnf", x);

        if (lastOnly) x = x.slice(0, seq - 1, seq).contiguous();

        // tied weights: the embedding table doubles as the output projection
        Tensor<float> logits = matmulNT(x, wte);
        if (trace) trace->add("logits", logits);
        return logits;
    }


    /* ===================================================================== //
        Sampling
    // ===================================================================== */

    static size_t argmax(const float* row, size_t n) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i) if (row[i] > row[best]) best = i;
        return best;
    }

    /*
        sampleTopK(row, n, temperature, topK, rng):
        - temperature divides the logits before the softmax: below 1 sharpens
          the distribution towards the argmax, above 1 flattens it.
        - only the topK most likely tokens are kept, which is what stops a
          long tail of individually-unlikely tokens from collectively
          derailing the text
    */
    static size_t sampleTopK(const float* row, size_t n, float temperature,
                             size_t topK, math::random::Random& rng) {
        if (temperature <= 0.0f || topK <= 1) return argmax(row, n);

        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;

        const size_t keep = std::min(topK, n);
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep), order.end(),
                          [row](size_t a, size_t b) { return row[a] > row[b]; });
        order.resize(keep);

        float best = row[order[0]] / temperature;
        std::vector<float> probs(keep);
        float total = 0.0f;
        for (size_t i = 0; i < keep; ++i) {
            probs[i] = std::exp(row[order[i]] / temperature - best);
            total += probs[i];
        }

        float pick = static_cast<float>(rng.randDouble(0.0, 1.0)) * total;
        for (size_t i = 0; i < keep; ++i) {
            pick -= probs[i];
            if (pick <= 0.0f) return order[i];
        }
        return order.back();
    }


    /*
        generate(prompt, count, temperature, topK, seed):
        - the whole prompt runs through the model again for every new token.
          A KV cache would keep the keys and values of the tokens already seen
          and only compute the new column, which is the difference between
          this and something usable at length. Correctness first
    */
    std::string generate(const std::string& prompt, size_t count,
                         float temperature = 0.8f, size_t topK = 40,
                         uint64_t seed = 0) const {
        math::random::Random rng{seed};

        std::vector<size_t> ids = tokenizer.encode(prompt);
        if (ids.empty()) ids.push_back(tokenizer.encoder.count("<|endoftext|>")
                                       ? static_cast<size_t>(tokenizer.encoder.at("<|endoftext|>"))
                                       : 0);

        for (size_t i = 0; i < count && ids.size() < cfg.nPositions; ++i) {
            Tensor<float> logits = forward(ids, true);
            ids.push_back(sampleTopK(logits.ptr, cfg.vocabSize, temperature, topK, rng));
        }

        return tokenizer.decode(ids);
    }
};

} // !namespace ai
