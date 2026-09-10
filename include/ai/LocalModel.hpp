#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "BPE.hpp"
#include "Kernels.hpp"
#include "Safetensors.hpp"
#include "Tensor.hpp"
#include "../files/JSON.hpp"
#include "../math/Random.hpp"

/*
    One loader and one forward pass for any dense decoder-only transformer.

    Every such model is the same residual stream - each block reads the bus,
    computes, and adds back - and they differ only in which pieces are dropped
    into four slots: how it normalises, how it encodes position, what its feed
    forward looks like, and how its weights happen to be laid out on disk.
    Those differences are DATA, held in an ArchSpec, not code.

    The trick that keeps the hot loop clean is that most of the variation is
    absorbed at load time and then never mentioned again:

        Conv1D [in,out] vs Linear [out,in]  transposed once at load
        fused qkv_proj vs split q/k/v       sliced into views, no copy
        fused gate_up_proj vs split         same
        tensor naming                       a table of format strings
        head_dim stated vs derived          resolved into the config
        tied vs separate lm_head            resolved to one pointer
        missing q_norm/k_norm               a null tensor that is skipped

    What is left is four enum branches evaluated once per layer, against
    matmuls that take milliseconds. They cost nothing.

    Covered: gpt2, llama, mistral, qwen2, qwen3, phi3, and anything else that
    is a dense decoder with the same parts.

    NOT covered, deliberately: mixture-of-experts routing, linear or hybrid
    attention, state-space blocks, and multimodal towers. Those are different
    machines, not different settings, and pretending otherwise is how an
    abstraction rots.
*/

namespace ai {

/* ========================================================================= //
    The axes of variation
// ========================================================================= */

enum class NormKind     { LayerNorm, RMSNorm };
enum class PosKind      { Learned, RoPE };
enum class MLPKind      { TwoMatrix, SwiGLU };
enum class ActKind      { GeluTanh, SiLU };
enum class LinearLayout { OutIn, InOut };      // nn.Linear vs GPT-2's Conv1D


/*
    NameMap:
    - every tensor name the loader might want, as a format string. "{L}" is
      replaced with the layer index. An empty entry means the model does not
      have that tensor, which is how optional pieces (biases, QK-norm, a
      separate lm_head) are expressed without a flag for each
*/
struct NameMap {
    std::string embed, posEmbed, lmHead;
    std::string finalNorm, finalNormBias;

    std::string inputNorm, inputNormBias;
    std::string postAttnNorm, postAttnNormBias;

    std::string q, k, v, o, qkv;               // qkv when the three are fused
    std::string qBias, kBias, vBias, oBias, qkvBias;
    std::string qNorm, kNorm;

    std::string gate, up, down, gateUp;        // gateUp when gate and up fuse
    std::string upBias, downBias;
};


/*
    ConfigKeys:
    - config.json does not agree on field names either. GPT-2 says n_layer
      where everything since says num_hidden_layers
*/
struct ConfigKeys {
    std::string nLayer   = "num_hidden_layers";
    std::string nHead    = "num_attention_heads";
    std::string nKVHead  = "num_key_value_heads";
    std::string hidden   = "hidden_size";
    std::string headDim  = "head_dim";
    std::string interm   = "intermediate_size";
    std::string vocab    = "vocab_size";
    std::string maxPos   = "max_position_embeddings";
    std::string normEps  = "rms_norm_eps";
    std::string ropeBase = "rope_theta";
    std::string tied     = "tie_word_embeddings";
};


struct ArchSpec {
    std::string  modelType;

    NormKind     norm   = NormKind::RMSNorm;
    PosKind      pos    = PosKind::RoPE;
    MLPKind      mlp    = MLPKind::SwiGLU;
    ActKind      act    = ActKind::SiLU;
    LinearLayout layout = LinearLayout::OutIn;

    bool         qkNorm     = false;
    bool         tiedByDefault = false;

    ConfigKeys   keys;
    NameMap      names;

    /* the numpy references dump under different names, so the trace follows
       whichever this model's oracle uses */
    std::string  tracePrefix    = "l{L}.";
    std::string  traceFinalNorm = "norm";
};


/* expand("model.layers.{L}.mlp.up_proj.weight", 7) */
inline std::string expandName(const std::string& tmpl, size_t layer) {
    const size_t at = tmpl.find("{L}");
    if (at == std::string::npos) return tmpl;
    return tmpl.substr(0, at) + std::to_string(layer) + tmpl.substr(at + 3);
}


/* ========================================================================= //
    The registry
// ========================================================================= */

/*
    llamaFamily():
    - the shape everything modern shares: RMSNorm, RoPE, SwiGLU, GQA, split
      projections, no biases anywhere. llama, mistral, qwen2 and qwen3 differ
      from this only in numbers and in whether QK-norm is present
*/
inline ArchSpec llamaFamily(const std::string& type) {
    ArchSpec s;
    s.modelType = type;

    s.names.embed         = "model.embed_tokens.weight";
    s.names.lmHead        = "lm_head.weight";
    s.names.finalNorm     = "model.norm.weight";

    s.names.inputNorm     = "model.layers.{L}.input_layernorm.weight";
    s.names.postAttnNorm  = "model.layers.{L}.post_attention_layernorm.weight";

    s.names.q             = "model.layers.{L}.self_attn.q_proj.weight";
    s.names.k             = "model.layers.{L}.self_attn.k_proj.weight";
    s.names.v             = "model.layers.{L}.self_attn.v_proj.weight";
    s.names.o             = "model.layers.{L}.self_attn.o_proj.weight";

    s.names.gate          = "model.layers.{L}.mlp.gate_proj.weight";
    s.names.up            = "model.layers.{L}.mlp.up_proj.weight";
    s.names.down          = "model.layers.{L}.mlp.down_proj.weight";

    return s;
}

inline ArchSpec qwen3Spec() {
    ArchSpec s = llamaFamily("qwen3");
    // the one thing Qwen3 adds: RMSNorm per head on Q and K, before RoPE
    s.qkNorm      = true;
    s.names.qNorm = "model.layers.{L}.self_attn.q_norm.weight";
    s.names.kNorm = "model.layers.{L}.self_attn.k_norm.weight";
    return s;
}

inline ArchSpec qwen2Spec() {
    ArchSpec s = llamaFamily("qwen2");
    // Qwen2 is the same but carries biases on the attention projections
    s.names.qBias = "model.layers.{L}.self_attn.q_proj.bias";
    s.names.kBias = "model.layers.{L}.self_attn.k_proj.bias";
    s.names.vBias = "model.layers.{L}.self_attn.v_proj.bias";
    return s;
}

inline ArchSpec phi3Spec() {
    ArchSpec s = llamaFamily("phi3");
    // Phi-3 fuses q/k/v into one matrix and gate/up into another
    s.names.q = s.names.k = s.names.v = "";
    s.names.qkv    = "model.layers.{L}.self_attn.qkv_proj.weight";
    s.names.gate   = s.names.up = "";
    s.names.gateUp = "model.layers.{L}.mlp.gate_up_proj.weight";
    return s;
}

/*
    gpt2Spec():
    - the odd one out, and the reason this abstraction is worth trusting.
      Conv1D weights stored [in,out], learned absolute positions rather than
      RoPE, LayerNorm with a bias rather than RMSNorm, a fused c_attn, a
      two-matrix GELU feed forward, and biases on everything
*/
inline ArchSpec gpt2Spec() {
    ArchSpec s;
    s.modelType = "gpt2";

    s.norm   = NormKind::LayerNorm;
    s.pos    = PosKind::Learned;
    s.mlp    = MLPKind::TwoMatrix;
    s.act    = ActKind::GeluTanh;
    s.layout = LinearLayout::InOut;
    s.tiedByDefault = true;

    s.keys.nLayer  = "n_layer";
    s.keys.nHead   = "n_head";
    s.keys.nKVHead = "n_head";
    s.keys.hidden  = "n_embd";
    s.keys.headDim = "";                      // derived from hidden / heads
    s.keys.interm  = "";                      // 4 * hidden
    s.keys.maxPos  = "n_positions";
    s.keys.normEps = "layer_norm_epsilon";
    s.keys.ropeBase = "";

    s.names.embed          = "wte.weight";
    s.names.posEmbed       = "wpe.weight";
    s.names.lmHead         = "";              // tied to wte
    s.names.finalNorm      = "ln_f.weight";
    s.names.finalNormBias  = "ln_f.bias";

    s.names.inputNorm         = "h.{L}.ln_1.weight";
    s.names.inputNormBias     = "h.{L}.ln_1.bias";
    s.names.postAttnNorm      = "h.{L}.ln_2.weight";
    s.names.postAttnNormBias  = "h.{L}.ln_2.bias";

    s.names.qkv      = "h.{L}.attn.c_attn.weight";
    s.names.qkvBias  = "h.{L}.attn.c_attn.bias";
    s.names.o        = "h.{L}.attn.c_proj.weight";
    s.names.oBias    = "h.{L}.attn.c_proj.bias";

    s.names.up       = "h.{L}.mlp.c_fc.weight";
    s.names.upBias   = "h.{L}.mlp.c_fc.bias";
    s.names.down     = "h.{L}.mlp.c_proj.weight";
    s.names.downBias = "h.{L}.mlp.c_proj.bias";

    s.tracePrefix    = "h.{L}.";
    s.traceFinalNorm = "lnf";

    return s;
}

inline const std::vector<ArchSpec>& specRegistry() {
    static const std::vector<ArchSpec> all = {
        gpt2Spec(),
        qwen3Spec(),
        qwen2Spec(),
        phi3Spec(),
        llamaFamily("llama"),
        llamaFamily("mistral"),
    };
    return all;
}

inline const ArchSpec* specFor(const std::string& modelType) {
    for (const ArchSpec& s : specRegistry()) if (s.modelType == modelType) return &s;
    return nullptr;
}


/* ========================================================================= //
    Config
// ========================================================================= */

struct LocalConfig {
    size_t nLayer = 0, nHead = 0, nKVHead = 0, headDim = 0;
    size_t hidden = 0, intermediate = 0, vocab = 0, maxPositions = 2048;
    float  normEps = 1e-5f;
    float  ropeTheta = 10000.0f;
    bool   tied = false;

    size_t qWidth()  const { return nHead * headDim; }
    size_t kvWidth() const { return nKVHead * headDim; }
    size_t group()   const { return nHead / nKVHead; }

    bool loadFrom(const files::json::Value& c, const ArchSpec& spec) {
        const ConfigKeys& k = spec.keys;

        nLayer  = c[k.nLayer].asSize(0);
        nHead   = c[k.nHead].asSize(0);
        hidden  = c[k.hidden].asSize(0);
        vocab   = c[k.vocab].asSize(0);
        maxPositions = c[k.maxPos].asSize(2048);

        // multi-head is grouped-query with one group, so there is one path
        nKVHead = c[k.nKVHead].asSize(nHead);

        headDim = k.headDim.empty() ? 0 : c[k.headDim].asSize(0);
        if (headDim == 0 && nHead > 0) headDim = hidden / nHead;

        intermediate = k.interm.empty() ? 0 : c[k.interm].asSize(0);
        if (intermediate == 0) intermediate = 4 * hidden;

        normEps   = static_cast<float>(c[k.normEps].asDouble(1e-5));
        ropeTheta = k.ropeBase.empty() ? 10000.0f
                                       : static_cast<float>(c[k.ropeBase].asDouble(10000.0));
        tied = c[k.tied].asBool(spec.tiedByDefault);

        return nLayer > 0 && nHead > 0 && nKVHead > 0 && headDim > 0
            && hidden > 0 && vocab > 0 && nHead % nKVHead == 0;
    }
};


/* ========================================================================= //
    Weights

    A weight is bfloat16 aliasing the file when the checkpoint is bfloat16,
    and float otherwise. Holding both and dispatching on which is populated
    keeps one call site rather than templating the whole model on dtype.
// ========================================================================= */

struct Weight {
    Tensor<float>    f32;
    Tensor<uint16_t> bf16;
    Tensor<float>    bias;                // empty when the model has none

    size_t outDim = 0, inDim = 0;

    bool valid()  const { return f32.storage != nullptr || bf16.storage != nullptr; }
    bool isBF16() const { return bf16.storage != nullptr; }
    bool hasBias() const { return bias.storage != nullptr; }
};


/*
    apply(x, W, out, rows):
    - out[rows, outDim] = x[rows, inDim] @ W^T, plus the bias if there is one.
      Both weight layouts have already been normalised to [out, in] at load,
      so this is the only orientation the kernels ever see
*/
inline void apply(const float* x, const Weight& W, float* out, size_t rows) {
    if (W.isBF16())
        kernels::matmulNTBF16(x, W.bf16.ptr, out, rows, W.inDim, W.outDim);
    else if (W.hasBias())
        kernels::matmulNTBias(x, W.f32.ptr, W.bias.ptr, out, rows, W.inDim, W.outDim);
    else
        kernels::matmulNT(x, W.f32.ptr, out, rows, W.inDim, W.outDim);

    // the bf16 path folds no bias, so add it here when both apply
    if (W.isBF16() && W.hasBias())
        for (size_t r = 0; r < rows; ++r)
            for (size_t j = 0; j < W.outDim; ++j)
                out[r * W.outDim + j] += W.bias.ptr[j];
}


struct LocalLayer {
    Tensor<float> inputNorm, inputNormBias;
    Tensor<float> postAttnNorm, postAttnNormBias;
    Tensor<float> qNorm, kNorm;
    Weight q, k, v, o;
    Weight gate, up, down;
};


/* ========================================================================= //
    Trace
// ========================================================================= */

struct LocalTrace {
    std::vector<std::pair<std::string, std::vector<float>>> items;

    void add(const std::string& name, const float* data, size_t n) {
        items.emplace_back(name, std::vector<float>(data, data + n));
    }

    const std::vector<float>* find(const std::string& name) const {
        for (const auto& i : items) if (i.first == name) return &i.second;
        return nullptr;
    }
};


/* ========================================================================= //
    KV cache
// ========================================================================= */

struct LocalCache {
    std::vector<std::vector<float>> keys, values;
    size_t length = 0, capacity = 0;

    void resize(const LocalConfig& cfg, size_t maxSeq) {
        capacity = maxSeq;
        length = 0;
        keys.assign(cfg.nLayer, std::vector<float>(maxSeq * cfg.kvWidth(), 0.0f));
        values.assign(cfg.nLayer, std::vector<float>(maxSeq * cfg.kvWidth(), 0.0f));
    }

    void reset() { length = 0; }
};


/* ========================================================================= //
    Shared primitives
// ========================================================================= */

inline void rmsNormRow(const float* x, const float* w, float* out, size_t n, float eps) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i] * x[i];

    const float inv = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

/*
    layerNormRow:
    - centres as well as scales, and carries a bias. The variance is the
      biased one, dividing by n rather than n-1
*/
inline void layerNormRow(const float* x, const float* w, const float* b,
                         float* out, size_t n, float eps) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += x[i];
    mean /= static_cast<float>(n);

    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) { const float d = x[i] - mean; var += d * d; }
    var /= static_cast<float>(n);

    const float inv = 1.0f / std::sqrt(var + eps);
    for (size_t i = 0; i < n; ++i) out[i] = (x[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}

inline float siluOf(float x)  { return x / (1.0f + std::exp(-x)); }

inline float geluTanhOf(float x) {
    const float k = 0.7978845608028654f;     // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

inline void ropeRow(float* vec, size_t heads, size_t headDim, size_t position, float theta) {
    const size_t half = headDim / 2;

    for (size_t j = 0; j < half; ++j) {
        const float freq =
            1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(headDim));
        const float angle = static_cast<float>(position) * freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);

        for (size_t h = 0; h < heads; ++h) {
            float* head = vec + h * headDim;
            const float a = head[j];
            const float b = head[j + half];
            head[j]        = a * c - b * s;
            head[j + half] = b * c + a * s;
        }
    }
}


/* ========================================================================= //
    LocalModel
// ========================================================================= */

struct LocalModel {
    ArchSpec                 spec;
    LocalConfig              cfg;
    ModelWeights             weights;
    Tokenizer                tokenizer;

    Weight                   embed, lmHead;
    Tensor<float>            posEmbed;                 // learned positions only
    Tensor<float>            finalNorm, finalNormBias;
    std::vector<LocalLayer>  layers;

    std::string              modelType;
    std::string              error;


    /* ===================================================================== //
        Loading
    // ===================================================================== */

    bool load(const fs::path& directory, const LoadProgress& onProgress = {}) {
        error.clear();

        const files::json::ParseResult config =
            files::json::parseFile(directory / "config.json");
        if (!config.ok) { error = "config.json: " + config.error; return false; }

        const files::json::Value& root = config.value;
        const files::json::Value& text =
            root["text_config"].isObject() ? root["text_config"] : root;

        modelType = root["model_type"].asString(text["model_type"].asString());
        const ArchSpec* found = specFor(modelType);
        if (!found) {
            error = "unsupported model_type \"" + modelType
                  + "\" (known: gpt2, llama, mistral, qwen2, qwen3, phi3)";
            return false;
        }
        spec = *found;

        if (!cfg.loadFrom(text, spec)) { error = "config.json: bad or missing fields"; return false; }

        // bfloat16 checkpoints are kept as-is; anything else is widened
        if (!weights.load(directory, false, onProgress)) { error = weights.error; return false; }
        if (!loadTokenizer(directory))       { return false; }

        /*
            the per-layer fetch is where a Conv1D checkpoint materialises its
            transposes, so it can allocate too and is worth guarding as well
        */
        try {
            if (!loadTopLevel()) return false;
            if (!loadLayers())   return false;
        } catch (const std::bad_alloc&) {
            error = "out of memory arranging the weights";
            return false;
        }

        return true;
    }


    bool loadTokenizer(const fs::path& directory) {
        if (fs::exists(directory / "tokenizer.json")) {
            if (tokenizer.loadJson(directory / "tokenizer.json")) return true;
            error = "tokenizer.json: " + tokenizer.error;
            return false;
        }

        if (fs::exists(directory / "vocab.json") && fs::exists(directory / "merges.txt")) {
            if (tokenizer.load(directory / "vocab.json", directory / "merges.txt")) return true;
            error = "vocab.json/merges.txt: " + tokenizer.error;
            return false;
        }

        error = "no tokenizer.json or vocab.json+merges.txt in " + directory.string();
        return false;
    }


    /*
        fetch(name):
        - a matrix in whichever dtype it was stored, with the Conv1D layouts
          already transposed into [out, in]. Everything downstream sees one
          orientation
    */
    Weight fetch(const std::string& name) const {
        Weight w;
        if (name.empty() || !weights.contains(name)) return w;

        Tensor<uint16_t> raw = weights.rawBF16(name);
        if (raw.storage) {
            w.bf16 = (spec.layout == LinearLayout::InOut) ? transposed(raw) : raw;
        } else {
            Tensor<float> f = weights.tensor(name);
            if (!f.storage) return w;
            w.f32 = (spec.layout == LinearLayout::InOut) ? transposed(f) : f;
        }

        const std::vector<size_t>& shape = w.isBF16() ? w.bf16.shape : w.f32.shape;
        if (shape.size() == 2) { w.outDim = shape[0]; w.inDim = shape[1]; }
        return w;
    }

    /*
        transposed(t):
        - materialises the transpose of a 2-D weight. Only GPT-2 needs it, and
          only once at load, which is why the hot path can assume one layout
          rather than branching per matmul
    */
    template <typename T>
    static Tensor<T> transposed(const Tensor<T>& t) {
        Tensor<T> out({t.shape[1], t.shape[0]});
        const size_t rows = t.shape[0], cols = t.shape[1];
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                out.ptr[j * rows + i] = t.ptr[i * cols + j];
        return out;
    }

    Tensor<float> fetchVector(const std::string& name) const {
        if (name.empty() || !weights.contains(name)) return Tensor<float>();
        return weights.tensor(name);
    }

    /* rowSlice(w, begin, end): rows of an [out,in] matrix as a view, which is
       how a fused qkv or gate_up is taken apart without copying */
    static Weight rowSlice(const Weight& w, size_t begin, size_t end) {
        Weight out;
        if (w.isBF16()) out.bf16 = w.bf16.slice(0, begin, end);
        else            out.f32  = w.f32.slice(0, begin, end);
        out.outDim = end - begin;
        out.inDim  = w.inDim;
        return out;
    }

    static Tensor<float> vectorSlice(const Tensor<float>& t, size_t begin, size_t end) {
        if (!t.storage) return Tensor<float>();
        return t.slice(0, begin, end).contiguous();
    }


    bool loadTopLevel() {
        embed = fetchEmbedding(spec.names.embed);
        if (!embed.valid()) { error = "missing " + spec.names.embed; return false; }

        finalNorm     = fetchVector(spec.names.finalNorm);
        finalNormBias = fetchVector(spec.names.finalNormBias);
        if (!finalNorm.storage) { error = "missing " + spec.names.finalNorm; return false; }

        if (spec.pos == PosKind::Learned) {
            posEmbed = fetchVector(spec.names.posEmbed);
            if (!posEmbed.storage) { error = "missing " + spec.names.posEmbed; return false; }
        }

        lmHead = (!spec.names.lmHead.empty() && weights.contains(spec.names.lmHead))
               ? fetchEmbedding(spec.names.lmHead)
               : embed;                      // tied

        return true;
    }

    /* the embedding table is [vocab, hidden] in both conventions, so it is
       never transposed even when the model's Linear layers are */
    Weight fetchEmbedding(const std::string& name) const {
        Weight w;
        if (name.empty() || !weights.contains(name)) return w;

        Tensor<uint16_t> raw = weights.rawBF16(name);
        if (raw.storage) w.bf16 = raw;
        else             w.f32  = weights.tensor(name);

        const std::vector<size_t>& shape = w.isBF16() ? w.bf16.shape : w.f32.shape;
        if (shape.size() == 2) { w.outDim = shape[0]; w.inDim = shape[1]; }
        return w;
    }


    bool loadLayers() {
        layers.resize(cfg.nLayer);

        for (size_t i = 0; i < cfg.nLayer; ++i) {
            LocalLayer& L = layers[i];

            L.inputNorm        = fetchVector(expandName(spec.names.inputNorm, i));
            L.inputNormBias    = fetchVector(expandName(spec.names.inputNormBias, i));
            L.postAttnNorm     = fetchVector(expandName(spec.names.postAttnNorm, i));
            L.postAttnNormBias = fetchVector(expandName(spec.names.postAttnNormBias, i));

            if (spec.qkNorm) {
                L.qNorm = fetchVector(expandName(spec.names.qNorm, i));
                L.kNorm = fetchVector(expandName(spec.names.kNorm, i));
            }

            /* ---- attention projections --------------------------------- */
            if (!spec.names.qkv.empty()) {
                // one matrix holding q, k and v stacked in that order
                Weight fused = fetch(expandName(spec.names.qkv, i));
                if (!fused.valid()) { error = "missing " + expandName(spec.names.qkv, i); return false; }

                const size_t qw = cfg.qWidth(), kw = cfg.kvWidth();
                L.q = rowSlice(fused, 0,           qw);
                L.k = rowSlice(fused, qw,          qw + kw);
                L.v = rowSlice(fused, qw + kw,     qw + 2 * kw);

                Tensor<float> bias = fetchVector(expandName(spec.names.qkvBias, i));
                if (bias.storage) {
                    L.q.bias = vectorSlice(bias, 0,       qw);
                    L.k.bias = vectorSlice(bias, qw,      qw + kw);
                    L.v.bias = vectorSlice(bias, qw + kw, qw + 2 * kw);
                }
            } else {
                L.q = fetch(expandName(spec.names.q, i));
                L.k = fetch(expandName(spec.names.k, i));
                L.v = fetch(expandName(spec.names.v, i));
                L.q.bias = fetchVector(expandName(spec.names.qBias, i));
                L.k.bias = fetchVector(expandName(spec.names.kBias, i));
                L.v.bias = fetchVector(expandName(spec.names.vBias, i));
            }

            L.o = fetch(expandName(spec.names.o, i));
            L.o.bias = fetchVector(expandName(spec.names.oBias, i));

            /* ---- feed forward ------------------------------------------ */
            if (!spec.names.gateUp.empty()) {
                Weight fused = fetch(expandName(spec.names.gateUp, i));
                if (!fused.valid()) { error = "missing " + expandName(spec.names.gateUp, i); return false; }
                L.gate = rowSlice(fused, 0, cfg.intermediate);
                L.up   = rowSlice(fused, cfg.intermediate, 2 * cfg.intermediate);
            } else {
                if (!spec.names.gate.empty()) L.gate = fetch(expandName(spec.names.gate, i));
                L.up = fetch(expandName(spec.names.up, i));
                L.up.bias = fetchVector(expandName(spec.names.upBias, i));
            }

            L.down = fetch(expandName(spec.names.down, i));
            L.down.bias = fetchVector(expandName(spec.names.downBias, i));

            if (!L.q.valid() || !L.o.valid() || !L.up.valid() || !L.down.valid()) {
                error = "missing weights for layer " + std::to_string(i);
                return false;
            }
        }

        return true;
    }


    size_t parameterCount() const { return weights.parameterCount(); }


    /* ===================================================================== //
        Forward
    // ===================================================================== */

    void normRow(const float* x, const Tensor<float>& w, const Tensor<float>& b,
                 float* out, size_t n) const {
        if (spec.norm == NormKind::RMSNorm) rmsNormRow(x, w.ptr, out, n, cfg.normEps);
        else layerNormRow(x, w.ptr, b.storage ? b.ptr : nullptr, out, n, cfg.normEps);
    }

    float activation(float x) const {
        return spec.act == ActKind::SiLU ? siluOf(x) : geluTanhOf(x);
    }

    static void widenRow(const Weight& table, size_t index, float* out, size_t n) {
        if (table.isBF16()) {
            const uint16_t* row = table.bf16.ptr + index * n;
            for (size_t i = 0; i < n; ++i)
                out[i] = std::bit_cast<float>(static_cast<uint32_t>(row[i]) << 16);
        } else {
            const float* row = table.f32.ptr + index * n;
            std::copy(row, row + n, out);
        }
    }


    void forward(const size_t* ids, size_t count, size_t startPos,
                 LocalCache& cache, float* logitsOut, LocalTrace* trace = nullptr) const {
        const size_t H  = cfg.hidden;
        const size_t D  = cfg.headDim;
        const size_t QW = cfg.qWidth();
        const size_t KW = cfg.kvWidth();

        std::vector<float> x(count * H), h(count * H), proj(count * H);
        std::vector<float> q(count * QW), k(count * KW), v(count * KW), attn(count * QW);
        std::vector<float> gate(count * cfg.intermediate), up(count * cfg.intermediate);
        std::vector<float> scores(startPos + count);

        for (size_t t = 0; t < count; ++t) widenRow(embed, ids[t], &x[t * H], H);

        /* learned absolute positions are simply added into the stream; RoPE
           instead rotates Q and K further down, and touches nothing here */
        if (spec.pos == PosKind::Learned)
            for (size_t t = 0; t < count; ++t)
                for (size_t i = 0; i < H; ++i)
                    x[t * H + i] += posEmbed.ptr[(startPos + t) * H + i];

        if (trace) trace->add("embed", x.data(), count * H);

        for (size_t l = 0; l < cfg.nLayer; ++l) {
            const LocalLayer& L = layers[l];
            const std::string tag = expandName(spec.tracePrefix, l);

            for (size_t t = 0; t < count; ++t)
                normRow(&x[t * H], L.inputNorm, L.inputNormBias, &h[t * H], H);
            if (trace) trace->add(tag + "ln1", h.data(), count * H);

            apply(h.data(), L.q, q.data(), count);
            apply(h.data(), L.k, k.data(), count);
            apply(h.data(), L.v, v.data(), count);

            if (spec.qkNorm) {
                for (size_t t = 0; t < count; ++t) {
                    for (size_t head = 0; head < cfg.nHead; ++head)
                        rmsNormRow(&q[t * QW + head * D], L.qNorm.ptr,
                                   &q[t * QW + head * D], D, cfg.normEps);
                    for (size_t head = 0; head < cfg.nKVHead; ++head)
                        rmsNormRow(&k[t * KW + head * D], L.kNorm.ptr,
                                   &k[t * KW + head * D], D, cfg.normEps);
                }
                if (trace) {
                    trace->add(tag + "qnorm", q.data(), count * QW);
                    trace->add(tag + "knorm", k.data(), count * KW);
                }
            }

            for (size_t t = 0; t < count; ++t) {
                if (spec.pos == PosKind::RoPE) {
                    ropeRow(&q[t * QW], cfg.nHead,   D, startPos + t, cfg.ropeTheta);
                    ropeRow(&k[t * KW], cfg.nKVHead, D, startPos + t, cfg.ropeTheta);
                }

                std::copy(&k[t * KW], &k[t * KW] + KW,
                          cache.keys[l].begin() + static_cast<long>((startPos + t) * KW));
                std::copy(&v[t * KW], &v[t * KW] + KW,
                          cache.values[l].begin() + static_cast<long>((startPos + t) * KW));
            }
            if (trace && spec.pos == PosKind::RoPE) {
                trace->add(tag + "qrope", q.data(), count * QW);
                trace->add(tag + "krope", k.data(), count * KW);
            }

            const float invSqrtD = 1.0f / std::sqrt(static_cast<float>(D));

            for (size_t t = 0; t < count; ++t) {
                const size_t position = startPos + t;

                for (size_t head = 0; head < cfg.nHead; ++head) {
                    const size_t kvHead = head / cfg.group();
                    const float* qh = &q[t * QW + head * D];

                    float best = -1e30f;
                    for (size_t j = 0; j <= position; ++j) {
                        const float s =
                            kernels::dot(qh, &cache.keys[l][j * KW + kvHead * D], D) * invSqrtD;
                        scores[j] = s;
                        best = std::max(best, s);
                    }

                    float sum = 0.0f;
                    for (size_t j = 0; j <= position; ++j) {
                        scores[j] = std::exp(scores[j] - best);
                        sum += scores[j];
                    }
                    const float invSum = 1.0f / sum;

                    float* out = &attn[t * QW + head * D];
                    for (size_t d = 0; d < D; ++d) out[d] = 0.0f;
                    for (size_t j = 0; j <= position; ++j) {
                        const float w = scores[j] * invSum;
                        const float* vh = &cache.values[l][j * KW + kvHead * D];
                        for (size_t d = 0; d < D; ++d) out[d] += w * vh[d];
                    }
                }
            }

            apply(attn.data(), L.o, proj.data(), count);
            if (trace) trace->add(tag + "attnout", proj.data(), count * H);

            for (size_t i = 0; i < count * H; ++i) x[i] += proj[i];
            if (trace) trace->add(tag + "res1", x.data(), count * H);

            for (size_t t = 0; t < count; ++t)
                normRow(&x[t * H], L.postAttnNorm, L.postAttnNormBias, &h[t * H], H);
            if (trace) trace->add(tag + "ln2", h.data(), count * H);

            if (spec.mlp == MLPKind::SwiGLU) {
                apply(h.data(), L.gate, gate.data(), count);
                apply(h.data(), L.up,   up.data(),   count);
                for (size_t i = 0; i < count * cfg.intermediate; ++i)
                    gate[i] = activation(gate[i]) * up[i];
            } else {
                apply(h.data(), L.up, gate.data(), count);
                if (trace) trace->add(tag + "fc", gate.data(), count * cfg.intermediate);
                for (size_t i = 0; i < count * cfg.intermediate; ++i)
                    gate[i] = activation(gate[i]);
                if (trace) trace->add(tag + "gelu", gate.data(), count * cfg.intermediate);
            }

            apply(gate.data(), L.down, proj.data(), count);
            if (trace) trace->add(tag + "mlpout", proj.data(), count * H);

            for (size_t i = 0; i < count * H; ++i) x[i] += proj[i];
            if (trace) trace->add(tag + "res2", x.data(), count * H);
        }

        cache.length = startPos + count;

        std::vector<float> last(H);
        normRow(&x[(count - 1) * H], finalNorm, finalNormBias, last.data(), H);
        if (trace) trace->add(spec.traceFinalNorm, last.data(), H);

        Weight head = lmHead;
        head.outDim = cfg.vocab;
        head.inDim  = H;
        apply(last.data(), head, logitsOut, 1);
        if (trace) trace->add("logits", logitsOut, cfg.vocab);
    }


    /* ===================================================================== //
        Sampling and chat
    // ===================================================================== */

    static size_t argmax(const float* row, size_t n) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i) if (row[i] > row[best]) best = i;
        return best;
    }

    static size_t sampleTopK(const float* row, size_t n, float temperature,
                             size_t topK, math::random::Random& rng) {
        if (temperature <= 0.0f || topK <= 1) return argmax(row, n);

        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;

        const size_t keep = std::min(topK, n);
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep), order.end(),
                          [row](size_t a, size_t b) { return row[a] > row[b]; });
        order.resize(keep);

        const float best = row[order[0]] / temperature;
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
        Instruction-tuned models do not agree on how a turn is marked, and
        feeding one the wrong markers produces text that looks almost right
        while quietly drifting. The format is detected from which special
        tokens the vocabulary actually contains rather than assumed.
    */
    enum class ChatFormat {
        None,           // a base model: prompt it as plain text
        ChatML,         // <|im_start|>role\n ... <|im_end|>
        ChatMLSep,      // the same but with <|im_sep|> between role and content
        MistralInst     // [INST] ... [/INST]
    };

    ChatFormat chatFormat() const {
        if (tokenizer.hasSpecial("[INST]"))       return ChatFormat::MistralInst;
        if (tokenizer.hasSpecial("<|im_sep|>"))   return ChatFormat::ChatMLSep;
        if (tokenizer.hasSpecial("<|im_start|>")) return ChatFormat::ChatML;
        return ChatFormat::None;
    }

    const char* chatFormatName() const {
        switch (chatFormat()) {
            case ChatFormat::MistralInst: return "[INST]";
            case ChatFormat::ChatMLSep:   return "ChatML+sep";
            case ChatFormat::ChatML:      return "ChatML";
            default:                      return "none (base model)";
        }
    }

    bool hasChatTemplate() const { return chatFormat() != ChatFormat::None; }

    std::string chatPrompt(const std::string& user, bool thinking = false) const {
        switch (chatFormat()) {
            case ChatFormat::MistralInst:
                return "[INST] " + user + " [/INST]";

            case ChatFormat::ChatMLSep:
                // Phi-4 separates the role from the content with a token
                // rather than a newline
                return "<|im_start|>user<|im_sep|>" + user
                     + "<|im_end|><|im_start|>assistant<|im_sep|>";

            case ChatFormat::ChatML: {
                std::string prompt = "<|im_start|>user\n" + user
                                   + "<|im_end|>\n<|im_start|>assistant\n";

                /*
                    The empty <think> block is how a reasoning model is told
                    to skip its scratchpad. A model without those tokens has
                    never seen them, so emitting the text anyway just puts
                    six stray tokens in front of the answer
                */
                if (!thinking && tokenizer.hasSpecial("<think>"))
                    prompt += "<think>\n\n</think>\n\n";

                return prompt;
            }

            default:
                return user;
        }
    }

    bool isStop(size_t id) const {
        static const char* ENDS[] = {"<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"};
        for (const char* e : ENDS)
            if (tokenizer.hasSpecial(e) && id == tokenizer.idFor(e)) return true;
        return false;
    }
};

} // !namespace ai
