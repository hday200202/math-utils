#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <optional>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "BPE.hpp"
#include "Kernels.hpp"
#include "Safetensors.hpp"
#include "gpu/Device.hpp"
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

/*
    Which mixer a layer uses. Everything before Qwen3.5 used softmax attention
    in every layer; the qwen3_5 family alternates, with three linear-attention
    layers for every full-attention one. The choice is per LAYER, not per
    model, so it lives in the config rather than the spec.
*/
enum class MixerKind    { Attention, GatedDeltaNet };

enum class NormKind     { LayerNorm, RMSNorm };
enum class PosKind      { Learned, RoPE };
/*
    MoE: the feed forward is a bank of small SwiGLU experts of which only a
    handful run per token, chosen by a router, plus one shared expert that
    always runs. The parameter count is enormous and the work per token is
    not, which is the whole trade: Qwen3-Next carries 80B weights and touches
    about 3B of them for each token it produces.
*/
enum class MLPKind      { TwoMatrix, SwiGLU, MoE };
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

    /* Gated DeltaNet layers. Empty on every architecture that has none */
    std::string linQKV, linA, linB, linZ;
    std::string linConv, linALog, linDtBias, linNorm, linOut;

    /* Mixture of experts. The expert templates take {E} as well as {L} */
    std::string router;                        // picks which experts run
    std::string expertGate, expertUp, expertDown;
    std::string sharedGate, sharedUp, sharedDown;
    std::string sharedScale;                   // [1, hidden], gates the shared expert
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

    /*
        attnOutputGate: q_proj emits twice the query width, the second half of
        each head being a gate applied to that head's attention output.
        The two halves INTERLEAVE per head rather than splitting the whole
        matrix in two, which is easy to get wrong and silently wrong if you do.
    */
    bool         attnOutputGate = false;

    /*
        normWeightOffset: what is added to a norm's weight before it scales.
        Almost every model stores the scale directly and this stays 0. The
        qwen3_5 family stores it as a DEVIATION from one, initialised to
        zero, and computes (1 + w) - so reading it directly multiplies every
        activation by zero and the whole model silently outputs nothing.
        Note it applies to the ordinary norms only; the gated norm inside a
        linear-attention layer stores its scale the usual way.
    */
    float        normWeightOffset = 0.0f;

    /*
        linFusedQKVZ: the linear-attention input projections arrive as two
        matrices rather than four - one holding q, k, v and z, the other b
        and a - and both are laid out PER KEY HEAD rather than as whole
        blocks, so they have to be de-interleaved rather than simply sliced.
    */
    bool         linFusedQKVZ = false;

    /* some tensor names sit under model.language_model. rather than model. */
    std::vector<std::string> prefixCandidates;

    /* branches of the checkpoint to read past entirely */
    std::vector<std::string> skipPrefixes;

    ConfigKeys   keys;
    NameMap      names;

    /* the numpy references dump under different names, so the trace follows
       whichever this model's oracle uses */
    std::string  tracePrefix    = "l{L}.";
    std::string  traceFinalNorm = "norm";
};


/* swaps the leading "model." of a template for whatever the file uses */
inline std::string applyPrefix(const std::string& tmpl, const std::string& prefix) {
    const std::string base = "model.";
    if (prefix == base || tmpl.compare(0, base.size(), base) != 0) return tmpl;
    return prefix + tmpl.substr(base.size());
}

/* expand("model.layers.{L}.mlp.up_proj.weight", 7) */
inline std::string expandName(const std::string& tmpl, size_t layer) {
    const size_t at = tmpl.find("{L}");
    if (at == std::string::npos) return tmpl;
    return tmpl.substr(0, at) + std::to_string(layer) + tmpl.substr(at + 3);
}

/* and again for the expert index, which only mixture-of-experts names carry */
inline std::string expandExpert(const std::string& tmpl, size_t expert) {
    const size_t at = tmpl.find("{E}");
    if (at == std::string::npos) return tmpl;
    return tmpl.substr(0, at) + std::to_string(expert) + tmpl.substr(at + 3);
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
/*
    qwen35Spec():
    - the qwen3_5 family. Three quarters of its layers replace softmax
      attention with a Gated DeltaNet recurrence, and the ones that keep
      attention add an output gate and rotate only part of each head.
    - the multimodal checkpoints nest the text tower under
      model.language_model., the text-only ones do not, so both prefixes are
      tried
*/
inline ArchSpec qwen35Spec(const std::string& type) {
    ArchSpec s = llamaFamily(type);

    s.qkNorm = true;
    s.attnOutputGate = true;
    s.normWeightOffset = 1.0f;
    s.prefixCandidates = {"model.", "model.language_model."};

    s.names.qNorm = "model.layers.{L}.self_attn.q_norm.weight";
    s.names.kNorm = "model.layers.{L}.self_attn.k_norm.weight";

    s.names.linQKV    = "model.layers.{L}.linear_attn.in_proj_qkv.weight";
    s.names.linA      = "model.layers.{L}.linear_attn.in_proj_a.weight";
    s.names.linB      = "model.layers.{L}.linear_attn.in_proj_b.weight";
    s.names.linZ      = "model.layers.{L}.linear_attn.in_proj_z.weight";
    s.names.linConv   = "model.layers.{L}.linear_attn.conv1d.weight";
    s.names.linALog   = "model.layers.{L}.linear_attn.A_log";
    s.names.linDtBias = "model.layers.{L}.linear_attn.dt_bias";
    s.names.linNorm   = "model.layers.{L}.linear_attn.norm.weight";
    s.names.linOut    = "model.layers.{L}.linear_attn.out_proj.weight";

    return s;
}

/*
    qwen3NextSpec():
    - the same alternating shape as qwen3_5, with two differences that matter.
      The linear-attention inputs arrive fused and interleaved per key head
      rather than as four separate matrices, and the feed forward is a bank
      of 512 experts of which ten run per token.
    - which layers attend is given as a stride rather than a list, and the
      checkpoint carries a speculative-decoding head that is read past
*/
inline ArchSpec qwen3NextSpec() {
    ArchSpec s = qwen35Spec("qwen3_next");

    s.mlp = MLPKind::MoE;
    s.linFusedQKVZ = true;
    s.prefixCandidates = {"model."};
    s.skipPrefixes = {"mtp."};

    s.names.linQKV = "model.layers.{L}.linear_attn.in_proj_qkvz.weight";
    s.names.linA   = "model.layers.{L}.linear_attn.in_proj_ba.weight";
    s.names.linB.clear();                  // folded into in_proj_ba
    s.names.linZ.clear();                  // folded into in_proj_qkvz

    s.names.router      = "model.layers.{L}.mlp.gate.weight";
    s.names.expertGate  = "model.layers.{L}.mlp.experts.{E}.gate_proj.weight";
    s.names.expertUp    = "model.layers.{L}.mlp.experts.{E}.up_proj.weight";
    s.names.expertDown  = "model.layers.{L}.mlp.experts.{E}.down_proj.weight";
    s.names.sharedGate  = "model.layers.{L}.mlp.shared_expert.gate_proj.weight";
    s.names.sharedUp    = "model.layers.{L}.mlp.shared_expert.up_proj.weight";
    s.names.sharedDown  = "model.layers.{L}.mlp.shared_expert.down_proj.weight";
    s.names.sharedScale = "model.layers.{L}.mlp.shared_expert_gate.weight";

    /* the dense gate/up/down templates would otherwise still be inherited */
    s.names.gate.clear();
    s.names.up.clear();
    s.names.down.clear();

    return s;
}

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
        qwen35Spec("qwen3_5"),
        qwen35Spec("qwen3_5_text"),
        qwen3NextSpec(),
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

    /*
        Only a fraction of each head's dimensions are rotated when this is
        below 1. Qwen3.5 rotates a quarter of 256, so 64 dims carry position
        and the remaining 192 are left alone
    */
    float  partialRotary = 1.0f;

    /* what an untrained matrix is drawn from, for the synthetic loader */
    float  initRange = 0.02f;

    /* Gated DeltaNet, when any layer uses it */
    size_t linKeyHeads = 0, linValueHeads = 0;
    size_t linKeyDim = 0, linValueDim = 0, convKernel = 0;

    /*
        Mixture of experts. numExperts is how many the layer holds,
        expertsPerToken how many actually run for each token
    */
    size_t numExperts = 0, expertsPerToken = 0;
    size_t moeIntermediate = 0, sharedIntermediate = 0;
    bool   normTopK = true;

    std::vector<MixerKind> mixers;      // one per layer

    size_t linKeyWidth()   const { return linKeyHeads * linKeyDim; }
    size_t linValueWidth() const { return linValueHeads * linValueDim; }
    size_t linConvWidth()  const { return linKeyWidth() * 2 + linValueWidth(); }
    size_t linGroup()      const { return linValueHeads / linKeyHeads; }

    bool isMoE() const { return numExperts > 0 && expertsPerToken > 0; }

    bool hasLinearLayers() const {
        for (MixerKind m : mixers) if (m == MixerKind::GatedDeltaNet) return true;
        return false;
    }
    size_t rotaryDims() const {
        const size_t d = static_cast<size_t>(float(headDim) * partialRotary);
        return (d / 2) * 2;             // rotation works in pairs
    }

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

        initRange = static_cast<float>(c["initializer_range"].asDouble(0.02));
        normEps   = static_cast<float>(c[k.normEps].asDouble(1e-5));
        ropeTheta = k.ropeBase.empty() ? 10000.0f
                                       : static_cast<float>(c[k.ropeBase].asDouble(10000.0));
        tied = c[k.tied].asBool(spec.tiedByDefault);

        /*
            qwen3_5 puts the rotary settings in a nested object rather than at
            the top level, so look there before falling back
        */
        const files::json::Value& rope = c["rope_parameters"];
        if (rope.isObject()) {
            ropeTheta = static_cast<float>(rope["rope_theta"].asDouble(ropeTheta));
            partialRotary = static_cast<float>(rope["partial_rotary_factor"].asDouble(1.0));
        }
        partialRotary = static_cast<float>(
            c["partial_rotary_factor"].asDouble(partialRotary));

        linKeyHeads   = c["linear_num_key_heads"].asSize(0);
        linValueHeads = c["linear_num_value_heads"].asSize(0);
        linKeyDim     = c["linear_key_head_dim"].asSize(0);
        linValueDim   = c["linear_value_head_dim"].asSize(0);
        convKernel    = c["linear_conv_kernel_dim"].asSize(0);

        numExperts         = c["num_experts"].asSize(0);
        expertsPerToken    = c["num_experts_per_tok"].asSize(0);
        moeIntermediate    = c["moe_intermediate_size"].asSize(0);
        sharedIntermediate = c["shared_expert_intermediate_size"].asSize(0);
        normTopK           = c["norm_topk_prob"].asBool(true);

        /* layer_types names the mixer for each layer; absent means all
           softmax attention, which is every architecture before this one */
        mixers.assign(nLayer, MixerKind::Attention);
        const files::json::Value& types = c["layer_types"];
        if (types.isArray()) {
            for (size_t i = 0; i < types.size() && i < nLayer; ++i)
                if (types[i].asString() == "linear_attention")
                    mixers[i] = MixerKind::GatedDeltaNet;
        } else {
            /*
                qwen3_next says the same thing as a stride instead of a list:
                every interval-th layer keeps full attention and the rest are
                linear. Counting from one, so with an interval of 4 it is
                layers 3, 7, 11 ... that attend
            */
            const size_t interval = c["full_attention_interval"].asSize(0);
            if (interval > 1)
                for (size_t i = 0; i < nLayer; ++i)
                    if ((i + 1) % interval != 0) mixers[i] = MixerKind::GatedDeltaNet;
        }

        if (isMoE() && (moeIntermediate == 0 || expertsPerToken > numExperts))
            return false;

        if (hasLinearLayers()
            && (linKeyHeads == 0 || linValueHeads == 0 || linKeyDim == 0
                || linValueDim == 0 || convKernel == 0
                || linValueHeads % linKeyHeads != 0))
            return false;

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

/*
    DevDType:
    - what is actually on the card, which the host tensors do not say on
      their own. Kept per weight, so a model with a handful of matrices the
      GPU cannot take is not a special case anywhere - each one falls back
      to the CPU by itself
*/
enum class DevDType { None, F32, BF16, Q8, Q4 };


/*
    OffloadPlan:
    - how much of the model to put on the card. Everything by default,
      as much as fits when it will not all go
*/
struct OffloadPlan {
    size_t vramBudget  = 0;                  // 0 = whatever is free, less the reserve
    size_t reserve     = size_t(512) << 20;  // left for scratch and the desktop
    bool   releaseHost = true;
};


struct Weight {
    Tensor<float>       f32;
    Tensor<uint16_t>    bf16;
    quant::QuantTensor  q;
    Tensor<float>       bias;             // empty when the model has none

    /*
        the same bytes in VRAM. Nothing is converted on the way: the
        quantiser has already run during load, so this is a straight copy
        of the codes and scales the CPU kernels read
    */
    gpu::BufferRef      dev;
    gpu::BufferRef      devScales;        // quantised only
    DevDType            devType = DevDType::None;

    size_t outDim = 0, inDim = 0;

    bool valid() const {
        return f32.storage != nullptr || bf16.storage != nullptr || q.valid();
    }
    bool isBF16() const { return bf16.storage != nullptr; }
    bool isQuant() const { return q.valid(); }
    bool hasBias() const { return bias.storage != nullptr; }
    bool onDevice() const { return devType != DevDType::None && dev && dev->valid(); }

    /* the codes, without the scales that go with them */
    size_t quantCodeBytes() const {
        const size_t weights = q.outDim * q.inDim;
        return (q.type == quant::Type::Q4) ? weights / 2 : weights;
    }
    size_t quantScaleBytes() const {
        return q.outDim * q.blocksPerRow() * sizeof(float);
    }

    /* the contiguous block a device copy would be made from */
    const void* hostBytes() const {
        if (isQuant()) return q.quants.get();
        if (isBF16())  return bf16.isContiguous() ? bf16.ptr : nullptr;
        if (f32.storage) return f32.isContiguous() ? f32.ptr : nullptr;
        return nullptr;
    }
    size_t deviceBytes() const {
        if (isQuant()) return quantCodeBytes() + quantScaleBytes();
        if (isBF16())  return bf16.numel() * sizeof(uint16_t);
        if (f32.storage) return f32.numel() * sizeof(float);
        return 0;
    }
};


/*
    apply(x, W, out, rows):
    - out[rows, outDim] = x[rows, inDim] @ W^T, plus the bias if there is one.
      Both weight layouts have already been normalised to [out, in] at load,
      so this is the only orientation the kernels ever see
*/
inline void apply(const float* x, const Weight& W, float* out, size_t rows,
                  bool threaded = true) {
    if (W.onDevice()) {
        switch (W.devType) {
            case DevDType::Q4:
                gpu::matmulNTQ4(x, *W.dev, *W.devScales, out, rows, W.inDim, W.outDim);
                break;
            case DevDType::Q8:
                gpu::matmulNTQ8(x, *W.dev, *W.devScales, out, rows, W.inDim, W.outDim);
                break;
            case DevDType::BF16:
                gpu::matmulNTBF16(x, *W.dev, out, rows, W.inDim, W.outDim);
                break;
            default:
                gpu::matmulNT(x, *W.dev, out, rows, W.inDim, W.outDim);
                break;
        }

        /* folded on the host: C has just been copied back anyway, so this
           is one pass over a vector that is already warm rather than a
           second kernel and a second launch */
        if (W.hasBias())
            for (size_t r = 0; r < rows; ++r)
                for (size_t j = 0; j < W.outDim; ++j)
                    out[r * W.outDim + j] += W.bias.ptr[j];
        return;
    }

    if (W.isQuant())
        kernels::matmulNTQuant(x, W.q, out, rows, W.inDim, W.outDim, threaded);
    else if (W.isBF16())
        kernels::matmulNTBF16(x, W.bf16.ptr, out, rows, W.inDim, W.outDim, threaded);
    else if (W.hasBias())
        kernels::matmulNTBias(x, W.f32.ptr, W.bias.ptr, out, rows, W.inDim, W.outDim, threaded);
    else
        kernels::matmulNT(x, W.f32.ptr, out, rows, W.inDim, W.outDim, threaded);

    // only the plain float kernel folds the bias in, so add it otherwise
    if ((W.isBF16() || W.isQuant()) && W.hasBias())
        for (size_t r = 0; r < rows; ++r)
            for (size_t j = 0; j < W.outDim; ++j)
                out[r * W.outDim + j] += W.bias.ptr[j];
}


/* one SwiGLU feed forward: the shape of both a dense MLP and every expert */
struct Expert {
    Weight gate, up, down;
};


struct LocalLayer {
    MixerKind     mixer = MixerKind::Attention;

    Tensor<float> inputNorm, inputNormBias;
    Tensor<float> postAttnNorm, postAttnNormBias;
    Tensor<float> qNorm, kNorm;
    Weight q, k, v, o;
    Weight gate, up, down;

    /* Gated DeltaNet, populated only on the layers that use it */
    Weight        linQKV, linA, linB, linZ, linOut;
    Tensor<float> linConv;          // [convWidth, 1, kernel], depthwise
    Tensor<float> linALog, linDtBias, linNorm;

    /*
        Mixture of experts. This is where nearly all of an 80B checkpoint
        actually sits: 512 experts a layer, of which ten run per token
    */
    Weight              router;         // [numExperts, hidden]
    Weight              sharedScale;    // [1, hidden], gates the shared expert
    Expert              shared;
    std::vector<Expert> experts;
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

    /*
        A linear-attention layer keeps no per-token history at all. Its whole
        memory is one [keyDim, valueDim] matrix per value head, plus the last
        few inputs the causal convolution still needs. That is CONSTANT in
        sequence length, which is the entire reason the architecture exists -
        and it means these layers contribute nothing to the growth that makes
        a long context expensive.
    */
    std::vector<std::vector<float>> recurrent;   // [layer][vHead * kDim * vDim]
    std::vector<std::vector<float>> convWindow;  // [layer][(kernel-1) * convWidth]

    size_t length = 0, capacity = 0;

    void resize(const LocalConfig& cfg, size_t maxSeq) {
        capacity = maxSeq;
        length = 0;

        keys.assign(cfg.nLayer, {});
        values.assign(cfg.nLayer, {});
        recurrent.assign(cfg.nLayer, {});
        convWindow.assign(cfg.nLayer, {});

        for (size_t l = 0; l < cfg.nLayer; ++l) {
            if (cfg.mixers[l] == MixerKind::GatedDeltaNet) {
                recurrent[l].assign(cfg.linValueHeads * cfg.linKeyDim * cfg.linValueDim, 0.0f);
                convWindow[l].assign((cfg.convKernel - 1) * cfg.linConvWidth(), 0.0f);
            } else {
                keys[l].assign(maxSeq * cfg.kvWidth(), 0.0f);
                values[l].assign(maxSeq * cfg.kvWidth(), 0.0f);
            }
        }
    }

    void reset() {
        length = 0;
        for (auto& r : recurrent) std::fill(r.begin(), r.end(), 0.0f);
        for (auto& c : convWindow) std::fill(c.begin(), c.end(), 0.0f);
    }
};


/* ========================================================================= //
    Shared primitives
// ========================================================================= */

inline void rmsNormRow(const float* x, const float* w, float* out, size_t n, float eps,
                       float weightOffset = 0.0f) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i] * x[i];

    const float inv = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * (w[i] + weightOffset);
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

inline float sigmoidOf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/* log(1 + e^x), computed so a large x does not overflow the exponential */
inline float softplusOf(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

/* scales a vector to unit length; the delta rule normalises q and k this way
   before the recurrence so the state cannot be driven by their magnitude */
inline void l2normalise(float* v, size_t n, float eps = 1e-6f) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += v[i] * v[i];

    const float inv = 1.0f / std::sqrt(sum + eps);
    for (size_t i = 0; i < n; ++i) v[i] *= inv;
}

inline float geluTanhOf(float x) {
    const float k = 0.7978845608028654f;     // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

/*
    ropeRow(vec, heads, headDim, position, theta, rotaryDims):
    - rotaryDims lets only the first part of each head carry position, which
      is what a partial rotary factor means. The dimensions beyond it are left
      exactly as they are rather than rotated by zero, and the pairing is
      within the rotated part, not across the whole head
*/
inline void ropeRow(float* vec, size_t heads, size_t headDim, size_t position,
                    float theta, size_t rotaryDims = 0) {
    const size_t span = (rotaryDims == 0 || rotaryDims > headDim) ? headDim : rotaryDims;
    const size_t half = span / 2;

    for (size_t j = 0; j < half; ++j) {
        const float freq =
            1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(span));
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
    quant::Type              quantType = quant::Type::None;
    ModelWeights             weights;
    Tokenizer                tokenizer;

    Weight                   embed, lmHead;
    Tensor<float>            posEmbed;                 // learned positions only
    Tensor<float>            finalNorm, finalNormBias;
    std::vector<LocalLayer>  layers;

    std::string              modelType;
    std::string              prefix = "model.";      // or model.language_model.
    std::string              error;


    /* ===================================================================== //
        Loading
    // ===================================================================== */

    /*
        plan(directory, quantise):
        - what loading this model will cost, before a byte is allocated.
          It goes through the spec rather than straight to the weights so
          that whatever branches an architecture reads past - a speculative
          decoding head, a vision tower - are not counted against it.
    */
    /*
        randomInitialiser(seed):
        - plausible values for a model that was never trained, so that a
          topology can be run without its weights.

        Getting the scale wrong here is not subtle. Draw every matrix from
        uniform [-1, 1] and the residual stream saturates within two or three
        layers, every logit comes back NaN, and the run tells you nothing
        about the architecture - which was the only thing being tested. So
        matrices are drawn from the same normal a real initialisation uses,
        and the tensors that are not matrices get the specific values their
        reference constructors give them:

          - a norm weight has to be the IDENTITY, which is 1 normally and 0
            under the qwen3_5 convention where the model computes (1 + w)
          - A_log holds log of a decay drawn from (0.01, 16]. Zero there
            means a decay of exactly one - a linear-attention state that
            never forgets. That is degenerate rather than immediately
            explosive, and a short prompt will not show it, which is
            precisely why it is worth setting correctly rather than trusting
            a run to catch it
          - dt_bias is ones, not zeros

        Everything else one-dimensional is a bias, and zero is right for it.
    */
    WeightInitialiser randomInitialiser(uint64_t seed) const {
        const float range = cfg.initRange;
        const float identity = 1.0f - spec.normWeightOffset;

        return [range, identity, seed](const std::string& name,
                                       const TensorInfo& info, float* out) {
            const size_t n = info.numel();

            const bool isNorm = name.find("norm") != std::string::npos;
            const bool isDecay = name.find("A_log") != std::string::npos;

            if (info.shape.size() < 2 && !isDecay) {
                const float value = (name.find("dt_bias") != std::string::npos) ? 1.0f
                                  : isNorm ? identity
                                  : 0.0f;
                for (size_t i = 0; i < n; ++i) out[i] = value;
                return;
            }

            /*
                A generator per fixed-size chunk rather than one for the whole
                model. Eighty billion normals is a quarter of an hour of
                Box-Muller on one core, and chunking is what lets all of them
                run at once.

                The chunk size is a constant and every chunk's seed comes from
                the tensor's NAME, so the same --seed gives the same model
                whatever the thread count and however the checkpoint happened
                to be split into shards. Threading a generator by handing
                slices of one stream to different threads would not: the
                answer would change with the scheduler.

                Keep it well BELOW the size of the tensors being filled. An
                expert matrix here is 2048x512, which is 1<<20 exactly - so a
                chunk of 1<<20 gave every one of the 73,728 of them a single
                chunk, generated on one thread while still paying for a
                24-thread wake-up. That measured 23 minutes where it should
                have been one.
            */
            constexpr size_t CHUNK = size_t(1) << 14;
            const size_t chunks = (n + CHUNK - 1) / CHUNK;
            const uint64_t base = seed ^ stableHash(name);

            kernels::Pool::instance().run(chunks, [&](size_t begin, size_t end) {
                for (size_t c = begin; c < end; ++c) {
                    math::random::Random rng(base + c * 0x9E3779B97F4A7C15ULL);

                    const size_t from = c * CHUNK;
                    const size_t to = std::min(from + CHUNK, n);

                    if (isDecay) {
                        for (size_t i = from; i < to; ++i)
                            out[i] = std::log(static_cast<float>(rng.randDouble(0.01, 16.0)));
                    } else {
                        for (size_t i = from; i < to; ++i)
                            out[i] = static_cast<float>(rng.normal(0.0, double(range)));
                    }
                }
            });
        };
    }


    /*
        perturbInitialiser(seed, epsilon):
        - every weight offset by an independent uniform draw from
          [-epsilon, +epsilon]. The trained structure is still there; the
          question is how much of it survives being jogged.

        The offset is ABSOLUTE, not a fraction of each weight, because that
        is what makes the experiment mean one thing: every weight is
        disturbed by the same amount regardless of how large it happens to
        be, so what the run measures is the model's tolerance and not a
        rescaling of it. A weight in one of these checkpoints is typically
        around 0.02, so an epsilon of 0.001 is a perturbation of a few
        percent and 0.02 is total.

        Same chunking and the same name-derived seeds as the synthetic
        initialiser, so --seed reproduces a perturbation exactly.
    */
    WeightInitialiser perturbInitialiser(uint64_t seed, float epsilon) const {
        return [seed, epsilon](const std::string& name,
                               const TensorInfo& info, float* out) {
            const size_t n = info.numel();

            constexpr size_t CHUNK = size_t(1) << 14;
            const size_t chunks = (n + CHUNK - 1) / CHUNK;
            const uint64_t base = seed ^ stableHash(name);

            kernels::Pool::instance().run(chunks, [&](size_t begin, size_t end) {
                for (size_t c = begin; c < end; ++c) {
                    math::random::Random rng(base + c * 0x9E3779B97F4A7C15ULL);

                    const size_t from = c * CHUNK;
                    const size_t to = std::min(from + CHUNK, n);

                    for (size_t i = from; i < to; ++i)
                        out[i] += static_cast<float>(
                            rng.randDouble(-double(epsilon), double(epsilon)));
                }
            });
        };
    }


    /* FNV-1a. std::hash makes no promise about being the same twice, and a
       seed that does not reproduce the same model defeats the point */
    static uint64_t stableHash(const std::string& text) {
        uint64_t h = 0xCBF29CE484222325ULL;
        for (unsigned char c : text) {
            h ^= c;
            h *= 0x100000001B3ULL;
        }
        return h;
    }


    static ModelWeights::Plan plan(const fs::path& directory, quant::Type quantise) {
        ModelWeights::Plan out;

        const files::json::ParseResult config =
            files::json::parseFile(directory / "config.json");
        if (!config.ok) {
            out.error = "config.json: " + config.error;
            return out;
        }

        const files::json::Value& root = config.value;
        const files::json::Value& text =
            root["text_config"].isObject() ? root["text_config"] : root;

        const ArchSpec* spec =
            specFor(root["model_type"].asString(text["model_type"].asString()));

        return ModelWeights::plan(directory, quantise,
                                  spec ? spec->skipPrefixes : std::vector<std::string>());
    }


    /*
        randomSeed: when set, the weights are interfered with.
        - epsilon of zero replaces them entirely: the topology, the tokenizer
          and every shape still come from the checkpoint, only the numbers
          are invented, and no payload is read at all.
        - epsilon above zero keeps the trained weights and offsets each one
          by a uniform draw from [-epsilon, +epsilon].
    */
    bool load(const fs::path& directory, const LoadProgress& onProgress = {},
              quant::Type quantise = quant::Type::None,
              const std::optional<uint64_t>& randomSeed = std::nullopt,
              float epsilon = 0.0f) {
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
            std::string known;
            for (const ArchSpec& k : specRegistry())
                known += (known.empty() ? "" : ", ") + k.modelType;
            error = "unsupported model_type \"" + modelType + "\" (known: " + known + ")";
            return false;
        }
        spec = *found;

        if (!cfg.loadFrom(text, spec)) { error = "config.json: bad or missing fields"; return false; }

        quantType = quantise;

        /* false: bfloat16 stays as-is rather than being widened. When
           quantising, each shard is converted and released as it is read */
        /* a speculative-decoding head carries a second full bank of experts
           that the ordinary forward pass never reads; do not pay for it */
        weights.skipPrefixes = spec.skipPrefixes;
        if (randomSeed && epsilon > 0.0f)
            weights.perturb = perturbInitialiser(*randomSeed, epsilon);
        else if (randomSeed)
            weights.synthesise = randomInitialiser(*randomSeed);

        if (!weights.load(directory, false, onProgress, quantise)) {
            error = weights.error;
            return false;
        }
        /*
            A multimodal checkpoint nests the text tower under
            model.language_model., a text-only one does not. Pick whichever
            prefix the file actually uses rather than guessing from the type
        */
        for (const std::string& candidate : spec.prefixCandidates)
            if (weights.contains(candidate + "norm.weight")) { prefix = candidate; break; }

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
    std::string named(const std::string& tmpl, size_t layer) const {
        return applyPrefix(expandName(tmpl, layer), prefix);
    }
    std::string named(const std::string& tmpl) const {
        return applyPrefix(tmpl, prefix);
    }
    std::string named(const std::string& tmpl, size_t layer, size_t expert) const {
        return applyPrefix(expandExpert(expandName(tmpl, layer), expert), prefix);
    }

    Weight fetch(const std::string& name) const {
        Weight w;
        if (name.empty() || !weights.contains(name)) return w;

        /* a quantised matrix is already [out, in] and never needs the Conv1D
           transpose: only GPT-2 uses that layout and it is not quantised */
        quant::QuantTensor q = weights.quantTensor(name);
        if (q.valid()) {
            w.q = q;
            w.outDim = q.outDim;
            w.inDim = q.inDim;
            return w;
        }

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
        embed = fetchEmbedding(named(spec.names.embed));
        if (!embed.valid()) { error = "missing " + named(spec.names.embed); return false; }

        finalNorm     = fetchVector(named(spec.names.finalNorm));
        finalNormBias = fetchVector(named(spec.names.finalNormBias));
        if (!finalNorm.storage) { error = "missing " + named(spec.names.finalNorm); return false; }

        if (spec.pos == PosKind::Learned) {
            posEmbed = fetchVector(named(spec.names.posEmbed));
            if (!posEmbed.storage) { error = "missing " + named(spec.names.posEmbed); return false; }
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

        quant::QuantTensor q = weights.quantTensor(name);
        if (q.valid()) {
            w.q = q;
            w.outDim = q.outDim;
            w.inDim = q.inDim;
            return w;
        }

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

            L.mixer = cfg.mixers.empty() ? MixerKind::Attention : cfg.mixers[i];

            L.inputNorm        = fetchVector(named(spec.names.inputNorm, i));
            L.inputNormBias    = fetchVector(named(spec.names.inputNormBias, i));
            L.postAttnNorm     = fetchVector(named(spec.names.postAttnNorm, i));
            L.postAttnNormBias = fetchVector(named(spec.names.postAttnNormBias, i));

            /* ---- Gated DeltaNet layers take an entirely different set ---- */
            if (L.mixer == MixerKind::GatedDeltaNet) {
                L.linQKV    = fetch(named(spec.names.linQKV, i));
                L.linA      = fetch(named(spec.names.linA, i));
                L.linB      = fetch(named(spec.names.linB, i));
                L.linZ      = fetch(named(spec.names.linZ, i));
                L.linOut    = fetch(named(spec.names.linOut, i));
                L.linConv   = fetchVector(named(spec.names.linConv, i));
                L.linALog   = fetchVector(named(spec.names.linALog, i));
                L.linDtBias = fetchVector(named(spec.names.linDtBias, i));
                L.linNorm   = fetchVector(named(spec.names.linNorm, i));

                if (!L.linQKV.valid() || !L.linOut.valid() || !L.linConv.storage
                    || !L.linALog.storage || !L.linDtBias.storage || !L.linNorm.storage) {
                    error = "missing linear-attention weights for layer " + std::to_string(i);
                    return false;
                }

                if (!loadFeedForward(L, i)) return false;
                continue;
            }

            if (spec.qkNorm) {
                L.qNorm = fetchVector(named(spec.names.qNorm, i));
                L.kNorm = fetchVector(named(spec.names.kNorm, i));
            }

            /* ---- attention projections --------------------------------- */
            if (!spec.names.qkv.empty()) {
                // one matrix holding q, k and v stacked in that order
                Weight fused = fetch(named(spec.names.qkv, i));
                if (!fused.valid()) { error = "missing " + named(spec.names.qkv, i); return false; }

                const size_t qw = cfg.qWidth(), kw = cfg.kvWidth();
                L.q = rowSlice(fused, 0,           qw);
                L.k = rowSlice(fused, qw,          qw + kw);
                L.v = rowSlice(fused, qw + kw,     qw + 2 * kw);

                Tensor<float> bias = fetchVector(named(spec.names.qkvBias, i));
                if (bias.storage) {
                    L.q.bias = vectorSlice(bias, 0,       qw);
                    L.k.bias = vectorSlice(bias, qw,      qw + kw);
                    L.v.bias = vectorSlice(bias, qw + kw, qw + 2 * kw);
                }
            } else {
                L.q = fetch(named(spec.names.q, i));
                L.k = fetch(named(spec.names.k, i));
                L.v = fetch(named(spec.names.v, i));
                L.q.bias = fetchVector(named(spec.names.qBias, i));
                L.k.bias = fetchVector(named(spec.names.kBias, i));
                L.v.bias = fetchVector(named(spec.names.vBias, i));
            }

            L.o = fetch(named(spec.names.o, i));
            L.o.bias = fetchVector(named(spec.names.oBias, i));

            if (!loadFeedForward(L, i)) return false;

            if (!L.q.valid() || !L.o.valid()) {
                error = "missing weights for layer " + std::to_string(i);
                return false;
            }
        }

        return true;
    }


    /*
        loadFeedForward(L, i):
        - the feed forward is the one part that does not care which mixer the
          layer uses, so both branches of loadLayers come through here.
          A mixture-of-experts layer is the same SwiGLU repeated 512 times
          with a router in front, which is why an Expert and a dense MLP are
          the same three matrices
    */
    bool loadFeedForward(LocalLayer& L, size_t i) {
        if (spec.mlp == MLPKind::MoE) {
            L.router      = fetch(named(spec.names.router, i));
            L.sharedScale = fetch(named(spec.names.sharedScale, i));
            L.shared.gate = fetch(named(spec.names.sharedGate, i));
            L.shared.up   = fetch(named(spec.names.sharedUp, i));
            L.shared.down = fetch(named(spec.names.sharedDown, i));

            if (!L.router.valid() || !L.sharedScale.valid()
                || !L.shared.gate.valid() || !L.shared.up.valid() || !L.shared.down.valid()) {
                error = "missing router or shared expert for layer " + std::to_string(i);
                return false;
            }

            L.experts.resize(cfg.numExperts);
            for (size_t e = 0; e < cfg.numExperts; ++e) {
                Expert& X = L.experts[e];
                X.gate = fetch(named(spec.names.expertGate, i, e));
                X.up   = fetch(named(spec.names.expertUp,   i, e));
                X.down = fetch(named(spec.names.expertDown, i, e));

                if (!X.gate.valid() || !X.up.valid() || !X.down.valid()) {
                    error = "missing expert " + std::to_string(e)
                          + " of layer " + std::to_string(i);
                    return false;
                }
            }
            return true;
        }

        if (!spec.names.gateUp.empty()) {
            Weight fused = fetch(named(spec.names.gateUp, i));
            if (!fused.valid()) { error = "missing " + named(spec.names.gateUp, i); return false; }
            L.gate = rowSlice(fused, 0, cfg.intermediate);
            L.up   = rowSlice(fused, cfg.intermediate, 2 * cfg.intermediate);
        } else {
            if (!spec.names.gate.empty()) L.gate = fetch(named(spec.names.gate, i));
            L.up = fetch(named(spec.names.up, i));
            L.up.bias = fetchVector(named(spec.names.upBias, i));
        }

        L.down = fetch(named(spec.names.down, i));
        L.down.bias = fetchVector(named(spec.names.downBias, i));

        if (!L.up.valid() || !L.down.valid()) {
            error = "missing mlp weights for layer " + std::to_string(i);
            return false;
        }
        return true;
    }


    size_t parameterCount() const { return weights.parameterCount(); }


    /* ===================================================================== //
        Moving the weights onto the device

        The weights were already narrowed on the way in, so there is nothing
        to convert here - the codes and scales the CPU kernels read are the
        same bytes the card gets. Uploading is a copy and a pointer.
    // ===================================================================== */

    struct DeviceReport {
        bool        ok         = false;
        size_t      matrices   = 0;
        size_t      bytes      = 0;
        size_t      hostFreed  = 0;     // RAM handed back after uploading
        size_t      budget     = 0;
        size_t      layersOnDevice = 0;
        size_t      layersTotal    = 0;
        bool        headOnDevice   = false;
        bool        partial        = false;
        std::string error;
    };

    /*
        forEachMatrix(visit):
        - every matrix a matmul reads. The embedding table is not here: it
          is row-gathered on the host and never multiplied by. When lm_head
          is tied it IS that table, and appears once, as lmHead.
        - the linear-attention and mixture-of-experts weights are included,
          so a Qwen3-Next layer uploads as readily as an attention one
    */
    template <typename Visit>
    static void forEachInLayer(LocalLayer& L, Visit visit) {
        visit(L.q); visit(L.k); visit(L.v); visit(L.o);
        visit(L.gate); visit(L.up); visit(L.down);

        visit(L.linQKV); visit(L.linA); visit(L.linB);
        visit(L.linZ);   visit(L.linOut);

        visit(L.router); visit(L.sharedScale);
        visit(L.shared.gate); visit(L.shared.up); visit(L.shared.down);
        for (Expert& e : L.experts) {
            visit(e.gate); visit(e.up); visit(e.down);
        }
    }

    template <typename Visit>
    void forEachMatrix(Visit visit) {
        visit(lmHead);
        for (LocalLayer& L : layers) forEachInLayer(L, visit);
    }

    /*
        deviceFootprint():
        - what toDevice() would allocate, so a model that will not fit can
          be turned away before half of it has crossed the bus
    */
    size_t deviceFootprint() {
        size_t total = 0;
        std::vector<const void*> seen;

        forEachMatrix([&](Weight& w) {
            const void* host = w.hostBytes();
            if (!host) return;
            for (const void* p : seen) if (p == host) return;
            seen.push_back(host);
            total += w.deviceBytes();
        });

        return total;
    }

    /*
        toDevice():
        - after this returns true, apply() routes through the GPU with no
          other change anywhere: the dispatch is a field on Weight, not a
          flag threaded through the forward pass.
        - deduplicated by host pointer. A tied lm_head is literally the
          embedding table, and on a 150k vocab that is over a gigabyte that
          would otherwise sit in VRAM twice
        - a weight the card will not take keeps devType None and runs on
          the CPU, which is why a partial upload is a slower model rather
          than a broken one
    */
    /*
        toDevice(plan):
        - uploads as much as the budget allows and leaves the rest on the
          CPU. Partial offload needs no machinery in the forward pass at
          all: apply() already dispatches on each weight's own devType, so
          a matrix that never made it across simply runs the CPU kernel,
          and a layer split down the middle is not a special case.
        - it is not fast. A token still reads every weight once, so the
          time is roughly cpuBytes/60GB/s + gpuBytes/400GB/s and the CPU
          half dominates the moment it is more than a sliver. The point is
          that a model one gigabyte too large runs a bit slower instead of
          not at all.
        - whole layers go or do not go. Splitting one would work, but "38
          of 40 layers" is something you can reason about and "72% of the
          matrices" is not.
    */
    DeviceReport toDevice(const OffloadPlan& plan = {}, const LoadProgress& onProgress = {}) {
        DeviceReport report;
        report.layersTotal = layers.size();

        if (!gpu::available()) { report.error = gpu::lastError(); return report; }

        const size_t needed = deviceFootprint();

        /*
            What is free, less a reserve. The reserve is not superstition:
            the scratch buffers that stage activations grow with the prompt
            (a 512-token prefill against a 150k vocab is 300MB by itself),
            and the desktop compositor wants its own VRAM back the moment
            anything else on the machine redraws.
        */
        size_t budget = plan.vramBudget;
        if (budget == 0) {
            const size_t free = gpu::freeMemory();
            budget = free > plan.reserve ? free - plan.reserve : 0;
        }
        report.budget = budget;

        /* what we expect to actually send, fixed here rather than read off
           `budget` - that shrinks as layers commit, and a progress total
           that shrinks while the numerator grows reports over 100% */
        const size_t plannedTotal = std::min(needed, budget);

        struct Uploaded {
            const void*    host;
            gpu::BufferRef dev;
            gpu::BufferRef scales;
            DevDType       type;
        };

        std::vector<Uploaded> seen;
        bool failed = false;

        /* what this weight would add, counting a shared matrix only once */
        auto costOf = [&](Weight& w) -> size_t {
            if (!w.valid() || w.onDevice()) return 0;
            const void* host = w.hostBytes();
            if (!host) return 0;
            for (const Uploaded& e : seen) if (e.host == host) return 0;
            return w.deviceBytes();
        };

        auto upload = [&](Weight& w) {
            if (failed || !w.valid() || w.onDevice()) return;

            const void* host = w.hostBytes();
            if (!host) return;              // a view that would need gathering

            for (const Uploaded& e : seen) {
                if (e.host != host) continue;
                w.dev = e.dev; w.devScales = e.scales; w.devType = e.type;
                return;
            }

            gpu::BufferRef buffer, scales;
            DevDType       type;

            if (w.isQuant()) {
                /* both kernels index scales per block of 32 along K, which
                   quant::quantize guarantees by refusing anything else */
                buffer = gpu::upload(w.q.quants.get(), w.quantCodeBytes());
                scales = buffer ? gpu::upload(w.q.scales.get(), w.quantScaleBytes()) : nullptr;
                type   = (w.q.type == quant::Type::Q4) ? DevDType::Q4 : DevDType::Q8;
            } else if (w.isBF16()) {
                buffer = gpu::upload(host, w.deviceBytes());
                type   = DevDType::BF16;
            } else {
                buffer = gpu::upload(host, w.deviceBytes());
                type   = DevDType::F32;
            }

            if (!buffer || (w.isQuant() && !scales)) {
                failed = true;
                report.error = gpu::lastError();
                return;
            }

            w.dev = buffer; w.devScales = scales; w.devType = type;
            seen.push_back({host, buffer, scales, type});

            ++report.matrices;
            report.bytes += w.deviceBytes();
            if (onProgress)
                onProgress(report.bytes, plannedTotal,
                           "uploading to " + gpu::deviceName());
        };

        /*
            The output projection goes first. It is read once per token like
            everything else, so nothing is gained by its position - but it
            is a single large matrix, and having it either wholly on or
            wholly off keeps the layer arithmetic below honest.
        */
        if (costOf(lmHead) <= budget) {
            report.bytes = 0;
            const size_t cost = costOf(lmHead);
            upload(lmHead);
            report.headOnDevice = !failed && lmHead.onDevice();
            if (report.headOnDevice) budget -= cost;
        }

        for (LocalLayer& L : layers) {
            if (failed) break;

            size_t cost = 0;
            forEachInLayer(L, [&](Weight& w) { cost += costOf(w); });

            if (cost > budget) break;       // stop at the first that will not fit

            forEachInLayer(L, upload);
            if (failed) break;

            budget -= cost;
            ++report.layersOnDevice;
        }

        if (failed) { toHost(); return report; }

        report.partial = report.layersOnDevice < report.layersTotal || !report.headOnDevice;

        /*
            Drop the host copy of everything the card now holds.

            Without this, --gpu costs exactly as much system RAM as running
            on the CPU did - the weights sit in VRAM and in the heap at the
            same time. Measured on Qwen3-8B that was 16.6 GB resident for a
            model the GPU was doing all the work for, which on a 32GB
            machine leaves no room for a desktop: the kernel OOM-killer
            takes the process the moment an editor starts.

            Only what actually went across is released, so under a partial
            offload the layers still running on the CPU keep their weights
            and stay usable.

            The embedding table is deliberately untouched. It is row-
            gathered on the host every token and never multiplied by, so it
            is not in forEachMatrix and nothing here can reach it. A tied
            lm_head shares its storage through a shared_ptr, so releasing
            the lm_head view leaves the table itself alive.

            The cost is that toHost() has nothing to fall back to
            afterwards. Pass releaseHost false when a run needs to compare
            the two paths against each other, which is what the test suite
            does.
        */
        if (plan.releaseHost) {
            forEachMatrix([&](Weight& w) {
                if (!w.onDevice()) return;
                w.bf16 = Tensor<uint16_t>();
                w.f32  = Tensor<float>();
                w.q    = quant::QuantTensor();
                // the bias stays: apply() still folds it on the host
            });

            /*
                and then at the level that actually owns the bytes. A Weight
                is a view - ModelWeights keeps the storage - so clearing the
                views above drops one reference each and frees nothing on
                its own. This is the line that returns the memory.
            */
            report.hostFreed = weights.releaseUnreferenced();
            hostReleased = true;
        }

        report.ok = true;
        return report;
    }

    /* true once toDevice() has dropped the host weights, after which the
       CPU path is no longer available without reloading */
    bool hostReleased = false;

    /*
        drops every device buffer and puts the forward pass back on the CPU.
        Only meaningful when the host copy was kept - see toDevice's
        releaseHost
    */
    void toHost() {
        forEachMatrix([](Weight& w) {
            w.dev.reset();
            w.devScales.reset();
            w.devType = DevDType::None;
        });
    }


    /* ===================================================================== //
        Forward
    // ===================================================================== */

    void normRow(const float* x, const Tensor<float>& w, const Tensor<float>& b,
                 float* out, size_t n) const {
        if (spec.norm == NormKind::RMSNorm)
            rmsNormRow(x, w.ptr, out, n, cfg.normEps, spec.normWeightOffset);
        else layerNormRow(x, w.ptr, b.storage ? b.ptr : nullptr, out, n, cfg.normEps);
    }

    float activation(float x) const {
        return spec.act == ActKind::SiLU ? siluOf(x) : geluTanhOf(x);
    }

    static void widenRow(const Weight& table, size_t index, float* out, size_t n) {
        if (table.isQuant()) { quant::dequantRow(table.q, index, out); return; }

        if (table.isBF16()) {
            const uint16_t* row = table.bf16.ptr + index * n;
            for (size_t i = 0; i < n; ++i)
                out[i] = std::bit_cast<float>(static_cast<uint32_t>(row[i]) << 16);
        } else {
            const float* row = table.f32.ptr + index * n;
            std::copy(row, row + n, out);
        }
    }


    /*
        attentionMixer(h, L, layer, count, startPos, cache, out):
        - ordinary softmax attention, with two options the newer models add.
          attnOutputGate makes q_proj emit twice the query width, the second
          half of EACH HEAD being a gate applied to that head's output - the
          halves interleave per head, they are not two halves of the matrix.
          A partial rotary factor rotates only the leading dimensions of each
          head and leaves the rest carrying no position at all.
    */
    void attentionMixer(const float* h, const LocalLayer& L, size_t layer,
                        size_t count, size_t startPos, LocalCache& cache,
                        float* out, const std::string& tag, LocalTrace* trace) const {
        const size_t D  = cfg.headDim;
        const size_t QW = cfg.qWidth();
        const size_t KW = cfg.kvWidth();
        const size_t qStride = spec.attnOutputGate ? D * 2 : D;
        const size_t total = startPos + count;

        std::vector<float> q(count * cfg.nHead * qStride);
        std::vector<float> k(count * KW), v(count * KW), attn(count * QW);
        std::vector<float> scores(total);

        /*
            the reference dumps queries without the gate interleaved into
            them, so the trace compacts the query halves back together rather
            than handing over the strided buffer as it stands
        */
        auto traceQ = [&](const std::string& name) {
            if (!trace) return;
            std::vector<float> flat(count * QW);
            for (size_t t = 0; t < count; ++t)
                for (size_t head = 0; head < cfg.nHead; ++head)
                    std::copy(&q[t * cfg.nHead * qStride + head * qStride],
                              &q[t * cfg.nHead * qStride + head * qStride] + D,
                              &flat[t * QW + head * D]);
            trace->add(name, flat.data(), flat.size());
        };

        apply(h, L.q, q.data(), count);
        apply(h, L.k, k.data(), count);
        apply(h, L.v, v.data(), count);

        for (size_t t = 0; t < count; ++t) {
            if (spec.qkNorm) {
                for (size_t head = 0; head < cfg.nHead; ++head) {
                    float* qh = &q[t * cfg.nHead * qStride + head * qStride];
                    rmsNormRow(qh, L.qNorm.ptr, qh, D, cfg.normEps, spec.normWeightOffset);
                }
                for (size_t head = 0; head < cfg.nKVHead; ++head)
                    rmsNormRow(&k[t * KW + head * D], L.kNorm.ptr,
                               &k[t * KW + head * D], D, cfg.normEps, spec.normWeightOffset);
            }
        }
        if (trace && spec.qkNorm) {
            traceQ(tag + "qnorm");
            trace->add(tag + "knorm", k.data(), count * KW);
        }

        /* rotate. With a gate present the query rows are strided, so each
           head is rotated on its own rather than as one contiguous run */
        if (spec.pos == PosKind::RoPE) {
            const size_t rot = cfg.rotaryDims();
            for (size_t t = 0; t < count; ++t) {
                for (size_t head = 0; head < cfg.nHead; ++head)
                    ropeRow(&q[t * cfg.nHead * qStride + head * qStride], 1, D,
                            startPos + t, cfg.ropeTheta, rot);
                ropeRow(&k[t * KW], cfg.nKVHead, D, startPos + t, cfg.ropeTheta, rot);
            }
            if (trace) {
                traceQ(tag + "qrope");
                trace->add(tag + "krope", k.data(), count * KW);
            }
        }

        for (size_t t = 0; t < count; ++t) {
            std::copy(&k[t * KW], &k[t * KW] + KW,
                      cache.keys[layer].begin() + static_cast<long>((startPos + t) * KW));
            std::copy(&v[t * KW], &v[t * KW] + KW,
                      cache.values[layer].begin() + static_cast<long>((startPos + t) * KW));
        }

        const float invSqrtD = 1.0f / std::sqrt(static_cast<float>(D));

        for (size_t t = 0; t < count; ++t) {
            const size_t position = startPos + t;

            for (size_t head = 0; head < cfg.nHead; ++head) {
                const size_t kvHead = head / cfg.group();
                const float* qh = &q[t * cfg.nHead * qStride + head * qStride];

                float best = -1e30f;
                for (size_t j = 0; j <= position; ++j) {
                    const float sc =
                        kernels::dot(qh, &cache.keys[layer][j * KW + kvHead * D], D) * invSqrtD;
                    scores[j] = sc;
                    best = std::max(best, sc);
                }

                float sum = 0.0f;
                for (size_t j = 0; j <= position; ++j) {
                    scores[j] = std::exp(scores[j] - best);
                    sum += scores[j];
                }
                const float invSum = 1.0f / sum;

                float* o = &attn[t * QW + head * D];
                for (size_t d = 0; d < D; ++d) o[d] = 0.0f;
                for (size_t j = 0; j <= position; ++j) {
                    const float w = scores[j] * invSum;
                    const float* vh = &cache.values[layer][j * KW + kvHead * D];
                    for (size_t d = 0; d < D; ++d) o[d] += w * vh[d];
                }

                // the gate sits in the back half of this head's query row
                if (spec.attnOutputGate) {
                    const float* g = qh + D;
                    for (size_t d = 0; d < D; ++d) o[d] *= sigmoidOf(g[d]);
                }
            }
        }

        apply(attn.data(), L.o, out, count);
        if (trace) trace->add(tag + "attnout", out, count * cfg.hidden);
    }


    /*
        gatedDeltaNetMixer(...):
        - linear attention. Instead of keeping every past key and value and
          scoring against all of them, each head carries ONE [keyDim, valueDim]
          matrix of state. A token decays that state, writes itself into it
          with a delta rule, and reads its output back out of it.
        - the consequence is that cost per token is constant rather than
          growing with the context, which is the whole reason the architecture
          exists. It also means these layers contribute nothing to the cache.
        - the causal convolution in front of it mixes each channel with the
          previous three timesteps, so the carried window is what makes
          decoding one token at a time agree with processing them together.
    */
    void gatedDeltaNetMixer(const float* h, const LocalLayer& L, size_t layer,
                            size_t count, LocalCache& cache,
                            float* out, const std::string& tag, LocalTrace* trace) const {
        const size_t KW = cfg.linKeyWidth();
        const size_t VW = cfg.linValueWidth();
        const size_t CW = cfg.linConvWidth();
        const size_t KD = cfg.linKeyDim;
        const size_t VD = cfg.linValueDim;
        const size_t nV = cfg.linValueHeads;
        const size_t group = cfg.linGroup();
        const size_t K = cfg.convKernel;

        std::vector<float> qkv(count * CW), z(count * VW);
        std::vector<float> a(count * nV), b(count * nV);

        if (spec.linFusedQKVZ) {
            /*
                Two matrices instead of four, and both laid out PER KEY HEAD:
                one key head's q, then its k, then its group of value heads'
                v, then that same group's z, and only then the next key head.
                Slicing the output into four blocks would interleave every
                part's heads with each other - the numbers stay plausible and
                the answer is wrong - so it is de-interleaved head by head.
            */
            const size_t nK = cfg.linKeyHeads;
            const size_t stride = 2 * KD + 2 * group * VD;
            const size_t baStride = 2 * group;

            std::vector<float> fused(count * nK * stride);
            std::vector<float> ba(count * nK * baStride);
            apply(h, L.linQKV, fused.data(), count);
            apply(h, L.linA,   ba.data(),    count);

            for (size_t t = 0; t < count; ++t)
                for (size_t kHead = 0; kHead < nK; ++kHead) {
                    const float* head = &fused[t * nK * stride + kHead * stride];
                    const float* pair = &ba[t * nK * baStride + kHead * baStride];

                    std::copy(head,      head + KD,     &qkv[t * CW + kHead * KD]);
                    std::copy(head + KD, head + 2 * KD, &qkv[t * CW + KW + kHead * KD]);

                    for (size_t j = 0; j < group; ++j) {
                        const size_t vHead = kHead * group + j;
                        const float* vSrc = head + 2 * KD + j * VD;
                        const float* zSrc = vSrc + group * VD;

                        std::copy(vSrc, vSrc + VD, &qkv[t * CW + 2 * KW + vHead * VD]);
                        std::copy(zSrc, zSrc + VD, &z[t * VW + vHead * VD]);

                        b[t * nV + vHead] = pair[j];
                        a[t * nV + vHead] = pair[group + j];
                    }
                }
        } else {
            apply(h, L.linQKV, qkv.data(), count);
            apply(h, L.linZ,   z.data(),   count);
            apply(h, L.linA,   a.data(),   count);
            apply(h, L.linB,   b.data(),   count);
        }

        if (trace) {
            trace->add(tag + "qkv", qkv.data(), count * CW);
            trace->add(tag + "z", z.data(), count * VW);
            trace->add(tag + "a", a.data(), count * nV);
            trace->add(tag + "b", b.data(), count * nV);
        }

        /* ---- causal depthwise convolution, kernel K, per channel -------- */
        std::vector<float>& window = cache.convWindow[layer];   // (K-1) * CW
        std::vector<float> conv(count * CW);

        for (size_t t = 0; t < count; ++t) {
            for (size_t c = 0; c < CW; ++c) {
                float acc = 0.0f;
                for (size_t j = 0; j < K; ++j) {
                    const long at = static_cast<long>(t) - static_cast<long>(K - 1)
                                  + static_cast<long>(j);
                    /* a negative index reaches back into the carried
                       window, whose last entry is the most recent input */
                    const float xv = (at >= 0)
                        ? qkv[static_cast<size_t>(at) * CW + c]
                        : window[static_cast<size_t>(static_cast<long>(K) - 1 + at) * CW + c];
                    acc += L.linConv.ptr[c * K + j] * xv;
                }
                conv[t * CW + c] = siluOf(acc);
            }
        }

        /* carry the last K-1 inputs forward for the next call */
        {
            std::vector<float> next((K - 1) * CW, 0.0f);
            for (size_t i = 0; i < K - 1; ++i) {
                const long at = static_cast<long>(count) - static_cast<long>(K - 1)
                              + static_cast<long>(i);
                const size_t from = (at >= 0) ? static_cast<size_t>(at)
                                              : static_cast<size_t>(static_cast<long>(K) - 1 + at);
                const float* src = (at >= 0) ? &qkv[from * CW] : &window[from * CW];
                std::copy(src, src + CW, &next[i * CW]);
            }
            window.swap(next);
        }
        if (trace) trace->add(tag + "conv", conv.data(), count * CW);

        /* ---- the recurrence -------------------------------------------- */
        std::vector<float>& state = cache.recurrent[layer];     // nV * KD * VD
        std::vector<float> core(count * VW);
        std::vector<float> qh(KD), kh(KD), kv(VD), delta(VD);

        for (size_t t = 0; t < count; ++t) {
            const float* qRow = &conv[t * CW];
            const float* kRow = qRow + KW;
            const float* vRow = kRow + KW;

            for (size_t vHead = 0; vHead < nV; ++vHead) {
                const size_t kHead = vHead / group;

                /* q and k are unit-normalised so the state cannot be driven
                   by their magnitude, then q is scaled by the head width */
                std::copy(qRow + kHead * KD, qRow + kHead * KD + KD, qh.begin());
                std::copy(kRow + kHead * KD, kRow + kHead * KD + KD, kh.begin());
                l2normalise(qh.data(), KD);
                l2normalise(kh.data(), KD);

                const float qScale = 1.0f / std::sqrt(static_cast<float>(KD));
                for (size_t i = 0; i < KD; ++i) qh[i] *= qScale;

                const float aPlusBias = a[t * nV + vHead] + L.linDtBias.ptr[vHead];
                const float g = -std::exp(L.linALog.ptr[vHead]) * softplusOf(aPlusBias);
                const float decay = std::exp(g);
                const float beta = sigmoidOf(b[t * nV + vHead]);

                float* S = &state[vHead * KD * VD];

                for (size_t i = 0; i < KD * VD; ++i) S[i] *= decay;

                for (size_t d = 0; d < VD; ++d) kv[d] = 0.0f;
                for (size_t i = 0; i < KD; ++i) {
                    const float ki = kh[i];
                    const float* row = S + i * VD;
                    for (size_t d = 0; d < VD; ++d) kv[d] += row[d] * ki;
                }

                const float* vh = vRow + vHead * VD;
                for (size_t d = 0; d < VD; ++d) delta[d] = (vh[d] - kv[d]) * beta;

                for (size_t i = 0; i < KD; ++i) {
                    const float ki = kh[i];
                    float* row = S + i * VD;
                    for (size_t d = 0; d < VD; ++d) row[d] += ki * delta[d];
                }

                float* o = &core[t * VW + vHead * VD];
                for (size_t d = 0; d < VD; ++d) o[d] = 0.0f;
                for (size_t i = 0; i < KD; ++i) {
                    const float qi = qh[i];
                    const float* row = S + i * VD;
                    for (size_t d = 0; d < VD; ++d) o[d] += row[d] * qi;
                }
            }
        }

        /* ---- normalise each head, gated by z, then project out ---------- */
        for (size_t t = 0; t < count; ++t)
            for (size_t vHead = 0; vHead < nV; ++vHead) {
                float* o = &core[t * VW + vHead * VD];
                /* the gated norm stores its scale directly, unlike the
                   ordinary norms in this same architecture */
                rmsNormRow(o, L.linNorm.ptr, o, VD, cfg.normEps);
                const float* gate = &z[t * VW + vHead * VD];
                for (size_t d = 0; d < VD; ++d) o[d] *= siluOf(gate[d]);
            }
        if (trace) trace->add(tag + "gatednorm", core.data(), count * VW);

        apply(core.data(), L.linOut, out, count);
        if (trace) trace->add(tag + "attnout", out, count * cfg.hidden);
    }


    /*
        moeBlock(h, L, count, out):
        - the feed forward as a bank of small experts. A router scores every
          one of them, the best few run, and their outputs are summed in
          proportion to those scores. One shared expert runs for every token
          regardless and is added on top, scaled by a gate of its own.
        - the softmax is taken over ALL the experts and the top few picked
          from the result - NOT the top few logits softmaxed among
          themselves. The denominators differ, so the two give different
          weights even though the same experts are chosen.
        - the weights are what makes this model 80B and the ten experts are
          what makes it behave like a 3B one: the parameters sit on the bus,
          and only the ones a token actually asked for are ever read.
    */
    void moeBlock(const float* h, const LocalLayer& L, size_t count,
                  float* out, const std::string& tag, LocalTrace* trace) const {
        const size_t H = cfg.hidden;
        const size_t E = cfg.numExperts;
        const size_t K = cfg.expertsPerToken;
        const size_t M = cfg.moeIntermediate;
        const size_t S = cfg.sharedIntermediate;

        /* ---- the shared expert, which every token pays for -------------- */
        std::vector<float> sg(count * S), su(count * S), scale(count);

        apply(h, L.shared.gate, sg.data(), count);
        apply(h, L.shared.up,   su.data(), count);
        for (size_t i = 0; i < count * S; ++i) sg[i] = activation(sg[i]) * su[i];
        apply(sg.data(), L.shared.down, out, count);

        apply(h, L.sharedScale, scale.data(), count);
        for (size_t t = 0; t < count; ++t) {
            const float g = sigmoidOf(scale[t]);
            for (size_t i = 0; i < H; ++i) out[t * H + i] *= g;
        }
        if (trace) trace->add(tag + "shared", out, count * H);

        /* ---- route ------------------------------------------------------ */
        std::vector<float>  logits(count * E);
        std::vector<size_t> chosen(count * K);
        std::vector<float>  share(count * K);
        std::vector<size_t> order(E);

        apply(h, L.router, logits.data(), count);

        for (size_t t = 0; t < count; ++t) {
            float* row = &logits[t * E];

            float best = -1e30f;
            for (size_t e = 0; e < E; ++e) best = std::max(best, row[e]);

            float sum = 0.0f;
            for (size_t e = 0; e < E; ++e) { row[e] = std::exp(row[e] - best); sum += row[e]; }

            const float inv = 1.0f / sum;
            for (size_t e = 0; e < E; ++e) row[e] *= inv;

            /* ten out of five hundred: a selection pass, not a sort */
            for (size_t e = 0; e < E; ++e) order[e] = e;
            std::nth_element(order.begin(), order.begin() + static_cast<long>(K), order.end(),
                             [row](size_t x, size_t y) { return row[x] > row[y]; });

            float total = 0.0f;
            for (size_t j = 0; j < K; ++j) {
                chosen[t * K + j] = order[j];
                share[t * K + j]  = row[order[j]];
                total += share[t * K + j];
            }

            /* the chosen few carry only part of the distribution, so they
               are rescaled to sum to one rather than to whatever they got */
            if (cfg.normTopK && total > 0.0f) {
                const float invTotal = 1.0f / total;
                for (size_t j = 0; j < K; ++j) share[t * K + j] *= invTotal;
            }
        }

        /*
            Run each expert once over every token that chose it, rather than
            once per token. At decode there is one token and this is a plain
            mat-vec either way; across a prompt it is the difference between
            reading an expert's weights once and reading them once per token
            that wanted it.
        */
        std::vector<size_t> tally(E, 0);
        for (size_t i = 0; i < count * K; ++i) tally[chosen[i]]++;

        std::vector<size_t> start(E + 1, 0);
        for (size_t e = 0; e < E; ++e) start[e + 1] = start[e] + tally[e];

        std::vector<size_t> fill(start.begin(), start.end() - 1);
        std::vector<size_t> slots(count * K);
        for (size_t i = 0; i < count * K; ++i) slots[fill[chosen[i]]++] = i;

        std::vector<size_t> active;
        active.reserve(E);
        for (size_t e = 0; e < E; ++e) if (tally[e]) active.push_back(e);

        /* one expert, start to finish, writing its share into `into` */
        const auto runExpert = [&](size_t which, float* into, bool threaded) {
            const size_t e = active[which];
            const size_t n = tally[e];

            std::vector<float> xg(n * H), eg(n * M), eu(n * M), eo(n * H);

            for (size_t r = 0; r < n; ++r) {
                const size_t t = slots[start[e] + r] / K;
                std::copy(h + t * H, h + t * H + H, &xg[r * H]);
            }

            apply(xg.data(), L.experts[e].gate, eg.data(), n, threaded);
            apply(xg.data(), L.experts[e].up,   eu.data(), n, threaded);
            for (size_t i = 0; i < n * M; ++i) eg[i] = activation(eg[i]) * eu[i];
            apply(eg.data(), L.experts[e].down, eo.data(), n, threaded);

            for (size_t r = 0; r < n; ++r) {
                const size_t slot = slots[start[e] + r];
                const size_t t = slot / K;
                const float  w = share[slot];
                for (size_t i = 0; i < H; ++i) into[t * H + i] += w * eo[r * H + i];
            }
        };

        const size_t lanes = std::min(active.size(), kernels::Pool::instance().size());

        if (count == 1 && lanes > 1) {
            /*
                Decoding: the experts themselves are the parallel axis rather
                than the rows inside each of their matmuls. Ten half-megabyte
                experts is not enough work to pay for thirty pool wake-ups per
                layer, and a thread that streams one whole expert reads it
                contiguously - which at batch one, where this is all memory
                traffic and barely any arithmetic, is the entire game.

                Each lane accumulates into its own row and the rows are summed
                after, since two experts the same token chose would otherwise
                write the same elements at once.
            */
            std::vector<std::vector<float>> partial(lanes, std::vector<float>(H, 0.0f));
            std::atomic<size_t> claim{0};

            kernels::Pool::instance().run(active.size(), [&](size_t begin, size_t end) {
                float* into = partial[claim.fetch_add(1, std::memory_order_relaxed)].data();
                for (size_t i = begin; i < end; ++i) runExpert(i, into, false);
            });

            for (const std::vector<float>& part : partial)
                for (size_t i = 0; i < H; ++i) out[i] += part[i];
        } else {
            /* Prefilling: every expert has many rows to do, so there is
               ample work inside each matmul and one accumulator will do */
            for (size_t i = 0; i < active.size(); ++i) runExpert(i, out, true);
        }

        if (trace) trace->add(tag + "mlpout", out, count * H);
    }


    void forward(const size_t* ids, size_t count, size_t startPos,
                 LocalCache& cache, float* logitsOut, LocalTrace* trace = nullptr) const {
        const size_t H = cfg.hidden;

        std::vector<float> x(count * H), h(count * H), proj(count * H);

        /* a mixture-of-experts layer sizes its own scratch per expert, so
           the dense feed-forward buffers are not wanted at all */
        const size_t ff = (spec.mlp == MLPKind::MoE) ? 0 : cfg.intermediate;
        std::vector<float> gate(count * ff), up(count * ff);

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

            /* the only place the two layer kinds differ: everything either
               side of this - the norms, the residual, the feed forward - is
               identical whichever mixer sits in the middle */
            if (L.mixer == MixerKind::GatedDeltaNet)
                gatedDeltaNetMixer(h.data(), L, l, count, cache, proj.data(), tag, trace);
            else
                attentionMixer(h.data(), L, l, count, startPos, cache,
                               proj.data(), tag, trace);

            for (size_t i = 0; i < count * H; ++i) x[i] += proj[i];
            if (trace) trace->add(tag + "res1", x.data(), count * H);

            for (size_t t = 0; t < count; ++t)
                normRow(&x[t * H], L.postAttnNorm, L.postAttnNormBias, &h[t * H], H);
            if (trace) trace->add(tag + "ln2", h.data(), count * H);

            if (spec.mlp == MLPKind::MoE) {
                moeBlock(h.data(), L, count, proj.data(), tag, trace);
            } else {
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
            }

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

    /*
        chatPrompt(user, thinking, system):
        - `system` is emitted only when it is non-empty, and belongs to the
          START of a conversation. A follow-up turn appended to a live cache
          must not repeat it: the earlier turns are already in the KV cache
          and re-rendering them would put the whole preamble in twice.
        - it is worth having at all because an instruct model trained on
          agentic traces will otherwise reach for tools it does not have.
          Ask one of those to write a program and it answers by listing the
          directory and opening package.json, which is not a malfunction -
          it is the behaviour it was taught, applied to a prompt that never
          said there is no repository and no tools to call
    */
    std::string chatPrompt(const std::string& user, bool thinking = false,
                           const std::string& system = {}) const {
        switch (chatFormat()) {
            case ChatFormat::MistralInst:
                /* the format has no system role, and the convention is to
                   put it at the top of the first instruction block */
                return "[INST] " + (system.empty() ? "" : system + "\n\n")
                     + user + " [/INST]";

            case ChatFormat::ChatMLSep:
                // Phi-4 separates the role from the content with a token
                // rather than a newline
                return (system.empty() ? ""
                        : "<|im_start|>system<|im_sep|>" + system + "<|im_end|>")
                     + "<|im_start|>user<|im_sep|>" + user
                     + "<|im_end|><|im_start|>assistant<|im_sep|>";

            case ChatFormat::ChatML: {
                std::string prompt =
                    (system.empty() ? "" : "<|im_start|>system\n" + system + "<|im_end|>\n")
                    + "<|im_start|>user\n" + user
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
