"""
Reference for the qwen3_next architecture (Qwen3-Next-80B-A3B), via
transformers itself.

Two things separate it from qwen3_5, and both are the kind of mistake that
produces confident nonsense rather than an error:

  * the linear-attention inputs arrive as two FUSED matrices, in_proj_qkvz and
    in_proj_ba, laid out per KEY head - one head's q, its k, its group of
    value heads' v, then that same group's z, and only then the next head.
    Slicing the output into four blocks instead of de-interleaving it gives
    numbers of the right magnitude and the wrong answer.

  * the feed forward is a bank of experts. A router softmaxes over ALL of
    them, the best few run, and those few are rescaled to sum to one.
    (With norm_topk_prob set, as it is here and in the real model, the
    rescaling cancels the softmax denominator entirely - so softmaxing only
    the chosen few would land on the same numbers. It does not when that flag
    is off, which is why the order is worth writing down.) What does not
    cancel is which experts are chosen and how the chosen are weighted
    against one another.

Writing a numpy reference from the same reading of the paper would prove
nothing - two self-consistent wrong implementations agree perfectly - so the
reference is transformers' own module, on a TINY randomly initialised model
rather than the real 151GB checkpoint.

    py tools/qwen3next_oracle.py --out models/tiny-qwen3next --dump out/qwen3next-oracle

The weights are written in the same PER-EXPERT layout the hub checkpoint uses
(mlp.experts.{E}.gate_proj.weight and friends) rather than the stacked form
transformers keeps in memory, so the C++ loader opens this exactly as it would
the real thing.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM


FULL_ATTENTION_INTERVAL = 2


def tiny_config():
    """
    Small enough to run instantly, and still exercising every axis that can
    be got wrong: both layer kinds, grouped heads on both the linear and the
    full attention (so the per-key-head interleaving actually interleaves),
    a head_dim that is not hidden/heads, a partial rotary factor, and more
    experts than run per token.
    """
    return Qwen3NextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=FULL_ATTENTION_INTERVAL,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act="silu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )


class Recorder:
    """Hooks every submodule and keeps the tensor it returned."""

    def __init__(self, dump_dir):
        self.dir = dump_dir
        self.entries = []
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    def add(self, name, tensor):
        if not self.dir or tensor is None:
            return
        a = np.ascontiguousarray(tensor.detach().to(torch.float32).numpy())
        a.tofile(os.path.join(self.dir, name + ".bin"))
        self.entries.append({"name": name, "shape": list(a.shape)})

    def finish(self):
        if not self.dir:
            return
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(self.entries, f, indent=2)


def attach(model, rec):
    """
    One hook per interesting module. The names match what the C++ trace
    emits, so the two compare entry by entry.
    """
    handles = []
    lm = model.model

    def hook(name):
        def fn(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            rec.add(name, out)
        return fn

    handles.append(lm.embed_tokens.register_forward_hook(hook("embed")))

    for i, layer in enumerate(lm.layers):
        tag = "l" + str(i) + "."
        handles.append(layer.input_layernorm.register_forward_hook(hook(tag + "ln1")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook(tag + "ln2")))
        handles.append(layer.mlp.register_forward_hook(hook(tag + "mlpout")))

        mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        handles.append(mixer.register_forward_hook(hook(tag + "attnout")))

        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            handles.append(la.conv1d.register_forward_hook(hook(tag + "conv")))
            handles.append(la.norm.register_forward_hook(hook(tag + "gatednorm")))

    handles.append(lm.norm.register_forward_hook(hook("norm")))
    return handles


def per_expert_state(model, cfg):
    """
    transformers keeps the experts stacked - gate_up_proj is [E, 2M, H] and
    down_proj is [E, H, M] - while the checkpoint on the hub stores each
    expert as its own pair of matrices. Write the hub's form, since that is
    what the loader under test has to read.
    """
    state = {}
    M = cfg.moe_intermediate_size

    for key, value in model.state_dict().items():
        if key.endswith("experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            for e in range(value.shape[0]):
                state[base + str(e) + ".gate_proj.weight"] = value[e, :M].clone()
                state[base + str(e) + ".up_proj.weight"] = value[e, M:].clone()
        elif key.endswith("experts.down_proj"):
            base = key[: -len("down_proj")]
            for e in range(value.shape[0]):
                state[base + str(e) + ".down_proj.weight"] = value[e].clone()
        else:
            state[key] = value

    return {k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/tiny-qwen3next")
    ap.add_argument("--dump", default="out/qwen3next-oracle")
    ap.add_argument("--tokenizer", default="models/qwen3-0.6b/tokenizer.json")
    ap.add_argument("--tokens", default="5,17,42,7,300,12")
    args = ap.parse_args()

    torch.manual_seed(1234)

    cfg = tiny_config()
    model = Qwen3NextForCausalLM(cfg).eval()

    """
    A random init leaves A_log and dt_bias at their constructor values and
    the router at zeros - which would make every expert equally likely and
    hide any routing mistake completely. Give both something with structure.
    """
    with torch.no_grad():
        for layer in model.model.layers:
            la = getattr(layer, "linear_attn", None)
            if la is not None:
                la.A_log.copy_(torch.log(torch.empty_like(la.A_log).uniform_(0.5, 8.0)))
                la.dt_bias.copy_(torch.empty_like(la.dt_bias).uniform_(-1.0, 1.0))

            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.weight.normal_(0.0, 0.5)

    os.makedirs(args.out, exist_ok=True)
    save_file(per_expert_state(model, cfg), os.path.join(args.out, "model.safetensors"))

    """
    Qwen3NextConfig derives layer_types in its constructor and serialises the
    result, dropping the full_attention_interval it derived them FROM. The
    real checkpoint's config.json carries the interval and no list at all, so
    write it that way - otherwise the stride is never exercised and an
    off-by-one in reading it goes unnoticed.
    """
    written = json.loads(cfg.to_json_string())
    written.pop("layer_types", None)
    written["full_attention_interval"] = FULL_ATTENTION_INTERVAL

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(written, f, indent=2)

    if os.path.exists(args.tokenizer):
        shutil.copyfile(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    else:
        print("  no tokenizer at " + args.tokenizer + " - copy one in by hand")

    ids = [int(t) for t in args.tokens.split(",")]

    rec = Recorder(args.dump)
    handles = attach(model, rec)

    with torch.no_grad():
        out = model(torch.tensor([ids]), use_cache=False)

    for h in handles:
        h.remove()

    logits = out.logits[0, -1].to(torch.float32).numpy()
    rec.add("logits", torch.tensor(logits))
    rec.finish()

    print("wrote " + args.out)
    print("  layers      " + str(cfg.layer_types))
    print("  experts     " + str(cfg.num_experts) + ", " + str(cfg.num_experts_per_tok) + " per token")
    print("  ids         " + str(ids))
    print("  argmax      " + str(int(logits.argmax())))
    print("  logits[:6]  " + np.array2string(logits[:6], precision=5))
    if args.dump:
        print("  dumped " + str(len(rec.entries)) + " tensors to " + args.dump)


if __name__ == "__main__":
    main()
