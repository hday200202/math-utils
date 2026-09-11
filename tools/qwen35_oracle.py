"""
Reference for the qwen3_5 architecture (Qwen3.8), via transformers itself.

Qwen3.8-27B is not a dense transformer. Three of every four layers are
GATED DELTANET linear attention - a recurrent mechanism with a running state
matrix, a causal convolution, and per-head decay - and the remaining
full-attention layers carry an output gate and only rotate a quarter of each
head's dimensions.

Guessing at that and writing a numpy "oracle" from the same guess would prove
nothing: two self-consistent wrong implementations agree perfectly. So the
reference here is transformers' own module, and the model is a TINY randomly
initialised one rather than the real 51.7GB checkpoint - the algorithm is
identical, it loads instantly, and every intermediate can be dumped.

    py tools/qwen35_oracle.py --out models/tiny-qwen35 --dump out/qwen35-oracle

Writes a complete little model (config.json + model.safetensors + tokenizer)
that the C++ loader can open exactly as it would the real thing.
"""

import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def tiny_config():
    """
    Small enough to run instantly, but every axis that matters is still
    exercised: both layer types, grouped key/value heads on both the linear
    and the full attention, a head_dim that is not hidden/heads, and a
    partial rotary factor.
    """
    return Qwen3_5TextConfig(
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
        layer_types=["linear_attention", "linear_attention",
                     "full_attention", "linear_attention"],
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        attn_output_gate=True,
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


def textModel(model):
    """
    A text-only config gives a Qwen3_5TextModel directly; the multimodal one
    nests it under .language_model. The real 27B checkpoint is the latter.
    """
    inner = model.model
    return getattr(inner, "language_model", inner)


def attach(model, rec):
    """
    One hook per interesting module. The names match what the C++ trace will
    emit, so the two can be compared entry by entry.
    """
    handles = []
    lm = textModel(model)

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

        # the pieces inside the linear mixer, which is the part being ported
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            handles.append(la.in_proj_qkv.register_forward_hook(hook(tag + "qkv")))
            handles.append(la.in_proj_z.register_forward_hook(hook(tag + "z")))
            handles.append(la.in_proj_a.register_forward_hook(hook(tag + "a")))
            handles.append(la.in_proj_b.register_forward_hook(hook(tag + "b")))
            handles.append(la.conv1d.register_forward_hook(hook(tag + "conv")))
            handles.append(la.norm.register_forward_hook(hook(tag + "gatednorm")))
        else:
            sa = layer.self_attn
            handles.append(sa.q_proj.register_forward_hook(hook(tag + "qproj")))
            handles.append(sa.k_proj.register_forward_hook(hook(tag + "kproj")))
            handles.append(sa.v_proj.register_forward_hook(hook(tag + "vproj")))

    handles.append(lm.norm.register_forward_hook(hook("norm")))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/tiny-qwen35")
    ap.add_argument("--dump", default="out/qwen35-oracle")
    ap.add_argument("--tokens", default="5,17,42,7,300,12")
    args = ap.parse_args()

    torch.manual_seed(1234)

    cfg = tiny_config()
    model = Qwen3_5ForCausalLM(cfg).eval()

    # random init leaves A_log and dt_bias at their constructor values, which
    # are uniform; give them something less degenerate so the decay actually
    # varies across heads
    with torch.no_grad():
        for layer in textModel(model).layers:
            la = getattr(layer, "linear_attn", None)
            if la is None:
                continue
            la.A_log.copy_(torch.log(torch.empty_like(la.A_log).uniform_(0.5, 8.0)))
            la.dt_bias.copy_(torch.empty_like(la.dt_bias).uniform_(-1.0, 1.0))

    os.makedirs(args.out, exist_ok=True)

    state = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(args.out, "model.safetensors"))

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(json.loads(cfg.to_json_string()), f, indent=2)

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
    print("  hidden      " + str(cfg.hidden_size))
    print("  ids         " + str(ids))
    print("  argmax      " + str(int(logits.argmax())))
    print("  logits[:6]  " + np.array2string(logits[:6], precision=5))
    if args.dump:
        print("  dumped " + str(len(rec.entries)) + " tensors to " + args.dump)


if __name__ == "__main__":
    main()
