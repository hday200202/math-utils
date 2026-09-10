"""
Qwen3 (dense) reference implementation in numpy.

Same job as gpt2_oracle.py: run the forward pass and dump every intermediate,
so the C++ can be diffed stage by stage instead of judged on whether its
output reads well.

    py tools/qwen3_oracle.py --model models/qwen3-0.6b --ids 3838,374,279,6722
    py tools/qwen3_oracle.py --ids ... --dump out/qwen-oracle

Token ids are passed in rather than tokenised here, so a mismatch in the
tokenizer cannot be confused with a mismatch in the model.
"""

import argparse
import glob
import json
import os
import struct

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        blob = np.memmap(path, dtype=np.uint8, mode="r")

    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        raw = blob[base + begin: base + end]

        if meta["dtype"] == "BF16":
            # bfloat16 is the top 16 bits of a float32, so widening is a shift
            u16 = raw.view(np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif meta["dtype"] == "F32":
            arr = raw.view(np.float32)
        elif meta["dtype"] == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        else:
            raise ValueError("unhandled dtype " + meta["dtype"])

        out[name] = arr.reshape(meta["shape"]).astype(np.float32, copy=True)

    return out


def load_model(directory):
    weights = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        weights.update(load_safetensors(path))
    return weights


class Dumper:
    def __init__(self, directory):
        self.dir = directory
        self.entries = []
        if directory:
            os.makedirs(directory, exist_ok=True)

    def __call__(self, name, arr):
        if not self.dir:
            return arr
        a = np.ascontiguousarray(arr, dtype=np.float32)
        a.tofile(os.path.join(self.dir, name + ".bin"))
        self.entries.append({"name": name, "shape": list(a.shape)})
        return arr

    def finish(self):
        if not self.dir:
            return
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(self.entries, f, indent=2)


# ==========================================================================
#   Pieces
# ==========================================================================

def rmsnorm(x, weight, eps):
    """No mean subtraction and no bias, unlike LayerNorm: only the scale."""
    scale = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x / scale) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(seq_begin, seq_len, head_dim, theta):
    """
    HuggingFace 'rotate half' layout: the cos/sin tables are the half-width
    frequencies concatenated with themselves, so element j and element
    j + head_dim/2 share an angle and form one rotating pair.
    """
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seq_begin, seq_begin + seq_len, dtype=np.float64)
    freqs = np.outer(pos, inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    # x is [seq, heads, head_dim]; the tables broadcast over the head axis
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def softmax_last(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


# ==========================================================================
#   Forward
# ==========================================================================

def forward(ids, w, cfg, dump):
    n_layer = cfg["num_hidden_layers"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hidden = cfg["hidden_size"]
    head_dim = cfg.get("head_dim", hidden // n_head)
    eps = cfg["rms_norm_eps"]
    theta = float(cfg["rope_theta"])
    group = n_head // n_kv

    seq = len(ids)
    x = w["model.embed_tokens.weight"][ids]
    dump("embed", x)

    cos, sin = rope_tables(0, seq, head_dim, theta)
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)

    for i in range(n_layer):
        p = "model.layers." + str(i) + "."

        h = rmsnorm(x, w[p + "input_layernorm.weight"], eps)
        dump("l" + str(i) + ".ln1", h)

        # HuggingFace Linear weights are [out, in], so this is x @ W.T
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(seq, n_head, head_dim)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(seq, n_kv, head_dim)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(seq, n_kv, head_dim)

        # QK-norm, per head, BEFORE the rotation. Qwen3 specific
        q = rmsnorm(q, w[p + "self_attn.q_norm.weight"], eps)
        k = rmsnorm(k, w[p + "self_attn.k_norm.weight"], eps)
        dump("l" + str(i) + ".qnorm", q)
        dump("l" + str(i) + ".knorm", k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        dump("l" + str(i) + ".qrope", q)
        dump("l" + str(i) + ".krope", k)

        # GQA: query head h reads key/value head h // group
        kf = np.repeat(k, group, axis=1)
        vf = np.repeat(v, group, axis=1)

        qh = q.transpose(1, 0, 2)      # [head, seq, dim]
        kh = kf.transpose(1, 0, 2)
        vh = vf.transpose(1, 0, 2)

        scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(head_dim)
        scores = np.where(mask, np.float32(-1e30), scores)
        weights = softmax_last(scores)
        dump("l" + str(i) + ".attnw", weights)

        out = (weights @ vh).transpose(1, 0, 2).reshape(seq, n_head * head_dim)
        out = out @ w[p + "self_attn.o_proj.weight"].T
        dump("l" + str(i) + ".attnout", out)

        x = x + out
        dump("l" + str(i) + ".res1", x)

        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], eps)
        dump("l" + str(i) + ".ln2", h)

        # SwiGLU: three matrices, the gate squashed and multiplying the other
        gate = h @ w[p + "mlp.gate_proj.weight"].T
        up = h @ w[p + "mlp.up_proj.weight"].T
        ff = (silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
        dump("l" + str(i) + ".mlpout", ff)

        x = x + ff
        dump("l" + str(i) + ".res2", x)

    x = rmsnorm(x, w["model.norm.weight"], eps)
    dump("norm", x)

    head = w.get("lm_head.weight", w["model.embed_tokens.weight"])
    logits = x[-1:] @ head.T
    dump("logits", logits)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b")
    ap.add_argument("--ids", default="3838,374,279,6722,315,9625,30")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    if "text_config" in cfg:
        cfg = cfg["text_config"]

    ids = [int(t) for t in args.ids.split(",") if t.strip()]
    w = load_model(args.model)

    dump = Dumper(args.dump)
    logits = forward(ids, w, cfg, dump)
    dump.finish()

    last = logits[-1]
    order = np.argsort(-last)[: args.topk]
    print("ids   : " + str(ids))
    print("argmax: " + str(int(last.argmax())))
    print("top " + str(args.topk) + ":")
    for r, t in enumerate(order):
        print("  %d. id=%-8d logit=%9.4f" % (r + 1, t, last[t]))

    if args.dump:
        print("dumped " + str(len(dump.entries)) + " tensors to " + args.dump)


if __name__ == "__main__":
    main()
