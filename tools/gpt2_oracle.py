"""
GPT-2 reference implementation in numpy.

This exists to be the ORACLE for the C++ inference engine: it runs the same
forward pass and dumps every intermediate activation to disk, so the C++ can
be diffed stage by stage instead of debugged by squinting at generated text.

A transformer that is subtly wrong does not crash, it produces fluent garbage.
Layer-by-layer diffing against a known-good reference is the only practical
way to find that class of bug.

Usage:
    py tools/gpt2_oracle.py --prompt "The capital of France is"
    py tools/gpt2_oracle.py --prompt "..." --dump out/oracle
    py tools/gpt2_oracle.py --prompt "..." --generate 20

The --dump directory gets:
    manifest.json   list of {name, shape} in write order
    <name>.bin      raw little-endian float32, C-contiguous

Nothing here is optimised. It is meant to be read.
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np


# ==========================================================================
#   safetensors
#
#   The format is deliberately trivial:
#       [8 bytes]  little-endian uint64: length of the JSON header
#       [N bytes]  JSON: name -> {dtype, shape, data_offsets: [begin, end]}
#       [rest]     raw tensor bytes, offsets relative to the end of the header
# ==========================================================================

DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": None,     # numpy has no bfloat16; handled by widening below
    "I64": np.int64,
    "I32": np.int32,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def load_safetensors(path):
    """name -> np.ndarray. BF16 is widened to float32 on the way in."""
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
            # bfloat16 is the top 16 bits of a float32, so widening is just a
            # left shift into the high half of a 32-bit word
            u16 = raw.view(np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        else:
            dt = DTYPES[meta["dtype"]]
            if dt is None:
                raise ValueError("unhandled dtype " + meta["dtype"])
            arr = raw.view(dt)

        out[name] = arr.reshape(meta["shape"]).astype(np.float32, copy=True)

    return out


# ==========================================================================
#   Byte-level BPE tokenizer
#
#   GPT-2 does not tokenise characters. It maps the raw UTF-8 BYTES of the
#   text into a private printable-unicode alphabet, then repeatedly merges the
#   highest-priority adjacent pair according to merges.txt. The byte mapping
#   exists so that every possible byte has a printable symbol and the merge
#   table never has to deal with control characters or invalid UTF-8.
# ==========================================================================

def bytes_to_unicode():
    """The GPT-2 byte <-> printable-codepoint bijection."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# GPT-2's pre-tokenizer splits on this before BPE runs. The reference uses the
# `regex` module for \p{L}/\p{N}; this ASCII-equivalent covers English prompts,
# which is all the oracle needs.
PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[^\sA-Za-z0-9]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class Tokenizer:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[1:]      # first line is a version banner
        merges = [tuple(l.split()) for l in lines if len(l.split()) == 2]
        self.ranks = {m: i for i, m in enumerate(merges)}

        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = list(token)
        while len(word) > 1:
            # find the adjacent pair with the best (lowest) merge rank
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break

            a, b = best
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged

        self.cache[token] = word
        return word

    def encode(self, text):
        ids = []
        for chunk in PAT.findall(text):
            mapped = "".join(self.b2u[b] for b in chunk.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self._bpe(mapped))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(self.u2b[c] for c in text).decode("utf-8", errors="replace")


# ==========================================================================
#   Activation dumping
# ==========================================================================

class Dumper:
    """Writes float32 blobs plus a manifest, or does nothing if disabled."""

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
#   The model
# ==========================================================================

def layernorm(x, weight, bias, eps=1e-5):
    """
    Per-token normalisation. Note the variance is the BIASED one (divide by n,
    not n-1) -- getting this wrong shifts every activation slightly and is
    invisible until the logits are subtly off.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def gelu_new(x):
    """
    The tanh approximation of GELU, which is what GPT-2 actually shipped with
    (config says "gelu_new"). The exact erf-based GELU differs by ~1e-3, which
    is enough to fail a tight layer-by-layer diff.
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    """Max-subtracted: exp() of a large logit overflows to inf without it."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(x, w, prefix, n_head, dump):
    """
    x is [seq, d_model]. Returns [seq, d_model].

    GPT-2 stores these as Conv1D weights, meaning [in, out] rather than the
    [out, in] that nn.Linear uses. So it is x @ W, NOT x @ W.T -- this single
    transpose is the most common porting bug in the whole model.
    """
    seq, d_model = x.shape
    d_head = d_model // n_head

    qkv = x @ w[prefix + ".attn.c_attn.weight"] + w[prefix + ".attn.c_attn.bias"]
    dump(prefix + ".qkv", qkv)

    q, k, v = np.split(qkv, 3, axis=-1)                       # each [seq, d_model]

    # [seq, d_model] -> [n_head, seq, d_head]
    q = q.reshape(seq, n_head, d_head).transpose(1, 0, 2)
    k = k.reshape(seq, n_head, d_head).transpose(1, 0, 2)
    v = v.reshape(seq, n_head, d_head).transpose(1, 0, 2)

    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)     # [n_head, seq, seq]

    # causal mask: token i may not attend to any j > i
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    scores = np.where(mask, np.float32(-1e10), scores)
    dump(prefix + ".scores", scores)

    weights = softmax(scores, axis=-1)
    dump(prefix + ".attnw", weights)

    out = weights @ v                                         # [n_head, seq, d_head]
    out = out.transpose(1, 0, 2).reshape(seq, d_model)
    out = out @ w[prefix + ".attn.c_proj.weight"] + w[prefix + ".attn.c_proj.bias"]
    dump(prefix + ".attnout", out)
    return out


def mlp(x, w, prefix, dump):
    h = x @ w[prefix + ".mlp.c_fc.weight"] + w[prefix + ".mlp.c_fc.bias"]
    dump(prefix + ".fc", h)
    h = gelu_new(h)
    dump(prefix + ".gelu", h)
    h = h @ w[prefix + ".mlp.c_proj.weight"] + w[prefix + ".mlp.c_proj.bias"]
    dump(prefix + ".mlpout", h)
    return h


def forward(ids, w, cfg, dump):
    """
    ids is a python list of token ids. Returns logits [seq, vocab].

    The residual stream is the whole architecture: x is a fixed-width bus that
    every sub-block reads from and ADDS back into. Nothing ever replaces it.
    """
    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    eps = cfg["layer_norm_epsilon"]

    tok = w["wte.weight"][ids]                       # [seq, d_model]
    pos = w["wpe.weight"][: len(ids)]                # [seq, d_model]
    x = tok + pos
    dump("embed", x)

    for i in range(n_layer):
        p = "h." + str(i)

        h = layernorm(x, w[p + ".ln_1.weight"], w[p + ".ln_1.bias"], eps)
        dump(p + ".ln1", h)
        x = x + attention(h, w, p, n_head, dump)
        dump(p + ".res1", x)

        h = layernorm(x, w[p + ".ln_2.weight"], w[p + ".ln_2.bias"], eps)
        dump(p + ".ln2", h)
        x = x + mlp(h, w, p, dump)
        dump(p + ".res2", x)

    x = layernorm(x, w["ln_f.weight"], w["ln_f.bias"], eps)
    dump("lnf", x)

    # lm_head is tied to the token embedding matrix, so there is no separate
    # tensor for it -- the same [vocab, d_model] table is reused transposed
    logits = x @ w["wte.weight"].T
    dump("logits", logits)
    return logits


# ==========================================================================
#   CLI
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--dump", default=None, help="directory for activation blobs")
    ap.add_argument("--generate", type=int, default=0, help="greedy tokens to emit")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tok = Tokenizer(os.path.join(args.model, "vocab.json"),
                    os.path.join(args.model, "merges.txt"))
    w = load_safetensors(os.path.join(args.model, "model.safetensors"))

    ids = tok.encode(args.prompt)
    print("prompt : " + repr(args.prompt))
    print("ids    : " + str(ids))
    print("roundtrip: " + repr(tok.decode(ids)))
    if tok.decode(ids) != args.prompt:
        print("WARNING: tokenizer did not round-trip", file=sys.stderr)

    dump = Dumper(args.dump)
    logits = forward(ids, w, cfg, dump)
    dump.finish()

    last = logits[-1]
    order = np.argsort(-last)[: args.topk]
    print("")
    print("top " + str(args.topk) + " next tokens:")
    probs = softmax(last)
    for r, t in enumerate(order):
        print("  %d. %-14s id=%-6d logit=%8.4f  p=%.4f"
              % (r + 1, repr(tok.decode([int(t)])), t, last[t], probs[t]))

    if args.generate:
        cur = list(ids)
        for _ in range(args.generate):
            lg = forward(cur, w, cfg, Dumper(None))
            cur.append(int(np.argmax(lg[-1])))
        print("")
        print("greedy: " + repr(tok.decode(cur)))

    if args.dump:
        print("")
        print("dumped " + str(len(dump.entries)) + " tensors to " + args.dump)


if __name__ == "__main__":
    main()
