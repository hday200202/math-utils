"""Generate tiny random checkpoints and independent NumPy logits, not language quality.

Requires NumPy only. No downloaded weights, tokenizers or frameworks.
Usage: python3 tools/make_gpu_fixture.py /tmp/math-utils-gpu-fixture
"""
import argparse
import json
import pathlib
import struct

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--model-type", choices=["qwen3", "qwen2", "llama", "mixtral", "qwen2_moe"], default="qwen3")
    parser.add_argument("--rope", choices=["default", "linear", "llama3"], default="default")
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    config = dict(model_type="qwen3", hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, vocab_size=4096, max_position_embeddings=2048,
                  rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
                  bos_token_id=255, eos_token_id=254, torch_dtype="bfloat16")
    config["model_type"] = args.model_type
    if args.rope != "default":
        config["rope_scaling"] = dict(rope_type=args.rope, factor=8., low_freq_factor=1.,
                                      high_freq_factor=4., original_max_position_embeddings=64)
    if args.window:
        config.update(sliding_window=args.window, use_sliding_window=True, max_window_layers=0)
    if args.model_type == "mixtral":
        config.update(num_local_experts=4, num_experts_per_tok=2)
    if args.model_type == "qwen2_moe":
        config.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
                      shared_expert_intermediate_size=128, decoder_sparse_step=1,
                      mlp_only_layers=[0])
    (args.directory / "config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(1234)
    tensors = {}

    def weight(name, shape, norm=False):
        values = np.ones(shape, dtype=np.float32) if norm else rng.normal(0, 0.02, shape).astype(np.float32)
        tensors[name] = (values.view(np.uint32) >> 16).astype("<u2")

    weight("model.embed_tokens.weight", (4096, 64))
    weight("lm_head.weight", (4096, 64))
    weight("model.norm.weight", (64,), True)
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        weight(prefix + "input_layernorm.weight", (64,), True)
        weight(prefix + "post_attention_layernorm.weight", (64,), True)
        for name, rows in (("q", 64), ("k", 32), ("v", 32), ("o", 64)):
            weight(prefix + f"self_attn.{name}_proj.weight", (rows, 64))
        if args.model_type == "qwen3":
            for name in ("q", "k"):
                weight(prefix + f"self_attn.{name}_norm.weight", (16,), True)
        if args.model_type in ("qwen2", "qwen2_moe"):
            for name, rows in (("q", 64), ("k", 32), ("v", 32)):
                weight(prefix + f"self_attn.{name}_proj.bias", (rows,))
        def mlp(base, names=("gate_proj", "up_proj", "down_proj")):
            for name, shape in zip(names, ((128, 64), (128, 64), (64, 128))):
                weight(base + name + ".weight", shape)
        if args.model_type == "mixtral":
            weight(prefix + "block_sparse_moe.gate.weight", (4, 64))
            for expert in range(4):
                mlp(prefix + f"block_sparse_moe.experts.{expert}.", ("w1", "w3", "w2"))
        elif args.model_type == "qwen2_moe" and layer == 1:
            weight(prefix + "mlp.gate.weight", (4, 64))
            weight(prefix + "mlp.shared_expert_gate.weight", (1, 64))
            mlp(prefix + "mlp.shared_expert.")
            for expert in range(4):
                mlp(prefix + f"mlp.experts.{expert}.")
        else:
            mlp(prefix + "mlp.")
    header, offset = {}, 0
    for name, tensor in tensors.items():
        header[name] = dict(dtype="BF16", shape=list(tensor.shape), data_offsets=[offset, offset + tensor.nbytes])
        offset += tensor.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with (args.directory / "model.safetensors").open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for tensor in tensors.values():
            out.write(tensor.tobytes())
    # An independent full-sequence FP64 oracle, using the actual BF16 bytes.
    # Compare suffix logits from incremental CPU/GPU decoding to this causal pass.
    state = {name: (value.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
             for name, value in tensors.items()}
    ids = np.arange(1, 22)
    x = state["model.embed_tokens.weight"][ids].copy()
    def norm(x, name):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * state[name]
    def linear(x, name):
        result = np.einsum("...i,ji->...j", x, state[name + ".weight"], optimize=False)
        if name + ".bias" in state:
            result += state[name + ".bias"]
        return result
    def ffn(x, base, names=("gate_proj", "up_proj", "down_proj")):
        gate = linear(x, base + names[0])
        return linear(gate / (1 + np.exp(-gate)) * linear(x, base + names[1]), base + names[2])
    frequencies = 10000. ** (-np.arange(0, 16, 2) / 16.)
    if args.rope == "linear":
        frequencies /= 8.
    if args.rope == "llama3":
        wavelengths = 2 * np.pi / frequencies
        blend = np.clip((64 / wavelengths - 1) / 3, 0, 1)
        frequencies *= (1 - blend) / 8 + blend
    angles = np.arange(len(ids))[:, None, None] * frequencies[None, None, :]
    def rotate(x):
        a, b = x[..., :8], x[..., 8:]
        return np.concatenate((a * np.cos(angles) - b * np.sin(angles),
                               b * np.cos(angles) + a * np.sin(angles)), axis=-1)
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        h = norm(x, prefix + "input_layernorm.weight")
        q = linear(h, prefix + "self_attn.q_proj").reshape(-1, 4, 16)
        k = linear(h, prefix + "self_attn.k_proj").reshape(-1, 2, 16)
        v = linear(h, prefix + "self_attn.v_proj").reshape(-1, 2, 16)
        if args.model_type == "qwen3":
            q = norm(q, prefix + "self_attn.q_norm.weight")
            k = norm(k, prefix + "self_attn.k_norm.weight")
        q, k = rotate(q), rotate(k)
        attended = np.empty_like(q)
        for token in range(len(ids)):
            first = max(0, token + 1 - args.window) if args.window else 0
            for head in range(4):
                scores = k[first:token + 1, head // 2] @ q[token, head] / 4.
                probs = np.exp(scores - scores.max()); probs /= probs.sum()
                attended[token, head] = probs @ v[first:token + 1, head // 2]
        x += linear(attended.reshape(-1, 64), prefix + "self_attn.o_proj")
        h = norm(x, prefix + "post_attention_layernorm.weight")
        if args.model_type == "mixtral" or (args.model_type == "qwen2_moe" and layer == 1):
            base = prefix + ("block_sparse_moe." if args.model_type == "mixtral" else "mlp.")
            scores = linear(h, base + "gate")
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            chosen = np.argsort(-probs, axis=-1)[:, :2]
            selected = np.take_along_axis(probs, chosen, axis=-1)
            if args.model_type == "mixtral":
                selected /= selected.sum(axis=-1, keepdims=True)
            result = np.zeros_like(h)
            for token in range(len(ids)):
                for expert, share in zip(chosen[token], selected[token]):
                    names = ("w1", "w3", "w2") if args.model_type == "mixtral" else ("gate_proj", "up_proj", "down_proj")
                    result[token] += share * ffn(h[token], base + f"experts.{expert}.", names)
            if args.model_type == "qwen2_moe":
                result += ffn(h, base + "shared_expert.") / (1 + np.exp(-linear(h, base + "shared_expert_gate")))
            x += result
        else:
            x += ffn(h, prefix + "mlp.")
    logits = linear(norm(x, "model.norm.weight"), "lm_head")
    (args.directory / "oracle-logits.bin").write_bytes(logits.astype("<f4").tobytes())
    (args.directory / "oracle.json").write_text(json.dumps(dict(ids=ids.tolist(), vocab=4096)))
    # GPT-2's reversible byte alphabet, with one minimal merge. The unused
    # vocabulary rows remain valid targets for numerical/logit comparisons.
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {b: chr(b) for b in direct}
    for index, byte in enumerate(b for b in range(256) if b not in alphabet):
        alphabet[byte] = chr(256 + index)
    vocab = {alphabet[b]: b for b in range(256)}
    vocab["ab"] = 256
    tokenizer = dict(model=dict(type="BPE", vocab=vocab, merges=[["a", "b"]]))
    (args.directory / "tokenizer.json").write_text(json.dumps(tokenizer))
    print(f"Wrote {offset:,} bytes of synthetic BF16 weights to {args.directory}")


if __name__ == "__main__":
    main()
