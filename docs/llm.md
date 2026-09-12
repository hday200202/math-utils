# llm

Run an open-weight model locally. One binary, CPU or GPU, no framework.

Point it at a folder holding `config.json`, the `.safetensors`, and a
tokenizer. The architecture is read out of `config.json` rather than
compiled in, so one build runs every supported model.

```
./bin/release/llm models/qwen3-8b --q4 --gpu
```

---

## Building

```
make release-llm
```

There is one `llm` binary, not a CPU one and a GPU one. `include/ai/gpu/Device.hpp`
is plain C++ with no HIP types in it, so every translation unit compiles the
same way everywhere and only the *backend* changes at link time:

| | |
|---|---|
| ROCm present | `src/gpu/Kernels.hip` through `hipcc`, linked in |
| ROCm absent | `src/gpu/Stub.cpp`, where `available()` returns false |

The stub is not a stripped-down build. `--gpu` still parses, still asks for a
device, and is told no — the same path taken when a card is present but out of
memory. One behaviour to reason about instead of two.

```
make release-llm                    use the GPU if this machine has one
make release-llm GPU=0              no-device build even where ROCm exists
make release-llm GPU_ARCH=gfx1030   a card that is not RDNA3
make release-llm HIPCC=/path/hipcc  ROCm somewhere other than /opt/rocm
```

`GPU_ARCH` must match the card. `rocminfo | grep gfx` will tell you.

Nothing outside `src/gpu/` knows the GPU exists, so on a machine with no ROCm
at all the build is exactly what it was before any of this: `g++ -std=c++26`,
no flags, no dependencies.

## Getting a model

```
py tools/fetch_model.py Qwen/Qwen3-8B
py tools/fetch_model.py Qwen/Qwen3-32B --check      # inspect, download nothing
```

Downloads only what is needed: `config.json`, a tokenizer, `tokenizer_config.json`
for its chat template, and the shards named by the index. `--check` prints the
size and the estimated speed before committing to the download.

Supported architectures: `gpt2`, `llama`, `mistral`, `qwen2`, `qwen3`,
`qwen3_5`, `qwen3_next`, `phi3`. Anything else is refused at load with the
reason, rather than running and producing nonsense.

---

## Fitting a model

This is the part that decides whether anything else matters. Three levers,
in the order worth reaching for them.

### 1. Narrow the weights

`--q8` and `--q4` quantise the weight matrices as they load. The checkpoint on
disk is untouched; nothing is written back.

| | bytes/weight | vs bf16 |
|---|---|---|
| bfloat16 (default) | 2.0 | — |
| `--q8` | 1.125 | 1.8x smaller |
| `--q4` | 0.625 | 3.2x smaller |

What it costs, measured on Qwen3-0.6B against its own bfloat16 output —
a *small* model, and therefore the harsh case, since there is less redundancy
to absorb the noise:

| | top-1 agreement | where the model was confident | perplexity |
|---|---|---|---|
| `--q8` | 98.9% | 100% | **−0.39%** |
| `--q4` | 77.4% | 96.6% | +4.09% |

`--q8` is free in every sense that matters — the perplexity moved *down*,
which is quantisation noise occasionally helping. `--q4` is a real trade and
you can see it; use it when it is the difference between running and not.

The raw top-1 number is misleading on purpose: most flips happen where the
model was torn between two near-tied continuations and nothing meaningful
changed. The confident-only column is the one to read.

### 2. Put it on the GPU

`--gpu` uploads whatever the weights already are — bfloat16, int8 or int4 —
and runs the matmuls on the card. It combines with `--q8` / `--q4`.

Measured on Qwen3-14B at `--q4`, this machine (7900 XT, 9800X3D):

```
CPU only                    5.0 tok/s
--gpu                      33.5 tok/s
```

Host memory drops too, because once the card has a weight the copy in RAM is
released:

```
8B  --q4 --gpu     1.7 GB resident
14B --q4 --gpu     1.9 GB resident
27B --q4 --gpu    14.9 GB on the card
```

Peak during *loading* is higher than steady state — the model is materialised
before it is uploaded — so a `--q4` 14B peaks near 8.8 GB and settles at 1.9.

### 3. Spill the rest to the CPU

`--vram N` caps the card at N gigabytes. Whole layers that do not fit stay on
the CPU. There is no mode switch: each weight carries its own dispatch, so a
model split across both devices is not a special case anywhere.

Qwen3-14B at `--q4`, which wants 8.1 GB:

```
--vram 2     8/40 layers on GPU     6.4 tok/s
--vram 4    18/40 layers            8.5 tok/s
--vram 6    28/40 layers           12.7 tok/s
--vram 9    all 40 layers          33.5 tok/s
```

Note the curve is **convex, not linear**. A token reads every weight exactly
once, so the time is roughly `cpuBytes/60GB/s + gpuBytes/400GB/s` and the CPU
term dominates until it is nearly gone. Half on each device lands much closer
to CPU speed than to the midpoint.

So this is a fallback, not a dial worth tuning. If you are more than a couple
of layers short, **a smaller quant beats partial offload** — it puts the whole
model on the fast path instead of leaving a slow tail.

---

## Options

### Where it runs

| | |
|---|---|
| `--gpu` | run the matmuls on the GPU |
| `--vram N` | cap the GPU at N gigabytes; the rest stays on the CPU |

### How big

| | |
|---|---|
| `--q8` / `--q4` | quantise the weights as they load |
| `--ctx N` | context window. Clamped to the model's own maximum |
| `--force` | load anyway when the size check says it will not fit |

### How it answers

| | |
|---|---|
| `--effort low\|medium\|xhigh` | let it think before answering |
| `--system TEXT` | replace the preamble |
| `--no-system` | drop the preamble entirely |
| `--raw` | no chat template at all — raw continuation |

**`--effort`** is off by default: the scratchpad is closed and the answer comes
straight out. Only Qwen3.5 and later have the dial, and it is implemented as a
sentence at the top of the system turn, copied verbatim from the model's own
chat template. `medium` deliberately carries no instruction — it is the
model's own baseline, not a middle setting. The reasoning prints along with
the answer.

It is worth more than it looks. On Qwen3.8-27B, *"a train leaves at 3:15 PM
and takes 95 minutes"*: with no thinking it opened `95 minutes = 1 hour + 45
minutes` and was heading for the wrong answer; at `--effort low` it worked out
1h35m and arrived at 4:50 PM. The instruction costs 24 tokens, once.

**The preamble** exists because an instruct model trained on agentic coding
traces does not answer "write me a renderer" by writing the renderer — it
answers by listing the directory and opening `package.json`, calling tools
this program does not have. The default system message tells it not to.

Describing the situation does not work. *"There are no tools, no file system,
nothing to inspect"* was acknowledged and ignored by Qwen3.5-27B, which went
on calling tools anyway. Naming the literal syntax it must not emit worked
first try. If you replace it with `--system`, keep that property.

`--no-system` is right for a model that was never trained to call tools.

### Sampling

| | |
|---|---|
| `--tokens N` | maximum to generate (default 256) |
| `--temp F` | 0 is greedy and deterministic (default 0.7) |
| `--topk N` | sample from the top N (default 20) |
| `--seed N` | reproduce a run exactly |
| `--prompt TEXT` | answer once and exit, instead of the chat loop |

### Testing the engine itself

| | |
|---|---|
| `--random` | run the topology with invented weights, reading no payload |
| `--random EPS` | keep the trained weights, offset each by ±EPS |

`--seed` reproduces either exactly. Useful for separating "the engine is
wrong" from "the weights are wrong" — a model that still forms sentences at
`--random 0.02` is doing something structurally right.

---

## Interactive commands

Running without `--prompt` drops into a chat loop. The KV cache carries across
turns, so a follow-up only pays for its own tokens.

| | |
|---|---|
| `/reset` | forget the conversation |
| `/raw` | stop wrapping input in the chat template |
| `/chat` | go back to the chat template |
| `/stats` | timings for the last turn |
| `/quit` | exit |

---

## Examples

```bash
# smallest useful thing
./bin/release/llm models/qwen3-0.6b

# an 8B on the card, one question, deterministic
./bin/release/llm models/qwen3-8b --q4 --gpu --temp 0 \
    --prompt "Explain a red-black tree in one paragraph."

# a 14B that only just fits, thinking enabled
./bin/release/llm models/qwen3-14b --q4 --gpu --effort low

# a 27B with the card capped, leaving room for a desktop
./bin/release/llm models/qwen3.8-27b --q4 --gpu --vram 16

# raw continuation, no chat wrapper, no preamble
./bin/release/llm models/qwen3-8b --raw --no-system \
    --prompt "The three laws of robotics are"
```

---

## When it goes wrong

**Killed with no message.** The kernel OOM-killer, almost certainly. Check
`journalctl -k | grep -i "killed process"`. Large contexts are the usual
cause: the KV cache is allocated in full at load, and on a 64-layer model
`--ctx 100000` is tens of gigabytes of *system* RAM before a single token is
generated. Lower `--ctx` first.

**`--gpu` seems to do nothing.** Check the `device` line printed at startup.
If it says `CPU`, either the flag was omitted or the build has no backend —
`nm -C bin/release/llm | grep -c hip` returns 0 on a stub build.

**Runs, but the output is nonsense or loops.** Suspect the checkpoint before
the engine. A shard can be the right size, pass its header check, and still
hold wrong bytes — an interrupted or doubled download does exactly that.
Verify against the published hashes:

```bash
cd models/<model>
python3 -c "
import json,urllib.request
for e in json.load(urllib.request.urlopen(
        'https://huggingface.co/api/models/<org>/<repo>/tree/main')):
    if e['path'].endswith('.safetensors') and e.get('lfs'):
        print(e['lfs']['oid'], e['path'])" > expected.txt
sha256sum *.safetensors
```

Delete any shard that does not match and re-fetch. This is worth checking
*early* — a corrupt shard produces a healthy-looking residual stream with
wrong content, which mimics every other kind of bug.

**It answers by calling tools.** See `--system` above. `--no-system` is not
the fix; the default preamble is.

**Refuses to load, saying it will not fit.** The size check is about what has
to stay resident, which `--q8` / `--q4` reduce. Try those before `--force`.

---

## Test suites

```
make test                       architecture and model correctness
make gpu-test                   GPU kernels, CPU/GPU agreement, quality
```

`llm-testing` builds against the stub and needs no ROCm — it tests the loader
and the model. `gpu-testing` genuinely needs a card and says so if one is
absent.

Two failures in `llm-testing` are expected on a fresh checkout: they need
fixtures that are not in the repo (`models/gpt2`, and a numpy oracle dump
generated by `tools/gpt2_oracle.py` / `tools/qwen3_oracle.py`).
