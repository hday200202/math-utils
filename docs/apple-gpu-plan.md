# Apple GPU backend

Status: implemented and tested on an M5 Pro. The file keeps its original
name so existing links remain valid. Resident decode is selected by supported
operations and shapes, not by model name.

## Build and run

```
make release-llm BACKEND=metal
./bin/release/llm models/qwen3-8b --q4 --gpu --ctx 1024
```

Requires an Apple GPU (M1 or newer), Apple Clang with the project's C++26
support, and `xcrun metal` / `xcrun metallib`. The default macOS build selects
Metal when those tools are present. `BACKEND=cpu` or `GPU=0` needs no Metal
SDK. `BACKEND=hip` preserves the AMD backend on its supported platforms.

`src/gpu/Metal.mm` implements the plain C interface in `DeviceABI.h` and owns
all Objective-C objects. `Kernels.metal` is compiled offline and embedded by
`tools/embed_metal.py`; the executable has no runtime shader-file dependency.
Running without `--gpu` uses the ARM CPU kernels in the same executable.

## Execution

The general backend implements `A[M,K] @ B[N,K]^T` for F32, BF16, Q8 and Q4.
Weights stay in their original packed representation. Q4 lanes consume both
nibble halves of each packed-byte load and share scales per 32 weights.
Decode uses vectorized SIMD-group reductions; prompts use 32x32 tiled
SIMD-group matrix multiplication, with F32 inputs and accumulation. Each K
tile is reduced separately to limit floating-point accumulation error.

Shared Metal buffers hold weights and reusable input/output scratch. The
public matmul interface remains synchronous, and a mutex protects its scratch
from concurrent callers such as routed experts. Allocation failures return
an error; invalid dimensions and GPU command failures raise a C++ exception
rather than returning uninitialized outputs. Initialization failure follows
the existing CPU fallback.

For compatible dense transformers, single-token decoding keeps the residual
stream, intermediate tensors and KV cache on the GPU. It records an indirect
compute program once per conversation and replays it in one command buffer
per token. Positions and attention grids are updated between tokens.
Resource declarations and explicit indirect-command barriers cover dependencies.
From position 256, the value reduction runs over independent 128-token
partitions and merges their local softmaxes using the local maxima and sums.
This keeps the long-history value walk parallel without an attention
approximation. The program is recorded again once at that transition.

The resident program fuses Q/K/V projections, head normalization with RoPE
and KV insertion, softmax with the value reduction, gate/up projections, and
residual additions with the next normalization. Only the embedding row and
rotary coefficients enter from the CPU; only logits return for sampling.

This path requires a fully offloaded dense RMSNorm/RoPE/SwiGLU
model with supported dimensions and no attention output gate, norm-weight
offset or activation clamp. Optional Q/K normalization and partial RoPE are
supported. Projection biases are copied into resident buffers, enabling Qwen2
as well as Qwen3. Default, linear and Llama 3 frequency scaling use the same
precomputed frequencies on CPU and GPU. Sliding attention windows are honored
per layer. Qwen3-8B exercises the path. Other architectures and partial
offloads use the per-operation backend; hybrid recurrent attention and MoE
still perform their non-matmul work on the CPU.

## Conversation lifetime and memory

A conversation owns its resident state. Copying a `LocalCache` materializes
an independent CPU KV history. A multirow follow-up prompt or a traced forward
pass restores CPU storage and downloads the GPU-generated KV entries before
using the ordinary path. While resident decode owns the history, duplicate
CPU KV storage is released; copying a conversation materializes an independent
CPU history without changing the source.
Reset and resize discard resident state. Offload-generation checks invalidate
resident weight bindings when a model is moved between devices.

CPU and GPU share physical RAM. The backend caps its advertised headroom by
both the recommended GPU working set and the host's available-memory estimate;
it does not treat unified memory as two independent pools. Those are snapshots,
not guaranteed allocations. Scratch grows only as needed and is reused.

Weight upload currently copies into Metal-owned buffers before the model's
host matrix storage is released. Loading/offload therefore has a transient
memory peak. CPU KV storage is released after the resident decoder copies the
prompt history, and recreated only for a copy, reset or CPU fallback.
Nonresident and hybrid models retain their CPU caches. Long contexts and a
27B model on a 24 GiB Mac can still cause memory pressure. Metal does not solve that capacity limit.

## Measurements

M5 Pro, 24 GiB unified memory, Qwen3-8B at Q4, 18 CPU workers, context 256,
74 prompt tokens and 48 generated tokens, temperature zero. Both executions
produced the same continuation. Single-run measurements, excluding loading:

| | ARM CPU | Metal |
|---|---:|---:|
| Prompt processing | 51.6 tokens/s | 77.1 tokens/s |
| Generation | 36.4 tokens/s | 39.5 tokens/s |

The decode gain is modest because the optimized CPU and GPU share the memory
controller. A standalone 512 MiB BF16 weight-bandwidth test reached roughly
230–270 GB/s inside the GPU kernel across runs. Whole-model rates include
attention, sampling, dispatch and synchronization and should not be inferred
from the standalone bandwidth. Larger contexts and other models need their
own measurements.

A subsequent matched comparison used an 833-token prompt, 32 generated tokens,
context 2048 and the same Q4 model, core count and greedy decoding. The previous
resident reduction generated **27.8 tokens/s**, versus **35.1 tokens/s** with
partitioned attention (about **26.5% faster**). Both produced the same text.
Prompt processing remained around 138 tokens/s in this comparison. Short
contexts remain around 39 tokens/s; these single-run measurements do not
imply a universal speedup.

CPU attention now distributes query/head pairs across the persistent worker
pool, preserving each output's summation order. An isolated M5 Pro benchmark
(128 queries after 384 cached tokens, 32 query heads, 8 KV heads, dimension 128)
measured 35.8 ms serial versus 3.2 ms parallel. This measures the attention
kernel, not whole-model token throughput.

The CLI reports dispatch count, GPU execution time and activation-copy time
for GPU runs. Copy counters measure logical CPU memory copies; they are not
PCIe transfers on Apple silicon. Sampling, embedding gathering and prompt
attention remain on the CPU; prompt matmuls run on the tiled GPU kernel.

## Validation

```
make test-unit
make release-gpu-test
python3 tools/make_gpu_fixture.py /tmp/math-utils-gpu-fixture
./bin/release/gpu-testing /tmp/math-utils-gpu-fixture
./bin/release/gpu-testing --resident-only models/qwen3-8b
```

The fixture generator also supports `--model-type qwen2`, `llama`, `mixtral`
and `qwen2_moe`, `--rope linear|llama3`, and `--window N`. It writes an
independent FP64 NumPy causal-model oracle using the BF16 checkpoint bytes.
`gpu-testing --fixture DIRECTORY` checks CPU and GPU prefill plus incremental
logits against that oracle. For example:

```
python3 tools/make_gpu_fixture.py /tmp/math-utils-qwen2-fixture \
  --model-type qwen2 --rope llama3 --window 7
./bin/release/gpu-testing --fixture /tmp/math-utils-qwen2-fixture
./bin/release/gpu-testing --resident-only /tmp/math-utils-qwen2-fixture
```

Mixtral and Qwen2 MoE pass the independent fixture checks; complete trained
checkpoints of those families have not been downloaded or language-tested here.
The Qwen3-8B resident checks cover the position-256 program transition, independent
copies, reset, offload changes and exact CPU history restoration.

The generated fixture requires NumPy only and contains small random BF16
weights plus a minimal byte tokenizer. It tests numerical behavior, not
language quality. Kernel checks cover all formats, ragged dimensions,
concurrent dispatch, buffer reuse and invalid buffer sizes. Full-model fixture
checks cover BF16/Q8/Q4 and quantization differences. The real 8B resident test
compares against the established per-operation GPU path, including independent
cache copies, follow-up prompts and reset. These tests also run with Metal
API validation; the resident fixture runs with shader validation.

Quantized GPU kernels use float activations. The CPU's small-row integer fast
path additionally narrows activations to int8. Numerical reference checks
therefore evaluate the GPU's float-activation operation explicitly, including
the final one-row output projection, rather than loosening tolerances to
conceal the difference.

## Further work

The largest remaining opportunities are resident multirow prompt attention,
resident hybrid/recurrent mixers and MoE, less transient memory during weight
upload, and bounded or compressed KV storage for large contexts. Tune further
attention and matrix kernels using model-level measurements. Additional optimized
paths must preserve traceability and the existing correctness checks.

References: [resource storage](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus),
[working-set guidance](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize),
[Metal capabilities](https://developer.apple.com/metal/capabilities/).

Model semantics were checked against the primary Transformers implementations:
[Mixtral](https://github.com/huggingface/transformers/blob/v4.52.1/src/transformers/models/mixtral/modeling_mixtral.py),
[Qwen2 MoE](https://github.com/huggingface/transformers/blob/v4.52.1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py),
and [RoPE scaling](https://github.com/huggingface/transformers/blob/v4.52.1/src/transformers/modeling_rope_utils.py).
