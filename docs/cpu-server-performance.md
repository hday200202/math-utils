# CPU server performance

GLM-5.3 Flash Next uses both recurrent Kimi Delta Attention and latent attention,
plus routed experts. The CPU stages can have different bottlenecks. RAM capacity
alone does not determine generation speed, and improvements measured on a Mac
cannot establish a throughput gain on a dual-socket Xeon.

## Build and measure on the server

Build on the machine that will run inference so `-march=native` targets its CPU:

```sh
make release-llm BACKEND=cpu
make release-llm-test BACKEND=cpu
./bin/release/llm-testing --unit
```

Keep your current model path, quantization, context, prompt, seed and token count.
Add `--profile` to report the compiled integer kernel, loading times and separate
prompt/decode stage timings. Compare `--cores all` with `--cores physical`.
The latter reads Linux socket/core topology and selects one allowed hardware
thread per physical core across the available sockets. It honors the process's
CPU affinity; missing topology falls back to all allowed CPUs. No default
worker-count change is imposed.

Add `--load-only --profile` to the same invocation to measure loading without
allocating a conversation cache or running inference. Loading reports:

- **read/place:** payload reads; the non-streaming path also includes placement
  copies into the buffers that inference owns.
- **conversion:** normal streaming conversion and quantization. This does not
  include disk-read time or the optional random/perturbed-weight path.
- **arranging:** tokenizer and layer setup, including matrices expanded for
  latent attention after payload loading.

Headers, scale-tensor preloading, allocation and other setup can account for a
remaining difference from total `loaded in` time. Stage timers are optional and
do not retain activation traces. Decode timing in the normal throughput line
also includes sampling and terminal output; stage totals cover forward work.

Collect `lscpu`, `numactl --hardware` if installed, your exact command and the
profile output before choosing a further optimization. During a slow run,
`vmstat 1` shows system paging, and `numastat -p PID` shows process placement.
These are read-only diagnostics; do not drop the server's page cache for a
benchmark on a shared machine. Warm and cold storage measurements are different
experiments and must be labeled.

## Implemented changes

### Loading

The INT4/INT8 streaming loader reads tensors in payload-offset order. Large
reads use reusable positioned-read descriptors and at least 1 MiB per reader;
small reads stay serial. The scale prepass already runs inside the worker pool
and deliberately uses the serial reader to avoid nested dispatch. Read buffers
are uninitialized and reused. Short/truncated reads fail instead of exposing
partially initialized weights, and POSIX reads retry interrupted calls.

FP8-to-INT4/INT8 conversion decodes one row per worker directly into final
quantized storage. It no longer allocates and zero-fills a complete float32
copy of the tensor. Scales and packed codes are byte-identical to the previous
float-staging path. The allocation planner includes per-worker row scratch.
Fully overwritten float tensors also avoid zero-filling; the ordinary `Tensor`
constructor retains its zero-initialization contract. Dense latent-attention
expansion runs across workers instead of a serial row loop.

On Linux, once a streamed tensor has been converted into owned storage, a
checkpoint larger than half physical RAM allows its copied file pages to be
evicted with `POSIX_FADV_DONTNEED`. This mirrors the existing whole-shard
loader and avoids retaining a second cached copy under memory pressure. Smaller
checkpoints keep normal page-cache behavior. No persistent quantized cache is
written: a later process still reads and converts the checkpoint.

### Generation

Kimi Delta Attention and Gated DeltaNet fuse state decay with prediction, and
the rank-one state update with output accumulation. This reduces four state
traversals to two without an algebraic approximation or changing each key
channel's reduction order. Tests cover scalar/per-channel decay, rectangular
states, repeated updates and GLM prompt/decode chunk continuity.

AVX2 activation and weight quantization now uses SIMD rounding and packing,
including ties away from zero and values immediately adjacent to ties. Q8 and
Q4 scales and codes preserve the scalar quantizer's rules.

AVX-512 BW integer dots process two 32-element blocks per vector and maintain
independent accumulation chains. VNNI is used when targeted by the compiler;
otherwise the BW multiply/add sequence is used. AVX2, ARM and portable fallbacks
remain. To compare the full-width path with the previous AVX2-width dot kernels
on the same server:

```sh
make release-llm BACKEND=cpu CPPFLAGS=-DAI_QUANT_AVX512=0
```

Rebuild normally without that override to restore full-width dots. Physical
core count, AVX width and memory bandwidth interact; benchmark both configurations
rather than assuming that wider vectors or SMT are universally faster.

## Validation and measurement limits

Native ARM and translated x86 AVX2 unit checks exercise rounding, packed layout,
direct FP8 conversion, streamed FP8 scale application, truncated parallel reads,
topology selection and recurrent state updates. Both AVX-512 BW and VNNI paths
compile to x86 machine code, but have not been executed on this Mac. Run the
unit suite on the server before benchmarking them. The full GLM checkpoint and
the dual-Xeon server were not available for local throughput measurement.

An isolated M5 Pro test converting a 2048x4096 FP8 tensor to INT4 averaged
1.85 ms with full float staging versus 1.09 ms with row scratch (four iterations,
18 workers). This excludes storage reads and does not predict server loading
or generation rates. An isolated 128x128 recurrent-state test showed a modest
local improvement; its importance depends on the server's stage profile.

References: [Linux NUMA allocation](https://cdn.kernel.org/doc/html/latest/driver-api/cxl/allocation/page-allocator.html),
[Intel AVX-512 and VNNI tuning](https://cdrdv2-public.intel.com/684945/deep-learning-tuning-guide-en.pdf).
