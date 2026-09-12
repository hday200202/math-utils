#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../include/ai/LocalModel.hpp"
#include "../include/math/Random.hpp"

/* ========================================================================= //
    GPU backend test suite.

    llm-testing already proves the quantiser round trips and that a narrowed
    model still answers. This file asks the questions that only exist once
    there are two machines running the same weights:

      does the GPU kernel compute what the CPU kernel computes, on bytes
      neither of them produced

      does a whole forward pass agree, end to end

      and what does it cost - separating kernel time from bus time, because
      at this stage the activations still live in host memory and the
      answer is uncomfortable

        make release-gpu-test
        ./bin/release/gpu-testing
        ./bin/release/gpu-testing models/qwen3-0.6b
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_DIM   = "\033[90m";
static constexpr const char* COLOR_RESET = "\033[0m";

static int failures = 0;

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] "
              << COLOR_RESET << label << std::endl;
    if (!pass) ++failures;
}

static void section(const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
}

static void note(const std::string& text) {
    std::cout << COLOR_DIM << "       " << text << COLOR_RESET << std::endl;
}

static std::string mb(size_t bytes) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.1f MB", double(bytes) / (1024.0 * 1024.0));
    return buffer;
}


/* ========================================================================= //
    Comparison

    The obvious metric - per-element relative error - is the wrong one, and
    wrong in a way that looks like a bug for a while. A dot product over
    thousands of signed terms mostly cancels, so some outputs land near zero
    and the two reduction orders disagree in the last bits of a number whose
    magnitude is already tiny. That reads as a huge relative error while
    every other row agrees to seven digits.

    So the error is measured against the scale of the RESULT VECTOR. That is
    the quantity the next layer sees, and it is what "the same answer" has
    to mean for a reduction allowed to associate differently.
// ========================================================================= */

struct Diff {
    double maxAbs = 0.0;
    double rms    = 0.0;
    double error  = 0.0;
};

static Diff compare(const std::vector<float>& reference, const std::vector<float>& other) {
    Diff d;
    double square = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        d.maxAbs = std::max(d.maxAbs, std::fabs(double(reference[i]) - double(other[i])));
        square  += double(reference[i]) * double(reference[i]);
    }
    d.rms   = std::sqrt(square / double(reference.size() ? reference.size() : 1));
    d.error = d.rms > 0.0 ? d.maxAbs / d.rms : d.maxAbs;
    return d;
}

static math::random::Random rng(1234);

static std::vector<float> randomFloats(size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = float(rng.randDouble(-1.0, 1.0));
    return out;
}

/* the top 16 bits of a float32 - what a bfloat16 checkpoint would have held,
   so the quantiser is fed the same kind of input it sees in practice */
static std::vector<uint16_t> narrowToBF16(const std::vector<float>& values) {
    std::vector<uint16_t> out(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        out[i] = uint16_t(std::bit_cast<uint32_t>(values[i]) >> 16);
    return out;
}


/* ========================================================================= //
    Device
// ========================================================================= */

static bool testDevice() {
    section("device");

    const bool up = ai::gpu::available();
    check("a HIP device is present", up);
    if (!up) { note(ai::gpu::lastError()); return false; }

    std::cout << "       " << ai::gpu::deviceName()
              << ", " << (ai::gpu::totalMemory() >> 20) << " MB total, "
              << (ai::gpu::freeMemory() >> 20) << " MB free" << std::endl;

    const std::vector<float> sent = randomFloats(4096);
    std::vector<float> back(sent.size(), 0.0f);

    ai::gpu::BufferRef buffer = ai::gpu::upload(sent.data(), sent.size() * sizeof(float));
    check("upload allocates", buffer != nullptr && buffer->valid());
    if (!buffer) { note(ai::gpu::lastError()); return false; }

    check("download returns the same bytes",
          ai::gpu::download(back.data(), *buffer, back.size() * sizeof(float))
          && std::memcmp(sent.data(), back.data(), back.size() * sizeof(float)) == 0);

    const size_t held = ai::gpu::allocatedBytes();
    buffer.reset();
    check("dropping the last reference frees the VRAM", ai::gpu::allocatedBytes() < held);

    return true;
}


/* ========================================================================= //
    GEMV against the CPU kernels

    The reference is ai::kernels / ai::quant - the code that actually runs
    when there is no card - fed the identical bytes. For the quantised
    cases that means ai::quant::quantize runs once and both sides read its
    output, so what is under test is the kernel and not the format.
// ========================================================================= */

static void testMatmulBF16(size_t M, size_t K, size_t N) {
    const std::vector<float>    A  = randomFloats(M * K);
    const std::vector<uint16_t> B  = narrowToBF16(randomFloats(N * K));

    std::vector<float> host(M * N, 0.0f), device(M * N, 0.0f);
    ai::kernels::matmulNTBF16(A.data(), B.data(), host.data(), M, K, N);

    ai::gpu::BufferRef w = ai::gpu::upload(B.data(), B.size() * sizeof(uint16_t));
    if (!w) { check("bf16 upload", false); return; }
    ai::gpu::matmulNTBF16(A.data(), *w, device.data(), M, K, N);

    const Diff d = compare(host, device);
    char label[160];
    std::snprintf(label, sizeof(label),
                  "bf16 M=%zu K=%zu N=%zu   maxAbs %.2e  vs rms %.2e  =  %.1e",
                  M, K, N, d.maxAbs, d.rms, d.error);
    check(label, d.error < 1e-5);
}

static void testMatmulQuant(size_t M, size_t K, size_t N, ai::quant::Type type) {
    const std::vector<float>    A  = randomFloats(M * K);
    const std::vector<uint16_t> B  = narrowToBF16(randomFloats(N * K));

    const ai::quant::QuantTensor q = ai::quant::quantize(B.data(), N, K, type);
    if (!q.valid()) { check("quantise", false); return; }

    std::vector<float> host(M * N, 0.0f), device(M * N, 0.0f);
    ai::kernels::matmulNTQuant(A.data(), q, host.data(), M, K, N);

    const size_t codeBytes = (type == ai::quant::Type::Q4) ? N * K / 2 : N * K;
    ai::gpu::BufferRef codes  = ai::gpu::upload(q.quants.get(), codeBytes);
    ai::gpu::BufferRef scales = ai::gpu::upload(q.scales.get(),
                                                N * q.blocksPerRow() * sizeof(float));
    if (!codes || !scales) { check("quant upload", false); return; }

    if (type == ai::quant::Type::Q4)
        ai::gpu::matmulNTQ4(A.data(), *codes, *scales, device.data(), M, K, N);
    else
        ai::gpu::matmulNTQ8(A.data(), *codes, *scales, device.data(), M, K, N);

    const Diff d = compare(host, device);
    char label[160];
    std::snprintf(label, sizeof(label),
                  "%-4s M=%zu K=%zu N=%zu   maxAbs %.2e  vs rms %.2e  =  %.1e",
                  ai::quant::typeName(type), M, K, N, d.maxAbs, d.rms, d.error);
    check(label, d.error < 1e-5);
}

static void testKernels() {
    section("GEMV vs the CPU kernels");
    note("both sides read identical bytes; the only difference is reduction order");

    testMatmulBF16 (1, 4096, 4096);
    testMatmulBF16 (1, 4096, 12288);
    testMatmulQuant(1, 4096, 4096,  ai::quant::Type::Q8);
    testMatmulQuant(1, 4096, 12288, ai::quant::Type::Q8);
    testMatmulQuant(1, 1536, 1024,  ai::quant::Type::Q8);
    testMatmulQuant(1, 4096, 4096,  ai::quant::Type::Q4);
    testMatmulQuant(1, 4096, 12288, ai::quant::Type::Q4);
    testMatmulQuant(1, 1536, 1024,  ai::quant::Type::Q4);

    section("prefill (M > 1)");
    note("same kernel with a second grid dimension - correct, not yet fast");
    testMatmulBF16 (8,  4096, 4096);
    testMatmulQuant(8,  4096, 4096, ai::quant::Type::Q8);
    testMatmulQuant(37, 1024, 2048, ai::quant::Type::Q4);
}


/* ========================================================================= //
    Bandwidth
// ========================================================================= */

static void testBandwidth() {
    section("bandwidth");

    /*
        512 MB of weights, not the 112 MB a single Qwen3 matrix would be.
        This machine has 96 MB of L3, so a matrix that size is largely cache
        resident and the CPU measures nearly 200 GB/s - a number no real
        model will ever see. Sizing past the cache is what makes the
        comparison honest.
    */
    constexpr size_t K = 4096, N = 65536, REPEATS = 8;

    const std::vector<float>    A = randomFloats(K);
    const std::vector<uint16_t> B = narrowToBF16(randomFloats(N * K));
    const size_t bytes = B.size() * sizeof(uint16_t);

    std::vector<float> out(N, 0.0f);
    using clock = std::chrono::steady_clock;

    const clock::time_point cpuBegin = clock::now();
    for (size_t i = 0; i < REPEATS; ++i)
        ai::kernels::matmulNTBF16(A.data(), B.data(), out.data(), 1, K, N);
    const double cpuSeconds = std::chrono::duration<double>(clock::now() - cpuBegin).count();

    ai::gpu::BufferRef w = ai::gpu::upload(B.data(), bytes);
    if (!w) { check("bandwidth upload", false); return; }

    ai::gpu::matmulNTBF16(A.data(), *w, out.data(), 1, K, N);   // warm
    ai::gpu::resetCounters();

    const clock::time_point gpuBegin = clock::now();
    for (size_t i = 0; i < REPEATS; ++i)
        ai::gpu::matmulNTBF16(A.data(), *w, out.data(), 1, K, N);
    const double gpuSeconds = std::chrono::duration<double>(clock::now() - gpuBegin).count();

    const ai::gpu::Counters& c = ai::gpu::counters();
    const double moved   = double(bytes) * double(REPEATS);
    const double cpuRate = moved / cpuSeconds / 1e9;
    const double gpuRate = moved / c.kernelSeconds / 1e9;

    char line[256];
    std::snprintf(line, sizeof(line), "%s of weights, %zu times", mb(bytes).c_str(), REPEATS);
    note(line);
    std::snprintf(line, sizeof(line), "CPU            %7.1f GB/s   (%.3f s)", cpuRate, cpuSeconds);
    note(line);
    std::snprintf(line, sizeof(line), "GPU kernel     %7.1f GB/s   (%.3f s)", gpuRate, c.kernelSeconds);
    note(line);
    std::snprintf(line, sizeof(line), "GPU wall clock %7.1f GB/s   (%.3f s, %.3f s of it on the bus)",
                  moved / gpuSeconds / 1e9, gpuSeconds, c.transferSeconds);
    note(line);

    check("the GPU kernel beats the CPU on weight bandwidth", gpuRate > cpuRate);
}


/* ========================================================================= //
    Quality, position by position

    "Does it produce the same text" is the wrong question and always answers
    no: greedy decoding is a chain, so one changed token re-conditions
    everything after it. These three mean something instead, measured over
    the SAME forced prefix so the runs never drift apart.

      top-1 agreement  what you would notice as a different word
      KL divergence    how far the whole distribution moved, in nats
      perplexity       the model's own loss on real text - the only one of
                       the three that says whether it got WORSE rather than
                       merely DIFFERENT
// ========================================================================= */

static std::vector<float> softmax(const float* logits, size_t n) {
    std::vector<float> out(n);
    float best = logits[0];
    for (size_t i = 1; i < n; ++i) best = std::max(best, logits[i]);

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) { out[i] = std::exp(logits[i] - best); sum += out[i]; }
    for (size_t i = 0; i < n; ++i) out[i] = float(double(out[i]) / sum);
    return out;
}

static std::vector<std::vector<float>> teacherForced(const ai::LocalModel& model,
                                                     const std::vector<size_t>& ids) {
    ai::LocalCache cache;
    cache.resize(model.cfg, ids.size() + 8);

    std::vector<std::vector<float>> out;
    out.reserve(ids.size() - 1);

    std::vector<float> logits(model.cfg.vocab, 0.0f);
    for (size_t t = 0; t + 1 < ids.size(); ++t) {
        model.forward(&ids[t], 1, t, cache, logits.data());
        out.push_back(logits);
    }
    return out;
}

struct Quality {
    double agreement = 0.0;
    double meanKL    = 0.0;
    double worstKL   = 0.0;
    double perplexityRef  = 0.0;
    double perplexityTest = 0.0;

    /*
        agreement restricted to positions where the reference was sure.
        Raw top-1 punishes a harmless change severely: where the model is
        torn between two continuations at 0.31 and 0.30, any perturbation
        flips the argmax and nothing meaningful has happened. Where it was
        at 0.9 and the argmax still moves, something did.
    */
    double confidentAgreement = 0.0;
    size_t confidentPositions = 0;
};

static Quality assess(const std::vector<std::vector<float>>& reference,
                      const std::vector<std::vector<float>>& test,
                      const std::vector<size_t>& ids) {
    Quality q;
    const size_t positions = reference.size();
    if (positions == 0) return q;

    size_t agreed = 0, confident = 0, confidentAgreed = 0;
    double totalKL = 0.0, nllRef = 0.0, nllTest = 0.0;

    for (size_t t = 0; t < positions; ++t) {
        const std::vector<float> p = softmax(reference[t].data(), reference[t].size());
        const std::vector<float> r = softmax(test[t].data(), test[t].size());

        const size_t refTop = ai::LocalModel::argmax(reference[t].data(), reference[t].size());
        const bool   same   = refTop == ai::LocalModel::argmax(test[t].data(), test[t].size());
        if (same) ++agreed;
        if (p[refTop] > 0.5f) { ++confident; if (same) ++confidentAgreed; }

        double kl = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            if (p[i] <= 1e-12) continue;
            kl += double(p[i]) * (std::log(double(p[i])) - std::log(std::max(double(r[i]), 1e-12)));
        }
        totalKL += kl;
        q.worstKL = std::max(q.worstKL, kl);

        const size_t truth = ids[t + 1];
        nllRef  -= std::log(std::max(double(p[truth]), 1e-12));
        nllTest -= std::log(std::max(double(r[truth]), 1e-12));
    }

    q.agreement          = double(agreed) / double(positions);
    q.confidentPositions = confident;
    q.confidentAgreement = confident ? double(confidentAgreed) / double(confident) : 1.0;
    q.meanKL             = totalKL / double(positions);
    q.perplexityRef      = std::exp(nllRef  / double(positions));
    q.perplexityTest     = std::exp(nllTest / double(positions));
    return q;
}

static const char* PASSAGE =
    "The city had grown outward from the river in uneven rings, each one "
    "marking a decade when money had been plentiful and confidence high. "
    "Engineers who arrived later found the older districts difficult to "
    "work in, because the drawings had been lost and the assumptions "
    "behind them were no longer written down anywhere. What remained was "
    "the buildings themselves, which had to be measured again by hand "
    "before anything could be altered. It was slow work, and it was the "
    "only kind that could be trusted.";


/* ========================================================================= //
    End to end
// ========================================================================= */

static void testEndToEnd(const std::string& directory) {
    section("end to end: " + directory);

    ai::LocalModel model;
    if (!model.load(directory)) { check("load", false); note(model.error); return; }

    char line[256];
    std::snprintf(line, sizeof(line), "%s, %zu layers, hidden %zu, vocab %zu",
                  model.modelType.c_str(), model.cfg.nLayer, model.cfg.hidden, model.cfg.vocab);
    note(line);

    const std::vector<size_t> prompt = { 3838, 374, 279, 6722, 315, 9625, 30 };
    ai::LocalCache cache;
    cache.resize(model.cfg, 256);

    std::vector<float> cpuLogits(model.cfg.vocab, 0.0f);
    model.forward(prompt.data(), prompt.size(), 0, cache, cpuLogits.data());

    const size_t footprint = model.deviceFootprint();
    std::snprintf(line, sizeof(line), "weights on device would be %s, %zu MB free",
                  mb(footprint).c_str(), ai::gpu::freeMemory() >> 20);
    note(line);

    using clock = std::chrono::steady_clock;
    const clock::time_point uploadBegin = clock::now();
    const ai::LocalModel::DeviceReport report = model.toDevice({ .releaseHost = false });
    const double uploadSeconds = std::chrono::duration<double>(clock::now() - uploadBegin).count();

    check("weights uploaded", report.ok);
    if (!report.ok) { note(report.error); return; }

    std::snprintf(line, sizeof(line), "%zu matrices, %s, %.1f s (%.1f GB/s over PCIe)",
                  report.matrices, mb(report.bytes).c_str(), uploadSeconds,
                  double(report.bytes) / uploadSeconds / 1e9);
    note(line);

    cache.reset();
    ai::gpu::resetCounters();

    std::vector<float> gpuLogits(model.cfg.vocab, 0.0f);
    model.forward(prompt.data(), prompt.size(), 0, cache, gpuLogits.data());

    const Diff d = compare(cpuLogits, gpuLogits);
    std::snprintf(line, sizeof(line), "bf16 logits agree   maxAbs %.2e  vs rms %.2e  =  %.1e",
                  d.maxAbs, d.rms, d.error);
    check(line, d.error < 1e-3);
    check("the top token is the same",
          ai::LocalModel::argmax(cpuLogits.data(), cpuLogits.size()) ==
          ai::LocalModel::argmax(gpuLogits.data(), gpuLogits.size()));

    /* ---- tokens per second and where they go ------------------------ */
    section("tokens per second");

    auto rate = [&](size_t tokens) {
        std::vector<float> logits(model.cfg.vocab, 0.0f);
        cache.reset();
        model.forward(prompt.data(), prompt.size(), 0, cache, logits.data());
        ai::gpu::resetCounters();

        const clock::time_point begin = clock::now();
        size_t next = ai::LocalModel::argmax(logits.data(), logits.size());
        for (size_t i = 0; i < tokens; ++i) {
            model.forward(&next, 1, cache.length, cache, logits.data());
            next = ai::LocalModel::argmax(logits.data(), logits.size());
        }
        return std::chrono::duration<double>(clock::now() - begin).count();
    };

    const double gpuSeconds = rate(32);
    const ai::gpu::Counters c = ai::gpu::counters();
    model.toHost();
    const double cpuSeconds = rate(32);

    std::snprintf(line, sizeof(line), "CPU  %6.2f tok/s", 32.0 / cpuSeconds);
    note(line);
    std::snprintf(line, sizeof(line), "GPU  %6.2f tok/s   (%.2fx)",
                  32.0 / gpuSeconds, cpuSeconds / gpuSeconds);
    note(line);
    check("the GPU path is faster end to end", gpuSeconds < cpuSeconds);

    section("where a token goes on the GPU");
    const double other = gpuSeconds - c.kernelSeconds - c.transferSeconds;
    const double each  = 100.0 / gpuSeconds;

    std::snprintf(line, sizeof(line), "%zu matmuls per token, each one a separate round trip",
                  c.launches / 32);
    note(line);
    std::snprintf(line, sizeof(line), "kernels          %6.3f s   %4.1f%%", c.kernelSeconds, c.kernelSeconds * each);
    note(line);
    std::snprintf(line, sizeof(line), "PCIe both ways   %6.3f s   %4.1f%%", c.transferSeconds, c.transferSeconds * each);
    note(line);
    std::snprintf(line, sizeof(line),
                  "everything else  %6.3f s   %4.1f%%   (norms, RoPE, attention - still on the CPU)",
                  other, other * each);
    note(line);
    std::snprintf(line, sizeof(line),
                  "kernels alone would be %.1f tok/s, so the port is roughly %.0f%% done by time",
                  32.0 / c.kernelSeconds, c.kernelSeconds * each);
    note(line);
}


/* ========================================================================= //
    The CPU and the GPU reading the same codes

    The quantiser is shared, so both start from identical bytes. That is
    necessary and not sufficient: two consumers can still disagree about
    what a byte means - which nibble comes first, whether a scale is per row
    or per block, whether the bias goes on before or after. This is the test
    that would catch it.
// ========================================================================= */

static void testHostAgainstDevice(const std::string& directory) {
    section("CPU and GPU on the same codes");

    const ai::quant::Type TYPES[2] = { ai::quant::Type::Q8, ai::quant::Type::Q4 };

    for (ai::quant::Type type : TYPES) {
        ai::LocalModel model;
        if (!model.load(directory, {}, type)) {
            check(std::string(ai::quant::typeName(type)) + " loads", false);
            note(model.error);
            return;
        }

        const std::vector<size_t> prompt = { 3838, 374, 279, 6722, 315, 9625, 30 };
        ai::LocalCache cache;
        cache.resize(model.cfg, 64);

        std::vector<float> cpuLogits(model.cfg.vocab, 0.0f);
        model.forward(prompt.data(), prompt.size(), 0, cache, cpuLogits.data());

        if (!model.toDevice({ .releaseHost = false }).ok) { check("upload", false); return; }

        cache.reset();
        std::vector<float> gpuLogits(model.cfg.vocab, 0.0f);
        model.forward(prompt.data(), prompt.size(), 0, cache, gpuLogits.data());

        const Diff d = compare(cpuLogits, gpuLogits);
        char label[192];
        std::snprintf(label, sizeof(label),
                      "%s: CPU and GPU logits agree   maxAbs %.2e  vs rms %.2e  =  %.1e",
                      ai::quant::typeName(type), d.maxAbs, d.rms, d.error);
        check(label, d.error < 1e-4);
        check(std::string(ai::quant::typeName(type)) + ": and pick the same token",
              ai::LocalModel::argmax(cpuLogits.data(), cpuLogits.size()) ==
              ai::LocalModel::argmax(gpuLogits.data(), gpuLogits.size()));
    }
}


/* ========================================================================= //
    What narrowing costs the output
// ========================================================================= */

static void testQuantisationQuality(const std::string& directory) {
    section("what quantisation costs the output");

    ai::LocalModel wide;
    if (!wide.load(directory)) { check("bf16 loads", false); note(wide.error); return; }

    const std::vector<size_t> ids = wide.tokenizer.encode(PASSAGE, false);
    char line[256];
    std::snprintf(line, sizeof(line), "%zu tokens, teacher forced, on the card", ids.size());
    note(line);

    if (!wide.toDevice({ .releaseHost = false }).ok) { check("bf16 uploads", false); return; }
    const std::vector<std::vector<float>> reference = teacherForced(wide, ids);
    wide.toHost();

    const ai::quant::Type TYPES[2] = { ai::quant::Type::Q8, ai::quant::Type::Q4 };
    Quality results[2];

    for (int pass = 0; pass < 2; ++pass) {
        ai::LocalModel narrow;
        if (!narrow.load(directory, {}, TYPES[pass])) { check("load", false); return; }
        if (!narrow.toDevice({ .releaseHost = false }).ok) { check("upload", false); return; }

        results[pass] = assess(reference, teacherForced(narrow, ids), ids);
        const Quality& q = results[pass];

        std::snprintf(line, sizeof(line), "%-4s  %s on the card",
                      ai::quant::typeName(TYPES[pass]), mb(narrow.deviceFootprint()).c_str());
        note(line);
        std::snprintf(line, sizeof(line),
                      "      top-1 agreement  %5.1f%%     confident-only %5.1f%% (%zu)",
                      q.agreement * 100.0, q.confidentAgreement * 100.0, q.confidentPositions);
        note(line);
        std::snprintf(line, sizeof(line), "      mean KL          %.4f nats   worst %.4f",
                      q.meanKL, q.worstKL);
        note(line);
        std::snprintf(line, sizeof(line), "      perplexity       %.4f      (%+.2f%% vs bf16 %.4f)",
                      q.perplexityTest,
                      100.0 * (q.perplexityTest - q.perplexityRef) / q.perplexityRef,
                      q.perplexityRef);
        note(line);
    }

    /*
        The int4 bars are calibrated to what round-to-nearest at four bits
        actually delivers on this model, measured, not to what would be
        nice. They are regression guards, not a claim that this is good
        enough for any particular use.
    */
    check("int8: confident-position agreement at least 99%", results[0].confidentAgreement >= 0.99);
    check("int8: mean KL under 0.01 nats", results[0].meanKL < 0.01);
    check("int8: perplexity within 1%",
          std::fabs(results[0].perplexityTest - results[0].perplexityRef)
              / results[0].perplexityRef < 0.01);

    check("int4: confident-position agreement at least 90%", results[1].confidentAgreement >= 0.90);
    check("int4: mean KL under 0.3 nats", results[1].meanKL < 0.3);
    check("int4: perplexity within 10%",
          std::fabs(results[1].perplexityTest - results[1].perplexityRef)
              / results[1].perplexityRef < 0.10);
}


int main(int argc, char** argv) {
    if (!testDevice()) {
        std::cout << "\nno device, nothing further to test\n";
        return 1;
    }

    testKernels();
    testBandwidth();

    for (int i = 1; i < argc; ++i) {
        testEndToEnd(argv[i]);
        testHostAgainstDevice(argv[i]);
        testQuantisationQuality(argv[i]);
    }

    std::cout << "\n" << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " failed" : "all passed")
              << COLOR_RESET << std::endl;

    return failures ? 1 : 0;
}
