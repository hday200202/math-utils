#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../include/ai/gpu/DeviceABI.h"
#include "../../include/platform/Memory.hpp"
#include "metal_library.h"

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Buffer { id<MTLBuffer> storage; uint64_t bytes; };
struct Shape { uint32_t m, k, n; };

struct Context {
    std::mutex gate;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> vec[4], matrix[4], multi[4];
    id<MTLComputePipelineState> dense[8];
    id<MTLBuffer> input, output, dummy;
    uint64_t allocated = 0;
    AiGpuCounters stats{};
    std::string name, error;
    bool ok = false;

    Context() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) { error = "no Metal device found"; return; }
            if (![device supportsFamily:MTLGPUFamilyApple7]) {
                error = "Metal backend requires an Apple GPU with SIMD-group matrix support";
                return;
            }
            name = device.name.UTF8String;
            queue = [device newCommandQueue];
            NSError* issue = nil;
            dispatch_data_t data = dispatch_data_create(METAL_LIBRARY, sizeof(METAL_LIBRARY),
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{});
            id<MTLLibrary> library = [device newLibraryWithData:data error:&issue];
            if (!library || !queue) { error = issue ? issue.localizedDescription.UTF8String : "Metal initialization failed"; return; }
            const auto pipelineFor = [&](id<MTLFunction> fn) -> id<MTLComputePipelineState> {
                if (!fn) return nil;
                MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
                descriptor.computeFunction = fn;
                descriptor.supportIndirectCommandBuffers = YES;
                descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                return [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&issue];
            };
            for (uint16_t type = 0; type < 4; ++type) {
                MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
                [constants setConstantValue:&type type:MTLDataTypeUShort atIndex:0];
                for (int tiled = 0; tiled < 2; ++tiled) {
                    id<MTLFunction> fn = [library newFunctionWithName:tiled ? @"gemm" : @"gemv"
                        constantValues:constants error:&issue];
                    id<MTLComputePipelineState> pipeline = pipelineFor(fn);
                    if (!pipeline || pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 128) {
                        error = issue ? issue.localizedDescription.UTF8String : "unsupported Metal compute limits";
                        return;
                    }
                    if (tiled) matrix[type] = pipeline; else vec[type] = pipeline;
                }
                id<MTLFunction> combined = [library newFunctionWithName:@"gemv_multi" constantValues:constants error:&issue];
                multi[type] = pipelineFor(combined);
                if (!multi[type]) { error = issue.localizedDescription.UTF8String; return; }
            }
            NSArray<NSString*>* names = @[@"dense_norm", @"dense_qk_cache", @"dense_scores", @"dense_softmax_values", @"dense_swiglu", @"dense_bias", @"dense_partial_values", @"dense_merge_values"];
            for (unsigned i = 0; i < 8; ++i) {
                id<MTLFunction> fn = [library newFunctionWithName:names[i]];
                dense[i] = pipelineFor(fn);
                if (!dense[i] || dense[i].threadExecutionWidth != 32 || dense[i].maxTotalThreadsPerThreadgroup < 128) {
                    error = issue ? issue.localizedDescription.UTF8String : "could not create resident decode pipeline";
                    return;
                }
            }
            dummy = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
            if (!dummy) { error = "could not allocate Metal dummy buffer"; return; }
            ok = true;
        }
    }

    static Context& get() { static Context value; return value; }
    bool grow(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer && buffer.length >= bytes) return true;
        if (bytes > device.maxBufferLength) { error = "Metal scratch exceeds maximum buffer length"; return false; }
        // No command can still reference scratch: each call completes under gate.
        allocated -= buffer.length;
        buffer = nil;
        buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) { error = "Metal scratch allocation failed"; return false; }
        allocated += buffer.length;
        return true;
    }
};

uint64_t product(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::invalid_argument("Metal matrix dimensions overflow");
    return a * b;
}

void matmul(const float* a, const void* weight, const void* scales, float* out,
            uint64_t m, uint64_t k, uint64_t n, unsigned type) {
    if (!m || !n) return;
    auto& ctx = Context::get();
    std::lock_guard<std::mutex> lock(ctx.gate);
    @autoreleasepool {
        auto fail = [&](const std::string& error) {
            ctx.error = error;
            throw std::runtime_error("Metal matmul: " + error);
        };
        if (!ctx.ok) fail(ctx.error);
        if (!a || !out || !weight || !k || m > UINT32_MAX || k > UINT32_MAX || n > UINT32_MAX)
            fail("invalid dimensions or null input");
        if (type >= 2 && (k % 32 || !scales)) fail("quantized weights require K divisible by 32 and block scales");
        const Buffer& b = *static_cast<const Buffer*>(weight);
        const Buffer* s = static_cast<const Buffer*>(scales);
        const uint64_t codes = product(n, k);
        const uint64_t expected = type == 3 ? codes / 2 : product(codes, type == 0 ? 4 : type == 1 ? 2 : 1);
        if (b.bytes < expected || (s && s->bytes < product(codes / 32, 4))) fail("weight or scale buffer too small");
        const uint64_t inputBytes = product(product(m, k), 4), outputBytes = product(product(m, n), 4);
        if (!ctx.grow(ctx.input, inputBytes) || !ctx.grow(ctx.output, outputBytes)) fail(ctx.error);
        auto start = Clock::now();
        std::memcpy(ctx.input.contents, a, inputBytes);
        ctx.stats.hostToDeviceBytes += inputBytes;
        ctx.stats.transferSeconds += seconds(start);
        // A ragged row is not aligned for the vector loads; the tiled path
        // uses scalar bounds-checked loads and also handles tiny matrices.
        const bool tiled = m >= 8 || k % 4;
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) fail("could not create compute encoder");
        [encoder setComputePipelineState:tiled ? ctx.matrix[type] : ctx.vec[type]];
        [encoder setBuffer:ctx.input offset:0 atIndex:0];
        [encoder setBuffer:b.storage offset:0 atIndex:1];
        [encoder setBuffer:s ? s->storage : ctx.dummy offset:0 atIndex:2];
        [encoder setBuffer:ctx.output offset:0 atIndex:3];
        const Shape shape{uint32_t(m), uint32_t(k), uint32_t(n)};
        [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
        const MTLSize grid = tiled ? MTLSizeMake((n + 31) / 32, (m + 31) / 32, 1)
                                  : MTLSizeMake((n + 3) / 4, m, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            fail(command.error ? command.error.localizedDescription.UTF8String : "GPU command failed");
        ++ctx.stats.launches;
        ctx.stats.kernelSeconds += command.GPUEndTime - command.GPUStartTime;
        start = Clock::now();
        std::memcpy(out, ctx.output.contents, outputBytes);
        ctx.stats.deviceToHostBytes += outputBytes;
        ctx.stats.transferSeconds += seconds(start);
        ctx.error.clear();
    }
}

struct DenseWeight {
    Buffer codes, scales;
    id<MTLBuffer> bias;
    uint32_t type, inDim, outDim;
    explicit DenseWeight(const AiGpuDenseWeight& w) : type(w.type), inDim(w.inDim), outDim(w.outDim) {
        if (!w.codes || type > 3 || !inDim || !outDim || inDim % 4)
            throw std::invalid_argument("invalid resident weight");
        codes = *static_cast<const Buffer*>(w.codes);
        scales = w.scales ? *static_cast<const Buffer*>(w.scales) : Buffer{nil, 0};
        const uint64_t elements = product(inDim, outDim);
        const uint64_t bytes = type == 3 ? elements / 2 : product(elements, type == 0 ? 4 : type == 1 ? 2 : 1);
        if (codes.bytes < bytes || (type >= 2 && (inDim % 32 || scales.bytes < product(elements / 32, 4))))
            throw std::invalid_argument("invalid resident weight storage");
    }
};
struct DenseLayer {
    DenseWeight q, k, v, o, gate, up, down;
    uint32_t window = 0;
    id<MTLBuffer> inputNorm, postNorm, qNorm, kNorm, keys, values;
    explicit DenseLayer(const AiGpuDenseLayer& l) : q(l.q), k(l.k), v(l.v), o(l.o), gate(l.gate), up(l.up), down(l.down) {}
};
struct DenseParams {
    uint32_t width, rows, heads, kvHeads, dim, capacity, position, rotary;
    float epsilon;
    uint32_t mode, window;
};
struct DenseContext {
    AiGpuDenseConfig cfg;
    DenseWeight head;
    std::vector<DenseLayer> layers;
    id<MTLBuffer> x, normed, q, k, v, attn, proj, gate, up, scores, trig, logits, finalNorm, partials;
    id<MTLIndirectCommandBuffer> program;
    id<MTLBuffer> parameters;
    uint32_t commandCount = 0;
    std::vector<uint32_t> dynamicParameters, scoreCommands, partitionCommands;
    std::vector<id<MTLResource>> resources;
    uint64_t held = 0;
    uint32_t length = 0;
    std::vector<float> frequencies;
    explicit DenseContext(const AiGpuDenseConfig& c) : cfg(c), head(c.head) {}
    ~DenseContext() { Context::get().allocated -= held; }
    id<MTLBuffer> buffer(uint64_t bytes, const void* data = nullptr, uint64_t copied = 0) {
        auto& ctx = Context::get();
        if (!bytes || bytes > ctx.device.maxBufferLength || copied > bytes)
            throw std::invalid_argument("resident buffer exceeds Metal limits");
        id<MTLBuffer> b = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("could not allocate resident decoder buffer");
        if (data && copied) std::memcpy(b.contents, data, copied);
        held += bytes;
        ctx.allocated += bytes;
        return b;
    }
};

// Record the static transformer once. Only positions and the attention
// grid change between tokens; replay avoids hundreds of CPU encoder calls.
struct Recorder {
    Context& ctx;
    DenseContext& d;
    id<MTLBuffer> bindings[10];
    void resource(id<MTLResource> r) {
        if (std::find(d.resources.begin(), d.resources.end(), r) == d.resources.end()) d.resources.push_back(r);
    }
    void emit(id<MTLComputePipelineState> pipeline, const void* params, size_t bytes,
              MTLSize groups, bool dynamic = false, bool scores = false, unsigned parameterIndex = 4) {
        const uint32_t index = d.commandCount++;
        if (index >= d.cfg.layers * 24 + 4) throw std::runtime_error("resident program capacity exceeded");
        id<MTLIndirectComputeCommand> command = [d.program indirectComputeCommandAtIndex:index];
        [command setComputePipelineState:pipeline];
        for (unsigned i = 0; i < parameterIndex; ++i) {
            [command setKernelBuffer:bindings[i] offset:0 atIndex:i];
            resource(bindings[i]);
        }
        const size_t offset = size_t(index) * 256;
        std::memcpy(static_cast<char*>(d.parameters.contents) + offset, params, bytes);
        [command setKernelBuffer:d.parameters offset:offset atIndex:parameterIndex];
        [command concurrentDispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [command setBarrier];
        if (dynamic) d.dynamicParameters.push_back(index);
        if (scores) d.scoreCommands.push_back(index);
    }
    void matmul(id<MTLBuffer> a, const DenseWeight& w, id<MTLBuffer> out) {
        bindings[0] = a; bindings[1] = w.codes.storage;
        bindings[2] = w.scales.storage ? w.scales.storage : ctx.dummy; bindings[3] = out;
        const Shape shape{1, w.inDim, w.outDim};
        emit(ctx.vec[w.type], &shape, sizeof(shape), MTLSizeMake((w.outDim + 3) / 4, 1, 1));
        bias(w, out);
    }
    void bias(const DenseWeight& w, id<MTLBuffer> out) {
        if (!w.bias) return;
        bindings[0] = out; bindings[1] = w.bias;
        const DenseParams p{w.outDim, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        emit(ctx.dense[5], &p, sizeof(p), MTLSizeMake((w.outDim + 127) / 128, 1, 1));
    }
    void combined(id<MTLBuffer> a, const DenseWeight& w0, const DenseWeight& w1,
        const DenseWeight* w2, id<MTLBuffer> o0, id<MTLBuffer> o1, id<MTLBuffer> o2) {
        if (w0.type != w1.type || (w2 && w0.type != w2->type)) {
            matmul(a, w0, o0); matmul(a, w1, o1); if (w2) matmul(a, *w2, o2);
            return;
        }
        bindings[0] = a; bindings[1] = w0.codes.storage; bindings[2] = w1.codes.storage;
        bindings[3] = w2 ? w2->codes.storage : ctx.dummy;
        bindings[4] = w0.scales.storage ? w0.scales.storage : ctx.dummy;
        bindings[5] = w1.scales.storage ? w1.scales.storage : ctx.dummy;
        bindings[6] = w2 && w2->scales.storage ? w2->scales.storage : ctx.dummy;
        bindings[7] = o0; bindings[8] = o1; bindings[9] = o2;
        const uint32_t shape[]{w0.inDim, w0.outDim, w1.outDim, w2 ? w2->outDim : 0};
        const uint64_t rows = uint64_t(shape[1]) + shape[2] + shape[3];
        emit(ctx.multi[w0.type], shape, sizeof(shape), MTLSizeMake((rows + 3) / 4, 1, 1), false, false, 10);
        bias(w0, o0); bias(w1, o1); if (w2) bias(*w2, o2);
    }

};

void recordProgram(DenseContext& d) {
    auto& ctx = Context::get();
    const auto& c = d.cfg;
    const uint32_t position = d.length;
    MTLIndirectCommandBufferDescriptor* descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritBuffers = NO; descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = 11;
    const uint64_t count = uint64_t(c.layers) * 24 + 4;
    d.program = [ctx.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:count options:MTLResourceStorageModeShared];
    if (!d.program) throw std::runtime_error("could not allocate resident indirect commands");
    d.parameters = d.buffer(count * 256);
    Recorder r{ctx, d, {ctx.dummy, ctx.dummy, ctx.dummy, ctx.dummy}};
    DenseParams p{c.hidden, 1, c.heads, c.kvHeads, c.headDim, c.capacity, position, c.rotaryDims, c.epsilon, 0, 0};
    const auto bind = [&](unsigned index, id<MTLBuffer> b) { r.bindings[index] = b; };
    const auto dispatch = [&](unsigned pipeline, uint64_t groups, uint64_t y = 1, unsigned parameterIndex = 4) {
        const uint32_t slot = d.commandCount;
        r.emit(ctx.dense[pipeline], &p, sizeof(p), MTLSizeMake(groups, y, 1), true, pipeline == 2, parameterIndex);
        if (pipeline == 6) d.partitionCommands.push_back(slot);
    };
    const auto norm = [&](id<MTLBuffer> x, id<MTLBuffer> gamma, id<MTLBuffer> out,
        uint32_t width, uint32_t rows, bool add = false) {
        bind(0, x); bind(1, gamma); bind(2, out); bind(3, add ? d.proj : ctx.dummy);
        p.width = width; p.rows = rows; p.mode = add;
        dispatch(0, (rows + 3) / 4);
        p.mode = 0;
    };
    bool addPrevious = false;
    for (size_t layer = 0; layer < d.layers.size(); ++layer) {
        const auto& l = d.layers[layer];
        p.window = 0;
        norm(d.x, l.inputNorm, d.normed, c.hidden, 1, addPrevious);
        r.combined(d.normed, l.q, l.k, &l.v, d.q, d.k, d.v);
        bind(0, d.q); bind(1, d.k); bind(2, d.v);
        bind(3, l.qNorm ? l.qNorm : ctx.dummy); bind(4, l.kNorm ? l.kNorm : ctx.dummy);
        bind(5, l.keys); bind(6, l.values); bind(7, d.trig);
        p.mode = (l.qNorm ? 1 : 0) | (l.kNorm ? 2 : 0);
        dispatch(1, (uint64_t(c.heads) + c.kvHeads + 3) / 4, 1, 8);
        p.mode = 0;
        p.window = l.window;
        bind(0, d.q); bind(1, l.keys); bind(2, d.scores);
        dispatch(2, (position + 4) / 4, c.heads);
        bind(0, d.scores); bind(1, l.values); bind(2, d.attn);
        if (position >= 256) {
            bind(2, d.partials); p.rows = uint32_t((uint64_t(c.capacity) + 127) / 128);
            const uint32_t length = p.window ? std::min(position + 1, p.window) : position + 1;
            dispatch(6, (c.heads + 3) / 4, (length + 127) / 128);
            bind(0, d.partials); bind(1, d.attn);
            dispatch(7, (c.heads + 3) / 4);
        } else dispatch(3, (c.heads + 3) / 4);
        r.matmul(d.attn, l.o, d.proj);
        norm(d.x, l.postNorm, d.normed, c.hidden, 1, true);
        r.combined(d.normed, l.gate, l.up, nullptr, d.gate, d.up, ctx.dummy);
        bind(0, d.gate); bind(1, d.up); p.width = c.intermediate;
        dispatch(4, (c.intermediate + 127) / 128);
        r.matmul(d.gate, l.down, d.proj);
        addPrevious = true;
    }
    norm(d.x, d.finalNorm, d.normed, c.hidden, 1, true);
    r.matmul(d.normed, d.head, d.logits);
    r.resource(d.parameters);
}

void denseDecode(DenseContext& d, const float* embedding, float* logits, uint32_t position) {
    auto& ctx = Context::get();
    const auto& c = d.cfg;
    if (!embedding || !logits || position >= c.capacity || position != d.length)
        throw std::invalid_argument("resident decode position does not match KV cache");
    auto start = Clock::now();
    std::memcpy(d.x.contents, embedding, c.hidden * sizeof(float));
    auto* trig = static_cast<float*>(d.trig.contents);
    for (uint32_t j = 0; j < c.rotaryDims / 2; ++j) {
        const float angle = float(position) * d.frequencies[j];
        trig[2 * j] = std::cos(angle); trig[2 * j + 1] = std::sin(angle);
    }
    ctx.stats.hostToDeviceBytes += (c.hidden + c.rotaryDims) * sizeof(float);
    ctx.stats.transferSeconds += seconds(start);
    id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
    if (!e) throw std::runtime_error("could not create resident decode encoder");
    if (d.program && position == 256) {
        // Switch once the history can feed several independent partitions.
        const uint64_t bytes = d.parameters.length;
        ctx.allocated -= bytes; d.held -= bytes;
        d.program = nil; d.parameters = nil; d.commandCount = 0;
        d.resources.clear(); d.dynamicParameters.clear(); d.scoreCommands.clear(); d.partitionCommands.clear();
    }
    if (!d.program) recordProgram(d);
    for (uint32_t slot : d.dynamicParameters) {
        auto* params = reinterpret_cast<DenseParams*>(static_cast<char*>(d.parameters.contents) + size_t(slot) * 256);
        params->position = position;
    }
    for (uint32_t slot : d.scoreCommands) {
        [[d.program indirectComputeCommandAtIndex:slot] concurrentDispatchThreadgroups:MTLSizeMake((position + 4) / 4, c.heads, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    for (uint32_t slot : d.partitionCommands) {
        const auto* params = reinterpret_cast<const DenseParams*>(static_cast<const char*>(d.parameters.contents) + size_t(slot) * 256);
        const uint32_t length = params->window ? std::min(position + 1, params->window) : position + 1;
        [[d.program indirectComputeCommandAtIndex:slot] concurrentDispatchThreadgroups:MTLSizeMake((c.heads + 3) / 4, (length + 127) / 128, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [e useResources:d.resources.data() count:d.resources.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [e executeCommandsInBuffer:d.program withRange:NSMakeRange(0, d.commandCount)];
    ctx.stats.launches += d.commandCount;
    [e endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error ? command.error.localizedDescription.UTF8String : "resident GPU command failed");
    ctx.stats.kernelSeconds += command.GPUEndTime - command.GPUStartTime;
    start = Clock::now();
    std::memcpy(logits, d.logits.contents, uint64_t(c.vocab) * sizeof(float));
    ctx.stats.deviceToHostBytes += uint64_t(c.vocab) * sizeof(float);
    ctx.stats.transferSeconds += seconds(start);
    ++d.length;
}
} // namespace

extern "C" {
int aigpu_available() { return Context::get().ok; }
const char* aigpu_device_name() { return Context::get().name.c_str(); }
const char* aigpu_last_error() {
    auto& c = Context::get();
    std::lock_guard<std::mutex> lock(c.gate);
    static thread_local std::string snapshot;
    snapshot = c.error;
    return snapshot.c_str();
}
uint64_t aigpu_total_memory() { return Context::get().device.recommendedMaxWorkingSetSize; }
uint64_t aigpu_free_memory() {
    auto& c = Context::get();
    if (!c.ok) return 0;
    const uint64_t used = c.device.currentAllocatedSize, budget = c.device.recommendedMaxWorkingSetSize;
    return std::min<uint64_t>(budget > used ? budget - used : 0, platform::availableMemory());
}
uint64_t aigpu_allocated_bytes() {
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate); return c.allocated;
}
void* aigpu_upload(const void* host, uint64_t bytes) {
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate);
    @autoreleasepool {
        if (!c.ok || !host || !bytes || bytes > c.device.maxBufferLength) {
            c.error = "invalid Metal upload or buffer exceeds maximum length"; return nullptr;
        }
        auto start = Clock::now();
        id<MTLBuffer> buffer = [c.device newBufferWithBytes:host length:bytes options:MTLResourceStorageModeShared];
        if (!buffer) { c.error = "Metal weight allocation failed"; return nullptr; }
        c.allocated += bytes;
        c.stats.hostToDeviceBytes += bytes;
        c.stats.transferSeconds += seconds(start);
        c.error.clear();
        return new Buffer{buffer, bytes};
    }
}
void aigpu_release(void* handle, uint64_t) {
    if (!handle) return;
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate);
    auto* buffer = static_cast<Buffer*>(handle);
    c.allocated -= buffer->bytes;
    delete buffer;
}
int aigpu_download(void* host, const void* handle, uint64_t bytes) {
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate);
    if (!handle || !host || bytes > static_cast<const Buffer*>(handle)->bytes) {
        c.error = "invalid Metal download or buffer too small"; return 0;
    }
    auto start = Clock::now();
    std::memcpy(host, static_cast<const Buffer*>(handle)->storage.contents, bytes);
    c.stats.deviceToHostBytes += bytes;
    c.stats.transferSeconds += seconds(start);
    c.error.clear();
    return 1;
}
void aigpu_matmul_nt(const float* a, const void* b, float* c, uint64_t m, uint64_t k, uint64_t n) { matmul(a,b,nullptr,c,m,k,n,0); }
void aigpu_matmul_nt_bf16(const float* a, const void* b, float* c, uint64_t m, uint64_t k, uint64_t n) { matmul(a,b,nullptr,c,m,k,n,1); }
void aigpu_matmul_nt_q8(const float* a, const void* b, const void* s, float* c, uint64_t m, uint64_t k, uint64_t n) { matmul(a,b,s,c,m,k,n,2); }
void aigpu_matmul_nt_q4(const float* a, const void* b, const void* s, float* c, uint64_t m, uint64_t k, uint64_t n) { matmul(a,b,s,c,m,k,n,3); }
void aigpu_synchronize() {
    // Each public dispatch is synchronous. The lock also waits for calls
    // issued by other CPU threads (for example concurrently routed experts).
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate);
}
void aigpu_counters(AiGpuCounters* out) {
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate);
    if (out) *out = c.stats;
}
void aigpu_reset_counters() {
    auto& c = Context::get(); std::lock_guard<std::mutex> lock(c.gate);
    c.stats = {};
}
int aigpu_dense_supported() { return Context::get().ok; }
void* aigpu_dense_create(const AiGpuDenseConfig* c, const AiGpuDenseLayer* layers,
    const float* const* keys, const float* const* values, uint32_t length) {
    auto& ctx = Context::get(); std::lock_guard<std::mutex> lock(ctx.gate);
    @autoreleasepool {
        try {
            if (!ctx.ok || !c || !layers || !keys || !values || !c->hidden || !c->layers ||
                !c->heads || !c->kvHeads || c->heads % c->kvHeads || c->headDim % 4 ||
                !c->headDim || !c->rotaryDims || c->rotaryDims % 2 || c->rotaryDims > c->headDim ||
                !c->intermediate || !c->capacity || !c->vocab || length > c->capacity || !c->finalNorm)
                throw std::invalid_argument("invalid resident decoder configuration");
            auto d = std::make_unique<DenseContext>(*c);
            d->length = length;
            for (uint32_t j = 0; j < c->rotaryDims / 2; ++j) {
                const float f = c->rotaryFrequencies ? c->rotaryFrequencies[j]
                    : 1.0f / std::pow(c->theta, float(2 * j) / float(c->rotaryDims));
                if (!std::isfinite(f) || f <= 0) throw std::invalid_argument("invalid rotary frequency");
                d->frequencies.push_back(f);
            }
            const auto floats = [&](uint64_t count, const float* data = nullptr, uint64_t copy = 0) {
                return d->buffer(product(count, 4), data, product(copy, 4));
            };
            const uint64_t qw = product(c->heads, c->headDim), kw = product(c->kvHeads, c->headDim);
            if (d->head.inDim != c->hidden || d->head.outDim != c->vocab)
                throw std::invalid_argument("resident output projection dimensions do not match config");
            if (c->head.bias) d->head.bias = floats(c->vocab, c->head.bias, c->vocab);
            d->x = floats(c->hidden); d->normed = floats(c->hidden);
            d->q = floats(qw); d->k = floats(kw); d->v = floats(kw); d->attn = floats(qw);
            d->proj = floats(c->hidden); d->gate = floats(c->intermediate); d->up = floats(c->intermediate);
            d->scores = floats(product(c->heads, c->capacity));
            d->partials = floats(product(product(c->heads, (uint64_t(c->capacity) + 127) / 128), c->headDim + 4)); d->trig = floats(c->rotaryDims);
            d->logits = floats(c->vocab); d->finalNorm = floats(c->hidden, c->finalNorm, c->hidden);
            for (uint32_t i = 0; i < c->layers; ++i) {
                const auto& l = layers[i];
                d->layers.emplace_back(l);
                auto& out = d->layers.back();
                out.window = l.window;
                const AiGpuDenseWeight* inputs[]{&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down};
                DenseWeight* outputs[]{&out.q, &out.k, &out.v, &out.o, &out.gate, &out.up, &out.down};
                for (unsigned j = 0; j < 7; ++j)
                    if (inputs[j]->bias) outputs[j]->bias = floats(inputs[j]->outDim, inputs[j]->bias, inputs[j]->outDim);
                const auto shape = [](const DenseWeight& w, uint64_t in, uint64_t out) {
                    return w.inDim == in && w.outDim == out;
                };
                if (!shape(out.q, c->hidden, qw) || !shape(out.k, c->hidden, kw) ||
                    !shape(out.v, c->hidden, kw) || !shape(out.o, qw, c->hidden) ||
                    !shape(out.gate, c->hidden, c->intermediate) || !shape(out.up, c->hidden, c->intermediate) ||
                    !shape(out.down, c->intermediate, c->hidden))
                    throw std::invalid_argument("resident layer dimensions do not match config");
                if (!l.inputNorm || !l.postNorm || !keys[i] || !values[i])
                    throw std::invalid_argument("missing resident decoder norms or cache");
                out.inputNorm = floats(c->hidden, l.inputNorm, c->hidden);
                out.postNorm = floats(c->hidden, l.postNorm, c->hidden);
                if (l.qNorm) out.qNorm = floats(c->headDim, l.qNorm, c->headDim);
                if (l.kNorm) out.kNorm = floats(c->headDim, l.kNorm, c->headDim);
                out.keys = floats(product(c->capacity, kw), keys[i], product(length, kw));
                out.values = floats(product(c->capacity, kw), values[i], product(length, kw));
            }
            ctx.error.clear();
            return d.release();
        } catch (const std::exception& e) { ctx.error = e.what(); return nullptr; }
    }
}
void aigpu_dense_release(void* handle) {
    auto& ctx = Context::get(); std::lock_guard<std::mutex> lock(ctx.gate);
    delete static_cast<DenseContext*>(handle);
}
void aigpu_dense_decode(void* handle, const float* embedding, float* logits, uint32_t position) {
    auto& ctx = Context::get(); std::lock_guard<std::mutex> lock(ctx.gate);
    @autoreleasepool {
        if (!handle) throw std::invalid_argument("null resident decoder context");
        try { denseDecode(*static_cast<DenseContext*>(handle), embedding, logits, position); }
        catch (const std::exception& e) { ctx.error = e.what(); throw; }
    }
}
void aigpu_dense_read_cache(void* handle, float* const* keys, float* const* values, uint32_t length) {
    auto& ctx = Context::get(); std::lock_guard<std::mutex> lock(ctx.gate);
    if (!handle || !keys || !values) throw std::invalid_argument("invalid resident cache download");
    auto& d = *static_cast<DenseContext*>(handle);
    if (length != d.length) throw std::invalid_argument("resident cache length mismatch");
    auto start = Clock::now();
    const uint64_t bytes = product(product(length, d.cfg.kvHeads), d.cfg.headDim) * 4;
    for (size_t i = 0; i < d.layers.size(); ++i) {
        if (!keys[i] || !values[i]) throw std::invalid_argument("null resident cache download target");
        std::memcpy(keys[i], d.layers[i].keys.contents, bytes);
        std::memcpy(values[i], d.layers[i].values.contents, bytes);
    }
    ctx.stats.deviceToHostBytes += bytes * d.layers.size() * 2;
    ctx.stats.transferSeconds += seconds(start);
}
}
