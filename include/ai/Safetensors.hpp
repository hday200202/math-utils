#pragma once

#include <bit>
#include <cstdint>
#include <fstream>
#include <functional>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Quant.hpp"
#include "Tensor.hpp"
#include "../files/File.hpp"
#include "../files/JSON.hpp"

/*
    Reader for the .safetensors weight format.

    The layout is deliberately trivial:
        [8 bytes]  little-endian uint64, the length of the JSON header
        [N bytes]  JSON: name -> {dtype, shape, data_offsets: [begin, end]}
        [rest]     raw tensor bytes, offsets relative to the end of the header

    The whole file is read once into a single buffer held by a shared_ptr, and
    an F32 tensor is then a VIEW aliasing into that buffer rather than a copy.
    Loading 124M parameters therefore costs one read and no per-tensor
    allocation. A tensor stored in a narrower dtype has to be widened, so that
    one does allocate.
*/

namespace ai {

namespace fs = std::filesystem;

/*
    LoadProgress:
    - called with the bytes read so far, the total expected, and what is
      currently being read. Reading tens of gigabytes takes long enough that
      silence is indistinguishable from a hang
*/
using LoadProgress = std::function<void(size_t done, size_t total, const std::string& stage)>;


/*
    WeightInitialiser:
    - fills one tensor with synthetic values instead of reading it.

    ModelWeights knows how to put a tensor where it belongs; only the
    architecture knows what a plausible value for it IS, so the policy is
    handed in rather than guessed at here.
*/
struct TensorInfo;
using WeightInitialiser =
    std::function<void(const std::string& name, const TensorInfo& info, float* out)>;


/* ========================================================================= //
    Header entries
// ========================================================================= */

struct TensorInfo {
    std::string         dtype;
    std::vector<size_t> shape;
    size_t              begin = 0;      // byte offsets into the payload,
    size_t              end   = 0;      // not into the file

    size_t numel() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }

    size_t byteCount() const { return end - begin; }
};


/* ========================================================================= //
    Safetensors
// ========================================================================= */

struct Safetensors {
    /*
        the entire file, kept alive for as long as any tensor views into it.
        A std::string rather than a vector<uint8_t> only because files::File
        hands back a string and this avoids a second copy of half a gigabyte
    */
    std::shared_ptr<std::string>                 blob;
    size_t                                       payloadBegin = 0;
    std::vector<std::pair<std::string, TensorInfo>> entries;   // in file order

    /*
        name -> slot in entries. A linear scan is fine for the 150-odd tensors
        a dense model has and quite untenable for the 76,000 a mixture of
        experts has: the loader alone looks every tensor up once, so a scan
        makes loading quadratic and a large checkpoint never finishes at all.
    */
    std::unordered_map<std::string, size_t>      index;

    files::json::Value                           metadata;
    std::string                                  error;


    bool load(const fs::path& path, const LoadProgress& onProgress = {},
              size_t baseDone = 0, size_t grandTotal = 0) {
        error.clear();
        entries.clear();
        index.clear();
        blob.reset();

        files::File file;
        if (!file.open(path, files::OpenMode::Read)) {
            error = "could not open " + path.string();
            return false;
        }

        const std::string label = path.filename().string();

        /*
            bad_alloc rather than a crash. A model larger than memory fails
            here, at a named allocation of a known size, which is a far more
            useful place to find out than std::terminate with no message
        */
        try {
            blob = std::make_shared<std::string>(
                file.text([&](size_t done, size_t total) {
                    if (onProgress)
                        onProgress(baseDone + done, grandTotal ? grandTotal : total, label);
                }));
        } catch (const std::bad_alloc&) {
            error = label + ": out of memory reading "
                  + std::to_string(file.size() / (1024 * 1024)) + " MB";
            return false;
        }
        if (blob->size() < 8) {
            error = "file is too small to hold a header length";
            return false;
        }

        uint64_t headerLength = 0;
        std::memcpy(&headerLength, blob->data(), 8);

        if (headerLength == 0 || headerLength + 8 > blob->size()) {
            error = "header length " + std::to_string(headerLength) + " is out of range";
            return false;
        }

        payloadBegin = 8 + static_cast<size_t>(headerLength);

        return parseEntries(std::string_view(blob->data() + 8,
                                             static_cast<size_t>(headerLength)),
                            blob->size() - payloadBegin);
    }


    /*
        loadHeader(path):
        - the 8-byte length and the JSON that follows it, and nothing else.
          The payload stays on disk to be read a tensor at a time.
        - this is what makes a checkpoint larger than memory loadable: the
          whole-file load below has to hold one entire shard while it works,
          which for a 151GB model is 3.7GB of headroom needed on top of the
          finished weights. Read tensor by tensor and the headroom is the
          largest single tensor instead.
    */
    bool loadHeader(const fs::path& path) {
        error.clear();
        entries.clear();
        index.clear();
        blob.reset();

        std::ifstream in(path, std::ios::binary);
        if (!in) {
            error = "could not open " + path.string();
            return false;
        }

        uint64_t headerLength = 0;
        in.read(reinterpret_cast<char*>(&headerLength), 8);
        if (!in || in.gcount() != 8) {
            error = "file is too small to hold a header length";
            return false;
        }

        std::error_code ec;
        const size_t total = static_cast<size_t>(fs::file_size(path, ec));
        if (ec || headerLength == 0 || headerLength + 8 > total) {
            error = "header length " + std::to_string(headerLength) + " is out of range";
            return false;
        }

        std::string header(static_cast<size_t>(headerLength), '\0');
        in.read(header.data(), static_cast<std::streamsize>(headerLength));
        if (static_cast<size_t>(in.gcount()) != headerLength) {
            error = "header is truncated";
            return false;
        }

        payloadBegin = 8 + static_cast<size_t>(headerLength);
        return parseEntries(header, total - payloadBegin);
    }


    /*
        parseEntries(header, payloadSize):
        - the tensor table, however the header bytes were obtained. Both the
          whole-file and the header-only path end up here so that the
          validation is written once and cannot drift between them
    */
    bool parseEntries(std::string_view header, size_t payloadSize) {
        const files::json::ParseResult parsed = files::json::parse(header);
        if (!parsed.ok) {
            error = "header JSON: " + parsed.error;
            return false;
        }

        const files::json::Value& root = parsed.value;
        if (!root.isObject()) {
            error = "header is not a JSON object";
            return false;
        }

        for (const std::string& name : root.keys()) {
            /* __metadata__ is a free-form string map the format allows
               alongside the tensors, not a tensor itself */
            if (name == "__metadata__") { metadata = root[name]; continue; }

            const files::json::Value& e = root[name];

            TensorInfo info;
            info.dtype = e["dtype"].asString();
            info.shape = e["shape"].asSizes();

            const std::vector<size_t> offsets = e["data_offsets"].asSizes();
            if (offsets.size() != 2) {
                error = name + ": data_offsets is not a pair";
                return false;
            }
            info.begin = offsets[0];
            info.end   = offsets[1];

            if (info.end < info.begin || info.end > payloadSize) {
                error = name + ": data_offsets run past the end of the payload";
                return false;
            }

            const size_t width = dtypeSize(info.dtype);
            if (width == 0) {
                error = name + ": unsupported dtype " + info.dtype;
                return false;
            }
            if (info.byteCount() != info.numel() * width) {
                error = name + ": byte range does not match shape and dtype";
                return false;
            }

            index.emplace(name, entries.size());
            entries.emplace_back(name, info);
        }

        return true;
    }


    static size_t dtypeSize(const std::string& dtype) {
        if (dtype == "F64")  return 8;
        if (dtype == "F32")  return 4;
        if (dtype == "F16")  return 2;
        if (dtype == "BF16") return 2;
        if (dtype == "I64")  return 8;
        if (dtype == "I32")  return 4;
        if (dtype == "I16")  return 2;
        if (dtype == "I8")   return 1;
        if (dtype == "U8")   return 1;
        if (dtype == "BOOL") return 1;
        return 0;
    }


    bool isLoaded() const { return blob != nullptr; }
    size_t count() const { return entries.size(); }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(entries.size());
        for (const auto& e : entries) out.push_back(e.first);
        return out;
    }

    const TensorInfo* info(const std::string& name) const {
        const auto it = index.find(name);
        return it == index.end() ? nullptr : &entries[it->second].second;
    }

    bool contains(const std::string& name) const { return info(name) != nullptr; }


    /*
        tensor(name):
        - an F32 tensor is returned as a view aliasing the file buffer, so no
          copy happens and the shared_ptr keeps the whole blob alive.
        - anything narrower has to be widened into its own allocation. BF16 is
          the top 16 bits of an F32, so widening it is a shift rather than a
          conversion; F16 needs the exponent rebiased and subnormals handled
        - returns an empty tensor when the name is missing, so a caller that
          forgot a weight gets a shape mismatch rather than reading garbage
    */
    Tensor<float> tensor(const std::string& name) const {
        const TensorInfo* meta = info(name);
        if (!meta || !blob) return Tensor<float>();

        const uint8_t* base =
            reinterpret_cast<const uint8_t*>(blob->data()) + payloadBegin + meta->begin;

        if (meta->dtype == "F32") {
            /* alignment is not promised by the format, only by every writer
               in practice. Fall back to a copy rather than risk it */
            if (reinterpret_cast<uintptr_t>(base) % alignof(float) == 0) {
                Tensor<float> out;
                out.shape   = meta->shape;
                out.strides = Tensor<float>::defaultStrides(out.shape);

                float* typed = reinterpret_cast<float*>(const_cast<uint8_t*>(base));
                out.storage  = std::shared_ptr<float[]>(blob, typed);
                out.ptr      = typed;
                return out;
            }

            return widenFrom(*meta, base);
        }

        return widenFrom(*meta, base);
    }


    /*
        widenFrom(info, base):
        - one tensor as float, converted out of whatever bytes are handed in.
          Taking a pointer rather than a name is what lets the streaming
          loader convert a tensor it read a moment ago and will discard next,
          without the whole shard having to be resident to do it.
    */
    static Tensor<float> widenFrom(const TensorInfo& info, const uint8_t* base) {
        Tensor<float> out(info.shape);
        const size_t n = info.numel();

        if (info.dtype == "F32") {
            std::memcpy(out.ptr, base, info.byteCount());
            return out;
        }

        if (info.dtype == "BF16") {
            for (size_t i = 0; i < n; ++i) {
                uint16_t half = 0;
                std::memcpy(&half, base + i * 2, 2);
                const uint32_t bits = static_cast<uint32_t>(half) << 16;
                std::memcpy(&out.ptr[i], &bits, 4);
            }
            return out;
        }

        if (info.dtype == "F16") {
            for (size_t i = 0; i < n; ++i) {
                uint16_t half = 0;
                std::memcpy(&half, base + i * 2, 2);
                out.ptr[i] = halfToFloat(half);
            }
            return out;
        }

        if (info.dtype == "F64") {
            for (size_t i = 0; i < n; ++i) {
                double wide = 0.0;
                std::memcpy(&wide, base + i * 8, 8);
                out.ptr[i] = static_cast<float>(wide);
            }
            return out;
        }

        // integer payloads are not weights; widen them so shapes still line up
        for (size_t i = 0; i < n; ++i) out.ptr[i] = 0.0f;
        return out;
    }


    /*
        rawBF16(name):
        - a bfloat16 tensor as a view over the file bytes, with no conversion
          and no copy. The kernels widen each value as they read it.
        - returns empty for anything not stored as BF16, so the caller falls
          back to the widening path rather than reinterpreting the wrong bits
    */
    Tensor<uint16_t> rawBF16(const std::string& name) const {
        const TensorInfo* meta = info(name);
        if (!meta || !blob || meta->dtype != "BF16") return Tensor<uint16_t>();

        const uint8_t* base =
            reinterpret_cast<const uint8_t*>(blob->data()) + payloadBegin + meta->begin;

        Tensor<uint16_t> out;
        out.shape   = meta->shape;
        out.strides = Tensor<uint16_t>::defaultStrides(out.shape);

        if (reinterpret_cast<uintptr_t>(base) % alignof(uint16_t) == 0) {
            uint16_t* typed = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(base));
            out.storage = std::shared_ptr<uint16_t[]>(blob, typed);
            out.ptr     = typed;
            return out;
        }

        Tensor<uint16_t> copy(meta->shape);
        std::memcpy(copy.ptr, base, meta->byteCount());
        return copy;
    }


    /*
        halfToFloat(half):
        - IEEE 754 binary16 to binary32. The exponent bias changes from 15 to
          127, and the two edge cases have to be handled separately: an
          exponent of 0 is a subnormal (or zero), an exponent of 31 is an
          infinity or a NaN
    */
    static float halfToFloat(uint16_t half) {
        const uint32_t sign     = static_cast<uint32_t>(half & 0x8000u) << 16;
        const uint32_t exponent = (half >> 10) & 0x1Fu;
        const uint32_t mantissa = half & 0x3FFu;

        uint32_t bits = 0;

        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                // renormalise: shift the mantissa up until its top bit falls out
                uint32_t e = 1;
                uint32_t m = mantissa;
                while ((m & 0x400u) == 0) { m <<= 1; ++e; }
                m &= 0x3FFu;
                bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
            }
        } else if (exponent == 31) {
            bits = sign | 0x7F800000u | (mantissa << 13);
        } else {
            bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
        }

        float out = 0.0f;
        std::memcpy(&out, &bits, 4);
        return out;
    }
};



/* ========================================================================= //
    ModelWeights

    A model past a couple of gigabytes is split across numbered shards, with
    a model.safetensors.index.json naming which shard holds each tensor. This
    presents the whole set as one lookup so nothing above it has to care
    whether the checkpoint arrived as one file or eighteen.
// ========================================================================= */

/* ========================================================================= //
    ShardReader

    One tensor's bytes at a time out of a shard whose payload was never read.
    The buffer is reused, so what has to be resident while a shard is being
    converted is the largest SINGLE TENSOR rather than the largest shard - on
    a 151GB mixture-of-experts checkpoint that is 0.6GB against 3.7GB, and
    that difference decides whether a model which only just fits can be
    loaded at all.

    Reads go through std::ifstream rather than files::File because the format
    wants a seek to an arbitrary offset, which File does not offer and which
    is no business of a general-purpose file class.
// ========================================================================= */

struct ShardReader {
    std::ifstream in;
    size_t        payloadBegin = 0;
    std::string   buffer;
    std::string   error;

    bool open(const fs::path& path, size_t begin) {
        error.clear();
        payloadBegin = begin;
        in.open(path, std::ios::binary);
        if (!in) { error = "could not open " + path.string(); return false; }
        return true;
    }

    /*
        read(info):
        - the tensor's bytes, valid until the next call. Chunked for the same
          reason File::text is: a single read of more than 2GB comes back
          having moved nothing and sets no error flag, so a large tensor read
          in one go silently yields zeros
    */
    const uint8_t* read(const TensorInfo& info) {
        const size_t want = info.byteCount();
        if (buffer.size() < want) buffer.resize(want);

        in.clear();
        in.seekg(static_cast<std::streamoff>(payloadBegin + info.begin), std::ios::beg);
        if (!in) { error = "could not seek to the tensor"; return nullptr; }

        constexpr size_t CHUNK = size_t(1) << 24;           // 16MB
        size_t got = 0;
        while (got < want) {
            const size_t take = std::min(CHUNK, want - got);
            in.read(buffer.data() + got, static_cast<std::streamsize>(take));
            const size_t moved = static_cast<size_t>(in.gcount());
            if (moved == 0) { error = "tensor payload is truncated"; return nullptr; }
            got += moved;
        }

        return reinterpret_cast<const uint8_t*>(buffer.data());
    }
};


struct ModelWeights {
    /*
        Slot:
        - where every form of one tensor lives. A name is hashed once and the
          answer says which shard it came from and which list, if any, holds
          the materialised copy - so a lookup costs the same whether the
          checkpoint has 150 tensors or 76,000.
    */
    static constexpr size_t NONE = static_cast<size_t>(-1);

    struct Slot {
        size_t shard    = 0;
        size_t tensorAt = NONE;         // slot in tensors
        size_t rawAt    = NONE;         // slot in raw
        size_t quantAt  = NONE;         // slot in quantized
    };

    std::vector<Safetensors>                    shards;
    std::vector<std::pair<std::string, size_t>> where;      // tensor -> shard
    std::unordered_map<std::string, Slot>       slots;

    /*
        Every tensor, already widened to float. Materialising up front and
        dropping each shard's raw bytes as soon as it has been converted is
        what keeps peak memory near the size of the finished weights rather
        than the size of the weights PLUS the whole file: a bf16 checkpoint
        doubles when widened, so holding both at once costs 50% more than
        necessary and is the difference between an 8B model fitting and not.
    */
    std::vector<std::pair<std::string, Tensor<float>>> tensors;

    /*
        When widen is false the file bytes are kept and nothing is converted:
        a bfloat16 weight is handed out as a Tensor<uint16_t> aliasing the
        blob, and the kernels widen it as they read. Resident memory is then
        the size of the file rather than twice it, and loading is just the
        read - no per-tensor pass at all.
    */
    bool                                        widen = true;
    std::vector<std::pair<std::string, Tensor<uint16_t>>> raw;

    /*
        Quantised weight matrices. A shard is quantised and its raw bytes
        released before the next is opened, so peak memory is the finished
        quantised model plus one shard - never the whole bfloat16 file. That
        property is the entire point: a model that only fits once quantised
        could not otherwise be loaded at all.
    */
    quant::Type                                 quantType = quant::Type::None;
    std::vector<std::pair<std::string, quant::QuantTensor>> quantized;

    /*
        Tensors whose names start with any of these are read past and never
        materialised. A Qwen3-Next checkpoint ships a multi-token-prediction
        head with its own 512 experts - roughly a gigabyte once quantised -
        that the ordinary forward pass never touches, and a model this size
        is being loaded precisely because the memory is tight.

        Set before load().
    */
    std::vector<std::string>                    skipPrefixes;

    /*
        When set, no payload is read at all: the headers give every tensor's
        name and shape and this supplies the values. Loading an 80B topology
        then costs 41 header reads rather than 151GB, which is what makes it
        usable for checking that a new architecture is plumbed together
        before committing to the download.
    */
    WeightInitialiser                           synthesise;

    /*
        perturb: the same signature, but the buffer arrives holding the REAL
        weights and this changes them. Replacing a checkpoint and nudging one
        are the same mechanics and completely different experiments - the
        first asks what the architecture does with no information at all, the
        second asks how much the trained weights can be disturbed before the
        model stops working.
    */
    WeightInitialiser                           perturb;

    /*
        How many values were offered to perturb and how many actually came
        out different. The gap is not a bug, it is the answer to a question
        worth asking: bfloat16 keeps 8 mantissa bits, so it can only
        represent steps of about 0.4% of a weight's own magnitude, and an
        offset smaller than that rounds straight back to where it started.
        Reporting it beats letting a whole experiment quietly do nothing.
    */
    size_t                                      perturbedValues = 0;
    size_t                                      changedValues = 0;

    std::string                                 error;


    bool skipped(const std::string& name) const {
        for (const std::string& p : skipPrefixes)
            if (name.compare(0, p.size(), p) == 0) return true;
        return false;
    }


    /*
        worthQuantizing(info):
        - the big 2-D matrices and nothing else. Norms and biases are vectors,
          they are a rounding error in the total, and quantising them costs
          accuracy for no saving. The row length has to divide into blocks
    */
    static bool endsWith(const std::string& s, const std::string& tail) {
        return s.size() >= tail.size()
            && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
    }

    static bool worthQuantizing(const std::string& name, const TensorInfo& info) {
        /*
            A router is the one big matrix that must stay exact. Rounding any
            other weight only nudges its output; rounding the router changes
            WHICH experts run, and swapping one expert for another is a
            discontinuous change rather than a small one. Routers are a tenth
            of a percent of the weights, so exactness there is free.
        */
        if (endsWith(name, ".mlp.gate.weight")) return false;

        return info.dtype == "BF16"
            && info.shape.size() == 2
            && info.numel() >= 65536
            && info.shape[1] % quant::BLOCK == 0;
    }

    bool load(const fs::path& directory, bool widenToFloat = true,
              const LoadProgress& onProgress = {},
              quant::Type quantise = quant::Type::None) {
        error.clear();
        shards.clear();
        where.clear();
        slots.clear();
        perturbedValues = 0;
        changedValues = 0;
        tensors.clear();
        raw.clear();
        quantized.clear();
        widen = widenToFloat;
        quantType = quantise;

        const fs::path index = directory / "model.safetensors.index.json";
        std::vector<std::string> files;

        if (fs::exists(index)) {
            const files::json::ParseResult parsed = files::json::parseFile(index);
            if (!parsed.ok) { error = "index: " + parsed.error; return false; }

            const files::json::Value& map = parsed.value["weight_map"];
            if (!map.isObject()) { error = "index: no weight_map"; return false; }

            // the map repeats a shard name once per tensor; collect uniques
            for (const files::json::Member& m : map.object) {
                const std::string file = m.second.asString();
                bool seen = false;
                for (const std::string& f : files) if (f == file) { seen = true; break; }
                if (!seen) files.push_back(file);
            }
        } else if (fs::exists(directory / "model.safetensors")) {
            files.push_back("model.safetensors");
        } else {
            error = "no model.safetensors or index in " + directory.string();
            return false;
        }

        // total up front so the caller can show one bar across every shard
        size_t grandTotal = 0;
        for (const std::string& f : files) {
            std::error_code ec;
            grandTotal += static_cast<size_t>(fs::file_size(directory / f, ec));
        }

        shards.resize(files.size());
        size_t done = 0;

        /*
            Quantising converts every tensor into memory it owns, so the file
            bytes are only ever scratch - which means they need never all be
            resident at once. Read the header, then one tensor at a time.

            The plain bfloat16 path cannot do this: its tensors are VIEWS
            onto the file bytes, so those bytes have to stay.
        */
        const bool stream = (quantType != quant::Type::None) || synthesise || perturb;

        for (size_t i = 0; i < files.size(); ++i) {
            std::error_code ec;
            const size_t shardBytes = static_cast<size_t>(fs::file_size(directory / files[i], ec));

            const bool opened = stream
                ? shards[i].loadHeader(directory / files[i])
                : shards[i].load(directory / files[i], onProgress, done, grandTotal);

            if (!opened) {
                error = files[i] + ": " + shards[i].error;
                return false;
            }

            ShardReader reader;
            if (stream && !synthesise
                && !reader.open(directory / files[i], shards[i].payloadBegin)) {
                error = files[i] + ": " + reader.error;
                return false;
            }
            done += shardBytes;

            size_t shardDone = 0;

            for (const auto& e : shards[i].entries) {
                if (skipped(e.first)) continue;

                where.emplace_back(e.first, i);

                Slot slot;
                slot.shard = i;

                if (synthesise || perturb) {
                    const uint8_t* source = nullptr;
                    if (perturb) {
                        source = reader.read(e.second);
                        if (!source) {
                            error = e.first + ": " + reader.error;
                            return false;
                        }
                    }

                    if (!fillAdjusted(e.first, e.second, slot, source)) return false;

                    shardDone += e.second.byteCount();
                    if (onProgress)
                        onProgress(done - shardBytes + shardDone,
                                   grandTotal ? grandTotal : shardBytes, files[i]);

                    slots.emplace(e.first, slot);
                    continue;
                }

                if (stream) {
                    const uint8_t* bytes = reader.read(e.second);
                    if (!bytes) {
                        error = e.first + ": " + reader.error;
                        return false;
                    }

                    /* the bar advances per tensor now rather than per chunk,
                       which is finer-grained and costs nothing */
                    shardDone += e.second.byteCount();
                    if (onProgress)
                        onProgress(done - shardBytes + shardDone,
                                   grandTotal ? grandTotal : shardBytes, files[i]);

                    if (worthQuantizing(e.first, e.second)) {
                        slot.quantAt = quantized.size();
                        quantized.emplace_back(e.first,
                            quant::quantize(reinterpret_cast<const uint16_t*>(bytes),
                                            e.second.shape[0], e.second.shape[1], quantType));
                    } else {
                        // everything else stays float: small, worth the bits
                        slot.tensorAt = tensors.size();
                        tensors.emplace_back(e.first,
                                             Safetensors::widenFrom(e.second, bytes));
                    }
                } else if (widen) {
                    slot.tensorAt = tensors.size();
                    tensors.emplace_back(e.first, shards[i].tensor(e.first));
                } else {
                    slot.rawAt = raw.size();
                    raw.emplace_back(e.first, shards[i].rawBF16(e.first));
                }

                slots.emplace(e.first, slot);
            }

            /*
                When widening, release this shard's raw bytes before reading
                the next one: every tensor now owns its own memory, so the
                file bytes are genuinely freed and peak stays near the size of
                the finished weights rather than the sum of both.

                When NOT widening the blob has to stay: the bfloat16 views
                alias it, and the small tensors that are still converted on
                demand read from it too.
            */
            /* the streaming path never held the bytes in the first place;
               the widening one owns every tensor by now, so they can go */
            if (widen) shards[i].blob.reset();
        }

        return !where.empty();
    }

    /*
        fillAdjusted(name, info, slot, source):
        - one tensor's values, decided in floats and then placed exactly
          where a plain read would have put them.

        With source null the buffer starts empty and `synthesise` supplies
        the numbers: nothing is read. With source given it starts as the real
        weights and `perturb` changes them.

        Either way the result is narrowed back to bfloat16 before it is
        stored or quantised, because that is the precision the checkpoint
        itself has. It also means a perturbation smaller than bfloat16 can
        represent disappears, which is why the values that really changed
        are counted rather than assumed.
    */
    bool fillAdjusted(const std::string& name, const TensorInfo& info, Slot& slot,
                      const uint8_t* source) {
        const size_t n = info.numel();

        std::vector<float> values;
        try {
            values.resize(n);
        } catch (const std::bad_alloc&) {
            error = name + ": out of memory holding "
                  + std::to_string(n / (1024 * 1024)) + "M values";
            return false;
        }

        if (source) {
            Tensor<float> real = Safetensors::widenFrom(info, source);
            std::copy(real.ptr, real.ptr + n, values.data());
        }

        /* narrowing the original first means the comparison below is
           bfloat16 against bfloat16, and counts only the changes that
           actually survive into the stored weights */
        std::vector<uint16_t> before;
        if (source) {
            before.resize(n);
            for (size_t i = 0; i < n; ++i)
                before[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(values[i]) >> 16);
        }

        if (source) perturb(name, info, values.data());
        else        synthesise(name, info, values.data());

        /*
            Narrow in place. A uint16 is half the width of a float, so
            writing element i never reaches bytes that element i or later
            still has to be read from
        */
        uint16_t* narrowed = reinterpret_cast<uint16_t*>(values.data());
        for (size_t i = 0; i < n; ++i)
            narrowed[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(values[i]) >> 16);

        if (source) {
            perturbedValues += n;
            for (size_t i = 0; i < n; ++i)
                if (before[i] != narrowed[i]) ++changedValues;
        }

        if (quantType != quant::Type::None && worthQuantizing(name, info)) {
            slot.quantAt = quantized.size();
            quantized.emplace_back(name, quant::quantize(narrowed, info.shape[0],
                                                         info.shape[1], quantType));
            return true;
        }

        if (quantType == quant::Type::None && info.shape.size() == 2) {
            // matrices stay bfloat16, exactly as a real checkpoint's would
            Tensor<uint16_t> out(info.shape);
            std::copy(narrowed, narrowed + n, out.ptr);
            slot.rawAt = raw.size();
            raw.emplace_back(name, std::move(out));
            return true;
        }

        Tensor<float> out(info.shape);
        for (size_t i = 0; i < n; ++i)
            out.ptr[i] = std::bit_cast<float>(static_cast<uint32_t>(narrowed[i]) << 16);

        slot.tensorAt = tensors.size();
        tensors.emplace_back(name, std::move(out));
        return true;
    }


    size_t shardCount() const { return shards.size(); }
    size_t count() const { return where.size(); }

    const Slot* slot(const std::string& name) const {
        const auto it = slots.find(name);
        return it == slots.end() ? nullptr : &it->second;
    }

    bool contains(const std::string& name) const { return slots.count(name) != 0; }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(where.size());
        for (const auto& w : where) out.push_back(w.first);
        return out;
    }

    Tensor<float> tensor(const std::string& name) const {
        const Slot* s = slot(name);
        if (!s) return Tensor<float>();
        if (s->tensorAt != NONE) return tensors[s->tensorAt].second;

        // not widened at load time, so convert this one on demand
        return shards[s->shard].tensor(name);
    }

    Tensor<uint16_t> rawBF16(const std::string& name) const {
        const Slot* s = slot(name);
        return (s && s->rawAt != NONE) ? raw[s->rawAt].second : Tensor<uint16_t>();
    }

    quant::QuantTensor quantTensor(const std::string& name) const {
        const Slot* s = slot(name);
        return (s && s->quantAt != NONE) ? quantized[s->quantAt].second
                                         : quant::QuantTensor();
    }

    const TensorInfo* info(const std::string& name) const {
        const Slot* s = slot(name);
        return s ? shards[s->shard].info(name) : nullptr;
    }

    /*
        Both counts come from the header rather than from whichever list
        happens to be populated. An F32 checkpoint loaded with widen=false
        fills neither: its tensors are handed out as aliasing views on
        demand, so counting the lists would report zero for a model that is
        very much resident.
    */
    size_t parameterCount() const {
        size_t total = 0;
        for (const auto& w : where) {
            const TensorInfo* i = shards[w.second].info(w.first);
            if (i) total += i->numel();
        }
        return total;
    }

    /* the weights actually resident, which is what matters, not the file
       size on disk */
    size_t residentBytes() const {
        size_t total = 0;
        for (const auto& t : quantized) total += t.second.byteCount();
        for (const auto& t : raw)       total += t.second.numel() * sizeof(uint16_t);
        for (const auto& t : tensors)   total += t.second.numel() * sizeof(float);

        if (total) return total;

        // nothing materialised: F32 views handed out on demand
        for (const auto& w : where) {
            const TensorInfo* i = shards[w.second].info(w.first);
            if (i) total += widen ? i->numel() * sizeof(float) : i->byteCount();
        }
        return total;
    }


    /*
        estimateResident(directory, type):
        - what the weights will occupy once loaded, from the file sizes alone.
          Needed before anything is read, so the caller can refuse a model
          that will not fit rather than discovering it halfway through
    */
    static size_t estimateResident(const fs::path& directory, quant::Type type) {
        const size_t onDisk = weightBytesOnDisk(directory);
        return static_cast<size_t>(double(onDisk) * quant::bytesPerWeight(type) / 2.0);
    }

    /* the largest single shard, which is the transient cost during load */
    /*
        shardFiles(directory):
        - the shard names in order, from the index if there is one and the
          single file if there is not. Everything that wants to size a
          checkpoint up front needs this, so it is written once
    */
    static std::vector<std::string> shardFiles(const fs::path& directory) {
        std::error_code ec;
        std::vector<std::string> files;

        const fs::path index = directory / "model.safetensors.index.json";
        if (fs::exists(index, ec)) {
            const files::json::ParseResult parsed = files::json::parseFile(index);
            const files::json::Value& map = parsed.value["weight_map"];

            for (const files::json::Member& m : map.object) {
                const std::string file = m.second.asString();
                bool already = false;
                for (const std::string& f : files) if (f == file) { already = true; break; }
                if (!already) files.push_back(file);
            }
            return files;
        }

        if (fs::exists(directory / "model.safetensors", ec))
            files.push_back("model.safetensors");
        return files;
    }


    /*
        Plan:
        - what a load will actually cost, worked out from the headers before
          a byte of payload is touched.

        Scaling the on-disk total by a bytes-per-weight ratio is close but
        not close enough to refuse a model on: it credits neither the
        branches that are skipped nor the routers deliberately left exact,
        and on an 80B checkpoint that is most of a gigabyte of slack in a
        decision with only a few to spare. Reading 41 headers costs
        milliseconds and the answer is exact.
    */
    struct Plan {
        size_t      onDisk = 0;
        size_t      resident = 0;
        size_t      largestTensor = 0;     // the high-water mark while loading
        size_t      largestSynthetic = 0;  // and when the values are made up
        size_t      tensorCount = 0;
        bool        ok = false;
        std::string error;

        size_t peak() const { return resident + largestTensor; }

        /* synthesising works in floats before narrowing, so the one tensor
           in flight is twice the width it would be read at */
        size_t syntheticPeak() const { return resident + largestSynthetic; }
    };

    static Plan plan(const fs::path& directory, quant::Type type,
                     const std::vector<std::string>& skipPrefixes = {}) {
        Plan out;
        const std::vector<std::string> files = shardFiles(directory);
        if (files.empty()) {
            out.error = "no model.safetensors or index in " + directory.string();
            return out;
        }

        const bool streaming = (type != quant::Type::None);

        for (const std::string& file : files) {
            std::error_code ec;
            out.onDisk += static_cast<size_t>(fs::file_size(directory / file, ec));

            Safetensors shard;
            if (!shard.loadHeader(directory / file)) {
                out.error = file + ": " + shard.error;
                return out;
            }

            for (const auto& e : shard.entries) {
                bool skip = false;
                for (const std::string& prefix : skipPrefixes)
                    if (e.first.compare(0, prefix.size(), prefix) == 0) { skip = true; break; }
                if (skip) continue;

                ++out.tensorCount;

                if (worthQuantizing(e.first, e.second)) {
                    quant::QuantTensor shape;
                    shape.type   = type;
                    shape.outDim = e.second.shape[0];
                    shape.inDim  = e.second.shape[1];
                    out.resident += shape.byteCount();
                } else if (streaming) {
                    out.resident += e.second.numel() * sizeof(float);
                } else {
                    out.resident += e.second.byteCount();
                }

                /*
                    Streaming holds exactly one tensor beyond the finished
                    weights, so the high-water mark is the largest of them.
                    The bfloat16 path has no transient at all: the file bytes
                    ARE the weights, aliased rather than converted
                */
                if (streaming)
                    out.largestTensor = std::max(out.largestTensor, e.second.byteCount());

                out.largestSynthetic = std::max(out.largestSynthetic,
                                                e.second.numel() * sizeof(float));
            }
        }

        out.ok = true;
        return out;
    }


    static size_t largestShardBytes(const fs::path& directory) {
        std::error_code ec;
        size_t largest = 0;

        const fs::path index = directory / "model.safetensors.index.json";
        if (fs::exists(index, ec)) {
            const files::json::ParseResult parsed = files::json::parseFile(index);
            const files::json::Value& map = parsed.value["weight_map"];

            std::vector<std::string> seen;
            for (const files::json::Member& m : map.object) {
                const std::string file = m.second.asString();
                bool already = false;
                for (const std::string& f : seen) if (f == file) { already = true; break; }
                if (already) continue;
                seen.push_back(file);
                largest = std::max(largest,
                    static_cast<size_t>(fs::file_size(directory / file, ec)));
            }
            return largest;
        }

        return static_cast<size_t>(fs::file_size(directory / "model.safetensors", ec));
    }

    /* the bytes on disk, which is what has to be read and, for a bfloat16
       checkpoint kept as-is, roughly what ends up resident */
    static size_t weightBytesOnDisk(const fs::path& directory) {
        std::error_code ec;
        size_t total = 0;

        const fs::path index = directory / "model.safetensors.index.json";
        if (fs::exists(index, ec)) {
            const files::json::ParseResult parsed = files::json::parseFile(index);
            const files::json::Value& map = parsed.value["weight_map"];

            std::vector<std::string> seen;
            for (const files::json::Member& m : map.object) {
                const std::string file = m.second.asString();
                bool already = false;
                for (const std::string& f : seen) if (f == file) { already = true; break; }
                if (already) continue;
                seen.push_back(file);
                total += static_cast<size_t>(fs::file_size(directory / file, ec));
            }
            return total;
        }

        return static_cast<size_t>(fs::file_size(directory / "model.safetensors", ec));
    }

    size_t byteSize() const {
        size_t total = 0;
        for (const Safetensors& s : shards) total += s.blob ? s.blob->size() : 0;
        return total;
    }
};

} // !namespace ai
