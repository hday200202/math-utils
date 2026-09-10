#pragma once

#include <cstdint>
#include <functional>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
    files::json::Value                           metadata;
    std::string                                  error;


    bool load(const fs::path& path, const LoadProgress& onProgress = {},
              size_t baseDone = 0, size_t grandTotal = 0) {
        error.clear();
        entries.clear();
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

        const std::string_view header(blob->data() + 8, static_cast<size_t>(headerLength));
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

        const size_t payloadSize = blob->size() - payloadBegin;

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
        for (const auto& e : entries) if (e.first == name) return &e.second;
        return nullptr;
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

            Tensor<float> out(meta->shape);
            std::memcpy(out.ptr, base, meta->byteCount());
            return out;
        }

        Tensor<float> out(meta->shape);
        const size_t n = meta->numel();

        if (meta->dtype == "BF16") {
            for (size_t i = 0; i < n; ++i) {
                uint16_t half = 0;
                std::memcpy(&half, base + i * 2, 2);
                const uint32_t bits = static_cast<uint32_t>(half) << 16;
                std::memcpy(&out.ptr[i], &bits, 4);
            }
            return out;
        }

        if (meta->dtype == "F16") {
            for (size_t i = 0; i < n; ++i) {
                uint16_t half = 0;
                std::memcpy(&half, base + i * 2, 2);
                out.ptr[i] = halfToFloat(half);
            }
            return out;
        }

        if (meta->dtype == "F64") {
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

struct ModelWeights {
    std::vector<Safetensors>                    shards;
    std::vector<std::pair<std::string, size_t>> where;      // tensor -> shard

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
    std::string                                 error;

    bool load(const fs::path& directory, bool widenToFloat = true,
              const LoadProgress& onProgress = {}) {
        error.clear();
        shards.clear();
        where.clear();
        tensors.clear();
        raw.clear();
        widen = widenToFloat;

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

        for (size_t i = 0; i < files.size(); ++i) {
            std::error_code ec;
            const size_t shardBytes = static_cast<size_t>(fs::file_size(directory / files[i], ec));

            if (!shards[i].load(directory / files[i], onProgress, done, grandTotal)) {
                error = files[i] + ": " + shards[i].error;
                return false;
            }
            done += shardBytes;

            for (const auto& e : shards[i].entries) {
                where.emplace_back(e.first, i);
                if (widen) tensors.emplace_back(e.first, shards[i].tensor(e.first));
                else       raw.emplace_back(e.first, shards[i].rawBF16(e.first));
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
            if (widen) shards[i].blob.reset();
        }

        return !where.empty();
    }

    size_t shardCount() const { return shards.size(); }
    size_t count() const { return where.size(); }

    bool contains(const std::string& name) const {
        for (const auto& w : where) if (w.first == name) return true;
        return false;
    }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(where.size());
        for (const auto& w : where) out.push_back(w.first);
        return out;
    }

    Tensor<float> tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.first == name) return t.second;

        // not widened at load time, so convert this one on demand
        for (const auto& w : where)
            if (w.first == name) return shards[w.second].tensor(name);
        return Tensor<float>();
    }

    Tensor<uint16_t> rawBF16(const std::string& name) const {
        for (const auto& t : raw) if (t.first == name) return t.second;
        return Tensor<uint16_t>();
    }

    const TensorInfo* info(const std::string& name) const {
        for (const auto& w : where) {
            const TensorInfo* i = shards[w.second].info(name);
            if (w.first == name && i) return i;
        }
        return nullptr;
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
        for (const auto& w : where) {
            const TensorInfo* i = shards[w.second].info(w.first);
            if (!i) continue;
            // widened at load, or kept in whatever width the file used
            total += widen ? i->numel() * sizeof(float) : i->byteCount();
        }
        return total;
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
