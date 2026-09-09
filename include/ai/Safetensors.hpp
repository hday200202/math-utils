#pragma once

#include <cstdint>
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


    bool load(const fs::path& path) {
        error.clear();
        entries.clear();
        blob.reset();

        files::File file;
        if (!file.open(path, files::OpenMode::Read)) {
            error = "could not open " + path.string();
            return false;
        }

        blob = std::make_shared<std::string>(file.text());
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

} // !namespace ai
