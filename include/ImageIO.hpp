#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>

/*
    Image loading, saving, resizing and contact sheets, with no third-party
    code and nothing to install.

    Reading: Netpbm (P5/P6) and 24/32-bit uncompressed BMP are parsed directly.
    Anything else -- png, jpeg, heic, tiff, gif -- is handed to whichever
    converter the machine already has (sips on macOS, magick/convert or ffmpeg
    elsewhere) and re-read as BMP. So load() takes essentially any image.

    Writing: BMP, because it opens in every image viewer without a compression
    implementation. PGM is still available when a raw dump is wanted.

    Everything is greyscale floats internally; that is all the network sees.
*/

namespace img {

/* ========================================================================= //
    Image
// ========================================================================= */

struct Image {
    size_t width  = 0;
    size_t height = 0;
    std::vector<float> pixels;      // greyscale, nominally 0..1, row-major

    Image() = default;
    Image(size_t w, size_t h) : width(w), height(h), pixels(w * h, 0.f) {}

    size_t size() const { return pixels.size(); }
    bool empty() const { return pixels.empty(); }

    float& at(size_t x, size_t y) { return pixels[y * width + x]; }
    const float& at(size_t x, size_t y) const { return pixels[y * width + x]; }

    // clamped lookup, so filters can walk off the edge without branching
    float sample(long x, long y) const {
        x = std::clamp<long>(x, 0, static_cast<long>(width)  - 1);
        y = std::clamp<long>(y, 0, static_cast<long>(height) - 1);
        return pixels[static_cast<size_t>(y) * width + static_cast<size_t>(x)];
    }
};


inline float luma(float r, float g, float b) { return 0.299f*r + 0.587f*g + 0.114f*b; }


/* ========================================================================= //
    Readers
// ========================================================================= */

/*
    loadNetpbm(path):
    - binary PGM (P5) and PPM (P6). Returns an empty Image on anything else
*/
inline Image loadNetpbm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};

    auto token = [&]() -> std::string {
        std::string t;
        while (in >> t) {
            if (t.empty()) continue;
            if (t[0] == '#') { std::string skip; std::getline(in, skip); continue; }
            return t;
        }
        return {};
    };

    std::string magic = token();
    if (magic != "P5" && magic != "P6") return {};

    size_t w = 0, h = 0, maxval = 0;
    try {
        w = std::stoul(token());
        h = std::stoul(token());
        maxval = std::stoul(token());
    } catch (...) { return {}; }

    if (w == 0 || h == 0 || maxval == 0 || maxval > 255) return {};
    in.get();                                   // the single whitespace before the data

    const size_t ch = (magic == "P6") ? 3 : 1;
    std::vector<uint8_t> raw(w * h * ch);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (static_cast<size_t>(in.gcount()) != raw.size()) return {};

    Image out(w, h);
    for (size_t i = 0; i < w * h; ++i) {
        float v = (ch == 3) ? luma(raw[i*3], raw[i*3+1], raw[i*3+2]) : float(raw[i]);
        out.pixels[i] = v / float(maxval);
    }

    return out;
}


/*
    loadBMP(path):
    - 24 and 32 bit uncompressed BMP. Handles both row orders: a negative
      height means the rows are stored top-down, which is what sips emits
*/
inline Image loadBMP(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};

    std::vector<char> f((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (f.size() < 54 || f[0] != 'B' || f[1] != 'M') return {};

    auto u32 = [&](size_t o) { return uint32_t(uint8_t(f[o])) | (uint32_t(uint8_t(f[o+1]))<<8)
                                    | (uint32_t(uint8_t(f[o+2]))<<16) | (uint32_t(uint8_t(f[o+3]))<<24); };
    auto u16 = [&](size_t o) { return uint16_t(uint8_t(f[o]) | (uint8_t(f[o+1])<<8)); };

    uint32_t dataOffset = u32(10);
    int32_t  w          = int32_t(u32(18));
    int32_t  h          = int32_t(u32(22));
    uint16_t bpp        = u16(28);
    uint32_t comp       = u32(30);

    // 0 = plain BGR(X), 3 = BI_BITFIELDS, which is still uncompressed but
    // declares its own channel masks. sips emits the latter for anything with
    // an alpha channel, so both have to be handled
    if (w <= 0 || h == 0 || (comp != 0 && comp != 3) || (bpp != 24 && bpp != 32)) return {};

    uint32_t maskB = 0x000000FFu, maskG = 0x0000FF00u, maskR = 0x00FF0000u;
    if (comp == 3) {
        if (f.size() < 66) return {};
        maskR = u32(54); maskG = u32(58); maskB = u32(62);
        if (!maskR || !maskG || !maskB) { maskB = 0x000000FFu; maskG = 0x0000FF00u; maskR = 0x00FF0000u; }
    }

    // pulls one channel out and rescales it to 0..1, whatever width the mask is
    auto channel = [](uint32_t pixel, uint32_t mask) -> float {
        uint32_t v = pixel & mask;
        while (!(mask & 1u)) { mask >>= 1; v >>= 1; }
        return float(v) / float(mask);
    };

    const bool topDown = (h < 0);
    const size_t H = size_t(std::abs(h)), W = size_t(w);
    const size_t bytesPerPixel = bpp / 8;
    const size_t stride = ((W * bytesPerPixel + 3) / 4) * 4;      // rows pad to 4 bytes

    if (f.size() < dataOffset + stride * H) return {};

    Image out(W, H);
    for (size_t row = 0; row < H; ++row) {
        const size_t src = dataOffset + (topDown ? row : (H - 1 - row)) * stride;

        for (size_t x = 0; x < W; ++x) {
            const size_t p = src + x * bytesPerPixel;
            uint32_t pixel = uint32_t(uint8_t(f[p])) | (uint32_t(uint8_t(f[p+1])) << 8)
                           | (uint32_t(uint8_t(f[p+2])) << 16);
            if (bytesPerPixel == 4) pixel |= uint32_t(uint8_t(f[p+3])) << 24;

            out.at(x, row) = luma(channel(pixel, maskR), channel(pixel, maskG), channel(pixel, maskB));
        }
    }

    return out;
}


/*
    convertWith(tool, in, out):
    - runs one external converter, returning true only if it produced a file
*/
inline bool convertWith(const std::string& cmd) {
    return std::system((cmd + " >/dev/null 2>&1").c_str()) == 0;
}


/*
    load(path):
    - reads any image the machine can open. Netpbm and BMP are parsed here;
      everything else is converted to a temporary BMP first, by whichever of
      sips / magick / convert / ffmpeg exists
*/
inline Image load(const std::string& path) {
    if (!std::filesystem::exists(path)) return {};

    if (Image direct = loadNetpbm(path); !direct.empty()) return direct;
    if (Image direct = loadBMP(path);    !direct.empty()) return direct;

    const std::string tmp = (std::filesystem::temp_directory_path() / "mathutils-load.bmp").string();
    std::filesystem::remove(tmp);

    const std::string q = "\"" + path + "\"", t = "\"" + tmp + "\"";
    const std::string tools[] = {
        "sips -s format bmp " + q + " --out " + t,
        "magick " + q + " " + t,
        "convert " + q + " " + t,
        "ffmpeg -y -i " + q + " " + t
    };

    for (const auto& cmd : tools) {
        if (convertWith(cmd) && std::filesystem::exists(tmp)) {
            if (Image converted = loadBMP(tmp); !converted.empty()) return converted;
        }
    }

    return {};
}


/* ========================================================================= //
    Writers
// ========================================================================= */

/*
    normalized(image):
    - rescales so the image's own min is black and its max is white. Activation
      maps are rarely inside 0..1, so without this most of them come out flat
*/
inline std::vector<uint8_t> normalized(const Image& image, bool rescale) {
    float lo = 0.f, hi = 1.f;

    if (rescale && !image.empty()) {
        lo = hi = image.pixels[0];
        for (float v : image.pixels) { lo = std::min(lo, v); hi = std::max(hi, v); }
        if (hi - lo < 1e-12f) { lo = 0.f; hi = 1.f; }
    }

    std::vector<uint8_t> out(image.size());
    for (size_t i = 0; i < image.size(); ++i)
        out[i] = uint8_t(std::clamp((image.pixels[i] - lo) / (hi - lo), 0.f, 1.f) * 255.f + 0.5f);

    return out;
}


/*
    saveBMP(path, image, rescale):
    - 24-bit uncompressed BMP, which every image viewer opens
*/
inline bool saveBMP(const std::string& path, const Image& image, bool rescale = true) {
    if (image.empty()) return false;

    const size_t W = image.width, H = image.height;
    const size_t stride = ((W * 3 + 3) / 4) * 4;
    const size_t dataSize = stride * H;
    const size_t fileSize = 54 + dataSize;

    std::vector<uint8_t> grey = normalized(image, rescale);
    std::vector<uint8_t> buf(fileSize, 0);

    auto put32 = [&](size_t o, uint32_t v) {
        buf[o] = v & 0xFF; buf[o+1] = (v>>8) & 0xFF; buf[o+2] = (v>>16) & 0xFF; buf[o+3] = (v>>24) & 0xFF;
    };
    auto put16 = [&](size_t o, uint16_t v) { buf[o] = v & 0xFF; buf[o+1] = (v>>8) & 0xFF; };

    buf[0] = 'B'; buf[1] = 'M';
    put32(2, uint32_t(fileSize));
    put32(10, 54);                      // pixel data offset
    put32(14, 40);                      // BITMAPINFOHEADER
    put32(18, uint32_t(W));
    put32(22, uint32_t(H));             // positive: rows stored bottom-up
    put16(26, 1);
    put16(28, 24);
    put32(34, uint32_t(dataSize));

    for (size_t row = 0; row < H; ++row) {
        uint8_t* dst = buf.data() + 54 + (H - 1 - row) * stride;
        for (size_t x = 0; x < W; ++x) {
            uint8_t v = grey[row * W + x];
            dst[x*3 + 0] = v; dst[x*3 + 1] = v; dst[x*3 + 2] = v;
        }
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return static_cast<bool>(out);
}


/*
    savePGM(path, image, rescale):
    - the raw greyscale dump, kept for when a Netpbm file is wanted
*/
inline bool savePGM(const std::string& path, const Image& image, bool rescale = true) {
    if (image.empty()) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    std::vector<uint8_t> grey = normalized(image, rescale);
    out << "P5\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(grey.data()), static_cast<std::streamsize>(grey.size()));
    return static_cast<bool>(out);
}


/* ========================================================================= //
    Resampling and contact sheets
// ========================================================================= */

/*
    resize(image, w, h):
    - bilinear. Lets the caller take an image of any size and hand the network
      the fixed shape it expects
*/
inline Image resize(const Image& src, size_t w, size_t h) {
    if (src.empty() || w == 0 || h == 0) return {};
    if (src.width == w && src.height == h) return src;

    Image out(w, h);
    const float sx = float(src.width)  / float(w);
    const float sy = float(src.height) / float(h);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float fx = (float(x) + 0.5f) * sx - 0.5f;
            float fy = (float(y) + 0.5f) * sy - 0.5f;

            long x0 = long(std::floor(fx)), y0 = long(std::floor(fy));
            float tx = fx - float(x0), ty = fy - float(y0);

            float top    = src.sample(x0, y0)     * (1-tx) + src.sample(x0+1, y0)     * tx;
            float bottom = src.sample(x0, y0 + 1) * (1-tx) + src.sample(x0+1, y0 + 1) * tx;

            out.at(x, y) = top * (1-ty) + bottom * ty;
        }
    }

    return out;
}


/* a 3x5 bitmap digit font, so montage panels can label themselves */
inline constexpr uint8_t DIGIT_FONT[10][5] = {
    {7,5,5,5,7}, {2,6,2,2,7}, {7,1,7,4,7}, {7,1,7,1,7}, {5,5,7,1,1},
    {7,4,7,1,7}, {7,4,7,5,7}, {7,1,1,1,1}, {7,5,7,5,7}, {7,5,7,1,7}
};

inline void drawNumber(Image& canvas, size_t number, size_t x, size_t y, size_t scale, float value) {
    std::string text = std::to_string(number);

    for (size_t c = 0; c < text.size(); ++c) {
        const uint8_t* glyph = DIGIT_FONT[text[c] - '0'];

        for (size_t row = 0; row < 5; ++row)
            for (size_t col = 0; col < 3; ++col)
                if (glyph[row] & (1 << (2 - col)))
                    for (size_t dy = 0; dy < scale; ++dy)
                        for (size_t dx = 0; dx < scale; ++dx) {
                            size_t px = x + (c * 4 + col) * scale + dx;
                            size_t py = y + row * scale + dy;
                            if (px < canvas.width && py < canvas.height) canvas.at(px, py) = value;
                        }
    }
}


/*
    montage(panels, cols, cell):
    - lays every panel out on one image, each scaled to cell x cell and
      numbered in its top-left corner, so the whole run can be looked at in a
      single file instead of opening a directory
*/
inline Image montage(const std::vector<Image>& panels, size_t cols, size_t cell = 192, size_t pad = 6) {
    if (panels.empty() || cols == 0) return {};

    const size_t rows = (panels.size() + cols - 1) / cols;
    const size_t W = cols * (cell + pad) + pad;
    const size_t H = rows * (cell + pad) + pad;

    Image sheet(W, H);
    for (float& v : sheet.pixels) v = 1.f;                 // white background

    for (size_t i = 0; i < panels.size(); ++i) {
        // each panel is normalized on its own, so a dim map is still readable
        std::vector<uint8_t> grey = normalized(panels[i], true);
        Image scaled(panels[i].width, panels[i].height);
        for (size_t k = 0; k < scaled.size(); ++k) scaled.pixels[k] = grey[k] / 255.f;
        scaled = resize(scaled, cell, cell);

        const size_t ox = pad + (i % cols) * (cell + pad);
        const size_t oy = pad + (i / cols) * (cell + pad);

        for (size_t y = 0; y < cell; ++y)
            for (size_t x = 0; x < cell; ++x)
                sheet.at(ox + x, oy + y) = scaled.at(x, y);

        drawNumber(sheet, i, ox + 4, oy + 4, 3, 1.f);      // white, with a dark shadow under it
        drawNumber(sheet, i, ox + 3, oy + 3, 3, 0.f);
    }

    return sheet;
}

} // !namespace img
