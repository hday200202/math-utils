#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

#include "../include/ImageIO.hpp"
#include "../include/NeuralNetwork.hpp"

using Vecf = math::dyn::Vec<float>;
using NN   = math::dyn::NeuralNetwork<float>;

static constexpr size_t HIDDEN   = 8;        // hidden units, so HIDDEN feature maps
static constexpr size_t PATTERN  = 512;      // size of the generated stand-in image
static constexpr size_t MAX_SIDE = 4096;     // default cap on the longest side
static const std::string OUT     = "out";


/*
    testPattern():
    - a stand-in image when none is given: a diagonal gradient, a soft disc and
      a checkerboard, so there are edges, curves and flat regions to react to
*/
static img::Image testPattern() {
    img::Image out(PATTERN, PATTERN);

    for (size_t y = 0; y < PATTERN; ++y) {
        for (size_t x = 0; x < PATTERN; ++x) {
            float fx = float(x) / PATTERN, fy = float(y) / PATTERN;

            float gradient = (fx + fy) * 0.5f;
            float disc     = ((fx-0.5f)*(fx-0.5f) + (fy-0.5f)*(fy-0.5f) < 0.05f) ? 0.6f : 0.f;
            float checker  = (((x / 32) + (y / 32)) % 2) ? 0.15f : 0.f;

            out.at(x, y) = std::clamp(gradient + disc + checker, 0.f, 1.f);
        }
    }

    return out;
}


/*
    forwardCapture(nn, input):
    - like nn.forward, but keeps every layer's activation rather than only the
      last. Applies relu/sigmoid itself, matching what the config asks for
*/
static std::vector<Vecf> forwardCapture(const NN& nn, const Vecf& input) {
    std::vector<Vecf> activations;
    Vecf a = input;

    for (size_t i = 0; i < nn.layers.size(); ++i) {
        Vecf z = nn.layers[i].forward(a);
        a = (i + 1 == nn.layers.size()) ? math::dyn::sigmoid(z) : math::dyn::relu(z);
        activations.push_back(a);
    }

    return activations;
}


/*
    patchPass(image):
    - runs one small network at every pixel, over that pixel's 3x3
      neighbourhood. Because it is applied everywhere, each hidden unit
      produces a whole full-resolution map instead of a single number, which is
      thing worth looking at. Random weights alone land on blurs, edge
      detectors and embossers, with no training at all
*/
static void patchPass(const img::Image& image) {
    constexpr size_t PATCH = 3;

    NN nn({ PATCH * PATCH, HIDDEN, 1 });
    math::dyn::NNConfig<float> cfg;
    cfg.initStddev = 0.8f;
    nn.init(cfg);

    const size_t W = image.width, H = image.height;

    std::vector<img::Image> maps(HIDDEN, img::Image(W, H));
    img::Image output(W, H);
    Vecf patch(PATCH * PATCH);

    for (size_t y = 0; y < H; ++y) {
        if (H > 2000 && y % 512 == 0)
            std::cout << "    row " << y << " / " << H << "\r" << std::flush;

        for (size_t x = 0; x < W; ++x) {
            size_t k = 0;
            for (long dy = -1; dy <= 1; ++dy)
                for (long dx = -1; dx <= 1; ++dx)
                    patch[k++] = image.sample(long(x) + dx, long(y) + dy);

            std::vector<Vecf> acts = forwardCapture(nn, patch);

            for (size_t u = 0; u < HIDDEN; ++u) maps[u].at(x, y) = acts[0][u];
            output.at(x, y) = acts[1][0];
        }
    }

    for (size_t u = 0; u < HIDDEN; ++u)
        img::saveBMP(OUT + "/hidden" + std::to_string(u) + ".bmp", maps[u]);
    img::saveBMP(OUT + "/output.bmp", output);

    // everything on one sheet, so there is a single file to open
    std::vector<img::Image> panels;
    panels.push_back(image);
    for (const auto& m : maps) panels.push_back(m);
    panels.push_back(output);

    img::saveBMP(OUT + "/layers.bmp", img::montage(panels, 5), false);

    std::cout << "  0 = input, 1.." << HIDDEN << " = hidden units, "
              << HIDDEN + 1 << " = output\n";
}


int main(int argc, char* argv[]) {
    std::filesystem::create_directories(OUT);

    /* input: any format, any size ========================================== */
    img::Image image;

    if (argc > 1) {
        image = img::load(argv[1]);
        if (image.empty()) {
            std::cout << "could not read " << argv[1] << "\n";
            return 1;
        }
        std::cout << "loaded " << argv[1] << "  " << image.width << "x" << image.height << "\n";
    } else {
        std::cout << "no image given, using a generated test pattern\n";
        image = testPattern();
    }

    // the network reads a 3x3 patch regardless of image size, so the image is
    // used at its own resolution. The only limit is memory: every hidden unit
    // holds a full-size float map, so the cost is (HIDDEN + 2) * W * H * 4 bytes
    const size_t cap = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : MAX_SIDE;
    const size_t longest = std::max(image.width, image.height);

    if (cap > 0 && longest > cap) {
        const double scale = double(cap) / double(longest);
        const size_t w = size_t(image.width * scale), h = size_t(image.height * scale);
        std::cout << "longest side over " << cap << ", resized to " << w << "x" << h
                  << "  (pass 0 to lift the cap)\n";
        image = img::resize(image, w, h);
    }

    const double mb = double(HIDDEN + 2) * image.width * image.height * 4.0 / (1024*1024);
    std::cout << "working at " << image.width << "x" << image.height
              << "  (" << std::fixed << std::setprecision(0) << mb << " MB of maps)\n";

    img::saveBMP(OUT + "/input.bmp", image, false);

    /* run ================================================================== */
    std::cout << "\n3x3 patches -> " << HIDDEN << " feature maps:\n";
    patchPass(image);

    std::cout << "\nopen " << OUT << "/layers.bmp\n";
    return 0;
}
