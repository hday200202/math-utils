# Keep the default build usable from terminals, IDEs and CI alike.
.DEFAULT_GOAL := all
CXXSTD ?= c++26
PYTHON ?= python3
ifeq ($(shell uname -s),Darwin)
    ifeq ($(origin CXX),default)
        CXX := clang++
    endif
    # Vec3/Vec4 expose anonymous-union swizzles and intentionally support
    # flat aggregate initialization. Clang asks for redundant nested braces.
    ifneq ($(findstring clang,$(shell $(CXX) --version 2>/dev/null)),)
        MAC_WARN_FLAGS := -Wno-missing-braces
    endif
endif

.PHONY: all test test-unit gpu-test

# ========================================================================= #
#   GPU detection
#
#   There is one llm binary, not two. include/ai/gpu/Device.hpp is plain
#   C++ with no HIP types in it AND entirely inline, so every translation
#   unit compiles the same way everywhere and only the BACKEND differs at
#   link time:
#
#       macOS + Metal  src/gpu/Metal.mm and offline Kernels.metal
#       ROCm present   src/gpu/Kernels.hip through hipcc, linked in
#       ROCm absent    src/gpu/Stub.cpp, where available() is false
#
#   The stub is not a stripped-down build. --gpu still parses, still asks,
#   and is told no - which is the same path taken when a card is present
#   but busy, so there is one behaviour to reason about rather than two.
#
#       make release-llm            use the GPU if this machine has one
#       make release-llm GPU=0      no-device build even where ROCm exists
#       make release-llm GPU_ARCH=gfx1030   a card that is not RDNA3
#
#   LINUX AND WINDOWS DIFFER IN ONE PLACE
#
#   Linux links the HIP object straight into the executable: one GCC builds
#   both sides, so there is nothing to reconcile.
#
#   Windows cannot. hipcc there emits host code against the MSVC C++ ABI
#   and the rest of this project is MinGW g++ against libstdc++; a .o from
#   one will not link into the other. So the backend becomes a DLL and the
#   two sides meet across the C entry points in include/ai/gpu/DeviceABI.h,
#   which both compilers agree about. The MinGW side never sees an MSVC
#   import library at all - it links against our DLL and nothing else.
#
#   Everything above the boundary is byte-for-byte the same build on both.
# ========================================================================= #

ifeq ($(OS),Windows_NT)
    PLATFORM := windows
else
    PLATFORM := unix
endif

# WinHTTP on windows, libcurl elsewhere - see omni-bots/utils/Http.hpp
ifeq ($(PLATFORM),windows)
    HTTP_LIB := -lwinhttp
else
    HTTP_LIB := -lcurl
endif

GPU ?= auto
BACKEND ?= auto
ifeq ($(filter $(BACKEND),auto cpu hip metal),)
    $(error BACKEND must be auto, cpu, hip or metal)
endif
SELECTED_BACKEND := $(BACKEND)
ifeq ($(GPU),0)
    SELECTED_BACKEND := cpu
else ifeq ($(BACKEND),auto)
    ifeq ($(shell uname -s),Darwin)
        HAVE_METAL := $(shell xcrun -f metal >/dev/null 2>&1 && xcrun -f metallib >/dev/null 2>&1 && echo 1)
        SELECTED_BACKEND := $(if $(HAVE_METAL),metal,cpu)
    endif
endif
ifeq ($(SELECTED_BACKEND),metal)
    ifneq ($(shell uname -s),Darwin)
        $(error BACKEND=metal requires macOS)
    endif
endif

ifeq ($(PLATFORM),windows)
    # the HIP SDK puts itself on PATH; HIP_PATH is the variable it sets
    HIPCC ?= $(firstword $(shell command -v hipcc hipcc.bat 2>/dev/null))
else
    HIPCC ?= /opt/rocm/bin/hipcc
endif

# -e rather than -x: on Windows hipcc is a .bat, and whether MSYS2 calls
# that executable is not a question worth depending on. A file that exists
# but cannot run fails loudly at the compile step instead.
ifeq ($(SELECTED_BACKEND),cpu)
    HAVE_HIP :=
else ifeq ($(SELECTED_BACKEND),hip)
    HAVE_HIP := 1
else ifeq ($(GPU),auto)
    HAVE_HIP := $(shell test -e "$(HIPCC)" && echo 1)
else ifeq ($(GPU),0)
    HAVE_HIP :=
else
    HAVE_HIP := 1
endif

# ask the machine which card it has rather than making the user remember.
# gfx1100 is RDNA3 (7900 XT/XTX); this falls back to it so a build on a
# host with no query tool behaves exactly as it did before.
ifeq ($(HAVE_HIP),1)
    DETECTED_ARCH := $(shell rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || \
                             hipInfo  2>/dev/null | grep -om1 'gfx[0-9a-f]*')
endif
GPU_ARCH ?= $(if $(DETECTED_ARCH),$(DETECTED_ARCH),gfx1100)

ifeq ($(SELECTED_BACKEND),metal)
    GPU_NOTE = with the Metal backend
    GPU_FLAGS =
    GPU_LIBS = -framework Metal -framework Foundation
    GPU_RELEASE = bin/obj/metal-backend.o
    GPU_DEBUG = bin/obj/metal-backend.o
    GPU_DEP_RELEASE = $(GPU_RELEASE)
    GPU_DEP_DEBUG = $(GPU_DEBUG)
else ifeq ($(HAVE_HIP),1)
    GPU_NOTE = with the HIP backend ($(GPU_ARCH))
    ifeq ($(PLATFORM),windows)
        # named per configuration rather than shared, because a DLL has to
        # sit beside the executable that loads it
        GPU_FLAGS   = -DAI_GPU_USE_DLL
        GPU_LIBS    =
        GPU_RELEASE = bin/release/gpu-kernels.dll
        GPU_DEBUG   = bin/debug/gpu-kernels.dll
    else
        GPU_FLAGS   =
        GPU_LIBS    = -L/opt/rocm/lib -lamdhip64
        GPU_RELEASE = bin/obj/gpu-kernels.o
        GPU_DEBUG   = bin/obj/gpu-kernels.o
    endif
    GPU_DEP_RELEASE = $(GPU_RELEASE)
    GPU_DEP_DEBUG   = $(GPU_DEBUG)
else
    GPU_FLAGS       =
    GPU_LIBS        =
    GPU_RELEASE     = src/gpu/Stub.cpp
    GPU_DEBUG       = src/gpu/Stub.cpp
    GPU_DEP_RELEASE =
    GPU_DEP_DEBUG   =
    GPU_NOTE        = no GPU backend on this machine
endif


all: release-quat release-tensor release-json release-llm release-llm-test

test: release-quat release-tensor release-json release-llm-test
	./bin/release/quat-testing
	./bin/release/tensor-testing
	./bin/release/json-testing
	./bin/release/llm-testing

# Small NumPy fixtures only; no checkpoint downloads or PyTorch required.
test-unit: release-quat release-tensor release-json release-llm-test
	$(PYTHON) tools/make_fixtures.py
	./bin/release/quat-testing
	./bin/release/tensor-testing
	./bin/release/json-testing --unit
	./bin/release/llm-testing --unit

debug-demo:
	mkdir -p bin/debug
	echo "building demo debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/demo src/demo.cpp -Werror $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-demo:
	mkdir -p bin/release
	echo "building demo release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -o bin/release/demo src/demo.cpp -Werror $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-demo:
	mkdir -p bin/rdbi
	echo "building demo release with debug info..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/demo src/demo.cpp -Werror $(CXXFLAGS) $(MAC_WARN_FLAGS)

debug-time:
	mkdir -p bin/debug
	echo "building time demo debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/time src/time.cpp $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-time:
	mkdir -p bin/release
	echo "building time demo release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -o bin/release/time src/time.cpp $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-time:
	mkdir -p bin/rdbi
	echo "building time demo rdbi..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/time src/time.cpp $(CXXFLAGS) $(MAC_WARN_FLAGS)

debug-ocr:
	mkdir -p bin/debug
	echo "building ocr demo debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/ocr src/ocr.cpp -ltesseract -lleptonica -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-ocr:
	mkdir -p bin/release
	echo "building ocr demo release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -o bin/release/ocr src/ocr.cpp -ltesseract -lleptonica -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-ocr:
	mkdir -p bin/rdbi
	echo "building ocr demo rdbi..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/ocr src/ocr.cpp -ltesseract -lleptonica -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

debug-random:
	mkdir -p bin/debug
	echo "building random demo debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/random src/random.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-random:
	mkdir -p bin/release
	echo "building random demo release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -o bin/release/random src/random.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-random:
	mkdir -p bin/rdbi
	echo "building random demo rdbi..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/random src/random.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

debug-neuralnet:
	mkdir -p bin/debug
	echo "building neuralnet demo debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/neuralnet src/neuralnet.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-neuralnet:
	mkdir -p bin/release
	echo "building neuralnet demo release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -o bin/release/neuralnet src/neuralnet.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-neuralnet:
	mkdir -p bin/rdbi
	echo "building neuralnet demo rdbi..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/neuralnet src/neuralnet.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)
debug-quat:
	mkdir -p bin/debug
	echo "building quaternion tests debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/quat-testing src/quat-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-quat:
	mkdir -p bin/release
	echo "building quaternion tests release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -o bin/release/quat-testing src/quat-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-quat:
	mkdir -p bin/rdbi
	echo "building quaternion tests rdbi..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/quat-testing src/quat-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

debug-tensor:
	mkdir -p bin/debug
	echo "building tensor tests debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/tensor-testing src/tensor-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-tensor:
	mkdir -p bin/release
	echo "building tensor tests release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -march=native -o bin/release/tensor-testing src/tensor-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

rdbi-tensor:
	mkdir -p bin/rdbi
	echo "building tensor tests rdbi..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O2 -g -o bin/rdbi/tensor-testing src/tensor-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

debug-json:
	mkdir -p bin/debug
	echo "building json debug..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g -o bin/debug/json-testing src/json-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-json:
	mkdir -p bin/release
	echo "building json release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -march=native -o bin/release/json-testing src/json-testing.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)










debug-llm: $(GPU_DEP_DEBUG)
	mkdir -p bin/debug
	echo "building llm debug, $(GPU_NOTE)..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O0 -g $(GPU_FLAGS) -o bin/debug/llm src/LLM.cpp $(GPU_DEBUG) $(GPU_LIBS) -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

release-llm: $(GPU_DEP_RELEASE)
	mkdir -p bin/release
	echo "building llm release, $(GPU_NOTE)..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -march=native $(GPU_FLAGS) -o bin/release/llm src/LLM.cpp $(GPU_RELEASE) $(GPU_LIBS) -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

# the architecture suite, deliberately on the stub: it tests the model and
# the loader, and has no business needing ROCm to build
release-llm-test:
	mkdir -p bin/release
	echo "building llm tests release..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -march=native -o bin/release/llm-testing src/llm-testing.cpp src/gpu/Stub.cpp -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)


# ========================================================================= #
#   GPU kernels
#
#   Only src/gpu/Kernels.hip goes through hipcc; everything else stays on
#   $(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD).
#
#   GPU_ARCH is detected from rocminfo/hipInfo and falls back to gfx1100,
#   which is RDNA3 - the 7900 XT/XTX. Override it if the guess is wrong:
#       make release-llm GPU_ARCH=gfx1150
# ========================================================================= #

bin/obj/gpu-kernels.o: src/gpu/Kernels.hip include/ai/gpu/DeviceABI.h
	mkdir -p bin/obj
	echo "compiling HIP kernels for $(GPU_ARCH)..."
	$(HIPCC) --offload-arch=$(GPU_ARCH) -std=c++17 -O3 -c src/gpu/Kernels.hip -o bin/obj/gpu-kernels.o

# Windows: a DLL rather than an object, for the ABI reason at the top of
# this file. One per configuration, because it has to sit beside the
# executable that loads it.
bin/%/gpu-kernels.dll: src/gpu/Kernels.hip include/ai/gpu/DeviceABI.h
	mkdir -p $(dir $@)
	echo "compiling HIP kernels for $(GPU_ARCH) into $@..."
	$(HIPCC) --offload-arch=$(GPU_ARCH) -std=c++17 -O3 -shared \
	         -DAI_GPU_BUILDING_DLL src/gpu/Kernels.hip -o $@

# GPU correctness and bandwidth checks, on whichever backend was selected.
release-gpu-test: $(GPU_DEP_RELEASE)
ifeq ($(strip $(GPU_DEP_RELEASE)),)
	@echo "gpu-testing needs a Metal or HIP backend; CPU-only build selected."
	@false
endif
	mkdir -p bin/release
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Isrc -Iinclude -O3 -march=native $(GPU_FLAGS) -o bin/release/gpu-testing src/gpu-testing.cpp $(GPU_RELEASE) $(GPU_LIBS) -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

# Embed the offline library, so moving the binary never loses its shaders.
bin/obj/metal-kernels.air: src/gpu/Kernels.metal
	mkdir -p bin/obj
	xcrun metal -std=metal3.0 -O3 -fno-fast-math -c $< -o $@

bin/obj/metal-kernels.metallib: bin/obj/metal-kernels.air
	xcrun metallib $< -o $@

bin/obj/metal_library.h: bin/obj/metal-kernels.metallib tools/embed_metal.py
	$(PYTHON) tools/embed_metal.py $< $@

bin/obj/metal-backend.o: src/gpu/Metal.mm include/ai/gpu/DeviceABI.h bin/obj/metal_library.h
	clang++ -std=c++20 -O3 -fobjc-arc -Ibin/obj -Wall -Werror -c $< -o $@

gpu-test: release-gpu-test
	./bin/release/gpu-testing

# ========================================================================= #
#   OMNI-BOTS EXAMPLES
#
#   CPU-only: the agent stack talks to a model through llm::LLM, which needs
#   the gpu ABI resolved. Stub.cpp satisfies it without a card.
# ========================================================================= #

notes-agent:
	mkdir -p bin/release
	echo "building notes-agent example..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Iomni-bots -O2 -o bin/release/notes-agent \
		omni-bots/examples/notes-agent.cpp omni-bots/ai/gpu/Stub.cpp $(HTTP_LIB) -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)

watch-agent:
	mkdir -p bin/release
	echo "building watch-agent example..."
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -std=$(CXXSTD) -Iomni-bots -O2 -o bin/release/watch-agent \
		omni-bots/examples/watch-agent.cpp omni-bots/ai/gpu/Stub.cpp $(HTTP_LIB) -Werror -Wall $(CXXFLAGS) $(MAC_WARN_FLAGS)
