# ========================================================================= #
#   GPU detection
#
#   There is one llm binary, not two. include/ai/gpu/Device.hpp is plain
#   C++ with no HIP types in it AND entirely inline, so every translation
#   unit compiles the same way everywhere and only the BACKEND differs at
#   link time:
#
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

ifeq ($(PLATFORM),windows)
    # the HIP SDK puts itself on PATH; HIP_PATH is the variable it sets
    HIPCC ?= $(firstword $(shell command -v hipcc hipcc.bat 2>/dev/null))
else
    HIPCC ?= /opt/rocm/bin/hipcc
endif

# -e rather than -x: on Windows hipcc is a .bat, and whether MSYS2 calls
# that executable is not a question worth depending on. A file that exists
# but cannot run fails loudly at the compile step instead.
ifeq ($(GPU),auto)
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

ifeq ($(HAVE_HIP),1)
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

debug-demo:
	clear
	mkdir -p bin/debug
	echo "building demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/demo src/demo.cpp -Werror

release-demo:
	clear
	mkdir -p bin/release
	echo "building demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/demo src/demo.cpp -Werror

rdbi-demo:
	clear
	mkdir -p bin/rdbi
	echo "building demo release with debug info..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/demo src/demo.cpp -Werror

debug-time:
	clear
	mkdir -p bin/debug
	echo "building time demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/time src/time.cpp

release-time:
	clear
	mkdir -p bin/release
	echo "building time demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/time src/time.cpp

rdbi-time:
	clear
	mkdir -p bin/rdbi
	echo "building time demo rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/time src/time.cpp

debug-ocr:
	clear
	mkdir -p bin/debug
	echo "building ocr demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/ocr src/ocr.cpp -ltesseract -lleptonica -Werror -Wall

release-ocr:
	clear
	mkdir -p bin/release
	echo "building ocr demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/ocr src/ocr.cpp -ltesseract -lleptonica -Werror -Wall

rdbi-ocr:
	clear
	mkdir -p bin/rdbi
	echo "building ocr demo rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/ocr src/ocr.cpp -ltesseract -lleptonica -Werror -Wall

debug-random:
	clear
	mkdir -p bin/debug
	echo "building random demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/random src/random.cpp -Werror -Wall

release-random:
	clear
	mkdir -p bin/release
	echo "building random demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/random src/random.cpp -Werror -Wall

rdbi-random:
	clear
	mkdir -p bin/rdbi
	echo "building random demo rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/random src/random.cpp -Werror -Wall

debug-neuralnet:
	clear
	mkdir -p bin/debug
	echo "building neuralnet demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/neuralnet src/neuralnet.cpp -Werror -Wall

release-neuralnet:
	clear
	mkdir -p bin/release
	echo "building neuralnet demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/neuralnet src/neuralnet.cpp -Werror -Wall

rdbi-neuralnet:
	clear
	mkdir -p bin/rdbi
	echo "building neuralnet demo rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/neuralnet src/neuralnet.cpp -Werror -Wall
debug-quat:
	clear
	mkdir -p bin/debug
	echo "building quaternion tests debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/quat-testing src/quat-testing.cpp -Werror -Wall

release-quat:
	clear
	mkdir -p bin/release
	echo "building quaternion tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/quat-testing src/quat-testing.cpp -Werror -Wall

rdbi-quat:
	clear
	mkdir -p bin/rdbi
	echo "building quaternion tests rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/quat-testing src/quat-testing.cpp -Werror -Wall

debug-tensor:
	clear
	mkdir -p bin/debug
	echo "building tensor tests debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/tensor-testing src/tensor-testing.cpp -Werror -Wall

release-tensor:
	clear
	mkdir -p bin/release
	echo "building tensor tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/tensor-testing src/tensor-testing.cpp -Werror -Wall

rdbi-tensor:
	clear
	mkdir -p bin/rdbi
	echo "building tensor tests rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/tensor-testing src/tensor-testing.cpp -Werror -Wall

debug-json:
	clear
	mkdir -p bin/debug
	echo "building json debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/json-testing src/json-testing.cpp -Werror -Wall

release-json:
	clear
	mkdir -p bin/release
	echo "building json release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/json-testing src/json-testing.cpp -Werror -Wall










debug-llm: $(GPU_DEP_DEBUG)
	clear
	mkdir -p bin/debug
	echo "building llm debug, $(GPU_NOTE)..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g $(GPU_FLAGS) -o bin/debug/llm src/LLM.cpp $(GPU_DEBUG) $(GPU_LIBS) -Werror -Wall

release-llm: $(GPU_DEP_RELEASE)
	clear
	mkdir -p bin/release
	echo "building llm release, $(GPU_NOTE)..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native $(GPU_FLAGS) -o bin/release/llm src/LLM.cpp $(GPU_RELEASE) $(GPU_LIBS) -Werror -Wall

# the architecture suite, deliberately on the stub: it tests the model and
# the loader, and has no business needing ROCm to build
release-llm-test:
	clear
	mkdir -p bin/release
	echo "building llm tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/llm-testing src/llm-testing.cpp src/gpu/Stub.cpp -Werror -Wall


# ========================================================================= #
#   GPU kernels
#
#   Only src/gpu/Kernels.hip goes through hipcc; everything else stays on
#   g++ -std=c++26.
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

# unlike llm, this one genuinely cannot be built without a card to test
release-gpu-test:
ifneq ($(HAVE_HIP),1)
	@echo "gpu-testing needs ROCm: $(HIPCC) not found."
	@echo "set HIPCC=/path/to/hipcc, or build 'release-llm' for a CPU-only binary."
	@false
endif
	clear
	mkdir -p bin/release
	$(MAKE) $(GPU_RELEASE)
	echo "building gpu tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native $(GPU_FLAGS) -o bin/release/gpu-testing src/gpu-testing.cpp $(GPU_RELEASE) $(GPU_LIBS) -Werror -Wall

gpu-test: release-gpu-test
	./bin/release/gpu-testing

# ========================================================================= #
#   OMNI-BOTS EXAMPLES
#
#   CPU-only: the agent stack talks to a model through llm::LLM, which needs
#   the gpu ABI resolved. Stub.cpp satisfies it without a card.
# ========================================================================= #

notes-agent:
	clear
	mkdir -p bin/release
	echo "building notes-agent example..."
	g++ -std=c++26 -Iomni-bots -O2 -o bin/release/notes-agent \
		omni-bots/examples/notes-agent.cpp omni-bots/ai/gpu/Stub.cpp $(HTTP_LIB) -Werror -Wall

watch-agent:
	clear
	mkdir -p bin/release
	echo "building watch-agent example..."
	g++ -std=c++26 -Iomni-bots -O2 -o bin/release/watch-agent \
		omni-bots/examples/watch-agent.cpp omni-bots/ai/gpu/Stub.cpp $(HTTP_LIB) -Werror -Wall
