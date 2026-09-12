# ========================================================================= #
#   GPU detection
#
#   There is one llm binary, not two. include/ai/gpu/Device.hpp is plain
#   C++ with no HIP types in it, so every translation unit compiles the
#   same way everywhere and only the BACKEND differs at link time:
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
# ========================================================================= #

HIPCC    ?= /opt/rocm/bin/hipcc
GPU_ARCH ?= gfx1100
GPU      ?= auto

ifeq ($(GPU),auto)
    HAVE_HIP := $(shell test -x $(HIPCC) && echo 1)
else ifeq ($(GPU),0)
    HAVE_HIP :=
else
    HAVE_HIP := 1
endif

ifeq ($(HAVE_HIP),1)
    GPU_BACKEND = bin/obj/gpu-kernels.o
    GPU_DEP     = bin/obj/gpu-kernels.o
    GPU_LIBS    = -L/opt/rocm/lib -lamdhip64
    GPU_NOTE    = with the HIP backend
else
    GPU_BACKEND = src/gpu/Stub.cpp
    GPU_DEP     =
    GPU_LIBS    =
    GPU_NOTE    = no GPU backend on this machine
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










debug-llm: $(GPU_DEP)
	clear
	mkdir -p bin/debug
	echo "building llm debug, $(GPU_NOTE)..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/llm src/LLM.cpp $(GPU_BACKEND) $(GPU_LIBS) -Werror -Wall

release-llm: $(GPU_DEP)
	clear
	mkdir -p bin/release
	echo "building llm release, $(GPU_NOTE)..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/llm src/LLM.cpp $(GPU_BACKEND) $(GPU_LIBS) -Werror -Wall

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
#   g++ -std=c++26 and links the object.
#
#   GPU_ARCH must match the card. gfx1100 is RDNA3 - the 7900 XT/XTX. Ask
#   rocminfo if unsure:  rocminfo | grep gfx
# ========================================================================= #

bin/obj/gpu-kernels.o: src/gpu/Kernels.hip include/ai/gpu/Device.hpp
	mkdir -p bin/obj
	echo "compiling HIP kernels for $(GPU_ARCH)..."
	$(HIPCC) --offload-arch=$(GPU_ARCH) -std=c++17 -O3 -c src/gpu/Kernels.hip -o bin/obj/gpu-kernels.o

# unlike llm, this one genuinely cannot be built without a card to test
release-gpu-test:
ifneq ($(HAVE_HIP),1)
	@echo "gpu-testing needs ROCm: $(HIPCC) not found."
	@echo "set HIPCC=/path/to/hipcc, or build 'release-llm' for a CPU-only binary."
	@false
endif
	clear
	mkdir -p bin/release
	$(MAKE) bin/obj/gpu-kernels.o
	echo "building gpu tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/gpu-testing src/gpu-testing.cpp bin/obj/gpu-kernels.o $(GPU_LIBS) -Werror -Wall

gpu-test: release-gpu-test
	./bin/release/gpu-testing
