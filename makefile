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










debug-llm:
	clear
	mkdir -p bin/debug
	echo "building llm debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/llm src/LLM.cpp src/gpu/Stub.cpp -Werror -Wall

release-llm:
	clear
	mkdir -p bin/release
	echo "building llm release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/llm src/LLM.cpp src/gpu/Stub.cpp -Werror -Wall

release-llm-test:
	clear
	mkdir -p bin/release
	echo "building llm tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/llm-testing src/llm-testing.cpp src/gpu/Stub.cpp -Werror -Wall


# ========================================================================= #
#   GPU
#
#   Only src/gpu/Kernels.hip goes through hipcc; everything else stays on
#   g++ -std=c++26 and links the object. include/ai/gpu/Device.hpp is plain
#   C++ on purpose so that boundary holds.
#
#   GPU_ARCH must match the card. gfx1100 is RDNA3 - the 7900 XT/XTX. Ask
#   rocminfo if unsure:  rocminfo | grep gfx
# ========================================================================= #

HIPCC    ?= /opt/rocm/bin/hipcc
GPU_ARCH ?= gfx1100
HIP_LIBS  = -L/opt/rocm/lib -lamdhip64

bin/obj/gpu-kernels.o: src/gpu/Kernels.hip include/ai/gpu/Device.hpp
	mkdir -p bin/obj
	echo "compiling HIP kernels for $(GPU_ARCH)..."
	$(HIPCC) --offload-arch=$(GPU_ARCH) -std=c++17 -O3 -c src/gpu/Kernels.hip -o bin/obj/gpu-kernels.o

release-llm-gpu: bin/obj/gpu-kernels.o
	clear
	mkdir -p bin/release
	echo "building llm release with GPU backend..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/llm-gpu src/LLM.cpp bin/obj/gpu-kernels.o $(HIP_LIBS) -Werror -Wall

release-gpu-test: bin/obj/gpu-kernels.o
	clear
	mkdir -p bin/release
	echo "building gpu tests release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/gpu-testing src/gpu-testing.cpp bin/obj/gpu-kernels.o $(HIP_LIBS) -Werror -Wall

gpu-test: release-gpu-test
	./bin/release/gpu-testing
