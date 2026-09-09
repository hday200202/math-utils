all: release-quat release-tensor release-json release-gpt2 release-gpt2demo

test: release-quat release-tensor release-json release-gpt2
	./bin/release/quat-testing
	./bin/release/tensor-testing
	./bin/release/json-testing
	./bin/release/gpt2-testing

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

debug-safetensors:
	clear
	mkdir -p bin/debug
	echo "building safetensors debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/safetensors-testing src/safetensors-testing.cpp -Werror -Wall

release-safetensors:
	clear
	mkdir -p bin/release
	echo "building safetensors release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/safetensors-testing src/safetensors-testing.cpp -Werror -Wall

debug-gpt2:
	clear
	mkdir -p bin/debug
	echo "building gpt2 debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/gpt2-testing src/gpt2-testing.cpp -Werror -Wall

release-gpt2:
	clear
	mkdir -p bin/release
	echo "building gpt2 release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/gpt2-testing src/gpt2-testing.cpp -Werror -Wall

debug-gpt2demo:
	clear
	mkdir -p bin/debug
	echo "building gpt2 demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/gpt2 src/gpt2.cpp -Werror -Wall

release-gpt2demo:
	clear
	mkdir -p bin/release
	echo "building gpt2 demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -march=native -o bin/release/gpt2 src/gpt2.cpp -Werror -Wall
