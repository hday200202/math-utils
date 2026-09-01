all: debug-demo release-demo rdbi-demo debug-nn-dyn release-nn-dyn rdbi-nn-dyn debug-nn-image release-nn-image debug-invp release-invp debug-ocr release-ocr

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

debug-demo-asm:
	clear
	mkdir -p bin/debug
	echo "building demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -S -o bin/debug/demo src/demo.cpp

release-demo-asm:
	clear
	mkdir -p bin/release
	echo "building demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -S -o bin/release/demo src/demo.cpp

rdbi-demo-asm:
	clear
	mkdir -p bin/rdbi
	echo "building demo release with debug info..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -S -o bin/rdbi/demo src/demo.cpp

debug-nn-dyn:
	clear
	mkdir -p bin/debug
	echo "building dynamic neural network demo..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/nn-demo-dyn src/nn-demo-dyn.cpp -Werror

release-nn-dyn:
	clear
	mkdir -p bin/release
	echo "building dynamic neural network release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/nn-demo-dyn src/nn-demo-dyn.cpp -Werror

rdbi-nn-dyn:
	clear
	mkdir -p bin/rdbi
	echo "building dynamic neural network rdbi..."
	g++ -std=c++26 -Isrc -Iinclude -O2 -g -o bin/rdbi/nn-demo-dyn src/nn-demo-dyn.cpp -Werror

debug-nn-image:
	clear
	mkdir -p bin/debug
	echo "building neural network image demo..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/nn-image-demo src/nn-image-demo.cpp -Werror

release-nn-image:
	clear
	mkdir -p bin/release
	echo "building neural network image release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/nn-image-demo src/nn-image-demo.cpp -Werror


debug-invp:
	clear
	mkdir -p bin/debug
	echo "building invoice parser debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/invoice-parser src/invoice-parser.cpp -Werror

release-invp:
	clear
	mkdir -p bin/release
	echo "building invoice parser release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/invoice-parser src/invoice-parser.cpp -Werror


debug-ocr:
	clear
	mkdir -p bin/debug
	echo "building ocr demo debug..."
	g++ -std=c++26 -Isrc -Iinclude -O0 -g -o bin/debug/ocr-demo src/ocr-demo.cpp -Werror

release-ocr:
	clear
	mkdir -p bin/release
	echo "building ocr demo release..."
	g++ -std=c++26 -Isrc -Iinclude -O3 -o bin/release/ocr-demo src/ocr-demo.cpp -Werror