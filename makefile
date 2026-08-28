all: debug-demo release-demo rdbi-demo
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
