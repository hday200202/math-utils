all: debug-demo release-demo rdbi-demo debug-time release-time rdbi-time

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
