#pragma once

#include <string>
#include <vector>

namespace platform {

struct CMDArg {
    void splitArgs(int argc, char* argv[]) {
        for (int i = 0; i < argc; ++i)
            args.push_back(argv[i]);
    }
    std::vector<std::string> args;
};

}