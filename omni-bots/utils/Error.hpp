#pragma once

#include <cstdlib>
#include <stdexcept>

namespace omnibot {

class Error : public std::runtime_error {
public:
    Error(const std::string& where, const std::string& info)
        : std::runtime_error(where + ": " + info) {}
};

}