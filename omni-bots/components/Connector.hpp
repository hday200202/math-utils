#pragma once

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <sstream>
#include "../utils/File.hpp"

namespace omnibot {

inline static const std::string INV_CMD = "Invalid command.";

class Connector {
public:
    Connector() = default;
    virtual ~Connector() = default;

    virtual const std::vector<std::string>& commands() const = 0;
    virtual std::string executeCmd(const std::string& cmd) = 0;
};

}