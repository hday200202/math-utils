#pragma once

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <sstream>
#include <concepts>
#include <unordered_map>
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


// The command NAME is the first token, everything after it lives 
// here because Interface and CommandBus both key on it.
inline std::string extractFirstToken(const std::string& cmd) {
    return cmd.substr(0, cmd.find(' '));
}


inline void connectorError(
    const std::string& connector,
    const std::string& info
) { std::cerr << connector << ": " << info << std::endl; }

}