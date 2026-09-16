#pragma once

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <sstream>
#include <concepts>
#include <unordered_map>

#include "../utils/Error.hpp"
#include "../utils/JSON.hpp"

namespace omnibot {

inline static const std::string INV_CMD = "Invalid command.";

class Command {
public:
    Command() = default;
    virtual ~Command() = default;

    const std::string& cmd() const { return m_cmd; }
    bool ok() const { return m_ok; }

    files::json::Value dump() const {
        using namespace files;
        json::Value v;
        v.type = json::Type::Object;

        json::Value name;
        name.type = json::Type::String;
        name.string = m_cmd;
        v.object.push_back({ "cmd", name });

        json::Value okay;
        okay.type = json::Type::Bool;
        okay.boolean = m_ok;
        v.object.push_back({ "ok", okay });

        for (const auto& m : fields().object)
            v.object.push_back(m);

        return v;
    }

protected:
    std::string m_cmd = "";
    bool        m_ok = false;

    virtual files::json::Value fields() const = 0;
};


class Connector {
public:
    Connector() = default;
    virtual ~Connector() = default;

    /*
        commands() and names() are INDEX-ALIGNED: commands()[i] is the json
        example an agent is shown, names()[i] is the bare name the routers
        key on. They are two lists rather than one because they are read by
        different things - the prompt wants the example, every router wants
        the name - and deriving one from the other would mean parsing the
        example back apart at every hop.
    */
    virtual const std::vector<std::string>& commands() const = 0;
    virtual const std::vector<std::string>& names() const = 0;

    /*
        executeCmd(cmd):
        - cmd is an object carrying at least "cmd", plus whatever fields that
          command takes. The reply is a Command::dump(), so it always has
          "cmd" and "ok" and the connector's own fields after them.
    */
    virtual files::json::Value executeCmd(const files::json::Value& cmd) = 0;
};


[[noreturn]] inline void connectorError(
    const std::string& connector,
    const std::string& info
) { throw Error(connector, info); }

}