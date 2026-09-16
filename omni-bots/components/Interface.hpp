#pragma once

#include <cstdlib>

#include "Connector.hpp"

namespace omnibot {

class Interface {
private:
    Interface() = default;
    Interface(
        const std::string& name, 
        const std::vector<Connector*>& connectors
    ) : m_name(name) {
        for (auto* c : connectors) {
            /* route on the NAME. commands() is the json example an agent is
               shown; names() is what actually arrives in cmd["cmd"] */
            for (const auto& name : c->names())
                m_cmdRouter[name] = c;
            m_connectors.emplace_back(c);
        }
    }

    // Can only create an interface through the factory function
    friend Interface* interface(
        const std::string& name, 
        const std::vector<Connector*>& connectors
    );

public:
    files::json::Value executeCmd(const files::json::Value& cmd) {
        using namespace files;

        const auto it = m_cmdRouter.find(cmd["cmd"].asString());
        if (it == m_cmdRouter.end() || !it->second)
            return json::jobject({
                { "cmd",    json::jstr(cmd["cmd"].asString()) },
                { "ok",     json::jbool(false) },
                { "result", json::jstr(m_name + ": " + INV_CMD) }
            });

        return it->second->executeCmd(cmd);
    }


    /* the bare names of everything reachable through this interface */
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const auto& c : m_connectors)
            for (const auto& n : c->names())
                out.push_back(n);
        return out;
    }

    const std::vector<std::string> commands() const {
        std::vector<std::string> validCmds;
        for (const auto& c : m_connectors)
            for (const auto& cmd : c->commands())
                validCmds.push_back(cmd);
        return validCmds;
    }

    std::string getName() const { return m_name; }

private:
    std::unordered_map<std::string, Connector*> m_cmdRouter;
    std::vector<std::unique_ptr<Connector>> m_connectors;
    std::string m_name;
};


[[noreturn]] inline void interfaceError(
    const std::string& interface,
    const std::string& info
) { throw Error(interface, info); }


// Interface factory function
inline Interface* interface(
    const std::string& name,
    const std::vector<Connector*>& connectors
) { 
    if (name.empty()) interfaceError("Interface: ", "Can not create an interface without name.");
    if (connectors.empty()) interfaceError(name + ": ", "Can not create an interface without connectors.");

    return new Interface(name, connectors); 
}

}