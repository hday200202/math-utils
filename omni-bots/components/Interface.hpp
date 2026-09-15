#pragma once

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
            for (const auto& cmd : c->commands())
                m_cmdRouter[extractFirstToken(cmd)] = c;
            m_connectors.emplace_back(c);
        }
    }

    // Can only create an interface through the factory function
    friend Interface* interface(
        const std::string& name, 
        const std::vector<Connector*>& connectors
    );

public:
    std::string executeCmd(const std::string& cmd) {
        auto it = m_cmdRouter.find(extractFirstToken(cmd));
        if (it == m_cmdRouter.end() || !it->second) return m_name + ": " + INV_CMD;
        return it->second->executeCmd(cmd);
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


inline void interfaceError(
    const std::string& interface,
    const std::string& info
) { std::cerr << interface << ": " << info << std::endl; }


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