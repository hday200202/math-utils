#pragma once

#include <queue>
#include <unordered_map>
#include <algorithm>

#include "Interface.hpp"

// Note: Container owns interfaces, Command Bus just gets a reference.

namespace omnibot {

inline static const std::string EMPTY_CMD_QUEUE = "The command queue is empty.";
inline static const std::string EMPTY_OUT_QUEUE = "The output queue is empty.";

class CommandBus {
public:
    CommandBus() = default;
    CommandBus(const std::string& name) : m_name(name) {}

    void write(const std::string& cmd) { 
        m_cmdQueue.push_back(cmd);
    }

    void execute() {
        while (!m_cmdQueue.empty()) {
            const std::string cmd = m_cmdQueue.front();
            m_cmdQueue.pop_front();

            auto it = m_cmdRouter.find(extractFirstToken(cmd));
            m_outQueue.push_back(
                (it != m_cmdRouter.end() && it->second) ? 
                    it->second->executeCmd(cmd) : m_name + ": " + INV_CMD
            );
        }
    }

    std::string read() {
        if (!hasOutput()) return m_name + ": " + EMPTY_OUT_QUEUE;
        auto s = m_outQueue.front(); 
        m_outQueue.pop_front();
        return s; 
    }

    bool makeRoute(const std::string& cmd, Interface* interface) {
        if (m_cmdRouter.contains(cmd)) return false;
        m_cmdRouter[cmd] = interface;

        if (std::find(m_interfaces.begin(), m_interfaces.end(), interface)
            == m_interfaces.end())
            m_interfaces.push_back(interface);

        return true;
    }

    std::vector<std::string> commands() const {
        std::vector<std::string> out;

        for (const Interface* i : m_interfaces)
            for (const std::string& cmd : i->commands()) {
                const auto it = m_cmdRouter.find(extractFirstToken(cmd));
                if (it != m_cmdRouter.end() && it->second == i)
                    out.push_back(cmd);
            }

        return out;
    }

    // for debug purposes, will write the cmdQueue,
    // outQueue, and router to string for printing
    std::string window() const {
        std::string out = "";

        out += "Commands: \n";
        for (const auto& cmd : m_cmdQueue)
            out += "\t" + cmd + "\n";
        
        out += "\nOutputs: \n";
        for (const auto& cmd : m_outQueue)
            out += "\t" + cmd + "\n";

        out += "\nRoutes: \n";
        for (const auto& route : m_cmdRouter)
            out += "\t" + route.first + " -> " + route.second->getName() + "\n";

        return out;
    }

    std::string getName() const { return m_name; }

    bool hasCommands() const { return !m_cmdQueue.empty(); }
    bool hasOutput() const { return !m_cmdQueue.empty(); }

private:
    std::deque<std::string> m_cmdQueue;     // in commands
    std::deque<std::string> m_outQueue;     // outputs from interfaces

    std::string m_name = "CommandBus";

    // per valid cmd routing
    std::unordered_map<std::string, Interface*> m_cmdRouter;

    // the distinct interfaces those routes point at, not owned
    std::vector<Interface*> m_interfaces;
};

}