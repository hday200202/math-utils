#pragma once

#include <queue>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#include "Interface.hpp"

// Note: Container owns interfaces, Command Bus just gets a reference.

namespace omnibot {

inline static const std::string EMPTY_CMD_QUEUE = "The command queue is empty.";
inline static const std::string EMPTY_OUT_QUEUE = "The output queue is empty.";

class CommandBus {
public:
    CommandBus() = default;
    CommandBus(const std::string& name) : m_name(name) {}

    void write(files::json::Value cmd) {
        m_cmdQueue.push_back(std::move(cmd));
    }

    /*
        write(text):
        - the agent hands over text, so the parse happens here and nothing
          above the bus has to know the wire format is json. Returns false on
          malformed input rather than queueing something that cannot route.
    */
    bool write(const std::string& text) {
        const files::json::ParseResult parsed = files::json::parse(text);
        if (!parsed.ok) return false;

        m_cmdQueue.push_back(parsed.value);
        return true;
    }

    void execute() {
        while (!m_cmdQueue.empty()) {
            const files::json::Value cmd = std::move(m_cmdQueue.front());
            m_cmdQueue.pop_front();

            const auto it = m_cmdRouter.find(cmd["cmd"].asString());

            if (it != m_cmdRouter.end() && it->second)
                m_outQueue.push_back(it->second->executeCmd(cmd));
            else
                m_outQueue.push_back(files::json::jobject({
                    { "cmd",    files::json::jstr(cmd["cmd"].asString()) },
                    { "ok",     files::json::jbool(false) },
                    { "result", files::json::jstr(m_name + ": " + INV_CMD) }
                }));

            record(cmd, m_outQueue.back());
        }
    }

    files::json::Value read() {
        if (!hasOutput()) return files::json::Value();

        files::json::Value v = std::move(m_outQueue.front());
        m_outQueue.pop_front();
        return v;
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

        for (const Interface* i : m_interfaces) {
            const std::vector<std::string> examples = i->commands();
            const std::vector<std::string> named    = i->names();

            for (size_t k = 0; k < examples.size() && k < named.size(); ++k) {
                const auto it = m_cmdRouter.find(named[k]);
                if (it != m_cmdRouter.end() && it->second == i)
                    out.push_back(examples[k]);
            }
        }

        return out;
    }

    // for debug purposes, will write the cmdQueue,
    // outQueue, and router to string for printing
    std::string window() const {
        std::string out = "";

        out += "Commands: \n";
        for (const auto& cmd : m_cmdQueue)
            out += "\t" + files::json::dump(cmd, 0) + "\n";
        
        out += "\nOutputs: \n";
        for (const auto& cmd : m_outQueue)
            out += "\t" + files::json::dump(cmd, 0) + "\n";

        out += "\nRoutes: \n";
        for (const auto& route : m_cmdRouter)
            out += "\t" + route.first + " -> " + route.second->getName() + "\n";

        out += "\nHistory: \n";
        for (const auto& [cmd, result] : m_log)
            out += "\t" + cmd + "\n\t  -> " + result + "\n";

        return out;
    }

    void clearHistory() { m_log.clear(); }

    /*
        trace(on):
        - echo every command and result to stdout the moment it is routed.

        window() can only be read between ticks, and a tick is many model
        turns long - minutes, on a real checkpoint. Watching a container work
        means seeing each command as it lands, not a summary once the episode
        is already over. Off by default; this is for a console someone is
        actually looking at.
    */
    static void trace(const bool on) { s_trace = on; }

    std::string getName() const { return m_name; }

    bool hasCommands() const { return !m_cmdQueue.empty(); }
    bool hasOutput() const { return !m_outQueue.empty(); }

private:
    /*
        record(cmd, result):
        - execute() drains both queues, so by the time anything outside the
          bus looks, window() has nothing left to show. This keeps the last
          LOG_MAX pairs, so a running container can be watched rather than
          only caught mid-tick. Bounded, because a long-lived container would
          otherwise grow it forever.
    */
    void record(const files::json::Value& cmd, const files::json::Value& result) {
        m_log.push_back({ files::json::dump(cmd, 0), files::json::dump(result, 0) });
        while (m_log.size() > LOG_MAX) m_log.pop_front();

        if (s_trace)
            std::cout << "  " << m_log.back().first
                      << "\n    -> " << m_log.back().second << std::endl;
    }

    static constexpr size_t LOG_MAX = 32;

    inline static bool s_trace = false;

    std::deque<std::pair<std::string, std::string>> m_log;   // executed, for window()

    std::deque<files::json::Value> m_cmdQueue;     // commands in
    std::deque<files::json::Value> m_outQueue;     // results from interfaces

    std::string m_name = "CommandBus";

    // per valid cmd routing
    std::unordered_map<std::string, Interface*> m_cmdRouter;

    // the distinct interfaces those routes point at, not owned
    std::vector<Interface*> m_interfaces;
};

}