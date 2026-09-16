#pragma once

#include <map>
#include <cstdlib>

#include "Agent.hpp"
#include "components/Interface.hpp"
#include "components/CommandBus.hpp"
#include "utils/Error.hpp"

namespace omnibot {

class Trigger   {};


/*
    Essentially just a wrapper to make it easier to do bus(name, {interface, ...})
    without the dev touching CommandBus itself, and allowing the container to handle
    CommandBus creation and routing.
*/
class Bus {
private:
    Bus() = default;
    Bus(const std::string& name, const std::vector<Interface*> interfaces) 
    : m_name(name), m_interfaces(interfaces) {}
    friend Bus bus(const std::string& name, const std::vector<Interface*> interfaces);

public:
    std::string getName() const { return m_name; }
    const std::vector<Interface*>& getInterfaces() const { return m_interfaces; }

private:
    std::string m_name;
    std::vector<Interface*> m_interfaces;
};


[[noreturn]] inline void busError(
    const std::string& busName,
    const std::string& info
) { throw Error(busName, info); }


[[noreturn]] inline void containerError(
    const std::string& funcName,
    const std::string& info
) { throw Error(funcName, info); }


class Container {
public:
    Container(const std::vector<Interface*>& interfaces) {
        CommandBus* mainBus = new CommandBus("Main Bus");
        for (auto* i : interfaces) {
            for (auto& name : i->names())
                if (!mainBus->makeRoute(name, i))
                    busError("Main Bus", "Interface valid command collision.");
            m_interfaces.emplace_back(i);
        }
        
        if (!addBus("Main Bus", mainBus)) busError("Main Bus", "Bus name collision.");
    }

    Container(const std::vector<Bus>& buses) {
        for (auto& b : buses) {
            CommandBus* newBus = new CommandBus(b.getName());
            for (auto* i : b.getInterfaces()) {
                for (auto& name : i->names())
                    if (!newBus->makeRoute(name, i))
                        busError(b.getName(), "Interface valid command collision.");
                
                bool inVec = false;
                for (auto& ptr : m_interfaces)
                    if (ptr.get() == i) inVec = true;

                if (!inVec) m_interfaces.emplace_back(i);
            }
            if (!addBus(b.getName(), newBus)) busError(b.getName(), "Bus name collision.");
        }
    }

    Container& spawnAgent(
        const std::string& name,
        const std::string& prompt,
        const Trigger& trigger
    ) {
        if (m_buses.size() != 1) 
            containerError(
                "Spawn Agent - Single Bus", 
                "Must assign a bus to an agent if multiple buses exist"
            );

        CommandBus* cmdBus = getBus("Main Bus");
        const std::string cmds = cmdBlock(cmdBus);
        
        if (!m_model) containerError(name, "No model loaded. Call model() first.");
        Agent* agent = new Agent(name, prompt, cmds, m_model.get());
        if (!addAgent(name, agent))
            containerError(name, "Agent name collision.");

        if (!addRoute(agent, cmdBus))
            containerError(agent->getName(), "Already assigned a bus");

        return *this;
    }

    Container& spawnAgent(
        const std::string& name,
        const std::string& bus,
        const std::string& prompt,
        const Trigger& trigger
    ) {
        if (bus.empty()) containerError(
            "Spawn Agent - Multi-Bus",
            "Must specify a bus if multiple buses are available."
        );

        CommandBus* cmdBus = getBus(bus);

        if (!cmdBus) containerError(
            "Spawn Agent - Multi-Bus",
            "Could not match \"" + bus + "\" with any available buses."
        );

        const std::string cmds = cmdBlock(cmdBus);

        if (!m_model) containerError(name, "No model loaded. Call model() first.");
        Agent* agent = new Agent(name, prompt, cmds, m_model.get());
        if (!addAgent(name, agent))
            containerError(name, "Agent name collision.");

        if (!addRoute(agent, cmdBus))
            containerError(agent->getName(), "Already assigned a bus");

        return *this;
    }


    Container& model(const std::string& root, const std::string& name) {
        m_model = std::make_unique<LocalLLM>(root, name, ai::quant::Type::Q4);
        return *this;
    }


    void tick() { for (auto& [name, agent] : m_agents) run(agent.get()); }


    std::string window() const {
        std::string out = "";
        for (const auto& [name, b] : m_buses)
            out += "[ " + name + " ]" + "\n" 
                + b->window() + "\n\n";
        return out;
    }

private:
    std::vector<std::unique_ptr<Interface>>                 m_interfaces;
    std::map<std::string, std::unique_ptr<CommandBus>>      m_buses;
    std::map<std::string, std::unique_ptr<Agent>>           m_agents;
    std::map<Agent*, CommandBus*>                           m_cmdBusRouter;

    std::string m_staticCmds = "{\"cmd\":\"done\"}\n"; // can add more, checked for in tick()

    inline static const std::string THINK_END = "</think>";

    std::unique_ptr<Model> m_model;

    /*
        cmdBlock(bus):
        - the command list as the agent is shown it: a heading, the examples,
          then what to do with them.

        The examples alone are not enough. Handed a bare run of json and a
        task, a model reads them as decoration and answers the task in
        prose - which extractCommands then finds nothing in, and the episode
        ends having done nothing. The heading and the rule after it are what
        turn a list into an instruction, and they are worth more than any
        amount of wording in the task itself.

        The rule says commands, plural, and says so HERE rather than in the
        agent's prompt. How to drive the bus is protocol - it is the same for
        every agent, it follows from the queues existing, and it is not
        something the person writing "read the inbox, add the items to the
        list" should have to know about. A task describes the work; this
        describes the machine the work runs on, and the two do not belong in
        the same string.
    */
    /*
        filled(example, sample):
        - one of the bus's own command examples with its <placeholders>
          replaced by real-looking values.

        A placeholder shows where a value goes; it does not show a value. A
        model given only <text> has nothing to copy and falls back to
        copying whatever is nearest to hand - in practice the entire result
        of the command before it. Substituting turns the example into a
        thing that can be imitated.

        Numbers are unquoted in the example, so they get a number; anything
        else is inside quotes already and gets the word.
    */
    std::string filled(const std::string& example, const std::string& sample) const {
        std::string out;

        for (size_t i = 0; i < example.size(); ++i) {
            if (example[i] != '<') { out += example[i]; continue; }

            const size_t close = example.find('>', i);
            if (close == std::string::npos) { out += example[i]; continue; }

            out += (example.compare(i, close - i + 1, "<number>") == 0)
                 ? std::string("1")
                 : sample;
            i = close;
        }

        return out;
    }


    /*
        batchExample(bus):
        - a worked reply, built from this bus's own commands.

        Told in prose to send several commands at once, both models tested
        sent one holding a list instead. Shown two lines that differ only
        in their value, neither did. The example has to be generated rather
        than written down because the command names belong to whatever
        connectors this bus happens to have, and nobody writing an agent's
        task is in a position to know them.

        A command taking a value is the one worth repeating - repeating a
        command that takes none demonstrates nothing - so that is the one
        picked when the bus has it.
    */
    std::string batchExample(const CommandBus* bus) const {
        const std::vector<std::string> examples = bus->commands();
        if (examples.empty()) return {};

        for (const std::string& e : examples) {
            if (e.find('<') == std::string::npos) continue;

            return "\nWhen a step covers several things, the reply looks like "
                   "this - one command each, same reply:\n"
                 + filled(e, "first thing") + "\n"
                 + filled(e, "second thing") + "\n";
        }

        if (examples.size() < 2) return {};

        return "\nWhen a step needs more than one command, send them in the "
               "same reply:\n"
             + examples[0] + "\n" + examples[1] + "\n";
    }


    std::string cmdBlock(const CommandBus* bus) const {
        std::string out = "Available commands:\n";

        for (const auto& cmd : bus->commands())
            out += cmd + "\n";
        out += m_staticCmds;

        out += batchExample(bus);

        out += "\nReply with commands from this list as json and nothing else. "
               "Replace every <placeholder> with a real value.\n"
               "\n"
               "One command does one thing. When a step involves several "
               "things, reply with several commands - one per line, all in "
               "the same reply. They run in order and their results come back "
               "together. Four things to do is four commands in one reply, "
               "never one command repeated over four replies.\n"
               "\n"
               "Do not stop after the first command. Before you finish a "
               "reply, check every thing the step asked for and make sure "
               "each one has its own command in it.\n"
               "\n"
               "A field holds one value, never a list. Where the task gives a "
               "format for a value, copy that format exactly, punctuation and "
               "escapes included.\n"
               "\n"
               "Reply {\"cmd\": \"done\"} only once every step is finished. A "
               "reply that contains done must not leave anything undone.\n";

        return out;
    }

    bool addBus(const std::string& name, CommandBus* bus) {
        if (m_buses.contains(name)) return false;
        m_buses[name] = std::unique_ptr<CommandBus>(bus);
        return true;
    }

    bool addAgent(const std::string& name, Agent* agent) {
        if (m_agents.contains(name)) return false;
        m_agents[name] = std::unique_ptr<Agent>(agent);
        return true;
    }

    CommandBus* getBus(const std::string& name) {
        if (!m_buses.contains(name)) return nullptr;
        return m_buses[name].get();
    }

    Agent* getAgent(const std::string& name) {
        if (!m_agents.contains(name)) return nullptr;
        return m_agents[name].get();
    }

    bool addRoute(Agent* agent, CommandBus* bus) {
        if (m_cmdBusRouter.contains(agent)) return false;
        m_cmdBusRouter[agent] = bus;
        return true;
    }

    /*
        run(agent):
        - statefulness is decided here rather than at spawn, because it is
          not a property of the agent. There is one conversation in the
          model, so a lone agent may keep it and any of several may not, and
          which of those is true changes the moment a second agent is
          spawned. Deciding it per run means that change is picked up instead
          of two agents quietly sharing one context.
    */
    void run(Agent* agent) {
        agent->setStateful(m_agents.size() == 1);

        CommandBus* cmdBus = m_cmdBusRouter.at(agent);
        std::string input = "Begin.";

        for (size_t turn = 0; turn < MAX_TURNS; ++turn) {
            const std::vector<files::json::Value> cmds = extractCommands(agent->send(input));

            if (cmds.empty()) {
                input = "No command found. Reply with commands from the list, "
                        "as json, and nothing else.";
                continue;
            }

            bool done = false;
            for (const auto& c : cmds) {
                if (c["cmd"].asString() == "done") { done = true; break; }
                cmdBus->write(c);
            }

            /*
                done is checked before the write because it was never routed,
                but returning here would be wrong: a reply that ends in done
                usually carries real commands ahead of it, and those are
                already queued. Execute first, then leave.
            */
            cmdBus->execute();

            /*
                Results go back labelled, not raw.

                Handed a bare result object as its next turn, a model reads
                it as something to continue rather than something to act on -
                the observed failure is copying a read's "result" straight
                into the next command's "data". Saying what the json is, and
                that a value has to be taken out of it rather than passed
                along, is the difference between a tool result and a line to
                autocomplete.
            */
            std::string results;
            while (cmdBus->hasOutput())
                results += files::json::dump(cmdBus->read(), 0) + "\n";

            if (done) return;

            input = "Results of your commands:\n" + results
                  + "\nUse these to decide what to do next. Take only the "
                    "value you need out of a result - never send a whole "
                    "result back as a command field. Continue, or reply "
                    "{\"cmd\": \"done\"} if the task is finished.";
        }
    }

    std::vector<files::json::Value> extractCommands(const std::string& cmdFromAgent) const {
        std::vector<files::json::Value> jsonCmds;

        /*
            Anything inside a reasoning block is not a command.

            A thinking model works the problem out before answering, and
            working it out means writing candidate commands down - "I should
            send {"cmd": "append_list", ...}" - which is deliberation, not a
            decision. Scanning it would execute every option the model
            considered and rejected. Only what comes after the block counts.
        */
        const std::string& reply = cmdFromAgent;
        size_t from = reply.rfind(THINK_END);
        from = (from == std::string::npos) ? 0 : from + THINK_END.size();

        std::string jsonStr = "";
        int  depth   = 0;
        bool inStr   = false;
        bool escaped = false;

        for (const char c : reply.substr(from)) {
            if (depth > 0) jsonStr += c;

            if (escaped)             { escaped = false; continue; }
            if (inStr && c == '\\')  { escaped = true;  continue; }
            if (c == '"')            { inStr = !inStr;  continue; }
            if (inStr)               continue;

            if (c == '{') { if (depth == 0) jsonStr = "{"; depth++; }
            if (c == '}' && depth > 0 && --depth == 0) {
                const files::json::ParseResult p = files::json::parse(jsonStr);
                if (p.ok) jsonCmds.push_back(p.value);
                jsonStr.clear();
            }
        }

        return jsonCmds;
    }

    static const uint8_t MAX_TURNS = 20;
};



inline Bus bus(
    const std::string& name, 
    const std::vector<Interface*> interfaces
) { 
    if (name.empty())       busError("Bus: ", "Can not create a bus without a name.");
    if (interfaces.empty()) busError(name + ": ", "Can not create a bus without interfaces.");
    return Bus(name, interfaces); 
}

}