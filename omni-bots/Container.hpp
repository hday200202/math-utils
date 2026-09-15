#pragma once

#include "Agent.hpp"
#include "components/Interface.hpp"
#include "components/CommandBus.hpp"

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


inline void busError(
    const std::string& busName,
    const std::string& info
) { std::cerr << busName << ": " << info << std::endl; }


inline void containerError(
    const std::string& funcName,
    const std::string& info
) { std::cerr << funcName << ": " << info << std::endl; }


class Container {
public:
    Container(const std::vector<Interface*>& interfaces) {
        CommandBus* mainBus = new CommandBus("Main Bus");
        for (auto* i : interfaces) {
            for (auto& cmd : i->commands())
                if (!mainBus->makeRoute(extractFirstToken(cmd), i))
                    busError("Main Bus", "Interface valid command collision.");
            m_interfaces.emplace_back(i);
        }
        m_buses.emplace_back(mainBus);
    }

    Container(const std::vector<Bus>& buses) {
        for (auto& b : buses) {
            CommandBus* newBus = new CommandBus(b.getName());
            for (auto* i : b.getInterfaces()) {
                for (auto& cmd : i->commands())
                    if (!newBus->makeRoute(extractFirstToken(cmd), i))
                        busError(b.getName(), "Interface valid command collision.");
                
                bool inVec = false;
                for (auto& ptr : m_interfaces)
                    if (ptr.get() == i) inVec = true;

                if (!inVec) m_interfaces.emplace_back(i);
            }
            m_buses.emplace_back(newBus);
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

        CommandBus* cmdBus = nullptr;
        for (auto& b : m_buses)
            if (b->getName() == bus) cmdBus = b.get();

        if (!cmdBus) containerError(
            "Spawn Agent - Multi-Bus",
            "Could not match \"" + bus + "\" with any available buses."
        );

        return *this;
    }


    std::string window() const {
        std::string out = "";
        for (const auto& bus : m_buses)
            out += "[ " + bus->getName() + " ]" + "\n" 
                + bus->window() + "\n\n";
        return out;
    }

private:
    std::vector<std::unique_ptr<Interface>>     m_interfaces;
    std::vector<std::unique_ptr<CommandBus>>    m_buses;
    std::vector<std::unique_ptr<Agent>>         m_agents;
    std::unordered_map<Agent*, CommandBus*>     m_cmdBusRouter;

    std::string cmdsForAgent(Agent* agent) const {
        std::string out = "done\n";
        for (auto& cmd : m_cmdBusRouter.at(agent)->commands())
            out += cmd;
        return out;
    }
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