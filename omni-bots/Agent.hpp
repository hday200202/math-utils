#pragma once

#include <queue>
#include <chrono>
#include <iostream>

#include "Model.hpp"

namespace omnibot {

class Agent {
public:
    Agent(
        const std::string& name, 
        const std::string& prompt,
        const std::string& cmdList,
        Model* model
    ) : m_name(name), 
        m_prompt(prompt), 
        m_cmdList(cmdList), 
        m_localModel(model) {}

    
    const std::string& getName() const { return m_name; }
    const std::string& transcript() const { return m_transcript; }

    /*
        clear():
        - forgets the conversation on both sides. Dropping the transcript
          without resetting the model would leave a stateful agent still
          carrying every earlier turn in the model's memory while believing
          it had started over.
    */
    void clear() {
        m_transcript.clear();
        m_turns = 0;
        if (m_stateful && m_localModel) m_localModel->reset();
    }

    bool stateful() const { return m_stateful; }

    /*
        setStateful(on):
        - only one agent may hold the model's conversation at a time, so this
          is the Container's call, not the agent's. Turning it on re-briefs
          from turn zero, since the model has no idea what this agent has
          been doing.
    */
    void setStateful(const bool on) {
        if (on == m_stateful) return;
        m_stateful = on;
        m_turns    = 0;

        m_system = m_stateful ? m_prompt + "\n\n" + m_cmdList : std::string();

        if (!m_localModel) return;

        m_localModel->setSystem(m_system);
        if (m_stateful) m_localModel->reset();
    }

    /* the standing briefing, empty when stateless - see send()'s trace */
    const std::string& system() const { return m_system; }

    /*
        send(input):
        - a stateful agent has the model to itself, so the turns it has
          already taken are still in the model's memory and a turn is only
          the new input. Its task and command list are the model's system
          prompt, set once in setStateful. Nothing is re-encoded, and the
          model sees a real exchange rather than a transcript quoted inside
          a single turn - which is most of why a small model does better
          here.
        - a stateless agent is sharing, so every turn has to carry its own
          briefing and its own history. m_transcript is what it carries.

        Either way the transcript is recorded. A stateful agent does not send
        it, but it is still what transcript() shows and what survives a model
        swap, so it is not bookkeeping for its own sake.
    */
    std::string send(const std::string& input) {
        std::string full =
            m_stateful
                ? input
                : m_prompt + "\n\n" + m_cmdList + "\n\n" +
                  m_transcript + "\nUser: " + input;

        /*
            The prompt goes out BEFORE the call, not alongside the reply.
            A turn can take minutes, and the whole question while waiting is
            what the model was handed - printing it afterwards answers that
            only once the wait is already over.

            The system block is printed too, and it has to be: a stateful
            turn sends only the new input, so printing `full` alone shows
            "Begin." and nothing else, which is not what the model is
            reading. It is reading this as well, every turn, whether or not
            this turn is the one that transmitted it.
        */
        if (s_trace) {
            std::cout << "\n---------------- [" << m_name << "] "
                      << (m_stateful ? "stateful" : "stateless") << " turn "
                      << m_turns << " ----------------\n";

            if (!m_system.empty())
                std::cout << "[system, " << m_system.size() << " chars"
                          << (m_turns == 0 ? ", sent this turn"
                                           : ", already in context")
                          << "]\n" << m_system << "\n\n";

            std::cout << "[input, " << full.size() << " chars]\n"
                      << full
                      << "\n---------------- end prompt ----------------"
                      << std::endl;
        }

        const auto t0 = std::chrono::steady_clock::now();
        std::string out = m_stateful ? m_localModel->prompt(full)
                                     : m_localModel->once(full);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        ++m_turns;

        if (s_trace)
            std::cout << "---------------- reply, " << ms << "ms, "
                      << out.size() << " chars ----------------\n"
                      << out
                      << "\n---------------- end reply ----------------"
                      << std::endl;

        m_transcript += "\nUser: " + input + "\nAssistant: " + out;

        if (m_transcript.size() > MAX_TRANSCRIPT) {
            const size_t cut = m_transcript.find("\nUser: ", m_transcript.size() - MAX_TRANSCRIPT);
            m_transcript = (cut == std::string::npos) ? std::string() : m_transcript.substr(cut);
        }

        return out;
    }

    static constexpr size_t MAX_TRANSCRIPT = 1000;

    /*
        trace(on):
        - echo every turn: what the agent was given, what the model said, and
          how long it took. A tick is many turns long and prints nothing
          until it ends, so without this a slow model and a hung one look
          exactly the same from the console.
    */
    static void trace(const bool on) { s_trace = on; }

private:
    std::string m_name;
    std::string m_prompt;
    std::string m_cmdList;
    std::string m_transcript;

    Model* m_localModel = nullptr;

    std::string m_system;        // the briefing held by the model, stateful only

    bool   m_stateful = false;   // set by Container; see setStateful
    size_t m_turns    = 0;       // since the last briefing

    inline static bool s_trace = false;
};

}