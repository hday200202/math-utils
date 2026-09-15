#pragma once

#include <queue>

#include "ai/LLM.hpp"

namespace omnibot {

class Agent {
public:
    Agent(
        const std::string& name, 
        const std::string& prompt,
        const std::string& cmdList,
        llm::LLM* model
    ) : m_name(name), 
        m_prompt(prompt), 
        m_cmdList(cmdList), 
        m_localModel(model) {}

    
    const std::string& getName() const { return m_name; }
    const std::string& transcript() const { return m_transcript; }
    void clear() { m_transcript.clear(); }

    std::string send(const std::string& input) {
        std::string full =
            m_prompt + "\n\n" + m_cmdList + "\n\n" +
            m_transcript + "\nUser: " + input;
        
        std::string out = m_localModel->once(full);
        m_transcript += "\nUser: " + input + "\nAssistant: " + out;

        if (m_transcript.size() > MAX_TRANSCRIPT) {
            const size_t cut = m_transcript.find("\nUser: ", m_transcript.size() - MAX_TRANSCRIPT);
            m_transcript = (cut == std::string::npos) ? std::string() : m_transcript.substr(cut);
        }

        return out;
    }

    static constexpr size_t MAX_TRANSCRIPT = 8000;

private:
    std::string m_name;
    std::string m_prompt;
    std::string m_cmdList;
    std::string m_transcript;

    llm::LLM* m_localModel = nullptr;
};

}