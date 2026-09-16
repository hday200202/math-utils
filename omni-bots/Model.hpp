#pragma once

#include "components/Connector.hpp"
#include "ai/LLM.hpp"

namespace omnibot {

[[noreturn]] inline void LLMError(
    const std::string& funcName,
    const std::string& info
) { throw Error(funcName, info); }

/*
    Model:
    - two ways to ask, because there are two situations.

      prompt() continues a conversation. The turns before it are still in the
      model's own memory, so the caller sends only what is new. That is much
      the better deal - nothing is re-encoded, and the model sees real turns
      instead of a transcript pasted into one of them - but there is only one
      such memory, so exactly one caller may use it.

      once() is self-contained. Whatever it is handed is the entire
      conversation, and nothing from an earlier call can reach the answer.
      Slower, since every turn re-encodes from the start, and it is on the
      caller to include whatever history matters. That is the price of
      several agents sharing one model.
*/
class Model {
public:
    virtual ~Model() = default;

    virtual std::string prompt(const std::string& input) = 0;
    virtual std::string once(const std::string& input)   = 0;

    /* forget the continuing conversation; no effect on once() */
    virtual void reset() = 0;

    /*
        setSystem(text):
        - the standing briefing, re-sent whenever a conversation starts over.

        This is not the same as sending the briefing as the first turn. A
        continuing conversation is bounded, and when it fills, the model
        starts a new one - at which point anything said only in turn one is
        gone, and the agent carries on answering with no idea what its task
        was. Held here, it comes back with the new conversation.
    */
    virtual void setSystem(const std::string& text) = 0;
};

class LocalLLM : public Model {
public:
    /*
        The default is Q4, not the library's own Type::None.

        None means bf16 - two bytes a weight, so an 8b checkpoint moves 16GB
        through memory for every token generated. On CPU that is the entire
        cost of a turn; nothing else in the loop is within an order of
        magnitude of it. Q4 is 0.625 bytes a weight, and an agent picking
        between a dozen command names is not the workload where the
        difference in quality shows up.

        Pass Type::None if you are chasing an answer rather than a loop.
    */
    LocalLLM(
        const std::string& root,
        const std::string& name,
        const ai::quant::Type quant = ai::quant::Type::Q4,
        const bool thinking = false
    ) : m_llm(llm::llm(root)) {
        llm::Options opts;
        opts.quant      = quant;
        opts.thinking   = thinking;

        /*
            The budget has to clear a whole batch, and the reasoning block
            when there is one.

            This is a ceiling, not a target - generation stops at the model's
            own stop token - so an unused allowance costs nothing, while too
            small a one truncates mid-json and arrives looking exactly like a
            model that cannot follow the format. A reply carrying four
            commands is already near 128, which is why that number, sized for
            one command a turn, quietly caps the batching it is supposed to
            allow.
        */
        opts.maxTokens  = thinking ? 1024 : 512;

        if (!m_llm->use(name, opts))
            LLMError("LocalLLM", "could not load \"" + name + "\"");
    }

    std::string prompt(const std::string& input) override { return m_llm->prompt(input); }
    std::string once(const std::string& input)   override { return m_llm->once(input); }

    void reset() override { m_llm->reset(); }

    void setSystem(const std::string& text) override { m_llm->options().system = text; }

private:
    std::unique_ptr<llm::LLM> m_llm;
};

}