#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "LocalModel.hpp"
#include "Quant.hpp"
#include "../files/JSON.hpp"
#include "../math/Random.hpp"

/*
    A small front door onto LocalModel.

    Everything underneath - tokenizers, KV caches, quantisation, sampling,
    chat templates - is machinery that anything embedding a model has to get
    right once and then never wants to see again. This is the once.

        llm::LLM* model = llm::llm("models");

        for (const auto& m : model->available())
            std::cout << m.name << "\n";

        model->use("qwen3-8b");
        std::string reply = model->prompt("What is the capital of France?");

    The object holds a conversation. prompt() appends to it and the KV cache
    carries between calls, so a second prompt() sees the first exchange;
    reset() starts over. That is the behaviour an agent wants - it asks,
    reads a tool result, and asks again - and it is why this is a class
    rather than a free function.

    Switching models frees the old one BEFORE loading the new one. With
    checkpoints measured in tens of gigabytes, holding two at once is the
    difference between working and swapping to disk, so the order matters
    more than it looks.
*/

namespace llm {

namespace fs = std::filesystem;


inline void llmError(const std::string& where, const std::string& info) {
    std::cerr << where << ": " << info << std::endl;
}


/*
    Available:
    - one model directory, as seen from outside. `name` is the folder name
      and is what use() takes; `supported` is whether this build has a spec
      for the architecture, so an unsupported model can be listed and
      explained rather than silently missing.
*/
struct Available {
    std::string name;
    std::string path;
    std::string type;
    bool        supported = false;
};


/*
    Options:
    - everything that changes how a model is loaded or how it answers.
      Defaults are the ones the CLI uses, so behaviour matches what you get
      from bin/release/llm with no flags.
*/
struct Options {
    ai::quant::Type quant       = ai::quant::Type::None;
    size_t          context     = 4096;
    size_t          maxTokens   = 512;
    float           temperature = 0.7f;
    size_t          topK        = 40;
    uint64_t        seed        = 0;
    std::string     system;          // system prompt, when the model takes one
    bool            thinking    = false;

    /*
        stateless: every prompt() is turn one.

        The default keeps the conversation, which is what an agent working
        through a task wants - ask, read a tool result, ask again. Set this
        when each call has to be independent, and the KV cache is cleared
        before every turn so nothing from an earlier one can reach the
        answer. It costs a fresh prefill each time, which is the price of
        the guarantee rather than an oversight.
    */
    bool            stateless   = false;
};


class LLM {
public:
    LLM(const LLM&)            = delete;
    LLM& operator=(const LLM&) = delete;

    /*
        available():
        - every directory under the configured roots holding a config.json,
          with the architecture it declares. Scanned once at construction and
          on refresh(), since the answer only changes when the disk does.
    */
    const std::vector<Available>& available() const { return m_available; }

    void refresh() { scan(); }

    bool loaded() const { return m_model != nullptr; }

    /* the name passed to use(), empty when nothing is loaded */
    const std::string& current() const { return m_current; }

    Options& options() { return m_options; }
    const Options& options() const { return m_options; }


    /*
        use(name):
        - loads a model by directory name, or by path if the name is not one
          of the scanned entries.
        - the currently loaded model is destroyed FIRST. Loading the new one
          on top would mean both resident at once, which for anything past a
          few billion parameters is what makes the machine unusable rather
          than merely slow.
    */
    bool use(const std::string& name) { return use(name, m_options); }

    bool use(const std::string& name, const Options& options) {
        const fs::path dir = resolve(name);
        if (dir.empty()) {
            llmError("llm::use", "no model called \"" + name + "\"");
            return false;
        }

        unload();
        m_options = options;

        auto fresh = std::make_unique<ai::LocalModel>();
        if (!fresh->load(dir, {}, m_options.quant)) {
            llmError("llm::use", name + ": " + fresh->error);
            return false;
        }

        m_model = std::move(fresh);
        m_current = name;

        m_cache.resize(m_model->cfg, std::min(m_options.context,
                                              m_model->cfg.maxPositions));
        m_logits.assign(m_model->cfg.vocab, 0.0f);
        m_turns = 0;

        return true;
    }


    /*
        unload():
        - drops the model and everything sized against it. Safe to call when
          nothing is loaded.
    */
    void unload() {
        m_model.reset();
        m_cache = ai::LocalCache{};
        m_logits.clear();
        m_current.clear();
        m_turns = 0;
    }


    /*
        reset():
        - forgets the conversation without unloading. The next prompt() is
          treated as the first, so the chat template emits its opening
          sequence and the system prompt again.
    */
    void reset() {
        m_cache.reset();
        m_turns = 0;
    }


    /*
        prompt(input):
        - one turn. Returns what the model said, with the chat template's
          own markers stripped; on any failure it returns an empty string and
          reports through llmError, so a caller that only wants text can
          treat empty as "nothing useful came back".
        - the conversation continues across calls. When the context fills,
          the cache is reset and the turn is re-encoded from scratch rather
          than silently truncating, since a half-dropped history produces
          confident nonsense.
    */
    std::string prompt(const std::string& input) {
        if (!m_model) { llmError("llm::prompt", "no model loaded"); return {}; }

        if (m_options.stateless) reset();

        const size_t limit = m_cache.capacity;

        const auto encode = [&] {
            const bool first = (m_turns == 0);
            const std::string text = m_model->hasChatTemplate()
                ? m_model->chatPrompt(input, m_options.thinking,
                                      first ? m_options.system : std::string(), first)
                : input;
            return m_model->tokenizer.encode(text, first);
        };

        std::vector<size_t> ids = encode();
        if (ids.empty()) { llmError("llm::prompt", "the prompt encoded to nothing"); return {}; }

        /* +8 leaves room for the handful of tokens a reply needs before it
           is worth starting at all */
        if (m_cache.length + ids.size() + 8 >= limit) {
            reset();
            ids = encode();
        }
        if (ids.size() + 8 >= limit) {
            llmError("llm::prompt", "the prompt alone does not fit in the context");
            return {};
        }

        m_model->forward(ids.data(), ids.size(), m_cache.length, m_cache, m_logits.data());
        ++m_turns;

        std::string out;
        for (size_t i = 0; i < m_options.maxTokens && m_cache.length < limit; ++i) {
            const size_t next = ai::LocalModel::sampleTopK(
                m_logits.data(), m_model->cfg.vocab,
                m_options.temperature, m_options.topK, m_rng);

            if (m_model->isStop(next)) break;

            /* only the very first token of a reply may have its leading
               space stripped - see Tokenizer::decode */
            out += m_model->tokenizer.decode({next}, out.empty());

            m_model->forward(&next, 1, m_cache.length, m_cache, m_logits.data());
        }

        return out;
    }


    /*
        once(input):
        - one self-contained turn, whatever `stateless` happens to be set to.

        Several callers can share one LLM - several agents, say - and each
        needs its own answer without the others' turns reaching it. Flipping
        the shared option around the call would leave it flipped for whoever
        runs next, so this saves and restores it instead.
    */
    std::string once(const std::string& input) {
        const bool was = m_options.stateless;
        m_options.stateless = true;
        std::string out = prompt(input);
        m_options.stateless = was;
        return out;
    }


private:
    LLM(const std::vector<fs::path>& roots) : m_roots(roots), m_rng(0) { scan(); }

    friend LLM* llm(const std::string& root);
    friend LLM* llm(const std::vector<std::string>& roots);


    /*
        scan():
        - a directory is a model when it holds a config.json. The
          architecture is read out of it so an unsupported checkpoint can be
          listed and named rather than just failing at use().
    */
    void scan() {
        m_available.clear();
        std::error_code ec;

        for (const fs::path& root : m_roots) {
            if (!fs::exists(root, ec)) continue;

            for (const auto& entry : fs::directory_iterator(root, ec)) {
                if (!entry.is_directory()) continue;
                if (!fs::exists(entry.path() / "config.json", ec)) continue;

                Available found;
                found.name = entry.path().filename().string();
                found.path = entry.path().generic_string();
                found.type = "?";

                const files::json::ParseResult parsed =
                    files::json::parseFile(entry.path() / "config.json");
                if (parsed.ok) {
                    const files::json::Value& v = parsed.value;
                    found.type = v["model_type"].asString(
                        v["text_config"]["model_type"].asString("?"));
                }

                found.supported = ai::specFor(found.type) != nullptr;
                m_available.push_back(found);
            }
        }
    }


    /* a scanned name first, then a path, so either spelling works */
    fs::path resolve(const std::string& name) const {
        for (const Available& a : m_available)
            if (a.name == name) return fs::path(a.path);

        std::error_code ec;
        const fs::path direct(name);
        if (fs::exists(direct / "config.json", ec)) return direct;

        return {};
    }


    std::vector<fs::path>               m_roots;
    std::vector<Available>              m_available;

    std::unique_ptr<ai::LocalModel>     m_model;
    ai::LocalCache                      m_cache;
    std::vector<float>                  m_logits;
    std::string                         m_current;
    size_t                              m_turns = 0;

    Options                             m_options;
    math::random::Random                m_rng;
};


// LLM factory functions
inline LLM* llm(const std::string& root) {
    if (root.empty()) llmError("llm", "no model root given");
    return new LLM(std::vector<fs::path>{ fs::path(root) });
}

inline LLM* llm(const std::vector<std::string>& roots) {
    if (roots.empty()) llmError("llm", "no model roots given");

    std::vector<fs::path> paths;
    paths.reserve(roots.size());
    for (const std::string& r : roots) paths.emplace_back(r);

    return new LLM(paths);
}

} // !namespace llm
