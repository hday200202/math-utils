#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../files/File.hpp"
#include "../files/JSON.hpp"

/*
    GPT-2's byte-level byte-pair-encoding tokenizer.

    Text is not split into characters or words. Its raw UTF-8 BYTES are first
    mapped into a private printable-unicode alphabet, then adjacent pairs are
    repeatedly merged in the priority order given by merges.txt until no
    merge applies. The byte mapping exists so every possible byte has a
    printable stand-in and the merge table never has to hold control
    characters or invalid UTF-8 - which is also why the vocabulary is full of
    odd-looking characters like the one standing in for a space.

    Two files are needed, both shipped alongside the weights:
        vocab.json   the mapped-alphabet string for each token id
        merges.txt   the merge rules, most-preferred first
*/

namespace ai {

namespace fs = std::filesystem;


/* ========================================================================= //
    UTF-8 helpers
// ========================================================================= */

inline void utf8Encode(uint32_t cp, std::string& out) {
    if (cp <= 0x7F) {
        out += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/* returns the code point starting at i and advances i past it */
inline uint32_t utf8Next(std::string_view s, size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);

    if (c < 0x80)        { ++i; return c; }
    if ((c >> 5) == 0x6) { uint32_t cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu); i += 2; return cp; }
    if ((c >> 4) == 0xE) {
        uint32_t cp = ((c & 0x0Fu) << 12)
                    | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6)
                    |  (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        i += 3; return cp;
    }
    uint32_t cp = ((c & 0x07u) << 18)
                | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12)
                | ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6)
                |  (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
    i += 4; return cp;
}


/* ========================================================================= //
    Tokenizer
// ========================================================================= */

/*
    Which pre-tokenisation rule set to apply. Both models use the same
    byte-level alphabet and the same merge algorithm; they differ only in how
    the text is chopped up before merging, and that difference changes every
    id produced.
*/
/*
    Two entirely different alphabets, not just two splitting rules.

    GPT2 and Qwen are BYTE LEVEL: the raw bytes are mapped into a private
    printable alphabet and the vocabulary is written in that alphabet, which
    is why it is full of characters like the one standing in for a space.

    Metaspace is what SentencePiece-derived tokenizers use - Mistral, Llama-2,
    and most things predating the byte-level convention. The vocabulary holds
    ordinary UTF-8, a space is written as U+2581, and any character missing
    from the vocabulary falls back to one <0xNN> token per byte.

    Decoding one with the other's rules does not fail loudly. It drops every
    space and mangles the prompt, and the model answers a question nobody
    asked.
*/
enum class PreTokenizer { GPT2, Qwen, Metaspace };


struct Tokenizer {
    std::unordered_map<std::string, int32_t> encoder;   // mapped string -> id
    std::vector<std::string>                 decoder;   // id -> mapped string
    std::unordered_map<std::string, int32_t> ranks;     // "left right" -> priority

    PreTokenizer                             style = PreTokenizer::GPT2;

    /*
        Tokens the file's post_processor wraps every sequence in - in
        practice a BOS marker, and for some models a trailing one too.
        These are NOT optional decoration. Mistral-Small-24B without its
        <s> emits nothing but whitespace; the 7B tolerates the omission,
        which is exactly the kind of difference that makes hardcoding per
        model a trap.
    */
    std::vector<size_t>                      prefixIds, suffixIds;

    /* Metaspace only: U+2581 as UTF-8, and whether a leading one is added */
    std::string                              metaspaceMark = "\xe2\x96\x81";
    bool                                     prependMark = true;
    bool                                     byteFallback = false;

    /* added_tokens live outside the BPE vocabulary and are matched as literal
       text before any splitting, so <|im_start|> stays a single token */
    std::vector<std::pair<std::string, size_t>> specials;
    std::unordered_map<size_t, std::string>     specialById;

    std::string byteToUnicode[256];                     // UTF-8 of the stand-in
    std::unordered_map<std::string, uint8_t> unicodeToByte;

    mutable std::unordered_map<std::string, std::vector<std::string>> cache;

    std::string error;


    /*
        bytesToUnicode():
        - the GPT-2 byte <-> printable-code-point bijection. The printable
          ASCII and Latin-1 ranges map to themselves; every remaining byte
          gets a code point from 256 upwards, in byte order
    */
    void buildByteMap() {
        std::vector<int> direct;
        for (int b = '!'; b <= '~'; ++b) direct.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) direct.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) direct.push_back(b);

        std::vector<uint32_t> points(direct.begin(), direct.end());

        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (int d : direct) if (d == b) { found = true; break; }
            if (found) continue;
            direct.push_back(b);
            points.push_back(static_cast<uint32_t>(256 + extra));
            ++extra;
        }

        for (size_t i = 0; i < direct.size(); ++i) {
            std::string encoded;
            utf8Encode(points[i], encoded);
            byteToUnicode[direct[i]] = encoded;
            unicodeToByte[encoded] = static_cast<uint8_t>(direct[i]);
        }
    }


    bool load(const fs::path& vocabPath, const fs::path& mergesPath) {
        error.clear();
        buildByteMap();

        const files::json::ParseResult vocab = files::json::parseFile(vocabPath);
        if (!vocab.ok) {
            error = "vocab.json: " + vocab.error;
            return false;
        }
        if (!vocab.value.isObject()) {
            error = "vocab.json is not an object";
            return false;
        }

        decoder.assign(vocab.value.size(), std::string());
        for (const files::json::Member& m : vocab.value.object) {
            const int32_t id = static_cast<int32_t>(m.second.asInt(-1));
            if (id < 0) continue;
            encoder[m.first] = id;
            if (static_cast<size_t>(id) >= decoder.size()) decoder.resize(static_cast<size_t>(id) + 1);
            decoder[static_cast<size_t>(id)] = m.first;
        }

        files::File merges;
        if (!merges.open(mergesPath, files::OpenMode::Read)) {
            error = "could not open " + mergesPath.string();
            return false;
        }

        const std::vector<std::string> lines = merges.lines();
        int32_t rank = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            // the first line is a version banner, not a merge
            if (i == 0 && !lines[i].empty() && lines[i][0] == '#') continue;
            if (lines[i].empty()) continue;

            const size_t space = lines[i].find(' ');
            if (space == std::string::npos) continue;

            ranks[lines[i]] = rank++;
        }

        return !encoder.empty() && !ranks.empty();
    }


    /*
        loadJson(tokenizer.json):
        - the HuggingFace single-file tokenizer. model.vocab is the same
          mapped-string to id table vocab.json holds, and model.merges the
          same rule list merges.txt holds, only as pairs rather than lines
    */
    bool loadJson(const fs::path& tokenizerPath) {
        error.clear();
        buildByteMap();
        style = PreTokenizer::Qwen;

        std::vector<std::string> prefixNames, suffixNames;

        const files::json::ParseResult parsed = files::json::parseFile(tokenizerPath);
        if (!parsed.ok) { error = "tokenizer.json: " + parsed.error; return false; }

        const files::json::Value& model = parsed.value["model"];
        const files::json::Value& vocab = model["vocab"];
        if (!vocab.isObject()) { error = "tokenizer.json: no model.vocab"; return false; }

        decoder.assign(vocab.size(), std::string());
        for (const files::json::Member& m : vocab.object) {
            const int32_t id = static_cast<int32_t>(m.second.asInt(-1));
            if (id < 0) continue;
            encoder[m.first] = id;
            if (static_cast<size_t>(id) >= decoder.size()) decoder.resize(static_cast<size_t>(id) + 1);
            decoder[static_cast<size_t>(id)] = m.first;
        }

        /*
            which alphabet this vocabulary is written in. Getting it wrong is
            silent: every space disappears and the ids are meaningless
        */
        const files::json::Value& pre = parsed.value["pre_tokenizer"];
        if (pre["type"].asString() == "Metaspace") {
            style = PreTokenizer::Metaspace;
            if (pre["replacement"].isString()) metaspaceMark = pre["replacement"].asString();
            prependMark = pre["prepend_scheme"].asString("first") != "never";
        }
        byteFallback = model["byte_fallback"].asBool(false);

        /*
            post_processor says what wraps a sequence. TemplateProcessing's
            "single" is a list of SpecialToken and Sequence entries; anything
            before the Sequence is a prefix, anything after is a suffix
        */
        const files::json::Value& post = parsed.value["post_processor"];
        if (post["type"].asString() == "TemplateProcessing") {
            const files::json::Value& single = post["single"];
            bool seenSequence = false;

            for (size_t i = 0; i < single.size(); ++i) {
                const files::json::Value& item = single[i];

                if (item.contains("Sequence")) { seenSequence = true; continue; }
                if (!item.contains("SpecialToken")) continue;

                const std::string name = item["SpecialToken"]["id"].asString();
                if (name.empty()) continue;

                // resolved after added_tokens are read, below
                (seenSequence ? suffixNames : prefixNames).push_back(name);
            }
        }

        const files::json::Value& merges = model["merges"];
        if (!merges.isArray()) { error = "tokenizer.json: no model.merges"; return false; }

        for (size_t i = 0; i < merges.size(); ++i) {
            const files::json::Value& m = merges[i];
            if (m.isArray() && m.size() == 2)
                ranks[m[0].asString() + " " + m[1].asString()] = static_cast<int32_t>(i);
            else if (m.isString())
                ranks[m.asString()] = static_cast<int32_t>(i);
        }

        const files::json::Value& added = parsed.value["added_tokens"];
        for (size_t i = 0; i < added.size(); ++i) {
            const std::string content = added[i]["content"].asString();
            const size_t id = added[i]["id"].asSize();
            if (content.empty()) continue;
            specials.emplace_back(content, id);
            specialById[id] = content;
            if (id >= decoder.size()) decoder.resize(id + 1);
        }

        // resolve the wrapper names now that added_tokens have been read
        for (const std::string& n : prefixNames)
            if (hasSpecial(n)) prefixIds.push_back(idFor(n));
        for (const std::string& n : suffixNames)
            if (hasSpecial(n)) suffixIds.push_back(idFor(n));

        return !encoder.empty() && !ranks.empty();
    }


    size_t vocabSize() const { return decoder.size(); }

    size_t idFor(const std::string& special) const {
        for (const auto& s : specials) if (s.first == special) return s.second;
        return 0;
    }

    /*
        hasSpecial(name):
        - separate from idFor because id 0 is a real token in some
          vocabularies - Mistral puts <unk> there - so "returned 0" cannot
          double as "not present"
    */
    bool hasSpecial(const std::string& name) const {
        for (const auto& s : specials) if (s.first == name) return true;
        return false;
    }


    /* ===================================================================== //
        Pre-tokenisation

        GPT-2 splits the text before BPE runs, so a merge can never straddle a
        word boundary. The reference uses a unicode-property regex; this is
        the ASCII-equivalent of the same alternation, applied in the same
        order because the order is what decides which rule wins.
    // ===================================================================== */

    static bool isLetter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    static bool isDigit(char c)  { return c >= '0' && c <= '9'; }
    static bool isSpace(char c)  { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

    /*
        preTokenizeQwen(text):
        - the ASCII equivalent of Qwen's alternation, applied in the same
          order because the order decides which rule wins.
        - two differences from GPT-2 matter. Contractions match case
          insensitively, and DIGITS ARE SPLIT ONE AT A TIME rather than in
          runs, so 1234 becomes four tokens. The prefix a word may absorb is
          also any non-letter non-digit, not only a space
    */
    static std::vector<std::string> preTokenizeQwen(std::string_view text) {
        static const char* CONTRACTIONS[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

        auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
        auto isBreak = [](char c) { return c == '\r' || c == '\n'; };

        std::vector<std::string> out;
        size_t i = 0;

        while (i < text.size()) {
            bool matched = false;

            for (const char* c : CONTRACTIONS) {
                const size_t n = std::char_traits<char>::length(c);
                if (text.size() - i < n) continue;
                bool same = true;
                for (size_t k = 0; k < n; ++k)
                    if (lower(text[i + k]) != c[k]) { same = false; break; }
                if (same) { out.emplace_back(text.substr(i, n)); i += n; matched = true; break; }
            }
            if (matched) continue;

            // an optional single non-letter, non-digit, non-newline prefix
            size_t j = i;
            if (!isBreak(text[j]) && !isLetter(text[j]) && !isDigit(text[j])) ++j;
            if (j < text.size() && isLetter(text[j])) {
                while (j < text.size() && isLetter(text[j])) ++j;
                out.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }

            if (isDigit(text[i])) { out.emplace_back(text.substr(i, 1)); ++i; continue; }

            j = i;
            if (text[j] == ' ') ++j;
            if (j < text.size() && !isSpace(text[j]) && !isLetter(text[j]) && !isDigit(text[j])) {
                while (j < text.size() && !isSpace(text[j]) && !isLetter(text[j]) && !isDigit(text[j])) ++j;
                while (j < text.size() && isBreak(text[j])) ++j;
                out.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }

            // whitespace ending in at least one line break
            j = i;
            while (j < text.size() && isSpace(text[j]) && !isBreak(text[j])) ++j;
            if (j < text.size() && isBreak(text[j])) {
                while (j < text.size() && isBreak(text[j])) ++j;
                out.emplace_back(text.substr(i, j - i));
                i = j;
                continue;
            }

            // the trailing-whitespace rules, as in GPT-2
            size_t k = i;
            while (k < text.size() && isSpace(text[k])) ++k;
            if (k == text.size()) { out.emplace_back(text.substr(i, k - i)); i = k; continue; }
            if (k - i > 1)        { out.emplace_back(text.substr(i, k - i - 1)); i = k - 1; continue; }

            out.emplace_back(text.substr(i, 1));
            i += 1;
        }

        return out;
    }


    static std::vector<std::string> preTokenizeGPT2(std::string_view text) {
        static const char* CONTRACTIONS[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

        std::vector<std::string> out;
        size_t i = 0;

        while (i < text.size()) {
            bool matched = false;

            // 's 't 're 've 'm 'll 'd
            for (const char* c : CONTRACTIONS) {
                const size_t n = std::char_traits<char>::length(c);
                if (text.size() - i >= n && text.compare(i, n, c) == 0) {
                    out.emplace_back(text.substr(i, n));
                    i += n;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;

            // an optional single leading space belongs to the token after it,
            // which is why " the" and "the" are different tokens
            const size_t start = i;
            size_t j = i;
            if (j < text.size() && text[j] == ' ') ++j;

            if (j < text.size() && isLetter(text[j])) {
                while (j < text.size() && isLetter(text[j])) ++j;
            } else if (j < text.size() && isDigit(text[j])) {
                while (j < text.size() && isDigit(text[j])) ++j;
            } else if (j < text.size() && !isSpace(text[j]) && !isLetter(text[j]) && !isDigit(text[j])) {
                while (j < text.size() && !isSpace(text[j]) && !isLetter(text[j]) && !isDigit(text[j])) ++j;
            } else {
                /* a whitespace run. The reference pattern hands every space
                   but the last to a token of its own and leaves the last one
                   to be absorbed by the word that follows */
                size_t k = i;
                while (k < text.size() && isSpace(text[k])) ++k;

                if (k == text.size()) { out.emplace_back(text.substr(i, k - i)); i = k; continue; }
                if (k - i > 1)        { out.emplace_back(text.substr(i, k - i - 1)); i = k - 1; continue; }

                out.emplace_back(text.substr(i, 1));
                i += 1;
                continue;
            }

            out.emplace_back(text.substr(start, j - start));
            i = j;
        }

        return out;
    }


    /* ===================================================================== //
        The merge loop
    // ===================================================================== */

    /*
        bpe(token):
        - repeatedly merges the adjacent pair with the best (lowest) rank
          until no pair in the word appears in the table. Ties never happen:
          every pair has a distinct rank
    */
    const std::vector<std::string>& bpe(const std::string& token) const {
        auto cached = cache.find(token);
        if (cached != cache.end()) return cached->second;

        std::vector<std::string> word;
        for (size_t i = 0; i < token.size(); ) {
            const size_t begin = i;
            utf8Next(token, i);
            word.emplace_back(token.substr(begin, i - begin));
        }

        while (word.size() > 1) {
            int32_t bestRank = -1;
            size_t bestAt = 0;

            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto found = ranks.find(word[i] + " " + word[i + 1]);
                if (found == ranks.end()) continue;
                if (bestRank < 0 || found->second < bestRank) {
                    bestRank = found->second;
                    bestAt = i;
                }
            }

            if (bestRank < 0) break;

            std::vector<std::string> merged;
            merged.reserve(word.size() - 1);
            for (size_t i = 0; i < word.size(); ) {
                if (i == bestAt) {
                    merged.push_back(word[i] + word[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(word[i]);
                    i += 1;
                }
            }
            word = std::move(merged);
        }

        return cache.emplace(token, std::move(word)).first->second;
    }


    std::vector<std::string> preTokenize(std::string_view text) const {
        return style == PreTokenizer::Qwen ? preTokenizeQwen(text) : preTokenizeGPT2(text);
    }


    /*
        encode(text):
        - special tokens are found as literal substrings first and the text
          between them encoded normally, so a chat template's <|im_start|>
          survives rather than being merged into what surrounds it
    */
    /*
        encode(text, addSpecials):
        - addSpecials wraps the result in whatever the file's post_processor
          declares, which is what HuggingFace does by default. It must be
          false for anything continuing an existing sequence - a second
          conversation turn appended to a KV cache already has its BOS
    */
    std::vector<size_t> encode(std::string_view text, bool addSpecials = true) const {
        std::vector<size_t> ids;

        if (addSpecials)
            ids.insert(ids.end(), prefixIds.begin(), prefixIds.end());

        encodeBody(text, ids);

        if (addSpecials)
            ids.insert(ids.end(), suffixIds.begin(), suffixIds.end());

        return ids;
    }


    void encodeBody(std::string_view text, std::vector<size_t>& ids) const {
        if (!specials.empty()) {
            size_t cursor = 0;
            while (cursor < text.size()) {
                size_t at = std::string_view::npos;
                size_t which = 0;

                for (size_t s = 0; s < specials.size(); ++s) {
                    const size_t found = text.find(specials[s].first, cursor);
                    if (found < at) { at = found; which = s; }
                }
                if (at == std::string_view::npos) break;

                if (at > cursor) encodePlain(text.substr(cursor, at - cursor), ids);
                ids.push_back(specials[which].second);
                cursor = at + specials[which].first.size();
            }

            if (cursor < text.size()) encodePlain(text.substr(cursor), ids);
            return;
        }

        encodePlain(text, ids);
    }


    /*
        encodeMetaspace(text, ids):
        - spaces become U+2581 and the whole string is fed to BPE in one
          piece, because this pre-tokenizer has split=false. There is no byte
          mapping: the vocabulary is plain UTF-8
        - a piece the vocabulary does not contain is emitted as one <0xNN>
          token per byte, which is what byte_fallback means
    */
    void encodeMetaspace(std::string_view text, std::vector<size_t>& ids) const {
        /* prepend_scheme "first" adds the mark unconditionally, and the
           decoder unconditionally strips one leading space to undo it.
           Adding it only when the text does not already begin with a space
           breaks that symmetry and eats a real leading space */
        std::string marked;
        if (prependMark) marked += metaspaceMark;

        for (char c : text) {
            if (c == ' ') marked += metaspaceMark;
            else          marked += c;
        }

        for (const std::string& piece : bpe(marked)) {
            auto found = encoder.find(piece);
            if (found != encoder.end()) { ids.push_back(static_cast<size_t>(found->second)); continue; }

            if (!byteFallback) continue;

            static const char* HEX = "0123456789ABCDEF";
            for (unsigned char b : piece) {
                std::string name = "<0x";
                name += HEX[(b >> 4) & 0xF];
                name += HEX[b & 0xF];
                name += '>';
                auto byteToken = encoder.find(name);
                if (byteToken != encoder.end())
                    ids.push_back(static_cast<size_t>(byteToken->second));
            }
        }
    }


    void encodePlain(std::string_view text, std::vector<size_t>& ids) const {
        if (style == PreTokenizer::Metaspace) { encodeMetaspace(text, ids); return; }

        for (const std::string& chunk : preTokenize(text)) {
            std::string mapped;
            for (char c : chunk) mapped += byteToUnicode[static_cast<unsigned char>(c)];

            for (const std::string& piece : bpe(mapped)) {
                auto found = encoder.find(piece);
                if (found != encoder.end()) ids.push_back(static_cast<size_t>(found->second));
            }
        }
    }


    /* unmapBytes(mapped): printable stand-ins back to the raw bytes */
    std::string unmapBytes(const std::string& mapped) const {
        std::string out;
        for (size_t i = 0; i < mapped.size(); ) {
            const size_t begin = i;
            utf8Next(mapped, i);
            const std::string symbol = mapped.substr(begin, i - begin);

            auto found = unicodeToByte.find(symbol);
            if (found != unicodeToByte.end()) out += static_cast<char>(found->second);
        }
        return out;
    }


    /*
        decode(ids):
        - a special token's text is literal rather than byte-mapped, so the
          run of ordinary tokens before it has to be flushed first
    */
    /*
        decodeMetaspace(ids):
        - the reverse of the decoder chain the file declares: replace U+2581
          with a space, turn <0xNN> back into a raw byte, concatenate, then
          strip the single leading space the prepend added
    */
    std::string decodeMetaspace(const std::vector<size_t>& ids, bool trimLeading) const {
        std::string out;

        for (size_t id : ids) {
            auto special = specialById.find(id);
            if (special != specialById.end()) { out += special->second; continue; }
            if (id >= decoder.size()) continue;

            const std::string& piece = decoder[id];

            // <0xNN> is one raw byte, not five characters
            if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0'
                                  && piece[2] == 'x' && piece[5] == '>') {
                const auto digit = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return -1;
                };
                const int hi = digit(piece[3]), lo = digit(piece[4]);
                if (hi >= 0 && lo >= 0) { out += static_cast<char>((hi << 4) | lo); continue; }
            }

            for (size_t i = 0; i < piece.size(); ) {
                if (piece.compare(i, metaspaceMark.size(), metaspaceMark) == 0) {
                    out += ' ';
                    i += metaspaceMark.size();
                } else {
                    out += piece[i];
                    ++i;
                }
            }
        }

        if (trimLeading && prependMark && !out.empty() && out[0] == ' ') out.erase(0, 1);
        return out;
    }


    /*
        decode(ids, trimLeading):
        - trimLeading undoes the mark the Metaspace encoder prepends, and so
          applies to a WHOLE sequence, not to a piece of one. Streaming a
          generation token by token has to pass false after the first, or
          every token loses its leading space and the output arrives as one
          run-on word
    */
    std::string decode(const std::vector<size_t>& ids, bool trimLeading = true) const {
        if (style == PreTokenizer::Metaspace) return decodeMetaspace(ids, trimLeading);

        std::string out;
        std::string mapped;

        for (size_t id : ids) {
            auto special = specialById.find(id);
            if (special != specialById.end()) {
                out += unmapBytes(mapped);
                mapped.clear();
                out += special->second;
                continue;
            }
            if (id < decoder.size()) mapped += decoder[id];
        }

        out += unmapBytes(mapped);
        return out;
    }


    std::string decodeOne(size_t id) const { return decode({id}); }
};

} // !namespace ai
