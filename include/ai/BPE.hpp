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

struct Tokenizer {
    std::unordered_map<std::string, int32_t> encoder;   // mapped string -> id
    std::vector<std::string>                 decoder;   // id -> mapped string
    std::unordered_map<std::string, int32_t> ranks;     // "left right" -> priority

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


    size_t vocabSize() const { return decoder.size(); }


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

    static std::vector<std::string> preTokenize(std::string_view text) {
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


    std::vector<size_t> encode(std::string_view text) const {
        std::vector<size_t> ids;

        for (const std::string& chunk : preTokenize(text)) {
            std::string mapped;
            for (char c : chunk) mapped += byteToUnicode[static_cast<unsigned char>(c)];

            for (const std::string& piece : bpe(mapped)) {
                auto found = encoder.find(piece);
                if (found != encoder.end()) ids.push_back(static_cast<size_t>(found->second));
            }
        }

        return ids;
    }


    std::string decode(const std::vector<size_t>& ids) const {
        std::string mapped;
        for (size_t id : ids)
            if (id < decoder.size()) mapped += decoder[id];

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


    std::string decodeOne(size_t id) const { return decode({id}); }
};

} // !namespace ai
