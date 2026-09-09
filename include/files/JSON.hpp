#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "File.hpp"

/*
    A small, dependency-free JSON reader.

    Written for two jobs: a model's config.json, and the header of a
    .safetensors file (a length-prefixed JSON blob mapping tensor names to
    dtype/shape/byte-offsets). Both are machine-generated and small, so this
    favours being readable and hard to misuse over being fast.

    Deliberately not supported: comments, trailing commas, NaN/Infinity
    literals, duplicate-key resolution (the first wins). Numbers are all held
    as double, which is exact for integers up to 2^53 - far past any byte
    offset a model file will produce.

    Missing lookups return a null Value rather than throwing, so a chain like
    doc["model"]["layers"][3]["size"].asSize(0) is safe on malformed input.
*/

namespace files {
namespace json {

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value;
using Member = std::pair<std::string, Value>;

struct Value {
    Type type = Type::Null;

    bool        boolean = false;
    double      number  = 0.0;
    std::string string;
    std::vector<Value>  array;
    std::vector<Member> object;      // insertion-ordered, so iteration is stable

    bool isNull()   const { return type == Type::Null; }
    bool isBool()   const { return type == Type::Bool; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray()  const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    bool        asBool(bool fallback = false)      const { return isBool() ? boolean : fallback; }
    double      asDouble(double fallback = 0.0)    const { return isNumber() ? number : fallback; }
    int64_t     asInt(int64_t fallback = 0)        const { return isNumber() ? static_cast<int64_t>(number) : fallback; }
    size_t      asSize(size_t fallback = 0)        const { return isNumber() ? static_cast<size_t>(number) : fallback; }

    std::string asString(const std::string& fallback = {}) const
    { return isString() ? string : fallback; }

    /* size(): element count for arrays and objects, 0 for everything else */
    size_t size() const {
        if (isArray())  return array.size();
        if (isObject()) return object.size();
        return 0;
    }

    bool contains(std::string_view key) const {
        if (!isObject()) return false;
        for (const Member& m : object) if (m.first == key) return true;
        return false;
    }

    /* the shared null returned for every missing lookup, so chained access
       on absent keys degrades to a default instead of undefined behaviour */
    static const Value& null() {
        static const Value nothing;
        return nothing;
    }

    const Value& operator[](std::string_view key) const {
        if (!isObject()) return null();
        for (const Member& m : object) if (m.first == key) return m.second;
        return null();
    }

    const Value& operator[](size_t i) const {
        if (!isArray() || i >= array.size()) return null();
        return array[i];
    }

    /* keys(): the object's member names in file order */
    std::vector<std::string> keys() const {
        std::vector<std::string> out;
        if (!isObject()) return out;
        out.reserve(object.size());
        for (const Member& m : object) out.push_back(m.first);
        return out;
    }

    /*
        asSizes(): an array of numbers as a shape vector. Returns empty for
        anything that is not an array of numbers, so a caller checking size()
        catches a malformed "shape" field
    */
    std::vector<size_t> asSizes() const {
        std::vector<size_t> out;
        if (!isArray()) return out;
        out.reserve(array.size());
        for (const Value& v : array) {
            if (!v.isNumber()) return {};
            out.push_back(static_cast<size_t>(v.number));
        }
        return out;
    }
};


struct ParseResult {
    Value       value;
    bool        ok = false;
    std::string error;
    size_t      position = 0;       // byte offset where parsing stopped

    explicit operator bool() const { return ok; }
};


namespace detail {

/* recursion guard: a pathological input of nothing but '[' would otherwise
   run the stack out before it ran out of text */
constexpr int MAX_DEPTH = 64;

struct Parser {
    std::string_view src;
    size_t pos = 0;
    std::string error;

    bool fail(const std::string& message) {
        if (error.empty()) error = message;
        return false;
    }

    bool atEnd() const { return pos >= src.size(); }
    char peek() const { return pos < src.size() ? src[pos] : '\0'; }

    void skipWhitespace() {
        while (pos < src.size()) {
            const char c = src[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
            else break;
        }
    }

    bool literal(std::string_view word) {
        if (src.size() - pos < word.size()) return false;
        if (src.compare(pos, word.size(), word) != 0) return false;
        pos += word.size();
        return true;
    }

    /* appends one code point to out as UTF-8 */
    static void encodeUtf8(uint32_t cp, std::string& out) {
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

    bool parseHex4(uint32_t& out) {
        if (src.size() - pos < 4) return fail("truncated \\u escape");

        out = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = src[pos++];
            uint32_t digit = 0;

            if (c >= '0' && c <= '9')       digit = static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')  digit = static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')  digit = static_cast<uint32_t>(c - 'A' + 10);
            else return fail("bad hex digit in \\u escape");

            out = (out << 4) | digit;
        }
        return true;
    }

    bool parseString(std::string& out) {
        if (peek() != '"') return fail("expected a string");
        ++pos;

        out.clear();
        while (true) {
            if (atEnd()) return fail("unterminated string");

            const char c = src[pos++];
            if (c == '"') return true;

            if (c != '\\') {
                out += c;
                continue;
            }

            if (atEnd()) return fail("unterminated escape");
            const char e = src[pos++];
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parseHex4(cp)) return false;

                    /* a code point above the BMP arrives as a surrogate pair;
                       combine it rather than emitting two broken halves */
                    if (cp >= 0xD800 && cp <= 0xDBFF
                        && src.size() - pos >= 2 && src[pos] == '\\' && src[pos + 1] == 'u') {
                        const size_t save = pos;
                        pos += 2;
                        uint32_t low = 0;
                        if (parseHex4(low) && low >= 0xDC00 && low <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        else
                            pos = save;
                    }

                    encodeUtf8(cp, out);
                    break;
                }
                default: return fail("unknown escape sequence");
            }
        }
    }

    bool parseNumber(Value& out) {
        const size_t begin = pos;

        if (peek() == '-' || peek() == '+') ++pos;
        while (pos < src.size()) {
            const char c = src[pos];
            const bool part = (c >= '0' && c <= '9') || c == '.'
                           || c == 'e' || c == 'E' || c == '+' || c == '-';
            if (!part) break;
            ++pos;
        }

        if (begin == pos) return fail("expected a number");

        double parsed = 0.0;
        const char* first = src.data() + begin;
        const char* last  = src.data() + pos;
        const std::from_chars_result r = std::from_chars(first, last, parsed);

        if (r.ec != std::errc() || r.ptr != last) {
            pos = begin;
            return fail("malformed number");
        }

        out.type = Type::Number;
        out.number = parsed;
        return true;
    }

    bool parseValue(Value& out, int depth) {
        if (depth > MAX_DEPTH) return fail("nesting too deep");

        skipWhitespace();
        if (atEnd()) return fail("unexpected end of input");

        switch (peek()) {
            case 'n':
                if (!literal("null")) return fail("expected null");
                out = Value{};
                return true;

            case 't':
                if (!literal("true")) return fail("expected true");
                out = Value{}; out.type = Type::Bool; out.boolean = true;
                return true;

            case 'f':
                if (!literal("false")) return fail("expected false");
                out = Value{}; out.type = Type::Bool; out.boolean = false;
                return true;

            case '"': {
                std::string s;
                if (!parseString(s)) return false;
                out = Value{}; out.type = Type::String; out.string = std::move(s);
                return true;
            }

            case '[': {
                ++pos;
                out = Value{}; out.type = Type::Array;

                skipWhitespace();
                if (peek() == ']') { ++pos; return true; }

                while (true) {
                    Value element;
                    if (!parseValue(element, depth + 1)) return false;
                    out.array.push_back(std::move(element));

                    skipWhitespace();
                    if (peek() == ',') { ++pos; continue; }
                    if (peek() == ']') { ++pos; return true; }
                    return fail("expected ',' or ']' in array");
                }
            }

            case '{': {
                ++pos;
                out = Value{}; out.type = Type::Object;

                skipWhitespace();
                if (peek() == '}') { ++pos; return true; }

                while (true) {
                    skipWhitespace();

                    std::string key;
                    if (!parseString(key)) return fail("expected a key string in object");

                    skipWhitespace();
                    if (peek() != ':') return fail("expected ':' after key");
                    ++pos;

                    Value member;
                    if (!parseValue(member, depth + 1)) return false;
                    out.object.emplace_back(std::move(key), std::move(member));

                    skipWhitespace();
                    if (peek() == ',') { ++pos; continue; }
                    if (peek() == '}') { ++pos; return true; }
                    return fail("expected ',' or '}' in object");
                }
            }

            default:
                return parseNumber(out);
        }
    }
};

} // !namespace detail


/*
    parse(text):
    - trailing whitespace is fine, trailing content is not: a header followed
      by binary payload must be sliced to its declared length before it gets
      here, or the leftover bytes are reported as an error
*/
inline ParseResult parse(std::string_view text) {
    ParseResult result;

    detail::Parser p{text, 0, {}};
    if (!p.parseValue(result.value, 0)) {
        result.ok = false;
        result.error = p.error.empty() ? "parse failed" : p.error;
        result.position = p.pos;
        return result;
    }

    p.skipWhitespace();
    if (!p.atEnd()) {
        result.ok = false;
        result.error = "trailing content after the top-level value";
        result.position = p.pos;
        return result;
    }

    result.ok = true;
    result.position = p.pos;
    return result;
}


/*
    parseFile(path):
    - reads through files::File so the whole library goes through one file
      abstraction. OpenMode::Read is binary, so nothing translates CRLF
*/
inline ParseResult parseFile(const fs::path& path) {
    File file;
    if (!file.open(path, OpenMode::Read)) {
        ParseResult result;
        result.error = "could not open " + path.string();
        return result;
    }

    return parse(file.text());
}


/* ========================================================================= //
    Writing - only what a round-trip test and debug printing need
// ========================================================================= */

inline void escapeInto(std::string_view s, std::string& out) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* HEX = "0123456789abcdef";
                    out += "\\u00";
                    out += HEX[(c >> 4) & 0xF];
                    out += HEX[c & 0xF];
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

inline void dumpInto(const Value& v, std::string& out, int indent = 2, int level = 0) {
    const std::string pad(static_cast<size_t>(indent * (level + 1)), ' ');
    const std::string padEnd(static_cast<size_t>(indent * level), ' ');
    const char* newline = indent > 0 ? "\n" : "";

    switch (v.type) {
        case Type::Null:   out += "null"; return;
        case Type::Bool:   out += v.boolean ? "true" : "false"; return;
        case Type::String: escapeInto(v.string, out); return;

        case Type::Number: {
            /* integers print without a decimal point, which keeps a shape
               array reading as [50257, 768] rather than [50257.0, 768.0] */
            char buffer[32];
            if (v.number == static_cast<double>(static_cast<int64_t>(v.number))) {
                const auto r = std::to_chars(buffer, buffer + sizeof(buffer),
                                             static_cast<int64_t>(v.number));
                out.append(buffer, r.ptr);
            } else {
                const auto r = std::to_chars(buffer, buffer + sizeof(buffer), v.number);
                out.append(buffer, r.ptr);
            }
            return;
        }

        case Type::Array: {
            if (v.array.empty()) { out += "[]"; return; }
            out += '['; out += newline;
            for (size_t i = 0; i < v.array.size(); ++i) {
                if (indent > 0) out += pad;
                dumpInto(v.array[i], out, indent, level + 1);
                if (i + 1 < v.array.size()) out += ',';
                out += newline;
            }
            if (indent > 0) out += padEnd;
            out += ']';
            return;
        }

        case Type::Object: {
            if (v.object.empty()) { out += "{}"; return; }
            out += '{'; out += newline;
            for (size_t i = 0; i < v.object.size(); ++i) {
                if (indent > 0) out += pad;
                escapeInto(v.object[i].first, out);
                out += ':';
                if (indent > 0) out += ' ';
                dumpInto(v.object[i].second, out, indent, level + 1);
                if (i + 1 < v.object.size()) out += ',';
                out += newline;
            }
            if (indent > 0) out += padEnd;
            out += '}';
            return;
        }
    }
}

inline std::string dump(const Value& v, int indent = 2) {
    std::string out;
    dumpInto(v, out, indent, 0);
    return out;
}

} // !namespace json
} // !namespace files
