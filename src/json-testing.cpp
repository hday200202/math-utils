#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/files/JSON.hpp"

/* ========================================================================= //
    JSON test suite.

    The last two sections parse the real files this exists for -- GPT-2's
    config.json and the header of its .safetensors -- so a passing run means
    the model loader has a working front end, not just that the toy cases
    work.

        make debug-json && ./bin/debug/json-testing
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

static int failures = 0;

namespace json = files::json;
using json::Value;
using json::Type;

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] "
              << COLOR_RESET << label << std::endl;
    if (!pass) ++failures;
}

static void section(const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
}

static std::string modelDir = "models/gpt2";


/* ========================================================================= //
    Primitives
// ========================================================================= */

static void testPrimitives() {
    section("primitives");

    check("null parses", json::parse("null").value.isNull());
    check("true parses", json::parse("true").value.asBool());
    check("false parses", json::parse("false").value.isBool()
          && !json::parse("false").value.asBool());

    check("an integer parses", json::parse("42").value.asInt() == 42);
    check("a negative integer parses", json::parse("-7").value.asInt() == -7);
    check("zero parses", json::parse("0").value.asInt() == 0);
    check("a float parses", json::parse("3.5").value.asDouble() == 3.5);

    // config.json writes layer_norm_epsilon as 1e-05
    check("exponent notation parses", json::parse("1e-05").value.asDouble() == 1e-05);
    check("a negative exponent with a mantissa parses",
          json::parse("-2.5E3").value.asDouble() == -2500.0);

    // a byte offset in a large model file exceeds 32 bits
    check("a large integer survives as a double",
          json::parse("154389504").value.asSize() == 154389504u);
    check("an integer past 2^32 is exact",
          json::parse("6871947673").value.asSize() == 6871947673ull);

    check("a string parses", json::parse("\"hello\"").value.asString() == "hello");
    check("an empty string parses", json::parse("\"\"").value.asString() == "");

    check("leading and trailing whitespace is skipped",
          json::parse("  \n\t 42 \n ").value.asInt() == 42);
}


/* ========================================================================= //
    Escapes
// ========================================================================= */

static void testEscapes() {
    section("escapes");

    check("\\\" unescapes", json::parse("\"a\\\"b\"").value.asString() == "a\"b");
    check("\\\\ unescapes", json::parse("\"a\\\\b\"").value.asString() == "a\\b");
    check("\\n unescapes", json::parse("\"a\\nb\"").value.asString() == "a\nb");
    check("\\t unescapes", json::parse("\"a\\tb\"").value.asString() == "a\tb");
    check("\\/ unescapes", json::parse("\"a\\/b\"").value.asString() == "a/b");

    check("a \\u escape becomes UTF-8",
          json::parse("\"\\u00e9\"").value.asString() == "\xc3\xa9");

    /* GPT-2's vocab.json is byte-level, so it is full of these -- the mapped
       printable stand-ins for raw bytes live in the U+0100 block */
    check("a \\u escape above 0x7FF becomes three bytes",
          json::parse("\"\\u0120\"").value.asString() == "\xc4\xa0");

    // a code point above the BMP arrives as a surrogate pair
    check("a surrogate pair combines into one code point",
          json::parse("\"\\ud83d\\ude00\"").value.asString() == "\xf0\x9f\x98\x80");
}


/* ========================================================================= //
    Arrays and objects
// ========================================================================= */

static void testContainers() {
    section("arrays and objects");

    Value a = json::parse("[1, 2, 3]").value;
    check("an array parses", a.isArray() && a.size() == 3);
    check("array elements read back", a[0].asInt() == 1 && a[2].asInt() == 3);
    check("an empty array parses", json::parse("[]").value.isArray());
    check("an empty array has size 0", json::parse("[]").value.size() == 0);

    Value o = json::parse("{\"a\": 1, \"b\": \"two\"}").value;
    check("an object parses", o.isObject() && o.size() == 2);
    check("object members read back", o["a"].asInt() == 1 && o["b"].asString() == "two");
    check("an empty object parses", json::parse("{}").value.isObject());
    check("contains finds a present key", o.contains("a"));
    check("contains rejects an absent key", !o.contains("zzz"));

    check("keys come back in file order", [&] {
        std::vector<std::string> k = o.keys();
        return k.size() == 2 && k[0] == "a" && k[1] == "b";
    }());

    Value nested = json::parse(
        "{\"outer\": {\"inner\": [10, 20, {\"deep\": true}]}}").value;
    check("nesting parses", nested["outer"]["inner"][1].asInt() == 20);
    check("deep nesting parses", nested["outer"]["inner"][2]["deep"].asBool());

    check("mixed types in one array", [] {
        Value m = json::parse("[1, \"two\", null, false, [3]]").value;
        return m.size() == 5 && m[0].isNumber() && m[1].isString()
            && m[2].isNull() && m[3].isBool() && m[4].isArray();
    }());

    check("asSizes reads a shape array", [] {
        std::vector<size_t> s = json::parse("[50257, 768]").value.asSizes();
        return s.size() == 2 && s[0] == 50257 && s[1] == 768;
    }());
    check("asSizes rejects a non-numeric array",
          json::parse("[1, \"x\"]").value.asSizes().empty());
}


/* ========================================================================= //
    Missing lookups

    Every miss returns a null Value, so a chain over absent keys degrades to
    the caller's default rather than reading through a bad reference.
// ========================================================================= */

static void testMissing() {
    section("missing lookups");

    Value o = json::parse("{\"a\": 1}").value;

    check("a missing key is null", o["nope"].isNull());
    check("a missing key falls back", o["nope"].asInt(-1) == -1);
    check("indexing an object is null", o[size_t(0)].isNull());
    check("indexing past the end of an array is null",
          json::parse("[1]").value[size_t(5)].isNull());
    check("indexing a scalar is null", json::parse("42").value["x"].isNull());

    check("a chain through missing keys is safe",
          o["a"]["b"]["c"][size_t(2)].asSize(99) == 99);

    check("asString on a number falls back", json::parse("42").value.asString("d") == "d");
    check("asInt on a string falls back", json::parse("\"x\"").value.asInt(5) == 5);
    check("size() of a scalar is 0", json::parse("42").value.size() == 0);
}


/* ========================================================================= //
    Malformed input

    Every one of these must report a failure rather than returning a
    half-built value that looks usable.
// ========================================================================= */

static void testErrors() {
    section("malformed input");

    auto bad = [](const std::string& text) { return !json::parse(text).ok; };

    check("empty input fails", bad(""));
    check("whitespace only fails", bad("   "));
    check("an unterminated string fails", bad("\"abc"));
    check("an unterminated array fails", bad("[1, 2"));
    check("an unterminated object fails", bad("{\"a\": 1"));
    check("a missing colon fails", bad("{\"a\" 1}"));
    check("a non-string key fails", bad("{a: 1}"));
    check("a trailing comma in an array fails", bad("[1, 2, ]"));
    check("a trailing comma in an object fails", bad("{\"a\": 1, }"));
    check("an unknown escape fails", bad("\"a\\qb\""));
    check("a truncated \\u escape fails", bad("\"\\u12\""));
    check("a bad hex digit fails", bad("\"\\u12zz\""));
    check("a bare identifier fails", bad("nul"));
    check("trailing content after the value fails", bad("42 43"));

    /* the safetensors header is a JSON blob followed by binary payload, so
       trailing content must be an error and not silently ignored -- the
       caller has to slice to the declared length first */
    check("a value followed by binary is rejected", bad("{\"a\":1}\x01\x02"));

    check("runaway nesting fails instead of overflowing the stack", [] {
        return !json::parse(std::string(500, '[')).ok;
    }());

    check("a failure reports an error message", [] {
        json::ParseResult r = json::parse("{\"a\" 1}");
        return !r.ok && !r.error.empty();
    }());

    check("operator bool reflects ok", [] {
        return static_cast<bool>(json::parse("1")) && !static_cast<bool>(json::parse("{"));
    }());
}


/* ========================================================================= //
    Round trip
// ========================================================================= */

static void testRoundTrip() {
    section("round trip");

    const std::string text =
        "{\"name\": \"gpt2\", \"n_layer\": 12, \"eps\": 1e-05, \"flag\": true, "
        "\"none\": null, \"shape\": [50257, 768], \"nest\": {\"a\": [1, 2]}}";

    json::ParseResult first = json::parse(text);
    check("the sample parses", first.ok);

    json::ParseResult second = json::parse(json::dump(first.value));
    check("dump output re-parses", second.ok);

    check("the round trip preserves every field", [&] {
        const Value& a = first.value;
        const Value& b = second.value;
        return a["name"].asString() == b["name"].asString()
            && a["n_layer"].asInt() == b["n_layer"].asInt()
            && a["eps"].asDouble() == b["eps"].asDouble()
            && a["flag"].asBool() == b["flag"].asBool()
            && b["none"].isNull()
            && a["shape"].asSizes() == b["shape"].asSizes()
            && a["nest"]["a"][1].asInt() == b["nest"]["a"][1].asInt();
    }());

    /* integers must not pick up a decimal point on the way out, or a shape
       array stops reading as a shape */
    check("integers dump without a decimal point",
          json::dump(json::parse("[50257, 768]").value, 0) == "[50257,768]");

    check("dump with indent 0 is compact",
          json::dump(json::parse("{\"a\":1}").value, 0) == "{\"a\":1}");

    check("escapes survive a round trip", [] {
        json::ParseResult r = json::parse("\"a\\\"b\\nc\\\\d\"");
        return json::parse(json::dump(r.value)).value.asString() == r.value.asString();
    }());
}


/* ========================================================================= //
    The real config.json
// ========================================================================= */

static void testConfigFile() {
    section("gpt2 config.json");

    json::ParseResult r = json::parseFile(modelDir + "/config.json");
    if (!r.ok) {
        check("config.json parses (" + r.error + ")", false);
        return;
    }
    check("config.json parses", true);

    const Value& c = r.value;
    check("n_layer is 12", c["n_layer"].asSize() == 12);
    check("n_embd is 768", c["n_embd"].asSize() == 768);
    check("n_head is 12", c["n_head"].asSize() == 12);
    check("vocab_size is 50257", c["vocab_size"].asSize() == 50257);
    check("n_positions is 1024", c["n_positions"].asSize() == 1024);
    check("layer_norm_epsilon is 1e-05", c["layer_norm_epsilon"].asDouble() == 1e-05);
    check("activation_function is gelu_new",
          c["activation_function"].asString() == "gelu_new");

    check("architectures is an array of one string",
          c["architectures"].size() == 1
          && c["architectures"][0].asString() == "GPT2LMHeadModel");

    // summary_activation is null in this file, which must not read as missing
    check("an explicit null is present but null",
          c.contains("summary_activation") && c["summary_activation"].isNull());

    check("a nested task_specific_params reads",
          c["task_specific_params"]["text-generation"]["max_length"].asSize() == 50);

    check("booleans read", c["summary_use_proj"].asBool());
}


/* ========================================================================= //
    The real safetensors header

    Layout: 8 bytes little-endian header length, then that many bytes of JSON
    mapping tensor name -> {dtype, shape, data_offsets}, then raw payload.
// ========================================================================= */

static void testSafetensorsHeader() {
    section("gpt2 safetensors header");

    std::ifstream in(modelDir + "/model.safetensors", std::ios::binary);
    if (!in) {
        check("model.safetensors opens", false);
        return;
    }

    uint64_t headerLength = 0;
    char lengthBytes[8] = {};
    in.read(lengthBytes, 8);
    std::memcpy(&headerLength, lengthBytes, 8);

    check("the header length is sane",
          headerLength > 0 && headerLength < 100u * 1024u * 1024u);

    std::string header(static_cast<size_t>(headerLength), '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerLength));
    check("the header reads", static_cast<bool>(in));

    json::ParseResult r = json::parse(header);
    if (!r.ok) {
        check("the header parses (" + r.error + " at " + std::to_string(r.position) + ")", false);
        return;
    }
    check("the header parses", true);

    const Value& h = r.value;
    check("the header holds 161 entries", h.size() == 161);

    const Value& wte = h["wte.weight"];
    check("wte.weight is present", wte.isObject());
    check("wte.weight is F32", wte["dtype"].asString() == "F32");
    check("wte.weight has shape [50257,768]",
          wte["shape"].asSizes() == std::vector<size_t>({50257, 768}));

    /* the offsets have to match the shape: 50257 * 768 floats at 4 bytes */
    check("wte.weight offsets match its element count", [&] {
        const std::vector<size_t> off = wte["data_offsets"].asSizes();
        if (off.size() != 2) return false;
        return (off[1] - off[0]) == 50257ull * 768ull * 4ull;
    }());

    check("wpe.weight has shape [1024,768]",
          h["wpe.weight"]["shape"].asSizes() == std::vector<size_t>({1024, 768}));

    /* GPT-2 stores linears as Conv1D, so this is [in, out] and NOT the
       [out, in] a normal Linear layer uses */
    check("h.0.attn.c_attn.weight is [768,2304] (Conv1D, in-major)",
          h["h.0.attn.c_attn.weight"]["shape"].asSizes()
          == std::vector<size_t>({768, 2304}));
    check("h.0.mlp.c_fc.weight is [768,3072]",
          h["h.0.mlp.c_fc.weight"]["shape"].asSizes()
          == std::vector<size_t>({768, 3072}));

    check("ln_f.weight is [768]",
          h["ln_f.weight"]["shape"].asSizes() == std::vector<size_t>({768}));

    // lm_head is tied to wte, so there is no separate tensor for it
    check("there is no lm_head tensor", !h.contains("lm_head.weight"));

    check("every one of the 12 blocks has all 12 tensors", [&] {
        const char* suffixes[] = {
            ".attn.bias", ".attn.c_attn.bias", ".attn.c_attn.weight",
            ".attn.c_proj.bias", ".attn.c_proj.weight",
            ".ln_1.bias", ".ln_1.weight", ".ln_2.bias", ".ln_2.weight",
            ".mlp.c_fc.bias", ".mlp.c_fc.weight", ".mlp.c_proj.bias", ".mlp.c_proj.weight"
        };
        for (int layer = 0; layer < 12; ++layer)
            for (const char* s : suffixes)
                if (!h.contains("h." + std::to_string(layer) + s)) return false;
        return true;
    }());

    check("every tensor is F32", [&] {
        for (const std::string& k : h.keys()) {
            if (k == "__metadata__") continue;
            if (h[k]["dtype"].asString() != "F32") return false;
        }
        return true;
    }());

    /* the regions must tile the payload exactly: sorted by start, each one
       begins where the previous ended, with no gap and no overlap */
    check("the tensor regions tile the payload with no gaps or overlaps", [&] {
        std::vector<std::pair<size_t, size_t>> spans;
        for (const std::string& k : h.keys()) {
            if (k == "__metadata__") continue;
            const std::vector<size_t> off = h[k]["data_offsets"].asSizes();
            if (off.size() != 2 || off[1] < off[0]) return false;
            spans.emplace_back(off[0], off[1]);
        }
        std::sort(spans.begin(), spans.end());
        if (spans.empty() || spans.front().first != 0) return false;
        for (size_t i = 1; i < spans.size(); ++i)
            if (spans[i].first != spans[i - 1].second) return false;
        return true;
    }());

    check("the declared payload matches the file size", [&] {
        size_t end = 0;
        for (const std::string& k : h.keys()) {
            if (k == "__metadata__") continue;
            const std::vector<size_t> off = h[k]["data_offsets"].asSizes();
            if (off.size() == 2) end = std::max(end, off[1]);
        }
        in.seekg(0, std::ios::end);
        const size_t fileSize = static_cast<size_t>(in.tellg());
        return fileSize == 8 + headerLength + end;
    }());
}


int main(int argc, char** argv) {
    const bool unitOnly = argc == 2 && std::string(argv[1]) == "--unit";
    if (argc > 1 && !unitOnly) modelDir = argv[1];

    testPrimitives();
    testEscapes();
    testContainers();
    testMissing();
    testErrors();
    testRoundTrip();
    if (!unitOnly) {
        testConfigFile();
        testSafetensorsHeader();
    } else {
        std::cout << "\nUnit mode: GPT-2 checkpoint integration tests were not run.\n";
    }

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    return failures;
}
