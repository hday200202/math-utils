#pragma once

#include "../Connector.hpp"

namespace omnibot {

inline static const std::string FILE_OPEN_FAIL     = "Could not open file.";
inline static const std::string FILE_READ_FAIL     = "Could not read file.";
inline static const std::string FILE_WRITE_FAIL    = "Could not write to file.";
inline static const std::string FILE_APP_FAIL      = "Could not append to file.";
inline static const std::string FILE_INS_FAIL      = "Could not insert in file.";
inline static const std::string FILE_WRITE_SUCCESS = "Successfully wrote to file.";
inline static const std::string FILE_APP_SUCCESS   = "Successfully appended to file.";
inline static const std::string FILE_INS_SUCCESS   = "Successfully inserted in file.";



class FileCmd : public Command {
public:
    FileCmd(std::string name) { m_cmd = std::move(name); }

    void setData(std::string data) { m_data = std::move(data); m_hasData = true; }
    void setIdx(uint32_t i) { m_idx = i; m_hasIdx = true; }

    void succeeded(std::string text) { m_ok = true; m_result = std::move(text); }
    void failed(std::string why) { m_ok = false; m_result = std::move(why); }

    const std::string& result() const { return m_result; }

private:
    files::json::Value fields() const override {
        files::json::Value v = files::json::jobject({
            { "result", files::json::jstr(m_result) } 
        });
        if (m_hasIdx)  v.object.push_back({ "idx",  files::json::jnum(m_idx) });
        if (m_hasData) v.object.push_back({ "data", files::json::jstr(m_data) });
        return v;
    }

    std::string m_result    = "";
    std::string m_data      = "";
    uint32_t    m_idx       = 0;

    bool m_hasData = false;
    bool m_hasIdx = false;
};



enum class FilePerms { Read, Write, Append, Insert };

class FileConnector : public Connector {
private:
    FileConnector() = default;
    FileConnector(
        const std::string& label, 
        const std::string& path, 
        const std::vector<FilePerms>& perms
    ) {
        m_perms = perms;

        m_readCmd   = "read_"    + label;
        m_writeCmd  = "write_"   + label;
        m_appendCmd = "append_"  + label;
        m_insertCmd = "insert_"  + label;

        /*
            The example IS the schema. A model copies a shape it can see far
            more reliably than it infers one from a type description, so what
            the agent is shown is exactly what it should send back.
        */
        for (auto& perm : m_perms) {
            switch (perm) {
                case FilePerms::Read:
                    offer(m_readCmd, R"({"cmd": ")" + m_readCmd + R"("})");
                    break;

                case FilePerms::Write:
                    offer(m_writeCmd,
                          R"({"cmd": ")" + m_writeCmd + R"(", "data": "<text>"})");
                    break;

                case FilePerms::Append:
                    offer(m_appendCmd,
                          R"({"cmd": ")" + m_appendCmd + R"(", "data": "<text>"})");
                    break;

                case FilePerms::Insert:
                    offer(m_insertCmd,
                          R"({"cmd": ")" + m_insertCmd +
                          R"(", "idx": <number>, "data": "<text>"})");
                    break;

                default: break;
            }
        }

        m_path = path;
        m_label = label;
    }

    friend FileConnector* fileConnector(
        const std::string& label,
        const std::string& path,
        const std::vector<FilePerms>& perms
    );

public:
    const std::vector<std::string>& commands() const override { return m_cmds; }
    const std::vector<std::string>& names()    const override { return m_names; }

    /*
        executeCmd(cmd):
        - the name says which fields the command carries, so there is nothing
          to parse: the json already arrived structured. Everything the old
          text parser needed - from_chars, substring offsets, length guards -
          is gone with it.
        - a field that must be present is checked with contains() rather than
          read with a fallback. operator[] hands back a shared null for a
          missing key, so asString() would quietly turn an absent "data" into
          an empty write.
    */
    files::json::Value executeCmd(const files::json::Value& cmd) override {
        using namespace files;

        const std::string name = cmd["cmd"].asString();
        FileCmd reply(name);

        if (name == m_readCmd) {
            const std::string text = read(reply);
            if (reply.ok()) reply.succeeded(text);
            return reply.dump();
        }

        if (name == m_writeCmd || name == m_appendCmd) {
            if (!cmd.contains("data")) {
                reply.failed(m_label + ": " + name + " requires \"data\"");
                return reply.dump();
            }

            const std::string data = cmd["data"].asString();
            reply.setData(data);

            if (name == m_writeCmd) write(reply, data);
            else                    append(reply, data);
            return reply.dump();
        }

        if (name == m_insertCmd) {
            if (!cmd.contains("idx") || !cmd.contains("data")) {
                reply.failed(m_label + ": " + name + " requires \"idx\" and \"data\"");
                return reply.dump();
            }

            const uint32_t idx  = static_cast<uint32_t>(cmd["idx"].asSize(0));
            const std::string data = cmd["data"].asString();
            reply.setIdx(idx);
            reply.setData(data);

            insert(reply, data, idx);
            return reply.dump();
        }

        reply.failed(m_label + ": " + INV_CMD);
        return reply.dump();
    }

private:
    std::vector<FilePerms>      m_perms;
    std::vector<std::string>    m_cmds;     // json examples, shown to the agent
    std::vector<std::string>    m_names;    // bare names, what routers key on
    std::string                 m_path;
    std::string                 m_label;

    std::string                 m_readCmd;
    std::string                 m_writeCmd;
    std::string                 m_appendCmd;
    std::string                 m_insertCmd;

    /*
        offer(name, example):
        - the two lists only mean anything together, so nothing adds to one
          without the other
    */
    void offer(const std::string& name, const std::string& example) {
        m_names.push_back(name);
        m_cmds.push_back(example);
    }


    /*
        The permission check stays here rather than only in executeCmd. The
        command list is the gate an agent sees, but executeCmd takes whatever
        json it is handed, so the grant is re-checked at the point of use.
    */
    std::string read(FileCmd& reply) {
        if (!hasPerm(FilePerms::Read)) { reply.failed(m_label + ": " + FILE_READ_FAIL); return {}; }

        files::File file;
        file.open(m_path, files::OpenMode::Read);
        if (!file.isOpen()) { reply.failed(m_label + ": " + FILE_OPEN_FAIL); return {}; }

        reply.succeeded({});
        return file.text();
    }

    void write(FileCmd& reply, const std::string& text) {
        if (!hasPerm(FilePerms::Write)) { reply.failed(m_label + ": " + FILE_WRITE_FAIL); return; }

        files::File file;
        file.open(m_path, files::OpenMode::Write);
        if (!file.isOpen()) { reply.failed(m_label + ": " + FILE_OPEN_FAIL); return; }

        if (file.write(text)) reply.succeeded(m_label + ": " + FILE_WRITE_SUCCESS);
        else                  reply.failed(m_label + ": " + FILE_WRITE_FAIL);
    }

    void append(FileCmd& reply, const std::string& text) {
        if (!hasPerm(FilePerms::Append)) { reply.failed(m_label + ": " + FILE_APP_FAIL); return; }

        files::File file;
        file.open(m_path, files::OpenMode::Append);
        if (!file.isOpen()) { reply.failed(m_label + ": " + FILE_OPEN_FAIL); return; }

        if (file.append(text)) reply.succeeded(m_label + ": " + FILE_APP_SUCCESS);
        else                   reply.failed(m_label + ": " + FILE_APP_FAIL);
    }

    void insert(FileCmd& reply, const std::string& text, const uint32_t idx) {
        if (!hasPerm(FilePerms::Insert)) { reply.failed(m_label + ": " + FILE_INS_FAIL); return; }

        files::File file;
        file.open(m_path, files::OpenMode::ReadWrite);
        if (!file.isOpen()) { reply.failed(m_label + ": " + FILE_OPEN_FAIL); return; }

        if (file.insert(idx, text)) reply.succeeded(m_label + ": " + FILE_INS_SUCCESS);
        else                        reply.failed(m_label + ": " + FILE_INS_FAIL);
    }

    bool hasPerm(const FilePerms perm) {
        return (std::find(m_perms.begin(), m_perms.end(), perm)) != m_perms.end();
    }
};


// FileConnector factory function
FileConnector* fileConnector(
    const std::string& label,
    const std::string& path,
    const std::vector<FilePerms>& perms
) {
    if (label.empty())  connectorError("FileConnector: ",   "Can not create a file connector with an empty label.");
    if (path.empty())   connectorError(label + ": ",        "Can not create a file connector with no specified path.");
    if (perms.empty())  connectorError(label + ": ",        "Can not create a file connector with no permissions.");

    return new FileConnector(label, path, perms); 
}

}