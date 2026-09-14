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



enum class FilePerms { Read, Write, Append, Insert };

class FileConnector : public Connector {
public:
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

        for (auto& perm : m_perms) {
            switch (perm) {
                case FilePerms::Read:   m_cmds.push_back(m_readCmd); break;
                case FilePerms::Write:  m_cmds.push_back(m_writeCmd + " \"data\""); break;
                case FilePerms::Append: m_cmds.push_back(m_appendCmd + " \"data\""); break;
                case FilePerms::Insert: m_cmds.push_back(m_insertCmd + " [idx] " + "\"data\""); break;
                default: break;
            }
        }

        m_path = path;
        m_label = label;
    }

    const std::vector<std::string>& commands() const override { return m_cmds; }

    std::string executeCmd(const std::string& cmd) override {
        std::string token   = "";
        std::string index   = "";

        for (auto& c : cmd) {
            if (c == ' ') break;
            token += c;
        }

        if (token == m_readCmd) return read();

        if (token == m_writeCmd) {
            if (cmd.size() <= m_writeCmd.size() + 1) return INV_CMD;
            return write(cmd.substr(m_writeCmd.size() + 1));
        }

        if (token == m_appendCmd) {
            if (cmd.size() <= m_appendCmd.size() + 1) return INV_CMD;
            return append(cmd.substr(m_appendCmd.size() + 1));
        }

        if (cmd.size() <= m_insertCmd.size() + 1) return INV_CMD;

        const char* begin   = cmd.data() + m_insertCmd.size() + 1;
        const char* end     = cmd.data() + cmd.size();
        uint32_t idx = 0;
        auto [ptr, ec] = std::from_chars(begin, end, idx);

        if (ec != std::errc{})  return INV_CMD;
        if (ptr == end)         return INV_CMD;
        if (*ptr != ' ')        return INV_CMD;

        const std::string data(ptr + 1, end);

        if (token == m_insertCmd) return insert(data, idx);

        return INV_CMD;
    }

private:
    std::vector<FilePerms>      m_perms;
    std::vector<std::string>    m_cmds;
    std::string                 m_path;
    std::string                 m_label;

    std::string                 m_readCmd;
    std::string                 m_writeCmd;
    std::string                 m_appendCmd;
    std::string                 m_insertCmd;

    std::string read() {
        if (!hasPerm(FilePerms::Read)) return FILE_READ_FAIL;

        files::File file; 
        file.open(m_path, files::OpenMode::Read);
        if (!file.isOpen()) return FILE_OPEN_FAIL;
        return file.text();
    }

    std::string write(const std::string& text) {
        if (!hasPerm(FilePerms::Write)) return FILE_WRITE_FAIL;

        files::File file; 
        file.open(m_path, files::OpenMode::Write);
        if (!file.isOpen()) return FILE_OPEN_FAIL;
        if (file.write(text)) return FILE_WRITE_SUCCESS;
        return FILE_WRITE_FAIL;
    }

    std::string append(const std::string& text) {
        if (!hasPerm(FilePerms::Append)) return FILE_APP_FAIL;

        files::File file; 
        file.open(m_path, files::OpenMode::Append);
        if (!file.isOpen()) return FILE_OPEN_FAIL;
        if (file.append(text)) return FILE_APP_SUCCESS;
        return FILE_APP_FAIL;
    }

    std::string insert(const std::string& text, const uint32_t idx) {
        if (!hasPerm(FilePerms::Insert)) return FILE_INS_FAIL;

        files::File file;
        file.open(m_path, files::OpenMode::ReadWrite);
        if (!file.isOpen()) return FILE_OPEN_FAIL;
        if (file.insert(idx, text)) return FILE_INS_SUCCESS;

        return FILE_INS_FAIL;
    }

    bool hasPerm(const FilePerms perm) {
        return (std::find(m_perms.begin(), m_perms.end(), perm)) != m_perms.end();
    }
};



}