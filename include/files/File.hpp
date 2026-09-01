#pragma once

#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <chrono>

namespace files {

namespace fs = std::filesystem;
using oss = std::ostringstream;


template <typename T> inline bool contains(const std::vector<T>& vec, T item) {
    auto it = std::find(vec.begin(), vec.end(), item);
    return it != vec.end();
}


enum class ReadAs { ASCII, Bytes };


class File {
public:
    File() = default;
    ~File() { if (isOpen()) m_file.close(); }


    bool open(const fs::path& path) {
        m_file.open(
            path.string(), std::ios::in | std::ios::out
        );
        return isOpen();
    }


    bool isOpen() const { return m_file.is_open(); }
    fs::path path() const { return m_path; }
    std::string fileName() const { return m_path.filename().string(); }
    std::string extension() const { return m_path.extension().string();}
    fs::path parentDirectory() { return m_path.parent_path(); }
    size_t size() const { return fs::file_size(m_path); }
    fs::perms permissions() const { return fs::status(m_path).permissions(); }
    
    uint64_t lastModified() const {
        auto ftime = fs::last_write_time(m_path);
        auto sysTime = std::chrono::clock_cast<std::chrono::system_clock>(ftime);
        std::time_t seconds = std::chrono::system_clock::to_time_t(sysTime);
        return static_cast<uint64_t>(seconds);
    }

    std::string text() const {
        oss out;
        out << m_file.rdbuf();
        return out.str();
    }

    const char* bytes() const {
        oss out;
        out << m_file.rdbuf();
        return out.str().c_str();
    }

    void tokenize(std::vector<std::string>* tokBuf, const std::vector<char>& delimiters) const {
        std::string token = "";
        std::string data = text();

        for (char c : data) {
            if (contains(delimiters, c)) {
                tokBuf->push_back(token);
                token = "";
            }

            token += c;
        }
    }

private:
    fs::path m_path;
    std::fstream m_file;
};
    
};