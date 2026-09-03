#pragma once

#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <system_error>

#include "../datetime/Datetime.hpp"

namespace files {


namespace fs = std::filesystem;
using oss = std::ostringstream;



/*
    OpenMode:
    - every mode is binary. Text mode translates CRLF on Windows and treats
      0x1A as end of file, which silently corrupts anything that is not text
*/
enum class OpenMode {
    Read,           // read only, fails if the file does not exist
    Write,          // write only, creates the file or empties an existing one
    ReadWrite,      // both, fails if the file does not exist
    Append          // write only, creates the file, always writes at the end
};



/*
    toStdOpenMode(mode):
    - maps an OpenMode onto the std::ios flags it stands for
*/
inline std::ios::openmode toStdOpenMode(const OpenMode mode) {
    switch (mode) {
        case OpenMode::Read:      return std::ios::in  | std::ios::binary;
        case OpenMode::Write:     return std::ios::out | std::ios::trunc | std::ios::binary;
        case OpenMode::ReadWrite: return std::ios::in  | std::ios::out   | std::ios::binary;
        case OpenMode::Append:    return std::ios::out | std::ios::app   | std::ios::binary;
    }

    return std::ios::in | std::ios::binary;
}




class File {
public:
    File() = default;


    /*
        ~File():
        - closing is idempotent, so this is safe on a File that was never
          opened or was closed by hand
    */
    ~File() { close(); }


    /*
        - an fstream cannot be copied, only moved. Declaring that explicitly
          keeps the error at the call site instead of somewhere inside <fstream>
    */
    File(const File&)            = delete;
    File& operator=(const File&) = delete;
    File(File&&)                 = default;
    File& operator=(File&&)      = default;


    /*
        open(path, mode):
        - ReadWrite, the default, requires an existing writable file. Use Read
          for anything read-only, and Write or Append to create
        - any previously open file is closed first
    */
    bool open(const fs::path& path, const OpenMode mode = OpenMode::ReadWrite) {
        close();

        m_path = path;
        m_mode = mode;
        m_file.open(path.string(), toStdOpenMode(mode));

        return isOpen();
    }


    /*
        close():
        - clear() drops any failbit left behind, so the same File object can be
          opened again afterwards
    */
    void close() {
        if (isOpen()) m_file.close();
        m_file.clear();
    }


    /*
        flush():
        - pushes buffered writes to disk without closing
    */
    void flush() { if (isOpen()) m_file.flush(); }


    bool isOpen()               const { return m_file.is_open(); }
    OpenMode mode()             const { return m_mode; }
    fs::path path()             const { return m_path; }
    std::string fileName()      const { return m_path.filename().string(); }
    std::string extension()     const { return m_path.extension().string(); }
    fs::path parentDirectory()  const { return m_path.parent_path(); }


    /*
        exists():
        - takes an error_code so a bad path reports false rather than throwing
    */
    bool exists() const {
        std::error_code ec;
        return fs::exists(m_path, ec) && !ec;
    }


    /*
        size():
        - the file's size in bytes, or 0 if it cannot be read
    */
    size_t size() const {
        std::error_code ec;
        const auto bytes = fs::file_size(m_path, ec);
        return ec ? 0 : static_cast<size_t>(bytes);
    }


    /*
        permissions():
        - the file's permission bits, or perms::none if they cannot be read
    */
    fs::perms permissions() const {
        std::error_code ec;
        const auto status = fs::status(m_path, ec);
        return ec ? fs::perms::none : status.permissions();
    }


    /*
        lastModified():
        - the file's write time, or the epoch if it cannot be read
    */
    dtm::Timestamp lastModified() const {
        std::error_code ec;
        const auto written = fs::last_write_time(m_path, ec);
        return ec ? dtm::Timestamp(int64_t(0)) : dtm::Timestamp(written);
    }


    /*
        text():
        - the whole file as a string. Rewinds first, so calling it twice gives
          the same answer both times, and clears eofbit afterwards since
          reading to the end sets it
        - not const: moving the stream position is a mutation
    */
    std::string text() {
        if (!rewindForRead()) return {};

        oss out;
        out << m_file.rdbuf();
        m_file.clear();

        return out.str();
    }


    /*
        bytes():
        - the whole file as raw bytes, returned by value. Handing back
          out.str().c_str() would be a pointer into a temporary string that is
          destroyed before the caller can read it
    */
    std::vector<uint8_t> bytes() {
        const std::string data = text();
        return std::vector<uint8_t>(data.begin(), data.end());
    }


    /*
        lines():
        - splits on '\n' and drops a trailing '\r', so a file written on
          Windows does not leave carriage returns on the end of every line
    */
    std::vector<std::string> lines() {
        std::vector<std::string> out;
        std::string line;

        std::istringstream in(text());
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            out.push_back(line);
        }

        return out;
    }


    /*
        write(data):
        - replaces the whole file. Seeking to the front is not enough on its
          own: writing fewer bytes than the file already holds leaves the tail
          of the old contents behind, so the file is resized afterwards
        - note that m_file.clear() resets the stream's error flags and has
          nothing to do with emptying the file
    */
    bool write(const std::string& data) {
        if (!isOpen() || m_mode == OpenMode::Read) return false;

        m_file.clear();
        if (m_mode != OpenMode::Append) m_file.seekp(0, std::ios::beg);
        m_file.write(data.data(), static_cast<std::streamsize>(data.size()));
        m_file.flush();

        if (m_mode != OpenMode::Append) {
            std::error_code ec;
            fs::resize_file(m_path, data.size(), ec);
            if (ec) return false;
        }

        return m_file.good();
    }


    /*
        write(data):
        - the raw byte overload
    */
    bool write(const std::vector<uint8_t>& data) {
        return write(std::string(data.begin(), data.end()));
    }


    /*
        append(data):
        - seeks to the end first. Without that the put pointer sits at zero on
          a freshly opened file, so appending would overwrite from the start
    */
    bool append(const std::string& data) {
        if (!isOpen() || m_mode == OpenMode::Read) return false;

        m_file.clear();
        m_file.seekp(0, std::ios::end);
        m_file.write(data.data(), static_cast<std::streamsize>(data.size()));
        m_file.flush();

        return m_file.good();
    }


    /*
        append(data):
        - the raw byte overload
    */
    bool append(const std::vector<uint8_t>& data) {
        return append(std::string(data.begin(), data.end()));
    }


private:
    /*
        rewindForRead():
        - the seek is what lets a second read work, and it is also what makes a
          read legal after a write: switching direction on an fstream without
          an intervening seek or flush is undefined behaviour
    */
    bool rewindForRead() {
        if (!isOpen() || m_mode == OpenMode::Write || m_mode == OpenMode::Append) return false;

        m_file.clear();
        m_file.seekg(0, std::ios::beg);

        return m_file.good();
    }


    fs::path        m_path;
    std::fstream    m_file;
    OpenMode        m_mode = OpenMode::ReadWrite;
};




}
