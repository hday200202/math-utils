#pragma once

#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <system_error>
#include <functional>
#include <algorithm>

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
        - the string is sized from the file up front and read into directly.
          Streaming through an ostringstream instead costs two extra copies of
          the whole file: the stream's buffer reallocates as it grows, and
          str() then hands back a copy of it. Measured on a 4GB file that is
          6.15s against 0.88s, and the transient memory matters as much as the
          time once a file is large enough for the difference to show
        - the read is CHUNKED. A single read() of more than 2GB returns
          nothing at all here - gcount() comes back 0 with no error flag set -
          so a whole-file read in one call silently produces an empty string
          on exactly the files where it matters
        - the loop accumulates gcount() rather than trusting the request, and
          the final resize reports what was actually read, so a file that
          shrank underneath us truncates instead of leaving a tail of zeros
    */
    std::string text(const std::function<void(size_t, size_t)>& onProgress = {}) {
        if (!rewindForRead()) return {};

        std::string out(size(), '\0');

        constexpr size_t CHUNK = size_t(1) << 24;   // 16MB
        size_t got = 0;

        while (got < out.size() && m_file) {
            const size_t want = std::min(CHUNK, out.size() - got);
            m_file.read(out.data() + got, static_cast<std::streamsize>(want));
            got += static_cast<size_t>(m_file.gcount());
            if (onProgress) onProgress(got, out.size());
        }

        out.resize(got);
        m_file.clear();

        return out;
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


    /*
        insert(at, data):
        - splices data in at a byte offset, keeping everything already there.
          at == size() is legal and means the same as append; anything past
          that is refused rather than silently landing at the end, since an
          index that is out of range is a mistake worth hearing about
        - ReadWrite is the only mode that can do this. Write cannot read back
          what it is about to shift, and Append can neither read nor seek, so
          checking the mode here is what stops a silent half-operation
        - the whole file goes through memory, which is the same bargain
          write() already makes: there is no way to shift the tail forward
          without rewriting everything from the insertion point on
    */
    bool insert(const size_t at, const std::string& data) {
        if (!isOpen() || m_mode != OpenMode::ReadWrite) return false;
        if (data.empty()) return true;

        std::string whole = text();
        if (at > whole.size()) return false;

        whole.insert(at, data);

        return write(whole);
    }


    /*
        insert(at, data):
        - the raw byte overload
    */
    bool insert(const size_t at, const std::vector<uint8_t>& data) {
        return insert(at, std::string(data.begin(), data.end()));
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
