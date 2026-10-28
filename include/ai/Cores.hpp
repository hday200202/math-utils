#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <sys/syscall.h>
    #include <unistd.h>
#endif

/*
    Which processors this program runs on, and how it shares them with other
    copies of itself.

    Every matmul is split across one worker per CPU and waits for the slowest
    of them. Two instances that each start a worker on every CPU therefore do
    not simply halve each other's speed: the scheduler keeps descheduling one
    instance's workers to run the other's, and every step waits on whichever
    worker is currently parked. Giving each instance its own CPUs is what
    keeps two of them from making each other far slower than either alone.

    So each instance pins its workers to a set of CPUs and records that set
    in a small claims file beside the executable. An instance asked to pick
    its own ("auto") takes whatever the others have not claimed.
*/

namespace ai {
namespace cores {

namespace fs = std::filesystem;


/* ========================================================================= //
    Naming CPUs
// ========================================================================= */

/*
    format(cpus): "0-15,32-47" - the same notation parse() reads, Linux's
    cpu lists use, and taskset accepts
*/
inline std::string format(std::vector<size_t> cpus) {
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());

    std::string out;
    for (size_t i = 0; i < cpus.size(); ) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) ++j;
        if (!out.empty()) out += ",";
        out += std::to_string(cpus[i]);
        if (j > i) out += "-" + std::to_string(cpus[j]);
        i = j + 1;
    }
    return out;
}

/*
    parseList(text, out, error):
    - "0-15,32,40-47" into the CPUs it names, sorted and without repeats.
      Purely syntactic: whether those CPUs exist is parse()'s question
*/
inline bool parseList(const std::string& text, std::vector<size_t>& out, std::string& error) {
    out.clear();
    std::stringstream in(text);
    std::string part;

    const auto number = [&](const std::string& s, size_t& value) {
        if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
        value = std::stoull(s);
        return true;
    };

    while (std::getline(in, part, ',')) {
        part.erase(0, part.find_first_not_of(" \t\n"));
        part.erase(part.find_last_not_of(" \t\n") + 1);
        if (part.empty()) continue;

        const size_t dash = part.find('-');
        size_t first = 0, last = 0;
        const bool ok = (dash == std::string::npos)
            ? number(part, first) && ((last = first), true)
            : number(part.substr(0, dash), first) && number(part.substr(dash + 1), last);

        if (!ok || last < first) {
            error = "\"" + part + "\" is not a CPU number or a range like 0-15";
            return false;
        }
        for (size_t c = first; c <= last; ++c) out.push_back(c);
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.empty()) { error = "no CPUs named"; return false; }
    return true;
}


/* ========================================================================= //
    The machine
// ========================================================================= */

/*
    allowed():
    - every CPU this process may run on, numbered the way the OS numbers
      them. On Linux that is the process's own affinity mask, so a run
      started under taskset or inside a cgroup cpuset sees only what it was
      given rather than every CPU in the machine. Windows numbers CPUs
      across processor groups in order, 64 to a group at most.
    - falls back to 0..hardware_concurrency-1 if the OS will not say
*/
inline std::vector<size_t> allowed() {
    std::vector<size_t> out;

#if defined(_WIN32)
    const DWORD total = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    for (DWORD i = 0; i < total; ++i) out.push_back(i);
#else
    /* the raw syscall rather than sched_getaffinity(): that one and the
       CPU_SET macros are only declared under _GNU_SOURCE, which not every
       compiler defines */
    std::vector<unsigned long> mask(1024 / sizeof(unsigned long), 0);   // 8192 CPUs
    const long bytes = syscall(SYS_sched_getaffinity, 0,
                               mask.size() * sizeof(unsigned long), mask.data());
    if (bytes > 0) {
        const size_t bits = 8 * sizeof(unsigned long);
        for (size_t c = 0; c < static_cast<size_t>(bytes) * 8; ++c)
            if (mask[c / bits] & (1UL << (c % bits))) out.push_back(c);
    }
#endif

    if (out.empty()) {
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < n; ++i) out.push_back(i);
    }
    return out;
}

/*
    parse(spec, cpus, error):
    - "all" for every CPU allowed(), or a list - which must name only CPUs
      this process can actually run on, since pinning a thread to one that
      does not exist fails quietly and leaves it wherever the OS likes
*/
inline bool parse(const std::string& spec, std::vector<size_t>& cpus, std::string& error) {
    const std::vector<size_t> usable = allowed();
    if (spec == "all") { cpus = usable; return true; }

    if (!parseList(spec, cpus, error)) return false;

    for (size_t c : cpus)
        if (!std::binary_search(usable.begin(), usable.end(), c)) {
            error = "CPU " + std::to_string(c) + " is not available to this process (it may use "
                  + format(usable) + ")";
            return false;
        }
    return true;
}

/*
    pinCurrentThread(cpu):
    - keeps the calling thread on that one CPU. A worker that stays put
      keeps its slice of the weights warm in that core's cache, and cannot
      wander onto a CPU another instance has claimed
*/
inline bool pinCurrentThread(size_t cpu) {
#if defined(_WIN32)
    const WORD groups = GetActiveProcessorGroupCount();
    for (WORD g = 0; g < groups; ++g) {
        const DWORD n = GetActiveProcessorCount(g);
        if (cpu < n) {
            GROUP_AFFINITY affinity{};
            affinity.Group = g;
            affinity.Mask  = KAFFINITY(1) << cpu;
            return SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr) != 0;
        }
        cpu -= n;
    }
    return false;
#else
    /* pid 0 in the raw syscall is the calling THREAD, not the process */
    const size_t bits = 8 * sizeof(unsigned long);
    std::vector<unsigned long> mask(cpu / bits + 1, 0);
    mask[cpu / bits] |= 1UL << (cpu % bits);
    return syscall(SYS_sched_setaffinity, 0, mask.size() * sizeof(unsigned long), mask.data()) == 0;
#endif
}


/* ========================================================================= //
    Processes
// ========================================================================= */

inline uint64_t selfPid() {
#if defined(_WIN32)
    return GetCurrentProcessId();
#else
    return static_cast<uint64_t>(getpid());
#endif
}

/*
    processStamp(pid):
    - when that process started, as an opaque number, or 0 if it is not
      running. A pid alone is not enough to tell whether a claim is still
      live: pids are reused, and a claim left by a run that was killed
      would otherwise be honoured by whatever unrelated process inherits
      its number. Pid AND start time identify one process.
*/
inline uint64_t processStamp(uint64_t pid) {
#if defined(_WIN32)
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (!h) return 0;

    uint64_t stamp = 0;
    DWORD code = 0;
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetExitCodeProcess(h, &code) && code == STILL_ACTIVE
        && GetProcessTimes(h, &created, &exited, &kernel, &user))
        stamp = (uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;

    CloseHandle(h);
    return stamp;
#else
    /* field 22 of /proc/<pid>/stat is the start time in clock ticks. The
       command name in field 2 can contain spaces and parentheses, so count
       from the LAST ')' rather than splitting the whole line */
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(in, line)) return 0;

    const size_t close = line.rfind(')');
    if (close == std::string::npos) return 0;

    std::stringstream rest(line.substr(close + 1));
    std::string field;
    for (int i = 3; i <= 22; ++i)
        if (!(rest >> field)) return 0;

    return std::stoull(field) + 1;          // +1 so a real start time is never 0
#endif
}


/* ========================================================================= //
    Claims
// ========================================================================= */

struct Claim {
    uint64_t            pid = 0, stamp = 0;
    std::vector<size_t> cpus;
    std::string         label;
};

/*
    Registry:
    - the claims file, one line per running instance:
          <pid> <start stamp> <cpus> <label>
      Lines whose process has gone are dropped whenever the file is read, so
      a run that was killed rather than exiting cleanly leaves nothing
      behind that matters.
    - every read-modify-write happens under a lock, taken by creating a
      directory beside the file. Creating a directory either happens or
      fails atomically on every filesystem this will meet, which makes it
      the portable lock - and a lock left by a process killed inside the few
      milliseconds it is held is broken after a few seconds
*/
class Registry {
public:
    explicit Registry(fs::path file) : m_file(std::move(file)) {}
    ~Registry() { release(); }

    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    const fs::path& file() const { return m_file; }

    /* the live claims of every OTHER process */
    std::vector<Claim> others(std::string& error) {
        Lock lock(m_file);
        if (!lock.held) { error = lock.error; return {}; }
        std::vector<Claim> out;
        for (Claim& c : readLive()) if (c.pid != selfPid()) out.push_back(std::move(c));
        return out;
    }

    /*
        claim(choose, label, error):
        - choose sees what the other live instances hold and returns the
          CPUs to take. Choosing and recording happen under one lock, so
          two instances started together cannot both pick the same "free"
          CPUs
    */
    bool claim(const std::function<std::vector<size_t>(const std::vector<Claim>&)>& choose,
               const std::string& label, std::vector<size_t>& taken, std::string& error) {
        Lock lock(m_file);
        if (!lock.held) { error = lock.error; return false; }

        std::vector<Claim> live = readLive(), others;
        for (const Claim& c : live) if (c.pid != selfPid()) others.push_back(c);

        taken = choose(others);

        Claim mine;
        mine.pid   = selfPid();
        mine.stamp = processStamp(mine.pid);
        mine.cpus  = taken;
        mine.label = label;
        others.push_back(mine);

        if (!write(others, error)) return false;
        m_held = true;
        return true;
    }

    /* drops this process's line. Called on the way out; harmless to repeat */
    void release() {
        if (!m_held) return;
        m_held = false;

        Lock lock(m_file);
        if (!lock.held) return;

        std::vector<Claim> keep;
        for (Claim& c : readLive()) if (c.pid != selfPid()) keep.push_back(std::move(c));

        std::string ignored;
        if (keep.empty()) {
            std::error_code ec;
            fs::remove(m_file, ec);
        } else {
            write(keep, ignored);
        }
    }

private:
    struct Lock {
        fs::path    dir;
        bool        held = false;
        std::string error;

        explicit Lock(const fs::path& file) : dir(file.string() + ".lock") {
            using namespace std::chrono;
            const auto deadline = steady_clock::now() + seconds(10);

            while (true) {
                std::error_code ec;
                if (fs::create_directory(dir, ec)) { held = true; return; }

                if (ec && ec != std::errc::file_exists) {
                    error = "cannot create " + dir.string() + ": " + ec.message();
                    return;
                }

                /* a lock this old belongs to a process that died holding it */
                const auto when = fs::last_write_time(dir, ec);
                if (!ec && fs::file_time_type::clock::now() - when > seconds(5)) {
                    fs::remove(dir, ec);
                    continue;
                }

                if (steady_clock::now() > deadline) {
                    error = "timed out waiting for " + dir.string();
                    return;
                }
                std::this_thread::sleep_for(milliseconds(10));
            }
        }

        ~Lock() {
            if (!held) return;
            std::error_code ec;
            fs::remove(dir, ec);
        }
    };

    std::vector<Claim> readLive() const {
        std::vector<Claim> out;
        std::ifstream in(m_file);
        std::string line;

        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::stringstream fields(line);
            Claim c;
            std::string cpus, ignored;
            if (!(fields >> c.pid >> c.stamp >> cpus)) continue;
            if (!parseList(cpus, c.cpus, ignored)) continue;
            std::getline(fields >> std::ws, c.label);

            if (c.stamp != 0 && processStamp(c.pid) == c.stamp) out.push_back(std::move(c));
        }
        return out;
    }

    /* written beside the file and renamed over it, so a reader never sees
       half a file */
    bool write(const std::vector<Claim>& claims, std::string& error) const {
        const fs::path temp = m_file.string() + ".tmp";
        {
            std::ofstream out(temp, std::ios::trunc);
            if (!out) { error = "cannot write " + temp.string(); return false; }

            out << "# CPUs claimed by running llm processes, one per line:\n"
                << "#   pid  start-stamp  cpus  model\n"
                << "# lines whose process has exited are dropped automatically\n";
            for (const Claim& c : claims)
                out << c.pid << " " << c.stamp << " " << format(c.cpus) << " " << c.label << "\n";
            if (!out) { error = "cannot write " + temp.string(); return false; }
        }

        std::error_code ec;
        fs::rename(temp, m_file, ec);
        if (ec) { error = "cannot replace " + m_file.string() + ": " + ec.message(); return false; }
        return true;
    }

    fs::path m_file;
    bool     m_held = false;
};

} // !namespace cores
} // !namespace ai
