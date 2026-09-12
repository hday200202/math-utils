#pragma once

#include <cstddef>
#include <cstdio>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
#else
    #include <unistd.h>
#endif

namespace platform {

/*
    Physical memory, in bytes.

    availableMemory() is what can be handed out without the machine starting
    to page, which is the number that matters before reading a model: the
    allocation will "succeed" long past this point by growing the page file,
    and then every weight read touches disk instead of RAM. That is not a
    slow model, it is an unusable one.

    Both return 0 when the platform cannot say, which callers should treat as
    "no opinion" rather than "no memory".
*/

inline size_t totalMemory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return static_cast<size_t>(status.ullTotalPhys);

#elif defined(__APPLE__)
    int64_t bytes = 0;
    size_t len = sizeof(bytes);
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    if (sysctl(mib, 2, &bytes, &len, nullptr, 0) != 0) return 0;
    return static_cast<size_t>(bytes);

#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page  = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 0;
    return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
}


inline size_t availableMemory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return static_cast<size_t>(status.ullAvailPhys);

#elif defined(__APPLE__)
    mach_port_t host = mach_host_self();
    vm_size_t pageSize = 0;
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (host_page_size(host, &pageSize) != KERN_SUCCESS) return 0;
    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&stats), &count) != KERN_SUCCESS)
        return 0;

    // free plus what the cache would give back under pressure
    return static_cast<size_t>(stats.free_count + stats.inactive_count)
         * static_cast<size_t>(pageSize);

#else
    /*
        MemAvailable, not _SC_AVPHYS_PAGES.

        _SC_AVPHYS_PAGES is MemFree, which counts only pages nobody is
        using at all. Page cache is not in it. On a machine that has just
        read a checkpoint that is almost the whole story: the file is in
        cache, MemFree reads a few hundred MB, and a loader that trusts it
        refuses to open a model the machine has ample room for.

        MemAvailable is the kernel's own estimate of what a new allocation
        could get, cache reclaim included - which is the question being
        asked here. It has been published since Linux 3.14; the older path
        stays as a fallback for anything that does not have it.
    */
    if (std::FILE* meminfo = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        size_t kilobytes = 0;
        bool found = false;

        while (std::fgets(line, sizeof(line), meminfo)) {
            unsigned long long value = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
                kilobytes = static_cast<size_t>(value);
                found = true;
                break;
            }
        }

        std::fclose(meminfo);
        if (found) return kilobytes * 1024;
    }

    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page  = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 0;
    return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
}

} // !namespace platform
