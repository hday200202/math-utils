#ifndef AI_GPU_DEVICE_ABI_H
#define AI_GPU_DEVICE_ABI_H

#include <stddef.h>
#include <stdint.h>

/*
    The backend boundary, in C.

    Device.hpp is the interface the model uses and it is still C++. This is
    the narrower thing underneath it: the symbols that actually cross from
    one compiler to another.

    WHY IT HAS TO BE C

    On Linux the backend and the rest of the library are both built by the
    same GCC, so a C++ boundary works and did. On Windows it cannot: hipcc
    emits host code against the MSVC C++ ABI, and the rest of this project
    is built by MinGW g++ against the Itanium ABI and libstdc++. Those two
    disagree about name mangling, about how a std::string is laid out, and
    about which runtime frees what. A std::string crossing that line is not
    a near miss - it will not even link, and if it did it would corrupt.

    C has none of those problems. There is one calling convention on x86-64
    Windows, structs of plain integers and doubles have one layout, and a
    function name is its own symbol. So everything that crosses is a C
    function taking scalars and pointers, and every C++ convenience -
    std::string, shared_ptr, the Buffer destructor - lives in Device.hpp as
    an inline wrapper on the caller's side, compiled by the caller's
    compiler.

    Nothing about this is Windows-only in effect. Linux links the same
    symbols from the same file; it just never needed the discipline.
*/

/*
    AI_GPU_ABI:
    - plain on Linux and for a statically linked stub, dllexport/dllimport
      when the backend is a Windows DLL. Three cases rather than two,
      because on Windows the stub is compiled into the executable and must
      NOT be marked dllimport.
*/
#if defined(_WIN32)
  #if defined(AI_GPU_BUILDING_DLL)
    #define AI_GPU_ABI __declspec(dllexport)
  #elif defined(AI_GPU_USE_DLL)
    #define AI_GPU_ABI __declspec(dllimport)
  #else
    #define AI_GPU_ABI
  #endif
#else
  #define AI_GPU_ABI
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
    Counters, as one flat struct of scalars.

    Passed by pointer and never by value: returning a struct is the one
    place where the two ABIs could still differ on where the result goes,
    and there is no reason to find out.
*/
typedef struct AiGpuCounters {
    uint64_t launches;
    uint64_t hostToDeviceBytes;
    uint64_t deviceToHostBytes;
    double   kernelSeconds;
    double   transferSeconds;
} AiGpuCounters;


/* 1 when a usable device was found. Safe to call first, and safe against
   the stub, which answers 0 */
AI_GPU_ABI int aigpu_available(void);

/* Borrowed, NUL-terminated, owned by the backend and valid until the next
   call that could change it. The wrapper in Device.hpp copies immediately
   so no std::string ever crosses */
AI_GPU_ABI const char* aigpu_device_name(void);
AI_GPU_ABI const char* aigpu_last_error(void);

AI_GPU_ABI uint64_t aigpu_total_memory(void);
AI_GPU_ABI uint64_t aigpu_free_memory(void);
AI_GPU_ABI uint64_t aigpu_allocated_bytes(void);

/*
    aigpu_upload / aigpu_release:
    - the device pointer is opaque to the caller. It is freed by whoever
      allocated it, which is the rule that makes a cross-compiler boundary
      safe: the backend's allocator and the caller's never meet.
    - the byte count comes back in on release so the backend can keep its
      allocation total without a lookup table.
*/
AI_GPU_ABI void* aigpu_upload(const void* host, uint64_t bytes);
AI_GPU_ABI void  aigpu_release(void* device, uint64_t bytes);
AI_GPU_ABI int   aigpu_download(void* host, const void* device, uint64_t bytes);

AI_GPU_ABI void aigpu_matmul_nt(const float* a, const void* b, float* c,
                                uint64_t m, uint64_t k, uint64_t n);

AI_GPU_ABI void aigpu_matmul_nt_bf16(const float* a, const void* b, float* c,
                                     uint64_t m, uint64_t k, uint64_t n);

AI_GPU_ABI void aigpu_matmul_nt_q8(const float* a, const void* b,
                                   const void* scales, float* c,
                                   uint64_t m, uint64_t k, uint64_t n);

AI_GPU_ABI void aigpu_matmul_nt_q4(const float* a, const void* b,
                                   const void* scales, float* c,
                                   uint64_t m, uint64_t k, uint64_t n);

AI_GPU_ABI void aigpu_synchronize(void);

AI_GPU_ABI void aigpu_counters(AiGpuCounters* out);
AI_GPU_ABI void aigpu_reset_counters(void);

#ifdef __cplusplus
} // !extern "C"
#endif

#endif // !AI_GPU_DEVICE_ABI_H
