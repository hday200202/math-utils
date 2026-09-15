#include "DeviceABI.h"

/*
    The no-device backend.

    Linked instead of Kernels.hip when the build has no ROCm, so that
    LocalModel and everything above it compile and run identically. Every
    call site already has to handle available() == false - a card can be
    absent at runtime too - so there is nothing here but honest zeroes.

    It implements the C entry points rather than the C++ interface, because
    that interface is now entirely inline in Device.hpp and has no symbols
    of its own to satisfy. Note there is no dllexport here even on Windows:
    the stub is compiled straight into the executable, never into a DLL,
    which is why AI_GPU_ABI has a third case for exactly this.
*/

extern "C" {

namespace {
    const char NO_NAME[] = "";
    const char NO_DEVICE[] =
        "built without a GPU backend: no ROCm on the machine this was compiled on";
}

int aigpu_available(void) { return 0; }

const char* aigpu_device_name(void) { return NO_NAME; }
const char* aigpu_last_error(void)  { return NO_DEVICE; }

uint64_t aigpu_total_memory(void)    { return 0; }
uint64_t aigpu_free_memory(void)     { return 0; }
uint64_t aigpu_allocated_bytes(void) { return 0; }

void* aigpu_upload(const void*, uint64_t) { return nullptr; }
void  aigpu_release(void*, uint64_t) {}
int   aigpu_download(void*, const void*, uint64_t) { return 0; }

void aigpu_matmul_nt(const float*, const void*, float*,
                     uint64_t, uint64_t, uint64_t) {}

void aigpu_matmul_nt_bf16(const float*, const void*, float*,
                          uint64_t, uint64_t, uint64_t) {}

void aigpu_matmul_nt_q8(const float*, const void*, const void*, float*,
                        uint64_t, uint64_t, uint64_t) {}

void aigpu_matmul_nt_q4(const float*, const void*, const void*, float*,
                        uint64_t, uint64_t, uint64_t) {}

void aigpu_synchronize(void) {}

void aigpu_counters(AiGpuCounters* out) {
    if (out) *out = AiGpuCounters{};
}

void aigpu_reset_counters(void) {}

} // !extern "C"
