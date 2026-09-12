#include "../../include/ai/gpu/Device.hpp"

/*
    The no-device backend.

    Linked instead of Kernels.hip when the build has no ROCm, so that
    LocalModel and everything above it compile and run identically. Every
    call site already has to handle available() == false - a card can be
    absent at runtime too - so there is nothing here but honest zeroes.
*/

namespace ai {
namespace gpu {

namespace {
    const std::string NO_NAME;
    const std::string NO_DEVICE = "built without a GPU backend (see release-llm-gpu)";
    const Counters    EMPTY;
}

Buffer::~Buffer() = default;

bool available() { return false; }

const std::string& deviceName() { return NO_NAME; }
const std::string& lastError()  { return NO_DEVICE; }

size_t totalMemory()    { return 0; }
size_t freeMemory()     { return 0; }
size_t allocatedBytes() { return 0; }

BufferRef upload(const void*, size_t) { return nullptr; }
bool      download(void*, const Buffer&, size_t) { return false; }

void matmulNT    (const float*, const Buffer&, float*, size_t, size_t, size_t) {}
void matmulNTBF16(const float*, const Buffer&, float*, size_t, size_t, size_t) {}
void matmulNTQ8(const float*, const Buffer&, const Buffer&, float*, size_t, size_t, size_t) {}
void matmulNTQ4(const float*, const Buffer&, const Buffer&, float*, size_t, size_t, size_t) {}

void synchronize() {}

const Counters& counters() { return EMPTY; }
void resetCounters() {}

} // !namespace gpu
} // !namespace ai
