#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/*
    The hot loops, kept apart from Tensor because they trade readability for
    speed and because they are where the threading lives.

    The thing to understand about generating one token at a time: it is bound
    by MEMORY BANDWIDTH, not arithmetic. Every weight in the model is read
    exactly once per token and each one is used for a single multiply-add, so
    the processor spends its time waiting on RAM. That is why the kernels
    below are shaped the way they are - every access walks forward through
    contiguous memory, and the work is split so each thread owns a distinct
    slab of the weights rather than sharing cache lines.

    No intrinsics. The inner loops are written so that -O3 -march=native
    vectorises them (AVX-512 with FMA on this machine); four running
    accumulators keep the FMA pipeline fed instead of stalling on a single
    serial dependency.
*/

namespace ai {
namespace kernels {

/* ========================================================================= //
    Thread pool

    Threads are created once and parked. Spawning per matmul would cost more
    than the matmul itself at these sizes - a token needs several hundred
    dispatches.
// ========================================================================= */

class Pool {
public:
    static Pool& instance() {
        static Pool pool;
        return pool;
    }

    size_t size() const { return m_workers.size() + 1; }

    /*
        run(count, body):
        - calls body(i) for i in [0, count), split into contiguous ranges one
          per thread. The calling thread takes a share too rather than idling
        - blocks until every range is done
    */
    void run(size_t count, const std::function<void(size_t, size_t)>& body) {
        const size_t threads = std::min(size(), count ? count : 1);
        if (threads <= 1 || count == 0) { if (count) body(0, count); return; }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_body = &body;
            m_count = count;
            m_threads = threads;
            m_next.store(0, std::memory_order_relaxed);
            m_remaining.store(threads - 1, std::memory_order_release);
            ++m_epoch;
        }
        m_wake.notify_all();

        consume();

        std::unique_lock<std::mutex> lock(m_doneMutex);
        m_done.wait(lock, [&] { return m_remaining.load(std::memory_order_acquire) == 0; });
        m_body = nullptr;
    }

private:
    Pool() {
        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware == 0) hardware = 1;

        for (unsigned i = 1; i < hardware; ++i)
            m_workers.emplace_back([this] { worker(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
            ++m_epoch;
        }
        m_wake.notify_all();
        for (std::thread& t : m_workers) if (t.joinable()) t.join();
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    /*
        consume():
        - grabs chunks by atomic increment rather than handing each thread a
          fixed slice up front. Rows of a matmul cost the same, but the last
          chunk of a ragged split would otherwise leave a thread idle
    */
    void consume() {
        const size_t count = m_count;
        const size_t chunks = m_threads;
        const size_t span = (count + chunks - 1) / chunks;

        while (true) {
            const size_t index = m_next.fetch_add(1, std::memory_order_relaxed);
            const size_t begin = index * span;
            if (begin >= count) break;
            (*m_body)(begin, std::min(begin + span, count));
        }
    }

    void worker() {
        size_t seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait(lock, [&] { return m_stopping || m_epoch != seen; });
                if (m_stopping) return;
                seen = m_epoch;
            }

            consume();

            if (m_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(m_doneMutex);
                m_done.notify_one();
            }
        }
    }

    std::vector<std::thread> m_workers;

    std::mutex               m_mutex;
    std::condition_variable  m_wake;
    size_t                   m_epoch = 0;
    bool                     m_stopping = false;

    const std::function<void(size_t, size_t)>* m_body = nullptr;
    size_t                   m_count = 0;
    size_t                   m_threads = 1;
    std::atomic<size_t>      m_next{0};

    std::atomic<size_t>      m_remaining{0};
    std::mutex               m_doneMutex;
    std::condition_variable  m_done;
};


/* ========================================================================= //
    Dot product
// ========================================================================= */

/*
    dot(a, b, n):
    - LANES independent accumulators, held as an array so the compiler keeps
      them in vector registers.
    - the count matters more than it looks. Floating point addition is not
      associative, so the compiler may only reduce as widely as the code
      explicitly permits: four named accumulators license a 4-wide reduction
      and no more, which on AVX-512 leaves seven eighths of the width idle.
      Measured on this machine, 4 accumulators give 8.1 GFLOP/s and 32 give
      65.3 - and 16 gives 6.9, because one zmm register cannot hide the
      four-cycle FMA latency. 32 is two registers deep, which can.
    - summing in a different order than strictly left to right changes the
      rounding slightly. That is unavoidable in any parallel reduction, and
      the layer-by-layer diff against the reference is what confirms it
      stays within tolerance
*/
constexpr size_t LANES = 32;

inline float dot(const float* a, const float* b, size_t n) {
    float acc[LANES] = {};

    size_t i = 0;
    for (; i + LANES <= n; i += LANES)
        for (size_t j = 0; j < LANES; ++j)
            acc[j] += a[i + j] * b[i + j];

    float sum = 0.0f;
    for (size_t j = 0; j < LANES; ++j) sum += acc[j];
    for (; i < n; ++i) sum += a[i] * b[i];

    return sum;
}


/*
    dotBF16(a, b, n):
    - the same reduction with the weights left in bfloat16 and widened as they
      are read. bfloat16 IS the top 16 bits of a float32, so widening is a
      shift into the high half rather than a conversion - no exponent
      rebiasing, no subnormal handling, no table.
    - the point is not the arithmetic, it is the memory. Generating a token
      reads every weight in the model exactly once, so halving the bytes very
      nearly halves the time. The accuracy given up is the mantissa bits that
      were never there: the checkpoint was bfloat16 on disk, and widening it
      to float32 at load time added zeroes, not information
*/
inline float dotBF16(const float* a, const uint16_t* b, size_t n) {
    float acc[LANES] = {};

    size_t i = 0;
    for (; i + LANES <= n; i += LANES)
        for (size_t j = 0; j < LANES; ++j)
            acc[j] += a[i + j] * std::bit_cast<float>(static_cast<uint32_t>(b[i + j]) << 16);

    float sum = 0.0f;
    for (size_t j = 0; j < LANES; ++j) sum += acc[j];
    for (; i < n; ++i)
        sum += a[i] * std::bit_cast<float>(static_cast<uint32_t>(b[i]) << 16);

    return sum;
}


/* ========================================================================= //
    Matmul against a transposed right operand

    C[M,N] = A[M,K] @ B[N,K]^T, with B stored row-major as [N,K].

    Every weight matrix in a HuggingFace Linear layer is stored [out, in],
    which is exactly this orientation - so no transpose is ever materialised
    and both operands are walked along their rows.

    The split is over N, the output columns, because N is the large dimension
    during generation: M is 1 for every token after the prompt, so splitting
    over rows would leave every thread but one asleep.
// ========================================================================= */

inline void matmulNT(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N, bool threaded = true) {
    /*
        j outermost, i innermost. The other order re-reads the whole weight
        matrix once per row of A, so a 32-token prompt streams every weight
        32 times and prefill runs no faster than decoding those tokens one
        at a time. This way each weight row is loaded once and reused across
        every row of A, which is what makes processing a prompt in one pass
        worth doing at all - A is only M*K and stays in cache.
    */
    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const float* brow = B + j * K;
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dot(A + i * K, brow, K);
        }
    };

    /* below a few thousand output columns the dispatch costs more than the
       work, and these are called several hundred times per token */
    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}


/*
    matmulNTBias(...):
    - the same, plus a bias row broadcast across the output. Folding it in
      here saves a second pass over C, which at these sizes is another trip
      through memory for no arithmetic
*/
/* matmulNT with the right operand left in bfloat16 */
inline void matmulNTBF16(const float* A, const uint16_t* B, float* C,
                         size_t M, size_t K, size_t N, bool threaded = true) {
    // j outermost so each weight row is streamed once for all M inputs
    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const uint16_t* brow = B + j * K;
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dotBF16(A + i * K, brow, K);
        }
    };

    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}


inline void matmulNTBias(const float* A, const float* B, const float* bias, float* C,
                         size_t M, size_t K, size_t N, bool threaded = true) {
    const auto body = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) {
            const float* brow = B + j * K;
            const float b = bias[j];
            for (size_t i = 0; i < M; ++i)
                C[i * N + j] = dot(A + i * K, brow, K) + b;
        }
    };

    if (!threaded || N < 512) { body(0, N); return; }

    Pool::instance().run(N, body);
}

} // !namespace kernels
} // !namespace ai
