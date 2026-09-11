#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/*
    A persistent pool, in its own header so that both the matmul kernels and
    the quantiser can use it. Threads are created once and parked; spawning
    per call would cost more than the work at these sizes.
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
        if (count == 0) return;

        /*
            EVERY worker is woken and every worker decrements the counter the
            caller waits on, so the split has to be into one range per worker
            even when there is less work than that. Sizing it to the work
            instead - the obvious thing - leaves the extra workers decrementing
            a counter that was only ever set for a few of them: it runs past
            zero, wraps, and the caller waits for a value that will never come
            back round. Nothing noticed while every call had hundreds of rows
            to hand out; the first one with ten hung immediately.
        */
        const size_t threads = size();
        if (threads <= 1) { body(0, count); return; }

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


} // !namespace kernels
} // !namespace ai
