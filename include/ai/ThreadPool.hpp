#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "Cores.hpp"

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

    /*
        configure(cpus):
        - one thread per listed CPU, each pinned to its own: the caller to
          the first, a worker to each of the rest. Has to come before
          anything calls instance(), since the pool is built once; returns
          false if it is too late.
        - without it the pool is one unpinned thread per hardware thread,
          which is what every program that never asks still gets
    */
    static bool configure(const std::vector<size_t>& cpus) {
        if (started() || cpus.empty()) return false;
        planned() = cpus;
        if (!cores::pinCurrentThread(cpus[0])) pinFailures().fetch_add(1);
        return true;
    }

    /* the CPUs the pool was configured for; empty means unpinned */
    static const std::vector<size_t>& cpus() { return planned(); }

    /* threads that could not be pinned where they were asked to go */
    static size_t unpinned() { return pinFailures().load(); }

    size_t size() const { return m_workers.size() + 1; }

    /*
        slice(count, index, threads):
        - the range of [0, count) that thread `index` of `threads` handles.
          The caller is thread 0 and worker i is thread i, and that never
          changes, which is the point: see run()
    */
    static std::pair<size_t, size_t> slice(size_t count, size_t index, size_t threads) {
        const size_t span  = (count + threads - 1) / threads;
        const size_t begin = std::min(index * span, count);
        return {begin, std::min(begin + span, count)};
    }

    /*
        run(count, body):
        - calls body(begin, end) over [0, count), split into one contiguous
          range per thread. The calling thread takes a share too rather than
          idling
        - blocks until every range is done
        - the split is STATIC: thread i always gets slice(count, i, size()).
          A matmul over the same matrix therefore has the same thread read
          the same rows every token, so the loader can place those rows in
          that thread's own NUMA node's memory and they stay local - on a
          two-socket server the difference between reading at local and at
          cross-socket bandwidth. Grabbing chunks dynamically balanced a
          ragged last chunk slightly better and threw that away
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
            m_remaining.store(threads - 1, std::memory_order_relaxed);
            m_epoch.fetch_add(1, std::memory_order_release);
        }
        m_wake.notify_all();

        work(0);

        /* the workers usually finish within microseconds of the caller, so
           spin for them before paying for a sleep and a wake */
        if (!spinUntil([&] { return m_remaining.load(std::memory_order_acquire) == 0; })) {
            std::unique_lock<std::mutex> lock(m_doneMutex);
            m_done.wait(lock, [&] { return m_remaining.load(std::memory_order_acquire) == 0; });
        }
        m_body = nullptr;
    }

private:
    Pool() {
        started() = true;
        const std::vector<size_t>& cpus = planned();

        if (!cpus.empty()) {
            /* the caller already sits on cpus[0]; a worker for each of the rest */
            for (size_t i = 1; i < cpus.size(); ++i)
                m_workers.emplace_back([this, i, cpu = cpus[i]] {
                    if (!cores::pinCurrentThread(cpu)) pinFailures().fetch_add(1);
                    worker(i);
                });
            return;
        }

        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware == 0) hardware = 1;

        for (unsigned i = 1; i < hardware; ++i)
            m_workers.emplace_back([this, i] { worker(i); });
    }

    static bool& started()                        { static bool s = false; return s; }
    static std::vector<size_t>& planned()         { static std::vector<size_t> c; return c; }
    static std::atomic<size_t>& pinFailures()     { static std::atomic<size_t> n{0}; return n; }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping.store(true, std::memory_order_relaxed);
            m_epoch.fetch_add(1, std::memory_order_release);
        }
        m_wake.notify_all();
        for (std::thread& t : m_workers) if (t.joinable()) t.join();
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    /* this thread's one slice of the current job */
    void work(size_t index) {
        const auto [begin, end] = slice(m_count, index, m_threads);
        if (begin < end) (*m_body)(begin, end);
    }

    static void relax() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }

    /*
        spinUntil(done):
        - polls for up to a fifth of a millisecond before giving up. A token
          is several hundred dispatches separated by a few microseconds of
          single-threaded work, and a condition-variable sleep and wake costs
          tens of microseconds each time - more than many of the dispatches
          themselves. Bounded, so a pool waiting on the user's next prompt
          goes to sleep rather than burning every CPU it owns
    */
    template <typename Done>
    static bool spinUntil(Done done) {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        for (size_t i = 0; ; ++i) {
            if (done()) return true;
            relax();
            if ((i & 255) == 255 && std::chrono::steady_clock::now() > limit) return false;
        }
    }

    void worker(size_t index) {
        size_t seen = 0;
        while (true) {
            const auto ready = [&] { return m_epoch.load(std::memory_order_acquire) != seen; };
            if (!spinUntil(ready)) {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait(lock, ready);
            }
            if (m_stopping.load(std::memory_order_relaxed)) return;
            seen = m_epoch.load(std::memory_order_acquire);

            work(index);

            if (m_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(m_doneMutex);
                m_done.notify_one();
            }
        }
    }

    std::vector<std::thread> m_workers;

    std::mutex               m_mutex;
    std::condition_variable  m_wake;
    std::atomic<size_t>      m_epoch{0};
    std::atomic<bool>        m_stopping{false};

    const std::function<void(size_t, size_t)>* m_body = nullptr;
    size_t                   m_count = 0;
    size_t                   m_threads = 1;

    std::atomic<size_t>      m_remaining{0};
    std::mutex               m_doneMutex;
    std::condition_variable  m_done;
};


} // !namespace kernels
} // !namespace ai
