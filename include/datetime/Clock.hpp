#pragma once

#include <chrono>
#include "Timestamp.hpp"

namespace dtm {

class Clock {
public:
    class Duration { 
    public:
        Duration() = default;
        explicit Duration(int64_t nanoseconds) : m_nanoseconds(nanoseconds) {};

        double asSeconds() const { return m_nanoseconds / 1000000000.0; }
        double asMillis() const { return m_nanoseconds / 1000000.0; }
        double asMicros() const { return m_nanoseconds / 1000.0; }
        int64_t asNanos() const { return m_nanoseconds; }

    private:
        int64_t m_nanoseconds = 0;
    };

    Clock() = default;

    /*
        now():
        - wall-clock time, as nanoseconds since the unix epoch. For a calendar
          value: Timestamp(Clock::now().asSeconds())
        - system_clock, so it can step when the machine's clock is corrected.
          That is fine for "what time is it" and wrong for measuring an
          interval, which is why the stopwatch below uses steadyNanos instead
    */
    static Duration now() {
        using namespace std::chrono;
        return Duration(
            duration_cast<std::chrono::nanoseconds>(
                system_clock::now().time_since_epoch()
            ).count()
        );
    }

    /*
        start():
        - begins timing. Does nothing if already running, so the interval in
          flight is never silently discarded
    */
    void start() {
        if (m_started) return;
        m_start = steadyNanos();
        m_started = true;
    }

    /*
        stop():
        - adds the interval in flight to the running total and halts. Does
          nothing if already stopped, so a second call cannot double count
    */
    Duration stop() {
        if (!m_started) return Duration(m_nanoseconds);

        m_nanoseconds += steadyNanos() - m_start;
        m_started = false;
        return Duration(m_nanoseconds);
    }

    /*
        reset():
        - clears the running total without stopping. The interval in flight is
          discarded too, so a running clock starts counting again from zero
    */
    void reset() {
        m_nanoseconds = 0;
        m_start = steadyNanos();
    }

    /*
        elapsed():
        - everything accumulated so far, plus the interval in flight if the
          clock is currently running. Reads zero on a clock that was never
          started, and stays frozen once stopped
    */
    Duration elapsed() const {
        return Duration(m_nanoseconds + (m_started ? steadyNanos() - m_start : 0));
    }

    /*
        lap():
        - the elapsed time, then reset, without stopping. What you want inside
          a loop when each iteration should be timed on its own
    */
    Duration lap() {
        const Duration total = elapsed();
        reset();
        return total;
    }

    bool isRunning() const { return m_started; }

private:
    /*
        steadyNanos():
        - steady_clock is monotonic, so an interval measured with it can never
          come out negative the way one measured with system_clock can. Its
          epoch is arbitrary, which is why it is only ever used for differences
    */
    static int64_t steadyNanos() {
        using namespace std::chrono;
        return duration_cast<std::chrono::nanoseconds>(
            steady_clock::now().time_since_epoch()
        ).count();
    }

    int64_t     m_start             = 0;
    int64_t     m_nanoseconds       = 0;
    bool        m_started           = false;
};

}
