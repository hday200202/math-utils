#pragma once

#include <cstdint>
#include <cmath>

#include "Vector.hpp"
#include "Matrix.hpp"

namespace math::random {

/* ========================================================================= //
    Random Generation
// ========================================================================= */

class Random {
public:
    using result_type = uint64_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }
    result_type operator()() { return next(); }

    explicit Random(const uint64_t s) { seed(s); }
    void seed(uint64_t s) {
        for (uint64_t& w : m_state) w = splitmix64(s);
        m_hasSpare = false;
    }

    double u01() { return (next() >> 11) * 0x1.0p-53; }

    double randDouble(const double lo, const double hi) { return lo + u01() * (hi - lo); }

    int64_t randInt(const int64_t lo, const int64_t hi) {
        uint64_t span = uint64_t(hi) - uint64_t(lo) + 1;
        if (span == 0) return int64_t(next());
        uint64_t limit = UINT64_MAX - (UINT64_MAX % span);
        uint64_t x;
        do { x = next(); } while (x >= limit);
        return lo + int64_t(x % span);
    }

    double normal(const double mean = 0.0, const double stddev = 1.0) {
        if (m_hasSpare) {
            m_hasSpare = false;
            return mean + stddev * m_spare;
        }

        double u1 = 1.0 - u01();
        double u2 = u01();

        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;

        m_spare = r * std::sin(theta);
        m_hasSpare = true;

        return mean + stddev * (r * std::cos(theta));
    }

    bool chance(const double p) { return u01() < p; }
    double exponential(const double lambda) { return -std::log(1.0 - u01()) / lambda; }

private:
    uint64_t m_state[4] = {};
    double m_spare = 0.0;
    bool m_hasSpare = false;

    uint64_t next() {
        uint64_t result = rotl(m_state[0] + m_state[3], 23) + m_state[0];

        uint64_t t = m_state[1] << 17;
        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];
        m_state[2] ^= t;
        m_state[3] = rotl(m_state[3], 45);

        return result;
    }

    static inline uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
};



/* ========================================================================= //
    Random initialization
// ========================================================================= */

/*
    randomVec(rng, n, mean, stddev):
    - builds a length-n vector whose entries are drawn from a normal
      distribution; pass an explicit rng so results are seedable and
      reproducible rather than silently random every time
*/
template <typename T> math::Vec<T> randomVec(Random& rng, size_t n, T mean = T(0), T stddev = T(1)) {
    math::Vec<T> out(n);

    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<T>(rng.normal(static_cast<double>(mean), static_cast<double>(stddev)));

    return out;
}

/*
    randomMat(rng, n, m, mean, stddev):
    - builds an n x m matrix whose entries are drawn from a normal
      distribution; same reproducibility note as randomVec. Built one row
      at a time since Mat<T> is stored as n rows of Vec<T>
*/
template <typename T> math::Mat<T> randomMat(Random& rng, size_t n, size_t m, T mean = T(0), T stddev = T(1)) {
    math::Mat<T> out(n, m);

    for (size_t i = 0; i < n; ++i)
        out[i] = randomVec<T>(rng, m, mean, stddev);

    return out;
}

} // !namespace math::random
