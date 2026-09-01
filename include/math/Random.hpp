#pragma once

#include <random>

#include "Vector.hpp"
#include "Matrix.hpp"

namespace math::random {

/* ========================================================================= //
    Random initialization
// ========================================================================= */

/*
    randomVec(rng, n, mean, stddev):
    - builds a length-n vector whose entries are drawn from a normal
      distribution; pass an explicit rng so results are seedable and
      reproducible rather than silently random every time
*/
template <typename T> math::Vec<T> randomVec(std::mt19937& rng, size_t n, T mean = T(0), T stddev = T(1)) {
    std::normal_distribution<T> dist(mean, stddev);
    math::Vec<T> out(n);

    for (size_t i = 0; i < n; ++i)
        out[i] = dist(rng);

    return out;
}

/*
    randomMat(rng, n, m, mean, stddev):
    - builds an n x m matrix whose entries are drawn from a normal
      distribution; same reproducibility note as randomVec. Built one row
      at a time since Mat<T> is stored as n rows of Vec<T>
*/
template <typename T> math::Mat<T> randomMat(std::mt19937& rng, size_t n, size_t m, T mean = T(0), T stddev = T(1)) {
    math::Mat<T> out(n, m);

    for (size_t i = 0; i < n; ++i)
        out[i] = randomVec<T>(rng, m, mean, stddev);

    return out;
}

} // !namespace math::utils
