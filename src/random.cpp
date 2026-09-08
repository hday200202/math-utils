#include "../include/math/Random.hpp"
#include "../include/datetime/Clock.hpp"
#include <iostream>
#include <algorithm>

int main() {
    using namespace math::random;
    Random rng = Random(397);
    rng.seed(dtm::Clock::now().asNanos());
    std::vector<int64_t> randoms;
    randoms.reserve(10);

    for (int i = 0; i < 10; ++i)
        randoms.push_back(rng.randInt(0, 2));

    std::cout << std::endl;
    for (auto& r : randoms) std::cout << r << std::endl;
    std::cout << std::endl;

    std::sort(randoms.begin(), randoms.end());

    std::cout << std::endl;
    for (auto& r : randoms) std::cout << r << std::endl;
    std::cout << std::endl;
}