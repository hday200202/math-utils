#include <iostream>
#include <chrono>

#include "../include/datetime/Datetime.hpp"
#include "../include/structs/LinkedList.hpp"

int main(int argc, char* argv[]) {
    using namespace dtm;

    std::vector<float> a;
    std::vector<float> b;

    Clock idxClock;
    Clock itrClock;

    structs::LinkedList<float> ll;
    for (int i = 0; i < 1000000; ++i) ll.append(i);

    idxClock.start();
    // for (size_t i = 0; i < ll.size(); ++i) a.push_back(ll[i]->data());
    idxClock.stop();

    std::cout << std::endl;

    itrClock.start();
    for (auto& node : ll) b.push_back(node);
    itrClock.stop();

    std::cout << "   Index Time: " << idxClock.elapsed().asMillis() << " ms" << std::endl;
    std::cout << "Iterator Time: " << itrClock.elapsed().asMillis() << " ms" << std::endl;
    std::cout 
        << "Iterator is " << 
            idxClock.elapsed().asMillis() / itrClock.elapsed().asMillis() 
        << "X faster. " << std::endl;

    return 0;
}