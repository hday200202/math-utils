#include <iostream>
#include <chrono>

#include "../include/datetime/Datetime.hpp"
#include "../include/structs/LinkedList.hpp"

int main(int argc, char* argv[]) {
    using namespace dtm;

    Timestamp ts(Clock::now().asSeconds());
    std::cout << ts.toString(DTMFormat::LE | DTMFormat::HM12) << std::endl;

    
    structs::LinkedList<float> ll;
    
    std::cout << ll.getRoot() << std::endl;

    ll.append(1);
    ll.append(2);
    ll.append(3);
    ll.append(4);

    for (size_t i = 0; i < ll.size(); ++i)
        std::cout << ll[i] << std::endl;

    ll.erase(0);

    for (size_t i = 0; i < ll.size(); ++i)
        std::cout << ll[i] << std::endl;

    return 0;
}