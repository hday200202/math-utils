#include <iostream>
#include <chrono>

#include "../include/datetime/Datetime.hpp"
#include "../include/structs/LinkedList.hpp"

int main(int argc, char* argv[]) {
    using namespace dtm;

    Timestamp ts(Clock::now().asSeconds());
    std::cout << ts.toString(DTMFormat::LE | DTMFormat::HM12) << std::endl;

    structs::LinkedList<float> ll;
    ll.append(2);
    ll.prepend(1);
    ll.append(3);

    std::cout << ll[2] << std::endl;

    return 0;
}