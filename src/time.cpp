#include <iostream>
#include <chrono>

#include "../include/datetime/Datetime.hpp"

int main(int argc, char* argv[]) {
    using namespace dtm;

    Timestamp ts(Clock::now().asSeconds());
    std::cout << ts.toString(DTMFormat::LE | DTMFormat::HM12) << std::endl;

    return 0;
}