#pragma once

#include <chrono>
#include <unordered_map>
#include <iostream>

namespace utils::datetime {

inline bool isLeapYear(const uint16_t year) { return ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0); }

// 1-indexed (Jan=1 .. Dec=12), matching daysFromCivil/civilFromDays' convention below
inline uint8_t getNumDays(const uint8_t month, const uint16_t year) {
    switch (month) {
        case  1:    return 31;
        case  2:    return 28 + isLeapYear(year);
        case  3:    return 31;
        case  4:    return 30;
        case  5:    return 31;
        case  6:    return 30;
        case  7:    return 31;
        case  8:    return 31;
        case  9:    return 30;
        case 10:    return 31;
        case 11:    return 30;
        case 12:    return 31;
        default:
            std::cerr                           <<
                "Month: "                       <<
                static_cast<int>(month)         <<
                " is not a month use [1-12]."   <<
            std::endl;
            return -1;
    }
}

struct Date { uint32_t year; uint32_t month; uint32_t day; };

inline int64_t daysFromCivil(const uint8_t day, const uint8_t month, const uint16_t year) {
    auto d = day;
    auto m = month;
    auto y = year;
    y -= (m <= 2) ? 1 : 0;
    auto era = (y >= 0 ? y : y - 399) / 400;
    auto yoe = y - era * 400;
    auto doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    auto doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    return era * 146097 + doe - 719468;
}

inline Date civilFromDays(const uint32_t daysSinceEpoch) {
    auto z = daysSinceEpoch + 719468;
    auto era = (z >= 0 ? z : z - 146096) / 146097;
    auto doe = z - era * 146097;
    auto yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    auto y = yoe + era * 400;
    auto doy = doe - (365*yoe + yoe/4 - yoe/100);
    auto mp = (5*doy + 2) / 153;
    auto d = doy - (153*mp + 2)/5 + 1;
    auto m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2) ? 1 : 0;
    return {y, m, d};
}

struct Week {
    static const uint8_t    days    = 7;
    static const uint8_t    hours   = 168;
    static const uint16_t   minutes = 10080;
    static const uint32_t   seconds = 604800;
};

struct Day {
    static const uint8_t    hours   = 24;
    static const uint16_t   minutes = 1440;
    static const uint32_t   seconds = 86400;
};

struct Hour {
    static const uint8_t    minutes = 60;
    static const uint16_t   seconds = 3600;
};

struct Minute {
    static const uint16_t   seconds = 60;
};

class Timestamp {
public:
    Timestamp() {}
    Timestamp(const uint64_t seconds) : m_seconds(seconds) {}
    Timestamp(const uint8_t day, const uint8_t month, const uint16_t year) {}

private:
    uint64_t m_seconds = 0;
};

}