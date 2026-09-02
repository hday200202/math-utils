#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <iostream>
#include <format>

namespace dtm {

struct Week {
    static constexpr int64_t    days    = 7;        // days in a week
    static constexpr int64_t    hours   = 168;      // hours in a week
    static constexpr int64_t    minutes = 10080;    // minutes in a week
    static constexpr int64_t    seconds = 604800;   // seconds in a week
};

struct Day {
    static constexpr int64_t    hours   = 24;       // hours in a day
    static constexpr int64_t    minutes = 1440;     // minutes in a day
    static constexpr int64_t    seconds = 86400;    // seconds in a day
};

struct Hour {
    static constexpr int64_t    minutes = 60;       // minutes in an hour
    static constexpr int64_t    seconds = 3600;     // seconds in an hour
};

struct Minute {
    static constexpr int64_t    seconds = 60;       // seconds in a minute
};

inline bool isLeapYear(const int64_t year) { return ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0); }

inline int64_t getNumDays(const int64_t month, const int64_t year) {
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
            std::cerr                               <<
                "Month: "                           <<
                month                               <<
                " is not a month. Must be [1-12]."  <<
            std::endl;
            return -1;
    }
}

inline std::string monthToString(const int64_t month) {
    switch (month) {
        case  1:    return "January";
        case  2:    return "February";
        case  3:    return "March";
        case  4:    return "April";
        case  5:    return "May";
        case  6:    return "June";
        case  7:    return "July";
        case  8:    return "August";
        case  9:    return "September";
        case 10:    return "October";
        case 11:    return "November";
        case 12:    return "December";
        default:
            std::cerr                               <<
                "Month: "                           <<
                month                               <<
                " is not a month. Must be [1-12]."  <<
            std::endl;
            return "";
    }
}

struct Date { 
    int64_t year; 
    int64_t month; 
    int64_t day;
    int64_t hours;
    int64_t minutes;
    int64_t seconds;
};

inline int64_t daysFromCivil(const int64_t day, const int64_t month, const int64_t year) {
    int64_t d = day;
    int64_t m = month;
    int64_t y = year;
    y -= (m <= 2) ? 1 : 0;
    int64_t era = (y >= 0 ? y : y - 399) / 400;
    int64_t yoe = y - era * 400;
    int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    int64_t doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    return era * 146097 + doe - 719468;
}

inline Date civilFromDays(const int64_t daysSinceEpoch) {
    int64_t z = daysSinceEpoch + 719468;
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    int64_t doe = z - era * 146097;
    int64_t yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    int64_t y = yoe + era * 400;
    int64_t doy = doe - (365*yoe + yoe/4 - yoe/100);
    int64_t mp = (5*doy + 2) / 153;
    int64_t d = doy - (153*mp + 2)/5 + 1;
    int64_t m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2) ? 1 : 0;
    return {y, m, d, 0, 0, 0};
}

inline int64_t floorDiv(const int64_t a, const int64_t b) {
    int64_t q = a / b;
    if (a % b != 0 && (a < 0) != (b < 0)) --q;
    return q;
}

inline int64_t secondsToDays(const int64_t seconds) {
    return floorDiv(seconds, Day::seconds);
}

enum class DateFormat {
    LE,         // DD-MM-YYYY
    BE,         // YYYY-MM-DD
    ME,         // MM-DD-YYYY
    Long        // Month, D, YYYY
};

class Timestamp {
public:
    Timestamp() {}
    Timestamp(const int64_t seconds) : m_seconds(seconds) {}
    Timestamp(const Date& date) {
        m_seconds =
            (daysFromCivil(date.day, date.month, date.year) * Day::seconds )    +
            (date.hours                                     * Hour::seconds)    +
            (date.minutes                                   * Minute::seconds)  +
            date.seconds;
    }

    int64_t asSeconds()     const { return m_seconds; }
    int64_t asMinutes()     const { return m_seconds / Minute::seconds; }
    int64_t asHours()       const { return m_seconds / Hour::seconds; }
    int64_t asDays()        const { return m_seconds / Day::seconds; }
    int64_t asWeeks()       const { return m_seconds / Week::seconds; }

    Date toDate() const {
        int64_t days    = secondsToDays(m_seconds);
        int64_t tod     = m_seconds - days * Day::seconds;

        Date d          = civilFromDays(days);
        d.hours         = tod / Hour::seconds;
        d.minutes       = (tod % Hour::seconds) / Minute::seconds;
        d.seconds       = tod % Minute::seconds;

        return d;
    }

    std::string toString(
        const DateFormat dateFormat = DateFormat::ME, 
        const char separator = '-'
    ) const {
        using df = DateFormat;
        Date date = toDate();

        switch (dateFormat) {
            case df::BE:    return std::format("{:04}{}{:02}{}{:02}", date.year, separator, date.month, separator, date.day);
            case df::LE:    return std::format("{:02}{}{:02}{}{:04}", date.day, separator, date.month, separator, date.year);
            case df::ME:    return std::format("{:02}{}{:02}{}{:04}", date.month, separator, date.day, separator, date.year);
            case df::Long:  return std::format("{} {:02}, {:04}", monthToString(date.month), date.day, date.year);
            default:        return std::format("{:04}{}{:02}{}{:02}", date.year, separator, date.month, separator, date.day);
        }
    };

    Timestamp operator+(const Timestamp& t)     const { return Timestamp(m_seconds + t.m_seconds); }
    Timestamp operator-(const Timestamp& t)     const { return Timestamp(m_seconds - t.m_seconds); }
    Timestamp& operator+=(const Timestamp& t)   { m_seconds += t.m_seconds; return *this; };
    Timestamp& operator-=(const Timestamp& t)   { m_seconds -= t.m_seconds; return *this; }
    
    bool operator<(const Timestamp& t)          const { return m_seconds < t.m_seconds; }
    bool operator>(const Timestamp& t)          const { return m_seconds > t.m_seconds; }
    bool operator==(const Timestamp& t)         const { return m_seconds == t.m_seconds; }
    bool operator!=(const Timestamp& t)         const { return m_seconds != t.m_seconds; }
    bool operator<=(const Timestamp& t)         const { return m_seconds <= t.m_seconds; }
    bool operator>=(const Timestamp& t)         const { return m_seconds >= t.m_seconds; }

private:
    int64_t m_seconds = 0;
};

}