#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <cstdint>
#include <string>
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

/*
    isValidDate(date):
    - true only if every field is in range for that specific date, so
      2023-02-29 is rejected while 2024-02-29 is accepted.
    - nothing calls this for you. Timestamp will happily accept a nonsense Date
      and normalise it into a real one -- month 13 rolls into the next year,
      Feb 30 becomes Mar 2 -- which is fine for arithmetic and terrible for
      parsing, where a typo would silently become a plausible wrong date. Check
      before constructing whenever the Date came from outside the program
    - the month is tested first so an out-of-range month never reaches
      getNumDays, which reports to cerr
    - seconds stop at 59: unix time has no leap seconds, so neither does this
    - the year is deliberately unbounded; the civil-date algorithm handles
      negative years, and clamping to a range would be an arbitrary policy
*/
inline bool isValidDate(const Date& date) {
    if (date.month   < 1 || date.month   > 12) return false;
    if (date.day     < 1 || date.day     > getNumDays(date.month, date.year)) return false;
    if (date.hours   < 0 || date.hours   > 23) return false;
    if (date.minutes < 0 || date.minutes > 59) return false;
    if (date.seconds < 0 || date.seconds > 59) return false;

    return true;
}

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

/*
    DTMFormat:
    - bit flags combined with |, e.g. DTMFormat::LE | DTMFormat::HMS12 | DTMFormat::TimeDate
    - the flags fall into three groups that are each mutually exclusive: a date
      layout, a time layout, and the order the two are joined in. Nothing stops
      you setting two flags from the same group, so the rule is first match wins
      in the order listed below
    - omit every time flag for a date-only string, or every date flag for a
      time-only one
*/
enum class DTMFormat : uint16_t {
    None        = 0,

    // date layout, pick one
    ISO8601     = 1 << 0,           // 2026-09-01T14:30:00Z, overrides every other flag
    Long        = 1 << 1,           // September 01, 2026
    BE          = 1 << 2,           // YYYY-MM-DD
    LE          = 1 << 3,           // DD-MM-YYYY
    ME          = 1 << 4,           // MM-DD-YYYY

    // time layout, pick at most one
    HMS12       = 1 << 5,           // 2:30:00 PM
    HM12        = 1 << 6,           // 2:30 PM
    HMS         = 1 << 7,           // 14:30:00
    HM          = 1 << 8,           // 14:30

    // arrangement, defaults to DateTime
    DateTime    = 1 << 9,           // date first
    TimeDate    = 1 << 10,          // time first

    DateMask    = ISO8601 | Long | BE | LE | ME,
    TimeMask    = HMS12 | HM12 | HMS | HM,
    OrderMask   = DateTime | TimeDate,
};

constexpr DTMFormat operator|(const DTMFormat a, const DTMFormat b) {
    return static_cast<DTMFormat>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

constexpr DTMFormat operator&(const DTMFormat a, const DTMFormat b) {
    return static_cast<DTMFormat>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

constexpr DTMFormat operator~(const DTMFormat a) {
    return static_cast<DTMFormat>(~static_cast<uint16_t>(a));
}

constexpr DTMFormat& operator|=(DTMFormat& a, const DTMFormat b) { a = a | b; return a; }
constexpr DTMFormat& operator&=(DTMFormat& a, const DTMFormat b) { a = a & b; return a; }

constexpr bool dtmFmtHas(const DTMFormat value, const DTMFormat flag) {
    return (static_cast<uint16_t>(value) & static_cast<uint16_t>(flag)) != 0;
}



/*
    TimeZone:
    - a FIXED offset from UTC, in seconds. A Timestamp always stores UTC; the
      offset is applied only when converting to a Date or printing
    - seconds rather than hours because not every zone is a whole hour: India
      is +05:30 and Nepal is +05:45
    - daylight saving is NOT applied for you. That needs a tz database and a
      date to look up, so standard and daylight variants are listed separately
      and you pick the right one. Several share an offset (PDT and MST are both
      -07:00), which is why printing shows a numeric offset rather than a name
    - for a zone that is not listed, cast directly: static_cast<TimeZone>(-9000)
*/
enum class TimeZone : int32_t {
    UTC     =        0,
    GMT     =        0,

    HST     = -10 * 3600,           // Hawaii
    AKST    =  -9 * 3600,           // Alaska, standard
    PST     =  -8 * 3600,           // US Pacific, standard
    PDT     =  -7 * 3600,           // US Pacific, daylight
    MST     =  -7 * 3600,           // US Mountain, standard
    MDT     =  -6 * 3600,           // US Mountain, daylight
    CST     =  -6 * 3600,           // US Central, standard
    CDT     =  -5 * 3600,           // US Central, daylight
    EST     =  -5 * 3600,           // US Eastern, standard
    EDT     =  -4 * 3600,           // US Eastern, daylight
    AST     =  -4 * 3600,           // Atlantic, standard
    BRT     =  -3 * 3600,           // Brazil
    CET     =   1 * 3600,           // Central Europe, standard
    CEST    =   2 * 3600,           // Central Europe, summer
    EET     =   2 * 3600,           // Eastern Europe, standard
    EEST    =   3 * 3600,           // Eastern Europe, summer
    MSK     =   3 * 3600,           // Moscow
    IST     =   5 * 3600 + 1800,    // India, +05:30
    NPT     =   5 * 3600 + 2700,    // Nepal, +05:45
    ICT     =   7 * 3600,           // Indochina
    CST_CN  =   8 * 3600,           // China, named apart from the US CST
    JST     =   9 * 3600,           // Japan
    AEST    =  10 * 3600,           // Australia Eastern, standard
    AEDT    =  11 * 3600,           // Australia Eastern, daylight
    NZST    =  12 * 3600,           // New Zealand
};

constexpr int64_t timeZoneOffset(const TimeZone zone) { return static_cast<int64_t>(zone); }

/*
    timeZoneToString(zone):
    - the ISO 8601 offset, "Z" for UTC and otherwise "+05:30" / "-08:00"
*/
inline std::string timeZoneToString(const TimeZone zone) {
    const int64_t offset = timeZoneOffset(zone);
    if (offset == 0) return "Z";

    const int64_t magnitude = (offset < 0) ? -offset : offset;
    return std::format("{}{:02}:{:02}",
        (offset < 0) ? '-' : '+',
        magnitude / Hour::seconds,
        (magnitude % Hour::seconds) / Minute::seconds);
}


/*
    systemTimeZone():
    - the machine's current offset from UTC, including whatever daylight saving
      is in effect right now.
    - std::chrono::current_zone() would be the modern answer, but it needs
      __cpp_lib_chrono >= 201907 and is still missing on plenty of toolchains,
      so this uses the portable C route: render the instant as UTC fields, ask
      mktime to read those fields back as local time, and the gap between the
      two is the offset.
    - computed once. The result is cached in a function-local static, which is
      thread-safe to initialise and means gmtime is only ever called once
      (it returns a pointer to shared storage, so calling it repeatedly from
      several threads would be a race). The cost is that a process running
      across a daylight-saving boundary keeps the offset it started with
*/
inline TimeZone systemTimeZone() {
    static const TimeZone cached = []() -> TimeZone {
        const std::time_t now = std::time(nullptr);
        std::tm utc{};

    #if defined(_WIN32)
        gmtime_s(&utc, &now);
    #else
        gmtime_r(&now, &utc);
    #endif

        utc.tm_isdst = -1;
        return static_cast<TimeZone>(static_cast<int32_t>(now - std::mktime(&utc)));
    }();

    return cached;
}

class Timestamp {
public:
    Timestamp() {}
    Timestamp(const int64_t seconds) : m_seconds(seconds) {}


    /*
        - the Date is read as a local time in `zone`, and UTC is what gets
          stored. `zone` also becomes this Timestamp's display zone
    */
    Timestamp(const Date& date, const TimeZone zone = systemTimeZone()) {
        set(date, zone);
    }


    /*
        - from a std::filesystem::file_time_type, ie. what last_write_time
          returns. `zone` is display only, as everywhere else
    */
    Timestamp(
        const std::filesystem::file_time_type& fileTime, 
        const TimeZone zone = systemTimeZone()
    ) { set(fileTime, zone); }

    void set(const int64_t seconds) { m_seconds = seconds; }


    /*
        - file_time_type is measured by file_clock, a different clock from
          system_clock with its own epoch, so the two cannot just be cast into
          one another.
        - C++20 added file_clock::to_sys for exactly this, but it needs
          __cpp_lib_chrono >= 201907 and is still missing on plenty of
          toolchains, so the fallback reads both clocks now and shifts by the
          gap between them. That samples them microseconds apart, which is far
          below the second resolution stored here
    */
    void set(const std::filesystem::file_time_type& fileTime, const TimeZone zone = systemTimeZone()) {
        using namespace std::chrono;

        #if defined(__cpp_lib_chrono) && __cpp_lib_chrono >= 201907L
        const auto sys = file_clock::to_sys(fileTime);
        #else
        const auto sys = time_point_cast<system_clock::duration>(
            fileTime - std::filesystem::file_time_type::clock::now() + system_clock::now());
        #endif

        m_zone    = zone;
        m_seconds = duration_cast<std::chrono::seconds>(sys.time_since_epoch()).count();
    }


    void set(const Date& date, const TimeZone zone = systemTimeZone()) {
        m_zone = zone;
        m_seconds =
            (daysFromCivil(date.day, date.month, date.year) * Day::seconds )    +
            (date.hours                                     * Hour::seconds)    +
            (date.minutes                                   * Minute::seconds)  +
            date.seconds
            - timeZoneOffset(zone);
    }


    /*
        - the zone is a display preference only. It never changes the instant,
          so it takes no part in comparison or arithmetic
    */
    void     setTimeZone(const TimeZone zone)   { m_zone = zone; }
    TimeZone getTimeZone()              const   { return m_zone; }

    int64_t asSeconds()     const { return m_seconds; }
    int64_t asMinutes()     const { return m_seconds / Minute::seconds; }
    int64_t asHours()       const { return m_seconds / Hour::seconds; }
    int64_t asDays()        const { return m_seconds / Day::seconds; }
    int64_t asWeeks()       const { return m_seconds / Week::seconds; }


    /*
        - the offset is applied before the split, so a zone that pushes the
          time past midnight rolls the date with it
    */
    Date toDate() const { return toDate(m_zone); }

    Date toDate(const TimeZone zone) const {
        int64_t local   = m_seconds + timeZoneOffset(zone);
        int64_t days    = secondsToDays(local);
        int64_t tod     = local - days * Day::seconds;

        Date d          = civilFromDays(days);
        d.hours         = tod / Hour::seconds;
        d.minutes       = (tod % Hour::seconds) / Minute::seconds;
        d.seconds       = tod % Minute::seconds;

        return d;
    }


    /*
        - prints in this Timestamp's own zone, which came from the system
          unless setTimeZone changed it
    */
    std::string toString(
        const DTMFormat format      = DTMFormat::ISO8601,
        const char      separator   = '-'
    ) const { return toString(format, m_zone, separator); }


    /*
        - prints in `zone`, leaving this Timestamp's own zone alone
    */
    std::string toString(
        const DTMFormat format,
        const TimeZone  zone,
        const char      separator   = '-'
    ) const {
        const Date date = toDate(zone);

        // ISO 8601 pins the layout, the order and the suffix, so it ignores
        // every other flag rather than trying to combine with them
        if (dtmFmtHas(format, DTMFormat::ISO8601))
            return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}",
                date.year, date.month, date.day,
                date.hours, date.minutes, date.seconds,
                timeZoneToString(zone));

        const std::string dateStr = formatDate(date, format, separator);
        const std::string timeStr = formatTime(date, format, zone);

        if (dateStr.empty()) return timeStr;
        if (timeStr.empty()) return dateStr;

        return dtmFmtHas(format, DTMFormat::TimeDate)
            ? timeStr + " " + dateStr
            : dateStr + " " + timeStr;
    }


    Timestamp operator+(const Timestamp& t)     const { Timestamp r(m_seconds + t.m_seconds); r.setTimeZone(m_zone); return r; }
    Timestamp operator-(const Timestamp& t)     const { Timestamp r(m_seconds - t.m_seconds); r.setTimeZone(m_zone); return r; }
    Timestamp& operator+=(const Timestamp& t)   { m_seconds += t.m_seconds; return *this; }
    Timestamp& operator-=(const Timestamp& t)   { m_seconds -= t.m_seconds; return *this; }
    

    bool operator<(const Timestamp& t)          const { return m_seconds < t.m_seconds; }
    bool operator>(const Timestamp& t)          const { return m_seconds > t.m_seconds; }
    bool operator==(const Timestamp& t)         const { return m_seconds == t.m_seconds; }
    bool operator!=(const Timestamp& t)         const { return m_seconds != t.m_seconds; }
    bool operator<=(const Timestamp& t)         const { return m_seconds <= t.m_seconds; }
    bool operator>=(const Timestamp& t)         const { return m_seconds >= t.m_seconds; }

private:
    int64_t  m_seconds = 0;
    TimeZone m_zone    = systemTimeZone();


    /*
        - first flag set wins, so a format with two date flags is still defined.
          Returns "" when no date flag is set at all, giving a time-only string
    */
    static std::string formatDate(const Date& date, const DTMFormat format, const char separator) {
        if (dtmFmtHas(format, DTMFormat::Long))
            return std::format("{} {:02}, {:04}", monthToString(date.month), date.day, date.year);
        if (dtmFmtHas(format, DTMFormat::BE))
            return std::format("{:04}{}{:02}{}{:02}", date.year, separator, date.month, separator, date.day);
        if (dtmFmtHas(format, DTMFormat::LE))
            return std::format("{:02}{}{:02}{}{:04}", date.day, separator, date.month, separator, date.year);
        if (dtmFmtHas(format, DTMFormat::ME))
            return std::format("{:02}{}{:02}{}{:04}", date.month, separator, date.day, separator, date.year);

        return {};
    }


    /*
        - same first-wins rule. Returns "" when no time flag is set, giving a
          date-only string. The zone is appended only when it is not UTC
    */
    static std::string formatTime(const Date& date, const DTMFormat format, const TimeZone zone) {
        const int64_t     hour12   = ((date.hours % 12) == 0) ? 12 : (date.hours % 12);
        const char* const meridiem = (date.hours < 12) ? "AM" : "PM";
        const std::string suffix   = (timeZoneOffset(zone) == 0) ? std::string{}
                                                                 : " " + timeZoneToString(zone);

        if (dtmFmtHas(format, DTMFormat::HMS12))
            return std::format("{}:{:02}:{:02} {}", hour12, date.minutes, date.seconds, meridiem) + suffix;
        if (dtmFmtHas(format, DTMFormat::HM12))
            return std::format("{}:{:02} {}", hour12, date.minutes, meridiem) + suffix;
        if (dtmFmtHas(format, DTMFormat::HMS))
            return std::format("{:02}:{:02}:{:02}", date.hours, date.minutes, date.seconds) + suffix;
        if (dtmFmtHas(format, DTMFormat::HM))
            return std::format("{:02}:{:02}", date.hours, date.minutes) + suffix;

        return {};
    }
};

}