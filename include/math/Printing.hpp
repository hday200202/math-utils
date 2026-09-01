#pragma once

#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <vector>

#include "Matrix.hpp"

namespace math {

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Mat2<T>& m) {
    std::string cells[2][2];
    size_t width = 0;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::ostringstream cell;
            cell << m[i][j];
            cells[i][j] = cell.str();
            width = std::max(width, cells[i][j].size());
        }
    }

    std::ios_base::fmtflags oldFlags = os.flags();
    os << std::right;

    for (int i = 0; i < 2; ++i) {
        os << "\n| ";
        for (int j = 0; j < 2; ++j) {
            os << std::setw(static_cast<int>(width)) << cells[i][j];
            if (j < 1) os << ", ";
        }
        os << " |";
    }

    os.flags(oldFlags);
    return os;
}

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Mat3<T>& m) {
    std::string cells[3][3];
    size_t width = 0;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::ostringstream cell;
            cell << m[i][j];
            cells[i][j] = cell.str();
            width = std::max(width, cells[i][j].size());
        }
    }

    std::ios_base::fmtflags oldFlags = os.flags();
    os << std::right;

    for (int i = 0; i < 3; ++i) {
        os << "\n| ";
        for (int j = 0; j < 3; ++j) {
            os << std::setw(static_cast<int>(width)) << cells[i][j];
            if (j < 2) os << ", ";
        }
        os << " |";
    }

    os.flags(oldFlags);
    return os;
}

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Mat4<T>& m) {
    std::string cells[4][4];
    size_t width = 0;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::ostringstream cell;
            cell << m[i][j];
            cells[i][j] = cell.str();
            width = std::max(width, cells[i][j].size());
        }
    }

    std::ios_base::fmtflags oldFlags = os.flags();
    os << std::right;

    for (int i = 0; i < 4; ++i) {
        os << "\n| ";
        for (int j = 0; j < 4; ++j) {
            os << std::setw(static_cast<int>(width)) << cells[i][j];
            if (j < 3) os << ", ";
        }
        os << " |";
    }

    os.flags(oldFlags);
    return os;
}

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Vec2<T>& v)
{ return os << "<" << v.x << ", " << v.y << ">"; }

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Vec3<T>& v)
{ return os << "<" << v.x << ", " << v.y << ", " << v.z << ">"; }

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Vec4<T>& v)
{ return os << "<" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ">"; }


/* ========================================================================= //
    VecN / MatNxM (runtime-sized) - same math::utils namespace as Vec2/3/4
// ========================================================================= */

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Mat<T>& m) {
    const size_t N = m.numRows();
    const size_t M = m.numCols();

    std::vector<std::vector<std::string>> cells(N, std::vector<std::string>(M));
    size_t width = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            std::ostringstream cell;
            cell << m[i][j];
            cells[i][j] = cell.str();
            width = std::max(width, cells[i][j].size());
        }
    }

    std::ios_base::fmtflags oldFlags = os.flags();
    os << std::right;

    for (size_t i = 0; i < N; ++i) {
        os << "\n| ";
        for (size_t j = 0; j < M; ++j) {
            os << std::setw(static_cast<int>(width)) << cells[i][j];
            if (j + 1 < M) os << ", ";
        }
        os << " |";
    }

    os.flags(oldFlags);
    return os;
}

template <typename T> std::ostream& operator<<(std::ostream& os, const math::Vec<T>& v) {
    os << "<";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << ">";
    return os;
}

} // !namespace math::utils
