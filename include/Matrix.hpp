#pragma once

#include "Vector.hpp"

namespace math {

// !!! All matrices are ROW-MAJOR !!!

/* ========================================================================= //
    Mat2
// ========================================================================= */

template <typename T> struct Mat2 {
    Vec2<T> rows[2] = {{1, 0}, {0, 1}};

    T* data() { return &rows[0][0]; }
    const T* data() const { return &rows[0][0]; }

    static Mat2 identity() { return { 1, 0, 0, 1 }; }

    Mat2 transpose() const {
        Vec2<T> row1 = { rows[0][0], rows[1][0] };
        Vec2<T> row2 = { rows[0][1], rows[1][1] };

        return { row1, row2 };
    }

    T determinant() const 
    { return rows[0][0]*rows[1][1] - rows[0][1]*rows[1][0]; }

    Mat2 inverse() const {
        T det = determinant();

        return Mat2{
             rows[1][1], -rows[0][1],
            -rows[1][0],  rows[0][0]
        } / det;
    }

    Vec2<T>& operator[](int i) { return rows[i]; }
    const Vec2<T>& operator[](int i) const { return rows[i]; }

    Mat2 operator+(const Mat2& m) const 
    { return { rows[0] + m.rows[0], rows[1] + m.rows[1] }; }

    Mat2 operator-(const Mat2& m) const 
    { return { rows[0] - m.rows[0], rows[1] - m.rows[1] }; }

    Mat2 operator*(const T s) const 
    { return { rows[0] * s, rows[1] * s }; }

    Mat2 operator/(const T s) const 
    { return { rows[0] / s, rows[1] / s }; }

    Mat2 operator*(const Mat2& m) const {
        Mat2 res;

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                T sum = 0;
                for (int k = 0; k < 2; ++k)
                    sum += rows[i][k] * m[k][j];
                res[i][j] = sum;
            }
        }

        return res;
    }

    Vec2<T> operator*(const Vec2<T>& v) const 
    { return { v.dot(rows[0]), v.dot(rows[1]) }; }

    bool operator==(const Mat2& m) const 
    { return rows[0] == m.rows[0] && rows[1] == m.rows[1]; }

    bool operator!=(const Mat2& m) const 
    { return rows[0] != m.rows[0] || rows[1] != m.rows[1]; }

    Mat2 operator+=(const Mat2& m)
    { return { rows[0] += m.rows[0], rows[1] += m.rows[1] }; }

    Mat2 operator-=(const Mat2& m)
    { return { rows[0] -= m.rows[0], rows[1] -= m.rows[1] }; }

    Mat2 operator*=(const T s)
    { return { rows[0] *= s, rows[1] *= s }; }

    Mat2 operator/=(const T s)
    { return { rows[0] /= s, rows[1] /= s }; }
};

using Mat2f = Mat2<float>;
using Mat2d = Mat2<double>;
using Mat2i = Mat2<int32_t>;
using Mat2u = Mat2<uint32_t>;

static_assert(std::is_standard_layout_v<Mat2f>, "Mat2 must be standard-layout");
static_assert(sizeof(Mat2f) == sizeof(float) * 4, "Mat2 has unexpected padding");



/* ========================================================================= //
    Mat3
// ========================================================================= */

template <typename T> struct Mat3 {
    Vec3<T> rows[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    T* data() { return &rows[0][0]; }
    const T* data() const { return &rows[0][0]; }

    static Mat3 identity() { return { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; }

    Mat3 transpose() const {
        Mat3 res;

        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                res[row][col] = rows[col][row];

        return res;
    }

    T determinant() const {
        T a, b, c, d, e, f, g, h, i;

        a = rows[0][0]; b = rows[0][1]; c = rows[0][2];
        d = rows[1][0]; e = rows[1][1]; f = rows[1][2];
        g = rows[2][0]; h = rows[2][1]; i = rows[2][2];

        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    }

    Mat3 inverse() const {
        T det = determinant();
        T a, b, c, d, e, f, g, h, i;

        a = rows[0][0]; b = rows[0][1]; c = rows[0][2];
        d = rows[1][0]; e = rows[1][1]; f = rows[1][2];
        g = rows[2][0]; h = rows[2][1]; i = rows[2][2];

        return Mat3{
            Vec3<T>{  (e*i - f*h), -(b*i - c*h),  (b*f - c*e) },
            Vec3<T>{ -(d*i - f*g),  (a*i - c*g), -(a*f - c*d) },
            Vec3<T>{  (d*h - e*g), -(a*h - b*g),  (a*e - b*d) }
        } / det;
    }

    Vec3<T>& operator[](int i) { return rows[i]; }
    const Vec3<T>& operator[](int i) const { return rows[i]; }

    Mat3 operator+(const Mat3& m) const 
    { return { rows[0] + m.rows[0], rows[1] + m.rows[1], rows[2] + m.rows[2] }; }

    Mat3 operator-(const Mat3& m) const 
    { return { rows[0] - m.rows[0], rows[1] - m.rows[1], rows[2] - m.rows[2] }; }

    Mat3 operator*(const T s) const 
    { return { rows[0] * s, rows[1] * s, rows[2] * s }; }

    Mat3 operator/(const T s) const 
    { return { rows[0] / s, rows[1] / s, rows[2] / s }; }

    Mat3 operator*(const Mat3& m) const {
        Mat3 res;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T sum = 0;
                for (int k = 0; k < 3; ++k)
                    sum += rows[i][k] * m[k][j];
                res[i][j] = sum;
            }
        }

        return res;
    }

    Vec3<T> operator*(const Vec3<T>& v) const 
    { return { v.dot(rows[0]), v.dot(rows[1]), v.dot(rows[2]) }; }

    bool operator==(const Mat3& m) const 
    { return rows[0] == m.rows[0] && rows[1] == m.rows[1] && rows[2] == m.rows[2]; }

    bool operator!=(const Mat3& m) const 
    { return rows[0] != m.rows[0] || rows[1] != m.rows[1] || rows[2] != m.rows[2]; }

    Mat3 operator+=(const Mat3& m)
    { return { rows[0] += m.rows[0], rows[1] += m.rows[1], rows[2] += m.rows[2] }; }

    Mat3 operator-=(const Mat3& m)
    { return { rows[0] -= m.rows[0], rows[1] -= m.rows[1], rows[2] -= m.rows[2] }; }

    Mat3 operator*=(const T s)
    { return { rows[0] *= s, rows[1] *= s, rows[2] *= s }; }

    Mat3 operator/=(const T s)
    { return { rows[0] /= s, rows[1] /= s, rows[2] /= s }; }
};

using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat3i = Mat3<int32_t>;
using Mat3u = Mat3<uint32_t>;

static_assert(std::is_standard_layout_v<Mat3f>, "Mat3 must be standard-layout");
static_assert(sizeof(Mat3f) == sizeof(float) * 9, "Mat3 has unexpected padding");



/* ========================================================================= //
    Mat4
// ========================================================================= */

template <typename T> struct Mat4 {
    Vec4<T> rows[4] = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };

    T* data() { return &rows[0][0]; }
    const T* data() const { return &rows[0][0]; }

    static Mat4 identity() { return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }; }

    Mat4 transpose() const {
        Mat4 res;

        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                res[row][col] = rows[col][row];

        return res;
    }

    T determinant() const {
        T a, b, c, d,
          e, f, g, h,
          i, j, k, l,
          m, n, o, p;

        a = rows[0][0]; b = rows[0][1]; c = rows[0][2]; d = rows[0][3];
        e = rows[1][0]; f = rows[1][1]; g = rows[1][2]; h = rows[1][3];
        i = rows[2][0]; j = rows[2][1]; k = rows[2][2]; l = rows[2][3];
        m = rows[3][0]; n = rows[3][1]; o = rows[3][2]; p = rows[3][3];

        Mat3<T> minora = { f, g, h, j, k, l, n, o, p };
        Mat3<T> minorb = { e, g, h, i, k, l, m, o, p };
        Mat3<T> minorc = { e, f, h, i, j, l, m, n, p };
        Mat3<T> minord = { e, f, g, i, j, k, m, n, o };

        return   a * minora.determinant()
               - b * minorb.determinant()
               + c * minorc.determinant()
               - d * minord.determinant();
    }

    Mat4 inverse() const {
        T a, b, c, d,
          e, f, g, h,
          i, j, k, l,
          m, n, o, p;

        T A, B, C, D,
          E, F, G, H,
          I, J, K, L,
          M, N, O, P;

        T det = determinant();

        a = rows[0][0]; b = rows[0][1]; c = rows[0][2]; d = rows[0][3];
        e = rows[1][0]; f = rows[1][1]; g = rows[1][2]; h = rows[1][3];
        i = rows[2][0]; j = rows[2][1]; k = rows[2][2]; l = rows[2][3];
        m = rows[3][0]; n = rows[3][1]; o = rows[3][2]; p = rows[3][3];

        Mat3<T> minora = { f, g, h, j, k, l, n, o, p };
        Mat3<T> minorb = { e, g, h, i, k, l, m, o, p };
        Mat3<T> minorc = { e, f, h, i, j, l, m, n, p };
        Mat3<T> minord = { e, f, g, i, j, k, m, n, o };

        Mat3<T> minore = { b, c, d, j, k, l, n, o, p };
        Mat3<T> minorf = { a, c, d, i, k, l, m, o, p };
        Mat3<T> minorg = { a, b, d, i, j, l, m, n, p };
        Mat3<T> minorh = { a, b, c, i, j, k, m, n, o };

        Mat3<T> minori = { b, c, d, f, g, h, n, o, p };
        Mat3<T> minorj = { a, c, d, e, g, h, m, o, p };
        Mat3<T> minork = { a, b, d, e, f, h, m, n, p };
        Mat3<T> minorl = { a, b, c, e, f, g, m, n, o };

        Mat3<T> minorm = { b, c, d, f, g, h, j, k, l };
        Mat3<T> minorn = { a, c, d, e, g, h, i, k, l };
        Mat3<T> minoro = { a, b, d, e, f, h, i, j, l };
        Mat3<T> minorp = { a, b, c, e, f, g, i, j, k };

        A =  minora.determinant(); B = -minorb.determinant(); C =  minorc.determinant(); D = -minord.determinant();
        E = -minore.determinant(); F =  minorf.determinant(); G = -minorg.determinant(); H =  minorh.determinant();
        I =  minori.determinant(); J = -minorj.determinant(); K =  minork.determinant(); L = -minorl.determinant();
        M = -minorm.determinant(); N =  minorn.determinant(); O = -minoro.determinant(); P =  minorp.determinant();

        return Mat4{
            A, E, I, M,
            B, F, J, N,
            C, G, K, O,
            D, H, L, P
        } / det;
    }

    Vec4<T>& operator[](int i) { return rows[i]; }
    const Vec4<T>& operator[](int i) const { return rows[i]; }

    Mat4 operator+(const Mat4& m) const 
    { return { rows[0] + m.rows[0], rows[1] + m.rows[1], rows[2] + m.rows[2], rows[3] + m.rows[3] }; }

    Mat4 operator-(const Mat4& m) const 
    { return { rows[0] - m.rows[0], rows[1] - m.rows[1], rows[2] - m.rows[2], rows[3] - m.rows[3] }; }

    Mat4 operator*(const T s) const 
    { return { rows[0] * s, rows[1] * s, rows[2] * s, rows[3] * s }; }

    Mat4 operator/(const T s) const 
    { return { rows[0] / s, rows[1] / s, rows[2] / s, rows[3] / s }; }

    Mat4 operator*(const Mat4& m) const {
        Mat4 res;

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                T sum = 0;
                for (int k = 0; k < 4; ++k)
                    sum += rows[i][k] * m[k][j];
                res[i][j] = sum;
            }
        }

        return res;
    }

    Vec4<T> operator*(const Vec4<T>& v) const 
    { return { v.dot(rows[0]), v.dot(rows[1]), v.dot(rows[2]), v.dot(rows[3]) }; }

    bool operator==(const Mat4& m) const 
    { return rows[0] == m.rows[0] && rows[1] == m.rows[1] && rows[2] == m.rows[2] && rows[3] == m.rows[3]; }

    bool operator!=(const Mat4& m) const 
    { return rows[0] != m.rows[0] || rows[1] != m.rows[1] || rows[2] != m.rows[2] || rows[3] != m.rows[3]; }

    Mat4 operator+=(const Mat4& m)
    { return { rows[0] += m.rows[0], rows[1] += m.rows[1], rows[2] += m.rows[2], rows[3] += m.rows[3] }; }

    Mat4 operator-=(const Mat4& m)
    { return { rows[0] -= m.rows[0], rows[1] -= m.rows[1], rows[2] -= m.rows[2], rows[3] -= m.rows[3] }; }

    Mat4 operator*=(const T s)
    { return { rows[0] *= s, rows[1] *= s, rows[2] *= s, rows[3] *= s }; }

    Mat4 operator/=(const T s)
    { return { rows[0] /= s, rows[1] /= s, rows[2] /= s, rows[3] /= s }; }
};

using Mat4f = Mat4<float>;
using Mat4d = Mat4<double>;
using Mat4i = Mat4<int32_t>;
using Mat4u = Mat4<uint32_t>;

static_assert(std::is_standard_layout_v<Mat4f>, "Mat4 must be standard-layout");
static_assert(sizeof(Mat4f) == sizeof(float) * 16, "Mat4 has unexpected padding");



/* ========================================================================= //
    MatNxM
// ========================================================================= */

template <size_t N, size_t M, typename T> struct Mat {
    Vec<M, T> rows[N] = {0};

    T* data() { return &rows[0][0]; }
    const T* data() const { return &rows[0][0]; }

    static Mat identity() requires ( N == M ) {
        Mat res;
        int oneIdx = 0;
        for (int row = 0; row < N; ++row)
            res[row][row] = 1;
        return res;
    }

    Mat<M, N, T> transpose() const {
        Mat<M, N, T> res;

        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < M; ++col)
                res[col][row] = rows[row][col];
        }

        return res;
    }

    T determinant() const requires ( N == M ) {
        Mat copy = *this;
        T det = T(1);
        int sign = 1;

        for (size_t col = 0; col < N; ++col) {
            size_t pivotRow = col;
            while (pivotRow < N && copy[pivotRow][col] == T(0)) ++pivotRow;

            if (pivotRow == N) return T(0);

            if (pivotRow != col) {
                std::swap(copy.rows[pivotRow], copy.rows[col]);
                sign = -sign;
            }

            for (size_t row = col + 1; row < N; ++row) {
                T factor = copy[row][col] / copy[col][col];
                for (size_t k = col; k < N; ++k)
                    copy[row][k] -= factor * copy[col][k];
            }
        }

        for (size_t i = 0; i < N; ++i) det *= copy[i][i];
        return sign * det;
    }

    Mat inverse() const requires ( N == M ) {
        Mat copy = *this;
        Mat result;
        for (size_t i = 0; i < N; ++i) result[i][i] = T(1);

        for (size_t col = 0; col < N; ++col) {
            size_t pivotRow = col;
            while (pivotRow < N && copy[pivotRow][col] == T(0)) ++pivotRow;

            if (pivotRow != col) {
                std::swap(copy.rows[pivotRow], copy.rows[col]);
                std::swap(result.rows[pivotRow], result.rows[col]);
            }

            T pivot = copy[col][col];
            for (size_t k = 0; k < N; ++k) {
                copy[col][k]   /= pivot;
                result[col][k] /= pivot;
            }

            for (size_t row = 0; row < N; ++row) {
                if (row == col) continue;
                T factor = copy[row][col];
                for (size_t k = 0; k < N; ++k) {
                    copy[row][k]   -= factor * copy[col][k];
                    result[row][k] -= factor * result[col][k];
                }
            }
        }

        return result;
    }

    Vec<M, T>& operator[](int i) { return rows[i]; }
    const Vec<M, T>& operator[](int i) const { return rows[i]; }

    Mat operator+(const Mat& m) const {
        Mat res;
        for (int row = 0; row < N; ++row)
            res[row] = rows[row] + m[row];
        return res;
    }

    Mat operator-(const Mat& m) const {
        Mat res;
        for (int row = 0; row < N; ++row)
            res[row] = rows[row] - m[row];
        return res;
    }

    Mat operator*(const T s) const {
        Mat res;
        for (int row = 0; row < N; ++row)
            res[row] = rows[row] * s;
        return res;
    }

    Mat operator/(const T s) const {
        Mat res;
        for (int row = 0; row < N; ++row)
            res[row] = rows[row] / s;
        return res;
    }

    template <size_t M2>
    Mat<N, M2, T> operator*(const Mat<M, M2, T>& m) const {
        Mat<N, M2, T> res;

        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < M2; ++col) {
                T sum = 0;
                for (int k = 0; k < M; ++k)
                    sum += rows[row][k] * m[k][col];
                res[row][col] = sum;
            }
        }

        return res;
    }

    Vec<N, T> operator*(const Vec<M, T>& v) const {
        Vec<N, T> res;
        for (int row = 0; row < N; ++row)
            res[row] = rows[row].dot(v);
        return res;
    }

    bool operator==(const Mat& m) {
        bool res = true;
        for (int row = 0; row < N; ++row)
            res &= (rows[row] == m[row]);
        return res;
    }

    bool operator!=(const Mat& m) {
        return !(*this == m);
    }


};



/* ========================================================================= //
    Common Matrix Operations
// ========================================================================= */

/* scalar * Matrix ========================================================= */
template <typename T> inline Mat2<T> operator*(const T s, const Mat2<T>& m)
{ return { s*m.rows[0], s*m.rows[1] }; }

template <typename T> inline Mat3<T> operator*(const T s, const Mat3<T>& m)
{ return { s*m.rows[0], s*m.rows[1], s*m.rows[2] }; }

template <typename T> inline Mat4<T> operator*(const T s, const Mat4<T>& m)
{ return { s*m.rows[0], s*m.rows[1], s*m.rows[2], s*m.rows[3] }; }

}