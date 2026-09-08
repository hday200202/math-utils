#pragma once

#include <limits>
#include "Matrix.hpp"

namespace math {

template <typename T> struct Quat {
    Vec3<T> v = {T(0), T(0), T(0)}; 
    T w = T(1);

    T* data() { return &v.x; }
    const T* data() const { return &v.x; }

    Quat operator*(const Quat& q) const {
        return { w * q.v + q.w * v + v.cross(q.v), w * q.w - v.dot(q.v) };
    }

    Quat operator*(const T s) const { return {v * s, w * s}; }
    Quat operator/(const T s) const { return {v / s, w / s}; }
    Quat operator+(const Quat& q) const { return {v + q.v, w + q.w}; }
    Quat operator-(const Quat& q) const { return {v - q.v, w - q.w}; }
    Quat operator-() const { return {-v, -w}; }
    T dot(const Quat& q) const { return v.dot(q.v) + w * q.w; }

    bool operator==(const Quat& q) const { return (v == q.v && w == q.w); }
    bool operator!=(const Quat& q) const { return !(*this == q); }

    void normalize() { T n = norm(); v /= n; w /= n; }
    T normSq() const { return v.magnitudeSq() + (w*w); }
    T norm() const { return std::sqrt(normSq()); }
    Quat conjugate() const { return {-v, w}; }
    Quat inverse() const { return conjugate() / normSq(); }
    Quat inverseUnit() const { return conjugate(); }

    Vec3<T> rotate(const Vec3<T>& p) const {
        Vec3<T> t = v.cross(p) + w*p;
        return p + v.cross(t) * T(2);
    }

    Mat3<T> toMat3() const {
        return {{
            {1 - 2*(v.y*v.y + v.z*v.z), 2*(v.x*v.y - w*v.z),        2*(v.x*v.z + w*v.y)},
            {2*(v.x*v.y + w*v.z),       1 - 2*(v.x*v.x + v.z*v.z),  2*(v.y*v.z - w*v.x)},
            {2*(v.x*v.z - w*v.y),       2*(v.y*v.z + w*v.x),        1 - 2*(v.x*v.x + v.y*v.y)},
        }};
    }

    Mat4<T> toMat4() const {
        return Mat4<T>::fromMat3(toMat3());
    }
    
    static Quat identity() { return {{0, 0, 0}, T(1)}; }
    static Quat fromAxisAngle(const Vec3<T>& axis, const T theta) {
        T half = theta / T(2);
        return {math::normalize(axis) * std::sin(half), std::cos(half)};
    }
    void toAxisAngle(Vec3<T>& axis, T& angle) const {
        T len = v.magnitude();
        if (len < std::numeric_limits<T>::epsilon() * T(10)) {
            axis = {T(1), T(0), T(0)};
            angle = T(0);
            return;
        }
        angle = T(2) * std::atan2(len, w);
        axis = v / len;
    }
    static Quat fromMat3(const Mat3<T>& m) {
        T trace = m[0][0] + m[1][1] + m[2][2];
        T s;

        if (trace > 0) {
            s = std::sqrt(trace + 1) * 2;
            return {
                {(m[2][1] - m[1][2]) / s, 
                (m[0][2] - m[2][0]) / s, 
                (m[1][0] - m[0][1]) / s}, 
                s / 4
            };
        }

        else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
            s = std::sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
            return {
                {s / 4, 
                (m[0][1] + m[1][0]) / s, 
                (m[0][2] + m[2][0]) / s}, 
                (m[2][1] - m[1][2]) / s
            };
        }

        else if (m[1][1] > m[2][2]) {
            s = std::sqrt(1 - m[0][0] + m[1][1] - m[2][2]) * 2;
            return {
                {(m[0][1] + m[1][0]) / s,
                s / 4,
                (m[1][2] + m[2][1]) / s},
                (m[0][2] - m[2][0]) / s
            };
        }

        else {
            s = std::sqrt(1 - m[0][0] - m[1][1] + m[2][2]) * 2;
            return {
                {(m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                s / 4},
                (m[1][0] - m[0][1]) / s
            };
        }
    }
};

using Quatf = Quat<float>;
using Quatd = Quat<double>;
using Quati = Quat<int32_t>;
using Quatu = Quat<uint32_t>;

static_assert(sizeof(Quat<float>) == sizeof(float) * 4);

template <typename T> inline Quat<T> slerp(Quat<T> a, const Quat<T>& b, T t) {
    T d = a.dot(b);
    if (d < T(0)) { a = -a; d = -d; }
    if (d > T(0.9995)) {
        Quat<T> r = a + (b - a) * t;
        r.normalize();
        return r;
    }

    T theta = std::acos(d);
    T s = std::sin(theta);
    return a * (std::sin((T(1) - t) * theta) / s) + b * (std::sin(t * theta) / s);
}

template <typename T> inline Quat<T> normalize(const Quat<T>& q) {
    T n = q.norm(); 
    return {q.v / n, q.w / n};
}

}