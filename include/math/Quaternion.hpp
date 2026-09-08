#pragma once

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
    
    static Quat identity() { return {{0, 0, 0}, T(1)}; }
    static Quat fromAxisAngle(const Vec3<T>& axis, const T theta) {
        T half = theta / T(2);
        return {math::normalize(axis) * std::sin(half), std::cos(half)};
    }
};

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

}