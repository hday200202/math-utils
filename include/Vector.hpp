#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <array>
#include <numeric>

namespace math {

/* ========================================================================= //
    Vec2
// ========================================================================= */

template <typename T> struct Vec2 {
    T x, y;

    T* data() { return &x; }
    const T* data() const { return &x; }

    T magnitudeSq() const { return x*x + y*y; }
    T magnitude() const { return sqrt(magnitudeSq()); }
    void normalize() { T mag = magnitude(); x /= mag; y /= mag; }
    T dot(const Vec2& v) const { return x*v.x + y*v.y; }

    Vec2 operator+(const Vec2& v) const { return { x+v.x, y+v.y }; }
    Vec2 operator-(const Vec2& v) const { return { x-v.x, y-v.y }; }
    Vec2 operator*(const T s) const { return { x*s, y*s }; }
    Vec2 operator/(const T s) const { return { x/s, y/s }; }
    Vec2 operator-() const { return { -x, -y }; }

    Vec2 operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
    Vec2 operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
    Vec2 operator*=(const T s) { x *= s; y *= s; return *this; }
    Vec2 operator/=(const T s) { x /= s; y /= s; return *this; }

    bool operator==(const Vec2& v) const { return x == v.x && y == v.y; }
    bool operator!=(const Vec2& v) const { return !(*this == v); }

    T& operator[](int i) { return data()[i]; }
    const T& operator[](int i) const { return data()[i]; }
};

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int32_t>;
using Vec2u = Vec2<uint32_t>;

static_assert(std::is_standard_layout_v<Vec2f>, "Vec2 must be standard-layout");
static_assert(sizeof(Vec2f) == sizeof(float) * 2, "Vec2 has unexpected padding");



/* ========================================================================= //
    Vec3
// ========================================================================= */

template <typename T> struct Vec3 { 
    T x, y, z;

    T* data() { return &x; }
    const T* data() const { return &x; }

    T magnitudeSq() const { return x*x + y*y + z*z; }
    T magnitude() const { return sqrt(magnitudeSq()); }
    void normalize() { T mag = magnitude(); x /= mag; y /= mag; z /= mag; }
    T dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vec3 cross(const Vec3& v) const { return { y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x }; }
    
    Vec3 operator+(const Vec3& v) const { return { x+v.x, y+v.y, z+v.z }; }
    Vec3 operator-(const Vec3& v) const { return { x-v.x, y-v.y, z-v.z }; }
    Vec3 operator*(const T s) const { return { x*s, y*s, z*s }; }
    Vec3 operator/(const T s) const { return { x/s, y/s, z/s }; }
    Vec3 operator-() const { return { -x, -y, -z }; }

    Vec3 operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3 operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3 operator*=(const T s) { x *= s; y *= s; z *= s; return *this; }
    Vec3 operator/=(const T s) { x /= s; y /= s; z /= s; return *this; }

    bool operator==(const Vec3& v) const { return x == v.x && y == v.y && z == v.z; }
    bool operator!=(const Vec3& v) const { return !(*this == v); }

    T& operator[](int i) { return data()[i]; }
    const T& operator[](int i) const { return data()[i]; }
};

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int32_t>;
using Vec3u = Vec3<uint32_t>;

static_assert(std::is_standard_layout_v<Vec3f>, "Vec3 must be standard-layout");
static_assert(sizeof(Vec3f) == sizeof(float) * 3, "Vec3 has unexpected padding");



/* ========================================================================= //
    Vec4
// ========================================================================= */

template <typename T> struct Vec4 { 
    T x, y, z, w;
    
    T* data() { return &x; }
    const T* data() const { return &x; }

    T magnitudeSq() const { return x*x + y*y + z*z + w*w; }
    T magnitude() const { return sqrt(magnitudeSq()); }
    void normalize() { T mag = magnitude(); x /= mag; y /= mag; z /= mag; w /= mag; }
    T dot(const Vec4& v) const { return x*v.x + y*v.y + z*v.z + w*v.w; }
    Vec4 cross(const Vec4& v) const { return { y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x, 0 }; }

    Vec4 operator+(const Vec4& v) const { return { x+v.x, y+v.y, z+v.z, w+v.w }; }
    Vec4 operator-(const Vec4& v) const { return { x-v.x, y-v.y, z-v.z, w-v.w }; }
    Vec4 operator*(const T s) const { return { x*s, y*s, z*s, w*s }; }
    Vec4 operator/(const T s) const { return { x/s, y/s, z/s, w/s }; }
    Vec4 operator-() const { return { -x, -y, -z, -w }; }

    Vec4 operator+=(const Vec4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
    Vec4 operator-=(const Vec4& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
    Vec4 operator*=(const T s) { x *= s; y *= s; z *= s; w *= s; return *this; }
    Vec4 operator/=(const T s) { x /= s; y /= s; z /= s; w /= s; return *this; }

    bool operator==(const Vec4& v) const { return x == v.x && y == v.y && z == v.z && w == v.w; }
    bool operator!=(const Vec4& v) const { return !(*this == v); }

    T& operator[](int i) { return data()[i]; }
    const T& operator[](int i) const { return data()[i]; }
};

using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int32_t>;
using Vec4u = Vec4<uint32_t>;

static_assert(std::is_standard_layout_v<Vec4f>, "Vec4 must be standard-layout");
static_assert(sizeof(Vec4f) == sizeof(float) * 4, "Vec4 has unexpected padding");



/* ========================================================================= //
    VecN
// ========================================================================= */

template <size_t N, typename T> struct Vec {
    std::array<T, N> v;

    size_t size() const { return v.size(); }

    T* data() { return &v[0]; }
    const T* data() const { return &v[0]; }

    T magnitudeSq() const {
        T sum = 0;
        for (int i = 0; i < v.size(); ++i)
            sum += v[i] * v[i];
        return sum;
    }

    T magnitude() const { return sqrt(magnitudeSq()); }

    void normalize() {
        T mag = magnitude();
        for (int i = 0; i < v.size(); ++i) v[i] /= mag;
    }

    T dot(const Vec& vec) const {
        T product = 0;
        for (int i = 0; i < v.size(); ++i) product += (v[i]*vec[i]);
        return product;
    }

    Vec operator+(const Vec& vec) const {
        Vec res;
        for (int i = 0; i < v.size(); ++i) res[i] = v[i] + vec[i];
        return res;
    }

    Vec operator-(const Vec& vec) const {
        Vec res;
        for (int i = 0; i < v.size(); ++i) res[i] = v[i] - vec[i];
        return res;
    }

    Vec operator*(const T s) const {
        Vec res;
        for (int i = 0; i < v.size(); ++i) res[i] = v[i]*s;
        return res;
    }

    Vec operator/(const T s) const {
        Vec res;
        for (int i = 0; i < v.size(); ++i) res[i] = v[i]/s;
        return res;
    }

    Vec operator-() const {
        Vec res;
        for (int i = 0; i < v.size(); ++i) res[i] = -v[i];
        return res;
    }

    Vec operator+=(const Vec& vec) {
        for (int i = 0; i < v.size(); ++i) v[i] += vec[i];
        return *this;
    }

    Vec operator-=(const Vec& vec) {
        for (int i = 0; i < v.size(); ++i) v[i] -= vec[i];
        return *this;
    }

    Vec operator*=(const T s) {
        for (int i = 0; i < v.size(); ++i) v[i] *= s;
        return *this;
    }

    Vec operator/=(const T s) {
        for (int i = 0; i < v.size(); ++i) v[i] /= s;
        return *this;
    }

    bool operator==(const Vec& vec) const {
        bool res = true;
        for (int i = 0; i < v.size(); ++i) res &= (v[i] == vec[i]);
        return res;
    }

    bool operator!=(const Vec& vec) const {
        return !(*this == vec);
    }

    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};

static_assert(std::is_standard_layout_v<Vec<5, float>>, "Vec<5, float> must be standard-layout");
static_assert(sizeof(Vec<5, float>) == sizeof(float) * 5, "Vec4 has unexpected padding");



/* ========================================================================= //
    Common Vector Operations
// ========================================================================= */

/* lerp ==================================================================== */
template <typename T> inline Vec2<T> lerp(const Vec2<T>& a, const Vec2<T>& b, T t) 
{ Vec2<T> res = a + ((b - a) * t); return res; }

template <typename T> inline Vec3<T> lerp(const Vec3<T>& a, const Vec3<T>& b, T t) 
{ Vec3<T> res = a + ((b - a) * t); return res; }

template <typename T> inline Vec4<T> lerp(const Vec4<T>& a, const Vec4<T>& b, T t) 
{ Vec4<T> res = a + ((b - a) * t); return res; }

template <size_t N, typename T> inline Vec<N, T> lerp(const Vec<N, T>& a, const Vec<N, T>& b, T t)
{ Vec<N, T> res = a + ((b - a) * t); return res; }


/* normalize =============================================================== */
template <typename T> inline Vec2<T> normalize(const Vec2<T>& v) 
{ T mag = v.magnitude(); return { v.x / mag, v.y / mag }; }

template <typename T> inline Vec3<T> normalize(const Vec3<T>& v) 
{ T mag = v.magnitude(); return { v.x / mag, v.y / mag, v.z / mag }; }

template <typename T> inline Vec4<T> normalize(const Vec4<T>& v) 
{ T mag = v.magnitude(); return { v.x / mag, v.y / mag, v.z / mag, v.w / mag }; }

template <size_t N, typename T> inline Vec<N, T> normalize(const Vec<N, T>& v) { 
    Vec<N, T> res; 
    T mag = v.magnitude();
    for (int i = 0; i < v.size(); ++i) res[i] = v[i] / mag;
    return res;
}


/* distance ================================================================ */
template <typename T> inline T distance(const Vec2<T>& a, const Vec2<T>& b) 
{ return (b - a).magnitude(); }

template <typename T> inline T distance(const Vec3<T>& a, const Vec3<T>& b)
{ return (b - a).magnitude(); }

template <typename T> inline T distance(const Vec4<T>& a, const Vec4<T>& b)
{ return (b - a).magnitude(); }

template <size_t N, typename T> inline T distance(const Vec<N, T>& a, const Vec<N, T>& b)
{ return (b - a).magnitude(); }


/* distance squared ======================================================== */
template <typename T> inline T distanceSq(const Vec2<T>& a, const Vec2<T>& b)
{ return (b - a).magnitudeSq(); }

template <typename T> inline T distanceSq(const Vec3<T>& a, const Vec3<T>& b)
{ return (b - a).magnitudeSq(); }

template <typename T> inline T distanceSq(const Vec4<T>& a, const Vec4<T>& b)
{ return (b - a).magnitudeSq(); }

template <size_t N, typename T> inline T distanceSq(const Vec<N, T>& a, const Vec<N, T>& b)
{ return (b - a).magnitudeSq(); }


/* scalar * vec ============================================================ */
template <typename T> inline Vec2<T> operator*(const T s, const Vec2<T>& v)
{ return v * s; }

template <typename T> inline Vec3<T> operator*(const T s, const Vec3<T>& v)
{ return v * s; }

template <typename T> inline Vec4<T> operator*(const T s, const Vec4<T>& v)
{ return v * s; }

template <size_t N, typename T> inline Vec<N, T> operator*(const T s, const Vec<N, T>& v)
{ return v * s; }


/* reflect ================================================================= */
template <typename T> inline Vec2<T> reflect(const Vec2<T>& v, const Vec2<T>& normal)
{ return v - (2 * v.dot(normal)) * normal; }

template <typename T> inline Vec3<T> reflect(const Vec3<T>& v, const Vec3<T>& normal)
{ return v - (2 * v.dot(normal)) * normal; }

template <typename T> inline Vec4<T> reflect(const Vec4<T>& v, const Vec4<T>& normal)
{ return v - (2 * v.dot(normal)) * normal; }

template <size_t N, typename T> inline Vec<N, T> reflect(const Vec<N, T>& v, const Vec<N, T> normal)
{ return v - (2 * v.dot(normal)) * normal;}


/* min ===================================================================== */
template <typename T> inline Vec2<T> min(const Vec2<T>& a, const Vec2<T>& b) 
{ return { std::min<T>(a.x, b.x), std::min<T>(a.y, b.y) }; }

template <typename T> inline Vec3<T> min(const Vec3<T>& a, const Vec3<T>& b) 
{ return { std::min<T>(a.x, b.x), std::min<T>(a.y, b.y), std::min<T>(a.z, b.z) }; }

template <typename T> inline Vec4<T> min(const Vec4<T>& a, const Vec4<T>& b) 
{ return { std::min<T>(a.x, b.x), std::min<T>(a.y, b.y), std::min<T>(a.z, b.z), std::min<T>(a.w, b.w) }; }

template <size_t N, typename T> inline Vec<N, T> min(const Vec<N, T>& a, const Vec<N, T>& b)
{ Vec<N, T> res; for (int i = 0; i < N; ++i) res[i] = std::min(a[i], b[i]); return res; }


/* max ===================================================================== */
template <typename T> inline Vec2<T> max(const Vec2<T>& a, const Vec2<T>& b) 
{ return { std::max<T>(a.x, b.x), std::max<T>(a.y, b.y) }; }

template <typename T> inline Vec3<T> max(const Vec3<T>& a, const Vec3<T>& b) 
{ return { std::max<T>(a.x, b.x), std::max<T>(a.y, b.y), std::max<T>(a.z, b.z) }; }

template <typename T> inline Vec4<T> max(const Vec4<T>& a, const Vec4<T>& b) 
{ return { std::max<T>(a.x, b.x), std::max<T>(a.y, b.y), std::max<T>(a.z, b.z), std::max<T>(a.w, b.w) }; }

template <size_t N, typename T> inline Vec<N, T> max(const Vec<N, T>& a, const Vec<N, T>& b)
{ Vec<N, T> res; for (int i = 0; i < N; ++i) res[i] = std::max(a[i], b[i]); return res; }


/* clamp =================================================================== */
template <typename T> inline Vec2<T> clamp(const Vec2<T>& v, const Vec2<T>& lo, const Vec2<T>& hi) 
{ return { std::clamp<T>(v.x, lo.x, hi.x), std::clamp<T>(v.y, lo.y, hi.y) }; }

template <typename T> inline Vec3<T> clamp(const Vec3<T>& v, const Vec3<T>& lo, const Vec3<T>& hi) 
{ return { std::clamp<T>(v.x, lo.x, hi.x), std::clamp<T>(v.y, lo.y, hi.y), std::clamp<T>(v.z, lo.z, hi.z) }; }

template <typename T> inline Vec4<T> clamp(const Vec4<T>& v, const Vec4<T>& lo, const Vec4<T>& hi) 
{ return { std::clamp<T>(v.x, lo.x, hi.x), std::clamp<T>(v.y, lo.y, hi.y), std::clamp<T>(v.z, lo.z, hi.z), std::clamp<T>(v.w, lo.w, hi.w) }; }

template <size_t N, typename T> inline Vec<N, T> clamp(const Vec<N, T>& v, const Vec<N, T>& lo, const Vec<N, T>& hi)
{ Vec<N, T> res; for (int i = 0; i < N; ++i) res[i] = std::clamp<T>(v[i], lo[i], hi[i]); return res; }


/* angle =================================================================== */
template <typename T> inline T angle(const Vec2<T>& a, const Vec2<T>& b)
{ return std::acos(a.dot(b) / (a.magnitude() * b.magnitude())); }

template <typename T> inline T angle(const Vec3<T>& a, const Vec3<T>& b)
{ return std::acos(a.dot(b) / (a.magnitude() * b.magnitude())); }

template <typename T> inline T angle(const Vec4<T>& a, const Vec4<T>& b)
{ return std::acos(a.dot(b) / (a.magnitude() * b.magnitude())); }

template <size_t N, typename T> inline T angle(const Vec<N, T>& a, const Vec<N, T>& b)
{ return std::acos(a.dot(b) / (a.magnitude() * b.magnitude())); }


/* project ================================================================= */
template <typename T> inline Vec2<T> project(const Vec2<T>& a, const Vec2<T>& b)
{ return (a.dot(b) / b.magnitudeSq()) * b; }

template <typename T> inline Vec3<T> project(const Vec3<T>& a, const Vec3<T>& b)
{ return (a.dot(b) / b.magnitudeSq()) * b; }

template <typename T> inline Vec4<T> project(const Vec4<T>& a, const Vec4<T>& b)
{ return (a.dot(b) / b.magnitudeSq()) * b; }

template <size_t N, typename T> inline Vec<N, T> project(const Vec<N, T>& a, const Vec<N, T>& b)
{ return (a.dot(b) / b.magnitudeSq()) * b; }


/* reject ================================================================== */
template <typename T> inline Vec2<T> reject(const Vec2<T>& a, const Vec2<T>& b)
{ return a - project(a, b); }

template <typename T> inline Vec3<T> reject(const Vec3<T>& a, const Vec3<T>& b)
{ return a - project(a, b); }

template <typename T> inline Vec4<T> reject(const Vec4<T>& a, const Vec4<T>& b)
{ return a - project(a, b); }

template <size_t N, typename T> inline Vec<N, T> reject(const Vec<N, T>& a, const Vec<N, T>& b)
{ return a - project(a, b); }


/* face forward ============================================================ */
template <typename T> inline Vec2<T> faceforward(const Vec2<T>& n, const Vec2<T>& i, const Vec2<T>& nref)
{ return (nref.dot(i) < 0) ? n : -n; }

template <typename T> inline Vec3<T> faceforward(const Vec3<T>& n, const Vec3<T>& i, const Vec3<T>& nref)
{ return (nref.dot(i) < 0) ? n : -n; }

template <typename T> inline Vec4<T> faceforward(const Vec4<T>& n, const Vec4<T>& i, const Vec4<T>& nref)
{ return (nref.dot(i) < 0) ? n : -n; }

template <size_t N, typename T> inline Vec<N, T> faceforward(const Vec<N, T>& n, const Vec<N, T>& i, const Vec<N, T>& nref)
{ return (nref.dot(i) < 0) ? n : -n; }



} // !namespace math