#include <iostream>
#include <cmath>
#include <string>

#include "../include/math/Quaternion.hpp"

/* ========================================================================= //
    Quaternion test suite.

    Every check prints a green [PASS] or red [FAIL]. Tests are grouped by the
    operation they exercise; the block at the bottom lists what is not built
    yet, as commented-out tests to uncomment as each piece lands.
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

static int failures = 0;

using Quatf = math::Quat<float>;
using math::Vec3f;
using math::Vec4f;
using math::Mat3f;
using math::Mat4f;

static constexpr float PI = 3.14159265358979f;

inline bool isEqual(float a, float b)             { return std::fabs(a - b) < 1e-4f; }
inline bool isEqual(const Vec3f& a, const Vec3f& b)
{ return isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z); }
inline bool isEqual(const Quatf& a, const Quatf& b)
{ return isEqual(a.v, b.v) && isEqual(a.w, b.w); }
inline bool isEqual(const Mat3f& a, const Mat3f& b) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!isEqual(a[i][j], b[i][j])) return false;
    return true;
}
inline bool isEqual(const Mat4f& a, const Mat4f& b) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (!isEqual(a[i][j], b[i][j])) return false;
    return true;
}

/*
    sameRotation(a, b):
    - q and -q are the same rotation (double cover), so a comparison that
      cares about the rotation rather than the components has to accept both
*/
inline bool sameRotation(const Quatf& a, const Quatf& b)
{ return isEqual(a, b) || isEqual(a, -b); }

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] "
              << COLOR_RESET << label << std::endl;
    if (!pass) ++failures;
}

static void section(const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
}


/* ========================================================================= //
    Construction and basic properties
// ========================================================================= */

static void testConstruction() {
    section("construction");

    Quatf id = Quatf::identity();
    check("identity is {0,0,0,1}", isEqual(id, Quatf{{0, 0, 0}, 1.f}));
    check("identity has unit norm", isEqual(id.norm(), 1.f));

    // an unnormalized axis must still give a unit quaternion
    Quatf q = Quatf::fromAxisAngle({0, 0, 5}, PI / 2);
    check("fromAxisAngle normalizes the axis", isEqual(q.norm(), 1.f));
    check("fromAxisAngle(Z, 90) w == cos(45)", isEqual(q.w, std::cos(PI / 4)));
    check("fromAxisAngle(Z, 90) v == (0,0,sin(45))",
          isEqual(q.v, Vec3f{0.f, 0.f, std::sin(PI / 4)}));

    check("fromAxisAngle(axis, 0) is identity",
          sameRotation(Quatf::fromAxisAngle({1, 2, 3}, 0.f), Quatf::identity()));

    // 360 degrees returns -identity: the same rotation, the other cover
    Quatf full = Quatf::fromAxisAngle({0, 1, 0}, 2 * PI);
    check("fromAxisAngle(axis, 360) is identity as a rotation", sameRotation(full, Quatf::identity()));

    check("data() is laid out x,y,z,w", [] {
        Quatf t{{1, 2, 3}, 4};
        const float* d = t.data();
        return d[0] == 1 && d[1] == 2 && d[2] == 3 && d[3] == 4;
    }());
}


/* ========================================================================= //
    Norm, conjugate, inverse
// ========================================================================= */

static void testNormAndInverse() {
    section("norm / conjugate / inverse");

    Quatf q{{1, 2, 3}, 4};

    check("normSq == x^2+y^2+z^2+w^2", isEqual(q.normSq(), 30.f));
    check("norm == sqrt(normSq)", isEqual(q.norm(), std::sqrt(30.f)));

    Quatf n = q;
    n.normalize();
    check("normalize gives unit norm", isEqual(n.norm(), 1.f));
    check("normalize preserves direction", sameRotation(n, q * (1.f / q.norm())));

    check("conjugate negates only the vector part",
          isEqual(q.conjugate(), Quatf{{-1, -2, -3}, 4}));
    check("conjugate is an involution", isEqual(q.conjugate().conjugate(), q));

    // holds for any quaternion, normalized or not
    check("q * q.inverse() == identity", isEqual(q * q.inverse(), Quatf::identity()));
    check("q.inverse() * q == identity", isEqual(q.inverse() * q, Quatf::identity()));

    // for a unit quaternion the conjugate IS the inverse
    Quatf u = Quatf::fromAxisAngle({1, 1, 0}, 0.9f);
    check("inverseUnit == inverse for unit quaternions", isEqual(u.inverseUnit(), u.inverse()));
    check("u * u.inverseUnit() == identity", isEqual(u * u.inverseUnit(), Quatf::identity()));
}


/* ========================================================================= //
    Arithmetic
// ========================================================================= */

static void testArithmetic() {
    section("arithmetic");

    Quatf a{{1, 2, 3}, 4};
    Quatf b{{5, 6, 7}, 8};

    check("operator+", isEqual(a + b, Quatf{{6, 8, 10}, 12}));
    check("operator-", isEqual(b - a, Quatf{{4, 4, 4}, 4}));
    check("unary minus", isEqual(-a, Quatf{{-1, -2, -3}, -4}));
    check("scalar multiply", isEqual(a * 2.f, Quatf{{2, 4, 6}, 8}));
    check("scalar divide", isEqual(a / 2.f, Quatf{{0.5f, 1.f, 1.5f}, 2.f}));
    check("dot", isEqual(a.dot(b), 1.f*5 + 2.f*6 + 3.f*7 + 4.f*8));

    check("q * identity == q", isEqual(a * Quatf::identity(), a));
    check("identity * q == q", isEqual(Quatf::identity() * a, a));

    // the property that makes rotation order matter
    check("multiplication is NOT commutative", !isEqual(a * b, b * a));

    Quatf c{{-2, 0.5f, 1}, 3};
    check("multiplication is associative", isEqual((a * b) * c, a * (b * c)));

    check("norm is multiplicative: |ab| == |a||b|",
          isEqual((a * b).norm(), a.norm() * b.norm()));
}


/* ========================================================================= //
    Rotating vectors
// ========================================================================= */

static void testRotation() {
    section("vector rotation");

    Quatf rz = Quatf::fromAxisAngle({0, 0, 1}, PI / 2);
    Quatf rx = Quatf::fromAxisAngle({1, 0, 0}, PI / 2);
    Quatf ry = Quatf::fromAxisAngle({0, 1, 0}, PI / 2);

    check("Rz(90) maps X to Y", isEqual(rz.rotate({1, 0, 0}), Vec3f{0, 1, 0}));
    check("Rx(90) maps Y to Z", isEqual(rx.rotate({0, 1, 0}), Vec3f{0, 0, 1}));
    check("Ry(90) maps Z to X", isEqual(ry.rotate({0, 0, 1}), Vec3f{1, 0, 0}));

    check("identity leaves a vector alone",
          isEqual(Quatf::identity().rotate({3, -2, 7}), Vec3f{3, -2, 7}));

    check("rotating about an axis leaves that axis fixed",
          isEqual(rz.rotate({0, 0, 4}), Vec3f{0, 0, 4}));

    Quatf q = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    Vec3f p{0.3f, -0.9f, 2.f};
    check("rotate then inverse-rotate returns the original",
          isEqual(q.inverseUnit().rotate(q.rotate(p)), p));

    check("rotation preserves length",
          isEqual(q.rotate(p).magnitude(), p.magnitude()));

    // q * v * q^-1 the long way, to confirm the optimized form agrees
    Quatf pq{p, 0.f};
    check("rotate() agrees with q * (0,p) * q^-1",
          isEqual(q.rotate(p), (q * pq * q.inverseUnit()).v));

    // composing rotations: applying b then a equals rotating by (a*b)
    Quatf a2 = Quatf::fromAxisAngle({0, 1, 0}, 0.4f);
    Quatf b2 = Quatf::fromAxisAngle({1, 0, 1}, -1.2f);
    check("(a*b).rotate(p) == a.rotate(b.rotate(p))",
          isEqual((a2 * b2).rotate(p), a2.rotate(b2.rotate(p))));
}


/* ========================================================================= //
    Matrix conversion
// ========================================================================= */

static void testToMat3() {
    section("toMat3");

    check("identity gives the identity matrix",
          isEqual(Quatf::identity().toMat3(), Mat3f::identity()));

    Quatf rz = Quatf::fromAxisAngle({0, 0, 1}, PI / 2);
    check("Rz(90) matrix maps X to Y", isEqual(rz.toMat3() * Vec3f{1, 0, 0}, Vec3f{0, 1, 0}));

    Quatf rx = Quatf::fromAxisAngle({1, 0, 0}, PI / 2);
    check("Rx(90) matrix maps Y to Z", isEqual(rx.toMat3() * Vec3f{0, 1, 0}, Vec3f{0, 0, 1}));

    Quatf ry = Quatf::fromAxisAngle({0, 1, 0}, PI / 2);
    check("Ry(90) matrix maps Z to X", isEqual(ry.toMat3() * Vec3f{0, 0, 1}, Vec3f{1, 0, 0}));

    Quatf q = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);

    // a rotation matrix is orthonormal, so its determinant is exactly +1.
    // -1 would mean a reflection crept in, anything else means a scale
    check("determinant == 1", isEqual(q.toMat3().determinant(), 1.f));

    check("transpose == inverse (matrix is orthonormal)",
          isEqual(q.toMat3().transpose(), q.toMat3().inverse()));

    check("toMat3 agrees with rotate()",
          isEqual(q.toMat3() * Vec3f{0.3f, -0.9f, 2.f}, q.rotate({0.3f, -0.9f, 2.f})));

    // the strongest single check: conversion is a homomorphism, which only
    // holds if multiplication order and matrix convention agree
    Quatf a = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    Quatf b = Quatf::fromAxisAngle({-2, 1, 0.5f}, -1.1f);
    check("toMat3(a*b) == toMat3(a) * toMat3(b)",
          isEqual((a * b).toMat3(), a.toMat3() * b.toMat3()));
}


/* ========================================================================= //
    Slerp
// ========================================================================= */

static void testSlerp() {
    section("slerp");

    Quatf a = Quatf::fromAxisAngle({0, 0, 1}, 0.f);
    Quatf b = Quatf::fromAxisAngle({0, 0, 1}, PI / 2);

    check("slerp(a,b,0) == a", sameRotation(math::slerp(a, b, 0.f), a));
    check("slerp(a,b,1) == b", sameRotation(math::slerp(a, b, 1.f), b));

    Quatf mid = math::slerp(a, b, 0.5f);
    check("slerp midpoint stays unit length", isEqual(mid.norm(), 1.f));
    check("slerp midpoint is the half rotation",
          sameRotation(mid, Quatf::fromAxisAngle({0, 0, 1}, PI / 4)));

    // rotating X by the midpoint should land halfway between X and Y
    check("slerp midpoint rotates X to 45 degrees",
          isEqual(mid.rotate({1, 0, 0}), Vec3f{std::cos(PI / 4), std::sin(PI / 4), 0}));

    // near-identical inputs take the nlerp branch; without it sin(theta) ~ 0
    Quatf c = Quatf::fromAxisAngle({0, 1, 0}, 0.30000f);
    Quatf d = Quatf::fromAxisAngle({0, 1, 0}, 0.30001f);
    Quatf tiny = math::slerp(c, d, 0.5f);
    check("slerp of near-identical inputs is finite (no NaN)",
          std::isfinite(tiny.w) && std::isfinite(tiny.v.x) &&
          std::isfinite(tiny.v.y) && std::isfinite(tiny.v.z));
    check("slerp of near-identical inputs stays unit", isEqual(tiny.norm(), 1.f));

    // double cover: b and -b are the same rotation, so interpolating toward
    // either must take the same short path rather than the 360-degree way
    Quatf viaPos = math::slerp(a, b, 0.5f);
    Quatf viaNeg = math::slerp(a, -b, 0.5f);
    check("slerp takes the shortest path regardless of sign",
          sameRotation(viaPos, viaNeg));

    check("slerp result is always unit length across t", [&] {
        for (int i = 0; i <= 10; ++i)
            if (!isEqual(math::slerp(a, b, i / 10.f).norm(), 1.f)) return false;
        return true;
    }());
}


/* ========================================================================= //
    fromMat3 (matrix -> quaternion)
// ========================================================================= */

static void testFromMat3() {
    section("fromMat3");

    check("fromMat3(identity) is identity",
          sameRotation(Quatf::fromMat3(Mat3f::identity()), Quatf::identity()));

    Quatf q = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    check("fromMat3 round-trips", sameRotation(Quatf::fromMat3(q.toMat3()), q));

    // Shepperd's method branches on which of w/x/y/z is largest. The three
    // axis-aligned 180 degree rotations are the trace <= 0 cases, one per
    // vector-component branch -- these are what the naive sqrt(1 + trace)
    // form divides by zero on.
    Quatf flipX = Quatf::fromAxisAngle({1, 0, 0}, PI);
    Quatf flipY = Quatf::fromAxisAngle({0, 1, 0}, PI);
    Quatf flipZ = Quatf::fromAxisAngle({0, 0, 1}, PI);
    check("fromMat3 handles 180 degrees about X (x branch)",
          sameRotation(Quatf::fromMat3(flipX.toMat3()), flipX));
    check("fromMat3 handles 180 degrees about Y (y branch)",
          sameRotation(Quatf::fromMat3(flipY.toMat3()), flipY));
    check("fromMat3 handles 180 degrees about Z (z branch)",
          sameRotation(Quatf::fromMat3(flipZ.toMat3()), flipZ));

    // An axis-aligned flip leaves three of the four components at zero, so a
    // branch can be wrong in a way those cases cannot see. Near-180 degrees
    // about a skewed axis keeps every component nonzero and still selects the
    // same branch, which is what actually pins each formula down.
    const float NEAR_PI = 179.f * PI / 180.f;
    Quatf skewX = Quatf::fromAxisAngle({3, 2, 1}, NEAR_PI);   // |x| largest
    Quatf skewY = Quatf::fromAxisAngle({2, 3, 1}, NEAR_PI);   // |y| largest
    Quatf skewZ = Quatf::fromAxisAngle({1, 2, 3}, NEAR_PI);   // |z| largest
    check("fromMat3 handles a skewed near-180 rotation (x branch)",
          sameRotation(Quatf::fromMat3(skewX.toMat3()), skewX));
    check("fromMat3 handles a skewed near-180 rotation (y branch)",
          sameRotation(Quatf::fromMat3(skewY.toMat3()), skewY));
    check("fromMat3 handles a skewed near-180 rotation (z branch)",
          sameRotation(Quatf::fromMat3(skewZ.toMat3()), skewZ));

    check("fromMat3 returns a unit quaternion",
          isEqual(Quatf::fromMat3(skewX.toMat3()).norm(), 1.f));

    // the other direction: matrix -> quat -> matrix must be a no-op
    Mat3f m = q.toMat3();
    check("toMat3(fromMat3(m)) == m", isEqual(Quatf::fromMat3(m).toMat3(), m));

    // sweep a range of axes and angles so no branch goes unvisited
    check("fromMat3 round-trips across a sweep of rotations", [] {
        const Vec3f axes[] = {
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1},
            {0, 1, 1}, {1, 1, 1}, {3, 2, 1}, {1, 2, 3}, {-2, 1, -0.5f}
        };
        for (const Vec3f& axis : axes) {
            for (int deg = -180; deg <= 180; deg += 5) {
                Quatf r = Quatf::fromAxisAngle(axis, deg * PI / 180.f);
                if (!sameRotation(Quatf::fromMat3(r.toMat3()), r)) return false;
            }
        }
        return true;
    }());

    // the interpolate-between-two-rotation-matrices path this exists for
    check("slerp between two matrices stays a rotation", [] {
        Mat3f a = Quatf::fromAxisAngle({0, 1, 0}, 0.f).toMat3();
        Mat3f b = Quatf::fromAxisAngle({0.3f, 1, 0.2f}, 150.f * PI / 180.f).toMat3();
        Quatf qa = Quatf::fromMat3(a);
        Quatf qb = Quatf::fromMat3(b);
        for (int i = 0; i <= 10; ++i) {
            Mat3f m = math::slerp(qa, qb, i / 10.f).toMat3();
            if (!isEqual(m.determinant(), 1.f)) return false;
            if (!isEqual(m.transpose(), m.inverse())) return false;
        }
        return true;
    }());
    check("slerp between two matrices hits its endpoints", [] {
        Mat3f a = Quatf::fromAxisAngle({1, 0, 0}, 0.4f).toMat3();
        Mat3f b = Quatf::fromAxisAngle({0, 0, 1}, 1.9f).toMat3();
        Quatf qa = Quatf::fromMat3(a);
        Quatf qb = Quatf::fromMat3(b);
        return isEqual(math::slerp(qa, qb, 0.f).toMat3(), a)
            && isEqual(math::slerp(qa, qb, 1.f).toMat3(), b);
    }());
}


/* ========================================================================= //
    toMat4
// ========================================================================= */

static void testToMat4() {
    section("toMat4");

    check("toMat4(identity) is the identity matrix",
          isEqual(Quatf::identity().toMat4(), Mat4f::identity()));

    Quatf q = Quatf::fromAxisAngle({1, 2, 3}, 0.9f);
    Mat4f m = q.toMat4();

    check("toMat4 upper-left equals toMat3", isEqual(m.upperLeft(), q.toMat3()));

    check("toMat4 last row and column are 0,0,0,1",
          isEqual(m[0][3], 0.f) && isEqual(m[1][3], 0.f) && isEqual(m[2][3], 0.f) &&
          isEqual(m[3][0], 0.f) && isEqual(m[3][1], 0.f) && isEqual(m[3][2], 0.f) &&
          isEqual(m[3][3], 1.f));

    // a point (w == 1) must come out rotated, with w untouched
    Vec3f p = {0.4f, -1.3f, 2.f};
    Vec3f expected = q.rotate(p);
    Vec4f out = m * Vec4f{p.x, p.y, p.z, 1.f};
    check("toMat4 rotates a point and leaves w alone",
          isEqual(out.xyz, expected) && isEqual(out.w, 1.f));

    // no translation, so a direction (w == 0) rotates the same way
    Vec4f dir = m * Vec4f{p.x, p.y, p.z, 0.f};
    check("toMat4 rotates a direction the same as a point",
          isEqual(dir.xyz, expected) && isEqual(dir.w, 0.f));

    check("toMat4 determinant is 1", isEqual(m.determinant(), 1.f));

    check("toMat4(a*b) == toMat4(a) * toMat4(b)", [] {
        Quatf a = Quatf::fromAxisAngle({1, 0, 0}, 0.6f);
        Quatf b = Quatf::fromAxisAngle({-2, 1, 0.5f}, -1.1f);
        return isEqual((a * b).toMat4(), a.toMat4() * b.toMat4());
    }());

    check("Mat4::fromMat3 agrees with toMat4",
          isEqual(Mat4f::fromMat3(q.toMat3()), m));
}


/* ========================================================================= //
    toAxisAngle
// ========================================================================= */

static void testToAxisAngle() {
    section("toAxisAngle");

    Vec3f axis;
    float angle;

    // identity has no meaningful axis, so the stand-in is returned instead
    Quatf::identity().toAxisAngle(axis, angle);
    check("toAxisAngle of identity gives angle 0", isEqual(angle, 0.f));
    check("toAxisAngle of identity gives the stand-in axis", isEqual(axis, Vec3f{1, 0, 0}));

    Quatf q = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    q.toAxisAngle(axis, angle);
    check("toAxisAngle recovers the axis", isEqual(axis, math::normalize(Vec3f{1, 2, 3})));
    check("toAxisAngle recovers the angle", isEqual(angle, 0.7f));
    check("toAxisAngle returns a unit axis", isEqual(axis.magnitude(), 1.f));
    check("toAxisAngle round-trips", sameRotation(Quatf::fromAxisAngle(axis, angle), q));

    // w == 0 here, the case acos-based formulations get wrong
    Quatf flip = Quatf::fromAxisAngle({0, 1, 0}, PI);
    flip.toAxisAngle(axis, angle);
    check("toAxisAngle of a 180 degree rotation gives PI", isEqual(angle, PI));
    check("toAxisAngle of a 180 degree rotation keeps the axis",
          isEqual(axis, Vec3f{0, 1, 0}));

    /*
        Small angles need a RELATIVE tolerance. isEqual is absolute at 1e-4, so
        comparing a 1e-5 angle against 1e-5 passes even if the function returns
        zero -- which is exactly the failure an acos(w) formulation produces.
    */
    const float SMALL = 1e-5f;
    Quatf tiny = Quatf::fromAxisAngle({0, 0, 1}, SMALL);
    tiny.toAxisAngle(axis, angle);
    check("toAxisAngle keeps precision at a small angle",
          std::fabs(angle - SMALL) < SMALL * 0.01f);
    check("toAxisAngle keeps the axis at a small angle", isEqual(axis, Vec3f{0, 0, 1}));

    // below the epsilon guard the axis is noise, so the stand-in takes over
    Quatf tinier = Quatf::fromAxisAngle({0, 0, 1}, 1e-9f);
    tinier.toAxisAngle(axis, angle);
    check("toAxisAngle below the epsilon guard gives angle 0", isEqual(angle, 0.f));
    check("toAxisAngle below the epsilon guard gives the stand-in axis",
          isEqual(axis, Vec3f{1, 0, 0}));

    // an exactly-zero vector part would be 0/0 without the guard
    Quatf zeroV = {{0, 0, 0}, 1.f};
    zeroV.toAxisAngle(axis, angle);
    check("toAxisAngle never returns NaN",
          std::isfinite(angle) && std::isfinite(axis.x) &&
          std::isfinite(axis.y) && std::isfinite(axis.z));

    /*
        atan2(k*len, k*w) == atan2(len, w) and the axis divides by its own
        magnitude, so neither result depends on the quaternion being unit.
    */
    Quatf scaled = q * 3.f;
    Vec3f sAxis; float sAngle;
    scaled.toAxisAngle(sAxis, sAngle);
    check("toAxisAngle is scale-invariant",
          isEqual(sAxis, math::normalize(Vec3f{1, 2, 3})) && isEqual(sAngle, 0.7f));

    // the range is [0, 2PI]: -q describes the same rotation the long way round
    Quatf neg = -q;
    neg.toAxisAngle(axis, angle);
    check("toAxisAngle of -q is the same rotation",
          sameRotation(Quatf::fromAxisAngle(axis, angle), q));
    check("toAxisAngle stays within [0, 2PI]", [] {
        for (int deg = -360; deg <= 360; deg += 7) {
            Quatf r = Quatf::fromAxisAngle({1, -2, 0.5f}, deg * PI / 180.f);
            Vec3f a; float t;
            r.toAxisAngle(a, t);
            if (t < 0.f || t > 2.f * PI + 1e-4f) return false;
        }
        return true;
    }());

    check("toAxisAngle round-trips across a sweep of rotations", [] {
        const Vec3f axes[] = {
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}, {3, 2, 1}, {-2, 1, -0.5f}
        };
        for (const Vec3f& ax : axes) {
            for (int deg = -180; deg <= 180; deg += 5) {
                Quatf r = Quatf::fromAxisAngle(ax, deg * PI / 180.f);
                Vec3f a; float t;
                r.toAxisAngle(a, t);
                if (!sameRotation(Quatf::fromAxisAngle(a, t), r)) return false;
                if (!isEqual(a.magnitude(), 1.f)) return false;
            }
        }
        return true;
    }());
}


/* ========================================================================= //
    operator== / operator!=
// ========================================================================= */

static void testEquality() {
    section("operator== / operator!=");

    Quatf q     = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    Quatf same  = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    Quatf other = Quatf::fromAxisAngle({1, 2, 3}, 0.8f);

    check("q == q", q == q);
    check("q == an identically built quaternion", q == same);
    check("q != a different quaternion", q != other);
    check("operator!= is the negation of operator==", (q != same) == !(q == same));
    check("identity() == identity()", Quatf::identity() == Quatf::identity());

    /*
        Comparison is COMPONENT-WISE, not rotational: q and -q are the same
        rotation but compare unequal. sameRotation() is the helper to use when
        the rotation is what matters.
    */
    check("q != -q (comparison is component-wise, not rotational)", q != -q);
    check("q and -q are still the same rotation", sameRotation(q, -q));

    // both operators are const, so they work on const objects and temporaries
    const Quatf cq = q;
    const Quatf cSame = same;
    check("operator== works on const quaternions", cq == cSame);
    check("operator!= works on const quaternions", !(cq != cSame));

    // exact float comparison: a value that differs by one ulp is not equal
    Quatf nudged = q;
    nudged.w = std::nextafter(q.w, 2.f);
    check("operator== is exact, not tolerant", q != nudged);
}


/* ========================================================================= //
    normalize (free function, matching Vec)
// ========================================================================= */

static void testNormalize() {
    section("normalize");

    Quatf unit = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
    Quatf big  = unit * 5.f;

    check("normalize returns a unit quaternion", isEqual(math::normalize(big).norm(), 1.f));
    check("normalize preserves the rotation", sameRotation(math::normalize(big), unit));
    check("normalize of a unit quaternion is a no-op",
          isEqual(math::normalize(unit), unit));

    // the whole reason for a free function: the member version mutates
    Quatf original = big;
    Quatf result = math::normalize(big);
    check("normalize leaves its argument alone", big == original);
    check("normalize returns the normalized value", isEqual(result.norm(), 1.f));

    // member normalize() mutates in place and must agree with the free version
    Quatf mutated = big;
    mutated.normalize();
    check("member normalize() agrees with the free normalize()",
          isEqual(mutated, math::normalize(original)));
    check("member normalize() mutates in place", isEqual(mutated.norm(), 1.f));

    // usable in an expression on a const quaternion, which the member cannot be
    const Quatf cbig = big;
    check("normalize works on a const quaternion", isEqual(math::normalize(cbig).norm(), 1.f));

    check("normalize of a small quaternion still gives unit length",
          isEqual(math::normalize(unit * 1e-6f).norm(), 1.f));
}


int main() {
    testConstruction();
    testNormAndInverse();
    testArithmetic();
    testRotation();
    testToMat3();
    testFromMat3();
    testToMat4();
    testToAxisAngle();
    testEquality();
    testNormalize();
    testSlerp();

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    /* ===================================================================== //
        NOT YET IMPLEMENTED

        Uncomment each block as the corresponding piece is added.

        --- nlerp ---------------------------------------------------------
        Cheaper than slerp, non-constant angular velocity. Currently only
        exists inline inside slerp's near-parallel branch.

            check("nlerp endpoints", ...);
            check("nlerp stays unit", ...);

        --- Euler conversion ----------------------------------------------
        Pick an order (XYZ, ZYX, ...) and document it; the conversions are
        not interchangeable. Expect gimbal lock at +/-90 on the middle axis.

            check("fromEuler/toEuler round-trip away from gimbal lock", ...);

        --- operator<< ----------------------------------------------------
        Matching what Vec/Mat already have in Printing.hpp.
    // ===================================================================== */

    return failures;
}
