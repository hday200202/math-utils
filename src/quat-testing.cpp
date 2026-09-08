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
using math::Mat3f;

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


int main() {
    testConstruction();
    testNormAndInverse();
    testArithmetic();
    testRotation();
    testToMat3();
    testSlerp();

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    /* ===================================================================== //
        NOT YET IMPLEMENTED

        Uncomment each block as the corresponding piece is added.

        --- fromMat3 (matrix -> quaternion) -------------------------------
        Use Shepperd's method: pick whichever of the four diagonal cases has
        the largest magnitude. The naive sqrt(1 + trace) form divides by
        something near zero for 180-degree rotations and produces NaN.

            Quatf q = Quatf::fromAxisAngle({1, 2, 3}, 0.7f);
            check("fromMat3 round-trips", sameRotation(Quatf::fromMat3(q.toMat3()), q));
            check("fromMat3(identity) is identity",
                  sameRotation(Quatf::fromMat3(Mat3f::identity()), Quatf::identity()));
            // the case the naive formula fails on:
            Quatf flip = Quatf::fromAxisAngle({0, 1, 0}, PI);
            check("fromMat3 handles a 180 degree rotation",
                  sameRotation(Quatf::fromMat3(flip.toMat3()), flip));

        --- toMat4 --------------------------------------------------------
        The 3x3 rotation in the upper-left, 1 in the bottom-right corner.

            check("toMat4 upper-left equals toMat3", ...);
            check("toMat4 last row/column is 0,0,0,1", ...);

        --- toAxisAngle ---------------------------------------------------
        Inverse of fromAxisAngle. Watch the near-identity case, where the
        axis is undefined and sin(theta/2) approaches zero.

            Vec3f axis; float angle;
            q.toAxisAngle(axis, angle);
            check("toAxisAngle round-trips", sameRotation(Quatf::fromAxisAngle(axis, angle), q));
            check("toAxisAngle of identity gives angle 0", ...);

        --- operator== / operator!= ---------------------------------------
        Decide whether q and -q compare equal, since they are the same
        rotation, and say which in a comment. Both answers are defensible.

            check("q == q", ...);
            check("q != other", ...);

        --- normalized() const --------------------------------------------
        normalize() mutates; a const version returning a copy is needed for
        expressions. fromAxisAngle currently needs the axis by value only
        because Vec3 lacks the same thing.

            check("normalized() leaves the original alone", ...);

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
