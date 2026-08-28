#include <iostream>
#include <cmath>
#include <cstdint>
#include <string>

#include "../include/Matrix.hpp"
#include "../include/Printing.hpp"


/* ========================================================================= //
    Test-check helpers: prints a green [PASS] or red [FAIL] line for each
    checked value, comparing against a hand-verified expected value.
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

inline bool isEqual(float a, float b) { return std::fabs(a - b) < 1e-4f; }
inline bool isEqual(bool a, bool b) { return a == b; }

// Component-wise, epsilon-tolerant — needed because expected values below
// are typed as rounded literals (e.g. 0.707107f), which won't bit-exactly
// match a runtime-computed sqrt()/division result.
template <typename T> bool isEqual(const math::Vec2<T>& a, const math::Vec2<T>& b)
{ return isEqual(a.x, b.x) && isEqual(a.y, b.y); }

template <typename T> bool isEqual(const math::Vec3<T>& a, const math::Vec3<T>& b)
{ return isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z); }

template <typename T> bool isEqual(const math::Vec4<T>& a, const math::Vec4<T>& b)
{ return isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z) && isEqual(a.w, b.w); }

template <size_t N, typename T> bool isEqual(const math::Vec<N, T>& a, const math::Vec<N, T>& b) {
    for (size_t i = 0; i < N; ++i)
        if (!isEqual(a[i], b[i])) return false;
    return true;
}

template <typename T> bool isEqual(const math::Mat2<T>& a, const math::Mat2<T>& b)
{ return isEqual(a[0], b[0]) && isEqual(a[1], b[1]); }

template <typename T> bool isEqual(const math::Mat3<T>& a, const math::Mat3<T>& b)
{ return isEqual(a[0], b[0]) && isEqual(a[1], b[1]) && isEqual(a[2], b[2]); }

template <typename T> bool isEqual(const math::Mat4<T>& a, const math::Mat4<T>& b)
{ return isEqual(a[0], b[0]) && isEqual(a[1], b[1]) && isEqual(a[2], b[2]) && isEqual(a[3], b[3]); }

// a.rows[i] rather than a[i] — Mat<N,M,T> has no operator[] yet
template <size_t N, size_t M, typename T> bool isEqual(const math::Mat<N, M, T>& a, const math::Mat<N, M, T>& b) {
    for (size_t i = 0; i < N; ++i)
        if (!isEqual(a.rows[i], b.rows[i])) return false;
    return true;
}

template <typename T> bool isEqual(const T& a, const T& b) { return a == b; }

template <typename T>
void check(const std::string& label, const T& actual, const T& expected) {
    bool pass = isEqual(actual, expected);
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] " << COLOR_RESET
               << label << ": " << actual << std::endl;
    // actual/expected may be matrices, which print with a leading newline and
    // no trailing one — giving this its own std::endl (rather than appending
    // inline) keeps it from running into the last row of `actual` above.
    if (!pass) std::cout << "  expected: " << expected << std::endl;
}


#define TEST_VEC2 0
#define TEST_VEC3 0
#define TEST_VEC4 0
#define TEST_VECN 0
#define TEST_MAT2 0
#define TEST_MAT3 0
#define TEST_MAT4 0
#define TEST_MATNXM 1


#if TEST_VEC2
void testVec2();
#endif

#if TEST_VEC3
void testVec3();
#endif

#if TEST_VEC4
void testVec4();
#endif

#if TEST_VECN
void testVecN();
#endif

#if TEST_MAT2
void testMat2();
#endif

#if TEST_MAT3
void testMat3();
#endif

#if TEST_MATNXM
void testMatNxM();
#endif

#if TEST_MAT4
void testMat4();
#endif


int main() {
#if TEST_VEC2
    testVec2();
#endif

#if TEST_VEC3
    testVec3();
#endif

#if TEST_VEC4
    testVec4();
#endif

#if TEST_VECN
    testVecN();
#endif

#if TEST_MAT2
    testMat2();
#endif

#if TEST_MAT3
    testMat3();
#endif

#if TEST_MAT4
    testMat4();
#endif

#if TEST_MATNXM
    testMatNxM();
#endif

    return 0;
}


#if TEST_VEC2
void testVec2() {
    math::Vec2 V = {2.f, 2.f};
    math::Vec2 U = {1.f, 1.f};
    math::Vec2 W = {0.f, 1.f};   // non-parallel to V, for cross/angle/project/reject/reflect/faceforward

    math::Vec2 add = V + U;
    math::Vec2 sub = V - U;
    math::Vec2 mul = V * 2;
    math::Vec2 div = V / 2;
    math::Vec2 neg = -V;
    math::Vec2 scalarMul = 2.f * V;

    bool eq  = (V == V);
    bool neq = (V != U);

    float dotVU = V.dot(U);
    float magSq = V.magnitudeSq();
    float mag   = V.magnitude();

    math::Vec2 norm = V; norm.normalize();
    math::Vec2 normFree = math::normalize(V);

    math::Vec2 lerped = math::lerp(V, U, 0.5f);

    float dist   = math::distance(V, U);
    float distSq = math::distanceSq(V, U);

    math::Vec2 reflected   = math::reflect(V, W);
    math::Vec2 minVU       = math::min(V, U);
    math::Vec2 maxVU       = math::max(V, U);
    math::Vec2 clamped     = math::clamp(V * 3.f, U, V);
    float      ang         = math::angle(V, W);
    math::Vec2 projected   = math::project(V, W);
    math::Vec2 rejected    = math::reject(V, W);
    math::Vec2 faced       = math::faceforward(W, V, U);

    std::cout << std::endl;

    std::cout << "Vec2:" << std::endl;

    std::cout << "   V: " << V << std::endl;
    std::cout << "   U: " << U << std::endl;
    std::cout << "   W: " << W << std::endl;

    std::cout << std::endl;

    check("add",        add,       math::Vec2f{3.f, 3.f});
    check("sub",        sub,       math::Vec2f{1.f, 1.f});
    check("mul",        mul,       math::Vec2f{4.f, 4.f});
    check("div",        div,       math::Vec2f{1.f, 1.f});
    check("neg",        neg,       math::Vec2f{-2.f, -2.f});
    check("scalar mul", scalarMul, math::Vec2f{4.f, 4.f});

    std::cout << std::endl;

    check("V == V", eq,  true);
    check("V != U", neq, true);

    std::cout << std::endl;

    check("dot(V, U)",       dotVU, 4.f);
    check("magnitudeSq(V)",  magSq, 8.f);
    check("magnitude(V)",    mag,   2.828427f);

    std::cout << std::endl;

    check("normalize (member, in-place)", norm,     math::Vec2f{0.707107f, 0.707107f});
    check("normalize (free function)",    normFree, math::Vec2f{0.707107f, 0.707107f});

    std::cout << std::endl;

    check("lerp(V, U, 0.5)", lerped, math::Vec2f{1.5f, 1.5f});

    std::cout << std::endl;

    check("distance(V, U)",        dist,      1.414214f);
    check("distanceSq(V, U)",      distSq,    2.f);
    check("reflect(V, W)",         reflected, math::Vec2f{2.f, -2.f});
    check("min(V, U)",             minVU,     math::Vec2f{1.f, 1.f});
    check("max(V, U)",             maxVU,     math::Vec2f{2.f, 2.f});
    check("clamp(V*3, U, V)",      clamped,   math::Vec2f{2.f, 2.f});
    check("angle(V, W)",           ang,       0.785398f);
    check("project(V, W)",         projected, math::Vec2f{0.f, 2.f});
    check("reject(V, W)",          rejected,  math::Vec2f{2.f, 0.f});
    check("faceforward(W, V, U)",  faced,     math::Vec2f{0.f, -1.f});

    std::cout << std::endl;

    check("W[0]", W[0], 0.f);
    check("W[1]", W[1], 1.f);
}
#endif


#if TEST_VEC3
void testVec3() {
    math::Vec3 V = {2.f, 2.f, 2.f};
    math::Vec3 U = {1.f, 1.f, 1.f};
    math::Vec3 W = {0.f, 1.f, 0.f};   // non-parallel to V, for cross/angle/project/reject/reflect/faceforward

    math::Vec3 add = V + U;
    math::Vec3 sub = V - U;
    math::Vec3 mul = V * 2;
    math::Vec3 div = V / 2;
    math::Vec3 neg = -V;
    math::Vec3 scalarMul = 2.f * V;

    bool eq  = (V == V);
    bool neq = (V != U);

    float dotVU = V.dot(U);
    math::Vec3 crossVW = V.cross(W);
    float magSq = V.magnitudeSq();
    float mag   = V.magnitude();

    math::Vec3 norm = V; norm.normalize();
    math::Vec3 normFree = math::normalize(V);

    math::Vec3 lerped = math::lerp(V, U, 0.5f);

    float dist   = math::distance(V, U);
    float distSq = math::distanceSq(V, U);

    math::Vec3 reflected   = math::reflect(V, W);
    math::Vec3 minVU       = math::min(V, U);
    math::Vec3 maxVU       = math::max(V, U);
    math::Vec3 clamped     = math::clamp(V * 3.f, U, V);
    float      ang         = math::angle(V, W);
    math::Vec3 projected   = math::project(V, W);
    math::Vec3 rejected    = math::reject(V, W);
    math::Vec3 faced       = math::faceforward(W, V, U);

    std::cout << std::endl;

    std::cout << "Vec3:" << std::endl;

    std::cout << "   V: " << V << std::endl;
    std::cout << "   U: " << U << std::endl;
    std::cout << "   W: " << W << std::endl;

    std::cout << std::endl;

    check("add",        add,       math::Vec3f{3.f, 3.f, 3.f});
    check("sub",        sub,       math::Vec3f{1.f, 1.f, 1.f});
    check("mul",        mul,       math::Vec3f{4.f, 4.f, 4.f});
    check("div",        div,       math::Vec3f{1.f, 1.f, 1.f});
    check("neg",        neg,       math::Vec3f{-2.f, -2.f, -2.f});
    check("scalar mul", scalarMul, math::Vec3f{4.f, 4.f, 4.f});

    std::cout << std::endl;

    check("V == V", eq,  true);
    check("V != U", neq, true);

    std::cout << std::endl;

    check("dot(V, U)",       dotVU,   6.f);
    check("cross(V, W)",     crossVW, math::Vec3f{-2.f, 0.f, 2.f});
    check("magnitudeSq(V)",  magSq,   12.f);
    check("magnitude(V)",    mag,     3.464102f);

    std::cout << std::endl;

    check("normalize (member, in-place)", norm,     math::Vec3f{0.57735f, 0.57735f, 0.57735f});
    check("normalize (free function)",    normFree, math::Vec3f{0.57735f, 0.57735f, 0.57735f});

    std::cout << std::endl;

    check("lerp(V, U, 0.5)", lerped, math::Vec3f{1.5f, 1.5f, 1.5f});

    std::cout << std::endl;

    check("distance(V, U)",        dist,      1.732051f);
    check("distanceSq(V, U)",      distSq,    3.f);
    check("reflect(V, W)",         reflected, math::Vec3f{2.f, -2.f, 2.f});
    check("min(V, U)",             minVU,     math::Vec3f{1.f, 1.f, 1.f});
    check("max(V, U)",             maxVU,     math::Vec3f{2.f, 2.f, 2.f});
    check("clamp(V*3, U, V)",      clamped,   math::Vec3f{2.f, 2.f, 2.f});
    check("angle(V, W)",           ang,       0.955317f);
    check("project(V, W)",         projected, math::Vec3f{0.f, 2.f, 0.f});
    check("reject(V, W)",          rejected,  math::Vec3f{2.f, 0.f, 2.f});
    check("faceforward(W, V, U)",  faced,     math::Vec3f{0.f, -1.f, 0.f});

    std::cout << std::endl;
}
#endif


#if TEST_VEC4
void testVec4() {
    math::Vec4 V = {2.f, 2.f, 2.f, 2.f};
    math::Vec4 U = {1.f, 1.f, 1.f, 1.f};
    math::Vec4 W = {0.f, 1.f, 0.f, 0.f};   // non-parallel to V, for cross/angle/project/reject/reflect/faceforward

    math::Vec4 add = V + U;
    math::Vec4 sub = V - U;
    math::Vec4 mul = V * 2;
    math::Vec4 div = V / 2;
    math::Vec4 neg = -V;
    math::Vec4 scalarMul = 2.f * V;

    bool eq  = (V == V);
    bool neq = (V != U);

    float dotVU = V.dot(U);
    math::Vec4 crossVW = V.cross(W);
    float magSq = V.magnitudeSq();
    float mag   = V.magnitude();

    math::Vec4 norm = V; norm.normalize();
    math::Vec4 normFree = math::normalize(V);

    math::Vec4 lerped = math::lerp(V, U, 0.5f);

    float dist   = math::distance(V, U);
    float distSq = math::distanceSq(V, U);

    math::Vec4 reflected   = math::reflect(V, W);
    math::Vec4 minVU       = math::min(V, U);
    math::Vec4 maxVU       = math::max(V, U);
    math::Vec4 clamped     = math::clamp(V * 3.f, U, V);
    float      ang         = math::angle(V, W);
    math::Vec4 projected   = math::project(V, W);
    math::Vec4 rejected    = math::reject(V, W);
    math::Vec4 faced       = math::faceforward(W, V, U);

    std::cout << std::endl;

    std::cout << "Vec4:" << std::endl;

    std::cout << "   V: " << V << std::endl;
    std::cout << "   U: " << U << std::endl;
    std::cout << "   W: " << W << std::endl;

    std::cout << std::endl;

    check("add",        add,       math::Vec4f{3.f, 3.f, 3.f, 3.f});
    check("sub",        sub,       math::Vec4f{1.f, 1.f, 1.f, 1.f});
    check("mul",        mul,       math::Vec4f{4.f, 4.f, 4.f, 4.f});
    check("div",        div,       math::Vec4f{1.f, 1.f, 1.f, 1.f});
    check("neg",        neg,       math::Vec4f{-2.f, -2.f, -2.f, -2.f});
    check("scalar mul", scalarMul, math::Vec4f{4.f, 4.f, 4.f, 4.f});

    std::cout << std::endl;

    check("V == V", eq,  true);
    check("V != U", neq, true);

    std::cout << std::endl;

    check("dot(V, U)",       dotVU,   8.f);
    check("cross(V, W)",     crossVW, math::Vec4f{-2.f, 0.f, 2.f, 0.f});
    check("magnitudeSq(V)",  magSq,   16.f);
    check("magnitude(V)",    mag,     4.f);

    std::cout << std::endl;

    check("normalize (member, in-place)", norm,     math::Vec4f{0.5f, 0.5f, 0.5f, 0.5f});
    check("normalize (free function)",    normFree, math::Vec4f{0.5f, 0.5f, 0.5f, 0.5f});

    std::cout << std::endl;

    check("lerp(V, U, 0.5)", lerped, math::Vec4f{1.5f, 1.5f, 1.5f, 1.5f});

    std::cout << std::endl;

    check("distance(V, U)",        dist,      2.f);
    check("distanceSq(V, U)",      distSq,    4.f);
    check("reflect(V, W)",         reflected, math::Vec4f{2.f, -2.f, 2.f, 2.f});
    check("min(V, U)",             minVU,     math::Vec4f{1.f, 1.f, 1.f, 1.f});
    check("max(V, U)",             maxVU,     math::Vec4f{2.f, 2.f, 2.f, 2.f});
    check("clamp(V*3, U, V)",      clamped,   math::Vec4f{2.f, 2.f, 2.f, 2.f});
    check("angle(V, W)",           ang,       1.047198f);
    check("project(V, W)",         projected, math::Vec4f{0.f, 2.f, 0.f, 0.f});
    check("reject(V, W)",          rejected,  math::Vec4f{2.f, 0.f, 2.f, 2.f});
    check("faceforward(W, V, U)",  faced,     math::Vec4f{0.f, -1.f, 0.f, 0.f});

    std::cout << std::endl;
}
#endif


#if TEST_VECN
void testVecN() {
    math::Vec<5, float> V = {2.f, 2.f, 2.f, 2.f, 2.f};
    math::Vec<5, float> U = {1.f, 1.f, 1.f, 1.f, 1.f};
    math::Vec<5, float> W = {0.f, 1.f, 0.f, 0.f, 0.f};   // non-parallel to V, for angle/project/reject/reflect/faceforward

    math::Vec<5, float> add = V + U;
    math::Vec<5, float> sub = V - U;
    math::Vec<5, float> mul = V * 2.f;
    math::Vec<5, float> div = V / 2.f;
    math::Vec<5, float> neg = -V;
    math::Vec<5, float> scalarMul = 2.f * V;

    bool eq  = (V == V);
    bool neq = (V != U);

    float dotVU = V.dot(U);
    float magSq = V.magnitudeSq();
    float mag   = V.magnitude();

    math::Vec<5, float> norm = V; norm.normalize();
    math::Vec<5, float> normFree = math::normalize(V);

    math::Vec<5, float> lerped = math::lerp(V, U, 0.5f);

    float dist   = math::distance(V, U);
    float distSq = math::distanceSq(V, U);

    math::Vec<5, float> reflected   = math::reflect(V, W);
    math::Vec<5, float> minVU       = math::min(V, U);
    math::Vec<5, float> maxVU       = math::max(V, U);
    math::Vec<5, float> clamped     = math::clamp(V * 3.f, U, V);
    float                ang         = math::angle(V, W);
    math::Vec<5, float> projected   = math::project(V, W);
    math::Vec<5, float> rejected    = math::reject(V, W);
    math::Vec<5, float> faced       = math::faceforward(W, V, U);

    std::cout << std::endl;

    std::cout << "VecN (N=5):" << std::endl;

    std::cout << "   V: " << V << std::endl;
    std::cout << "   U: " << U << std::endl;
    std::cout << "   W: " << W << std::endl;

    std::cout << std::endl;

    check("add",        add,       math::Vec<5, float>{3.f, 3.f, 3.f, 3.f, 3.f});
    check("sub",        sub,       math::Vec<5, float>{1.f, 1.f, 1.f, 1.f, 1.f});
    check("mul",        mul,       math::Vec<5, float>{4.f, 4.f, 4.f, 4.f, 4.f});
    check("div",        div,       math::Vec<5, float>{1.f, 1.f, 1.f, 1.f, 1.f});
    check("neg",        neg,       math::Vec<5, float>{-2.f, -2.f, -2.f, -2.f, -2.f});
    check("scalar mul", scalarMul, math::Vec<5, float>{4.f, 4.f, 4.f, 4.f, 4.f});

    std::cout << std::endl;

    check("V == V", eq,  true);
    check("V != U", neq, true);

    std::cout << std::endl;

    check("dot(V, U)",       dotVU, 10.f);
    check("magnitudeSq(V)",  magSq, 20.f);
    check("magnitude(V)",    mag,   4.472136f);

    std::cout << std::endl;

    check("normalize (member, in-place)", norm,     math::Vec<5, float>{0.447214f, 0.447214f, 0.447214f, 0.447214f, 0.447214f});
    check("normalize (free function)",    normFree, math::Vec<5, float>{0.447214f, 0.447214f, 0.447214f, 0.447214f, 0.447214f});

    std::cout << std::endl;

    check("lerp(V, U, 0.5)", lerped, math::Vec<5, float>{1.5f, 1.5f, 1.5f, 1.5f, 1.5f});

    std::cout << std::endl;

    check("distance(V, U)",        dist,      2.236068f);
    check("distanceSq(V, U)",      distSq,    5.f);
    check("reflect(V, W)",         reflected, math::Vec<5, float>{2.f, -2.f, 2.f, 2.f, 2.f});
    check("min(V, U)",             minVU,     math::Vec<5, float>{1.f, 1.f, 1.f, 1.f, 1.f});
    check("max(V, U)",             maxVU,     math::Vec<5, float>{2.f, 2.f, 2.f, 2.f, 2.f});
    check("clamp(V*3, U, V)",      clamped,   math::Vec<5, float>{2.f, 2.f, 2.f, 2.f, 2.f});
    check("angle(V, W)",           ang,       1.107149f);
    check("project(V, W)",         projected, math::Vec<5, float>{0.f, 2.f, 0.f, 0.f, 0.f});
    check("reject(V, W)",          rejected,  math::Vec<5, float>{2.f, 0.f, 2.f, 2.f, 2.f});
    check("faceforward(W, V, U)",  faced,     math::Vec<5, float>{0.f, -1.f, 0.f, 0.f, 0.f});

    std::cout << std::endl;

    check("size()", V.size(), size_t(5));
    check("W[0]",   W[0], 0.f);
    check("W[1]",   W[1], 1.f);
}
#endif


#if TEST_MAT2
void testMat2() {
    math::Mat2f A = { 1.f, 2.f, 3.f, 4.f };
    math::Mat2f B = { 5.f, 6.f, 7.f, 8.f };
    math::Vec2f v = {1.f, 1.f};

    math::Mat2f add       = A + B;
    math::Mat2f sub       = A - B;
    math::Mat2f mul       = A * 2.f;
    math::Mat2f div       = A / 2.f;
    math::Mat2f matMul    = A * B;
    math::Vec2f vecMul    = A * v;

    bool eq                     = (A == A);
    bool neq                    = (A != B);

    math::Mat2f t         = A.transpose();
    float       det       = A.determinant();

    std::cout << std::endl;

    std::cout << "Mat2:" << std::endl;

    std::cout << "A:" << A << std::endl;
    std::cout << "B:" << B << std::endl;
    std::cout << "v: " << v << std::endl;

    std::cout << std::endl;

    check("add", add, math::Mat2f{ 6.f,  8.f, 10.f, 12.f});
    check("sub", sub, math::Mat2f{-4.f, -4.f, -4.f, -4.f});
    check("mul", mul, math::Mat2f{ 2.f,  4.f,  6.f,  8.f});
    check("div", div, math::Mat2f{0.5f,  1.f, 1.5f,  2.f});

    std::cout << std::endl;

    check("A*B", matMul, math::Mat2f{19.f, 22.f, 43.f, 50.f});
    check("A*v", vecMul, math::Vec2f{3.f, 7.f});

    std::cout << std::endl;

    check("A == A", eq,  true);
    check("A != B", neq, true);

    std::cout << std::endl;

    check("transpose(A)",   t,   math::Mat2f{1.f, 3.f, 2.f, 4.f});
    check("determinant(A)", det, -2.f);

    std::cout << std::endl;

    // Inverse: one case with no inverse, one case with an inverse,
    // verified by checking that M * M.inverse() comes back to identity.
    math::Mat2f singular = { 1.f, 2.f, 2.f, 4.f };   // det = 1*4 - 2*2 = 0
    std::cout << "singular:" << singular << std::endl;
    check("determinant(singular) (0 -> no inverse, not calling .inverse())", singular.determinant(), 0.f);

    std::cout << std::endl;

    math::Mat2f invertible = A;   // det = -2, non-singular
    math::Mat2f inv        = invertible.inverse();
    std::cout << "invertible:" << invertible << std::endl;
    check("determinant(invertible)",              invertible.determinant(), -2.f);
    check("inverse(invertible)",                  inv,                      math::Mat2f{-2.f, 1.f, 1.5f, -0.5f});
    check("invertible * inverse(invertible)",     invertible * inv,         math::Mat2f::identity());

    std::cout << std::endl;

    check("data()[0]", A.data()[0], 1.f);
    check("data()[1]", A.data()[1], 2.f);
    check("data()[2]", A.data()[2], 3.f);
    check("data()[3]", A.data()[3], 4.f);

    std::cout << std::endl;

    std::cout << "Mat2f is " << sizeof(math::Mat2f) / sizeof(uint8_t) << " bytes. " << std::endl;
    std::cout << "Mat2d is " << sizeof(math::Mat2d) / sizeof(uint8_t) << " bytes. " << std::endl;
    std::cout << "Mat2i is " << sizeof(math::Mat2i) / sizeof(uint8_t) << " bytes. " << std::endl;
    std::cout << "Mat2u is " << sizeof(math::Mat2u) / sizeof(uint8_t) << " bytes. " << std::endl;

    std::cout << std::endl;
}
#endif


#if TEST_MAT3
void testMat3() {
    math::Mat3f A = { 
        1.f, 2.f, 3.f, 
        4.f, 5.f, 6.f, 
        7.f, 8.f, 9.f 
    };
    math::Mat3f B = { 
        9.f, 8.f, 7.f, 
        6.f, 5.f, 4.f, 
        3.f, 2.f, 1.f 
    };
    math::Vec3f v = {1.f, 1.f, 1.f};

    math::Mat3f add       = A + B;
    math::Mat3f sub       = A - B;
    math::Mat3f mul       = A * 2.f;
    math::Mat3f div       = A / 2.f;
    math::Mat3f matMul    = A * B;
    math::Vec3f vecMul    = A * v;

    bool eq                     = (A == A);
    bool neq                    = (A != B);

    math::Mat3f t         = A.transpose();
    float       det       = A.determinant();

    std::cout << std::endl;

    std::cout << "Mat3:" << std::endl;

    std::cout << "A:" << A << std::endl;
    std::cout << "B:" << B << std::endl;
    std::cout << "v: " << v << std::endl;

    std::cout << std::endl;

    check("add", add, math::Mat3f{10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f});
    check("sub", sub, math::Mat3f{-8.f, -6.f, -4.f, -2.f,  0.f,  2.f,  4.f,  6.f,  8.f});
    check("mul", mul, math::Mat3f{ 2.f,  4.f,  6.f,  8.f, 10.f, 12.f, 14.f, 16.f, 18.f});
    check("div", div, math::Mat3f{0.5f,  1.f, 1.5f,  2.f, 2.5f,  3.f, 3.5f,  4.f, 4.5f});

    std::cout << std::endl;

    check("A*B", matMul, math::Mat3f{30.f, 24.f, 18.f, 84.f, 69.f, 54.f, 138.f, 114.f, 90.f});
    check("A*v", vecMul, math::Vec3f{6.f, 15.f, 24.f});

    std::cout << std::endl;

    check("A == A", eq,  true);
    check("A != B", neq, true);

    std::cout << std::endl;

    check("transpose(A)", t, math::Mat3f{1.f, 4.f, 7.f, 2.f, 5.f, 8.f, 3.f, 6.f, 9.f});

    std::cout << std::endl;

    // Inverse: one case with no inverse (A itself is singular, det = 0
    // since its rows are in arithmetic progression), one case with an
    // inverse, verified by checking that M * M.inverse() comes back to
    // identity.
    check("determinant(A) (0 -> no inverse, not calling .inverse())", det, 0.f);

    std::cout << std::endl;

    math::Mat3f invertible = {
        1.f, 2.f, 3.f,
        0.f, 1.f, 4.f,
        5.f, 6.f, 0.f
    };   // det = 1
    math::Mat3f inv = invertible.inverse();
    std::cout << "invertible:" << invertible << std::endl;
    check("determinant(invertible)",          invertible.determinant(), 1.f);
    check("inverse(invertible)",              inv, math::Mat3f{-24.f, 18.f, 5.f, 20.f, -15.f, -4.f, -5.f, 4.f, 1.f});
    check("invertible * inverse(invertible)", invertible * inv, math::Mat3f::identity());

    std::cout << std::endl;

    check("data()[0]", A.data()[0], 1.f);
    check("data()[1]", A.data()[1], 2.f);
    check("data()[2]", A.data()[2], 3.f);
    check("data()[3]", A.data()[3], 4.f);
    check("data()[4]", A.data()[4], 5.f);
    check("data()[5]", A.data()[5], 6.f);
    check("data()[6]", A.data()[6], 7.f);
    check("data()[7]", A.data()[7], 8.f);
    check("data()[8]", A.data()[8], 9.f);

    std::cout << std::endl;

    std::cout << "Mat3f is " << sizeof(math::Mat3f) / sizeof(uint8_t) << " bytes. " << std::endl;
    std::cout << "Mat3d is " << sizeof(math::Mat3d) / sizeof(uint8_t) << " bytes. " << std::endl;
    std::cout << "Mat3i is " << sizeof(math::Mat3i) / sizeof(uint8_t) << " bytes. " << std::endl;
    std::cout << "Mat3u is " << sizeof(math::Mat3u) / sizeof(uint8_t) << " bytes. " << std::endl;
}
#endif


#if TEST_MAT4
void testMat4() {
    math::Mat4<float> A = {
        1.f,  2.f,  3.f,  4.f,
        5.f,  6.f,  7.f,  8.f,
        9.f, 10.f, 11.f, 12.f,
       13.f, 14.f, 15.f, 16.f
    };
    math::Mat4<float> B = {
       16.f, 15.f, 14.f, 13.f,
       12.f, 11.f, 10.f,  9.f,
        8.f,  7.f,  6.f,  5.f,
        4.f,  3.f,  2.f,  1.f
    };
    math::Vec4f v = {1.f, 1.f, 1.f, 1.f};

    math::Mat4<float> add    = A + B;
    math::Mat4<float> sub    = A - B;
    math::Mat4<float> mul    = A * 2.f;
    math::Mat4<float> div    = A / 2.f;
    math::Mat4<float> matMul = A * B;
    math::Vec4f       vecMul = A * v;

    bool eq  = (A == A);
    bool neq = (A != B);

    math::Mat4<float> t   = A.transpose();
    float              det = A.determinant();

    std::cout << std::endl;

    std::cout << "Mat4:" << std::endl;

    std::cout << "A:" << A << std::endl;
    std::cout << "B:" << B << std::endl;
    std::cout << "v: " << v << std::endl;

    std::cout << std::endl;

    check("add", add, math::Mat4<float>{
        17.f, 17.f, 17.f, 17.f,
        17.f, 17.f, 17.f, 17.f,
        17.f, 17.f, 17.f, 17.f,
        17.f, 17.f, 17.f, 17.f
    });
    check("sub", sub, math::Mat4<float>{
        -15.f, -13.f, -11.f, -9.f,
         -7.f,  -5.f,  -3.f, -1.f,
          1.f,   3.f,   5.f,  7.f,
          9.f,  11.f,  13.f, 15.f
    });
    check("mul", mul, math::Mat4<float>{
         2.f,  4.f,  6.f,  8.f,
        10.f, 12.f, 14.f, 16.f,
        18.f, 20.f, 22.f, 24.f,
        26.f, 28.f, 30.f, 32.f
    });
    check("div", div, math::Mat4<float>{
        0.5f, 1.f, 1.5f, 2.f,
        2.5f, 3.f, 3.5f, 4.f,
        4.5f, 5.f, 5.5f, 6.f,
        6.5f, 7.f, 7.5f, 8.f
    });

    std::cout << std::endl;

    check("A*B", matMul, math::Mat4<float>{
         80.f,  70.f,  60.f,  50.f,
        240.f, 214.f, 188.f, 162.f,
        400.f, 358.f, 316.f, 274.f,
        560.f, 502.f, 444.f, 386.f
    });
    check("A*v", vecMul, math::Vec4f{10.f, 26.f, 42.f, 58.f});

    std::cout << std::endl;

    check("A == A", eq,  true);
    check("A != B", neq, true);

    std::cout << std::endl;

    check("transpose(A)", t, math::Mat4<float>{
        1.f, 5.f,  9.f, 13.f,
        2.f, 6.f, 10.f, 14.f,
        3.f, 7.f, 11.f, 15.f,
        4.f, 8.f, 12.f, 16.f
    });

    std::cout << std::endl;

    // Inverse: A itself is singular (its rows are in arithmetic progression,
    // same reasoning as the Mat3 test — rank < 4, so det = 0, no inverse).
    // A separate diagonal matrix is used for the invertible case since its
    // inverse can be checked by hand trivially (diag(a,b,c,d)^-1 = diag(1/a,1/b,1/c,1/d)),
    // verified by checking that M * M.inverse() comes back to identity.
    check("determinant(A) (0 -> no inverse, not calling .inverse())", det, 0.f);

    std::cout << std::endl;

    math::Mat4<float> invertible = {
        2.f, 0.f, 0.f, 0.f,
        0.f, 3.f, 0.f, 0.f,
        0.f, 0.f, 4.f, 0.f,
        0.f, 0.f, 0.f, 5.f
    };   // det = 2*3*4*5 = 120
    math::Mat4<float> inv = invertible.inverse();
    std::cout << "invertible:" << invertible << std::endl;
    check("determinant(invertible)", invertible.determinant(), 120.f);
    check("inverse(invertible)", inv, math::Mat4<float>{
        0.5f,      0.f,  0.f, 0.f,
         0.f, 1.f/3.f,  0.f, 0.f,
         0.f,      0.f, 0.25f, 0.f,
         0.f,      0.f,  0.f, 0.2f
    });
    check("invertible * inverse(invertible)", invertible * inv, math::Mat4<float>::identity());

    std::cout << std::endl;

    for (int i = 0; i < 16; ++i)
        check("data()[" + std::to_string(i) + "]", A.data()[i], static_cast<float>(i + 1));

    std::cout << std::endl;

    check("Mat4<float> size (bytes)",    sizeof(math::Mat4<float>)    / sizeof(uint8_t), size_t(64));
    check("Mat4<double> size (bytes)",   sizeof(math::Mat4<double>)   / sizeof(uint8_t), size_t(128));
    check("Mat4<int32_t> size (bytes)",  sizeof(math::Mat4<int32_t>)  / sizeof(uint8_t), size_t(64));
    check("Mat4<uint32_t> size (bytes)", sizeof(math::Mat4<uint32_t>) / sizeof(uint8_t), size_t(64));

    std::cout << std::endl;
}
#endif


#if TEST_MATNXM
void testMatNxM() {
    // Square (N == M) case — mirrors the Mat2/Mat3/Mat4 pattern exactly,
    // same numbers as the Mat3 test, so the expected values below are the
    // same hand-verified ones.
    math::Mat<3, 3, float> A = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f
    };
    math::Mat<3, 3, float> B = {
        9.f, 8.f, 7.f,
        6.f, 5.f, 4.f,
        3.f, 2.f, 1.f
    };
    math::Vec<3, float> v = {1.f, 1.f, 1.f};

    math::Mat<3, 3, float> add    = A + B;
    math::Mat<3, 3, float> sub    = A - B;
    math::Mat<3, 3, float> mul    = A * 2.f;
    math::Mat<3, 3, float> div    = A / 2.f;
    math::Mat<3, 3, float> matMul = A * B;
    math::Vec<3, float>    vecMul = A * v;

    bool eq  = (A == A);
    bool neq = (A != B);

    math::Mat<3, 3, float> t   = A.transpose();
    float                  det = A.determinant();

    std::cout << std::endl;

    std::cout << "MatNxM (square, 3x3):" << std::endl;

    std::cout << "   A:" << A << std::endl;
    std::cout << "   B:" << B << std::endl;
    std::cout << "   v: " << v << std::endl;

    std::cout << std::endl;

    check("add", add, math::Mat<3, 3, float>{10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f});
    check("sub", sub, math::Mat<3, 3, float>{-8.f, -6.f, -4.f, -2.f, 0.f, 2.f, 4.f, 6.f, 8.f});
    check("mul", mul, math::Mat<3, 3, float>{2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f, 16.f, 18.f});
    check("div", div, math::Mat<3, 3, float>{0.5f, 1.f, 1.5f, 2.f, 2.5f, 3.f, 3.5f, 4.f, 4.5f});

    std::cout << std::endl;

    check("A*B", matMul, math::Mat<3, 3, float>{30.f, 24.f, 18.f, 84.f, 69.f, 54.f, 138.f, 114.f, 90.f});
    check("A*v", vecMul, math::Vec<3, float>{6.f, 15.f, 24.f});

    std::cout << std::endl;

    check("A == A", eq,  true);
    check("A != B", neq, true);

    std::cout << std::endl;

    check("transpose(A)", t, math::Mat<3, 3, float>{1.f, 4.f, 7.f, 2.f, 5.f, 8.f, 3.f, 6.f, 9.f});

    std::cout << std::endl;

    // Inverse: A itself is singular (rows in arithmetic progression, same as
    // the Mat3 test), a separate matrix is used for the invertible case,
    // verified by checking that M * M.inverse() comes back to identity.
    check("determinant(A) (0 -> no inverse, not calling .inverse())", det, 0.f);

    std::cout << std::endl;

    math::Mat<3, 3, float> invertible = {
        1.f, 2.f, 3.f,
        0.f, 1.f, 4.f,
        5.f, 6.f, 0.f
    };   // det = 1
    math::Mat<3, 3, float> inv = invertible.inverse();
    std::cout << "invertible:" << invertible << std::endl;
    check("determinant(invertible)",          invertible.determinant(), 1.f);
    check("inverse(invertible)",              inv, math::Mat<3, 3, float>{-24.f, 18.f, 5.f, 20.f, -15.f, -4.f, -5.f, 4.f, 1.f});
    check("invertible * inverse(invertible)", invertible * inv, math::Mat<3, 3, float>::identity());

    std::cout << std::endl;

    check("data()[0]", A.data()[0], 1.f);
    check("data()[1]", A.data()[1], 2.f);
    check("data()[2]", A.data()[2], 3.f);
    check("data()[3]", A.data()[3], 4.f);
    check("data()[4]", A.data()[4], 5.f);
    check("data()[5]", A.data()[5], 6.f);
    check("data()[6]", A.data()[6], 7.f);
    check("data()[7]", A.data()[7], 8.f);
    check("data()[8]", A.data()[8], 9.f);

    std::cout << std::endl;

    // Rectangular (N != M) case — construction/basic-op sanity check only.
    // determinant()/inverse() are correctly *not* callable here (both are
    // gated with `requires (N == M)`), so deliberately not tested.
    math::Mat<2, 3, float> R = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    std::cout << "R (2x3):" << R << std::endl;

    check("R data()[0]", R.data()[0], 1.f);
    check("R data()[5]", R.data()[5], 6.f);
}
#endif
