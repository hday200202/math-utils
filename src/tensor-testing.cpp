#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>

#include "../include/ai/Tensor.hpp"

/* ========================================================================= //
    Tensor test suite.

    Grows alongside Tensor.hpp. Everything above the NOT YET IMPLEMENTED
    block at the bottom is live; uncomment each section there as the
    corresponding piece lands.

    Expected values come from numpy (tools/make_fixtures.py -> out/fixtures),
    not from hand calculation. A value worked out by hand using the same wrong
    mental model that produced the bug agrees with the bug.

    Run from the repo root so the fixture paths resolve:
        make debug-tensor && ./bin/debug/tensor-testing
// ========================================================================= */

static constexpr const char* COLOR_GREEN = "\033[32m";
static constexpr const char* COLOR_RED   = "\033[31m";
static constexpr const char* COLOR_RESET = "\033[0m";

static int failures = 0;

using Tensorf = ai::Tensor<float>;
using Shape   = std::vector<size_t>;

inline bool isEqual(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

static void check(const std::string& label, bool pass) {
    std::cout << (pass ? COLOR_GREEN : COLOR_RED) << "[" << (pass ? "PASS" : "FAIL") << "] "
              << COLOR_RESET << label << std::endl;
    if (!pass) ++failures;
}

static void section(const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
}


/* ========================================================================= //
    Fixture loading

    out/fixtures/<name>.txt is:
        <ndim> <dim0> <dim1> ...
        <values, whitespace separated, C-contiguous>
// ========================================================================= */

struct Fixture {
    Shape shape;
    std::vector<float> values;
    bool ok = false;

    size_t numel() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }
};

static std::string fixtureDir = "out/fixtures";

static Fixture loadFixture(const std::string& name) {
    Fixture f;

    std::ifstream in(fixtureDir + "/" + name + ".txt");
    if (!in) {
        std::cout << COLOR_RED << "[FAIL] " << COLOR_RESET
                  << "fixture not found: " << fixtureDir << "/" << name << ".txt"
                  << "  (run: py tools/make_fixtures.py)" << std::endl;
        ++failures;
        return f;
    }

    size_t ndim = 0;
    in >> ndim;
    f.shape.resize(ndim);
    for (size_t i = 0; i < ndim; ++i) in >> f.shape[i];

    f.values.resize(f.numel());
    for (size_t i = 0; i < f.values.size(); ++i) in >> f.values[i];

    f.ok = static_cast<bool>(in);
    return f;
}


/* ========================================================================= //
    Step 1: construction, shape, strides
// ========================================================================= */

static void testConstruction() {
    section("construction");

    Tensorf t(Shape{3, 4});

    check("ndim is the number of dimensions", t.ndim() == 2);
    check("shape is kept", t.shape == Shape({3, 4}));
    check("numel is the product of the shape", t.numel() == 12);
    check("storage is allocated", t.storage != nullptr);
    check("ptr points at the start of storage", t.data() == t.storage.get());
    check("a fresh tensor is contiguous", t.isContiguous());

    // make_shared<T[]> value-initialises, so a new tensor is all zeros
    check("a fresh tensor is zeroed", [&] {
        for (size_t i = 0; i < t.numel(); ++i)
            if (t.data()[i] != 0.f) return false;
        return true;
    }());

    // a scalar: no dimensions, but exactly one element -- numel must not be 0
    Tensorf s(Shape{});
    check("a rank-0 tensor has ndim 0", s.ndim() == 0);
    check("a rank-0 tensor has numel 1", s.numel() == 1);
    check("a rank-0 tensor is contiguous", s.isContiguous());

    Tensorf v(Shape{7});
    check("a 1-D tensor has numel equal to its length", v.numel() == 7);
    check("a 1-D tensor is contiguous", v.isContiguous());

    Tensorf big(Shape{2, 3, 4, 5});
    check("a 4-D tensor computes numel", big.numel() == 120);
    check("a 4-D tensor is contiguous", big.isContiguous());
}


/* ========================================================================= //
    Step 1: default strides

    strides[n-1] = 1, then strides[d] = strides[d+1] * shape[d+1].
    Advancing index d by one skips a whole block of everything below it.
// ========================================================================= */

static void testDefaultStrides() {
    section("defaultStrides");

    check("rank 0 has no strides", Tensorf::defaultStrides({}) == Shape({}));
    check("rank 1 has stride 1", Tensorf::defaultStrides({7}) == Shape({1}));
    check("[3,4] gives (4,1)", Tensorf::defaultStrides({3, 4}) == Shape({4, 1}));
    check("[3,5] gives (5,1)", Tensorf::defaultStrides({3, 5}) == Shape({5, 1}));

    // the numpy-verified case from the walkthrough
    check("[2,3,4] gives (12,4,1)", Tensorf::defaultStrides({2, 3, 4}) == Shape({12, 4, 1}));
    check("[4,2,3] gives (6,3,1)", Tensorf::defaultStrides({4, 2, 3}) == Shape({6, 3, 1}));
    check("[2,3,4,5] gives (60,20,5,1)",
          Tensorf::defaultStrides({2, 3, 4, 5}) == Shape({60, 20, 5, 1}));

    // the attention head split: [seq,d_model] -> [seq,heads,d_head]
    check("[5,12] gives (12,1)", Tensorf::defaultStrides({5, 12}) == Shape({12, 1}));
    check("[5,3,4] gives (12,4,1)", Tensorf::defaultStrides({5, 3, 4}) == Shape({12, 4, 1}));

    // a size-1 dimension still gets a stride, it just never gets used
    check("[1,4] gives (4,1)", Tensorf::defaultStrides({1, 4}) == Shape({4, 1}));
    check("[4,1] gives (1,1)", Tensorf::defaultStrides({4, 1}) == Shape({1, 1}));

    // the constructor must use the same values
    Tensorf t(Shape{2, 3, 4});
    check("the constructor installs the default strides", t.strides == Shape({12, 4, 1}));
}


/* ========================================================================= //
    Step 1: offset

    offset is the dot product of the index vector with the stride vector.
    Everything else in Tensor is built on this being right.
// ========================================================================= */

static void testOffset() {
    section("offset");

    Tensorf t(Shape{3, 4});     // strides (4,1)

    check("offset of the first element is 0", t.offset({0, 0}) == 0);
    check("offset({1,2}) is 1*4 + 2 = 6", t.offset({1, 2}) == 6);
    check("offset({2,3}) is 2*4 + 3 = 11", t.offset({2, 3}) == 11);
    check("offset of the last element is numel-1", t.offset({2, 3}) == t.numel() - 1);

    Tensorf u(Shape{2, 3, 4});  // strides (12,4,1)
    check("3-D offset({0,0,0}) is 0", u.offset({0, 0, 0}) == 0);
    check("3-D offset({1,0,0}) is 12", u.offset({1, 0, 0}) == 12);
    check("3-D offset({0,1,0}) is 4", u.offset({0, 1, 0}) == 4);
    check("3-D offset({0,0,1}) is 1", u.offset({0, 0, 1}) == 1);
    check("3-D offset({1,2,3}) is 12+8+3 = 23", u.offset({1, 2, 3}) == 23);

    /*
        Every index must map to a distinct slot, and the whole set must cover
        0..numel-1 exactly once. This catches a swapped or duplicated stride
        that the handful of spot checks above would walk straight past.
    */
    check("offsets over a contiguous tensor are a permutation of 0..numel-1", [] {
        Tensorf x(Shape{2, 3, 4});
        std::vector<int> seen(x.numel(), 0);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k) {
                    size_t o = x.offset({i, j, k});
                    if (o >= x.numel()) return false;
                    ++seen[o];
                }
        for (int c : seen) if (c != 1) return false;
        return true;
    }());

    /*
        For a contiguous tensor, walking the indices in row-major order must
        produce offsets 0,1,2,... in sequence. If this passes, the layout
        genuinely matches what reshape and matmul will later assume.
    */
    check("row-major index order gives consecutive offsets", [] {
        Tensorf x(Shape{5, 3, 4});
        size_t expected = 0;
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    if (x.offset({i, j, k}) != expected++) return false;
        return true;
    }());

    // writing through data() and reading back through offset() must agree
    check("offset addresses the value written at that flat position", [] {
        Tensorf x(Shape{3, 4});
        for (size_t i = 0; i < x.numel(); ++i) x.data()[i] = static_cast<float>(i);
        return isEqual(x.data()[x.offset({1, 2})], 6.f)
            && isEqual(x.data()[x.offset({2, 3})], 11.f);
    }());
}


/* ========================================================================= //
    Step 1: data() and storage lifetime
// ========================================================================= */

static void testData() {
    section("data and storage");

    Tensorf t(Shape{2, 2});
    t.data()[0] = 1.f;
    t.data()[3] = 4.f;
    check("data() is writable", isEqual(t.data()[0], 1.f) && isEqual(t.data()[3], 4.f));

    const Tensorf& ct = t;
    check("data() const reads the same memory", isEqual(ct.data()[3], 4.f));

    // copying a Tensor shares the buffer -- this is intentional, it is what
    // makes views cheap, but it means a copy is NOT a deep copy
    Tensorf copy = t;
    check("a copy shares the same storage", copy.storage == t.storage);
    check("a copy sees writes through the original", [&] {
        t.data()[1] = 9.f;
        return isEqual(copy.data()[1], 9.f);
    }());

    // the buffer must outlive the tensor it was created from
    check("storage survives the original going out of scope", [] {
        Tensorf outer;
        {
            Tensorf inner(Shape{4});
            inner.data()[2] = 7.f;
            outer = inner;
        }
        return isEqual(outer.data()[2], 7.f);
    }());
}


/* ========================================================================= //
    Fixture plumbing

    Not a Tensor test -- it checks the harness itself, so that a later failure
    is unambiguously the tensor code and not a bad path or a stale fixture.
// ========================================================================= */

static void testFixtures() {
    section("fixtures");

    Fixture f = loadFixture("matmul_a");
    check("a fixture loads", f.ok);
    check("matmul_a has shape [3,4]", f.shape == Shape({3, 4}));
    check("matmul_a has 12 values", f.values.size() == 12);

    Fixture ids = loadFixture("gather_ids");
    check("gather_ids has shape [4]", ids.shape == Shape({4}));
    check("gather_ids holds 3 0 5 3", ids.values.size() == 4
          && isEqual(ids.values[0], 3.f) && isEqual(ids.values[1], 0.f)
          && isEqual(ids.values[2], 5.f) && isEqual(ids.values[3], 3.f));
}


/* ========================================================================= //
    Fixture -> Tensor helpers
// ========================================================================= */

static Tensorf loadTensor(const std::string& name) {
    Fixture f = loadFixture(name);
    if (!f.ok) return Tensorf(Shape{1});
    return ai::fromValues<float>(f.shape, f.values);
}

/* compares by SHAPE and by value in logical order, so a view holding the
   right numbers in the wrong layout still fails */
static bool matches(const Tensorf& t, const std::string& name, float tol = 1e-5f) {
    Fixture f = loadFixture(name);
    if (!f.ok) return false;
    if (t.shape != f.shape) return false;

    Tensorf c = t.contiguous();
    for (size_t i = 0; i < f.values.size(); ++i)
        if (!isEqual(c.data()[i], f.values[i], tol)) return false;
    return true;
}


/* ========================================================================= //
    Step 2: at() / operator()
// ========================================================================= */

static void testAt() {
    section("at / operator()");

    Tensorf t(Shape{3, 4});
    for (size_t i = 0; i < t.numel(); ++i) t.data()[i] = static_cast<float>(i);

    check("at({1,2}) reads what data()[6] holds", isEqual(t.at({1, 2}), 6.f));
    check("at({2,3}) reads the last element", isEqual(t.at({2, 3}), 11.f));
    check("operator()(1,2) agrees with at({1,2})", isEqual(t(1, 2), t.at({1, 2})));

    t.at({0, 1}) = 99.f;
    check("at() is writable", isEqual(t.data()[1], 99.f));
    t(2, 0) = 55.f;
    check("operator() is writable", isEqual(t.data()[8], 55.f));

    const Tensorf& ct = t;
    check("at() const reads the same memory", isEqual(ct.at({0, 1}), 99.f));
    check("operator() const reads the same memory", isEqual(ct(2, 0), 55.f));

    Tensorf u(Shape{2, 3, 4});
    for (size_t i = 0; i < u.numel(); ++i) u.data()[i] = static_cast<float>(i);
    check("3-D operator()(1,2,3) is element 23", isEqual(u(1, 2, 3), 23.f));
}


/* ========================================================================= //
    Step 3: permute / transpose
// ========================================================================= */

static void testPermute() {
    section("permute / transpose");

    Tensorf a(Shape{2, 3, 4});
    for (size_t i = 0; i < a.numel(); ++i) a.data()[i] = static_cast<float>(i);

    Tensorf p = a.permute({2, 0, 1});
    check("permute(2,0,1) of [2,3,4] has shape [4,2,3]", p.shape == Shape({4, 2, 3}));
    check("permute(2,0,1) of [2,3,4] has strides (1,12,4)", p.strides == Shape({1, 12, 4}));
    check("permute shares storage with the original", p.storage == a.storage);
    check("a permuted tensor reports non-contiguous", !p.isContiguous());

    /* new dim 0 is old dim 2, new dim 1 is old dim 0, new dim 2 is old dim 1,
       so p(x,y,z) must be a(y,z,x) */
    check("permuted.at(x,y,z) == original.at(y,z,x)", [&] {
        for (size_t x = 0; x < 4; ++x)
            for (size_t y = 0; y < 2; ++y)
                for (size_t z = 0; z < 3; ++z)
                    if (!isEqual(p(x, y, z), a(y, z, x))) return false;
        return true;
    }());

    check("permuting three times by (2,0,1) restores the original", [&] {
        Tensorf q = a.permute({2, 0, 1}).permute({2, 0, 1}).permute({2, 0, 1});
        if (q.shape != a.shape || q.strides != a.strides) return false;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    if (!isEqual(q(i, j, k), a(i, j, k))) return false;
        return true;
    }());

    Tensorf b(Shape{3, 5});
    for (size_t i = 0; i < b.numel(); ++i) b.data()[i] = static_cast<float>(i);
    Tensorf bt = b.transpose();
    check("transpose of [3,5] has shape [5,3]", bt.shape == Shape({5, 3}));
    check("transpose of [3,5] has strides (1,5)", bt.strides == Shape({1, 5}));
    check("transposed.at(i,j) == original.at(j,i)", [&] {
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = 0; j < 3; ++j)
                if (!isEqual(bt(i, j), b(j, i))) return false;
        return true;
    }());

    Tensorf c(Shape{2, 3, 4});
    check("transpose of [2,3,4] leaves the batch dim alone",
          c.transpose().shape == Shape({2, 4, 3}));
}


/* ========================================================================= //
    Step 4: contiguous
// ========================================================================= */

static void testContiguous() {
    section("contiguous");

    Tensorf px = loadTensor("permute_x");
    check("permute_x loads as [2,3,4]", px.shape == Shape({2, 3, 4}));

    check("contiguous() of permute(2,0,1) matches permute_201",
          matches(px.permute({2, 0, 1}).contiguous(), "permute_201"));
    check("contiguous() of permute(1,2,0) matches permute_120",
          matches(px.permute({1, 2, 0}).contiguous(), "permute_120"));

    Tensorf tx = loadTensor("transpose_x");
    check("contiguous() of a transposed tensor matches transpose_xt",
          matches(tx.transpose().contiguous(), "transpose_xt"));

    check("contiguous() reports isContiguous",
          px.permute({2, 0, 1}).contiguous().isContiguous());
    check("contiguous() of an already-packed tensor keeps the values",
          matches(px.contiguous(), "permute_x"));

    check("contiguous() of a permuted tensor does not alias the source", [&] {
        Tensorf p = px.permute({2, 0, 1});
        return p.contiguous().storage != p.storage;
    }());

    check("clone() always copies, even when already packed", [&] {
        return px.clone().storage != px.storage;
    }());
}


/* ========================================================================= //
    Step 5: reshape
// ========================================================================= */

static void testReshape() {
    section("reshape");

    Tensorf x(Shape{5, 12});
    for (size_t i = 0; i < x.numel(); ++i) x.data()[i] = static_cast<float>(i);

    Tensorf r = x.reshape({5, 3, 4});
    check("reshape [5,12] -> [5,3,4] keeps the shape", r.shape == Shape({5, 3, 4}));
    check("reshape recomputes strides to (12,4,1)", r.strides == Shape({12, 4, 1}));
    check("reshape shares storage", r.storage == x.storage);
    check("reshape keeps every value in flat order", [&] {
        for (size_t i = 0; i < x.numel(); ++i)
            if (!isEqual(r.data()[i], static_cast<float>(i))) return false;
        return true;
    }());

    check("reshape to a flat vector works", x.reshape({60}).shape == Shape({60}));
    check("reshape round-trips", x.reshape({60}).reshape({5, 12}).shape == Shape({5, 12}));

    /* the attention head split: reshape THEN permute. The other order would
       need a reshape of non-contiguous data, which is exactly what the
       assert in reshape exists to stop */
    Tensorf hs = loadTensor("heads_x");
    check("the head reshape [5,12] -> [3,5,4] matches heads_y",
          matches(hs.reshape({5, 3, 4}).permute({1, 0, 2}).contiguous(), "heads_y"));
}


/* ========================================================================= //
    Step 6: slice / row
// ========================================================================= */

static void testSlice() {
    section("slice / row");

    Tensorf x = loadTensor("slice_x");
    check("slice_x loads as [4,6]", x.shape == Shape({4, 6}));

    Tensorf rows = x.slice(0, 1, 3);
    check("slice along dim 0 gives shape [2,6]", rows.shape == Shape({2, 6}));
    check("slice along dim 0 matches slice_rows_1_3", matches(rows, "slice_rows_1_3"));
    check("a row slice is still contiguous", rows.isContiguous());
    check("slice shares storage", rows.storage == x.storage);
    check("slice moves ptr, not storage", rows.data() == x.data() + 6);

    Tensorf cols = x.slice(1, 2, 5);
    check("slice along dim 1 gives shape [4,3]", cols.shape == Shape({4, 3}));
    check("slice along dim 1 matches slice_cols_2_5", matches(cols, "slice_cols_2_5"));
    check("a column slice is NOT contiguous", !cols.isContiguous());
    check("a column slice keeps the original strides", cols.strides == x.strides);

    check("slicing the whole extent is a no-op", matches(x.slice(0, 0, 4), "slice_x"));

    Tensorf r2 = x.row(2);
    check("row(2) drops the leading dimension", r2.shape == Shape({6}));
    check("row(2) reads row 2", [&] {
        for (size_t j = 0; j < 6; ++j)
            if (!isEqual(r2(j), x(2, j))) return false;
        return true;
    }());
}


/* ========================================================================= //
    Step 7: gather
// ========================================================================= */

static void testGather() {
    section("gather");

    Tensorf table = loadTensor("gather_table");
    Fixture ids = loadFixture("gather_ids");

    std::vector<size_t> idx;
    for (float v : ids.values) idx.push_back(static_cast<size_t>(v));

    Tensorf g = ai::gather(table, idx);
    check("gather gives shape [4,4]", g.shape == Shape({4, 4}));
    check("gather picks rows 3,0,5,3 of the table", matches(g, "gather_y"));

    check("gather handles a repeated index", [&] {
        for (size_t j = 0; j < 4; ++j)
            if (!isEqual(g(0, j), g(3, j))) return false;
        return true;
    }());
}


/* ========================================================================= //
    Step 8-9: matmul
// ========================================================================= */

static void testMatmul() {
    section("matmul");

    Tensorf a = loadTensor("matmul_a");
    Tensorf b = loadTensor("matmul_b");
    Tensorf c = ai::matmul(a, b);

    check("matmul [3,4] @ [4,5] gives [3,5]", c.shape == Shape({3, 5}));
    check("matmul_a @ matmul_b == matmul_c", matches(c, "matmul_c"));

    /* fed a transposed operand: [3,5]^T @ [3,2] -> [5,2]. This is where a
       matmul that assumes contiguity without checking goes wrong */
    Tensorf tx = loadTensor("transpose_x");
    Tensorf pb = loadTensor("matmul_perm_b");
    Tensorf pc = ai::matmul(tx.transpose(), pb);
    check("matmul of a transposed operand gives [5,2]", pc.shape == Shape({5, 2}));
    check("matmul of a transposed operand == matmul_perm_c", matches(pc, "matmul_perm_c"));

    check("multiplying by the identity is a no-op", [&] {
        Tensorf id(Shape{4, 4});
        for (size_t i = 0; i < 4; ++i) id(i, i) = 1.f;
        Tensorf r = ai::matmul(a, id);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                if (!isEqual(r(i, j), a(i, j))) return false;
        return true;
    }());

    // output must start zeroed, since the inner loop accumulates into it
    check("matmul does not accumulate into stale output", [&] {
        Tensorf r1 = ai::matmul(a, b);
        Tensorf r2 = ai::matmul(a, b);
        for (size_t i = 0; i < r1.numel(); ++i)
            if (!isEqual(r1.data()[i], r2.data()[i])) return false;
        return true;
    }());
}


static void testBatchedMatmul() {
    section("batched matmul");

    Tensorf a = loadTensor("bmm_a");
    Tensorf b = loadTensor("bmm_b");
    Tensorf c = ai::matmul(a, b);

    check("batched matmul [2,3,4] @ [2,4,3] gives [2,3,3]", c.shape == Shape({2, 3, 3}));
    check("bmm_a @ bmm_b == bmm_c", matches(c, "bmm_c"));

    /* each batch is an independent matmul -- a bug that ignores the batch
       stride still produces the right shape */
    check("batched matmul keeps the batches independent", [&] {
        Tensorf a0 = a.slice(0, 0, 1).contiguous().reshape({3, 4});
        Tensorf b0 = b.slice(0, 0, 1).contiguous().reshape({4, 3});
        Tensorf c0 = ai::matmul(a0, b0);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                if (!isEqual(c0(i, j), c(0, i, j))) return false;
        return true;
    }());

    // the attention product: Q [H,S,D] @ K^T [H,D,S] -> [H,S,S]
    check("the attention product shape works out", [] {
        Tensorf q(Shape{3, 5, 4});
        Tensorf k(Shape{3, 5, 4});
        return ai::matmul(q, k.transpose()).shape == Shape({3, 5, 5});
    }());
}


/* ========================================================================= //
    Step 10: elementwise
// ========================================================================= */

static void testElementwise() {
    section("elementwise");

    check("addBias broadcasts [4] over [3,4]",
          matches(ai::addBias(loadTensor("addbias_x"), loadTensor("addbias_b")), "addbias_y"));

    check("gelu matches gelu_y across [-4,4]",
          matches(ai::gelu(loadTensor("gelu_x")), "gelu_y"));

    check("softmax over the last axis matches softmax_y",
          matches(ai::softmaxLastAxis(loadTensor("softmax_x")), "softmax_y"));

    /* that row holds a 900: without subtracting the row max, exp() overflows
       to inf and the whole row becomes nan */
    Tensorf ov = ai::softmaxLastAxis(loadTensor("softmax_overflow_x"));
    check("softmax subtracts the row max (no NaN on a huge logit)", [&] {
        for (size_t i = 0; i < ov.numel(); ++i)
            if (!std::isfinite(ov.data()[i])) return false;
        return true;
    }());
    check("softmax of a huge logit matches softmax_overflow_y",
          matches(ov, "softmax_overflow_y"));

    check("softmax rows sum to 1", [] {
        Tensorf s = ai::softmaxLastAxis(loadTensor("softmax_x"));
        for (size_t r = 0; r < s.shape[0]; ++r) {
            float sum = 0.f;
            for (size_t j = 0; j < s.shape[1]; ++j) sum += s(r, j);
            if (!isEqual(sum, 1.f)) return false;
        }
        return true;
    }());

    check("layernorm matches layernorm_y",
          matches(ai::layernorm(loadTensor("layernorm_x"),
                                loadTensor("layernorm_g"),
                                loadTensor("layernorm_b")), "layernorm_y"));

    /* unit weight and zero bias isolates the normalisation itself, and this
       is the test that catches an unbiased (n-1) variance */
    check("layernorm uses the biased variance", [] {
        Tensorf g(Shape{8}); g.fill(1.f);
        Tensorf b(Shape{8});
        return matches(ai::layernorm(loadTensor("layernorm_x"), g, b), "layernorm_plain_y");
    }());

    check("layernorm rows have zero mean and unit variance", [] {
        Tensorf g(Shape{8}); g.fill(1.f);
        Tensorf b(Shape{8});
        Tensorf y = ai::layernorm(loadTensor("layernorm_x"), g, b);
        for (size_t r = 0; r < y.shape[0]; ++r) {
            float mean = 0.f;
            for (size_t j = 0; j < 8; ++j) mean += y(r, j);
            mean /= 8.f;
            float var = 0.f;
            for (size_t j = 0; j < 8; ++j) { float d = y(r, j) - mean; var += d * d; }
            var /= 8.f;
            if (!isEqual(mean, 0.f, 1e-4f) || !isEqual(var, 1.f, 1e-3f)) return false;
        }
        return true;
    }());

    check("causalMask matches causal_masked",
          matches(ai::causalMask(loadTensor("causal_x")), "causal_masked", 1e-2f));

    check("causalMask then softmax matches causal_y",
          matches(ai::softmaxLastAxis(ai::causalMask(loadTensor("causal_x"))), "causal_y"));

    /* a token must not attend to anything ahead of it, so everything above
       the diagonal has to come out as exactly zero weight */
    check("causal attention weights are lower-triangular", [] {
        Tensorf w = ai::softmaxLastAxis(ai::causalMask(loadTensor("causal_x")));
        for (size_t m = 0; m < 2; ++m)
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = i + 1; j < 4; ++j)
                    if (!isEqual(w(m, i, j), 0.f)) return false;
        return true;
    }());

    check("add is elementwise over the same shape", [] {
        Tensorf a = loadTensor("addbias_x");
        Tensorf s = ai::add(a, a);
        for (size_t i = 0; i < a.numel(); ++i)
            if (!isEqual(s.data()[i], a.data()[i] * 2.f)) return false;
        return true;
    }());

    check("split into thirds matches split_q/k/v", [] {
        std::vector<Tensorf> parts = ai::split(loadTensor("split_x"), 3, 1);
        return parts.size() == 3
            && matches(parts[0], "split_q")
            && matches(parts[1], "split_k")
            && matches(parts[2], "split_v");
    }());

    check("split returns views, not copies", [] {
        Tensorf x = loadTensor("split_x");
        return ai::split(x, 3, 1)[0].storage == x.storage;
    }());
}


int main(int argc, char** argv) {
    if (argc > 1) fixtureDir = argv[1];

    testConstruction();
    testDefaultStrides();
    testOffset();
    testData();
    testFixtures();
    testAt();
    testPermute();
    testContiguous();
    testReshape();
    testSlice();
    testGather();
    testMatmul();
    testBatchedMatmul();
    testElementwise();

    std::cout << "\n"
              << (failures ? COLOR_RED : COLOR_GREEN)
              << (failures ? std::to_string(failures) + " FAILURE(S)" : "ALL PASS")
              << COLOR_RESET << std::endl;

    return failures;
}
