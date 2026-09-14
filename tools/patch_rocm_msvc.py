"""
Make ROCm's HIP headers compile against MSVC 14.51 (Visual Studio 18).

THE PROBLEM

An EMPTY .hip file does not compile on this combination. It has nothing to do
with any source in this repository:

    __clang_hip_cmath.h:109: error: __device__ function 'isgreater' cannot
    overload __host__ __device__ function 'isgreater'

MSVC's <cmath> grew a block guarded by a bare `#ifdef __clang__` that declares
isgreater, isgreaterequal, isless, islessequal, islessgreater and isunordered
as `constexpr`, to give clang users the efficient builtins (microsoft/STL
issue 4609). In HIP a constexpr function is implicitly __host__ __device__.
ROCm then declares the same six as plain __device__ - in two headers - and a
__device__ function cannot overload a __host__ __device__ one.

Exactly those six and nothing else. The neighbouring isfinite/isinf/isnan are
declared by MSVC with _CONSTEXPR23, which is `inline` rather than `constexpr`
under -std=c++17, so they are never implicitly __host__ __device__ and never
collide. That is why the guards below are so narrow.

Confirmed on ROCm 7.2.1 (clang 22) and the June 2026 nightly (clang 23), so
it is not a version that will pass on its own.

THE FIX

Skip ROCm's six declarations when MSVC's STL has already supplied them. The
MSVC versions are constexpr __builtin_* calls, which compile for amdgcn
perfectly well, so device code keeps working - and src/gpu/Kernels.hip calls
none of the six anyway. It uses no math library at all, only intrinsics
(__shfl_down, __uint_as_float, __syncthreads).

THIS IS A PATCH TO A VENDOR HEADER, which means:

  - any `pip install --upgrade` of the ROCm SDK undoes it. Re-run this.
  - it is idempotent; it looks for its own marker before touching anything.
  - `--revert` restores from the .orig copy it keeps beside each header.

The real fix is on AMD's side, or an older MSVC toolset. This exists so the
GPU backend can be built today without one.

    py tools/patch_rocm_msvc.py            apply
    py tools/patch_rocm_msvc.py --check    report only
    py tools/patch_rocm_msvc.py --revert   undo
"""

import argparse
import glob
import os
import shutil
import site
import sys

MARKER = "util-libraries: guarded for MSVC STL"

NOTE = """/* %s
   MSVC's <cmath> declares these as constexpr, which HIP makes implicitly
   __host__ __device__; the __device__ declarations below then cannot
   overload them. MSVC's are __builtin_* calls and compile for amdgcn, so
   skipping ROCm's loses nothing. See tools/patch_rocm_msvc.py. */
""" % MARKER

GUARD_OPEN = NOTE + "#if !defined(_MSVC_STL_VERSION)\n"
GUARD_CLOSE = "#endif // !defined(_MSVC_STL_VERSION) - %s\n" % MARKER

"""
The exact runs to wrap, per header. They are runs rather than one block
because ROCm interleaves them with isfinite/isinf/isnan, which must stay.
"""
BLOCKS = {
    "__clang_hip_cmath.h": ["""__DEVICE__ __CONSTEXPR__ bool isgreater(float __x, float __y) {
  return __builtin_isgreater(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool isgreater(double __x, double __y) {
  return __builtin_isgreater(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool isgreaterequal(float __x, float __y) {
  return __builtin_isgreaterequal(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool isgreaterequal(double __x, double __y) {
  return __builtin_isgreaterequal(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool isless(float __x, float __y) {
  return __builtin_isless(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool isless(double __x, double __y) {
  return __builtin_isless(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool islessequal(float __x, float __y) {
  return __builtin_islessequal(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool islessequal(double __x, double __y) {
  return __builtin_islessequal(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool islessgreater(float __x, float __y) {
  return __builtin_islessgreater(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool islessgreater(double __x, double __y) {
  return __builtin_islessgreater(__x, __y);
}
""",
        # isnormal sits between these two runs and must NOT be guarded: it
        # is a BUILTIN1 on the MSVC side, declared _CONSTEXPR23, which is
        # `inline` under -std=c++17 and so never collides
        """__DEVICE__ __CONSTEXPR__ bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
__DEVICE__ __CONSTEXPR__ bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
"""],

    "__clang_cuda_math_forward_declares.h": [
        """__DEVICE__ bool isgreater(double, double);
__DEVICE__ bool isgreaterequal(double, double);
__DEVICE__ bool isgreaterequal(float, float);
__DEVICE__ bool isgreater(float, float);
""",
        """__DEVICE__ bool isless(double, double);
__DEVICE__ bool islessequal(double, double);
__DEVICE__ bool islessequal(float, float);
__DEVICE__ bool isless(float, float);
__DEVICE__ bool islessgreater(double, double);
__DEVICE__ bool islessgreater(float, float);
""",
        """__DEVICE__ bool isunordered(double, double);
__DEVICE__ bool isunordered(float, float);
""",
    ],
}


def headers():
    """Every affected header in every installed ROCm SDK."""
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    found = []
    for root in roots:
        for name in BLOCKS:
            found.extend(glob.glob(os.path.join(
                root, "_rocm_sdk_core", "lib", "llvm", "lib", "clang", "*",
                "include", name)))
    return sorted(set(found))


def patch(path, revert=False, check=False):
    text = open(path, newline="").read()
    backup = path + ".orig"

    """
    Match the FILE's line endings, not this script's. The headers ship with
    LF; if this file was checked out with CRLF, every anchor misses silently
    and the patch reports success having changed nothing.
    """
    crlf, lf = chr(13) + chr(10), chr(10)
    native = crlf if crlf in text else lf
    fit = lambda t: t.replace(crlf, lf).replace(lf, native)

    if revert:
        if not os.path.exists(backup):
            return "no backup"
        shutil.copyfile(backup, path)
        os.remove(backup)
        return "reverted"

    if MARKER in text:
        return "already ok"
    if check:
        return "NEEDS PATCH"

    blocks = [fit(b) for b in BLOCKS[os.path.basename(path)]]
    missing = [b for b in blocks if b not in text]
    if missing:
        return "SKIPPED: %d/%d anchors not found - header differs from expected" \
               % (len(missing), len(blocks))

    if not os.path.exists(backup):
        shutil.copyfile(path, backup)

    for b in blocks:
        text = text.replace(b, fit(GUARD_OPEN) + b + fit(GUARD_CLOSE), 1)

    open(path, "w", newline="").write(text)
    return "patched %d block(s)" % len(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    found = headers()
    if not found:
        print("no ROCm SDK found in site-packages")
        print("  pip install --user --index-url "
              "https://rocm.nightlies.amd.com/v2/gfx1150/ rocm[devel]")
        return 1

    for path in found:
        print("%-28s %s" % (patch(path, args.revert, args.check),
                            os.path.basename(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
