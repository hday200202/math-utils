"""
Generate ground-truth fixtures for the Tensor<T> unit tests.

Hand-computed expected values are dangerous: if you work out a matmul by hand
using the same wrong mental model that produced the bug, the test agrees with
the code and both are wrong. These come from numpy instead.

Format is deliberately dumb so the C++ side needs no JSON parser yet -- one
text file per tensor, readable with istringstream >> :

    <ndim> <dim0> <dim1> ...
    <values, whitespace separated, row-major/C-contiguous>

Integer tensors (token ids) use the same layout with integral values.

    py tools/make_fixtures.py            writes out/fixtures/
"""

import os

import numpy as np

OUT = os.path.join("out", "fixtures")
rng = np.random.default_rng(42)

written = []


def emit(name, arr):
    a = np.ascontiguousarray(arr)
    is_int = np.issubdtype(a.dtype, np.integer)
    a32 = a if is_int else a.astype(np.float32)

    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        f.write(str(a32.ndim) + " " + " ".join(str(d) for d in a32.shape) + "\n")
        flat = a32.reshape(-1)
        if is_int:
            f.write(" ".join(str(int(v)) for v in flat))
        else:
            # 9 significant digits round-trips a float32 exactly
            f.write(" ".join("%.9g" % v for v in flat))
        f.write("\n")

    written.append(name)
    return a32


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


# --------------------------------------------------------------------------
#   reference ops -- these must match the oracle in gpt2_oracle.py exactly
# --------------------------------------------------------------------------

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # biased: /n, not /(n-1)
    return (x - mean) / np.sqrt(var + eps) * g + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- matmul [M,K] @ [K,N] --------------------------------------------
    # non-square and M != K != N on purpose: a transposed-index bug in a
    # square matmul still produces correctly shaped output and hides
    a = emit("matmul_a", rand(3, 4))
    b = emit("matmul_b", rand(4, 5))
    emit("matmul_c", a @ b)

    # ---- batched matmul [H,S,D] @ [H,D,S] --------------------------------
    # the shape attention needs. Different H slices hold different data, so
    # a bug that ignores the batch stride shows up immediately
    ba = emit("bmm_a", rand(2, 3, 4))
    bb = emit("bmm_b", rand(2, 4, 3))
    emit("bmm_c", ba @ bb)

    # ---- permute ---------------------------------------------------------
    # all three dims distinct so a wrong axis order cannot accidentally match
    px = emit("permute_x", rand(2, 3, 4))
    emit("permute_201", np.ascontiguousarray(px.transpose(2, 0, 1)))
    emit("permute_120", np.ascontiguousarray(px.transpose(1, 2, 0)))

    # ---- transpose of a 2D view ------------------------------------------
    tx = emit("transpose_x", rand(3, 5))
    emit("transpose_xt", np.ascontiguousarray(tx.T))

    # ---- matmul on a PERMUTED operand ------------------------------------
    # catches the case where matmul assumes contiguity but is handed a view.
    # transpose_x is [3,5], so its transpose [5,3] times [3,2] gives [5,2]
    mb = emit("matmul_perm_b", rand(3, 2))
    emit("matmul_perm_c", np.ascontiguousarray(tx.T) @ mb)

    # ---- softmax over the last axis --------------------------------------
    sx = emit("softmax_x", rand(3, 5) * 3.0)
    emit("softmax_y", softmax(sx))

    # a row with a huge value: without max-subtraction exp() overflows to inf
    ox = np.array([[1.0, 2.0, 900.0, 3.0]], dtype=np.float32)
    emit("softmax_overflow_x", ox)
    emit("softmax_overflow_y", softmax(ox))

    # ---- layernorm -------------------------------------------------------
    lx = emit("layernorm_x", rand(3, 8) * 2.0 + 1.0)
    lg = emit("layernorm_g", rand(8))
    lb = emit("layernorm_b", rand(8))
    emit("layernorm_y", layernorm(lx, lg, lb))

    # unit weight / zero bias isolates the normalisation itself
    emit("layernorm_plain_y",
         layernorm(lx, np.ones(8, np.float32), np.zeros(8, np.float32)))

    # ---- gelu ------------------------------------------------------------
    gx = emit("gelu_x", np.linspace(-4, 4, 17).astype(np.float32))
    emit("gelu_y", gelu_new(gx))

    # ---- causal mask + softmax -------------------------------------------
    cx = emit("causal_x", rand(2, 4, 4))
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    cm = np.where(mask, np.float32(-1e10), cx)
    emit("causal_masked", cm)
    emit("causal_y", softmax(cm))

    # ---- bias broadcast --------------------------------------------------
    abx = emit("addbias_x", rand(3, 4))
    abb = emit("addbias_b", rand(4))
    emit("addbias_y", abx + abb)

    # ---- slice along dim 0 -----------------------------------------------
    slx = emit("slice_x", rand(4, 6))
    emit("slice_rows_1_3", np.ascontiguousarray(slx[1:3]))

    # ---- slice along dim 1 (non-contiguous result) -----------------------
    emit("slice_cols_2_5", np.ascontiguousarray(slx[:, 2:5]))

    # ---- row gather (embedding lookup) -----------------------------------
    tbl = emit("gather_table", rand(6, 4))
    ids = emit("gather_ids", np.array([3, 0, 5, 3], dtype=np.int64))
    emit("gather_y", tbl[ids])

    # ---- split into 3 along the last axis (the qkv split) ----------------
    qkv = emit("split_x", rand(2, 9))
    q, k, v = np.split(qkv, 3, axis=-1)
    emit("split_q", np.ascontiguousarray(q))
    emit("split_k", np.ascontiguousarray(k))
    emit("split_v", np.ascontiguousarray(v))

    # ---- the head reshape: [S,D] -> [S,H,dh] -> [H,S,dh] -----------------
    # exactly what attention does; a wrong reshape here is the most common
    # attention bug and it still produces plausible-looking output
    hs = emit("heads_x", rand(5, 12))                    # seq 5, d_model 12
    heads = hs.reshape(5, 3, 4).transpose(1, 0, 2)       # 3 heads of width 4
    emit("heads_y", np.ascontiguousarray(heads))

    with open(os.path.join(OUT, "index.txt"), "w") as f:
        f.write("\n".join(written) + "\n")

    print("wrote " + str(len(written)) + " fixtures to " + OUT)


if __name__ == "__main__":
    main()
