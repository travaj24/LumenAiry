"""H4 BOR-pencil accuracy probe.

For every parametrisation of ``test_h4_bor_pencil_eigh_accuracy`` this reports

  * the measured relative error against the Bessel-zero oracle,
  * the per-mode relative errors (which mode carries the max),
  * an INDEPENDENT predicted SCALE for that error from the actual assembled
    matrices: the LAPACK bound for a symmetric-definite pencil reduced by
    Cholesky, |lam_hat - lam| <~ p(n) eps ||A||_2 ||M^-1||_2, turned into a
    relative error on sqrt(lam) by dividing by 2*lam_i,
  * a refinement reading (degree 12 / 24 elements) that separates the SEM
    TRUNCATION error from the arithmetic: if the error does not fall, the
    reading is arithmetic-dominated and is a BLAS/LAPACK-build quantity.

Usage:  python probe_h4_pencil.py <out.json>
"""
import json
import os
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jn_zeros, jnp_zeros

import lumenairy
from lumenairy.elements.bor.radial_eigensolver import (
    _gll_nodes_weights, _graded_boundaries, _lagrange_vals_derivs,
    radial_spectrum)

assert "lum_reds" in lumenairy.__file__, lumenairy.__file__

CASES = [(0, "dirichlet", "jn"), (1, "dirichlet", "jn"), (3, "dirichlet", "jn"),
         (1, "neumann", "jnp"), (3, "neumann", "jnp")]


def assemble(m, R, degree, n_el, bc):
    """The SAME weak form radial_spectrum assembles -- reproduced here so the
    matrix norms below are read off the actual pencil."""
    ref_nodes, _w = _gll_nodes_weights(degree)
    p = degree + 1
    bnds = _graded_boundaries(0.0, R, n_el)
    n_el_a = len(bnds) - 1
    l2g = np.zeros((n_el_a, p), dtype=int)
    gid = 0
    for e in range(n_el_a):
        for a in range(p):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, p - 1]
            else:
                l2g[e, a] = gid
                gid += 1
    n_glob = gid
    xq, wq = leggauss(degree + 10)
    V, D = _lagrange_vals_derivs(ref_nodes, xq)
    A = np.zeros((n_glob, n_glob))
    M = np.zeros((n_glob, n_glob))
    for e in range(n_el_a):
        xl, xr = bnds[e], bnds[e + 1]
        J = 0.5 * (xr - xl)
        rq = 0.5 * (xr + xl) + J * xq
        wp = wq * J
        Dp = D / J
        idx = l2g[e]
        for a in range(p):
            for b in range(p):
                A[idx[a], idx[b]] += (np.sum(wp * rq * Dp[:, a] * Dp[:, b])
                                      + m * m * np.sum(wp * (1.0 / rq)
                                                       * V[:, a] * V[:, b]))
                M[idx[a], idx[b]] += np.sum(wp * rq * V[:, a] * V[:, b])
    drop = set()
    if bc == "dirichlet":
        drop.add(n_glob - 1)
    if m != 0:
        drop.add(0)
    keep = np.array([i for i in range(n_glob) if i not in drop])
    return A[np.ix_(keep, keep)], M[np.ix_(keep, keep)]


out = {"python": sys.version.split()[0], "numpy": np.__version__,
       "lumenairy_file": lumenairy.__file__,
       "env": {k: os.environ.get(k) for k in
               ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS")},
       "cases": {}}
try:
    import threadpoolctl
    out["threadpool"] = threadpoolctl.threadpool_info()
except Exception as exc:                                   # pragma: no cover
    out["threadpool"] = str(exc)
try:
    import scipy
    out["scipy"] = scipy.__version__
except Exception:
    pass

eps = float(np.finfo(float).eps)
for m, bc, zf in CASES:
    ref = (jn_zeros(m, 6) if zf == "jn" else jnp_zeros(m, 6))
    ev = np.asarray(radial_spectrum(m, 1.0, 8, 12, bc=bc, n_low=6))
    per = np.abs(np.sqrt(np.abs(ev)) / ref - 1.0)
    rel = float(np.max(per))

    Ak, Mk = assemble(m, 1.0, 8, 12, bc)
    n = Ak.shape[0]
    nA = float(np.linalg.norm(Ak, 2))
    nM = float(np.linalg.norm(Mk, 2))
    nMinv = float(np.linalg.norm(np.linalg.inv(Mk), 2))
    kM = nM * nMinv
    lam_all = np.linalg.eigvalsh(np.linalg.solve(Mk, Ak)
                                 if False else Ak)          # unused branch
    # full pencil spectrum for lam_max, via the same Cholesky reduction
    L = np.linalg.cholesky(Mk)
    Li = np.linalg.inv(L)
    C = Li @ Ak @ Li.T
    lam_pencil = np.linalg.eigvalsh(0.5 * (C + C.T))
    lam_max = float(lam_pencil[-1])

    # LAPACK symmetric-definite bound: |dlam| <~ n eps ||A||_2 ||M^-1||_2
    abs_bound = n * eps * nA * nMinv
    # ... and the weaker/tighter form scaled by the pencil's own top eigenvalue
    abs_bound_lmax = n * eps * lam_max
    pred_rel = float(np.max(abs_bound / (2.0 * np.abs(ev))))
    pred_rel_lmax = float(np.max(abs_bound_lmax / (2.0 * np.abs(ev))))

    ev_fine = np.asarray(radial_spectrum(m, 1.0, 12, 24, bc=bc, n_low=6))
    rel_fine = float(np.max(np.abs(np.sqrt(np.abs(ev_fine)) / ref - 1.0)))

    out["cases"][f"{m}-{bc}-{zf}"] = dict(
        rel=rel, per_mode=[float(v) for v in per],
        argmax_mode=int(np.argmax(per)),
        ev=[float(v) for v in ev], ref=[float(v) for v in ref],
        n_dof=int(n), normA=nA, normM=nM, normMinv=nMinv, condM=float(kM),
        lam_max=lam_max, lam_min=float(lam_pencil[0]),
        cond_pencil=float(lam_max / lam_pencil[0]),
        abs_bound=float(abs_bound), abs_bound_lmax=float(abs_bound_lmax),
        pred_rel=pred_rel, pred_rel_lmax=pred_rel_lmax,
        ratio_pred_over_meas=float(pred_rel / rel) if rel else None,
        rel_refined_deg12_el24=rel_fine,
    )
    del lam_all

json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out["cases"].items():
    print(f"{k:22s} rel={v['rel']:.3e} argmax={v['argmax_mode']} "
          f"pred={v['pred_rel']:.3e} pred_lmax={v['pred_rel_lmax']:.3e} "
          f"refined={v['rel_refined_deg12_el24']:.3e} n={v['n_dof']} "
          f"condM={v['condM']:.3e} lam_max={v['lam_max']:.4e}")
