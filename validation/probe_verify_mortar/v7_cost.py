"""V7 -- MEMORY and COST.

``python v7_cost.py [xmass device]``

* ``xmass``  the factored vs dense cross-mass: memory ratio and apply speed
  across grid pairs (the build claims 900x memory at ``N = (6, 12), M = 6``
  and that the FACTORED apply is SLOWER below ~100x100).
* ``device`` per-layer vs shared wall time and peak RSS on the F2 device pair
  (corner-dominated 2-D pillars) at matched DOF.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc
import json
import sys
import time
import warnings

import numpy as np
import psutil

import lumenairy
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Basis1D,
    _stag_cross_mass_1d,
    _stag_kron_apply,
)

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
P = 1.2
WL = 0.85
EPS_P, EPS_H = 6.0, 2.25
PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / 1024 ** 2


def sec_xmass():
    out = {}
    rng = np.random.default_rng(7)
    for (Na, Nb), M in (((2, 3), 6), ((3, 6), 6), ((4, 6), 8), ((6, 12), 6),
                        ((8, 16), 6)):
        ba, bb = Basis1D(P, Na, M, 1.0 + 0j), Basis1D(P, Nb, M, 1.0 + 0j)
        Cx = _stag_cross_mass_1d(ba, bb, "B")
        Cy = _stag_cross_mass_1d(ba, bb, "Btilde")
        dense_bytes = Cx.shape[0] * Cy.shape[0] * Cx.shape[1] * Cy.shape[1] * 16
        fact_bytes = Cx.nbytes + Cy.nbytes
        X = (rng.standard_normal((Cy.shape[1] * Cx.shape[1], 3))
             + 1j * rng.standard_normal((Cy.shape[1] * Cx.shape[1], 3)))
        D = np.kron(Cy, Cx)
        Y2 = D @ X
        Y1 = _stag_kron_apply(Cy, Cx, X)
        rel = float(np.max(np.abs(Y1 - Y2))) / float(np.max(np.abs(Y2)))
        nrep = max(3, int(2e7 / max(D.size, 1)))
        t0 = time.perf_counter()
        for _ in range(nrep):
            _stag_kron_apply(Cy, Cx, X)
        tf = (time.perf_counter() - t0) / nrep
        t0 = time.perf_counter()
        for _ in range(nrep):
            D @ X
        td = (time.perf_counter() - t0) / nrep
        out[f"N{Na}x{Nb}_M{M}"] = {
            "dense_MB": dense_bytes / 1024 ** 2,
            "factors_MB": fact_bytes / 1024 ** 2,
            "memory_ratio": dense_bytes / fact_bytes,
            "dense_dim": int(D.shape[0]),
            "apply_speedup_factored_over_dense": td / tf,
            "identity_rel": rel, "nrep": nrep}
        r = out[f"N{Na}x{Nb}_M{M}"]
        print(f"[xmass] ({Na},{Nb}) M={M}: dense {r['dense_MB']:.3f} MB "
              f"({D.shape[0]}x{D.shape[1]}) vs factors "
              f"{r['factors_MB']:.4f} MB -> {r['memory_ratio']:.0f}x memory; "
              f"apply {r['apply_speedup_factored_over_dense']:.2f}x; identity "
              f"{rel:.2e}", flush=True)
        del D
        gc.collect()
    RES["xmass"] = out


def _pil(N, lo, hi):
    c = np.full((N, N), EPS_H + 0j)
    c[lo:hi, lo:hi] = EPS_P
    return c


def sec_device():
    """The F2 corner-dominated pillar pair at MATCHED DOF: wall time and peak
    RSS, both arms, measured one at a time in a warm process."""
    out = {}
    cA6 = np.repeat(np.repeat(_pil(2, 0, 1), 3, 0), 3, 1)
    cB6 = np.repeat(np.repeat(_pil(3, 1, 2), 2, 0), 2, 1)
    th, ph = 0.18, 0.35

    def timed(fn):
        gc.collect()
        base = rss_mb()
        peak = base
        t0 = time.perf_counter()
        r = fn()
        w = time.perf_counter() - t0
        peak = max(peak, rss_mb())
        return w, base, peak, r

    for Mu in (3, 4, 5):
        q = 6 * (Mu - 1)
        MA, MB = 3 * Mu - 2, 2 * Mu - 1

        def union():
            st = PMM2DStackPure(P, n_modes=Mu, n_orders=1)
            st.add_layer(0.30, eps_cell=cA6)
            st.add_layer(0.22, eps_cell=cB6)
            st.set_source(WL, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return st.solve(jones=False)

        def per():
            st = PMM2DStackPure(P, n_modes=max(MA, MB), n_orders=1,
                                layer_grids="per-layer")
            st.add_layer(0.30, eps_cell=_pil(2, 0, 1), n_modes=MA)
            st.add_layer(0.22, eps_cell=_pil(3, 1, 2), n_modes=MB)
            st.set_source(WL, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return st.solve(jones=False)

        wu, bu, pu, _ = timed(union)
        wp, bp, pp, _ = timed(per)
        out[f"q{q}"] = {"Mu": Mu, "MA": MA, "MB": MB, "eig_dim": 2 * q * q,
                        "wall_union": wu, "wall_perlayer": wp,
                        "wall_ratio": wp / wu,
                        "rss_base_union_MB": bu, "rss_peak_union_MB": pu,
                        "rss_base_perlayer_MB": bp,
                        "rss_peak_perlayer_MB": pp,
                        "delta_rss_union_MB": pu - bu,
                        "delta_rss_perlayer_MB": pp - bp}
        print(f"[device] q={q} eig dim {2*q*q}: union {wu:.2f}s "
              f"(RSS {bu:.0f} -> {pu:.0f} MB) | per-layer {wp:.2f}s "
              f"(RSS {bp:.0f} -> {pp:.0f} MB)  ratio {wp/wu:.2f}x",
              flush=True)
    # the STAIRCASE cost claim: eig work at production modal counts
    claim = {}
    for label, Ns, M in (("union N=6 vs per-layer 2/3/6", (2, 3, 6), 8),
                         ("sibling union N=12 vs 2/3/4", (2, 3, 4), 8)):
        Nu = int(np.lcm.reduce(Ns))
        q_u = Nu * (M - 1)
        dim_u = 2 * q_u ** 2
        dims = [2 * (N * (M - 1)) ** 2 for N in Ns]
        claim[label] = {
            "N_union": Nu, "q_union": q_u, "eig_dim_union": dim_u,
            "eig_dims_perlayer": dims,
            "eig_work_ratio_cubed": (dim_u ** 3) / sum(d ** 3 for d in dims),
            "union_matrix_GB": 16 * dim_u ** 2 / 1024 ** 3,
            "perlayer_matrix_GB": 16 * max(dims) ** 2 / 1024 ** 3}
        c = claim[label]
        print(f"[device] {label} at M={M}: union N={Nu} dim {dim_u} "
              f"({c['union_matrix_GB']:.2f} GB) vs per-layer dims {dims} "
              f"({c['perlayer_matrix_GB']:.3f} GB) -> eig-work ratio "
              f"{c['eig_work_ratio_cubed']:.0f}x", flush=True)
    out["staircase_arithmetic"] = claim
    RES["device"] = out


SECTIONS = {"xmass": sec_xmass, "device": sec_device}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v7_cost.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            old = {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
