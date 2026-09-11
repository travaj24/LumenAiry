"""E-SLIVER -- class C (mortar / sliver) ladders.  MEASUREMENT ONLY.

C1  1-D union grid (``PMMStack``, ``lumenairy/elements/pmm/stack.py``; walls
    unioned across layers by ``lumenairy/elements/pmm/_core.py:4467
    _pmm_union_grid``).  Ladder: two layers whose walls differ by ``delta`` of
    the period.  ALREADY GUARDED (``PMM_SLIVER_GUARD``, shipped state), so the
    question is WHICH arm fires and at which ``delta``.

C2  per-layer 2-D grids.
    * ``PMM2DStackPure`` (``stack2d_pure.py``) -- an L2 MORTAR couples adjacent
      per-layer grids; guarded by the minimum-segment contract
      ``_STAG_MIN_SEG_FRAC`` (refuse) + the degradation band
      ``_STAG_SLIVER_BAND_FRAC`` (warn) + ``_guarded_mortar_solve``.
    * ``PMM2DStackHybrid`` (``stack2d.py``) -- NO union grid and NO mortar
      (every layer is Fourier-Galerkin projected into the shared Rayleigh
      basis), but each layer still gets its OWN spectral-element axis from
      ``pmm/twod.py:203 _build_axis``, which carries NO minimum-width contract.
      Ladder: an INTRA-layer sliver strip of width ``delta * period`` placed by
      ``add_tapered_pillar`` (exact walls, no pixel snapping), instrumented with
      the SEM mass matrix that ``_assemble_2d`` inverts unguarded
      (``pmm/twod.py:393 Minv = np.linalg.inv(M)``,
      ``pmm/twod.py:408 np.linalg.solve(P_inv, M)``).

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=n MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_sliver.py out.json
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

WL = 0.85e-6
PX = 1.2e-6
NS = 1.0
EPS_H, EPS_P = 2.25, 9.0
DELTAS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6)
#: the hybrid has NO width contract at all, so its ladder is pushed until the
#: SEM mass matrix that ``_assemble_2d`` inverts unguarded actually breaks.
DELTAS_DEEP = DELTAS + (1e-7, 1e-8, 1e-10, 1e-12, 1e-14)


def _capture(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res, err = fn(), None
        except Exception as exc:
            res, err = None, "%s: %s" % (type(exc).__name__,
                                         str(exc).replace("\n", " ")[:260])
    return res, err, [dict(cls=type(x.message).__name__,
                           msg=str(x.message).replace("\n", " ")[:220])
                      for x in w]


def _cl(res):
    v = E.rt_per_pol(res)
    return dict(rt=[float(x) for x in v],
                closure=float(np.max(np.abs(v - 1.0))))


# ------------------------------------------------------------------ C1 1-D
def c1_pmmstack():
    from lumenairy import PMMStack
    A0, B0 = 0.2917, 0.6109
    dz = 0.08e-6

    def solve(delta, degree=14):
        s = PMMStack(PX, n_superstrate=NS, n_substrate=NS, degree=degree)
        for (a, b) in ((A0, B0), (A0 - delta, B0 + delta)):
            s.add_layer(dz, segments=[(a, EPS_H), (b - a, EPS_P),
                                      (1.0 - b, EPS_H)])
        s.set_source(WL, theta=0.15)
        o, R, T = s.solve()[:3]
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        return o[i], np.atleast_2d(R)[-1][i], np.atleast_2d(T)[-1][i]

    ref, referr, _ = _capture(lambda: solve(0.0))
    rows = []
    for d in DELTAS:
        res, err, warns = _capture(lambda dd=d: solve(dd))
        row = dict(delta=d, sliver_m=d * PX, raised=err, warnings=warns)
        if res is not None:
            R, T = res[1], res[2]
            row["closure"] = float(abs(R.sum() + T.sum() - 1.0))
            if ref is not None:
                row["shift_vs_delta0"] = float(
                    max(np.abs(R - ref[1]).max(), np.abs(T - ref[2]).max()))
        rows.append(row)
    return dict(reference_raised=referr, rows=rows)


# --------------------------------------------------------------- C2 pure 2-D
def c2_pure():
    from lumenairy.elements.pmm import PMM2DStackPure

    def solve(delta):
        s = PMM2DStackPure(PX, PX, n_substrate=NS, n_superstrate=NS,
                           degree=6, n_orders=3, layer_grids="per-layer")
        s.add_layer(0.1e-6, eps=EPS_H)
        # SQUARE segment counts are a contract of the staggered basis, so the
        # y axis carries the same number of walls as the slivered x axis.
        xw = [0.30 * PX, (0.30 + delta) * PX, 0.70 * PX]
        yw = [0.25 * PX, 0.50 * PX, 0.75 * PX]
        tile = np.full((4, 4), EPS_H, dtype=complex)
        tile[1, 1:3] = EPS_P
        tile[2, 1:3] = EPS_P
        s.add_layer(0.2e-6, eps_cell=tile, x_walls=xw, y_walls=yw)
        s.add_layer(0.1e-6, eps=EPS_H)
        r = s.set_source(WL, theta=0.15).solve()
        return (r[0], np.atleast_2d(r[1]), np.atleast_2d(r[2]))

    rows = []
    for d in DELTAS:
        res, err, warns = _capture(lambda dd=d: solve(dd))
        row = dict(delta=d, seg_frac=d, raised=err, warnings=warns)
        if res is not None:
            row.update(_cl(res))
        rows.append(row)
    return rows


# ------------------------------------------------------------- C2 hybrid 2-D
def c2_hybrid():
    from lumenairy.elements.pmm import PMM2DStackHybrid
    from lumenairy.elements.pmm.twod import _assemble_2d, _build_axis

    def solve(delta, degree=7, n_orders=3):
        s = PMM2DStackHybrid(PX, PX, n_substrate=NS, n_superstrate=NS,
                             degree=degree, n_orders=n_orders, symmetry=False)
        s.add_layer(0.1e-6, eps=EPS_H)
        if delta > 0.0:
            s.add_tapered_pillar(0.2e-6, eps_pillar=EPS_P, eps_host=EPS_H,
                                 x_bounds_bottom=(0.30 * PX,
                                                  (0.30 + delta) * PX),
                                 y_bounds_bottom=(0.30 * PX, 0.70 * PX),
                                 n_slices=1)
        else:
            s.add_layer(0.2e-6, eps=EPS_H)
        s.add_layer(0.1e-6, eps=EPS_H)
        r = s.set_source(WL, theta=0.15).solve()
        return (r[0], np.atleast_2d(r[1]), np.atleast_2d(r[2]))

    def sem_conditioning(delta, degree=7):
        ax = _build_axis(PX, [0.30 * PX, (0.30 + delta) * PX], degree, 1, False)
        ay = _build_axis(PX, [0.30 * PX, 0.70 * PX], degree, 1, False)
        tile = np.full((3, 3), EPS_H, dtype=complex)
        tile[1, 1] = EPS_P
        ops = _assemble_2d(ax, ay, tile, 2 * np.pi / WL)
        return dict(n_dof=int(ops["M"].shape[0]),
                    cond_mass_M=float(np.linalg.cond(ops["M"])),
                    cond_axis_Mx=float(np.linalg.cond(ax["M"])),
                    max_abs_Gx=float(np.max(np.abs(ops["Gx"]))),
                    widest_over_narrowest=float(
                        max(b - a for a, b, _s in ax["elem_bnds"])
                        / min(b - a for a, b, _s in ax["elem_bnds"])))

    ref, referr, _ = _capture(lambda: solve(0.0))
    rows = []
    for d in DELTAS_DEEP:
        res, err, warns = _capture(lambda dd=d: solve(dd))
        row = dict(delta=d, sliver_m=d * PX, raised=err, warnings=warns)
        try:
            row["sem"] = sem_conditioning(d)
        except Exception as exc:
            row["sem"] = dict(raised=repr(exc)[:160])
        if res is not None:
            row.update(_cl(res))
            if ref is not None:
                row["shift_vs_nopillar"] = float(
                    max(np.abs(res[1] - ref[1]).max(),
                        np.abs(res[2] - ref[2]).max()))
        rows.append(row)

    # HIGH-DEGREE arm: the C1 spurious wavenumber scales as N(N+1)/(k0 J), so
    # degree amplifies whatever a narrow element does.
    ref13, _e13, _w13 = _capture(lambda: solve(0.0, degree=13))
    rows13 = []
    for d in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        res, err, warns = _capture(lambda dd=d: solve(dd, degree=13))
        row = dict(delta=d, degree=13, raised=err, warnings=warns)
        try:
            row["sem"] = sem_conditioning(d, degree=13)
        except Exception as exc:
            row["sem"] = dict(raised=repr(exc)[:160])
        if res is not None:
            row.update(_cl(res))
            if ref13 is not None:
                row["shift_vs_nopillar"] = float(
                    max(np.abs(res[1] - ref13[1]).max(),
                        np.abs(res[2] - ref13[2]).max()))
        rows13.append(row)
    return dict(reference_raised=referr, rows=rows, rows_degree13=rows13)


def main():
    E.pin_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "e_sliver.json"
    res = {}
    for key, fn in (("C1_pmmstack_union", c1_pmmstack),
                    ("C2_pure_mortar", c2_pure),
                    ("C2_hybrid_intralayer", c2_hybrid)):
        try:
            res[key] = fn()
        except Exception as exc:
            res[key] = dict(fatal=repr(exc)[:300])
        print("done", key, flush=True)
    E.dump(out, res)


if __name__ == "__main__":
    main()
