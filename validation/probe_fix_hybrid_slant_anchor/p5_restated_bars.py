"""P5 -- re-measure the FOUR thin bars the slant verification flagged in
``tests/unit/test_pmm2d_staggered_slant.py`` (D2 / S8.2 of
``docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10.md``), on BOTH builds, so
their restatements carry their own measurement:

  1. ``cost_min_ratio`` -- the ONE quantity with real cross-build/under-load
     spread (a wall clock).  The test cites ``0.96x .. 1.07x``; the
     verification read ``1.167`` (WIN) / ``1.105`` (WSL) at ``M = 6`` under
     load.  Re-measured here BOTH ways -- with this box otherwise idle and
     with a deliberate co-running load -- so the docstring can state an
     envelope instead of an unloaded snapshot.
  2. ``m4c_*_against_ratio`` -- 0.875 / 0.986 against a 0.8 bar (a 9% margin).
  3. ``m4_*_plus_monotone`` -- rung-by-rung strictness on a cross-engine
     ladder; the first-to-last ratio is what the claim actually needs.
  4. ``b3_conj_over_none`` -- 1.79 against a 1.5 bar; the docstring says
     "about TWICE" while the SMALLEST row is 1.787.

The fixtures are IMPORTED from the test file itself, so this re-measures the
file's own bars and not a lookalike.
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from _lib import arm, dump  # noqa: I001,E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "unit"))
import test_pmm2d_staggered_slant as TS  # noqa: E402


def cost_ratios(reps=3):
    from lumenairy.elements.pmm.stack2d_pure import Granet2DTransverseE, _region_modes_oop
    k0 = 2.0 * np.pi
    a0 = (0.25 * k0, 0.18 * k0)
    out = {}
    for M in (4, 5, 6, 7):
        def _t(fn):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))
        to = _t(lambda: _region_modes_oop(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, TS.OOPC, alpha0x=a0[0], alpha0y=a0[1], k0=k0)))
        ts = _t(lambda: _region_modes_oop(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, TS.SCA, alpha0x=a0[0], alpha0y=a0[1], k0=k0,
            slant=(0.75, 0.0))))
        out[f"M{M}"] = ts / to
    return out


def m4c(theta, phi):
    base = TS._pure_pillar([TS._pillar3()], [(TS._PT, 0.0)], [TS._PDEP], 3,
                           theta, phi)
    got = {}
    for direc, lab in ((+1, "with"), (-1, "against")):
        got[lab] = []
        for n in (1, 2):
            cells = [TS._pillar3(direc * k * (2 // n)) for k in range(n)]
            v = TS._pure_pillar(cells, [None] * n, [TS._PDEP / n] * n, 3,
                                theta, phi)
            got[lab].append(float(np.max(np.abs(v - base))))
    return dict(with_=got["with"], against=got["against"],
                with_ratio=got["with"][1] / got["with"][0],
                against_ratio=got["against"][1] / got["against"][0],
                separation=((got["against"][1] / got["against"][0])
                            / (got["with"][1] / got["with"][0])))


def m4(theta, phi):
    import warnings

    from lumenairy.elements.pmm import PMM2DStackHybrid
    ref = TS._pure_pillar([TS._pillar3()], [(TS._PT, 0.0)], [TS._PDEP], 4,
                          theta, phi)
    got = {}
    for sgn, lab in ((+1.0, "plus"), (-1.0, "minus")):
        got[lab] = []
        for no in (3, 5, 7):
            hs = PMM2DStackHybrid(TS._PPX, TS._PPX, n_superstrate=TS.NSUP,
                                  n_substrate=TS.NSUB, n_orders=no)
            hs.add_layer(TS._PDEP, eps_cell=TS._pillar3(),
                         slant=(sgn * TS._PT, 0.0))
            hs.set_source(TS._PWL, theta=theta, phi=phi)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oh, Rh, Th, Jh = hs.solve()
            got[lab].append(float(np.max(np.abs(TS._pvec(oh, Rh, Th) - ref))))
    got["plus_first_over_last"] = got["plus"][0] / got["plus"][-1]
    got["minus_last_over_first"] = got["minus"][-1] / got["minus"][0]
    got["plus_step_ratios"] = [got["plus"][0] / got["plus"][1],
                               got["plus"][1] / got["plus"][2]]
    return got


def b3():
    """The conjugate / none ratio over the 12 rows of B3's own parametrization,
    on the uniform null where the truth is exact."""
    rows = []
    for tname, tv in (("iso", 2.25), ("oop", TS.TIL)):
        for theta, phi in ((np.deg2rad(25.0), 0.0),
                           (np.deg2rad(25.0), np.deg2rad(40.0))):
            (o0, R0, T0, J0), st0 = TS._stack(
                [(TS.DEP, {"eps": complex(tv) if np.ndim(tv) == 0
                           else np.asarray(tv, dtype=complex)})], 5,
                theta=theta, phi=phi)
            Jt0 = TS._transmission_jones(st0)
            for sname, sv in (("x10", (TS.T10, 0.0)), ("x35", (TS.T35, 0.0)),
                              ("diag35", TS.DIAG35)):
                kw = {"eps": complex(tv) if np.ndim(tv) == 0
                      else np.asarray(tv, dtype=complex), "slant": sv}
                (o1, R1, T1, J1), st1 = TS._stack([(TS.DEP, kw)], 5,
                                                  theta=theta, phi=phi)
                Jt = TS._transmission_jones(st1)
                shx = -sum(L.get("slant", (0.0, 0.0))[0] * L["thickness"]
                           for L in st1._layers)
                shy = -sum(L.get("slant", (0.0, 0.0))[1] * L["thickness"]
                           for L in st1._layers)
                k0 = 2.0 * np.pi / st1._modal["wavelength"]
                p0 = st1._modal["p0"]
                ph0 = np.exp(-1j * k0 * (st1._modal["kx"][p0] * shx
                                         + st1._modal["ky"][p0] * shy))
                shipped = float(np.max(np.abs(Jt - Jt0)))
                none_ = float(np.max(np.abs(Jt / ph0 - Jt0)))
                plus = float(np.max(np.abs(Jt / ph0 / ph0 - Jt0)))
                rows.append(dict(tensor=tname, mount=f"{theta:.2f}/{phi:.2f}",
                                 slant=sname, shipped=shipped, none=none_,
                                 conj=plus, ratio=plus / none_))
    return dict(rows=rows,
                ratio_min=min(r["ratio"] for r in rows),
                ratio_max=max(r["ratio"] for r in rows),
                shipped_max=max(r["shipped"] for r in rows),
                none_min=min(r["none"] for r in rows))


def main():
    out = {}
    out["cost_idle"] = cost_ratios()
    out["m4c"] = {m: m4c(*a) for m, a in
                  (("normal", (0.0, 0.0)),
                   ("conical", (np.deg2rad(20.0), np.deg2rad(35.0))))}
    out["m4"] = {m: m4(*a) for m, a in
                 (("normal", (0.0, 0.0)),
                  ("conical", (np.deg2rad(20.0), np.deg2rad(35.0))))}
    out["b3"] = b3()
    print("cost ratios (idle):", {k: round(v, 4)
                                  for k, v in out["cost_idle"].items()})
    for m, r in out["m4c"].items():
        print(f"m4c {m}: with {r['with_']} ratio {r['with_ratio']:.5f} | "
              f"against {r['against']} ratio {r['against_ratio']:.5f} | "
              f"separation {r['separation']:.3f}")
    for m, r in out["m4"].items():
        print(f"m4 {m}: +t {['%.3e' % v for v in r['plus']]} first/last "
              f"{r['plus_first_over_last']:.4f} steps "
              f"{['%.3f' % v for v in r['plus_step_ratios']]} | -t "
              f"{['%.3e' % v for v in r['minus']]} last/first "
              f"{r['minus_last_over_first']:.4f}")
    b = out["b3"]
    print(f"b3: conj/none over {len(b['rows'])} rows = {b['ratio_min']:.4f} .."
          f" {b['ratio_max']:.4f}; shipped max {b['shipped_max']:.3e}; none "
          f"min {b['none_min']:.3e}")
    dump("p5_restated_bars", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
