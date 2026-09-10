"""T1 -- converged-reference study for the two BOUNDED out-of-plane 2-D cases.

The two cases are open item 1 of
``docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md`` and the "could not
verify" of ``docs/audits/VERIFY_PMM2D_STAGGERED_OOP_2026_09_09.md`` section 10:

  ``corner``  the (3,3) L cell with a RE-ENTRANT 270-degree corner
              (``validation/probe_pmm2d_staggered_oop/g0_corner_gate.py``),
              normal AND conical 20/35.
  ``chiral``  the (3,3) CHIRAL cell of
              ``validation/probe_verify_staggered_oop/v2b_gauge_chiral_2d.py``
              at CONICAL 25/40 (and normal, as the control).

For each fixture x mount, THREE independent truncation ladders are run in the
LIBRARY gauge (the shipped entry points, not the prototype driver):

  * ``pmm_jones_2d_staggered``  -- the staggered modal degree M
  * ``pmm_jones_2d``            -- the hybrid's ``n_orders``, BOTH E_z rules
  * ``rcwa_jones_2d``           -- the Fourier ``n_orders`` (pixel-upsampled)

Each ladder is then fitted to ``f(x) = f_inf + C x^-p`` (the algebraic rate is
FITTED, not assumed), Richardson/Aitken-extrapolated, and given an uncertainty
derived from the fit's own residual, its stability against dropping the first
ladder point, and its distance from the independent Aitken extrapolant.  The
question the tables answer is whether the engines' extrapolated limits agree
within their COMBINED uncertainties, per order.

Run (each fixture separately; both are long):
  cd /c/tmp/lum_oopfast && PYTHONPATH=/c/tmp/lum_oopfast OMP_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python validation/probe_pmm2d_staggered_oop_reference/t1_ladders.py corner
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d, uniaxial_tensor  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

#: per-solve wall-time budget: a ladder stops once its last rung exceeds this
BUDGET_S = 180.0

FIXTURES = {
    # the M9 L of the prototype, in the library gauge
    "corner": dict(
        px=1.2e-6, py=1.2e-6, wl=1.0e-6, depth=0.4e-6, nsub=1.5, nsup=1.0,
        mounts=(("normal", 0.0, 0.0),
                ("conical20_35", np.deg2rad(20.0), np.deg2rad(35.0))),
        M_ladder=(5, 6, 7, 8, 9, 10),
        hyb_orders=(7, 9, 11, 13), rcwa_orders=(5, 7, 9, 11),
        hyb_degree=9),
    # the v2b chiral cell: two DIFFERENT out-of-plane tensors + an isotropic
    # pixel, so the 180-degree image differs in pattern AND tensor
    "chiral": dict(
        px=1.10e-6, py=1.10e-6, wl=0.68e-6, depth=0.36e-6, nsub=1.5, nsup=1.0,
        mounts=(("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0)),
                ("normal", 0.0, 0.0)),
        M_ladder=(5, 6, 7, 8, 9, 10),
        hyb_orders=(7, 9, 11, 13), rcwa_orders=(5, 7, 9, 11),
        hyb_degree=9),
}


def cell_corner():
    er = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = np.eye(3)
    for i, j in ((0, 0), (1, 0), (0, 1)):
        e[i, j] = er
    return e


def cell_chiral():
    tA = uniaxial_tensor(1.46, 1.74, 0.58, phi=0.31)
    tC = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = np.eye(3)
    e[0, 0] = tA
    e[1, 0] = tC
    e[1, 2] = 2.25 * np.eye(3)
    return e


CELLS = {"corner": cell_corner, "chiral": cell_chiral}

#: the observables every ladder is fitted on: per-order R and T for the
#: propagating set, both incident polarizations, plus the four Jones entries.
KEEP = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1),
        (1, -1), (-1, 1)]


def extract(o, Rm, Tm, J):
    """Flat dict of scalar observables, keyed by name."""
    idx = {tuple(int(v) for v in row): j for j, row in enumerate(np.asarray(o))}
    d = {}
    for k in KEEP:
        j = idx.get(k)
        if j is None:
            continue
        for pol in (0, 1):
            d[f"R{k}p{pol}"] = float(np.asarray(Rm)[pol, j])
            d[f"T{k}p{pol}"] = float(np.asarray(Tm)[pol, j])
    d["sumR0"] = float(np.sum(np.asarray(Rm)[0]))
    d["sumR1"] = float(np.sum(np.asarray(Rm)[1]))
    d["sumT0"] = float(np.sum(np.asarray(Tm)[0]))
    d["sumT1"] = float(np.sum(np.asarray(Tm)[1]))
    for a in range(2):
        for b in range(2):
            d[f"Jre{a}{b}"] = float(np.real(np.asarray(J)[a, b]))
            d[f"Jim{a}{b}"] = float(np.imag(np.asarray(J)[a, b]))
    return d


def fit_limit(x, f):
    """Fit ``f(x) = f_inf + C x^-p`` (p scanned, (f_inf, C) linear) and return
    ``(f_inf, sigma, p, rel_resid)``.

    ``sigma`` is the MAXIMUM of three independently derived quantities, none of
    them assumed:

      * the linear least-squares standard error of the ``f_inf`` coefficient at
        the best ``p``, with the fit's own residual as the noise estimate;
      * the shift in ``f_inf`` when the FIRST ladder rung is dropped (a
        stability check against the pre-asymptotic regime);
      * the distance to the independent Aitken ``delta^2`` extrapolant built
        from the last three rungs.
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    if x.size < 3:
        return float(f[-1]), float("inf"), float("nan"), float("nan")

    def _lin(xx, ff, p):
        A = np.stack([np.ones_like(xx), xx ** (-p)], axis=1)
        coef, *_ = np.linalg.lstsq(A, ff, rcond=None)
        res = ff - A @ coef
        dof = max(1, xx.size - 2)
        s2 = float(res @ res) / dof
        cov = s2 * np.linalg.pinv(A.T @ A)
        return float(coef[0]), float(np.sqrt(max(cov[0, 0], 0.0))), \
            float(np.sqrt(float(res @ res)))

    grid = np.linspace(0.25, 10.0, 400)
    best = None
    for p in grid:
        fi, se, rn = _lin(x, f, p)
        if best is None or rn < best[3]:
            best = (fi, se, rn, rn, p)
    f_inf, se, rn, _, p = best
    # stability: drop the first rung
    if x.size >= 4:
        best2 = None
        for pp in grid:
            fi2, _se2, rn2 = _lin(x[1:], f[1:], pp)
            if best2 is None or rn2 < best2[1]:
                best2 = (fi2, rn2)
        drop = abs(best2[0] - f_inf)
    else:
        drop = 0.0
    # Aitken delta^2 on the last three rungs
    f0, f1, f2 = f[-3], f[-2], f[-1]
    den = (f2 - f1) - (f1 - f0)
    aitken = f2 - (f2 - f1) ** 2 / den if abs(den) > 0 else f2
    dait = abs(aitken - f_inf)
    scale = max(abs(f_inf), 1e-12)
    sigma = max(se, drop, dait)
    return f_inf, sigma, p, rn / scale


def run_ladders(fix, mount, out):
    name, th, ph = mount
    cfg = FIXTURES[fix]
    cell = CELLS[fix]()
    px, py, wl = cfg["px"], cfg["py"], cfg["wl"]
    dep, nsub, nsup = cfg["depth"], cfg["nsub"], cfg["nsup"]
    rec = {"fixture": fix, "mount": name, "ladders": {}}
    out["cases"].append(rec)

    def save():
        with open(os.path.join(OUT, f"t1_{fix}.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    # ---------------- staggered M ladder --------------------------------
    lad = {"x": [], "obs": [], "t": []}
    for M in cfg["M_ladder"]:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = pmm_jones_2d_staggered(px, py, cell, nsub, nsup, dep, wl,
                                       degree=M, n_orders=7, theta=th, phi=ph)
        dt = time.time() - t0
        lad["x"].append(M)
        lad["obs"].append(extract(*r))
        lad["t"].append(dt)
        print(f"[{fix}/{name}] staggered M={M} dim={4*(3*(M-1))**2} "
              f"sumR={lad['obs'][-1]['sumR0']:.9f} [{dt:.1f}s]", flush=True)
        save()
        if dt > BUDGET_S:
            print(f"[{fix}/{name}] staggered ladder stops at M={M} "
                  f"({dt:.0f}s > {BUDGET_S:.0f}s budget)", flush=True)
            break
    rec["ladders"]["staggered"] = lad
    save()

    # ---------------- hybrid n_orders ladders ---------------------------
    for form in ("laurent", "li"):
        lad = {"x": [], "obs": [], "t": []}
        for no in cfg["hyb_orders"]:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = pmm_jones_2d(px, py, cell, nsub, nsup, dep, wl,
                                 degree=cfg["hyb_degree"], n_orders=no,
                                 formulation=form, stabilize=True,
                                 theta=th, phi=ph)
            dt = time.time() - t0
            lad["x"].append(no)
            lad["obs"].append(extract(*r))
            lad["t"].append(dt)
            print(f"[{fix}/{name}] hybrid-{form} n_orders={no} "
                  f"sumR={lad['obs'][-1]['sumR0']:.9f} [{dt:.1f}s]", flush=True)
            save()
            if dt > BUDGET_S:
                print(f"[{fix}/{name}] hybrid-{form} ladder stops at "
                      f"n_orders={no}", flush=True)
                break
        rec["ladders"][f"hybrid-{form}"] = lad
        save()

    # ---------------- rcwa n_orders ladder ------------------------------
    lad = {"x": [], "obs": [], "t": []}
    for no in cfg["rcwa_orders"]:
        k = int(np.ceil((4 * no + 1) / 3))
        fine = np.repeat(np.repeat(cell, k, axis=0), k, axis=1)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = rcwa_jones_2d(px, py, fine, nsub, nsup, dep, wl,
                              n_orders_x=no, n_orders_y=no, theta=th, phi=ph)
        dt = time.time() - t0
        lad["x"].append(no)
        lad["obs"].append(extract(*r))
        lad["t"].append(dt)
        print(f"[{fix}/{name}] rcwa n_orders={no} (upsample {k}x) "
              f"sumR={lad['obs'][-1]['sumR0']:.9f} [{dt:.1f}s]", flush=True)
        save()
        if dt > BUDGET_S:
            print(f"[{fix}/{name}] rcwa ladder stops at n_orders={no}",
                  flush=True)
            break
    rec["ladders"]["rcwa"] = lad
    save()

    # ---------------- fits ----------------------------------------------
    fits = {}
    keys = sorted(set.intersection(*[set(la["obs"][0])
                                     for la in rec["ladders"].values()]))
    for eng, la in rec["ladders"].items():
        fits[eng] = {}
        for k in keys:
            vals = [o[k] for o in la["obs"]]
            fi, sg, p, rr = fit_limit(la["x"], vals)
            fits[eng][k] = dict(f_inf=fi, sigma=sg, p=p, rel_resid=rr,
                                last=vals[-1],
                                last_step=(abs(vals[-1] - vals[-2])
                                           if len(vals) > 1 else float("nan")))
    rec["fits"] = fits
    save()

    print(f"\n[{fix}/{name}] EXTRAPOLATED LIMITS (per observable)")
    hdr = "   observable    " + "".join(f"{e:>26s}" for e in fits)
    print(hdr)
    engs = list(fits)
    for k in keys:
        if not (k.startswith("R(") or k.startswith("T(") or k.startswith("sum")):
            continue
        line = f"   {k:12s}  "
        for e in engs:
            line += f"{fits[e][k]['f_inf']:14.8f} +/-{fits[e][k]['sigma']:8.1e}"
        print(line)
    print(f"\n[{fix}/{name}] PAIRWISE: |limit_A - limit_B| vs the combined "
          f"uncertainty sqrt(sA^2 + sB^2)")
    pw = {}
    for i, a in enumerate(engs):
        for b in engs[i + 1:]:
            worst = None
            for k in keys:
                d = abs(fits[a][k]["f_inf"] - fits[b][k]["f_inf"])
                c = float(np.hypot(fits[a][k]["sigma"], fits[b][k]["sigma"]))
                z = d / c if c > 0 else float("inf")
                if worst is None or z > worst[3]:
                    worst = (k, d, c, z)
            n_ok = sum(1 for k in keys
                       if abs(fits[a][k]["f_inf"] - fits[b][k]["f_inf"])
                       <= float(np.hypot(fits[a][k]["sigma"],
                                         fits[b][k]["sigma"])))
            pw[f"{a}|{b}"] = dict(worst_key=worst[0], worst_d=worst[1],
                                  worst_c=worst[2], worst_z=worst[3],
                                  n_within=n_ok, n_total=len(keys))
            print(f"   {a:16s} vs {b:16s}: within combined sigma on "
                  f"{n_ok}/{len(keys)} observables; worst {worst[0]} "
                  f"d={worst[1]:.3e} sigma={worst[2]:.3e} z={worst[3]:.2f}")
    rec["pairwise"] = pw
    save()


def main():
    fix = sys.argv[1] if len(sys.argv) > 1 else "corner"
    out = {"lumenairy": lumenairy.__file__, "version": lumenairy.__version__,
           "fixture": fix, "cases": []}
    for mount in FIXTURES[fix]["mounts"]:
        run_ladders(fix, mount, out)
    print(f"\nwrote results/t1_{fix}.json")


if __name__ == "__main__":
    main()
