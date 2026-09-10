"""W3 -- the three SOLVE-FREE bars, re-measured on my own populations.

  * ``_SLIVER_OWN_SCALE_RATIO`` = 100: the own/w ratio an ORDINARY
    non-conforming stack reaches (the fix: <= 25.43 over 4,192 random
    two-layer geometries).  Re-drawn here with a different RNG stream, a
    different layer count (2, 3 AND 4 layers) and a different gap law.
  * ``_PASSIVE_ANTIHERM_DEADBAND`` = 16 ULP: the worst spurious
    ``-lam_min(A)/max|eps|`` a rotated uniaxial director produces in floats
    (the fix: 0.77 ULP over 200,000 lossless, 0.00 over 50,000 lossy), and
    what a GAIN payload reads (the fix: 5.8e+09 ULP at ``Im(n)`` = 1e-6).
  * ``_SLIVER_Q_EXCESS`` = 1e+6: the predicted spurious ``|q| / n_max`` of an
    ORDINARY owned cell vs a liner that actually breaks (the fix: <= 3.322e+03
    vs >= 8.793e+06).

Solve-free except the q-excess ladder's own reference solves.

    python w3_bars.py [out.json]
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lumenairy                                               # noqa: E402
from lumenairy.elements.pmm import PMMStack                    # noqa: E402
from lumenairy.elements.pmm import stack as ps                 # noqa: E402
from w_fixtures import shared_move, unguarded, uniaxial        # noqa: E402

ULP = float(np.finfo(float).eps)


# ------------------------------------------------------------- (d) ratio ----
def ratio_population(n=6000, seed=20260911):
    """Random ORDINARY non-conforming stacks: 2-4 layers, every wall a real
    feature and every CROSS-LAYER wall gap a real 1-8 % of a period.  The
    quantity is the screen's own ``own / w`` on every MANUFACTURED cell --
    scored UNBARRED (the screen's 100x filter is what is being tested, so it
    cannot be applied first)."""
    from lumenairy.elements.pmm._core import _pmm_union_grid
    rng = np.random.default_rng(seed)
    vals, drawn = [], 0
    for _ in range(n):
        nl = int(rng.integers(2, 5))
        walls = []
        for _k in range(nl):
            for _try in range(400):
                a = float(rng.uniform(0.08, 0.45))
                b = float(rng.uniform(a + 0.12, 0.92))
                flat = [x for w in walls for x in w]
                gaps = [abs(c - x) for c in (a, b) for x in flat]
                if all(0.01 <= g <= 0.08 or g > 0.08 for g in gaps) and all(
                        g >= 0.01 for g in gaps):
                    walls.append((a, b))
                    break
            else:
                break
        if len(walls) != nl:
            continue
        drawn += 1
        segs = [[(a, 4.0), (b - a, 1.0), (1.0 - b, 2.25)] for a, b in walls]
        uw, _e, owners = _pmm_union_grid(segs, 1e-12, return_owners=True,
                                         warn=False)
        own = min(abs(float(sg[0])) for sl in segs for sg in sl
                  if float(sg[0]) > 0.0)
        for i, w in enumerate(np.asarray(uw, dtype=float)):
            w = float(w)
            if w <= 0.0 or (owners[i] & owners[i + 1]):
                continue                    # some ONE layer asked for it
            vals.append(own / w)
    v = np.sort(np.asarray(vals))
    return dict(n_drawn=drawn, n_manufactured_cells=int(v.size),
                max=float(v[-1]) if v.size else None,
                p999=float(np.percentile(v, 99.9)) if v.size else None,
                p99=float(np.percentile(v, 99)) if v.size else None,
                median=float(np.median(v)) if v.size else None,
                bar=ps._SLIVER_OWN_SCALE_RATIO,
                headroom=(ps._SLIVER_OWN_SCALE_RATIO / float(v[-1])
                          if v.size else None),
                n_over_bar=int((v >= ps._SLIVER_OWN_SCALE_RATIO).sum()))


# ---------------------------------------------------------- (e) deadband ----
def antiherm_ulp(M):
    A = (M - np.conjugate(M).T) / 2.0j
    scale = max(float(np.max(np.abs(M))), 1.0)
    return -float(np.min(np.linalg.eigvalsh(A))) / scale / ULP


def deadband_population(n_lossless=200_000, n_lossy=50_000, seed=771):
    rng = np.random.default_rng(seed)
    worst_ll, worst_ly = -np.inf, -np.inf
    fails_ll = fails_ly = 0
    bar = ps._PASSIVE_ANTIHERM_DEADBAND / ULP
    for _ in range(n_lossless):
        no = float(rng.uniform(1.2, 4.0))
        ne = float(rng.uniform(1.2, 4.0))
        M = uniaxial(no, ne, float(rng.uniform(0, np.pi)),
                     azim=float(rng.uniform(0, 2 * np.pi)))
        u = antiherm_ulp(M)
        worst_ll = max(worst_ll, u)
        if not ps._segment_passive(M):
            fails_ll += 1
    for _ in range(n_lossy):
        no = complex(rng.uniform(1.2, 4.0), rng.uniform(1e-4, 0.4))
        ne = complex(rng.uniform(1.2, 4.0), rng.uniform(1e-4, 0.4))
        M = uniaxial(no, ne, float(rng.uniform(0, np.pi)),
                     azim=float(rng.uniform(0, 2 * np.pi)))
        u = antiherm_ulp(M)
        worst_ly = max(worst_ly, u)
        if not ps._segment_passive(M):
            fails_ly += 1
    gains = {}
    for imn in (1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11,
                1e-9, 1e-8, 1e-7, 1e-6, 1e-4):
        n = 1.5 - 1j * imn                    # Im(n) < 0 is GAIN in Im>=0
        M = uniaxial(n, n * 1.1, 0.4, azim=0.7)
        gains[f"gain_Im(n)={imn:g}"] = dict(
            ulp=antiherm_ulp(M), passive=bool(ps._segment_passive(M)))
    return dict(bar_ulp=bar, worst_lossless_ulp=worst_ll,
                worst_lossy_ulp=worst_ly, n_lossless=n_lossless,
                n_lossy=n_lossy, refused_lossless=fails_ll,
                refused_lossy=fails_ly,
                headroom=bar / max(worst_ll, worst_ly, 1e-300), gains=gains)


# ----------------------------------------------------------- (f) q-excess ---
def qexcess_populations():
    """Ordinary OWNED cells vs the liners that actually break.  The predictor
    is the library's own; ``n_max`` is read off a concrete stack."""
    period, wl = 1.2e-6, 0.85e-6
    n_max = 3.0                                  # sqrt(9.0), the ridge index
    ordinary = []
    for w in np.geomspace(1e-3, 0.30, 40):
        for deg in (8, 10, 12, 14, 16):
            ordinary.append(ps._sliver_q_predictor(float(w), period, deg, wl)
                            / n_max)
    ladder = {}
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        q = {deg: ps._sliver_q_predictor(d, period, deg, wl) / n_max
             for deg in (8, 12, 14, 16)}
        ladder[f"{d:g}"] = q
    # the SOLVED ladder: err/d and R+T for a liner ONE layer owns
    solved = {}
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        row = {}
        for deg in (8, 12, 14, 16):
            st = PMMStack(period, n_substrate=1.0, degree=deg,
                          far_field_orders=21, min_feature=period * 1e-12)
            for _k in range(2):
                st.add_layer(0.08e-6,
                             segments=[(0.30, 2.25), (d, 9.0),
                                       (0.70 - d, 2.25)])
            st.set_source(wl, theta=0.15)
            ref = PMMStack(period, n_substrate=1.0, degree=deg,
                           far_field_orders=21, min_feature=period * 1e-12)
            for _k in range(2):
                ref.add_layer(0.08e-6, segments=[(0.30, 2.25), (0.70, 2.25)])
            ref.set_source(wl, theta=0.15)
            a, b = unguarded(st), unguarded(ref)
            row[deg] = dict(err=shared_move(a, b, pol=1),
                            err_over_d=shared_move(a, b, pol=1) / d,
                            worst=a["worst"])
        solved[f"{d:g}"] = row
    return dict(bar=ps._SLIVER_Q_EXCESS, ordinary_min=min(ordinary),
                ordinary_max=max(ordinary), n_ordinary=len(ordinary),
                predicted_ladder=ladder, solved_ladder=solved)


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w3_bars.json"))
    print("lumenairy:", os.path.abspath(lumenairy.__file__))
    t0 = time.time()
    out = dict(lumenairy=os.path.abspath(lumenairy.__file__),
               python=sys.version.split()[0], numpy=np.__version__)
    out["ratio"] = ratio_population()
    print("  ratio:", {k: v for k, v in out["ratio"].items()})
    out["deadband"] = deadband_population()
    print("  deadband:", {k: v for k, v in out["deadband"].items()
                          if k != "gains"})
    print("  gains:", out["deadband"]["gains"])
    out["qexcess"] = qexcess_populations()
    q = out["qexcess"]
    print(f"  q-excess ordinary: {q['ordinary_min']:.4g} .. "
          f"{q['ordinary_max']:.4g} (bar {q['bar']:g})")
    for d, row in q["solved_ladder"].items():
        print(f"    liner {d}: q@14 = {q['predicted_ladder'][d][14]:.4g}, "
              + ", ".join(f"deg{k} err/d={v['err_over_d']:.3g} "
                          f"R+T={v['worst']:.6g}" for k, v in row.items()))
    out["wall"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("->", out_path, f"{out['wall']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
