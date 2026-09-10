"""ROUND 2, probe 6 -- conjunct (a)'s ratio bar, and the coated-taper class.

The verification REFUTED the fix's claim that the M2 audit-class 2 deg coated
pillar taper sits at attribution ratio ~1.7e+02: it reads **12.11** (the
own-scale is the 5 nm conformal COAT, not the 200 nm ridge), so that whole
device class is OUTSIDE conjunct (a) and the guard can never fire on it.

Three questions, measured:

A. What ratio does that taper actually read, and is its answer defective?
B. Does a ratio-12 manufactured cell carry the defect AT ALL?  Built directly:
   the O-11 stack with an OWNED liner of width ``s`` in every layer (so the
   own-scale is ``s``) and the walls opened by ``delta`` -- ratio ``s/delta``
   at a FIXED, known-catastrophic ``delta``.
C. What ratios do ORDINARY non-conforming stacks produce?  That is the
   population the bar must stay above, and lowering the bar costs exactly the
   headroom over it.

    python validation/probe_pmmstack_sliver_round2/r6_ratio.py [out.json]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps

HERE = os.path.dirname(os.path.abspath(__file__))
NM = 1e-9
M2_PERIOD, M2_WL = 700 * NM, 1310 * NM
M2_SIDE, M2_H1, M2_COAT, M2_WTOP = (np.deg2rad(2.0), 310 * NM, 5.0 * NM,
                                    340 * NM)
M2_CORE, M2_CO, M2_GR = (3.48 + 0j) ** 2, (1.76 + 0j) ** 2, 1.0 + 0j
P, WL, TH = 1.2e-6, 0.85e-6, 0.15
DZ = 0.32e-6 / 4
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505


def m2_segments(w_core, coat=M2_COAT):
    w_out = 0.5 * (w_core + 2.0 * coat) / M2_PERIOD
    w_in = 0.5 * w_core / M2_PERIOD
    g = 0.5 - w_out
    c = w_out - w_in
    return [(g, M2_GR), (c, M2_CO), (2 * w_in, M2_CORE), (c, M2_CO), (g, M2_GR)]


def m2_layers(ns):
    return [(M2_H1 / ns,
             m2_segments(2.0 * (0.5 * M2_WTOP - ((k + 0.5) / ns) * M2_H1
                                * np.tan(M2_SIDE))))
            for k in range(ns)]


def run(st, guard=True):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return dict(ok=True, m=o[i], R=np.asarray(R)[1][i],
                    T=np.asarray(T)[1][i], worst=float(np.max(tot)))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(ok=False, msg=str(exc)[:140])
    finally:
        ps.PMM_SLIVER_GUARD = was


def gap(a, b):
    c = np.intersect1d(a["m"], b["m"])
    ia, ib = np.searchsorted(a["m"], c), np.searchsorted(b["m"], c)
    return float(max(np.abs(a["R"][ia] - b["R"][ib]).max(),
                     np.abs(a["T"][ia] - b["T"][ib]).max()))


def raw_ratio(layer_segments, mff):
    """The attribution ratio with NO bar -- what conjunct (a) would read if
    ``_SLIVER_OWN_SCALE_RATIO`` were 1.  ``None`` when the union manufactured
    no cell at all."""
    from lumenairy.elements.pmm._core import _pmm_union_grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uw, _e, own_sets = _pmm_union_grid(layer_segments, mff,
                                           return_owners=True, warn=False)
    own = min(abs(float(sg[0])) for segs in layer_segments for sg in segs
              if abs(float(sg[0])) > 0.0)
    best = None
    for i, w in enumerate(np.asarray(uw, dtype=float)):
        if w <= 0.0 or (own_sets[i] & own_sets[i + 1]):
            continue
        r = own / float(w)
        if best is None or r > best[0]:
            best = (r, float(w))
    return None if best is None else dict(ratio=best[0], w=best[1], own=own)


def part_a():
    out = []
    for ns in (2, 3, 6, 8):
        segs = [s for _t, s in m2_layers(ns)]
        for tag, mff in (("library default", 1e-5), ("no snap", 1e-12)):
            raw = raw_ratio(segs, mff)
            hit = ps._cross_layer_sliver(segs, mff)
            ratio = None if hit is None else hit[4] / hit[0]
            out.append(dict(ns=ns, min_feature=tag, ratio=ratio,
                            raw_ratio=None if raw is None else raw["ratio"],
                            raw_w_m=None if raw is None else raw["w"] * M2_PERIOD,
                            raw_own=None if raw is None else raw["own"],
                            w_m=None if hit is None else hit[0] * M2_PERIOD,
                            own=None if hit is None else hit[4],
                            n_cells=None if hit is None else hit[5]))
            rs = "None" if ratio is None else format(ratio, ".4g")
            rr = ("None" if raw is None
                  else f"{raw['ratio']:.4g} (w {raw['w'] * M2_PERIOD:.4g} m, "
                       f"own {raw['own']:.4g})")
            print(f"  ns={ns} {tag:16s} screen={rs}  UNBARRED ratio={rr}")
    return out


def _liner_layers(s, dd):
    segs = []
    for (a, b) in [(A0, B0), (A0 - dd, B0 + dd)]:
        if s < min(a, 1.0 - b) / 2.0:
            segs.append([(a - s, EH), (s, EP), (b - a, EP),
                         (1.0 - b - s, EH), (s, EP)])
        else:
            segs.append([(a, EH), (b - a, EP), (1.0 - b, EH)])
    return segs


def part_b():
    """The O-11 hazard at delta = 1e-4 (catastrophic, measured), with an OWNED
    liner of width ``s`` in every layer so the own-scale -- and therefore the
    attribution ratio -- is whatever we choose."""
    d = 1e-4
    rows = []
    for s in (0.2786, 1e-1, 3e-2, 1e-2, 3e-3, 1.2e-3, 1e-3, 5e-4):
        segs = _liner_layers(s, d)
        hit = ps._cross_layer_sliver(segs, 1e-12)
        ratio = None if hit is None else hit[4] / hit[0]
        st = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=14,
                      min_feature=P * 1e-12)
        for sg in segs:
            st.add_layer(DZ, segments=sg)
        st.set_source(WL, theta=TH)
        st0 = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=14,
                       min_feature=P * 1e-12)
        for sg in _liner_layers(s, 0.0):
            st0.add_layer(DZ, segments=sg)
        st0.set_source(WL, theta=TH)
        cur, ref = run(st, False), run(st0, False)
        e = gap(cur, ref) if (cur["ok"] and ref["ok"]) else None
        rows.append(dict(liner=s, delta=d, ratio=ratio, err=e,
                         err_over_delta=None if e is None else e / d,
                         RplusT=cur.get("worst"),
                         fires_at_100=bool(ratio is not None and ratio >= 100),
                         fires_at_10=bool(ratio is not None and ratio >= 10)))
        rs = "None" if ratio is None else format(ratio, ".4g")
        es = "n/a" if e is None else format(e, ".3e")
        xs = "n/a" if e is None else format(e / d, ".1f")
        print(f"  own-liner {s:.4g} -> ratio {rs}: err {es} ({xs}x delta)  "
              f"R+T {cur.get('worst')}")
    return rows


def part_c():
    """ORDINARY non-conforming stacks: two layers whose walls differ by REAL
    features.  The population conjunct (a)'s bar must stay above."""
    rng = np.random.default_rng(20260911)
    hits = []
    for _ in range(6000):
        n = int(rng.integers(2, 5))
        wa = np.sort(rng.uniform(0.02, 0.98, n))
        off = rng.uniform(0.01, 0.08, n) * rng.choice((-1.0, 1.0), n)
        wb = np.sort(np.clip(wa + off, 0.005, 0.995))
        if min(float(np.diff(wa).min(initial=1.0)),
               float(np.diff(wb).min(initial=1.0)), float(wa.min()),
               float(wb.min()), 1.0 - float(wa.max()),
               1.0 - float(wb.max())) < 0.01:
            continue                    # not an ORDINARY feature any more

        def mk(w):
            out, prev = [], 0.0
            for i, x in enumerate(list(w) + [1.0]):
                out.append((x - prev, EP if i % 2 else EH))
                prev = x
            return [s for s in out if s[0] > 0]

        cross = min(abs(float(x) - float(y)) for x in wa for y in wb)
        if cross < 0.01:                # walls this close ARE a sliver
            continue
        raw = raw_ratio([mk(wa), mk(wb)], 1e-12)
        if raw is not None:
            hits.append(raw["ratio"])
    h = np.array(hits)
    print(f"  {len(h)} ordinary non-conforming stacks with a cross-layer cell;"
          f" ratio max {h.max():.4g}, p99.9 {np.percentile(h, 99.9):.4g}, "
          f"p99 {np.percentile(h, 99):.4g}, median {np.median(h):.4g}")
    return dict(n=int(h.size), max=float(h.max()),
                p999=float(np.percentile(h, 99.9)),
                p99=float(np.percentile(h, 99)),
                median=float(np.median(h)))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r6_ratio.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()
    print("\n== A: the M2 coated taper's own ratio ==")
    a = part_a()
    print("\n== B: does a ratio-12 manufactured cell carry the defect? ==")
    b = part_b()
    print("\n== C: the ORDINARY non-conforming population ==")
    c = part_c()
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       m2=a, liner_ratio=b, ordinary=c,
                       wall_s=time.time() - t0), f, indent=1, default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
