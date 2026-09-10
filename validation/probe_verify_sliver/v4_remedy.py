"""VERIFY task 4 -- the REMEDY, and open item A (the asymmetric snap).

A. THE REMEDY.  On TWO fixtures of my own (a 0.9 um period / 0.62 um and a
   1.55 um period / 1.31 um telecom one) at four degrees each, the
   ``min_feature`` the refusal prescribes is scored against the EXACT
   ``delta -> 0`` reference: ``err``, ``err/delta`` and the ``2 delta`` bar,
   plus the snapped solve's own closure.  The O-11 fixture is scored too, so
   the audit's 1.152-1.154 slope has a same-fixture arm.

B. OPEN ITEM A -- the ASYMMETRIC SNAP.  ``_pmm_union_grid`` merges a pair when
   ``d < mf`` with ``d`` the FLOATING separation.  For a symmetric geometry
   (both wall pairs nominally ``delta`` apart) the two computed ``d`` straddle
   ``mf`` by ~1e-10 relative, so one pair merges and the other does not: a
   geometry nobody asked for.  Measured here: how far the two separations sit
   from ``mf``, how many ``delta`` in a sweep produce an ODD merge count, and
   what the asymmetric grid does to the ANSWER.

C. CANDIDATE FIX.  A local re-implementation of the snap loop with a RELATIVE
   slack, scored for (i) does it cure the asymmetry, (ii) is it bit-identical
   off the threshold.

    python validation/probe_verify_sliver/v4_remedy.py [out.json]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps
from lumenairy.elements.pmm._core import _pmm_union_grid

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, period, wl, theta, dz, eps_lo, eps_hi, wall_a, wall_b, n_sup, n_sub)
FIXTURES = [
    ("o11", 1.2e-6, 0.85e-6, 0.15, 0.32e-6 / 4, 2.25, 9.0, 0.27865, 0.62505,
     1.0, 1.0),
    ("mine_vis", 0.9e-6, 0.62e-6, 0.21, 70e-9, 2.0, 6.5, 0.311, 0.688,
     1.0, 1.46),
    ("mine_tel", 1.55e-6, 1.31e-6, 0.08, 110e-9, 2.1, 11.9, 0.1907, 0.5533,
     1.5, 3.48),
]


def _segs(a, b, eh, ep):
    return [(a, eh), (b - a, ep), (1.0 - b, eh)]


def _solve(fx, d, deg, mf=None, guard=False):
    _n, P, WL, TH, DZ, EH, EP, A, B, NS, NB = fx
    kw = {} if mf is None else dict(min_feature=mf)
    st = PMMStack(P, n_superstrate=NS, n_substrate=NB, degree=deg, **kw)
    for (a, b) in [(A, B), (A - d, B + d)]:
        st.add_layer(DZ, segments=_segs(a, b, EH, EP))
    st.set_source(WL, theta=TH)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot))


def _err(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


NO_SNAP = 1e-12       # FRACTION -- below _pmm_union_grid's own 1e-9 dedup tol


# ==========================================================================
def part_a():
    rows = []
    for fx in FIXTURES:
        name, P = fx[0], fx[1]
        for deg in (10, 12, 14, 18):
            ref = _solve(fx, 0.0, deg, mf=P * NO_SNAP)
            for d in (1e-4, 5e-5, 3e-5, 1e-5):
                # what would the refusal prescribe here?
                try:
                    _solve(fx, d, deg, mf=P * NO_SNAP, guard=True)
                    continue                      # not in the band
                except ValueError as exc:
                    if "NEAR-COINCIDENT-WALL SLIVER" not in str(exc):
                        raise
                    mf = float(str(exc).split("min_feature=")[1].split(" ")[0])
                fixed = _solve(fx, d, deg, mf=mf, guard=True)
                e = _err(fixed, ref)
                rows.append(dict(fixture=name, degree=deg, delta=d,
                                 prescribed_mf_m=mf, err=e, err_over_delta=e / d,
                                 bar_2delta=2.0 * d, within_bar=bool(e <= 2.0 * d),
                                 RplusT=fixed[3],
                                 closure=abs(fixed[3] - 1.0)))
                print(f"  {name:9s} deg {deg:2d} delta {d:.1e} mf {mf:.4g} "
                      f"err {e:.4e} err/delta {e/d:.4f} "
                      f"<=2delta {rows[-1]['within_bar']} "
                      f"R+T {fixed[3]:.8f}", flush=True)
    slopes = [r["err_over_delta"] for r in rows]
    print(f"  rows {len(rows)}; err/delta in [{min(slopes):.4f}, "
          f"{max(slopes):.4f}]; within 2delta "
          f"{sum(r['within_bar'] for r in rows)}/{len(rows)}; "
          f"max closure {max(r['closure'] for r in rows):.3e}")
    return rows


# ==========================================================================
def _merge_census(fx, d, mf_frac):
    """(n_union_cells, the two pair separations, how many merged)."""
    _n, P, _WL, _TH, _DZ, EH, EP, A, B, _NS, _NB = fx
    segs = [_segs(A, B, EH, EP), _segs(A - d, B + d, EH, EP)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uw, _e = _pmm_union_grid(segs, mf_frac)
    sep_left = A - (A - d)
    sep_right = (B + d) - B
    return dict(n_cells=int(uw.size), sep_left=float(sep_left),
                sep_right=float(sep_right),
                left_lt_mf=bool(sep_left < mf_frac),
                right_lt_mf=bool(sep_right < mf_frac),
                merged=5 - int(uw.size))


def part_b():
    out = {}
    for fx in FIXTURES:
        name, P = fx[0], fx[1]
        mf_frac = 1e-5                       # the LIBRARY DEFAULT fraction
        rows = []
        # a fine multiplicative walk of delta straight through mf
        for f in np.geomspace(0.5, 2.0, 121):
            d = float(mf_frac * f)
            c = _merge_census(fx, d, mf_frac)
            c["delta"] = d
            c["ratio_to_mf"] = f
            rows.append(c)
        odd = [r for r in rows if r["merged"] == 1]
        both = [r for r in rows if r["merged"] == 2]
        none = [r for r in rows if r["merged"] == 0]
        # what does the asymmetry do to the answer at delta == mf exactly?
        d0 = mf_frac
        ref = _solve(fx, 0.0, 14, mf=P * NO_SNAP)
        asym = _solve(fx, d0, 14)                 # library DEFAULT min_feature
        sym = _solve(fx, d0, 14, mf=P * 2.0 * d0)  # both pairs snapped
        raw = _solve(fx, d0, 14, mf=P * NO_SNAP)   # neither snapped
        out[name] = dict(
            n_rows=len(rows), n_odd=len(odd), n_both=len(both), n_none=len(none),
            odd_span=[min(r["ratio_to_mf"] for r in odd),
                      max(r["ratio_to_mf"] for r in odd)] if odd else None,
            at_mf=_merge_census(fx, d0, mf_frac),
            err_asym=_err(asym, ref), RT_asym=asym[3],
            err_sym=_err(sym, ref), RT_sym=sym[3],
            err_raw=_err(raw, ref), RT_raw=raw[3],
            rows=rows)
        c = out[name]["at_mf"]
        print(f"  {name:9s} delta==mf: cells {c['n_cells']} merged "
              f"{c['merged']}; sep_left {c['sep_left']!r} (<mf {c['left_lt_mf']}) "
              f"sep_right {c['sep_right']!r} (<mf {c['right_lt_mf']})")
        print(f"      over 121 deltas 0.5..2.0 x mf: merged 0 in "
              f"{len(none)}, 1 in {len(odd)} (ASYMMETRIC), 2 in {len(both)}"
              + (f"; asymmetric span {out[name]['odd_span'][0]:.6f}"
                 f"..{out[name]['odd_span'][1]:.6f} x mf" if odd else ""))
        print(f"      answers at delta==mf: asym err {out[name]['err_asym']:.4e} "
              f"(R+T {asym[3]:.6g}) | both-snapped err {out[name]['err_sym']:.4e} "
              f"(R+T {sym[3]:.6g}) | unsnapped err {out[name]['err_raw']:.4e} "
              f"(R+T {raw[3]:.6g})")
    return out


# ==========================================================================
def _union_grid_patched(layer_segments, min_feature=None, *, rel=1e-9):
    """The shipped loop with a RELATIVE slack on the threshold, re-implemented
    locally so the library is not touched while the candidate is scored."""
    walls = {0.0, 1.0}
    wall_owners = {0.0: set(), 1.0: set()}
    cums = []
    for li, segs in enumerate(layer_segments):
        w = np.asarray([float(s[0]) for s in segs], dtype=float)
        cw = np.concatenate([[0.0], np.cumsum(w)])
        cw[-1] = 1.0
        cums.append(cw)
        for x in cw:
            x = float(x)
            walls.add(x)
            wall_owners.setdefault(x, set()).add(li)
    uwalls = np.array(sorted(walls))
    if uwalls.size > 2:
        tol = 1e-9
        keep = [uwalls[0]]
        owners = [wall_owners.get(float(uwalls[0]), set())]
        for w in uwalls[1:]:
            if w - keep[-1] > tol:
                keep.append(w)
                owners.append(wall_owners.get(float(w), set()))
            else:
                owners[-1] = owners[-1] | wall_owners.get(float(w), set())
        if keep[-1] < uwalls[-1]:
            keep[-1] = uwalls[-1]
        if min_feature is not None and float(min_feature) > tol:
            mf = float(min_feature) * (1.0 + rel)          # <-- the candidate
            out_w, out_o = [keep[0]], [owners[0]]
            for w, ow in zip(keep[1:], owners[1:]):
                d = w - out_w[-1]
                interior = 0.0 < out_w[-1] and w < 1.0
                if d < mf and interior and not (out_o[-1] & ow):
                    out_w[-1] = 0.5 * (out_w[-1] + w)
                    out_o[-1] = out_o[-1] | ow
                else:
                    out_w.append(w)
                    out_o.append(ow)
            keep = out_w
        uwalls = np.array(keep)
    return np.diff(uwalls)


def part_c():
    """Score the relative-slack candidate: does it cure the asymmetry, and is
    it bit-identical off the threshold?"""
    out = {}
    for fx in FIXTURES:
        name = fx[0]
        EH, EP, A, B = fx[5], fx[6], fx[7], fx[8]
        mf_frac = 1e-5
        cured, ident, tested = 0, 0, 0
        odd_after = []
        for f in np.geomspace(0.5, 2.0, 121):
            d = float(mf_frac * f)
            segs = [_segs(A, B, EH, EP), _segs(A - d, B + d, EH, EP)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base = _pmm_union_grid(segs, mf_frac)[0]
            new = _union_grid_patched(segs, mf_frac)
            tested += 1
            if np.array_equal(base, new):
                ident += 1
            if int(new.size) == 4:
                odd_after.append(f)
            if int(base.size) == 4 and int(new.size) != 4:
                cured += 1
        # off-threshold bit-identity on a WIDE sweep (0.01x .. 100x mf) and on
        # several min_feature values
        off_ident, off_tested, off_diff = 0, 0, []
        for mfq in (1e-5, 3e-5, 1e-4, 1e-3):
            for f in np.geomspace(1e-2, 1e2, 400):
                d = float(mfq * f)
                if 0.999 < f < 1.001:
                    continue                         # the threshold itself
                segs = [_segs(A, B, EH, EP), _segs(A - d, B + d, EH, EP)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base = _pmm_union_grid(segs, mfq)[0]
                new = _union_grid_patched(segs, mfq)
                off_tested += 1
                if np.array_equal(base, new):
                    off_ident += 1
                else:
                    off_diff.append((mfq, f, int(base.size), int(new.size)))
        out[name] = dict(threshold_rows=tested, cured=cured,
                         still_odd=len(odd_after),
                         off_threshold_rows=off_tested,
                         off_threshold_identical=off_ident,
                         off_threshold_diffs=off_diff[:10])
        print(f"  {name:9s} threshold walk: cured {cured}/{tested}, "
              f"still asymmetric {len(odd_after)}")
        print(f"      OFF-threshold (0.01..100 x mf, 4 min_features, "
              f"{off_tested} rows): bit-identical {off_ident}/{off_tested}"
              + ("" if not off_diff else f"; first diffs {off_diff[:3]}"))
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v4_remedy.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    print("\n== A: the prescribed min_feature vs the exact delta -> 0 limit ==")
    a = part_a()
    print("\n== B: open item A -- the ASYMMETRIC snap ==")
    b = part_b()
    print("\n== C: the relative-slack candidate ==")
    c = part_c()
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       part_a=a, part_b=b, part_c=c), f, indent=1, default=str)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
