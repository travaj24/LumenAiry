"""ROUND 2, probe 9 -- the ARBITER on every path, and what it costs.

The verification's S10 listed five things it could not check.  Round 2 wires a
RE-SOLVE into the guard, so each of them is now a live question -- can the
arbiter run there, and does it re-solve at the RIGHT physics?

A. ``solve()``: classical, CONICAL, slant, per-layer window grids,
   ``stabilize='slices'``.
B. ``solve_vs_wavelength``: the sweep never writes ``stack._src``, so the
   arbiter is handed the sweep's own wavelength EXPLICITLY -- asserted by
   spying on the probe.  Also the dispersive case, where the re-solve cannot
   be materialised and the verdict must be ``'unknown'`` (round-1 behaviour).
C. ``prepare().solve``: ``set_source`` is never required there.
D. COST: how long one arbitration takes, against the solve it guards.

    python validation/probe_pmmstack_sliver_round2/r9_paths.py [out.json]
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
P, WL, TH = 1.2e-6, 0.85e-6, 0.15
DZ = 0.32e-6 / 4
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505
NO_SNAP = P * 1e-12


def build(d, degree=14, *, phi=0.0, slant=0.0, per_layer=False, nl=2,
          eps=EP, source=True):
    kw = dict(layer_grids="per-layer") if per_layer else {}
    st = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=degree,
                  min_feature=NO_SNAP, **kw)
    for k in range(nl):
        dd = d * (k % 2)
        a, b = A0 - dd, B0 + dd
        st.add_layer(DZ, segments=[(a, EH), (b - a, eps), (1.0 - b, EH)],
                     slant_angle=slant)
    if source:
        st.set_source(WL, theta=TH, phi=phi)
    return st


def attempt(fn):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fn()
        return dict(refused=False,
                    warned=[str(w.message)[:70] for w in rec][:2])
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        return dict(refused=True,
                    sliver="NEAR-COINCIDENT-WALL SLIVER" in str(exc),
                    attributed="ATTRIBUTION, MEASURED ON THIS CALL" in str(exc),
                    msg=str(exc)[:120])
    finally:
        ps.PMM_SLIVER_GUARD = was


def spy():
    """Wrap the probe so the wavelength it is handed can be read back."""
    seen = []
    real = ps._sliver_probe_solve

    def _s(stack, mf, src):
        seen.append(None if src is None else float(src["wl"]))
        return real(stack, mf, src)
    ps._sliver_probe_solve = _s
    return seen, real


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r9_paths.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    out = {}

    print("\n== A: solve() routings ==")
    rows = []
    for tag, kw in (("classical", {}),
                    ("conical phi=0.62", dict(phi=0.62)),
                    ("slant 0.17", dict(slant=0.17)),
                    ("per-layer grids, 5 layers", dict(per_layer=True, nl=5))):
        for d, expect in ((1e-4, "?"), (3e-5, "?"), (3e-3, "return")):
            seen, real = spy()
            try:
                r = attempt(lambda: build(d, **kw).solve())
            finally:
                ps._sliver_probe_solve = real
            rows.append(dict(path=tag, delta=d, probed=len(seen), **r))
            print(f"  {tag:26s} d {d:.0e}: probe run {len(seen)}x -> "
                  f"{'REFUSED' if r['refused'] else 'returned'}"
                  f"{' (attributed)' if r.get('attributed') else ''}"
                  f"{'  ' + (r.get('msg') or '')[:60] if r['refused'] and not r.get('sliver') else ''}")
            del expect
    # stabilize='slices' -- the guard raises before the consensus probe
    seen, real = spy()
    try:
        r = attempt(lambda: build(1e-4).solve(stabilize="slices"))
    finally:
        ps._sliver_probe_solve = real
    rows.append(dict(path="stabilize='slices'", delta=1e-4, probed=len(seen),
                     **r))
    print(f"  {'stabilize=slices':26s} d 1e-04: probe run {len(seen)}x -> "
          f"{'REFUSED' if r['refused'] else 'returned'}")
    out["solve_paths"] = rows

    print("\n== B: solve_vs_wavelength ==")
    st = build(1e-4)
    st.set_source(0.4e-6, theta=TH)          # a STALE, different source
    seen, real = spy()
    try:
        r = attempt(lambda: st.solve_vs_wavelength([WL], angle=TH,
                                                   max_workers=1))
    finally:
        ps._sliver_probe_solve = real
    print(f"  sweep at {WL:g} with a stale set_source(4e-07): probe was handed "
          f"{seen} -> {'REFUSED' if r['refused'] else 'returned'}")
    out["sweep"] = dict(handed=seen, stale_src=0.4e-6, **r)
    assert seen == [WL], seen

    # the DISPERSIVE sweep: the re-solve cannot materialise the callables
    std = build(1e-4, source=False)
    std._layers = [(t, [(w, (lambda wl, e=e: e)) for w, e in segs], sl)
                   for t, segs, sl in std._layers]
    seen, real = spy()
    try:
        rd = attempt(lambda: std.solve_vs_wavelength([WL], angle=TH,
                                                     max_workers=1))
    finally:
        ps._sliver_probe_solve = real
    passive = ps._stack_provably_passive(std)
    print(f"  dispersive sweep: provably_passive={passive} -> the geometric "
          f"screen is never reached (a callable eps cannot be resolved), so "
          f"probe run {len(seen)}x and the solve "
          f"{'REFUSED' if rd['refused'] else 'RETURNED under the plain warning'}"
          f" -- unchanged from round 1")
    out["sweep_dispersive"] = dict(probed=len(seen), provably_passive=passive,
                                   **rd)

    print("\n== C: prepare().solve ==")
    stp = build(1e-4, source=False)
    prep = stp.prepare()
    seen, real = spy()
    try:
        rp = attempt(lambda: prep.solve(wavelength=WL, angle=TH))
    finally:
        ps._sliver_probe_solve = real
    print(f"  prepared (no set_source at all): probe was handed {seen} -> "
          f"{'REFUSED' if rp['refused'] else 'returned'}"
          f"{' (attributed)' if rp.get('attributed') else ''}")
    out["prepared"] = dict(handed=seen, **rp)

    print("\n== D: cost ==")
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    t0 = time.perf_counter()
    for _ in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(1e-4).solve()
    t_solve = (time.perf_counter() - t0) / 10.0
    ps.PMM_SLIVER_GUARD = was
    st = build(1e-4)
    rec = ps._sliver_screen(st)
    mf = 2.0 * rec[0][3] * P
    t0 = time.perf_counter()
    for _ in range(10):
        ps._sliver_probe_solve(st, mf, dict(st._src))
    t_arb = (time.perf_counter() - t0) / 10.0
    print(f"  one guarded solve  {t_solve * 1e3:.1f} ms")
    print(f"  one ARBITRATION    {t_arb * 1e3:.1f} ms  "
          f"({t_arb / t_solve:.2f}x the solve it guards)")
    out["cost"] = dict(solve_s=t_solve, arbiter_s=t_arb,
                       ratio=t_arb / t_solve)

    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__), **out), f, indent=1,
                  default=str)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
