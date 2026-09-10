"""W6 -- the RESONANT counter-fixture (round 2's open item R2-D).

R2-D: ``move > 100 * w_wide`` compares an EFFICIENCY difference with a PERIOD
fraction, i.e. it silently assumes ``dR/dx = O(1)``.  The round-2 report says
"a device with dR/dx above ~50 could in principle move a correct answer past
the bar.  No such device was found; a resonant fixture would be the way to
attack it."  This probe builds those fixtures.

Three families, each with its steepness DIALED and measured:

  GMR   a guided-mode-resonance grating: a weak-contrast (eps 3.6 / 4.0)
        grating on a high-index slab, at a wavelength on the resonance flank.
        The linewidth is ~0.5 nm, so ``dR/d(duty)`` is enormous.
  FP    a Fabry-Perot cavity: two identical gratings separated by a low-index
        spacer, at a spacer thickness where the cavity is on resonance.
  WOOD  a near-Wood-anomaly mount: the wavelength set just inside the
        Rayleigh cutoff of the substrate order.

The steepness statistic is the device's OWN smooth-regime continuity slope
``s = err / delta``, measured where the answer is stationary in BOTH degree
and ``min_feature`` -- the library's own stationarity advice used as the
oracle.  A row is then

    CORRECT for this device   err <=  3 s delta
    WRONG   for this device   err  >  30 s delta

and the two things being hunted are

  (a) a CORRECT row that the library REFUSES  -- a false refusal;
  (b) a WRONG row the library RETURNS as 'truncation' because the snapped
      super-unity SURVIVED -- a false attribution the other way.

    python w6_resonant.py [out.json]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lumenairy                                              # noqa: E402
from lumenairy.elements.pmm import PMMStack                   # noqa: E402
from lumenairy.elements.pmm import stack as ps                # noqa: E402
from w_fixtures import prescribed, shared_move, snapped       # noqa: E402
from w_fixtures import unguarded                              # noqa: E402

TRIG, CLOSURE, MOVE = 1.0e-3, 1.0e-5, 100.0


# --------------------------------------------------------------- fixtures ---
def gmr(d, deg, *, wl=1.7490e-6, duty=0.5, period=1.0e-6, th=0.10,
        e_lo=3.6, e_hi=4.0, t_gr=0.30e-6, t_slab=0.10e-6, nsub=1.45,
        nsup=1.0, mf=None, nl=2):
    """The GMR grating, split into ``nl`` z-slices whose walls open by ``d``
    -- so slice ``k``'s duty is ``duty + 2 d k / (nl-1)``."""
    a = 0.5 - duty / 2.0
    st = PMMStack(period, n_substrate=nsub, n_superstrate=nsup, degree=deg,
                  far_field_orders=15,
                  min_feature=(period * 1e-12 if mf is None else float(mf)))
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(t_gr / nl, segments=[(a - dd, e_lo),
                                          (duty + 2 * dd, e_hi),
                                          (1.0 - a - duty - dd, e_lo)])
    st.add_layer(t_slab, eps=e_hi)
    st.set_source(wl, theta=th)
    return st


def fabry(d, deg, *, wl=1.55e-6, duty=0.5, period=0.9e-6, th=0.12,
          e_hi=6.25, t_gr=0.14e-6, t_gap=0.62e-6, nsub=1.45, nsup=1.0,
          mf=None, nl=2):
    """Two gratings around a low-index spacer -- a cavity whose resonance
    moves with the gratings' duty."""
    a = 0.5 - duty / 2.0
    st = PMMStack(period, n_substrate=nsub, n_superstrate=nsup, degree=deg,
                  far_field_orders=15,
                  min_feature=(period * 1e-12 if mf is None else float(mf)))
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(t_gr / nl, segments=[(a - dd, 1.0),
                                          (duty + 2 * dd, e_hi),
                                          (1.0 - a - duty - dd, 1.0)])
    st.add_layer(t_gap, eps=1.0)
    st.add_layer(t_gr, segments=[(a, 1.0), (duty, e_hi),
                                 (1.0 - a - duty, e_hi * 0 + 1.0)])
    st.set_source(wl, theta=th)
    return st


FAMILIES = {"GMR": gmr, "FP": fabry}


def measure_slope(fn, deg, kw, deltas=(3e-3, 1e-3, 3e-4)):
    """The device's own smooth-regime continuity slope, taken where the
    answer is stationary in BOTH degree and min_feature (the library's own
    stationarity rule).  Returns ``(slope, stationary)``."""
    ref = unguarded(fn(0.0, deg, **kw))
    ss, ok = [], True
    for d in deltas:
        a = unguarded(fn(d, deg, **kw))
        b = unguarded(fn(d, deg + 6, **kw))
        refb = unguarded(fn(0.0, deg + 6, **kw))
        e = shared_move(a, ref, pol=1)
        eb = shared_move(b, refb, pol=1)
        ss.append(e / d)
        if eb == 0.0 or not (0.5 <= (e / d) / max(eb / d, 1e-300) <= 2.0):
            ok = False
    return float(np.median(ss)), ok


def guarded(fn, d, deg, kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn(d, deg, **kw).solve()
        except ValueError as exc:
            return ("REFUSED" if "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
                    else "raised"), [str(w.message) for w in rec]
    return "returned", [str(w.message) for w in rec]


def scan_family(name, fn, kws, degrees, deltas):
    rows = []
    for tag, kw in kws.items():
        for deg in degrees:
            try:
                slope, stat = measure_slope(fn, deg, kw)
                ref = unguarded(fn(0.0, deg, **kw))
            except (ValueError, NotImplementedError, RuntimeError):
                continue
            for d in deltas:
                try:
                    st = fn(float(d), deg, **kw)
                    cur = unguarded(st)
                except (ValueError, NotImplementedError, RuntimeError):
                    continue
                e = shared_move(cur, ref, pol=1)
                pre = prescribed(st)
                row = dict(fam=name, tag=tag, deg=deg, delta=float(d),
                           slope=slope, slope_stationary=bool(stat), err=e,
                           err_over_d=e / d,
                           err_over_slope_d=e / max(slope * d, 1e-300),
                           worst=cur["worst"],
                           has_sliver=pre is not None)
                if pre is not None and cur["worst"] - 1.0 > TRIG:
                    snp = snapped(st, pre["mf"])
                    row.update(su_snap=max(snp["worst"] - 1.0, 0.0),
                               move_ratio=(shared_move(cur, snp)
                                           / pre["w_wide"]),
                               w_wide=pre["w_wide"])
                v, warns = guarded(fn, float(d), deg, kw)
                row["verdict"] = v
                row["truncation_note"] = any("is NOT what moved" in w
                                             for w in warns)
                rows.append(row)
    return rows


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w6_resonant.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    t0 = time.time()
    rows = []
    # GMR: the resonance flank sampled, and the same device pushed into a
    # super-unity mount (dense superstrate, grazing, low degree).
    gmr_kws = {}
    for wl in (1.74880e-6, 1.74892e-6, 1.74900e-6, 1.74904e-6, 1.74908e-6,
               1.74924e-6, 1.74932e-6, 1.7490e-6):
        gmr_kws[f"wl{wl:.6e}"] = dict(wl=wl)
    for nsup, th in ((2.4, 1.22), (3.2, 1.33)):
        for wl in (1.749e-6, 1.55e-6, 0.93e-6):
            gmr_kws[f"sup{nsup}_th{th}_wl{wl:.3e}"] = dict(
                wl=wl, nsup=nsup, th=th, nsub=1.45 + 0.05j)
    for nl in (3, 4):
        gmr_kws[f"nl{nl}"] = dict(wl=1.7490e-6, nl=nl)
    deltas = list(np.geomspace(3e-3, 3e-6, 26))
    rows += scan_family("GMR", gmr, gmr_kws, (8, 12, 14, 18), deltas)
    print(f"  GMR: {len(rows)} rows, {time.time() - t0:.0f} s")
    fp_kws = {}
    for tg in (0.58e-6, 0.60e-6, 0.62e-6, 0.64e-6, 0.66e-6):
        fp_kws[f"gap{tg:.2e}"] = dict(t_gap=tg)
    for nsup, th in ((2.4, 1.22),):
        fp_kws[f"sup{nsup}"] = dict(nsup=nsup, th=th, nsub=1.45 + 0.05j)
    rows += scan_family("FP", fabry, fp_kws, (8, 12, 16), deltas)
    print(f"  +FP: {len(rows)} rows, {time.time() - t0:.0f} s")

    steep = [r for r in rows if r["slope"] > 20 and r["slope_stationary"]]
    refused = [r for r in rows if r["verdict"] == "REFUSED"]
    false_refusals = [r for r in refused
                      if r["err_over_slope_d"] <= 3.0]
    false_truncations = [
        r for r in rows
        if r["verdict"] == "returned" and r["truncation_note"]
        and r["err_over_slope_d"] > 30.0]
    near_miss = sorted((r for r in rows if r.get("move_ratio") is not None
                        and r["err_over_slope_d"] <= 3.0),
                       key=lambda r: -r["move_ratio"])[:15]
    summary = dict(
        lumenairy=lib, python=sys.version.split()[0], numpy=np.__version__,
        n_rows=len(rows), wall=time.time() - t0,
        max_slope=max(r["slope"] for r in rows),
        max_slope_stationary=max((r["slope"] for r in rows
                                  if r["slope_stationary"]), default=None),
        n_steep_rows=len(steep), n_refused=len(refused),
        n_false_refusals=len(false_refusals),
        n_false_truncations=len(false_truncations),
        n_arbitrated=sum(1 for r in rows if r.get("move_ratio") is not None),
        max_move_ratio_on_correct=max(
            (r["move_ratio"] for r in rows
             if r.get("move_ratio") is not None
             and r["err_over_slope_d"] <= 3.0), default=None),
        min_move_ratio_on_wrong=min(
            (r["move_ratio"] for r in rows
             if r.get("move_ratio") is not None
             and r["err_over_slope_d"] > 30.0), default=None),
        false_refusals=false_refusals[:20],
        false_truncations=false_truncations[:20],
        near_miss=near_miss)
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, rows=rows), fh, indent=1)
    for k, v in summary.items():
        if k in ("false_refusals", "false_truncations", "near_miss"):
            print(f"  {k}: {len(v)}")
            for r in v[:8]:
                print(f"    {r['fam']}/{r['tag']} deg {r['deg']} "
                      f"d={r['delta']:.3e} slope={r['slope']:.1f} "
                      f"err/(s d)={r['err_over_slope_d']:.2f} "
                      f"R+T-1={r['worst'] - 1:+.3e} "
                      f"su_snap={r.get('su_snap')} "
                      f"move={r.get('move_ratio')} -> {r['verdict']}")
        else:
            print(f"  {k:30s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
