"""ROUND 2, probe 3 -- the TRIGGER bar as a FAMILY property, three fixtures.

Round 1's bar (1e-2) was derived on one 46-point sample of one fixture and the
verification refuted the margins on a denser grid of the SAME fixture.  Rule 5
of TESTING_STANDARDS asks for the family's envelope, not the sample's, so this
measures both populations on THREE fixtures -- the O-11 stack, a visible-band
one and a telecom one, which read continuity slopes 1.15 / 4.44 / 1.04 -- and
scores the round-2 trigger ladder on all of them together.

For every row that reads super-unity above the lowest trigger considered, the
ARBITER is run: one re-solve on the prescribed ``min_feature`` grid.

    python validation/probe_pmmstack_sliver_round2/r3_trigger.py [out.json]
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
TRIGGERS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
LOW = min(TRIGGERS)

# (name, period, wl, theta, dz, eps_lo, eps_hi, wall_a, wall_b, n_sup, n_sub)
FIXTURES = [
    ("o11", 1.2e-6, 0.85e-6, 0.15, 0.32e-6 / 4, 2.25, 9.0, 0.27865, 0.62505,
     1.0, 1.0),
    ("vis", 0.9e-6, 0.62e-6, 0.21, 70e-9, 2.0, 6.5, 0.311, 0.688, 1.0, 1.46),
    ("tel", 1.55e-6, 1.31e-6, 0.08, 110e-9, 2.1, 11.9, 0.1907, 0.5533,
     1.5, 3.48),
]


def build(fx, d, deg, mf=None):
    _n, P, WL, TH, DZ, EH, EP, A, B, NS, NB = fx
    st = PMMStack(P, n_superstrate=NS, n_substrate=NB, degree=deg,
                  min_feature=(P * 1e-12 if mf is None else mf))
    for (a, b) in [(A, B), (A - d, B + d)]:
        st.add_layer(DZ, segments=[(a, EH), (b - a, EP), (1.0 - b, EH)])
    st.set_source(WL, theta=TH)
    return st


def raw(st):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return (o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot)),
            np.asarray(R)[:, i], np.asarray(T)[:, i])


def err(a, b):
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[1][ia] - b[1][ib]).max(),
                     np.abs(a[2][ia] - b[2][ib]).max()))


def move_both(a, b):
    """The move the LIBRARY computes: BOTH incident polarizations."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[4][:, ia] - b[4][:, ib]).max(),
                     np.abs(a[5][:, ia] - b[5][:, ib]).max()))


def prescribed(st):
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 float(st.min_feature) / float(st.period))
    return None if hit is None else 2.0 * hit[3] * float(st.period)


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r3_trigger.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()
    deltas = [float(x) for x in np.geomspace(3e-3, 1e-6, 80)]
    rows = []
    for fx in FIXTURES:
        for deg in (10, 14, 18):
            ref = raw(build(fx, 0.0, deg))
            for d in deltas:
                st = build(fx, d, deg)
                base = raw(st)
                e = err(base, ref)
                r = dict(fixture=fx[0], degree=deg, delta=d, err=e,
                         err_over_delta=e / d, RplusT=base[3],
                         kind=("wrong" if e > 100.0 * d else
                               "right" if e <= 10.0 * d else "grey"))
                mf = prescribed(st)
                r["screen_hit"] = mf is not None
                r["passive"] = bool(ps._stack_provably_passive(st))
                if mf is not None and r["passive"] and base[3] > 1.0 + LOW:
                    snap = raw(build(fx, d, deg, mf=mf))
                    r["snapped_RplusT"] = snap[3]
                    r["move"] = err(base, snap)
                    r["move_both"] = move_both(base, snap)
                rows.append(r)
        print(f"  fixture {fx[0]} done ({time.time() - t0:.0f} s)", flush=True)

    summ = {}
    for name in [f[0] for f in FIXTURES] + ["ALL"]:
        sel = [r for r in rows if name == "ALL" or r["fixture"] == name]
        right = [abs(r["RplusT"] - 1.0) for r in sel if r["kind"] == "right"]
        wrong = [r["RplusT"] - 1.0 for r in sel if r["kind"] == "wrong"]
        summ[name] = dict(n=len(sel), n_right=len(right), n_wrong=len(wrong),
                          n_grey=sum(1 for r in sel if r["kind"] == "grey"),
                          max_absRT1_right=max(right) if right else None,
                          min_RT1_wrong=min(wrong) if wrong else None)
        print(f"  {name:4s} n={len(sel):4d} right {len(right):3d} wrong "
              f"{len(wrong):3d} grey {summ[name]['n_grey']:2d}  "
              f"max|R+T-1|_right {summ[name]['max_absRT1_right']:.4e}  "
              f"min(R+T-1)_wrong {summ[name]['min_RT1_wrong']:.4e}")

    table = {}
    for trig in TRIGGERS:
        rw = rr = rg = tw = tg = 0
        for r in rows:
            fires = (r["screen_hit"] and r["passive"]
                     and r["RplusT"] > 1.0 + trig)
            att = fires and r.get("snapped_RplusT", 9e9) <= 1.0 + trig
            if att:
                rw += r["kind"] == "wrong"
                rr += r["kind"] == "right"
                rg += r["kind"] == "grey"
            else:
                tw += r["kind"] == "wrong"
                tg += r["kind"] == "grey"
        table[f"{trig:g}"] = dict(refused_wrong=rw, refused_right=rr,
                                  refused_grey=rg, returned_wrong=tw,
                                  returned_grey=tg)
        print(f"  trigger {trig:g}: refuse wrong {rw}/{summ['ALL']['n_wrong']}"
              f" RIGHT {rr}/{summ['ALL']['n_right']} grey {rg}  |  "
              f"RETURNED wrong {tw} grey {tg}")

    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       rows=rows, summary=summ, decisions=table,
                       wall_s=time.time() - t0), f, indent=1, default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
