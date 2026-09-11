"""ROUND 4, P2 driver -- run the candidate measurement over every population
the decision has to be right on, and write one JSON per (build, kernel) arm.

POPULATIONS
  ladder    the five two-slice staircases of round 3 (continuity slopes 0.47
            .. 31.4) x degrees 12/14/20 x 22 wall steps -- the CORRECT and
            WRONG populations the guard is scored against.
  census    the 648-configuration realistic staircase box, sub-sampled -- the
            round-1 FALSE-POSITIVE population (every row correct), and the
            only multi-slice family here.
  d5        the five D-5 mounts of round 3 (GMR, grazing staircase,
            Fabry-Perot, near-Wood) -- restorable WRONG answers.
  v4        the degree-4 mount of verification defect V-4 -- three CORRECT
            answers rounds 2 and 3 both REFUSE.
  steep     the high-contrast-grating Fano resonance and the many-slice
            taper -- CORRECT rows whose move/w_wide runs past the shipped bar.
  tensor    the in-plane and out-of-plane liquid-crystal directors.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_fixtures as G  # noqa: E402
import numpy as np  # noqa: E402
import p2_candidates as C  # noqa: E402
import v_fixtures as V  # noqa: E402

G.assert_tree()

LAD_DELTAS = [float(x) for x in np.geomspace(3e-3, 1e-6, 22)]
D5_DELTAS = [float(x) for x in np.geomspace(1e-4, 1e-6, 14)]


def ladder(rows):
    for name, cfg in G.FIXTURES.items():
        for deg in (12, 14, 20):
            ref = G.unguarded(G.wbuild(0.0, deg, **cfg))
            for d in LAD_DELTAS:
                rows.append(C.measure(G.wbuild(d, deg, **cfg), ref, d,
                                      label=f"ladder:{name}", degree=deg))
        print("ladder", name, flush=True)


def census(rows):
    subs = [complex(1.45, 0.08), complex(2.0, 0.35), complex(3.4, 1.7)]
    for nsub in subs:
        for nsup in (2.4, 3.2):
            for th in (1.22, 1.44):
                for deg in (6, 10):
                    for nl in (2, 4):
                        ref = G.unguarded(G.cbuild(0.0, deg, nsub, nsup, th,
                                                   nl, 10.5))
                        for d in (3e-3, 1e-3, 3e-4):
                            rows.append(C.measure(
                                G.cbuild(d, deg, nsub, nsup, th, nl, 10.5),
                                ref, d, label="census", degree=deg))
        print("census", nsub, flush=True)


D5 = {
    "gmr_deg6": (V.vgmr, dict(degree=6)),
    "gmr_deg8": (V.vgmr, dict(degree=8)),
    "fp_deg8": (V.vfp, dict(degree=8)),
    "wood_deg8": (V.vwood, dict(degree=8)),
    "graze_deg6": (lambda d, **k: V.vstair(
        d, period=1.18e-6, wl=0.72e-6, theta=1.31, a0=0.2870, b0=0.7150,
        e_lo=3.24, e_hi=4.41, dz=0.19e-6, nl=2, nsup=2.28,
        nsub=complex(1.46, 0.06), ffo=15, **k), dict(degree=6)),
    "graze_deg8": (lambda d, **k: V.vstair(
        d, period=1.18e-6, wl=0.72e-6, theta=1.31, a0=0.2870, b0=0.7150,
        e_lo=3.24, e_hi=4.41, dz=0.19e-6, nl=2, nsup=2.28,
        nsub=complex(1.46, 0.06), ffo=15, **k), dict(degree=8)),
}


def d5(rows):
    for name, (fn, kw) in D5.items():
        ref = G.unguarded(fn(0.0, **kw))
        for d in D5_DELTAS:
            rows.append(C.measure(fn(d, **kw), ref, d, label=f"d5:{name}",
                                  degree=kw["degree"]))
        print("d5", name, flush=True)


def _v4(d, **kw):
    return V.vstair(d, period=1.02e-6, wl=0.633e-6, theta=1.35, a0=0.3120,
                    b0=0.6790, e_lo=2.56, e_hi=12.25, dz=0.12e-6, nl=3,
                    nsup=3.10, nsub=complex(2.90, 1.10), ffo=21, **kw)


def v4(rows):
    ref = G.unguarded(_v4(0.0, degree=4))
    for d in [1.6622e-05, 1.2690e-05, 7.3955e-06] + \
             [float(x) for x in np.geomspace(3e-5, 3e-6, 10)]:
        rows.append(C.measure(_v4(d, degree=4), ref, d, label="v4",
                              degree=4))
    print("v4", flush=True)


def steep(rows):
    for deg in (12, 14):
        ref = G.unguarded(V.vhcg(0.0, degree=deg, duty=0.7203))
        for d in [float(x) for x in np.geomspace(1e-3, 1e-6, 12)]:
            rows.append(C.measure(V.vhcg(d, degree=deg, duty=0.7203), ref, d,
                                  label="steep:hcg", degree=deg))
    cfg = dict(G.FIXTURES["O11"])
    for nl in (6, 12):
        for deg in (10, 14):
            ref = G.unguarded(G.wbuild(0.0, deg, nl=nl, **cfg))
            for d in [float(x) for x in np.geomspace(3e-3, 1e-5, 10)]:
                rows.append(C.measure(G.wbuild(d, deg, nl=nl, **cfg), ref, d,
                                      label=f"steep:taper{nl}", degree=deg))
    print("steep", flush=True)


def tensor(rows):
    for name, fn in (("tensor", V.vtensor), ("oop", V.voop)):
        ref = G.unguarded(fn(0.0))
        for d in [float(x) for x in np.geomspace(3e-4, 1e-6, 12)]:
            rows.append(C.measure(fn(d), ref, d, label=f"tensor:{name}",
                                  degree=10))
        print("tensor", name, flush=True)


PARTS = dict(ladder=ladder, census=census, d5=d5, v4=v4, steep=steep,
             tensor=tensor)


def main():
    want = sys.argv[1:] or list(PARTS)
    rows = []
    for k in want:
        PARTS[k](rows)
    G.dump(dict(rows=rows, parts=want), "p2_" + "_".join(want)
           if len(want) < len(PARTS) else "p2_all")


if __name__ == "__main__":
    main()
