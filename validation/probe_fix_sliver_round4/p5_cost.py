"""ROUND 4, P5 -- what the three extra solves cost, and where they are paid.

Round 2 measured its ONE arbitration at 0.26x of the solve it guards, paid
only on a stack that already read super-unity above the trigger.  Round 4 pays
THREE, and pays them on every stack the GEOMETRIC screen fires on -- which is
the whole point, and is therefore the number that has to be stated rather than
described.

Two measurements, both on the wall clock and both repeated:

* the RATIO, on a stack that IS screened: a guarded solve against the same
  solve with ``PMM_SLIVER_GUARD = False``.  The three probe grids are COARSER
  than the sliver grid (the snap and the closure both remove cells), so each
  probe is cheaper than the solve it guards;
* the FLOOR, on a stack that is NOT screened: the guard must cost nothing
  measurable there, because the screen is pure geometry and answers ``None``.

The third number is the FIRING RATE -- how many of a realistic population are
screened at all -- because a ratio on a rare stack and a ratio on a common one
are different costs.
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_fixtures as G  # noqa: E402
import p3_bars as P  # noqa: E402

from lumenairy.elements.pmm import stack as ps  # noqa: E402

G.assert_tree()
REPS = 5


def _time(build, guard):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    best = float("inf")
    try:
        for _ in range(REPS):
            st = build()
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    st.solve()
                except ValueError:
                    pass
            best = min(best, time.perf_counter() - t0)
    finally:
        ps.PMM_SLIVER_GUARD = was
    return best


def main():
    cfg = G.FIXTURES["O11"]
    out = {}
    cases = {
        # SCREENED: a manufactured sliver is present
        "o11_deg14_d1e-4": lambda: G.wbuild(1e-4, 14, **cfg),
        "o11_deg20_d1e-5": lambda: G.wbuild(1e-5, 20, **cfg),
        "census_deg6_nl4": lambda: G.cbuild(1e-3, 6, complex(1.45, 0.08), 2.4,
                                            1.22, 4, 10.5),
        "taper16_deg8": lambda: G.wbuild(3e-4, 8, nl=16, **cfg),
        # NOT screened: nothing manufactured (own/w = 92.7 < 100), and the
        # coincident-wall limit
        "o11_deg14_d3e-3": lambda: G.wbuild(3e-3, 14, **cfg),
        "o11_deg14_d0": lambda: G.wbuild(0.0, 14, **cfg),
    }
    for name, build in sorted(cases.items()):
        st = build()
        screened = ps._sliver_screen(st, require_passive=False) is not None
        off = _time(build, False)
        on = _time(build, True)
        out[name] = dict(screened=screened, guard_off_s=off, guard_on_s=on,
                         ratio=on / off, extra_s=on - off)
        print(f"{name:20s} screened={screened!s:5s} off={off * 1e3:8.2f} ms "
              f"on={on * 1e3:8.2f} ms  ratio={on / off:6.3f}", flush=True)

    # the FIRING RATE on the realistic staircase box: how much of a real
    # population pays anything at all
    n = fired = 0
    for nsub in (complex(1.45, 0.08), complex(2.0, 0.35), complex(3.4, 1.7)):
        for nsup in (2.4, 3.2):
            for th in (1.22, 1.44):
                for deg in (6, 10):
                    for nl in (2, 4):
                        for d in (3e-3, 1e-3, 3e-4):
                            st = G.cbuild(d, deg, nsub, nsup, th, nl, 10.5)
                            n += 1
                            if ps._sliver_screen(st,
                                                 require_passive=False):
                                fired += 1
    out["firing_rate_census_box"] = dict(n=n, screened=fired,
                                         fraction=fired / n)
    print("census box screened:", fired, "/", n, flush=True)

    # ... and on a population of ORDINARY non-conforming stacks, where the
    # own-scale ratio bar is what keeps the guard off
    import numpy as np
    rng = np.random.default_rng(20260911)
    n2 = fired2 = 0
    for _ in range(400):
        a = float(rng.uniform(0.15, 0.35))
        b = float(rng.uniform(0.55, 0.80))
        gap = float(rng.uniform(0.01, 0.08))
        st = P.V.PMMStack(1.2e-6, degree=10, min_feature=1.2e-6 * 1e-12)
        st.add_layer(0.1e-6, segments=[(a, 2.25), (b - a, 9.0), (1.0 - b, 2.25)])
        st.add_layer(0.1e-6, segments=[(a + gap, 2.25), (b + gap - a - gap, 9.0),
                                       (1.0 - b - gap, 2.25)])
        st.set_source(0.85e-6, theta=0.15)
        n2 += 1
        if ps._sliver_screen(st, require_passive=False):
            fired2 += 1
    out["firing_rate_ordinary"] = dict(n=n2, screened=fired2,
                                       fraction=fired2 / n2)
    print("ordinary non-conforming screened:", fired2, "/", n2, flush=True)
    G.dump(dict(reps=REPS, **out), "p5_cost")


if __name__ == "__main__":
    main()
