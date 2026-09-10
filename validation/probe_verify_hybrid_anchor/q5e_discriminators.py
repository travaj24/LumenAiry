"""Q5e -- O2, the four DISCRIMINATORS that decide what the blow-up depends on.

q5d pins the proximate cause (``cond(T22)`` in the generalized interface
solve).  This one varies one axis at a time to find what CONTROLS it, and in
particular tests the attribution the fix audit gives (and the library's own
``_EnergyWarning`` text repeats): "the documented near-degenerate layer <->
region mode match", whose stated remedy is to DETUNE a coincident permittivity
by ~1e-6.

Axes, all on the exact O2 fixture at ``n_orders = 5``, oblique 25 unless
scanned:

  1. DETUNE the cell's ``eps = 1.0`` (= the superstrate) by 1e-6 / 1e-4, and
     detune the SUPERSTRATE instead;
  2. THETA, which moves the ``+1`` order relative to the ``|alpha| = 1``
     cut-off;
  3. the SLANT magnitude, from 1e-4 to 1.0;
  4. the spectral-element DEGREE.
"""
from __future__ import annotations

import math
import time
import warnings

import _lib as L
import numpy as np
from q5_o2_blowup import Probe
from q5c_mechanism import D1, DF, FEPS, NSUB, NSUP, PX, PY, TSL, WL, cell

BASE = cell()
PR = Probe()


def run(*, c=None, M=5, theta_deg=25.0, tsl=TSL, degree=11, nsup=NSUP,
        film=True):
    import lumenairy.elements.pmm.stack2d as S2
    st = S2.PMM2DStackHybrid(PX, PY, n_superstrate=nsup, n_substrate=NSUB,
                             n_orders=M, degree=degree)
    st.add_layer(D1, eps_cell=BASE if c is None else c,
                 slant=((tsl, 0.0) if tsl else None))
    if film:
        st.add_layer(DF, eps=FEPS)
    st.set_source(WL, theta=math.radians(theta_deg), phi=0.0)
    PR.reset()
    PR.install(S2)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _o, R, T, _J = st.solve()
        rt = float(np.max(np.asarray(R).sum(axis=1)
                          + np.asarray(T).sum(axis=1)))
        warned = bool(w)
    finally:
        PR.restore(S2)
    s = PR.summary()
    return dict(RT=rt, warned=warned, cond_T22=s.get("max_cond_T22"),
                cond_Mb=s.get("max_cond_Mb"),
                max_prop=s.get("max_prop_factor"))


def main():
    t0 = time.time()
    out = {}

    # 1 -- DETUNE
    det = {"as_written": run()}
    for eps in (1e-6, 1e-4):
        c = BASE.copy()
        c[c == 1.0] = 1.0 + eps
        det["cell_eps_detuned_%.0e" % eps] = run(c=c)
    det["superstrate_detuned_1e-06"] = run(nsup=1.0 + 1e-6 + 0j)
    out["detune"] = det

    # 2 -- THETA (the near-cut-off attribution)
    th = {}
    for t in (15, 20, 22, 24, 25, 26, 28, 30, 35):
        r = run(theta_deg=t)
        r["alpha_plus1"] = math.sin(math.radians(t)) + WL / PX
        th["theta%d" % t] = r
    out["theta"] = th

    # 3 -- the SLANT magnitude
    sl = {"vertical": run(tsl=None)}
    for t in (1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.35, 0.5, 1.0):
        sl["slant_%g" % t] = run(tsl=t)
    out["slant"] = sl

    # 4 -- the spectral DEGREE
    dg = {}
    for d in (7, 9, 11, 13, 15):
        dg["degree%d" % d] = run(degree=d)
    out["degree"] = dg

    for grp in ("detune", "theta", "slant", "degree"):
        print("==", grp)
        for k, v in out[grp].items():
            print("   %-28s RT=%-13.5g cond(T22)=%-12.4g warned=%s"
                  % (k, v["RT"], v["cond_T22"] or float("nan"), v["warned"]))
    out["_seconds"] = round(time.time() - t0, 1)
    L.dump("q5e_discriminators", out)


if __name__ == "__main__":
    main()
