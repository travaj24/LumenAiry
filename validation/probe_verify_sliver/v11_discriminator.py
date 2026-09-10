"""VERIFY -- a CANDIDATE fix for V-1 / open item F, scored two-sided.

The refusal cannot tell a sliver-caused super-unity from a truncation-caused
one, and its first-named remedy silences the guard without changing the number
(S4.5 of the verification report).  There is a discriminator that costs one
extra solve at the point where the library is about to RAISE anyway:

    re-solve on the grid the prescribed ``min_feature`` would produce.
    If the super-unity VANISHES, the sliver was the cause.
    If it SURVIVES, it was not -- name degree / n_slices instead.

Scored here on both arms:

* TRUE POSITIVES  -- the O-11 hazard band (lossless, degrees 12/14/20).
  Super-unity must VANISH.
* FALSE POSITIVES -- the truncation family of S4.5 (lossy substrate, large
  angle, low degree, a HARMLESS sliver).  Super-unity must SURVIVE.

Nothing is changed in the library; this is evidence for the suggestion.

    python validation/probe_verify_sliver/v11_discriminator.py [out.json]
"""
import itertools
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
P, WL = 1.2e-6, 0.85e-6
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505
DZ = 0.32e-6 / 4
NO = P * 1e-12
BAR = ps._STACK_SUPERUNITY_BAR


def build(d, deg, nsub, nsup, th, nl=2, eps=EP, mf=NO):
    st = PMMStack(P, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=mf, far_field_orders=31)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(DZ, segments=[(A0 - dd, EH),
                                   (B0 + dd - (A0 - dd), eps),
                                   (1.0 - (B0 + dd), EH)])
    st.set_source(WL, theta=th)
    return st


def worst(st):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = st.solve()
        return float(np.max(np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)))
    finally:
        ps.PMM_SLIVER_GUARD = was


def prescribed_mf(d, nl=2):
    """The `min_feature` the refusal would print, from the screen itself."""
    segs = []
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        segs.append([(A0 - dd, EH), (B0 + dd - (A0 - dd), EP),
                     (1.0 - (B0 + dd), EH)])
    hit = ps._cross_layer_sliver(segs, NO / P)
    return None if hit is None else 2.0 * hit[3] * P


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v11_discriminator.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    t0 = time.time()

    tp = []
    for deg, d in itertools.product((12, 14, 20), (1e-4, 5e-5, 3e-5, 1e-5)):
        w0 = worst(build(d, deg, 1.0, 1.0, 0.15))
        if w0 <= 1.0 + BAR:
            continue
        mf = prescribed_mf(d)
        w1 = worst(build(d, deg, 1.0, 1.0, 0.15, mf=mf))
        tp.append(dict(degree=deg, delta=d, RplusT=w0, snapped_RplusT=w1,
                       survives=bool(w1 > 1.0 + BAR)))
        print(f"  TP deg {deg} d {d:.0e}: R+T {w0:.6g} -> snapped "
              f"{w1:.8g}  survives={tp[-1]['survives']}")
    tp_ok = sum(1 for r in tp if not r["survives"])
    print(f"  correct on {tp_ok}/{len(tp)} TRUE POSITIVES "
          f"(super-unity must VANISH)")

    fp = []
    for nsub, nsup, th, deg, eps, nl, d in itertools.product(
            (1.5 + 0.05j, 1.5 + 0.2j), (2.5, 3.5), (1.2, 1.3), (6, 8),
            (12.0, EP), (2, 4), (3e-3, 1e-3, 3e-4)):
        w0 = worst(build(d, deg, nsub, nsup, th, nl, eps))
        if w0 <= 1.0 + BAR:
            continue
        mf = prescribed_mf(d, nl)
        if mf is None:
            continue
        w1 = worst(build(d, deg, nsub, nsup, th, nl, eps, mf=mf))
        fp.append(dict(nsub=str(nsub), nsup=nsup, theta=th, degree=deg,
                       eps=str(eps), n_layers=nl, delta=d, RplusT=w0,
                       snapped_RplusT=w1, survives=bool(w1 > 1.0 + BAR)))
    fp_ok = sum(1 for r in fp if r["survives"])
    print(f"  correct on {fp_ok}/{len(fp)} TRUNCATION rows "
          f"(super-unity must SURVIVE)")
    print(f"  total wall {time.time() - t0:.0f} s for "
          f"{2 * (len(tp) + len(fp)) + len(fp)} solves")
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__, bar=BAR),
                       true_positives=tp, tp_correct=tp_ok,
                       truncation=fp, fp_correct=fp_ok), f, indent=1,
                  default=str)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
