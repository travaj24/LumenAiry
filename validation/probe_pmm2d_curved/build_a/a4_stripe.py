"""A4 -- stretch self-consistency on the TE stripe against the exact 1-D oracle
(and the STOP condition of the plan: TE at a = 0.05 p must reach 1e-5 of the
oracle by M = 10).

  python validation/probe_pmm2d_curved/build_a/a4_stripe.py oracle
  python validation/probe_pmm2d_curved/build_a/a4_stripe.py ladder <a_frac> <M_lo> <M_hi>
  python validation/probe_pmm2d_curved/build_a/a4_stripe.py nocof <a_frac> <M>

Outputs: a4_oracle1d.json, a4_stripe_a<a>.json, a4_nocof_a<a>_M<M>.json.
Per row: the max |dR|, |dT| over the nine orders |m|, |n| <= 1 against the
1-D oracle, SEPARATELY for TE (incident E along y, row 1) and TM (row 0),
and the lossless closure.  a = 0 is the UNMAPPED shipped solver on the same
physical walls (see _common.solve).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402

ORACLE = os.path.join(C.HERE, "a4_oracle1d.json")


def oracle():
    from lumenairy.elements.pmm.oned import pmm_efficiency_1d
    out = {}
    for pol in ("te", "tm"):
        vals = {}
        for deg in (30, 40):
            o, R, T = pmm_efficiency_1d(C.P, 2.0, 1.0, C.N_SUB, C.N_SUP,
                                        C.DEPTH, 0.6 / C.P, C.WL,
                                        polarization=pol, degree=deg,
                                        far_field_orders=7)
            vals[str(deg)] = {str(int(m)): [float(R[i]), float(T[i])]
                              for i, m in enumerate(np.asarray(o))}
        out[pol] = vals
    self_gap = max(abs(out[p]["30"][k][i] - out[p]["40"][k][i])
                   for p in out for k in out[p]["40"] for i in (0, 1))
    res = {"env": C.env_record(), "oracle": out, "self_gap_30_40": self_gap}
    with open(ORACLE, "w") as f:
        json.dump(res, f, indent=1)
    print("oracle self-gap 30 vs 40:", self_gap)


def ref_vec(pol):
    o = json.load(open(ORACLE))["oracle"][pol]["40"]
    R = [o.get(str(m), [0, 0])[0] if n == 0 else 0.0 for m, n in C.ORD9]
    T = [o.get(str(m), [0, 0])[1] if n == 0 else 0.0 for m, n in C.ORD9]
    return np.array(R + T)


def errs(o, R, T):
    out = {}
    for row, pol in ((1, "te"), (0, "tm")):
        v = C.vec(o, R[row:row + 1], T[row:row + 1])
        out[f"err_{pol}"] = float(np.max(np.abs(v - ref_vec(pol))))
        out[f"closure_{pol}"] = float(abs(R[row].sum() + T[row].sum() - 1))
    return out


def ladder(a, m_lo, m_hi):
    path = os.path.join(C.HERE, f"a4_stripe_a{a}.json")
    res = {"env": C.env_record(), "a_over_p": a, "fixture": "stripe",
           "rows": []}
    for M in range(m_lo, m_hi + 1):
        t0 = time.perf_counter()
        o, R, T, _J, _st = C.solve("stripe", a, M)
        row = {"M": M, "dof": 2 * (3 * (M - 1)) ** 2,
               "t": time.perf_counter() - t0, **errs(o, R, T)}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


def nocof(a, M):
    o, R, T, _J, _st = C.solve("stripe", a, M)
    good = errs(o, R, T)
    with C.no_cofactor():
        o2, R2, T2, _J2, _st2 = C.solve("stripe", a, M)
    bad = errs(o2, R2, T2)
    res = {"env": C.env_record(), "a_over_p": a, "M": M, "correct": good,
           "no_cofactor": bad,
           "no_cofactor_vs_correct": float(np.max(np.abs(
               C.vec(o2, R2, T2) - C.vec(o, R, T))))}
    with open(os.path.join(C.HERE, f"a4_nocof_a{a}_M{M}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["correct"]), json.dumps(res["no_cofactor"]),
          res["no_cofactor_vs_correct"])


if __name__ == "__main__":
    if sys.argv[1] == "oracle":
        oracle()
    elif sys.argv[1] == "ladder":
        ladder(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        nocof(float(sys.argv[2]), int(sys.argv[3]))
