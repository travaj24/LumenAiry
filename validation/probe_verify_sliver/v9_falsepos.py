"""VERIFY task 3, the FALSE-POSITIVE census -- and what the refusal tells the
caller to do about it.

Open item F of the fix audit says the refusal "cannot separate the sliver from
the quasi-resonance on a real taper", and mitigates it with remedy (4).  This
measures how wide that hole is and what remedy (1) actually does inside it.

A. Conjunct (b)'s premise, attacked directly: 960 SLIVER-FREE stacks that
   ``_stack_provably_passive`` accepts, over lossy substrates, superstrate
   indices, angles, layer permittivities and degrees.  How far above the bar
   does an ordinary UNDER-CONVERGED solve read?

B. The census: over a product of realistic staircase parameters, count the rows
   where the guard REFUSES a solve whose answer tracks the exact ``delta -> 0``
   limit within the fix's own "right" rule -- i.e. where the sliver is present,
   harmless, and blamed anyway.

C. The remedy trace on one such row: what remedy (1) (the ``min_feature`` the
   message prescribes, named first and with a number) does to the answer,
   against what remedy (4) (degree) does.

    python validation/probe_verify_sliver/v9_falsepos.py [out.json]
"""
import itertools
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

HERE = os.path.dirname(os.path.abspath(__file__))
P, WL = 1.2e-6, 0.85e-6
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505
DZ = 0.32e-6 / 4
NO_SNAP = P * 1e-12


def build(d, deg, nsub, nsup, th, nl=2, eps=EP, mf=NO_SNAP):
    """A staircase of ``nl`` slices whose walls open by ``d`` in total."""
    st = PMMStack(P, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=mf, far_field_orders=31)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(DZ, segments=[(A0 - dd, EH),
                                   (B0 + dd - (A0 - dd), eps),
                                   (1.0 - (B0 + dd), EH)])
    st.set_source(WL, theta=th)
    return st


def run(st, guard=True):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = guard
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, _J = st.solve()
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        return dict(ok=True, worst=float(np.max(tot)), m=o[i],
                    R=np.asarray(R)[1][i], T=np.asarray(T)[1][i],
                    warned=[str(w.message)[:60] for w in rec][:1])
    except ValueError as exc:
        return dict(ok=False, sliver="SLIVER" in str(exc), msg=str(exc))
    finally:
        ps.PMM_SLIVER_GUARD = was


def gap(a, b):
    c = np.intersect1d(a["m"], b["m"])
    ia, ib = np.searchsorted(a["m"], c), np.searchsorted(b["m"], c)
    return float(max(np.abs(a["R"][ia] - b["R"][ib]).max(),
                     np.abs(a["T"][ia] - b["T"][ib]).max()))


# ==========================================================================
def part_a():
    worst, n_pass, n = (0.0, None), 0, 0
    over = 0
    for nsub, nsup, th, eps, deg in itertools.product(
            (1.5 + 0.05j, 1.5 + 0.5j, 3.0 + 2.0j, 1.0 + 8.0j, 0.05 + 4.0j,
             4.0 + 0.0j),
            (1.0, 1.5, 2.5, 3.5), (0.0, 0.3, 0.9, 1.3, 1.55),
            (4.0, (2.0 + 0.5j) ** 2, 12.0, (0.2 + 3.5j) ** 2), (8, 12)):
        st = PMMStack(P, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                      far_field_orders=21)
        st.add_layer(0.32e-6, segments=[(0.4, eps), (0.6, 1.0)])
        st.add_layer(0.11e-6, segments=[(0.4, 2.25), (0.6, 1.0)])
        st.set_source(WL, theta=th)
        n += 1
        if not ps._stack_provably_passive(st):
            continue
        n_pass += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = st.solve()
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        w = float(np.max(tot)) - 1.0
        if w > ps._STACK_SUPERUNITY_BAR:
            over += 1
        if w > worst[0]:
            worst = (w, dict(nsub=str(nsub), nsup=nsup, theta=th,
                             eps=str(eps), degree=deg))
    print(f"  {n} sliver-free stacks, {n_pass} PROVABLY PASSIVE, "
          f"{over} of them read super-unity ABOVE the bar")
    print(f"  worst R+T-1 = {worst[0]:.4e} at {worst[1]}")
    return dict(n=n, n_passive=n_pass, n_over_bar=over, worst=worst[0],
                worst_at=worst[1])


def part_b():
    hits, tried = [], 0
    for nsub, nsup, th, deg, eps, nl in itertools.product(
            (1.5 + 0.05j, 1.5 + 0.2j, 3.0 + 2.0j), (2.5, 3.5),
            (1.2, 1.3, 1.45), (6, 8, 10), (12.0, EP), (2, 4)):
        for d in (3e-3, 1e-3, 3e-4):
            tried += 1
            ref = run(build(0.0, deg, nsub, nsup, th, nl, eps), False)
            cur = run(build(d, deg, nsub, nsup, th, nl, eps), False)
            if not (ref["ok"] and cur["ok"]):
                continue
            e = gap(cur, ref)
            if e > 10.0 * d:            # not provably sliver-harmless -> skip
                continue
            g = run(build(d, deg, nsub, nsup, th, nl, eps), True)
            if not g["ok"] and g["sliver"]:
                hits.append(dict(nsub=str(nsub), nsup=nsup, theta=th,
                                 degree=deg, eps=str(eps), n_layers=nl,
                                 delta=d, err=e, err_over_delta=e / d,
                                 RplusT=cur["worst"]))
    print(f"  tried {tried}; FALSE POSITIVES {len(hits)} "
          f"({100.0 * len(hits) / tried:.1f}%)")
    if hits:
        print(f"  err/delta among them: "
              f"{min(h['err_over_delta'] for h in hits):.2f} .. "
              f"{max(h['err_over_delta'] for h in hits):.2f} "
              f"(the fix's own CORRECT rule is <= 10)")
        print(f"  R+T among them: {min(h['RplusT'] for h in hits):.5f} .. "
              f"{max(h['RplusT'] for h in hits):.5f}")
        for h in hits[:4]:
            print(f"    nsub={h['nsub']} nsup={h['nsup']} th={h['theta']} "
                  f"deg={h['degree']} eps={h['eps']} nl={h['n_layers']} "
                  f"delta={h['delta']:.0e}: err {h['err']:.3e} "
                  f"({h['err_over_delta']:.2f}x delta) R+T {h['RplusT']:.6g}")
    return dict(tried=tried, n_hits=len(hits), hits=hits)


def part_c():
    d = 1e-3
    g = run(build(d, 6, 1.5 + 0.05j, 2.5, 1.2, 2, 12.0))
    assert not g["ok"], "the case must refuse"
    mf = float(g["msg"].split("min_feature=")[1].split(" ")[0])
    out = dict(prescribed_min_feature=mf)
    r1 = run(build(d, 6, 1.5 + 0.05j, 2.5, 1.2, 2, 12.0, mf=mf))
    out["remedy1_ok"] = r1["ok"]
    out["remedy1_RplusT"] = r1.get("worst")
    r2 = run(build(0.0, 6, 1.5 + 0.05j, 2.5, 1.2, 2, 12.0))
    out["remedy2_RplusT"] = r2.get("worst")
    out["remedy4_ladder"] = {}
    for deg in (6, 8, 10, 12, 14, 16):
        r = run(build(d, deg, 1.5 + 0.05j, 2.5, 1.2, 2, 12.0))
        out["remedy4_ladder"][deg] = (r["worst"] if r["ok"] else "REFUSED")
    out["unguarded_deg6_snap_off"] = run(
        build(d, 6, 1.5 + 0.05j, 2.5, 1.2, 2, 12.0), False)["worst"]
    print(f"  prescribed min_feature = {mf:g} m")
    print(f"  remedy (1): solved={r1['ok']} R+T = {r1.get('worst'):.6g} "
          f"(unguarded, unsnapped: {out['unguarded_deg6_snap_off']:.6g}) "
          f"-> the refusal is silenced, the NUMBER is unchanged")
    print(f"  remedy (2): R+T = {out['remedy2_RplusT']:.6g}")
    print("  remedy (4): " + " ".join(
        f"deg{k}:{v if isinstance(v, str) else format(v, '.6g')}"
        for k, v in out["remedy4_ladder"].items()))
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v9_falsepos.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    print("\n== A: conjunct (b)'s premise on sliver-free passive stacks ==")
    a = part_a()
    print("\n== B: the false-positive census ==")
    b = part_b()
    print("\n== C: what the prescribed remedy does ==")
    c = part_c()
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       part_a=a, part_b=b, part_c=c), f, indent=1, default=str)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
