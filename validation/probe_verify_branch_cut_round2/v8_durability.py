"""TASK 6 -- test durability: every constant in
``tests/unit/test_fix_branch_cut_round2.py`` and the restated X-1 / M1 tests,
re-measured on ITS OWN fixture across a thread ladder and both builds.

For each bar the probe reports what the bar is gating, the POST envelope (the
floor it must clear), and the smallest PRE reading (the defect it must catch),
so the two-sided margin can be read off rather than taken from a comment.

Run with a thread count on the command line, e.g.
  OPENBLAS_NUM_THREADS=4 PYTHONPATH=. python .../v8_durability.py out.json
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v8.json"

# ---- the test file's OWN fixture constants, transcribed ------------------
WL, P, D = 0.6e-6, 0.5e-6, 0.2e-6
HOST, N_SUB = 2.25, 1.63
WEAK = HOST * (1.0 + 1e-6)
S = 6


def cell(pillar=WEAK):
    c = np.full((S, S), HOST + 0j)
    c[2:4, 2:4] = pillar
    return c


def pmm_stack(n_orders=4, pillar=WEAK, spacer=HOST, n_sub=N_SUB):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    st = PMM2DStackHybrid(P, P, n_substrate=n_sub, n_superstrate=1.0,
                          degree=7, n_orders=n_orders, symmetry=False)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    st.add_layer(D, eps_cell=cell(pillar))
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.set_source(WL, theta=0.0).solve()


def closure(res):
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 2.0)


def pixel_stack(n_orders=3, spacer=HOST, Sp=32):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    e = np.full((Sp, Sp), HOST + 0j)
    x = (np.arange(Sp) + 0.5) / Sp - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    e[m] = WEAK
    st = PMM2DStackHybrid(P, P, n_substrate=N_SUB, n_superstrate=1.0,
                          degree=7, n_orders=n_orders, symmetry=False)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    st.add_layer(D, eps_cell=e)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(WL, theta=0.0).solve()
    return float(np.sum(np.asarray(R)) + np.sum(np.asarray(T)))


def rcwa_reference(n_orders=6, rep=12):
    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(P, period_y=P, n_substrate=N_SUB, n_superstrate=1.0,
                   n_orders=n_orders, n_orders_y=n_orders)
    st.add_layer(0.1e-6, eps=HOST)
    st.add_layer(D, eps_cell=np.kron(cell(), np.ones((rep, rep),
                                                     dtype=complex)))
    st.add_layer(0.1e-6, eps=HOST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.set_source(WL, theta=0.0).solve().efficiencies()


def order_map(res):
    o = np.asarray(res[0])
    R = np.asarray(res[1], dtype=float)
    T = np.asarray(res[2], dtype=float)
    if R.ndim == 2:
        R, T = R.sum(axis=0), T.sum(axis=0)
    return {(int(a), int(b)): (float(R[i]), float(T[i]))
            for i, (a, b) in enumerate(o)}


def per_order_gap(a, b):
    keys = set(a) & set(b)
    return max(max(abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1]))
               for k in keys)


def pre_round1_sqrt_decay(x, xp=None, band=1e-8):
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


class PreArm:
    def __init__(self):
        self._saved = []

    def __enter__(self):
        import lumenairy.elements as EL
        base = Path(EL.__file__).parent
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(base).with_suffix("")
            name = "lumenairy.elements." + ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if callable(getattr(mod, "_sqrt_decay", None)):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = pre_round1_sqrt_decay
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        self._saved = []
        return False


def run():
    out = {}

    # ---- _CLOSURE_BAR = 1e-9, gates gate 3 -----------------------------
    post = {str(M): closure(pmm_stack(M)) for M in (3, 4, 5)}
    with PreArm():
        pre = {str(M): closure(pmm_stack(M)) for M in (3, 4, 5)}
        ctrl = {str(M): closure(pmm_stack(M, spacer=HOST * 1.01))
                for M in (3, 4, 5)}
    out["CLOSURE_BAR"] = {
        "bar": 1e-9, "post": post, "pre_engineered": pre,
        "pre_detuned_control": ctrl,
        "post_envelope": max(abs(v) for v in post.values()),
        "pre_worst": max(abs(v) for v in pre.values()),
        "pre_smallest": min(abs(v) for v in pre.values()),
        "control_worst": max(abs(v) for v in ctrl.values()),
    }

    # ---- _PASSIVITY_BAR = 1e-8, gates gate 4 ---------------------------
    lad = {}
    for M in (2, 3, 4, 5, 6):
        try:
            lad[f"post_spacer_M{M}"] = pixel_stack(M)
        except Exception as exc:
            lad[f"post_spacer_M{M}"] = f"{type(exc).__name__}"
        try:
            lad[f"post_nospacer_M{M}"] = pixel_stack(M, spacer=None)
        except Exception as exc:
            lad[f"post_nospacer_M{M}"] = f"{type(exc).__name__}"
    with PreArm():
        for M in (2, 3, 4, 5, 6):
            for nm, sp in (("pre_spacer", HOST), ("pre_nospacer", None)):
                try:
                    lad[f"{nm}_M{M}"] = pixel_stack(M, spacer=sp)
                except Exception as exc:
                    lad[f"{nm}_M{M}"] = f"{type(exc).__name__}"
    num = {k: v for k, v in lad.items() if isinstance(v, float)}
    out["PASSIVITY_BAR"] = {
        "bar": 1e-8, "ladder": lad,
        "post_envelope": max((abs(v - 2.0) for k, v in num.items()
                              if k.startswith("post")), default=None),
        "pre_nospacer_envelope": max((abs(v - 2.0) for k, v in num.items()
                                      if k.startswith("pre_nospacer")),
                                     default=None),
        "pre_spacer_smallest": min((abs(v - 2.0) for k, v in num.items()
                                    if k.startswith("pre_spacer")),
                                   default=None),
        "pre_spacer_worst": max((abs(v - 2.0) for k, v in num.items()
                                 if k.startswith("pre_spacer")), default=None),
    }

    # ---- _REFERENCE_BAR = 1e-9, gates gate 5 ---------------------------
    ref6 = order_map(rcwa_reference(6))
    ref8 = order_map(rcwa_reference(8))
    postr = {str(M): per_order_gap(order_map(pmm_stack(M)), ref6)
             for M in (3, 4, 5)}
    with PreArm():
        prer = {str(M): per_order_gap(order_map(pmm_stack(M)), ref6)
                for M in (3, 4, 5)}
    out["REFERENCE_BAR"] = {
        "bar": 1e-9, "reference_self_convergence_6_vs_8":
            per_order_gap(ref6, ref8),
        "post": postr, "pre_engineered": prer,
        "post_envelope": max(postr.values()),
        "pre_smallest": min(prer.values()),
    }

    # ---- X-1's _X1_CLOSED_CLOSURE = 1e-8 -------------------------------
    sys.path.insert(0, str(VC.TREE / "validation"
                           / "probe_verify_branch_cut_round2"))
    import v5_x1_m1 as V5
    post_l = V5.x1_ladder("shipped")
    with V5.PreArm():
        pre_l = V5.x1_ladder("pre-round-1 engineered")
    out["X1_CLOSED_CLOSURE"] = {
        "bar": 1e-8,
        "post_worst_closure": max(post_l[p]["worst_closure"]
                                  for p in ("te", "tm")),
        "pre_n_bad_te": sum(1 for r in pre_l["te"]["rows"]
                            if r["raised"] or r["close"] > 1e-6),
        "pre_n_flagged_te": pre_l["te"]["n_flagged_cells"],
        "pre_worst_relative_sumR_te": pre_l["te"]["worst_relative_sumR"],
        "pre_smallest_manifesting_closure_te": min(
            [r["close"] for r in pre_l["te"]["rows"]
             if np.isfinite(r["close"]) and r["close"] > 1e-6] or [None]),
    }
    VC.dump(OUT, out)

    st = VC.stamp()
    print(f"ARM {st['arm']}  py {st['python']}  "
          f"OPENBLAS={st['OPENBLAS_NUM_THREADS']}")
    for k, v in out.items():
        print(f"\n== {k}  bar = {v['bar']:.0e}")
        for kk, vv in v.items():
            if kk == "bar":
                continue
            print(f"   {kk}: {vv}")


if __name__ == "__main__":
    run()
