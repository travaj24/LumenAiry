"""TASK 2 -- the hybrid wrong-answer reproducer, on MY OWN fixtures.

Runs UNCHANGED on the pre-round-2 tree (2898767) and on the round-2 tree, so
the ARM is the tree, not an in-process patch.  ``_vcommon.arm()`` reads the arm
off the LIVE source (how many ``_sqrt_decay`` bodies exist), and every JSON
carries it.

What it measures, per fixture:
  * lossless closure ``sum R + sum T - 2`` (a Jones return) -- an INDEPENDENT
    oracle needing no reference solve;
  * per-order distance to an independent ``RCWAStack`` solve of the SAME
    device, with the reference's OWN convergence (an n_orders ladder and a
    cell-replication ladder) reported so the comparison carries its
    uncertainty;
  * ``cond(a + b)`` at every guarded interface mode-match, captured by wrapping
    ``_guarded_inverse`` in BOTH ``rcwa/_core`` and ``pmm/_core``;
  * an analytic three-layer Fresnel oracle for the zeroth order, valid to
    O(modulation) because the cell is weakly modulated.

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v2_hybrid.py out.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v2.json"
WL = 0.5321e-6


# ---------------------------------------------------------------------------
# cond(a+b) instrument -- wraps the shared guarded inverse in BOTH bindings
# ---------------------------------------------------------------------------
class CondTap:
    def __init__(self):
        self.rows = []
        self._saved = []

    def __enter__(self):
        from lumenairy.elements.pmm import _core as pc
        from lumenairy.elements.rcwa import _core as rc
        for mod in (rc, pc):
            orig = mod._guarded_inverse
            self._saved.append((mod, orig))

            def wrapped(A, site, *a, _orig=orig, **kw):
                try:
                    A_np = np.asarray(A)
                    if A_np.ndim == 2 and A_np.shape[0] == A_np.shape[1] \
                            and np.all(np.isfinite(A_np)):
                        self.rows.append((str(site), int(A_np.shape[0]),
                                          float(np.linalg.cond(A_np))))
                except Exception:
                    pass
                return _orig(A, site, *a, **kw)

            mod._guarded_inverse = wrapped
        return self

    def __exit__(self, *exc):
        for mod, orig in self._saved:
            mod._guarded_inverse = orig
        self._saved = []
        return False

    def summary(self):
        if not self.rows:
            return {"n": 0}
        by = {}
        for site, n, c in self.rows:
            by.setdefault(site, []).append(c)
        return {"n": len(self.rows),
                "max_cond": max(c for _, _, c in self.rows),
                "per_site_max": {k: max(v) for k, v in by.items()},
                "per_site_n": {k: len(v) for k, v in by.items()}}


# ---------------------------------------------------------------------------
# analytic three-layer Fresnel oracle (exact, no solver involved)
# ---------------------------------------------------------------------------
def fresnel_stack(eps_list, thick_list, eps_sup, eps_sub, wl, theta=0.0,
                  pol="s"):
    """Exact TMM for a stack of isotropic layers, ``exp(-i w t)`` convention.
    Returns (R, T) power coefficients."""
    k0 = 2 * np.pi / wl
    kt = np.sqrt(complex(eps_sup)) * np.sin(theta) * k0
    eps_all = [eps_sup] + list(eps_list) + [eps_sub]
    kz = [np.sqrt(complex(e) * k0 ** 2 - kt ** 2 + 0j) for e in eps_all]
    kz = [k if k.imag >= 0 else -k for k in kz]
    if pol == "s":
        eta = kz
    else:
        eta = [kz[i] / complex(eps_all[i]) for i in range(len(eps_all))]
    # transfer-matrix fold
    M = np.eye(2, dtype=complex)
    for i in range(1, len(eps_all) - 1):
        d = thick_list[i - 1]
        ph = kz[i] * d
        Mi = np.array([[np.cos(ph), -1j * np.sin(ph) / eta[i]],
                       [-1j * eta[i] * np.sin(ph), np.cos(ph)]], dtype=complex)
        M = M @ Mi
    e0, eN = eta[0], eta[-1]
    denom = (M[0, 0] * e0 + M[0, 1] * e0 * eN + M[1, 0] + M[1, 1] * eN)
    r = (M[0, 0] * e0 + M[0, 1] * e0 * eN - M[1, 0] - M[1, 1] * eN) / denom
    t = 2 * e0 / denom
    R = float(abs(r) ** 2)
    T = float(abs(t) ** 2 * (eN / e0).real)
    return R, T


# ---------------------------------------------------------------------------
# fixtures -- MINE
# ---------------------------------------------------------------------------
def cellA(rel):
    """8 x 8, eps 2.25, a 3 x 2 block lifted by ``rel``."""
    c = np.full((8, 8), 2.25, dtype=complex)
    c[1:4, 2:4] = 2.25 * (1.0 + rel)
    return c


def cellB(rel):
    """6 x 6, eps 4.0, an L-shaped region lifted by ``rel``."""
    c = np.full((6, 6), 4.0, dtype=complex)
    c[0:2, 0:4] = 4.0 * (1.0 + rel)
    c[0:4, 0:2] = 4.0 * (1.0 + rel)
    return c


def cellC(rel):
    """10 x 10, eps 2.89, a centred 4 x 3 block."""
    c = np.full((10, 10), 2.89, dtype=complex)
    c[3:7, 4:7] = 2.89 * (1.0 + rel)
    return c


FIX = {
    # ---- coincident-spacer stacks (the claim's shape, MY geometries) -------
    "S1_spacer_8px": dict(cell=cellA, rel=1e-6, bg=2.25, px=0.62e-6,
                          py=0.58e-6, spacers=(0.12e-6, 0.09e-6),
                          d=0.23e-6, nsub=1.63, nsup=1.0, th=0.0, ph=0.0,
                          spacer_eps=None),
    "S2_spacer_eps4": dict(cell=cellB, rel=2e-6, bg=4.0, px=0.44e-6,
                           py=0.44e-6, spacers=(0.05e-6, 0.05e-6),
                           d=0.31e-6, nsub=1.71, nsup=1.0, th=0.0, ph=0.0,
                           spacer_eps=None),
    "S3_spacer_oblique": dict(cell=cellA, rel=1e-6, bg=2.25, px=0.62e-6,
                              py=0.58e-6, spacers=(0.12e-6, 0.09e-6),
                              d=0.23e-6, nsub=1.63, nsup=1.0,
                              th=np.deg2rad(12.0), ph=0.0, spacer_eps=None),
    "S4_spacer_conical": dict(cell=cellA, rel=1e-6, bg=2.25, px=0.62e-6,
                              py=0.58e-6, spacers=(0.12e-6, 0.09e-6),
                              d=0.23e-6, nsub=1.63, nsup=1.0,
                              th=np.deg2rad(15.0), ph=np.deg2rad(33.0),
                              spacer_eps=None),
    "S5_spacer_10px": dict(cell=cellC, rel=5e-7, bg=2.89, px=0.71e-6,
                           py=0.66e-6, spacers=(0.17e-6, 0.04e-6),
                           d=0.19e-6, nsub=1.42, nsup=1.0, th=0.0, ph=0.0,
                           spacer_eps=None),
    # ---- controls ----------------------------------------------------------
    "C1_nospacer_8px": dict(cell=cellA, rel=1e-6, bg=2.25, px=0.62e-6,
                            py=0.58e-6, spacers=(), d=0.23e-6, nsub=1.63,
                            nsup=1.0, th=0.0, ph=0.0, spacer_eps=None),
    "C2_nospacer_eps4": dict(cell=cellB, rel=2e-6, bg=4.0, px=0.44e-6,
                             py=0.44e-6, spacers=(), d=0.31e-6, nsub=1.71,
                             nsup=1.0, th=0.0, ph=0.0, spacer_eps=None),
    # ---- LOSSY spacer at oblique: the sign is physics, nothing may move ----
    "L1_lossy_spacer_oblique": dict(
        cell=cellA, rel=1e-6, bg=2.25, px=0.62e-6, py=0.58e-6,
        spacers=(0.12e-6, 0.09e-6), d=0.23e-6, nsub=1.63, nsup=1.0,
        th=np.deg2rad(12.0), ph=np.deg2rad(20.0), spacer_eps=2.25 + 0.01j),
    "L2_lossy_cell_spacer": dict(
        cell=cellA, rel=1e-6, bg=2.25, px=0.62e-6, py=0.58e-6,
        spacers=(0.12e-6, 0.09e-6), d=0.23e-6, nsub=1.63, nsup=1.0,
        th=0.0, ph=0.0, spacer_eps=2.25 + 1e-3j, lossy_cell=True),
}


def build_hybrid(spec, n_orders, degree=7, detune=0.0, cell_rel=None,
                 cell_override=None):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    rel = spec["rel"] if cell_rel is None else cell_rel
    cell = spec["cell"](rel) if cell_override is None else cell_override
    if spec.get("lossy_cell"):
        cell = cell + 1e-3j
    st = PMM2DStackHybrid(spec["px"], spec["py"], n_superstrate=spec["nsup"],
                          n_substrate=spec["nsub"], degree=degree,
                          n_orders=n_orders, symmetry=False)
    sp_eps = spec["spacer_eps"] if spec["spacer_eps"] is not None \
        else spec["bg"]
    sp_eps = sp_eps * (1.0 + detune)
    if len(spec["spacers"]) >= 1:
        st.add_layer(spec["spacers"][0], eps=sp_eps)
    st.add_layer(spec["d"], eps_cell=cell)
    if len(spec["spacers"]) >= 2:
        st.add_layer(spec["spacers"][1], eps=sp_eps)
    st.set_source(WL, theta=spec["th"], phi=spec["ph"])
    return st


def solve_hybrid(spec, n_orders, degree=7, detune=0.0, tap=True, **kw):
    ctx = CondTap() if tap else None
    out = {}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            if ctx is not None:
                ctx.__enter__()
            st = build_hybrid(spec, n_orders, degree, detune, **kw)
            res = st.solve()
            R = np.asarray(res[1] if isinstance(res, tuple) else res.R)
            T = np.asarray(res[2] if isinstance(res, tuple) else res.T)
            out["R"] = R.tolist()
            out["T"] = T.tolist()
            out["sumRT"] = float(R.sum() + T.sum())
            out["closure"] = float(R.sum() + T.sum() - 2.0)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if ctx is not None:
                ctx.__exit__()
        out["warnings"] = sorted({type(w.message).__name__ for w in ws})
    if ctx is not None:
        out["cond"] = ctx.summary()
    return out


# ---------------------------------------------------------------------------
# independent reference: RCWAStack on the SAME device
# ---------------------------------------------------------------------------
def rcwa_reference(spec, n_orders, rep=8, detune=0.0, cell_rel=None):
    from lumenairy.elements.rcwa import RCWAStack
    rel = spec["rel"] if cell_rel is None else cell_rel
    cell = spec["cell"](rel)
    if spec.get("lossy_cell"):
        cell = cell + 1e-3j
    big = np.kron(cell, np.ones((rep, rep)))
    sp_eps = (spec["spacer_eps"] if spec["spacer_eps"] is not None
              else spec["bg"]) * (1.0 + detune)
    st = RCWAStack(spec["px"], period_y=spec["py"], n_superstrate=spec["nsup"],
                   n_substrate=spec["nsub"], n_orders=n_orders)
    if len(spec["spacers"]) >= 1:
        st.add_layer(spec["spacers"][0], eps=sp_eps)
    st.add_layer(spec["d"], eps_cell=big)
    if len(spec["spacers"]) >= 2:
        st.add_layer(spec["spacers"][1], eps=sp_eps)
    st.set_source(WL, theta=spec["th"], phi=spec["ph"])
    o, R, T = st.solve().efficiencies()
    return np.asarray(R), np.asarray(T), np.asarray(o)


def order_map(orders):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(orders)}


def compare_per_order(Rp, Tp, op, Rr, Tr, orr):
    """Per-order distance on the orders BOTH solves carry, summing the two
    polarizations of a Jones return against the reference's scalar pair."""
    mp, mr = order_map(op), order_map(orr)
    keys = sorted(set(mp) & set(mr))
    dR = dT = 0.0
    for k in keys:
        i, j = mp[k], mr[k]
        rp = Rp[:, i].sum() if Rp.ndim == 2 else Rp[i]
        tp = Tp[:, i].sum() if Tp.ndim == 2 else Tp[i]
        rr = Rr[:, j].sum() if Rr.ndim == 2 else Rr[j]
        tr = Tr[:, j].sum() if Tr.ndim == 2 else Tr[j]
        dR = max(dR, abs(float(rp) - float(rr)))
        dT = max(dT, abs(float(tp) - float(tr)))
    return {"n_common_orders": len(keys), "max_dR": dR, "max_dT": dT,
            "max_per_order": max(dR, dT)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run():
    payload = {"fixtures": {}, "reference": {}, "detune": {},
               "modulation": {}, "fresnel": {}}

    # --- the reference own convergence, so the comparison carries its bound --
    for name in ("S1_spacer_8px", "S2_spacer_eps4", "S3_spacer_oblique",
                 "S5_spacer_10px"):
        spec = FIX[name]
        conv = {}
        base = None
        for M, rep in ((4, 8), (6, 8), (8, 8), (6, 4), (6, 12)):
            try:
                R, T, o = rcwa_reference(spec, M, rep=rep)
                key = f"M{M}_rep{rep}"
                conv[key] = {"closure": float(R.sum() + T.sum() - 2.0),
                             "n_orders_returned": int(o.shape[0])}
                if base is None:
                    base = (R, T, o)
                else:
                    conv[key]["vs_M4rep8"] = compare_per_order(
                        R, T, o, base[0], base[1], base[2])["max_per_order"]
            except Exception as exc:
                conv[f"M{M}_rep{rep}"] = {"error": repr(exc)}
        payload["reference"][name] = conv

    # --- the fixtures, at three truncations, with the reference ------------
    for name, spec in FIX.items():
        rows = {}
        for M in (3, 4, 5):
            row = solve_hybrid(spec, M)
            try:
                Rr, Tr, orr = rcwa_reference(spec, max(M + 2, 6), rep=8)
                st = build_hybrid(spec, M)
                res = st.solve()
                Rp = np.asarray(res[1])
                Tp = np.asarray(res[2])
                op = np.asarray(res[0])
                row["vs_reference"] = compare_per_order(Rp, Tp, op,
                                                        Rr, Tr, orr)
                row["reference_closure"] = float(Rr.sum() + Tr.sum() - 2.0)
            except Exception as exc:
                row["vs_reference"] = {"error": repr(exc)}
            rows[f"M{M}"] = row
        payload["fixtures"][name] = rows

    # --- the analytic Fresnel oracle: the device is uniform to O(rel) -------
    for name in ("S1_spacer_8px", "S2_spacer_eps4", "C1_nospacer_8px"):
        spec = FIX[name]
        layers = list(spec["spacers"][:1]) + [spec["d"]] + \
            list(spec["spacers"][1:2])
        sp = spec["spacer_eps"] if spec["spacer_eps"] is not None else spec["bg"]
        eps_l = ([sp] if len(spec["spacers"]) >= 1 else []) + [spec["bg"]] + \
            ([sp] if len(spec["spacers"]) >= 2 else [])
        Rs, Ts = fresnel_stack(eps_l, layers, spec["nsup"] ** 2,
                               spec["nsub"] ** 2, WL, spec["th"], "s")
        payload["fresnel"][name] = {"R_s": Rs, "T_s": Ts,
                                    "closure": Rs + Ts - 1.0}

    # --- the SPACER DETUNE ladder: where does the PRE answer become right? --
    lad = {}
    for d in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5,
              1e-4, 3e-4, 1e-3, 1e-2, 1e-1):
        row = {}
        for M in (3, 4):
            r = solve_hybrid(FIX["S1_spacer_8px"], M, detune=d, tap=False)
            row[f"M{M}"] = {k: r.get(k) for k in
                            ("closure", "sumRT", "warnings", "error")}
            try:
                Rr, Tr, orr = rcwa_reference(FIX["S1_spacer_8px"], M + 3,
                                             rep=8, detune=d)
                st = build_hybrid(FIX["S1_spacer_8px"], M, detune=d)
                res = st.solve()
                row[f"M{M}"]["vs_reference"] = compare_per_order(
                    np.asarray(res[1]), np.asarray(res[2]), np.asarray(res[0]),
                    Rr, Tr, orr)["max_per_order"]
            except Exception as exc:
                row[f"M{M}"]["vs_reference"] = repr(exc)
        lad[f"{d:.0e}"] = row
    payload["detune"] = lad

    # --- the MODULATION ladder: the window in which the defect is visible ---
    mod = {}
    for rel in (1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1):
        row = {}
        for fx in ("S1_spacer_8px", "C1_nospacer_8px"):
            r = solve_hybrid(FIX[fx], 4, cell_rel=rel, tap=False)
            row[fx] = {k: r.get(k) for k in ("closure", "sumRT", "error")}
        mod[f"{rel:.0e}"] = row
    payload["modulation"] = mod

    VC.dump(OUT, payload)

    st = VC.stamp()
    print(f"ARM = {st['arm']}  ({st['n_sqrt_decay_definitions']} definitions)")
    print(f"{'fixture':26s} {'M':>2s} {'closure':>13s} {'sum R+T':>13s} "
          f"{'vs ref':>12s} {'cond(a+b)':>11s}")
    for name, rows in payload["fixtures"].items():
        for M, row in rows.items():
            vr = row.get("vs_reference", {})
            vr = vr.get("max_per_order") if isinstance(vr, dict) else None
            cd = row.get("cond", {}).get("max_cond")
            print(f"{name:26s} {M:>2s} "
                  f"{row.get('closure', float('nan')):13.4e} "
                  f"{row.get('sumRT', float('nan')):13.6f} "
                  f"{(vr if vr is not None else float('nan')):12.4e} "
                  f"{(cd if cd is not None else float('nan')):11.3e}"
                  f"  {row.get('warnings') or ''} {row.get('error') or ''}")
    print("\nDETUNE ladder (S1, spacer relative detune):")
    for d, row in payload["detune"].items():
        m4 = row["M4"]
        print(f"  d={d:>9s}  M4 closure {m4.get('closure', float('nan')):12.4e}"
              f"  sumRT {m4.get('sumRT', float('nan')):12.6f}"
              f"  vs ref {m4.get('vs_reference')}")
    print("\nMODULATION ladder (M=4):")
    for rel, row in payload["modulation"].items():
        print(f"  rel={rel:>8s}  spacer {row['S1_spacer_8px'].get('closure')}"
              f"   none {row['C1_nospacer_8px'].get('closure')}")


if __name__ == "__main__":
    run()
