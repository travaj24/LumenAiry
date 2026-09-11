"""TASK C -- the nodal passivity refusal, and the FLAGGED gate.

THE QUESTION THE FLAG ASKS.  The shipped gate
``test_bor_solve::test_structured_stack_energy_floor_nodal`` asserted that the
legacy nodal basis's "documented ~1-4% floor" held (``max|R+T-1| < 0.05``) on a
lossless three-layer stack, and it read 1.02882.  5.45.1 replaces that
assertion with a REFUSAL.  Either

  (i) the nodal answer at that fixture is within its OWN discretisation error
      of the truth -- in which case 2.9 % super-unity is the floor doing what
      the docstring said, the refusal is a FALSE POSITIVE, and changing the
      gate is wrong; or
  (ii) the nodal answer is WRONG -- per-channel, not merely in its sum -- in
      which case the old gate was rationalising a defect.

This probe decides that by measuring the nodal answer against three
independent references on the SAME geometry:

  * its STAGGERED twin through the same ``bor_solve`` cascade (div-conforming,
    closes energy to ~1e-12);
  * a CONVERGED LADDER in N on the staggered basis (N = 100 .. 500) -- the
    reference the nodal answer must lie within its own error of;
  * the UNIFORM-slab analytic limit, on a fixture where the "grating" layer is
    made uniform so a closed form exists (a control that certifies the
    cascade, the channel bookkeeping and this probe's own channel matching).

It then re-derives the two-sided bar on ITS OWN populations, checks the switch
restores the pre-fix number and the pre-fix assertion exactly, and checks a
LOSSY nodal stack -- where ``R + T < 1`` is legitimate -- is never refused.

Usage:  python v3_nodal.py <pre|post> [outdir]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def _stack(basis, m=1, R=4.0, N=200, k0=2.0, prof=None, thick=0.5,
           e_out=2.0):
    from lumenairy.elements.bor.bor_solve import build_layer
    prof = prof if prof is not None else _ring(0.8, 2.0, 6.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return k0, [build_layer(m, R, N, _uni(e_out), k0, basis=basis),
                    build_layer(m, R, N, prof, k0, thickness=thick,
                                basis=basis),
                    build_layer(m, R, N, _uni(e_out), k0, basis=basis)]


def _solve_disarmed(layers, k0):
    """Solve with the guard (if present) DISARMED, so the row is the number
    the solver RETURNS -- the population must be the pre-fix one on both
    builds or the comparison is not a comparison."""
    import lumenairy.elements.bor.bor_solve as bs
    had = hasattr(bs, "BOR_NODAL_PASSIVITY_GUARD")
    prev = getattr(bs, "BOR_NODAL_PASSIVITY_GUARD", None)
    if had:
        bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bs.solve(layers, k0)
    finally:
        if had:
            bs.BOR_NODAL_PASSIVITY_GUARD = prev


def _match_channels(res_a, res_b, k0, rtol=2e-2):
    """Pair incident channels of two solves by their axial index qn = q/k0.

    The two bases select their propagating sets by different screens (the
    nodal basis adds a divergence tag), so the channel ARRAYS are not
    index-aligned and a per-channel comparison must match by physics."""
    qa = np.asarray(res_a["q_inc"]) / k0
    qb = np.asarray(res_b["q_inc"]) / k0
    pairs = []
    used = set()
    for i, va in enumerate(qa):
        d = np.abs(qb - va) / max(abs(va), 1e-300)
        order = np.argsort(d)
        for j in order:
            if j in used:
                continue
            if d[j] <= rtol:
                pairs.append((i, int(j), float(d[j])))
                used.add(int(j))
            break
    return pairs, qa, qb


def the_flagged_fixture():
    """The shipped gate's stack, measured against three references."""
    out = {}
    k0, ln = _stack("nodal")
    rn = _solve_disarmed(ln, k0)
    Rn, Tn = np.asarray(rn["R"]), np.asarray(rn["T"])
    out["nodal"] = dict(n_inc=int(Rn.size), R=Rn.tolist(), T=Tn.tolist(),
                        energy=(Rn + Tn).tolist(),
                        max_energy=float(np.max(Rn + Tn)),
                        qn_inc=(np.asarray(rn["q_inc"]) / k0).tolist())

    k0s, ls = _stack("staggered")
    rs = _solve_disarmed(ls, k0s)
    Rs, Ts = np.asarray(rs["R"]), np.asarray(rs["T"])
    out["staggered"] = dict(n_inc=int(Rs.size), R=Rs.tolist(), T=Ts.tolist(),
                            energy=(Rs + Ts).tolist(),
                            max_energy=float(np.max(Rs + Ts)),
                            qn_inc=(np.asarray(rs["q_inc"]) / k0s).tolist())

    pairs, qa, qb = _match_channels(rn, rs, k0)
    per = []
    for i, j, d in pairs:
        per.append(dict(qn=float(qa[i]), dq_rel=d,
                        R_nodal=float(Rn[i]), R_stag=float(Rs[j]),
                        T_nodal=float(Tn[i]), T_stag=float(Ts[j]),
                        dR=float(Rn[i] - Rs[j]), dT=float(Tn[i] - Ts[j]),
                        dR_abs=float(abs(Rn[i] - Rs[j])),
                        dT_abs=float(abs(Tn[i] - Ts[j]))))
    out["per_channel_vs_staggered"] = per
    out["matched"] = len(per)
    out["worst_dR"] = max((p["dR_abs"] for p in per), default=float("nan"))
    out["worst_dT"] = max((p["dT_abs"] for p in per), default=float("nan"))

    # --- converged ladder on the staggered basis --------------------------
    lad = []
    for N in (100, 150, 200, 300, 400, 500):
        k0n, lN = _stack("staggered", N=N)
        rN = _solve_disarmed(lN, k0n)
        lad.append(dict(N=N, n_inc=int(np.asarray(rN["R"]).size),
                        max_energy=float(np.max(np.asarray(rN["R"])
                                                + np.asarray(rN["T"]))),
                        R=np.asarray(rN["R"]).tolist(),
                        qn=(np.asarray(rN["q_inc"]) / k0n).tolist()))
    out["staggered_ladder"] = lad

    # nodal in N -- the "does NOT improve with N" claim, re-measured
    nlad = []
    for N in (100, 150, 200, 300, 400):
        k0n, lN = _stack("nodal", N=N)
        rN = _solve_disarmed(lN, k0n)
        RN, TN = np.asarray(rN["R"]), np.asarray(rN["T"])
        nlad.append(dict(N=N, n_inc=int(RN.size),
                         max_energy=float(np.max(RN + TN)),
                         min_energy=float(np.min(RN + TN)),
                         R=RN.tolist(),
                         qn=(np.asarray(rN["q_inc"]) / k0n).tolist()))
    out["nodal_ladder"] = nlad
    return out


def uniform_control():
    """A UNIFORM 'grating' layer: the same cascade, the same channel
    bookkeeping, but a geometry whose per-channel R and T have a closed form
    (each radial mode is a planar Fresnel problem at its own oblique angle).
    Certifies that a per-channel comparison of this shape can detect a wrong
    answer at all."""
    out = {}
    for basis in ("nodal", "staggered"):
        k0, L = _stack(basis, prof=_uni(6.0))
        r = _solve_disarmed(L, k0)
        R, T = np.asarray(r["R"]), np.asarray(r["T"])
        qn = np.asarray(r["q_inc"]) / k0
        # analytic scalar slab per mode: gamma^2 = eps k0^2 - q^2 shared
        e_out, e_l, d = 2.0, 6.0, 0.5
        ana_R, ana_T = [], []
        for v in qn:
            g2 = e_out * k0 ** 2 - (v * k0) ** 2          # transverse^2
            q_out = np.sqrt(complex(e_out * k0 ** 2 - g2))
            q_l = np.sqrt(complex(e_l * k0 ** 2 - g2))
            # TE (scalar) Fresnel for a symmetric slab
            r12 = (q_out - q_l) / (q_out + q_l)
            ph = np.exp(2j * q_l * d)
            rr = r12 * (1 - ph) / (1 - r12 ** 2 * ph)
            tt = (1 - r12 ** 2) * np.exp(1j * q_l * d) / (1 - r12 ** 2 * ph)
            ana_R.append(float(abs(rr) ** 2))
            ana_T.append(float(abs(tt) ** 2))
        out[basis] = dict(qn=qn.tolist(), R=R.tolist(), T=T.tolist(),
                          energy=(R + T).tolist(),
                          max_energy=float(np.max(R + T)),
                          ana_R_TE=ana_R, ana_T_TE=ana_T,
                          dR_TE=[float(abs(a - b))
                                 for a, b in zip(R, ana_R)],
                          dT_TE=[float(abs(a - b))
                                 for a, b in zip(T, ana_T)])
    return out


def population():
    """MY OWN nodal / staggered populations, guard disarmed: the two-sided bar
    re-derived rather than read."""
    rows = []
    for basis in ("nodal", "staggered"):
        for m in (0, 1, 2):
            for N in (120, 200):
                for rl in (0.5, 1.0, 2.0, 4.0, 8.0, 14.0):
                    k0 = 2.0
                    R = rl * 2.0 * np.pi / k0
                    for fam, prof, thk in (
                            ("uniform", _uni(4.0), 0.4),
                            ("ring", _ring(0.8, 2.0, 6.0), 0.5)):
                        try:
                            k0v, L = _stack(basis, m=m, R=R, N=N, k0=k0,
                                            prof=prof, thick=thk)
                            r = _solve_disarmed(L, k0v)
                            E = np.asarray(r["R"]) + np.asarray(r["T"])
                            rows.append(dict(
                                basis=basis, m=m, N=N, Rbig_over_lambda=rl,
                                family=fam, n_inc=int(E.size),
                                max_energy=float(np.max(E)) if E.size else
                                float("nan"),
                                excess=float(np.max(E)) - 1.0 if E.size
                                else float("nan")))
                        except BaseException as e:   # noqa: BLE001
                            rows.append(dict(basis=basis, m=m, N=N,
                                             Rbig_over_lambda=rl, family=fam,
                                             raised=type(e).__name__))
    return rows


def lossy_nodal():
    """A LOSSY nodal stack: ``R + T < 1`` is legitimate, so the screen must
    never judge it.  Also a lossy stack whose nodal cascade BLOWS UP -- the
    honest question being whether disarming on loss hides real damage."""
    rows = []
    for imn in (1e-1, 1e-2, 1e-4, 1e-8, 1e-13, 1e-14, 0.0):
        e = complex(2.0, 2.0 * imn)
        for rl in (1.0, 4.0, 12.0):
            k0 = 2.0
            R = rl * 2.0 * np.pi / k0
            rec = dict(im_rel=imn, Rbig_over_lambda=rl)
            try:
                k0v, L = _stack("nodal", R=R, k0=k0,
                                prof=_ring(0.8, e, 6.0 + 0j), e_out=e)
                import lumenairy.elements.bor.bor_solve as bs
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = bs.solve(L, k0v)
                E = np.asarray(r["R"]) + np.asarray(r["T"])
                rec.update(raised=None,
                           max_energy=float(np.max(E)) if E.size else
                           float("nan"),
                           n_warn=len(w),
                           warn=[str(x.message)[:120] for x in w])
                rd = _solve_disarmed(L, k0v)
                Ed = np.asarray(rd["R"]) + np.asarray(rd["T"])
                rec["max_energy_disarmed"] = (float(np.max(Ed)) if Ed.size
                                              else float("nan"))
            except BaseException as exc:             # noqa: BLE001
                rec.update(raised=type(exc).__name__, msg=str(exc)[:160])
            rows.append(rec)
    return rows


def switch_restores():
    """The switch must return the PRE-fix number bit for bit AND satisfy the
    pre-fix assertion."""
    import lumenairy.elements.bor.bor_solve as bs
    k0, L = _stack("nodal")
    out = {}
    if hasattr(bs, "BOR_NODAL_PASSIVITY_GUARD"):
        prev = bs.BOR_NODAL_PASSIVITY_GUARD
        bs.BOR_NODAL_PASSIVITY_GUARD = False
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = bs.solve(L, k0)
            out["switch_off"] = dict(
                hash=_vh.hash_arrays(np.asarray(r["R"]), np.asarray(r["T"])),
                n_inc=int(np.asarray(r["R"]).size),
                max_energy=float(np.max(np.asarray(r["R"])
                                        + np.asarray(r["T"]))),
                warnings=[str(x.message)[:160] for x in w])
        finally:
            bs.BOR_NODAL_PASSIVITY_GUARD = prev
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r2 = bs.solve(L, k0)
            out["switch_on"] = dict(raised=None,
                                    hash=_vh.hash_arrays(
                                        np.asarray(r2["R"]),
                                        np.asarray(r2["T"])))
        except BaseException as e:                   # noqa: BLE001
            out["switch_on"] = dict(raised=type(e).__name__, msg=str(e)[:400])
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = bs.solve(L, k0)
        out["prefix"] = dict(
            hash=_vh.hash_arrays(np.asarray(r["R"]), np.asarray(r["T"])),
            n_inc=int(np.asarray(r["R"]).size),
            max_energy=float(np.max(np.asarray(r["R"])
                                    + np.asarray(r["T"]))),
            warnings=[str(x.message)[:160] for x in w])
    return out


def main():
    build = sys.argv[1]
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    res = dict(arm=a, build=build)
    with _vh.timed("flagged"):
        res["flagged"] = the_flagged_fixture()
    print("  nodal max energy %.6f" % res["flagged"]["nodal"]["max_energy"])
    print("  stag  max energy %.12f"
          % res["flagged"]["staggered"]["max_energy"])
    print("  matched channels %d  worst dR %.4g  worst dT %.4g"
          % (res["flagged"]["matched"], res["flagged"]["worst_dR"],
             res["flagged"]["worst_dT"]))
    with _vh.timed("uniform_control"):
        res["uniform_control"] = uniform_control()
    with _vh.timed("population"):
        res["population"] = population()
    with _vh.timed("lossy"):
        res["lossy"] = lossy_nodal()
    with _vh.timed("switch"):
        res["switch"] = switch_restores()
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    _vh.dump("%s/v3_nodal_%s_%s_%s_t%s.json"
             % (out, build, a["platform"], a["loaded_kernel"],
                a["blas_threads"]), res)


if __name__ == "__main__":
    main()
