"""TASK F -- aggregate every vf_*.json into the cross-arm envelope table."""
from __future__ import annotations

import glob
import json
import math
import pathlib

D = pathlib.Path(r"C:/tmp/lum_vbor/validation/probe_verify_bor_guards")


def load(pat):
    out = []
    for p in sorted(glob.glob(str(D / pat))):
        try:
            out.append((pathlib.Path(p).name, json.load(open(p))))
        except Exception as e:                                # noqa: BLE001
            print("  (unreadable %s: %s)" % (p, e))
    return out


def dec(x, bar, below=True):
    if x is None or x == 0 or not math.isfinite(x):
        return float("inf")
    return math.log10(bar / x) if below else math.log10(x / bar)


def env(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None, None, None
    return min(vals), max(vals), (max(vals) / min(vals) if min(vals) else None)


print("#" * 78)
print("# PROBE A -- orientation kernel / classifier band")
rows = load("vf_a_orient_post_*.json")
print("arms:", len(rows))
hdr = ("%-22s %12s %12s %12s %12s %12s %8s"
       % ("arm", "closure", "ladderworst", "noise_ord", "noise_cut",
          "sig1e-6", "counts"))
print(hdr)
cl, lw, no, nc, s6, cnts, tot, dis, rmin = [], [], [], [], [], set(), [], [], []
for name, d in rows:
    a = "%s/%s/t%s" % (d["arm"]["platform"], d["arm"]["loaded_kernel"],
                       d["arm"]["blas_threads"])
    cl.append(d["near_cutoff_closure"]["closure"])
    lw.append(d["ladder"]["worst"])
    no.append(d["noise_ordinary"]["worst"])
    nc.append(d["noise_cutoff"]["worst"])
    s6.append(d["signal_side"]["imn_1e-06"]["smallest"])
    cnts.update(d["ladder"]["counts"])
    tot.append(d["widening"]["total"])
    dis.append(d["widening"]["disagree"])
    rmin.append(d["widening"]["relmin"])
    print("%-22s %12.4e %12.4e %12.4e %12.4e %12.4e %8s"
          % (a, cl[-1], lw[-1], no[-1], nc[-1], s6[-1], d["ladder"]["counts"]))
for label, vals, bar, below in (
        ("near_cutoff closure   (bar 1e-8)", cl, 1e-8, True),
        ("ladder worst closure  (bar 1e-6)", lw, 1e-6, True),
        ("noise ordinary sigma  (bar 1e-11)", no, 1e-11, True),
        ("noise cutoff sigma    (bar 5e-9)", nc, 5e-9, True),
        ("signal Im(n)=1e-6     (bar 2e-8)", s6, 2e-8, False)):
    lo, hi, sp = env(vals)
    worst = hi if below else lo
    print("  %-34s env %.4e .. %.4e (%.1fx)  worst margin %.2fx = %.2f dec"
          % (label, lo, hi, sp, (bar / worst) if below else (worst / bar),
             dec(worst, bar, below)))
print("  ladder channel counts seen across all arms:", sorted(cnts))
print("  widening total env", env(tot)[:2], "disagree", set(dis),
      "relmin", env(rmin)[:2])

print()
print("#" * 78)
print("# PROBE B -- nodal passivity")
rows = load("vf_b_nodal_post_*.json")
print("arms:", len(rows))
print("%-22s %10s %10s %10s %10s %10s %9s"
      % ("arm", "stag_twin", "healthy", "mild", "floorRT", "floormaxdev",
         "refusals"))
st, hl, ml, fr, fm, msgok = [], [], [], [], [], set()
for name, d in rows:
    a = "%s/%s/t%s" % (d["arm"]["platform"], d["arm"]["loaded_kernel"],
                       d["arm"]["blas_threads"])
    st.append(d["staggered_twin"]["worst"])
    hl.append(d["healthy_uniform_small_cell"]["worst"])
    ml.append(d["mildest_broken_row"]["violation"])
    fr.append(d["bor_solve_floor_stack"]["max_RT"])
    fm.append(d["bor_solve_floor_stack"]["max_abs_dev"])
    msgok.add(d["bor_solve_floor_stack"]["msg_contains_1_0288"])
    nref = sum(1 for r in d["nodal_five_layer_armed"] if r["raised"])
    print("%-22s %10.4e %10.4e %10.4e %10.6g %10.6g %5d/5"
          % (a, st[-1], hl[-1], ml[-1], fr[-1], fm[-1], nref))
for label, vals, bar, below in (
        ("staggered twin closure (bar 1e-9)", st, 1e-9, True),
        ("healthy nodal          (bar 1e-5)", hl, 1e-5, True),
        ("mildest broken row     (bar 1e-2)", ml, 1e-2, False),
        ("floor stack max|E-1|   (bar 0.05)", fm, 0.05, True)):
    lo, hi, sp = env(vals)
    worst = hi if below else lo
    print("  %-36s env %.6g .. %.6g (%.4gx)  worst margin %.3gx = %.2f dec"
          % (label, lo, hi, sp, (bar / worst) if below else (worst / bar),
             dec(worst, bar, below)))
print("  message contains '1.0288' on every arm:", msgok)
print("  max(R+T) env:", env(fr)[:2])
bt = [d["bor_solve_staggered_twin"]["closure"] for _n, d in rows]
lo, hi, sp = env(bt)
print("  bor_solve staggered twin closure env %.4e .. %.4e (%.3gx) bar 1e-9 "
      "-> %.2f dec" % (lo, hi, sp, dec(hi, 1e-9, True)))
n5 = {d["nodal_five_layer_disarmed"][0]["violation"] for _n, d in rows}
mv = min(min(r["violation"] for r in d["nodal_five_layer_disarmed"])
         for _n, d in rows)
print("  five-layer nodal: mildest violation across all arms %.6g "
      "(refusal bar 1e-3 -> %.2f dec below it)" % (mv, dec(mv, 1e-3, False)))
print("  proxy warned pattern:",
      {tuple(r["proxy_warned"] for r in d["nodal_five_layer_armed"])
       for _n, d in rows})

print()
print("#" * 78)
print("# PROBE C -- SEM contract (delta ladder / census hook)")
rows = load("vf_c_sem_post_*.json")
print("arms:", len(rows))
print("%-22s %8s %10s %12s %12s %6s %10s %10s"
      % ("arm", "verdicts", "fu(1e-6)", "q(1e-6)", "q(1e-7)", "n_inv",
         "min_rcond", "max_resid"))
rc, rs, ninv, verd, q6 = [], [], [], set(), []
for name, d in rows:
    a = "%s/%s/t%s" % (d["arm"]["platform"], d["arm"]["loaded_kernel"],
                       d["arm"]["blas_threads"])
    lad = {r["delta_frac"]: r for r in d["delta_ladder"]}
    tbl = "".join({"ok": "o", "warn": "w", "refuse": "R"}[
        lad[k]["verdict"]] for k in sorted(lad, reverse=True))
    verd.add(tbl)
    rc.append(d["inv_census"]["min_rcond"])
    rs.append(d["inv_census"]["max_resid"])
    ninv.append(d["inv_census"]["n"])
    q6.append(lad[1e-6]["q_excess"])
    print("%-22s %8s %10.6g %12.6g %12.6g %6d %10.4e %10.4e"
          % (a, tbl, lad[1e-6]["w_min_union_frac"], lad[1e-6]["q_excess"],
             lad[1e-7]["q_excess"], d["inv_census"]["n"],
             d["inv_census"]["min_rcond"], d["inv_census"]["max_resid"]))
print("  distinct verdict tables (1e-1..1e-7):", verd)
print("  n inverses env:", env([float(x) for x in ninv])[:2], " (bar > 200)")
lo, hi, sp = env(rc)
print("  min rcond env %.4e .. %.4e (%.3gx)  worst margin %.2f dec above 1e-9"
      % (lo, hi, sp, dec(lo, 1e-9, False)))
lo, hi, sp = env(rs)
print("  max resid env %.4e .. %.4e (%.3gx)  worst margin %.2f dec below 1e-9"
      % (lo, hi, sp, dec(hi, 1e-9, True)))
lo, hi, sp = env(q6)
print("  q_excess at delta=1e-6 env %.6g .. %.6g (%.3gx), bar 1e4 -> %.3gx"
      % (lo, hi, sp, lo / 1e4))

print()
print("#" * 78)
print("# PROBE D -- SEM ordinary census + taper")
for name, d in load("vf_d_census_post_*.json"):
    a = "%s/%s/t%s" % (d["arm"]["platform"], d["arm"]["loaded_kernel"],
                       d["arm"]["blas_threads"])
    print("--", a, name)
    for r in d.get("ordinary_census", []):
        print("   census %-6s d%-3d fams=%d narrowest=%.4e (%.4gx; assert "
              "> %.1e) worst_q=%.6g (assert < %.4g) bad=%s"
              % (r["group"], r["degree"], r["n_families"], r["narrowest"],
                 r["narrowest_over_bar"], r["assert_narrow_bar"],
                 r["worst_q"], r["assert_q_bar"], r["bad"]))
    for r in d.get("taper", []):
        print("   taper n=%-4d d%-3d k0=%-4g v=%s w=%.6e (assert > %.1e, "
              "%.3gx) q=%.6g (assert < %.5g, %.3gx)"
              % (r["n_slices"], r["degree"], r["k0"], r["verdicts"], r["w"],
                 r["assert_w_bar"], r["w"] / r["assert_w_bar"], r["q"],
                 r["assert_q_bar"], r["assert_q_bar"] / r["q"]))

print()
print("#" * 78)
print("# PROBE E -- EME branch cut")
for name, d in load("vf_e_eme_post_*.json"):
    a = "%s/%s/t%s" % (d["arm"]["platform"], d["arm"]["loaded_kernel"],
                       d["arm"]["blas_threads"])
    lm = d["layer_modes_narrowed"]
    sf = d["strip_flip"]
    pf = d.get("prefix_pin_narrowed", {})
    print("%-22s flip=%d/%d worst/tol=%.3e | modes %d vs %d dqz2=%.4e "
          "(atol 1e-6) | prefix-pin %s vs %s"
          % (a, sf["flipped"], sf["n"], sf["worst_ratio_to_tol"],
             lm["n_real"], lm["n_tiny"], lm["worst_dqz2"],
             pf.get("n_real"), pf.get("n_tiny")))

print()
print("#" * 78)
print("# PROBE F -- forced premise")
for name, d in load("vf_f_forced_post_*.json"):
    a = "%s/%s/t%s" % (d["arm"]["platform"], d["arm"]["loaded_kernel"],
                       d["arm"]["blas_threads"])
    print("--", a, name)
    for k in ("ncut_N", "ncount", "taper512", "nodalN", "floormsg"):
        for r in d.get(k, []):
            print("   %-9s %s" % (k, r))
