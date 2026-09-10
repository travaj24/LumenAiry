"""V4 -- the MOVE bar and the deliberate FALSE-REFUSAL attack (tasks 2 and 4).

Two questions, measured on devices chosen to make them hard.

**The move bar.**  ``_SLIVER_MOVE_FACTOR`` = 100 asks the answer to move more
than 100 widest-manufactured-cells before the move counts as more than the
geometric perturbation the snap describes.  The prescribed snap displaces a
wall by at most half the widest manufactured cell, so for an answer that is
still tracking its geometry the move is about ``s * w_wide``, where ``s`` is
the DEVICE's own slope ``dR/dx`` in per-order efficiency per unit wall
fraction.  A device whose ``s`` exceeds 100 therefore puts a CORRECT row past
the bar with no numerical pathology at all -- which is open item R2-D.

Stage 1 hunts for such a device the direct way: it scans the WALL POSITION
itself on a fine grid (the same coordinate the snap moves) and takes the
largest local slope, at two degrees so a slope that is a discretisation
artefact rather than the device is visible.  Resonant families are used
because that is where a large slope lives: a weak-contrast guided-mode
resonance grating, a Fabry-Perot cavity, and a near-Wood mount.

**The false refusal.**  A refusal needs BOTH arms: ``su_snapped`` inside the
closure (equivalently ``drop >= 100`` once the relative arm binds) AND
``move > 100 w_wide``.  The fix claims no CORRECT row meets both, over 1,369
rows.  Stage 2 scans wall steps and degrees on the steep mounts, on
MANY-SLICE tapered staircases (where the super-unity is the documented
many-interface quasi-resonance and is strongly grid-dependent), and on the
near-Wood family, and reports the joint approach: the largest
``min(drop / 100, move_w / 100)`` over the CORRECT population is how close
the attack came, and any row above 1 on both arms IS a false refusal.

    python v4_move.py out.json [--fast]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import v_fixtures as F  # noqa: E402


# ------------------------------------------------------- stage 1: slope ----
def wall_scan(mk, xs, deg):
    """``R``/``T`` over a fine grid of WALL SHIFTS -- the coordinate the
    prescribed snap moves.  Returns the per-point solves."""
    out = []
    for x in xs:
        try:
            out.append(F.unguarded(mk(float(x), deg)))
        except (ValueError, NotImplementedError):
            out.append(None)
    return out


def local_slope(solves, xs):
    """``max |dR/dx|`` over adjacent grid points, and where it occurs."""
    best, at = 0.0, None
    for i in range(len(xs) - 1):
        a, b = solves[i], solves[i + 1]
        if a is None or b is None:
            continue
        d = F.move_shared(a, b)
        if d is None:
            continue
        s = d / abs(float(xs[i + 1]) - float(xs[i]))
        if s > best:
            best, at = s, 0.5 * (float(xs[i]) + float(xs[i + 1]))
    return best, at


def r_zero(st):
    """The pol-1 zeroth-order reflectance -- the scalar the resonance is
    located on."""
    r = F.unguarded(st)
    j = int(np.argmin(np.abs(r["o"])))
    return float(r["R"][1, j])


HCG_SEEDS = ((0.65, 0.35e-6), (0.65, 0.45e-6), (0.55, 0.35e-6),
             (0.72, 0.35e-6))


def hunt(fast=False):
    """LOCATE the steepest devices, measuring rather than pinning.

    For each high-contrast-grating seed: a coarse wavelength scan finds the
    interval where the zeroth-order reflectance swings most, a fine scan
    inside it finds the resonance, and a DUTY scan at that wavelength gives
    the device's own ``dR/d(duty)``.  The near-Wood family is scanned in the
    wall coordinate directly.  Every slope is re-measured one degree up, so a
    slope that is a discretisation artefact rather than the device is
    visible."""
    out = []
    seeds = HCG_SEEDS[:1] if fast else HCG_SEEDS
    for duty0, tg in seeds:
        def mk(wl, du, g, tg=tg):
            return F.vhcg(0.0, wl=wl, duty=du, t_gr=tg, degree=g)
        wls = np.linspace(1.30e-6, 1.80e-6, 121 if fast else 251)
        try:
            v = np.array([r_zero(mk(w, duty0, 12)) for w in wls])
        except (ValueError, NotImplementedError):
            continue
        k = int(np.argmax(np.abs(np.diff(v))))
        fine = np.linspace(max(wls[k] - 1.1e-8, 1.2e-6), wls[k] + 1.1e-8,
                           121 if fast else 441)
        fv = np.array([r_zero(mk(w, duty0, 12)) for w in fine])
        kk = int(np.argmax(np.abs(np.diff(fv))))
        wl_star = float(0.5 * (fine[kk] + fine[kk + 1]))
        h = 1.0e-4 if fast else 5.0e-5
        duties = np.arange(duty0 - 0.05, duty0 + 0.05, h)
        dv = np.array([r_zero(mk(wl_star, float(x), 12)) for x in duties])
        sl = np.abs(np.diff(dv)) / h
        j = int(np.argmax(sl))
        duty_star = float(0.5 * (duties[j] + duties[j + 1]))
        # the same slope one degree up, locally -- the stationarity check
        loc = np.arange(duty_star - 5 * h, duty_star + 5 * h, h)
        d14 = np.array([r_zero(mk(wl_star, float(x), 14)) for x in loc])
        s14 = float(np.max(np.abs(np.diff(d14))) / h)
        d12 = np.array([r_zero(mk(wl_star, float(x), 12)) for x in loc])
        s12 = float(np.max(np.abs(np.diff(d12))) / h)
        out.append(dict(name=f"hcg_duty{duty0:g}_tg{tg * 1e9:.0f}",
                        kind="hcg", wl=wl_star, duty=duty_star, t_gr=tg,
                        slope=float(sl[j]), slope_local_deg12=s12,
                        slope_local_deg14=s14,
                        stationary=bool(abs(s14 - s12)
                                        <= 0.5 * max(s12, 1e-30)),
                        at=duty_star))
    for ins in ((5e-4, 4e-3) if not fast else (2e-3,)):
        xs = np.linspace(0.0, 0.06, 61 if fast else 241)
        def mkw(x, g, ins=ins):
            return F.vwood(0.0, inside=ins, a0=0.2410 - x, b0=0.7040 + x,
                           degree=g)
        try:
            sv = [F.unguarded(mkw(float(x), 12)) for x in xs]
        except (ValueError, NotImplementedError):
            continue
        best, at = 0.0, None
        for i in range(len(xs) - 1):
            dd = F.move_shared(sv[i], sv[i + 1])
            if dd is None:
                continue
            sl2 = dd / abs(float(xs[i + 1]) - float(xs[i]))
            if sl2 > best:
                best, at = sl2, 0.5 * (float(xs[i]) + float(xs[i + 1]))
        if at is None:
            continue
        loc = np.linspace(at - 5e-4, at + 5e-4, 9)
        def _loc(g):
            sv2 = [F.unguarded(mkw(float(x), g)) for x in loc]
            return max((F.move_shared(sv2[i], sv2[i + 1])
                        / abs(float(loc[i + 1]) - float(loc[i])))
                       for i in range(len(loc) - 1))
        s12, s14 = _loc(12), _loc(14)
        out.append(dict(name=f"wood_in{ins:g}", kind="wood", inside=ins,
                        slope=best, at=at, slope_local_deg12=s12,
                        slope_local_deg14=s14,
                        stationary=bool(abs(s14 - s12)
                                        <= 0.5 * max(s12, 1e-30))))
    return out


# ------------------------------------------- stage 2: the delta x degree ---
def scan(name, build, deg, deltas):
    try:
        ref = F.unguarded(build(0.0, deg))
    except (ValueError, NotImplementedError):
        return []
    rows = []
    for d in deltas:
        try:
            st = build(d, deg)
            cur = F.unguarded(st)
        except (ValueError, NotImplementedError):
            continue
        pre = F.prescribed(st)
        if pre is None:
            continue
        e = F.move_shared(cur, ref, pol=1)
        sn = F.snapped(st, pre["mf"])
        su = max(sn["worst"] - 1.0, 0.0)
        mv = F.move_shared(cur, sn)
        es = F.move_shared(sn, ref, pol=1)
        rows.append(dict(
            mount=name, degree=deg, delta=d, worst=cur["worst"],
            err_d=e / d, kind=F.classify(e, d), w_wide=pre["w_wide"],
            su_snap=su, drop=F.drop(cur["worst"], su),
            move_w=(None if mv is None else mv / pre["w_wide"]),
            err_snap_d=es / d, kind_snap=F.classify(es, d),
            arbitrated=bool(cur["worst"] - 1.0 > F.ps._SLIVER_TRIGGER_BAR),
        ))
    return rows


def attack_mounts(fast=False):
    """The mounts the false-refusal attack is run on beyond the hunted ones:
    MANY-SLICE tapered staircases and the near-Wood family."""
    out = []
    for nl in ((4, 8, 12) if not fast else (8,)):
        out.append((f"taper_nl{nl}",
                    (lambda d, g, nl=nl: F.vstair(
                        d, period=1.35e-6, wl=1.064e-6, theta=1.18,
                        a0=0.3120, b0=0.6790, e_lo=2.56, e_hi=12.25,
                        dz=0.24e-6 / nl, nl=nl, degree=g, nsup=2.05,
                        nsub=complex(2.90, 1.10), ffo=31))))
        out.append((f"taper_gmr_nl{nl}",
                    (lambda d, g, nl=nl: F.vgmr(d, nl=nl, degree=g))))
    for ins in ((5e-4, 8e-3) if not fast else (2e-3,)):
        out.append((f"wood_in{ins:g}",
                    (lambda d, g, ins=ins: F.vwood(d, inside=ins, degree=g))))
    return out


def summarise(rows, hunted):
    arb = [r for r in rows if r["arbitrated"]]
    right = [r for r in arb if r["kind"] == "right"]
    grey = [r for r in arb if r["kind"] == "grey"]

    def env(seq, key):
        v = [r[key] for r in seq if r.get(key) is not None
             and np.isfinite(r[key])]
        return max(v) if v else None

    def approach(seq):
        best, who = 0.0, None
        for r in seq:
            if r.get("move_w") is None:
                continue
            dv = r["drop"] if np.isfinite(r["drop"]) else 1e300
            a = min(dv / 100.0, r["move_w"] / 100.0)
            if a > best:
                best, who = a, r
        return best, who

    a_r, who_r = approach(right)
    a_g, who_g = approach(grey)
    false_ref = [r for r in right
                 if r.get("move_w") is not None and r["move_w"] > 100.0
                 and (not np.isfinite(r["drop"]) or r["drop"] >= 100.0)]
    s = dict(
        rows=len(rows), arbitrated=len(arb), arb_right=len(right),
        arb_grey=len(grey),
        arb_wrong=sum(1 for r in arb if r["kind"] == "wrong"),
        slope_max=max((h["slope"] for h in hunted), default=None),
        slope_over_100=sum(1 for h in hunted if h["slope"] > 100.0),
        slope_top=[dict(name=h["name"], slope=h["slope"], at=h["at"],
                        local12=h["slope_local_deg12"],
                        local14=h["slope_local_deg14"],
                        stationary=h["stationary"])
                   for h in sorted(hunted, key=lambda h: -h["slope"])[:8]],
        correct_move_w_envelope=env(right, "move_w"),
        correct_move_w_over_100=sum(
            1 for r in right if r.get("move_w") is not None
            and r["move_w"] > 100.0),
        correct_move_w_over_100_rows=[
            F.jsonable(r) for r in right
            if r.get("move_w") is not None and r["move_w"] > 100.0][:12],
        correct_finite_drop_envelope=env(right, "drop"),
        correct_infinite_drop=sum(1 for r in right
                                  if not np.isfinite(r["drop"])),
        grey_move_w_envelope=env(grey, "move_w"),
        grey_finite_drop_envelope=env(grey, "drop"),
        false_refusals=len(false_ref),
        false_refusal_rows=[F.jsonable(r) for r in false_ref[:12]],
        closest_approach_correct=a_r,
        closest_approach_correct_row=(None if who_r is None
                                      else F.jsonable(who_r)),
        closest_approach_grey=a_g,
        closest_approach_grey_row=(None if who_g is None
                                   else F.jsonable(who_g)),
        correct_top_move=[F.jsonable(r) for r in sorted(
            [r for r in right if r.get("move_w") is not None],
            key=lambda r: -r["move_w"])[:6]],
        correct_top_drop=[F.jsonable(r) for r in sorted(
            [r for r in right if np.isfinite(r["drop"])],
            key=lambda r: -r["drop"])[:6]],
    )
    return s


def main():
    fast = "--fast" in sys.argv
    dest = ([a for a in sys.argv[1:] if not a.startswith("--")]
            or ["v4_move.json"])[0]
    t0 = time.perf_counter()
    hunted = hunt(fast)
    degs = (6, 8, 10) if not fast else (8,)
    deltas = list(np.logspace(-2.5, -6.0, 10 if fast else 22))
    rows = []
    steep = sorted(hunted, key=lambda h: -h["slope"])[:(1 if fast else 4)]
    mounts = []
    for h in steep:
        if h["kind"] == "hcg":
            for nl in ((2, 3, 4) if not fast else (3,)):
                mounts.append((
                    f"RES_{h['name']}_nl{nl}",
                    (lambda d, g, h=h, nl=nl: F.vhcg(
                        d, wl=h["wl"], duty=h["duty"], t_gr=h["t_gr"],
                        degree=g, nl=nl))))
        else:
            for nl in ((2, 3) if not fast else (2,)):
                mounts.append((
                    f"RES_{h['name']}_nl{nl}",
                    (lambda d, g, h=h, nl=nl: F.vwood(
                        d, inside=h["inside"], a0=0.2410 - h["at"],
                        b0=0.7040 + h["at"], degree=g, nl=nl))))
    mounts += attack_mounts(fast)
    for name, build in mounts:
        for g in degs:
            rows += scan(name, build, g, deltas)
    s = summarise(rows, hunted)
    s["wall"] = time.perf_counter() - t0
    doc = dict(build=F.build_info(), summary=s, hunted=hunted, rows=rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(doc), fh, indent=1)
    for k in sorted(s):
        if not isinstance(s[k], (list, dict)):
            print(f"{k:36s} {s[k]}")
    print("top slopes:", [(h["name"], round(h["slope"], 1))
                          for h in s["slope_top"][:5]])
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
