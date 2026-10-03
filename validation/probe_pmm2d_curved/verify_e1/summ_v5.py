"""Collect the V5 pillar readings: ladders, topologies, staircases, the
exact-disk RCWA (rates on >= 4 points and Richardson), the z-staircase
(rates in N and in N_z on >= 4 points each, double extrapolation), the
mirror identity, reciprocity / Onsager.  Output v5_summary.json."""
import glob
import json
import os

import _ve1common as V
import numpy as np

H = V.HERE


def load(pat):
    out = []
    for fn in sorted(glob.glob(os.path.join(H, pat))):
        d = json.load(open(fn))
        d["_f"] = os.path.basename(fn)
        d["vec"] = np.asarray(d["vec"])
        out.append(d)
    return out


def dist(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


S = {}
for mat, sx in (("oop", 0.0), ("eps35", 0.15)):
    tagc = f"s{sx:+.2f}"
    c3 = {d["M"]: d for d in load(f"v5_curved_{mat}_c3_t0_p0_{tagc}_M*.json")}
    c5 = {d["M"]: d for d in load(f"v5_curved_{mat}_c5_t0_p0_{tagc}_M*.json")}
    if not c3:
        continue
    top3 = c3[max(c3)]
    ref = top3["vec"]
    r = {"c3": {M: dict(dist_top=dist(d["vec"], ref), clo=d["clo"],
                        step=(dist(d["vec"], c3[M - 1]["vec"])
                              if M - 1 in c3 else None))
                for M, d in sorted(c3.items())},
         "c5": {M: dict(dist_c3top=dist(d["vec"], ref), clo=d["clo"],
                        step=(dist(d["vec"], c5[M - 1]["vec"])
                              if M - 1 in c5 else None))
                for M, d in sorted(c5.items())},
         "c3_top": max(c3), "c5_top": max(c5) if c5 else None}
    ref5 = c5[max(c5)]["vec"] if c5 else ref
    r["topologies"] = dist(ref, ref5)
    st = load(f"v5_stair_{mat}_k*_{tagc}_M*.json")
    r["stair"] = {}
    for d in st:
        r["stair"].setdefault(d["k"], {})[d["M"]] = dict(
            dist_c5top=dist(d["vec"], ref5), clo=d["clo"])
    if mat == "oop":
        rc = {d["n"]: d for d in load("v5_rcwa_oop_t0.000_p0.000_n*.json")}
        ns = sorted(rc)
        r["rcwa"] = {n: dict(dist_c5top=dist(rc[n]["vec"], ref5),
                             nxd=dist(rc[n]["vec"], ref5) * (2 * n + 1),
                             clo=rc[n]["clo"], ff=rc[n]["ffcheck"],
                             calls=rc[n]["calls"]) for n in ns}
        # local rates between consecutive N (N = 2n + 1 harmonics per axis)
        rates = []
        for a, b in zip(ns, ns[1:]):
            da, db = dist(rc[a]["vec"], ref5), dist(rc[b]["vec"], ref5)
            rates.append(float(np.log(da / db) / np.log((2 * b + 1)
                                                         / (2 * a + 1))))
        r["rcwa_rates_vs_curved"] = rates
        # self-convergence rates (no reference): |v_n - v_{n+1}|
        sc = []
        for a, b, c in zip(ns, ns[1:], ns[2:]):
            d1 = dist(rc[a]["vec"], rc[b]["vec"])
            d2 = dist(rc[b]["vec"], rc[c]["vec"])
            sc.append(float(np.log(d1 / d2) / np.log((2 * c + 1)
                                                     / (2 * a + 1))))
        r["rcwa_self_rates"] = sc
        # Richardson in 1 / N on consecutive pairs
        rich = {}
        for a, b in zip(ns, ns[1:]):
            Na, Nb = 2 * a + 1, 2 * b + 1
            ext = (Nb * rc[b]["vec"] - Na * rc[a]["vec"]) / (Nb - Na)
            rich[f"{a}-{b}"] = dist(ext, ref5)
        r["rcwa_richardson_1overN"] = rich
    if mat == "eps35":
        zs = load("v5_zstair_s+0.15_Nz*_n*.json")
        Z = {}
        for d in zs:
            Z[(d["Nz"], d["n"])] = d["vec"]
        Nzs = sorted({k[0] for k in Z})
        ns = sorted({k[1] for k in Z})
        r["zstair_raw"] = {f"{a}_{b}": dist(v, ref5) for (a, b), v in
                           sorted(Z.items())}
        # Richardson in 1 / N per N_z (last pair), then in 1 / N_z^2
        perNz = {}
        for a in Nzs:
            have = [n for n in ns if (a, n) in Z]
            if len(have) < 2:
                continue
            n1, n2 = have[-2], have[-1]
            N1, N2 = 2 * n1 + 1, 2 * n2 + 1
            perNz[a] = (N2 * Z[(a, n2)] - N1 * Z[(a, n1)]) / (N2 - N1)
            # rates in N at this N_z (self-convergence, >= 3 points)
            if len(have) >= 3:
                rr = []
                for x, y, z in zip(have, have[1:], have[2:]):
                    d1 = dist(Z[(a, x)], Z[(a, y)])
                    d2 = dist(Z[(a, y)], Z[(a, z)])
                    rr.append(float(np.log(d1 / d2)
                                    / np.log((2 * z + 1) / (2 * x + 1))))
                r.setdefault("zstair_rates_in_N", {})[a] = rr
        r["zstair_richN"] = {a: dist(v, ref5) for a, v in perNz.items()}
        ks = sorted(perNz)
        # rates in N_z on the Richardson-in-N values (self-convergence)
        rz = []
        for x, y, z in zip(ks, ks[1:], ks[2:]):
            d1, d2 = dist(perNz[x], perNz[y]), dist(perNz[y], perNz[z])
            rz.append(float(np.log(d1 / d2) / np.log(z / x)))
        r["zstair_rates_in_Nz"] = rz
        dbl = {}
        for x, y in zip(ks, ks[1:]):
            ext = (y * y * perNz[y] - x * x * perNz[x]) / (y * y - x * x)
            dbl[f"{x}-{y}"] = dist(ext, ref5)
        r["zstair_double_extrapolated"] = dbl
        # the mirror identity: R(m, n; -t) = R(-m, n; +t)
        for kind in ("c3", "c5"):
            mm = load(f"v5_curved_eps35_{kind}_t0_p0_s-0.15_M*.json")
            for d in mm:
                M = d["M"]
                pos = (c3 if kind == "c3" else c5).get(M)
                if pos is None:
                    continue
                perm = [V.ORD9.index((-m, n)) for m, n in V.ORD9]
                v = pos["vec"].reshape(4, 9)[:, perm].ravel()
                r.setdefault("mirror", {})[f"{kind}_M{M}"] = dict(
                    mirror_identity=dist(d["vec"], v),
                    minus_vs_plus=dist(d["vec"], pos["vec"]))
        zm = load("v5_zstair_s-0.15_Nz16_n13.json")
        if zm and (16, 13) in Z:
            perm = [V.ORD9.index((-m, n)) for m, n in V.ORD9]
            v = Z[(16, 13)].reshape(4, 9)[:, perm].ravel()
            r["zstair_mirror_identity"] = dist(zm[0]["vec"], v)
    S[mat] = r
# conical / oblique rungs and reciprocity
for mat, sx in (("oop", 0.0), ("eps35", 0.15)):
    tagc = f"s{sx:+.2f}"
    c3 = {d["M"]: d for d in load(f"v5_curved_{mat}_c3_t22_p38_{tagc}_M*.json")}
    c5 = {d["M"]: d for d in load(f"v5_curved_{mat}_c5_t22_p38_{tagc}_M*.json")}
    if c3:
        S.setdefault(mat, {})["conical"] = {
            "c3_steps": {M: dist(d["vec"], c3[M - 1]["vec"])
                         for M, d in c3.items() if M - 1 in c3},
            "c5_steps": {M: dist(d["vec"], c5[M - 1]["vec"])
                         for M, d in c5.items() if M - 1 in c5},
            "topologies": (dist(c3[max(c3)]["vec"], c5[max(c5)]["vec"])
                           if c5 else None),
            "clo_c3": {M: d["clo"] for M, d in c3.items()},
            "clo_c5": {M: d["clo"] for M, d in c5.items()}}
rec = {}
for fn in sorted(glob.glob(os.path.join(H, "v5_recip_*.json"))):
    d = json.load(open(fn))
    rec[os.path.basename(fn)[9:-5]] = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("sv")}
        for k, v in d.items() if k not in ("env", "wall")}
S["recip"] = rec
V.dump("v5_summary.json", S)
print(json.dumps(S, indent=1, default=V._js)[:6000])
