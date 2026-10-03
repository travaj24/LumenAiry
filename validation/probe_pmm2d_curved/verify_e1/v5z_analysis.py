"""The exact-disk RCWA z-staircase of the slanted eps-3.5 disk, analysed for
legitimacy before it is extrapolated: local rates in N (harmonics per axis
2n + 1) at fixed N_z on >= 4 points, local rates in N_z at fixed n on >= 4
points, then the double extrapolation (Richardson in 1 / N on the last pair
at each N_z, then in 1 / N_z^2) and an independent separable fit
v(N, N_z) = v_inf + a / N + b / N_z^2 by least squares over every point.
Distances are the 36-vector max-abs to the curved composite answer (c5 top
rung).  Output v5z_analysis.json."""
import glob
import json
import os

import _ve1common as V
import numpy as np

H = V.HERE
Z = {}
for fn in glob.glob(os.path.join(H, "v5_zstair_s+0.15_Nz*_n*.json")):
    d = json.load(open(fn))
    Z[(d["Nz"], d["n"])] = np.asarray(d["vec"])
c5 = {}
for fn in glob.glob(os.path.join(H, "v5_curved_eps35_c5_t0_p0_s+0.15_M*.json")):
    d = json.load(open(fn))
    c5[d["M"]] = np.asarray(d["vec"])
c3 = {}
for fn in glob.glob(os.path.join(H, "v5_curved_eps35_c3_t0_p0_s+0.15_M*.json")):
    d = json.load(open(fn))
    c3[d["M"]] = np.asarray(d["vec"])
ref = c5[max(c5)]


def dist(a, b):
    return float(np.abs(a - b).max())


out = {"ref": f"c5 M{max(c5)}", "raw": {f"{a}_{b}": dist(v, ref)
                                         for (a, b), v in sorted(Z.items())}}
Nzs = sorted({k[0] for k in Z})
ns = sorted({k[1] for k in Z})
# self-convergence rates in N at fixed N_z (needs >= 3 consecutive points)
rN = {}
for a in Nzs:
    have = [n for n in ns if (a, n) in Z]
    r = []
    for x, y, z in zip(have, have[1:], have[2:]):
        d1, d2 = dist(Z[(a, x)], Z[(a, y)]), dist(Z[(a, y)], Z[(a, z)])
        Nx, Nyy, Nzz = 2 * x + 1, 2 * y + 1, 2 * z + 1
        # for e ~ C / N: |v_x - v_y| = C (1/Nx - 1/Ny); the rate p of
        # d ~ N^-p estimated from the two differences
        r.append(float(np.log(d1 / d2) / np.log(((Nyy + Nzz) / 2)
                                                 / ((Nx + Nyy) / 2))))
    rN[a] = {"points": len(have), "rates": r}
out["rates_in_N"] = rN
rZ = {}
for n in ns:
    have = [a for a in Nzs if (a, n) in Z]
    r = []
    for x, y, z in zip(have, have[1:], have[2:]):
        d1, d2 = dist(Z[(x, n)], Z[(y, n)]), dist(Z[(y, n)], Z[(z, n)])
        r.append(float(np.log(d1 / d2) / np.log(y / x)))
    rZ[n] = {"points": len(have), "rates": r}
out["rates_in_Nz"] = rZ
# double extrapolation: Richardson 1/N on the LAST available pair per N_z
# that every N_z has (common pair), then 1/N_z^2 on consecutive N_z
common = [n for n in ns if all((a, n) in Z for a in Nzs)]
dbl = {}
if len(common) >= 2:
    n1, n2 = common[-2], common[-1]
    N1, N2 = 2 * n1 + 1, 2 * n2 + 1
    rich = {a: (N2 * Z[(a, n2)] - N1 * Z[(a, n1)]) / (N2 - N1) for a in Nzs}
    out["richN_pair"] = [n1, n2]
    out["richN"] = {a: dist(v, ref) for a, v in rich.items()}
    for x, y in zip(Nzs, Nzs[1:]):
        ext = (y * y * rich[y] - x * x * rich[x]) / (y * y - x * x)
        dbl[f"{x}-{y}"] = dist(ext, ref)
out["double_extrapolated"] = dbl
# separable least-squares fit over every point
keys = sorted(Z)
A = np.array([[1.0, 1.0 / (2 * n + 1), 1.0 / (a * a)] for a, n in keys])
Y = np.array([Z[k] for k in keys])
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
vinf = coef[0]
res = Y - A @ coef
out["fit"] = {"vinf_to_ref": dist(vinf, ref),
              "max_residual": float(np.abs(res).max()),
              "n_points": len(keys),
              "vinf_to_c3top": dist(vinf, c3[max(c3)])}
# the curved answer's own uncertainty
out["curved"] = {"c5_vs_c3top": dist(ref, c3[max(c3)]),
                 "c5_last_step": dist(c5[max(c5)], c5[max(c5) - 1])}
V.dump("v5z_analysis.json", out)
print(json.dumps(out, indent=1, default=V._js))
