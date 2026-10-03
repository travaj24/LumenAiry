"""A3 -- a UNIFORM film under a stretch is exact (oracle: the Airy slab).

The film is passed as a constant PATTERNED cell, so it takes its OWN mapped
region eig (plain-Gram H partner) while the half-spaces ride the mapped
geometric eig -- the realistic mix of the two helpers.  Arms:

  correct      : the library as built;
  no_cofactor  : the far projector handed the no-cofactor view of the map
                 (_common.no_cofactor) -- the fail-before;
  oblique      : theta = 25 deg (phi = 0) and conical theta = 25 deg,
                 phi = 40 deg, against the s / p Airy reflectance (the
                 incident lab-basis E_x / E_y split into s and p power);
                 NOT probed by the planning campaign.

  python validation/probe_pmm2d_curved/build_a/a3_film.py

Output: a3_film.json.  err = max over both incident polarizations and all
orders of |R - R_exact|, |T - T_exact| (normal incidence: per order, the
specular order carrying the Airy values and every other order zero; oblique:
on the order sums, T_exact = 1 - R_exact for the lossless film).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402

N2 = 2.0


def airy_sp(theta):
    """|r_s|^2, |r_p|^2 of the n = 2 film between n = 1 and n = 1.45."""
    k0 = 2 * np.pi / C.WL
    ns = (C.N_SUP, N2, C.N_SUB)
    st = C.N_SUP * np.sin(theta)
    kz = [np.sqrt(complex(n * n - st * st)) for n in ns]

    def slab(r12, r23):
        ph = np.exp(2j * kz[1] * k0 * C.DEPTH)
        return abs((r12 + r23 * ph) / (1 + r12 * r23 * ph)) ** 2

    rs = slab((kz[0] - kz[1]) / (kz[0] + kz[1]),
              (kz[1] - kz[2]) / (kz[1] + kz[2]))
    e = [n * n for n in ns]
    rp = slab((e[1] * kz[0] - e[0] * kz[1]) / (e[1] * kz[0] + e[0] * kz[1]),
              (e[2] * kz[1] - e[1] * kz[2]) / (e[2] * kz[1] + e[1] * kz[2]))
    return rs, rp


def r_exact_rows(theta, phi):
    """Exact total reflectance for incident lab E_x (row 0) and E_y (row 1):
    E_t = (1, 0) / (0, 1) = a s_t + b cos(theta) k_t with s_t = (-sin phi,
    cos phi), k_t = (cos phi, sin phi)."""
    rs, rp = airy_sp(theta)
    out = []
    for et in ((1.0, 0.0), (0.0, 1.0)):
        a = -np.sin(phi) * et[0] + np.cos(phi) * et[1]
        b = (np.cos(phi) * et[0] + np.sin(phi) * et[1]) / np.cos(theta)
        out.append((a * a * rs + b * b * rp) / (a * a + b * b))
    return np.array(out)


def main():
    res = {"env": C.env_record(), "airy_normal": C.airy(N2), "rows": []}
    Rex, Tex = C.airy(N2)
    for a in (0.05, 0.15):
        for M in (4, 5, 6, 7):
            t0 = time.perf_counter()
            o, R, T, _J, _st = C.solve("film", a, M)
            i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            R = R.copy()
            T = T.copy()
            R[:, i0] -= Rex
            T[:, i0] -= Tex
            row = {"a": a, "M": M, "arm": "correct",
                   "err_tm_Ex": float(max(np.abs(R[0]).max(),
                                          np.abs(T[0]).max())),
                   "err_te_Ey": float(max(np.abs(R[1]).max(),
                                          np.abs(T[1]).max())),
                   "t": time.perf_counter() - t0}
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
    for a, M in ((0.05, 6), (0.15, 6)):
        with C.no_cofactor():
            o, R, T, _J, _st = C.solve("film", a, M)
        i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        R = R.copy()
        T = T.copy()
        R[:, i0] -= Rex
        T[:, i0] -= Tex
        row = {"a": a, "M": M, "arm": "no_cofactor",
               "err": float(max(np.abs(R).max(), np.abs(T).max()))}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    th = np.deg2rad(25.0)
    for phi_deg in (0.0, 40.0):
        ph = np.deg2rad(phi_deg)
        Rx = r_exact_rows(th, ph)
        for a in (0.0, 0.05, 0.15):
            for M in (4, 5, 6, 7):
                o, R, T, _J, _st = C.solve("film", a, M, theta=th, phi=ph)
                Rs = R.sum(axis=1)
                Ts = T.sum(axis=1)
                row = {"a": a, "M": M, "arm": "oblique", "theta_deg": 25.0,
                       "phi_deg": phi_deg,
                       "err": float(max(np.max(np.abs(Rs - Rx)),
                                        np.max(np.abs(Ts - (1 - Rx)))))}
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
    with open(os.path.join(C.HERE, "a3_film.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
