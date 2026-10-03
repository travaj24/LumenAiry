"""A5 -- the eps-free GEOMETRIC SPLIT survives a map: -R == [eps'_t] / eps
for a uniform isotropic region (plan 2.4).  Fail-before: a uniform MAGNETIC
region (no map; the shipped magnetic route), where -R carries chi_t and the
split is genuinely broken.

  python validation/probe_pmm2d_curved/build_a/a5_split.py

Output: a5_split.json.  rel = max|[eps_t]/eps - (-R)| / max|R| with
[eps_t] the four ASSEMBLED blocks of the region.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402

TS = C.TS
K0 = 2 * np.pi / C.WL


def split_rel(s, eps):
    qq = s.q * s.q
    E = np.zeros_like(s.Rmat)
    E[:qq, :qq], E[qq:, qq:] = s.Et_blocks
    if s.Et_offdiag is not None:
        E[:qq, qq:], E[qq:, :qq] = s.Et_offdiag
    return float(np.max(np.abs(E / eps + s.Rmat)) / np.max(np.abs(s.Rmat)))


def main():
    res = {"env": C.env_record(), "mapped": [], "fail_before": []}
    for a in (0.0, 0.05, 0.15):
        for eps in (1.0 + 0j, 2.1025 + 0j, 4.0 + 0.5j):
            for M in (4, 6):
                cm = (C.IdentityMap(C.XW, C.YW["stripe"]) if a == 0
                      else C.stretch_map(a))
                s = TS.Granet2DTransverseE(
                    C.P, C.P, cm.u_walls, cm.v_walls, M,
                    np.full((3, 3), eps), k0=K0, cmap=cm)
                row = {"a": a, "eps": [eps.real, eps.imag], "M": M,
                       "rel": split_rel(s, eps)}
                res["mapped"].append(row)
                print(json.dumps(row), flush=True)
    EYE = np.eye(3, dtype=complex)
    for label, mu in (("scalar_mu_2", np.full((2, 2), 2.0 + 0j)),
                      ("uniaxial_mu_diag_2_1_1",
                       np.broadcast_to(np.diag([2.0, 1.0, 1.0]).astype(complex),
                                       (2, 2, 3, 3)).copy()),
                      ("mu_1_control", np.full((2, 2), 1.0 + 0j))):
        for M in (4, 6):
            s = TS.Granet2DTransverseE(C.P, C.P, 2, 2, M,
                                       np.full((2, 2), 2.1025 + 0j), k0=K0,
                                       mu_cell=mu)
            row = {"case": label, "M": M, "rel": split_rel(s, 2.1025)}
            res["fail_before"].append(row)
            print(json.dumps(row), flush=True)
    del EYE
    res["mapped_max"] = max(r["rel"] for r in res["mapped"])
    with open(os.path.join(C.HERE, "a5_split.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("mapped max", res["mapped_max"])


if __name__ == "__main__":
    main()
