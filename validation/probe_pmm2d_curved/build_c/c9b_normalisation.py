"""C9b -- which incident decomposition: the arms, on the film (exact answer)
and on the circle pillar (window independence, round-off floor).

  c9b_normalisation.py <M> film|pillar

Arms (all through the library's stack; the helper
``twod_staggered._stag_incident_coeffs_mapped`` is swapped per arm):
  lstsq   -- the shipped least-squares overlap (cinc = lstsq(Hsup, delta));
  l2      -- the exact L2 modal decomposition, W0^-1 G^-1 b;
  l2norm  -- the L2 decomposition renormalised by the 2 x 2 matrix that
             makes ITS order-0 far field exactly the input (E_x, E_y):
             cinc = C (H0 C)^-1, H0 the two order-0 rows of the projector.
film:   max |R, T - Airy| over every order and both inputs (n_orders 3)
filmobl: the same at OBLIQUE (25 deg, 0) and CONICAL (25 deg, 40 deg)
        incidence on the circle map AND on the identity map of the same
        walls, against the s / p Airy reflectances
pillar: n_orders 2 / 3 / 5 against 8, and the 1e-15 half-space perturbation
Output: c9b_<kind>_M<M>.json
"""
import sys

import _common as C
import numpy as np

TS = C.TS
SP = C.SP
ORIG = TS._stag_incident_coeffs_mapped


def set_arm(name):
    if name == "lstsq":
        SP._stag_incident_coeffs_mapped = lambda *a, **k: None
    elif name == "l2":
        SP._stag_incident_coeffs_mapped = lambda *a, **k: ORIG(*a)
    else:
        SP._stag_incident_coeffs_mapped = ORIG


def airy(n2=2.0):
    k0 = 2 * np.pi / C.WL
    ns = (C.N_SUP, n2, C.N_SUB)
    kz = [complex(n) for n in ns]
    # normal incidence: s = p
    r01 = (kz[0] - kz[1]) / (kz[0] + kz[1])
    r12 = (kz[1] - kz[2]) / (kz[1] + kz[2])
    ph = np.exp(2j * kz[1] * k0 * C.DEPTH)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    return abs(r) ** 2


def film(M):
    cm, _eps = C.circle3()
    eps = np.full((3, 3), 4.0 + 0j)
    Rx = airy()
    res = {"env": C.env_record(), "M": M, "kind": "film"}
    for a in ("lstsq", "l2", "l2norm"):
        set_arm(a)
        o, R, T, J = C.solve(cm, eps, M)
        i0 = C.idx(o, [(0, 0)])[0]
        R = R.copy()
        T = T.copy()
        R[:, i0] -= Rx
        T[:, i0] -= 1.0 - Rx
        res[a] = float(max(np.abs(R).max(), np.abs(T).max()))
    print(res, flush=True)
    C.dump(f"c9b_film_M{M}.json", res)


def pillar(M):
    cm, eps = C.circle3()
    res = {"env": C.env_record(), "M": M, "kind": "pillar"}
    for a in ("lstsq", "l2", "l2norm"):
        set_arm(a)
        vs = {n: C.vec(*C.solve(cm, eps, M, n_orders=n)[:3])
              for n in (2, 3, 5, 8)}
        res[a] = {"orders": max(float(np.max(np.abs(vs[n] - vs[8])))
                                for n in (2, 3, 5)),
                  "vec3": vs[3].tolist()}
    for a in ("lstsq", "l2", "l2norm"):
        res[a]["vs_l2norm"] = float(np.max(np.abs(
            np.asarray(res[a]["vec3"]) - np.asarray(res["l2norm"]["vec3"]))))
    print({a: (res[a]["orders"], res[a]["vs_l2norm"])
           for a in ("lstsq", "l2", "l2norm")}, flush=True)
    C.dump(f"c9b_pillar_M{M}.json", res)


def airy_rows(theta, phi, n2=2.0):
    """Exact reflectance of the uniform n2 film for incident lab E_x / E_y
    (s / p split of the incident transverse field) -- Phase B's
    ``_airy_rows`` of tests/unit/test_pmm2d_staggered_curved_b.py."""
    k0 = 2 * np.pi / C.WL
    ns = (C.N_SUP, n2, C.N_SUB)
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
    out = []
    for et in ((1.0, 0.0), (0.0, 1.0)):
        a = -np.sin(phi) * et[0] + np.cos(phi) * et[1]
        b = (np.cos(phi) * et[0] + np.sin(phi) * et[1]) / np.cos(theta)
        out.append((a * a * rs + b * b * rp) / (a * a + b * b))
    return np.array(out)


def filmobl(M):
    cm, _eps = C.circle3()
    idm = C.CM.TransfiniteMap(cm.u_walls, cm.v_walls)
    eps = np.full((3, 3), 4.0 + 0j)
    res = {"env": C.env_record(), "M": M, "kind": "filmobl"}
    for th, ph in ((25.0, 0.0), (25.0, 40.0)):
        Rx = airy_rows(np.deg2rad(th), np.deg2rad(ph))
        for mname, m in (("circle", cm), ("identity", idm)):
            for a in ("lstsq", "l2norm"):
                set_arm(a)
                o, R, T, J = C.solve(m, eps, M, theta=np.deg2rad(th),
                                     phi=np.deg2rad(ph))
                i0 = C.idx(o, [(0, 0)])[0]
                R = R.copy()
                T = T.copy()
                R[:, i0] -= Rx
                T[:, i0] -= 1.0 - Rx
                res[f"{mname}_t{th}_p{ph}_{a}"] = float(max(
                    np.abs(R).max(), np.abs(T).max()))
    print(res, flush=True)
    C.dump(f"c9b_filmobl_M{M}.json", res)


if __name__ == "__main__":
    M = int(sys.argv[1])
    {"film": film, "pillar": pillar, "filmobl": filmobl}[sys.argv[2]](M)
