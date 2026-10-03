"""E1-3m -- an OUT-OF-PLANE eps WITH a material mu (the wall the magnetic
build left standing: "an OOP x mu layer refuses"): a uniform (eps, mu) slab
against the independent (eps, mu) Berreman 4x4 oracle of ``_mu_oracle.py``,
unmapped and under maps (where the material chi_t composes with the map's).

The oracle is validated first (``oracle``): with mu = I against the shipped
``berreman_jones_1d`` (R, T, both Jones; the out-of-plane tensors at normal,
oblique and conical incidence), and on an isotropic (eps, mu) slab against
the analytic Airy formula (TE and TM at 25 deg).

Permeabilities: aniso (real symmetric block-form), gyro (Hermitian,
non-reciprocal: m12 = -m21 = 0.3i), lossy (non-Hermitian -> the QZ branch of
_region_modes_oop).

usage: python e3m_mu.py oracle
       python e3m_mu.py ladder <map> <tensor> <mu> <angle> [M,..]
writes e3m_oracle.json / e3m_<map>_<tensor>_<mu>_<angle>.json
"""
import sys
import time

import _e1common as E
import _mu_oracle as MO
import numpy as np

from lumenairy.elements.berreman import berreman_jones_1d

MUS = {"aniso": np.array([[1.5, 0.2, 0], [0.2, 1.3, 0], [0, 0, 1.2]],
                         complex),
       "gyro": np.array([[1.6, 0.3j, 0], [-0.3j, 1.6, 0], [0, 0, 1.2]],
                        complex),
       "lossy": np.array([[1.5 + 0.08j, 0.2, 0], [0.2, 1.3 + 0.05j, 0],
                          [0, 0, 1.2 + 0.03j]], complex)}


def airy(eps, mu, d, nsub, wl, th):
    """Analytic isotropic (eps, mu) slab in air: (R_TM, T_TM), (R_TE, T_TE)
    (PUBLIC exp(-i w t): the layer matrix with -i)."""
    k0, s = 2 * np.pi / wl, np.sin(th)
    out = []
    for pol in ("TM", "TE"):
        kz1 = np.sqrt(1 - s ** 2 + 0j)
        kz = np.sqrt(eps * mu - s ** 2 + 0j)
        kz3 = np.sqrt(nsub ** 2 - s ** 2 + 0j)
        Y1, Y, Y3 = ((1 / kz1, eps / kz, nsub ** 2 / kz3) if pol == "TM"
                     else (kz1, kz / mu, kz3))
        dd = k0 * d * kz
        Mm = np.array([[np.cos(dd), -1j * np.sin(dd) / Y],
                       [-1j * Y * np.sin(dd), np.cos(dd)]])
        B, Cc = Mm @ np.array([1.0, Y3])
        r = (Y1 * B - Cc) / (Y1 * B + Cc)
        out.append((abs(r) ** 2,
                    float(4 * np.real(Y1) * np.real(Y3)
                          / abs(Y1 * B + Cc) ** 2)))
    return out


def oracle():
    f = E.SLAB
    out = {}
    for tn in ("oop", "nonrec", "lnonrec"):
        for an, (th, ph) in E.ANG.items():
            Rb, Tb, jr, jt = berreman_jones_1d(
                [(E.TENSORS[tn], f["DEP"])], f["NSUB"], f["NSUP"], f["WL"],
                angle=np.deg2rad(th), phi=np.deg2rad(ph))
            R, T, Jr, Jt = MO.berreman_mu(E.TENSORS[tn], np.eye(3), f["DEP"],
                                          f["NSUB"], f["NSUP"], f["WL"],
                                          np.deg2rad(th), np.deg2rad(ph))
            out[f"{tn}_{an}"] = {
                "dRT": float(max(np.abs(R - Rb).max(), np.abs(T - Tb).max())),
                "dJr": float(np.abs(Jr - jr).max()),
                "dJt": float(np.abs(Jt - jt).max())}
            print(tn, an, out[f"{tn}_{an}"], flush=True)
    th = np.deg2rad(25.0)
    for eps, mu in ((2.0, 1.8), (2.0 + 0.1j, 1.8 + 0.05j)):
        R, T, _Jr, _Jt = MO.berreman_mu(eps * np.eye(3), mu * np.eye(3),
                                        f["DEP"], f["NSUB"], f["NSUP"],
                                        f["WL"], th, 0.0)
        (Rtm, Ttm), (Rte, Tte) = airy(eps, mu, f["DEP"], f["NSUB"], f["WL"],
                                      th)
        key = f"airy_eps{eps}_mu{mu}"
        out[key] = {"dR": float(max(abs(R[0] - Rtm), abs(R[1] - Rte))),
                    "dT": float(max(abs(T[0] - Ttm), abs(T[1] - Tte)))}
        print(key, out[key], flush=True)
    E.dump("e3m_oracle.json", out)


def ladder(mapname, tname, muname, ang, Ms):
    f = E.SLAB
    cm = E.make_map(mapname, f["P"])
    th, ph = E.ANG[ang]
    t33, mu = E.TENSORS[tname], MUS[muname]
    R0, T0, jr, jt = MO.berreman_mu(t33, mu, f["DEP"], f["NSUB"], f["NSUP"],
                                    f["WL"], np.deg2rad(th), np.deg2rad(ph))
    rows = []
    for M in Ms:
        t0 = time.perf_counter()
        st = E.PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                              n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                              cmap=cm)
        st.add_layer(f["DEP"], eps=t33, mu=mu)
        st.set_source(f["WL"], theta=np.deg2rad(th), phi=np.deg2rad(ph))
        _o, R, T, J = st.solve(jones=True)
        R, T = np.asarray(R), np.asarray(T)
        r = {"M": M,
             "dRT": float(max(np.abs(R.sum(1) - R0).max(),
                              np.abs(T.sum(1) - T0).max())),
             "dJr": float(np.abs(np.asarray(J) - jr).max()),
             "dJt": float(np.abs(E.jones_t(st) - jt).max()),
             "closure": float(np.abs(R.sum(1) + T.sum(1) - 1).max()),
             "t": time.perf_counter() - t0}
        rows.append(r)
        print(mapname, tname, muname, ang,
              {k: (f"{v:.2e}" if isinstance(v, float) else v)
               for k, v in r.items()}, flush=True)
    E.dump(f"e3m_{mapname}_{tname}_{muname}_{ang}.json",
           {"rows": rows, "map": mapname, "tensor": tname, "mu": muname,
            "theta": th, "phi": ph})


if __name__ == "__main__":
    if sys.argv[1] == "oracle":
        oracle()
    elif sys.argv[1] == "ladder":
        Ms = ([int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6
              else [4, 5, 6, 7, 8])
        ladder(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], Ms)
    else:
        raise SystemExit(__doc__)
