"""V0 -- validate the verifier's own (eps, mu) slab oracle before use.

(a) mu = I: against the shipped ``berreman_jones_1d`` on the verifier's
    out-of-plane tensors (general director, its non-reciprocal twin, the
    lossy gyrotropic rotated tensor) at the three mounts: R, T, Jr, Jt.
(b) an ISOTROPIC (eps, mu) slab, lossless and lossy, against the Airy /
    characteristic-matrix formula (tilted admittances eta_TE = kz / mu,
    eta_TM = eps / kz), s and p, at 0 / 30 / 55 deg.
(c) against the BUILDER's oracle (``build_e1/_mu_oracle.py``) with a
    material mu (gyro, lossy gyro, anisotropic) on the out-of-plane tensors:
    two independent formulations (expm vs eigen + z-referenced system).
Output v0_oracle.json.
"""
import importlib.util
import os

import _ve1common as V
import numpy as np

out = {"a": {}, "b": {}, "c": {}}
f = V.SLAB
for tn in ("dirgen", "nrgen", "gyrol"):
    t = V.TENSORS[tn]
    for mn, (th, ph) in V.MOUNTS.items():
        Rb, Tb, jr, jt = V.berreman_jones_1d([(t, f["DEP"])], f["NSUB"],
                                             f["NSUP"], f["WL"], angle=th,
                                             phi=ph)
        R, T, Jr, Jt = V.eps_mu_slab(t, None, th, ph)
        out["a"][f"{tn}_{mn}"] = [float(np.abs(R - Rb).max()),
                                  float(np.abs(T - Tb).max()),
                                  float(np.abs(Jr - jr).max()),
                                  float(np.abs(Jt - jt).max())]


def airy(eps, mu, th, pol):
    k0d = 2 * np.pi / f["WL"] * f["DEP"]
    K = f["NSUP"] * np.sin(th)

    def kz(e, m):
        v = np.sqrt(complex(e * m - K * K))
        return v if v.imag >= 0 else -v

    def eta(e, m):
        k = kz(e, m)
        return k / m if pol == "s" else e / k
    e0, es = f["NSUP"] ** 2, f["NSUB"] ** 2
    n0, ns = eta(e0, 1.0), eta(es, 1.0)
    nl = eta(eps, mu)
    d = kz(eps, mu) * k0d
    Mc = np.array([[np.cos(d), -1j * np.sin(d) / nl],
                   [-1j * nl * np.sin(d), np.cos(d)]])
    B, C = Mc @ np.array([1.0, ns])
    r = (n0 * B - C) / (n0 * B + C)
    T = 4 * n0.real * ns.real / abs(n0 * B + C) ** 2
    return abs(r) ** 2, T


for tag, e, m in (("lossless", 2.6, 1.7), ("lossy", 2.6 + 0.3j,
                                            1.7 + 0.15j)):
    for thd in (0.0, 30.0, 55.0):
        th = np.deg2rad(thd)
        R, T, _jr, _jt = V.eps_mu_slab(e * V.EYE, m * V.EYE, th, 0.0)
        Rp, Tp = airy(e, m, th, "p")
        Rs, Ts = airy(e, m, th, "s")
        out["b"][f"{tag}_{thd:g}"] = [float(abs(R[0] - Rp)),
                                      float(abs(T[0] - Tp)),
                                      float(abs(R[1] - Rs)),
                                      float(abs(T[1] - Ts))]

# (c) the builder's oracle
p = os.path.join(V.HERE, "..", "build_e1", "_mu_oracle.py")
spec = importlib.util.spec_from_file_location("_mu_oracle_b", p)
B = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(B)
    for tn in ("dirgen", "nrgen", "gyrol"):
        for mn_, m in (("gyro", V.MU_GYRO), ("gyrol", V.MU_GYRO_LOSSY),
                       ("aniso", V.MU_ANISO)):
            for mo, (th, ph) in V.MOUNTS.items():
                a = V.eps_mu_slab(V.TENSORS[tn], m, th, ph)
                b = B.berreman_mu(V.TENSORS[tn], m, f["DEP"], f["NSUB"],
                                  f["NSUP"], f["WL"], theta=th, phi=ph)
                out["c"][f"{tn}_{mn_}_{mo}"] = [
                    float(np.abs(np.asarray(x) - np.asarray(y)).max())
                    for x, y in zip(a, b)]
except Exception as exc:     # pragma: no cover
    out["c"]["error"] = repr(exc)
V.dump("v0_oracle.json", out)
for k in ("a", "b", "c"):
    print(k, max(max(v) for v in out[k].values()))
