"""V0 -- the verifier's own eps+mu Berreman against the shipped
berreman_jones_1d on eps-only stacks (power AND complex reflection Jones),
normal / oblique / conical, every tensor class used later; then the
verifier's oracle with mu against the closed-form (eps, mu) Airy slab.
Output v0_oracle.json."""
import numpy as np
from _vdcommon import (
    G3,
    GYRO_V,
    LOSSY,
    RASYM,
    berreman_eps_mu,
    berreman_jones_1d,
    biaxial,
    dump,
    lc,
)

rows = {}
worst = 0.0
for tn, t in {"lc0": lc(0), "lc30": lc(30), "lc45": lc(45), "lc90": lc(90),
              "biax": biaxial(), "gyro": GYRO_V, "rasym": RASYM,
              "lossy": LOSSY}.items():
    for th, ph in ((0.0, 0.0), (0.4, 0.0), (0.35, 0.7), (0.5, -1.2)):
        Rb, Tb, jr, _ = berreman_jones_1d([(t, G3["DEP"])], G3["NSUB"],
                                          G3["NSUP"], G3["WL"], angle=th,
                                          phi=ph)
        R, T, r = berreman_eps_mu([(t, 1.0, G3["DEP"])], G3["NSUB"],
                                  G3["NSUP"], G3["WL"], th, ph)
        d = max(np.abs(R - Rb).max(), np.abs(T - Tb).max(),
                np.abs(r - jr).max())
        worst = max(worst, d)
        rows[f"{tn}_{th}_{ph}"] = float(d)
# isotropic (eps, mu) slab, normal incidence, vs Airy (verifier's own algebra)
eps, mu, dd = 2.0, 1.8, 0.5
n2 = np.sqrt(eps * mu)
k0 = 2 * np.pi
a1, a2, a3 = 1.0, n2 / mu, 1.45
r12, r23 = (a1 - a2) / (a1 + a2), (a2 - a3) / (a2 + a3)
t12, t23 = 2 * a1 / (a1 + a2), 2 * a2 / (a2 + a3)
ph = np.exp(1j * n2 * k0 * dd)
den = 1 + r12 * r23 * ph ** 2
Ra = abs((r12 + r23 * ph ** 2) / den) ** 2
Ta = abs(t12 * t23 * ph / den) ** 2 * a3 / a1
R, T, _r = berreman_eps_mu([(eps, mu, dd)], 1.45, 1.0, 1.0)
airy = float(max(abs(R[0] - Ra), abs(T[0] - Ta)))
dump("v0_oracle.json", {"eps_only_vs_shipped": rows, "worst": worst,
                        "airy_mu": airy})
print("worst eps-only vs shipped", worst, "airy mu", airy)
