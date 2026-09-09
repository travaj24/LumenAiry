"""Probe 2 (G2) -- a UNIFORM ISOTROPIC MAGNETIC slab against the ANALYTIC
Airy / characteristic-matrix formula with the wave impedance.

The oracle is written here, not imported: for a uniform slab of relative
permittivity ``eps`` and permeability ``mu`` between vacuum half-spaces,

    kz = sqrt(eps*mu - sin^2 th0)        (Im >= 0, the decaying branch)
    Y_TE = kz / mu ,   Y_TM = eps / kz   (relative admittances; vacuum:
                                          Y0_TE = cos th0, Y0_TM = 1/cos th0)
    M    = [[cos d, -(i/Y) sin d], [-i Y sin d, cos d]] ,  d = k0 * t * kz
    [B; C] = M [1; Ys] ,  r = (Y0 B - C)/(Y0 B + C) ,  t = 2 Y0/(Y0 B + C)
    R = |r|^2 ,  T = Re(Ys)/Re(Y0) |t|^2

Abeles / Born & Wolf, written for the module's PUBLIC ``exp(-i w t)``: the
textbook layer matrix carries ``+i`` and belongs to ``exp(+i w t)``, where
``Im(eps) > 0`` is GAIN -- MEASURED here, the ``+i`` form overshot a lossy slab
by ``dT = 4.4`` (it amplified) while the ``-i`` form lands at 1e-15.  The two
forms are IDENTICAL for real eps and mu, so only a LOSSY arm can pin that
bridge, which is why one is included.

The formula's convention is pinned on the NONMAGNETIC arm FIRST (``mu = 1``,
where the shipped scalar solver is already validated), and only then used as
the oracle for ``mu != 1``: the magnetic claim then rests on a formula this
build has itself verified, R and T and the complex Jones alike.
"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_mag"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

P, WL, DEP = 0.70e-6, 0.55e-6, 0.28e-6


def airy(eps, mu, th0, t_over_wl, pol):
    """(R, T, r) of a uniform slab between vacuum half-spaces."""
    eps, mu = complex(eps), complex(mu)
    s2 = np.sin(th0) ** 2
    kz = np.sqrt(complex(eps * mu - s2))
    if kz.imag < 0:
        kz = -kz
    c0 = np.cos(th0)
    if pol == "te":
        Y, Y0 = kz / mu, complex(c0)
    else:
        Y, Y0 = eps / kz, complex(1.0 / c0)
    d = 2.0 * np.pi * t_over_wl * kz
    M = np.array([[np.cos(d), -1j * np.sin(d) / Y],
                  [-1j * Y * np.sin(d), np.cos(d)]], dtype=complex)
    B, C = M @ np.array([1.0, Y0], dtype=complex)
    r = (Y0 * B - C) / (Y0 * B + C)
    tr = 2.0 * Y0 / (Y0 * B + C)
    return abs(r) ** 2, (Y0.real / Y0.real) * abs(tr) ** 2, r


def engine(eps, mu, th0, M, mu_none=False):
    cell = np.full((2, 2), complex(eps))
    kw = {} if mu_none else dict(mu_cell=np.full((2, 2), complex(mu)))
    o, R, T, J = pmm_jones_2d_staggered(
        P, P, cell, 1.0, 1.0, DEP, WL, degree=M, n_orders=2, theta=th0,
        phi=0.0, **kw)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    # phi = 0: incident Ey = TE (s), incident Ex = TM (p, tangential ratio)
    return dict(R_tm=float(R[0, p0]), T_tm=float(T[0, p0]),
                R_te=float(R[1, p0]), T_te=float(T[1, p0]),
                j_tm=complex(J[0, 0]), j_te=complex(J[1, 1]),
                Rtot=float(R.sum(axis=1).max()), Ttot=float(T.sum(axis=1).max()))


def row(tag, eps, mu, th0, M, mu_none=False):
    e = engine(eps, mu, th0, M, mu_none)
    out = {}
    for pol in ("te", "tm"):
        Ra, Ta, ra = airy(eps, mu, th0, DEP / WL, pol)
        out["dR_" + pol] = abs(e["R_" + pol] - Ra)
        out["dT_" + pol] = abs(e["T_" + pol] - Ta)
        out["|r|_" + pol] = abs(ra)
        out["dJp_" + pol] = abs(e["j_" + pol] - ra)
        out["dJm_" + pol] = abs(e["j_" + pol] + ra)
    print(f"{tag:<42s} M={M} "
          f"dR_te={out['dR_te']:.3e} dT_te={out['dT_te']:.3e} "
          f"dR_tm={out['dR_tm']:.3e} dT_tm={out['dT_tm']:.3e} "
          f"dJ_te(+/-)={out['dJp_te']:.2e}/{out['dJm_te']:.2e} "
          f"dJ_tm(+/-)={out['dJp_tm']:.2e}/{out['dJm_tm']:.2e} "
          f"R+T={e['Rtot'] + e['Ttot']:.12f}")
    return out


if __name__ == "__main__":
    print("=== A. CONVENTION PIN: the formula vs the SHIPPED nonmagnetic path")
    for th in (0.0, 0.35):
        row(f"eps=4  mu=1  (mu_cell OMITTED) th={th}", 4.0, 1.0, th, 8, True)
        row(f"eps=4  mu=1  (forced magnetic) th={th}", 4.0, 1.0, th, 8)
    print()
    print("=== B. MAGNETIC: lossless, M ladder")
    for eps, mu in ((4.0, 2.0), (2.25, 3.0), (1.0, 4.0)):
        for th in (0.0, 0.35):
            for M in (5, 7, 8):
                row(f"eps={eps} mu={mu} th={th}", eps, mu, th, M)
    print()
    print("=== C. MAGNETIC: LOSSY mu (Im mu > 0 = loss in the public gauge)")
    for eps, mu in ((4.0, 2.0 + 0.3j), (4.0 + 0.2j, 2.0)):
        for th in (0.0, 0.35):
            for M in (5, 7, 8):
                row(f"eps={eps} mu={mu} th={th}", eps, mu, th, M)
