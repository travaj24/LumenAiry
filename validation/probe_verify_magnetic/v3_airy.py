"""V3 -- G2: a UNIFORM MAGNETIC slab against an ANALYTIC oracle written here.

The oracle is a 2 x 2 transfer matrix derived from scratch in the LIBRARY's
public ``exp(-i w t)`` convention (``Im(eps) > 0`` = loss), NOT copied from a
textbook:

    curl E = +i w mu0 mu H,  curl H = -i w eps0 eps E          (exp(-i w t))

TE (E = E_y y^):   H~_x = (kz / mu) E_y  for a FORWARD wave  ->  Y_TE = kz/mu
TM (H = H_y y^):   H_y  = (eps / kz) E~_x                     ->  Y_TM = eps/kz
with ``kz = sqrt(eps mu - sin^2 th0)`` on the ``Im(kz) >= 0`` branch (a forward
wave ``exp(+i k0 kz z)`` then DECAYS, which is what ``exp(-i w t)`` loss means).

Writing ``U`` for the tangential E and ``V`` for the tangential H (normalised
by ``sqrt(eps0/mu0)``), the FORWARD transfer over a thickness ``t`` is

    [U(t); V(t)] = [[cos d,   i sin d / Y],
                    [i Y sin d,   cos d ]] [U(0); V(0)],   d = k0 t kz

-- the ``+i`` form here is the z-FORWARD transfer; the ``-i`` textbook matrix is
its inverse (the t -> 0 characteristic matrix).  The two coincide for real
``eps``/``mu``, so only a LOSSY arm pins the convention: run
``python v3_airy.py convention`` to see the wrong branch read dT ~ 4 on a lossy
slab (it amplifies).

Matching ``U(0) = 1 + r``, ``V(0) = Y_sup (1 - r)``, ``U(t) = tau``,
``V(t) = Y_sub tau`` gives r and tau in closed form (see :func:`slab_rt`).
"""
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm.twod_staggered import pmm_jones_2d_staggered

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__


def _kz(eps, mu, sin2):
    v = np.sqrt(complex(eps) * complex(mu) - sin2)
    return v if v.imag >= 0 else -v


def slab_rt(eps, mu, thickness, wl, theta, pol, eps_sup=1.0, eps_sub=1.0,
            forward=True):
    """Analytic (r, tau, R, T) of a uniform slab.  ``forward=False`` uses the
    OTHER sign branch (the exp(+i w t) transfer), kept so the convention can be
    measured rather than asserted."""
    k0 = 2.0 * np.pi / wl
    sin2 = complex(eps_sup) * np.sin(theta) ** 2
    kzs, kzl, kzb = (_kz(eps_sup, 1.0, sin2), _kz(eps, mu, sin2),
                     _kz(eps_sub, 1.0, sin2))
    if pol == "te":
        ysup, ylay, ysub = kzs / 1.0, kzl / complex(mu), kzb / 1.0
    else:
        ysup, ylay, ysub = (complex(eps_sup) / kzs, complex(eps) / kzl,
                            complex(eps_sub) / kzb)
    d = k0 * thickness * kzl
    sgn = 1.0 if forward else -1.0
    cd, sd = np.cos(d), np.sin(d)
    # (1+r) * p1 = (1-r) * p2  with  U(t) = tau, V(t) = ysub tau
    p1 = ysub * cd - sgn * 1j * ylay * sd
    p2 = ysup * (cd - sgn * 1j * (ysub / ylay) * sd)
    r = (p2 - p1) / (p2 + p1)
    tau = cd * (1.0 + r) + sgn * 1j * sd * ysup * (1.0 - r) / ylay
    rr = float(abs(r) ** 2)
    tt = float(abs(tau) ** 2 * (ysub.real / ysup.real))
    return r, tau, rr, tt


def _library(eps, mu, thickness, wl, theta, m, n_orders=3, px=0.90e-6,
             magnetic_path=True):
    """The specular (0,0) R, T and Jones of the same slab, through the engine."""
    cell = np.full((2, 2), complex(eps))
    kw = {}
    if magnetic_path:
        mc = np.zeros((2, 2, 3, 3), dtype=complex)
        mc[...] = np.eye(3) * complex(mu)
        kw["mu_cell"] = mc
    o, r, t, j = pmm_jones_2d_staggered(px, px, cell, 1.0, 1.0, thickness, wl,
                                        n_modes=m, n_orders=n_orders,
                                        theta=theta, phi=0.0, **kw)
    o = np.asarray(o)
    i0 = int(np.flatnonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0])
    return r[:, i0], t[:, i0], j


def airy_residual(eps, mu, thickness, wl, theta, m, magnetic_path=True,
                  forward=True, want=None):
    """max relative residual over {R, T, Jones} x {TE, TM}.

    ROW/COLUMN convention MEASURED on the NONMAGNETIC arm, then reused: at
    ``phi = 0`` the incident ``Ey`` drive is TE and the incident ``Ex`` drive is
    TM, and the order-0 reflection Jones is DIAGONAL with ``J[1,1] = r_TE`` and
    ``J[0,0] = r_TM`` -- no extra sign, because :func:`slab_rt` solves for the
    TANGENTIAL amplitude ratio directly (``U(0) = 1 + r``, ``U`` = tangential
    E), which is exactly what the library's Jones is.  (Asserting the OTHER
    p^-hat convention, ``J[0,0] = -r_TM``, reads 7.5e-01 on the nonmagnetic
    slab, so this is a measurement, not an assumption.)"""
    rl, tl, j = _library(eps, mu, thickness, wl, theta, m,
                         magnetic_path=magnetic_path)
    out = {}
    for pol, row, jd in (("tm", 0, (0, 0)), ("te", 1, (1, 1))):
        r, tau, rr, tt = slab_rt(eps, mu, thickness, wl, theta, pol,
                                 forward=forward)
        del tau
        jr = r
        out["R_" + pol] = abs(float(rl[row]) - rr)
        out["T_" + pol] = abs(float(tl[row]) - tt)
        out["J_" + pol] = abs(complex(j[jd]) - jr)
    out["J_offdiag"] = max(abs(complex(j[0, 1])), abs(complex(j[1, 0])))
    if want:
        return out
    return max(out.values())


def _table(title, rows):
    print(title)
    for r in rows:
        print("   " + r)
    print()


def main():
    wl, t = 0.55e-6, 0.30e-6
    if len(sys.argv) > 1 and sys.argv[1] == "convention":
        print("--- the exp(-i w t) branch is MEASURED, not assumed ---")
        for fwd in (True, False):
            res = airy_residual(4.0 + 0.20j, 2.0, t, wl, 0.35, 8, want=True,
                                forward=fwd)
            tag = "exp(-iwt) (+i forward transfer)" if fwd else \
                "exp(+iwt) (the other branch)"
            print(f"  {tag:34s} " + "  ".join(
                f"{k} {v:.3e}" for k, v in sorted(res.items())))
        return

    rows = []
    for th in (0.0, 0.35):
        a = airy_residual(4.0, 1.0, t, wl, th, 8, magnetic_path=False)
        b = airy_residual(4.0, 1.0, t, wl, th, 8, magnetic_path=True)
        rows.append(f"theta {th:4.2f}   mu omitted {a:.4e}   "
                    f"mu=1 forced through the magnetic path {b:.4e}")
    _table("--- V3A: pin the oracle on the NONMAGNETIC arm (eps=4, M=8) ---",
           rows)

    rows = []
    for eps, mu, tag in ((4.0, 2.0, "eps 4.0        mu 2.0       "),
                         (1.0, 4.0, "eps 1.0        mu 4.0       "),
                         (4.0, 2.0 + 0.3j, "eps 4.0        mu 2.0+0.3i  "),
                         (4.0 + 0.2j, 2.0, "eps 4.0+0.2i   mu 2.0       "),
                         (2.25, 0.60, "eps 2.25       mu 0.60      "),
                         (9.0 + 1.0j, 1.8 + 0.4j,
                          "eps 9.0+1.0i   mu 1.8+0.4i ")):
        r0 = airy_residual(eps, mu, t, wl, 0.0, 8)
        r1 = airy_residual(eps, mu, t, wl, 0.35, 8)
        rows.append(f"{tag} theta 0 {r0:.4e}   theta 0.35 {r1:.4e}")
    _table("--- V3B: MAGNETIC slabs vs the analytic oracle (M=8) ---", rows)

    rows = []
    for m in (5, 6, 7, 8):
        rows.append(f"M={m}  {airy_residual(4.0, 2.0, t, wl, 0.35, m):.4e}")
    _table("--- V3C: spectral convergence, eps 4 mu 2, theta 0.35 ---", rows)


if __name__ == "__main__":
    main()
