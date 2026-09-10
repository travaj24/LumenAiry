"""V5 -- G4: the MAGNETIC stripe against the two 1-D engines, through duality.

NO 1-D diffraction engine in the library accepts a permeability (see the census
in :func:`census`), so the magnetic stripe cannot be compared directly.  It can
be compared through the EXACT duality of V4: the y-uniform MAGNETIC stripe
``(eps = vacuum, mu = P)`` is the dual of the y-uniform ELECTRIC stripe
``(eps = P, mu = 1)``, which BOTH 1-D engines solve.

At a CLASSICAL mount (``phi = 0``, so ``ky = 0``) the duality map of V4 is
anti-diagonal, ``A(k^) = [[0, -k^z], [1/k^z, 0]]``, and

    T'_m = A(k^_m) T_m A(k^_inc)^-1,   A_i = [[0, -kzi], [1/kzi, 0]],
                                       A_r,m = [[0, kzm], [-1/kzm, 0]]

(the REFLECTED branch carries ``k^z = -kz_m``, which is where the overall MINUS
below comes from), i.e.

    T'_m = [[ -(kzm/kzi) T22 ,  kzm kzi T21 ],
            [  T12/(kzm kzi) , -(kzi/kzm) T11 ]]

so at the SPECULAR order (``kzm = kzi = kz``)

    J' = [[-J22, kz^2 J21], [J12/kz^2, -J11]]

-- the OVERALL SIGN is load-bearing and measurable: dropping it leaves the
residual at 2|J| = 6.7e-01 instead of 5e-06 (it never converges in M).

and the EFFICIENCY rows swap EXACTLY at any theta: the dual x-drive (unit
tangential ``(1,0)``, ``|E| = 1/kz``) corresponds to the original tangential
``A_i^-1 (1,0) = (0, -1/kz)``, the same ``|E|``, so the power ratio is
unchanged -- ``R'[0] = R[1]``, ``R'[1] = R[0]`` (likewise T).

Step 0 pins the GEOMETRY and the Jones BASIS on the nonmagnetic control (the
electric stripe through the 2-D staggered engine vs the 1-D engines), so any
residual in the magnetic arm is the magnetic assembly and not an alignment.
"""
import subprocess
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm.oned import pmm_jones_1d
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.rcwa.oned import rcwa_jones_1d

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__

_PX = 0.90e-6
_WL = 0.55e-6
_T = 0.30e-6


def _lc(no, ne, psi):
    c, s = np.cos(psi), np.sin(psi)
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ d @ rz.T


_RIDGE = _lc(1.50, 1.72, 0.40)
_GROOVE = np.eye(3, dtype=complex) * 2.10


def _stripe_cell(ridge, groove, nx=2):
    """A y-UNIFORM (Nx, Nx, 3, 3) cell: x-segment 0 = ridge (duty 0.5)."""
    c = np.zeros((nx, nx, 3, 3), dtype=complex)
    c[0, :] = ridge
    c[1, :] = groove
    return c


def _two_d(eps_cell, mu_cell, m, theta, n_orders=5):
    st = PMM2DStackPure(_PX, _PX, n_superstrate=1.0, n_substrate=1.0,
                        n_modes=m, n_orders=n_orders)
    if mu_cell is None:
        st.add_layer(_T, eps_cell=eps_cell)
    else:
        st.add_layer(_T, eps_cell=eps_cell, mu_cell=mu_cell)
    st.set_source(_WL, theta=theta, phi=0.0)
    o, r, t, j = st.solve(jones=True)
    o = np.asarray(o)
    return {"orders": o, "R": r, "T": t, "J": j,
            "kz": st.per_order_amplitudes("reflection")["kz"],
            "kz_inc": st.per_order_amplitudes("reflection")["kz_inc"]}


def _one_d(engine, theta, **kw):
    o, r, t, j = engine(_PX, _RIDGE, _GROOVE, 1.0, 1.0, _T, 0.5, _WL,
                        theta=theta, **kw)
    return {"orders": np.asarray(o), "R": np.asarray(r), "T": np.asarray(t),
            "J": np.asarray(j)}


def _row_by_order(res2d, res1d):
    """Line up the 2-D (m, n) rows with the 1-D (m,) orders; also return the
    worst amplitude found on a y-FORBIDDEN (n != 0) order."""
    o2, o1 = res2d["orders"], res1d["orders"]
    if o2.ndim == 1:                     # 1-D vs 1-D (the two oracles)
        idx = [int(np.flatnonzero(o2 == mm)[0]) if np.any(o2 == mm) else -1
               for mm in o1]
        keep = [k for k, v in enumerate(idx) if v >= 0]
        return (np.array([idx[k] for k in keep]), np.array(keep), 0.0)
    idx, leak = [], 0.0
    for i in range(o2.shape[0]):
        if o2[i, 1] != 0:
            leak = max(leak, float(np.max(np.abs(res2d["R"][:, i]))),
                       float(np.max(np.abs(res2d["T"][:, i]))))
    for mm in o1:
        w = np.flatnonzero((o2[:, 0] == mm) & (o2[:, 1] == 0))
        idx.append(int(w[0]) if w.size else -1)
    keep = [k for k, v in enumerate(idx) if v >= 0]
    return np.array([idx[k] for k in keep]), np.array(keep), leak


def compare(res2d, res1d, swap_rows, kz_specular=None):
    """max |dR|, |dT| per order and the Jones residual."""
    i2, i1, leak = _row_by_order(res2d, res1d)
    r2, t2 = res2d["R"][:, i2], res2d["T"][:, i2]
    r1, t1 = res1d["R"][:, i1], res1d["T"][:, i1]
    if swap_rows:
        r1, t1 = r1[::-1], t1[::-1]
    drt = max(float(np.max(np.abs(r2 - r1))), float(np.max(np.abs(t2 - t1))))
    j1 = res1d["J"]
    if swap_rows:
        kz = complex(kz_specular)
        j1 = np.array([[-j1[1, 1], kz ** 2 * j1[1, 0]],
                       [j1[0, 1] / kz ** 2, -j1[0, 0]]])
    dj = float(np.max(np.abs(res2d["J"] - j1)))
    return drt, dj, leak


def census():
    print("--- ENGINE CENSUS: does ANY 1-D engine take a permeability? ---")
    cmd = ["grep", "-rn", "--include=*.py", "-iE",
           r"permeability|\bmu_|\bmu\b|\bmu=", "lumenairy/elements/rcwa",
           "lumenairy/elements/pmm", "lumenairy/elements/berreman.py"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         cwd="C:\\tmp\\lum_vmag").stdout.splitlines()
    hits = {}
    for line in out:
        f = line.split(":", 1)[0]
        hits[f] = hits.get(f, 0) + 1
    for f, n in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4d}  {f}")
    print("   (twod_staggered / stack2d_pure are THIS build; every other hit "
          "is inspected in the report)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "census":
        census()
        return
    print("--- V5A: pin the GEOMETRY + Jones basis on the NONMAGNETIC arm ---")
    print("    (the same ELECTRIC stripe through the 2-D staggered engine and "
          "the two 1-D engines)")
    for theta in (0.0, 0.22):
        for m in (6, 8):
            a = _two_d(_stripe_cell(_RIDGE, _GROOVE), None, m, theta)
            p = _one_d(pmm_jones_1d, theta, degree=24, far_field_orders=21)
            r = _one_d(rcwa_jones_1d, theta, n_orders=81)
            dp = compare(a, p, False)
            dr = compare(a, r, False)
            pr = compare(p, r, False)
            print(f"  theta {theta:4.2f} M={m}   vs pmm_1d dRT {dp[0]:.3e} "
                  f"dJ {dp[1]:.3e}   vs rcwa_1d dRT {dr[0]:.3e} dJ "
                  f"{dr[1]:.3e}   [1d oracles vs each other dRT {pr[0]:.3e} "
                  f"dJ {pr[1]:.3e}]   y-leak {dp[2]:.2e}")
    print()
    print("--- V5B: the MAGNETIC stripe (eps = vacuum, mu = the pattern) ---")
    vac = np.zeros((2, 2, 3, 3), dtype=complex)
    vac[...] = np.eye(3)
    for theta in (0.0, 0.22):
        p = _one_d(pmm_jones_1d, theta, degree=24, far_field_orders=21)
        r = _one_d(rcwa_jones_1d, theta, n_orders=81)
        for m in (5, 6, 7, 8):
            b = _two_d(vac, _stripe_cell(_RIDGE, _GROOVE), m, theta)
            i2, _i1, _leak = _row_by_order(b, p)
            kzs = b["kz_inc"]
            dp = compare(b, p, True, kzs)
            dr = compare(b, r, True, kzs)
            ctrl = compare(b, p, False)
            print(f"  theta {theta:4.2f} M={m}   vs pmm_1d dRT {dp[0]:.3e} "
                  f"dJ {dp[1]:.3e}   vs rcwa_1d dRT {dr[0]:.3e} dJ "
                  f"{dr[1]:.3e}   y-leak {dp[2]:.2e}   [unswapped control "
                  f"dRT {ctrl[0]:.3e}]")
            del i2
    print()
    census()


if __name__ == "__main__":
    main()
