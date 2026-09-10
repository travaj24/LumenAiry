"""V4 -- G3: ELECTROMAGNETIC DUALITY, with the transform re-derived here.

For ``exp(-i w t)`` Maxwell (``curl E = i w mu0 mu H``,
``curl H = -i w eps0 eps E``),

    (E, H, eps, mu)  ->  (Z0 H, -E / Z0, mu, eps),   Z0 = sqrt(mu0/eps0)

is an EXACT symmetry (substitute and use ``Z0 eps0 = mu0 / Z0 = sqrt(mu0 eps0)``).
With VACUUM (self-dual) half-spaces the swapped grating ``(mu, eps)`` is the
SAME physical problem, so its per-order response is the original's composed
with the duality map on amplitudes.

THE MAP, in the LAB TANGENTIAL basis the library reports (no (s,p) frame and no
p^-hat convention needed).  For a plane wave in vacuum, ``Z0 H = k^ x E`` and
``E_z = -(k^x Ex + k^y Ey)/k^z``, so

    E'_x = -(k^x k^y / k^z) Ex - ((k^y^2 + k^z^2)/k^z) Ey
    E'_y =  ((k^x^2 + k^z^2)/k^z) Ex + (k^x k^y / k^z) Ey

    A(k^) = (1/k^z) [[ -k^x k^y , -(k^y^2 + k^z^2) ],
                     [  k^x^2 + k^z^2 ,  k^x k^y   ]]

(classical mount ``k^y = 0``: ``A = [[0, -k^z], [1/k^z, 0]]``; normal incidence:
``A = [[0,-1],[1,0]] = D``).  Since ``A`` depends on ``k^z`` only through the
prefactor and even powers, the REFLECTED branch gives ``A_r = -A_i`` at the
specular order.

If ``T_m`` is the 2x2 per-order response (columns = incident lab Ex / Ey drives,
rows = order-m tangential Ex / Ey), then

    T'_m = A(k^_m) T_m A(k^_inc)^-1

with ``k^_m = (kx_m, ky_m, -kz_m)`` on the REFLECTION port and
``(kx_m, ky_m, +kz_m)`` on TRANSMISSION.  At normal incidence this collapses to
``T' = -D T D^-1 = D T D`` and the two EFFICIENCY rows simply SWAP.

Measured here: the residual ``max_m |T'_m - A_m T_m A_i^-1|`` over all orders,
its no-rotation control, and (at normal incidence) the row-swapped R/T.
"""
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__

_PX = 0.90e-6
_WL = 0.55e-6
_T = 0.30e-6


def _amap(kx, ky, kz):
    kz = complex(kz)
    return np.array([[-kx * ky, -(ky ** 2 + kz ** 2)],
                     [kx ** 2 + kz ** 2, kx * ky]], dtype=complex) / kz


def _lc(no, ne, psi):
    c, s = np.cos(psi), np.sin(psi)
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ d @ rz.T


def _solve(eps_spec, mu_spec, m, theta, phi, nx=2):
    st = PMM2DStackPure(_PX, _PX, n_superstrate=1.0, n_substrate=1.0,
                        n_modes=m, n_orders=3)
    st.add_layer(_T, eps_cell=eps_spec, mu_cell=mu_spec)
    st.set_source(_WL, theta=theta, phi=phi)
    o, r, t, j = st.solve(jones=True)
    out = {"orders": np.asarray(o), "R": r, "T": t, "J": j}
    for port in ("reflection", "transmission"):
        d = st.per_order_amplitudes(port)
        out[port] = d
    return out


def _resp(d):
    """(N, 2, 2) per-order response matrices, columns = incident Ex / Ey."""
    ex, ey = d["Ex"], d["Ey"]
    n = ex.shape[1]
    tm = np.zeros((n, 2, 2), dtype=complex)
    tm[:, 0, 0], tm[:, 0, 1] = ex[0], ex[1]
    tm[:, 1, 0], tm[:, 1, 1] = ey[0], ey[1]
    return tm


def duality_residual(a, b, rotate=True, evanescent=False):
    """max over ports and orders of |T'_m - A_m T_m A_i^-1|, normalised by the
    largest |T_m| in play.

    PROPAGATING orders only by default.  The relation is exact for evanescent
    orders too, but those are NEAR-FIELD amplitudes and the staggered basis is
    not self-dual (``E3`` sits in V3 while ``H3`` lands in Vw), so they carry
    the discretization defect amplified: on fixture C at M=6 the worst
    evanescent order reads 1.2e+00 against 5.4e-02 on the worst propagating
    one, and both fall together in M.  ``evanescent=True`` measures that set."""
    worst, scale = 0.0, 0.0
    for port, zsign in (("reflection", -1.0), ("transmission", +1.0)):
        da, db = a[port], b[port]
        ai = _amap(da["kx0"], da["ky0"], da["kz_inc"])
        aiinv = np.linalg.inv(ai)
        ta, tb = _resp(da), _resp(db)
        prop = (np.real(da["kz"]) > 0.10) & (np.abs(np.imag(da["kz"])) < 1e-9)
        keep = ~prop if evanescent else prop
        for i in np.flatnonzero(keep):
            am = _amap(da["kx"][i], da["ky"][i], zsign * da["kz"][i])
            pred = (am @ ta[i] @ aiinv) if rotate else ta[i]
            worst = max(worst, float(np.max(np.abs(tb[i] - pred))))
            scale = max(scale, float(np.max(np.abs(tb[i]))),
                        float(np.max(np.abs(ta[i]))))
    return worst / max(scale, 1e-300)


def _row_swap_rt(a, b):
    """At NORMAL incidence the duality map is a signed permutation, so the two
    EFFICIENCY rows must swap exactly."""
    return max(float(np.max(np.abs(b["R"][0] - a["R"][1]))),
               float(np.max(np.abs(b["R"][1] - a["R"][0]))),
               float(np.max(np.abs(b["T"][0] - a["T"][1]))),
               float(np.max(np.abs(b["T"][1] - a["T"][0]))))


def _vac(nx=2):
    return np.ones((nx, nx), dtype=complex)


def _patterned_e(nx=2):
    c = np.zeros((nx, nx, 3, 3), dtype=complex)
    c[...] = np.eye(3) * (2.10 + 0.0j)
    c[0, 0] = _lc(1.50, 1.72, 0.40)
    return c


def _patterned_m(nx=2):
    c = np.zeros((nx, nx, 3, 3), dtype=complex)
    c[...] = np.eye(3) * (1.60 + 0.0j)
    c[0, 0] = _lc(1.05, 1.28, -0.30)
    return c


def _uniform_e():
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[...] = _lc(1.50, 1.72, 0.40)
    return c


def _uniform_m():
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[...] = _lc(1.05, 1.28, -0.30)
    return c


_CASES = {
    "A uniform (eps, mu) tensor pair": (_uniform_e(), _uniform_m()),
    "B patterned eps, mu = vacuum (purely electric <-> purely magnetic)":
        (_patterned_e(), np.ones((2, 2), dtype=complex)),
    "C BOTH patterned and anisotropic": (_patterned_e(), _patterned_m()),
}

_INC = [("normal", 0.0, 0.0), ("theta 0.30", 0.30, 0.0),
        ("conical 0.30 / 0.70", 0.30, 0.70)]


def main():
    ms = [int(x) for x in (sys.argv[1:] or ["5", "6", "7", "8"])]
    for name, (eps, mu) in _CASES.items():
        print(f"--- {name} ---")
        for tag, th, ph in _INC:
            cells = []
            for m in ms:
                a = _solve(eps, mu, m, th, ph)
                b = _solve(mu, eps, m, th, ph)
                cells.append(f"M={m} {duality_residual(a, b):.3e}")
            ctrl = duality_residual(a, b, rotate=False)
            evan = duality_residual(a, b, evanescent=True)
            extra = ""
            if tag == "normal":
                extra = f"   R/T row-swap {_row_swap_rt(a, b):.3e}"
            print(f"  {tag:22s} " + "  ".join(cells)
                  + f"   [control {ctrl:.3e}, evanescent(M={ms[-1]}) "
                  f"{evan:.3e}]" + extra)
        print()


if __name__ == "__main__":
    main()
