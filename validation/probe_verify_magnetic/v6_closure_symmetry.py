"""V6 -- G5 closure, G6 the x<->y transpose, G7 the guards, and composition.

G5.  A HERMITIAN permeability absorbs NOTHING: the Poynting dissipation term
carries the anti-Hermitian parts of eps AND mu, so a gyrotropic
``m12 = -m21 = i b`` (Hermitian, complex) is lossless while ``Im(m11) > 0``
(anti-Hermitian) is not.  Measured two-sided: the Hermitian arm closes, the
lossy arm closes strictly BELOW one, and the closure tripwire
(``_STAG_CLOSURE_TOL = 5e-2``) stays silent on a resolved fixture and FIRES on a
deliberately under-resolved one.

G6.  Transposing the cell about ``x = y`` is an exact symmetry when the grid
axes, the PERIODS, the incidence azimuth (``phi -> pi/2 - phi``) and the TENSOR
components are all transposed: ``e11 <-> e22``, ``e12 <-> e21`` -- and the SAME
on mu.  Orders map ``(m, n) -> (n, m)`` and the tangential Jones conjugates by
``P = [[0,1],[1,0]]``: ``J_T = P J P``.  Deliberately swapping ONLY ``m12/m21``
in one arm must BREAK it; swapping only ``e12/e21`` must NOT (a rotated-LC
permittivity has ``e12 = e21``), which is why the permeability needs its own
GYROTROPIC discriminator.

G7.  Every documented guard, plus the negative side: float noise in ``m13`` must
NOT trip the block-form gate, and must leave R BIT-IDENTICAL.
"""
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    _homog_geom_cache,
    pmm_jones_2d_staggered,
)

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__

_PX = 0.90e-6
_WL = 0.55e-6
_T = 0.30e-6


def _lc(no, ne, psi):
    c, s = np.cos(psi), np.sin(psi)
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ d @ rz.T


def _tilt(no=1.50, ne=1.72, psi=0.35, tilt=0.30):
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    cz, sz = np.cos(psi), np.sin(psi)
    cy, sy = np.cos(tilt), np.sin(tilt)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    q = rz @ ry
    return q @ d @ q.T


def _gyro(b=0.40, m=1.20, m33=1.10):
    """HERMITIAN: m12 = i b, m21 = -i b = conj(m12)."""
    return np.array([[m, 1j * b, 0.0], [-1j * b, m, 0.0], [0.0, 0.0, m33]],
                    dtype=complex)


def _lossy_mu(imag=0.25):
    return np.array([[1.30 + 1j * imag, 0.22, 0.0], [0.22, 1.75, 0.0],
                     [0.0, 0.0, 1.45]], dtype=complex)


def _cell(fill, back, nx=2):
    c = np.zeros((nx, nx, 3, 3), dtype=complex)
    c[...] = back
    c[0, 0] = fill
    return c


def _solve(eps, mu, m, theta, phi, px=_PX, py=None, record=False):
    st = PMM2DStackPure(px, py or px, n_superstrate=1.0, n_substrate=1.0,
                        n_modes=m, n_orders=4)
    st.add_layer(_T, eps_cell=eps, mu_cell=mu)
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, r, t, j = st.solve(jones=True)
    hits = [str(x.message) for x in w if "closure" in str(x.message)]
    out = {"orders": np.asarray(o), "R": r, "T": t, "J": j,
           "closure": np.abs(r.sum(axis=1) + t.sum(axis=1) - 1.0),
           "total": r.sum(axis=1) + t.sum(axis=1), "warnings": hits}
    return out if record else out


# ---------------------------------------------------------------------- G5
def g5():
    print("--- V6/G5A: a HERMITIAN mu absorbs nothing (|sum R + sum T - 1|) ---")
    iso = np.eye(3, dtype=complex) * 2.10
    fixtures = {
        "uniform LC eps, uniform gyrotropic mu":
            (_cell(_lc(1.50, 1.72, 0.40), _lc(1.50, 1.72, 0.40)),
             _cell(_gyro(), _gyro())),
        "patterned LC/iso eps, gyro/1.2 mu":
            (_cell(_lc(1.50, 1.72, 0.40), iso),
             _cell(_gyro(), np.eye(3, dtype=complex) * 1.2)),
        "VACUUM eps, patterned mu":
            (np.ones((2, 2), dtype=complex),
             _cell(_gyro(), np.eye(3, dtype=complex) * 1.6)),
        "real scalar mu = 0.7 (mu < 1)":
            (_cell(_lc(1.50, 1.72, 0.40), iso),
             np.array([[0.70, 2.30], [1.40, 0.90]], dtype=complex)),
    }
    for name, (e, m) in fixtures.items():
        cells = []
        for mm in (6, 8):
            for th, ph, tag in ((0.0, 0.0, "normal"), (0.25, 0.60, "conical")):
                d = _solve(e, m, mm, th, ph)
                cells.append(f"M={mm} {tag} {float(np.max(d['closure'])):.3e}")
        print(f"  {name:38s} " + "  ".join(cells))
    print()

    print("--- V6/G5B: a LOSSY mu (Im m11 = +0.25) must close BELOW one ---")
    e = _cell(_lc(1.50, 1.72, 0.40), iso)
    m = _cell(_lossy_mu(), np.eye(3, dtype=complex) * 1.2)
    for mm in (6, 8):
        d = _solve(e, m, mm, 0.25, 0.60)
        print(f"  M={mm}  sum R+T (Ex / Ey) = {d['total'][0]:.6f} / "
              f"{d['total'][1]:.6f}   warnings {len(d['warnings'])}")
    print("  (control: the SAME cell with Im(m11) removed ->", end=" ")
    m0 = m.copy()
    m0[0, 0, 0, 0] = 1.30
    d0 = _solve(e, m0, 8, 0.25, 0.60)
    print(f"total {d0['total'][0]:.6f} / {d0['total'][1]:.6f})")
    print()

    print("--- V6/G5C: the closure TRIPWIRE, both sides ---")
    for mm, tag in ((8, "RESOLVED"), (3, "under-resolved (M=3, the minimum)")):
        d = _solve(e, m0, mm, 0.25, 0.60)
        print(f"  Hermitian mu, M={mm} {tag:34s} defect "
              f"{float(np.max(d['closure'])):.3e}   warnings raised "
              f"{len(d['warnings'])}")
    d = _solve(e, m, 8, 0.25, 0.60)
    print(f"  LOSSY mu, M=8 (no unity claim is made)         warnings raised "
          f"{len(d['warnings'])}")
    print()


# ---------------------------------------------------------------------- G6
def _transpose_tensor(t):
    """Swap the 1<->2 tensor components (e11<->e22, e12<->e21), leaving e33."""
    o = t.copy()
    o[..., 0, 0], o[..., 1, 1] = t[..., 1, 1], t[..., 0, 0]
    o[..., 0, 1], o[..., 1, 0] = t[..., 1, 0], t[..., 0, 1]
    return o


def _transpose_cell(c):
    c = np.asarray(c)
    if c.ndim == 2:
        return np.ascontiguousarray(c.T)
    return np.ascontiguousarray(_transpose_tensor(np.swapaxes(c, 0, 1)))


def _g6_residual(eps, mu, m, theta, phi, break_mu=False, break_eps=False):
    a = _solve(eps, mu, m, theta, phi)
    et, mt = _transpose_cell(eps), _transpose_cell(mu)
    if break_mu:
        mt = mt.copy()
        mt[..., 0, 1], mt[..., 1, 0] = mt[..., 1, 0].copy(), mt[..., 0, 1].copy()
    if break_eps:
        et = et.copy()
        et[..., 0, 1], et[..., 1, 0] = et[..., 1, 0].copy(), et[..., 0, 1].copy()
    b = _solve(et, mt, m, theta, np.pi / 2.0 - phi)
    oa, ob = a["orders"], b["orders"]
    worst = 0.0
    for i in range(oa.shape[0]):
        w = np.flatnonzero((ob[:, 0] == oa[i, 1]) & (ob[:, 1] == oa[i, 0]))
        k = int(w[0])
        worst = max(worst,
                    float(np.max(np.abs(a["R"][:, i] - b["R"][::-1, k]))),
                    float(np.max(np.abs(a["T"][:, i] - b["T"][::-1, k]))))
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    dj = float(np.max(np.abs(a["J"] - p @ b["J"] @ p)))
    return worst, dj


def g6():
    print("--- V6/G6: the x<->y transpose with the mu blocks swapped ---")
    iso = np.eye(3, dtype=complex) * 2.10
    eps = _cell(_lc(1.50, 1.72, 0.40), iso)
    mu = _cell(_gyro(), np.eye(3, dtype=complex) * 1.2)
    for m in (5, 6):
        a = _g6_residual(eps, mu, m, 0.25, 0.60)
        b = _g6_residual(eps, mu, m, 0.25, 0.60, break_mu=True)
        c = _g6_residual(eps, mu, m, 0.25, 0.60, break_eps=True)
        print(f"  M={m}  correct placement  dRT {a[0]:.3e}  dJ {a[1]:.3e}")
        print(f"        m12/m21 SWAPPED     dRT {b[0]:.3e}  dJ {b[1]:.3e}")
        print(f"        e12/e21 swapped     dRT {c[0]:.3e}  dJ {c[1]:.3e}"
              "   (control: a rotated-LC eps has e12 == e21, so this is a "
              "no-op)")
    print()


# ---------------------------------------------------------------------- G7
def _raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as exc:            # noqa: BLE001
        return type(exc).__name__, str(exc)[:70]
    return "NO RAISE", ""


def g7():
    print("--- V6/G7: guards ---")
    iso = np.eye(3, dtype=complex) * 2.10
    eps = _cell(_lc(1.50, 1.72, 0.40), iso)
    good = _cell(_gyro(), np.eye(3, dtype=complex) * 1.2)
    oop_mu = good.copy()
    oop_mu[0, 0, 0, 2] = oop_mu[0, 0, 2, 0] = 0.30
    sing = good.copy()
    sing[0, 0, :2, :2] = np.array([[1.0, 1.0], [1.0, 1.0]])
    zero33 = good.copy()
    zero33[0, 0, 2, 2] = 0.0
    oop_eps = _cell(_tilt(), iso)
    j = pmm_jones_2d_staggered
    checks = [
        ("out-of-plane mu",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=oop_mu,
                   n_modes=4)),
        ("mu with an out-of-plane eps",
         lambda: j(_PX, _PX, oop_eps, 1.0, 1.0, _T, _WL, mu_cell=good,
                   n_modes=4)),
        ("mu_superstrate != 1",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=good,
                   mu_superstrate=1.6, n_modes=4)),
        ("mu_substrate != 1",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=good,
                   mu_substrate=2.0, n_modes=4)),
        ("PMM2DStackPure(mu_superstrate=)",
         lambda: PMM2DStackPure(_PX, mu_superstrate=1.6)),
        ("singular [mu_t] (det = 0)",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=sing,
                   n_modes=4)),
        ("m33 = 0",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=zero33,
                   n_modes=4)),
        ("mu_cell of a bad shape",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL,
                   mu_cell=np.ones((2, 2, 2, 2)), n_modes=4)),
        ("mu_cell grid != eps_cell grid",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL,
                   mu_cell=np.ones((3, 3)), n_modes=4)),
        ("non-square mu_cell",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL,
                   mu_cell=np.ones((2, 3)), n_modes=4)),
        ("zero scalar mu_cell",
         lambda: j(_PX, _PX, eps, 1.0, 1.0, _T, _WL,
                   mu_cell=np.zeros((2, 2)), n_modes=4)),
        ("both mu and mu_cell",
         lambda: PMM2DStackPure(_PX, n_modes=4).add_layer(
             _T, eps_cell=eps, mu=1.4, mu_cell=good)),
        ("uniform mu of a bad shape",
         lambda: PMM2DStackPure(_PX, n_modes=4).add_layer(
             _T, eps_cell=eps, mu=np.ones((2, 2)))),
        ("uniform scalar mu = 0",
         lambda: PMM2DStackPure(_PX, n_modes=4).add_layer(
             _T, eps_cell=eps, mu=0.0)),
        ("add_layer(eps_cell=out-of-plane) with mu",
         lambda: PMM2DStackPure(_PX, n_modes=4).add_layer(
             _T, eps_cell=oop_eps, mu=1.4)),
        ("add_layer(eps=out-of-plane (3,3)) with mu",
         lambda: PMM2DStackPure(_PX, n_modes=4).add_layer(
             _T, eps=_tilt(), mu=1.4)),
        ("_homog_geom_cache on a magnetic solver",
         lambda: _homog_geom_cache(Granet2DTransverseE(
             _PX, _PX, 2, 2, 4, np.full((2, 2), 4.0 + 0j),
             mu_cell=np.full((2, 2), 1.4 + 0j)))),
    ]
    for name, fn in checks:
        kind, msg = _raises(fn)
        print(f"  {name:42s} {kind:20s} {msg}")

    print("  --- the NEGATIVE side of the floor ---")
    for name, val in (("mu_superstrate=1.0 accepted", 1.0),
                      ("mu_substrate=1 accepted", 1)):
        kind, _m = _raises(
            j, _PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=good, n_modes=4,
            **({"mu_superstrate": val} if "super" in name
               else {"mu_substrate": val}))
        print(f"  {name:42s} {kind}")
    stray = good.copy()
    stray[0, 0, 0, 2] = 1e-16
    stray[0, 0, 2, 1] = -1e-16
    a = j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=good, n_modes=5,
          n_orders=3)
    b = j(_PX, _PX, eps, 1.0, 1.0, _T, _WL, mu_cell=stray, n_modes=5,
          n_orders=3)
    print(f"  1e-16 stray in m13/m32: passes the gate, max|dR| = "
          f"{float(np.max(np.abs(a[1] - b[1]))):.1e}, "
          f"bit-identical R/T/J = "
          f"{bool(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3]))}")
    print()


# -------------------------------------------------------------- composition
def composition():
    print("--- V6/C1: layer_absorption on a LOSSY MAGNETIC layer ---")
    iso = np.eye(3, dtype=complex) * 2.10
    eps = _cell(_lc(1.50, 1.72, 0.40), iso)
    mu = _cell(_lossy_mu(), np.eye(3, dtype=complex) * 1.2)
    for m in (5, 6, 7, 8):
        st = PMM2DStackPure(_PX, _PX, n_superstrate=1.0, n_substrate=1.0,
                            n_modes=m, n_orders=4)
        st.add_layer(_T, eps_cell=eps, mu_cell=mu)
        st.add_layer(0.15e-6, eps=2.25)
        st.set_source(_WL, theta=0.20, phi=0.40)
        o, r, t, jj = st.solve(jones=True, retain_internal=True)
        a = np.asarray(st.layer_absorption())
        lhs = a.sum(axis=0) if a.ndim > 1 else a
        rhs = 1.0 - r.sum(axis=1) - t.sum(axis=1)
        print(f"  M={m}  sum A {np.atleast_1d(lhs)[0]:.6f} / "
              f"{np.atleast_1d(lhs)[-1]:.6f}   1-R-T {rhs[0]:.6f} / "
              f"{rhs[1]:.6f}   closure "
              f"{float(np.max(np.abs(np.atleast_1d(lhs) - rhs))):.3e}")
        del o, jj
    print()
    print("--- V6/C2: a MAGNETIC layer beside an OUT-OF-PLANE layer ---")
    oop = _cell(_tilt(), iso)
    for m in (4, 5, 6, 7):
        st = PMM2DStackPure(_PX, _PX, n_superstrate=1.0, n_substrate=1.0,
                            n_modes=m, n_orders=4)
        st.add_layer(0.20e-6, eps_cell=oop)
        st.add_layer(0.20e-6, eps=2.25, mu=np.eye(3, dtype=complex) * 1.6)
        st.set_source(_WL, theta=0.20, phi=0.40)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            o, r, t, jj = st.solve(jones=True)
        tot = r.sum(axis=1) + t.sum(axis=1)
        print(f"  M={m}  |sum R + sum T - 1| = "
              f"{float(np.max(np.abs(tot - 1.0))):.3e}   warnings "
              f"{len([x for x in w if 'closure' in str(x.message)])}")
        del o, jj
    print()


def main():
    which = sys.argv[1:] or ["g5", "g6", "g7", "comp"]
    if "g5" in which:
        g5()
    if "g6" in which:
        g6()
    if "g7" in which:
        g7()
    if "comp" in which:
        composition()


if __name__ == "__main__":
    main()
