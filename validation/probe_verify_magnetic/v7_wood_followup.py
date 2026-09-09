"""V7 -- FOLLOW-UP 5: the Wood-anomaly nudge list must carry a MAGNETIC layer's
own cut-offs, ``kt^2 = Re(eps mu)``.

A Rayleigh cut-off inside a layer sits where the LAYER's longitudinal
wavenumber vanishes.  For a nonmagnetic layer that is ``kt^2 = Re(eps)``, which
is what ``_wood_eps_reals`` already lists; for a magnetic one it is
``kt^2 = Re(eps mu)`` (index ``n = sqrt(eps mu)``), so a magnetic layer sitting
exactly on its own cut-off was NOT being nudged.

Two arms:

* ``hash``  -- every NONMAGNETIC call site, hashed, so the change can be shown
  byte-identical before and after (run it on both sides of the edit and diff).
  It also hashes a magnetic layer with ``mu = 1`` (whose eps*mu product is
  ``eps`` exactly in float64) and a magnetic layer OFF any cut-off.
* ``cutoff`` -- the two-sided behaviour: a magnetic layer ON its ``eps*mu``
  cut-off is nudged; the same layer with ``mu = 1`` is not (its ``eps`` alone
  is off every cut-off), and neither is either half-space.
"""
import hashlib
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__

# NORMAL incidence, px = py: kt^2(m, n) = (m^2 + n^2) (wl/px)^2.  With
# wl = px sqrt(EPS_L * MU_L) the (+/-1, 0) and (0, +/-1) orders sit EXACTLY on
# the magnetic layer's own cut-off, while EPS_L, MU_L and both half-spaces are
# far from every kt^2 -- so the permittivity-only rule cannot see it.
_PX = 0.50e-6
_EPS_L, _MU_L = 4.0, 2.25
_WL_CUT = _PX * np.sqrt(_EPS_L * _MU_L)
_N_SUP, _N_SUB = 1.0, 1.5
_DEPTH = 0.28e-6
_M, _NO = 5, 3


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        m.update(str(arr.shape).encode())
        m.update(arr.tobytes())
    return m.hexdigest()[:24]


def _orders(n_orders=_NO):
    mo = np.arange(-int(n_orders), int(n_orders) + 1)
    return np.tile(mo, len(mo)), np.repeat(mo, len(mo))


def _nudge(wl, eps_list, px=_PX):
    mx, my = _orders()
    return _grazing_safe_wavelength(float(wl), 0.0, 0.0, mx, my, px, px,
                                    list(eps_list))


def _stack(wl, *, eps, mu=None, uniform_eps=False, uniform_mu=True,
           theta=0.0, phi=0.0, n_modes=_M):
    s = PMM2DStackPure(_PX, _PX, n_superstrate=_N_SUP, n_substrate=_N_SUB,
                       n_modes=n_modes, n_orders=_NO)
    kw = {"eps": eps} if uniform_eps else {"eps_cell": eps}
    if mu is not None:
        kw["mu" if uniform_mu else "mu_cell"] = mu
    s.add_layer(_DEPTH, **kw)
    s.set_source(float(wl), theta=theta, phi=phi)
    o, r, t, j = s.solve(jones=True)
    return r, t, j


def _lc(no, ne, psi):
    c, s = np.cos(psi), np.sin(psi)
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ d @ rz.T


def arm_hash():
    """Every call site that must NOT move, plus the mu = 1 magnetic identity."""
    out = {}
    wl = 0.55e-6
    cell = np.array([[4.0, 1.0], [1.0, 2.25]], dtype=complex)
    tens = np.zeros((2, 2, 3, 3), dtype=complex)
    tens[...] = np.eye(3) * 2.10
    tens[0, 0] = _lc(1.50, 1.72, 0.40)
    for pol in ("te", "tm"):
        o, r, t = pmm_efficiency_2d_staggered(
            _PX, _PX, cell, _N_SUB, _N_SUP, _DEPTH, wl, degree=_M,
            n_orders=_NO, polarization=pol, theta=0.20, phi=0.30)[:3]
        out["eff_" + pol] = _h(r, t)
        del o
    o, r, t, j = pmm_jones_2d_staggered(_PX, _PX, cell, _N_SUB, _N_SUP,
                                        _DEPTH, wl, n_modes=_M, n_orders=_NO,
                                        theta=0.20, phi=0.30)
    out["jones_scalar"] = _h(r, t, j)
    o, r, t, j = pmm_jones_2d_staggered(_PX, _PX, tens, _N_SUB, _N_SUP,
                                        _DEPTH, wl, n_modes=_M, n_orders=_NO,
                                        theta=0.20, phi=0.30)
    out["jones_tensor"] = _h(r, t, j)
    out["stack_patterned"] = _h(*_stack(wl, eps=cell))
    out["stack_uniform"] = _h(*_stack(wl, eps=4.0, uniform_eps=True))
    out["stack_uniform_tensor"] = _h(
        *_stack(wl, eps=_lc(1.50, 1.72, 0.40), uniform_eps=True))
    out["stack_tensor_cell"] = _h(*_stack(wl, eps=tens))
    # magnetic, but mu = 1 in all four spellings: eps*mu == eps EXACTLY
    out["magnetic_mu1_scalar"] = _h(*_stack(wl, eps=cell, mu=1.0))
    out["magnetic_mu1_tensor"] = _h(
        *_stack(wl, eps=cell, mu=np.eye(3, dtype=complex)))
    out["magnetic_mu1_cell"] = _h(
        *_stack(wl, eps=cell, mu=np.ones((2, 2), dtype=complex),
                uniform_mu=False))
    out["magnetic_mu1_tcell"] = _h(
        *_stack(wl, eps=tens, uniform_mu=False,
                mu=np.broadcast_to(np.eye(3, dtype=complex),
                                   (2, 2, 3, 3)).copy()))
    # magnetic and OFF every cut-off: must not move either
    out["magnetic_offcut"] = _h(*_stack(wl, eps=cell, mu=1.7))
    out["magnetic_offcut_tensor"] = _h(
        *_stack(wl, eps=tens, mu=_lc(1.05, 1.28, -0.30)))
    out["magnetic_offcut_gyro"] = _h(*_stack(
        wl, eps=cell, mu=np.array([[1.2, 0.4j, 0.0], [-0.4j, 1.2, 0.0],
                                   [0.0, 0.0, 1.1]], dtype=complex)))
    for k in sorted(out):
        print(f"{k:26s} {out[k]}")


def arm_cutoff():
    kt2 = np.array([1.0, 2.0, 4.0, 5.0]) * (_WL_CUT / _PX) ** 2
    print("--- the fixture ---")
    print(f"  wl_cut / px = {_WL_CUT / _PX:.6f}, eps*mu = "
          f"{_EPS_L * _MU_L:.6f}, (wl/px)^2 - eps*mu = "
          f"{(_WL_CUT / _PX) ** 2 - _EPS_L * _MU_L!r}")
    for name, v in (("eps_sup", _N_SUP ** 2), ("eps_sub", _N_SUB ** 2),
                    ("eps_layer", _EPS_L), ("mu_layer", _MU_L),
                    ("eps*mu", _EPS_L * _MU_L)):
        print(f"  min |{name:9s} - kt^2| = "
              f"{float(np.min(np.abs(v - np.append(kt2, 0.0)))):.6f}")
    print("--- the RULES, driven directly ---")
    half = [_N_SUP ** 2, _N_SUB ** 2]
    for tag, lst in (("half-spaces only", half),
                     ("+ eps only (pre-follow-up)", half + [_EPS_L]),
                     ("+ eps and mu separately", half + [_EPS_L, _MU_L]),
                     ("+ eps*mu (follow-up)", half + [_EPS_L * _MU_L])):
        wl = _nudge(_WL_CUT, lst)
        print(f"  {tag:28s} nudged: {wl != _WL_CUT}   "
              f"dwl/wl = {(wl - _WL_CUT) / _WL_CUT:.3e}")
    print("--- the PUBLIC surface (a solve AT the cut-off vs AT the nudge) ---")
    cell = np.full((2, 2), _EPS_L + 0j)
    wl_nudged = _nudge(_WL_CUT, half + [_EPS_L * _MU_L])
    a = _stack(_WL_CUT, eps=cell, mu=_MU_L)
    b = _stack(wl_nudged, eps=cell, mu=_MU_L)
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    print(f"  MAGNETIC  mu = {_MU_L}:  solve(wl_cut) == solve(wl_nudged) "
          f"bit for bit -> {same}   (max|dR| "
          f"{float(np.max(np.abs(a[0] - b[0]))):.3e})")
    c = _stack(_WL_CUT, eps=cell, mu=1.0)
    d = _stack(wl_nudged, eps=cell, mu=1.0)
    same1 = all(np.array_equal(x, y) for x, y in zip(c, d))
    print(f"  the SAME layer with mu = 1: solve(wl_cut) == solve(wl_nudged) "
          f"-> {same1}   (max|dR| "
          f"{float(np.max(np.abs(c[0] - d[0]))):.3e})")
    e = _stack(_WL_CUT, eps=cell)
    d2 = max(float(np.max(np.abs(x - y))) for x, y in zip(c, e))
    print(f"  mu = 1 magnetic vs the NONMAGNETIC layer at wl_cut: max|d| "
          f"{d2:.3e}   (re-summation only; the two took the SAME nudge)")


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "cutoff"
    (arm_hash if arm == "hash" else arm_cutoff)()


if __name__ == "__main__":
    main()
