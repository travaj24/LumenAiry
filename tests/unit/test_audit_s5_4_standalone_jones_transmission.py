"""Audit S5-4 [P2][seam]: the standalone 1-D Jones functions return tuples
whose POSITIONS mean different things across engines, and the transmitted Jones
-- the observable for a transmissive metasurface -- was UNAVAILABLE from the
standalone ``rcwa`` / ``pmm`` jones functions (only from the ``Stack`` classes).

The fix (1) adds an opt-in ``return_jones_transmission=False`` kwarg to
``rcwa_jones_1d`` / ``rcwa_jones_1d_segments`` that appends the zeroth-order
TRANSMISSION Jones (the transmitted amplitudes are already solved and were
previously squared into ``T_eff`` then discarded -- near-zero extra cost, and
the default keeps the 4-tuple arity), and (2) documents the cross-engine
positional divergence (``result[3]`` = REFLECTION for rcwa/pmm but TRANSMISSION
for berreman; ``result[2]`` = efficiency ARRAY vs a Jones MATRIX) on all three
standalone entry points.

The tests below are INDEPENDENT-ORACLE, not tautologies:

* ``test_..._matches_berreman_uniform_slab`` -- a grating whose ridge == groove
  is a homogeneous anisotropic slab, which the Berreman 4x4 method (a wholly
  different formulation) solves exactly; the new transmission Jones magnitude
  must match ``berreman_jones_1d``'s ``jones_t`` elementwise.  |J_t| is
  reference-plane invariant for a lossless substrate, so this compares physics,
  not phase bookkeeping.
* ``test_..._flux_consistency_normal`` -- at normal incidence the transmitted
  power the solver already reports (``T_eff`` at order 0) must equal
  ``(n_sub/n_sup) * (|J_t[0]|^2 + |J_t[1]|^2)``; a wrong order index, or
  stacking the REFLECTION amplitudes by mistake, would break it.
* ``test_..._arity_backcompat`` -- the default call is byte-unchanged (4-tuple);
  the kwarg appends a (2, 2) complex matrix.

No JAX / GUI dependency -- pure NumPy rcwa + NumPy berreman.
"""
import numpy as np

from lumenairy.elements import berreman_jones_1d
from lumenairy.elements.pmm import pmm_jones_1d
from lumenairy.elements.rcwa import rcwa_jones_1d, rcwa_jones_1d_segments


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _inplane_tensor(exx=2.6, eyy=2.1, exy=0.3, ezz=2.35):
    """A lossless (real-symmetric) IN-PLANE permittivity tensor with x<->y
    cross-coupling (``exy``), diagonal ``ezz`` -- routes the rcwa/berreman
    in-plane path and produces a genuinely 2x2 (cross-polarizing) Jones."""
    return np.array([[exx, exy, 0.0],
                     [exy, eyy, 0.0],
                     [0.0, 0.0, ezz]], dtype=complex)


_P = 0.7e-6          # sub-wavelength period (uniform layer: value is immaterial)
_DEPTH = 0.42e-6
_WL = 0.55e-6
_NSUP = 1.0
_NSUB = 1.5


# --------------------------------------------------------------------------- #
# (1) opt-in kwarg: arity back-compat + shape/dtype of the new observable
# --------------------------------------------------------------------------- #
def test_rcwa_jones_transmission_arity_backcompat():
    eps = _inplane_tensor()
    # DEFAULT: historical 4-tuple, unchanged.
    out4 = rcwa_jones_1d(_P, eps, eps, _NSUB, _NSUP, _DEPTH, 0.5, _WL,
                         n_orders=5)
    assert len(out4) == 4
    orders, R, T, J_r = out4
    assert J_r.shape == (2, 2)

    # OPT-IN: a fifth element, a (2, 2) complex transmission Jones; the first
    # four entries are byte-identical to the default call.
    out5 = rcwa_jones_1d(_P, eps, eps, _NSUB, _NSUP, _DEPTH, 0.5, _WL,
                         n_orders=5, return_jones_transmission=True)
    assert len(out5) == 5
    J_t = out5[4]
    assert J_t.shape == (2, 2)
    assert np.iscomplexobj(J_t)
    for a, b in zip(out4, out5[:4]):
        assert np.array_equal(a, b)

    # the segmented sibling shares the same core and the same opt-in.
    seg = [(0.4, eps), (0.6, eps)]
    s4 = rcwa_jones_1d_segments(_P, seg, _NSUB, _NSUP, _DEPTH, _WL, n_orders=5)
    s5 = rcwa_jones_1d_segments(_P, seg, _NSUB, _NSUP, _DEPTH, _WL, n_orders=5,
                                return_jones_transmission=True)
    assert len(s4) == 4 and len(s5) == 5
    assert s5[4].shape == (2, 2) and np.iscomplexobj(s5[4])


# --------------------------------------------------------------------------- #
# (2) independent oracle: uniform (ridge == groove) slab vs the Berreman 4x4
#     method -- an entirely different formulation solving the identical
#     homogeneous anisotropic layer.
# --------------------------------------------------------------------------- #
def test_rcwa_jones_transmission_matches_berreman_uniform_slab():
    eps = _inplane_tensor()

    # NORMAL incidence, anisotropic (cross-polarizing) uniform slab.
    _o, _R, _T, _Jr, J_t = rcwa_jones_1d(
        _P, eps, eps, _NSUB, _NSUP, _DEPTH, 0.5, _WL,
        angle=0.0, n_orders=6, return_jones_transmission=True)
    _Rb, _Tb, _Jrb, J_tb = berreman_jones_1d(
        [(eps, _DEPTH)], _NSUB, _NSUP, _WL, angle=0.0)
    # magnitude (reference-plane invariant for a lossless substrate); the full
    # 2x2 structure incl. the cross terms must agree with the independent engine.
    assert np.allclose(np.abs(J_t), np.abs(J_tb), atol=1e-4), (
        f"normal: |rcwa J_t|=\n{np.abs(J_t)}\n|berreman J_t|=\n{np.abs(J_tb)}")

    # OBLIQUE incidence, isotropic uniform slab (diagonal Jones): still must
    # match the independent engine (exercises the flux/kx0 bookkeeping).
    eps_iso = 2.25 * np.eye(3, dtype=complex)
    ang = np.deg2rad(22.0)
    _o, _R, _T, _Jr, J_t_ob = rcwa_jones_1d(
        _P, eps_iso, eps_iso, _NSUB, _NSUP, _DEPTH, 0.5, _WL,
        angle=ang, n_orders=6, return_jones_transmission=True)
    _Rb, _Tb, _Jrb, J_tb_ob = berreman_jones_1d(
        [(eps_iso, _DEPTH)], _NSUB, _NSUP, _WL, angle=ang)
    assert np.allclose(np.abs(J_t_ob), np.abs(J_tb_ob), atol=1e-4), (
        f"oblique: |rcwa J_t|=\n{np.abs(J_t_ob)}\n"
        f"|berreman J_t|=\n{np.abs(J_tb_ob)}")


# --------------------------------------------------------------------------- #
# (3) flux self-consistency: the returned T_eff (order 0) is exactly the flux
#     carried by the new transmission Jones -- proves it is the TRANSMITTED (not
#     reflected) amplitude at the right order.
# --------------------------------------------------------------------------- #
def test_rcwa_jones_transmission_flux_consistency_normal():
    eps = _inplane_tensor()
    M = 6
    orders, _R, T, _Jr, J_t = rcwa_jones_1d(
        _P, eps, eps, _NSUB, _NSUP, _DEPTH, 0.5, _WL,
        angle=0.0, n_orders=M, return_jones_transmission=True)
    m0 = int(np.where(orders == 0)[0][0])
    # at normal incidence the order-0 z-flux weight is Re(kz_sub/kz_inc)=n_sub/n_sup
    # and the incident |E|^2 factor is 1, so T_eff[col, 0] == weight*|J_t col|^2.
    weight = _NSUB / _NSUP
    for col in range(2):
        lhs = float(T[col, m0])
        rhs = weight * (abs(J_t[0, col]) ** 2 + abs(J_t[1, col]) ** 2)
        assert abs(lhs - rhs) < 1e-9, (
            f"col={col}: T_eff[0]={lhs} != n_sub/n_sup*|J_t|^2={rhs}")
    # sanity: with real (lossless) media the slab is passive per incident pol.
    assert np.all(T.sum(axis=1) <= 1.0 + 1e-6)


# --------------------------------------------------------------------------- #
# (4) the cross-engine positional divergence is DOCUMENTED on every standalone
#     entry point (guards the documentation half of the fix from silent removal).
# --------------------------------------------------------------------------- #
def test_standalone_jones_return_seam_documented():
    for fn in (rcwa_jones_1d, rcwa_jones_1d_segments, pmm_jones_1d,
               berreman_jones_1d):
        assert fn.__doc__ is not None
        assert "CROSS-ENGINE SEAM" in fn.__doc__, (
            f"{fn.__name__} lost the S5-4 cross-engine seam note")
    # the rcwa entry points advertise the opt-in transmission observable.
    assert "return_jones_transmission" in rcwa_jones_1d.__doc__
    assert "return_jones_transmission" in rcwa_jones_1d_segments.__doc__
