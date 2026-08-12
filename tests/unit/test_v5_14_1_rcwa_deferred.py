"""v5.14.1 -- the RCWA roadmap's deferred items (audit 2026-06-10).

* **LEV-1** planar TE/TM decouple: the 1-D solver runs the whole pipeline at
  size N instead of 2N (Ky = 0 makes P@Q exactly block-diagonal); ~x4-8 at
  large N, per-order equal to the 2N (Jones) machinery, and it CURED two of
  the three pinned large-period blow-ups (see test_rcwa.py).
* **LEV-2** diagonal-aware propagation star: algebraically identical to
  starring the pure-propagation S-matrix, without the ~10-zgemm chain.
* **GAP3** per-layer ``formulation='li'`` in ``RCWAStack.add_layer`` -- the
  Li-1997 sequential rule per isotropic patterned layer.
* **GAP5** ``RCWAStack.solve_vs_wavelength`` + dispersive ``wl -> value``
  callables for every material slot.
* **GAP1** sheared (parallelogram) sidewalls in ``add_tapered_grating``.
* **GAP2** out-of-plane (full-3x3) tensors in ``rcwa_jones_2d`` via the
  pointwise ezz-Schur fold + the generalized forward/backward cascade,
  validated against the conical Berreman 4x4 oracle at machine precision.
* P3 hygiene: non-finite indices raise with a named culprit; non-finite
  energy totals raise; the 1-D sweep wrapper rejects JAX inputs loudly.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_1d,
    rcwa_jones_2d,
)
from lumenairy.elements.rcwa._core import (
    _C,
    _propagation_smatrix,
    _propagation_smatrix_general,
    _propagation_star,
    _propagation_star_general,
    _redheffer_star,
    uniaxial_tensor,
)

from .test_v5_14_1_rcwa_audit_fixes import _berreman_jones_conical

_WL = 0.633e-6


# --------------------------------------------------------------------------- #
# LEV-2: diagonal-aware propagation star
# --------------------------------------------------------------------------- #

def test_propagation_star_identity():
    rng = np.random.default_rng(11)
    n = 101
    S = tuple((rng.standard_normal((n, n))
               + 1j * rng.standard_normal((n, n))).astype(_C) for _ in range(4))
    lam = (0.3 * rng.standard_normal(n)
           + 1j * rng.standard_normal(n)).astype(_C)
    lam = np.where(lam.real < 0, -lam, lam)
    A = _redheffer_star(S, _propagation_smatrix(lam, 1.7))
    B = _propagation_star(S, lam, 1.7)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(A, B)) < 1e-12
    lam_b = -lam + 0.01j
    A = _redheffer_star(S, _propagation_smatrix_general(lam, lam_b, 0.9))
    B = _propagation_star_general(S, lam, lam_b, 0.9)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(A, B)) < 1e-12


# --------------------------------------------------------------------------- #
# LEV-1: the decoupled planar path equals the 2N (Jones) machinery
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pol,row", [("te", 1), ("tm", 0)])
@pytest.mark.parametrize("ang", [0.0, 0.35])
def test_decoupled_1d_matches_2n_jones_machinery(pol, row, ang):
    """A scalar-isotropic tensor through rcwa_jones_1d runs the FULL 2N
    machinery; the decoupled N-path of rcwa_efficiency_1d must reproduce it
    per-order (TE = incident-Ey row, TM = incident-Ex row)."""
    eps = np.diag([4.41, 4.41, 4.41]).astype(complex)
    o2, R2, T2, _ = rcwa_jones_1d(1.5e-6, eps, np.eye(3, dtype=complex),
                                  1.45, 1.0, 0.3e-6, 0.5, _WL, angle=ang,
                                  n_orders=15)
    o1, R1, T1 = rcwa_efficiency_1d(1.5e-6, 2.1, 1.0, 1.45, 1.0, 0.3e-6, 0.5,
                                    _WL, angle=ang, polarization=pol,
                                    n_orders=15)
    assert np.max(np.abs(np.asarray(R1) - np.asarray(R2)[row])) < 1e-10
    assert np.max(np.abs(np.asarray(T1) - np.asarray(T2)[row])) < 1e-10


def test_decoupled_1d_lossless_energy_and_metal():
    for pol in ("te", "tm"):
        o, R, T = rcwa_efficiency_1d(1.5e-6, 2.1, 1.0, 1.45, 1.0, 0.3e-6, 0.5,
                                     _WL, polarization=pol, n_orders=41)
        assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-9
    n_metal = np.sqrt(-8.97 + 1.08j)
    o, R, T = rcwa_efficiency_1d(1.0e-6, n_metal, 1.0, 1.45, 1.0, 0.4e-6, 0.5,
                                 _WL, polarization="tm", n_orders=24,
                                 formulation="li")
    A = 1.0 - float(R.sum() + T.sum())
    assert abs(A - 0.0866) < 1e-2          # the audited converged band


# --------------------------------------------------------------------------- #
# GAP3: per-layer formulation in RCWAStack
# --------------------------------------------------------------------------- #

def _stripe_1d_cell(S=256):
    cell = np.full((S, 1), 1.0 + 0j)
    cell[:S // 2, 0] = -8.97 + 1.08j
    return cell


def test_stack_li_layer_matches_direct_1d():
    """A 1-D stack metal layer with formulation='li' reproduces the direct
    1-D 'li' solver per-order (TM = row 0); 'laurent' stays the default."""
    n_metal = np.sqrt(-8.97 + 1.08j)
    for form, tol in (("laurent", 1e-3), ("li", 2e-4)):
        st = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, n_orders=8)
        st.add_layer(0.4e-6, eps_cell=_stripe_1d_cell(), formulation=form)
        o, R, T = st.set_source(_WL, theta=0.0).solve().efficiencies()
        o1, R1, T1 = rcwa_efficiency_1d(1.0e-6, n_metal, 1.0, 1.45, 1.0,
                                        0.4e-6, 0.5, _WL, polarization="tm",
                                        n_orders=8, formulation=form)
        i0 = len(o1) // 2
        d = max(max(abs(float(R[0][j]) - float(R1[i0 + int(m)])),
                    abs(float(T[0][j]) - float(T1[i0 + int(m)])))
                for j, m in enumerate(np.asarray(o)))
        assert d < tol, f"{form}: {d}"
    # and the li layer's absorptance is the accurate one (laurent ~2x high
    # at this truncation on the audited stripe)
    A = {}
    for form in ("laurent", "li"):
        st = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, n_orders=8)
        st.add_layer(0.4e-6, eps_cell=_stripe_1d_cell(), formulation=form)
        o, R, T = st.set_source(_WL, theta=0.0).solve().efficiencies()
        A[form] = 1.0 - float(R[0].sum() + T[0].sum())
    assert abs(A["li"] - 0.0866) < 2e-2
    assert A["laurent"] > A["li"] + 5e-2


def test_stack_li_layer_matches_direct_2d():
    cell = np.full((64, 64), 1.0 + 0j)
    cell[:32, :32] = -8.97 + 1.08j
    st = RCWAStack(0.6e-6, period_y=0.6e-6, n_superstrate=1.0,
                   n_substrate=1.45, n_orders=4, n_orders_y=4)
    st.add_layer(0.2e-6, eps_cell=cell, formulation="li")
    o, R, T = st.set_source(_WL, theta=0.0).solve().efficiencies()
    o2, R2, T2 = rcwa_efficiency_2d(0.6e-6, 0.6e-6, cell, 1.45, 1.0, 0.2e-6,
                                    _WL, polarization="tm", n_orders_x=4,
                                    n_orders_y=4, formulation="li")
    pos = {tuple(x): j for j, x in enumerate(np.asarray(o2))}
    d = max(abs(float(R[1][j]) - float(R2[pos[tuple(x)]]))
            for j, x in enumerate(np.asarray(o)))
    assert d < 1e-11


def test_stack_formulation_validation():
    st = RCWAStack(1.0e-6, n_orders=4)
    with pytest.raises(ValueError, match="formulation"):
        st.add_layer(0.1e-6, eps=2.25, formulation="bogus")
    with pytest.raises(ValueError, match="isotropic patterned"):
        st.add_layer(0.1e-6, eps=2.25, formulation="li")


# --------------------------------------------------------------------------- #
# GAP5: stack wavelength sweep + dispersion
# --------------------------------------------------------------------------- #

def test_stack_sweep_matches_per_wavelength():
    st = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, n_orders=6)
    st.add_layer(0.4e-6, eps_cell=_stripe_1d_cell())
    st.add_layer(0.1e-6, eps=2.25)
    st.set_source(0.6e-6)
    wls = (0.55e-6, 0.7e-6)
    o, R, T, J = st.solve_vs_wavelength(wls)
    assert R.shape == (2, 2, len(o)) and J.shape == (2, 2, 2)
    for i, w in enumerate(wls):
        res = st.set_source(float(w)).solve()
        o1, R1, T1 = res.efficiencies()
        assert np.max(np.abs(R[i] - np.asarray(R1))) == 0.0
        assert np.max(np.abs(T[i] - np.asarray(T1))) == 0.0
        assert np.max(np.abs(J[i] - np.asarray(res.jones_reflection()))) == 0.0


def test_stack_dispersive_materials():
    st = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, n_orders=5)
    st.add_layer(0.3e-6, eps=lambda wl: (1.45 + 0.05 * (wl / 1e-6)) ** 2)
    loss = lambda wl: 10.0 * (wl / 1e-6)               # noqa: E731
    st.add_layer(0.4e-6, eps_cell=lambda wl: np.where(
        np.arange(64)[:, None] < 32, 4.0 + loss(wl) * 1j, 1.0).astype(complex))
    st.set_source(0.6e-6)
    with pytest.raises(ValueError, match="DISPERSIVE"):
        st.solve()
    wls = (0.5e-6, 0.7e-6)
    o, R, T, J = st.solve_vs_wavelength(wls)
    a = [1.0 - float(R[i, 0].sum() + T[i, 0].sum()) for i in range(2)]
    assert 0.0 < a[0] < a[1] < 1.0      # loss grows with wavelength
    # hand-materialized control at wls[1]
    st2 = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, n_orders=5)
    st2.add_layer(0.3e-6, eps=(1.45 + 0.05 * 0.7) ** 2)
    st2.add_layer(0.4e-6, eps_cell=np.where(
        np.arange(64)[:, None] < 32, 4.0 + 7.0j, 1.0).astype(complex))
    o2, R2, T2 = st2.set_source(0.7e-6).solve().efficiencies()
    assert np.max(np.abs(R[1] - np.asarray(R2))) < 1e-12


# --------------------------------------------------------------------------- #
# GAP1: sheared sidewalls
# --------------------------------------------------------------------------- #

#: Relative bar for the two SYMMETRY residuals of the sheared grating, scored
#: against the diffraction order each is a symmetry OF rather than against an
#: absolute number (``FIX_RUNNER_PINS_2026_08_12`` S6).  Both residuals are
#: round-off magnitudes out of a 13x13 cascade through an 8-slice staircase, so
#: they move with the BLAS reduction order, while the order itself is O(0.3):
#: the absolute ``1e-12`` this pair carried was a bar calibrated on one runner,
#: and CI grazed it at 1.4306e-12 on the ubuntu py3.12 and py3.13 shards
#: (``|0.33449837513783104 - 0.3344983751392616|``).  Measured worst residual,
#: relative to the order:
#:
#:     mount / runner              vertical sym    mirror shear    relative
#:     Windows py3.14 OpenBLAS     0.0             4.4498e-13      1.3303e-12
#:     WSL py3.12 OpenBLAS         5.5511e-17      1.5654e-14      4.6799e-14
#:     CI ubuntu py3.12 / py3.13   --              1.4306e-12      4.2777e-12
#:
#: (Both mounts read identically at OPENBLAS_NUM_THREADS 1 / 2 / default, so
#: the axis here is the LAPACK build, not the reduction width.)  1e-9 is an
#: ENVELOPE over all three with 234x headroom above the worst, and it sits
#: 5.0e6x BELOW the smallest shear error the check has to be able to see: one
#: ``n_x = 256`` raster pixel, ``d(shear) = 3.9e-3``, already moves this order
#: by 1.6753e-03 = 5.0e-3 relative (identical on both mounts).
_SHEAR_SYM_REL = 1e-9


def _shear_build(shear):
    st = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, n_orders=6)
    st.add_tapered_grating(0.4e-6, eps_ridge=4.0, eps_groove=1.0,
                           duty_bottom=0.5, n_slices=8, shear=shear)
    return st.set_source(_WL).solve().efficiencies()


def test_tapered_grating_shear_relative_envelope_still_sees_one_raster_pixel():
    """THE FAIL-BEFORE for the relative envelope above.

    ``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S6.  Re-deriving an absolute
    bar as a relative one is only safe if the relative one still sees the
    defect class the absolute one was there for, so the smallest shear error
    the builder can even represent is injected: ONE ``n_x = 256`` raster pixel,
    ``d(shear) = 1/256``.  Anything smaller does not move a pixel under
    ``raster='hard'`` and is not a distinguishable geometry at all (measured:
    ``d(shear)`` of 1e-9 through 1e-4 leave the answer bit-identical), so this
    is the true floor of the check's job.

    It reads 1.6753e-03 = 5.0e-3 relative, identical on both mounts -- 5.0e6x
    above the envelope, and 1.2e9x above this box's round-off residual.  The
    band between "round-off" and "a real shear error" is nine decades wide,
    which is why the bar never needed to be absolute."""
    o, _Rs, Ts = _shear_build(0.3)
    ip1 = int(np.where(np.asarray(o) == 1)[0][0])
    im1 = int(np.where(np.asarray(o) == -1)[0][0])
    o, _Rm, Tm = _shear_build(-(0.3 + 1.0 / 256.0))
    mir = abs(float(Tm[0, im1]) - float(Ts[0, ip1]))
    rel = mir / abs(float(Ts[0, ip1]))
    assert rel > 1e-3, (
        f'one raster pixel of shear error moves the mirrored order by only '
        f'{rel:.4e} relative (measured 5.0e-3), so the {_SHEAR_SYM_REL:g} '
        f'envelope would not separate it from round-off')
    with pytest.raises(AssertionError, match='mirror the asymmetry'):
        assert mir < _SHEAR_SYM_REL * abs(float(Ts[0, ip1])), (
            'mirroring the shear must mirror the asymmetry exactly')


def test_tapered_grating_shear():
    build = _shear_build
    o, R0, T0 = build(0.0)
    o, Rs, Ts = build(0.3)
    ip1 = int(np.where(np.asarray(o) == 1)[0][0])
    im1 = int(np.where(np.asarray(o) == -1)[0][0])
    # lossless staircase conserves energy with or without shear (1e-7: one
    # CI runner's BLAS closes this 8-slice staircase at 4.5e-8)
    assert abs(float(R0.sum() + T0.sum()) - 2.0) < 1e-7
    assert abs(float(Rs.sum() + Ts.sum()) - 2.0) < 1e-7
    # the vertical grating is +/-1 symmetric; the sheared one is NOT.  Both
    # symmetry claims are scored RELATIVE to the order they are about
    # (_SHEAR_SYM_REL) -- see that constant for the cross-runner table.
    sym0 = abs(float(T0[0, ip1] - T0[0, im1]))
    assert sym0 < _SHEAR_SYM_REL * abs(float(T0[0, ip1])), (
        f'the UNSHEARED grating is +/-1 symmetric by construction, but the '
        f'residual {sym0:.4e} is {sym0 / abs(float(T0[0, ip1])):.4e} of the '
        f'order itself ({float(T0[0, ip1]):.6f}), above the {_SHEAR_SYM_REL:g} '
        f'relative envelope')
    assert abs(float(Ts[0, ip1] - Ts[0, im1])) > 1e-2
    # mirror shear mirrors the asymmetry
    o, Rm, Tm = build(-0.3)
    mir = abs(float(Tm[0, im1]) - float(Ts[0, ip1]))
    assert mir < _SHEAR_SYM_REL * abs(float(Ts[0, ip1])), (
        f'mirroring the shear must mirror the asymmetry exactly, but '
        f'T(-shear)[-1] and T(+shear)[+1] differ by {mir:.4e} = '
        f'{mir / abs(float(Ts[0, ip1])):.4e} of the order '
        f'({float(Ts[0, ip1]):.6f}), above the {_SHEAR_SYM_REL:g} relative '
        f'envelope.  One n_x raster pixel of shear error reads 5.0e-3 '
        f'relative, so a residual in this band is round-off, not geometry')


# --------------------------------------------------------------------------- #
# GAP2: 2-D out-of-plane tensors
# --------------------------------------------------------------------------- #

_OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(30.0), phi=np.deg2rad(20.0))
_OOP_LOSSY = uniaxial_tensor(1.5 + 0.05j, 1.7 + 0.08j, np.deg2rad(35.0),
                             phi=np.deg2rad(40.0))


@pytest.mark.parametrize("name,eps,th_d,ph_d", [
    ("oop-planar", _OOP, 20.0, 0.0),
    ("oop-conical", _OOP, 20.0, 25.0),
    ("oop-normal", _OOP, 0.0, 0.0),
    ("oop-lossy-conical", _OOP_LOSSY, 20.0, 25.0),
])
def test_jones_2d_oop_uniform_vs_conical_berreman(name, eps, th_d, ph_d):
    """Uniform full-3x3 cells against the independent (fixed) conical
    Berreman 4x4 oracle -- machine precision (measured <= 2.2e-15)."""
    S = 16
    cell = np.broadcast_to(eps, (S, S, 3, 3)).copy()
    th, ph = np.deg2rad(th_d), np.deg2rad(ph_d)
    o, R, T, J = rcwa_jones_2d(0.4e-6, 0.4e-6, cell, 1.5, 1.0, 0.4e-6, _WL,
                               theta=th, phi=ph, n_orders_x=2, n_orders_y=2)
    Jr_o, _kx, _ky = _oracle_jones(eps, th, ph)
    assert float(np.max(np.abs(np.asarray(J) - Jr_o))) < 1e-12
    if eps is _OOP:                         # lossless: exact closure
        for row in (0, 1):
            assert abs(float(R[row].sum() + T[row].sum()) - 1.0) < 1e-9


def _oracle_jones(eps, th, ph):
    Jr, Jt, Kx, Ky = _berreman_jones_conical(eps, 1.0, 1.5, 0.4e-6, _WL,
                                             th, ph)
    return Jr, Kx, Ky


def test_jones_2d_oop_patterned_vs_1d():
    """A y-uniform PATTERNED out-of-plane cell against the 1-D OOP Jones
    solver (the library's validated 1-D generalized cascade)."""
    S, M = 64, 6
    cell = np.empty((S, 1, 3, 3), complex)
    cell[:S // 2, 0] = _OOP
    cell[S // 2:, 0] = 2.1 * np.eye(3)
    o2, R2, T2, J2 = rcwa_jones_2d(1.0e-6, 1.0e-6, np.broadcast_to(
        cell, (S, max(1, 4 * 2 + 1), 3, 3)).copy(), 1.45, 1.0, 0.3e-6, _WL,
        n_orders_x=M, n_orders_y=2)
    o1, R1, T1, J1 = rcwa_jones_1d(1.0e-6, _OOP,
                                   2.1 * np.eye(3, dtype=complex), 1.45, 1.0,
                                   0.3e-6, 0.5, _WL, n_orders=M)
    m0 = o2[:, 1] == 0
    pos = {int(m): j for j, m in enumerate(o2[m0, 0])}
    i0 = len(o1) // 2
    dmax = max(max(abs(float(R2[r][m0][pos[m]]) - float(R1[r][i0 + m])),
                   abs(float(T2[r][m0][pos[m]]) - float(T1[r][i0 + m])))
               for r in (0, 1) for m in range(-M, M + 1))
    leak = float(R2[:, ~m0].sum() + T2[:, ~m0].sum())
    assert dmax < 5e-4          # pixel-sampling floor (S=64)
    assert leak < 1e-10
    assert np.max(np.abs(np.asarray(J2) - np.asarray(J1))) < 1e-3


def test_jones_2d_oop_zero_ezz_raises():
    S = 8
    bad = np.broadcast_to(_OOP, (S, S, 3, 3)).copy()
    bad[..., 2, 2] = 0.0
    with pytest.raises(ValueError, match="e_zz"):
        rcwa_jones_2d(0.4e-6, 0.4e-6, bad, 1.5, 1.0, 0.4e-6, _WL,
                      n_orders_x=2, n_orders_y=2)


# --------------------------------------------------------------------------- #
# P3 hygiene
# --------------------------------------------------------------------------- #

def test_nonfinite_index_named_culprit():
    with pytest.raises(ValueError, match="n_substrate is not finite"):
        rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, np.nan, 1.0, 0.3e-6, 0.5,
                           1.55e-6, n_orders=5)


def test_sweep_rejects_jax_inputs():
    jnp = pytest.importorskip("jax.numpy")
    from lumenairy.elements.rcwa import rcwa_efficiency_vs_wavelength
    with pytest.raises(NotImplementedError, match="vmap"):
        rcwa_efficiency_vs_wavelength(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5,
                                      jnp.asarray([0.6e-6, 0.7e-6]), order=0)
