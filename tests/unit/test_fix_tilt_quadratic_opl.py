"""FIX_TILT_QUADRATIC_OPL (2026-08-11) -- the traced element must hand back an
ABSOLUTE optical path, and a tilted congruence's chief-ray PISTON must match an
exact skew ray trace.

**The defect this pins closed.**  ``apply_real_lens_traced`` referenced its
traced OPL grid to the ray launched at the LAUNCH-LATTICE AXIS
(``opl_grid -= opl_grid[i_axis, i_axis]``, with ``xs_in`` axis-centred) and
never re-applied the removed constant.  That constant is

    Lam(0) = W(0, 0) + a_fit(0, 0) + P(0, 0)

-- the launched congruence's entrance eikonal at the axis (the hammer-H6 and
niche-C6 terms added to ``final.opd``) plus the geometric path of the axis ray
through the group.  On an UNTILTED, UNDECENTRED congruence the axis IS the
chief ray, so dropping it cost only an unobservable global phase.  Under a
:class:`~lumenairy.TiltedCarrier` the axis is *not* the chief ray and both
pieces become functions of the tilt (``W(0,0) = -theta^2 z_c + O(theta^4)``,
``P(0,0) = P_0 + theta^2 T_g + O(theta^4)``), so every traced group silently
subtracted a pure TILT-QUADRATIC piston from its own chief ray.

**Why nothing caught it.**  Every shipped acceptance metric on this path is
PISTON-BLIND -- per-frame ``|F|^2`` on separated DOE frames -- so a defect
worth up to half a wave of inter-beam phase passed every gate.
``docs/audits/PROBE_CHAIN_LADDER_PISTON_2026_08_11.md`` S3.7 measured it on
design 121: the chain reproduced a CONSTANT **95.2 %** of the tilt-quadratic
chief-ray optical path at the first post-DOE group, a 4.8 % deficit held to
0.2 points across a 66x span in angle and INVARIANT to ``N``, ``dx``,
``ray_subsample``, ``fit_radius_beam_factor``, ``newton_poly_order`` and
``newton_max_iters`` -- the signature of a missing TERM, not a discretisation
error.  Through the 6-group relay it reached 0.050-0.416 waves of inter-order
piston error, 5x to 42x outside the lambda/100 bar an aggregating consumer
needs.

**What is pinned here, and why in this shape.**  The observable is the one a
consumer reads -- the phase at the CHIEF RAY of the carrier- and ramp-
referenced envelope -- and the reference is an INDEPENDENT exact skew ray
trace of the same chief ray through the same surfaces, sharing no FFT grid, no
carrier convention and no propagator with the code under test.  Two rules,
both needed:

* an ENVELOPE rule -- the residual against the ray oracle stays inside
  lambda/100 (2*pi/100 rad), the bar the sum-at-aperture consumer states; and
* a COMPARATIVE rule -- the residual must NOT scale as ``theta^2``.  This is
  the rule with teeth: the fixture's genuine, irreducible residual (a
  diffraction/aberration difference a geometric ray trace cannot carry) is
  tilt-INDEPENDENT, while ANY re-introduction of this defect class is
  quadratic in the tilt by construction.  Measured on the two-group fixture
  below, the pre-fix residual grows 0.741 -> 2.924 rad for a 2x tilt (3.95x,
  i.e. ``theta^2`` to two digits) where the fixed version moves
  0.01704 -> 0.01709 rad (0.3 %).

Fail-before, on this file's own fixtures with the library at ``main``
(v5.34.0, commit 755ad99):

===========================================  ============  ============
measurement                                  pre-fix       fixed
===========================================  ============  ============
one group, chief-ray residual @ 6 mrad       -2.078 rad    -3.95e-07 rad
one group, chief-ray residual @ 24 mrad      -1.808 rad    -4.45e-05 rad
one group, tilt-quadratic ratio @ 24 mrad     0.8547        0.9999964
two groups, residual @ 6 / 12 mrad           +0.741 /      +0.01704 /
                                             +2.924 rad    +0.01709 rad
element, on-axis phase vs k0 * traced OPL    +2.850 rad    -1.88e-08 rad
===========================================  ============  ============
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import TiltedCarrier, get_glass_index
from lumenairy.propagators.carrier import (
    _tilt_ramp,
    carrier_referenced_envelope,
)
from lumenairy.raytrace import Surface, make_ray, trace
from lumenairy.raytrace.trace import surfaces_from_prescription

# ---------------------------------------------------------------------------
# A synthetic stand-in -- no .zmx, no data file.  Flat entrance + ``K = -n^2``
# conic exit is the exact Fermat singlet for a collimated bundle, so the
# element's own wavefront error is negligible and what the oracle comparison
# measures is the PISTON BOOKKEEPING rather than the fit.
# ---------------------------------------------------------------------------
_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GLASS = 'N-BK7'
_W0 = 1.0e-3                 # collimated 1/e^2 beam radius at the launch plane
_APER = 5.0e-3
_THICK = 2.0e-3
_NGRID = 512
_EXTENT = 10.0e-3            # holds the optical axis AND the tilted beam
_DX = _EXTENT / _NGRID
_RS = 4                      # ray_subsample (the chain default)

#: lambda/100 in radians -- the bar an aggregating (sum-at-aperture) consumer
#: states for inter-beam phase, and the bar the probe scored against.
_BAR_PISTON = 2.0 * np.pi / 100.0
#: The tilt-quadratic optical path must be reproduced to this RELATIVE
#: accuracy.  The defect read 0.952 on design 121 and 0.855 / 0.348 here.
_BAR_RATIO = 1e-3
#: A residual that MOVES with the tilt is the defect class returning.  A
#: ``theta^2`` term changes by 4x over the 2x tilt step below; the fixture's
#: own tilt-independent residual moves by 5e-05 rad.
_BAR_TILT_DRIFT = 2.0 * np.pi / 1000.0


def _n_glass() -> float:
    return float(get_glass_index(_GLASS, _WL))


def _surface(radius, glass_before, glass_after, conic):
    return {'radius': float(radius), 'glass_before': glass_before,
            'glass_after': glass_after, 'conic': float(conic),
            'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _prescription(f):
    n = _n_glass()
    return {'name': f'singlet f={f * 1e3:.0f}mm', 'aperture_diameter': _APER,
            'thicknesses': [_THICK],
            'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                         _surface(-(n - 1.0) * f, _GLASS, 'air', -n * n)]}


def _groups_one():
    """One group behind one free leg -- the configuration the audit localised
    the per-group deficit on (``PL_NGROUPS=1``)."""
    return [{'prescription': _prescription(40.0e-3), 'gap_before': 25.0e-3}]


def _groups_two():
    """Two groups -- pins that the piston ACCUMULATES correctly, which is the
    property design 121's six-group relay turns into 0.4 waves."""
    return [{'prescription': _prescription(0.200), 'gap_before': 25.0e-3},
            {'prescription': _prescription(0.200), 'gap_before': 30.0e-3}]


# ---------------------------------------------------------------------------
# The oracle: an exact skew ray trace of the chief ray, launch plane -> the
# last group's back vertex.  ``opd`` is the geometric optical path.
# ---------------------------------------------------------------------------
def _oracle_surfaces(groups):
    surfs = [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                     glass_before='air', glass_after='air', is_mirror=False,
                     thickness=float(groups[0]['gap_before']),
                     label='launch')]
    for j, g in enumerate(groups):
        sf = [dataclasses.replace(s, semi_diameter=np.inf)
              for s in surfaces_from_prescription(g['prescription'])]
        sf[-1] = dataclasses.replace(
            sf[-1], thickness=(float(groups[j + 1]['gap_before'])
                               if j + 1 < len(groups) else 0.0))
        surfs.extend(sf)
    surfs.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                         glass_before='air', glass_after='air',
                         is_mirror=False, thickness=0.0, label='img'))
    return surfs


def _oracle_opl(groups, L, M):
    im = trace(make_ray(0.0, 0.0, L, M, wavelength=_WL),
               _oracle_surfaces(groups), _WL, output_filter='last').image_rays
    assert bool(np.asarray(im.alive).ravel()[0]), 'oracle chief ray vignetted'
    return float(np.asarray(im.opd).ravel()[0])


# ---------------------------------------------------------------------------
# The measurement: the chain's own chief-ray piston.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _chain_piston(groups_key, L, M):
    """Phase at the CHIEF RAY of the exit envelope, with the carrier sphere
    AND the tilt ramp divided out -- so it is the accumulated optical path and
    nothing else.  Both references are identically zero at the chief ray, and
    the chain re-centres its envelope on the chief ray, so this is the grid's
    centre pixel."""
    key = (groups_key, L, M)
    if key in _CACHE:
        return _CACHE[key]
    groups = _groups_one() if groups_key == 'one' else _groups_two()
    x = (np.arange(_NGRID) - _NGRID / 2) * _DX
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / _W0 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env0, groups, _WL, _DX,
            r_in=TiltedCarrier(np.inf, L, M, 0.0, 0.0),
            ray_subsample=_RS, n_workers=1, final_distance=0.0)
    field = np.asarray(res.field)
    env = carrier_referenced_envelope(field, res.R, _WL, res.dx)
    tilted = [s for s in res.stages if s.get('L_out') is not None]
    if tilted:
        ramp = _tilt_ramp(field.shape, res.dx, _WL,
                          float(tilted[-1]['L_out']),
                          float(tilted[-1]['M_out']), 0.0, 0.0, -1)
        if ramp is not None:
            env = env * ramp
    out = float(np.angle(env[_NGRID // 2, _NGRID // 2]))
    _CACHE[key] = out
    return out


def _residual(groups_key, theta):
    """``(oracle, wrapped residual)`` in radians for one tilt, both referenced
    to the same chain / oracle at zero tilt."""
    groups = _groups_one() if groups_key == 'one' else _groups_two()
    oracle = _K0 * (_oracle_opl(groups, theta, 0.0)
                    - _oracle_opl(groups, 0.0, 0.0))
    measured = float(np.angle(np.exp(
        1j * (_chain_piston(groups_key, theta, 0.0)
              - _chain_piston(groups_key, 0.0, 0.0)))))
    return oracle, float(np.angle(np.exp(1j * (measured - oracle))))


# ---------------------------------------------------------------------------
# 1.  ENVELOPE RULE -- the chief-ray piston against the exact ray trace.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('theta', [6.0e-3, 12.0e-3, 24.0e-3])
def test_tilted_chief_ray_piston_matches_an_exact_skew_ray_trace(theta):
    oracle, resid = _residual('one', theta)
    assert abs(oracle) > 0.5, (
        f'fixture is not exercising the tilt-quadratic path: the oracle reads '
        f'only {oracle:.3e} rad at {theta * 1e3:.1f} mrad')
    assert abs(resid) < _BAR_PISTON, (
        f'tilted chief-ray piston is {resid:+.4e} rad '
        f'({resid / (2 * np.pi):+.5f} waves) away from the exact skew ray '
        f'trace at {theta * 1e3:.1f} mrad -- outside lambda/100 = '
        f'{_BAR_PISTON:.4e} rad.  Pre-fix this read -2.078 rad at 6 mrad; see '
        f'docs/audits/FIX_TILT_QUADRATIC_OPL_2026_08_11.md.')


# ---------------------------------------------------------------------------
# 2.  THE FINGERPRINT -- the tilt-quadratic optical path, in full.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('theta', [12.0e-3, 24.0e-3])
def test_the_tilt_quadratic_optical_path_is_reproduced_in_full(theta):
    """The defect's signature was a CONSTANT FRACTION of the ``theta^2``
    optical path -- 0.952 on design 121, 0.348 / 0.855 on this fixture.  A
    fraction is what a missing TERM looks like, so the fraction is what is
    pinned."""
    oracle, resid = _residual('one', theta)
    ratio = (oracle + resid) / oracle
    assert abs(ratio - 1.0) < _BAR_RATIO, (
        f'the chain reproduces {ratio * 100:.4f} % of the tilt-quadratic '
        f'chief-ray optical path at {theta * 1e3:.1f} mrad (want 100 % to '
        f'{_BAR_RATIO:.0e}).  A fixed FRACTION over a span of tilt is a '
        f'missing or mis-scaled theta^2 TERM, not a discretisation error.')


# ---------------------------------------------------------------------------
# 3.  COMPARATIVE RULE -- the residual must not be quadratic in the tilt.
# ---------------------------------------------------------------------------
def test_the_multi_group_piston_residual_does_not_scale_with_tilt_squared():
    """Two groups, so the per-group bookkeeping has to ACCUMULATE.

    The fixture keeps an irreducible residual -- the beam sits at a different
    transverse position in the second group at each tilt, so the diffraction /
    aberration difference a geometric ray trace cannot carry is real.  That
    residual is tilt-INDEPENDENT.  A dropped or mis-scaled ``theta^2`` piston
    is not: it grows 4x over this 2x tilt step (measured pre-fix: 0.741 ->
    2.924 rad, 3.95x)."""
    o_lo, r_lo = _residual('two', 6.0e-3)
    o_hi, r_hi = _residual('two', 12.0e-3)
    assert o_hi / o_lo > 3.5, (
        'fixture check: the oracle path itself must be ~quadratic in the '
        f'tilt, saw a ratio of {o_hi / o_lo:.3f}')
    assert abs(r_lo) < _BAR_PISTON and abs(r_hi) < _BAR_PISTON, (
        f'two-group chief-ray piston residuals {r_lo:+.4e} / {r_hi:+.4e} rad '
        f'against lambda/100 = {_BAR_PISTON:.4e} rad')
    assert abs(r_hi - r_lo) < _BAR_TILT_DRIFT, (
        f'the piston residual MOVED with the tilt: {r_lo:+.4e} rad at 6 mrad '
        f'-> {r_hi:+.4e} rad at 12 mrad (drift {r_hi - r_lo:+.4e} rad against '
        f'{_BAR_TILT_DRIFT:.4e}).  A residual that scales with theta^2 is the '
        f'tilt-quadratic OPL defect, whatever its absolute size.')


# ---------------------------------------------------------------------------
# 4.  THE MECHANISM ITSELF -- the element hands back an ABSOLUTE path.
# ---------------------------------------------------------------------------
def test_traced_element_returns_the_absolute_optical_path():
    """The tilt-dependence above is a symptom; this is the cause, measured on
    an on-axis, untilted call where the answer is a single ray trace.

    ``apply_real_lens_traced`` must return a field whose ON-AXIS phase is
    ``k0 *`` the traced axial optical path, vertex plane to vertex plane.
    Before the fix it returned ``0`` there for every element -- the OPL grid's
    axis reference was subtracted and never restored -- which is exactly the
    constant that becomes tilt-dependent under a ``TiltedCarrier``.  The
    prescription is slow (f/40) so the traced fit's own residual is 3e-09
    waves and the assertion is about the piston and nothing else."""
    presc = _prescription(0.200)
    x = (np.arange(_NGRID) - _NGRID / 2) * _DX
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / _W0 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens_traced(
            env, prescription=presc, wavelength=_WL, dx=_DX,
            ray_subsample=2, carrier=np.inf,
            amplitude_model='ray_density', preserve_input_phase='remap')
    surfs = [dataclasses.replace(s, semi_diameter=np.inf)
             for s in surfaces_from_prescription(presc)]
    surfs[-1] = dataclasses.replace(surfs[-1], thickness=0.0)
    surfs.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                         glass_before='air', glass_after='air',
                         is_mirror=False, thickness=0.0, label='vertex'))
    axial = float(np.asarray(trace(
        make_ray(0.0, 0.0, 0.0, 0.0, wavelength=_WL), surfs, _WL,
        output_filter='last').image_rays.opd).ravel()[0])
    assert axial > 2.0e-3, 'fixture check: the axial path must be non-trivial'
    got = float(np.angle(out[_NGRID // 2, _NGRID // 2]))
    want = float(np.angle(np.exp(1j * (_K0 * axial))))
    resid = float(np.angle(np.exp(1j * (got - want))))
    assert abs(resid) < 1.0e-3, (
        f'apply_real_lens_traced returned on-axis phase {got:+.9f} rad where '
        f'the traced axial optical path {axial * 1e3:.9f} mm demands '
        f'{want:+.9f} rad (residual {resid:+.4e} rad).  The element must '
        f'carry its ABSOLUTE optical path -- the chain composes legs on it, '
        f'and its own docstrings promise it in four places.')
