"""WP-A3 / audit 2026-09-11: the traced-lens CAUSTIC siblings.

Covers S1 (the exit-vertex P0), T2 (the silent zero field / one-sided energy
tripwire), S9 (the uniform dark-fill restriction), S11 (the KMAH wrap, the dead
turning-point counter, the rebound ``L0``/``M0``) and the new
``caustic='wave'`` ray-to-wave hand-off.

EVERY physics fixture here has a CURVED REAR SURFACE.  That is the point: the
entire multibranch / uniform corpus in this repository uses plano-rear singlets
(``test_niche_k1_kmah_caustic._caustic_prescription``, its ``caustic_fold_ref``
npz, ``test_niche_k4_uniform_caustic``, ``test_v5_21_lens_accuracy_extensions.
_mini_fast_singlet``), where ``sag == 0`` identically and the P0 this file pins
is INVISIBLE -- ``pytest -k "multibranch or uniform or ludwig"`` passed 8/8 on
the unfixed code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.raytrace as rt
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements._lens_traced_multibranch import (
    _shift_clamped,
    _trace_launch_grid,
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements._lens_traced_uniform import (
    _AIRY_TAIL_CELLS,
    _count_interior_turning_points,
)

# Dispersionless model glass so every oracle below is a closed-form number.
# Registered / removed by tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_A3GLASS': lambda wl: 1.5168}

_WL = 1.31e-6


# ===========================================================================
# S1 -- the exit-vertex transfer in the branch-enumeration siblings
# ===========================================================================
def _curved_rear(r1=float('inf'), r2=-25e-3, aperture=20e-3, thick=4e-3):
    """Plano-convex with a CURVED rear face -- the geometry the whole
    multibranch/uniform fixture corpus lacks."""
    return {
        'surfaces': [
            {'radius': r1, 'glass_before': 'air', 'glass_after': '_A3GLASS'},
            {'radius': r2, 'glass_before': '_A3GLASS', 'glass_after': 'air'},
        ],
        'thicknesses': [thick],
        'aperture_diameter': aperture,
    }


def _independent_exit_plane_trace(presc, heights, d_out, wavelength=_WL):
    """Exit-plane (x, OPL) for a collimated meridional fan, computed WITHOUT
    the library's exit-vertex helper.

    Oracle construction: trace with ``rt.trace`` (the surface intersections and
    Snell's law are not what is under test), then apply the vertex transfer by
    hand from the DEFINITION -- a ray at ``(x, z)`` travelling along
    ``(L, N)`` reaches the plane ``z = 0`` after a signed parametric distance
    ``t = -z/N``, accumulating ``n_exit * t`` of optical path -- and then the
    free-space leg ``d_out / N``.  Error floor: float64 round-off on ~1e-2 m of
    OPL, i.e. ~1e-18 m.
    """
    n = len(heights)
    rays = rt.RayBundle(
        x=np.asarray(heights, dtype=float).copy(), y=np.zeros(n),
        z=np.zeros(n), L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
        wavelength=wavelength, alive=np.ones(n, dtype=bool), opd=np.zeros(n))
    surfaces = rt.surfaces_from_prescription(presc)
    ex = rt.trace(rays, surfaces, wavelength).image_rays
    from lumenairy.glass import get_glass_index
    n_exit = get_glass_index(surfaces[-1].glass_after, wavelength)
    t_v = -ex.z / ex.N
    x_v = ex.x + ex.L * t_v
    opl_v = ex.opd + n_exit * t_v
    t_d = float(d_out) / ex.N
    return x_v + ex.L * t_d, opl_v + 1.0 * t_d, ex.alive


@pytest.mark.parametrize('r2,aperture,d_out', [
    (-25e-3, 20e-3, 0.0),          # the audit's worst case, AT the vertex
    (-25e-3, 20e-3, 40e-3),
    (-100e-3, 20e-3, 0.0),
    (-60e-3, 12e-3, 0.0),          # biconvex-ish, a second curvature
])
def test_s1_launch_grid_lands_on_a_plane_not_on_the_sag(r2, aperture, d_out):
    """``_trace_launch_grid`` must evaluate on the OUTPUT PLANE.

    Pre-fix it advanced ``image_rays`` -- which sit at ``z = sag(rho)`` of the
    last surface -- by ``output_plane_distance / N``, so every ray was read at
    ``z = sag(rho) + d``: a ray-DEPENDENT longitudinal position.  Measured
    against the oracle below on R2 = -25 mm over a 20 mm aperture, the
    pre-fix error was 477 um of transverse position and 3501 WAVES of OPD, at
    ``d = 0`` (the default) exactly as much as through focus.

    BAR.  Transverse: 1e-9 m.  The oracle and the library now run the same
    closed-form transfer in float64 over |x| <= 1e-2 m, so their difference is
    round-off (~1e-18 m, measured 1.7e-12 um = 1.7e-18 m); the pre-fix signal
    is 4.8e-4 m.  1e-9 m sits ~9 decades above the floor and ~6 below the
    defect.  OPD: 1e-4 waves against a measured ~1e-11 waves floor and a
    3.5e+03-wave defect -- 7 decades of gap on both sides.
    """
    presc = _curved_rear(r2=r2, aperture=aperture)
    lr = 0.5 * aperture * 0.98
    n_launch = 41
    g = _trace_launch_grid(presc, _WL, lr, n_launch, d_out, 1.0)

    # the meridional row of the launch lattice (y_in = 0 is an exact sample
    # because n_launch is odd)
    j0 = n_launch // 2
    xs = g['xs_in']
    got_x = g['x_out'][:, j0]
    got_opl = g['opl'][:, j0]
    alive = g['alive'][:, j0]

    exp_x, exp_opl, exp_alive = _independent_exit_plane_trace(
        presc, xs, d_out)
    m = alive & exp_alive & np.isfinite(got_x) & np.isfinite(exp_x)
    assert m.sum() > 20, f'only {int(m.sum())} live rays -- fixture broken'

    dx_err = float(np.max(np.abs(got_x[m] - exp_x[m])))
    dopd_waves = float(np.max(np.abs(got_opl[m] - exp_opl[m]))) / _WL
    assert dx_err < 1e-9, f'transverse error {dx_err:.3e} m'
    assert dopd_waves < 1e-4, f'OPD error {dopd_waves:.3e} waves'


def test_s1_fail_before_the_uncorrected_transfer_is_hundreds_of_waves():
    """FAIL-BEFORE, constructed from the running build's own geometry.

    Re-derives what the PRE-FIX construction would have produced -- advancing
    ``image_rays`` (at the sag) by ``d / N`` -- and shows it disagrees with the
    same oracle by a margin the bar above rejects by many decades.  No
    monkeypatching: the pre-fix expression is three lines, written out here.
    """
    presc = _curved_rear(r2=-25e-3, aperture=20e-3)
    heights = np.linspace(0.2e-3, 0.49 * 20e-3, 41)
    n = heights.size
    rays = rt.RayBundle(
        x=heights.copy(), y=np.zeros(n), z=np.zeros(n), L=np.zeros(n),
        M=np.zeros(n), N=np.ones(n), wavelength=_WL,
        alive=np.ones(n, dtype=bool), opd=np.zeros(n))
    surfaces = rt.surfaces_from_prescription(presc)
    ex = rt.trace(rays, surfaces, _WL).image_rays
    # ---- the PRE-FIX construction, verbatim ----
    d_out = 0.0
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = d_out / Nz
    pre_x = ex.x + t * ex.L
    pre_opl = ex.opd + 1.0 * t
    # ---- the oracle ----
    exp_x, exp_opl, _ = _independent_exit_plane_trace(presc, heights, d_out)
    m = ex.alive & np.isfinite(pre_x)
    assert m.sum() > 20
    assert float(np.max(np.abs(pre_x[m] - exp_x[m]))) > 1e-5, (
        'the pre-fix construction should be hundreds of microns off on a '
        'curved rear surface; if it is not, the fixture has a flat rear face '
        'and cannot see this defect class at all')
    assert float(np.max(np.abs(pre_opl[m] - exp_opl[m]))) / _WL > 100.0


def test_s1_multibranch_field_agrees_with_the_single_valued_path_at_the_vertex():
    """END TO END, on the public API, with a CURVED rear surface.

    At ``output_plane_distance = 0`` and away from any caustic, the
    multibranch field IS the single-valued traced field -- one branch per
    pixel, same eikonal, same amplitude model.  Pre-fix the two disagreed by
    1.80 rad rms with max = pi (a fully decorrelated wrapped phase) on a
    curved rear surface, while agreeing to 0.005 rad rms on a flat one.

    BAR.  0.10 rad rms over the bright core.  The flat-rear CONTROL in the
    same audit measures 0.0050 rad rms (the rasteriser's own interpolation
    floor), so the bar sits 20x above the floor and 18x below the 1.80 rad
    defect -- and 1.81 rad is the rms of a UNIFORMLY WRAPPED difference, i.e.
    the defect is at the mathematical ceiling.
    """
    N, dx = 256, 25e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E_in = np.exp(-(X ** 2 + Y ** 2) / (3.5e-3) ** 2).astype(np.complex128)
    presc = _curved_rear(r2=-100e-3, aperture=20e-3, thick=4e-3)
    kw = dict(prescription=presc, wavelength=_WL, dx=dx,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_single = apply_real_lens_traced(E_in, **kw)
        E_mb = apply_real_lens_traced(
            E_in, amplitude_model='ray_density', caustic='multibranch',
            caustic_ray_subsample=2, **kw)
    core = (np.abs(E_single) > 0.2 * np.abs(E_single).max()) & (np.abs(E_mb) > 0)
    assert int(core.sum()) > 5000, 'bright core too small -- fixture broken'
    d = np.angle(E_mb[core] * np.conj(E_single[core]))
    rms = float(np.sqrt(np.mean(d ** 2)))
    assert rms < 0.10, f'multibranch vs single-valued phase rms {rms:.4f} rad'


# ===========================================================================
# T2 -- the silent zero field at an axial focus, and the energy tripwire
# ===========================================================================
def _fast_singlet(aperture=2.0e-3):
    return {
        'surfaces': [
            {'radius': 0.025, 'glass_before': 'air', 'glass_after': '_A3GLASS'},
            {'radius': -0.025, 'glass_before': '_A3GLASS',
             'glass_after': 'air'},
        ],
        'thicknesses': [2e-3],
        'aperture_diameter': aperture,
    }


def _paraxial_bfl(presc, wavelength=_WL):
    """BFL from a real ray trace of THIS prescription (never a constant)."""
    surfaces = rt.surfaces_from_prescription(presc)
    h = 0.05e-3
    rays = rt.RayBundle(x=np.array([h]), y=np.zeros(1), z=np.zeros(1),
                        L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                        wavelength=wavelength, alive=np.ones(1, dtype=bool),
                        opd=np.zeros(1))
    ex = rt.trace(rays, surfaces, wavelength).at_exit_vertex()
    return float(-ex.x[0] / ex.L[0])


def _focus_fixture():
    N, dx = 512, 4e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E_in = np.exp(-(X ** 2 + Y ** 2) / (0.30e-3) ** 2).astype(np.complex128)
    return E_in, N, dx


def test_t2_multibranch_refuses_instead_of_returning_an_empty_field():
    """At an axial point focus the branch sum has no answer -- say so.

    Pre-fix, ``apply_real_lens_traced(caustic='multibranch',
    output_plane_distance=BFL)`` returned an IDENTICALLY ZERO field with ZERO
    warnings: every mapped triangle collapses below ``caustic_min_area_ratio``
    or compresses below one pixel, so the rasteriser covers nothing.  The
    caller could not distinguish "no light" from "this method cannot represent
    this plane".

    This is a DECISION test, not a reading: it asserts that a refusal happens
    and that it names the remedy, with no numeric bar at all.
    """
    E_in, N, dx = _focus_fixture()
    presc = _fast_singlet()
    bfl = _paraxial_bfl(presc)
    assert 20e-3 < bfl < 30e-3, f'BFL {bfl} m outside the expected range'
    with pytest.raises(RuntimeError) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_traced_multibranch(
                E_in, prescription=presc, wavelength=_WL, dx=dx,
                output_plane_distance=bfl, ray_subsample=4)
    msg = str(exc.value)
    for needle in ('identically ZERO', "caustic='wave'", 'min_area_ratio'):
        assert needle in msg, f'missing {needle!r} in:\n{msg}'


def test_t2_energy_tripwire_is_two_sided_and_sees_the_pre_focus_band():
    """The tripwire fired only on a GAIN above 10x, so two regimes were mute.

    Measured in the audit: 4x and 8x of the input aperture power at
    z = 24.70 / 24.74 mm on a 24.83 mm BFL, both silent; and an 11 % LOSS on a
    resolved fold, invisible by construction.  Both arms now warn.

    DECISION test: the plane is chosen by SCANNING toward the focus until the
    reconstructed power leaves the band, so nothing depends on one build's
    exact z.  Hard-fails only if the ladder is exhausted.
    """
    E_in, N, dx = _focus_fixture()
    presc = _fast_singlet()
    bfl = _paraxial_bfl(presc)
    p_in = float((np.abs(E_in) ** 2).sum())
    fired = False
    for frac in (0.960, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995):
        z = frac * bfl
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                E = apply_real_lens_traced_multibranch(
                    E_in, prescription=presc, wavelength=_WL, dx=dx,
                    output_plane_distance=z, ray_subsample=4)
        except RuntimeError:
            continue                      # the total-collapse refusal
        ratio = float((np.abs(E) ** 2).sum()) / p_in
        # the ENERGY arm specifically -- the degenerate-triangle census warns
        # from the same module with a different message, and accepting that
        # one would let a broken energy tripwire pass (VERIFY-A3).
        msgs = [str(w.message) for w in rec
                if 'reconstructed grid power is' in str(w.message)]
        if ratio > 2.0 or ratio < 0.5:
            assert msgs, (
                f'P_out/P_in = {ratio:.4g} at z = {z * 1e3:.4f} mm with NO '
                f'energy warning -- the tripwire is still one-sided or '
                f'mis-tuned (other multibranch warnings seen: '
                + '; '.join(str(w.message)[:60] for w in rec) + ')')
            fired = True
            break
    assert fired, (
        'the scan never left the [0.5, 2.0] energy band, so this fixture '
        'cannot exercise the tripwire; widen the ladder')


# ===========================================================================
# caustic='wave' -- the ray-to-wave hand-off
# ===========================================================================
def test_wave_at_zero_distance_is_the_single_valued_field_bit_for_bit():
    """``caustic='wave'`` with ``output_plane_distance = 0`` must be EXACTLY
    the ordinary traced call: the hand-off is one recursion plus an ASM leg,
    and a zero-length leg is the identity.  Bit-identity, no tolerance."""
    E_in, N, dx = _focus_fixture()
    presc = _fast_singlet()
    kw = dict(prescription=presc, wavelength=_WL, dx=dx,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = apply_real_lens_traced(E_in, **kw)
        b = apply_real_lens_traced(E_in, caustic='wave',
                                   output_plane_distance=0.0, **kw)
    assert np.array_equal(a, b)


def test_wave_reaches_the_axial_focus_the_branch_sum_cannot():
    """The hand-off is exact where the branch sum is empty.

    Gated against an INDEPENDENT oracle built from the other lens model
    entirely -- ``apply_real_lens`` (the analytic split-step; a different
    phase model, not this one) propagated to the same plane with the same
    band-limited ASM.

    BARS.  Energy: |P_out/P_in - 1| < 1e-3.  The ASM is unitary on a
    band-limited field, so the only loss is the aperture mask the traced call
    already applied; measured 1e-6.  Peak and EE(25 um): 5 % relative.  The
    two models' EXIT fields differ by the traced OPD refinement (the audit
    measures 1.06e-01 relative on a fast singlet), so a few per cent at the
    focus is the physics, not the propagator -- and the branch sum reads 0 and
    0, i.e. 100 % off, so the bar has 20x of gap on that side.
    """
    E_in, N, dx = _focus_fixture()
    presc = _fast_singlet()
    bfl = _paraxial_bfl(presc)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    kw = dict(prescription=presc, wavelength=_WL, dx=dx,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent')
    from lumenairy.elements import apply_real_lens
    from lumenairy.propagators.asm import angular_spectrum_propagate
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_wave = apply_real_lens_traced(
            E_in, caustic='wave', output_plane_distance=bfl, **kw)
        E_oracle = angular_spectrum_propagate(
            apply_real_lens(E_in, prescription=presc, wavelength=_WL, dx=dx),
            z=bfl, wavelength=_WL, dx=dx)
    p_in = float((np.abs(E_in) ** 2).sum())
    assert abs(float((np.abs(E_wave) ** 2).sum()) / p_in - 1.0) < 1e-3

    pk_w = float(np.abs(E_wave).max())
    pk_o = float(np.abs(E_oracle).max())
    assert abs(pk_w - pk_o) / pk_o < 0.05, f'peak {pk_w} vs oracle {pk_o}'

    r2 = X ** 2 + Y ** 2
    ee_w = float((np.abs(E_wave[r2 <= (25e-6) ** 2]) ** 2).sum()) / p_in
    ee_o = float((np.abs(E_oracle[r2 <= (25e-6) ** 2]) ** 2).sum()) / p_in
    assert abs(ee_w - ee_o) / ee_o < 0.05, f'EE25 {ee_w} vs oracle {ee_o}'
    # ...and the branch sum is empty at this very plane (the contrast that
    # makes the mode worth having).
    with pytest.raises(RuntimeError):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_traced_multibranch(
                E_in, prescription=presc, wavelength=_WL, dx=dx,
                output_plane_distance=bfl, ray_subsample=4)


def test_wave_tracks_the_multibranch_away_from_any_caustic():
    """Two independent constructions of the same field must agree where BOTH
    are valid -- a short leg past the exit pupil, far from focus.

    BAR.  5 % max relative difference over the bright core.  The two are
    genuinely different methods (a coherent branch sum on a triangulated ray
    map vs a band-limited ASM of the pupil field), so they are not expected to
    agree to round-off; the audit's own comparison of the multibranch against
    an ASM oracle reads 0.206 of peak rms ON a resolved fold, and here -- with
    a single branch everywhere -- the measured difference is 0.017.
    """
    E_in, N, dx = _focus_fixture()
    presc = _fast_singlet()
    kw = dict(prescription=presc, wavelength=_WL, dx=dx,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_w = apply_real_lens_traced(
            E_in, caustic='wave', output_plane_distance=2.0e-3, **kw)
        E_mb = apply_real_lens_traced(
            E_in, caustic='multibranch', amplitude_model='ray_density',
            caustic_ray_subsample=4, output_plane_distance=2.0e-3, **kw)
    core = np.abs(E_mb) > 0.2 * np.abs(E_mb).max()
    assert int(core.sum()) > 1000
    rel = (float(np.abs(E_w[core] - E_mb[core]).max())
           / float(np.abs(E_mb[core]).max()))
    assert rel < 0.05, f'wave vs multibranch max rel diff {rel:.4f}'


def test_wave_rejects_the_configurations_it_cannot_serve():
    """Enumeration + GPU refusal, and ``output_plane_distance`` acceptance."""
    E_in, N, dx = _focus_fixture()
    presc = _fast_singlet()
    kw = dict(prescription=presc, wavelength=_WL, dx=dx,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent')
    with pytest.raises(ValueError, match='caustic must be'):
        apply_real_lens_traced(E_in, caustic='wavelet', **kw)
    with pytest.raises(ValueError, match='CPU path'):
        apply_real_lens_traced(E_in, caustic='wave', use_gpu=True, **kw)
    # ``output_plane_distance != 0`` is refused for every mode that cannot
    # honour it, and accepted for this one.
    with pytest.raises(ValueError, match='output_plane_distance'):
        apply_real_lens_traced(E_in, output_plane_distance=1e-3, **kw)
    # ...and the hand-off's last step is a PROPAGATION, not a multiplier, so it
    # cannot be handed back as an input-independent screen.
    with pytest.raises(ValueError, match='return_screen'):
        apply_real_lens_traced(E_in, caustic='wave', return_screen=True,
                               output_plane_distance=1e-3, **kw)


# ===========================================================================
# S9 -- the uniform dark fill is restricted to the annulus the tail occupies
# ===========================================================================
def test_s9_dark_fill_is_restricted_to_the_airy_annulus():
    """The CFU kernel must not be evaluated on pixels whose value is zero.

    ``Ai(x)`` decays as ``exp(-2/3 x^{3/2})``: at ``_AIRY_TAIL_CELLS`` = 20
    Airy lengths past the caustic it is 4.0e-16 of its value at the ring,
    below the double-precision resolution of the returned field.  Pre-fix the
    fill ran over EVERY pixel outside r_c (measured 4 193 535 of 4 194 304 =
    100.0 % at N = 2048, 25.4 s against the multibranch's 0.85 s).

    This asserts the CONSTANT and the decay it is derived from -- an operation
    count, not a wall time (TESTING_STANDARDS S1: never assert timings).
    """
    from scipy.special import airy
    ai_ring = float(airy(0.0)[0])
    ai_cut = float(airy(float(_AIRY_TAIL_CELLS))[0])
    assert ai_cut / ai_ring < 1e-15, (
        f'Ai({_AIRY_TAIL_CELLS}) / Ai(0) = {ai_cut / ai_ring:.3e} is not below '
        f'double precision -- the dark-fill cut would truncate real tail')
    # ...and the cut sits beyond where the tail stops being representable at
    # any useful level: solve ``Ai(x)/Ai(0) = 1e-12`` on the running build
    # rather than quoting a constant.  (Measured here: x ~ 1.1e+01, against
    # the shipped cut of 20 -- so the constant has ~9 Airy lengths of margin,
    # and the field beyond it is 1e-24 of the ring value.)
    xs = np.linspace(0.0, float(_AIRY_TAIL_CELLS), 4001)
    ratio = np.abs(airy(xs)[0]) / ai_ring
    below = np.nonzero(ratio < 1e-12)[0]
    assert below.size, 'Ai never falls below 1e-12 inside the cut'
    x_1e12 = float(xs[below[0]])
    assert x_1e12 < float(_AIRY_TAIL_CELLS), (
        f'Ai/Ai(0) reaches 1e-12 only at {x_1e12:.2f} Airy lengths, which is '
        f'not inside the {_AIRY_TAIL_CELLS} the fill covers')
    # ...and the cut is not so wide that the restriction buys nothing: the
    # overflow guard already clamps the Airy argument at 50.
    assert float(_AIRY_TAIL_CELLS) < 50.0


# ===========================================================================
# S11 -- the small correctness items
# ===========================================================================
def test_s11_kmah_nan_fill_does_not_wrap_across_the_pupil():
    """``np.roll`` made the launch lattice a TORUS, so a left-edge node's
    "nearest valid neighbour" was the RIGHT-edge node -- 2*launch_radius away,
    on the other side of the pupil and possibly on a different sheet.
    ``_shift_clamped`` repeats the boundary instead.

    Exact, not tolerance-based: compare against the closed-form edge-clamped
    shift written out here.
    """
    a = np.arange(20, dtype=np.int64).reshape(4, 5)
    for axis in (0, 1):
        for sh in (1, -1, 2, -2):
            got = _shift_clamped(a, sh, axis)
            idx = np.arange(a.shape[axis]) - sh
            idx = np.clip(idx, 0, a.shape[axis] - 1)
            exp = np.take(a, idx, axis=axis)
            assert np.array_equal(got, exp), (axis, sh, got, exp)
            # ...and it is NOT the wrapping form wherever the two differ
            rolled = np.roll(a, sh, axis=axis)
            assert not np.array_equal(got, rolled), (
                'edge-clamped and wrapped shifts coincide on this fixture, '
                'so it cannot demonstrate the difference')


@pytest.mark.parametrize('xo,expected', [
    # one clean extremum -> one fold
    (np.array([0.0, 1.0, 2.0, 1.0, 0.0]), 1),
    # the SAME extremum with an isolated exactly-flat slope sample: the raw
    # diff(sign(diff)) form counts '+ -> 0 -> -' as TWO sign changes and
    # misclassifies a fold as a cusp, routing it away from the Airy
    # completion it qualifies for.
    (np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0]), 1),
    (np.array([0.0, 1.0, 2.0, 1.0, 2.0, 3.0]), 2),
    (np.array([0.0, 1.0, 2.0, 3.0]), 0),
])
def test_s11_turning_point_counter_is_robust_to_a_flat_sample(xo, expected):
    """The robust counter was written for this and then never called."""
    assert _count_interior_turning_points(xo) == expected


def test_s11_flat_sample_case_breaks_the_raw_construction():
    """FAIL-BEFORE for the row above: the raw form really does double-count."""
    xo = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    raw = int(np.where(np.diff(np.sign(np.diff(xo))) != 0)[0].size)
    assert raw == 2 and _count_interior_turning_points(xo) == 1
