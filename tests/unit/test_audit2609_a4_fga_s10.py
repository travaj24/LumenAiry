"""WP-A4 regression pins for FGA dispatch routing -- audit finding S10.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §2.5 S10: ``apply_real_lens_universal``
routed a TILTED but perfectly collimated, single-valued high-NA plane AT ITS
FOCUS to ``phase_screen``, because both discriminators in that decision are
blind to a global tilt:

* ``_caustic_zone`` scored each exit ray's crossing with the optical AXIS
  (``z = -x_exit / u_exit``).  A global input tilt moves the focus off axis by
  ~``f * theta``, so the axis crossings measure a different quantity and the
  zone came back as junk.
* the final escape hatch asked ``_carrier_residual_rms(E_in, None, ...)``, which
  returns EXACTLY the tilt magnitude for a pure tilt -- so a 0.05 rad (2.9 deg)
  tilted PLANE WAVE read as more "non-collimated" than an R = 3 mm converging
  wavefront.

The oracle used here is NOT the dispatcher: :func:`_ray_focal_zone` traces real
rays with ``raytrace.trace`` (a different code path from the
``ray_transfer_jacobian`` differential fan ``_caustic_zone`` uses), carries them
to the exit vertex with the shared WP-A1 operator, and finds the axial range
over which the bundle's RMS transverse spread ABOUT ITS OWN CENTROID is near its
minimum.  That is the geometric focal zone by definition.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.elements._lens_traced import (
    _NONCOLLIMATED_RESID_THRESH,
    _carrier_residual_rms,
)
from lumenairy.propagators.fga import (
    _ABERRATION_MAX_RAD,
    _SEIDEL_SA_MAX_RAD,
    _caustic_zone,
    _global_mean_tilt,
    _remove_global_tilt,
    _system_na,
    _universal_route,
)

LAM = 1.0e-6
AP = 0.30e-3
N, DX = 192, 2.0e-6
_X1 = (np.arange(N) - N / 2) * DX
X, Y = np.meshgrid(_X1, _X1)
K0 = 2.0 * np.pi / LAM

# f = 1.2 mm biconvex N-BK7, 0.30 mm aperture -> NA 0.1452 (> the 0.12
# na_threshold), focus 1.027 mm past the exit vertex: the audit's fixture.
FAST = la.make_singlet(1.2e-3, -1.2e-3, 0.8e-3, 'N-BK7', aperture=AP)
SLOW = la.make_singlet(100e-3, -100e-3, 3e-3, 'N-BK7', aperture=0.20e-3)
Z_FOCUS = 1.027e-3


def _beam(w=100e-6, tilt=0.0, curv=0.0, dec=0.0):
    E = np.exp(-((X - dec) ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    if tilt:
        E = E * np.exp(1j * K0 * tilt * X)
    if curv:
        E = E * np.exp(1j * K0 * (X ** 2 + Y ** 2) / (2.0 * curv))
    return E


def _ray_focal_zone(prescription, tilt=0.0, decentre=0.0, n=121, frac=0.20):
    """INDEPENDENT ORACLE -- the geometric focal zone from real rays.

    Returns ``(z_near, z_far, z_best)``: the axial range over which the traced
    bundle's RMS transverse spread about its own centroid is within
    ``(1 + frac)`` of its minimum, and the minimising plane.  Uses
    ``raytrace.trace`` + ``TraceResult.at_exit_vertex`` only -- nothing from
    ``fga.py``.
    """
    surfs = rt.surfaces_from_prescription(prescription)
    h = np.linspace(-0.49 * AP, 0.49 * AP, n) + decentre
    L = np.full(n, float(tilt))
    M = np.zeros(n)
    Nz = np.sqrt(np.maximum(1.0 - L * L - M * M, 0.0))
    bundle = rt.RayBundle(x=h.copy(), y=np.zeros(n), z=np.zeros(n),
                          L=L, M=M, N=Nz, wavelength=LAM,
                          alive=np.ones(n, bool), opd=np.zeros(n))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ex = rt.trace(bundle, surfs, LAM).at_exit_vertex()
    ok = np.asarray(ex.alive, bool)
    assert ok.sum() > n // 2, 'premise: most oracle rays must survive'
    x0 = np.asarray(ex.x)[ok]
    u0 = (np.asarray(ex.L) / np.asarray(ex.N))[ok]
    zs = np.linspace(0.2e-3, 3.0e-3, 4000)
    xz = x0[None, :] + u0[None, :] * zs[:, None]
    spread = np.std(xz - xz.mean(axis=1, keepdims=True), axis=1)
    inside = spread <= (1.0 + frac) * float(spread.min())
    return (float(zs[inside][0]), float(zs[inside][-1]),
            float(zs[int(np.argmin(spread))]))


def _route(E, prescription, opd):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _universal_route(E, prescription, LAM, DX, DX, opd, 0.12, 3.0,
                                None, 0.06, _ABERRATION_MAX_RAD,
                                _SEIDEL_SA_MAX_RAD)


# ===========================================================================
# S10a -- _caustic_zone must find the caustic of a TILTED beam
# ===========================================================================

@pytest.mark.parametrize('tilt,decentre,pre_fix_zone_mm', [
    (0.00, 0.0, (1.021056, 1.032698)),      # unchanged by the fix
    (0.02, 0.0, (1.318251, 10.987635)),
    (0.05, 0.0, (2.002278, 11.019568)),
    (0.05, 80e-6, (1.996611, 28.212060)),
])
def test_s10_caustic_zone_brackets_the_real_ray_focus(tilt, decentre,
                                                      pre_fix_zone_mm):
    """The reported zone must contain the plane where real rays actually focus.

    Oracle: :func:`_ray_focal_zone`.  MEASURED best-focus planes (mm, at this
    helper's n = 121 rays / 4000 z samples): 1.021305 (collimated), 1.020605
    (tilt 0.02), 1.018505 (tilt 0.05), 1.019905 (tilt 0.05 + 80 um decentre)
    -- i.e. tilting a collimated beam moves the focus by at most **0.27 %**,
    because the focal DISTANCE of a collimated beam does not care which way
    the beam points.

    PRE-FIX ``_caustic_zone`` reported the zones in ``pre_fix_zone_mm``: the
    three tilted rows are 1.3-28 mm, off by up to a factor **27**, and none of
    them contains the real focus even with the router's own
    ``3 lambda / NA^2`` = 0.142 mm pad.  POST-FIX all four contain it.
    """
    _lo, _hi, z_best = _ray_focal_zone(FAST, tilt=tilt, decentre=decentre)
    assert 0.9e-3 < z_best < 1.2e-3, (
        f'premise: the oracle focus must be near 1.02 mm, got {z_best:.6e} m')
    E = _beam(tilt=tilt, dec=decentre, w=(60e-6 if decentre else 100e-6))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        zone = _caustic_zone(E, DX, FAST, LAM)
    assert zone is not None, 'a converging beam must report a caustic zone'
    pad = 3.0 * LAM / (_system_na(FAST, LAM) ** 2)     # the router's own pad
    assert (zone[0] - pad) <= z_best <= (zone[1] + pad), (
        f'tilt={tilt} dec={decentre}: reported caustic '
        f'[{zone[0]*1e3:.6f}, {zone[1]*1e3:.6f}] mm (+-{pad*1e3:.3f} pad) does '
        f'not contain the real-ray focus at {z_best*1e3:.6f} mm.  Pre-fix this '
        f'row reported [{pre_fix_zone_mm[0]:.6f}, {pre_fix_zone_mm[1]:.6f}] mm.')


def test_s10_caustic_zone_is_tilt_invariant_to_the_physical_shift():
    """The zone must move with the beam's real focus, not with the frame.

    Oracle: the real-ray best focus moves 1.021305 -> 1.018505 mm between
    tilt 0 and tilt 0.05, i.e. **-0.27 %**.  MEASURED zone centres:
    post-fix 1.026877 -> 1.024326 mm, **-0.25 %** -- it tracks the oracle.
    PRE-FIX the centre went 1.026877 -> 6.510923 mm, **+534.1 %**.

    Bar: 3 %.  Ten times the measured 0.25 % (and above the oracle's own
    0.27 %), and two decades below the pre-fix 534 %.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        z0 = _caustic_zone(_beam(), DX, FAST, LAM)
        zt = _caustic_zone(_beam(tilt=0.05), DX, FAST, LAM)
    assert z0 is not None and zt is not None
    c0 = 0.5 * (z0[0] + z0[1])
    ct = 0.5 * (zt[0] + zt[1])
    moved = abs(ct - c0) / c0
    _, _, b0 = _ray_focal_zone(FAST, tilt=0.0)
    _, _, bt = _ray_focal_zone(FAST, tilt=0.05)
    oracle_move = abs(bt - b0) / b0
    assert oracle_move < 0.01, (
        f'premise: the real focus barely moves under tilt, got {oracle_move:.4f}')
    assert moved < 0.03, (
        f'caustic-zone centre moved {moved*100:.2f} % under a 0.05 rad tilt '
        f'while the real focus moved {oracle_move*100:.2f} % '
        f'(pre-fix: +534 %)')


# ===========================================================================
# S10b -- the collimation discriminator must be tilt-invariant
# ===========================================================================

@pytest.mark.parametrize('tilt', [0.005, 0.02, 0.05, 0.1])
def test_s10_a_tilted_plane_wave_reads_as_collimated(tilt):
    """A tilted plane wave IS collimated.

    MEASURED raw ``_carrier_residual_rms``: 5.000000e-03 / 2.000000e-02 /
    5.000000e-02 / 1.000000e-01 at tilt 0.005 / 0.02 / 0.05 / 0.1 -- i.e.
    exactly the tilt, so everything at or above 0.02 rad lost to the
    ``_NONCOLLIMATED_RESID_THRESH = 0.02`` gate.  After
    :func:`_remove_global_tilt`: 1.257379e-09 / 5.029515e-09 / 1.257379e-08 /
    2.514758e-08 -- six to seven decades below the gate.

    Bar: 1e-6.  Two decades above the worst measured residual and four decades
    below the threshold it has to clear.
    """
    E = _beam(w=80e-6, tilt=tilt)
    raw = float(_carrier_residual_rms(E, None, LAM, DX))
    det = float(_carrier_residual_rms(
        _remove_global_tilt(E, DX, DX, LAM), None, LAM, DX))
    assert raw == pytest.approx(tilt, rel=1e-5), (
        f'premise: the raw residual is the tilt itself; got {raw:.6e} for '
        f'tilt {tilt}')
    assert det < 1e-6, (
        f'de-tilted residual {det:.6e} for a pure {tilt} rad tilt '
        f'(measured 1.3e-09 .. 2.5e-08)')
    assert det < _NONCOLLIMATED_RESID_THRESH


def test_s10_detilting_preserves_a_real_divergence_reading():
    """Removing a global tilt must not remove CURVATURE.

    A global linear phase changes the reference direction; it moves no energy
    and changes no local wavefront curvature, so the divergence reading has to
    survive it.  MEASURED on an R = 3 mm converging wavefront: raw
    3.238183e-02, de-tilted 3.238526e-02 (1.1e-04 relative).  With a 0.05 rad
    tilt ADDED the raw reading inflates to 5.956998e-02 -- 84 % high -- and
    de-tilts back to 3.238525900e-02, matching the untilted beam's
    3.238525887e-02 to **4.0e-09 relative** (the conjugate-product tilt
    estimator's own precision once a ramp is added).  Bar 1e-6: 2.5 decades
    above that and 5 decades below the 84 % the raw reading inflates by.
    """
    conv = _beam(w=80e-6, curv=-3e-3)
    conv_t = _beam(w=80e-6, curv=-3e-3, tilt=0.05)
    raw_c = float(_carrier_residual_rms(conv, None, LAM, DX))
    det_c = float(_carrier_residual_rms(
        _remove_global_tilt(conv, DX, DX, LAM), None, LAM, DX))
    raw_ct = float(_carrier_residual_rms(conv_t, None, LAM, DX))
    det_ct = float(_carrier_residual_rms(
        _remove_global_tilt(conv_t, DX, DX, LAM), None, LAM, DX))
    assert raw_c > _NONCOLLIMATED_RESID_THRESH, 'premise: R=3 mm is diverging'
    assert det_c == pytest.approx(raw_c, rel=1e-3), (
        f'de-tilting changed a genuine divergence reading: {raw_c:.6e} -> '
        f'{det_c:.6e}')
    assert det_c > _NONCOLLIMATED_RESID_THRESH, (
        'the de-tilted converging beam must STILL read as non-collimated')
    assert raw_ct > 1.5 * raw_c, (
        f'premise: adding a tilt inflates the raw reading (measured 3.24e-02 '
        f'-> 5.96e-02); got {raw_c:.3e} -> {raw_ct:.3e}')
    assert det_ct == pytest.approx(det_c, rel=1e-6), (
        f'the same physical beam, tilted, must read the same divergence: '
        f'{det_c:.9e} vs {det_ct:.9e}')


def test_s10_global_mean_tilt_recovers_a_known_ramp():
    """:func:`_global_mean_tilt` against a ramp built here."""
    for tx, ty in ((0.03, 0.0), (0.0, -0.02), (0.017, 0.011)):
        E = np.exp(-(X ** 2 + Y ** 2) / (80e-6) ** 2).astype(np.complex128)
        E = E * np.exp(1j * K0 * (tx * X + ty * Y))
        gx, gy = _global_mean_tilt(E, DX, DX, LAM)
        assert gx == pytest.approx(tx, abs=1e-6), f'tx {tx} -> {gx}'
        assert gy == pytest.approx(ty, abs=1e-6), f'ty {ty} -> {gy}'
    # A symmetric converging beam carries no net tilt beyond the half-pixel
    # asymmetry of an even grid (measured 3.333e-04 at R = 3 mm, which is
    # dx/(2R) = 3.333e-04 exactly).
    gx, gy = _global_mean_tilt(_beam(w=80e-6, curv=-3e-3), DX, DX, LAM)
    assert abs(gx) < 1e-3 and abs(gy) < 1e-3, (gx, gy)


# ===========================================================================
# S10c -- the routing decision itself must be frame-invariant
# ===========================================================================

@pytest.mark.parametrize('tilt,pre_fix_route', [
    (0.02, 'traced'),
    (0.05, 'phase_screen'),
])
def test_s10_router_is_invariant_under_a_global_tilt(tilt, pre_fix_route):
    """The same physical beam, viewed in a tilted frame, must get the same
    propagator.

    This is the finding in one line: pre-fix ``apply_real_lens_universal``
    answered ``'fga'`` for a collimated beam at its focus and
    ``{pre_fix_route!r}`` for THE SAME BEAM tilted by {tilt} rad -- two
    different physical models for one piece of physics, chosen by the observer's
    frame.

    The two members are not interchangeable here: measured at this focus, the
    ``phase_screen`` and ``fga`` fields differ by 3.9 um in intensity centroid
    and by a factor 3.9 in intensity-rms width (12.72 um vs 3.27 um), and that
    disagreement is the SAME at tilt 0 (12.61 vs 3.17 um) -- so it is the
    members' own accuracy question, not a tilt artefact, and it is the size of
    what the frame-dependence was buying.
    """
    base = _route(_beam(), FAST, Z_FOCUS)
    assert base == 'fga', (
        f'premise: the untilted high-NA plane at its focus routes to fga '
        f'(the audit accepted this regime); got {base!r}')
    got = _route(_beam(tilt=tilt), FAST, Z_FOCUS)
    assert got == base, (
        f'tilt={tilt}: router returned {got!r} for the tilted beam and '
        f'{base!r} for the same beam untilted (pre-fix: {pre_fix_route!r})')


def test_s10_router_is_invariant_for_a_tilted_decentred_beam():
    """Decentre + tilt together -- the case the pre-fix axis metric got most
    wrong (caustic [1.997, 28.212] mm against a real focus at 1.020 mm)."""
    base = _route(_beam(w=60e-6, dec=80e-6), FAST, Z_FOCUS)
    got = _route(_beam(w=60e-6, dec=80e-6, tilt=0.05), FAST, Z_FOCUS)
    assert base == 'fga', f'premise: the decentred beam routes to fga, got {base!r}'
    assert got == base, (
        f'decentred+tilted routed to {got!r} vs {base!r} untilted '
        f"(pre-fix: 'phase_screen')")


def test_s10_untilted_regimes_are_unchanged():
    """The four regimes the audit found correctly routed must stay that way.

    ``_caustic_zone`` is BIT-IDENTICAL to the pre-fix value on all three
    symmetric fixtures (measured: [98.021603, 98.021702] mm slow / exit,
    [1.021056, 1.032698] mm fast / focus, [0.277843, 3.274212] mm
    multi-valued), because a centred beam's chief ray is the axis.
    """
    assert _route(_beam(w=80e-6), SLOW, 0.0) == 'phase_screen'
    assert _route(_beam(), FAST, Z_FOCUS) == 'fga'
    assert _route(_beam(w=60e-6, dec=80e-6), FAST, Z_FOCUS) == 'fga'
    two_beams = _beam(tilt=0.05) + _beam(tilt=-0.05)
    assert _route(two_beams, FAST, Z_FOCUS) == 'fga'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        z_slow = _caustic_zone(_beam(w=80e-6), DX, SLOW, LAM)
        z_fast = _caustic_zone(_beam(), DX, FAST, LAM)
        z_mv = _caustic_zone(two_beams, DX, FAST, LAM)
    for got, want, nm in ((z_slow, (98.021603e-3, 98.021702e-3), 'slow'),
                          (z_fast, (1.021056e-3, 1.032698e-3), 'fast'),
                          (z_mv, (0.277843e-3, 3.274212e-3), 'multi-valued')):
        assert got[0] == pytest.approx(want[0], rel=1e-6), (nm, got)
        assert got[1] == pytest.approx(want[1], rel=1e-6), (nm, got)


# ===========================================================================
# S10d -- the FGA vector wrapper's Jones normalisation (GUARD, already correct)
# ===========================================================================

def test_s10_fga_vector_normalises_the_jones_components_jointly():
    """GREEN pre- and post-fix -- a guard on the defect class the SIBLING has.

    S10's third sub-item is that ``apply_real_lens_maslov_vector`` passes
    ``normalize_output`` through with its ``'power'`` default and applies it
    INDEPENDENTLY to ``E_x`` and ``E_y``, so any differential transmission,
    vignetting or clipping between the two is normalised away and the output
    polarization ratio is forced back to the input's.  ``apply_real_lens_fga_vector``
    does NOT have that defect -- it applies one joint scale to ``(ex, ey, ez)``
    -- and this pins that it stays so.

    MEASURED on a (2, 48, 48) Jones input with ``P_x/P_y = 4`` exactly, through
    an f = 1.2 mm singlet to 1 mm past the vertex: ``normalize_output='none'``
    and ``'power'`` return ``P_x/P_y = 3.9999998702357877`` and
    ``3.999999870235787`` -- the same to **2.2e-16** (1 ulp) -- while the total
    power is restored to 1.0000000000000002.  The 3.2e-08 departure from the
    input ratio is the s/p diattenuation the system really applies, and it
    SURVIVES the normalisation.
    """
    pytest.importorskip('numba')
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    n, dx = 48, 4.0e-6
    xs = (np.arange(n) - n / 2) * dx
    xx, yy = np.meshgrid(xs, xs)
    g = np.exp(-(xx ** 2 + yy ** 2) / (60e-6) ** 2).astype(np.complex128)
    E_vec = np.stack([g, 0.5 * g])                   # P_x/P_y = 4 exactly
    rx = la.make_singlet(1.2e-3, -1.2e-3, 0.8e-3, 'N-BK7', aperture=0.20e-3)

    def ratio(E):
        return float(np.sum(np.abs(E[0]) ** 2) / np.sum(np.abs(E[1]) ** 2))

    kw = dict(prescription=rx, wavelength=LAM, dx=dx,
              output_plane_distance=1.0e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = np.asarray(apply_real_lens_fga_vector(
            E_vec.copy(), normalize_output='none', **kw))
        nrm = np.asarray(apply_real_lens_fga_vector(
            E_vec.copy(), normalize_output='power', **kw))
    assert ratio(raw) == pytest.approx(ratio(nrm), rel=1e-12), (
        f'normalize_output changed the polarization ratio: {ratio(raw)!r} -> '
        f'{ratio(nrm)!r}; the two Jones components must share ONE scale')
    p_in = float(np.sum(np.abs(E_vec[0]) ** 2 + np.abs(E_vec[1]) ** 2))
    p_out = float(np.sum(np.abs(nrm[0]) ** 2 + np.abs(nrm[1]) ** 2))
    assert p_out == pytest.approx(p_in, rel=1e-9), (
        "normalize_output='power' must restore the total power")
