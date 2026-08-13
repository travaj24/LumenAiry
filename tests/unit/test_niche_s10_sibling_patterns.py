"""S10 -- SIBLING-PATTERN sweep of the traced-carrier stack (2026-07-25).

The traced-carrier campaign (audits ``AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22``,
``AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24``,
``AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24``) closed a set of bugs that were
INSTANCES OF GENERAL PATTERNS.  This file pins the sibling instances found by
re-auditing ``propagators/carrier.py`` + ``elements/**`` for the same patterns:

1. **Paraxial-parabola vs exact-sphere convention mixing at API boundaries.**
   ``carrier_referenced_fit_radius`` fits the PARABOLA, so on an exact-sphere
   field it reads ``1 + NA^2/2`` too flat -- pinned as a documented, measured,
   dx-INVARIANT offset (contrast pattern 3 below, which is dx-DEPENDENT).
2. **inf/NaN sentinels silently disabling logic.**  ``carrier=+/-inf`` (the
   chain's documented collimated launch) used to build an all-NaN exact-sphere
   eikonal, which made the engage test read NaN, skipped the carrier machinery
   with no diagnostic, and emitted "invalid value encountered in subtract" +
   "All-NaN slice encountered" RuntimeWarnings on every such call.  Now the
   analytic ``|R| -> inf`` limit (``W == 0``) is returned: same output, no
   warnings.
3. **Pixel-unit parameters crossing grids.**  ``_fit_carrier_inv``'s default
   central-difference estimator reads ``sin(h)`` where the physics has the
   per-pixel carrier phase step ``h``, so the SAME physical field fits a
   different R on every grid.  ``estimator='increment'`` is dx-exact; the
   ``on_aliased`` guard fires when the beam's own tilt passes the grid
   Nyquist.  ``smooth_sigma_px`` is measured and pinned as report-only.
4. **Degenerate-branch contract violations.**  ``_fourier_upsample_crop`` (the
   helper F-A itself was found in) silently returned an ``n_fine``-shaped array
   built from a 1x1 crop when ``n_crop > env.shape[-1]`` -- the same
   wrong-window class as F-A, now rejected.  And the traced element's ROW-BAND
   assembly (``sag_chunk_rows``, auto-ON at N >= 4096, documented
   BYTE-IDENTICAL) silently downgraded the R7 CUBIC OPL upsample to linear
   whenever a carrier was engaged.
5. **Double-application sharp edges.**  ``window_factor`` is consumed twice on
   the exact final leg; the compounding is measured, documented and pinned
   (exact no-op from wf=4 up, so the shipping default 7.0 is unaffected).

Every fix is either an error-path change, a documented opt-in, or a restoration
of an existing byte-identity contract; the design-121 acceptance is unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_traced import (
    _compute_carrier,
    apply_real_lens_traced,
    apply_real_lens_traced_multi,
    prepare_real_lens_traced,
)
from lumenairy.propagators.carrier import (
    _carrier_fit_alias_fraction,
    _envelope_amp_radius,
    _exact_sphere_eikonal,
    _fourier_upsample_crop,
    _radial_carrier_phase,
    carrier_referenced_exact_focus_readout,
    carrier_referenced_fit_radius,
)

_WL = 1.31e-6


def _singlet(R1, R2, d, glass, ap, name='s'):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _gauss(N, dx, w, R=None, sphere=True):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    A = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    if R is None:
        return A.astype(np.complex128)
    if sphere:
        ph = np.exp(1j * (2 * np.pi / _WL)
                    * _exact_sphere_eikonal((N, N), dx, dx, _WL, R))
    else:
        ph = np.asarray(_radial_carrier_phase((N, N), dx, dx, _WL, R, +1))
    return (A * ph).astype(np.complex128)


# ===========================================================================
# Pattern 2 -- the collimated-carrier all-NaN eikonal sentinel.
# ===========================================================================
def test_compute_carrier_collimated_is_the_plane_wave_not_nan():
    """``carrier=+/-inf`` must give the analytic plane-wave limit ``W == 0``
    with zero gradient -- NOT ``inf - inf = NaN`` over the whole grid, whose
    only effect was to make every reduction of it silently degenerate."""
    N, dx = 64, 4.0e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = _gauss(N, dx, 60e-6)
    for c in (np.inf, -np.inf):
        with warnings.catch_warnings():
            warnings.simplefilter('error')          # no NaN-arithmetic warnings
            W, grad, wfn = _compute_carrier(c, E, _WL, dx, X, Y)
        assert np.isfinite(W).all()
        assert np.array_equal(W, np.zeros((N, N)))
        L, M = grad(np.array([0.0, 1e-4]), np.array([0.0, 2e-4]))
        assert np.array_equal(L, np.zeros(2)) and np.array_equal(M, np.zeros(2))
        assert np.array_equal(wfn(np.array([1e-4]), np.array([2e-4])),
                              np.zeros(1))


def test_traced_collimated_carrier_is_silent_and_equals_carrier_none():
    """The element call with ``carrier=inf`` emits NO RuntimeWarning (it used
    to emit "invalid value encountered in subtract" + "All-NaN slice
    encountered") and stays byte-identical to the plane-wave path it already
    degenerated to."""
    presc = _singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', 1.0e-3)
    N, dx = 128, 4.0e-6
    E = _gauss(N, dx, 150e-6)
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent', parallel_amp=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out_inf = np.asarray(apply_real_lens_traced(E, carrier=np.inf, **kw))
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)], \
        [str(w.message) for w in rec]
    out_none = np.asarray(apply_real_lens_traced(E, carrier=None, **kw))
    assert np.array_equal(out_inf, out_none)


def test_exact_sphere_eikonal_collimated_limit():
    """The carrier-side twin of the same expression: ``R = +/-inf`` (and the
    degenerate ``R == 0``) return zeros instead of an all-NaN grid."""
    for R in (np.inf, -np.inf, 0.0):
        S = _exact_sphere_eikonal((8, 8), 1e-6, 1e-6, _WL, R)
        assert S.shape == (8, 8)
        assert np.array_equal(S, np.zeros((8, 8)))
    # finite R is untouched (spot-check against the closed form)
    S = _exact_sphere_eikonal((4, 4), 1e-3, 1e-3, _WL, -2e-3)
    assert S[2, 2] == pytest.approx(0.0, abs=1e-18)
    assert S[0, 0] == pytest.approx(-(np.sqrt(2 * (2e-3) ** 2 + 4e-6) - 2e-3))


# ===========================================================================
# Pattern 3 -- pixel-unit / grid-dependent curvature estimator.
# ===========================================================================
_FIT_ROWS = [                       # (N, h rad/px, gradient R/R, increment R/R)
    (4096, 0.187, 1.00440, 1.00000),
    (2048, 0.375, 1.01771, 1.00000),
    (1024, 0.749, 1.07276, 1.00000),
]


@pytest.mark.parametrize('N,h_exp,g_exp,i_exp', _FIT_ROWS)
def test_fit_radius_gradient_is_dx_dependent_increment_is_not(N, h_exp, g_exp,
                                                              i_exp):
    """FIXED physical field (Gaussian w=200 um, parabola R=-2 mm, 1.6 mm
    window), pitch swept: the default 'gradient' estimator drifts +0.4% ->
    +7.3% purely with the grid (it reads sin(h) for h), while 'increment' is
    exact to 1e-5 at every pitch."""
    w, R, extent = 200e-6, -2.0e-3, 8 * 200e-6
    dx = extent / N
    E = _gauss(N, dx, w, R, sphere=False)
    h = 2 * np.pi / _WL * dx * w / abs(R)
    assert h == pytest.approx(h_exp, rel=2e-3)
    g = carrier_referenced_fit_radius(E, _WL, dx, on_aliased='silent') / R
    i = carrier_referenced_fit_radius(E, _WL, dx, estimator='increment',
                                      on_aliased='silent') / R
    assert g == pytest.approx(g_exp, rel=2e-4), (N, g)
    assert i == pytest.approx(i_exp, abs=2e-5), (N, i)
    # the drift the grid alone introduces, vs the increment's flatness
    assert abs(g - 1.0) > 20 * abs(i - 1.0) or abs(g - 1.0) > 4e-3


def test_fit_radius_sphere_offset_is_the_convention_not_the_grid():
    """Pattern 1: on an EXACT-SPHERE field the parabolic fit is flat by
    ``1 + NA^2/2`` (0.5% at NA 0.10) -- and with the dx-exact estimator that
    offset is the SAME at every pitch, which is what makes it a convention
    error rather than a sampling error."""
    w, R, extent = 200e-6, -2.0e-3, 8 * 200e-6
    na = w / abs(R)
    vals = []
    for N in (1024, 2048, 4096):
        dx = extent / N
        E = _gauss(N, dx, w, R, sphere=True)
        vals.append(carrier_referenced_fit_radius(
            E, _WL, dx, estimator='increment', on_aliased='silent') / R)
    for v in vals:
        assert v == pytest.approx(1.0 + 0.5 * na ** 2, rel=2e-3), vals
    assert max(vals) - min(vals) < 1e-5, vals        # dx-INVARIANT


def test_fit_radius_alias_guard_fires_only_past_nyquist():
    """The default ``on_aliased='warn'`` guard is silent on a well-sampled
    field and fires once the beam's own carrier tilt reaches half the grid
    Nyquist step, where NEITHER estimator can recover R (measured R_fit/R = 92
    at h ~ 6 rad/px)."""
    w, R, extent = 200e-6, -2.0e-3, 8 * 200e-6
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        carrier_referenced_fit_radius(_gauss(2048, extent / 2048, w, R),
                                      _WL, extent / 2048)
    assert not rec, [str(w_.message) for w_ in rec]
    dx = extent / 128
    with pytest.warns(RuntimeWarning, match='per-pixel phase step'):
        bad = carrier_referenced_fit_radius(_gauss(128, dx, w, R), _WL, dx)
    assert abs(bad / R) > 10                        # meaningless, as warned
    # ... and acknowledging it silences the guard without changing the number
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        same = carrier_referenced_fit_radius(_gauss(128, dx, w, R), _WL, dx,
                                             on_aliased='silent')
    assert not rec
    assert same == bad
    assert _carrier_fit_alias_fraction(_gauss(128, dx, w, R)) > 0.05


def test_fit_radius_argument_validation():
    E = _gauss(32, 4e-6, 40e-6)
    with pytest.raises(ValueError, match='estimator'):
        carrier_referenced_fit_radius(E, _WL, 4e-6, estimator='bogus')
    with pytest.raises(ValueError, match='on_aliased'):
        carrier_referenced_fit_radius(E, _WL, 4e-6, on_aliased='bogus')
    with pytest.raises(ValueError, match='refit_estimator'):
        la.carrier_referenced_aperture(E, np.inf, _WL, 4e-6, radius=5e-5,
                                       refit_carrier=True,
                                       refit_estimator='bogus')


def test_fit_radius_default_estimator_is_bit_identical_to_pre_v5_29():
    """The default path must not move: 'gradient' + the historical call
    signature reproduce the old arithmetic exactly (the reference values are
    the pre-change readings of this same field)."""
    N, dx, w = 512, 4e-6, 160e-6
    R = 80e-3
    E = _gauss(N, dx, w, R, sphere=False)
    assert carrier_referenced_fit_radius(E, _WL, dx) == pytest.approx(
        R, rel=5e-3)
    rx, ry = carrier_referenced_fit_radius(E, _WL, dx, astigmatic=True)
    assert rx == pytest.approx(R, rel=5e-3)
    assert ry == pytest.approx(R, rel=5e-3)
    assert np.isinf(carrier_referenced_fit_radius(_gauss(N, dx, w), _WL, dx))
    assert np.isinf(carrier_referenced_fit_radius(
        _gauss(N, dx, w), _WL, dx, estimator='increment'))


def test_smooth_sigma_px_is_pixel_unit_but_core_tilt_is_dx_invariant():
    """Report-only pin for the F-B suspect ``smooth_sigma_px=4``: on a
    single-mode field the launch tilt at the beam core is dx-invariant to 6
    digits (the smoothing is the documented no-op), while a two-mode field is
    strongly dx-dependent -- and so is the raw quantity, with no convergent
    target for either a pixel or a physical sigma."""
    from lumenairy.elements._lens_traced import _sample_local_tilts
    extent, w, tilt = 1.024e-3, 150e-6, 0.06
    xl = np.linspace(-0.2 * w, 0.2 * w, 5)
    XL, YL = np.meshgrid(xl, xl)
    k = 2 * np.pi / _WL

    def _field(N, dx, two):
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        A = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
        if two:
            return (A * (np.exp(1j * k * tilt * X)
                         + np.exp(-1j * k * tilt * X))).astype(np.complex128)
        return (A * np.exp(1j * k * tilt * X)).astype(np.complex128)

    core = []
    two = []
    for N in (256, 512, 1024):
        dx = extent / N
        L, _M = _sample_local_tilts(_field(N, dx, False), _WL, dx, XL, YL)
        core.append(float(L[2, 2]))
        L2, _ = _sample_local_tilts(_field(N, dx, True), _WL, dx, XL, YL)
        two.append(float(np.sqrt(np.mean(L2 ** 2))))
    for c in core:
        assert c == pytest.approx(tilt, abs=1e-6), core
    assert max(core) - min(core) < 1e-6, core
    # multi-mode: no dx limit (monotone decay toward the collimated launch)
    assert two[0] > 2 * two[-1], two


# ===========================================================================
# Pattern 4 -- degenerate-branch contracts.
# ===========================================================================
def test_fourier_upsample_crop_rejects_a_crop_larger_than_the_grid():
    """F-A's own helper: ``n_crop > env.shape[-1]`` used to return an
    ``n_fine``-shaped array built from a 1x1 crop -- shape invariant satisfied,
    physical window wrong by 66x, values garbage.  Same silent-wrong-pitch
    class as F-A; now an explicit error."""
    env = _gauss(64, 1.0, 8.0)
    for n_crop, n_fine in ((66, 128), (66, 256), (128, 128), (70, 128)):
        with pytest.raises(ValueError, match='n_crop'):
            _fourier_upsample_crop(env, n_crop, n_fine)
    with pytest.raises(ValueError, match='n_crop'):
        _fourier_upsample_crop(env, 1, 64)
    with pytest.raises(ValueError, match='n_fine'):
        _fourier_upsample_crop(env, 32, 1)


@pytest.mark.parametrize('n_crop,n_fine', [(64, 64), (64, 128), (64, 32),
                                           (32, 32), (32, 64), (32, 16)])
def test_fourier_upsample_crop_legal_branches_keep_shape_and_parseval(n_crop,
                                                                      n_fine):
    """The in-contract branches are unchanged: exact shape, and power (value x
    pitch^2) preserved in BOTH directions."""
    env = _gauss(64, 1.0, 8.0)
    out = _fourier_upsample_crop(env, n_crop, n_fine)
    assert out.shape == (n_fine, n_fine)
    c0 = 32 - n_crop // 2
    P_in = float(np.sum(np.abs(env[c0:c0 + n_crop, c0:c0 + n_crop]) ** 2))
    P_out = float(np.sum(np.abs(out) ** 2)) * (n_crop / n_fine) ** 2
    assert P_out == pytest.approx(P_in, rel=2e-3)


def test_row_band_assembly_matches_whole_grid_under_a_carrier():
    """``sag_chunk_rows`` is documented BYTE-IDENTICAL and auto-ON at
    N >= 4096, but the band loop hard-coded ``order=1`` while the whole-grid
    path uses the R7 CUBIC OPL upsample whenever a carrier is engaged -- so
    banding silently disabled R7.  Measured pre-fix: 4.17e-2 max / 2.92e-2 rms
    relative field change at the beam core, power unchanged (pure phase).
    map_coordinates is pointwise in the OUTPUT at any order (the coarse input
    and its prefilter are never banded), so the orders can and must match.
    Tolerance is the FP floor (the library is FP-floor reproducible, not
    bit-reproducible, across band decompositions) -- 13 orders below the
    defect.

    2026-08-13 -- WHY THIS PIN NAMES ``inverse_map=False``, AND WHY THAT IS
    THE HONEST SCOPE RATHER THAN A WORKAROUND.

    The claim above is about ONE algorithm evaluated in two decompositions:
    the OPL upsample's interpolation ORDER must not depend on how the exit
    field is banded.  Since ``FIX_G8_PROBE_2026_08_12`` the whole-grid path
    at the shipped default does not run that upsample at all -- the
    inverse-characteristic evaluator supplies the OPL per exit pixel -- and
    the band path REFUSES the evaluator BY CONSTRUCTION: ``_imap_domain_gate``
    (``_lens_traced.py`` :8466) carries ``not _chunk_assembly``, because "the
    band path exists to never materialise a full-grid float64; handing it one
    would undo the memory fix it is" (:10311).  So at the shipped default the
    two sides are two DIFFERENT inversions, measured 2.1886e-02 apart on the
    carrier arm and 2.4216e-01 on the ``carrier=None`` arm, and no
    interpolation order is being compared.  With the evaluator off they are
    the same algorithm again and both arms read EXACTLY 0.0 -- byte identity,
    not merely the 1e-12 FP floor this pin asks for.

    That structural exclusion is asserted below rather than assumed, so the
    scoping cannot outlive its reason: if the band path ever opens the gate
    (or the whole-grid path ever closes it) this test says so.  See
    ``docs/audits/FIX_RELEASE_FIFTEEN_2026_08_13.md`` S2.2, which also
    records the consequence the gate has for large N (banding is AUTO-ON at
    N >= 4096 on the non-ray-density branch, so those calls keep the
    incumbent coarse-Newton inversion -- the pre-5.35 behaviour, refused
    rather than degraded)."""
    presc = _singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', 1.6e-3)
    N, dx, conj = 256, 4.0e-6, 30.0e-3
    E = _gauss(N, dx, 250e-6, conj, sphere=True)
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
              carrier=conj, on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent', parallel_amp=False, n_workers=1,
              inverse_map=False)

    def _rel(a, b):
        return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))),
                                                 1e-300))

    whole = np.asarray(apply_real_lens_traced(E, sag_chunk_rows=0, **kw))
    for cr in (32, 128, 7):                     # divisor and non-divisor bands
        band = np.asarray(apply_real_lens_traced(E, sag_chunk_rows=cr, **kw))
        assert _rel(band, whole) < 1e-12, (cr, _rel(band, whole))
    # ... and the carrier=None default stays on order-1 for both paths
    kw_nc = dict(kw)
    kw_nc['carrier'] = None
    w0 = np.asarray(apply_real_lens_traced(E, sag_chunk_rows=0, **kw_nc))
    b0 = np.asarray(apply_real_lens_traced(E, sag_chunk_rows=64, **kw_nc))
    assert np.array_equal(w0, b0)
    # ... and the reason the scoping is needed, at the SHIPPED default: the
    # whole-grid call builds the evaluator, the banded call never opens its
    # gate.  Structural, so it is asserted on the gate itself and not on a
    # field difference that a fixture change could quieten.
    kw_d = dict(kw)
    del kw_d['inverse_map']
    rec_w, rec_b = {}, {}
    apply_real_lens_traced(E, sag_chunk_rows=0, _imap_out=rec_w, **kw_d)
    apply_real_lens_traced(E, sag_chunk_rows=32, _imap_out=rec_b, **kw_d)
    assert rec_w['gate_open'] and rec_w['engaged'], rec_w
    assert not rec_b['gate_open'] and not rec_b['engaged'], rec_b


# ===========================================================================
# Pattern 5 -- window_factor applied twice on the exact final leg.
# ===========================================================================
_WF_ROWS = [                       # (wf, 2nd cut / w, win2/win1)
    (2.0, 0.8796, 0.8800),
    (3.0, 0.9866, 0.9867),
    (4.0, 0.9995, 1.0000),
    (7.0, 1.0000, 1.0000),
]


@pytest.mark.parametrize('wf,rho_exp,ratio_exp', _WF_ROWS)
def test_window_factor_double_crop_compounding(wf, rho_exp, ratio_exp):
    """The retrace crops to ``wf * w_entrance`` and the readout crops AGAIN to
    ``wf * w`` re-measured on the truncated field, so the second cut lands at
    ``wf * rho`` beam radii.  Pinned: material below wf ~ 3, an exact no-op
    from wf = 4 up (so the shipping default 7.0 is unaffected)."""
    N, dx, w, R = 1024, 2.0e-6, 200e-6, -2.0e-3
    E = _gauss(N, dx, w, R, sphere=True)
    w0 = _envelope_amp_radius(E, dx, dx)

    def _crop(F, factor):
        n = F.shape[-1]
        ww = _envelope_amp_radius(F, dx, dx)
        win = min(factor * ww, n * dx) if ww > 0 else n * dx
        nc = int(min(max(int(2 * round((win / dx) / 2)), 2), n))
        c0 = n // 2 - nc // 2
        return np.ascontiguousarray(F[c0:c0 + nc, c0:c0 + nc]), nc * dx

    Ec, win1 = _crop(E, wf)                       # _fine_trace_group_exit
    _Ed, win2 = _crop(Ec, wf)                     # exact readout's own crop
    rho = _envelope_amp_radius(Ec, dx, dx) / w0
    assert rho == pytest.approx(rho_exp, abs=2e-4)
    assert win2 / win1 == pytest.approx(ratio_exp, abs=2e-4)
    if wf >= 4.0:
        assert win2 == win1                        # exact no-op


def test_window_factor_double_crop_is_a_measured_no_op_at_the_default():
    """End-to-end through the real readout: at the library default
    ``window_factor=7`` the pre-cropped and un-cropped inputs give the SAME
    focal field, while at wf=2 the compounding costs ~8 EE3 points."""
    N, dx, w, R = 1024, 2.0e-6, 200e-6, -2.0e-3
    E = _gauss(N, dx, w, R, sphere=True)
    P_in = float(np.sum(np.abs(E) ** 2)) * dx * dx

    def _pre(factor):
        ww = _envelope_amp_radius(E, dx, dx)
        win = min(factor * ww, N * dx)
        nc = int(min(max(int(2 * round((win / dx) / 2)), 2), N))
        c0 = N // 2 - nc // 2
        return np.ascontiguousarray(E[c0:c0 + nc, c0:c0 + nc])

    def _ee3(F, factor):
        out = carrier_referenced_exact_focus_readout(
            F, R, abs(R), _WL, dx, dx_out=0.05e-6, N_out=512,
            window_factor=factor, ram_budget=float('inf'))
        I = np.abs(out) ** 2
        iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
        xx = (np.arange(512) - ix) * 0.05e-6
        yy = (np.arange(512) - iy) * 0.05e-6
        rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
        return float(I[rr <= 3e-6].sum()) * (0.05e-6) ** 2 / P_in

    assert _ee3(_pre(7.0), 7.0) == pytest.approx(_ee3(E, 7.0), abs=1e-4)
    single2, double2 = _ee3(E, 2.0), _ee3(_pre(2.0), 2.0)
    assert single2 - double2 > 0.05, (single2, double2)


# ===========================================================================
# Pattern 1 -- the multi / prepared boundaries cannot silently run legacy.
# ===========================================================================
def _multi_case():
    presc = _singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', 1.6e-3)
    N, dx, conj = 128, 4.0e-6, 30.0e-3
    E = _gauss(N, dx, 200e-6, conj, sphere=True)
    return [E], dict(prescription=presc, wavelength=_WL, dx=dx,
                     ray_subsample=8, on_undersample='silent',
                     on_noncollimated='off'), conj


def test_multi_rejects_a_conflicting_preserve_input_phase():
    """It used to POP ``preserve_input_phase`` silently, so ``'remap'``
    returned a field byte-identical to ``True`` (0.0 difference) while the
    direct element call differs by 7.5e-3 -- the caller could not tell."""
    fields, kw, conj = _multi_case()
    for bad in ('remap', False):
        with pytest.raises(ValueError, match='preserve_input_phase'):
            apply_real_lens_traced_multi(fields, carriers=conj,
                                         reuse_prepared=False,
                                         preserve_input_phase=bad,
                                         amplitude_model='ray_density', **kw)
    # the forced value itself is still accepted (no behaviour change)
    a = apply_real_lens_traced_multi(fields, carriers=conj,
                                     reuse_prepared=False,
                                     preserve_input_phase=True, **kw)
    b = apply_real_lens_traced_multi(fields, carriers=conj,
                                     reuse_prepared=False, **kw)
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-20)


def test_multi_reports_the_modes_a_prepared_screen_cannot_hold():
    """``amplitude_model`` / ``fit_radius_beam_factor`` used to raise an opaque
    ``TypeError: prepare_real_lens_traced() got an unexpected keyword
    argument`` on the DEFAULT reuse path while working on the per-emitter one.
    Now: a clear error naming the remedy, and they work with
    ``reuse_prepared=False``."""
    fields, kw, conj = _multi_case()
    for bad in ({'amplitude_model': 'ray_density'},
                {'fit_radius_beam_factor': 2.0}):
        with pytest.raises(ValueError, match='reuse_prepared=False'):
            apply_real_lens_traced_multi(fields, carriers=conj,
                                         reuse_prepared=True, **bad, **kw)
        out = apply_real_lens_traced_multi(fields, carriers=conj,
                                           reuse_prepared=False, **bad, **kw)
        assert np.isfinite(np.asarray(out)).all()
    # 'auto' / ndarray carriers never route to a screen, so they are unaffected
    out = apply_real_lens_traced_multi(fields, carriers='auto',
                                       reuse_prepared=True,
                                       amplitude_model='ray_density', **kw)
    assert np.isfinite(np.asarray(out)).all()


def test_chain_partial_override_of_the_default_triple_reports_its_own_default():
    """The v5.29 default flip left a boundary footgun: the CHAIN supplies
    ``preserve_input_phase='remap'``, which the element rejects without
    ``amplitude_model='ray_density'`` -- so overriding the amplitude ALONE
    produced an element-level error naming an option the caller never wrote.
    The chain now raises first, naming where 'remap' came from and the legacy
    escape hatch."""
    presc = _singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', 1.6e-3)
    N, dx = 128, 4.0e-6
    env = _gauss(N, dx, 200e-6)
    groups = [{'prescription': presc, 'gap_before': 0.0}]
    common = dict(wavelength=_WL, dx=dx, r_in=30e-3, ray_subsample=8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for gs, tkw in ((groups, {'amplitude_model': 'screen'}),
                        ([{'prescription': presc, 'gap_before': 0.0,
                           'traced_kwargs': {'amplitude_model': 'screen'}}],
                         None)):
            with pytest.raises(ValueError, match='CHAIN-DEFAULT'):
                la.propagate_traced_carrier_chain(
                    env, gs, common['wavelength'], common['dx'],
                    r_in=common['r_in'],
                    ray_subsample=common['ray_subsample'],
                    traced_kwargs=tkw)
        # the documented legacy configuration (all three together) is accepted
        res = la.propagate_traced_carrier_chain(
            env, groups, common['wavelength'], common['dx'],
            r_in=common['r_in'], ray_subsample=common['ray_subsample'],
            carrier_reference='parabola',
            traced_kwargs={'amplitude_model': 'screen',
                           'preserve_input_phase': True})
        assert np.isfinite(np.asarray(res.field)).all()
        # ... and so are the shipping defaults
        res = la.propagate_traced_carrier_chain(
            env, groups, common['wavelength'], common['dx'],
            r_in=common['r_in'], ray_subsample=common['ray_subsample'])
        assert np.isfinite(np.asarray(res.field)).all()


def test_prepare_rejects_the_input_dependent_modes_by_name():
    presc = _singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', 1.6e-3)
    base = dict(prescription=presc, wavelength=_WL, dx=4.0e-6, N=64,
                carrier=30.0e-3, on_undersample='silent')
    with pytest.raises(ValueError, match='INPUT-DEPENDENT'):
        prepare_real_lens_traced(amplitude_model='ray_density', **base)
    with pytest.raises(ValueError, match='fit_radius_beam_factor'):
        prepare_real_lens_traced(fit_radius_beam_factor=2.0, **base)
    # the screen-compatible values are accepted (and are the defaults)
    prep = prepare_real_lens_traced(amplitude_model='screen',
                                    fit_radius_beam_factor=None, **base)
    assert prep.screen.shape == (64, 64)
    prep.release()
