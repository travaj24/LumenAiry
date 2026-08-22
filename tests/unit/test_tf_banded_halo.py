"""Byte-identity of the row-banded TANGENT-FACET family
(``surface_model='tangent_facet'`` and ``'tangent_facet_remap'``).

v5.36.0 / the remap rung shipped both models WHOLE-GRID ONLY, and both builds
refused the band loop explicitly rather than approximating into it: route 3
differentiates a gradient twice, so an exact band needs a 3-row sag halo AND a
halo on the persistent momentum accumulator, and the remap resamples the field
itself.  ``sag_chunk_rows`` was pinned INERT with ``np.array_equal`` in both
suites precisely so a later change could not let the models into the band loop
silently.  This file is the byte-identity argument that change has to arrive
with.

THE HALOS, DERIVED (the module comment above ``_tf_sl`` carries the same
derivation, backwards from what a band must produce):

* route 3 -- the accumulator is ``-grad(opd)``, ``opd`` is built from five
  gradients of quantities that are themselves built from ``grad sag``, so the
  band needs ``sag`` on a **3-row** halo and the momentum on a **2-row** one;
* the remap rung -- the accumulator is ``p_out`` in CLOSED FORM (R3), so the
  deepest chain is only the (R4)/(R5) Hessian: ``sag`` on a **2-row** halo and
  the momentum on **none at all**.  The rung that costs more memory needs the
  narrower halo, which is (R3) showing up in the arithmetic;
* the gap transport takes one gradient of the accumulator: a **1-row** halo;
* the flat-face skip ``bool(xp.any(sag))`` is a reduction, evaluated band-wise
  with a short circuit -- it is observable, because it is what keeps the
  accumulator a pair of PYTHON floats through a leading plate;
* the ``all(ok)`` reduction that decides whether the thin-screen fallback runs
  is whole-grid in the whole-grid path and band-local here.  Those agree
  element for element because ``xp.where`` on an all-True mask returns the left
  operand exactly AND at its own dtype, and the tangent-facet screen's dtype
  (``result_type(sag, p)``) is never narrower than the thin screen's
  (``sag``'s) -- pinned below on float32 geometry, where a promotion would
  show;
* the numexpr-vs-numpy phase-screen choice stays gated on the WHOLE ``E.size``,
  the ``_slant_narrow_chunk`` precedent, because numexpr's ``exp`` differs from
  numpy's in the last bit.

WHAT IS NOT BANDED, AND IS TESTED AS A REFUSAL RATHER THAN AS A FEATURE: the
remap rung's second half, the pull-back that resamples the field at ``x + W``.
Its halo is the WALK -- a length, not a row count, measured at 93.0 / 67.3 /
31.7 um on a design-121-like doublet, i.e. 15 / 27 / 50 rows at dx = 8 / 4 /
2 um -- and three of its steps are globally coupled on top of that
(``spline_filter``'s IIR, the whole-grid least-squares moments of the
demodulating eikonal, and ``min(det)``, which must refuse the CALL).  The
screen half bands; the apply half runs whole-grid and says so through
``progress``.

Determinism note: the models propagate through glass with an FFT-based ASM
between surfaces, so byte-equality across separate calls needs FFT plan
determinism -- the fixture pins auto-promote off, matching the sibling suites.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as LR
from lumenairy.elements._lens_real import apply_real_lens

LAM = 1.31e-6
MODELS = ('tangent_facet', 'tangent_facet_remap')


@pytest.fixture(autouse=True)
def _deterministic_fft():
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


# ---------------------------------------------------------------------------
# prescriptions
# ---------------------------------------------------------------------------
def _surf(radius, glass_before, glass_after, **extra):
    d = {'radius': radius, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass_before, 'glass_after': glass_after}
    d.update(extra)
    return d


def _biconvex(R=12.6e-3, glass='N-SSK2', **extra0):
    """Both faces powered, so the momentum accumulator is a live grid at
    surface 1 and the gap transport has something to resample."""
    return {'surfaces': [_surf(R, 'AIR', glass, **extra0),
                         _surf(-R, glass, 'AIR')],
            'thicknesses': [4.0e-3], 'aperture_diameter': 3.0e-3}


def _planoconvex():
    """One FLAT face: the flat-face skip has to fire on surface 1, band-wise."""
    return {'surfaces': [_surf(19.6e-3, 'AIR', 'N-SSK2'),
                         _surf(np.inf, 'N-SSK2', 'AIR')],
            'thicknesses': [4.0e-3], 'aperture_diameter': 3.0e-3}


def _leading_plate():
    """A flat plate in FRONT of the powered pair: the accumulator must stay a
    pair of Python floats through it and promote mid-prescription."""
    return {'surfaces': [_surf(np.inf, 'AIR', 'N-BK7'),
                         _surf(np.inf, 'N-BK7', 'AIR'),
                         _surf(19.6e-3, 'AIR', 'N-SSK2'),
                         _surf(-27.4e-3, 'N-SSK2', 'AIR')],
            'thicknesses': [1.0e-3, 2.0e-3, 4.0e-3],
            'aperture_diameter': 3.0e-3}


def _oblate():
    """k = +4: the conic domain edge falls INSIDE the grid, so the NaN
    sentinel annulus crosses every band boundary."""
    return {'surfaces': [_surf(6.0e-3, 'AIR', 'N-BK7', conic=4.0),
                         _surf(np.inf, 'N-BK7', 'AIR')],
            'thicknesses': [3.0e-3]}


def _folding(n=129, amp=12.0e-6, period=40.0e-6):
    """BUILD_TF_REMAP S4's ENGINEERED folding prescription: a lens interior
    does not fold, so the departure is injected as a ``form_error``
    corrugation (``sag_callable`` is the displaced path's hook, not this
    model's -- that was the first attempt and it is recorded as refuted)."""
    dx = period * 8 / n
    ax = (np.arange(n) - n // 2) * dx
    fe = amp * np.cos(2 * np.pi * ax[None, :] / period) + 0.0 * ax[:, None]
    return dx, {'surfaces': [
        _surf(np.inf, 'AIR', 'N-SF57', form_error=fe),
        _surf(np.inf, 'N-SF57', 'AIR')],
        'thicknesses': [200.0e-6]}


def _field(N, dx, dy=None, w=None, dtype=np.complex128):
    dy = dy if dy is not None else dx
    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(x, y)
    w = w if w is not None else 0.22 * N * min(dx, dy)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(dtype)


def _run(E, presc, dx, model, cr, dy=None, **kw):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        out = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                              dy=dy, surface_model=model,
                              sag_chunk_rows=cr, **kw)
    return out, tuple(sorted(str(w.message)[:80] for w in wl))


def _same(E, presc, dx, model, crs, dy=None, **kw):
    """Whole grid (``cr=0``) against every band size in ``crs``, on the
    returned field's BYTES, its dtype and the warning set."""
    ref, ref_w = _run(E, presc, dx, model, 0, dy=dy, **kw)
    for cr in crs:
        got, got_w = _run(E, presc, dx, model, cr, dy=dy, **kw)
        assert got.dtype == ref.dtype, (model, cr)
        assert got_w == ref_w, (model, cr, ref_w, got_w)
        assert np.array_equal(ref.view(np.uint8), got.view(np.uint8)), (
            model, cr, float(np.max(np.abs(ref - got))))
    return ref


# ---------------------------------------------------------------------------
# 1.  THE CORE MATRIX
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('presc_name', ['biconvex', 'planoconvex',
                                        'leading_plate'])
@pytest.mark.parametrize('carrier_name', ['none', 'tilt', 'sphere'])
def test_core_matrix_byte_identical(model, presc_name, carrier_name):
    """{route 3, remap} x {both faces powered, one flat face, leading plate}
    x {no carrier, collimated tilt, finite-radius sphere}, at band sizes 1, 3,
    N and N+7 -- ``cr=1`` being the one where EVERY row is simultaneously a
    band interior and a band boundary, so no gradient stencil is ever purely
    interior to a band."""
    N, dx = 96, 25e-6
    presc = {'biconvex': _biconvex, 'planoconvex': _planoconvex,
             'leading_plate': _leading_plate}[presc_name]()
    kw = {}
    if carrier_name == 'tilt':
        kw['carrier'] = la.TiltedCarrier(np.inf, 0.05, 0.0)
    elif carrier_name == 'sphere':
        kw['carrier'] = la.TiltedCarrier(-40e-3, 0.03, 0.02)
    _same(_field(N, dx), presc, dx, model, [1, 3, N, N + 7], **kw)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('opt', [
    {'fresnel': True},
    {'fresnel': True, 'absorption': True},
    {'bandlimit': True},
    {'absorption': True},
])
def test_the_orthogonal_option_axes_are_byte_identical(model, opt):
    """The tangent-facet screen REPLACES the paraxial OPD but leaves the
    Fresnel amplitude, the TIR mask and the propagator alone, so the band has
    to reach both the ``_narrow_chunk`` gate (no fresnel) and the
    ``_slant_narrow_chunk`` one (fresnel on) and reproduce the whole grid from
    both.  ``slant_correction`` is not in this list because it is REFUSED with
    these models -- both replace the same facet coefficient."""
    N, dx = 96, 25e-6
    _same(_field(N, dx), _biconvex(), dx, model, [1, 3, N, N + 7],
          carrier=la.TiltedCarrier(np.inf, 0.05, 0.0), **opt)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('cdt', [np.complex128, np.complex64])
@pytest.mark.parametrize('sdt', [np.float64, np.float32])
def test_dtype_matrix_byte_identical(model, cdt, sdt):
    """The dtype sweep is not decoration.  ``BUILD_OBL_BANDED_HALO`` S6.1's one
    real wrong answer was a scalar accumulator promoted to a float32 ARRAY at
    band 0, which under NEP 50 silently dropped a momentum triangle from
    float64 to float32 -- invisible at the float64 default and worth 5e-6 of
    field at ``sag_dtype='float32'``.  The same hazard exists here (the
    tangent-facet accumulator is a pair of Python floats until the first
    powered surface), and float32 geometry is also where a thin-screen-fallback
    dtype promotion would show."""
    N, dx = 96, 25e-6
    _same(_field(N, dx, dtype=cdt), _biconvex(), dx, model, [1, 3, N + 7],
          carrier=la.TiltedCarrier(np.inf, 0.05, 0.0), sag_dtype=sdt)


@pytest.mark.parametrize('model', MODELS)
def test_odd_n_and_anisotropic_grid_byte_identical(model):
    """Odd N (so the last band is short) and ``dx != dy`` (so the two gradient
    spacings differ and a transposed halo would show)."""
    for N, dx, dy in ((65, 31e-6, 31e-6), (65, 18e-6, 25e-6),
                      (72, 25e-6, 19e-6)):
        _same(_field(N, dx, dy), _biconvex(), dx, model, [1, 3, 7, N, N + 7],
              dy=dy, carrier=la.TiltedCarrier(np.inf, 0.05, 0.0))


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('cr', [1, 3, 7, 17])
def test_nan_sentinel_at_band_boundaries(model, cr):
    """An oblate (k = +4) conic whose domain edge falls inside the grid: the
    sag carries a NaN sentinel on an annulus that crosses EVERY band boundary
    at these band sizes.  The zeroing has to be applied to the HALO, not the
    band, or the halo rows feed NaN back into the interior stencil."""
    N, dx = 96, 25e-6
    _same(_field(N, dx), _oblate(), dx, model, [cr],
          carrier=la.TiltedCarrier(np.inf, 0.05, 0.0))


@pytest.mark.parametrize('model', MODELS)
def test_clear_aperture_and_stop_byte_identical(model):
    """A per-surface clear aperture and the aperture stop are applied PER BAND
    on the fresnel band path and after the loop on the whole-grid one; for the
    remap rung they must also land BEFORE the walk, because the aperture is a
    property of the surface."""
    N, dx = 96, 25e-6
    E = _field(N, dx)
    _same(E, _biconvex(clear_aperture=2.2e-3), dx, model, [1, 3, N + 7],
          fresnel=True)
    presc = _biconvex()
    presc['aperture_diameter'] = 2.0e-3
    presc['stop_index'] = 1
    _same(E, presc, dx, model, [1, 3, N + 7], fresnel=True)
    _same(E, presc, dx, model, [1, 3, N + 7])


@pytest.mark.parametrize('model', MODELS)
def test_a_face_that_falls_through_mid_prescription(model):
    """A DECENTERED face disqualifies both band gates, so that surface runs
    whole-grid while its neighbours band.  The accumulator has to hand back and
    forth across the boundary -- and the gap transport has to reach the
    identical step from either side, which is why the whole-grid path routes
    through ``_tf_gap_transport`` too."""
    N, dx = 96, 25e-6
    presc = _leading_plate()
    presc['surfaces'][2]['decenter'] = (12e-6, -8e-6)
    _same(_field(N, dx), presc, dx, model, [1, 3, N + 7],
          carrier=la.TiltedCarrier(np.inf, 0.05, 0.0))


# ---------------------------------------------------------------------------
# 2.  THE FOLD GUARD FIRES IDENTICALLY
# ---------------------------------------------------------------------------
def test_the_fold_guard_refuses_identically_banded_and_whole():
    """The remap's guard is a whole-grid ``min(det)`` reduction, and that is
    the point: it must refuse the CALL, not a band.  On the engineered folding
    prescription the banded path must raise the SAME refusal at the SAME
    surface as the whole-grid path -- a band that ran because it had not seen
    the folding row would be exactly the silent wrong answer the model exists
    to refuse."""
    dx, presc = _folding()
    E = _field(129, dx)
    with pytest.raises(ValueError) as ref:
        _run(E, presc, dx, 'tangent_facet_remap', 0)
    for cr in (1, 3, 7, 129, 136):
        with pytest.raises(ValueError) as got:
            _run(E, presc, dx, 'tangent_facet_remap', cr)
        assert 'REFUSES at surface' in str(got.value)
        assert str(got.value) == str(ref.value), cr


def test_below_the_fold_the_corrugation_still_runs_byte_identically():
    """The control for the test above: at an amplitude the guard passes, the
    banded and whole-grid fields are byte-identical -- so the refusal above is
    the guard firing, not the corrugation breaking the band."""
    dx, presc = _folding(amp=1.6e-6)
    _same(_field(129, dx), presc, dx, 'tangent_facet_remap', [1, 7, 129, 136])
    _same(_field(129, dx), presc, dx, 'tangent_facet', [1, 7, 129, 136])


# ---------------------------------------------------------------------------
# 3.  THE BAND LOOP IS ACTUALLY TAKEN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('model', MODELS)
def test_the_band_loop_is_taken_not_silently_skipped(model):
    """A byte-identity test passes trivially if the banded arm quietly falls
    back to the whole grid.  Two structural checks that it does not: the
    full-grid meshgrids are never built, and ``np.gradient`` is called more
    often than the whole-grid path calls it (once per band per level rather
    than once per level)."""
    N, dx = 96, 25e-6
    E = _field(N, dx)
    presc = _biconvex()

    calls = {'grad': 0, 'mesh': 0}
    real_grad, real_mesh = np.gradient, np.meshgrid

    def spy_grad(*a, **k):
        calls['grad'] += 1
        return real_grad(*a, **k)

    def spy_mesh(*a, **k):
        calls['mesh'] += 1
        return real_mesh(*a, **k)

    np.gradient, np.meshgrid = spy_grad, spy_mesh
    try:
        calls['grad'] = calls['mesh'] = 0
        _run(E, presc, dx, model, 0)
        whole = dict(calls)
        calls['grad'] = calls['mesh'] = 0
        _run(E, presc, dx, model, 8)
        band = dict(calls)
    finally:
        np.gradient, np.meshgrid = real_grad, real_mesh

    assert band['grad'] > whole['grad'], (whole, band)
    assert band['mesh'] < whole['mesh'], (whole, band)


@pytest.mark.parametrize('model', MODELS)
def test_the_flat_face_skip_keeps_the_accumulator_scalar(model):
    """The flat-face reduction is OBSERVABLE, not an optimisation: it is what
    keeps the momentum accumulator a pair of PYTHON floats through a leading
    plate.  Banded, it becomes a band scan with a short circuit, and this pins
    that the scan reaches the same answer -- the gap transport behind the two
    plate faces must still be handed scalars."""
    seen = []
    real = LR._tangent_facet_transport

    def spy(px, py, t, n_gap, dx, dy, xp):
        seen.append((getattr(px, 'ndim', 0), getattr(py, 'ndim', 0)))
        return real(px, py, t, n_gap, dx, dy, xp)

    LR._tangent_facet_transport = spy
    try:
        for cr in (0, 1, 5, 96):
            seen.clear()
            _run(_field(96, 25e-6), _leading_plate(), 25e-6, model, cr)
            # gaps 0 and 1 sit behind the two FLAT plate faces
            assert seen[:2] == [(0, 0), (0, 0)], (model, cr, seen)
    finally:
        LR._tangent_facet_transport = real


# ---------------------------------------------------------------------------
# 4.  THE REFUSAL IS PRICED AND PRINTED
# ---------------------------------------------------------------------------
def test_the_pull_back_refusal_is_priced_and_printed():
    """The remap rung's SCREEN half bands; its pull-back does not.  That is a
    refusal, so it is reported with numbers -- the measured ``max|W|``, the
    halo it implies in ROWS, and that as a percentage of the band actually in
    use -- through ``progress``, which costs nothing when nobody is listening.
    """
    msgs = []

    def prog(stage, frac, message):
        msgs.append(message)

    N, dx = 96, 25e-6
    _run(_field(N, dx), _biconvex(), dx, 'tangent_facet_remap', 8,
         progress=prog)
    hits = [m for m in msgs if 'pull-back NOT banded' in m]
    assert hits, msgs
    m = hits[0]
    assert 'max|W|' in m and 'halo rows' in m and '8-row band' in m
    assert 'spline_filter IIR' in m and 'min(det)' in m
    # ...and the SCREEN half is not quietly refused with it
    assert 'SCREEN half IS banded' in m


def test_the_refusal_costs_nothing_without_a_callback():
    """The pricing takes two whole-grid reductions over the walk, so it is not
    taken at all when there is no callback -- pinned structurally rather than
    timed."""
    calls = {'n': 0}
    real = np.max

    def spy(*a, **k):
        calls['n'] += 1
        return real(*a, **k)

    N, dx = 96, 25e-6
    E = _field(N, dx)
    np.max = spy
    try:
        calls['n'] = 0
        _run(E, _biconvex(), dx, 'tangent_facet_remap', 8)
        silent = calls['n']
        calls['n'] = 0
        _run(E, _biconvex(), dx, 'tangent_facet_remap', 8,
             progress=lambda *a: None)
        loud = calls['n']
    finally:
        np.max = real
    assert loud > silent, (silent, loud)


# ---------------------------------------------------------------------------
# 5.  THE HALOS ARE THE DERIVED ONES -- CONTROLS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('model,narrow', [('tangent_facet', 2),
                                          ('tangent_facet_remap', 1)])
def test_a_narrower_halo_breaks_it(model, narrow):
    """A guard that is never observed to fail is indistinguishable from a
    guard that does nothing.  Narrow the sag halo by one row and the banded
    path must stop reproducing the whole grid -- either by moving the field
    (route 3, whose deepest level still has rows to read) or by running out of
    rows outright (the remap rung at halo 1, where the Hessian level has none
    left).  Both are failures to reproduce; neither is allowed to pass."""
    N, dx = 96, 25e-6
    E = _field(N, dx)
    presc = _biconvex()
    ref, _ = _run(E, presc, dx, model, 0)
    ok, _ = _run(E, presc, dx, model, 8)
    assert np.array_equal(ref.view(np.uint8), ok.view(np.uint8))

    real3, real2 = LR._TF_SAG_HALO_ROWS, LR._TF_REMAP_SAG_HALO_ROWS
    LR._TF_SAG_HALO_ROWS, LR._TF_REMAP_SAG_HALO_ROWS = narrow, narrow
    try:
        bad = None
        try:
            bad, _ = _run(E, presc, dx, model, 8)
        except ValueError:
            pass                      # ran out of rows: also a non-reproduction
    finally:
        LR._TF_SAG_HALO_ROWS, LR._TF_REMAP_SAG_HALO_ROWS = real3, real2
    assert bad is None or not np.array_equal(
        ref.view(np.uint8), bad.view(np.uint8)), (
        'a one-row-narrower sag halo must change the answer, or the shipped '
        'width is not the derived one')
    # ...and restoring it restores byte-identity, so the control is the halo
    again, _ = _run(E, presc, dx, model, 8)
    assert np.array_equal(ref.view(np.uint8), again.view(np.uint8))


def test_a_narrower_gap_transport_halo_breaks_it():
    """The gap transport's own 1-row halo, controlled the same way: at zero
    rows the banded accumulator reads a one-sided stencil where the whole grid
    reads a central one."""
    N, dx = 96, 25e-6
    E = _field(N, dx)
    presc = _leading_plate()
    ref, _ = _run(E, presc, dx, 'tangent_facet', 0)
    real = LR._TF_GAP_HALO_ROWS
    LR._TF_GAP_HALO_ROWS = 0
    try:
        bad, _ = _run(E, presc, dx, 'tangent_facet', 8)
    finally:
        LR._TF_GAP_HALO_ROWS = real
    assert not np.array_equal(ref.view(np.uint8), bad.view(np.uint8))
    again, _ = _run(E, presc, dx, 'tangent_facet', 8)
    assert np.array_equal(ref.view(np.uint8), again.view(np.uint8))


# ---------------------------------------------------------------------------
# 6.  THE ROWS FORMS ARE THE WHOLE-GRID FORMS
# ---------------------------------------------------------------------------
def test_the_rows_screen_equals_the_whole_grid_screen_exactly():
    """``_tangent_facet_screen`` is now literally ``_tangent_facet_screen_rows``
    at ``lo, hi = 0, Ny``, and a slab with one row of margin reproduces the
    whole-grid rows bit for bit.  Pinned as an equality of bytes, not a
    tolerance -- if it were a tolerance the band would be a different model."""
    rng = np.random.default_rng(11)
    ny, nx = 41, 23
    dx = dy = 25e-6
    sag = 3e-5 * rng.standard_normal((ny, nx))
    gy, gx = np.gradient(sag, dy, dx)
    px = 0.03 * rng.standard_normal((ny, nx))
    py = 0.02 * rng.standard_normal((ny, nx))
    whole, ok_w = LR._tangent_facet_screen(sag, gx, gy, px, py, 1.0, 1.5,
                                           dx, dy, np)
    for r0, r1 in ((5, 9), (0, 4), (ny - 4, ny)):
        a0, a1 = max(0, r0 - 1), min(ny, r1 + 1)
        band, ok_b = LR._tangent_facet_screen_rows(
            sag[a0:a1], gx[a0:a1], gy[a0:a1], px[a0:a1], py[a0:a1],
            1.0, 1.5, dx, dy, np, r0 - a0, r0 - a0 + (r1 - r0))
        assert np.array_equal(band.view(np.uint8),
                              whole[r0:r1].view(np.uint8)), (r0, r1)
        assert np.array_equal(ok_b, ok_w[r0:r1])


def test_the_rows_screen_accepts_the_scalar_accumulator():
    """The accumulator is a pair of PYTHON floats until the first powered
    surface, and every band has to read that SAME object -- promoting it to an
    array of zeros is the NEP 50 hazard that produced this family's one
    documented wrong answer.  ``_tf_sl`` passes scalars through, and this pins
    that the rows form is exact with them."""
    rng = np.random.default_rng(3)
    ny, nx = 33, 17
    dx = dy = 25e-6
    sag = 2e-5 * rng.standard_normal((ny, nx))
    gy, gx = np.gradient(sag, dy, dx)
    whole, _ = LR._tangent_facet_screen(sag, gx, gy, 0.0, 0.0, 1.0, 1.5,
                                        dx, dy, np)
    r0, r1 = 7, 12
    a0, a1 = r0 - 1, r1 + 1
    band, _ = LR._tangent_facet_screen_rows(
        sag[a0:a1], gx[a0:a1], gy[a0:a1], 0.0, 0.0, 1.0, 1.5, dx, dy, np,
        r0 - a0, r0 - a0 + (r1 - r0))
    assert np.array_equal(band.view(np.uint8), whole[r0:r1].view(np.uint8))


def test_the_rows_transport_equals_the_whole_grid_transport_exactly():
    rng = np.random.default_rng(5)
    ny, nx = 37, 19
    dx = dy = 25e-6
    px = 0.04 * rng.standard_normal((ny, nx))
    py = 0.03 * rng.standard_normal((ny, nx))
    wx, wy = LR._tangent_facet_transport(px, py, 4e-3, 1.6, dx, dy, np)
    for r0, r1 in ((6, 11), (0, 5), (ny - 5, ny)):
        a0, a1 = max(0, r0 - 1), min(ny, r1 + 1)
        bx, by = LR._tangent_facet_transport_rows(
            px[a0:a1], py[a0:a1], 4e-3, 1.6, dx, dy, np,
            r0 - a0, r0 - a0 + (r1 - r0))
        assert np.array_equal(bx.view(np.uint8), wx[r0:r1].view(np.uint8))
        assert np.array_equal(by.view(np.uint8), wy[r0:r1].view(np.uint8))


def test_rows_grad_drops_only_the_rows_whose_stencil_is_wrong():
    """``_tf_rows_grad`` is the primitive the whole halo argument rests on:
    np.gradient's interior stencil does not know how tall the array is, so a
    slab reproduces the whole-grid gradient exactly except at a slab edge that
    is not also a GRID edge -- and those rows are dropped rather than
    trusted."""
    rng = np.random.default_rng(13)
    ny, nx = 40, 11
    a = rng.standard_normal((ny, nx))
    gy_w, gx_w = np.gradient(a, 2.0, 3.0)
    # an interior slab loses one row at each end
    gy, gx, b0, b1 = LR._tf_rows_grad(a[10:20], 10, 20, ny, 2.0, 3.0, np)
    assert (b0, b1) == (11, 19)
    assert np.array_equal(gy.view(np.uint8), gy_w[11:19].view(np.uint8))
    assert np.array_equal(gx.view(np.uint8), gx_w[11:19].view(np.uint8))
    # a slab that touches row 0 keeps it (the one-sided stencil is the RIGHT
    # one there, which is why the halo is clipped rather than padded)
    gy, gx, b0, b1 = LR._tf_rows_grad(a[0:8], 0, 8, ny, 2.0, 3.0, np)
    assert (b0, b1) == (0, 7)
    assert np.array_equal(gy.view(np.uint8), gy_w[0:7].view(np.uint8))
    # ...and likewise the last row
    gy, gx, b0, b1 = LR._tf_rows_grad(a[32:ny], 32, ny, ny, 2.0, 3.0, np)
    assert (b0, b1) == (33, ny)
    assert np.array_equal(gy.view(np.uint8), gy_w[33:ny].view(np.uint8))


# ---------------------------------------------------------------------------
# 7.  THE GPU PATH STILL FALLS THROUGH
# ---------------------------------------------------------------------------
def test_gpu_namespace_still_falls_through_to_whole_grid():
    """Both band gates keep ``and xp is np``, and ``_chunk_grids`` requires it
    too, so a CuPy field keeps taking the whole-grid path -- pinned
    structurally (by reading the source) rather than by needing a GPU."""
    import inspect
    # v5.40 RETARGET (not a relaxation): ``apply_real_lens`` became a thin
    # wrapper that owns the accumulator-store context and forwards to
    # ``_apply_real_lens_impl``, which is where the band gates now live.  The
    # assertion itself is unchanged -- the same three source markers, the same
    # ``xp is np`` requirement.  The companion check below pins the split, so
    # a future move of the body cannot make this test read an empty string and
    # pass vacuously.
    src = inspect.getsource(LR._apply_real_lens_impl)
    assert '_apply_real_lens_impl' in inspect.getsource(LR.apply_real_lens), (
        "apply_real_lens no longer forwards to _apply_real_lens_impl; this "
        "test is reading the wrong function")
    i = src.index('_narrow_chunk = (')
    j = src.index('_slant_narrow_chunk = (')
    assert 'xp is np' in src[i:i + 400]
    assert 'xp is np' in src[j:j + 400]
    k = src.index('_chunk_grids = (')
    assert 'xp is np' in src[k:k + 200]


# ---------------------------------------------------------------------------
# 8.  THE AUTO THRESHOLD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('model', MODELS)
def test_the_auto_convention_now_covers_this_family(model, monkeypatch):
    """``sag_chunk_rows=None`` -> AUTO is the convention every other screen
    follows, and this family now follows it: byte-identical (this whole file)
    and strictly lighter (the build note's table).  Tested with the threshold
    monkeypatched down, so the test does not need a multi-GB reference."""
    N, dx = 96, 25e-6
    E = _field(N, dx)
    presc = _biconvex()
    ref, _ = _run(E, presc, dx, model, 0)
    monkeypatch.setattr(LR, '_SAG_CHUNK_AUTO_MIN_N', 32)
    monkeypatch.setattr(LR, '_SAG_CHUNK_AUTO_MIN_ROWS', 8)
    got, _ = _run(E, presc, dx, model, None)
    assert LR._resolve_sag_chunk_rows(None, N) == 8
    assert np.array_equal(ref.view(np.uint8), got.view(np.uint8))


# ---------------------------------------------------------------------------
# 9.  THE OBLIQUITY BLOCK IS OFF UNDER THIS FAMILY
# ---------------------------------------------------------------------------
def test_the_screen_obliquity_block_is_dead_under_this_family():
    """``_check_screen_obliquity_support`` already returned False for these
    models and the guard was already gated off, so the obliquity block was
    computing a correction nobody added and accumulating a momentum field
    nobody read.  It is gated off at the source now; this pins that the delta
    is never even called, which is the structural form of "dead"."""
    calls = {'n': 0}
    real = LR._screen_obliquity_delta

    def spy(*a, **k):
        calls['n'] += 1
        return real(*a, **k)

    N, dx = 96, 25e-6
    E = _field(N, dx)
    car = la.TiltedCarrier(-40e-3, 0.03, 0.02)
    LR._screen_obliquity_delta = spy
    try:
        for model in MODELS:
            for cr in (0, 8):
                calls['n'] = 0
                _run(E, _biconvex(), dx, model, cr, carrier=car)
                assert calls['n'] == 0, (model, cr)
        # ...and the control: with the default 'thin' screen it IS called
        calls['n'] = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E, prescription=_biconvex(), wavelength=LAM,
                            dx=dx, carrier=car, sag_chunk_rows=8)
        assert calls['n'] > 0
    finally:
        LR._screen_obliquity_delta = real


def test_the_carrier_still_seeds_the_accumulator():
    """The control for the test above -- gating the obliquity block off must
    not gate the CARRIER off.  ``carrier=`` seeds the tangent-facet momentum
    accumulator, so the field must still depend on it."""
    N, dx = 96, 25e-6
    E = _field(N, dx)
    for model in MODELS:
        for cr in (0, 8):
            blind, _ = _run(E, _biconvex(), dx, model, cr)
            car, _ = _run(E, _biconvex(), dx, model, cr,
                          carrier=la.TiltedCarrier(np.inf, 0.05, 0.0))
            assert not np.array_equal(blind.view(np.uint8),
                                      car.view(np.uint8)), (model, cr)
