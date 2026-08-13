"""Byte-identity of the row-banded ANGLE-TRUE (screen-obliquity) sag path.

v5.35.0 gave ``apply_real_lens`` an angle-true screen: pass ``carrier=`` and
each surface's ``(n2-n1)*sag`` OPD picks up equation (4)'s angular term plus
the R1 drift term.  Both are built out of GRADIENTS of the sag, and the
row-banded sag paths (``_narrow_chunk`` / ``_slant_narrow_chunk``) were
therefore disqualified outright -- ``carrier=`` forced the whole-grid path,
which is what made a 32k angle-aware run cost ~+129 GB over the banded-slant
baseline and OOM the box.

v5.35.3 bands them with a halo instead.  The pieces that are NOT pointwise:

* ``xp.gradient(sag)`` -- 1-row halo (the precedent
  ``test_slant_chunk_byte_identical`` already pins for the refraction leg);
* ``xp.gradient(e_err)`` inside the R1 term -- a gradient OF a gradient, so a
  2-row sag halo when the drift is live;
* ``xp.gradient(p0)`` in the inter-surface re-imaging step -- 1-row halo on
  the persistent momentum grids;
* ``bool(xp.any(sag))`` (the flat-face skip) and the carrier's ``all(q == 0)``
  -- reductions, evaluated band-wise with a short circuit;
* the numexpr-vs-numpy phase-screen choice -- gated on the WHOLE ``E.size``,
  exactly as the whole-grid path does, because numexpr's ``exp`` differs from
  numpy's in the last bit.

Everything else is element-wise, so the banded output must be BYTE-IDENTICAL
to the whole-grid output -- pinned here with ``np.array_equal`` (not
``allclose``), together with the estimator accumulator the accuracy guard
scores and the guard's own message.

Determinism note: the model does an FFT-based ASM propagation through the
glass between surfaces, so byte-equality across separate calls needs FFT plan
determinism -- the fixture pins auto-promote off (matching
``test_slant_chunk_byte_identical`` and ``test_screen_obliquity``).
"""
import gc
import inspect
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as LR
from lumenairy.elements._lens_real import apply_real_lens

LAM = 1.31e-6


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


def _biconvex(**extra0):
    """Both faces POWERED, so the momentum accumulator is a live grid at
    surface 1 and the R1 drift term (which needs the gap behind surface 0)
    actually fires."""
    return {'name': 'biconvex', 'aperture_diameter': 3e-3,
            'thicknesses': [4e-3],
            'surfaces': [_surf(19.6e-3, 'air', 'N-SSK2', **extra0),
                         _surf(-19.6e-3, 'N-SSK2', 'air')]}


def _planoconvex():
    """One FLAT face -- exercises the ``bool(xp.any(sag))`` skip that keeps a
    plane face from touching the accumulators at all."""
    return {'name': 'planoconvex', 'aperture_diameter': 3e-3,
            'thicknesses': [4e-3],
            'surfaces': [_surf(19.6e-3, 'air', 'N-SSK2'),
                         _surf(np.inf, 'N-SSK2', 'air')]}


def _leading_plate():
    """A FLAT plate in front of the powered pair: the plate's gap advances the
    drift while ``p0`` is still the scalar seed, so the first powered surface
    runs the R1 term against a SCALAR momentum -- the promotion corner."""
    return {'name': 'plate+lens', 'aperture_diameter': 3e-3,
            'thicknesses': [1e-3, 2e-3, 4e-3],
            'surfaces': [_surf(np.inf, 'air', 'N-BK7'),
                         _surf(np.inf, 'N-BK7', 'air'),
                         _surf(19.6e-3, 'air', 'N-SSK2'),
                         _surf(-19.6e-3, 'N-SSK2', 'air')]}


def _steep_conic():
    """An OBLATE conic (k = +4) whose domain edge (norm >= 0.9999) falls INSIDE
    the grid, so ``surface_sag_general`` returns its NaN sentinel on an annulus
    that crosses band boundaries at every band size."""
    return {'name': 'oblate', 'aperture_diameter': 3e-3,
            'thicknesses': [4e-3],
            'surfaces': [_surf(1.6e-3, 'air', 'N-BK7', conic=4.0),
                         _surf(-19.6e-3, 'N-BK7', 'air')]}


def _stop_presc():
    p = _biconvex()
    p['aperture_diameter'] = 2.0e-3
    p['stop_index'] = 1
    return p


PLANE = la.TiltedCarrier(float('inf'), 0.05, 0.0)
SPHERE = la.TiltedCarrier(0.25, 0.05, 0.0)


def _field(N, dx, dy=None, dtype=np.complex128):
    dy = dx if dy is None else dy
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    env = np.exp(-(X ** 2 + Y ** 2) / (0.8e-3) ** 2)
    return (env * np.exp(1j * 2 * np.pi / LAM * 0.01 * X)).astype(dtype)


# ---------------------------------------------------------------------------
# the comparison harness: field, dtype, guard message AND the estimator
# accumulator the guard scores (captured at the guard's own call site).
# ---------------------------------------------------------------------------
def _run(presc, E, dx, dy, chunk, monkeypatch=None, **kw):
    seen = {}
    real_rms = LR._screen_obliquity_rms_waves

    def spy(field, X, Y, r_pupil, wavelength, xp):
        seen['total'] = np.array(field, copy=True)
        return real_rms(field, X, Y, r_pupil, wavelength, xp)

    LR._screen_obliquity_rms_waves = spy
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = apply_real_lens(E.copy(), prescription=presc, wavelength=LAM,
                                  dx=dx, dy=dy, sag_chunk_rows=chunk, **kw)
            msgs = sorted(str(w.message) for w in caught
                          if 'angle-blind' in str(w.message)
                          or 'grazing' in str(w.message))
    finally:
        LR._screen_obliquity_rms_waves = real_rms
    return out, msgs, seen.get('total')


def _assert_identical(presc, E, dx, chunk, label, dy=None, **kw):
    Ew, mw, tw = _run(presc, E, dx, dy, 0, **kw)
    Eb, mb, tb = _run(presc, E, dx, dy, chunk, **kw)
    assert Ew.dtype == Eb.dtype, (
        f"{label}: dtype {Ew.dtype} (whole) != {Eb.dtype} (band{chunk})")
    assert np.array_equal(Ew, Eb), (
        f"{label}: band{chunk} field != whole-grid "
        f"(max |diff| {np.nanmax(np.abs(Ew - Eb)):.3e})")
    assert mw == mb, (
        f"{label}: band{chunk} guard messages differ:\n{mw}\n{mb}")
    if tw is None:
        assert tb is None, f"{label}: banded scored an estimator, whole did not"
    else:
        assert tb is not None, f"{label}: banded skipped the estimator"
        assert np.array_equal(tw, tb), (
            f"{label}: band{chunk} estimator accumulator != whole-grid "
            f"(max |diff| {np.nanmax(np.abs(tw - tb)):.3e})")


# ---------------------------------------------------------------------------
# 1. the core matrix: screen x carrier x policy x apply-vs-estimate,
#    on a fully powered pair AND on a singlet with one flat face.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("presc_fn,label", [(_biconvex, 'biconvex'),
                                            (_planoconvex, 'planoconvex')])
@pytest.mark.parametrize("slant,fres,screen", [(False, False, 'paraxial'),
                                               (True, False, 'slant'),
                                               (False, True, 'fresnel')])
@pytest.mark.parametrize("carrier,cname", [(PLANE, 'plane'),
                                           (SPHERE, 'sphere')])
@pytest.mark.parametrize("policy", ['warn', 'silent'])
@pytest.mark.parametrize("screen_obliquity", [True, False])
def test_core_matrix_byte_identical(presc_fn, label, slant, fres, screen,
                                    carrier, cname, policy, screen_obliquity):
    """{paraxial, slant, fresnel} x {plane, sphere carrier} x {warn, silent} x
    {correction applied, estimator-only} -- field, dtype, guard message and
    estimator accumulator all byte-identical between banded and whole-grid."""
    N, dx = 128, 6e-6
    _assert_identical(
        presc_fn(), _field(N, dx), dx, 17,
        f"{label}/{screen}/{cname}/{policy}/so={screen_obliquity}",
        carrier=carrier, screen_obliquity=screen_obliquity,
        on_screen_obliquity=policy, slant_correction=slant, fresnel=fres)


# ---------------------------------------------------------------------------
# 2. adversarial geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [1, 3, 7, 17])
@pytest.mark.parametrize("slant", [False, True])
def test_nan_sentinel_at_band_boundaries(chunk, slant):
    """An oblate conic whose domain edge falls inside the grid puts NaN
    sentinel pixels on an annulus that crosses EVERY band boundary at these
    band sizes.  The whole-grid block zeroes them on the WHOLE sag before
    taking the gradient; the banded block has to zero them on the HALO (not
    just the band) or the halo rows feed NaN back into the interior stencil."""
    N, dx = 96, 2.0e-5
    _assert_identical(
        _steep_conic(), _field(N, dx), dx, chunk,
        f"nan-sentinel/cr={chunk}/slant={slant}",
        carrier=SPHERE, screen_obliquity=True, on_screen_obliquity='warn',
        slant_correction=slant)


@pytest.mark.parametrize("presc_fn,label", [
    (lambda: _biconvex(clear_aperture=1.8e-3), 'clear_aperture'),
    (_stop_presc, 'stop_surface'),
    (_leading_plate, 'leading_flat_plate'),
    (lambda: _biconvex(decenter=(5e-6, 0.0)), 'decentered_falls_through'),
])
@pytest.mark.parametrize("carrier,cname", [(PLANE, 'plane'), (SPHERE, 'sphere')])
def test_masks_and_mixed_paths_byte_identical(presc_fn, label, carrier, cname):
    """A per-surface clear aperture and an aperture STOP (both handled per band
    in the slant path, both a whole-grid fall-through in the paraxial one), a
    LEADING FLAT PLATE (the drift goes live while the momentum accumulator is
    still a scalar), and a DECENTERED face (which falls through to the
    whole-grid path mid-prescription, so the two paths have to hand the
    accumulators back and forth within one call)."""
    N, dx = 96, 8e-6
    for slant in (False, True):
        _assert_identical(
            presc_fn(), _field(N, dx), dx, 13,
            f"{label}/{cname}/slant={slant}",
            carrier=carrier, screen_obliquity=True,
            on_screen_obliquity='warn', slant_correction=slant)


@pytest.mark.parametrize("chunk", [1, 3, 65, 72])
def test_odd_n_anisotropic_grid_byte_identical(chunk):
    """Odd N (no band divides it evenly) on an ANISOTROPIC grid (dx != dy, so
    the gradient's per-axis spacing matters) at band sizes 1, 3, N and N+7 --
    single-row bands, a ragged tail, one whole-grid band, and a band wider
    than the grid."""
    N, dx, dy = 65, 1.2e-5, 1.0e-5
    cr = {65: N, 72: N + 7}.get(chunk, chunk)
    _assert_identical(
        _biconvex(), _field(N, dx, dy), dx, cr, f"oddN/cr={cr}", dy=dy,
        carrier=SPHERE, screen_obliquity=True, on_screen_obliquity='warn',
        slant_correction=True, fresnel=True)


@pytest.mark.parametrize("chunk", [1, 3, 515, 522])
def test_large_odd_n_byte_identical(chunk):
    """The same at N = 515 (odd, > 512), which is where a ragged final band
    also contains the true array edge row."""
    N, dx = 515, 3e-6
    _assert_identical(
        _biconvex(), _field(N, dx), dx, chunk, f"N515/cr={chunk}",
        carrier=SPHERE, screen_obliquity=True, on_screen_obliquity='silent',
        slant_correction=True)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("sag_dtype", [None, 'float32'])
def test_dtype_matrix_byte_identical(dtype, sag_dtype):
    """complex64 (which the fresnel amplitude promotes to complex128) and the
    opt-in float32 GEOMETRY dtype.

    float32 sag is the case that caught the first draft: promoting the scalar
    momentum seed to a full grid at band 0 made bands 1..n read a float32
    ARRAY of zeros where band 0 and the whole grid read a PYTHON float, which
    under NEP 50 drops the momentum triangle from float64 to float32 (5e-6 of
    field error).  The build pins the source per surface instead."""
    N, dx = 128, 6e-6
    kw = dict(carrier=SPHERE, screen_obliquity=True,
              on_screen_obliquity='warn', slant_correction=True, fresnel=True)
    if sag_dtype is not None:
        kw['sag_dtype'] = sag_dtype
    _assert_identical(_biconvex(), _field(N, dx, dtype=dtype), dx, 17,
                      f"dtype={dtype.__name__}/sag={sag_dtype}", **kw)


@pytest.mark.parametrize("carrier,cname", [
    (0.3, 'scalar_conjugate'), ('auto', 'auto_fit'), (None, 'ndarray')])
def test_other_carrier_vocabularies_byte_identical(carrier, cname):
    """The carrier vocabularies WITHOUT a closed row form (a signed scalar
    conjugate, an 'auto' fit, an explicit wavefront ndarray).  These keep the
    materialised momentum field and are simply sliced per band -- no memory
    win, but the byte-identity claim is unconditional."""
    N, dx = 128, 6e-6
    if carrier is None:                     # explicit wavefront ndarray
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        carrier = (X ** 2 + Y ** 2) / (2 * 0.3) + 0.02 * X
    _assert_identical(
        _biconvex(), _field(N, dx), dx, 17, f"carrier={cname}",
        carrier=carrier, screen_obliquity=True, on_screen_obliquity='warn',
        slant_correction=True)


# ---------------------------------------------------------------------------
# 3. the AUTO band threshold
# ---------------------------------------------------------------------------
def test_auto_threshold_resolution_unchanged():
    """``sag_chunk_rows=None`` still resolves to whole-grid below N=4096 and to
    ``max(256, N//16)`` at or above it -- the obliquity fix does not move the
    auto threshold, and the runner preflight prices the two branches
    differently."""
    assert LR._resolve_sag_chunk_rows(None, 2048) is None
    assert LR._resolve_sag_chunk_rows(None, 4095) is None
    assert LR._resolve_sag_chunk_rows(None, 4096) == 256
    assert LR._resolve_sag_chunk_rows(None, 16384) == 1024


def test_auto_band_engages_byte_identical(monkeypatch):
    """AUTO mode (``sag_chunk_rows=None``) with a carrier, on both sides of the
    threshold.

    BELOW: auto resolves to whole-grid, so ``None`` and ``0`` are the same
    call.  ABOVE: auto bands.  The shipped threshold is 4096, where a
    whole-grid angle-aware reference run costs several GB, so the ABOVE arm
    monkeypatches the threshold down -- the code under test is the same band
    loop with the same multi-band split (32-row bands over 128 rows)."""
    N, dx = 128, 6e-6
    E = _field(N, dx)
    kw = dict(carrier=SPHERE, screen_obliquity=True,
              on_screen_obliquity='warn', slant_correction=True)
    # below the shipped threshold: AUTO == whole-grid
    assert LR._resolve_sag_chunk_rows(None, N) is None
    _assert_identical(_biconvex(), E, dx, None, 'auto/below-threshold', **kw)
    # above a lowered threshold: AUTO bands, in 32-row bands
    monkeypatch.setattr(LR, '_SAG_CHUNK_AUTO_MIN_N', 64)
    monkeypatch.setattr(LR, '_SAG_CHUNK_AUTO_MIN_ROWS', 32)
    assert LR._resolve_sag_chunk_rows(None, N) == 32
    _assert_identical(_biconvex(), E, dx, None, 'auto/above-threshold', **kw)


# ---------------------------------------------------------------------------
# 4. the band loop is really taken (not silently falling through)
# ---------------------------------------------------------------------------
def test_carrier_no_longer_disqualifies_the_band(monkeypatch):
    """The pre-v5.35.3 behaviour was structural: ``carrier=`` forced the
    whole-grid path, which ALWAYS builds the X/Y/h_sq meshgrids.  With a
    SINGLE surface (no glass propagation, hence no ASM meshgrid) and the
    guard silenced (the guard's own estimator legitimately needs X/Y), NO
    FULL-GRID meshgrid may be built any more.

    Band-sized meshgrids are expected and fine -- the finite-radius carrier's
    momentum field is evaluated a band at a time -- so the spy scores the
    SHAPES, not the call count."""
    N, dx = 128, 6e-6
    E = _field(N, dx)
    one = {'name': 'one', 'aperture_diameter': 3e-3, 'thicknesses': [],
           'surfaces': [_surf(19.6e-3, 'air', 'N-SSK2')]}

    shapes = []
    real_mesh = np.meshgrid

    def counting_mesh(*a, **k):
        out = real_mesh(*a, **k)
        shapes.append(np.shape(out[0]))
        return out

    monkeypatch.setattr(np, 'meshgrid', counting_mesh)

    def _run_one(**kw):
        shapes.clear()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E.copy(), prescription=one, wavelength=LAM, dx=dx,
                            sag_chunk_rows=32, **kw)
        return list(shapes)

    for slant in (False, True):
        got = _run_one(carrier=PLANE, screen_obliquity=True,
                       on_screen_obliquity='silent', slant_correction=slant)
        assert got == [], (
            f"plane carrier, slant={slant}: banded angle-aware run built "
            f"meshgrids {got}")
        got = _run_one(carrier=SPHERE, screen_obliquity=True,
                       on_screen_obliquity='silent', slant_correction=slant)
        assert got and all(s[0] <= 32 for s in got), (
            f"sphere carrier, slant={slant}: the carrier angle field "
            f"materialised a non-band grid (shapes {got})")


def test_banded_takes_one_gradient_pair_per_band(monkeypatch):
    """Count np.gradient calls: the whole-grid angle-aware surface takes a
    fixed number per surface, the banded one takes that many per BAND.  A
    silent fall-through to the whole-grid path would show equal counts."""
    N, dx = 128, 6e-6
    E = _field(N, dx)
    kw = dict(prescription=_biconvex(), wavelength=LAM, dx=dx, carrier=PLANE,
              screen_obliquity=True, on_screen_obliquity='silent')

    counts = {'n': 0}
    real_grad = np.gradient

    def counting_grad(*a, **k):
        counts['n'] += 1
        return real_grad(*a, **k)

    monkeypatch.setattr(np, 'gradient', counting_grad)

    counts['n'] = 0
    apply_real_lens(E.copy(), sag_chunk_rows=0, **kw)
    whole = counts['n']
    counts['n'] = 0
    apply_real_lens(E.copy(), sag_chunk_rows=32, **kw)
    band = counts['n']
    assert whole >= 3, "whole-grid angle-aware path took no gradients"
    assert band == whole * (N // 32), (
        f"banded gradient count {band} != {whole} * {N // 32}; the per-band "
        f"obliquity path may not have executed")


def test_gpu_namespace_still_falls_through_to_whole_grid():
    """The two banded gates must still require ``xp is np``.  numexpr has no
    GPU backend and the halo slicing is written against NumPy semantics, so a
    CuPy field has to keep taking the whole-grid path -- and the carrier's row
    evaluator must only be built when the banded path is live."""
    src = inspect.getsource(LR.apply_real_lens)
    assert src.count('and xp is np and not slant_correction') == 1
    assert src.count('and xp is np and (slant_correction or fresnel)') == 1
    # the row evaluator is gated on _chunk_grids, which itself requires np
    assert '_chunk_grids = (sag_chunk_rows is not None' in src
    assert 'and xp is np)' in src
    assert 'if _chunk_grids:\n            _obl_q_rows_fn' in src


def test_row_evaluator_declines_non_analytic_carriers():
    """The row evaluator is a memory optimisation with a narrow contract: only
    a finite-radius TiltedCarrier (analytic and pointwise in x, y).  Everything
    else returns None and keeps the materialised field."""
    N, dx = 32, 6e-6
    assert LR._screen_obliquity_row_evaluator(PLANE, dx, dx, N, N) is None
    assert LR._screen_obliquity_row_evaluator(0.3, dx, dx, N, N) is None
    assert LR._screen_obliquity_row_evaluator('auto', dx, dx, N, N) is None
    assert LR._screen_obliquity_row_evaluator(
        np.zeros((N, N)), dx, dx, N, N) is None
    rows = LR._screen_obliquity_row_evaluator(SPHERE, dx, dx, N, N, n_medium=1.5)
    assert rows is not None
    # and it reproduces the whole-grid field row for row
    qx, qy = LR._screen_obliquity_angle_field(
        SPHERE, np.zeros((N, N), dtype=np.complex128), LAM, dx, dx, N, N,
        n_medium=1.5)
    for r0 in (0, 5, 17):
        r1 = min(N, r0 + 7)
        bx, by = rows(r0, r1)
        assert np.array_equal(bx, qx[r0:r1]), f"qx band [{r0}:{r1}] differs"
        assert np.array_equal(by, qy[r0:r1]), f"qy band [{r0}:{r1}] differs"


# ---------------------------------------------------------------------------
# 5. the memory claim (generous margin -- the byte-identity tests above carry
#    the CORRECTNESS claim unconditionally, this one only has to show the
#    banded path is the leaner one on whatever runner it lands on)
# ---------------------------------------------------------------------------
def test_banded_peak_is_much_smaller_than_whole_grid():
    """tracemalloc peak of the banded angle-aware run at N=2048 /
    sag_chunk_rows=128 against the same call whole-grid.

    No per-build absolute number is asserted -- runners differ in BLAS
    scratch, FFT planner and allocator behaviour.  What must hold on any of
    them is the SHAPE of the fix: the whole-grid path holds the sag, its
    gradient pair, the delta, the R1 gradients and the carrier angle field
    simultaneously, and the banded path holds one band of each, so the ratio
    has to sit well under 1.  0.6 is a deliberately loose bar (measured 0.50
    on the reference box)."""
    N, dx = 2048, 4.0e-3 / 2048
    presc = _biconvex()
    E = _field(N, dx)
    kw = dict(prescription=presc, wavelength=LAM, dx=dx, carrier=SPHERE,
              screen_obliquity=True, on_screen_obliquity='warn',
              slant_correction=True)

    def peak(chunk):
        # warm up first: the FIRST apply_real_lens of a process also pays FFT
        # plan / lazy-import allocations, which land in the peak and would
        # flatter whichever arm ran second.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E.copy(), sag_chunk_rows=chunk, **kw)
            gc.collect()
            tracemalloc.start()
            apply_real_lens(E.copy(), sag_chunk_rows=chunk, **kw)
            _, p = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        gc.collect()
        return p

    p_whole = peak(0)
    p_band = peak(128)
    assert p_band < 0.6 * p_whole, (
        f"banded peak {p_band / 1e6:.0f} MB is not < 0.6 x whole-grid "
        f"{p_whole / 1e6:.0f} MB (ratio {p_band / p_whole:.2f}); the "
        f"row-banded obliquity path may not be engaging")
