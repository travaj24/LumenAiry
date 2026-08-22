"""The v5.40 memory levers: memory-mapped persistent accumulators, the banded
carrier seed, and the streamed ASM transfer function.

WHY THERE ARE THREE OF THEM, AND WHY THE MIDDLE ONE IS THE ONE THAT PAYS.

``PROBE_D121_ANALYTIC_32K_FOOTPRINT_2026_08_17`` measured 11.23 float64 grids
of residual footprint on a design-121 group under ``surface_model=
'tangent_facet'`` + ``carrier='auto'`` and attributed it to "the persistent
momentum accumulator, the sag gradients feeding it, and the ASM work arrays".
Re-measured with a time-resolved sampler rather than a warmed peak, the peak of
that call stands somewhere else entirely: inside the SET-UP of the carrier
momentum field, before the first surface is touched.  So:

* **the accumulators** (``_tf_px`` / ``_tf_py``, the fresh destination pair
  each surface writes into, the remap walk components, the screen-obliquity
  momentum / drift / carrier pairs and the guard accumulator) are real,
  simultaneously-live, whole-grid state that banding cannot remove -- 4 float64
  grids on route 3 with a carrier, 34.4 GB at N = 32768 -- and
  ``accumulator_store='memmap'`` spills them;
* **the seed** is what the accumulator is INITIALISED from, and
  ``_screen_obliquity_angle_field`` builds it whole-grid: 7 float64 grids live
  to deliver 2.  Banding it (``_screen_obliquity_rows_any``) is what lets the
  memmap lever reach the peak at all;
* **the ASM transfer function** is a full complex grid plus a second one for
  the product, both avoidable when H is generated during the multiply.

Every assertion below is a byte-identity or a decision, never a reading.  The
one quantitative claim -- that the store actually MOVES the accumulators out of
the Python heap -- is asserted as an inequality with a derivation and decades
of margin, not as a measured number (docs/TESTING_STANDARDS.md, rule 5).
"""
import glob
import os
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as LR
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements._lens_traced import TiltedCarrier, _compute_carrier
from lumenairy.propagators.asm import angular_spectrum_propagate

LAM = 1.31e-6
MODELS = ('tangent_facet', 'tangent_facet_remap')


@pytest.fixture(autouse=True)
def _deterministic_fft():
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


# ---------------------------------------------------------------------------
# fixtures -- the same shapes the sibling banded suite uses
# ---------------------------------------------------------------------------
def _surf(radius, glass_before, glass_after, **extra):
    d = {'radius': radius, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass_before, 'glass_after': glass_after}
    d.update(extra)
    return d


def _biconvex(R=12.6e-3, glass='N-SSK2'):
    return {'surfaces': [_surf(R, 'AIR', glass), _surf(-R, glass, 'AIR')],
            'thicknesses': [4.0e-3], 'aperture_diameter': 3.0e-3}


def _triplet():
    """Three refracting surfaces of two glasses -- the shape of the
    design-121 groups the probe measured, where the accumulator survives two
    gap transports."""
    return {'surfaces': [_surf(19.6e-3, 'AIR', 'N-SSK2'),
                         _surf(-27.4e-3, 'N-SSK2', 'N-SF57'),
                         _surf(-40.0e-3, 'N-SF57', 'AIR')],
            'thicknesses': [3.0e-3, 2.0e-3], 'aperture_diameter': 0.30e-3}


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
    """The engineered folding prescription: the remap rung must REFUSE, and
    must refuse identically with the levers on."""
    dx = period * 8 / n
    ax = (np.arange(n) - n // 2) * dx
    fe = amp * np.cos(2 * np.pi * ax[None, :] / period) + 0.0 * ax[:, None]
    return dx, {'surfaces': [
        _surf(np.inf, 'AIR', 'N-SF57', form_error=fe),
        _surf(np.inf, 'N-SF57', 'AIR')],
        'thicknesses': [200.0e-6]}


def _field(N, dx, dy=None, w=None, dtype=np.complex128, tilt=0.0):
    dy = dy if dy is not None else dx
    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(x, y)
    w = w if w is not None else 0.22 * N * min(dx, dy)
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    if tilt:
        E = E * np.exp(1j * (2 * np.pi / LAM) * tilt * X)
    return E.astype(dtype)


#: Every congruence ``_screen_obliquity_rows_any`` has to reproduce, plus the
#: two that must fall through to the caller's existing fast paths.
CARRIERS = [
    pytest.param(None, id='none'),
    pytest.param('auto', id='auto'),
    pytest.param(0.030, id='conj_pos'),
    pytest.param(-0.045, id='conj_neg'),
    pytest.param(TiltedCarrier(R=0.05, L=0.02, M=-0.01), id='tilt_finiteR'),
    pytest.param(TiltedCarrier(R=np.inf, L=0.02, M=-0.01), id='tilt_collim'),
    pytest.param(TiltedCarrier(R=np.inf, L=0.0, M=0.0), id='tilt_zero'),
]


def _run(E, presc, dx, dy=None, **kw):
    """``(outcome, payload, dtype, warnings)`` -- a REFUSAL is an outcome, not
    a failure, and is compared as one."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        try:
            out = apply_real_lens(E, prescription=presc, wavelength=LAM,
                                  dx=dx, dy=dy, **kw)
        except Exception as exc:            # noqa: BLE001 - compared, not swallowed
            return ('raise', (type(exc).__name__, str(exc)), None,
                    tuple(sorted(str(w.message)[:120] for w in wl)))
        return ('ok', out, out.dtype,
                tuple(sorted(str(w.message)[:120] for w in wl)))


def _assert_same(ref, got, what):
    assert ref[0] == got[0], (what, ref[0], got[0], ref[1], got[1])
    assert ref[3] == got[3], (what, ref[3], got[3])
    if ref[0] == 'raise':
        assert ref[1] == got[1], (what, ref[1], got[1])
        return
    assert ref[2] == got[2], (what, ref[2], got[2])
    assert np.array_equal(ref[1].view(np.uint8), got[1].view(np.uint8)), (
        what, float(np.max(np.abs(ref[1] - got[1]))))


def _carrier_kw(carrier):
    if carrier is None:
        return {}
    return dict(carrier=carrier, screen_obliquity=False,
                on_screen_obliquity='silent')


# ---------------------------------------------------------------------------
# 1.  BYTE IDENTITY -- the load-bearing claim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('carrier', CARRIERS)
@pytest.mark.parametrize('cr', [0, 1, 3, 96, 103])
def test_every_lever_is_byte_null_against_the_shipped_answer(
        tmp_path, model, carrier, cr):
    """The SHIPPED answer is ``sag_chunk_rows=0`` with both levers off -- what
    ``origin/main`` produces for the same call, since neither keyword exists
    there.  Every combination of band size, store and streaming must reproduce
    it byte for byte, or reproduce its REFUSAL with the same type and message.

    Band sizes 1 and 3 are the stress: every row is simultaneously a band
    interior and a band boundary, which is where a halo error shows.  103 is
    coprime with the grid so the last band is short.
    """
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64, tilt=0.01)
    presc = _triplet()
    base = dict(slant_correction=False, surface_model=model,
                **_carrier_kw(carrier))
    ref = _run(E, presc, dx, sag_chunk_rows=0, **base)
    for store in ('ram', 'memmap'):
        for stream in (False, True):
            got = _run(E, presc, dx, sag_chunk_rows=cr,
                       accumulator_store=store,
                       scratch_dir=str(tmp_path),
                       stream_transfer_function=stream, **base)
            _assert_same(ref, got, (model, carrier, cr, store, stream))


@pytest.mark.parametrize('presc_fn', [_biconvex, _triplet, _leading_plate,
                                      _oblate])
@pytest.mark.parametrize('carrier', ['auto', 0.030,
                                     TiltedCarrier(R=0.05, L=0.02, M=-0.01)])
def test_byte_null_across_prescription_shapes(tmp_path, presc_fn, carrier):
    """The four shapes that exercise the paths a banded seed could break: a
    plain doublet, a three-surface group (two gap transports), a LEADING PLATE
    (the accumulator stays two Python floats through it and promotes
    mid-prescription) and an oblate conic (a NaN sentinel annulus inside the
    grid)."""
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    presc = presc_fn()
    base = dict(slant_correction=False, surface_model='tangent_facet',
                **_carrier_kw(carrier))
    ref = _run(E, presc, dx, sag_chunk_rows=0, **base)
    for cr in (1, 7, N):
        got = _run(E, presc, dx, sag_chunk_rows=cr,
                   accumulator_store='memmap', scratch_dir=str(tmp_path),
                   stream_transfer_function=True, **base)
        _assert_same(ref, got, (presc_fn.__name__, carrier, cr))


@pytest.mark.parametrize('N,dyf', [(65, 1.0), (72, 1.5), (129, 0.75)])
@pytest.mark.parametrize('cdtype', [np.complex64, np.complex128])
def test_byte_null_on_odd_grids_and_anisotropic_pitch(tmp_path, N, dyf,
                                                      cdtype):
    """Odd ``N`` and ``dy != dx`` are where a half-bin index convention or a
    ``meshgrid`` axis order goes wrong: the banded seed rebuilds the y axis as
    ``arange(r0, r1) - Ny/2``, and the streamed H rebuilds the frequency
    vectors, so both have their own chance to disagree with the whole-grid
    layout on an odd grid."""
    dx = 4.0e-6
    dy = dx * dyf
    E = _field(N, dx, dy=dy, dtype=cdtype)
    presc = _triplet()
    base = dict(slant_correction=False, surface_model='tangent_facet',
                carrier='auto', screen_obliquity=False,
                on_screen_obliquity='silent')
    ref = _run(E, presc, dx, dy=dy, sag_chunk_rows=0, **base)
    got = _run(E, presc, dx, dy=dy, sag_chunk_rows=13,
               accumulator_store='memmap', scratch_dir=str(tmp_path),
               stream_transfer_function=True, **base)
    _assert_same(ref, got, (N, dyf, cdtype))


@pytest.mark.parametrize('opts', [
    pytest.param(dict(fresnel=True), id='fresnel'),
    pytest.param(dict(fresnel=True, absorption=True), id='fresnel_absorb'),
    pytest.param(dict(bandlimit=False), id='no_bandlimit'),
    pytest.param(dict(sag_dtype=np.float32), id='sag_f32'),
    pytest.param(dict(fresnel=True, sag_dtype=np.float32), id='fresnel_f32'),
])
def test_byte_null_across_the_option_combinations(tmp_path, opts):
    """``fresnel=True`` is what routes the family through the SECOND band gate
    and through the Fresnel/TIR path; ``bandlimit=False`` changes which
    branch the streamed H takes; ``sag_dtype=float32`` is the arm where a
    scalar-vs-array promotion in the accumulator seed would show as a dtype
    change rather than as a value change."""
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    presc = _triplet()
    base = dict(slant_correction=False, surface_model='tangent_facet',
                carrier='auto', screen_obliquity=False,
                on_screen_obliquity='silent', **opts)
    ref = _run(E, presc, dx, sag_chunk_rows=0, **base)
    got = _run(E, presc, dx, sag_chunk_rows=11,
               accumulator_store='memmap', scratch_dir=str(tmp_path),
               stream_transfer_function=True, **base)
    _assert_same(ref, got, opts)


def test_the_fold_guard_refuses_identically_with_every_lever_on(tmp_path):
    """``min(det)`` is a WHOLE-GRID reduction, so a lever that quietly made it
    band-local would run and return a field where the shipped call refuses.
    The engineered folding prescription must raise the SAME error with the
    same message; the control at an amplitude the guard passes must be
    byte-identical."""
    dx, presc = _folding()
    N = 129
    E = _field(N, dx, dtype=np.complex64)
    base = dict(slant_correction=False, surface_model='tangent_facet_remap',
                carrier='auto', screen_obliquity=False,
                on_screen_obliquity='silent')
    ref = _run(E, presc, dx, sag_chunk_rows=0, **base)
    assert ref[0] == 'raise', "the fixture no longer folds -- re-engineer it"
    assert 'folds' in ref[1][1]
    for cr in (1, 17, N):
        got = _run(E, presc, dx, sag_chunk_rows=cr,
                   accumulator_store='memmap', scratch_dir=str(tmp_path),
                   stream_transfer_function=True, **base)
        _assert_same(ref, got, ('fold', cr))

    dx_ok, presc_ok = _folding(amp=0.05e-6)
    E_ok = _field(N, dx_ok, dtype=np.complex64)
    ref_ok = _run(E_ok, presc_ok, dx_ok, sag_chunk_rows=0, **base)
    assert ref_ok[0] == 'ok', "the control fixture must PASS the guard"
    got_ok = _run(E_ok, presc_ok, dx_ok, sag_chunk_rows=17,
                  accumulator_store='memmap', scratch_dir=str(tmp_path),
                  stream_transfer_function=True, **base)
    _assert_same(ref_ok, got_ok, 'fold-control')


# ---------------------------------------------------------------------------
# 2.  THE SEED COLLAPSE -- the NEP-50 pin
# ---------------------------------------------------------------------------
def test_a_constant_momentum_field_still_collapses_to_two_python_floats():
    """``_screen_obliquity_angle_field`` collapses a CONSTANT momentum field
    to two Python floats, and that is observable rather than cosmetic: under
    NEP 50 a Python float and a float64 array of the same value promote
    ``float32`` geometry differently, so a banded seed that returned a
    constant GRID would change the screen's arithmetic dtype.

    Engineered rather than hoped for: a collimated tilt has constant direction
    cosines by construction, and ``sag_dtype=float32`` is the configuration
    where the promotion would show.  Asserted as byte-identity of the
    float32-geometry output against the whole-grid path, which is a decision
    (same dtype, same bits) and not a reading.
    """
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    presc = _triplet()
    base = dict(slant_correction=False, surface_model='tangent_facet',
                carrier=TiltedCarrier(R=np.inf, L=0.02, M=-0.01),
                screen_obliquity=False, on_screen_obliquity='silent',
                sag_dtype=np.float32)
    ref = _run(E, presc, dx, sag_chunk_rows=0, **base)
    got = _run(E, presc, dx, sag_chunk_rows=7, **base)
    _assert_same(ref, got, 'collimated-collapse')

    # And directly: the row evaluator declines the collimated tilt outright,
    # so the caller keeps its two-float fast path rather than filling a grid
    # with a constant.
    assert LR._screen_obliquity_rows_any(
        TiltedCarrier(R=np.inf, L=0.02, M=-0.01), E, LAM, dx, dx, N, N) is None


def test_the_banded_seed_reproduces_the_whole_grid_momentum_field_exactly():
    """Directly, without the lens: every band of ``_screen_obliquity_rows_any``
    equals the corresponding rows of ``_screen_obliquity_angle_field``.

    This is the claim the byte-identity tests above rest on, isolated so that
    a failure here names the seed rather than the screen."""
    N, dx, dy = 65, 4.0e-6, 5.5e-6
    E = _field(N, dx, dy=dy, dtype=np.complex64, tilt=0.01)
    for carrier in ('auto', 0.030, -0.045,
                    TiltedCarrier(R=0.05, L=0.02, M=-0.01)):
        for n1 in (1.0, 1.62):
            whole = LR._screen_obliquity_angle_field(
                carrier, E, LAM, dx, dy, N, N, n_medium=n1)
            rows = LR._screen_obliquity_rows_any(
                carrier, E, LAM, dx, dy, N, N, n_medium=n1)
            assert rows is not None, carrier
            qx, qy = whole
            assert getattr(qx, 'ndim', 0) == 2, (carrier, 'expected a grid')
            for r0 in range(0, N, 11):
                r1 = min(N, r0 + 11)
                bx, by = rows(r0, r1)
                assert bx.dtype == qx.dtype and by.dtype == qy.dtype
                assert np.array_equal(bx, qx[r0:r1]), (carrier, n1, r0)
                assert np.array_equal(by, qy[r0:r1]), (carrier, n1, r0)


def test_need_W_false_changes_nothing_the_caller_reads():
    """``_compute_carrier(need_W=False)`` skips the full-grid potential and
    returns ``None`` for it.  The GRADIENT closures must be unchanged bit for
    bit -- they are the only thing the seed reads."""
    N, dx = 64, 4.0e-6
    E = _field(N, dx, dtype=np.complex64, tilt=0.01)
    ax = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    for carrier in ('auto', 0.030, TiltedCarrier(R=0.05, L=0.02, M=-0.01)):
        W_a, g_a, w_a = _compute_carrier(carrier, E, LAM, dx, X, Y)
        W_b, g_b, w_b = _compute_carrier(carrier, E, LAM, dx, X, Y,
                                         need_W=False)
        assert W_a is not None and W_b is None, carrier
        La, Ma = g_a(X, Y)
        Lb, Mb = g_b(X, Y)
        assert np.array_equal(np.asarray(La), np.asarray(Lb)), carrier
        assert np.array_equal(np.asarray(Ma), np.asarray(Mb)), carrier
        # w_fn survives need_W=False -- it is a closure, not the grid
        assert np.array_equal(np.asarray(w_a(X, Y)),
                              np.asarray(w_b(X, Y))), carrier
        # and the coefficients themselves are untouched by the in-place
        # weighting of the design matrix
        assert np.array_equal(np.asarray(W_a),
                              np.asarray(_compute_carrier(
                                  carrier, E, LAM, dx, X, Y)[0])), carrier


# ---------------------------------------------------------------------------
# 3.  THE STREAMED TRANSFER FUNCTION
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('N', [64, 65, 96])
@pytest.mark.parametrize('cdtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('bandlimit', [True, False])
@pytest.mark.parametrize('z', [3.0e-3, -2.0e-3, 0.0])
def test_streamed_H_is_byte_identical_to_the_materialised_H(N, cdtype,
                                                            bandlimit, z):
    """Elementwise in (row, column), so the band width cannot matter -- but
    that is an argument, and this is the measurement of it.  ``z = 0`` is the
    exact-identity branch, which must not be routed through the band loop at
    all; negative ``z`` is the backward propagator; odd ``N`` is where the
    ``ifftshift`` of the frequency vectors could pick up a half-bin offset."""
    rng = np.random.default_rng(11)
    E = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(cdtype)
    kw = dict(wavelength=LAM, dx=1.2e-6, dy=1.45e-6, bandlimit=bandlimit)
    ref = angular_spectrum_propagate(E, z, **kw)
    got = angular_spectrum_propagate(E, z, stream_transfer_function=True, **kw)
    assert got.dtype == ref.dtype
    assert np.array_equal(ref.view(np.uint8), got.view(np.uint8)), (
        N, cdtype, bandlimit, z, float(np.max(np.abs(ref - got))))


def test_the_streamed_band_width_is_a_free_choice():
    """The saving depends on the band width; the ANSWER must not.  Sweep the
    module constant over three decades and require the bytes to be identical
    every time -- if any width changed a bit, the bit-identity claim above
    would be an accident of one default."""
    import lumenairy.propagators.asm as ASM
    N = 96
    rng = np.random.default_rng(5)
    E = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    kw = dict(wavelength=LAM, dx=1.2e-6, dy=1.2e-6, bandlimit=True)
    ref = angular_spectrum_propagate(E, 3.0e-3, **kw)
    prev = ASM._ASM_STREAM_BAND_ELEMS
    try:
        for elems in (1, N, N * 7, N * N, 1 << 22):
            ASM._ASM_STREAM_BAND_ELEMS = elems
            got = angular_spectrum_propagate(
                E, 3.0e-3, stream_transfer_function=True, **kw)
            assert np.array_equal(ref.view(np.uint8), got.view(np.uint8)), \
                elems
    finally:
        ASM._ASM_STREAM_BAND_ELEMS = prev


def test_streaming_is_declined_where_the_caller_wants_H_back():
    """``return_transfer_function=True`` asks for exactly the grid the lever
    avoids building, so the plain path is taken and H is still returned --
    silently degrading to ``None`` would be a contract break."""
    N = 64
    rng = np.random.default_rng(2)
    E = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    out, H = angular_spectrum_propagate(
        E, 3.0e-3, LAM, 1.2e-6, return_transfer_function=True,
        stream_transfer_function=True)
    assert H.shape == (N, N)
    ref, H_ref = angular_spectrum_propagate(
        E, 3.0e-3, LAM, 1.2e-6, return_transfer_function=True)
    assert np.array_equal(out.view(np.uint8), ref.view(np.uint8))
    assert np.array_equal(H.view(np.uint8), H_ref.view(np.uint8))


# ---------------------------------------------------------------------------
# 4.  THE STORE -- refusals, teardown, and that it actually moves the bytes
# ---------------------------------------------------------------------------
def test_warnings_still_point_at_the_callers_line_after_the_split():
    """The v5.40 split put the public wrapper between the caller and the body,
    which silently moved every warning's apparent origin one frame INTO the
    library -- a warning whose reported source is
    ``return _apply_real_lens_impl(...)`` tells you where the library called
    itself and nothing about your code.

    A DECISION, not a reading: the recorded warning's filename must be THIS
    test file.  Exercised through the aperture guard, which is the warning a
    user meets first and which routes through a helper with its own
    ``stacklevel``, so it pins both paths at once."""
    N, dx = 64, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    presc = _biconvex()          # 3 mm aperture on a 0.256 mm grid -> warns
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
    hits = [w for w in wl if 'aperture' in str(w.message)]
    assert hits, "the aperture guard did not fire; the fixture no longer warns"
    assert os.path.abspath(hits[0].filename) == os.path.abspath(__file__), (
        f"warning reported from {hits[0].filename}:{hits[0].lineno}, not from "
        f"the caller -- _WARN_STACKLEVEL is out of step with the wrapper")


def test_an_unknown_store_is_refused_by_name():
    N, dx = 32, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    with pytest.raises(ValueError, match='accumulator_store'):
        apply_real_lens(E, prescription=_biconvex(), wavelength=LAM, dx=dx,
                        accumulator_store='disk')


def test_memmap_is_refused_on_a_device_backend_rather_than_ignored():
    """A silent fall-back to RAM would make the preflight's memmap credit a
    lie, so the store refuses.  Exercised through ``bind`` because the device
    itself need not be present for the decision to be testable."""
    store = LR._AccumulatorStore('memmap')
    with pytest.raises(ValueError, match='NumPy backend'):
        store.bind(object())
    store.close()
    assert LR._AccumulatorStore('ram').bind(object()) is not None


def test_the_scratch_files_exist_while_the_call_runs_and_not_after(tmp_path):
    """Two claims, and the first is what makes the second meaningful: the
    store must actually CREATE files (a teardown test on a store that never
    allocated is vacuous), and none may survive the call."""
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    seen = []

    def spy(*a, **k):
        seen.append(len(glob.glob(os.path.join(str(tmp_path), '*'))))

    apply_real_lens(E, prescription=_triplet(), wavelength=LAM, dx=dx,
                    slant_correction=False, surface_model='tangent_facet',
                    carrier='auto', screen_obliquity=False,
                    on_screen_obliquity='silent', sag_chunk_rows=11,
                    accumulator_store='memmap', scratch_dir=str(tmp_path),
                    progress=spy)
    assert max(seen) >= 2, (
        f"the store never allocated ({seen}); this teardown test would be "
        f"vacuous")
    assert glob.glob(os.path.join(str(tmp_path), '*')) == []


def test_the_scratch_footprint_is_bounded_by_what_is_LIVE(tmp_path):
    """A store that only released at ``close()`` would grow to every
    accumulator the call ever made, not the handful that are live.

    The tangent-facet path allocates a FRESH destination pair per surface and
    a fresh pair per gap transport and rebinds the accumulator to it; the
    previous pair is garbage immediately.  On a three-surface group that is
    twelve mappings over the call against four or so live -- at N = 32768,
    ~103 GB of scratch instead of ~34.  The store reaps each mapping when its
    view is collected, so the directory stays bounded.

    The bar is DERIVED, not measured: the route can hold the momentum pair,
    the destination pair and (for the remap rung) the walk pair at once, so
    six is the structural maximum; eight leaves a factor to spare and is still
    far below the twelve a release-at-close store would reach.
    """
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    seen = []

    def spy(*a, **k):
        seen.append(len(glob.glob(os.path.join(str(tmp_path), '*'))))

    apply_real_lens(E, prescription=_triplet(), wavelength=LAM, dx=dx,
                    slant_correction=False, surface_model='tangent_facet',
                    carrier='auto', screen_obliquity=False,
                    on_screen_obliquity='silent', sag_chunk_rows=11,
                    accumulator_store='memmap', scratch_dir=str(tmp_path),
                    progress=spy)
    assert max(seen) >= 1, f"the store never allocated ({seen})"
    assert max(seen) <= 8, (
        f"scratch files peaked at {max(seen)} on a 3-surface group; the "
        f"per-view reaping is not releasing rebound accumulators")
    assert glob.glob(os.path.join(str(tmp_path), '*')) == []


def test_the_scratch_files_are_removed_when_the_call_raises(tmp_path):
    """The refusal path is the one a ``finally`` exists for.  The folding
    prescription raises mid-prescription, after the store has allocated."""
    dx, presc = _folding()
    E = _field(129, dx, dtype=np.complex64)
    with pytest.raises(ValueError, match='folds'):
        apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                        slant_correction=False,
                        surface_model='tangent_facet_remap', carrier='auto',
                        screen_obliquity=False, on_screen_obliquity='silent',
                        sag_chunk_rows=17, accumulator_store='memmap',
                        scratch_dir=str(tmp_path))
    assert glob.glob(os.path.join(str(tmp_path), '*')) == []


def test_a_private_scratch_directory_leaves_no_data_behind(tmp_path):
    """``scratch_dir=None`` makes its own temporary directory and clears it.

    THE ASSERTION IS ON THE DATA, NOT ON THE INODE, and that is deliberate.
    The accumulator FILES are always unlinked -- that is the library's
    decision and it is what "no data left behind" means.  Whether the empty
    DIRECTORY has also disappeared by the time this line runs is a Windows
    scheduling artefact: a just-unlinked entry lingers in a pending-delete
    state, ``rmdir`` fails on a directory ``listdir`` already reports as
    empty, and how long that lasts depends on what else the box is doing
    (observed passing 8/8 idle and failing under a concurrent 16 GB job).
    Asserting the directory is gone would be asserting the machine's load.

    So the claim is two-sided and build-free: **no file survives**, and any
    surviving directory is EMPTY and was reported through a warning rather
    than left silently."""
    import tempfile
    N, dx = 96, 4.0e-6
    E = _field(N, dx, dtype=np.complex64)
    # HERMETIC: point ``tempfile`` at this test's own directory.  Globbing the
    # real ``%TEMP%`` would also see the private directories of any CONCURRENT
    # process using this feature -- which is not a hypothetical, it is how this
    # test first failed (a two-tree sweep running beside it), and a test that
    # reads another process's live state is testing the box, not the library.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    pat = os.path.join(str(tmp_path), 'lumenairy_accum_*')
    before = set(glob.glob(pat))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        apply_real_lens(E, prescription=_triplet(), wavelength=LAM, dx=dx,
                        slant_correction=False, surface_model='tangent_facet',
                        carrier='auto', screen_obliquity=False,
                        on_screen_obliquity='silent', sag_chunk_rows=11,
                        accumulator_store='memmap')
    new = set(glob.glob(pat)) - before
    for d in new:
        assert os.listdir(d) == [], (
            f"accumulator DATA left behind in {d}: {os.listdir(d)[:3]}")
    if new:
        # An empty directory may only survive if the library said so.
        assert any('scratch directory' in str(w.message) for w in wl), (
            f"{len(new)} empty scratch directory/ies survived with no "
            f"warning; the store is not reporting what it could not remove")
        for d in new:                       # do not litter the next test
            try:
                os.rmdir(d)
            except OSError:
                pass
    monkeypatch.undo()


def test_closing_the_store_twice_is_harmless():
    store = LR._AccumulatorStore('memmap')
    a = store.zeros((8, 8), np.float64)
    assert isinstance(a, np.ndarray) and type(a) is np.ndarray, (
        "the store must hand out a BASE-CLASS view, or a ufunc could "
        "dispatch on np.memmap and produce a differently-typed result")
    del a
    store.close()
    store.close()


def test_the_store_moves_the_accumulators_off_the_python_heap():
    """The claim the memmap lever makes, stated as a DECISION with a derived
    bar rather than as a measured number.

    ``tracemalloc`` counts Python allocations; a mapped page is not one.  So
    if the accumulators really move, the tracemalloc peak must fall by AT
    LEAST the size of the accumulators the run is known to hold, and the two
    outputs must still be byte-identical (asserted elsewhere).

    The bar is DERIVED from the route rather than measured: route 3 with a
    non-collimated carrier holds the momentum pair AND the fresh destination
    pair -- four ``float64`` grids of ``8 * Ny * Nx`` bytes -- simultaneously
    live at the moment a surface's destination is complete.  Requiring a fall
    of at least TWO of those four (a 2x margin on the claim, so a change in
    which pair happens to dominate cannot flip the test) is far outside any
    build-to-build allocator variation, which moves kilobytes, not
    hundred-megabyte grids.
    """
    import gc
    import tracemalloc
    N, dx = 512, 1.0e-6
    grid = 8.0 * N * N
    E = _field(N, dx, dtype=np.complex64)
    presc = _triplet()
    kw = dict(prescription=presc, wavelength=LAM, dx=dx,
              slant_correction=False, surface_model='tangent_facet',
              carrier=0.030, screen_obliquity=False,
              on_screen_obliquity='silent', sag_chunk_rows=64)

    def peak(**extra):
        apply_real_lens(E, **kw, **extra)          # warm
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        apply_real_lens(E, **kw, **extra)
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return pk

    ram = peak()
    mm = peak(accumulator_store='memmap')
    assert (ram - mm) >= 2.0 * grid, (
        f"memmap saved only {(ram - mm) / grid:.2f} float64 grids of Python "
        f"heap; route 3 with a carrier holds four, and the bar is two")
