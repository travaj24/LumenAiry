"""Audit W9 -- same-process byte-determinism of the traced-lens pipeline.

THE DEFECT (v5.30 and earlier).  ``apply_real_lens_traced`` on ONE FIXED
input returned one bit pattern for its first two calls and a DIFFERENT one
(max|d| ~ 2.8e-15) for every call after that, in a fresh process, with no
input changing.  All later calls agreed with each other, so the pipeline was
not "noisy" -- it *switched regimes* partway through the session.

ROOT CAUSE.  ``fft_infra`` shipped ESTIMATE -> MEASURE plan auto-promote ON
by default (4.12).  Every ``(direction, shape, dtype, threads)`` plan-cache
entry counts its calls, and on the 5th the plan is rebuilt with
``FFTW_MEASURE``.  One traced-lens call performs exactly four transforms at
one 256^2 key (fwd, inv, fwd, inv), so the counter reads 2 after call 0, 4
after call 1, and trips at 5 *inside* call 2 -- exactly the observed
boundary.  Two things made this worse than an ULP wobble:

  * the counter is GLOBAL per key, so an unrelated earlier caller at the
    same shape moves the boundary.  That is how this first surfaced: a
    byte-identity pin in ``test_niche_s12_remap_sampling.py`` failed on one
    pytest collection layout and passed on three others.
  * ``FFTW_MEASURE`` chooses its algorithm by TIMING candidate plans, so the
    winner depends on machine noise.  Measured: four fresh processes gave
    four DIFFERENT post-promotion results where ESTIMATE gave one.  The
    library was therefore not run-to-run reproducible either.

THE FIX (v5.30.1).  Auto-promote is now OPT-IN
(``_PYFFTW_AUTO_PROMOTE_SHIPPED = False``).  Only ``FFTW_ESTIMATE`` is a
deterministic planner, and neither result is more accurate, so the default
buys reproducibility.  The throughput (measured 1.4x @256^2 .. 4.6x @4096^2)
is one call away via ``set_pyfftw_planner('FFTW_MEASURE')``, which plans at
FIRST use and so is itself byte-consistent within a process.

These pins fence all four legs: the shipped default, the traced pipeline's
call-to-call byte identity at that default, the ``memory.py`` companion
lock, and -- so nobody "fixes" a future flake by deleting the feature --
that the promote machinery still works when explicitly opted in.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.memory import _LOW_MEMORY_SHIPPED_DEFAULTS
from lumenairy.propagators import fft_infra as FI

LAM = 1.31e-6


@pytest.fixture(autouse=True)
def _shipped_fft_planner_state(shipped_fft_dispatch):
    """Run every test from the SHIPPED planner configuration and restore.

    2026-08-01: also takes ``shipped_fft_dispatch`` (tests/conftest.py).
    This fixture already owned the planner flag, auto-promote and wisdom;
    what it did NOT own is the rest of the dispatch configuration --
    ``USE_PYFFTW``, ``FFTW_MIN_SIZE``, ``PYFFTW_FALLBACK_ON_ERROR``, the
    double-buffer mode and the plan-cache size.  With the pyFFTW path
    switched off (or ``FFTW_MIN_SIZE`` raised) by an earlier test in the
    shard, ``_fft2`` never reaches the plan cache at all and
    ``test_auto_promote_still_promotes_when_opted_in`` reads ``None`` for
    every flag -- passing alone, failing in a full sweep.  See that
    fixture's docstring.

    Applies ``_PYFFTW_AUTO_PROMOTE_SHIPPED`` (not a hardcoded ``False``) so
    the byte-identity tests below still fail if the shipped default ever
    regresses to ``True``, while staying immune to whatever an earlier test
    in the shard left behind.  ``set_pyfftw_planner`` clears the plan cache,
    which also discards any entry an earlier test already promoted.

    It also snapshots and restores FFTW **wisdom**, which the plan cache
    does NOT own.  Wisdom is process-global inside libfftw3 and sticky:
    once a MEASURE plan exists for a problem size, later ESTIMATE plans at
    that size reuse the wisdom-recorded algorithm and produce different
    bits than they would in a clean process (measured, audit W9).  Two
    tests here deliberately plan under MEASURE, so without this the file
    would silently perturb every later 256^2 transform in the same pytest
    worker -- re-creating the collection-order coupling this module exists
    to eliminate.
    """
    prev_promote = la.get_fft_auto_promote()
    prev_planner = la.get_pyfftw_planner()
    wisdom = None
    if FI.PYFFTW_AVAILABLE and FI._ensure_pyfftw_loaded():
        wisdom = FI.pyfftw.export_wisdom()
    la.set_pyfftw_planner('FFTW_ESTIMATE')
    la.set_fft_auto_promote(FI._PYFFTW_AUTO_PROMOTE_SHIPPED)
    try:
        yield
    finally:
        la.set_fft_auto_promote(prev_promote)
        la.set_pyfftw_planner(prev_planner)
        if wisdom is not None:
            FI.pyfftw.forget_wisdom()
            FI.pyfftw.import_wisdom(wisdom)


def _presc():
    """Strong two-surface N-BK7 singlet -- the s12 stress prescription."""
    surfaces = [
        {'radius': 3.1e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': -3.1e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': 'strong', 'aperture_diameter': 1.2e-3,
            'surfaces': surfaces, 'thicknesses': [1.0e-3]}


def _field(n=256, dx=4.0e-6, w=200e-6, a=6.0, rc=-0.02):
    x = (np.arange(n) - n // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k = 2.0 * np.pi / LAM
    sag = np.sign(rc) * (np.sqrt(r2 + rc ** 2) - abs(rc))
    return (np.exp(-r2 / w ** 2) * np.exp(1j * k * sag)
            * np.exp(1j * a * (r2 / w ** 2) ** 2)).astype(np.complex128)


def _traced(E, dx=4.0e-6, rc=-0.02):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E, prescription=_presc(), wavelength=LAM, dx=dx, carrier=rc,
            amplitude_model='ray_density', preserve_input_phase=True,
            parallel_amp=False, on_undersample='silent',
            on_noncollimated='silent', ray_subsample=4,
            remap_sampling='lattice'))


# ---------------------------------------------------------------------------
# 1. the shipped contract
# ---------------------------------------------------------------------------

def test_auto_promote_ships_off():
    """Auto-promote is OPT-IN as of v5.30.1.

    Source-declared constant, so this is immune to test ordering: it pins
    the shipped contract, not the live process state.
    """
    assert FI._PYFFTW_AUTO_PROMOTE_SHIPPED is False, (
        "ESTIMATE->MEASURE auto-promote must ship OFF: it swaps the plan "
        "mid-session (changing results for one fixed input partway through "
        "a run) and FFTW_MEASURE picks its algorithm by timing, so it is "
        "not reproducible run-to-run either. See the module docstring.")


def test_live_default_matches_shipped_default():
    """A fresh process starts at the shipped value (the autouse fixture
    reasserts it, so this also proves the setter round-trips)."""
    assert la.get_fft_auto_promote() is FI._PYFFTW_AUTO_PROMOTE_SHIPPED


def test_low_memory_shipped_default_is_companion_locked():
    """``memory.py``'s restore table must track the fft_infra default.

    ``set_low_memory(False)`` with no enable on record restores from this
    table, so a stale ``True`` would silently opt a caller INTO the
    non-reproducible planner -- the exact defect, re-entered by the back
    door.
    """
    assert (_LOW_MEMORY_SHIPPED_DEFAULTS['fft_auto_promote']
            is FI._PYFFTW_AUTO_PROMOTE_SHIPPED), (
        "memory._LOW_MEMORY_SHIPPED_DEFAULTS['fft_auto_promote'] drifted "
        "from fft_infra._PYFFTW_AUTO_PROMOTE_SHIPPED; update both together.")


# ---------------------------------------------------------------------------
# 2. the regression fence: the reported defect
# ---------------------------------------------------------------------------

def test_traced_lens_is_byte_identical_across_calls():
    """12 identical calls, no warm-up, all byte-identical.

    Pre-fix this failed at call 2 with max|d| ~ 2.8e-15 (12 calls x 4
    transforms trips the 5-call promote threshold inside call 2).  There is
    deliberately NO warm-up fixture here -- warming up would hide exactly
    the regression this pins.
    """
    E = _field()
    ref = _traced(E)
    for i in range(1, 12):
        out = _traced(E)
        assert np.array_equal(out, ref), (
            f"apply_real_lens_traced call {i} is not byte-identical to call "
            f"0 on a FIXED input (max|d| = {np.max(np.abs(out - ref)):.3e}). "
            f"Same-process byte-determinism is a house contract; several "
            f"pins assert np.array_equal across repeat calls.")


def test_repeat_calls_survive_a_foreign_caller_at_the_same_shape():
    """The regime switch used to be reachable from an UNRELATED caller.

    The plan-cache call counter is keyed on (direction, shape, dtype,
    threads) only, so plain ASM propagations at the same 256^2 shape used
    to advance the traced pipeline's counter and move its boundary --
    making byte-identity pins fail as a function of pytest collection
    order.  Interleaving the two must now be inert.
    """
    E = _field()
    ref = _traced(E)
    probe = _field(w=120e-6)
    for i in range(1, 6):
        # a foreign consumer of the very same plan key
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.angular_spectrum_propagate(
                probe, z=1e-3, wavelength=LAM, dx=4.0e-6)
        out = _traced(E)
        assert np.array_equal(out, ref), (
            f"a foreign ASM caller at the same 256^2 plan key perturbed "
            f"traced-lens repeat {i} (max|d| = "
            f"{np.max(np.abs(out - ref)):.3e})")


# ---------------------------------------------------------------------------
# 3. the opt-in routes still work (and are themselves in-process stable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FI.PYFFTW_AVAILABLE, reason="requires pyFFTW")
def test_auto_promote_still_promotes_when_opted_in():
    """Machinery meta-pin: the feature is opt-in, not deleted.

    Structural assert on the cache entry's planner flag rather than on
    output bits -- MEASURE *may* happen to pick the ESTIMATE algorithm, so
    a bit-difference assert would be a flake.
    """
    la.set_fft_auto_promote(True)
    # shape[0] must clear FFTW_MIN_SIZE or _fft2 never reaches the plan
    # cache at all (it falls back to scipy/numpy and this pins nothing).
    shape = (FI.FFTW_MIN_SIZE, FI.FFTW_MIN_SIZE)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape)
         + 1j * rng.standard_normal(shape)).astype(np.complex128)

    def _fwd_flag():
        """Planner flag of the fwd entry at this shape (scan, don't
        reconstruct the key -- the threads component is dispatch-internal)."""
        for k, entry in FI._PYFFTW_PLAN_CACHE.items():
            if k[0] == 'fwd' and k[1] == shape:
                return entry['flag']
        return None

    flags = []
    for _ in range(FI._PYFFTW_AUTO_PROMOTE_THRESHOLD + 1):
        FI._fft2(x)
        flags.append(_fwd_flag())
    assert flags[0] == 'FFTW_ESTIMATE', (
        f"first call should plan under ESTIMATE, got {flags[0]!r} "
        f"(cache keys: {list(FI._PYFFTW_PLAN_CACHE)})")
    assert flags[-1] == 'FFTW_MEASURE', (
        f"opting in must still promote by call "
        f"{FI._PYFFTW_AUTO_PROMOTE_THRESHOLD}; flags were {flags}")


@pytest.mark.skipif(not FI.PYFFTW_AVAILABLE, reason="requires pyFFTW")
def test_explicit_measure_planner_is_stable_from_the_first_call():
    """The recommended fast path is in-process byte-consistent.

    ``set_pyfftw_planner('FFTW_MEASURE')`` plans every key at FIRST use and
    never swaps it, so unlike auto-promote there is no mid-session regime
    change -- call 0 already agrees with call N.  (Run-to-run
    reproducibility is still traded away; that is inherent to a
    timing-based planner and is now an explicit, informed opt-in.)
    """
    la.set_pyfftw_planner('FFTW_MEASURE')
    E = _field()
    ref = _traced(E)
    for i in range(1, 4):
        out = _traced(E)
        assert np.array_equal(out, ref), (
            f"explicit-MEASURE call {i} diverged from call 0 (max|d| = "
            f"{np.max(np.abs(out - ref)):.3e}); the opt-in fast path must "
            f"not swap plans mid-session")
