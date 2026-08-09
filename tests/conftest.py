"""Shared pytest fixtures for the lumenairy unit-test suite.

Provides small grids, standard wavelengths, and minimal optical
elements so individual unit-test modules don't repeat the same setup.

Design goals
------------
* **Fast**: every fixture targets N=64 by default so a unit-test
  module finishes in under a second per file.
* **No external deps**: nothing here requires Zemax, rayoptics,
  Optiland, h5py, or matplotlib.  Unit tests should run on a fresh
  checkout with only the base library dependencies.
* **Standard SI units everywhere**: meters, radians, real Hz/m
  spatial frequencies.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

import lumenairy as la

# ---------------------------------------------------------------------------
# BLAS THREADING -- read this before "fixing" the per-file guards
# ---------------------------------------------------------------------------
# 282 files in ``tests/unit`` open with some form of::
#
#     for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
#         os.environ.setdefault(_v, "1")
#
# Under pytest every one of those is INERT.  pytest imports THIS conftest
# before any test module, and the ``import numpy`` / ``import lumenairy``
# above have already made OpenBLAS read its thread count from the environment
# by the time a test module's guard runs (AUDIT_CI_TEST_TIME_2026_08_03 §1.1,
# reproduced with ``threadpoolctl.threadpool_info()`` before and after).  They
# only ever do anything when a file is executed as a script.
#
# The pin is therefore made where it actually takes effect -- in the workflow
# ``env:`` block, BEFORE the interpreter starts.  ``slow-tests`` pins (the
# eig-heavy EME convergence tests were flake-prone under multi-threaded BLAS
# reduction order) and ``jax-unit`` pins (JAX + multi-threaded OpenBLAS
# deadlock on the first large ``lstsq``).  The fast ``unit`` matrix does NOT,
# deliberately: §1.2 of the same audit measured single-threading to be up to
# ~29x FASTER on many-small-solve files but 2.76x SLOWER on the large-eig
# convergence files, netting +27% WORSE across one 75-file range -- so a
# blanket pin is a real trade, not a free win, and it belongs behind a
# CI-hardware A/B rather than a copy-pasted env block.
#
# A NO-OP GUARD IS NOT A PIN: do not read the per-file blocks as evidence that
# a given lane runs single-threaded.  Check the workflow.  To pin locally,
# set the variables in the shell that launches pytest.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grid / wavelength / k0
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def N_small() -> int:
    """Small grid dimension for fast unit tests (64x64)."""
    return 64


@pytest.fixture(scope='session')
def N_med() -> int:
    """Medium grid dimension when 64x64 is too coarse (128x128)."""
    return 128


@pytest.fixture(scope='session')
def dx_m() -> float:
    """Standard test grid spacing (5 microns)."""
    return 5e-6


@pytest.fixture(scope='session')
def wavelength_m() -> float:
    """Standard test wavelength (1.31 microns, telecom O-band)."""
    return 1.31e-6


@pytest.fixture(scope='session')
def k0(wavelength_m) -> float:
    """Wave-number magnitude k = 2*pi / lambda."""
    return 2.0 * np.pi / wavelength_m


# ---------------------------------------------------------------------------
# Field fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plane_wave(N_small) -> np.ndarray:
    """Unit-amplitude plane wave on the small grid."""
    return np.ones((N_small, N_small), dtype=np.complex128)


@pytest.fixture
def gaussian_beam(N_small, dx_m, wavelength_m) -> np.ndarray:
    """Tiny Gaussian beam: sigma = 30 microns, peak-normalized.

    ``create_gaussian_beam`` returns ``(E, x, y)``; the fixture
    unpacks the field for tests that only need the array.
    """
    E, _x, _y = la.create_gaussian_beam(N_small, dx_m, wavelength_m,
                                          w0=(30e-6) * np.sqrt(2))
    return E


# ---------------------------------------------------------------------------
# FFT dispatch isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def shipped_fft_dispatch():
    """Run a test from the SHIPPED FFT-dispatch configuration, then restore
    whatever the process had before.

    Opt-in (NOT autouse): request it from a module's own autouse fixture.

    WHY (release verification 2026-08-01).  The pyFFTW plan-cache pins --
    ``test_perf_v4_12_0_fft_infra.py::TestAutoPromote`` and
    ``test_niche_audit_w9_traced_determinism.py::
    test_auto_promote_still_promotes_when_opted_in`` -- all assert that the
    FIRST plan built at a key is ``FFTW_ESTIMATE`` and that a cache entry
    exists at all.  Both of those depend on process-global dispatch state
    that neither module's own fixture owned:

      * ``USE_PYFFTW`` -- ``False`` (e.g. left behind by any consumer of the
        UI dock's backend selector, which sets it unconditionally and only
        re-raises it for ``backend == 'pyfftw'``) makes ``_fft2`` skip the
        plan cache entirely, so the probed entry is ``None``;
      * ``FFTW_MIN_SIZE`` -- raised above the test's N has the same effect,
        and it is the ONE dispatch global ``snapshot_fft_state`` does not
        carry;
      * ``_PYFFTW_PLAN_FLAGS`` -- left at ``FFTW_MEASURE`` makes the first
        plan MEASURE, so "starts at ESTIMATE" fails;
      * ``PYFFTW_FALLBACK_ON_ERROR`` / ``_PYFFTW_DOUBLE_BUFFER`` /
        ``_PYFFTW_PLAN_CACHE_SIZE`` -- reachable via ``set_low_memory``.

    Each of those makes the pins pass alone and fail in a full sweep.  This
    fixture removes the coupling by CONSTRUCTION rather than by chasing the
    polluter: it forces every one of them to its shipped value, clears the
    plan cache and the bad-shape blacklist, and restores the caller's state
    (including ``FFTW_MIN_SIZE``, by hand) on the way out.

    It deliberately does NOT touch libfftw3 *wisdom* -- that is
    process-global inside the C library, affects bits rather than the plan
    flags asserted here, and the w9 module already snapshots it.
    """
    with shipped_fft_dispatch_state():
        yield


@contextlib.contextmanager
def shipped_fft_dispatch_state():
    """The body of :func:`shipped_fft_dispatch`, usable from a MODULE-scoped
    fixture too (niche C11).

    A function-scoped fixture cannot protect a module-scoped one: pytest builds
    the higher scope FIRST, so a module fixture that runs chains (e.g. niche
    D4's ``runs``) computes them BEFORE any function-scoped isolation is
    active.  Modules with expensive module-scoped chain fixtures must wrap
    those with this context manager instead.
    """
    from lumenairy.propagators import fft_infra as _fi

    state = _fi.snapshot_fft_state()
    prev_min_size = _fi.FFTW_MIN_SIZE
    prev_planner = _fi.get_pyfftw_planner()
    _fi.USE_PYFFTW = _fi.PYFFTW_AVAILABLE     # shipped: on iff importable
    _fi.USE_SCIPY_FFT = True                  # shipped
    _fi.PYFFTW_FALLBACK_ON_ERROR = True       # shipped
    _fi.FFTW_MIN_SIZE = 256                   # shipped
    if not _fi._PYFFTW_DOUBLE_BUFFER:         # shipped (clears plans on flip)
        _fi.set_fft_double_buffer(True)
    _fi.set_fft_plan_cache_size(8)            # shipped
    _fi.set_pyfftw_planner('FFTW_ESTIMATE')   # shipped; clears the plan cache
    _fi.reset_fft_backend()                   # + bad shapes, + call counters
    try:
        yield
    finally:
        _fi.FFTW_MIN_SIZE = prev_min_size
        _fi.restore_fft_state(state)
        _fi.set_pyfftw_planner(prev_planner)
        _fi.reset_fft_backend()


# ---------------------------------------------------------------------------
# Prescription fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def singlet_prescription():
    """Minimal singlet prescription dict (N-BK7 plano-convex,
    R1=25mm/R2=inf, 2.5mm thick, 5mm aperture)."""
    return la.make_singlet(
        R1=25e-3, R2=np.inf, d=2.5e-3,
        glass='N-BK7', aperture=5e-3,
    )


@pytest.fixture(scope='session')
def doublet_prescription():
    """Minimal cemented N-BK7 + N-SF2 doublet."""
    return la.make_doublet(
        R1=25e-3, R2=-20e-3, R3=-50e-3,
        d1=2.5e-3, d2=1.5e-3,
        glass1='N-BK7', glass2='N-SF2',
        aperture=5e-3,
    )


# ===========================================================================
# Niche C11 (2026-08-03) -- MODULE-FLAG LEAK GUARD
# ===========================================================================
# THE DEFECT THIS KILLS.  Most of this library's physics modes are module-level
# flags (``TILTED_CARRIER_EXACT_EIKONAL``, ``REMAP_STATIONARY_PHASE_LAUNCH``,
# ``SPHERE_PARAB_CONVERSION_EXACT``, ...), and dozens of tests legitimately
# toggle them to reach a fail-before arm.  Every one of those sites is supposed
# to restore in a ``finally``; a single site that does not leaves the flag
# dirty for every LATER test in the same process.
#
# That failure is invisible in isolation and invisible under a different shard
# layout -- the leaked state only reaches a victim when pytest-split happens to
# put the two in the same group.  It presents as a physics-mode-sized delta in
# an unrelated file (measured: 0.0661 against a 1e-4 bar) or as a
# ``DID NOT WARN`` in a guard test, on some shards and some Python versions
# only, which is the most expensive shape of CI failure this project has.
#
# The guard snapshots every known flag before each test and restores it after,
# so a leak cannot cross a test boundary.  Restoring SILENTLY is deliberate:
# the point is to make the suite order-independent, not to red it.  Set
# ``LUMEN_TEST_FLAG_LEAK_STRICT=1`` to FAIL the leaking test instead, which is
# how you find the culprit once you know the class is live.
#
# The flag set is DISCOVERED, not enumerated: every module-level scalar
# (bool / int / float / str / None) whose name is upper-case -- the convention
# this library uses for its mode flags and calibration constants -- in the two
# modules that carry physics modes.  A hand-written list is itself a defect
# surface: it silently stops covering a flag the day someone adds one, which is
# exactly the class being closed here.
#
# Cost: two ``vars()`` scans (~200 names) per test, no imports, no I/O.
# ``fft_infra`` is in this list for a REASON that cost a CI round: it carries
# ``DEFAULT_WAVE_PROPAGATOR`` (a physics mode -- 'asm' vs anything else changes
# every propagation in the process), ``DEFAULT_DY``, and the whole pyFFTW
# dispatch set (``USE_PYFFTW``, ``FFTW_MIN_SIZE``, ``_PYFFTW_PLAN_FLAGS``, the
# auto-promote counters).  The first version of this guard discovered flags in
# the two PHYSICS modules only, which is exactly why it did not catch a
# dispatch-global leak -- ``lumenairy/ui/waveoptics_dock.py`` clears
# ``USE_PYFFTW`` unconditionally, and ``shipped_fft_dispatch`` is opt-in.
_LEAK_GUARD_MODULES = (
    'lumenairy.elements._lens_traced',
    'lumenairy.propagators.carrier',
    'lumenairy.propagators.fft_infra',
)
#
# ``type`` and ``np.dtype`` are in the type filter for a REASON
# (2026-08-09, docs/audits/FIX_STEERING_FAMILY_2026_08_09.md).  The
# scalar-only filter silently EXCLUDED the two dtype steering knobs
# ``fft_infra.DEFAULT_REAL_DTYPE`` and ``fft_infra.DEFAULT_COMPLEX_DTYPE``
# -- they ship as the numpy scalar TYPES ``np.float64`` / ``np.complex128``
# (``isinstance(np.float64, (bool, int, float, str, NoneType))`` is False),
# and ``set_default_real_dtype`` normalises them to ``np.dtype`` INSTANCES,
# which is a third class the filter also missed.  So a test that called
# ``set_default_real_dtype(np.float32)`` and forgot to put it back handed
# every LATER test in the process a float32 OPL accumulator, and the guard
# whose whole job is to stop exactly that never looked.  Both spellings
# are listed because either can be the value in flight: the SHIPPED value
# is a type, every value that has been through a setter is a dtype.
#
# This does NOT make the guard trigger-happy about the type-vs-dtype
# SPELLING: the comparison below falls back to ``==``, and
# ``np.dtype('float64') == np.float64`` is True, so the benign
# renormalisation a save/restore round-trip performs is tolerated while a
# real float32/complex64 leak is caught and reverted.  Measured against
# the three guarded modules, adding these two admits the two dtype knobs
# and nothing else.
_LEAK_GUARD_TYPES = (bool, int, float, str, type(None), type, np.dtype)


def _leak_guard_snapshot():
    """``{(module, name): (module_object, value)}`` for every module-level
    scalar mode flag / calibration constant in the physics modules."""
    import importlib
    snap = {}
    for mod_name in _LEAK_GUARD_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for n, v in list(vars(mod).items()):
            if n.startswith('__'):
                continue
            # upper-case names only: the library's own convention for a knob.
            # ``_FOO_BAR`` counts, ``_foo`` and ``SomeClass`` do not.
            core = n.lstrip('_')
            if not core or not core.isupper():
                continue
            if isinstance(v, _LEAK_GUARD_TYPES):
                snap[(mod_name, n)] = (mod, v)
    return snap


# MODULE scope (2026-08-03): per-TEST restore erased module-scoped
# fixtures' legitimate flag setup mid-module -- the D4 hand-split test
# failed on MORE CI shards after the per-test guard landed.  Module
# scope still kills cross-module order dependence (the leak class)
# while letting a module's own fixtures mean what they say.
@pytest.fixture(autouse=True, scope='module')
def _module_flag_leak_guard():
    """Restore every physics-mode flag after each test (niche C11).

    Runs OUTSIDE the test's own fixtures (autouse fixtures are set up first and
    torn down last), so a ``monkeypatch`` undo has already happened by the time
    this checks -- what it sees is genuine leakage, not a pending restore.
    """
    import os
    before = _leak_guard_snapshot()
    yield
    leaked = []
    # The kwarg-defaults cache is introspection, not physics, but it is a
    # module-level CONTAINER and the guard restores scalars only -- so it is
    # cleared rather than compared.  Cheap (it is rebuilt lazily per signature).
    try:
        from lumenairy.elements import _lens_traced as _lt_mod
        _lt_mod._TRACED_KWARG_DEFAULTS_CACHE.clear()
    except Exception:
        pass
    for (mod_name, n), (mod, val) in before.items():
        now = getattr(mod, n, val)
        same = now is val
        if not same:
            try:
                same = bool(now == val)
            except Exception:
                same = False
        if not same:
            leaked.append(f'{mod_name}.{n}: {val!r} -> {now!r}')
            setattr(mod, n, val)
    if leaked and os.environ.get('LUMEN_TEST_FLAG_LEAK_STRICT'):
        raise AssertionError(
            'this test leaked module-level physics flags to every LATER test '
            'in the process (niche C11 leak guard): ' + '; '.join(leaked))


# ===========================================================================
# GLASS-REGISTRY LEAK GUARD (2026-08-06)
# ===========================================================================
# THE DEFECT THIS KILLS.  ``lumenairy.glass`` keeps its material tables --
# ``GLASS_REGISTRY``, ``SELLMEIER_COEFFICIENTS``, ``GLASS_VALIDITY`` -- as
# process-global dicts, and ~30 test modules need a dispersionless MODEL glass
# so their oracle is a closed-form number rather than a Sellmeier fit.  Every
# one of them used to write that glass straight into ``GLASS_REGISTRY`` AT
# MODULE SCOPE::
#
#     GLASS_REGISTRY['_G1CACHE'] = lambda wl: 1.5168     # never removed
#
# pytest imports every selected test module during COLLECTION, before the first
# test runs, so one such line poisons the tables for the whole session -- and a
# lambda does not pickle.  ``tests/unit/test_niche_d8_congruence_workers.py``
# asserts that a CLEAN worker-state snapshot has nothing unpicklable in it
# (a model glass cannot cross a process boundary, so the parallel path must
# degrade to serial and say so).  That assertion is right; what it was handed
# was not clean, so ``test_snapshot_is_picklable`` and
# ``test_a_clean_glass_snapshot_reports_nothing_unpicklable`` passed alone and
# failed in the full-suite order -- the same shape as the ``USE_PYFFTW`` leak
# fixed in 5.32.1, and the same class the niche C11 flag guard above closes for
# scalar mode flags.
#
# THE FIX IS OWNERSHIP, NOT DETECTION.  A module that needs a model glass
# declares it and does not register it::
#
#     MODULE_GLASSES = {'_G1CACHE': lambda wl: 1.5168}
#
# ``_module_glass_registry_guard`` below installs those entries for the length
# of that module and removes them afterwards, so the glass exists exactly where
# it is used and nowhere else.  The guard ALSO restores any mutation a test
# body or fixture made and forgot to undo, which is the same defect one scope
# down.
#
# Deliberately ONE shared fixture keyed off a module-level name rather than 30
# copies of a per-module fixture: a copied fixture is a copied defect surface,
# and the next module to need a model glass should have to write one dict entry
# and get the cleanup for free.  Restoring SILENTLY matches the flag guard --
# the goal is an order-independent suite, not a red one.  Set
# ``LUMEN_TEST_GLASS_LEAK_STRICT=1`` to FAIL the leaking module instead.
_GLASS_TABLE_NAMES = ('GLASS_REGISTRY', 'SELLMEIER_COEFFICIENTS',
                      'GLASS_VALIDITY')


def _glass_tables():
    """``[(name, dict)]`` for every runtime-mutable material table.

    Fetched live, never cached: other modules hold these dicts BY REFERENCE
    (``from lumenairy.glass import GLASS_REGISTRY``), so the objects must be
    mutated in place and never rebound.
    """
    from lumenairy import glass as _g
    out = []
    for n in _GLASS_TABLE_NAMES:
        t = getattr(_g, n, None)
        if isinstance(t, dict):
            out.append((n, t))
    return out


def _glass_snapshot():
    """``{table_name: {key: value}}`` -- a shallow copy, so the recorded values
    are the SAME objects and identity is a valid "was this rebound?" test."""
    return {n: dict(t) for n, t in _glass_tables()}


def _glass_owner(val):
    """Best-effort "who defined this?" for a registry value.

    A model glass is a callable, so ``__module__`` names the test module that
    created it exactly.  Catalogue entries are strings/tuples and have none.
    """
    return getattr(val, '__module__', None) or '<not a callable>'


def _is_library_user_fixed(val):
    """True for a ``('__user__', '__fixed__', '__fixed__')`` dispatch entry.

    THE ONE THING THESE GUARDS MUST NOT TOUCH.  ``raytrace.surfaces_from_
    elements`` registers a content-derived ``__spherical_<n>`` /
    ``__aspheric_<n>`` pseudo-glass for every numeric ``n_lens`` it is handed
    (``lumenairy/raytrace/trace.py:1502,1531``), and for those names
    ``glass._glass_cache`` is the AUTHORITATIVE value store rather than a cache
    -- ``glass._clear_glass_caches`` says so at ``lumenairy/glass.py:1716-1731``
    and deliberately preserves exactly this set for exactly this reason.
    Deleting the registry row would strand the cached ``_FixedIndex`` and make
    the next ``get_glass_index`` raise "flagged as user-fixed but has no
    _glass_cache entry".

    The growth is bounded and intentional by design (same content -> same name,
    audit P3-61), so it is library behaviour a test guard has no business
    reverting.  ``tests/unit/test_audit_p1_glass_registration.py`` sweeps ~200
    indices to PIN that behaviour and is what surfaced the need for this
    carve-out.
    """
    try:
        from lumenairy.glass import _USER_FIXED_SENTINEL
    except ImportError:                               # pragma: no cover - env
        return False
    return val == _USER_FIXED_SENTINEL


def _glass_restore(before):
    """Put the material tables back to ``before`` IN PLACE.

    Returns one description per difference, so a caller can report what leaked.
    """
    leaked = []
    for name, table in _glass_tables():
        prev = before.get(name)
        if prev is None:                              # pragma: no cover - env
            continue
        for k in [k for k in table if k not in prev]:
            if _is_library_user_fixed(table[k]):
                continue                              # library-owned; see above
            leaked.append(f'{name}[{k!r}] ADDED by {_glass_owner(table[k])}')
            del table[k]
        for k, v in prev.items():
            if k not in table:
                leaked.append(f'{name}[{k!r}] REMOVED')
                table[k] = v
            elif table[k] is not v:
                leaked.append(f'{name}[{k!r}] REBOUND')
                table[k] = v
    return leaked


@pytest.fixture(autouse=True, scope='module')
def _module_glass_registry_guard(request):
    """Install ``MODULE_GLASSES`` for this module; undo every table mutation.

    MODULE scope, matching :func:`_module_flag_leak_guard`: the registrations
    have to outlive a module's own module-scoped chain fixtures (which build
    prescriptions naming the model glass), and autouse fixtures at a given
    scope are set up before the non-autouse ones, so the glass exists before
    anything can look it up.
    """
    import os
    declared = dict(getattr(request.module, 'MODULE_GLASSES', None) or {})
    before = _glass_snapshot()
    if declared:
        from lumenairy import glass as _g
        _g.GLASS_REGISTRY.update(declared)            # in place; see above
    yield
    leaked = _glass_restore(before)
    stray = [d for d in leaked
             if not any(d.startswith(f'GLASS_REGISTRY[{k!r}] ADDED')
                        for k in declared)]
    if stray and os.environ.get('LUMEN_TEST_GLASS_LEAK_STRICT'):
        raise AssertionError(
            'this module leaked lumenairy.glass table entries to every LATER '
            'test in the process (glass-registry leak guard); declare model '
            'glasses in a module-level MODULE_GLASSES dict instead of writing '
            'them into GLASS_REGISTRY: ' + '; '.join(stray))


#: The material tables as SHIPPED, captured at conftest import -- pytest loads
#: this conftest before it imports a single test module, so this is the only
#: baseline that predates collection.
_PRISTINE_GLASS = _glass_snapshot()


def pytest_collection_modifyitems(session, config, items):
    """Strip IMPORT-TIME glass registrations -- the one leak the module guard
    structurally cannot catch.

    pytest imports EVERY selected test module during collection, before any
    fixture exists, so a module-scope ``GLASS_REGISTRY[...] = ...`` is already
    in the tables when the first module guard takes its snapshot.  That guard
    would then treat the pollution as part of the shipped baseline and preserve
    it for the whole session -- which is exactly the state that broke niche D8.
    This hook runs at the first moment all imports are done, diffs against the
    pre-collection baseline, and removes what a test module added.

    Removing rather than merely reporting is the point: it makes the NEXT
    module to register at import scope fail ITS OWN tests (its model glass is
    gone by the time they run) instead of silently failing an unrelated
    module's picklability assertion three thousand tests later.  Entries that
    are picklable AND not attributable to a test module are left alone and only
    reported -- those would be the library lazily loading a catalogue, which is
    not this hook's business.
    """
    import os
    import pickle
    import warnings
    offenders, removed = [], []
    for name, table in _glass_tables():
        prev = _PRISTINE_GLASS.get(name, {})
        for k in [k for k in table if k not in prev]:
            val = table[k]
            if _is_library_user_fixed(val):
                continue                              # library-owned; see above
            owner = _glass_owner(val)
            try:
                pickle.dumps(val)
                picklable = True
            except Exception:
                picklable = False
            ours = owner.startswith('tests.') or not picklable
            offenders.append(f'{name}[{k!r}] from {owner}'
                             f'{"" if picklable else " -- UNPICKLABLE"}'
                             f'{"" if ours else " (left in place)"}')
            if ours:
                del table[k]
                removed.append(f'{name}[{k!r}]')
    if not offenders:
        return
    msg = ('IMPORT-TIME glass-table pollution detected after collection: '
           + '; '.join(offenders)
           + '. Declare model glasses in a module-level MODULE_GLASSES dict '
             '(tests/conftest.py::_module_glass_registry_guard registers and '
             'removes them per module) instead of writing them into '
             'GLASS_REGISTRY at module scope.'
           + (f' REMOVED: {", ".join(removed)}.' if removed else ''))
    if os.environ.get('LUMEN_TEST_GLASS_LEAK_STRICT'):
        raise pytest.UsageError(msg)
    warnings.warn(msg, stacklevel=1)
    tr = config.pluginmanager.get_plugin('terminalreporter')
    if tr is not None:                                # pragma: no cover - env
        tr.write_line(msg, red=True, bold=True)


# ===========================================================================
# Niche C11 -- PROCESS-STATE DUMP for order-dependent failures
# ===========================================================================
#: The shipped baseline, captured at conftest import -- before any test runs.
_PRISTINE_FLAGS = _leak_guard_snapshot()


def describe_process_state():
    """Every process-global this suite's physics is known to depend on, as a
    printable block.

    Attach it to the assertion message of any test that PASSES in isolation and
    FAILS in a shard.  Such a failure names the victim and never the poisoner,
    and reproducing it needs the shard layout that produced it -- so the only
    cheap way to learn anything is to have the victim print what it was handed.

    Covers: the 91 discovered mode flags (only those differing from their
    shipped value are listed, so a clean run prints one line), the module-level
    CACHES the guard deliberately does not compare, the pyFFTW dispatch and
    plan-cache statistics, the active warnings filters, and the environment
    variables the library reads.
    """
    import os
    import warnings
    lines = []

    # --- mode flags: report only DEVIATIONS from the pristine values ------
    # ``_PRISTINE_FLAGS`` is captured at conftest IMPORT, i.e. before any test
    # has run, so it is the process's own shipped baseline.  A clean run prints
    # one line; a poisoned one prints exactly what was changed.
    for (mod_name, n), (mod, val) in sorted(_PRISTINE_FLAGS.items()):
        now = getattr(mod, n, val)
        same = now is val
        if not same:
            try:
                same = bool(now == val)
            except Exception:
                same = False
        if not same:
            lines.append(f'  FLAG {mod_name}.{n}: shipped {val!r} -> NOW {now!r}')
    if not lines:
        lines.append(f'  FLAGS: all {len(_PRISTINE_FLAGS)} discovered flags '
                     f'are at their process-start values')

    # --- caches the guard does not compare --------------------------------
    try:
        from lumenairy.elements import _lens_traced as _lt_mod
        c = _lt_mod._TRACED_KWARG_DEFAULTS_CACHE
        lines.append(f'  CACHE _TRACED_KWARG_DEFAULTS_CACHE: {len(c)} entries '
                     f'{sorted(c)[:6]}')
    except Exception as exc:                                  # pragma: no cover
        lines.append(f'  CACHE _lens_traced unavailable: {exc!r}')
    try:
        from lumenairy.propagators import fft_infra as _fi
        for name in ('_H_CACHE', '_FREQ_GRID_CACHE', '_BANDLIMIT_CACHE',
                     '_PYFFTW_PLAN_CACHE', '_PYFFTW_BAD_SHAPES'):
            obj = getattr(_fi, name, None)
            if obj is not None:
                lines.append(f'  CACHE fft_infra.{name}: {len(obj)} entries')
        lines.append(
            f'  FFT dispatch: USE_PYFFTW={_fi.USE_PYFFTW} '
            f'USE_SCIPY_FFT={_fi.USE_SCIPY_FFT} '
            f'FFTW_MIN_SIZE={_fi.FFTW_MIN_SIZE} '
            f'FFTW_THREADS={_fi.FFTW_THREADS} '
            f'FALLBACK={_fi.PYFFTW_FALLBACK_ON_ERROR}')
        lines.append(
            f'  FFT planner: {_fi.get_pyfftw_planner()!r} '
            f'double_buffer={_fi._PYFFTW_DOUBLE_BUFFER} '
            f'plan_cache_size={_fi._PYFFTW_PLAN_CACHE_SIZE} '
            f'auto_promote={_fi._PYFFTW_AUTO_PROMOTE}'
            f'/thresh={_fi._PYFFTW_AUTO_PROMOTE_THRESHOLD} '
            f'propagator={_fi.DEFAULT_WAVE_PROPAGATOR!r} '
            f'DEFAULT_DY={_fi.DEFAULT_DY!r}')
        try:
            lines.append(f'  FFT counters: {_fi.get_fft_backend_counts()!r}')
        except Exception:
            pass
    except Exception as exc:                                  # pragma: no cover
        lines.append(f'  FFT state unavailable: {exc!r}')

    # --- warnings filters and environment ---------------------------------
    lines.append(f'  WARNINGS filters ({len(warnings.filters)}): '
                 f'{[(f[0], getattr(f[2], "__name__", f[2])) for f in warnings.filters[:8]]}')
    envs = {k: v for k, v in os.environ.items()
            if k.startswith(('LUMEN', 'PYFFTW', 'OMP_', 'MKL_', 'OPENBLAS_',
                             'NUMBA_', 'JAX_', 'NUMEXPR_'))}
    lines.append(f'  ENV: {envs}')
    return ('\n---- process state (niche C11 order-dependence dump) ----\n'
            + '\n'.join(lines) + '\n' + '-' * 56)


@pytest.fixture(scope='session')
def fft_state_ctx():
    """The :func:`shipped_fft_dispatch_state` context manager, as a fixture.

    ``tests/conftest.py`` is not importable by name from a test module, so a
    module-scoped fixture that needs the context manager asks for this.
    """
    return shipped_fft_dispatch_state


@pytest.fixture(scope='session')
def process_state_dump():
    """:func:`describe_process_state`, as a fixture (same reason)."""
    return describe_process_state
