"""VERIFY-B14 -- the gaps the independent re-verification of WP-B14 found.

WP-B14 (`fix/known-reds-and-stacklevels`, e61c6467) closed four known reds, the
5.47.0 CI matrix and the warning-attribution sweep of the three chain modules.
Re-measuring it reproduced its numbers; this file pins the four properties the
re-measurement found to be ASSERTED NOWHERE, each with the measurement that
sets its bar.  The full evidence is
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B14.md`
and the probes under `validation/probe_verify_b14/`.

Every bar here is two-sided and derived from a measurement taken on BOTH
builds (Windows py3.14.6 / numpy 2.4.4 / OpenBLAS Haswell, and WSL py3.12.3 /
numpy 2.4.6), with the readings in the docstring of the test that carries it.
Nothing here pins a version's digits: the quantities are re-derived at runtime
from the running library.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import os
import pathlib
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]
_PKG = os.path.abspath(os.path.dirname(la.__file__))

#: The files of the ``warnings`` implementation itself, which sit between the
#: emitting line and ``showwarning``.  CPython 3.14 moved the Python half to
#: ``_py_warnings.py``, so the set is derived rather than written down.
_WARN_IMPL_FILES = {os.path.basename(warnings.__file__ or 'warnings.py'),
                    'warnings.py', '_py_warnings.py'}


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


# ===========================================================================
# 1.  The c7 / c8 stimulus must stay STATED, not inherited
# ===========================================================================
_C7_C8_FILES = ('tests/unit/test_niche_c7_ray_density_halo_check.py',
                'tests/unit/test_niche_c8_inverse_support_bound.py')


@pytest.mark.parametrize('rel', _C7_C8_FILES)
def test_the_halo_fixtures_state_the_fit_order_they_were_calibrated_at(rel):
    """WHY THIS EXISTS.  Both files manufacture their lobe with the order-10
    decentred ray fit they were calibrated against.  WP-A26 (`bbb6c02d`) moved
    the library default ``_DECENTRED_FIT_POLY_ORDER`` 10 -> 16 for an unrelated
    and sound reason and did not restate the fixtures, and the four ids went
    red on every arm.  Re-measured in `git archive` trees with ``PYTHONPATH``
    pinned (`validation/probe_verify_b14/probe_v1_halo_order.py`), the halo
    beyond three beam radii with the C8 bound off / on reads

        bbb6c02d^ and v5.45.1   4.594672922387141e-02 / 8.913411901995892e-04
        bbb6c02d and later      1.5217194665917584e-04 (both, ratio 1.00)

    bit for bit, so the bisection is exact.  WP-B14's repair was to STATE the
    order in each file's ``_BASE_KW``.  Nothing asserted that it stays stated,
    and the next default move would strand the fixtures the same way -- the
    failure would again present as four unexplained reds rather than as a
    changed default.  This is that assertion, and it is a source-level one so
    it costs nothing.

    Measured ladder on the C8 ``_GHOST`` fixture at this tree (halo beyond
    3 w, bound OFF): 1.463e-04 at orders 5, 6, 7, 8, 11 and 12; **4.595e-02 at
    order 9 and at order 10**; 1.53e-04 at 13/14; 1.52e-04 at 15/16; 1.54e-04
    at 17/18; 1.55e-04 at 20.  The manufacturing band is two orders wide, and
    the shipped default is outside it, which is exactly why the stimulus has
    to be a parameter of the fixture.
    """
    src = (_REPO / rel).read_text(encoding='utf-8')
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and \
                node.arg == 'decentred_fit_poly_order':
            if isinstance(node.value, ast.Constant):
                found.append(node.value.value)
    assert found, (
        f"{rel} no longer states decentred_fit_poly_order in its fixture "
        f"keywords.  The manufactured-lobe stimulus these files assert on is "
        f"reachable only inside a narrow band of the decentred fit order "
        f"(orders 9 and 10 on this geometry, measured), so inheriting the "
        f"library default makes the fail-before arm a hostage to a default "
        f"that has already moved once (WP-A26, 10 -> 16).  State it.")
    assert all(isinstance(v, int) and v > 0 for v in found), found


# ===========================================================================
# 2.  estimate_lens_memory's bound must not depend on process history
# ===========================================================================
_Z3_N = 512
_Z3_WARM_N = 256


def _z3_measure(drain_before):
    wl, dx = 633e-9, 30e-3 / _Z3_N
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                         aperture=25e-3)
    E, _, _ = la.create_gaussian_beam(_Z3_N, dx, wl, w0=5e-3,
                                      dtype=np.complex128)
    E = np.ascontiguousarray(E)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.apply_real_lens(E[:_Z3_WARM_N, :_Z3_WARM_N].copy(),
                           prescription=rx, wavelength=wl, dx=dx)
        # Whatever an earlier test in this process would have done: build the
        # N-sized FFT / ASM caches.
        la.apply_real_lens(E.copy(), prescription=rx, wavelength=wl, dx=dx)
        if drain_before:
            # the library's own registered drain; it empties the ASM caches
            # AND the pyFFTW plan cache, which is the pair the reading needs
            la.clear_asm_caches()
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del out
    gc.collect()
    est = la.estimate_lens_memory(_Z3_N, np.complex128, lens_model='real')
    return (retained / float(_Z3_N * _Z3_N * 16), peak, est, est / peak)


def test_the_lens_memory_reading_is_engineered_not_inherited():
    """WHY THIS EXISTS, and it is a live defect, not a hypothetical.
    ``test_audit2609_a11_polar_sources_infra.py::
    test_z3_estimate_lens_memory_real_bounds_apply_real_lens`` gained an
    unconditional guard that the measured call RETAINS 5.5 .. 7.0 full complex
    grids, "the 6.0 of N-sized FFT/ASM cache it builds".  That premise is a
    statement about PROCESS HISTORY: the call builds those caches only while
    they are cold for this N.  Measured, byte-identically on both builds
    (`validation/probe_verify_b14/probe_v8_a11_retain_state.py`):

        state     retained   peak      est       est/peak
        cold        6.09 g   46.5 MB   49.5 MB     1.064
        WARM        1.00 g   25.2 MB   49.5 MB     1.965
        drained     6.01 g   46.2 MB   49.5 MB     1.071

    -- so in the warm state BOTH bars fail, the RETAIN guard first.  It is
    reachable from a plain selection: on this tree
    ``pytest test_audit2609_a6_carrier.py <that id>`` is ``1 failed, 80
    passed`` while the id alone is ``1 passed`` and the base commit `96cb2096`
    is ``81 passed``.  Since the CI fast lane assigns files to shards by
    duration, whether the two land in one process is not the test's to decide.

    WHAT IS PINNED HERE is the repair, as a property rather than as a bar on
    one arm: after the caches are DRAINED through the library's own registered
    drains the reading is the cold one again, on any process history.  The bars
    are the a11 test's own, unchanged: the estimate must bound the peak
    (>= 1.0) and not over-reserve past 1.6x.  The warm arm is asserted too, as
    the fail-before that says the drain is doing work -- measured 1.965, bar
    > 1.7, which is 13 % under the measurement and 6 % over the bar it breaks.
    """
    held_w, _peak_w, _est_w, ratio_w = _z3_measure(drain_before=False)
    held_d, peak_d, est_d, ratio_d = _z3_measure(drain_before=True)

    assert 1.0 <= ratio_d <= 1.6, (
        f"after draining the ASM and FFT caches the estimate no longer bounds "
        f"apply_real_lens: est {est_d / 1e6:.1f} MB vs peak "
        f"{peak_d / 1e6:.1f} MB (ratio {ratio_d:.3f}), retained "
        f"{held_d:.2f} grids.  This is the a11 gate's own bar, measured here "
        f"with the state ENGINEERED rather than inherited.")
    assert 5.5 < held_d < 7.0, (
        f"drained, the call retained {held_d:.2f} grids, not the ~6.0 of "
        f"N-sized FFT/ASM cache it should have rebuilt.")
    # FAIL-BEFORE: without the drain the same measurement is out of band, so
    # the drain is not decoration.
    assert ratio_w > 1.7, (
        f"the warm-cache arm now reads est/peak = {ratio_w:.3f} (retained "
        f"{held_w:.2f} grids), i.e. a warm process no longer perturbs this "
        f"measurement.  If that is a real change, this test's premise is gone "
        f"-- do not simply relax the bar.")


# ===========================================================================
# 3.  The FFT dispatch IS a function of its inputs, in every buffer mode
# ===========================================================================
def test_the_fft_dispatch_is_a_function_of_its_inputs_in_every_buffer_mode():
    """WHY THIS EXISTS.  WP-B14 section 6.8 reports, as a library defect handed
    to the owner of ``fft_infra.py``, that "with the shipped pyFFTW two-buffer
    ping-pong ON, ``_ifft2(_fft2(E) * H)`` is not a function of its input
    values alone".  Re-measured, the transforms themselves ARE: repeated
    identical evaluations of the round trip, and of each half separately, give
    ONE byte image at n = 128, 256 and 512, in every mode, on both builds --
    eight repeats in
    `validation/probe_verify_b14/probe_v7_fft_determinism.py` and six here,
    which is the same claim at a cost this file can carry.

    The A/B the report saw is real but is made one level up, by NumPy: with the
    ping-pong on, ``_fft2`` hands back a NON-OWNING aligned view, which NumPy's
    temporary elision cannot claim, where the single-buffer path hands back
    ``buf.copy()``, which it can.  In the expression ``_fft2(E) * H`` that
    flips which operand is elided into, and on the Linux NumPy build the
    right-operand elision of a complex128 multiply moves the last bits
    (measured 1.0e-16 to 1.8e-16 relative on 16-17 % of the doubles at
    n >= 128; zero at every size on the Windows build --
    `probe_v7d_elision.py`).  So the ``fft_double_buffer`` knob's
    byte-identity claim has to be read as a claim about the TRANSFORM's values,
    and that is what is asserted here, unconditionally and on both builds.

    Both bars are exact equalities between two byte images of the same
    transform, so there is no tolerance to derive: the quantity either is a
    function of its input or it is not.
    """
    from lumenairy.propagators import fft_infra as F
    rng = np.random.default_rng(12345)
    prev = F.get_fft_double_buffer()
    try:
        for n in (128, 256, 512):
            E = (rng.normal(size=(n, n))
                 + 1j * rng.normal(size=(n, n))).astype(np.complex128)
            H = (rng.normal(size=(n, n))
                 + 1j * rng.normal(size=(n, n))).astype(np.complex128)
            for mode in (True, False):
                F.set_fft_double_buffer(mode)
                rt = {_md5(F._ifft2(F._fft2(E) * H)) for _ in range(6)}
                assert len(rt) == 1, (
                    f"n={n}, double_buffer={mode}: six identical evaluations "
                    f"of _ifft2(_fft2(E) * H) returned {len(rt)} distinct "
                    f"byte images.  The dispatch is not a function of its "
                    f"inputs.")
                fw = {_md5(F._fft2(E)) for _ in range(6)}
                assert len(fw) == 1, (
                    f"n={n}, double_buffer={mode}: _fft2 alone returned "
                    f"{len(fw)} distinct byte images from one operand.")
            # the knob's own claim, correctly scoped: the TRANSFORM's values
            # do not depend on the buffer mode.
            F.set_fft_double_buffer(True)
            a = np.array(F._fft2(E), copy=True)
            F.set_fft_double_buffer(False)
            b = np.array(F._fft2(E), copy=True)
            assert np.array_equal(a, b), (
                f"n={n}: _fft2's VALUES moved with set_fft_double_buffer, "
                f"which is the one thing the knob's documentation says it "
                f"cannot do.  max|da| = {float(np.max(np.abs(a - b))):.3e}")
            F.set_fft_double_buffer(True)
            c = np.array(F._ifft2(E), copy=True)
            F.set_fft_double_buffer(False)
            d = np.array(F._ifft2(E), copy=True)
            assert np.array_equal(c, d), (
                f"n={n}: _ifft2's VALUES moved with set_fft_double_buffer.")
    finally:
        F.set_fft_double_buffer(prev)


# ===========================================================================
# 4.  The dense GBD budget: what 'measured' does and does not buy
# ===========================================================================
def _gbd_run(mode, bundle, N, budget_mb):
    from lumenairy.propagators import gbd as G
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = G.reconstruct_field_from_beamlets(
                bundle, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
                chunk_beamlets=4096, mem_budget_mb=budget_mb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return np.asarray(out), int(peak - base)


def test_the_measured_accounting_bounds_the_budget_only_above_one_column():
    """WHY THIS EXISTS.  ``test_wave5_gbd_dense_mem_budget.py`` asserts, on one
    cell, that ``DENSE_MEM_BUDGET_ACCOUNTING = 'measured'`` "makes the budget a
    bound", and the module note says the switch is what makes ``mem_budget_mb``
    bound the loop.  Re-measured over a wider ladder that is TRUE ONLY WHERE
    THE BUDGET BUYS MORE THAN ONE BEAMLET COLUMN.  The chunk floors at 1, and
    one column plus the loop's fixed arrays is the smallest transient the loop
    can have, whatever constant the chunk sizing uses.

    Independent model, fitted on this tree and agreeing with every ladder row
    to 0.2 % (`validation/probe_verify_b14/probe_v4_gbd_budget.py`):

        peak  ~  Ny * Nx * (48 + C * chunk) bytes,  C = 72 for a single chunk,
                                                    C = 96 for two or more

    i.e. the shipped ``_DENSE_CELL_BYTES_LEGACY = 16`` under-counts the
    per-column term by exactly 6.0x = 96/16 (which is where the report's
    saturation comes from), and there is a FIXED ~48 B/cell term the chunk
    arithmetic does not model at all.  Measured, N = 256, 'measured' mode:

        budget 512 MB -> chunk 61, peak  387.1 MB, 0.76x  (bounded)
        budget  16 MB -> chunk  1, peak    9.6 MB, 0.60x  (bounded, just)
        budget   4 MB -> chunk  1, peak    9.6 MB, 2.39x  (NOT bounded)
        budget   1 MB -> chunk  1, peak    9.6 MB, 9.58x  (NOT bounded)

    Both claims are pinned, two-sided: above the floor the switch bounds the
    budget (bar 1.0x against a measured 0.76x), and below it no accounting
    constant can (bar > 1.5x against a measured 2.39x, and the repaired state
    would be < 1.0x, so the gap is a factor of 1.5 either way).  That is the
    scope the note should carry, and it is what a caller sizing a small box
    needs to know.
    """
    from lumenairy.propagators import gbd as G
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    N = 256
    try:
        rng = np.random.default_rng(0)
        nb = 1024
        bundle = G.BeamletBundle(
            positions=rng.normal(0.0, 2.0e-4, size=(nb, 3)),
            directions=np.zeros((nb, 3)),
            Q=np.full(nb, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128),
            amplitude=(rng.normal(size=nb)
                       + 1j * rng.normal(size=nb)).astype(np.complex128),
            waist0=np.full(nb, 1.0e-3))

        # the floor the budget cannot go under, derived from the loop's own
        # shape rather than from a reading: one column plus the fixed arrays.
        floor_mb = N * N * (48.0 + G._DENSE_CELL_BYTES_MEASURED) / 1e6

        _f, peak_big = _gbd_run('measured', bundle, N, 512.0)
        assert peak_big / (512.0 * 1e6) < 1.0, (
            f"'measured' no longer bounds a budget well above one column: "
            f"peak {peak_big / 1e6:.1f} MB against 512 MB.")

        small_mb = 4.0
        assert small_mb < floor_mb, (
            f"the probe's small budget {small_mb} MB is no longer below the "
            f"loop's own floor ({floor_mb:.1f} MB); re-derive it.")
        _f2, peak_small = _gbd_run('measured', bundle, N, small_mb)
        ratio_small = peak_small / (small_mb * 1e6)
        assert ratio_small > 1.5, (
            f"a {small_mb:.0f} MB budget is now bounded by the 'measured' "
            f"accounting (peak {peak_small / 1e6:.1f} MB, {ratio_small:.2f}x) "
            f"-- the chunk floor at 1 must have been addressed.  If so this "
            f"test's premise is gone and the scope note above should go with "
            f"it; do not relax the bar.")
    finally:
        G.DENSE_MEM_BUDGET_ACCOUNTING = old


# ===========================================================================
# 5.  The swept chain modules name the caller at more than one depth
# ===========================================================================
def _library_frame(nest):
    """A forwarder through ``nest`` frames that are library code to both
    consumers of that notion: ``caller_stacklevel`` reads ``co_filename`` and
    ``warnings`` reads the frame globals' ``__file__``."""
    if nest <= 0:
        return lambda fn, a, kw: fn(*a, **kw)
    fake = os.path.join(_PKG, 'propagators', '_verify_b14_synth_depth.py')
    cur = None

    def _compile(body):
        ns = {'_inner': cur, '__file__': fake,
              '__name__': 'lumenairy.propagators._verify_b14_synth_depth'}
        exec(compile(body, fake, 'exec'), ns)
        return ns['_w']

    cur = _compile('def _w(fn, a, kw):\n    return fn(*a, **kw)\n')
    for _ in range(nest - 1):
        cur = _compile('def _w(fn, a, kw):\n    return _inner(fn, a, kw)\n')
    return cur


def test_the_swept_chain_warn_sites_name_the_caller_at_three_depths():
    """WHY THIS EXISTS.  The b11 ratchet's new two-caller fixture covers ONE
    warn site (the tilt-inert notice) at two depths.  The sweep touched 21
    literal sites across three modules, and a literal is wrong exactly when a
    site is reachable at more than one depth -- so one site is a sample, not a
    measurement.

    The instrument here needs no second real entry point: a function COMPILED
    with a ``co_filename`` under the package directory, whose globals carry a
    matching ``__file__``, is library code to ``caller_stacklevel``'s rule and
    to ``warnings``' filename bookkeeping alike, so every site can be reached
    at depth 0, 1 and 2 from this file.

    MEASURED with it over a 13-case battery
    (`validation/probe_verify_b14/probe_v5_two_caller.py`), counting a warning
    as misattributed when the file it names is inside the package:

        base commit 96cb2096   26 of 36 emissions misattributed
        this tree               6 of 36, all of them ONE site in
                                ``propagators/asm.py`` -- an unswept module,
                                reached through carrier.py, and exactly the
                                open work section 5.3 records.

    So the claim asserted here is the one the sweep earns: no warning raised
    out of ``carrier.py``, ``system.py`` or ``carrier_field.py`` names a file
    inside the package, at any of the three depths.  Warnings from OTHER
    package modules are counted and reported but not asserted, because they
    are the recorded open work rather than this package's claim.
    """
    from lumenairy.propagators import carrier as CA
    from lumenairy.propagators import carrier_field as CF
    from lumenairy.propagators import system as SY

    swept = {'carrier.py', 'system.py', 'carrier_field.py'}
    n, dx, wl = 64, 8e-6, 633e-9
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    env = np.exp(-(X ** 2 + Y ** 2) / (60e-6 ** 2)).astype(np.complex128)

    def _cf():
        return CF.CarrierField(
            envelope=env.copy(),
            grid=CF.FieldGrid(shape=env.shape, dx=dx, dy=dx,
                              origin=(0.0, 0.0)),
            carrier=CF.CarrierSpec(R=-0.05, centre=(0.0, 0.0),
                                   tilt=(0.0, 0.0), piston=0.0),
            wavelength=wl)

    cases = {
        'carrier.tilt_inert': (
            CA.propagate_carrier_referenced, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, gap_kernel='fresnel',
                 tilt=(0.12, 0.0))),
        'carrier.focus_readout_tilt_inert': (
            CA.carrier_referenced_focus_readout, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, gap_kernel='fresnel',
                 tilt=(0.12, 0.0), dx_out=dx, N_out=32)),
        'carrier.replica': (
            CA.carrier_referenced_focus_readout, (env, -0.05, 5e-3),
            dict(wavelength=wl, dx=dx, dx_out=dx * 8.0, N_out=256,
                 on_replica='warn', on_focus_containment='warn')),
        'carrier_field.re_reference': (
            CF.re_reference,
            (_cf(), CF.CarrierSpec(R=2e-4, centre=(0.0, 0.0),
                                   tilt=(0.0, 0.0), piston=0.0),
             CF.FieldGrid(shape=(64, 64), dx=dx, dy=dx, origin=(0.0, 0.0))),
            dict(on_nyquist='warn', on_window='warn')),
        'system.fresnel_leg': (
            SY.propagate_through_system, (env,),
            dict(elements=[{'type': 'propagate', 'z': 0.5,
                            'method': 'fresnel'}], wavelength=wl, dx=dx)),
    }

    total, seen, bad, elsewhere = 0, 0, [], []
    for label, (fn, args, kw) in cases.items():
        for nest in (0, 1, 2):
            fwd = _library_frame(nest)
            # Record the EMITTING module alongside the file the warning NAMES.
            # The two differ exactly when an unswept module's literal points at
            # its caller: ``asm.py``'s window notice reached through
            # ``carrier.py`` names carrier.py while being emitted from asm.py,
            # and that is the recorded open work, not this sweep's claim.
            # CPython reaches ``showwarning`` through one or two frames of
            # ``warnings.py`` itself, so the emitting line is the first frame
            # out that is not in that module.
            caught = []

            def _hook(message, category, filename, lineno, *a, **k):
                import sys as _sys
                fr, emitter = _sys._getframe(1), '?'
                while fr is not None:
                    b = os.path.basename(fr.f_code.co_filename or '')
                    if b not in _WARN_IMPL_FILES:
                        emitter = b
                        break
                    fr = fr.f_back
                caught.append((str(message), category, filename, lineno,
                               emitter))

            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.showwarning = _hook
                try:
                    fwd(fn, args, dict(kw))
                except Exception:                 # pragma: no cover
                    pass
            for msg, _cat, filename, lineno, emitter in caught:
                total += 1
                f = os.path.abspath(filename)
                if not f.startswith(_PKG):
                    seen += 1
                    continue
                if emitter in swept:
                    bad.append((label, nest, emitter, os.path.basename(f),
                                lineno, msg[:70]))
                else:
                    elsewhere.append((label, nest, emitter,
                                      os.path.basename(f)))
    # The instrument has to be alive before its verdict means anything, and
    # it is counted over EVERY emission rather than over the correctly
    # attributed ones -- otherwise a tree on which everything is misattributed
    # would trip the instrument guard instead of the claim.
    assert total >= 6, (
        f"the battery produced only {total} warnings at all; it is no longer "
        f"reaching the swept sites (cases: {list(cases)}).")
    assert not bad, (
        f"{len(bad)} warning(s) raised out of the SWEPT chain modules name a "
        f"file inside the package instead of the caller.  A literal "
        f"stacklevel is right for one call depth and silently wrong at every "
        f"other; use caller_stacklevel().  {bad}\n"
        f"(unswept modules seen, not asserted: {sorted(set(elsewhere))})")
