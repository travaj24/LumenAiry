"""Wave-5 D item (3): rebuild a11's Z3 peak-array gate.

Run with ``PYTHONPATH=validation/probe_known_reds python <this file>`` from the
worktree root.  Idempotent-checked: it refuses if the anchor is not found.
"""
from _patch import patch

P = "tests/unit/test_audit2609_a11_polar_sources_infra.py"

OLD = '''def test_z3_stokes_and_dop_peak_arrays():
    """Peak, in full-grid REAL arrays (N*N*8 B), measured 2026-09-12 at
    N = 2048: stokes 7.00 -> 6.00, dop 8.25 -> 6.00 (both pre-fix arms are
    measured in this same test).  Bar 6.5: above the post-fix reading by 8 %
    and below both pre-fix readings by 7 % / 21 %.  tracemalloc peaks of
    straight-line NumPy are exact allocation counts with no cross-build
    spread.  6.00 is the floor for a bit-identical implementation: the four
    outputs plus the one complex cross-term the exact S2/S3 need."""
    N = 1024
    unit = N * N * 8
    f = _pathological_field(N, np.complex128)
    f = JonesField(np.nan_to_num(f.Ex, posinf=3.0),
                   np.nan_to_num(f.Ey, posinf=3.0), 1e-6, 1e-6)
    peaks = {}
    for tag, fn in (('stokes_new', stokes_parameters), ('stokes_old', _old_stokes),
                    ('dop_new', degree_of_polarization), ('dop_old', _old_dop)):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        res = fn(f)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del res
        gc.collect()
        peaks[tag] = peak / unit
    assert peaks['stokes_new'] < 6.5 < peaks['stokes_old'], peaks
    assert peaks['dop_new'] < 6.5 < peaks['dop_old'], peaks
'''

NEW = '''#: How far above a whole number of full grids a tracemalloc peak may sit.
#: DERIVED, two-sided: the reading is an EXACT allocation count in units of one
#: full grid plus tracemalloc's own bookkeeping, and that bookkeeping measured
#: 1960 .. 2984 B over 5 repeats on each of two arms (Windows py3.14 /
#: numpy 2.4.4 and WSL py3.12 / numpy 2.4.6), i.e. at most 3.6e-04 grids at
#: N = 1024.  0.05 is two decades above that spread and 1.3 decades below the
#: 1.0 that separates one allocation count from the next -- so it can absorb
#: any bookkeeping and can never absorb an array.
_Z3_PEAK_SLACK = 0.05


def _z3_peak_units(fn, unit):
    """Peak transient of ``fn()`` in units of one full-grid REAL array."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    res = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del res
    gc.collect()
    return peak / unit


def _numpy_elides_binary_temporaries(field, unit):
    """MEASURED premise: does this build rewrite ``Ex * conj(Ey)`` into the
    ``conj`` temporary's own buffer instead of allocating a second array?

    NumPy's ``temp_elide.c`` does exactly that when an operand is an
    unreferenced temporary, but only where the optimisation is compiled in (it
    needs ``backtrace()``) and only when its stack walk can confirm the
    temporary came from the interpreter.  Both are BUILD properties: measured
    ACTIVE on WSL (py3.12, numpy 2.4.6) and on the py3.11 and py3.14 Linux
    runners of CI run 34914295323, INACTIVE on Windows (py3.14, numpy 2.4.4).
    So the peak of an expression written with a free temporary is a per-build
    quantity and has to be read on the running arm, never assumed from it.

    Returns ``(free, held)`` in full-grid REAL arrays.  ``held`` binds the
    temporary to a name, which lifts its reference count to 2 and puts elision
    out of reach on every build: it is 2.00 everywhere, and is asserted, so a
    reading of "no elision here" can never come from an instrument that
    measured nothing at all.  ``free`` is 2.00 where elision is unavailable
    and 1.00 where it is.
    """
    free = _z3_peak_units(lambda: field.Ex * np.conj(field.Ey), unit)

    def _held():
        c = np.conj(field.Ey)             # named -> refcount 2 -> not elidable
        return field.Ex * c

    held = _z3_peak_units(_held, unit)
    assert 1.9 < held < 2.1, (
        f"the elision instrument read {held:.3f} full grids for a complex "
        f"product that must cost exactly 2.00 -- it is not measuring what it "
        f"claims, so no premise can be drawn from it")
    return free, held


def _z3_peaks(N=1024):
    """The four peak readings, in full-grid REAL arrays, with the field."""
    unit = N * N * 8
    f = _pathological_field(N, np.complex128)
    f = JonesField(np.nan_to_num(f.Ex, posinf=3.0),
                   np.nan_to_num(f.Ey, posinf=3.0), 1e-6, 1e-6)
    peaks = {tag: _z3_peak_units(lambda fn=fn: fn(f), unit)
             for tag, fn in (('stokes_new', stokes_parameters),
                             ('stokes_old', _old_stokes),
                             ('dop_new', degree_of_polarization),
                             ('dop_old', _old_dop))}
    return f, unit, peaks


def test_z3_stokes_and_dop_peak_arrays():
    """The UNCONDITIONAL half of the peak-array claim, in full-grid REAL
    arrays (N*N*8 B).

    6.00 is the derived FLOOR for a bit-identical Stokes implementation: the
    four outputs, plus the one complex cross term (2 real grids) the exact
    S2 / S3 need while S3 is being written.  Both shipped functions sit ON
    that floor, and the DOP costs no more than the ``stokes_parameters`` call
    inside it, which is the whole content of the in-place accumulation.  The
    pre-fix DOP does not: it holds 8.25.

    Every claim here holds on both arms of the two-arm ladder -- Windows
    py3.14 / numpy 2.4.4 (no temporary elision) and WSL py3.12 / numpy 2.4.6
    (elision active) -- and on every CI runner, because none of them can be
    reached by eliding a temporary.  MEASURED, 5 repeats per arm::

        stokes_new  6.000237 .. 6.000345 (Win)  6.000234 .. 6.000333 (WSL)
        dop_new     6.000237             (Win)  6.000234             (WSL)
        dop_old     8.250353 .. 8.250356 (Win)  8.250346 .. 8.250349 (WSL)

    The one reading that IS build-dependent -- the pre-fix Stokes form's
    seventh grid -- is premise-gated in
    :func:`test_z3_the_pre_fix_stokes_form_holds_a_seventh_grid`.
    """
    _f, _unit, peaks = _z3_peaks()
    # the four outputs + the complex cross term, and not one array more
    assert 6.0 <= peaks['stokes_new'] < 6.0 + _Z3_PEAK_SLACK, peaks
    assert 6.0 <= peaks['dop_new'] < 6.0 + _Z3_PEAK_SLACK, peaks
    # the DOP accumulates over the very arrays its own Stokes call returned
    assert peaks['dop_new'] <= peaks['stokes_new'] + _Z3_PEAK_SLACK, peaks
    # ... which the pre-fix DOP did not: 2.25 grids of avoidable transient
    assert peaks['dop_old'] > 6.5, peaks
    assert peaks['dop_old'] - peaks['dop_new'] > 2.0, peaks
    # and no arm may make the shipped Stokes form the more expensive one
    assert peaks['stokes_new'] <= peaks['stokes_old'] + _Z3_PEAK_SLACK, peaks


def test_z3_the_pre_fix_stokes_form_holds_a_seventh_grid():
    """PREMISE-GATED (TESTING_STANDARDS S3).  The pathology the shipped Stokes
    form removed -- the second full-grid COMPLEX temporary that
    ``Ex * conj(Ey)``, written twice, costs -- is observable only on a build
    where NumPy does not elide that temporary away.

    The premise is measured on the running arm by
    :func:`_numpy_elides_binary_temporaries`, never assumed from the platform.
    Where it holds, the pre-fix form peaks at 7.00 grids against the shipped
    form's 6.00 (Windows py3.14 / numpy 2.4.4: 7.000074 vs 6.000237).  Where
    NumPy elides, the pre-fix form is rewritten into the same 6.00 and there
    is no seventh grid to see: WSL py3.12 / numpy 2.4.6 and the py3.11 and
    py3.14 Linux runners of CI run 34914295323 all read 6.000234, which is
    what red this gate's previous ``6.5 < stokes_old`` form.  That arm asserts
    the collapse explicitly and then skips WITH the reading, so it can never
    pass silently.

    The DOP half of the claim needs no gate and is asserted unconditionally in
    the test above: 8.25 -> 6.00 on every arm measured.
    """
    f, unit, peaks = _z3_peaks()
    free, held = _numpy_elides_binary_temporaries(f, unit)
    if free < held - 0.5:
        assert 6.0 <= peaks['stokes_old'] < 6.0 + _Z3_PEAK_SLACK, (
            f"NumPy elided the pre-fix form's complex temporary, so it must "
            f"land on the same 6.00-grid floor as the shipped one, but it "
            f"read {peaks['stokes_old']:.6f}: {peaks}")
        pytest.skip(
            f"premise absent on this arm: NumPy's temporary elision is ACTIVE "
            f"(Ex*conj(Ey) peaks at {free:.3f} full grids free, {held:.3f} "
            f"with the temporary name-bound), so the pre-fix Stokes form "
            f"allocates no seventh grid -- measured stokes_old="
            f"{peaks['stokes_old']:.6f} against stokes_new="
            f"{peaks['stokes_new']:.6f}, both on the 6.00 floor")
    assert 7.0 <= peaks['stokes_old'] < 7.0 + _Z3_PEAK_SLACK, (
        f"elision is inactive here (free={free:.3f}, held={held:.3f}), so the "
        f"pre-fix form must hold its seventh grid: {peaks}")
    assert peaks['stokes_old'] - peaks['stokes_new'] > 0.95, peaks
'''

patch(P, OLD, NEW)
