"""Wave-5 D item (4): rebuild b8's apply_jones_matrix peak-array gate."""
from _patch import patch

P = "tests/unit/test_audit2609_b8_analysis_sources.py"

OLD = '''def test_b8_apply_jones_matrix_peak_full_grid_arrays():
    """Derived count, in units of ONE full-grid complex array.

    Before: the first component's result plus the second's two products
    and their sum = 4.  After: two results plus one shared scratch = 3,
    which is the floor (neither result can be written before both its
    terms exist, and the inputs belong to the caller).  Bar at 3.5."""
    N = 1024
    grid = N * N * 16
    rng = np.random.default_rng(1)
    Ex = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Ey = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    J = np.array([[0.3 + 0.4j, -0.5 + 0.1j],
                  [0.2 - 0.7j, 0.9 + 0.05j]], dtype=complex)
    old = _peak_grids(lambda: _pre_fix_jones(J, Ex, Ey), grid)
    new = _peak_grids(lambda: apply_jones_matrix(
        JonesField(Ex, Ey, 1e-6, WL), J), grid)
    assert old > 3.5, old
    assert new < 3.5, new
'''

NEW = '''#: How far above a whole number of full grids a tracemalloc peak may sit.
#: DERIVED, two-sided: the reading is an EXACT allocation count in units of one
#: full COMPLEX grid plus tracemalloc's own bookkeeping, and that bookkeeping
#: measured 448 .. 7712 B over 9 repeats on each of two arms (Windows py3.14 /
#: numpy 2.4.4 and WSL py3.12 / numpy 2.4.6), i.e. at most 4.6e-04 grids at
#: N = 1024.  0.05 is two decades above that spread and 1.3 decades below the
#: 1.0 that separates one allocation count from the next.
_B8_PEAK_SLACK = 0.05


def _numpy_elides_the_sum_of_two_temporaries(J, Ex, Ey, grid):
    """MEASURED premise: does this build rewrite ``a*X + b*Y`` into one of the
    two products' own buffers instead of allocating a third array?

    NumPy's ``temp_elide.c`` does that when an operand of a binary op is an
    unreferenced temporary, but only where the optimisation is compiled in (it
    needs ``backtrace()``) and only when its stack walk can confirm the
    temporary came from the interpreter.  Both are BUILD properties, and they
    do NOT follow the operating system: on CI run 34914295323 this sum was
    elided on the py3.14 Linux runner and NOT elided on the py3.12 and py3.13
    ones, while a DIFFERENT elidable pattern (``Ex * conj(Ey)``, gated in
    ``test_audit2609_a11_polar_sources_infra.py``) was elided on py3.11.  Two
    patterns, two premises, each measured where it is used -- never inferred
    from the platform, and never from each other.

    Returns ``(free, held)`` in full-grid COMPLEX arrays.  ``held`` binds both
    products to names, which lifts their reference counts to 2 and puts
    elision out of reach on every build: it is 3.00 everywhere, and is
    asserted, so a reading of "no elision here" can never come from an
    instrument that measured nothing at all.  ``free`` is 3.00 where elision
    is unavailable and 2.00 where it is.  MEASURED: 3.000 / 3.000 on Windows
    py3.14, 2.000 / 3.000 on WSL py3.12.
    """
    free = _peak_grids(lambda: J[0, 0] * Ex + J[0, 1] * Ey, grid)

    def _held():
        a = J[0, 0] * Ex                  # named -> refcount 2 -> not elidable
        b = J[0, 1] * Ey
        return a + b

    held = _peak_grids(_held, grid)
    assert 2.9 < held < 3.1, (
        f"the elision instrument read {held:.3f} full grids for two products "
        f"and their sum, which must cost exactly 3.00 -- it is not measuring "
        f"what it claims, so no premise can be drawn from it")
    return free, held


def _b8_jones_peaks(N=1024):
    """``(grid_bytes, J, Ex, Ey, old, new)`` -- the two peak readings, in units
    of ONE full-grid complex array."""
    grid = N * N * 16
    rng = np.random.default_rng(1)
    Ex = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Ey = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    J = np.array([[0.3 + 0.4j, -0.5 + 0.1j],
                  [0.2 - 0.7j, 0.9 + 0.05j]], dtype=complex)
    old = _peak_grids(lambda: _pre_fix_jones(J, Ex, Ey), grid)
    new = _peak_grids(lambda: apply_jones_matrix(
        JonesField(Ex, Ey, 1e-6, WL), J), grid)
    return grid, J, Ex, Ey, old, new


def test_b8_apply_jones_matrix_peak_full_grid_arrays():
    """The UNCONDITIONAL half: the shipped path sits ON its derived floor, in
    units of ONE full-grid complex array.

    3.00 is that floor -- two results plus one shared scratch.  Neither result
    can be written before both of its terms exist, and the inputs belong to
    the caller, so nothing can do better and the reading is two-sided: at
    least 3.00 and less than 4.00.

    MEASURED, 9 repeats on each arm: 3.000448 .. 3.000457 on Windows py3.14 /
    numpy 2.4.4 and 3.000448 .. 3.000457 on WSL py3.12 / numpy 2.4.6, i.e. the
    exact count plus <= 7.7 kB of tracemalloc bookkeeping on both.  No arm may
    make the shipped path the more expensive of the two, which is asserted
    here as well and holds whether or not NumPy elides the pre-fix form's
    third array.

    The reading that IS build-dependent -- the pre-fix expression's fourth
    grid -- is premise-gated in
    :func:`test_b8_the_pre_fix_jones_expression_holds_a_fourth_grid`.
    """
    _grid, _J, _Ex, _Ey, old, new = _b8_jones_peaks()
    assert 3.0 <= new < 3.0 + _B8_PEAK_SLACK, (old, new)
    assert new <= old + _B8_PEAK_SLACK, (old, new)


def test_b8_the_pre_fix_jones_expression_holds_a_fourth_grid():
    """PREMISE-GATED (TESTING_STANDARDS S3).  The pathology the shipped
    ``apply_jones_matrix`` removed -- the third full-grid array that
    ``J[1,0]*Ex + J[1,1]*Ey`` costs while the first component's result is
    still live -- is observable only on a build where NumPy does not elide
    that sum into one of its own operands.

    The premise is measured on the running arm by
    :func:`_numpy_elides_the_sum_of_two_temporaries`, never assumed from the
    platform.  Where it holds, the pre-fix expression peaks at 4.00 grids
    against the shipped path's 3.00 (Windows py3.14 / numpy 2.4.4: 4.000027 vs
    3.000460; and the py3.12 and py3.13 Linux runners of CI run 34914295323,
    which read 4.00 and passed).  Where NumPy elides, the pre-fix expression
    is rewritten into the same 3.00 and there is no fourth grid to see: WSL
    py3.12 / numpy 2.4.6 reads 3.000032 and so did the py3.14 runner of that CI
    run, which is what red this gate's previous ``old > 3.5`` form.  That arm
    asserts the collapse explicitly and then skips WITH the reading, so it can
    never pass silently.
    """
    grid, J, Ex, Ey, old, new = _b8_jones_peaks()
    free, held = _numpy_elides_the_sum_of_two_temporaries(J, Ex, Ey, grid)
    if free < held - 0.5:
        assert 3.0 <= old < 3.0 + _B8_PEAK_SLACK, (
            f"NumPy elided the pre-fix expression's third array, so it must "
            f"land on the same 3.00-grid floor as the shipped path, but it "
            f"read {old:.6f} (new={new:.6f})")
        pytest.skip(
            f"premise absent on this arm: NumPy's temporary elision is ACTIVE "
            f"for a*X + b*Y (it peaks at {free:.3f} full grids free, "
            f"{held:.3f} with both products name-bound), so the pre-fix "
            f"expression allocates no fourth grid -- measured old={old:.6f} "
            f"against new={new:.6f}, both on the 3.00 floor")
    assert 4.0 <= old < 4.0 + _B8_PEAK_SLACK, (
        f"elision is inactive here (free={free:.3f}, held={held:.3f}), so the "
        f"pre-fix expression must hold its fourth grid: old={old:.6f}, "
        f"new={new:.6f}")
    assert old - new > 0.95, (old, new)
'''

patch(P, OLD, NEW)
