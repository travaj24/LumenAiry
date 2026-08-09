"""Threaded RCWAStack.solve_vs_wavelength (parallel, byte-identical).

Per-wavelength solves are independent and NumPy releases the GIL inside LAPACK,
so RCWAStack.solve_vs_wavelength runs them on a bounded thread pool (each on a
private clone of the stack).  Results are stored by index, so the output is
BYTE-IDENTICAL to a serial sweep regardless of worker count.  These tests pin
that identity, the single-wavelength edge case, and the NaN/unstable handling.

WHY THE BLAS CAP IS PINNED STRUCTURALLY HERE (M4, 2026-08-04).  Byte-identity
is not free: it holds only because every solve, serial or threaded, runs at the
SAME BLAS thread count.  The sweep used to enter ``_blas_threads_quiet`` inside
each worker, on the belief that the cap was thread-local.  It is not --
``threadpoolctl`` applies a cap on OpenBLAS via ``openblas_set_num_threads()``,
which is PROCESS-GLOBAL (measured directly: a worker thread entering
``threadpool_limits(1)`` takes the main thread's reported pool to 1 too).  N
concurrent enter/exit pairs on one global therefore race, and the first worker
to exit restored the environment's 24-thread pool while its siblings were still
inside ``solve`` -> a different GEMM/LAPACK reduction order -> different last
bits.

MEASURED before the fix, Windows / scipy-openblas 0.3.31 / 24 threads /
py3.14 / numpy 2.4.4, this exact 12-point sweep, four consecutive runs:
``max |T(4 workers) - T(serial)|`` = 5.88e-15, 1.89e-15, 0.0, 0.0 (R and the
Jones matrix stayed bit-exact throughout).  The test below failed on 4 of 6
runs.  With ``OPENBLAS_NUM_THREADS=1`` the same code was 4/4 bit-exact,
because then the racing save/restore is 1 -> 1 -> 1.  After the fix (one cap
around the whole dispatch): 10/10 bit-exact at 4 and 8 workers, and the SERIAL
result is unchanged -- sha256 of (R, T, jones) is identical for
{1, 4} workers x {default pool, OPENBLAS_NUM_THREADS=1}, so the fix removed
nondeterminism without moving a single bit.

The race needs BOTH ``threadpoolctl`` installed AND an environment pool > 1, so
it CANNOT reproduce on the 2-core CI runner or on a build without
``threadpoolctl``.  A byte-identity assertion alone is therefore green on CI
whether or not the bug is present -- which is how it survived.  Hence
``test_threaded_sweep_applies_exactly_one_blas_cap``: it pins the FIX (one cap
application per sweep, not one per worker) rather than its symptom, and it is
build-independent.
"""
from __future__ import annotations

import warnings

import numpy as np

from lumenairy.elements.rcwa import RCWAStack


def _pillar(S, hw=0.2, eps=8.0, bg=2.1):
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = np.full((S, S), bg, complex)
    c[(np.abs(X) < hw) & (np.abs(Y) < hw)] = eps
    return c


def _stack():
    st = RCWAStack(period=0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=5, n_orders_y=5)
    st.add_layer(0.08e-6, eps=2.25)
    st.add_layer(0.12e-6, eps_cell=_pillar(40))
    return st.set_source(0.55e-6, theta=0.1, phi=0.2)


def test_threaded_sweep_is_byte_identical_to_serial():
    """Serial and threaded agree bit for bit, across SEVERAL worker counts.

    Widened from a single 1-vs-4 comparison (M4): the pre-fix race fired
    nondeterministically, so one comparison passed by luck on ~1 run in 3 and
    the pin reported green while the contract was broken.  Byte-identity is
    asserted as a TOLERANCE AT 0.0 on the max absolute difference (the repo's
    standing rule) rather than ``array_equal``, so a failure reports the
    magnitude -- the number that told us this was a few-ULP BLAS reduction
    reorder and not a logic error.

    COST IS HELD AT PARITY with the single-comparison version it replaces: 12
    wavelengths x 2 sweeps (serial + 4 workers) = 24 solves before, 6 x 4
    (serial + 2 + 4 + 8) = 24 solves now.  That matters on a build with no
    ``threadpoolctl``, where the per-sweep BLAS cap is inert and 8 concurrent
    uncapped solves thrash a 24-thread pool -- at 12 wavelengths this file did
    not finish inside 25 minutes on the WSL/OpenBLAS build, against a repo that
    has already lost a release-verify shard to a 30-minute cap.
    """
    wls = np.linspace(0.50e-6, 0.62e-6, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o1, R1, T1, J1 = _stack().solve_vs_wavelength(wls, max_workers=1)
        ref = {}
        for nw in (2, 4, 8):
            ref[nw] = _stack().solve_vs_wavelength(wls, max_workers=nw)
    for nw, (o2, R2, T2, J2) in ref.items():
        assert np.array_equal(np.asarray(o1), np.asarray(o2)), nw
        # NaN-safe byte comparison (unstable points, if any, are NaN in both)
        for name, a, b in (("R", R1, R2), ("T", T1, T2), ("jones", J1, J2)):
            d = float(np.max(np.abs(np.nan_to_num(a) - np.nan_to_num(b))))
            assert d == 0.0, f"{name} moved by {d:.3e} at max_workers={nw}"
        assert np.array_equal(np.isnan(R1), np.isnan(R2)), nw


def test_threaded_sweep_applies_exactly_one_blas_cap():
    """The BLAS cap is applied ONCE per sweep, not once per worker.

    This is the build-INDEPENDENT pin on the byte-identity fix.  The race it
    guards against needs a >1-thread environment pool AND ``threadpoolctl``
    installed, neither of which holds on the 2-core CI runner or on the
    WSL/OpenBLAS build, so the byte-identity assertion above is green there
    whether or not the bug is present.  Counting cap applications works on any
    machine.

    It needs NO real limiter to be INSTALLED: substituting a counting
    stand-in for ``_get_blas_controller`` makes ``_blas_limit`` route through
    it regardless of whether ``threadpoolctl`` is present, and counting is the
    whole point -- no numerics are involved.  Counting there catches EVERY
    application, the sweep-level one and any nested per-solve one, because
    ``_core._blas_limit`` resolves the controller by module-global lookup at
    call time, from whichever module calls it.

    The stand-in DELEGATES to the real controller when there is one.  That is
    not incidental: a stand-in that merely counts and returns a null context
    silently DISABLES the cap for the duration, so the sweep's worker threads
    each run BLAS at the environment pool (24 here) -- 4 solver threads x 24 =
    96 OpenBLAS threads in one process, against a library built
    ``MAX_THREADS=24``.  Under ``pytest -n 6`` that reliably took a worker down
    ("node down: Not properly terminated") while passing alone and at ``-n 2``.
    An instrument must not switch off the thing it is instrumenting.

    Exactly 1 on BOTH the serial and the threaded branch is the load-bearing
    part: the serial loop runs on the caller's own thread, where the
    sweep-level request is set, so without the explicit clear in ``_solve_one``
    it nested one enter/exit per wavelength (measured: 7 for a 6-point serial
    sweep) inside the sweep-level one.

    The stack here is deliberately TINY (``n_orders=1``, 3 wavelengths): this
    test asserts a control-flow invariant, not a physical result, so it has no
    business running a physics-scale solve.
    """
    import contextlib

    from lumenairy.elements.rcwa import _core
    real = _core._get_blas_controller()          # None without threadpoolctl
    calls = []

    class _CountingController:
        def limit(self, **kw):
            calls.append(kw)
            # delegate, so the cap this test is measuring stays IN FORCE
            return real.limit(**kw) if real is not None \
                else contextlib.nullcontext()

    st = RCWAStack(period=0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=1, n_orders_y=1)
    st.add_layer(0.08e-6, eps=2.25)
    st.add_layer(0.12e-6, eps_cell=_pillar(6))
    st.set_source(0.55e-6, theta=0.1, phi=0.2)
    wls = np.linspace(0.54e-6, 0.56e-6, 3)

    prev = _core._get_blas_controller
    try:
        _core._get_blas_controller = lambda: _CountingController()
        for nw in (1, 2, 4):
            calls.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.solve_vs_wavelength(wls, max_workers=nw)
            assert len(calls) == 1, (
                f"max_workers={nw}: BLAS cap applied {len(calls)} times, "
                f"expected exactly 1 (once around the whole sweep).  More "
                f"than one application means concurrent enter/exit pairs on "
                f"a PROCESS-GLOBAL setting -- the byte-identity race.")
            assert calls[0]["user_api"] == "blas"
            assert calls[0]["limits"] == 1        # blas_per_worker default
    finally:
        _core._get_blas_controller = prev


def test_threaded_sweep_single_wavelength():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = _stack().solve_vs_wavelength([0.55e-6])   # auto workers
    assert R.shape[0] == 1 and T.shape[0] == 1 and J.shape[0] == 1
    # a single point -> serial path (max_workers collapses to 1); energy sane
    assert np.isfinite(R).any()


def test_threaded_sweep_matches_per_wavelength_solve():
    """A threaded sweep point matches a direct single-wavelength solve()."""
    wls = np.array([0.53e-6, 0.57e-6, 0.61e-6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = _stack().solve_vs_wavelength(wls, max_workers=3)
        direct = _stack().set_source(0.57e-6, theta=0.1, phi=0.2).solve()
    od, Rd, Td = direct.efficiencies()
    assert np.max(np.abs(np.nan_to_num(R[1]) - np.asarray(Rd))) < 1e-12
    assert np.max(np.abs(np.nan_to_num(T[1]) - np.asarray(Td))) < 1e-12
