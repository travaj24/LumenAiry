"""Wave-5 D item (5): de-contaminate a11's estimate_lens_memory('real') gate."""
from _patch import patch

P = "tests/unit/test_audit2609_a11_polar_sources_infra.py"

OLD = '''def test_z3_estimate_lens_memory_real_bounds_apply_real_lens():
    """``lens_model='real'`` is the DOCUMENTED model for ``apply_real_lens``;
    it under-predicted the measured peak by 1.6x (default parallel_amp=True)
    to 2.8x (parallel_amp=False), i.e. a pre-flight budget under-reserved --
    the exact failure ``check_sim_memory`` exists to prevent.

    Bar: the estimate must BOUND the measurement (>= 1.0, the fail-safe
    direction) and not by more than 1.6x (an estimate that over-reserves by
    more than that stops being useful).  Measured 2026-09-12 at N = 512 /
    1024 / 2048 and both complex dtypes: 1.06-1.07.  The measured peak is
    pure N^2 with 1.4 % spread across those grids, so the reading has no
    per-build knife edge; a future reduction of ``apply_real_lens``'s own
    peak moves the ratio UP (still fail-safe) and is what the upper bar is
    there to surface.
    """
    N, dt = 512, np.complex128
    wl, dx = 633e-9, 30e-3 / N
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                         aperture=25e-3)
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3, dtype=dt)
    E = np.ascontiguousarray(E)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.apply_real_lens(E[:64, :64].copy(), prescription=rx,
                           wavelength=wl, dx=dx)          # warm the caches
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del out
    gc.collect()
'''

NEW = '''#: Grid the warm-up call runs on, in pixels.  DERIVED, not chosen: at 64 it is
#: below the size at which ``apply_real_lens`` takes its deferred-import branch,
#: so ~11.2 MB of one-time module imports (9.55 MB of bytecode at
#: ``<frozen importlib._bootstrap_external>``, plus dask / jinja2 / pathlib /
#: inspect objects) landed INSIDE the measured region on Windows, where those
#: modules are not already resident -- 0.45 MB of the same line on Linux, which
#: is why the bar below held there and read 0.81 here.  At 256 the warm-up takes
#: the same branch and the measured peak is BYTE-IDENTICAL on the two arms.
_Z3_LENS_WARM_N = 256


def test_z3_estimate_lens_memory_real_bounds_apply_real_lens():
    """``lens_model='real'`` is the DOCUMENTED model for ``apply_real_lens``;
    it under-predicted the measured peak by 1.6x (default parallel_amp=True)
    to 2.8x (parallel_amp=False), i.e. a pre-flight budget under-reserved --
    the exact failure ``check_sim_memory`` exists to prevent.

    Bar: the estimate must BOUND the measurement (>= 1.0, the fail-safe
    direction) and not by more than 1.6x (an estimate that over-reserves by
    more than that stops being useful).  Neither bar is moved here; what is
    fixed is the MEASUREMENT, which was picking up allocations that are not
    ``apply_real_lens``'s working set at all (see ``_Z3_LENS_WARM_N``).

    MEASURED after, in FRESH processes, one (N, dtype) each, on Windows
    py3.14 / numpy 2.4.4 AND WSL py3.12 / numpy 2.4.6 -- every number below is
    byte-identical on the two arms, which is the point::

        N     dtype       estimate   first call   est/peak   retained
        512   complex128    49.5 MB     46.5 MB     1.064     6.09 grids
        1024  complex128   198.0 MB    159.8 MB     1.239     6.02 grids
        512   complex64     33.8 MB     31.5 MB     1.072     6.01 grids
        1024  complex64    135.1 MB    102.1 MB     1.323     6.04 grids

    so the gate's own row (512, complex128) sits 6.4 % above the fail-safe
    bar and 33 % below the upper one, and the worst row of the four is still
    21 % inside the upper bar.

    The RETAINED column is asserted too, as the premise that the reading is
    clean: the call keeps the N-sized FFT and ASM caches it built
    (``propagators/fft_infra.py`` 4 complex grids, ``propagators/asm.py`` 1 + 1)
    and nothing else, 6.01 .. 6.09 grids on every row and both arms.  With the
    old 64-pixel warm-up the same reading was 9.49 grids on Windows -- the
    deferred imports -- so this guard catches exactly the contamination that
    red this test, two-sided: 15 % above the worst clean reading and 36 %
    below the contaminated one.
    """
    N, dt = 512, np.complex128
    wl, dx = 633e-9, 30e-3 / N
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                         aperture=25e-3)
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3, dtype=dt)
    E = np.ascontiguousarray(E)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.apply_real_lens(E[:_Z3_LENS_WARM_N, :_Z3_LENS_WARM_N].copy(),
                           prescription=rx, wavelength=wl, dx=dx)
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del out
    gc.collect()
    held = retained / float(N * N * 16)
    assert 5.5 < held < 7.0, (
        f"the measured call retained {held:.2f} full complex grids, not the "
        f"6.0 of N-sized FFT/ASM cache it builds -- something else is being "
        f"allocated inside the measured region (with a 64-pixel warm-up this "
        f"reads 9.49 on Windows: deferred module imports), so the peak below "
        f"is not apply_real_lens's working set")
'''

patch(P, OLD, NEW)
