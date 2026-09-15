"""The dense GBD reconstruction's memory budget is a bound -- measured.

WHY THIS FILE EXISTS.  ``HANDOFF_2026_09_14.md`` section 1 records an
unexplained interpreter crash: a long unit run on the maintainer's box died
twice with ``Windows fatal exception: access violation``, once inside the dense
``reconstruct_field_from_beamlets`` path of ``lumenairy/propagators/gbd.py``,
"which passes alone in 30 s".  A native fault in a loop that contains no native
code of its own, appearing only beside other heavy jobs, is an ALLOCATION
story, so the question asked here is the one that can be answered by
measurement: does that loop's ``mem_budget_mb`` bound what it allocates?

WHAT WAS MEASURED (2026-09-14, ``validation/probe_known_reds/
probe_gbd_dense_budget.py``, py3.14.6 / numpy 2.4.4, tracemalloc, a
64/128/192/256 grid ladder x {512, 64} MB budgets x {512, 1024} beamlets):

    declared cost   16.0 B per (output cell x beamlet-column)
    measured peak   72.0 to 96.8 B per (cell x beamlet-column)
    overrun         1.18x to 6.05x, saturating at 6.0x = 96/16
    worst cell      mem_budget_mb=512, 256^2 grid, 1024 beamlets -> 3 073 MB

The shipped constant is the size of ONE complex128 element where the comment
claims it covers three float64 buffers plus one complex128 ("the
dX/dY/rho2/phase buffers").  The WINDOWED sibling does the same accounting
correctly (``_WINDOWED_CELL_BYTES = 32.0``, with a written per-array tally of
~26 B/cell measured), which is the in-repo precedent for the shape of the fix.

WHAT IS PINNED HERE.

  1. The default is ``'legacy'`` and is BYTE-IDENTICAL -- correcting the
     constant moves the chunk boundary, hence the summation order of the
     per-chunk reductions, hence the output bytes, so it ships opt-in.
  2. The overrun is REAL and is not a fixture artefact: with the legacy
     accounting the measured peak exceeds the requested budget by a wide,
     stated factor.  This is the fail-before arm, asserted rather than assumed.
  3. With ``'measured'`` the peak is UNDER the requested budget -- i.e. the
     switch makes the budget a bound, which is the whole claim.
  4. The two arms differ only by round-off, which is the evidence that the
     cost of the fix is bit-identity and nothing else.

THE BARS ARE TWO-SIDED AND DERIVED.  ``tracemalloc`` counts Python-level
allocations, which is reproducible across arms in a way RSS is not, but it is
still a measurement with spread, so neither bar sits on a measured digit:

  * the fail-before bar is 3.0x the budget, half of the 6.0x the ladder
    saturates at and 2.5x above the smallest overrun the ladder measured on
    this cell -- a real repair drops the overrun below 1.0x, so the gap
    between the bar and the repaired state is a factor of three;
  * the repaired bar is 1.0x the budget (the budget must actually bound),
    measured 0.76x, so it clears by 1.3x;
  * the round-off bar is 1e-12 relative against a measured 2.1e-17, eight
    orders of margin, and it is an UPPER bound on a quantity a real change of
    physics would push far above it.

NOT ASSERTED, deliberately: that this is the cause of the access violation.
No fault was reproduced here.  What is established is that the transient is up
to six times the size the caller asked for, which is a defect on its own terms.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from lumenairy.propagators import gbd as G

_BUDGET_MB = 512.0
_N = 256
_NB = 1024


@pytest.fixture
def _restore_accounting():
    old = G.DENSE_MEM_BUDGET_ACCOUNTING
    yield
    G.DENSE_MEM_BUDGET_ACCOUNTING = old


def _bundle(n=_NB, seed=0):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 2.0e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.0e-3))


def _run(mode, bundle, budget_mb=_BUDGET_MB, N=_N):
    """``(field, peak_bytes)`` for one accounting mode."""
    G.DENSE_MEM_BUDGET_ACCOUNTING = mode
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = G.reconstruct_field_from_beamlets(
            bundle, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
            chunk_beamlets=4096, mem_budget_mb=budget_mb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return np.asarray(out), int(peak - base)


def test_the_shipped_default_is_the_legacy_accounting():
    """The switch exists; its default is the state that reproduces the
    previous release.  A correction that silently moved a default path's bytes
    would be the wrong shape for this defect, however right the arithmetic."""
    assert G.DENSE_MEM_BUDGET_ACCOUNTING == 'legacy'
    assert G._DENSE_CELL_BYTES_LEGACY == 16.0
    # The honest constant must be above every measurement on the ladder
    # (96.8 B/cell-col) and is carried with the same margin the windowed
    # sibling ships.  Both facts are what make it a bar rather than a reading.
    assert G._DENSE_CELL_BYTES_MEASURED >= 96.8
    assert G._DENSE_CELL_BYTES_MEASURED >= G._WINDOWED_CELL_BYTES


def test_the_legacy_accounting_does_not_bound_the_loop(_restore_accounting):
    """FAIL-BEFORE, asserted.  ``mem_budget_mb`` is documented as a cap on the
    dense path's per-chunk working set.  Measured, it is not one: the peak
    overruns by 6.0x on this cell.  The bar is 3.0x -- half the measured
    overrun, and three times the repaired state -- so it cannot be tripped by
    allocator noise in either direction."""
    _field, peak = _run('legacy', _bundle())
    overrun = peak / (_BUDGET_MB * 1e6)
    assert overrun > 3.0, (
        f"the dense path no longer overruns its budget ({overrun:.2f}x of "
        f"{_BUDGET_MB:.0f} MB, peak {peak / 1e6:.1f} MB).  If the accounting "
        f"was repaired in place, this test's premise is gone and the switch "
        f"below should go with it -- do not simply relax this bar.")


def test_the_measured_accounting_makes_the_budget_a_bound(_restore_accounting):
    """THE CLAIM.  With the honest per-cell cost the peak sits UNDER what the
    caller asked for.  Measured 0.76x of the budget; the bar is 1.0x, which is
    the definition of the word 'budget' rather than a tuned number."""
    _field, peak = _run('measured', _bundle())
    ratio = peak / (_BUDGET_MB * 1e6)
    assert ratio < 1.0, (
        f"peak {peak / 1e6:.1f} MB against a {_BUDGET_MB:.0f} MB budget "
        f"({ratio:.2f}x)")


def test_the_two_accountings_differ_only_by_summation_order(
        _restore_accounting):
    """The COST of the repair, pinned: the chunk boundary moves, so the order
    the per-chunk reductions are summed in moves, and floating-point addition
    is not associative.  Nothing else changes -- the fields agree to round-off
    (measured 2.1e-17 relative), which is what says the switch is an
    allocation decision and not a physics one."""
    b = _bundle()
    a, _pa = _run('legacy', b)
    c, _pc = _run('measured', b)
    scale = float(np.max(np.abs(a)))
    assert scale > 0.0
    rel = float(np.max(np.abs(a - c))) / scale
    assert rel < 1.0e-12, (
        f"the two accountings differ by {rel:.3e} relative, far above the "
        f"summation-order round-off this can only be.  Something other than "
        f"the chunk boundary moved.")


def test_an_unknown_accounting_is_treated_as_legacy(_restore_accounting):
    """The switch is read, not validated, at a hot site, so an unrecognised
    value must fall to the SHIPPED behaviour rather than to the new one: a
    typo must not silently move a user's bytes."""
    b = _bundle(n=128)
    a, _ = _run('legacy', b, N=64)
    G.DENSE_MEM_BUDGET_ACCOUNTING = 'not-a-mode'
    c, _ = _run('not-a-mode', b, N=64)
    assert np.array_equal(a, c)
