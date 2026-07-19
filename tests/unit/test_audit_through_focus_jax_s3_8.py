"""Regression test for audit finding S3-8 (v5.24.5) + its S3-19 sequel.

**S3-8 (v5.24.5).**  ``through_focus_scan_jax`` computed its
``best_focus_spot`` summary with an unguarded ``float(np.nanmin(rms_r))``.
On an all-zero scan the OLD inline metric path left ``rms_r`` at its
``np.full(n_z, np.nan)`` default (each plane was skipped by an
``I_sum > 0`` guard), so ``np.nanmin`` over an all-NaN slice emitted
``RuntimeWarning: All-NaN slice encountered`` -- and RAISED under
warnings-as-errors -- whereas the pure-NumPy twin ``through_focus_scan``
guards the same reduction and returns cleanly.  The v5.24.5 fix mirrored
the NumPy twin's ``if np.any(np.isfinite(rms_r)) else float('nan')`` guard
on the JAX path, so a dark scan produced ``best_focus_spot`` with no
``All-NaN slice`` warning.

**S3-19 sequel (v5.24.6).**  The S3-19 metric-consolidation then routed the
JAX backend's per-plane metrics through the SAME ``single_plane_metrics``
the NumPy backend uses, eliminating the hand-inlined twin.  On a zero
field ``single_plane_metrics`` -> ``beam_d4sigma`` returns 0 (not NaN), so
``rms_r`` is now all-**0** (finite) on a dark scan and
``best_focus_spot`` is **0.0** on BOTH backends -- the former JAX
NaN-vs-NumPy 0.0 divergence is gone.  The all-NaN guard remains in place as
defence (it is simply no longer exercised by a dark scan), so the
``All-NaN slice`` warning STILL cannot fire.

Independent oracle: the pure-NumPy ``through_focus_scan`` (a different code
path) is driven on the identical all-zero input; the JAX path must (a) not
emit the ``All-NaN slice`` warning and (b) return the SAME
``best_focus_spot`` the NumPy backend does (0.0).

Measured (WSL venv):
    backend / call                 pre-S3-8         post-S3-8        post-S3-19
    through_focus_scan (NumPy)      clean; 0.0       clean; 0.0       clean; 0.0
    through_focus_scan_jax          warn/raises      clean; NaN       clean; 0.0
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _jax_or_skip():
    """Import jax, skipping the whole test when it is absent -- mirrors the
    sibling ``test_audit_through_focus_jax_x64._jax_or_skip`` importorskip
    guard so the JAX-only regression is skipped, not errored, on a
    numpy-only install."""
    return pytest.importorskip("jax")


def test_through_focus_scan_jax_all_zero_no_all_nan_warning():
    """All-zero scan: the JAX ``best_focus_spot`` reduction must stay guarded
    like the NumPy twin -- no ``All-NaN slice`` RuntimeWarning (which raises
    under warnings-as-errors) -- and, post-S3-19, must return the SAME value
    the NumPy backend does (0.0), not the pre-S3-19 NaN.

    Fail-before/pass-after: pre-S3-8 the unguarded ``np.nanmin(rms_r)``
    emitted ``RuntimeWarning: All-NaN slice encountered`` and raised it under
    ``-W error``; the guard suppresses it.  Post-S3-19 the dark-scan
    ``best_focus_spot`` is 0.0 on both backends (was NaN on the JAX path),
    pinning cross-backend parity on a dark scan.
    """
    _jax_or_skip()
    from lumenairy.analysis.through_focus import (
        through_focus_scan,
        through_focus_scan_jax,
    )

    N, dx, wl = 64, 8e-6, 1.55e-6
    # complex64 -> the S3-2 x64 guard stays silent, so the ONLY RuntimeWarning
    # that can fire on this dark scan is the S3-8 "All-NaN slice" one.
    E = np.zeros((N, N), dtype=np.complex64)
    z = np.linspace(45e-3, 55e-3, 9)

    # Independent oracle: the pure-NumPy twin (different code path) completes
    # with no "All-NaN slice" warning on the identical all-zero input and
    # returns a clean (finite 0.0) best_focus_spot.
    with warnings.catch_warnings(record=True) as caught_np:
        warnings.simplefilter("always")
        ref = through_focus_scan(
            E, dx, wl, z, bandlimit=True, verbose=False)
    assert not [w for w in caught_np
                if "All-NaN slice" in str(w.message)], (
        "the NumPy oracle must not warn on an all-zero scan -- it guards "
        "best_focus_spot in through_focus_scan.")
    assert np.isfinite(ref.best_focus_spot), (
        "post-S3-19 the NumPy dark-scan best_focus_spot is a finite 0.0 "
        "(beam_d4sigma returns 0 on a zero field).")
    assert ref.best_focus_spot == 0.0

    # Strict fail-before/pass-after: promote ONLY the All-NaN slice warning to
    # an error so unrelated benign warnings don't confound the assertion.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "error", message="All-NaN slice encountered",
            category=RuntimeWarning)
        got = through_focus_scan_jax(E, dx, wl, z, bandlimit=True)

    # S3-19 backend parity: the JAX dark scan must match the NumPy backend
    # (0.0), not the pre-S3-19 NaN, and must be finite/clean.
    assert np.isfinite(got.best_focus_spot), (
        "post-S3-19 an all-zero JAX scan returns a finite best_focus_spot "
        "(rms_r is 0 via the shared single_plane_metrics), not NaN.")
    assert got.best_focus_spot == ref.best_focus_spot == 0.0, (
        "an all-zero JAX scan must return the SAME best_focus_spot as the "
        "NumPy backend (0.0) -- audit S3-19 metric consolidation.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
