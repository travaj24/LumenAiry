"""Regression test for audit finding S3-8 (v5.24.5).

``through_focus_scan_jax`` computed its ``best_focus_spot`` summary with an
unguarded ``float(np.nanmin(rms_r))`` (through_focus.py:1281).  On an all-zero
scan every plane has ``I_sum == 0`` so the JAX per-plane path never writes
``rms_r[i]`` and the whole array stays at its ``np.full(n_z, np.nan)`` default.
``np.nanmin`` over an all-NaN slice then emits ``RuntimeWarning: All-NaN slice
encountered`` -- and RAISES under warnings-as-errors -- whereas the pure-NumPy
twin ``through_focus_scan`` guards the same reduction (lines 461-462:
``float(np.nanmin(rms_r)) if np.any(np.isfinite(rms_r)) else float('nan')``)
and returns cleanly.  So a backend switch silently changed a clean summary into
a warning / potential crash.

The fix mirrors the NumPy twin's guard on the JAX path, so ``best_focus_spot``
is produced with no ``All-NaN slice`` warning on a dark scan, matching the
NumPy backend's crash-free behaviour.

Independent oracle: the pure-NumPy ``through_focus_scan`` (a different code
path) is driven on the identical all-zero input and shown to complete without
the ``All-NaN slice`` warning -- the JAX path must now do the same.

Measured (WSL venv):
    backend / call                 pre-fix                 post-fix
    through_focus_scan (NumPy)      clean, no warning       clean, no warning
    through_focus_scan_jax          "All-NaN slice" warn;   clean, no warning;
                                    raises under -W error   best_focus_spot=NaN
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
    """All-zero scan: the JAX ``best_focus_spot`` reduction must be guarded
    like the NumPy twin -- no ``All-NaN slice`` RuntimeWarning (which raises
    under warnings-as-errors) and a clean NaN result.

    Fail-before/pass-after: pre-fix ``through_focus_scan_jax`` on an all-zero
    field emitted ``RuntimeWarning: All-NaN slice encountered`` from the
    unguarded ``np.nanmin(rms_r)`` and raised it under ``-W error``; post-fix
    the guard suppresses it and returns ``best_focus_spot == nan``.
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
    # with no "All-NaN slice" warning on the identical all-zero input.
    with warnings.catch_warnings(record=True) as caught_np:
        warnings.simplefilter("always")
        ref = through_focus_scan(
            E, dx, wl, z, bandlimit=True, verbose=False)
    assert not [w for w in caught_np
                if "All-NaN slice" in str(w.message)], (
        "the NumPy oracle must not warn on an all-zero scan -- it guards "
        "best_focus_spot at through_focus_scan lines 461-462.")
    assert np.isfinite(ref.best_focus_spot) or np.isnan(ref.best_focus_spot)

    # Strict fail-before/pass-after: promote ONLY the All-NaN slice warning to
    # an error so unrelated benign warnings don't confound the assertion.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "error", message="All-NaN slice encountered",
            category=RuntimeWarning)
        got = through_focus_scan_jax(E, dx, wl, z, bandlimit=True)

    assert np.isnan(got.best_focus_spot), (
        "an all-zero JAX scan must return NaN best_focus_spot cleanly "
        "(audit S3-8 guard mirrors the NumPy twin).")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
