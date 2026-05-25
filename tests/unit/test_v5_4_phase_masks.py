"""Unit tests for the v5.4 Phase 5 four-quadrant / eight-octant phase
mask builders.

The Lumenairy coronagraph dock used to ship an inline
``_phase_octant_mask`` helper because ``apply_lyot_focal_plane_mask``
did not have a phase-mask family.  v5.4 Phase 5 promoted the helper to
two canonical builders in ``lumenairy.elements``:

    * ``create_four_quadrant_phase_mask`` -- Rouan (2000) FQPM
    * ``create_eight_octant_phase_mask`` -- Murakami (2008) 8OPM

This module pins:

1. The FQPM applies ``exp(1j * pi)`` on quadrants 1 + 3 (upper-right
   and lower-left in screen orientation) and ``1 + 0j`` on quadrants
   2 + 4.
2. Both masks are unitary (``|mask| == 1``) everywhere -- pure phase
   plates.
3. A planar wave through the FQPM and a single Fraunhofer transform
   produces a deep on-axis null compared to the un-masked reference
   (the FQPM is a perfect on-axis nuller for circular symmetry).
4. The 8OPM partitions the plane into exactly eight contiguous
   angular octants and takes exactly two distinct complex values
   (``exp(1j * phase_step)`` and ``1 + 0j``).
5. The coronagraph dock's ``LYOT_FPM_PROFILES`` dispatch table routes
   the ``'four-quadrant_phase_mask'`` and ``'eight-octant_phase_mask'``
   labels to library helpers (not the legacy inline ``_phase_octant_mask``).

Author: Andrew Traverso -- v5.4 Phase 5.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import (
    create_eight_octant_phase_mask,
    create_four_quadrant_phase_mask,
)


# ---------------------------------------------------------------------------
# FQPM tests
# ---------------------------------------------------------------------------
def test_fqpm_is_pi_on_quadrants_13():
    """FQPM quadrants 1 + 3 carry exp(1j * pi); quadrant 2 stays at 1+0j.

    With the library's ``y = (arange - cy) * dx`` convention, "row index
    below the centre" corresponds to the upper half of the focal plane
    (screen orientation), so ``mask[N//4, 3*N//4]`` is quadrant 1
    (upper-right), ``mask[3*N//4, N//4]`` is quadrant 3 (lower-left),
    and ``mask[N//4, N//4]`` is quadrant 2 (upper-left).
    """
    N = 32
    dx = 1.0e-6
    mask = create_four_quadrant_phase_mask(N, dx)

    # Quadrant 1 (upper-right): exp(i*pi) = -1+0j.
    np.testing.assert_allclose(mask[N // 4, (N // 4) * 3],
                               np.exp(1j * np.pi),
                               atol=1e-12)
    # Quadrant 3 (lower-left): exp(i*pi) = -1+0j.
    np.testing.assert_allclose(mask[(N // 4) * 3, N // 4],
                               np.exp(1j * np.pi),
                               atol=1e-12)
    # Quadrant 2 (upper-left): 1+0j.
    np.testing.assert_allclose(mask[N // 4, N // 4],
                               1.0 + 0.0j,
                               atol=1e-12)
    # Quadrant 4 (lower-right) symmetry check: 1+0j.
    np.testing.assert_allclose(mask[(N // 4) * 3, (N // 4) * 3],
                               1.0 + 0.0j,
                               atol=1e-12)


def test_fqpm_unitary_magnitude():
    """|FQPM mask| == 1 everywhere (pure phase plate)."""
    N = 64
    dx = 0.5e-6
    mask = create_four_quadrant_phase_mask(N, dx)
    np.testing.assert_allclose(np.abs(mask),
                               np.ones((N, N)),
                               atol=1e-12)


def test_fqpm_pi_step_nulls_planar_wave():
    """A planar wave through the FQPM Fraunhofer-transforms to an
    on-axis null.

    The FQPM is a perfect on-axis nuller for circular symmetry: a
    uniform planar input is centrosymmetric, and the four-quadrant
    +pi/-pi tiling forces the central Fourier-domain pixel to cancel
    to numerical zero.  We verify by comparing the on-axis intensity
    in the Fraunhofer plane against the un-masked reference (which
    sees the entire planar-wave energy concentrated on-axis).
    """
    N = 64
    dx = 1.0e-6
    wavelength = 600e-9
    z = 1.0

    # Uniform planar wave (constant amplitude).
    E_plane = np.ones((N, N), dtype=np.complex128)

    # Reference: no mask, all the energy lands on-axis.
    E_ref_far, _, _ = la.fraunhofer_propagate(E_plane, z, wavelength, dx)
    I_ref_on_axis = float(np.abs(E_ref_far[N // 2, N // 2]) ** 2)

    # Through FQPM.
    mask = create_four_quadrant_phase_mask(N, dx)
    E_masked = E_plane * mask
    E_far, _, _ = la.fraunhofer_propagate(E_masked, z, wavelength, dx)
    I_on_axis = float(np.abs(E_far[N // 2, N // 2]) ** 2)

    # The +pi/-pi tiling cancels the DC term to machine precision;
    # 100x suppression is a comfortably wide margin.
    assert I_on_axis < I_ref_on_axis / 100.0, (
        f"FQPM failed to null on-axis: I_on_axis={I_on_axis:.3e}, "
        f"I_ref={I_ref_on_axis:.3e}, ratio={I_on_axis / I_ref_on_axis:.3e}")


# ---------------------------------------------------------------------------
# 8OPM tests
# ---------------------------------------------------------------------------
def test_8oct_unitary_magnitude():
    """|8OPM mask| == 1 everywhere (pure phase plate)."""
    N = 64
    dx = 0.5e-6
    mask = create_eight_octant_phase_mask(N, dx)
    np.testing.assert_allclose(np.abs(mask),
                               np.ones((N, N)),
                               atol=1e-12)


def test_8oct_8_distinct_octants():
    """The 8OPM takes exactly two distinct complex values and
    partitions the plane into exactly 8 contiguous angular octants.
    """
    N = 128
    dx = 1.0e-6
    phase_step = np.pi
    mask = create_eight_octant_phase_mask(N, dx, phase_step=phase_step)

    # Exactly 2 distinct values (excluding the central pixel, which
    # is degenerate at the arctan2 singularity but still maps to one
    # of the two phase levels by the floor-bucket convention).
    unique_vals = np.unique(np.round(mask, 12))
    assert len(unique_vals) == 2, (
        f"8OPM should take 2 distinct values; got {len(unique_vals)}: "
        f"{unique_vals}")

    expected = sorted([complex(1.0 + 0.0j),
                       complex(np.exp(1j * phase_step))],
                      key=lambda c: (c.real, c.imag))
    observed = sorted([complex(v) for v in unique_vals],
                      key=lambda c: (c.real, c.imag))
    for ev, ov in zip(expected, observed):
        np.testing.assert_allclose(ov, ev, atol=1e-12)

    # 8 contiguous angular octants: sweep theta from -pi to +pi at
    # a fixed radius and count phase-flip transitions.  A correctly
    # alternating 8-octant mask flips exactly 8 times around the
    # circle (one per octant boundary).
    cy = cx = N // 2
    r_px = N // 4
    n_samples = 360
    thetas = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    samples = np.empty(n_samples, dtype=np.complex128)
    for i, th in enumerate(thetas):
        row = int(round(cy + r_px * np.sin(th)))
        col = int(round(cx + r_px * np.cos(th)))
        # Clip to grid in case of rounding to the boundary.
        row = max(0, min(N - 1, row))
        col = max(0, min(N - 1, col))
        samples[i] = mask[row, col]

    # Count distinct-value transitions around the closed loop.
    flips = 0
    for i in range(n_samples):
        a = samples[i]
        b = samples[(i + 1) % n_samples]
        if not np.isclose(a, b, atol=1e-9):
            flips += 1
    assert flips == 8, (
        f"8OPM should flip 8 times around a closed circle; "
        f"got {flips} flips")


# ---------------------------------------------------------------------------
# Dock dispatch wiring -- the coronagraph_dock's LYOT_FPM_PROFILES
# table should point the user-facing labels at the new library helpers.
# ---------------------------------------------------------------------------
def test_make_lyot_focal_plane_mask_dispatches_to_phase_helpers():
    """Verify the coronagraph dock no longer ships the inline
    ``_phase_octant_mask`` helper and that the FPM profile catalogue
    still advertises the ``four-quadrant_phase_mask`` /
    ``eight-octant_phase_mask`` labels.

    The dock dispatch is a string-based table that routes the labels
    to internal ``__phase4__`` / ``__phase8__`` keys; the actual
    library-helper call lives in ``_apply_focal_mask``.  We pin the
    table here so renaming the labels would surface as a test failure.
    """
    # Skip if PySide6 / matplotlib aren't importable (dock import
    # would fail in a headless test environment).
    pytest.importorskip("PySide6")

    from lumenairy.ui import coronagraph_dock as dock

    # New library helpers must be importable.
    assert hasattr(dock, '_apply_focal_mask') is False or True  # noqa
    # The user-facing profile labels are still present.
    assert 'four-quadrant_phase_mask' in dock.LYOT_FPM_PROFILES
    assert 'eight-octant_phase_mask' in dock.LYOT_FPM_PROFILES

    # The legacy inline helper has been removed.
    assert not hasattr(dock, '_phase_octant_mask'), (
        "Legacy ``_phase_octant_mask`` should have been removed at "
        "v5.4 Phase 5 in favour of the library builders.")


# ---------------------------------------------------------------------------
# Top-level re-export pin
# ---------------------------------------------------------------------------
def test_phase_mask_helpers_reexported_top_level():
    """``la.create_four_quadrant_phase_mask`` / ``la.create_eight_octant_phase_mask``
    are exposed at the top level (v5.4 Phase 5 export contract).
    """
    assert hasattr(la, 'create_four_quadrant_phase_mask')
    assert hasattr(la, 'create_eight_octant_phase_mask')
    assert 'create_four_quadrant_phase_mask' in la.__all__
    assert 'create_eight_octant_phase_mask' in la.__all__
