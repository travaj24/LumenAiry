"""G10 / S3-17 (AUDIT_V5_24_2) -- Source polarization/Jones channel.

Before the fix the ``Source`` dataclass had no polarization channel, so
vectorial pipelines had to bypass the container entirely.  The minimal
safe fix adds an OPTIONAL ``jones`` field (default ``None``) that:

  * is accepted by the constructor (keyword and via dataclass fields),
  * does not perturb any existing positional-construction call, and
  * is carried through ``Source.propagate`` UNCHANGED (the scalar
    propagators do not act on it -- it is preserved metadata).

These tests fail before the fix (``TypeError`` on the ``jones=`` kwarg /
``AttributeError`` on ``.jones``) and pass after.
"""
from __future__ import annotations

import numpy as np

import lumenairy as la


def _small_gaussian(N=32, dx=8e-6, w=40e-6):
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    return np.exp(-(Xg ** 2 + Yg ** 2) / w ** 2).astype(np.complex128), dx


def test_jones_defaults_to_none():
    """A scalar Source built the historical way has ``jones is None``."""
    E, dx = _small_gaussian()
    src = la.Source(E, dx, 0.633e-6)          # legacy positional form
    assert src.jones is None


def test_positional_construction_unchanged():
    """Adding ``jones`` LAST must not shift any positional argument."""
    E, dx = _small_gaussian()
    src = la.Source(E, dx, 0.633e-6, (1e-3, 2e-3), 'probe', 9e-6)
    assert src.dx == dx
    assert src.wavelength == 0.633e-6
    assert src.source_point == (1e-3, 2e-3)
    assert src.name == 'probe'
    assert src.dy == 9e-6
    assert src.jones is None


def test_jones_channel_stored_verbatim():
    """A uniform-polarization Jones 2-vector round-trips by identity."""
    E, dx = _small_gaussian()
    jvec = np.array([1.0, 1j], dtype=np.complex128)   # 45-deg circular-ish
    src = la.Source(E, dx, 0.633e-6, jones=jvec)
    assert src.jones is jvec
    assert np.array_equal(src.jones, jvec)


def test_jones_full_vector_field_stored():
    """A full (2, Ny, Nx) vectorial field is accepted as-is."""
    E, dx = _small_gaussian(N=16)
    vfield = np.stack([E, 0.5 * E]).astype(np.complex128)  # (2, 16, 16)
    src = la.Source(E, dx, 0.633e-6, jones=vfield)
    assert src.jones.shape == (2, 16, 16)
    assert np.array_equal(src.jones, vfield)


def test_propagate_preserves_jones_unchanged():
    """The scalar propagator transports E but hands ``jones`` through
    to the descendant Source byte-for-byte (identity object)."""
    E, dx = _small_gaussian()
    jvec = np.array([0.6, 0.8], dtype=np.complex128)
    src = la.Source(E, dx, 0.633e-6, name='src', jones=jvec)
    out = src.propagate(method='asm', z=1e-3)
    # E has changed (real propagation), jones is the SAME object.
    assert out.jones is jvec
    assert np.array_equal(out.jones, jvec)
    # A scalar Source (jones=None) still propagates to jones=None.
    src_scalar = la.Source(E, dx, 0.633e-6)
    out_scalar = src_scalar.propagate(method='asm', z=1e-3)
    assert out_scalar.jones is None
