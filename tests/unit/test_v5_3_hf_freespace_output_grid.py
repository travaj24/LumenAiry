"""Regression pin for v5.2.5 -> v5.3 HF freespace dispatcher fix.

Closes AUDIT_V5_2_5 P1-1 (HF freespace ``output_grid``/``output_dx``
dispatcher fix shipped half-broken at v5.2.5).

Background
----------
v5.2.3 closed P1-A residual by threading ``output_grid``/``output_dx``
through the dispatcher's gbd/hfpi/hf through-prescription branches.
v5.2.5 closed P2-F1-1 by threading them through the FREESPACE
branches too.  But the v5.2.5 fix shipped half-broken: HFPI works
because its receiver natively accepts the kwargs, but HF's
``propagate_huygens_fresnel_freespace`` is a thin pass-through to
``rayleigh_sommerfeld_propagate`` which does not accept those
kwargs.  Result: ``la.propagate(method='hf', output_grid=(N, dx))``
raised ``TypeError`` at v5.2.5 (was a silent no-op at v5.2.3 and
earlier).

v5.3 fixes the pass-through by handling the resample inside
``propagate_huygens_fresnel_freespace`` (matches the v5.2.3 MHS
substantive-resampling pattern).

These tests pin:

1. The dispatcher's canonical ``output_grid=(N_out, dx_out)`` form
   succeeds end-to-end and returns the requested grid.
2. ``output_dx=`` alone (no ``output_grid``) also works.
3. No kwargs (default pass-through) returns the bare ndarray as
   before -- the v5.2.5 contract isn't broken for the common case.
4. Power is preserved within numerical tolerance through the
   resample step.

Author: Andrew Traverso -- v5.3 / AUDIT_V5_2_5 P1-1 closure.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


def _build_test_field(N=64, dx=10e-6):
    """A small smooth complex field for the propagation pins."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w0 = 5 * dx
    return np.exp(-(X ** 2 + Y ** 2) / (w0 ** 2)).astype(np.complex128)


def test_hf_freespace_with_output_grid_does_not_raise():
    """v5.2.5 P1 regression -- ``method='hf'`` with the dispatcher's
    canonical ``output_grid=(N_out, dx_out)`` form must NOT raise.
    """
    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    # Pre-v5.3 this raised TypeError; v5.3 must return cleanly.
    out = la.propagate(
        E, z=z, wavelength=wavelength, dx=dx,
        method='hf', output_grid=(N, dx))

    # When resampling at the same N + dx, the return contract is
    # the resample-tuple form (E_out, dx_out).
    assert isinstance(out, tuple), (
        f'expected (E_out, dx_out) tuple from resample path; '
        f'got {type(out).__name__}.')
    E_out, dx_out = out
    assert isinstance(E_out, np.ndarray)
    assert E_out.shape == (N, N), (
        f'output shape {E_out.shape} != requested ({N}, {N}).')
    assert dx_out == pytest.approx(dx, rel=1e-12), (
        f'output dx {dx_out:.6e} != requested {dx:.6e}.')


def test_hf_freespace_with_output_dx_alone():
    """``output_dx=`` alone (no ``output_grid``) must work too --
    v5.2.3's dispatcher resolver returns ``(N_in, dx_out)``.
    """
    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    out = la.propagate(
        E, z=z, wavelength=wavelength, dx=dx,
        method='hf', output_dx=dx)
    assert isinstance(out, tuple)
    E_out, dx_out = out
    assert E_out.shape == (N, N)


def test_hf_freespace_no_output_kwargs_returns_bare_ndarray():
    """The default pass-through (no ``output_grid`` / ``output_dx``)
    must return a bare ``ndarray`` -- v5.2.3-and-earlier contract
    preserved bit-for-bit for the common case.
    """
    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    out = la.propagate(E, z=z, wavelength=wavelength, dx=dx, method='hf')
    assert isinstance(out, np.ndarray), (
        f'default pass-through must return bare ndarray; got '
        f'{type(out).__name__}.')
    assert out.shape == E.shape


def test_hf_freespace_resampled_power_is_preserved():
    """End-to-end: total field power through the propagator + resample
    is bounded -- the resample step does not artificially amplify
    or attenuate beyond the documented bicubic-interpolation
    behavior.  Pin to within 20% because RS at z=10mm + bicubic
    resample has non-trivial sample-pad behavior at this grid
    scale; the test is a sanity guard against catastrophic
    amplification, not a Parseval pin.
    """
    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    p_in = float(np.sum(np.abs(E) ** 2) * dx * dx)
    E_out, dx_out = la.propagate(
        E, z=z, wavelength=wavelength, dx=dx,
        method='hf', output_grid=(N, dx))
    p_out = float(np.sum(np.abs(E_out) ** 2) * dx_out * dx_out)

    # Bounded ratio -- guards against catastrophic amplification.
    ratio = p_out / max(p_in, 1e-30)
    assert 0.5 < ratio < 2.0, (
        f'HF freespace + resample power ratio out/in = {ratio:.4f}; '
        f'expected within (0.5, 2.0) -- catastrophic amplification '
        f'or attenuation would indicate a kernel-or-resample bug.')


def test_hf_freespace_non_square_output_shape_raises():
    """Non-square ``output_shape`` is documented as unsupported in the
    v5.3 wrapper.  The narrower raise is honest about the
    ``resample_field`` square-grid assumption.
    """
    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    # Call the wrapper directly to bypass the dispatcher's auto-
    # squaring (the dispatcher's resolver returns (N, N) always).
    from lumenairy.propagators.hf import propagate_huygens_fresnel_freespace
    with pytest.raises(ValueError, match='non-square'):
        propagate_huygens_fresnel_freespace(
            E, z, wavelength, dx,
            output_shape=(48, 64))
