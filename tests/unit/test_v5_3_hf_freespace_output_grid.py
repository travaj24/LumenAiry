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
    """End-to-end Parseval pin: total field power through the
    propagator + resample is preserved to within bicubic-interp
    noise on a well-confined Gaussian.

    v5.4 (audit P2): tightened from ``0.5 < ratio < 2.0`` (catastrophic-
    amplification sanity guard) to ``rel_err < 1e-3`` after the HF
    freespace wrapper gained the sqrt(p_in/p_out) Parseval
    renormalization step that mirrors mhs.py:602-606.  Pre-fix
    the edge-grazing case drifted ~1.6%; well-confined Gaussian
    drifts an order of magnitude less but the same renorm path
    bounds it to interp noise (<< 1e-3).
    """
    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    p_in = float(np.sum(np.abs(E) ** 2) * dx * dx)
    E_out, dx_out = la.propagate(
        E, z=z, wavelength=wavelength, dx=dx,
        method='hf', output_grid=(N, dx))
    p_out = float(np.sum(np.abs(E_out) ** 2) * dx_out * dx_out)

    # Tight Parseval pin -- the renorm step in
    # propagate_huygens_fresnel_freespace must restore total power.
    rel_err = abs(p_out - p_in) / max(p_in, 1e-30)
    assert rel_err < 1e-3, (
        f'HF freespace + resample Parseval drift rel_err = {rel_err:.6e}; '
        f'expected < 1e-3 -- the sqrt(p_in/p_out) renorm step in '
        f'propagate_huygens_fresnel_freespace (mirrors mhs.py:602-606) '
        f'must restore total power to within bicubic-interp noise.')


def test_hf_freespace_same_shape_short_circuit_is_bit_for_bit():
    """v5.4 (audit P2) pin (a): when the requested output grid exactly
    matches the input grid (same N, same dx), the resample branch
    must short-circuit and return the native RS-kernel output
    bit-for-bit -- no ``map_coordinates`` pass, no rescale, no
    dtype churn.  Mirrors mhs.py:583-587.
    """
    from lumenairy.propagators.hf import propagate_huygens_fresnel_freespace
    from lumenairy.propagators.propagation import rayleigh_sommerfeld_propagate

    N, dx = 64, 10e-6
    E = _build_test_field(N=N, dx=dx)
    z, wavelength = 0.01, 633e-9

    E_native = rayleigh_sommerfeld_propagate(E, z, wavelength, dx)
    E_short, dx_short = propagate_huygens_fresnel_freespace(
        E, z, wavelength, dx,
        output_shape=(N, N), output_dx=dx)

    assert dx_short == dx, (
        f'short-circuit must echo input dx exactly; got '
        f'{dx_short!r} vs {dx!r}.')
    assert np.array_equal(E_short, E_native), (
        'same-shape / same-dx short-circuit must return the native '
        'RS-kernel output bit-for-bit (no map_coordinates pass).')


def test_hf_freespace_edge_grazing_gaussian_parseval_renorm():
    """v5.4 (audit P2) pin (b): edge-grazing Gaussian upsample must
    preserve Parseval power to within 1e-3.  Pre-fix this drifted
    ~1.6e-2 because ``resample_field`` (bicubic) attenuates
    edge-pixel power when the support grazes the grid boundary.
    The sqrt(p_in/p_out) renorm step (mirrors mhs.py:602-606)
    restores total power.
    """
    from lumenairy.propagators.hf import propagate_huygens_fresnel_freespace

    N_in, dx = 256, 10e-6
    # Edge-grazing waist: w0 = 25 * dx places significant Gaussian
    # tail near the (N_in/2) * dx grid edge.
    w0 = 25.0 * dx
    x = (np.arange(N_in) - N_in / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (w0 ** 2)).astype(np.complex128)
    z, wavelength = 0.01, 633e-9

    # Upsample to N=512 at the same dx (covers double the extent).
    N_out, dx_out = 512, dx

    p_in = float(np.sum(np.abs(E) ** 2) * dx * dx)
    E_native = E  # we measure p_in on the input field for clarity.
    # Drive the resample path: native RS output is at the input grid,
    # but we request a larger N at the same dx.
    E_out, dx_returned = propagate_huygens_fresnel_freespace(
        E, z, wavelength, dx,
        output_shape=(N_out, N_out), output_dx=dx_out)

    # The Parseval invariant we actually pin is on the
    # RS-kernel-native vs resampled power -- it's the resample
    # step that the renorm targets.  Re-run the native kernel to
    # get the reference total.
    from lumenairy.propagators.propagation import rayleigh_sommerfeld_propagate
    E_ref = rayleigh_sommerfeld_propagate(E, z, wavelength, dx)
    p_ref = float(np.sum(np.abs(E_ref) ** 2) * dx * dx)
    p_out = float(np.sum(np.abs(E_out) ** 2) * dx_returned * dx_returned)

    rel_err = abs(p_out - p_ref) / max(p_ref, 1e-30)
    assert rel_err < 1e-3, (
        f'Edge-grazing Gaussian Parseval drift rel_err = {rel_err:.6e}; '
        f'expected < 1e-3.  Pre-fix this was ~1.6e-2 (1.6% drift) '
        f'because resample_field attenuates edge-pixel power; '
        f'the sqrt(p_in/p_out) renorm step in '
        f'propagate_huygens_fresnel_freespace (mirrors mhs.py:602-606) '
        f'must restore total power.')


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
