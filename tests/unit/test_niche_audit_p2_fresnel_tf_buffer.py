"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 (P2): ``fresnel_tf_propagate`` must
not hand back the live pyFFTW inverse ping-pong buffer.

The bug
-------
``fresnel_tf_propagate`` returned ``_ifft2(...)`` directly.  On the
NumPy/pyFFTW path that is the plan-cache-owned inverse output buffer, whose
contents the double-buffer contract guarantees only until the NEXT same-key
``_ifft2`` call -- so the 2nd subsequent same-shape call silently rewrote an
already-returned field.  Measured at N=512, complex128, pyFFTW active:
after three further calls at other distances, ``max|A - A_snapshot| =
0.497`` on a peak-1 field, with ``A`` becoming byte-identical to a LATER
leg's result and ``A.base is not None`` (``id(A)`` present in
``fft_infra._PYFFTW_PLAN_CACHE``'s buffer set).

ASM and RS were already safe: ASM's exit ``fftshift`` allocates, and
``rs.py`` slice-``.copy()``s with this exact hazard documented (audit F-3).
The in-library reach was ``carrier.propagate_carrier_referenced``, which
stores this array straight into the returned
``CarrierReferencedField.env``.

The fix: copy on the way out (the ``rs.py`` F-3 precedent).  Where the
caller's dtype differs, the existing ``astype`` already allocates, so no
second copy is paid.

These pins hold under BOTH backends -- with pyFFTW inactive (or below
``FFTW_MIN_SIZE``) the underlying ifft2 already returns a fresh array, so
the assertions are simply satisfied for a different reason.

Self-contained, ~2 s.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators import fft_infra as _fi
from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.fresnel import fresnel_tf_propagate
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

LAM = 1.0e-6
DX = 1.0e-6
# >= FFTW_MIN_SIZE (256) so the pyFFTW double-buffer path is the one under
# test wherever pyFFTW is installed.
N = 512


def _beam(dtype=np.complex128):
    x = (np.arange(N) - N / 2) * DX
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / (25e-6) ** 2).astype(dtype)


def _plan_buffer_ids():
    """Every array currently owned by the pyFFTW plan cache."""
    ids = set()
    with _fi._PYFFTW_PLAN_LOCK:
        for entry in _fi._PYFFTW_PLAN_CACHE.values():
            for buf in entry['bufs']:
                ids.add(id(buf))
    return ids


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64, np.float64])
def test_result_survives_three_more_same_shape_calls(dtype):
    """The audit's repro: snapshot the first result, then run three more
    same-shape propagations at other distances.  Pre-fix the 2nd of those
    recycled the returned buffer (max|delta| = 0.497)."""
    E = _beam(dtype)
    first = fresnel_tf_propagate(E, 1.0e-3, LAM, DX)
    snapshot = first.copy()
    for z in (2.0e-3, 3.0e-3, 4.0e-3):
        fresnel_tf_propagate(E, z, LAM, DX)
    assert np.array_equal(first, snapshot), (
        'the returned field was overwritten by a later same-shape call')


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64, np.float64])
def test_result_owns_its_memory(dtype):
    """Ownership contract: the returned array must not be a view into (or
    the identity of) a cache-owned pyFFTW workspace."""
    out = fresnel_tf_propagate(_beam(dtype), 1.0e-3, LAM, DX)
    assert out.base is None, 'result is a view into someone else\'s buffer'
    if _fi.PYFFTW_AVAILABLE and _fi.USE_PYFFTW:
        assert id(out) not in _plan_buffer_ids(), (
            'result IS a live pyFFTW plan buffer')


def test_z_zero_identity_still_owns_and_copies():
    """The ``z == 0`` short-circuit returns ``E_in.copy()``; make sure it is
    a copy, not the caller's array."""
    E = _beam()
    out = fresnel_tf_propagate(E, 0.0, LAM, DX)
    assert out is not E
    assert out.base is None
    assert np.array_equal(out, E)
    E[0, 0] = 12.5 + 0.5j
    assert out[0, 0] != E[0, 0]


def test_copy_did_not_change_the_values():
    """Guard against 'fixed the aliasing, changed the physics': the copied
    result must equal a freshly computed one, and the field must still be
    the matched-paraxial kernel applied to the input."""
    E = _beam()
    a = fresnel_tf_propagate(E, 1.5e-3, LAM, DX)
    b = fresnel_tf_propagate(E, 1.5e-3, LAM, DX)
    assert np.array_equal(a, b)
    k = 2 * np.pi / LAM
    kx_sq = (2 * np.pi * np.fft.fftfreq(N, DX)) ** 2
    z = 1.5e-3
    phase = k * z - (z / (2 * k)) * (kx_sq[:, None] + kx_sq[None, :])
    ref = np.fft.ifft2(np.fft.fft2(E) * np.exp(1j * phase))
    assert np.max(np.abs(a - ref)) / np.abs(ref).max() < 1.0e-12


def test_sibling_propagators_keep_their_ownership_contract():
    """ASM / RS were the audit's 'safe' comparison points -- pin them so a
    future refactor cannot regress them into the same class."""
    E = _beam()
    for fn in (angular_spectrum_propagate, rayleigh_sommerfeld_propagate):
        first = fn(E, 1.0e-3, LAM, DX)
        snapshot = first.copy()
        for z in (2.0e-3, 3.0e-3, 4.0e-3):
            fn(E, z, LAM, DX)
        assert np.array_equal(first, snapshot), f'{fn.__name__} clobbered'
        assert first.base is None, f'{fn.__name__} returned a view'


def test_carrier_referenced_env_is_not_a_live_buffer():
    """The in-library consumer: ``propagate_carrier_referenced`` stores the
    ``fresnel_tf_propagate`` result into the returned envelope."""
    carrier = pytest.importorskip('lumenairy.propagators.carrier')
    E = _beam()
    # R_carrier = inf is the collimated-carrier branch, i.e. the one that
    # stores a bare ``fresnel_tf_propagate`` result into the envelope.
    out = carrier.propagate_carrier_referenced(
        E, np.inf, 1.0e-3, LAM, DX)
    env = np.asarray(getattr(out, 'env', out[0]))
    snapshot = env.copy()
    for z in (2.0e-3, 3.0e-3, 4.0e-3):
        carrier.propagate_carrier_referenced(E, np.inf, z, LAM, DX)
    assert np.array_equal(env, snapshot), (
        'CarrierReferencedField.env was overwritten by a later call')
