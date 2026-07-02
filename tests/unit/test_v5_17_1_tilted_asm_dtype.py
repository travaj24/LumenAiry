"""v5.17.1: angular_spectrum_propagate_tilted honours the dtype-follows-input
contract.

Pre-fix the carrier was built unconditionally complex128, silently upcasting
the whole tilted pipeline for complex64 inputs (2x memory) and RETURNING
complex128.  Post-fix the carrier is built at the target dtype with the
carrier phase folded mod 2*pi in float64 before the float32 cast (the same
accuracy mitigation as the main ASM kernel), so the complex64 path is both
lean and accurate; the complex128 path is bit-identical to pre-fix.
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate_tilted


@pytest.fixture(autouse=True)
def _deterministic_fft():
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(True)


def _tilted_gauss(N, dx, tilt_x, lam, dtype):
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    k0 = 2 * np.pi / lam
    E = (np.exp(-(X**2 + Y**2) / (0.2e-3) ** 2)
         * np.exp(1j * k0 * np.sin(tilt_x) * X))
    return E.astype(dtype)


LAM = 1.31e-6


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_tilted_asm_output_dtype_follows_input(dtype):
    N, dx = 256, 2e-6
    E = _tilted_gauss(N, dx, np.deg2rad(15), LAM, dtype)
    out = angular_spectrum_propagate_tilted(
        E, z=0.5e-3, wavelength=LAM, dx=dx, tilt_x=np.deg2rad(15))
    assert out.dtype == np.dtype(dtype), (
        f"tilted ASM returned {out.dtype} for {np.dtype(dtype)} input")


def test_tilted_asm_c64_matches_c128_pipeline():
    """The complex64 path (mod-2*pi-folded float32 carrier) must agree with
    the complex128 pipeline to within the complex64 noise floor -- the
    carrier fold keeps the large carrier argument accurate."""
    N, dx = 512, 2e-6
    tilt = np.deg2rad(20)
    E128 = _tilted_gauss(N, dx, tilt, LAM, np.complex128)
    E64 = E128.astype(np.complex64)
    out128 = angular_spectrum_propagate_tilted(
        E128, z=1e-3, wavelength=LAM, dx=dx, tilt_x=tilt)
    out64 = angular_spectrum_propagate_tilted(
        E64, z=1e-3, wavelength=LAM, dx=dx, tilt_x=tilt)
    scale = np.abs(out128).max()
    assert np.abs(out64 - out128.astype(np.complex64)).max() / scale < 5e-5


def test_tilted_asm_c64_energy_conserved():
    N, dx = 512, 2e-6
    tilt = np.deg2rad(20)
    E = _tilted_gauss(N, dx, tilt, LAM, np.complex64)
    out = angular_spectrum_propagate_tilted(
        E, z=1e-3, wavelength=LAM, dx=dx, tilt_x=tilt)
    P_in = float(np.sum(np.abs(E) ** 2))
    P_out = float(np.sum(np.abs(out) ** 2))
    assert abs(P_out / P_in - 1.0) < 5e-3
