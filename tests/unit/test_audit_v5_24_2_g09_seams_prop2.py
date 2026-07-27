"""Regression pins for the v5.24.2 exhaustive-audit G09 seam group
(propagator seams, batch 2).

Each test targets one finding and is written to FAIL on the pre-fix code
and PASS after.  The oracle is INDEPENDENT of the code under test: at
z == 0 the field propagates zero distance, so the output must equal the
INPUT to FFT round-trip precision for ANY grid -- this is a physical
fact, not a restatement of the transfer-function formula.

Findings covered:
  * S2-11  z=0 ASM is the exact identity (evanescent bins NOT zeroed at
           z=0) for sub-wavelength grids, and the z=0 propagator path
           agrees with the dispatcher's ``propagate(z=None)`` copy.
"""
from __future__ import annotations

import numpy as np
import pytest

# ============================================================================
# S2-11 -- z=0 ASM is the exact identity, including sub-wavelength grids
# ============================================================================

# A deliberately SUB-wavelength grid: dx < lambda/2 so the centred spectrum
# contains evanescent bins (|f| up to 1/(2*dx) > 1/lambda).  On the pre-fix
# code these bins were zeroed even at z=0, so ASM(z=0) dropped ~70% of the
# field energy (probe rel err ~0.84).
_WAVELENGTH = 1.31e-6
_DX_SUB = 0.4e-6          # < lambda/2 = 0.655 um -> evanescent bins present
_DX_COARSE = 1.0e-6       # > lambda/2 -> no evanescent bins (control)
_N = 64


def _random_field(n, seed, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n))
            + 1j * rng.standard_normal((n, n))).astype(dtype)


class TestS2_11Z0Identity:
    def test_z0_identity_subwavelength(self):
        """z=0 must reproduce the input on a sub-wavelength grid.

        Oracle: the input field itself (zero-distance propagation is a
        no-op).  Pre-fix this failed with rel err ~0.84 because the
        evanescent bins were zeroed by the ``kz_sq > 0`` mask at z=0.
        """
        from lumenairy.propagators.asm import angular_spectrum_propagate

        E = _random_field(_N, 0)
        E_out = angular_spectrum_propagate(
            E, z=0.0, wavelength=_WAVELENGTH, dx=_DX_SUB)
        rel = np.linalg.norm(E_out - E) / np.linalg.norm(E)
        assert rel < 1e-13, f"z=0 not identity on sub-wavelength grid: {rel}"

    def test_z0_identity_coarse_control(self):
        """z=0 identity also holds on a coarse grid (already true pre-fix;
        kept as a control that the short-circuit did not regress the
        propagating-only case)."""
        from lumenairy.propagators.asm import angular_spectrum_propagate

        E = _random_field(_N, 1)
        E_out = angular_spectrum_propagate(
            E, z=0.0, wavelength=_WAVELENGTH, dx=_DX_COARSE)
        rel = np.linalg.norm(E_out - E) / np.linalg.norm(E)
        assert rel < 1e-13, f"z=0 not identity on coarse grid: {rel}"

    def test_build_asm_h_square_z0_is_all_ones(self):
        """The centered square builder returns H == 1 for EVERY bin at
        z=0, evanescent bins included.  Pre-fix, the sub-wavelength grid
        had its evanescent bins zeroed, so H was not all-ones."""
        from lumenairy.propagators.asm import _build_asm_H_square

        H = _build_asm_H_square(_N, _DX_SUB, 0.0, _WAVELENGTH)
        assert np.array_equal(H, np.ones((_N, _N), dtype=np.complex128)), (
            "z=0 _build_asm_H_square must be the exact all-ones identity")

    def test_z0_matches_dispatch_znone_copy(self):
        """The two no-propagation spellings must agree: the dispatcher's
        ``propagate(method='asm', z=None)`` (a pure copy) and the z=0
        propagator path.  Pre-fix they diverged on sub-wavelength grids
        (copy = identity vs z=0 = evanescent-filtered)."""
        from lumenairy.propagators.dispatch import propagate

        E = _random_field(_N, 2)
        # v5.30 (audit P5 / roadmap F1 flip): ``return_result=False`` names the
        # native-ndarray contract these two spellings are compared in; the
        # dispatcher's default is now a PropagationResult.
        E_none = propagate(E, z=None, wavelength=_WAVELENGTH,
                           dx=_DX_SUB, method='asm', return_result=False)
        E_z0 = propagate(E, z=0.0, wavelength=_WAVELENGTH,
                         dx=_DX_SUB, method='asm', return_result=False)
        max_diff = float(np.max(np.abs(E_none - E_z0)))
        assert max_diff < 1e-12, (
            f"z=None copy and z=0 propagate disagree: {max_diff}")

    def test_z0_return_transfer_function_is_all_ones(self):
        """With return_transfer_function=True the z=0 transfer function
        returned to the caller is the exact all-ones identity."""
        from lumenairy.propagators.asm import angular_spectrum_propagate

        E = _random_field(_N, 3)
        E_out, H = angular_spectrum_propagate(
            E, z=0.0, wavelength=_WAVELENGTH, dx=_DX_SUB,
            return_transfer_function=True)
        assert np.array_equal(H, np.ones((_N, _N), dtype=np.complex128))
        rel = np.linalg.norm(E_out - E) / np.linalg.norm(E)
        assert rel < 1e-13

    def test_z0_identity_complex64(self):
        """The complex64 path is also the identity at z=0 (to the float32
        round-trip floor) and preserves dtype."""
        from lumenairy.propagators.asm import angular_spectrum_propagate

        E = _random_field(_N, 4, dtype=np.complex64)
        E_out = angular_spectrum_propagate(
            E, z=0.0, wavelength=_WAVELENGTH, dx=_DX_SUB)
        assert E_out.dtype == np.complex64
        rel = np.linalg.norm(E_out - E) / np.linalg.norm(E)
        assert rel < 1e-5, f"complex64 z=0 not identity: {rel}"

    def test_z0_identity_batch_subwavelength(self):
        """The 3-D batch variant is also the identity at z=0 on a
        sub-wavelength grid (it shares _get_asm_H_natural)."""
        propagation = pytest.importorskip("lumenairy.propagators.propagation")
        batch = getattr(
            propagation, "angular_spectrum_propagate_batch", None)
        if batch is None:
            pytest.skip("angular_spectrum_propagate_batch not available")
        rng = np.random.default_rng(5)
        E = (rng.standard_normal((3, _N, _N))
             + 1j * rng.standard_normal((3, _N, _N))).astype(np.complex128)
        E_out = batch(E, 0.0, _WAVELENGTH, _DX_SUB)
        rel = np.linalg.norm(E_out - E) / np.linalg.norm(E)
        assert rel < 1e-13, f"batch z=0 not identity: {rel}"

    def test_z0_identity_jax_subwavelength(self):
        """The JAX path is also the identity at z=0 on a sub-wavelength
        grid (shares the host-built _get_asm_H_natural short-circuit)."""
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        jax.config.update("jax_enable_x64", True)
        from lumenairy.propagators.asm import angular_spectrum_propagate

        E_np = _random_field(_N, 6)
        E = jnp.asarray(E_np)
        E_out = angular_spectrum_propagate(
            E, z=0.0, wavelength=_WAVELENGTH, dx=_DX_SUB)
        rel = float(jnp.linalg.norm(E_out - E) / jnp.linalg.norm(E))
        assert rel < 1e-13, f"jax z=0 not identity: {rel}"
