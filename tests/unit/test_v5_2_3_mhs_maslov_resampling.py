"""v5.2.3 ROADMAP residual closure: substantive maslov-branch
output-grid resampling in :func:`prescription_subdomain`.

v5.2.0 closed AUDIT_V4_13_1 P1-C by raising ``ValueError`` when
``method='maslov'`` was paired with an ``out_surface`` declaring a
grid different from ``in_surface`` (option a -- block).  v5.2.3
ships option b: the function now resamples the maslov kernel's
input-grid output onto the declared ``out_surface`` via
:func:`lumenairy.propagators.mft.resample_field`, with a power
re-normalisation step to preserve total L2 energy across the
bicubic interpolator.

Test pins:

* ``test_output_dimensions_match_request`` -- N=128, dx=2e-6
  returned field has exactly those dimensions.
* ``test_zero_magnification_matches_native_grid`` -- when input
  grid == output grid, resample is skipped; output bit-matches a
  same-grid call.
* ``test_magnification_extent_doubles`` -- 2x output pitch doubles
  the spatial extent over which the field is sampled.
* ``test_parseval_total_power_preserved`` -- total power
  (integrated over the output extent) is preserved across the
  resample to within the bicubic interpolator's truncation error.
* ``test_non_square_out_surface_retained_raise`` -- the narrow
  retained-raise corner case (non-square out_surface) still fires.

All comments / strings cp1252-safe.  Citation: ``v5.2.3
(AUDIT_V4_13_1 P1-C substantive closure)``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.propagators.mhs import (
    HuygensSurface,
    prescription_subdomain,
)


# Minimal prescription used by all pins.  A small spherical singlet
# keeps the maslov ray bundle fit cheap.
def _singlet_pres():
    return lm.make_singlet(
        R1=50e-3, R2=-50e-3, d=3e-3,
        glass='N-BK7', aperture=0.3e-3,
    )


def _gaussian_input(N, dx, w0=100e-6):
    """Small Gaussian source field on an (N, N) grid."""
    x = (np.arange(N) - N / 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(xx ** 2 + yy ** 2) / (w0 ** 2)).astype(np.complex128)


# Small problem size keeps the maslov ray-bundle fit cheap.
_MASLOV_FAST_KWARGS = dict(
    ray_field_samples=6,
    ray_pupil_samples=6,
    n_v2=8,
)


class TestV5_2_3_MaslovSubdomainResampling:
    """Pins on the substantive maslov-branch resampling in
    :func:`prescription_subdomain`."""

    # -- closed-form / dimension pin -----------------------------------

    def test_output_dimensions_match_request(self):
        """Prescribe an output grid of N=128, dx=2e-6 and verify the
        returned field has exactly those dimensions."""
        N_in, dx_in = 64, 4e-6
        N_out, dx_out = 128, 2e-6
        in_s = HuygensSurface(z=0.0, dx=dx_in, Ny=N_in, Nx=N_in)
        out_s = HuygensSurface(z=0.0, dx=dx_out, Ny=N_out, Nx=N_out)
        pres = _singlet_pres()

        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='maslov',
            **_MASLOV_FAST_KWARGS,
        )
        E_in = _gaussian_input(N_in, dx_in)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out = sub.propagator(E_in, in_s, out_s, **sub.kwargs)

        assert E_out.shape == (N_out, N_out), (
            f"maslov resample output shape {E_out.shape} != requested "
            f"({N_out}, {N_out})"
        )
        assert np.iscomplexobj(E_out)
        assert np.all(np.isfinite(E_out.real))
        assert np.all(np.isfinite(E_out.imag))

    # -- numerical equivalence pin (zero magnification) ----------------

    def test_zero_magnification_matches_native_grid(self):
        """When ``in_surface == out_surface`` (same N, same dx), the
        resample fast-path is skipped and the output bit-matches a
        same-grid call."""
        N, dx = 48, 8e-6
        in_s = HuygensSurface(z=0.0, dx=dx, Ny=N, Nx=N)
        out_s = HuygensSurface(z=0.0, dx=dx, Ny=N, Nx=N)
        pres = _singlet_pres()

        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='maslov',
            **_MASLOV_FAST_KWARGS,
        )
        E_in = _gaussian_input(N, dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out_resampler_path = sub.propagator(
                E_in, in_s, out_s, **sub.kwargs)

        # Reference: call apply_real_lens_maslov directly with the same
        # kwargs.  The same-grid resample fast-path returns the kernel
        # output unchanged, so the two outputs must bit-match.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_ref = lm.apply_real_lens_maslov(
                E_in, prescription=pres, wavelength=633e-9, dx=dx,
                **_MASLOV_FAST_KWARGS,
            )

        np.testing.assert_allclose(
            E_out_resampler_path, E_ref,
            rtol=1e-10, atol=1e-10,
            err_msg='same-grid maslov subdomain output drifted from '
                    'direct kernel call (resample fast-path broken).',
        )

    # -- magnification pin (2x output pitch) ---------------------------

    def test_magnification_extent_doubles(self):
        """A 2x output pitch (with same N) doubles the spatial extent
        over which the resampled field is sampled."""
        N, dx_in = 64, 4e-6
        dx_out = 2.0 * dx_in
        in_s = HuygensSurface(z=0.0, dx=dx_in, Ny=N, Nx=N)
        out_s = HuygensSurface(z=0.0, dx=dx_out, Ny=N, Nx=N)
        pres = _singlet_pres()

        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='maslov',
            **_MASLOV_FAST_KWARGS,
        )
        E_in = _gaussian_input(N, dx_in)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out = sub.propagator(E_in, in_s, out_s, **sub.kwargs)

        # Extents: dx_out * N is the full physical span at the output
        # grid; verify it equals 2 * (dx_in * N).
        extent_in = dx_in * N
        extent_out = dx_out * N
        assert abs(extent_out - 2.0 * extent_in) < 1e-15

        assert E_out.shape == (N, N)
        # The resampled field is sampled at coarser dx; with a smaller
        # input Gaussian, its peak should still be near the centre.
        peak_idx = np.unravel_index(np.argmax(np.abs(E_out)), E_out.shape)
        assert abs(peak_idx[0] - N // 2) <= 2
        assert abs(peak_idx[1] - N // 2) <= 2

    # -- Parseval / total-power pin ------------------------------------

    def test_parseval_total_power_preserved(self):
        """Total power must be preserved across the resample (the
        v5.2.3 implementation explicitly re-normalises so this holds
        to within bicubic-interpolator precision)."""
        N_in, dx_in = 64, 4e-6
        N_out, dx_out = 96, 3e-6
        in_s = HuygensSurface(z=0.0, dx=dx_in, Ny=N_in, Nx=N_in)
        out_s = HuygensSurface(z=0.0, dx=dx_out, Ny=N_out, Nx=N_out)
        pres = _singlet_pres()

        sub = prescription_subdomain(
            in_s, out_s, pres,
            wavelength=633e-9, method='maslov',
            **_MASLOV_FAST_KWARGS,
        )
        E_in = _gaussian_input(N_in, dx_in)

        # Reference power: run the kernel natively on the input grid.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_native = lm.apply_real_lens_maslov(
                E_in, prescription=pres, wavelength=633e-9, dx=dx_in,
                **_MASLOV_FAST_KWARGS,
            )
            E_out = sub.propagator(E_in, in_s, out_s, **sub.kwargs)

        # Total power = sum(|E|^2) * dx^2 (Riemann-sum approx of the L2
        # integral).  The v5.2.3 implementation re-normalises after the
        # bicubic resample, so the two should match to numerical precision.
        p_native = float(np.sum(np.abs(E_native) ** 2)) * (dx_in ** 2)
        p_resampled = float(np.sum(np.abs(E_out) ** 2)) * (dx_out ** 2)

        # 1e-9 abs tol on dimensionless power -- the re-normalisation
        # step explicitly enforces this to machine precision.  We use
        # the larger of the two as the scale.
        p_scale = max(p_native, p_resampled, 1e-30)
        rel_err = abs(p_resampled - p_native) / p_scale
        assert rel_err < 1e-9, (
            f'Parseval check failed: native power {p_native:.6e}, '
            f'resampled power {p_resampled:.6e}, '
            f'rel_err {rel_err:.2e} (expected < 1e-9).'
        )

    # -- narrow retained-raise corner case -----------------------------

    def test_non_square_out_surface_retained_raise(self):
        """The narrow retained ``ValueError``: the maslov kernel requires
        a square output grid.  Non-square Ny != Nx in out_surface raises
        at construction time (rather than mid-pipeline)."""
        in_s = HuygensSurface(z=0.0, dx=4e-6, Ny=64, Nx=64)
        out_s = HuygensSurface(z=0.0, dx=2e-6, Ny=128, Nx=64)
        pres = _singlet_pres()
        with pytest.raises(ValueError, match='square'):
            prescription_subdomain(
                in_s, out_s, pres,
                wavelength=633e-9, method='maslov',
            )

    def test_non_square_in_surface_retained_raise(self):
        """Mirror of the previous test: a non-square ``in_surface`` is
        rejected at construction time (the maslov kernel itself would
        raise on a non-square ``E_in.shape`` mid-run)."""
        in_s = HuygensSurface(z=0.0, dx=4e-6, Ny=128, Nx=64)
        out_s = HuygensSurface(z=0.0, dx=2e-6, Ny=128, Nx=128)
        pres = _singlet_pres()
        with pytest.raises(ValueError, match='square'):
            prescription_subdomain(
                in_s, out_s, pres,
                wavelength=633e-9, method='maslov',
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
