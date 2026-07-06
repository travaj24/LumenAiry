"""v5.21 GBD/Maslov performance batch -- correctness gates.

Each item is a speedup that must not change the physics: validated against the
trusted reference (full grid / baked prescription / LAPACK / dense sum).
"""
import numpy as np
import pytest

LAM = 0.633e-6


def _singlet(last_gap):
    """Plano-convex N-BK7 singlet; ``last_gap`` = air gap after the lens."""
    return {'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, last_gap, 0.0]}


def _relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300))


def _gauss(N, dx, w0=1.8e-3):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


# --------------------------------------------------------------------------
# #2 Maslov focus-plane ROI (compose a free-space leg into the canonical map)
# --------------------------------------------------------------------------
def test_maslov_output_plane_distance_matches_baked_prescription():
    """Composing a free-space leg of distance ``d`` past the exit equals baking
    ``d`` into the prescription's last thickness (re-tracing) -- at a
    well-conditioned plane (away from the tight focus)."""
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    N, dx = 64, 5e-6
    E = _gauss(N, dx)
    base = 2e-3
    for d in (20e-3, 40e-3):
        F_compose = apply_real_lens_maslov(
            E, prescription=_singlet(base), wavelength=LAM, dx=dx,
            integration_method='quadrature', n_v2=48, output_plane_distance=d)
        F_baked = apply_real_lens_maslov(
            E, prescription=_singlet(base + d), wavelength=LAM, dx=dx,
            integration_method='quadrature', n_v2=48)
        assert _relerr(F_compose, F_baked) < 1e-7


def test_maslov_focus_roi_equals_full_grid_crop():
    """The ROI window on the composed focus plane is identical to the
    corresponding slice of the full-grid focus-plane field (both raw / no
    power-normalisation), at O(roi_n^2) instead of O(N^2) integrand evals."""
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    N, dx = 96, 5e-6
    E = _gauss(N, dx)
    base, d = 2e-3, 40e-3
    kw = dict(prescription=_singlet(base), wavelength=LAM, dx=dx,
              integration_method='quadrature', n_v2=48,
              output_plane_distance=d)
    F_full = apply_real_lens_maslov(E, normalize_output='none', **kw)
    # on-axis window
    F_roi = apply_real_lens_maslov(E, roi=(0.0, 0.0, 10 * dx), **kw)
    rn, c, h = F_roi.shape[0], N // 2, F_roi.shape[0] // 2
    assert _relerr(F_roi, F_full[c - h:c - h + rn, c - h:c - h + rn]) < 1e-12
    # off-axis window
    off = 30
    F_off = apply_real_lens_maslov(E, roi=(off * dx, 0.0, 6 * dx), **kw)
    rn2 = F_off.shape[0]
    crop = F_full[c - rn2 // 2:c - rn2 // 2 + rn2,
                  c + off - rn2 // 2:c + off - rn2 // 2 + rn2]
    assert _relerr(F_off, crop) < 1e-10


# --------------------------------------------------------------------------
# #9 GBD FFT-convolution reconstruction for uniform-Q bundles
# --------------------------------------------------------------------------
def test_fft_reconstruct_matches_dense_uniform_Q():
    """FFT-convolution reconstruction is machine-precision identical to the
    dense sum for a uniform-Q, uniform-direction, on-grid free-space bundle --
    and correctly falls back to the windowed sum for a per-beamlet-Q (after a
    lens) or skew bundle."""
    from lumenairy.propagators.gbd import (
        _fft_reconstruct_applicable,
        apply_thin_lens_to_beamlets,
        decompose_field_to_beamlets,
        propagate_beamlets_freespace,
        reconstruct_field_from_beamlets,
    )
    N, dx = 128, 5e-6
    E0 = _gauss(N, dx, w0=0.15e-3)

    # (a) free-space uniform-Q -> FFT applies, matches dense to ~1e-15
    b = decompose_field_to_beamlets(E0, dx, wavelength=LAM, sample_step=1,
                                    waist_factor=1.5)
    b = propagate_beamlets_freespace(b, 10e-3, LAM)
    assert _fft_reconstruct_applicable(b, N, N, dx, dx, (0.0, 0.0))
    dense = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx,
                                            wavelength=LAM, window=None)
    fft = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx,
                                          wavelength=LAM, window=5.0)
    assert _relerr(fft, dense) < 1e-9

    # (b) after a thin lens the per-beamlet direction kick breaks uniformity ->
    #     the FFT path must NOT engage (windowed still matches dense)
    b2 = decompose_field_to_beamlets(E0, dx, wavelength=LAM, sample_step=2,
                                     waist_factor=1.5)
    b2 = apply_thin_lens_to_beamlets(b2, 30e-3, LAM)
    b2 = propagate_beamlets_freespace(b2, 5e-3, LAM)
    assert not _fft_reconstruct_applicable(b2, N, N, dx, dx, (0.0, 0.0))
    d2 = reconstruct_field_from_beamlets(b2, Ny=N, Nx=N, dx=dx,
                                         wavelength=LAM, window=None)
    w2 = reconstruct_field_from_beamlets(b2, Ny=N, Nx=N, dx=dx,
                                         wavelength=LAM, window=5.0)
    assert _relerr(w2, d2) < 1e-9


def test_fft_reconstruct_anamorphic_diagonal_Q():
    """Uniform DIAGONAL tensor Q (anamorphic dy != dx) also takes the FFT path
    and matches the dense sum."""
    from lumenairy.propagators.gbd import (
        decompose_field_to_beamlets,
        propagate_beamlets_freespace,
        reconstruct_field_from_beamlets,
    )
    N, dx, dy = 128, 5e-6, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    ys = (np.arange(N) - N // 2) * dy
    Xa, Ya = np.meshgrid(xs, ys)
    E = np.exp(-(Xa ** 2 + Ya ** 2) / (0.15e-3) ** 2).astype(np.complex128)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, dy=dy, sample_step=1,
                                    waist_factor=1.5)
    b = propagate_beamlets_freespace(b, 8e-3, LAM)
    dense = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, dy=dy,
                                            wavelength=LAM, window=None)
    fft = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, dy=dy,
                                          wavelength=LAM, window=5.0)
    assert _relerr(fft, dense) < 1e-9


# --------------------------------------------------------------------------
# #5 backend-generic FFT-conv reconstruction (JAX / CuPy) + differentiable
# --------------------------------------------------------------------------
def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_fft_reconstruct_jax_matches_numpy_and_differentiable():
    """The FFT-conv reconstruction runs on the JAX backend (matches NumPy) and
    is jax.grad-differentiable through the whole reconstruct."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.propagators.gbd import (
        BeamletBundle,
        decompose_field_to_beamlets,
        propagate_beamlets_freespace,
        reconstruct_field_from_beamlets,
    )
    N, dx = 96, 5e-6
    E0 = _gauss(N, dx, w0=0.15e-3)
    b = decompose_field_to_beamlets(E0, dx, wavelength=LAM, sample_step=1,
                                    waist_factor=1.5)
    b = propagate_beamlets_freespace(b, 8e-3, LAM)
    fft_np = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx,
                                             wavelength=LAM, window=5.0)
    bj = BeamletBundle(
        positions=jnp.asarray(b.positions),
        directions=jnp.asarray(b.directions),
        Q=jnp.asarray(b.Q), amplitude=jnp.asarray(b.amplitude),
        waist0=jnp.asarray(b.waist0))
    fft_jax = reconstruct_field_from_beamlets(bj, Ny=N, Nx=N, dx=dx,
                                              wavelength=LAM, window=5.0)
    assert _relerr(np.asarray(fft_jax), fft_np) < 1e-9

    def loss(scale):
        bs = BeamletBundle(positions=bj.positions, directions=bj.directions,
                           Q=bj.Q, amplitude=bj.amplitude * scale,
                           waist0=bj.waist0)
        F = reconstruct_field_from_beamlets(bs, Ny=N, Nx=N, dx=dx,
                                            wavelength=LAM, window=5.0)
        return jnp.sum(jnp.abs(F) ** 2)
    g = float(jax.grad(loss)(1.0))
    fd = float((loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / 2e-5)
    assert abs(g - fd) / abs(fd) < 1e-6
