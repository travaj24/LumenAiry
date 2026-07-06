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
