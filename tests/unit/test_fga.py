"""FGA (Frozen Gaussian Approximation) caustic-accurate lens propagator.

Validates :func:`lumenairy.propagators.apply_real_lens_fga`:
  * through a real plano-convex singlet it matches BOTH the GBD propagator and
    the angular-spectrum oracle to high fidelity (correct through-surface
    transport, incl. the manual image-side leg + monodromy);
  * at a spherical-aberration CAUSTIC it BEATS GBD on the caustic peak-intensity
    error (the whole point of FGA);
  * energy is controllable via the frozen width and closes with ``normalize``.

Requires numba (skipped otherwise).
"""
import numpy as np
import pytest

numba = pytest.importorskip("numba")           # noqa: F841

from lumenairy.propagators import apply_real_lens_fga  # noqa: E402
from lumenairy.propagators.asm import angular_spectrum_propagate  # noqa: E402

# The FGA phase-space swarm sum + the GBD/ASM oracles are eig-free but heavy
# (numba JIT + 256^2 grids); run in the slow-tests CI job, not the fast gate.
pytestmark = pytest.mark.slow

_WL = 0.633e-6


def _fid(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)


def _singlet():
    """Plano-convex N-BK7 singlet (curved side first, flat exit -> no powered-
    exit-surface vertex-plane ambiguity).  f ~ 38.8 mm."""
    return {'name': 'pcx', 'aperture_diameter': 2.8e-3,
            'surfaces': [
                {'radius': 20e-3, 'conic': 0.0, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'semi_diameter': 1.4e-3},
                {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'semi_diameter': 1.4e-3}],
            'thicknesses': [2.5e-3]}


def _collimated_gaussian(N=256, dx=10e-6, w=0.9e-3):
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    return np.exp(-(Xg ** 2 + Yg ** 2) / w ** 2).astype(np.complex128), dx


def test_fga_through_singlet_matches_gbd_and_asm():
    """Through a real singlet, FGA matches GBD AND the ASM oracle in the smooth
    converging region -- the through-surface transport (trace + monodromy +
    image-leg) is correct."""
    from lumenairy.elements import apply_real_lens, apply_real_lens_gbd
    presc = _singlet()
    u0, dx = _collimated_gaussian()
    for zi in (25e-3,):
        fga = apply_real_lens_fga(u0, prescription=presc, wavelength=_WL, dx=dx,
                                  output_plane_distance=zi)
        gbd = apply_real_lens_gbd(u0, prescription=presc, wavelength=_WL, dx=dx,
                                  output_plane_distance=zi,
                                  beamlets_per_aperture=40)
        asm = angular_spectrum_propagate(
            apply_real_lens(u0, prescription=presc, wavelength=_WL, dx=dx),
            zi, _WL, dx)
        assert _fid(gbd, asm) > 0.99         # sanity: the oracle pair agrees
        assert _fid(fga, asm) > 0.99         # FGA matches the ASM oracle
        assert _fid(fga, gbd) > 0.99         # ... and GBD


def test_fga_beats_gbd_at_spherical_aberration_caustic():
    """At a spherical-aberration caustic (aberrated field, free-space leg), FGA
    renders the caustic peak far better than GBD.  Modeled as a lens+SA phase on
    the input propagated by a null (flat-window) prescription's image leg is
    awkward; instead compare against the exact ASM field for the aberrated input
    propagated in free space -- the canonical caustic test."""
    from lumenairy.propagators.gbd import propagate_gbd_freespace
    N, dx = 256, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    R2 = Xg ** 2 + Yg ** 2
    k0 = 2 * np.pi / _WL
    F, SA, W0 = 250e-6, 8.0, 22e-6
    Rn = np.sqrt(R2) / W0
    uc = (np.exp(-R2 / W0 ** 2) * np.exp(-1j * k0 * R2 / (2.0 * F))
          * np.exp(1j * 2.0 * np.pi * SA * Rn ** 4)).astype(np.complex128)
    # a null (flat) prescription so apply_real_lens_fga just free-space-props
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    for zf in (0.9,):
        z = zf * F
        asm = angular_spectrum_propagate(uc, z, _WL, dx)
        fga = apply_real_lens_fga(uc, prescription=flat, wavelength=_WL, dx=dx,
                                  output_plane_distance=z, w0_factor=4.0,
                                  p_max=0.14, n_p=25)
        gbd = propagate_gbd_freespace(uc, dx, z=z, wavelength=_WL,
                                      sample_step=2, waist_factor=2.0,
                                      direction_sampling=True)
        pk = np.abs(asm).max() ** 2
        sf = np.vdot(asm, fga) / np.vdot(fga, fga)
        sg = np.vdot(asm, gbd) / np.vdot(gbd, gbd)
        ef = abs((np.abs(sf * fga).max() ** 2 - pk) / pk)
        eg = abs((np.abs(sg * gbd).max() ** 2 - pk) / pk)
        assert ef < eg                       # FGA renders the caustic peak better
        assert ef < 0.15                     # and to the documented ~few-% floor


def test_fga_power_normalization_and_guards():
    presc = _singlet()
    u0, dx = _collimated_gaussian(N=128)
    out = apply_real_lens_fga(u0, prescription=presc, wavelength=_WL, dx=dx,
                              output_plane_distance=20e-3,
                              normalize_output="power")
    assert abs(np.sum(np.abs(out) ** 2) / np.sum(np.abs(u0) ** 2) - 1.0) < 1e-6
    with pytest.raises(ValueError, match="square"):
        apply_real_lens_fga(u0[:, :64], prescription=presc, wavelength=_WL,
                            dx=dx)
    with pytest.raises(ValueError, match="normalize_output"):
        apply_real_lens_fga(u0, prescription=presc, wavelength=_WL, dx=dx,
                            normalize_output="bogus")
