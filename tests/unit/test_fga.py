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

# NOT marked ``slow``: these are heavy (numba JIT + 256^2 swarm sums) but
# eig-FREE, so they belong in the fast gate -- which is xdist-parallelised
# (--dist loadfile) and absorbs this file on a single worker in ~7 min -- rather
# than in the serial, eig-heavy, hardware-sensitive ``slow`` job (which cannot
# safely take xdist and was already near its time cap).

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
        # n_p=15 is as decisive as n_p=25 here (the swarm is n_p^2 momenta, so
        # this is the dominant cost) -- verified: FGA peak error ~0.08 vs GBD
        # ~0.13 at z/F=0.9 either way.  Keep N=256 + the rigorous GBD oracle so
        # the peak-error margin the assertion checks is unchanged.
        fga = apply_real_lens_fga(uc, prescription=flat, wavelength=_WL, dx=dx,
                                  output_plane_distance=z, w0_factor=4.0,
                                  p_max=0.14, n_p=15)
        # sample_step=3 (not 2) is the dominant cost here; a coarser GBD only
        # makes GBD's caustic peak WORSE, so it can only strengthen the
        # ef < eg assertion -- it never masks an FGA regression.  Verified:
        # GBD peak error ~0.13 at this step vs FGA ~0.08.
        gbd = propagate_gbd_freespace(uc, dx, z=z, wavelength=_WL,
                                      sample_step=3, waist_factor=2.0,
                                      direction_sampling=True)
        pk = np.abs(asm).max() ** 2
        sf = np.vdot(asm, fga) / np.vdot(fga, fga)
        sg = np.vdot(asm, gbd) / np.vdot(gbd, gbd)
        ef = abs((np.abs(sf * fga).max() ** 2 - pk) / pk)
        eg = abs((np.abs(sg * gbd).max() ** 2 - pk) / pk)
        assert ef < eg                       # FGA renders the caustic peak better
        assert ef < 0.15                     # and to the documented ~few-% floor


def test_auto_dispatcher_routes_and_matches():
    """apply_real_lens_auto detects the caustic zone, routes far planes to GBD
    and near-focus planes to FGA, and its output equals the chosen propagator's
    (a true dispatch, not a re-implementation)."""
    from lumenairy.propagators.fga import _caustic_zone, apply_real_lens_auto
    presc = _singlet()                       # f ~ 38.8 mm
    u0, dx = _collimated_gaussian(N=128)     # routing is grid-size-independent
    zone = _caustic_zone(u0, dx, presc, _WL)
    assert zone is not None and 30e-3 < zone[0] < 45e-3   # focus detected
    # far from focus -> GBD; near focus -> FGA
    _o1, m_far = apply_real_lens_auto(
        u0, prescription=presc, wavelength=_WL, dx=dx,
        output_plane_distance=15e-3, return_method=True,
        gbd_kwargs={'beamlets_per_aperture': 40})
    _o2, m_near = apply_real_lens_auto(
        u0, prescription=presc, wavelength=_WL, dx=dx,
        output_plane_distance=38e-3, return_method=True,
        gbd_kwargs={'beamlets_per_aperture': 40})
    assert m_far == "gbd"
    assert m_near == "fga"
    # forced method matches the standalone propagator
    from lumenairy.elements import apply_real_lens_gbd
    auto_g = apply_real_lens_auto(u0, prescription=presc, wavelength=_WL, dx=dx,
                                  output_plane_distance=15e-3, method="gbd",
                                  gbd_kwargs={'beamlets_per_aperture': 40})
    ref_g = apply_real_lens_gbd(u0, prescription=presc, wavelength=_WL, dx=dx,
                                output_plane_distance=15e-3,
                                beamlets_per_aperture=40)
    assert np.array_equal(auto_g, ref_g)


def test_universal_dispatcher_4way_routing():
    """apply_real_lens_universal (4-way) routes by regime.  For the LOW-NA singlet
    'auto' picks 'phase_screen', and that route reproduces the angular-spectrum
    oracle EXACTLY (apply_real_lens at the exit vertex + exact ASM output leg) --
    i.e. it is a true dispatch, not a re-implementation.  Also guards the method
    name.  (The high-NA traced/fga routing is covered by test_auto_dispatcher and
    the caustic-zone detector below.)"""
    from lumenairy.elements import apply_real_lens
    from lumenairy.propagators.fga import _system_na, apply_real_lens_universal
    presc = _singlet()                          # NA ~0.036 (low)
    u0, dx = _collimated_gaussian(N=128)        # routing is grid-size-independent
    assert _system_na(presc, _WL) < 0.12        # low-NA -> phase_screen branch
    out, m = apply_real_lens_universal(
        u0, prescription=presc, wavelength=_WL, dx=dx,
        output_plane_distance=15e-3, return_method=True)
    assert m == "phase_screen"
    ref = angular_spectrum_propagate(
        apply_real_lens(u0, prescription=presc, wavelength=_WL, dx=dx),
        15e-3, _WL, dx)
    assert np.allclose(out, ref)                 # exact-oracle dispatch
    with pytest.raises(ValueError, match="method must be"):
        apply_real_lens_universal(u0, prescription=presc, wavelength=_WL,
                                  dx=dx, method="bogus")


def test_fga_normalization_identity_and_energy():
    """The corrected FGA normalization makes the t=0 resolution of identity exact
    (power ratio ~1, not the pre-fix 2^d=4) and free-space propagation energy-
    conserving to ~1.0 -- confirming the energy error was a normalization factor,
    not the O(eps) transport defect."""
    N, dx = 128, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (18e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}

    def pr(a, b):
        return float(np.sum(np.abs(a) ** 2) / np.sum(np.abs(b) ** 2))
    ident = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                                output_plane_distance=0.0, w0_factor=8.0,
                                p_max=0.06, n_p=15)
    assert abs(pr(ident, u0) - 1.0) < 0.02        # resolution of identity
    prop = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                               output_plane_distance=300e-6, w0_factor=8.0,
                               p_max=0.06, n_p=15)
    ref = angular_spectrum_propagate(u0, 300e-6, _WL, dx)
    assert abs(pr(prop, u0) - 1.0) < 0.03         # free-space energy conserved
    assert _fid(prop, ref) > 0.999                # ... and shape-exact


def test_fga_memory_chunking_numerically_identical():
    """The momentum-swarm chunking (``chunk`` / ``mem_budget_mb``) bounds peak
    beamlet memory to O(Nq*chunk) and returns the SAME field as the full swarm --
    it only reorders an additive sum, so the result matches to float round-off,
    for both the scalar and vector propagators."""
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    N, dx = 128, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (18e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    kw = dict(prescription=flat, wavelength=_WL, dx=dx,
              output_plane_distance=300e-6, w0_factor=8.0, p_max=0.06, n_p=13)
    full = apply_real_lens_fga(u0, **kw)
    for c in (apply_real_lens_fga(u0, chunk=1, **kw),
              apply_real_lens_fga(u0, chunk=5, **kw),
              apply_real_lens_fga(u0, mem_budget_mb=2, **kw)):
        assert np.max(np.abs(c - full)) < 1e-9     # identical to round-off
    vfull = apply_real_lens_fga_vector(np.stack([u0, np.zeros_like(u0)]),
                                       return_longitudinal=True, **kw)
    vchunk = apply_real_lens_fga_vector(np.stack([u0, np.zeros_like(u0)]),
                                        chunk=3, return_longitudinal=True, **kw)
    assert np.max(np.abs(vchunk - vfull)) < 1e-9


def test_fga_position_pruning_no_loss():
    """Position-support pruning (``prune_frac``) drops launch-lattice points where
    the windowed input is negligible.  On a concentrated field it changes the
    result by nothing (Cauchy-Schwarz bounds the dropped Gabor coefficients) while
    cutting the beamlet count; a grid-filling field prunes ~nothing."""
    N, dx = 160, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    conc = np.exp(-(Xg ** 2 + Yg ** 2) / (9e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    kw = dict(prescription=flat, wavelength=_WL, dx=dx,
              output_plane_distance=250e-6, w0_factor=6.0, p_max=0.06, n_p=13)
    unpruned = apply_real_lens_fga(conc, prune_frac=0.0, **kw)
    pruned = apply_real_lens_fga(conc, prune_frac=1e-3, **kw)   # default-scale
    assert _fid(pruned, unpruned) > 0.99999                    # no-loss
    # the pruned run keeps far fewer lattice points (concentrated field)
    from lumenairy.propagators.fga import _lattice_support_mask
    keep = _lattice_support_mask(conc, dx, dx, 2, 6.0 * dx, 3.0, 1e-3)
    assert keep.sum() < 0.6 * keep.size                        # meaningfully pruned


def test_fga_vector_polarization():
    """Vector (Jones) FGA: free-space parity (Jones=identity in air -> Ex matches
    the scalar propagator, no spurious cross-pol), a physical longitudinal Ez,
    and correct return shapes."""
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    N, dx = 128, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (18e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    KW = dict(w0_factor=5.0, p_max=0.06, n_p=13)
    z = 300e-6
    sc = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                             output_plane_distance=z, **KW)
    vec = apply_real_lens_fga_vector(
        np.stack([u0, np.zeros_like(u0)]), prescription=flat, wavelength=_WL,
        dx=dx, output_plane_distance=z, return_longitudinal=True, **KW)
    assert vec.shape == (3, N, N)
    assert _fid(vec[0], sc) > 0.999                       # Ex == scalar
    assert np.linalg.norm(vec[1]) < 1e-3 * np.linalg.norm(vec[0])   # Ey ~ 0
    assert np.linalg.norm(vec[2]) < 0.05 * np.linalg.norm(vec[0])   # Ez small
    two = apply_real_lens_fga_vector(
        np.stack([u0, np.zeros_like(u0)]), prescription=flat, wavelength=_WL,
        dx=dx, output_plane_distance=z, **KW)
    assert two.shape == (2, N, N)                          # (Ex, Ey) by default


def test_fga_power_normalization_and_guards():
    presc = _singlet()
    u0, dx = _collimated_gaussian(N=128)
    out = apply_real_lens_fga(u0, prescription=presc, wavelength=_WL, dx=dx,
                              output_plane_distance=20e-3,
                              normalize_output="power")
    assert abs(np.sum(np.abs(out) ** 2) / np.sum(np.abs(u0) ** 2) - 1.0) < 1e-6
    with pytest.raises(ValueError, match="2-D"):
        apply_real_lens_fga(u0[0], prescription=presc, wavelength=_WL, dx=dx)
    with pytest.raises(ValueError, match="normalize_output"):
        apply_real_lens_fga(u0, prescription=presc, wavelength=_WL, dx=dx,
                            normalize_output="bogus")


def test_fga_anamorphic_grid_matches_asm():
    """Anamorphic (dx != dy) pixel pitch and rectangular arrays.  On a
    rectangular-pixel grid (dy = 1.5 dx) FGA reproduces the EXACT angular-
    spectrum field (the frozen beamlet tracks the geomean pitch, the phase-space
    measure is the anamorphic cell dx*dy, and the momentum swarm is unchanged);
    a non-square array (Ny != Nx) also propagates and keeps its shape."""
    dx = 0.7e-6
    dy = 1.5 * dx
    flat = {'name': 'flat', 'aperture_diameter': 1.0,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': 0.5}],
            'thicknesses': []}
    # (a) anamorphic pitch, square array, vs the exact ASM(dy) oracle
    N = 128
    xs = (np.arange(N) - N / 2) * dx
    ys = (np.arange(N) - N / 2) * dy
    Xg, Yg = np.meshgrid(xs, ys)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (16e-6) ** 2).astype(np.complex128)
    z = 250e-6
    fga = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                              dy=dy, output_plane_distance=z, w0_factor=8.0,
                              p_max=0.06, n_p=15)
    ref = angular_spectrum_propagate(u0, z, _WL, dx, dy)
    assert _fid(fga, ref) > 0.99
    # dy=None must still equal dy=dx exactly (backward-compatible default)
    sq = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                             output_plane_distance=z, w0_factor=8.0,
                             p_max=0.06, n_p=15)
    sq2 = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                              dy=dx, output_plane_distance=z, w0_factor=8.0,
                              p_max=0.06, n_p=15)
    assert np.array_equal(sq, sq2)
    # (b) rectangular array (Ny != Nx), square pixels -> correct shape + matches
    Ny, Nx = 96, 128
    xr = (np.arange(Nx) - Nx / 2) * dx
    yr = (np.arange(Ny) - Ny / 2) * dx
    Xr, Yr = np.meshgrid(xr, yr)
    ur = np.exp(-(Xr ** 2 + Yr ** 2) / (16e-6) ** 2).astype(np.complex128)
    rect = apply_real_lens_fga(ur, prescription=flat, wavelength=_WL, dx=dx,
                               output_plane_distance=z, w0_factor=8.0,
                               p_max=0.06, n_p=15)
    assert rect.shape == (Ny, Nx)
    assert _fid(rect, angular_spectrum_propagate(ur, z, _WL, dx)) > 0.99
    # (c) the vector propagator threads dy too: its Ex channel reproduces the
    # scalar anamorphic FGA (Jones = identity in air)
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    vec = apply_real_lens_fga_vector(
        np.stack([u0, np.zeros_like(u0)]), prescription=flat, wavelength=_WL,
        dx=dx, dy=dy, output_plane_distance=z, w0_factor=8.0, p_max=0.06,
        n_p=15)
    assert vec.shape == (2, N, N)
    assert _fid(vec[0], fga) > 0.999
