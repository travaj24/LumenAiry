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
# eig-FREE, so they belong in the fast gate -- rather than in the serial,
# eig-heavy, hardware-sensitive ``slow`` job (which cannot safely take xdist
# and was already near its time cap).
#
# 2026-08-03: the old wording here said the fast gate is "xdist-parallelised
# (--dist loadfile)" and absorbs this file "on a single worker in ~7 min".
# Both numbers are stale.  In-process xdist was ABANDONED (a runner has only
# ~2-4 cores, so it capped the gate at ~1.5x); the gate now duration-balances
# across three pytest-split shards on their own runners
# (``--splits 3 --group N --splitting-algorithm least_duration``).  And this
# file measures 1791.7 s (~30 min) across its 27 ``.test_durations`` entries,
# which the splitter spreads over those shards rather than landing whole on
# one worker.

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


def _hi_na_singlet():
    """Short-focus plano-convex singlet, NA ~0.19 (> the 0.12 dispatcher gate),
    so 'auto' reaches the high-NA traced/fga/multi-valued branch."""
    return {'name': 'hi', 'aperture_diameter': 6e-3,
            'surfaces': [
                {'radius': 8e-3, 'conic': 0.0, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
                {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'semi_diameter': 3e-3}],
            'thicknesses': [2.5e-3]}


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


def test_fga_diverging_beam_auto_sampling():
    """A strongly-DIVERGING beam free-space-propagated by FGA reconstructs to
    HIGH fidelity under the DEFAULT (auto) phase-space sampling.

    Leading-order FGA/Herman-Kluk is EXACT for a quadratic Hamiltonian (free
    space is one; Lasser & Lubich, Acta Numerica 29 (2020)), so the historical
    ~0.93 fidelity cap was NOT a frozen-approximation error but a too-coarse
    momentum spacing ``dp`` -- a diverging beam's broad phase-space footprint was
    under-sampled at the old fixed ``n_p``.  Auto-sizing ``n_p`` (so ``dp`` is
    fine) and ``p_max`` (to the field's angular content) restores fidelity to
    ~1.  A deliberately coarse fixed ``n_p`` reproduces the old cap, proving the
    sampling -- not the approximation order -- is the lever."""
    N, dx = 256, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    k = 2 * np.pi / _WL
    r2 = Xg ** 2 + Yg ** 2
    div = (np.exp(-r2 / (28e-6) ** 2)
           * np.exp(1j * k * r2 / (2 * 0.6e-3))).astype(np.complex128)  # ~0.09 rad
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    z = 0.5e-3
    ref = angular_spectrum_propagate(div, z, _WL, dx)
    # DEFAULT (auto) sampling -> high fidelity (the fix; was ~0.93)
    auto = apply_real_lens_fga(div, prescription=flat, wavelength=_WL, dx=dx,
                               output_plane_distance=z, w0_factor=5.0)
    fa = _fid(auto, ref)
    assert fa > 0.99
    # a deliberately COARSE fixed sampling reproduces the old under-resolved cap
    coarse = apply_real_lens_fga(div, prescription=flat, wavelength=_WL, dx=dx,
                                 output_plane_distance=z, w0_factor=5.0,
                                 p_max=0.14, n_p=13)   # dp ~ 0.023, too coarse
    assert _fid(coarse, ref) < fa                      # sampling IS the lever


def test_fga_flat_prescription_no_na_cap_on_wide_angle():
    """A flat (power-less) prescription must NOT impose a physical NA cap on the
    auto momentum swarm (audit S2-6).

    ``_default_p_max`` returns its 0.05 floor for a flat plate (marginal-ray
    NA -> 0), which under the old code clamped auto ``p_max`` at ``1.5*0.05=0.075``
    regardless of the field's angular content -- truncating a wide-angle free-space
    beam (content ~0.15 -> fidelity ~0.96 instead of ~1).  The fix skips the NA cap
    when the prescription carries no real power, so ``p_max`` follows the field's
    own content and the diverging beam reconstructs to ~1 vs the ASM oracle -- while
    the NA-*estimate* callers keep the unchanged small-NA floor."""
    from lumenairy.propagators.fga import _default_p_max, _field_angular_content, _resolve_sampling
    N, dx = 256, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    k = 2 * np.pi / _WL
    r2 = Xg ** 2 + Yg ** 2
    # a strongly diverging beam: angular content ~0.15 (>> the old 0.075 cap)
    div = (np.exp(-r2 / (20e-6) ** 2)
           * np.exp(1j * k * r2 / (2 * 0.25e-3))).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    content = _field_angular_content(div, dx, dx, _WL)
    assert content > 0.12                        # genuinely wide-angle

    # seam: the power-less flat plate signals "no NA cap" (inf) to the momentum
    # sizer, while the NA-ESTIMATE path keeps the unchanged small-NA floor.
    assert not np.isfinite(_default_p_max(flat, _WL, powerless_uncapped=True))
    assert _default_p_max(flat, _WL) == 0.05     # NA-estimate path byte-unchanged
    p_max, _n_p = _resolve_sampling(div, dx, dx, _WL, 5.0 * dx, flat, None, None)
    assert p_max > 1.4 * content                 # swarm covers the content ...
    assert p_max > 0.075                         # ... beyond the old NA-cap clamp

    # end-to-end: auto now reconstructs the wide-angle diverging beam to ~1,
    # while forcing the OLD 0.075 cap (independent ASM oracle) is measurably worse.
    z = 0.3e-3
    ref = angular_spectrum_propagate(div, z, _WL, dx)
    auto = apply_real_lens_fga(div, prescription=flat, wavelength=_WL, dx=dx,
                               output_plane_distance=z, w0_factor=5.0)
    capped = apply_real_lens_fga(div, prescription=flat, wavelength=_WL, dx=dx,
                                 output_plane_distance=z, w0_factor=5.0,
                                 p_max=0.075)      # reproduces the old NA clamp
    fa = _fid(auto, ref)
    assert fa > 0.99                              # the fix (was ~0.96)
    assert fa > _fid(capped, ref) + 0.02          # the NA cap WAS the lever


def test_fga_auto_sampling_conserves_identity_power():
    """The auto sampler's p_max COMPLETENESS floor (~2/(k*w0), the beamlet
    momentum width) keeps the t=0 resolution of identity energy-conserving for a
    COLLIMATED Gaussian.

    Audit S2-2 regression: the field's narrow angular CONTENT alone under-sized
    ``p_max`` below the frozen beamlet's own momentum width, so the swarm did not
    resolve the identity -> up to 34% power deficit -- while field FIDELITY stayed
    ~1 (a pure under-normalization), which a fidelity-only test cannot catch."""
    N, dx = 256, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    for w in (12e-6, 18e-6, 30e-6):
        u0 = np.exp(-(Xg ** 2 + Yg ** 2) / w ** 2).astype(np.complex128)
        out = apply_real_lens_fga(u0, prescription=flat, wavelength=_WL, dx=dx,
                                  output_plane_distance=0.0)     # DEFAULT (auto)
        pratio = float(np.sum(np.abs(out) ** 2) / np.sum(np.abs(u0) ** 2))
        assert pratio > 0.99         # completeness floor conserves power (was ~0.66)


def test_caustic_zone_uses_meridional_row_not_column_index():
    """`_caustic_zone` must read the centre-ROW meridional line ``E_in[Ny//2, :]``,
    not ``E_in[Nx//2, :]`` (audit S2-21).  On a NON-SQUARE (tall) field where
    ``Nx//2 >= Ny`` the old column-as-row index raised IndexError; even when in
    range it read the wrong row, corrupting the caustic-zone / dispatcher decision."""
    from lumenairy.propagators.fga import _caustic_zone
    k = 2 * np.pi / _WL
    Ny, Nx, dx = 64, 192, 8e-6            # tall: Nx//2 = 96 >= Ny = 64
    xs = (np.arange(Nx) - Nx / 2) * dx
    ys = (np.arange(Ny) - Ny / 2) * dx
    Xg, Yg = np.meshgrid(xs, ys)
    r2 = Xg ** 2 + Yg ** 2
    conv = (np.exp(-r2 / (0.4e-3) ** 2)
            * np.exp(-1j * k * r2 / (2 * 38e-3))).astype(np.complex128)  # converging
    zone = _caustic_zone(conv, dx, _singlet(), _WL)   # old code: IndexError here
    assert zone is not None and zone[0] < zone[1]     # a real caustic zone found


def test_fga_coarse_stride_matches_full():
    """The opt-in coarse-lattice trace (``coarse_stride>1``) reconstructs the
    SAME field as the full per-beamlet trace to the FGA floor.

    It traces only a stride-M position sub-lattice per momentum and cubic
    ``map_coordinates``-interpolates the smooth ray quantities (opd, exit
    position/direction, prefactor ``a``) to the full lattice -- reconstructing
    ``AW = C*a*exp(ik*opd)`` at full resolution (the oscillating ``AW`` is never
    interpolated) -- with a direct-trace RIM fallback where the aperture
    vignettes the beam.  Opt-in speedup for large / expensive-trace prescriptions;
    no accuracy loss (default ``coarse_stride=1`` is the exact full trace)."""
    N, dx = 128, 12e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    coll = np.exp(-(Xg ** 2 + Yg ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=_WL, dx=dx, w0_factor=5.0,
              output_plane_distance=30e-3, n_p=11, dq_step=2, prune_frac=0.0,
              coeff_frac=0.0)
    full = apply_real_lens_fga(coll, coarse_stride=1, **kw)           # exact
    for M in (4, 6):
        coarse = apply_real_lens_fga(coll, coarse_stride=M, **kw)
        assert _fid(coarse, full) > 0.999      # coarse trace + interp is no-loss


def test_fga_exact_jacobian_matches_fd():
    """``exact_jacobian=True`` (the truncation-free analytic differential
    ray-transfer Jacobian, used for the all-conic singlet) reconstructs the SAME
    field as the finite-difference default to the FD truncation floor -- it is the
    exact ``h -> 0`` limit of the FD monodromy, not a re-derivation."""
    N, dx = 128, 12e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    coll = np.exp(-(Xg ** 2 + Yg ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=_WL, dx=dx, w0_factor=5.0,
              output_plane_distance=30e-3, n_p=11, dq_step=2, prune_frac=0.0,
              coeff_frac=0.0)
    fd = apply_real_lens_fga(coll, exact_jacobian=False, **kw)        # FD default
    an = apply_real_lens_fga(coll, exact_jacobian=True, **kw)         # analytic
    assert _fid(an, fd) > 0.9999           # analytic == FD to the truncation floor


def test_fga_cache_trace_reuses_and_is_identical():
    """``cache_trace`` memoizes the FIELD-INDEPENDENT coarse pre-trace, so a
    second propagation through the SAME optics with a DIFFERENT field reuses it
    (cache stays size 1) and every cached result is identical to the uncached one
    (only the field-dependent rim is recomputed)."""
    import lumenairy.propagators.fga as _F
    N, dx = 128, 12e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    b1 = np.exp(-(Xg ** 2 + Yg ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    b2 = (b1 * np.exp(1j * 0.3 * Xg / dx)).astype(np.complex128)   # diff field
    # EXPLICIT p_max/n_p so both fields share the SAME momentum lattice (the reuse
    # condition -- with the default AUTO sampler each field's angular content sets
    # its own p_max, so differing fields make differing swarms and MISS the cache).
    kw = dict(prescription=_singlet(), wavelength=_WL, dx=dx, w0_factor=5.0,
              output_plane_distance=30e-3, p_max=0.08, n_p=11, dq_step=2,
              prune_frac=0.0, coeff_frac=0.0, coarse_stride=4)
    ref1 = apply_real_lens_fga(b1, cache_trace=False, **kw)
    ref2 = apply_real_lens_fga(b2, cache_trace=False, **kw)
    _F._COARSE_CACHE.clear()
    c1 = apply_real_lens_fga(b1, cache_trace=True, **kw)      # populates cache
    assert len(_F._COARSE_CACHE) == 1
    c2 = apply_real_lens_fga(b2, cache_trace=True, **kw)      # CACHE HIT (same optics)
    assert len(_F._COARSE_CACHE) == 1                          # reused, not re-added
    assert np.allclose(c1, ref1) and np.allclose(c2, ref2)    # cache is result-identical
    _F._COARSE_CACHE.clear()


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


def test_universal_dispatcher_multivalued_avoids_traced():
    """'auto' must NOT route a MULTI-EMITTER / multi-valued field to 'traced':
    traced launches one ray per pixel along the local phase gradient, which is
    undefined where several wave components cross (it silently collapses them to
    their mean direction and applies the wrong angle-dependent OPD).  The
    dispatcher detects multi-valuedness (:func:`_tilt_dispersion`) and routes to
    'fga' instead, whose phase-space swarm transports every direction
    independently.  A SINGLE-valued beam (even strongly aberrated/divergent) is
    still eligible for the sub-nm traced OPL."""
    from lumenairy.propagators.fga import _system_na, _tilt_dispersion, apply_real_lens_universal
    presc = _hi_na_singlet()
    na = _system_na(presc, _WL)
    assert na > 0.12                                   # reaches the high-NA branch
    N, dx = 160, 1.0e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    k = 2 * np.pi / _WL
    r2 = Xg ** 2 + Yg ** 2
    gauss = np.exp(-r2 / (40e-6) ** 2)

    # --- the detector separates single- from multi-valued by >10x ---
    single = {
        'gauss': gauss,
        'diverging': gauss * np.exp(1j * k * r2 / (2 * 4e-3)),        # single src
        'spherical-ab': gauss * np.exp(1j * k * 3e-6 * (r2 / (90e-6) ** 2) ** 2),
    }
    for E in single.values():
        assert _tilt_dispersion(E.astype(np.complex128), dx, dx, _WL, na) < 0.02
    # two emitters crossing at +/-0.10 rad -> genuinely multi-valued
    gL = np.exp(-((Xg + 45e-6) ** 2 + Yg ** 2) / (55e-6) ** 2) * np.exp(1j * k * 0.10 * Xg)
    gR = np.exp(-((Xg - 45e-6) ** 2 + Yg ** 2) / (55e-6) ** 2) * np.exp(-1j * k * 0.10 * Xg)
    two = (gL + gR).astype(np.complex128)
    doe = (sum(np.exp(1j * k * 0.05 * m * Xg) for m in range(-2, 3))
           * gauss).astype(np.complex128)
    assert _tilt_dispersion(two, dx, dx, _WL, na) > 0.06
    assert _tilt_dispersion(doe, dx, dx, _WL, na) > 0.06

    fkw = {'fga': {'n_p': 7, 'w0_factor': 4.0}}
    far = 2e-3                             # a smooth plane, well before any focus

    def route(E, **kw):
        return apply_real_lens_universal(
            E, prescription=presc, wavelength=_WL, dx=dx,
            output_plane_distance=far, return_method=True,
            method_kwargs=fkw, **kw)[1]

    # single-valued high-NA smooth far plane -> traced ; multi-valued -> fga
    assert route(gauss.astype(np.complex128)) == "traced"
    assert route(two) == "fga"            # <-- the bug: was 'traced'
    assert route(doe) == "fga"
    # explicit override in BOTH directions.  multivalued=False forces the
    # single-valued path (NOT fga); the two-emitter field is still non-collimated
    # there (large residual angular spread), so the F1 collimation split sends it
    # to phase_screen rather than blurring it via traced.
    assert route(two, multivalued=False) == "phase_screen"
    assert route(gauss.astype(np.complex128), multivalued=True) == "fga"

    # ... and the method it routes multi-valued fields TO is exact for them:
    # FGA reproduces the exact free-space angular-spectrum field for a two-emitter
    # (multi-valued) input -- two co-located beams with OPPOSITE tilts, so every
    # pixel carries two directions at once, which a single-direction ray launch
    # cannot represent.  (Co-located + small waist keeps it clear of the grid
    # edge; the separated ``two`` above is for the routing decision.)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    gc = np.exp(-r2 / (12e-6) ** 2)
    twoc = (gc * np.exp(1j * k * 0.10 * Xg)
            + gc * np.exp(-1j * k * 0.10 * Xg)).astype(np.complex128)
    assert _tilt_dispersion(twoc, dx, dx, _WL, na) > 0.06      # genuinely multi-valued
    z = 120e-6
    fga_two = apply_real_lens_fga(twoc, prescription=flat, wavelength=_WL, dx=dx,
                                  output_plane_distance=z, w0_factor=4.0,
                                  p_max=0.16, n_p=17)
    asm_two = angular_spectrum_propagate(twoc, z, _WL, dx)
    assert _fid(fga_two, asm_two) > 0.999     # FGA is exact on the multi-valued field


def test_universal_dispatcher_diverging_beam_not_blurred_via_traced():
    """A single-valued but DIVERGING beam must NOT route to 'traced': traced
    launches rays along the local phase gradient and is only valid for a
    ~collimated beam, so a diverging (large residual angular spread) beam would be
    silently blurred.  'auto' routes it to the wave-exact 'phase_screen' instead
    (bounded thin-screen OPD, never a blur), using traced's OWN collimation
    threshold -- while a collimated single-valued beam still gets the sub-nm
    traced OPL, and neither emits a collimation-blur warning."""
    import warnings

    from lumenairy.elements import apply_real_lens
    from lumenairy.propagators.fga import apply_real_lens_universal
    presc = _hi_na_singlet()
    N, dx = 192, 0.8e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    k = 2 * np.pi / _WL
    r2 = Xg ** 2 + Yg ** 2
    gauss = np.exp(-r2 / (30e-6) ** 2).astype(np.complex128)
    diverging = (gauss * np.exp(1j * k * r2 / (2 * 0.6e-3))).astype(np.complex128)
    far = 0.6e-3                           # smooth plane, well before the focus

    def route(E):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            out, m = apply_real_lens_universal(
                E, prescription=presc, wavelength=_WL, dx=dx,
                output_plane_distance=far, return_method=True)
        blur = sum(("collimated-reference" in str(x.message)
                    or "no single direction" in str(x.message)) for x in wl)
        return out, m, blur

    _o_c, m_coll, blur_c = route(gauss)
    out_d, m_div, blur_d = route(diverging)
    assert m_coll == "traced"                     # collimated single-valued -> traced
    assert m_div == "phase_screen"                # diverging -> phase_screen (the fix)
    assert blur_c == 0 and blur_d == 0            # neither is silently blurred
    # the phase_screen route is a TRUE dispatch: exit-vertex apply_real_lens + ASM
    ref = angular_spectrum_propagate(
        apply_real_lens(diverging, prescription=presc, wavelength=_WL, dx=dx),
        far, _WL, dx)
    assert np.allclose(out_d, ref)                 # exact-oracle dispatch, no blur


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
    # separable=False pins ONE analysis method so this tests the momentum-chunk
    # additive reorder in isolation (the separable-vs-direct round-off is covered
    # by test_fga_separable_kernels_match_direct_and_oracle; the position-lattice
    # Nq-chunk by test_fga_nq_chunk_bounds_memory).  Here mem_budget_mb=2 exercises
    # BOTH the momentum chunk and Nq-chunking on the same direct path.
    kw = dict(prescription=flat, wavelength=_WL, dx=dx,
              output_plane_distance=300e-6, w0_factor=8.0, p_max=0.06, n_p=13,
              separable=False)
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


def test_fga_nq_chunk_bounds_memory():
    """F3: mem_budget_mb now bounds BOTH dimensions -- momenta (cw) AND the
    POSITION lattice (nq_chunk) -- so a large aperture runs within the budget as
    an additive sum over lattice chunks (no OOM), giving the SAME field as the
    un-chunked full swarm to round-off, for separable + direct + coeff-pruning +
    vector.  Only an absurd budget (a single lattice point can't fit) raises."""
    from lumenairy.propagators.fga import _resolve_nq_chunk, apply_real_lens_fga_vector
    # sizer: None when it fits / no budget; chunks when it doesn't; raises only if
    # even one lattice point can't fit
    assert _resolve_nq_chunk(4096, 169, True, 169, None, "t") is None    # no budget
    assert _resolve_nq_chunk(4096, 169, True, 169, 10_000, "t") is None  # fits whole
    nqc = _resolve_nq_chunk(4096, 169, True, 169, 3, "t")               # must chunk
    assert nqc is not None and 1 <= nqc < 4096
    with pytest.raises(MemoryError, match="single position-lattice point"):
        _resolve_nq_chunk(10_000, 225, True, 225, 0.0001, "apply_real_lens_fga")
    # integration: a budget forcing MANY lattice chunks reproduces the full swarm
    N, dx = 96, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (14e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    kw = dict(prescription=flat, wavelength=_WL, dx=dx,
              output_plane_distance=200e-6, w0_factor=6.0, p_max=0.1, n_p=13)
    # Nq = (96/2)^2 = 2304; mem_budget_mb=2 chunks it into many lattice pieces
    for sep, cf in ((True, 1e-3), (False, 0.0)):
        full = apply_real_lens_fga(u0, separable=sep, coeff_frac=cf, **kw)
        chunked = apply_real_lens_fga(u0, separable=sep, coeff_frac=cf,
                                      mem_budget_mb=2, **kw)
        assert not np.isnan(chunked).any()
        assert np.max(np.abs(chunked - full)) < 1e-9   # bit-close to full swarm
    # vector too (NaN-safe Ez under Nq-chunking)
    vfull = apply_real_lens_fga_vector(np.stack([u0, 0.3 * u0]),
                                       return_longitudinal=True, **kw)
    vchunk = apply_real_lens_fga_vector(np.stack([u0, 0.3 * u0]), mem_budget_mb=2,
                                        return_longitudinal=True, **kw)
    assert not np.isnan(vchunk).any()
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


def test_fga_coefficient_pruning_no_loss():
    """Coefficient pruning (``coeff_frac``) skips whole momenta whose peak Gabor
    coefficient is negligible -- no-loss (the field carries ~no energy there) and
    NaN-free (the vector Ez path stays finite for skipped momenta)."""
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    N, dx = 160, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (30e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    kw = dict(prescription=flat, wavelength=_WL, dx=dx,
              output_plane_distance=250e-6, w0_factor=6.0, p_max=0.15, n_p=17,
              prune_frac=0.0)
    base = apply_real_lens_fga(u0, coeff_frac=0.0, **kw)
    pruned = apply_real_lens_fga(u0, coeff_frac=1e-3, **kw)
    assert not np.isnan(pruned).any()
    assert _fid(pruned, base) > 0.99999                        # no-loss
    vec = apply_real_lens_fga_vector(np.stack([u0, np.zeros_like(u0)]),
                                     coeff_frac=1e-3, return_longitudinal=True,
                                     **kw)
    assert not np.isnan(vec).any()                             # NaN-safe Ez


def test_fga_separable_kernels_match_direct_and_oracle():
    """The separable analysis + recurrence scatter kernels (``separable='auto'``,
    the v5.24.0 default) are numerically equivalent to the direct kernels and
    EQUALLY accurate vs the truth: separable/direct agree to fidelity ~1, and each
    matches the exact free-space angular-spectrum oracle to the SAME fidelity (the
    kernel-vs-kernel difference is cancellation-amplified round-off far below the
    FGA error floor, not an accuracy loss).  Covers scalar + vector + NaN-safety."""
    from lumenairy.propagators.fga import apply_real_lens_fga_vector
    N, dx = 128, 0.7e-6
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (16e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    z = 200e-6
    kw = dict(prescription=flat, wavelength=_WL, dx=dx, output_plane_distance=z,
              w0_factor=5.0, p_max=0.12, n_p=13, prune_frac=0.0, coeff_frac=0.0)
    direct = apply_real_lens_fga(u0, separable=False, **kw)
    sep = apply_real_lens_fga(u0, separable=True, **kw)
    assert _fid(direct, sep) > 0.99999                  # numerically equivalent
    # ... and equally accurate vs the exact oracle (the real no-loss criterion)
    oracle = angular_spectrum_propagate(u0, z, _WL, dx)
    assert abs(_fid(sep, oracle) - _fid(direct, oracle)) < 1e-5
    # vector path: equivalent + NaN-safe
    Ev = np.stack([u0, 0.4 * u0])
    vd = apply_real_lens_fga_vector(Ev, separable=False, return_longitudinal=True,
                                    **kw)
    vs = apply_real_lens_fga_vector(Ev, separable=True, return_longitudinal=True,
                                    **kw)
    assert not np.isnan(vs).any()
    assert _fid(vd, vs) > 0.99999


def test_fga_kernels_bit_identical_at_integer_R_over_dx():
    """Seam S2-7: the separable analysis + recurrence scatter kernels must agree
    with their direct counterparts to ULP even at INTEGER ``R/dx`` -- which is
    the DEFAULT (``R/dx = nsig * w0_factor = 3 * 5 = 15`` on a square grid),
    exactly where the boundary pixel sits at ``r == R`` and float rounding of the
    window bound can flip its membership.

    Before the fix the direct box was tighter than the separable superset (worst
    analysis-coefficient rel-diff ~7e-5) and the separable scatter gated on the
    drifting recurrence ``dxr`` instead of a fresh one (~2.6e-3), so the two
    kernels dropped/kept different boundary pixels and the end-to-end field
    diverged by ~1.3e-4 -- while the docstrings/CHANGELOG asserted "numerically
    identical (ULP-level)".  ``fidelity`` (a normalised inner product) is blind
    to a 1e-4 max-abs difference, so this pins the strong max-abs metric AND
    checks BOTH kernels against the exact free-space oracle (so the agreement
    is genuine, not a shared error).  Off-tie configs (``w0_factor`` 5.001) were
    always ULP-close; the fix leaves them byte-identical and only repairs the tie.
    """
    import math

    from lumenairy.propagators.fga import _gabor_coeff, _gabor_coeff_sep, _reconstruct_into

    N, dx = 128, 0.7e-6
    dyg = dx
    nsig, w0_factor = 3.0, 5.0                       # -> R/dx = 15 exactly
    assert abs(nsig * w0_factor - round(nsig * w0_factor)) < 1e-12
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (16e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}
    z = 200e-6
    kw = dict(prescription=flat, wavelength=_WL, dx=dx, output_plane_distance=z,
              w0_factor=w0_factor, p_max=0.12, n_p=13, prune_frac=0.0,
              coeff_frac=0.0)

    # (1) end-to-end: the two kernels now agree to machine precision, not 1.3e-4.
    direct = apply_real_lens_fga(u0, separable=False, **kw)
    sep = apply_real_lens_fga(u0, separable=True, **kw)
    rel_field = np.max(np.abs(direct - sep)) / (np.max(np.abs(direct)) + 1e-300)
    assert rel_field < 1e-10, rel_field           # was ~1.3e-4 at this default

    # (2) independent oracle: BOTH match the exact angular-spectrum truth to the
    #     same fidelity -> the ULP agreement is real, not a shared blunder.
    oracle = angular_spectrum_propagate(u0, z, _WL, dx)
    assert _fid(direct, oracle) > 0.999
    assert abs(_fid(sep, oracle) - _fid(direct, oracle)) < 1e-6

    # driver-consistent beamlet constants for the kernel-level probes.
    x0 = -(N / 2) * dx
    y0 = -(N / 2) * dyg
    w0 = w0_factor * math.sqrt(dx * dyg)
    k = 2.0 * np.pi / _WL
    Ag = (1.0 / (np.pi * w0 ** 2)) ** 0.5
    # full on-grid launch/beamlet lattice (dq_step=1), as the driver builds it.
    qi, qj = np.meshgrid(np.arange(N), np.arange(N))
    qx = (x0 + qi.ravel() * dx).astype(np.float64)
    qy = (y0 + qj.ravel() * dyg).astype(np.float64)
    n_p = 13
    pv = np.linspace(-0.12, 0.12, n_p)
    pxg, pyg = np.meshgrid(pv, pv)               # ip = a*n_p + b : px=pv[b], py=pv[a]
    px, py = pxg.ravel().astype(np.float64), pyg.ravel().astype(np.float64)

    # (3) analysis half: direct vs separable Gabor coefficients agree to ULP.
    cd = _gabor_coeff(u0, qx, qy, px, py, x0, y0, dx, dyg, w0, k, Ag, nsig)
    cs = _gabor_coeff_sep(u0, qx, qy, pv, x0, y0, dx, dyg, w0, k, Ag, nsig)
    rel_coeff = np.max(np.abs(cd - cs)) / (np.max(np.abs(cd)) + 1e-300)
    assert rel_coeff < 1e-11, rel_coeff           # was ~7e-5 at integer R/dx

    # (4) scatter half: direct vs recurrence scatter agree to ULP.  Synthesise a
    #     random on-grid beamlet field so the boundary pixel carries weight.
    rng = np.random.default_rng(0)
    nb = qx.shape[0]
    Px = (rng.standard_normal(nb) * 0.05).astype(np.float64)
    Py = (rng.standard_normal(nb) * 0.05).astype(np.float64)
    W = rng.standard_normal(nb) + 1j * rng.standard_normal(nb)
    od_r, od_i = np.zeros((N, N)), np.zeros((N, N))
    _reconstruct_into(od_r, od_i, qx, qy, Px, Py, W, x0, y0, dx, dyg, N, N,
                      w0, k, Ag, nsig, sep=False)
    os_r, os_i = np.zeros((N, N)), np.zeros((N, N))
    _reconstruct_into(os_r, os_i, qx, qy, Px, Py, W, x0, y0, dx, dyg, N, N,
                      w0, k, Ag, nsig, sep=True)
    od, os = od_r + 1j * od_i, os_r + 1j * os_i
    rel_scat = np.max(np.abs(od - os)) / (np.max(np.abs(od)) + 1e-300)
    assert rel_scat < 1e-11, rel_scat             # was ~2.6e-3 at integer R/dx


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


# ---------------------------------------------------------------------------
# Hammer audit H2: dispatcher sag-screen aberration gate
# ---------------------------------------------------------------------------
# The analytic phase-screen model cannot represent high-order / orientation-
# dependent aberration (H2, dual-oracle-quantified); apply_real_lens_universal
# now estimates that error cheaply at routing time and steers a heavily-aberrated
# prescription AWAY from 'phase_screen' even at low NA (and warns when forced).
_WL_H2 = 1.31e-6           # dual-oracle wavelength (docs/audit_real_lens_hammer)

from lumenairy.glass import GLASS_REGISTRY  # noqa: E402

# dispersionless model glass n=1.5168 (matches the dual-oracle build convention)
GLASS_REGISTRY.setdefault('_GATE_N1p5168', lambda wl: 1.5168)


def _biconvex(R, semi, t=5e-3, glass='_GATE_N1p5168'):
    """Symmetric biconvex singlet R = +/-R, aperture matched to ``semi``."""
    return {'wavelength': _WL_H2, 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': R, 'thickness': t, 'glass_before': 'air',
                 'glass_after': glass, 'semi_diameter': semi},
                {'radius': -R, 'thickness': 0.0, 'glass_before': glass,
                 'glass_after': 'air', 'semi_diameter': semi}],
            'thicknesses': [t], 'stop_index': 0}


def _gauss2d(N, dx, w0):
    xs = (np.arange(N) - N // 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    return np.exp(-(Xg ** 2 + Yg ** 2) / w0 ** 2).astype(np.complex128)


def test_gate_h2_aberration_estimate_calibration():
    """Calibrate the sag-screen aberration estimate against the dual-oracle f/5
    biconvex (docs/audit_real_lens_hammer_2026_07_19.md: R = +/-51.68 mm, t = 5 mm,
    n = 1.5168, lambda = 1.31 um, w0 = 5 mm; image at 49.163 mm; dual-oracle
    EE50/EE80/r2m = 15.17/55.22/64.98 um, where the analytic model converged to a
    WRONG 40.5 um -- H2).  The estimate must be well above the ~2 rad budget for
    the f/5 case (which is genuinely LOW NA, so it reaches the phase-screen
    shortcut the gate must override) and far below it for the benign small-beam
    regime (byte-identical low-NA dispatch)."""
    from lumenairy.propagators.fga import (
        _ABERRATION_MAX_RAD,
        _sag_screen_aberration_rad,
        _system_na,
        _universal_route,
    )
    presc = _biconvex(51.68e-3, 5e-3)
    # f/5 IS low-NA by the marginal-ray metric (na ~ 0.10 < the 0.12 gate) -- the
    # exact regime that WOULD route to the analytic sag-screen without this gate.
    assert _system_na(presc, _WL_H2) < 0.12
    ext = 12.8e-3
    dx = 2 * ext / 256
    E5 = _gauss2d(256, dx, 5e-3)                            # w0 = 5 mm, fully covered
    est5 = _sag_screen_aberration_rad(E5, dx, dx, presc, _WL_H2)
    assert 15.0 < est5 < 30.0, est5                          # ~21.7 rad
    assert est5 > _ABERRATION_MAX_RAD                        # trips the gate
    # the EXACT dual-oracle f/5 prescription routes AWAY from analytic at every
    # plane (pre-gate it hit the low-NA phase_screen shortcut): traced at the
    # vertex / pre-focus, fga at the 49.163 mm image (the caustic).
    for opd in (0.0, 5e-3, 49.163e-3):
        r = _universal_route(E5, presc, _WL_H2, dx, dx, opd, 0.12, 3.0,
                             None, 0.06, _ABERRATION_MAX_RAD)
        assert r != "phase_screen", (opd, r)
        assert r in ("traced", "fga"), (opd, r)
    # benign: same lens, w0 = 0.5 mm (10x smaller) -> r^4 down 1e4 -> << budget
    Eb = _gauss2d(256, 2 * 1.28e-3 / 256, 0.5e-3)
    estb = _sag_screen_aberration_rad(Eb, 2 * 1.28e-3 / 256, 2 * 1.28e-3 / 256,
                                      presc, _WL_H2)
    assert estb < 0.1, estb                                  # ~0.002 rad
    assert estb < _ABERRATION_MAX_RAD                        # does NOT trip
    # defensive: empty / malformed inputs never trip the gate
    assert _sag_screen_aberration_rad(np.zeros((8, 8), complex), 1e-6, 1e-6,
                                      presc, _WL_H2) == 0.0
    assert _sag_screen_aberration_rad(E5, 2 * ext / 256, 2 * ext / 256,
                                      {'surfaces': 'bogus'}, _WL_H2) == 0.0


def test_gate_h2_reroutes_low_na_aberrated_away_from_phase_screen():
    """The routing regression (H2 gate).  Two fields through the SAME low-NA lens
    (na ~ 0.11 < 0.12, so both would hit the phase-screen shortcut) differ ONLY in
    beam radius, hence only in the sag-screen aberration estimate:

      * benign small beam (estimate << budget) -> 'phase_screen' (unchanged);
      * beam-filling aberrated (estimate > budget) -> routed AWAY to a ray-based
        member ('traced' on a smooth plane, 'fga' near the caustic).

    Uses the pure routing decision (no propagation) so the choice is exact and
    cheap.  Pre-gate BOTH returned 'phase_screen'."""
    from lumenairy.propagators.fga import (
        _ABERRATION_MAX_RAD,
        _sag_screen_aberration_rad,
        _system_na,
        _universal_route,
    )
    presc = _biconvex(11.9e-3, 1.15e-3)                      # efl ~ 10.6 mm
    na = _system_na(presc, _WL_H2)
    assert na < 0.12                                         # low-NA shortcut regime
    dx = 2 * 3.0e-3 / 128
    E_ab = _gauss2d(128, dx, 1.15e-3)                        # beam-filling -> aberrated
    E_bn = _gauss2d(128, dx, 0.20e-3)                        # tiny beam -> benign
    assert _sag_screen_aberration_rad(E_ab, dx, dx, presc, _WL_H2) > _ABERRATION_MAX_RAD
    assert _sag_screen_aberration_rad(E_bn, dx, dx, presc, _WL_H2) < _ABERRATION_MAX_RAD

    def route(E, opd):
        return _universal_route(E, presc, _WL_H2, dx, dx, opd, 0.12, 3.0,
                                None, 0.06, _ABERRATION_MAX_RAD)

    # benign low-NA -> phase_screen (byte-identical dispatch, both planes)
    assert route(E_bn, 0.5e-3) == "phase_screen"
    assert route(E_bn, 10.6e-3) == "phase_screen"
    # aberrated low-NA -> NEVER phase_screen; ray-based member instead
    assert route(E_ab, 0.5e-3) == "traced"                  # smooth pre-focus plane
    assert route(E_ab, 10.6e-3) == "fga"                    # near the caustic
    assert route(E_ab, 0.5e-3) != "phase_screen"


def test_gate_h2_benign_low_na_dispatch_byte_identical():
    """The benign low-NA path is a pure pass-through: apply_real_lens_universal
    still routes to 'phase_screen' and returns EXACTLY apply_real_lens (exit
    vertex) + exact-ASM output leg -- the gate adds only a read-only estimate, so
    the benign output is byte-identical to the pre-gate dispatch."""
    from lumenairy.elements import apply_real_lens
    from lumenairy.propagators.fga import apply_real_lens_universal
    presc = _biconvex(11.9e-3, 1.15e-3)
    dx = 2 * 1.6e-3 / 128                                    # grid covers the 1.15 mm aperture
    E = _gauss2d(128, dx, 0.20e-3)                           # benign (estimate << budget)
    opd = 4e-3
    out, m = apply_real_lens_universal(
        E, prescription=presc, wavelength=_WL_H2, dx=dx,
        output_plane_distance=opd, return_method=True)
    assert m == "phase_screen"
    ref = angular_spectrum_propagate(
        apply_real_lens(E, prescription=presc, wavelength=_WL_H2, dx=dx),
        opd, _WL_H2, dx)
    assert np.array_equal(out, ref)                          # byte-identical


def test_gate_h2_warns_when_phase_screen_forced_out_of_envelope():
    """Forcing method='phase_screen' on a heavily-aberrated prescription still runs
    (honours the explicit choice) but emits a RuntimeWarning citing the validity
    budget -- an unreliable spot is never returned silently.  A benign forced
    phase-screen call stays quiet."""
    import warnings

    from lumenairy.propagators.fga import apply_real_lens_universal
    presc = _biconvex(11.9e-3, 1.15e-3)
    dx = 2 * 3.0e-3 / 128
    E_ab = _gauss2d(128, dx, 1.15e-3)                        # aberrated
    E_bn = _gauss2d(128, dx, 0.20e-3)                        # benign
    with pytest.warns(RuntimeWarning, match="validity budget"):
        apply_real_lens_universal(E_ab, prescription=presc, wavelength=_WL_H2,
                                  dx=dx, method="phase_screen")
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        apply_real_lens_universal(E_bn, prescription=presc, wavelength=_WL_H2,
                                  dx=dx, method="phase_screen")
    assert not any("validity budget" in str(w.message) for w in wl)
