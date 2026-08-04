"""Regression tests for the v5.24.2 exhaustive-audit G06-perf group.

Each test pins ONE finding's fix with an INDEPENDENT oracle (a hand-built
reference / a conservation law / a call-count) rather than the library's own
formula, per the remediation contract.  All the perf fixes are advertised
"no-loss" (byte-identical or physically invariant), so the tests assert
byte-identity where the fix claims it and a physical invariant otherwise.

Findings covered:
  S5-8b  numexpr fused complex-exp in the ASM H build (byte-identical)
  S5-8c  pyFFTW default thread count capped at the oversubscription knee
  S5-8e  centred-meshgrid cache in analysis.beam_stats (byte-identical)
  S5-8f  idle pyfftw.interfaces.cache daemon stays disabled
  S5-8g  ASM H built directly in natural FFT order (byte-identical)
  S1-11  Berreman OOP-oblique eig/interface caching (fixed-angle wl sweep)
  S1-18  BOR nodal build_layer: single dense eig, reldiv harvested from it
  S3-18  make_shack_hartmann_wfs caches the Zernike reconstructor
  S2-16  HK slope->direction-cosine scalar-Jacobian limit is documented
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

# ===========================================================================
# ASM H-build helpers -- the INDEPENDENT reference (pre-S5-8b/S5-8g algorithm):
# build the transfer function on the CENTERED grid with plain ``np.exp`` and
# then ``ifftshift`` into natural layout.  The library now builds it directly
# in natural order (S5-8g) and fuses the exp through numexpr (S5-8b); both must
# reproduce this reference byte-for-byte.
# ===========================================================================

def _ref_H_natural(Ny, Nx, dy, dx, wavelength, z, bandlimit, cdtype):
    k = 2.0 * np.pi / wavelength
    fx = (np.arange(Nx) - Nx / 2) / (Nx * dx)
    fy = (np.arange(Ny) - Ny / 2) / (Ny * dy)
    kx_sq = (2 * np.pi * fx) ** 2
    ky_sq = (2 * np.pi * fy) ** 2
    kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
    if cdtype == np.complex128:
        H = np.where(prop, np.exp(1j * kz * z), 0).astype(cdtype)
    else:
        fdt = np.float32
        phase = np.mod(kz * z, 2.0 * np.pi)
        c = np.cos(phase).astype(fdt)
        s = np.sin(phase).astype(fdt)
        H = np.empty((Ny, Nx), dtype=cdtype)
        H.real[:] = np.where(prop, c, fdt(0))
        H.imag[:] = np.where(prop, s, fdt(0))
    if bandlimit and z != 0:
        Lx, Ly = Nx * dx, Ny * dy
        fxm = Lx / (2 * wavelength * abs(z))
        fym = Ly / (2 * wavelength * abs(z))
        mask = (np.abs(fx) < fxm)[None, :] & (np.abs(fy) < fym)[:, None]
        H = H * mask.astype(cdtype)
    return np.fft.ifftshift(H)


@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("z", [8e-3, -5e-3])
@pytest.mark.parametrize("bl", [True, False])
def test_s5_8g_natural_order_byte_identical(N, z, bl):
    """S5-8g: the natural-order H build == the old centered-build+ifftshift,
    byte-for-byte, for both complex128 and complex64."""
    from lumenairy.propagators import asm, fft_infra
    for cdtype in (np.complex128, np.complex64):
        fft_infra.clear_asm_caches()
        H = asm._get_asm_H_natural(N, N, 1e-5, 1e-5, 633e-9, z, bl,
                                   cdtype, np, is_jax=False)
        Href = _ref_H_natural(N, N, 1e-5, 1e-5, 633e-9, z, bl, cdtype)
        assert H.dtype == cdtype
        assert np.array_equal(H, Href), (
            f"natural-order H differs from reference (N={N}, z={z}, bl={bl}, "
            f"{cdtype}); max|diff|={np.max(np.abs(H - Href))}")


def test_s5_8b_numexpr_matches_numpy_byte_identical(monkeypatch):
    """S5-8b: the numexpr-fused complex-exp path is byte-identical to the
    pure-numpy path (importorskip numexpr; force each path explicitly)."""
    pytest.importorskip("numexpr")
    from lumenairy.propagators import asm, fft_infra
    assert asm.NUMEXPR_AVAILABLE, "numexpr importable but flag is False"

    N, z = 512, 7.5e-3
    # numpy path (numexpr disabled)
    fft_infra.clear_asm_caches()
    monkeypatch.setattr(asm, "NUMEXPR_AVAILABLE", False)
    H_np = asm._get_asm_H_natural(N, N, 1e-5, 1e-5, 633e-9, z, True,
                                  np.complex128, np, is_jax=False).copy()
    # numexpr path
    fft_infra.clear_asm_caches()
    monkeypatch.setattr(asm, "NUMEXPR_AVAILABLE", True)
    H_ne = asm._get_asm_H_natural(N, N, 1e-5, 1e-5, 633e-9, z, True,
                                  np.complex128, np, is_jax=False).copy()
    assert np.array_equal(H_np, H_ne), (
        f"numexpr H differs from numpy H; max|diff|={np.max(np.abs(H_np - H_ne))}")
    # ...and both equal the independent reference.
    Href = _ref_H_natural(N, N, 1e-5, 1e-5, 633e-9, z, True, np.complex128)
    assert np.array_equal(H_ne, Href)


def test_s5_8c_default_fftw_threads_capped():
    """S5-8c: the DEFAULT pyFFTW thread count is capped at the
    oversubscription knee (min(available_cpus, cap)), and the
    set_fft_threads(None) reset honours the same cap."""
    from lumenairy.memory import available_cpus
    from lumenairy.propagators import fft_infra as fi
    cap = fi._FFTW_DEFAULT_THREAD_CAP
    assert cap == 8
    expected = max(1, min(int(available_cpus()), cap))
    assert fi._default_fftw_threads() == expected
    assert fi.FFTW_THREADS <= cap
    prior = fi.get_fft_threads()
    try:
        fi.set_fft_threads(2)
        assert fi.FFTW_THREADS == 2
        fi.set_fft_threads(None)                 # reset -> capped default
        assert fi.FFTW_THREADS == expected
    finally:
        fi.set_fft_threads(prior if prior > 0 else None)


def test_s5_8f_pyfftw_interfaces_cache_daemon_disabled():
    """S5-8f: loading pyFFTW must NOT spin up the idle
    pyfftw.interfaces.cache keep-alive daemon (the library drives raw FFTW
    plans only, so the interfaces cache is never populated).

    AUDIT_CI_TEST_TIME_2026_08_03 §3: the substring probe below used to be
    ``"pyfftw" in n``, which made this test structurally UNPASSABLE whenever
    ``pytest-timeout`` was active -- its watchdog thread is named after the
    node ID it is guarding, and THIS test's own node ID contains the
    substring ``pyfftw``.  Reproduced as::

        AssertionError: a pyFFTW cache daemon thread is running:
        ['mainthread', 'pytest_timeout tests/unit/test_audit_g06_perf.py::
         test_s5_8f_pyfftw_interfaces_cache_daemon_disabled']

    That false-fail is the only thing blocking ``--timeout`` as a hang-guard
    on the main gate, so the probe now excludes any thread whose name is a
    pytest-timeout watchdog.  The real daemon this pin is about is created by
    ``pyfftw.interfaces.cache.enable()`` and is named by pyFFTW itself
    (``_Cache``'s thread carries ``pyfftw`` / ``fftwcache`` in its name and
    NEVER carries a node ID), so filtering the watchdog cannot mask it.
    """
    pytest.importorskip("pyfftw")
    from lumenairy.propagators import fft_infra as fi
    assert fi._ensure_pyfftw_loaded()
    names = [t.name.lower() for t in threading.enumerate()]
    # A pytest-timeout watchdog is named ``pytest_timeout <nodeid>``; it is
    # this harness's own thread, not a pyFFTW daemon.
    probe = [n for n in names
             if "pytest_timeout" not in n and "pytest-timeout" not in n]
    assert not any("pyfftw" in n or "fftwcache" in n for n in probe), (
        f"a pyFFTW cache daemon thread is running: {probe} "
        f"(all threads: {names})")


# ===========================================================================
# S5-8e -- centred-meshgrid cache in analysis.beam_stats
# ===========================================================================

def _ref_centroid(E, dx, dy):
    Ny, Nx = E.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    I = np.abs(E) ** 2
    t = I.sum()
    return float((X * I).sum() / t), float((Y * I).sum() / t)


def _ref_d4sigma(E, dx, dy):
    Ny, Nx = E.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    I = np.abs(E) ** 2
    t = I.sum()
    cx = (X * I).sum() / t
    cy = (Y * I).sum() / t
    vx = ((X - cx) ** 2 * I).sum() / t
    vy = ((Y - cy) ** 2 * I).sum() / t
    return float(4 * np.sqrt(vx)), float(4 * np.sqrt(vy))


def test_s5_8e_meshgrid_cache_byte_identical_and_hits():
    """S5-8e: cached meshgrid gives byte-identical centroid / d4sigma vs a
    fresh-meshgrid reference, and a second call at the same geometry reuses
    ONE cached grid (no rebuild)."""
    from lumenairy.analysis import beam_stats as bs
    rng = np.random.default_rng(0)
    E = rng.standard_normal((64, 80)) + 1j * rng.standard_normal((64, 80))
    dx, dy = 1e-5, 1.3e-5

    bs.clear_meshgrid_cache()
    c1 = bs.beam_centroid(E, dx, dy)
    assert c1 == _ref_centroid(E, dx, dy)
    d1 = bs.beam_d4sigma(E, dx, dy)
    assert d1 == _ref_d4sigma(E, dx, dy)

    # Same (Ny, Nx, dx, dy) across centroid + d4sigma -> exactly one entry.
    assert len(bs._MESHGRID_CACHE) == 1
    key = (64, 80, float(dx), float(dy))
    assert key in bs._MESHGRID_CACHE
    grid_first = bs._MESHGRID_CACHE[key]
    # A second call returns the SAME cached array objects (a genuine hit).
    bs.beam_centroid(E, dx, dy)
    assert bs._MESHGRID_CACHE[key] is grid_first


def test_s5_8e_cache_enrolled_in_registry():
    """S5-8e: the meshgrid cache is drained by the central clear_asm_caches
    (registry enrollment), so it cannot silently leak across sessions."""
    import lumenairy as la
    from lumenairy.analysis import beam_stats as bs
    E = np.ones((32, 32), dtype=complex)
    bs.beam_centroid(E, 1e-5)
    assert len(bs._MESHGRID_CACHE) >= 1
    la.clear_asm_caches()
    assert len(bs._MESHGRID_CACHE) == 0


# ===========================================================================
# S1-11 -- Berreman OOP-oblique eig/interface caching
# ===========================================================================

def _tilted_uniaxial_eps(eo=2.25, ee=2.89, tilt_deg=40.0):
    psi = np.deg2rad(tilt_deg)
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.diag([eo, eo, ee]) @ R.T).astype(complex)


def test_s1_11_offplane_oblique_eig_cached_over_wl_sweep():
    """S1-11: a fixed-angle wavelength sweep of a tilted-director (OOP tensor)
    stack routes the wl-independent eig through the shared mode cache -- one
    tagged entry serves the whole sweep -- and the result is byte-identical to
    a cache-cleared solve, with R+T=1 (independent energy oracle)."""
    from lumenairy.elements import berreman as br
    eps = _tilted_uniaxial_eps()
    assert abs(eps[0, 2]) > 1e-3          # genuinely out-of-plane
    layers = [(eps, 550e-9)]
    angle = np.deg2rad(30.0)
    wls = [500e-9, 550e-9, 600e-9, 650e-9]

    br._clear_berreman_mode_cache()
    warm = [br.berreman_jones_1d(layers, 1.5, 1.0, wl, angle=angle) for wl in wls]

    # The per-layer eig went through the cache exactly once (wl-independent).
    tagged = [k for k in br._MODE_CACHE
              if isinstance(k, tuple) and k and k[0] == "offplane_condensed_M"]
    assert len(tagged) == 1, (
        f"expected one wl-independent OOP eig cache entry, got {len(tagged)} "
        "(the generalized path is bypassing the eig cache -- S1-11 regression)")

    # Byte-identity: a cache-cleared solve reproduces the cached answer.
    br._clear_berreman_mode_cache()
    R0, T0, jr0, jt0 = br.berreman_jones_1d(layers, 1.5, 1.0, 550e-9, angle=angle)
    Rw, Tw, jrw, jtw = warm[1]
    assert np.array_equal(jr0, jrw) and np.array_equal(jt0, jtw)
    assert np.array_equal(R0, Rw) and np.array_equal(T0, Tw)

    # Independent physical oracle: the stack is lossless, so R + T = 1.
    for (Rk, Tk, _jr, _jt) in warm:
        assert np.allclose(Rk + Tk, 1.0, atol=1e-10)


# ===========================================================================
# S1-18 -- BOR nodal build_layer runs ONE dense eig, reldiv harvested from it
# ===========================================================================

def test_s1_18_bor_nodal_reldiv_single_eig_byte_identical():
    """S1-18: layer_modes(with_reldiv=True) harvests reldiv from the SAME
    eig(K,B) it runs for the modal basis; it must equal the standalone
    radial_coupled_modes reldiv byte-for-byte (independent second impl)."""
    from lumenairy.elements.bor.coupled_radial_eigensolver import (
        radial_coupled_modes,
    )
    from lumenairy.elements.bor.zcascade import layer_modes

    def eps_profile(rr):
        return np.where(rr <= 3e-6, 2.10 + 0j, 1.96 + 0j)

    m, Rbig, N, k0 = 1, 8e-6, 60, 2 * np.pi / 1.55e-6
    reldiv_new = layer_modes(m, Rbig, N, eps_profile, k0,
                             wall="pec", with_reldiv=True)["reldiv"]
    reldiv_ref = np.array([md["reldiv"] for md in
                           radial_coupled_modes(m, Rbig, N, eps_profile,
                                                k0, wall="pec")])
    assert reldiv_new.shape == reldiv_ref.shape
    assert np.array_equal(reldiv_new, reldiv_ref), (
        f"deduped reldiv diverged; max|diff|={np.max(np.abs(reldiv_new - reldiv_ref))}")


def test_s1_18_build_layer_nodal_still_tags_and_keeps_siblings():
    """S1-18: build_layer(basis='nodal') still returns the byte-identical
    reldiv tag AND the sibling S1-15/S1-16 additions (eps_ceiling)."""
    import warnings

    from lumenairy.elements.bor.bor_solve import build_layer
    from lumenairy.elements.bor.coupled_radial_eigensolver import (
        radial_coupled_modes,
    )

    def eps_profile(rr):
        return np.where(rr <= 3e-6, 2.10 + 0j, 1.96 + 0j)

    m, Rbig, N, k0 = 1, 8e-6, 60, 2 * np.pi / 1.55e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = build_layer(m, Rbig, N, eps_profile, k0, wall="pec",
                        basis="nodal", thickness=1e-6)
    reldiv_ref = np.array([md["reldiv"] for md in
                           radial_coupled_modes(m, Rbig, N, eps_profile,
                                                k0, wall="pec")])
    assert np.array_equal(L["reldiv"], reldiv_ref)
    assert "eps_ceiling" in L                      # S1-16 sibling preserved


# ===========================================================================
# S3-18 -- make_shack_hartmann_wfs caches the Zernike reconstructor
# ===========================================================================

def test_s3_18_wfs_reconstructor_cached_and_identical():
    """S3-18: the Zernike reconstructor (zernike_modal_basis: FD influence
    matrix + pinv) is built ONCE per WFS geometry, not per frame.  Verified by
    a call-count on zernike_modal_basis + byte-identical per-frame output."""
    from lumenairy.analysis import ao

    calls = {"n": 0}
    real_zmb = ao.zernike_modal_basis

    def counting_zmb(*a, **k):
        calls["n"] += 1
        return real_zmb(*a, **k)

    N = 96
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    resid = 0.3 * X + 0.2 * (X ** 2 + Y ** 2) - 0.15 * X * Y

    import unittest.mock as _mock
    with _mock.patch.object(ao, "zernike_modal_basis", counting_zmb):
        wfs = ao.make_shack_hartmann_wfs(
            subaperture_grid=12, n_modes=10, dx_pupil=1e-4,
            noise_sigma_pixels=0.0)
        m1 = wfs(resid)
        m2 = wfs(resid)
        m3 = wfs(resid)

    assert calls["n"] == 1, (
        f"zernike_modal_basis rebuilt {calls['n']} times across 3 AO frames "
        "(expected 1 -- S3-18 reconstructor cache regression)")
    assert np.array_equal(m1, m2) and np.array_equal(m1, m3)
    assert np.all(np.isfinite(m1))


# ===========================================================================
# S2-16 -- HK slope->direction-cosine scalar-Jacobian: documented limit
# ===========================================================================

def _exact_dircos_jacobian(ux, uy):
    """Hand-derived EXACT d(p_x, p_y)/d(u_x, u_y) for p = u/sqrt(1+u^2+v^2)."""
    g = 1.0 + ux ** 2 + uy ** 2
    f = g ** -1.5
    return f * np.array([[1.0 + uy ** 2, -ux * uy],
                         [-ux * uy, 1.0 + ux ** 2]])


def test_s2_16_scalar_jacobian_is_isotropic_part_and_limited():
    """S2-16: the code's scalar factor go = (1+u^2+v^2)^-1.5 is the on-axis /
    isotropic part of the true 2x2 direction-cosine Jacobian.  Pin that it is
    EXACT on-axis, negligibly wrong at moderate NA (why the validated regime is
    unaffected), and materially wrong for a skew high-NA ray (why the limit is
    documented, not silently 'fine')."""
    # On-axis: scalar == full 2x2 exactly.
    J0 = _exact_dircos_jacobian(0.0, 0.0)
    go0 = (1.0 + 0.0 + 0.0) ** -1.5
    assert np.allclose(J0, go0 * np.eye(2), atol=0.0)

    def rel_gap(ux, uy):
        J = _exact_dircos_jacobian(ux, uy)
        go = (1.0 + ux ** 2 + uy ** 2) ** -1.5
        return np.max(np.abs(J - go * np.eye(2))) / abs(go)

    # Moderate NA (|u| ~ 0.1): the dropped anisotropy/skew is ~1% -- below the
    # FGA moderate-NA validation floor (fidelity 0.997-0.999).
    assert rel_gap(0.1, 0.05) < 0.02
    # Skew high-NA (u = v = 0.5, |p| ~ 0.5): the scalar approximation is
    # materially wrong (tens of percent) -- the documented O(u^2) limit.
    assert rel_gap(0.5, 0.5) > 0.2

    # The limitation is actually documented at the code site.
    import inspect

    from lumenairy.propagators import fga
    src = inspect.getsource(fga._fga_coarse)
    assert "S2-16" in src and "2x2" in src


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
