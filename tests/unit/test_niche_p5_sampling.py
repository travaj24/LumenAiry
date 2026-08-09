"""P5 / N6b (2026-07-20): content-adaptive momentum sampling for FGA.

``apply_real_lens_fga(..., momentum_sampling='adaptive')`` importance-samples the
``n_p`` momentum nodes by the inverse-CDF of the input field's (beamlet-broadened,
symmetrized) marginal angular power, with the MATCHING per-node midpoint-cell
quadrature weights folded into the beamlet normalization + auto power-normalization.

HONEST FINDING (N6b, pinned here): this is a MEASURED NON-improvement -- the FGA
reconstruction integrand is the field spectrum convolved with the beamlet momentum
width, so it is too broad for importance sampling to beat uniform, which already
converges (see the hammer doc N6b curve).  The knob is shipped opt-in per the plan;
these tests pin its CORRECTNESS and its byte-identical default, not a fidelity win.

ADVERSARIAL SELF-CHECK (the common bug the plan flags): a NON-uniform p-measure
MUST carry the matching per-node quadrature weights, else the reconstruction
mis-normalizes.  ``test_matching_weights_are_a_valid_quadrature_wrong_ones_are_not``
is the teeth (pure quadrature, deterministic): the midpoint-cell weights integrate
a known function with a bounded, n_p-converging error, while forcing the naive
uniform ``dp`` weight on the SAME concentrated nodes over-counts by > 200%.

Defaults byte-identical: ``momentum_sampling='uniform'`` (default) takes the
scalar-``C`` (``wp=None``) path -- provably the prior-release code path; the
adaptive weights reduce to the uniform ``dp`` cell on a uniform grid.

H5 CONTRACTS (must stay green under the new flag): the near-collimated
content-sizing override and the identity/diverging floor path are decided in
``_resolve_sampling`` and are ORTHOGONAL to ``momentum_sampling`` -- re-asserted
here (mirrors ``test_fga_h4_h5.py``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.propagators import fga
from lumenairy.propagators.fga import (
    _adaptive_momentum_nodes,
    _midpoint_cell_weights,
    _swarm_lattice,
    apply_real_lens_fga,
)

WL = 0.633e-6
_K = 2.0 * np.pi / WL

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_P5_GLASS': lambda wl: 1.5168}


def _flat_presc(N, dx):
    return {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': N * dx / 2}],
            'thicknesses': []}


def _diverging(N, dx, sig=16e-6, Rdiv=90e-6):
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r2 = X ** 2 + Y ** 2
    return (np.exp(-r2 / sig ** 2)
            * np.exp(1j * _K * r2 / (2 * Rdiv))).astype(np.complex128)


# ==========================================================================
# FAST -- no propagation: node / weight construction + API contracts
# ==========================================================================
def test_midpoint_weights_reduce_to_dp_on_uniform_grid():
    """The byte-identity guarantee: on a UNIFORM node grid the midpoint-cell
    weights are the constant spacing ``dp`` for EVERY node (endpoints included),
    so ``outer(wv, wv) == dp**2`` -- exactly the scalar rectangle measure the
    uniform path uses."""
    for n_p in (7, 15, 32, 61):
        pv = np.linspace(-0.3, 0.3, n_p)
        dp = pv[1] - pv[0]
        wv = _midpoint_cell_weights(pv)
        assert np.allclose(wv, dp, rtol=0, atol=1e-15)
        assert np.allclose(np.outer(wv, wv), dp * dp, rtol=0, atol=1e-15)


def test_swarm_lattice_uniform_is_scalar_path():
    """``nodes=None`` (uniform, the default) returns ``wp=None`` -> the caller
    uses the scalar ``C`` (prior-release path); passing uniform nodes+weights
    reproduces the same tensor momentum grid and a ``dp**2`` cell area."""
    n_p, p_max = 15, 0.25
    qx, qy, px, py, dp, wp = _swarm_lattice(8, 8, 1e-6, 1e-6, -4e-6, -4e-6, 2,
                                            p_max, n_p, nodes=None)
    assert wp is None
    pv = np.linspace(-p_max, p_max, n_p)
    wv = _midpoint_cell_weights(pv)
    _, _, px2, py2, _, wp2 = _swarm_lattice(8, 8, 1e-6, 1e-6, -4e-6, -4e-6, 2,
                                            p_max, n_p, nodes=(pv, wv))
    assert np.array_equal(px, px2) and np.array_equal(py, py2)
    assert np.allclose(wp2, dp * dp, rtol=0, atol=1e-15)


def test_adaptive_nodes_span_monotone_and_concentrate():
    """Adaptive nodes strictly increase, span the full ``[-p_max, p_max]`` (so
    beamlet-completeness coverage is preserved), carry positive weights, and are
    MORE concentrated toward the spectral peak (p=0) than the uniform grid for a
    diverging beam whose angular spectrum peaks on-axis."""
    N, dx = 256, 0.7e-6
    E = _diverging(N, dx)
    w0 = 5.0 * dx
    p_max = 1.5 * fga._field_angular_content(E, dx, dx, WL)
    n_p = 41
    pv, wv = _adaptive_momentum_nodes(E, dx, dx, WL, w0, p_max, n_p)
    assert pv.size == n_p
    assert np.all(np.diff(pv) > 0)                       # strictly monotone
    assert pv[0] == pytest.approx(-p_max) and pv[-1] == pytest.approx(p_max)
    assert np.all(wv > 0)
    # concentration: the central spacing (near the spectral peak) is finer than
    # the uniform spacing; the tail spacing is coarser.
    dp_uni = 2 * p_max / (n_p - 1)
    mid = n_p // 2
    central = pv[mid + 1] - pv[mid]
    assert central < dp_uni                              # denser at the peak
    assert (pv[-1] - pv[-2]) > dp_uni                    # sparser in the tail


def test_matching_weights_are_a_valid_quadrature_wrong_ones_are_not():
    """ADVERSARIAL SELF-CHECK (the common bug the plan flags): a NON-uniform
    p-measure MUST carry the matching per-node quadrature weights.  The
    midpoint-cell weights integrate a known test function with a BOUNDED,
    ``n_p``-converging error; forcing the naive uniform ``dp`` weight on the SAME
    concentrated nodes GROSSLY over-counts (the dense-near-peak nodes each carry a
    full ``dp`` instead of their small cell) -- >100% error vs <30%.  Pure
    quadrature, deterministic, no FGA/numba."""
    N, dx = 256, 0.7e-6
    E = _diverging(N, dx, sig=40e-6, Rdiv=800e-6)         # concentrated spectrum
    w0 = 5.0 * dx

    def f(p):                                             # smooth test integrand
        return np.exp(-(p / 0.06) ** 2)
    pp = np.linspace(-0.45, 0.45, 20001)
    true_int = float(np.trapezoid(f(pp), pp))
    for p_max, n_p in [(0.30, 15), (0.30, 31), (0.45, 31)]:
        pv, wv = _adaptive_momentum_nodes(E, dx, dx, WL, w0, p_max, n_p)
        assert (np.max(np.diff(pv)) / np.min(np.diff(pv))) > 3.0   # concentrated
        err_correct = abs(np.sum(wv * f(pv)) - true_int) / true_int
        dp_nom = 2 * p_max / (n_p - 1)
        err_wrong = abs(np.sum(dp_nom * f(pv)) - true_int) / true_int
        assert err_correct < 0.30                         # matching = valid quad
        assert err_wrong > 1.0                            # naive uniform = broken
        assert err_wrong > 5 * err_correct                # unambiguous separation
    # and the matching quadrature converges as n_p grows (a valid rule)
    e15 = abs(np.sum(_adaptive_momentum_nodes(E, dx, dx, WL, w0, 0.30, 15)[1]
                     * f(_adaptive_momentum_nodes(E, dx, dx, WL, w0, 0.30, 15)[0]))
              - true_int) / true_int
    e61 = abs(np.sum(_adaptive_momentum_nodes(E, dx, dx, WL, w0, 0.30, 61)[1]
                     * f(_adaptive_momentum_nodes(E, dx, dx, WL, w0, 0.30, 61)[0]))
              - true_int) / true_int
    assert e61 < e15


def test_adaptive_flat_spectrum_stays_near_uniform():
    """A field whose angular spectrum is ~flat over the swarm range gives nodes
    close to uniform (nothing to concentrate)."""
    N, dx = 128, 0.5e-6
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    # white-ish phase noise -> broad flat spectrum
    rng = np.random.default_rng(0)
    E = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N))).astype(np.complex128)
    E *= np.exp(-(X ** 2 + Y ** 2) / (20e-6) ** 2)
    p_max, n_p = 0.3, 31
    pv, _ = _adaptive_momentum_nodes(E, dx, dx, WL, 5.0 * dx, p_max, n_p)
    uni = np.linspace(-p_max, p_max, n_p)
    assert np.max(np.abs(pv - uni)) < 0.25 * p_max       # stays near uniform


def test_invalid_momentum_sampling_raises():
    N, dx = 32, 1e-6
    E = np.ones((N, N), np.complex128)
    with pytest.raises(ValueError, match="momentum_sampling"):
        apply_real_lens_fga(E, prescription=_flat_presc(N, dx), wavelength=WL,
                            dx=dx, momentum_sampling='bogus')


# ==========================================================================
# FAST -- H5 content-override contracts stay green + orthogonal to the flag
# ==========================================================================
def test_h5_near_collimated_override_unchanged():
    """H5: a near-collimated input still trips the content-sizing override in
    ``_resolve_sampling`` (independent of ``momentum_sampling``)."""
    N, dx = 512, 12e-6
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (5e-3) ** 2).astype(np.complex128)
    w0 = 5.0 * dx
    presc = _flat_presc(N, dx)
    info = {}
    with pytest.warns(RuntimeWarning, match="near-collimated"):
        p_max, n_p = fga._resolve_sampling(E, dx, dx, WL, w0, presc, None, None,
                                           info=info)
    assert info['content_override'] is True
    content = fga._field_angular_content(E, dx, dx, WL)
    assert p_max == pytest.approx(fga._PMAX_CONTENT_TAIL * content, rel=1e-6)


def test_h5_diverging_keeps_floor_path():
    """H5: a diverging beam (content within an order of magnitude of the floor)
    keeps the floor path -- no override -- so the S2-2 completeness contract is
    untouched, regardless of the sampling flag."""
    N, dx = 256, 0.633e-6
    E = _diverging(N, dx, sig=28e-6, Rdiv=0.6e-3)
    w0 = 5.0 * dx
    info = {}
    fga._resolve_sampling(E, dx, dx, WL, w0, _flat_presc(N, dx), None, None,
                          info=info)
    assert info['content_override'] is False


# ==========================================================================
# NUMBA -- small propagations: weight correctness + byte-identical default
# ==========================================================================
def _wide_cone(N, dx):
    """A concentrated-spectrum beam (narrow divergence) sampled in a WIDE
    momentum cone -- the regime where adaptive sampling helps (p_max >> content)."""
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r2 = X ** 2 + Y ** 2
    return (np.exp(-r2 / (40e-6) ** 2)
            * np.exp(1j * _K * r2 / (2 * 800e-6))).astype(np.complex128)


@pytest.mark.slow
def test_uniform_flag_is_byte_identical_to_default():
    """The default (no ``momentum_sampling``) equals an explicit
    ``momentum_sampling='uniform'`` bit-for-bit -- the default IS the uniform
    scalar path (the adaptive plumbing does not perturb it)."""
    pytest.importorskip("numba")
    N, dx = 192, 0.7e-6
    E = _diverging(N, dx)
    presc = _flat_presc(N, dx)
    kw = dict(prescription=presc, wavelength=WL, dx=dx,
              output_plane_distance=50e-6, p_max=0.35, n_p=21, dq_step=2,
              mem_budget_mb=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = apply_real_lens_fga(E, **kw)
        b = apply_real_lens_fga(E, momentum_sampling='uniform', **kw)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_adaptive_is_power_normalized_and_a_valid_neutral_alternative():
    """The adaptive output is AUTO power-normalized to the input power (its
    non-uniform quadrature does not preserve the calibrated uniform scale), and
    -- the HONEST measured result (N6b) -- it is a VALID alternative that neither
    beats nor badly regresses uniform on a smooth diverging beam: the reconstruction
    integrand is the field spectrum CONVOLVED with the beamlet momentum width, so it
    is broad enough that uniform is already near-optimal (concentrating by the raw
    content under-samples the completeness skirt).  Adaptive stays within a few % of
    uniform's fidelity, not a win."""
    pytest.importorskip("numba")
    from lumenairy.propagators.asm import angular_spectrum_propagate
    N, dx = 256, 0.7e-6
    E = _wide_cone(N, dx)
    presc = _flat_presc(N, dx)
    z = 200e-6
    gt = angular_spectrum_propagate(E, z, WL, dx)
    pin = float(np.sum(np.abs(E) ** 2))

    def _fid(out):
        gg = gt.ravel()
        oo = np.asarray(out).ravel()
        return abs(np.vdot(oo, gg)) / (np.linalg.norm(oo) * np.linalg.norm(gg))

    kw = dict(prescription=presc, wavelength=WL, dx=dx, output_plane_distance=z,
              p_max=0.30, n_p=15, dq_step=2, mem_budget_mb=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uni = apply_real_lens_fga(E, momentum_sampling='uniform', **kw)
        ada = apply_real_lens_fga(E, momentum_sampling='adaptive', **kw)
    # adaptive auto-normalizes power to the input (do_norm on the adaptive path)
    assert abs(float(np.sum(np.abs(ada) ** 2)) / pin - 1.0) < 1e-6
    # neutral: within ~3% of uniform's fidelity (measured ~0.981 vs 0.989)
    assert _fid(ada) > _fid(uni) - 0.03


@pytest.mark.slow
def test_adaptive_agrees_with_uniform_at_high_np():
    """Adaptive is UNBIASED: at a converged ``n_p`` it agrees with uniform to the
    FGA floor (it redistributes nodes, not the answer).  Both power-normalized."""
    pytest.importorskip("numba")
    N, dx = 192, 0.7e-6
    E = _diverging(N, dx)
    presc = _flat_presc(N, dx)
    p_max = 1.5 * fga._field_angular_content(E, dx, dx, WL)
    kw = dict(prescription=presc, wavelength=WL, dx=dx,
              output_plane_distance=50e-6, p_max=p_max, n_p=61, dq_step=2,
              mem_budget_mb=2000, normalize_output='power')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ua = apply_real_lens_fga(E, momentum_sampling='uniform', **kw)
        aa = apply_real_lens_fga(E, momentum_sampling='adaptive', **kw)
    rel = np.linalg.norm(aa - ua) / np.linalg.norm(ua)
    assert rel < 0.03                                    # unbiased at high n_p
