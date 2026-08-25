"""P4 / N3 (2026-07-20): GBD strong-reconvergence frame re-expansion.

A converging input reconverged by a NEGATIVE element to a near real focus sheds
~6-12% of its power at the INPUT decomposition -- the flat-waist beamlet frame
cannot carry the input wavefront curvature, so its coherent sum is incomplete
(the loss is baked into the frame at the decomposition plane, BEFORE any
propagation, and is unrecoverable downstream).  ``reexpand='auto'`` re-decomposes
the input with a CARRIER REFERENCE (remove the smooth congruence, decompose the
compact residual, seed each beamlet's direction / curvature / piston from the
carrier -- the H7 machinery extended with the beamlet-Q curvature seeding plain
Husimi omits), closing the loss to > 0.99 power with windowed r2m within 0.3% of
``traced``.

Oracles (independent of the GBD implementation under test):
* ABCD Gaussian q-trace through the biconcave (the image plane / focus).
* ``apply_real_lens_traced`` (carrier-referenced per-pixel OPL) -- the H6-fixed
  reference model for the diverging/converging-input real-surface class -- for
  the windowed-r2m spatial cross-check.
* Parseval / power conservation on the re-decomposition (no double-count).

Defaults byte-identical: ``reexpand='off'`` (default) is the prior release; the
P4 numbers are opt-in.  The strong-reconvergence loss the DEFAULT still exhibits
is pinned in ``test_hammer_h7_gbd_diverging.py``
(``test_h7_converging_strong_reconvergence_frame_limit``).
"""
from __future__ import annotations

import hashlib
import os
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements.lenses_gbd import apply_real_lens_gbd
from lumenairy.propagators.gbd import (
    decompose_field_to_beamlets,
    frame_completeness,
)

_WL = 1.31e-6
_N_GLASS = 1.5168
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_P4_GLASS': lambda wl: _N_GLASS}
_k = 2 * np.pi / _WL

# Fast config that exhibits the reconvergence loss (naive ~0.87) and its close
# (~1.00): M5 biconcave, converging input R_in = -35 mm -> real image ~108 mm.
_N, _DX, _W_L, _SS = 384, 10e-6, 1.0e-3, 3
_R_IN = -35e-3


def _m5_biconcave():
    return {'wavelength': _WL, 'aperture_diameter': 24e-3,
            'surfaces': [
                {'radius': -51.68e-3, 'thickness': 3e-3, 'glass_before': 'air',
                 'glass_after': '_P4_GLASS', 'semi_diameter': 12e-3},
                {'radius': 51.68e-3, 'thickness': 0.0,
                 'glass_before': '_P4_GLASS', 'glass_after': 'air',
                 'semi_diameter': 12e-3}],
            'thicknesses': [3e-3], 'stop_index': 0}


def _singlet_positive():
    return {'wavelength': _WL, 'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_P4_GLASS', 'semi_diameter': 10e-3},
                {'radius': -51.68e-3, 'thickness': 0.0,
                 'glass_before': '_P4_GLASS', 'glass_after': 'air',
                 'semi_diameter': 10e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


def _conv_input(N, dx, w_L, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    env = np.exp(-r_sq / w_L ** 2)
    ph = (np.ones_like(env) if np.isinf(R_in)
          else np.exp(1j * _k * r_sq / (2.0 * R_in)))
    return (env * ph).astype(np.complex128)


def _abcd_biconcave():
    def refr(R, n1, n2):
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    return (refr(51.68e-3, _N_GLASS, 1.0) @ trans(3e-3)
            @ refr(-51.68e-3, 1.0, _N_GLASS))


def _q_image(R_in, w_L):
    q_inv = 1.0 / R_in - 1j * _WL / (np.pi * w_L ** 2)
    q0 = 1.0 / q_inv
    M = _abcd_biconcave()
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    return float(-q1.real)


def _r2m_windowed(I, dx, r_win=200e-6):
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    j, i = np.unravel_index(np.argmax(I), I.shape)
    R = np.sqrt((X - x[i]) ** 2 + (Y - x[j]) ** 2)
    m = R <= r_win
    return float(np.sqrt((I[m] * R[m] ** 2).sum() / I[m].sum()))


def _ee_at(E_exit, dx, z, N, r_ee=150e-6):
    E = la.angular_spectrum_propagate(E_exit, z, _WL, dx)
    I = np.abs(E) ** 2
    if not np.all(np.isfinite(I)):
        return float('nan')
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    return float(I[r <= r_ee].sum() / I.sum())


def _field_hash(E):
    return hashlib.sha256(
        np.ascontiguousarray(E).tobytes()).hexdigest()[:16]


def _byte_id_message(E_a, E_b, ctx=''):
    """Everything needed to adjudicate a byte-identity miss WITHOUT a re-run.

    S14.6's one unreproduced failure cost three re-runs precisely because the
    assertion carried none of this: how far apart, how many cells, and whether
    the two fields are the same object-shaped answer at all.
    """
    d = np.abs(np.asarray(E_a) - np.asarray(E_b))
    n_diff = int(np.count_nonzero(d))
    peak = float(np.max(np.abs(E_a))) or 1.0
    return (f"fields differ: max|diff|={float(d.max()):.6e} "
            f"({float(d.max()) / peak:.3e} of peak), {n_diff} of {d.size} "
            f"cells differ, hashes {_field_hash(E_a)} vs {_field_hash(E_b)} "
            f"{ctx}")


def _gbd(E, presc, *, dx=_DX, ss=_SS, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(apply_real_lens_gbd(
            E, prescription=presc, wavelength=_WL, dx=dx, sample_step=ss, **kw))


# ---------------------------------------------------------------------------
# The frame-completeness primitive
# ---------------------------------------------------------------------------
def test_frame_completeness_primitive_flat_vs_curved():
    """The published metric: a flat / collimated frame reconstructs its input to
    ~1 (a complete frame); a strongly-curved converging input decomposed by the
    same flat-waist beamlets falls well below 1 (the frame under-spans the
    curvature it cannot carry).  This is the mechanism P4 closes."""
    flat = _conv_input(_N, _DX, _W_L, np.inf)
    curved = _conv_input(_N, _DX, _W_L, _R_IN)
    bf = decompose_field_to_beamlets(flat, _DX, wavelength=_WL, dy=_DX,
                                     waist_factor=float(_SS), sample_step=_SS,
                                     direction_sampling=False)
    bc = decompose_field_to_beamlets(curved, _DX, wavelength=_WL, dy=_DX,
                                     waist_factor=float(_SS), sample_step=_SS,
                                     direction_sampling=True)
    c_flat = frame_completeness(bf, flat, _DX, wavelength=_WL, dy=_DX)
    c_curved = frame_completeness(bc, curved, _DX, wavelength=_WL, dy=_DX)
    assert c_flat > 0.99, f"flat frame incomplete: {c_flat:.4f}"
    assert c_curved < 0.97, (
        f"curved-input frame should be incomplete (< 0.97), got {c_curved:.4f}")


# ---------------------------------------------------------------------------
# Byte-identical defaults
# ---------------------------------------------------------------------------
def test_reexpand_off_is_default_and_byte_identical():
    """The new params default to a no-op: an explicit reexpand='off' (and the
    default with no reexpand kwarg) produce byte-for-byte identical output."""
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    E_default = _gbd(E, _m5_biconcave())
    E_off = _gbd(E, _m5_biconcave(), reexpand='off')
    assert np.array_equal(E_default, E_off), (
        "reexpand='off' diverged from the default "
        f"(max|diff|={np.max(np.abs(E_default - E_off)):.2e})")


def test_reexpand_off_reconvergence_still_lossy():
    """Fail-before anchor: the DEFAULT ('off') frame stays lossy on the strong
    reconvergence (0.80-0.95 power) -- the documented ~0.94 edge P4 closes."""
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    p_in = float(np.sum(np.abs(E) ** 2))
    E_off = _gbd(E, _m5_biconcave(), reexpand='off')
    pc = float(np.sum(np.abs(E_off) ** 2)) / p_in
    assert 0.80 < pc < 0.95, f"default reconvergence power {pc:.4f} off-envelope"


def test_reexpand_collimated_byte_identical():
    """A collimated (flat-wavefront) input is already complete (> threshold), so
    reexpand='auto' does NOT re-decompose -- its output is byte-identical to
    'off' (the re-expansion only adds a completeness measurement there)."""
    xs = (np.arange(256) - 128) * _DX
    X, Y = np.meshgrid(xs, xs)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex128)
    E_auto = _gbd(E0.copy(), _m5_biconcave(), ss=4, reexpand='auto')
    E_off = _gbd(E0.copy(), _m5_biconcave(), ss=4, reexpand='off')
    assert np.array_equal(E_auto, E_off), (
        "reexpand='auto' disturbed a collimated (already-complete) input "
        f"(max|diff|={np.max(np.abs(E_auto - E_off)):.2e})")


def test_reexpand_does_not_fire_on_diverging_positive():
    """A diverging input through a POSITIVE element already conserves power
    (completeness > threshold): reexpand='auto' must NOT fire and its output is
    byte-identical to 'off' -- the re-expansion is surgical to the edge."""
    E = _conv_input(_N, _DX, _W_L, 150e-3)
    diag = {}
    E_auto = _gbd(E, _singlet_positive(), reexpand='auto', diagnostics=diag)
    E_off = _gbd(E, _singlet_positive(), reexpand='off')
    assert diag['reexpanded'] is False, (
        f"re-expansion fired on an already-good case (diagnostics={diag})")
    assert np.array_equal(E_auto, E_off), _byte_id_message(
        E_auto, E_off, f"[diagnostics={diag}]")


# ---------------------------------------------------------------------------
# The headline: reexpand='auto' closes the reconvergence loss
# ---------------------------------------------------------------------------
def test_reexpand_auto_closes_reconvergence_power():
    """HEADLINE (plan N3 acceptance): reexpand='auto' on the strong reconvergence
    reaches power > 0.99 (vs the ~0.87 naive frame at this config), fires the
    carrier-referenced re-decomposition, and stays finite."""
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    p_in = float(np.sum(np.abs(E) ** 2))
    diag = {}
    E_re = _gbd(E, _m5_biconcave(), reexpand='auto', diagnostics=diag)
    pc = float(np.sum(np.abs(E_re) ** 2)) / p_in
    assert diag['reexpanded'] is True, "reexpand='auto' did not fire on the edge"
    assert pc > 0.99, f"reexpanded reconvergence power {pc:.4f} <= 0.99"
    assert np.all(np.isfinite(E_re))


def test_reexpand_auto_focuses_at_abcd_image():
    """The re-expansion preserves the correct LAUNCH: the reexpanded field
    focuses at the ABCD q-trace image, not the collimated focal plane."""
    z_img = _q_image(_R_IN, _W_L)
    assert z_img > 0
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    E_re = _gbd(E, _m5_biconcave(), reexpand='auto')
    zs = np.linspace(0.85 * z_img, 1.15 * z_img, 11)
    ees = [_ee_at(E_re, _DX, float(z), _N) for z in zs]
    z_bf = float(zs[int(np.nanargmax(ees))])
    assert abs(z_bf - z_img) / z_img < 0.06, (z_bf * 1e3, z_img * 1e3)
    assert np.nanmax(ees) > 0.9


def test_reexpand_auto_r2m_within_5pct_of_traced():
    """Plan N3 acceptance: windowed r2m at the ABCD image within 5% of the
    independent carrier-referenced ``traced`` propagator."""
    z_img = _q_image(_R_IN, _W_L)
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    E_re = _gbd(E, _m5_biconcave(), reexpand='auto')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_tr = np.asarray(apply_real_lens_traced(
            E, prescription=_m5_biconcave(), wavelength=_WL, dx=_DX,
            carrier=_R_IN, on_noncollimated='ignore'))
    Ef = la.angular_spectrum_propagate(E_re, z_img, _WL, _DX)
    Eft = la.angular_spectrum_propagate(E_tr, z_img, _WL, _DX)
    r_gbd = _r2m_windowed(np.abs(Ef) ** 2, _DX)
    r_tr = _r2m_windowed(np.abs(Eft) ** 2, _DX)
    assert abs(r_gbd / r_tr - 1.0) < 0.05, (
        f"r2m gbd={r_gbd*1e6:.2f}um traced={r_tr*1e6:.2f}um "
        f"ratio={r_gbd/r_tr:.4f} outside 5%")


# ---------------------------------------------------------------------------
# Parseval / diagnostics / grid-convergence
# ---------------------------------------------------------------------------
def test_reexpand_parseval_no_double_count():
    """Parseval audit on the re-decomposition: the carrier-referenced frame
    reconstructs the input to ~1 (0.99-1.01) -- it neither sheds power (naive
    frame) nor DOUBLE-COUNTS it (over-overlapped frame > 1.01)."""
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    diag = {}
    _gbd(E, _m5_biconcave(), reexpand='auto', diagnostics=diag)
    c_re = diag['frame_completeness_reexpanded']
    c_in = diag['frame_completeness_input']
    assert c_in < 0.97, f"naive input completeness {c_in:.4f} not < 0.97"
    assert 0.99 <= c_re <= 1.01, (
        f"re-decomposition completeness {c_re:.4f} not in [0.99, 1.01] "
        f"(power shed or double-counted)")


def test_frame_completeness_metric_published():
    """The P4 metric is published on the diagnostics dict; a None dict computes
    nothing (default zero-overhead) and the output is unchanged.

    INSTRUMENTED (P4 close-out, 2026-08-24).  This test failed ONCE, in a
    2526-test sweep under five-way concurrent load, and arrived without its
    assertion text because the batch piped pytest through a ``grep`` that kept
    the ``FAILED`` line and dropped the message above it
    (``BUILD_DETERMINISTIC_TRACED_FIT_2026_08_23.md`` S14.6).  Five arms never
    reproduced it.  It has since been bounded at **782 diagnosed/undiagnosed
    pairs (1564 calls), zero mismatches and ONE field hash** -- under a
    deliberate 6-to-8-way process load at ``OMP_NUM_THREADS=8``, at BLAS widths
    1/2/4/8/16, and in four concurrent in-process threads; the one
    BLAS-adjacent step on the path (``_compute_carrier``'s 82092 x 5 fit) held
    one coefficient hash over 40 000 solves under the same load.  See
    ``docs/audits/FIX_P4_TRACED_CLOSEOUT_2026_08_24.md``.

    So every assertion below now carries the number it read and the margin it
    had.  A filter tight enough to be readable is tight enough to throw away
    the evidence, and the cure that survives the filter is putting the evidence
    in the assertion itself.
    """
    # The windowed reconstruct's chunk boundaries -- and so the field's LAST
    # BITS -- are a function of the per-chunk memory budget, on which
    # ``LUMENAIRY_MEM_BUDGET_MB`` is a hard ceiling read at CALL time (measured
    # 2026-08-24: the field hash moves at 1/8/64 MB against the 512 MB
    # default, while ``frame_completeness`` does not move to 12 digits).  It
    # cannot break the pair below -- both calls read the same value -- but a
    # value leaked in by another test or by the runner changes every field bit
    # this test reads, so it is reported rather than assumed.
    budget = os.environ.get('LUMENAIRY_MEM_BUDGET_MB', '(unset)')
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    diag = {}
    E_a = _gbd(E, _m5_biconcave(), reexpand='auto', diagnostics=diag)
    ctx = f"[LUMENAIRY_MEM_BUDGET_MB={budget} diagnostics={diag}]"
    for key in ('frame_completeness', 'reexpanded', 'n_beamlets',
                'frame_completeness_input', 'frame_completeness_reexpanded'):
        # the last two keys exist only when the re-expansion GATE fired
        # (angular spread over the Husimi threshold) and, for the last, when
        # the carrier-referenced frame was accepted -- so a miss here names a
        # decision that moved, not a missing assignment.
        assert key in diag, f"diagnostics missing {key!r} {ctx}"
    assert diag['frame_completeness'] > 0.99, (
        f"frame_completeness {diag['frame_completeness']:.6f} <= 0.99 "
        f"(margin was +9.8e-03 on 2026-08-24) {ctx}")
    assert isinstance(diag['reexpanded'], bool), ctx
    assert diag['n_beamlets'] > 0, ctx
    # None (default) -> no metric, identical output.  This is a CALL-TO-CALL
    # DETERMINISM claim as much as a "diagnostics is read-only" one: the two
    # calls differ in nothing but the dict, and the dict is consumed AFTER the
    # field is built.
    E_b = _gbd(E, _m5_biconcave(), reexpand='auto')
    assert np.array_equal(E_a, E_b), _byte_id_message(E_a, E_b, ctx)
    # ...and the other way round, so the claim is two-sided: an UNDIAGNOSED
    # call followed by a diagnosed one returns the same bits too.
    E_c = _gbd(E, _m5_biconcave(), reexpand='auto')
    E_d = _gbd(E, _m5_biconcave(), reexpand='auto', diagnostics={})
    assert np.array_equal(E_c, E_d), _byte_id_message(E_c, E_d, ctx)
    assert np.array_equal(E_a, E_c), _byte_id_message(E_a, E_c, ctx)


def test_reexpand_grid_convergence():
    """Grid-convergence of the intermediate (entrance-plane) reconstruction: at
    the same physical extent and beamlet spacing, halving dx (N 384->768,
    dx 10->5 um, ss 3->6) keeps completeness > 0.99 and the windowed r2m within
    2%."""
    z_img = _q_image(_R_IN, _W_L)
    r2ms = {}
    comps = {}
    for N, dx, ss in [(384, 10e-6, 3), (768, 5e-6, 6)]:
        E = _conv_input(N, dx, _W_L, _R_IN)
        diag = {}
        E_re = _gbd(E, _m5_biconcave(), dx=dx, ss=ss, reexpand='auto',
                    diagnostics=diag)
        Ef = la.angular_spectrum_propagate(E_re, z_img, _WL, dx)
        r2ms[dx] = _r2m_windowed(np.abs(Ef) ** 2, dx)
        comps[dx] = diag['frame_completeness']
    assert min(comps.values()) > 0.99, comps
    ratio = r2ms[10e-6] / r2ms[5e-6]
    assert abs(ratio - 1.0) < 0.02, (
        f"r2m not grid-converged: {r2ms[10e-6]*1e6:.2f} vs "
        f"{r2ms[5e-6]*1e6:.2f} um (ratio {ratio:.4f})")


# ---------------------------------------------------------------------------
# Carrier vocabulary, normalize, and validation
# ---------------------------------------------------------------------------
def test_reexpand_scalar_carrier_matches_auto():
    """The known scalar conjugate (reexpand_carrier=R_in) and the auto-fit
    congruence both close the loss to > 0.99 and agree to ~1% in power."""
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    p_in = float(np.sum(np.abs(E) ** 2))
    E_auto = _gbd(E, _m5_biconcave(), reexpand='auto', reexpand_carrier='auto')
    E_scal = _gbd(E, _m5_biconcave(), reexpand='auto', reexpand_carrier=_R_IN)
    pc_a = float(np.sum(np.abs(E_auto) ** 2)) / p_in
    pc_s = float(np.sum(np.abs(E_scal) ** 2)) / p_in
    assert pc_a > 0.99 and pc_s > 0.99, (pc_a, pc_s)
    assert abs(pc_a - pc_s) < 0.01, (pc_a, pc_s)


def test_reexpand_with_normalize_power():
    """reexpand='auto' composes with normalize_output='power' (exact total
    power) and the published frame_completeness reflects the RAW (pre-normalize)
    frame -- so it stays > 0.99, not spuriously pinned to 1.0 by the rescale."""
    E = _conv_input(_N, _DX, _W_L, _R_IN)
    p_in = float(np.sum(np.abs(E) ** 2))
    diag = {}
    E_re = _gbd(E, _m5_biconcave(), reexpand='auto', normalize_output='power',
                diagnostics=diag)
    pc = float(np.sum(np.abs(E_re) ** 2)) / p_in
    assert abs(pc - 1.0) < 1e-6, f"normalize_output='power' total {pc:.6f}"
    assert diag['frame_completeness'] > 0.99   # raw frame, pre-normalize


def test_reexpand_invalid_value_raises():
    E = _conv_input(64, _DX, _W_L, _R_IN)
    with pytest.raises(ValueError, match="reexpand must be"):
        apply_real_lens_gbd(E, prescription=_m5_biconcave(), wavelength=_WL,
                            dx=_DX, sample_step=8, reexpand='bogus')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
