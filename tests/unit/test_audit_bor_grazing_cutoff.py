"""AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13 regression gates.

Commit fca4665's dimensionless propagating-mode classifier chose the
real-axis cutoff ``q/k0 > 0.05`` (an ANGULAR cutoff: theta < 88 deg in
n=1.41) to preserve bit-identity with the old absolute threshold at the
validated k0=2.0 scale.  That cutoff silently dropped genuinely propagating
near-grazing orders, biasing per-order R/T low and leaking energy (2.28e-2 on
the lossless ring-grating reproducer here; the dropped mode carried exactly
that power).  Fixed by flooring the real-axis test at the q ~ 0 degenerate
point only (``q/k0 > 1e-6``) in all THREE classifier twins
(``bor_stack.solve``'s ``prop()``, ``bor_solve._physical_propagating``,
``_jax_bor._mask``).

The floor is compatible with the flux normalizer's field-norm fallback
(``|P| <= 1e-10 * fnrm``): the modal flux ratio scales as ``P/fnrm = q/k0``
for the limiting polarization family (verified empirically), so every kept
mode sits >= 4 decades above the fallback -- kept implies flux-normalized and
``|S|^2`` stays a true power fraction.

Gates (audit section 5): the reproducer at three unit scales; the
near-grazing band is populated; the fundamental-mode R pin (the lossless-trap
guard -- a fix that "closes" energy by renormalizing instead of restoring the
missing order fails it); JAX twin parity; the ``bor_solve`` twin.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.bor import BORStack

pytestmark = pytest.mark.slow      # dense modal eigensolves (N=256 half-spaces)

_LAM = 1.0e-6      # 1 um design wavelength (physical scale)


def _reproducer(scale, ring_index=2.45 + 0j, N=256):
    """The audit's lossless concentric ring grating between index-matched
    half-spaces, m=1, at unit scale ``scale`` (1 = metres)."""
    k0 = 2.0 * np.pi / (_LAM * scale)
    s = BORStack(Rbig=48e-6 * scale, m=1, N=N,
                 n_superstrate=1.41 + 0j, n_substrate=1.41 + 0j)
    s.add_layer(0.5e-6 * scale, rings=(3.0e-6 * scale, 0.5, ring_index,
                                       1.41 + 0j))
    s.set_source(k0=k0)
    return s.solve(), k0


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e9], ids=["m", "um", "nm"])
def test_bor_grazing_reproducer_closes_at_all_scales(scale):
    """Gate 1: energy closure to 1e-9 with the full 319-mode incident set at
    m / um / nm unit scales.  Pre-fix: 318 modes, max|R+T-1| = 2.28e-2."""
    res, _k0 = _reproducer(scale)
    e = np.asarray(res["energy"], float)
    assert e.size == 319, f"incident-mode count {e.size} != 319"
    assert float(np.max(np.abs(e - 1.0))) < 1e-9


def test_bor_grazing_band_is_kept():
    """Gate 2: the near-grazing band the bug silenced (q/k0 in (1e-3, 0.05)
    with essentially-zero imag) is populated in the kept incident set -- the
    reproducer's mode comb has a clean propagating mode at q/k0 = 0.0493."""
    res, k0 = _reproducer(1e6)
    qn = np.asarray(res["q"], float) / k0
    band = (qn > 1e-3) & (qn < 0.05)
    assert np.any(band), "no kept incident mode in the near-grazing band"


def test_bor_fundamental_mode_reflectance_pin():
    """Gate 3 (the lossless-trap guard): pin the fundamental (near-axis)
    incident mode's R = 0.146135 -- the converged value re-derived in the
    audit from the SHIPPED S-matrix with the correct mode sets.  A fix that
    closes energy by renormalizing rather than by restoring the dropped
    order fails this (the shipped-bug value was 0.145113)."""
    res, _k0 = _reproducer(1e6)
    q = np.asarray(res["q"], float)
    R = np.asarray(res["R"], float)
    j = int(np.argmax(q))              # near-axis fundamental = largest q
    assert abs(R[j] - 0.146135) < 1e-4, f"fundamental R = {R[j]:.6f}"
    assert abs(float(np.asarray(res["energy"])[j]) - 1.0) < 1e-9


def test_bor_jax_twin_parity_on_reproducer():
    """Gate 4: the differentiable twin's mask carries the same constant --
    total reflected/transmitted power must match the NumPy path (its full-2N
    masked arrays sum over the same propagating sets)."""
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    res_np, _ = _reproducer(1e6)
    res_jx, _ = _reproducer(1e6, ring_index=jnp.asarray(2.45 + 0j))
    r_np = float(np.sum(np.asarray(res_np["R"])))
    t_np = float(np.sum(np.asarray(res_np["T"])))
    r_jx = float(np.sum(np.asarray(res_jx["R"])))
    t_jx = float(np.sum(np.asarray(res_jx["T"])))
    assert abs(r_np - r_jx) < 1e-8 * max(1.0, abs(r_np))
    assert abs(t_np - t_jx) < 1e-8 * max(1.0, abs(t_np))


def test_bor_solve_twin_classifier_keeps_grazing_band():
    """Gate 5: ``bor_solve._physical_propagating`` carries the same fixed
    constant.  Direct classifier unit test (synthetic mode table): the
    near-grazing band the bug silenced (q/k0 in (1e-6, 0.05), clean imag,
    low reldiv) must be KEPT; the q ~ 0 degenerate point, lossy modes, and
    high-reldiv (spurious) modes must still be rejected.

    (Historical note: this gate was originally classifier-only because the
    then-NODAL ``build_layer`` basis had a separate, PRE-EXISTING blow-up on
    large cells -- max|R+T-1| ~ 1e25..1e32 for Rbig >= ~12 lambda, IDENTICAL
    under the old and new classifier constants (A/B monkeypatch), traced to
    zero-flux spurious modes orienting by the sign of noise and making the
    interface transmission block singular.  ``build_layer`` now defaults to
    the spurious-free staggered basis, and gate 6 runs the cascade-level twin
    end-to-end.)"""
    from lumenairy.elements.bor.bor_solve import _physical_propagating

    k0 = 7.3            # arbitrary scale; classifier is dimensionless in q/k0
    qn = np.array([
        0.0493,          # the audit's dropped near-grazing mode -> KEEP
        2e-3,            # deep near-grazing band -> KEEP
        5e-7,            # below the q~0 degenerate floor -> reject
        0.5 + 1e-3j,     # lossy/evanescent (imag leg) -> reject
        0.8,             # ordinary propagating -> KEEP
        0.6,             # spurious (high reldiv) -> reject
    ])
    reldiv = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.9])
    L = {"q": qn * k0, "reldiv": reldiv}
    keep = _physical_propagating(L, k0)
    assert list(keep) == [True, True, False, False, True, False]


def test_bor_solve_twin_cascade_closes_on_reproducer():
    """Gate 6: the ``build_layer``/``solve`` cascade twin end-to-end on the
    audit reproducer (um scale).  On the staggered default basis this must
    reproduce the production ``BORStack`` numbers exactly: 319 incident
    modes, machine-precision closure, and the gate-3 fundamental-mode R pin
    (probe-verified per-mode R parity with ``BORStack`` is 0.0 -- same basis,
    same cascade, agreeing classifier sets).  Pre-fix (nodal basis) this
    configuration returned max|R+T-1| ~ 4e32."""
    from lumenairy.elements.bor.bor_solve import build_layer, solve

    scale = 1e6
    k0 = 2.0 * np.pi / (_LAM * scale)
    Rbig, N = 48e-6 * scale, 256
    period, duty = 3.0e-6 * scale, 0.5
    n_hi, n_lo = 2.45 + 0j, 1.41 + 0j

    def eps_bg(r):
        return np.full_like(r, n_lo ** 2, dtype=complex)

    def eps_rings(r):
        e = np.full_like(r, n_lo ** 2, dtype=complex)
        e[(r % period) < duty * period] = n_hi ** 2
        return e

    Lh = build_layer(1, Rbig, N, eps_bg, k0)
    Lg = build_layer(1, Rbig, N, eps_rings, k0, thickness=0.5e-6 * scale)
    res = solve([Lh, Lg, Lh], k0)
    e = np.asarray(res["energy"], float)
    assert e.size == 319, f"incident-mode count {e.size} != 319"
    assert float(np.max(np.abs(e - 1.0))) < 1e-9
    jf = int(np.argmax(np.real(res["q_inc"])))     # near-axis fundamental
    assert abs(res["R"][jf] - 0.146135) < 1e-4, f"fundamental R = {res['R'][jf]:.6f}"
