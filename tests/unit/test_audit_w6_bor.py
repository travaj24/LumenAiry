"""Wave-6 audit fixes, ``bor`` cluster (v5.17.0 deep audit).

Discriminating tests for the code fixes:

- P3-12: ``BORStack`` modal LRU -- identical-profile layers share ONE
  eigensolve per solve, repeated solve()/sweep calls reuse it, the cache is
  bounded, and cached results are byte-identical to the uncached path.
- P3-13: ``_fast_geig``'s QZ fallback now actually fires for NEAR-singular
  ``B`` (an LU pivot-ratio guard; LAPACK ``solve`` raises only on an exact
  zero pivot, so the old ``except LinAlgError`` hook was unreachable), and
  ``_assemble_staggered`` warns when the E_z-elimination operator is
  near-singular (a longitudinal resonance) instead of silently degrading.
- P3-14: staggered ``layer_modes`` exposes the FACE grid (``r_face``,
  ``wq_face``, ``wq_node``) that rows ``[:N]`` of ``W``/``V`` live on, so
  consumers can build the correct two-grid quadrature.

(P3-11 is a docstring-only fix in ``BORStack.solve``; P3-64 strengthens
``tests/unit/test_bor_solve.py`` itself.)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

import lumenairy.elements.bor.bor_stack as bor_stack_mod
from lumenairy.elements.bor.bor_stack import _MODAL_CACHE_SIZE, BORStack
from lumenairy.elements.bor.coupled_radial_eigensolver import (
    _assemble_staggered,
    _fast_geig,
    _fd_grid_staggered,
)
from lumenairy.elements.bor.zcascade import layer_modes


def _rings_stack(N=60, via_profile=False):
    """3 identical ring layers; ``via_profile`` builds them through three
    DISTINCT (but numerically identical) eps_profile callables, which have
    distinct cache fingerprints -> the uncached per-layer path."""
    s = BORStack(4.0, 1, n_substrate=1.4142, n_superstrate=1.4142, N=N)
    period, duty = 0.8, 0.5
    er, eg = complex(2.449) ** 2, complex(1.414) ** 2
    for _ in range(3):
        if via_profile:
            s.add_layer(0.5, eps_profile=lambda r, p=period, d=duty, a=er, b=eg:
                        np.where((r % p) < d * p, a, b).astype(complex))
        else:
            s.add_layer(0.5, rings=(period, duty, 2.449, 1.414))
    s.set_source(wavelength=2 * np.pi / 2.0)
    return s


def test_p12_identical_layers_share_one_eig(monkeypatch):
    """3 identical rings layers + matched super/substrate -> exactly 2
    layer_modes builds (1 half-space + 1 ring profile), and a repeated
    solve() rebuilds NOTHING."""
    calls = []
    orig = bor_stack_mod.layer_modes

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(bor_stack_mod, "layer_modes", counting)
    s = _rings_stack()
    s.solve()
    assert len(calls) == 2          # was 4 pre-fix (sup/sub + 3 mids)
    s.solve()
    assert len(calls) == 2          # full reuse across solve() calls
    s.set_source(k0=2.1)            # new k0 -> new eigs (distinct cache keys)
    s.solve()
    assert len(calls) == 4


def test_p12_cached_solve_byte_identical():
    """The dedup/cached path returns byte-identical R/T to the per-layer
    uncached path (three distinct-fingerprint but numerically identical
    profiles force one build per layer)."""
    res_cached = _rings_stack().solve()
    res_plain = _rings_stack(via_profile=True).solve()
    assert res_cached["R"].tobytes() == res_plain["R"].tobytes()
    assert res_cached["T"].tobytes() == res_plain["T"].tobytes()


def test_p12_cache_is_bounded():
    """More distinct (profile, k0) keys than slots -> the LRU stays at the
    bound (no unbounded growth over a sweep)."""
    s = BORStack(4.0, 1, n_substrate=1.4142, n_superstrate=1.4142, N=8)
    for i in range(_MODAL_CACHE_SIZE + 3):
        s.add_layer(0.1, eps=2.0 + 0.01 * i)
    s.set_source(k0=2.0)
    res = s.solve()
    assert np.all(np.isfinite(res["energy"]))
    assert len(s._modal_cache) == _MODAL_CACHE_SIZE


def test_p13_near_singular_B_routes_to_qz():
    """A NEAR-singular (not exactly singular) B never raises in the folded
    solve, so the old ``except LinAlgError`` fallback was dead code and the
    finite pencil root came out with O(1) error; the LU pivot-ratio guard
    routes it to the QZ, which recovers the root to machine precision."""
    e = 3e-16                                   # 1.0 + e is representable
    K = np.diag([2.0, 3.0]).astype(complex)
    B = np.array([[1.0, 1.0], [1.0, 1.0 + e]], dtype=complex)
    lam_ref = 6.0 / (5.0 + 2 * e)               # exact finite pencil root
    # the old folded path proceeds silently and loses the root entirely
    q2_old = np.linalg.eigvals(np.linalg.solve(B, K))
    assert np.min(np.abs(q2_old - lam_ref)) > 0.1
    q2_new, _ = _fast_geig(K, B)
    assert np.min(np.abs(q2_new - lam_ref)) < 1e-8


def test_p13_well_conditioned_fold_bit_identical():
    """Off the guard, _fast_geig is BIT-identical to the pre-fix folded
    eig(solve(Be, Ke)) path (lu_factor+lu_solve == LAPACK gesv)."""
    rng = np.random.default_rng(7)
    n = 24
    K = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    B = np.eye(n) + 0.1 * rng.standard_normal((n, n))
    B = B.astype(complex)
    d = np.sqrt(np.abs(np.diag(B)))
    d = np.where(d > 0, 1.0 / d, 1.0)
    Ke = (d[:, None] * K) * d[None, :]
    Be = (d[:, None] * B) * d[None, :]
    from scipy.linalg import eig as sla_eig
    q2_old, z_old = sla_eig(np.linalg.solve(Be, Ke))
    q2_new, z_new = _fast_geig(K, B)
    assert q2_new.tobytes() == q2_old.tobytes()
    assert z_new.tobytes() == (d[:, None] * z_old).tobytes()


def test_p13_assemble_warns_at_longitudinal_resonance():
    """At a discrete longitudinal resonance (k0^2*eps == an eigenvalue of
    -Lm) the E_z-elimination operand is near-singular: _assemble_staggered
    now WARNS there (pre-fix: silent ~9-order energy-accuracy loss), and
    stays silent off-resonance."""
    m, Rbig, N, eps = 1, 8.0, 60, 2.0
    r_n, r_f, h, Dn2f, Df2n, An2f, Af2n = _fd_grid_staggered(Rbig, N)
    Lm = (np.diag(1.0 / r_n) @ Df2n @ np.diag(r_f) @ Dn2f
          - np.diag((m / r_n) ** 2))
    lam = np.linalg.eigvals(Lm)
    lam = np.sort(lam.real[lam.real < -1.0])    # resonant k0^2*eps = -lam
    k0res = float(np.sqrt(-lam[len(lam) // 2] / eps))
    prof = lambda r: np.full_like(r, eps, dtype=complex)  # noqa: E731
    with pytest.warns(UserWarning, match="near-singular"):
        _assemble_staggered(m, Rbig, N, prof, k0res)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # off-resonance: no warning
        _assemble_staggered(m, Rbig, N, prof, 1.01 * k0res)


def test_p14_staggered_face_grid_returned():
    """Staggered layer_modes exposes the face grid + per-half quadrature
    weights; the two-grid flux is exactly the unit normalization while the
    legacy single-``wq`` quadrature (the P3-14 trap) is off by O(h)."""
    m, Rbig, N, k0 = 1, 8.0, 60, 2.0
    L = layer_modes(m, Rbig, N,
                    lambda r: np.full_like(r, 2.0, dtype=complex), k0,
                    staggered=True)
    h = Rbig / N
    assert np.array_equal(L["r_face"], (np.arange(N) + 1.0) * h)
    assert np.array_equal(L["wq_face"], (L["r_face"] * h).astype(complex))
    assert np.array_equal(L["wq_node"], L["wq"])          # alias, node half
    assert np.array_equal(L["r"], (np.arange(N) + 0.5) * h)

    qn = L["q"] / k0
    prop = np.where((np.abs(qn.imag) < 5e-5) & (qn.real > 0.05))[0]
    assert prop.size >= 4
    wf, wn = np.real(L["wq_face"]), np.real(L["wq_node"])
    for j in prop[:4]:
        Er, Ephi = L["W"][:N, j], L["W"][N:, j]
        hr, hphi = L["V"][:N, j], L["V"][N:, j]
        F_split = np.real(np.sum(Er * np.conj(hphi) * wf)
                          - np.sum(Ephi * np.conj(hr) * wn))
        F_wq = np.real(np.sum((Er * np.conj(hphi) - Ephi * np.conj(hr))
                              * np.real(L["wq"])))
        assert abs(F_split - 1.0) < 1e-9          # the correct two-grid flux
        assert abs(F_wq - 1.0) > 1e-3             # the single-wq trap is O(h)
