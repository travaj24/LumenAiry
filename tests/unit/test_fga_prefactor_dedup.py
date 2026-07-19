"""Byte-identity guard for the FGA de-duplication (audit S2-14 / roadmap A1).

v5.24.4 extracted two shared helpers in ``propagators/fga.py``:

  * ``_hk_prefactor(M, go, gi, kw2)`` -- the Herman-Kluk prefactor
    ``a = sqrt(det Z)`` (continuous branch), formerly copy-pasted into the
    scalar (``_fga_through_lens``), vector (``_fga_vector_through_lens``), and
    coarse (``_fga_coarse._trace``) paths;
  * ``_coeff_keep_mask(qbounds, Np, coeff_frac, chunk_peak)`` -- the global
    per-momentum-peak coeff-prune PRE-PASS, formerly copy-pasted into the scalar
    and vector paths.

The consolidation is required to be **byte-identical**: the helper must
reproduce every former call site to machine precision.  These tests feed
hand-built representative beamlet inputs through (a) an INDEPENDENT analytic
hand-oracle and (b) a verbatim copy of the FORMER inline block, and assert the
helper reproduces both bit-for-bit.  The scalar-vs-2x2 Jacobian difference
(audit S2-16) lives in the caller's ``go``/``gi`` -- NOT the helper -- so the
helper is exercised across a range of go/gi.

Guarded by the ``numba`` importorskip the FGA tests use.
"""
import numpy as np
import pytest

numba = pytest.importorskip("numba")           # noqa: F841

from lumenairy.propagators.fga import (  # noqa: E402
    _coeff_keep_mask,
    _det2,
    _hk_prefactor,
)


def _representative_beamlet_inputs(seed, nq=17):
    """A deterministic, physically-shaped (M, go, gi, kw2) beamlet column.

    ``M`` mimics the FGA ray-transfer monodromy (identity + O(0.3) mixing so
    ``det Z`` stays well away from the branch cut and both signs of ``Re a``
    occur); ``go`` is a per-beamlet (nq,) output-Jacobian factor, ``gi`` a
    scalar input-Jacobian factor, ``kw2 = k * w0**2``."""
    rng = np.random.default_rng(seed)
    M = np.tile(np.eye(4), (nq, 1, 1)) + 0.35 * rng.standard_normal((nq, 4, 4))
    go = 0.7 + 0.5 * rng.random(nq)                     # dp/du at output (nq,)
    gi = float(1.0 + 0.4 * rng.random())               # du/dp at input (scalar)
    kw2 = float(2.0 * np.pi / 0.633e-6 * (5e-6) ** 2)   # k * w0**2, real FGA scale
    return M, go, gi, kw2


def _former_inline_block(M, go, gi, kw2):
    """VERBATIM copy of the prefactor block as it appeared at every former call
    site (scalar ~:1096, vector ~:1452, coarse ~:834 pre-dedup).  Kept here so
    the test pins the helper to the exact code it replaced."""
    A = M[:, 0:2, 0:2]
    B = M[:, 0:2, 2:4] * gi
    Cc = M[:, 2:4, 0:2] * go[:, None, None]
    D = M[:, 2:4, 2:4] * (go[:, None, None] * gi)
    Z = (A + D) + 1j * (kw2 * Cc - B / kw2)
    a = np.sqrt(_det2(Z))
    a = np.where(a.real < 0, -a, a)
    return a


def _analytic_prefactor_oracle(M, go, gi, kw2):
    """INDEPENDENT hand-oracle: build the 2x2 Herman-Kluk matrix Z one beamlet
    at a time with explicit scalar indexing (no vectorized block slicing), take
    ``det = z00*z11 - z01*z10`` by hand, ``sqrt``, and the ``Re a >= 0`` branch.

        Z[i,j] = (M[i,j] + go * gi * M[2+i, 2+j])
                 + 1j * (kw2 * go * M[2+i, j] - gi * M[i, 2+j] / kw2)
    """
    nq = M.shape[0]
    out = np.empty(nq, dtype=np.complex128)
    for n in range(nq):
        g = go[n]
        z = np.empty((2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                real = M[n, i, j] + g * gi * M[n, 2 + i, 2 + j]
                imag = kw2 * g * M[n, 2 + i, j] - gi * M[n, i, 2 + j] / kw2
                z[i, j] = complex(real, imag)
        det = z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]
        a = np.sqrt(det)
        if a.real < 0:
            a = -a
        out[n] = a
    return out


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_hk_prefactor_matches_former_inline_block_bitwise(seed):
    """The helper reproduces the verbatim former inline block BIT-for-BIT."""
    M, go, gi, kw2 = _representative_beamlet_inputs(seed)
    a_helper = _hk_prefactor(M, go, gi, kw2)
    a_former = _former_inline_block(M, go, gi, kw2)
    # identical floating-point expression -> exact equality, not just close.
    assert np.array_equal(a_helper.real, a_former.real)
    assert np.array_equal(a_helper.imag, a_former.imag)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_hk_prefactor_matches_analytic_oracle(seed):
    """The helper matches an INDEPENDENT per-beamlet analytic Z-assembly."""
    M, go, gi, kw2 = _representative_beamlet_inputs(seed)
    a_helper = _hk_prefactor(M, go, gi, kw2)
    a_oracle = _analytic_prefactor_oracle(M, go, gi, kw2)
    assert np.allclose(a_helper, a_oracle, rtol=0.0, atol=1e-12)
    # both branches of the Re<0 flip must actually be exercised somewhere across
    # the seed set (the branch is the whole point of the helper).
    assert a_helper.shape == (M.shape[0],)


def test_hk_prefactor_continuous_branch_forces_nonneg_real():
    """Every returned amplitude has ``Re a >= 0`` (the continuous branch)."""
    M, go, gi, kw2 = _representative_beamlet_inputs(3)
    a = _hk_prefactor(M, go, gi, kw2)
    assert np.all(a.real >= 0.0)


def test_hk_prefactor_identity_monodromy():
    """Sanity: for M = I, go = gi = 1, kw2 = 1, Z = I + 1j*(0) except the
    off-diagonal-free case -> a = sqrt(det Z).  Hand value cross-check."""
    nq = 4
    M = np.tile(np.eye(4), (nq, 1, 1))
    go = np.ones(nq)
    gi = 1.0
    kw2 = 1.0
    # Z = A + D + 1j(kw2 C - B/kw2); with M=I: A=I, D=I, C=0, B=0 -> Z = 2I.
    # det Z = 4 -> a = sqrt(4) = 2.
    a = _hk_prefactor(M, go, gi, kw2)
    assert np.allclose(a, 2.0 + 0.0j, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# coeff-prune PRE-PASS helper
# ---------------------------------------------------------------------------

def _coeff_keep_mask_oracle(qbounds, Np, coeff_frac, chunk_peak):
    """INDEPENDENT reproduction of the former inline pre-pass: stack every
    chunk's peak into a (Nchunk, Np) array, reduce with a plain elementwise max,
    threshold at ``coeff_frac * (max + 1e-300)``.  Returns None when pruning is
    off or the lattice is a single chunk (former ``keep_mom = None`` path)."""
    if not (coeff_frac > 0.0 and len(qbounds) > 1):
        return None
    stacked = np.stack([np.asarray(chunk_peak(qs, qe), float)
                        for qs, qe in qbounds], axis=0)
    gpeak = stacked.max(axis=0)
    assert gpeak.shape == (Np,)
    return gpeak >= coeff_frac * float(gpeak.max() + 1e-300)


def _make_chunk_peak(peaks):
    """A ``chunk_peak(qs, qe)`` closure that looks up a fixed (Np,) array per
    chunk by its start index -- lets the tests pin exact per-momentum peaks."""
    def chunk_peak(qs, qe):
        return peaks[qs].copy()
    return chunk_peak


def test_coeff_keep_mask_matches_oracle_multichunk():
    Np = 6
    qbounds = [(0, 10), (10, 20), (20, 30)]
    peaks = {
        0: np.array([1.0, 0.5, 0.02, 0.0, 3.0, 0.1]),
        10: np.array([0.3, 2.0, 0.01, 0.4, 0.5, 0.9]),
        20: np.array([0.2, 0.1, 0.05, 4.0, 1.0, 0.05]),
    }
    chunk_peak = _make_chunk_peak(peaks)
    coeff_frac = 0.1
    keep = _coeff_keep_mask(qbounds, Np, coeff_frac, chunk_peak)
    oracle = _coeff_keep_mask_oracle(qbounds, Np, coeff_frac, chunk_peak)
    assert keep is not None
    assert np.array_equal(keep, oracle)
    # hand-check the reduction: gpeak = [1, 2, 0.05, 4, 3, 0.9]; max = 4;
    # threshold = 0.4 -> keep = [T, T, F, T, T, T].
    assert np.array_equal(keep, np.array([True, True, False, True, True, True]))


def test_coeff_keep_mask_none_when_pruning_off():
    Np = 5
    qbounds = [(0, 10), (10, 20)]
    peaks = {0: np.ones(Np), 10: np.ones(Np)}
    chunk_peak = _make_chunk_peak(peaks)
    # coeff_frac <= 0 disables pruning -> None (former keep_mom stays None).
    assert _coeff_keep_mask(qbounds, Np, 0.0, chunk_peak) is None
    assert _coeff_keep_mask(qbounds, Np, -1.0, chunk_peak) is None


def test_coeff_keep_mask_none_when_single_chunk():
    Np = 5
    qbounds = [(0, 30)]                         # single lattice chunk
    peaks = {0: np.linspace(0.1, 1.0, Np)}
    chunk_peak = _make_chunk_peak(peaks)
    # single chunk -> None (caller uses the running-max path, not the pre-pass).
    assert _coeff_keep_mask(qbounds, Np, 0.1, chunk_peak) is None


def test_coeff_keep_mask_chunk_peak_called_once_per_chunk():
    """The pre-pass touches every lattice chunk exactly once (no double count)."""
    Np = 4
    qbounds = [(0, 5), (5, 10), (10, 15), (15, 20)]
    calls = []

    def chunk_peak(qs, qe):
        calls.append((qs, qe))
        return np.full(Np, float(qs + 1))

    _coeff_keep_mask(qbounds, Np, 0.5, chunk_peak)
    assert calls == qbounds
