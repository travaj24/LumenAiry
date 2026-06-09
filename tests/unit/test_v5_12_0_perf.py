"""Stage 3a performance regression tests (2026-06-08 PMM/RCWA audit, section 3).

Both optimizations are BYTE-EXACT by construction; these tests pin that contract
so a future refactor can't silently perturb the numerics:

* P1 -- _redheffer_star skips the two 2N x 2N inverses when a propagation
  S-matrix (S11 = S22 = 0) is starred, because (I - 0) = I and inv(I) == I
  byte-for-byte.  The result must equal the explicit-inverse formula exactly.
* P4 -- _binary_grating_convolutions skips building the inverse-rule matrix
  (an O(n_orders^3) inverse) when use_li=False; the Laurent EPS must be
  byte-identical and EPS_II must be None.

See docs/audits/PMM_RCWA_AUDIT_2026_06_08.md section 3 (P1, P4).
"""
import numpy as np

from lumenairy.elements.rcwa import (
    _binary_grating_convolutions,
    _propagation_smatrix,
    _redheffer_star,
    rcwa_efficiency_1d,
)


def _ref_star(SA, SB):
    """Reference Redheffer star that ALWAYS inverts (the pre-P1 code path)."""
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    eye = np.eye(n, dtype=complex)
    D = np.linalg.inv(eye - B11 @ A22)
    F = np.linalg.inv(eye - A22 @ B11)
    return (A11 + A12 @ D @ B11 @ A21,
            A12 @ D @ B12,
            B21 @ F @ A21,
            B22 + B21 @ F @ A22 @ B12)


def test_p1_redheffer_skip_byte_identical_to_inverse_path():
    """A propagation S-matrix (zero S11/S22) starred onto an accumulated S must
    give byte-identical blocks whether the inverses are skipped (literal I) or
    taken (inv(I) == I)."""
    rng = np.random.default_rng(0)
    n = 12
    SA = tuple((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
               for _ in range(4))
    lam = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    SB = _propagation_smatrix(lam, 0.37)          # (Z, X, X, Z): S11 = S22 = 0
    assert not SB[0].any() and not SB[3].any()    # precondition: zero blocks
    got = _redheffer_star(SA, SB)                 # P1 skip fires here
    ref = _ref_star(SA, SB)
    for g, r in zip(got, ref):
        assert np.array_equal(g, r)


def test_p1_skip_also_fires_when_SA_is_propagation():
    """The other order -- propagation as the FIRST operand (zero A22)."""
    rng = np.random.default_rng(1)
    n = 9
    lam = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    SA = _propagation_smatrix(lam, 0.21)          # zero A22
    SB = tuple((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
               for _ in range(4))
    got = _redheffer_star(SA, SB)
    ref = _ref_star(SA, SB)
    for g, r in zip(got, ref):
        assert np.array_equal(g, r)


def test_p4_use_li_false_skips_inverse_rule_byte_identical():
    """use_li=False returns the same Laurent EPS byte-for-byte and EPS_II=None."""
    EPS_full, II_full = _binary_grating_convolutions(1.5, 1.0, 0.5, 20,
                                                     use_li=True)
    EPS_skip, II_skip = _binary_grating_convolutions(1.5, 1.0, 0.5, 20,
                                                     use_li=False)
    assert II_full is not None
    assert II_skip is None
    assert np.array_equal(EPS_full, EPS_skip)


def test_p4_default_use_li_true_unchanged():
    """The default (use_li omitted) still builds both -- back-compat for callers
    like the B2 test that unpack a 2-tuple of matrices."""
    EPS, II = _binary_grating_convolutions(1.5, 1.0, 0.5, 30)
    assert EPS.shape == (61, 61) and II.shape == (61, 61)


def test_p4_use_li_false_te_end_to_end_noop():
    """Audit F4: end-to-end validation of the use_li=False (Laurent) path where
    P4 skips EPS_II.  The wall-normal inverse-rule operator does NOT enter the TE
    problem, so formulation='laurent' (use_li=False, EPS_II skipped) is
    BYTE-IDENTICAL to formulation='li' for TE -- proving the skip changed nothing
    on the common path -- and conserves energy.  (For TM the two rules differ only
    in CONVERGENCE RATE; both reach the same answer, so laurent+tm is a valid, if
    slower, choice, NOT a guarded error -- hence no precondition raise.)"""
    args = dict(period=0.9e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.45e-6, duty_cycle=0.5,
                wavelength=0.633e-6, polarization="te", n_orders=31)
    o_l, R_l, T_l = rcwa_efficiency_1d(**args, formulation="laurent")
    o_i, R_i, T_i = rcwa_efficiency_1d(**args, formulation="li")
    assert np.array_equal(np.asarray(R_l), np.asarray(R_i))
    assert np.array_equal(np.asarray(T_l), np.asarray(T_i))
    assert abs(float(np.sum(R_l) + np.sum(T_l)) - 1.0) < 1e-6
