"""The RCWA modal branch-cut defect, and its fix.

WHAT BROKE.  ``tests/unit/test_v5_14_2_backlog_batch.py::
test_jones_2d_even_sector_matches_full`` read ``dR = 1.625072e-02`` on WSL
(py3.12 / OpenBLAS SkylakeX) and ``2.812111e-10`` on Windows (py3.14 /
OpenBLAS Haswell) from byte-identical code, with a LAPACK ``DLASCL`` complaint
alongside.  Both readings are the same defect seen from different reduction
orders: pinning ``OPENBLAS_NUM_THREADS`` moved the SAME build's reading from
7.7e-14 to 2.8e-10 (Windows) and from 1.9e-13 to 1.6e-02 (WSL).

WHAT THE DEFECT IS.  ``_core._sqrt_decay`` maps a layer's eigenvalue
``lam^2`` to its modal decay constant ``lam``.  A PROPAGATING mode of a
LOSSLESS layer has ``lam^2`` exactly real NEGATIVE -- exactly on the principal
square root's branch cut, where the two roots ``+i|kz|`` (outgoing, the branch
the region modes are built on) and ``-i|kz|`` (incoming) are separated only by
the SIGN of ``Im(lam^2)``.  For a value that came out of ``eig`` that sign is
the eigensolver's backward error, ~``eps_mach * ||M||``.  The pin meant to
force the outgoing root tested ``Re(sqrt(lam^2)) == 0`` EXACTLY, which an
``eig`` output never satisfies (its real part is ~1e-16, not 0), so the pin
never fired for a structured layer and the root was decided by a last bit.

A forward layer mode carrying the INCOMING root is, at an interface with a
region of the SAME permittivity, exactly that region's BACKWARD mode -- so the
interface mode-match ``a + b``, whose explicit inverse IS ``S12``, is exactly
singular.  Measured on the failing fixture (whose layer background ``2.25``
equals ``n_substrate^2 = 1.5^2`` exactly): ``cond(a + b) = 1.97e15`` at the
layer->substrate interface against ``1.5e4`` at the layer->superstrate one,
and a lossless-closure defect of ``3.2e-03``.  After the fix the same numbers
are ``5.6e3`` and ``6.0e-15``.

Evidence: ``docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md``; probes in
``validation/probe_fix_rcwa_even_sector_wsl/`` (both builds, both arms).

EVERY numeric bar below is a DECISION with measured room on both sides, per
``docs/TESTING_STANDARDS.md``.  The populations quoted were measured on
2026-09-11 on Windows py3.14 / numpy 2.4.4 AND WSL py3.12 / numpy 2.4.6, over
``OPENBLAS_NUM_THREADS`` in {1, 2, 3, 4, 6, 8, 16} and unpinned -- because the
thread count is precisely what moved the pre-fix readings.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import _core as _rc
from lumenairy.elements.rcwa import rcwa_jones_2d

# --------------------------------------------------------------- the fixture
# The failing test's cell, verbatim: an eps = 2.25 background carrying a
# centred square of uniaxial material whose optic axis is rotated 0.7 rad.
# ``no^2 = 1.5^2 = 2.25`` and ``n_substrate = 1.5``, so the layer background
# permittivity, the block's ``zz`` component and the SUBSTRATE permittivity are
# all EXACTLY 2.25 -- the coincidence that turns a mis-rooted layer mode into
# the substrate's own backward mode.
_P, _WL, _DEPTH = 0.5e-6, 0.6e-6, 0.2e-6
_N_SUB, _N_SUP = 1.5, 1.0
_S, _TWIST, _NO, _NE = 48, 0.7, 1.5, 1.7


def _cell():
    tc = np.zeros((_S, _S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = 2.25
    no2, ne2 = _NO ** 2, _NE ** 2
    c0, s0 = np.cos(_TWIST), np.sin(_TWIST)
    x = (np.arange(_S) + 0.5) / _S - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2
    return tc


def _solve(symmetry, theta=0.0, phi=0.0, n_orders=5, n_sub=_N_SUB):
    with warnings.catch_warnings():
        # The pre-fix arm raises _EnergyWarning here; this file asserts on the
        # closure NUMBER rather than on whether a warning was emitted, so the
        # two arms differ in the assertion and not in the plumbing.
        warnings.simplefilter("ignore")
        return rcwa_jones_2d(_P, _P, _cell(), n_sub, _N_SUP, _DEPTH, _WL,
                             theta=theta, phi=phi, n_orders_x=n_orders,
                             n_orders_y=n_orders, symmetry=symmetry)


def _closure_defect(res):
    """``sum R + T - 2``.  A provably lossless cell under the LAURENT rule
    conserves energy EXACTLY at any truncation, so this is an independent
    oracle with an error floor of the arithmetic itself -- it does not depend
    on a converged reference, on the other solve path, or on a prior reading.
    Two incident polarizations, hence 2 rather than 1."""
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 2.0)


# ==================================================================== BAR 1
# The selector itself, with no eigensolver in the loop: the state is
# ENGINEERED (rule 3 of TESTING_STANDARDS) rather than hoped for from a build.
#
# ``eta`` walks the measured backward-error range and four decades either side
# of it.  Every one of these lam^2 values is a propagating LOSSLESS mode --
# real negative up to ``eta`` -- so every returned root must be the OUTGOING
# one.  PRE-FIX this fails for every eta != 0 (the exact ``Re(r) == 0`` test
# cannot see them); it passes only at eta == 0.
_ETA_LADDER = (0.0, -1e-20, -5.7e-20, -1.6e-18, -2.7e-16, -2.9e-15, -5.5e-15,
               -1e-13, -1e-12, 1e-20, 2.9e-15, 1e-12)
_KZ = (0.9, 1.5, 2.0, 0.30, 8.7)


def test_sqrt_decay_pins_the_outgoing_root_through_eigensolver_noise():
    """A propagating lossless mode must get ``Im(lam) >= 0`` whatever sign the
    eigensolver's backward error put on ``Im(lam^2)``.

    DECISION, not a reading: the assertion is on the SIGN of the root, which
    is a discrete choice, not on any digit.  Measured on the failing fixture,
    the mis-rooted modes came back at ``lam = -1.500000000000j`` and
    ``-0.910599187849j`` while the substrate's own modes -- built through exact
    arithmetic, where the old pin did fire -- were ``+1.5000j`` / ``+0.9000j``.
    """
    lam2 = np.array([-kz ** 2 + 1j * eta for kz in _KZ
                     for eta in _ETA_LADDER], dtype=complex)
    lam = _rc._sqrt_decay(lam2)
    assert np.all(lam.imag >= 0.0), (
        "incoming root returned for %d of %d numerically-on-the-cut modes"
        % (int(np.sum(lam.imag < 0)), lam.size))
    # The other half of this function's contract, which the fix must not cost:
    # Re(lam) >= 0 is what makes exp(-lam k0 L) a CONTRACTION.
    assert np.all(lam.real >= 0.0)
    # And the root is still a root: |lam|^2 must reproduce |lam^2|.
    assert np.max(np.abs(lam ** 2 - lam2)) < 1e-12 * np.max(np.abs(lam2))


def test_sqrt_decay_leaves_an_evanescent_root_on_the_decaying_branch():
    """The failure mode the exact test was guarding against must stay guarded:
    a strongly evanescent mode (``lam^2`` real POSITIVE) must keep
    ``Re(lam) > 0``, or the propagator becomes ``exp(+|gamma| k0 L)``."""
    lam2 = np.array([g ** 2 + 1j * eta for g in (0.5, 3.0, 70.0)
                     for eta in _ETA_LADDER], dtype=complex)
    lam = _rc._sqrt_decay(lam2)
    assert np.all(lam.real > 0.0)


# ==================================================================== BAR 2
# The other side of the band.  ``_CUT_BAND_REL`` may only reach modes whose
# negative imaginary part is rounding.  Over 2150 ``Im(r) < 0`` modes on 51
# fixtures x 2 builds the discriminating ratio ``|Re(r)| / max(max|r|, 1)``
# splits into <= 2.245e-16 (WIN) / 1.511e-16 (WSL), all of them lossless
# propagating modes, and >= 8.267e-02 (WIN) / 7.947e-02 (WSL) -- with a loss
# ladder down to Im(eps) = 1e-8 contributing NOTHING to the low side.
# (validation/probe_fix_rcwa_even_sector_wsl/r11_band.py)
_SIGNAL_SIDE_MIN = 7.9e-2      # measured; the bar is 6.9 decades below it


def test_a_physically_signed_root_is_returned_bit_for_bit():
    """A mode whose ``Im(lam) < 0`` is PHYSICS -- a real decay rate, i.e.
    ``|Re(r)|`` a finite fraction of the spectrum rather than rounding -- must
    come back untouched.  Built at the measured signal-side minimum and a
    decade BELOW it, so the claim brackets the bar rather than sitting
    comfortably above it."""
    for frac in (_SIGNAL_SIDE_MIN, _SIGNAL_SIDE_MIN / 10.0):
        # Build r = a - i with |Re(r)| / |r| == frac exactly, so the entry
        # sits AT the measured signal-side minimum (and, second pass, a decade
        # below it -- still 5.9 decades above the bar).
        a = frac / np.sqrt(1.0 - frac ** 2)
        r = np.array([a - 1j], dtype=complex)
        scale = max(float(np.abs(r[0])), 1.0)
        assert abs(abs(r[0].real) / scale - frac) < 1e-12
        x = r ** 2
        lam = _rc._sqrt_decay(x)
        assert lam[0].imag < 0.0, (
            "a physically signed root at |Re(r)|/scale = %.3e was pinned"
            % frac)
        # "Untouched" means exactly the principal root, bit for bit.
        assert np.array_equal(lam, np.sqrt(x))


# ==================================================================== BAR 3
# The consequence, on the real solve.  Bar 1e-9.
#
#   pre-fix, FULL path, |sum R + T - 2| over 14 (build, thread) samples:
#     WIN  3.200e-03  1.003e-03  1.084e-06  1.264e-03  4.939e-06  1.170e-04
#     WSL  8.516e-04  1.942e-03  2.185e-04  1.448e-03  2.104e-03  3.651e-04
#          4.122e-05  1.784e-03
#     -> minimum 1.084e-06
#   post-fix, over the 9 (build, thread) samples measured: <= 6.0e-15, and
#     the even path reads EXACTLY 0.0 on both builds
#
# The bar sits 5.2 decades above the post-fix envelope and 3.0 decades below
# the smallest pre-fix reading.  It is NOT a tolerance on agreement between
# two code paths -- it is a physical conservation law with an arithmetic error
# floor, so it stays meaningful if either path changes.
_CLOSURE_BAR = 1e-9


@pytest.mark.parametrize("symmetry", [False, True])
def test_coincident_layer_and_region_permittivity_closes_energy(symmetry):
    """With the layer background permittivity EXACTLY equal to the substrate's,
    a lossless cell must still conserve energy."""
    d = _closure_defect(_solve(symmetry))
    assert abs(d) < _CLOSURE_BAR, (
        "lossless closure defect %+.3e on the symmetry=%s path" % (d, symmetry))


@pytest.mark.parametrize("theta,phi", [(0.3, 0.0), (0.2, 0.7)])
def test_the_defect_is_not_confined_to_normal_incidence(theta, phi):
    """The even-parity fold is only reachable at normal incidence, but the
    mis-rooted mode is a property of the LAYER, not of the fold: at oblique
    incidence the fold is skipped entirely and the FULL solve carried the same
    defect (pre-fix closure +7.88e-04 WIN / -3.11e-05 WSL at theta=0.3, and
    -1.17e-04 WIN / -7.70e-04 WSL at theta=0.2, phi=0.7; post-fix 4.97e-14 /
    1.08e-13 and 1.33e-15 / 1.47e-14)."""
    d = _closure_defect(_solve(True, theta=theta, phi=phi, n_orders=4))
    assert abs(d) < _CLOSURE_BAR, "closure defect %+.3e" % d


# ==================================================================== BAR 4
# The original assertion, restated with a bar derived from its own measured
# envelope rather than from one build's residual.
#
#   post-fix max|R_full - R_even| over 9 (build, thread) samples: <= 2.78e-16
#   post-fix max|J_full - J_even| over the same:                  <= 6.95e-16
#   pre-fix: 1.625e-02 (WSL, 1 thread) ... 7.73e-14 (WIN, 4 threads)
#
# The bar sits 4.5 decades above the post-fix envelope.  Note it does NOT have
# decades below every pre-fix reading -- the pre-fix reading is thread-order
# noise and spans twelve decades, which is exactly why the closure bar above,
# and not this one, is the discriminator.
_FOLD_BAR = 1e-11


def test_even_fold_matches_the_full_solve_at_the_coincidence():
    """The even-parity fold is an exact change of basis, so it must agree with
    the full ``2N`` solve to the arithmetic floor -- including on the cell that
    sits on the coincidence."""
    full, even = _solve(False), _solve(True)
    dR = float(np.max(np.abs(np.asarray(full[1]) - np.asarray(even[1]))))
    dJ = float(np.max(np.abs(np.asarray(full[3]) - np.asarray(even[3]))))
    assert dR < _FOLD_BAR and dJ < _FOLD_BAR, "dR %.3e dJ %.3e" % (dR, dJ)
    # The fold is a different basis, not the same arithmetic (see the
    # symmetry= docstring): it must not be bit-identical, or the test would be
    # passing because the fold silently did not engage.
    assert not np.array_equal(np.asarray(full[1]), np.asarray(even[1]))


# ==================================================================== BAR 5
# The invariant, read off the real operator rather than an engineered one.


def test_no_layer_mode_of_a_lossless_cell_carries_the_incoming_root(
        monkeypatch):
    """Every layer mode that is numerically on the cut must come back outgoing.

    This reads the eigenvalues of the solve's OWN system matrix (through the
    eig factory the layer solve resolves at call time), so it asserts the
    invariant where it matters instead of on a constructed array.  Pre-fix
    this same count read 10 of 16 on-cut modes on Windows and 8 of 16 on WSL,
    summed over the even and the full solve."""
    seen = []
    orig = _rc._eig_for

    def spy(xp):
        base = orig(xp)

        def wrapped(A):
            w, v = base(A)
            seen.append(np.asarray(w).astype(complex).copy())
            return w, v
        return wrapped

    monkeypatch.setattr(_rc, "_eig_for", spy)
    _solve(True)
    _solve(False)
    assert seen, "the layer eigenproblem did not run"
    bad = 0
    total = 0
    for lam2 in seen:
        lam = _rc._sqrt_decay(lam2)
        scale = max(float(np.max(np.abs(lam))), 1.0)
        # getattr, so that on a PRE-FIX tree (which has no such constant)
        # this test fails on the PHYSICS below rather than on a missing name.
        band = getattr(_rc, "_CUT_BAND_REL", 1e-8)
        on_cut = np.abs(lam.real) <= band * scale
        total += int(on_cut.sum())
        bad += int(np.sum(on_cut & (lam.imag < 0)))
    assert total > 0, "no mode of this cell was on the cut -- wrong fixture"
    assert bad == 0, "%d of %d on-cut modes carry the incoming root" % (bad,
                                                                       total)
