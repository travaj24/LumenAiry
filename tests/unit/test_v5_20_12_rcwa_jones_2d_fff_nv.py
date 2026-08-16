"""Full anisotropic off-diagonal FFF: rcwa_jones_2d(formulation='fff_nv').

``formulation='fff_nv'`` builds the Li-2003 successive full-tensor factorization
``ehat = L2 L1(eps)`` (J.Opt.A 5:345; the Smagin-Weiss-Dyakov 2026 ``l+-_tau``
operator), so ALL FOUR in-plane blocks -- including the off-diagonal
``Cxy``/``Cyx`` of a rotated in-plane director (``exy, eyx != 0``) -- get the
correct inverse-rule treatment (the ``'li'`` diagonal rule leaves the
off-diagonal Laurent-floored).  It reaches the same limit as ``'laurent'`` but
converges markedly faster on sharp anisotropic walls.

Unlike the normal-vector projector form ``[[eps.C]][[C]]^-1`` (which inverts an
ill-conditioned 2N x 2N matrix, ``cond ~ 1e7`` for a crossed pillar), the Li-2003
operator inverts ONLY scalar wall-normal elements (plus one N x N block), so it
is well-conditioned even for a CROSSED (both-axis-patterned) cell -- so crossed
anisotropic pillars now CONVERGE (rigorous for axis-aligned / Manhattan cells).
Out-of-plane tensors (`exz, eyz != 0`) are also handled -- the full-3x3 `L2 L1`
plus the `E_z` fold (Li 2003 Eq. 27) -- again converging far faster than laurent.
These tests pin: the operator reduction to the rigorous Li-1996 1-D rule on a
stripe, the exact reduction to the 1-D full-tensor solver + faster-than-Laurent
convergence on a stripe, the lossy absorptance SPLIT (the lossless-trap guard),
the crossed-cell convergence (monotone + beats laurent), the out-of-plane
fast convergence, the uniform-cell routing, and the JAX guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import rcwa_jones_2d
from lumenairy.elements.rcwa import twod as _twod
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments

# eig-heavy 2-D fff_nv (Li-2003 successive full-tensor); version-insensitive
# numerics -> run in the slow-tests job to keep the fast 4-Python gate under
# its cap (v5.21.1 fast-gate trim).
pytestmark = pytest.mark.slow


def _rot(phi, no, ne):
    """In-plane rotated uniaxial 3x3 (optic axis at angle phi in the x-y plane)."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    """y-uniform (x-periodic) two-material stripe of 3x3 tensors."""
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


PX = 0.7e-6
WL = 1.0e-6
DEPTH = 0.5e-6

#: The ``fff_nv`` lossless-closure envelope.  THIS IS A REGIME, NOT A
#: CALIBRATION (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S1, 2026-08-15).
#:
#: ``rcwa_jones_2d``'s own docstring states it, under INHERENT CLOSURE ERROR
#: (audit M5 2026-07-25): "the ``fff_nv`` in-plane operator is NON-Hermitian,
#: so there is no finite-truncation energy theorem behind it -- a LOSSLESS
#: ``fff_nv`` cell violates ``R+T = 1`` by ~1e-2..6e-2 ... at EVERY truncation,
#: and the resulting ``_EnergyWarning`` is a property of the formulation, not
#: an instability signal."  ``stabilize=True`` does not even count that warning
#: as a failed rung for this formulation, for exactly that reason.
#:
#: So a closure bar on an ``fff_nv`` result cannot be tight: there is no
#: theorem to be tight against.  The bar this replaces was 5e-3, calibrated on
#: one build's reading, and what it actually measured was a knife-edge
#: truncation.  At the ``n_orders=11`` this fixture uses, the closure residual
#: of ALL THREE arms swings by orders across builds -- measured 2026-08-15,
#: same worktree, same fixture:
#:
#:     arm                    Windows py3.14/np2.4.4   WSL py3.12/np2.5.1
#:     fff_nv                 2.987e-03                5.101e-03
#:     laurent                4.180e-05                6.247e-03
#:     rigorous 1-D (Li-1996) 1.742e-02                1.764e-05
#:
#: plus 5.325e-03 for the fff_nv arm on the CI ubuntu numpy-2.5 wheel, which
#: is what failed the 5e-3 bar.  The rigorous 1-D arm -- which DOES have an
#: energy theorem, and holds it to <1e-11 on a clean solve -- reading 1.7e-02
#: on one mount and 1.8e-05 on the other is the proof that this truncation is
#: a measure-zero mode-match coincidence for this geometry, not a property of
#: any formulation.  No bar tighter than the regime can survive that, and the
#: reduction claims below are therefore stated on ``sum(R)``, which agrees to
#: 4.5e-06 across the two mounts.
#:
#: 6e-2 is the upper end of the library's own documented envelope: 11x over
#: the worst closure reading any build has produced here (5.325e-03), and
#: still strictly INSIDE the solver's hard energy tripwire (which raises at
#: ``sum R+T > 1.05 * n_states``, i.e. a defect of 1e-1), so the assertion
#: still says something the library does not already enforce.
_FFF_NV_CLOSURE_ENVELOPE = 6e-2

#: A rigorous 1-D closure this clean means the Li-1996 energy theorem is
#: actually holding at that truncation.  ``_check_energy``'s own comment: "the
#: closure R+T = 1 is exact in this code (clean solves hold it to <1e-11)".
#: 1e-9 is 100x above that and 4+ decades below every poisoned reading in the
#: ladder measured below, so the search below cannot pick a bad order.
_ONED_SOUND_CLOSURE = 1e-9


def _sound_1d_reference(segments, n_first=11, n_last=41):
    """The rigorous 1-D solve at the first truncation where its OWN energy
    theorem actually holds, FOUND on the running build.

    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S1, 2026-08-15.  Li-1996
    1-D closure is EXACT for a lossless stack, so a closure defect is never
    truncation error -- it is the measure-zero layer<->region mode-match
    coincidence :func:`~lumenairy.elements.rcwa._core._check_energy`
    documents, at which the per-order answers are, in the library's own
    words, "suspect".  WHICH truncations are poisoned is a per-BUILD fact.
    Measured on this fixture 2026-08-15, closure defect by ``n_orders``::

        n_orders   11        13        15        17        19        21
        Windows    1.74e-02  2.01e-04  8.46e-03  3.06e-03  9.24e-03  1.51e-14
        WSL        1.76e-05  ...                                     (sound)

    -- i.e. the order the test used to compare against, 11, is a POISONED
    truncation on Windows and a merely mediocre one on WSL.  Comparing a
    2-D solve against a reference in that state is what left the Jones arm
    of ``test_fff_nv_stripe_reduces_to_rigorous_1d`` at
    ``max|Jf - J1| / max|Jl - J1|`` = 0.041 on one mount and 0.419 on the
    other -- a 10x swing on an assertion that needs only 1.0 to fail.

    Searching for a sound truncation instead makes every comparison against
    this reference build-free, and STRENGTHENS them: the same two ratios read
    0.034 (R) and 0.025 (Jones) against the converged reference, i.e. 29x and
    40x of margin instead of 2.4x.

    Raises rather than skipping if no truncation in the window is sound --
    that would mean the fixture itself is unusable and must be seen.
    """
    for n in range(int(n_first), int(n_last) + 1, 2):
        _o, R1, T1, J1 = rcwa_jones_1d_segments(
            PX, segments, 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=n)
        if abs(np.sum(R1) + np.sum(T1) - 2.0) < _ONED_SOUND_CLOSURE:
            return n, R1, T1, J1
    raise AssertionError(
        f"no truncation in {n_first}..{n_last} gave the rigorous 1-D solver "
        f"its own exact lossless closure on this build, so there is no sound "
        f"reference to compare against; the fixture, not the formulation, is "
        f"the problem")


def test_fff_nv_operator_reduces_to_li1996_on_stripe():
    """The Li-2003 successive operator L2 L1 reduces EXACTLY (machine precision)
    to the rigorous Li-1996 1-D factorization on a y-uniform stripe: the
    wall-normal diagonal Cxx == [[1/exx]]^-1."""
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)[:2, :2]
    eg = np.diag([2.25, 2.25]).astype(complex)
    Sx = 64
    xm = (np.arange(Sx) + 0.5) / Sx < 0.5
    cell = np.zeros((Sx, 8, 2, 2), complex)
    for ix in range(Sx):
        cell[ix, :] = er if xm[ix] else eg
    orders, _ = _twod._harmonic_orders_2d(9, 1)
    Cxx, Cxy, Cyx, Cyy = _twod._li_convolutions_2d_tensor(
        cell[:, :, 0, 0], cell[:, :, 0, 1], cell[:, :, 1, 0],
        cell[:, :, 1, 1], orders, 9, 1, np)
    inv_exx = np.linalg.inv(
        _twod._eps_convolution_2d(1.0 / cell[:, :, 0, 0], orders, 9, 1))
    assert np.max(np.abs(Cxx - inv_exx)) < 1e-12       # rigorous inverse rule


def test_fff_nv_stripe_reduces_to_rigorous_1d():
    """A y-uniform rotated-director stripe: fff_nv reduces to the rigorous 1-D
    full-tensor solver AND is more accurate than laurent at the same order."""
    er, eg = _rot(np.deg2rad(35.0), 1.5, 2.3), np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 11
    _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=No, n_orders_y=1,
                                   formulation="fff_nv", symmetry=False)
    _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=No, n_orders_y=1,
                                   formulation="laurent", symmetry=False)
    # The reference is the rigorous 1-D solver at a truncation where its own
    # energy theorem HOLDS on this build -- not at n_orders=No, which is a
    # poisoned truncation for this fixture on at least one shipped platform.
    # See _sound_1d_reference for the ladder and the margins this buys.
    n_ref, R1, T1, J1 = _sound_1d_reference([(0.5, er), (0.5, eg)])
    assert abs(np.sum(R1) + np.sum(T1) - 2.0) < _ONED_SOUND_CLOSURE, n_ref
    # ENERGY: physical, and inside the formulation's OWN documented envelope
    # -- fff_nv has no finite-truncation energy theorem, so there is nothing
    # tighter to assert here.  See _FFF_NV_CLOSURE_ENVELOPE for the regime and
    # for the three-arm cross-build table that rules a calibrated bar out.
    defect = abs((np.sum(Rf) + np.sum(Tf)) - 2.0)
    assert defect < _FFF_NV_CLOSURE_ENVELOPE, (
        f"fff_nv lossless closure is off by {defect:.3e}, outside the "
        f"non-Hermitian operator's documented ~1e-2..6e-2 envelope")
    # ...and the arm that actually catches a blown-up truncation, which needs
    # no calibration at all: every per-order efficiency is a real power
    # fraction, so a solve that has gone non-physical shows up here first.
    for name, arr in (("R", Rf), ("T", Tf)):
        assert np.all(np.isfinite(arr)), f"fff_nv {name} is not finite"
        assert np.all(arr >= 0.0), (
            f"fff_nv returned a NEGATIVE per-order {name} "
            f"(min {float(np.min(arr)):.3e})")
    # fff_nv tracks the rigorous 1-D solver, and MORE closely than laurent
    # does.  Both arms are compared against the SOUND reference above, so
    # neither ratio can be moved by the reference wandering: measured
    # ef/el = 0.0343 and jf/jl = 0.0247 (29x and 40x of margin), against
    # 0.053/0.041 [W] and 0.048/0.419 [M] when the reference was taken at the
    # poisoned n_orders=11.
    ef = abs(np.sum(Rf) - np.sum(R1))
    el = abs(np.sum(Rl) - np.sum(R1))
    assert ef < el, f"fff_nv err {ef:.2e} not < laurent err {el:.2e}"
    jf = np.max(np.abs(Jf - J1))
    jl = np.max(np.abs(Jl - J1))
    assert jf < jl, f"fff_nv Jones err {jf:.2e} not < laurent {jl:.2e}"


def test_fff_nv_beats_laurent_convergence():
    """fff_nv reaches a given accuracy at far lower order than laurent on a
    high-contrast rotated-director stripe (the off-diagonal FFF win)."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    cell = _stripe(er, eg)

    def sumR(No, form):
        _o, R, _T, _J = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=No, n_orders_y=1,
                                      formulation=form, symmetry=False)
        return np.sum(R)

    # converged reference: the rigorous 1-D solver at high order
    _o, Rref, _T, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=61)
    ref = np.sum(Rref)
    ef = abs(sumR(9, "fff_nv") - ref)
    el = abs(sumR(9, "laurent") - ref)
    assert ef < 0.5 * el, f"fff_nv {ef:.2e} not < half of laurent {el:.2e}"


def test_fff_nv_lossy_stripe_absorptance_split():
    """Lossless trap guard: on a LOSSY rotated-director stripe, fff_nv's
    absorptance (1 - R - T) tracks the rigorous 1-D solver's SPLIT more closely
    than laurent -- energy closure alone would not police this.

    2026-08-15 (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S1 sibling
    sweep).  This test was measuring something else entirely.  The 2-D arms
    were averaged PER INCIDENT POLARIZATION (``np.sum(R, 1)`` leaves a (2,)
    vector, then ``np.mean``), but the 1-D reference was built as
    ``1 - sum(R1) - sum(T1)`` with the sums taken over BOTH pols at once --
    so ``A1`` was ``2*A - 1``, not ``A``.  Both sides of the comparison were
    then dominated by that factor-of-two offset:

        A1 as written   -0.14089        A1 per-pol mean   0.42956
        |Af - A1|        0.57054        |Af - A1| fixed   9.094e-05
        |Al - A1|        0.57214        |Al - A1| fixed   1.694e-03
        ratio            0.99720        ratio fixed       0.05368

    -- i.e. the assertion reduced algebraically to ``Af > Al`` and passed on a
    0.28% margin that had nothing to do with tracking the reference.  Fixing
    the normalisation makes the test assert what its docstring always claimed
    AND turns a 1.003x coin flip into an 18.6x margin.
    """
    er = _rot(np.deg2rad(35.0), 1.5 + 0.15j, 2.3 + 0.15j)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 11

    def absorptance(form):
        _o, R, T, _J = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                     n_orders_x=No, n_orders_y=1,
                                     formulation=form, symmetry=False)
        return 1.0 - np.sum(R, 1) - np.sum(T, 1)      # (2,) per incident pol

    _o, R1, T1, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=No)
    # PER INCIDENT POL on both sides -- R1/T1 are (2, n_orders), exactly like
    # the 2-D arms, so the reduction must match theirs.
    A1 = float(np.mean(1.0 - np.sum(R1, 1) - np.sum(T1, 1)))
    Af = float(np.mean(absorptance("fff_nv")))
    Al = float(np.mean(absorptance("laurent")))
    # the fixture is genuinely absorbing, so "tracks the split" is a claim
    # about a real number and not about two ways of writing zero
    assert 0.05 < A1 < 0.95, f"the lossy reference absorbs {A1:.4f}"
    assert abs(Af - A1) < abs(Al - A1), (
        f"fff_nv absorptance err {abs(Af - A1):.3e} not < laurent "
        f"{abs(Al - A1):.3e} (A1 = {A1:.6f})")


def test_fff_nv_crossed_cell_converges_and_beats_laurent():
    """A CROSSED (both-axis-patterned) rotated-director pillar now CONVERGES
    under fff_nv (Li-2003 L2 L1, well-conditioned) -- monotone and markedly
    faster than laurent on a high-contrast lossy cell, energy closed."""
    th = np.deg2rad(45.0)                          # metal-like eps, rotated 45 deg
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    er = R @ np.diag([-8.0 + 1.2j, -2.0 + 0.8j, -2.0 + 0.8j]).astype(complex) @ R.T
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)

    def _sq(S):
        m = np.zeros((S, S), bool)
        m[S // 4:3 * S // 4, S // 4:3 * S // 4] = True
        c = np.zeros((S, S, 3, 3), complex)
        for i in range(S):
            for j in range(S):
                c[i, j] = er if m[i, j] else eg
        return c

    def sumR(No, form):
        _o, R, _T, _J = rcwa_jones_2d(0.5e-6, 0.5e-6, _sq(max(40, 4 * No + 4)),
                                      1.5, 1.0, 0.25e-6, WL, n_orders_x=No,
                                      n_orders_y=No, formulation=form,
                                      symmetry=False)
        return np.sum(R)

    ref = sumR(17, "fff_nv")                       # best-converging reference
    ef = [abs(sumR(No, "fff_nv") - ref) for No in (5, 7, 9, 11)]
    el = [abs(sumR(No, "laurent") - ref) for No in (5, 7, 9, 11)]
    assert ef[-1] < ef[0]                          # fff_nv converging
    # >~3x better than laurent.  2026-08-15 sibling sweep
    # (FIX_RUNNER_PINS_2_2026_08_15 S1): measured ef[-1]/el[-1] = 0.0952 on
    # BOTH Windows py3.14/np2.4.4 and WSL py3.12/np2.5.1 -- bit-identical to
    # four figures, so the 3.2x headroom is headroom over a DETERMINISTIC
    # convergence value, not over a build-dependent one.  Retained as-is.
    assert ef[-1] < 0.3 * el[-1]
    assert all(ef[i + 1] <= ef[i] + 1e-9 for i in range(len(ef) - 1))  # monotone


def test_fff_nv_uniform_routes_to_laurent():
    """A UNIFORM anisotropic tensor cell (no walls) + fff_nv routes to laurent
    (which is exact there) and matches it."""
    e = _rot(np.deg2rad(30.0), 1.5, 2.1)
    cell = np.broadcast_to(e, (32, 32, 3, 3)).copy()
    _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=5, n_orders_y=5,
                                   formulation="fff_nv", symmetry=False)
    _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=5, n_orders_y=5,
                                   formulation="laurent", symmetry=False)
    assert np.max(np.abs(Rf - Rl)) < 1e-12
    assert np.max(np.abs(Jf - Jl)) < 1e-12


def _oop_stripe(No):
    """y-uniform stripe of a tilted uniaxial (optic axis tilted about y ->
    exz, ezx != 0, out-of-plane)."""
    th = np.deg2rad(35.0)
    c, s = np.cos(th), np.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    er = Ry @ np.diag([1.5 ** 2, 1.5 ** 2, 2.4 ** 2]).astype(complex) @ Ry.T
    eg = np.diag([2.1, 2.1, 2.1]).astype(complex)
    Sx = max(64, 4 * No + 4)
    xm = (np.arange(Sx) + 0.5) / Sx < 0.5
    cell = np.zeros((Sx, 8, 3, 3), complex)
    for ix in range(Sx):
        cell[ix, :] = er if xm[ix] else eg
    return cell


def test_fff_nv_out_of_plane_converges_fast():
    """OUT-OF-PLANE (exz, ezx != 0) is now supported: the Li-2003 successive
    full-3x3 factorization + the E_z fold.  fff_nv converges FAST (nearly
    order-independent) while laurent climbs slowly TOWARD the same value -- so
    fff_nv reaches the true limit at far lower order.  Energy closes."""
    def sumR(No, form):
        _o, R, T, _J = rcwa_jones_2d(PX, PX, _oop_stripe(No), 1.5, 1.0, DEPTH,
                                     WL, n_orders_x=No, n_orders_y=1,
                                     formulation=form, symmetry=False)
        return np.sum(R), np.sum(R) + np.sum(T)

    conv, e0 = sumR(13, "fff_nv")                  # fff_nv converged value
    # ENERGY CLOSES -- and here that is a real claim, unlike the in-plane
    # sibling.  2026-08-15 (FIX_RUNNER_PINS_2_2026_08_15 S1): this fixture is
    # a TILTED-uniaxial stripe, so it carries exz/ezx but exy = eyx = 0, and
    # the non-Hermitian in-plane operator that costs fff_nv its energy
    # theorem (see _FFF_NV_CLOSURE_ENVELOPE) is simply not excited at normal
    # incidence.  Measured closure defect 2.043e-14 [Windows py3.14/np2.4.4]
    # and 7.994e-15 [WSL py3.12/np2.5.1] -- round-off, five decades under the
    # bar below, where the ROTATED-DIRECTOR sibling reads 3e-3..5e-3.  The old
    # 5e-3 bar here was copied from that sibling and asserted nothing (2e11 of
    # headroom); 1e-9 is 100x above the library's own "<1e-11 on a clean
    # solve" and 5e4 above the worst reading either mount produced.
    assert abs(e0 - 2.0) < 1e-9
    f7, _ = sumR(7, "fff_nv")
    l7, _ = sumR(7, "laurent")
    l15, _ = sumR(15, "laurent")
    # fff_nv ~converged by No=7.  2026-08-15 sibling sweep: measured
    # |f7 - conv| = 9.664963e-06 on BOTH mounts -- bit-identical to seven
    # figures.  2.07x is thin, but it is thin against a value that does not
    # move: this OOP stripe is well conditioned (the closure defect above is
    # round-off), so the residual is a deterministic truncation number, not
    # the LAPACK-dependent kind that failed the in-plane sibling.  Retained.
    assert abs(f7 - conv) < 2e-5
    assert abs(l15 - conv) < abs(l7 - conv)         # laurent climbs toward fff_nv
    # fff_nv converges markedly faster.  Margin recalibrated 0.3 -> 0.5 with
    # the OOP factor-i fix (AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14): the
    # corrected physics lands the ratio of these two already-tiny residuals
    # at 0.32 (was calibrated on the pre-fix values); the absolute
    # convergence gate above is the load-bearing one.
    # 2026-08-15 sibling sweep: measured ratio 0.3194 on BOTH mounts
    # (bit-identical), i.e. the 1.57x here is headroom over a deterministic
    # convergence value.  Retained; the absolute gate above is load-bearing.
    assert abs(f7 - conv) < 0.5 * abs(l7 - conv)


def test_out_of_plane_matches_berreman_uniform():
    """INDEPENDENT-METHOD check of the out-of-plane machinery: a UNIFORM tilted-
    uniaxial (exz, ezx != 0) slab solved by rcwa_jones_2d must match the Berreman
    4x4 method (an entirely different formalism) to machine precision, at normal
    AND conical incidence.  Compares the Jones-reflection singular values
    (basis-invariant), so it is convention-independent."""
    from lumenairy.elements.berreman import berreman_jones_1d
    th = np.deg2rad(35.0)
    c, s = np.cos(th), np.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    eps = Ry @ np.diag([1.5 ** 2, 1.5 ** 2, (2.4 + 0.05j) ** 2]) @ Ry.T
    cell = np.broadcast_to(eps, (8, 8, 3, 3)).copy()
    for ang in (0.0, 25.0, 45.0):
        a = np.deg2rad(ang)
        _o, _R, _T, Jr = rcwa_jones_2d(0.6e-6, 0.6e-6, cell, 1.5, 1.0, DEPTH, WL,
                                       theta=a, phi=0.0, n_orders_x=3,
                                       n_orders_y=3, formulation="laurent",
                                       symmetry=False)
        _Rb, _Tb, Jb, _Jt = berreman_jones_1d([(eps, DEPTH)], 1.5, 1.0, WL,
                                              theta=a, phi=0.0)
        sv_r = np.sort(np.linalg.svd(Jr, compute_uv=False))
        sv_b = np.sort(np.linalg.svd(np.asarray(Jb), compute_uv=False))
        assert np.max(np.abs(sv_r - sv_b)) < 1e-11, f"theta={ang}"


def test_fff_nv_out_of_plane_same_limit_as_laurent():
    """fff_nv and laurent converge to the SAME out-of-plane limit: their gap
    shrinks monotonically with n_orders (fff_nv reaches it fast, laurent slowly).
    Guards against the operator-Schur E_z fold converging to a WRONG limit."""
    def sumR(No, form):
        _o, R, _T, _J = rcwa_jones_2d(PX, PX, _oop_stripe(No), 1.5, 1.0, DEPTH,
                                      WL, n_orders_x=No, n_orders_y=1,
                                      formulation=form, symmetry=False)
        return np.sum(R)
    gaps = [abs(sumR(No, "fff_nv") - sumR(No, "laurent")) for No in (7, 15, 25)]
    assert gaps[0] > gaps[1] > gaps[2]              # converging to the same limit
    # gap closing meaningfully.  2026-08-15 sibling sweep: gaps measured
    # 3.9922e-05 / 1.7625e-05 / 1.1241e-05 and the ratio 0.2816 on BOTH
    # mounts, bit-identical -- 1.78x over a deterministic value.  Retained.
    assert gaps[2] < 0.5 * gaps[0]


def test_fff_nv_jax_raises():
    """fff_nv rejects a JAX-traced cell (host-side successive factorization)."""
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="JAX backend"):
        rcwa_jones_2d(PX, PX, jnp.asarray(cell), 1.5, 1.0, DEPTH, WL,
                      n_orders_x=9, n_orders_y=1, formulation="fff_nv",
                      symmetry=False)
