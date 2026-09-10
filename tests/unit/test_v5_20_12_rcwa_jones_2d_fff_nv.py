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
convergence on a stripe, that stripe fixture's freedom from the layer<->region
mode-match degeneracy the reduction reference depends on (2026-09-10 --
``_STRIPE_EPS_GROOVE``), the lossy absorptance SPLIT (the lossless-trap guard),
the crossed-cell convergence (monotone + beats laurent), the out-of-plane
fast convergence, the uniform-cell routing, and the JAX guard.
"""
from __future__ import annotations

import warnings

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

#: Isotropic groove permittivity of the LOSSLESS rotated-director stripe.
#:
#: DELIBERATELY NOT 2.25 (2026-09-10,
#: ``docs/audits/FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md`` Task H).  2.25 is the
#: director's OWN ordinary permittivity (``no^2``, ``no = 1.5``) AND the
#: substrate's (``n_sub = 1.5``), so with a 2.25 groove the ordinary channel of
#: the layer is a perfectly uniform medium identical to the substrate: the
#: layer then carries a set of modes EXACTLY degenerate with the region's, and
#: :func:`~lumenairy.elements.rcwa._core._check_energy`'s documented
#: "near-degenerate layer<->region mode-match at a measure-zero coincidence"
#: is not near -- it is exact.  The interface inverse there amplifies the
#: rounding floor by ~1e14, which is a per-BUILD quantity, so the rigorous 1-D
#: solver's OWN energy theorem reads anywhere from 1e-13 to 5e-02 depending on
#: the truncation, the LAPACK build and even the BLAS thread count.  MEASURED
#: on that coincidence, |sum R + sum T - 2| over n_orders = 11..41:
#:
#:     config                          worst      best   truncations < 1e-9
#:     Windows py3.14/np2.4.4, 1 thr   2.761e-02  4.612e-06     0 of 16
#:     Windows py3.14/np2.4.4, 4 thr   2.309e-02  2.722e-13     1 of 16
#:     WSL py3.12/np2.4.6, 1 thread    5.001e-02  2.061e-06     0 of 16
#:
#: Detuning ANY of the three coincident permittivities restores the theorem at
#: EVERY truncation: 1e-6 relative already brings the worst defect to 4.7e-10,
#: and this 2.10 groove reads <= 2.1e-13 over n_orders = 11..41 (16 of 16
#: sound) and <= 5.1e-13 out to 61, on all three configurations.  The 2-D arms
#: become build-free with it too: the two ratios this file asserts read
#: 0.0209 and 0.0182 -- to FIVE significant figures -- on all three, where on
#: the coincidence they read 0.0277 / 0.0200 [both Windows] and 0.0999 /
#: 1.1773 [WSL], i.e. the Jones ratio crosses the 1.0 that decides the test.
_STRIPE_EPS_GROOVE = 2.10

#: The coincident groove value (``= no^2 = n_sub^2``), kept as the NEGATIVE
#: arm of :func:`test_stripe_fixture_is_free_of_the_mode_match_degeneracy`.
_DEGENERATE_EPS_GROOVE = 2.25

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
#:
#: CORRECTION 2026-09-10 (``FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md`` Task H),
#: appended, not rewritten: the three-arm table above is a reading of the
#: INDEX-COINCIDENT fixture (see ``_STRIPE_EPS_GROOVE``), not of the
#: formulation.  On a y-uniform stripe ``fff_nv`` reduces to the rigorous 1-D
#: rule (the test right below this pins that), and for a REAL SYMMETRIC tensor
#: that rule's ``[[Cxx, Cxy], [Cyx, Cyy]]`` is Hermitian -- MEASURED
#: ``max|C - C^H| / max|C|`` <= 1.7e-16 at n_orders 5..61 -- so an energy
#: theorem does exist there and the closure IS machine-exact: at ``No = 11``
#: 1.421e-14 [Win 1 thr] / 2.354e-14 [Win 4 thr] / 5.373e-14 [WSL] on the
#: non-coincident stripe, and <= 5.4e-14 over ``No`` = 9, 11, 13 on all three,
#: against 2.3e-05..4.0e-04 on the coincident one.  The stripe test
#: therefore asserts ``_ONED_SOUND_CLOSURE`` now.  This 6e-2 envelope is
#: retained as the LIBRARY-WIDE statement for genuinely CROSSED cells, where
#: the operator is not a 1-D reduction and the non-Hermitian argument stands;
#: nothing measured here contradicts it for that regime.
_FFF_NV_CLOSURE_ENVELOPE = 6e-2

#: A rigorous 1-D closure this clean means the Li-1996 energy theorem is
#: actually holding at that truncation.  ``_check_energy``'s own comment: "the
#: closure R+T = 1 is exact in this code (clean solves hold it to <1e-11)".
#: 1e-9 is 100x above that and 4+ decades below every mode-degenerate reading
#: measured below, so nothing near the boundary can be mistaken for sound.
#: 2026-09-10: it is now ASSERTED on a fixture engineered to satisfy it at
#: every truncation (worst 2.1e-13 over n_orders 11..41 on three
#: configurations) rather than SEARCHED for -- see ``_STRIPE_EPS_GROOVE`` and
#: :func:`_rigorous_1d_reference`.  Its gap is therefore 4 decades below and
#: 4 decades above the two things it separates.
_ONED_SOUND_CLOSURE = 1e-9


#: Reference truncation for the rigorous 1-D arm.  CONVERGED, not merely
#: "sound": ``sum R`` measured 0.066447791681 (61), 0.066447219556 (81),
#: 0.066446952496 (101), 0.066446718238 (141) on the non-coincident stripe, so
#: at 81 the reference's own residual error is ~5.0e-07 in ``sum R`` and
#: ~1.9e-06 in the Jones -- 61x and 31x below the smallest quantity compared
#: against it (``ef`` = 3.0399e-05, ``jf`` = 5.8158e-05 at ``No = 11``, both
#: identical to five figures on all three configurations).  Costs 0.37 s.
_ONED_REF_ORDERS = 81


def _rigorous_1d_reference(segments, n_ref=_ONED_REF_ORDERS):
    """The rigorous 1-D full-tensor solve at a CONVERGED truncation, with its
    own energy theorem asserted rather than searched for.

    HISTORY (kept, because the replaced version's reasoning was half right).
    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S1, 2026-08-15, replaced a
    fixed ``n_orders = 11`` reference with a LADDER that scanned 11..41 for the
    first truncation whose lossless closure held to 1e-9, because 11 was a
    poisoned truncation on Windows and a mediocre one on WSL.  That correctly
    identified the symptom -- and made the test depend on the build PRODUCING a
    sound truncation, which is precondition-shaped (``docs/TESTING_STANDARDS.md``
    S3).  MEASURED 2026-09-10: on that fixture the window contains 0 sound
    truncations of 16 on this Windows box at 1 BLAS thread, 1 of 16 at 4
    threads, and 0 of 16 on WSL -- so the same code, same box, same fixture
    PASSED at ``OPENBLAS_NUM_THREADS=4`` and FAILED at 1.

    The cause was never the truncation: it was the fixture sitting on an EXACT
    layer<->region mode coincidence (see ``_STRIPE_EPS_GROOVE`` for the
    mechanism and the numbers).  With the coincidence removed, the rigorous 1-D
    closure holds at EVERY truncation on every configuration measured, so the
    reference needs no search at all -- it is taken at a converged order and
    its theorem is ASSERTED.  That assertion is the fixture's own tripwire: it
    fires if anyone reintroduces a degenerate cell here, which is exactly the
    state that must never go unnoticed.
    """
    _o, R1, T1, J1 = rcwa_jones_1d_segments(
        PX, segments, 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=int(n_ref))
    defect = abs(np.sum(R1) + np.sum(T1) - 2.0)
    assert defect < _ONED_SOUND_CLOSURE, (
        f"the rigorous 1-D reference violates its OWN exact lossless closure "
        f"by {defect:.3e} at n_orders={n_ref}, so its per-order answers are "
        f"suspect and nothing may be compared against them.  That is a "
        f"property of the FIXTURE, not of the formulation: check whether the "
        f"cell has re-acquired a layer<->region mode coincidence (see "
        f"_STRIPE_EPS_GROOVE)")
    return int(n_ref), R1, T1, J1


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
    full-tensor solver AND is more accurate than laurent at the same order.

    2026-09-10: the groove is ``_STRIPE_EPS_GROOVE`` (2.10), NOT the director's
    own ordinary permittivity -- see that constant for why an index-coincident
    groove made every arm of this test a reading of the build's rounding floor.
    All four numbers this test compares are now build-free: ``ef/el`` and
    ``jf/jl`` read 0.0209 / 0.0182 to FIVE significant figures on Windows at 1
    and at 4 BLAS threads and on WSL alike (48x and 55x of margin on the 1.0
    that "beats laurent" means), where on the old fixture ``jf/jl`` read
    0.0200 / 0.0200 / 1.1773 on those same three -- the WSL arm outright
    failing the claim.
    """
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    eg = np.diag([_STRIPE_EPS_GROOVE] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 11
    _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=No, n_orders_y=1,
                                   formulation="fff_nv", symmetry=False)
    _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=No, n_orders_y=1,
                                   formulation="laurent", symmetry=False)
    # The reference is the rigorous 1-D solver at a CONVERGED truncation whose
    # own energy theorem is asserted inside the helper (2026-09-10; the
    # build-scanning ladder it replaces, and why, are in its docstring).
    n_ref, R1, T1, J1 = _rigorous_1d_reference([(0.5, er), (0.5, eg)])
    assert abs(np.sum(R1) + np.sum(T1) - 2.0) < _ONED_SOUND_CLOSURE, n_ref
    # ENERGY.  On a y-UNIFORM stripe the fff_nv operator IS the rigorous 1-D
    # rule (the test above pins that reduction), and that rule is Hermitian for
    # a real symmetric tensor, so the closure is machine-exact here -- not
    # merely inside the crossed-cell envelope.  MEASURED 1.4e-14 [Win 1 thr] /
    # 2.4e-14 [Win 4 thr] / 5.4e-14 [WSL] at No = 9, 11, 13: five decades under
    # this bar, which is itself four decades under the 1e-04..1e-02 a
    # mode-degenerate cell produces.  See _FFF_NV_CLOSURE_ENVELOPE for the
    # correction this replaces and for the crossed-cell regime it still states.
    defect = abs((np.sum(Rf) + np.sum(Tf)) - 2.0)
    assert defect < _ONED_SOUND_CLOSURE, (
        f"fff_nv lossless closure is off by {defect:.3e}; on a y-uniform "
        f"stripe this operator reduces to the rigorous (Hermitian) 1-D rule, "
        f"whose closure is exact -- a defect here means the reduction broke "
        f"or the cell is mode-degenerate")
    assert defect < _FFF_NV_CLOSURE_ENVELOPE      # the library-wide envelope
    # ...and the arm that actually catches a blown-up truncation, which needs
    # no calibration at all: every per-order efficiency is a real power
    # fraction, so a solve that has gone non-physical shows up here first.
    for name, arr in (("R", Rf), ("T", Tf)):
        assert np.all(np.isfinite(arr)), f"fff_nv {name} is not finite"
        assert np.all(arr >= 0.0), (
            f"fff_nv returned a NEGATIVE per-order {name} "
            f"(min {float(np.min(arr)):.3e})")
    # fff_nv tracks the rigorous 1-D solver, and MORE closely than laurent
    # does.  Both arms are compared against the CONVERGED reference above, so
    # neither ratio can be moved by the reference wandering.
    #
    # THE BAR.  2026-09-10: the two ratios are DETERMINISTIC on a
    # non-degenerate cell -- ef/el = 0.0209 and jf/jl = 0.0182 to FIVE
    # significant figures on Windows py3.14/np2.4.4 at 1 AND at 4 BLAS threads
    # and on WSL py3.12/np2.4.6 (ef = 3.0399e-05, el = 1.4527e-03,
    # jf = 5.8158e-05, jl = 3.2042e-03 on every one of them), i.e. a
    # cross-configuration spread below the printed precision.  A bare
    # ``ef < el`` would therefore pass on 48x of margin and say nothing about a
    # 10x degradation; 0.2 keeps a 9.6x gap to the measurement below it and a
    # 5x gap to the 1.0 that "beats laurent" means above it.  (The same two
    # quantities on the index-coincident fixture this test used until
    # 2026-09-10, same reference order: ef/el 0.0277 / 0.0277 / 0.0999 and
    # jf/jl 0.0200 / 0.0200 / 1.1773 -- the WSL arm crossing 1.0 outright.
    # See _STRIPE_EPS_GROOVE.)
    ef = abs(np.sum(Rf) - np.sum(R1))
    el = abs(np.sum(Rl) - np.sum(R1))
    assert ef < 0.2 * el, f"fff_nv err {ef:.2e} not < 0.2 x laurent {el:.2e}"
    jf = np.max(np.abs(Jf - J1))
    jl = np.max(np.abs(Jl - J1))
    assert jf < 0.2 * jl, f"fff_nv Jones err {jf:.2e} not < 0.2 x {jl:.2e}"


#: The modules that BIND the shared ``_sqrt_decay``.  Rebinding the name in
#: each of them is how :class:`_pre_round1_branch` reaches every solver path;
#: patching only the definition module would leave the ``from ... import``
#: copies pointing at the shipped body.
_SQRT_DECAY_BOUND = ("lumenairy.elements.rcwa._core",
                     "lumenairy.elements.rcwa.oned",
                     "lumenairy.elements.rcwa.stack",
                     "lumenairy.elements.pmm.twod",
                     "lumenairy.elements.berreman")


def _pre_round1_sqrt_decay(x, xp=None, band=1e-8):
    """The PRE-5.45.0 branch body, re-typed: the EXACT ``Re(r) == 0`` on-cut
    test and the ``-r`` flip.

    An ``eig`` output never satisfies ``Re(r) == 0`` -- its real part is the
    eigensolver's backward error, ~1e-16 -- so this body never pins a
    structured layer's propagating mode, and the mode-match degeneracy at an
    index coincidence comes back.  That is what makes it a usable
    fail-before: the state is CONSTRUCTED, not hoped for from a build.
    """
    from lumenairy.backend.array import array_namespace  # noqa: PLC0415
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


class _pre_round1_branch:
    """Reinstate the pre-5.45.0 branch body for the duration of a block."""

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import importlib  # noqa: PLC0415
        for name in _SQRT_DECAY_BOUND:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _pre_round1_sqrt_decay
        assert self._saved, _SQRT_DECAY_BOUND
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def _worst_closure(eps_groove, er, ladder):
    """Worst ``|sum R + sum T - 2|`` over the truncation ladder, how many
    rungs stay sound, and the tripwire messages the ladder raised.

    The warnings are COLLECTED here rather than left to a ``pytest.warns``
    around the call: the recorder has to be inside the loop to attribute a
    warning to its rung, and a ``catch_warnings`` block consumes what it
    records, so an outer ``pytest.warns`` would see nothing.  Returning the
    messages keeps the assertion on the text without that trap.
    """
    worst, sound, msgs = 0.0, 0, []
    for n in ladder:
        eg = np.diag([eps_groove] * 3).astype(complex)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        d = abs(float(np.sum(R1) + np.sum(T1) - 2.0))
        worst = max(worst, d)
        sound += d < _ONED_SOUND_CLOSURE
        msgs += [str(w.message) for w in ws
                 if "lossless energy closure violated" in str(w.message)]
    return worst, sound, msgs


def test_stripe_fixture_is_free_of_the_mode_match_degeneracy():
    """The reference fixture admits its OWN energy theorem at EVERY truncation
    -- and the pre-5.45.0 solver, on the index-coincident cell it replaced,
    does not.  Two-sided, with the negative arm ENGINEERED.

    This is the property the whole reduction test rests on, so it is asserted
    directly instead of being hoped for: the rigorous 1-D solver's lossless
    closure is exact for a lossless stack, and on a non-degenerate cell it
    holds at every ``n_orders`` rather than at a lucky one.

    **RESTATED 2026-09-11 (CI KERNEL SWEEP,**
    ``docs/audits/CI_KERNEL_SWEEP_2026_09_11.md`` **).**  The NEGATIVE arm used
    to reconstruct the coincidence through the public API alone -- groove
    permittivity set to the director's own ``no^2`` = 2.25, which is also
    ``n_sub^2`` -- and then require the library to WARN about the closure it
    violated.  **That arm is dead, and it was killed by a fix in this very
    release.**  5.45.0's modal branch-cut repair (``_sqrt_decay``'s relative
    on-cut band) is precisely the repair of "a LAYER permittivity EXACTLY
    EQUAL to a REGION's ... degenerate at EVERY truncation", so the
    coincidence no longer bites: the degenerate arm now reads **2.498e-13,
    16 of 16 truncations sound, no warning**, against the 2.761e-02 and 0 of
    16 this docstring recorded on 2026-09-10.

    This is NOT a kernel-dependent test.  It fails identically on every arm
    measured -- Windows py3.14 and WSL py3.12, Haswell / Sandybridge /
    Nehalem / Katmai -- and it failed on CI (slow-gate shard 1) for the same
    reason it fails here.  It is recorded in the sweep because the sweep is
    what found it: the shard's ``DID NOT WARN`` looks exactly like the two
    genuinely kernel-decided warnings beside it in that matrix, and only a
    two-sided measurement separates them.

    The durability rule the old docstring wrote for exactly this event says
    re-derive, not widen.  So the negative arm is now CONSTRUCTED
    (``docs/TESTING_STANDARDS.md`` restatement 3): the pre-5.45.0 branch body
    is reinstated at every module that binds it, which reopens the degeneracy
    by NINE TO ELEVEN DECADES and makes the tripwire fire again.

    MEASURED 2026-09-11 at one thread on EIGHT ARMS -- Windows py3.14.6 /
    numpy 2.4.4 and WSL py3.12.3 / numpy 2.4.6, each under
    ``OPENBLAS_CORETYPE`` Haswell / Sandybridge / Nehalem / Katmai -- worst
    ``|sum R + sum T - 2|`` over the 16 rungs of ``n_orders`` 11..41:

        arm                 clean          coincident        rungs   warns
        SHIPPED, all 8    2.0-4.6e-13     2.5-6.4e-13        16/16     0
        PRE-FIX WIN-HAS   2.083e-13       2.761e-02           0/16    16
        PRE-FIX WIN-SBR   3.020e-13       4.690e-02           1/16    14
        PRE-FIX WIN-NEH   2.958e-13       8.158e-03           2/16    14
        PRE-FIX WIN-KAT   4.008e-13       7.289e-04           5/16    11
        PRE-FIX WSL-HAS   1.861e-13       5.001e-02           0/16    16
        PRE-FIX WSL-SBR   3.069e-13       4.802e-02           1/16    14
        PRE-FIX WSL-NEH   2.918e-13       2.955e-03           2/16    12
        PRE-FIX WSL-KAT   4.035e-13       1.584e-03           5/16    11

    THE COLUMN THAT MATTERS IS "rungs", and it is why this restatement was
    itself restated once.  The first version of this test asserted
    ``pre_coin_sound == 0`` -- an EXACT COUNT over nondeterministic machinery,
    ``docs/TESTING_STANDARDS.md`` shape S5 -- because 0/16 is what the Haswell
    arm reads.  It is 1/16 on Sandybridge, 2/16 on Nehalem and 5/16 on Katmai,
    on BOTH builds: WHICH rungs of the ladder happen to land on the wrong side
    of a rounding-level degeneracy is exactly the thing the kernel decides.
    That assertion failed on the WSL-Sandybridge arm of this sweep's own
    verification pass and has been withdrawn.

    What survives every arm is the READING, and it has room on both sides:
    the coincident worst is **7.289e-04 at its smallest**, seven decades above
    the clean population's 4.6e-13 ceiling and 73x above the 1e-05 bar
    asserted below, while the shipped solver keeps that same cell at 6.4e-13
    or better.  Both claims are unconditional on every arm: the shipped solver
    is sound on both cells, and the defect the fixture was moved off is real
    and reconstructible on demand.
    """
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    ladder = range(11, 42, 2)
    n_rungs = len(list(ladder))

    # ---- the POSITIVE arm, on the shipped solver
    clean, clean_sound, clean_msgs = _worst_closure(
        _STRIPE_EPS_GROOVE, er, ladder)
    assert clean < _ONED_SOUND_CLOSURE, (
        "the reference fixture violates its own exact lossless closure by "
        "%.3e somewhere in %d..%d" % (clean, ladder.start, ladder.stop - 1))
    assert clean_sound == n_rungs, (clean_sound, n_rungs, clean)
    assert clean_msgs == [], clean_msgs

    # ---- and the coincidence is sound on the shipped solver TOO, which is
    # what 5.45.0 bought.  Same bar as the reference fixture -- this is the
    # claim that used to be its opposite.
    coincident, coin_sound, coin_msgs = _worst_closure(
        _DEGENERATE_EPS_GROOVE, er, ladder)
    assert coincident < _ONED_SOUND_CLOSURE, (
        "the index-coincident cell no longer closes on the shipped solver "
        "(%.3e): the 5.45.0 branch-cut repair has regressed" % coincident)
    assert coin_sound == n_rungs, (coin_sound, n_rungs, coincident)
    assert coin_msgs == [], coin_msgs

    # ---- the NEGATIVE arm, ENGINEERED: reinstate the pre-5.45.0 branch body
    # and the degeneracy comes back, loudly.  Bars placed by the gap: the
    # shipped readings are ~2.5e-13 and the pre-fix coincident one 2.761e-02,
    # so the 1e5 ratio below leaves six decades of margin on each side.
    with _pre_round1_branch():
        pre_clean, _pre_clean_sound, _pre_clean_msgs = _worst_closure(
            _STRIPE_EPS_GROOVE, er, ladder)
        pre_coin, pre_coin_sound, pre_coin_msgs = _worst_closure(
            _DEGENERATE_EPS_GROOVE, er, ladder)
    # the library must SAY SO, and say what it measured
    assert pre_coin_msgs, "the tripwire stayed silent on the pre-fix arm"
    assert "lossless energy closure violated" in pre_coin_msgs[0]
    # the pre-fix arm must break the COINCIDENCE and only the coincidence --
    # otherwise it is demonstrating some other defect.  Clean stays sound on
    # all eight arms (worst 4.035e-13 against this 1e-09 bar).
    assert pre_clean < _ONED_SOUND_CLOSURE, (
        "the pre-fix branch body also breaks the NON-coincident fixture "
        "(%.3e), so it is not isolating the mode-match degeneracy"
        % pre_clean)
    # THE reading, bounded on both sides by the eight-arm census in the
    # docstring: coincident worst 7.289e-04 .. 5.001e-02 (73x to 5000x above
    # this bar) against a clean population that never leaves 4.6e-13.
    assert pre_coin > 1e-5, (
        "the index-coincident cell closed to %.3e even with the pre-5.45.0 "
        "branch body: the mode-match degeneracy this fixture was moved off "
        "can no longer be reconstructed, so the move (and this test) must be "
        "re-derived" % pre_coin)
    # ... and as a RATIO against the clean arm measured in the same run, which
    # is what makes it a defect rather than a scale: 1.82e+09 .. 2.69e+11 over
    # the eight arms, against this 1e+05.
    assert pre_coin > 1e5 * max(pre_clean, 1e-13), (pre_coin, pre_clean)
    # at least ONE rung of the ladder is broken.  Deliberately not a count:
    # 0 / 1 / 2 / 5 of 16 rungs stay sound on Haswell / Sandybridge / Nehalem
    # / Katmai, so any exact count here is a kernel reading (see the
    # docstring's "rungs" column).
    assert pre_coin_sound < n_rungs, (pre_coin_sound, n_rungs, pre_coin)
    # and the fix is worth decades on the same ladder, measured here.
    # Census: 1.6e+09 .. 6.3e+11 over the eight arms, against this 1e+07.
    assert pre_coin / max(coincident, 1e-13) > 1e7, (pre_coin, coincident)


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
    # bar below, where the ROTATED-DIRECTOR sibling reads 3e-3..5e-3.
    # (CORRECTION 2026-09-10: that sibling reading was the index coincidence,
    # not the non-Hermitian operator -- see _FFF_NV_CLOSURE_ENVELOPE and
    # _STRIPE_EPS_GROOVE.  It now reads 1.4e-14 too, so this test's bar is no
    # longer the odd one out; the measurement and the bar below stand.)  The old
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
