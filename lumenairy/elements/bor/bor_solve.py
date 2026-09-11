"""BOR-PMM Milestone 5: the high-level axisymmetric stack solver (the prototype
of the eventual ``BORStack`` public API).

Pipeline: per-layer radial vector modes (closed-wall clean half-spaces, M5a) ->
flux-normalized basis -> M4 z-cascade S-matrix -> physical-mode R/T efficiencies
+ cylindrical far-field orders (M5b Fourier-Bessel / vortex Hankel).

BASIS (follow-up to AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13):
``build_layer`` now defaults to the Yee div-conforming STAGGERED basis -- the
same spurious-free discretization production ``BORStack`` uses -- so the
cascade conserves energy to machine precision.  The historical NODAL FD basis
(``basis='nodal'``) is retained for its legacy gates but is catastrophically
unreliable on large cells: its spurious divergence-violating mode sea (~40-50%
of the basis at Rbig ~ 12 lambda) carries zero z-flux, so each spurious mode's
forward/backward orientation is decided by the SIGN OF NOISE.  When adjacent
layers share most of their cross-section, near-identical spurious modes can
orient OPPOSITELY, making a layer-a "forward" combination equal a layer-b
"backward" combination -- which renders the interface transmission block
``a + b`` numerically singular (measured cond ~ 2.6e15 at Rbig = 12 lambda)
and blows the cascade energy up to ~1e29 (small cells only leak the documented
~1-4% floor, which does NOT decrease with N).
"""
from __future__ import annotations

import warnings

import numpy as np

from ._orient import channel_core, flux_is_strong
from .zcascade import (
    interface_smatrix,
    layer_modes,
    propagation_smatrix,
    redheffer_star,
)

#: FAIL-BEFORE SWITCH for the nodal passivity refusal (5.45.1).  ``False``
#: restores the pre-fix behaviour bit for bit: the ``Rbig/lambda`` warning
#: below, and the wrong number returned.  A switch, not a policy -- it changes
#: nothing on any solve that does not trip the screen.
BOR_NODAL_PASSIVITY_GUARD = True

#: THE REFUSAL BAR on ``max(R + T) - 1``, for a stack the solver can PROVE
#: passive (every layer lossless, the incidence medium propagating).  On such a
#: stack ``R + T <= 1`` is a theorem, so any super-unity is numerical damage
#: and its SIZE is the detector.
#:
#: NOT a conditioning conjunction.  The scoping measured the nodal
#: ``inv(a + b)``'s equilibrated residual at 5.998e-13 against the Cartesian
#: guard's ``_INV_RESID_REFUSE = 1e-8`` -- four to five decades BELOW the bar --
#: so ``rcwa/_core._guarded_inverse`` as written refuses 0 of 6 broken rows on
#: every kernel.  The nodal ``a + b`` is not singular; it AMPLIFIES.  What
#: separates the two populations is passivity
#: (``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`` section 3.3).
#:
#: MEASURED POPULATIONS, 132 solves over four families -- the scoping's
#: five-layer ring stack at Rbig = 1 .. 16 vacuum wavelengths, an N-refinement
#: sweep, every nodal fixture the shipped suite builds, and a 48-row small-cell
#: census over m = 0,1,2 x N = 120/200 x Rbig/lambda = 0.5 .. 2.0
#: (``validation/probe_fix_bor_guards/s3_nodal_passivity.py``, with the guard
#: DISARMED so the population is the numbers the solver RETURNED):
#:
#:   STAGGERED, every row, every family     worst ``R+T-1`` = 3.7406e-12 (WIN)
#:                                                            1.9959e-11 (WSL)
#:   NODAL, the family that is accurate     worst ``R+T-1`` = 4.4336e-09
#:     (UNIFORM layers on a small cell)                       (BOTH builds)
#:   -------------------------------------  -------------------------------
#:   NODAL, everything else                 ``R+T-1`` = 2.8819e-02 .. 6899.3
#:
#: **A 6.81-decade gap with NOTHING in it, identical on both builds.**  1e-3
#: sits 5.35 decades above the healthy ceiling and 1.46 decades below the
#: mildest broken row -- two-sided, with well over a decade on the tight side.
#:
#: The bar is deliberately NOT the 1-D peer's ``_STACK_SUPERUNITY_BAR = 1e-2``:
#: that value carries only 0.46 decades here, because the mildest broken row
#: measured is the shipped ``test_structured_stack_energy_floor_nodal`` fixture
#: at 2.8819e-02 -- only 2.9x it.  That fixture is the reason the census had to
#: be measured BEFORE the bar was chosen: it was shipped as a demonstration of
#: the nodal basis's "~1-4% floor", and a 2.9% energy violation on a stack of
#: LOSSLESS media, where ``R + T <= 1`` is a theorem, is not a floor.  It is
#: now refused, and the switch below restores it.
#:
#: KERNEL-STABLE BY CONSTRUCTION.  Every nodal row reads the same
#: ``max(R + T)`` to four significant figures on all four thread counts per
#: build and on both builds: the nodal blow-up is a DETERMINISTIC
#: discretisation defect, not an arithmetic one, so a passivity screen against
#: it does not move with the kernel the way a Class-C energy screen does.
_BOR_NODAL_SUPERUNITY_BAR = 1.0e-3

#: The WARNING edge.  Between this and the refusal bar the answer is returned
#: with a ``UserWarning`` quoting the measured violation -- so a mildly damaged
#: nodal solve is never SILENT, which is what the retired ``Rbig/lambda > 4``
#: proxy allowed.  3.71 decades above the healthy ceiling measured above.
#:
#: ROUND 2 RESTATEMENT OF THAT MARGIN.  "3.71 decades" is against the ONE-SIDED
#: healthy ceiling (4.4336e-09).  On the TWO-SIDED measure this edge now reads
#: -- ``max(|R+T-1|)`` on the accurate family, which is what D2 armed -- the
#: ceiling is **2.714107e-07** (uniform nodal rows with the right channel set
#: and a silent index ceiling; ``r1_passivity_census`` -> ``r4_bars``), so the
#: edge carries **0.57 decades, not 3.71**.
#:
#: It is kept at 1e-6 anyway, and the reason is what this edge COSTS when it is
#: wrong: one ``UserWarning``, never a refusal.  The quantity is also
#: kernel-stable by construction -- the nodal blow-up is a deterministic
#: discretisation defect, not an arithmetic one -- so 3.7x is not the same kind
#: of thin margin a backward-error-driven bar would have at 3.7x.  The number is
#: recorded here rather than left at 3.71 so the next round re-derives from the
#: measurement.
_BOR_NODAL_SUPERUNITY_WARN = 1.0e-6

#: How close to real ``eps`` a layer must be for the stack to count as PROVABLY
#: LOSSLESS -- which is what makes ``R + T = 1`` an EQUALITY and arms the
#: DEFICIT half of the screen.  Relative to the profile's own ``max|eps|`` so
#: it is unit-invariant.
#:
#: 1e-12 admits only round-off, not a deliberately lossy layer (whose
#: ``R + T < 1`` is legitimate and whose deficit the screen must never judge).
#: MEASURED that it is far enough below the deficit bar to be safe: the
#: staggered twin of the shipped refusal fixture absorbs 2e-12 of the incident
#: power at exactly this ratio and 1.6e-09 at 1e-09
#: (``validation/probe_fix_bor_round2/r2_loss_ladder.py``, ladder A's
#: ``staggered`` column) -- nine decades below ``_BOR_NODAL_SUPERUNITY_BAR``,
#: so a stack this predicate calls lossless cannot absorb its way past the
#: deficit bar.
#:
#: ROUND 2 NOTE.  Until round 2 this ONE constant gated the WHOLE screen, in
#: both directions, and that was defect D1: see
#: :func:`_stack_is_provably_passive`.
_BOR_LOSSLESS_REL_IM = 1.0e-12

#: ROUND 2 (D1).  How far below zero a layer's ``Im(eps)`` may sit, relative to
#: that profile's own ``max|eps|``, and still count as ROUND-OFF rather than
#: GAIN.  16 ULP -- the sizing constant the 1-D peer
#: ``pmm/stack._PASSIVE_ANTIHERM_DEADBAND`` uses for the same decision, and
#: which ``pmm/stack._WALL_SNAP_DEADBAND`` uses for its own.
#:
#: It exists for how a permittivity is BUILT, not for how it is typed: a
#: lossless ``n`` squared into an ``eps`` can land an ULP either side of the
#: real axis.  It is nine decades below the smallest gain that could matter
#: (an ``Im(n)`` of 1e-6 on a real part of 4 reads -5e-7, i.e. 1.4e+08 ULP).
_BOR_PASSIVE_DEADBAND = 16.0 * float(np.finfo(float).eps)

#: ROUND 2 (D2) -- THE DETERMINISTIC CONJUNCT, and it carries no bar of its own.
#: How far a returned channel's axial index ``Re qn = Re q / k0`` may exceed the
#: half-space's own index ``Re sqrt(eps)`` before the channel SET is refused.
#:
#: WHY THIS IS EXACT AND NOT CALIBRATED.  ``q^2`` is an eigenvalue of
#: ``eps k0^2 + D`` with ``D`` the transverse operator; wherever ``D`` is
#: negative semi-definite -- the continuum problem, and any div-conforming
#: discretisation of it -- the Rayleigh quotient gives ``q^2 <= max(eps) k0^2``,
#: i.e. ``qn <= n``.  A channel above it has ``gamma^2 < 0``: a transverse
#: eigenvalue the PEC-walled cylinder does not have.  That is a CONTRADICTION,
#: not a magnitude, so it does not move with the BLAS kernel the way an energy
#: excess does, and it detects the half of the nodal defect the energy cannot
#: see (verification D3: 20 of 32 rows the energy screen called OK returned the
#: wrong channel count).
#:
#: 5e-10 is not a new constant: it is the SAME absolute slack the div-conforming
#: twins already apply as their own index ceiling (``bor_stack.solve``'s
#: ``prop()`` at both sites, ``_jax_bor._mask``, ``_jax_sem``:
#: ``Re sqrt(eps) - Re qn > -5e-10``).  The nodal channel gate deliberately does
#: NOT apply it as a FILTER -- audit S1-16 measured that over-filtering the
#: reldiv-screened set degrades the basis's own floor -- so round 2 reads it as
#: a REFUSAL instead, which filters nothing.
#:
#: MEASURED, 160 solves over both bases x {uniform, ring} x m = 0..3 x
#: N = 120/200 x Rbig/lambda = 0.5 .. 8
#: (``validation/probe_fix_bor_round2/r1_passivity_census.py``, guard disarmed):
#:
#:   STAGGERED, all 80 rows          worst ``Re qn / n - 1`` = -3.0273e-05
#:   NODAL, channel set matches the
#:     staggered twin, all 36 rows   worst ``Re qn / n - 1`` = -1.4137e-03
#:   ------------------------------  ----------------------------------------
#:   NODAL, channel set WRONG        +5.0266e-06 .. +1.2966e-03 on 40 of 44
#:
#: The two populations are on OPPOSITE SIDES OF ZERO.  In absolute ``qn`` the
#: mildest violation is 7.1e-06, which is **4.15 decades** above this slack,
#: and the closest approach from below is -4.3e-05, on the other side.  Zero
#: false positives on 116 rows that are not damaged.
_BOR_INDEX_CEILING_SLACK = 5.0e-10


class BORNodalPassivityError(ValueError):
    """Raised by :func:`solve` when the LEGACY NODAL cascade returns
    non-physical ``R + T`` on a stack that is provably passive.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers are
    unaffected.  The message names the basis, quotes the measured violation,
    and names the remedy -- ``basis='staggered'``, which is
    :func:`build_layer`'s default and conserves energy to machine precision at
    any cell size.
    """


def _stack_is_provably_passive(layers):
    """``True`` when ``R + T <= 1`` is a THEOREM for this stack: every layer
    PROVABLY PASSIVE (``Im eps >= 0`` in the library's ``exp(-i omega t)``
    convention, to :data:`_BOR_PASSIVE_DEADBAND`), and a LOSSLESS incidence
    half-space.

    ROUND 2 -- WHAT THIS USED TO SAY, AND WHY IT WAS DEFECT D1.  As first built
    it required every layer to be LOSSLESS to 1e-12 and disarmed otherwise, on
    the stated reasoning *"on a lossy stack there is no theorem to violate."*
    That reasoning covers the BELOW-unity direction only.  For a stack of
    passive media energy conservation reads ``R + T + A = 1`` with the absorbed
    fraction ``A >= 0``, so ``R + T <= 1`` holds on EVERY passive stack,
    absorbing or not -- and the guard had a discontinuity at ``Im(eps) = 0+`` of
    exactly the shape this same wave removed from the EME module.

    MEASURED (``validation/probe_fix_bor_round2/r2_loss_ladder.py``, ladder A):
    the shipped refusal fixture with a relative loss on the ring's high region.
    The returned violation is the SAME 2.9 % across the whole ladder -- the
    nodal excess reads 2.881869e-02 at ``Im/Re`` = 0, at 1e-12, at 3e-12 and out
    to 1e-04 -- while the pre-round-2 predicate refused the first four rungs and
    RETURNED every rung from 3e-12 up.  A "lossless" glass entered as
    ``n = 1.5 + 1e-8i``, or any dispersion fit with a residual imaginary part,
    took the whole guard out.

    WHAT STAYS.  The LOSSLESS requirement on the INCIDENCE half-space is kept,
    because that is where the theorem actually needs it: ``R`` and ``T`` are
    formed from a basis normalised to unit ``|z-flux|`` per mode, and in an
    absorbing incidence medium a mode's flux is not a conserved power, so the
    sums are not power fractions and there is nothing for a bar to mean.  The
    1-D peer ``pmm/stack._stack_provably_passive`` excludes an absorbing
    superstrate for the same reason and records the measured 1.00026 / 1.0152 /
    1.0303 super-unity ladder it legitimately produces.

    GAIN IS NOT PASSIVE and stays outside the screen entirely: a stack with
    ``Im eps < 0`` has no energy theorem in either direction, so both halves
    disarm.  (Behaviour change: an infinitesimal GAIN of -1e-12 relative, which
    the pre-round-2 predicate admitted because it tested ``|Im eps|``, is now
    outside.  That is the peer's convention -- ``_tensor_is_passive`` requires
    ``Im >= 0``, not ``|Im|`` small -- and it is the conservative direction.)

    ANYTHING THIS CANNOT RESOLVE ANSWERS ``False``, which leaves the solve
    exactly as it was.  On this path the permittivity reaches
    :func:`build_layer` as a 1-D complex array of radial samples, for which the
    peer's general statement -- the anti-Hermitian part ``(eps - eps^H) / 2i``
    is positive semi-definite -- reduces EXACTLY to ``Im(eps) >= 0``, sample by
    sample: a scalar (or diagonal) permittivity has ``(eps - eps^H)/2i =
    Im(eps)``.  So the test below IS the peer's test on the only payload the
    nodal cascade accepts; a layer that could not produce the fact answers
    ``None`` and disarms.
    """
    if not layers:
        return False
    for i, L in enumerate(layers):
        rel = L.get("min_rel_im_eps")
        if rel is None or not np.isfinite(float(rel)):
            return False
        if float(rel) < -_BOR_PASSIVE_DEADBAND:      # GAIN
            return False
        if i == 0 and not _layer_is_lossless(L):     # the incidence medium
            return False
    return True


def _layer_is_lossless(L):
    """``True`` when this layer absorbs nothing to round-off, so ``R + T = 1``
    is an EQUALITY rather than an inequality across it."""
    rel = L.get("max_rel_im_eps")
    if rel is None or not np.isfinite(float(rel)):
        return False
    return abs(float(rel)) <= _BOR_LOSSLESS_REL_IM


def _stack_is_provably_lossless(layers):
    """``True`` when EVERY layer is lossless, which is what arms the DEFICIT
    half of the screen (ROUND 2, defect D2).

    Inside a PEC wall a lossless passive stack has no absorption and no exit
    other than the propagating channels of its two half-spaces, so ``R + T = 1``
    is an EQUALITY and a DEFICIT is damage of exactly the same kind as an
    excess.  The shipped screen tested only ``max(R+T) - 1``, and the
    verification measured 5 of 48 provably lossless nodal rows returning
    ``R + T < 1`` silently, the worst at 0.537349.  Reproduced here at
    0.5373486 (``m = 2``, ``Rbig/lambda = 0.5``, ``N = 200``, ring layer;
    ``r1_passivity_census``), whose staggered twin returns 1 to 6.5e-12.
    """
    return all(_layer_is_lossless(L) for L in layers)


def _channel_index_excess(L, qn):
    """``max(Re qn) - Re sqrt(eps_ceiling)`` over one half-space's RETURNED
    channels: how far the worst channel's axial index exceeds the largest index
    the medium has.  Negative on every physical answer.  ``None`` when the layer
    cannot supply its own index ceiling."""
    eps = L.get("eps_ceiling")
    if eps is None:
        return None
    qn = np.asarray(qn)
    if qn.size == 0:
        return None
    try:
        n_max = float(np.real(np.sqrt(complex(eps))))
    except (TypeError, ValueError):               # pragma: no cover - payload
        return None
    if not np.isfinite(n_max) or n_max <= 0.0:
        return None
    return float(np.max(np.real(qn))) - n_max


def _check_nodal_passivity(layers, energy, channels=()):
    """Refuse (or warn about) a non-physical answer from the legacy nodal
    cascade.  No-op on the staggered basis, on a stack with GAIN, on a stack
    with an absorbing incidence medium, and when
    :data:`BOR_NODAL_PASSIVITY_GUARD` is ``False``.

    TWO DETECTORS, ROUND 2, and they are independent:

    1. THE ENERGY SCREEN.  ``R + T <= 1`` on any provably passive stack, and
       ``R + T = 1`` on a provably LOSSLESS one -- so the screen is ONE-SIDED
       where the stack absorbs and TWO-SIDED where it cannot (defects D1 and
       D2).  See :func:`_stack_is_provably_passive`,
       :func:`_stack_is_provably_lossless` and
       :data:`_BOR_NODAL_SUPERUNITY_BAR`.

    2. THE INDEX CEILING, which reads no energy at all: a returned channel whose
       axial index exceeds its own half-space's index has ``gamma^2 < 0``, which
       the PEC-walled cylinder cannot support.  See
       :data:`_BOR_INDEX_CEILING_SLACK`.

    WHY BOTH, MEASURED.  Neither dominates.  On the 160-solve census
    (``validation/probe_fix_bor_round2/r1_passivity_census.py``) the energy
    populations OVERLAP by ten decades when scored against a channel-set
    definition of damage -- nodal rows whose channel set matches the
    div-conforming twin reach ``|R+T-1|`` = 1.95e+03, while rows whose set is
    WRONG come as mild as 2.08e-07 -- so no single scalar bar separates them,
    and saying so is the honest reading of verification D3 ("the bar is a
    detector of catastrophe, not of correctness").  What the energy bar DOES
    have a two-sided gap against is the sub-population the ceiling cannot see:
    among nodal rows with the right channel set and a silent ceiling, the
    accurate family tops out at 2.7141e-07 and the next row up is 2.1730e-02 --
    **4.90 decades with nothing in it**, which is where
    :data:`_BOR_NODAL_SUPERUNITY_BAR` sits.

    ``channels`` is ``[(layer, qn), ...]`` for the half-spaces whose channels
    were returned; the ceiling conjunct reads it and nothing else.
    """
    if not BOR_NODAL_PASSIVITY_GUARD:
        return
    if not any(L.get("basis") == "nodal" for L in layers):
        return
    e = np.asarray(energy, dtype=float)
    if e.size == 0 or not _stack_is_provably_passive(layers):
        return
    nlam = [float(np.real(L.get("Rbig_over_lambda", float("nan"))))
            for L in layers if L.get("Rbig_over_lambda") is not None]
    where = (" (cell radius %.2f vacuum wavelengths)" % (nlam[0],)
             if nlam else "")

    # ---- detector 1: the energy, one- or two-sided ----
    excess = float(np.max(e)) - 1.0
    two_sided = _stack_is_provably_lossless(layers)
    deficit = 1.0 - float(np.min(e))
    worst = max(excess, deficit) if two_sided else excess
    sense = ("a DEFICIT" if (two_sided and deficit > excess) else "an EXCESS")
    law = ("R + T = 1 is an EQUALITY (no absorption, and a PEC wall leaves no "
           "other exit)" if two_sided else
           "R + T <= 1 is a theorem (R + T + A = 1 with the absorbed "
           "fraction A >= 0)")
    if worst > _BOR_NODAL_SUPERUNITY_BAR:
        raise BORNodalPassivityError(
            "bor_solve.solve(basis='nodal'): the cascade returned "
            "max(R + T) = %.6g on a PROVABLY PASSIVE %sstack%s -- a violation "
            "of %.4g against the bar %.0e.  It is %s: min(R + T) = %.6g, "
            "max(R + T) = %.6g, and on this stack %s.  "
            "This is the legacy NODAL FD basis's spurious "
            "divergence-violating mode sea: those modes carry zero z-flux, so "
            "their forward/backward orientation is decided by the sign of "
            "noise, adjacent layers orient near-identical spurious modes "
            "OPPOSITELY, and the interface transmission block acquires a null "
            "vector.  Measured over 93 solves, the nodal basis reads 2.88e-02 "
            "to 6899 here while its STAGGERED twin reads 1 + 3.7e-12 on every "
            "one of the same rows.  REMEDY: basis='staggered' (the "
            "build_layer default, div-conforming Yee), which conserves energy "
            "to machine precision at any cell size.  To restore the previous "
            "behaviour and receive this number instead of this error, set "
            "lumenairy.elements.bor.bor_solve.BOR_NODAL_PASSIVITY_GUARD = "
            "False."
            % (excess + 1.0, "LOSSLESS " if two_sided else "", where, worst,
               _BOR_NODAL_SUPERUNITY_BAR, sense, 1.0 - deficit, excess + 1.0,
               law))

    # ---- detector 2: the index ceiling.  It is exact, so it carries no bar
    #      of its own -- but it is checked AFTER the energy refusal, so a
    #      row the 5.45.1 screen already refuses keeps the message (and the
    #      quoted violation) it has always had, and this detector speaks
    #      only where the energy is silent, which is the population it was
    #      measured to add. ----
    for side, (L, qn) in zip(("incidence", "exit"), channels):
        # The Rayleigh bound is AIRTIGHT only where the half-space is lossless:
        # there ``eps k0^2 + D`` is real symmetric, its spectrum is real, and
        # ``q^2 <= max(eps) k0^2`` exactly.  With a complex ``eps`` the operator
        # is complex-symmetric and the same statement about ``Re q`` is only
        # approximate, so the conjunct stays silent rather than refusing on a
        # bound it cannot prove.  Every half-space of the 160-solve census is
        # lossless, so nothing measured is given up by this.
        exc = _channel_index_excess(L, qn) if _layer_is_lossless(L) else None
        if exc is None or exc <= _BOR_INDEX_CEILING_SLACK:
            continue
        n_max = float(np.real(np.sqrt(complex(L["eps_ceiling"]))))
        raise BORNodalPassivityError(
            "bor_solve.solve(basis='nodal'): the cascade returned a %s "
            "channel at axial index Re(q/k0) = %.9g on a medium whose largest "
            "index is %.9g%s -- an excess of %.4g against the %.0e slack.  An "
            "axial index above the medium's own means gamma^2 < 0, a "
            "transverse eigenvalue this PEC-walled cylinder does not have: "
            "q^2 is an eigenvalue of eps k0^2 + D, and wherever D is negative "
            "semi-definite the Rayleigh quotient bounds q^2 <= max(eps) k0^2.  "
            "The legacy NODAL FD basis is NOT divergence-conforming, so its D "
            "is not, and the channel SET it returns contains modes the medium "
            "cannot carry.  This is refused independently of the energy "
            "because the energy does not see it: measured over 160 solves, 20 "
            "of the rows whose R + T closes to better than 1e-06 return the "
            "wrong number of channels against the closed-form Bessel-zero "
            "count.  REMEDY: basis='staggered' (the build_layer default, "
            "div-conforming Yee), which returns the closed-form count and "
            "whose worst channel sits 3.0e-05 BELOW this ceiling on every row "
            "of that census.  To restore the previous behaviour and receive "
            "the channel set instead of this error, set lumenairy.elements."
            "bor.bor_solve.BOR_NODAL_PASSIVITY_GUARD = False."
            % (side, n_max + exc, n_max, where, exc,
               _BOR_INDEX_CEILING_SLACK))

    if worst <= _BOR_NODAL_SUPERUNITY_WARN:
        return
    warnings.warn(
        "bor_solve.solve(basis='nodal'): R + T spans [%.9g, %.9g] on a "
        "provably passive %sstack%s -- %s of %.3g, above the %.0e warning edge "
        "but below the %.0e refusal bar, where %s.  The legacy nodal FD "
        "basis leaks energy through its divergence-violating spurious mode "
        "sea; basis='staggered' (the default) reads 1 + 3.7e-12 on the same "
        "geometry." % (1.0 - deficit, excess + 1.0,
                       "lossless " if two_sided else "", where, sense, worst,
                       _BOR_NODAL_SUPERUNITY_WARN,
                       _BOR_NODAL_SUPERUNITY_BAR, law),
        stacklevel=3)


def _flux(L, j):
    N = L["N"]
    Er, Ephi = L["W"][:N, j], L["W"][N:, j]
    hr, hphi = L["V"][:N, j], L["V"][N:, j]
    return np.real(np.sum((Er * np.conj(hphi) - Ephi * np.conj(hr)) * L["wq"]))


def _flux_normalize(L):
    """Scale each mode column to unit |z-flux| (propagating) / unit field-norm
    (evanescent), so a flux-normalized S-matrix has ``|S|^2`` = power fraction."""
    L = dict(L)
    W, V = L["W"].copy(), L["V"].copy()
    N, wq = L["N"], np.real(L["wq"])
    for j in range(W.shape[1]):
        P = _flux(L, j)
        # flux threshold RELATIVE to the mode's own r*dr field norm (same
        # measure as the flux -> unit-invariant; absolute 1e-10 silently
        # mis-normalized meter-scale inputs)
        fnrm = np.sum((np.abs(W[:N, j]) ** 2 + np.abs(W[N:, j]) ** 2) * wq)
        s = (1.0 / np.sqrt(abs(P)) if flux_is_strong(P, fnrm, xp=np)
             else 1.0 / np.sqrt(np.sum(np.abs(W[:, j]) ** 2) + 1e-300))
        W[:, j] *= s
        V[:, j] *= s
    L["W"], L["V"] = W, V
    L["flux"] = np.array([_flux(L, j) for j in range(W.shape[1])])
    return L


def build_layer(m, Rbig, N, eps_profile, k0, *, wall="pec", thickness=None,
                basis="staggered"):
    """A flux-normalized layer with its physical-mode flag (reldiv-tagged).

    ``basis='staggered'`` (default) uses the Yee div-conforming discretization
    (spurious-free; the production ``BORStack`` basis) -- the cascade then
    conserves energy to machine precision at any cell size.  ``basis='nodal'``
    keeps the historical FD basis (see the module docstring for why it blows
    up on large cells); its spurious modes are tagged by ``reldiv`` for the
    ``_physical_propagating`` filter.  It now WARNS (audit S1-15) when the cell
    radius exceeds a few vacuum wavelengths -- the regime where the spurious-mode
    sea silently drives the cascade energy up to ~1e29.  The staggered wall is
    the closed Dirichlet wall, so ``wall`` must stay ``'pec'`` there.

    ``thickness`` is validated (audit W6-B11): the ``BORStack.add_layer`` sibling
    has guarded it since P3-10 ("a NEGATIVE thickness flips the propagation
    exponent exp(iqL) so forward-oriented evanescent modes GROW, silently
    destabilizing the Redheffer cascade instead of raising"), but this
    lower-level twin accepted ``thickness=-0.5`` and cascaded it silently.
    """
    if thickness is not None:
        thickness = float(thickness)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "build_layer: thickness must be > 0 and finite (or None for a "
                "semi-infinite half-space), got %r" % (thickness,))
    if basis == "staggered":
        if wall != "pec":
            raise ValueError("basis='staggered' builds in the closed Dirichlet "
                             "wall; wall must be 'pec' (got %r)" % (wall,))
        # already flux-normalized per column inside _layer_modes_staggered;
        # re-running _flux_normalize here would apply the single-grid ``wq``
        # measure to the two-grid basis (the audit-P3-14 half-cell error).
        L = dict(layer_modes(m, Rbig, N, eps_profile, k0, staggered=True))
        W, V = L["W"], L["V"]
        wq_f, wq_n = L["wq_face"], L["wq_node"]
        L["flux"] = np.real(
            np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
            - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
        # div-conforming by construction: no spurious sea to tag.
        L["reldiv"] = np.zeros(W.shape[1])
    elif basis == "nodal":
        # Large-cell blow-up guard (audit S1-15): the nodal FD basis grows a
        # divergence-violating spurious-mode sea (~40-50% of the basis at
        # Rbig ~ 12 vacuum wavelengths) whose zero-z-flux modes are oriented by
        # the sign of noise, driving the interface transmission block singular
        # (cond ~ 2.6e15) and blowing the cascade energy up to ~1e29 -- silently.
        # 5.45.1: this ``Rbig/lambda > 4`` test is RETIRED AS A DECISION and
        # kept only as an early, cheap hint.  It is a PROXY, and the scoping
        # measured how badly it misses its own population: at 1, 2 and 4
        # vacuum wavelengths the same five-layer stack reads max(R + T) =
        # 3.05, 114.4 and 37.91 and this warning does not fire, while a
        # UNIFORM nodal stack at 12 wavelengths -- which it does warn about --
        # reads 1.035.  The DECISION now lives in solve(), on the measured
        # passivity violation itself (_check_nodal_passivity).  Nothing keys
        # on the number below.
        rbig_lambda = float(np.real(Rbig)) * float(np.real(k0)) / (2.0 * np.pi)
        if rbig_lambda > 4.0:
            warnings.warn(
                "build_layer(basis='nodal'): the cell radius Rbig is "
                f"{rbig_lambda:.1f} vacuum wavelengths; the nodal FD basis "
                "develops a spurious divergence-violating mode sea on large "
                "cells that can render the interface transmission block singular "
                "and blow the cascade energy up to ~1e29.  Use the default "
                "basis='staggered' (div-conforming Yee), which conserves energy "
                "to machine precision at any cell size.",
                stacklevel=2)
        # S1-18: harvest the divergence tag from the SAME dense eig
        # ``layer_modes`` already runs (``with_reldiv=True``) instead of a
        # second byte-identical ``radial_coupled_modes`` eigensolve.  The two
        # nodal paths assemble byte-identical K/B, so ``reldiv`` is unchanged.
        _Lm = layer_modes(m, Rbig, N, eps_profile, k0, wall=wall,
                          with_reldiv=True)
        _reldiv = _Lm["reldiv"]
        L = _flux_normalize(_Lm)
        L["reldiv"] = _reldiv
    else:
        raise ValueError("basis must be 'staggered' or 'nodal' (got %r)"
                         % (basis,))
    L["thickness"] = thickness
    # S1-16: store the layer's index ceiling (the eps of maximum real part over
    # the radial profile) as a per-layer REFERENCE -- it is the axial-index
    # bound q/k0 <= sqrt(eps) that the staggered twins (bor_stack.solve's prop()
    # and _jax_bor._mask) enforce.  It is recorded here for cross-checks but is
    # DELIBERATELY NOT applied by the nodal ``_physical_propagating`` (whose
    # unique leg is reldiv): forcing the ceiling onto the nodal FD basis
    # over-filters and degrades its ~4% energy floor.  For the homogeneous
    # super/substrate (the only layers _physical_propagating ever classifies)
    # this is exactly that medium's eps.
    _eps_arr = np.asarray(eps_profile(L["r"]) if callable(eps_profile)
                          else eps_profile, dtype=complex).ravel()
    L["eps_ceiling"] = complex(_eps_arr[int(np.argmax(_eps_arr.real))])
    # The two facts the 5.45.1 passivity screen needs and could not otherwise
    # recover from the returned dict: WHICH basis built this layer, and whether
    # its permittivity is lossless (so ``R + T <= 1`` is a theorem rather than
    # merely a hope).  Both are recorded here because this is where the
    # profile is in scope; neither is read by anything else.
    L["basis"] = basis
    # ROUND 2 (D1): TWO facts, not one, because the screen asks two questions.
    # ``max_rel_im_eps`` answers "is this layer LOSSLESS", which decides whether
    # R + T = 1 is an EQUALITY (the deficit half of the screen).
    # ``min_rel_im_eps`` answers "is this layer PASSIVE" -- SIGNED, so GAIN is
    # distinguishable from loss, which is the distinction the pre-round-2
    # predicate collapsed by testing |Im eps|.  Both are relative to the
    # profile's own ``max|eps|`` (the complex modulus, not ``max|Re eps|``: a
    # near-ENZ or plasmonic profile can have a vanishing real part while being
    # perfectly well scaled), so both are unit- and amplitude-invariant.
    #
    # For this path's payload -- a 1-D complex array of radial samples -- the
    # 1-D peer's general passivity statement (the anti-Hermitian part
    # ``(eps - eps^H) / 2i`` is positive semi-definite,
    # ``pmm/stack._segment_passive``) reduces exactly to ``Im(eps) >= 0``
    # sample by sample, so ``min_rel_im_eps >= -deadband`` IS that test here.
    # An empty or non-finite profile answers ``inf`` / ``nan`` and disarms the
    # screen, which is the conservative direction.
    _den = float(np.max(np.abs(_eps_arr))) if _eps_arr.size else 0.0
    L["max_rel_im_eps"] = (float(np.max(np.abs(_eps_arr.imag))) / _den
                           if _den > 0.0 else float("inf"))
    L["min_rel_im_eps"] = (float(np.min(_eps_arr.imag)) / _den
                           if _den > 0.0 else float("nan"))
    L["Rbig_over_lambda"] = (float(np.real(Rbig)) * float(np.real(k0))
                             / (2.0 * np.pi))
    return L


def _physical_propagating(L, k0, reldiv_tol=0.5):
    # Dimensionless q/k0 classifier (audit P2-06): absolute thresholds on q
    # (units 1/length) silently emptied the propagating set for small-k0 unit
    # systems.
    #
    # AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13: the P2-06 constant
    # (0.05) was an ANGULAR cutoff that dropped genuinely propagating
    # near-grazing orders (energy leak 2.28e-2 on the ring-grating
    # reproducer).  The real-axis floor guards ONLY the q ~ 0 degenerate
    # point (1e-6); kept modes sit >= 4 decades above the flux normalizer's
    # field-norm fallback (P/fnrm = qn for the limiting family), so kept
    # implies flux-normalized.
    #
    # S1-16 (audit AUDIT_V5_24_2): the three BOR mode classifiers share a
    # {imag, real-floor, index-ceiling} CORE.  This one previously carried
    # the reldiv leg but NOT the index ceiling, while the staggered twins
    # (bor_stack.solve's prop() and _jax_bor._mask) carried the ceiling but
    # NOT reldiv -- so the "keep all three in lockstep" comment was false.
    # The reldiv leg is UNIQUE to this classifier on purpose: it filters the
    # divergence-violating spurious sea of the optional NODAL basis (staggered
    # sets reldiv == 0, so the leg is a no-op there); the twins are
    # staggered-only (div-conforming, spurious-free) and deliberately skip
    # the reldiv eigensolve.  The index ceiling (q/k0 <= sqrt(eps)) that the
    # staggered twins carry is DELIBERATELY NOT replicated on the nodal basis:
    # applying it here over-filters the reldiv-screened FD mode set and
    # degrades the documented ~4% nodal energy floor (measured 4% -> 10.7% on
    # test_structured_stack_energy_floor_nodal).  So S1-16 is resolved by making
    # this comment TRUE -- the three classifiers share the {imag, real-floor}
    # core and each carries ONE basis-specific leg (nodal: reldiv; staggered
    # twins: index-ceiling) -- rather than forcing a numeric lockstep the bases
    # do not physically share.
    qn = L["q"] / k0
    keep = channel_core(qn, xp=np) & (L["reldiv"] < reldiv_tol)
    return keep


def solve(layers, k0):
    """Cascade a list of ``build_layer`` layers (first/last = semi-infinite
    super/substrate) and return per-incident-mode R/T efficiencies over the
    physical propagating channels, plus the global S-matrix.

    Returns a dict: ``S`` (S-matrix), ``inc`` (superstrate physical-prop indices),
    ``out`` (substrate ...), ``R``/``T`` (arrays over ``inc``: total reflected /
    transmitted power fraction), ``energy`` (R+T per incident mode).

    Audit W6-B11: a MIDDLE layer left at the ``build_layer`` default
    ``thickness=None`` used to die inside ``propagation_smatrix`` with a bare
    ``TypeError: unsupported operand type(s) for *: 'complex' and 'NoneType'``.
    """
    if len(layers) < 2:
        raise ValueError(
            "bor_solve.solve needs at least the two semi-infinite half-spaces "
            "(got %d layer(s))" % (len(layers),))
    for i in range(1, len(layers) - 1):
        if layers[i].get("thickness") is None:
            raise ValueError(
                "bor_solve.solve: middle layer %d has no thickness -- pass "
                "thickness=... to build_layer for every layer between the "
                "half-spaces." % (i,))
    S = interface_smatrix(layers[0]["W"], layers[0]["V"],
                          layers[1]["W"], layers[1]["V"])
    for i in range(1, len(layers) - 1):
        S = redheffer_star(S, propagation_smatrix(layers[i]["q"],
                                                  layers[i]["thickness"]))
        S = redheffer_star(S, interface_smatrix(layers[i]["W"], layers[i]["V"],
                                                layers[i + 1]["W"],
                                                layers[i + 1]["V"]))
    S11, S12, S21, S22 = S
    inc = np.where(_physical_propagating(layers[0], k0))[0]
    out = np.where(_physical_propagating(layers[-1], k0))[0]
    R = np.array([np.sum([abs(S11[jp, j]) ** 2 for jp in inc]) for j in inc])
    T = np.array([np.sum([abs(S21[jp, j]) ** 2 for jp in out]) for j in inc])
    # PASSIVITY SCREEN (5.45.1, two-sided plus the index ceiling in ROUND 2).
    # On the legacy nodal basis this cascade returned R + T from 2.9e-02 to
    # 6899 on provably passive lossless stacks, and below four vacuum
    # wavelengths it returned it with NO warning at all.  See
    # _BOR_NODAL_SUPERUNITY_BAR for the energy populations and the 4.90-decade
    # gap between them, _BOR_INDEX_CEILING_SLACK for the deterministic conjunct
    # that reads the channel SET instead, and _check_nodal_passivity for why
    # both are needed.
    _check_nodal_passivity(
        layers, R + T,
        channels=((layers[0], layers[0]["q"][inc] / k0),
                  (layers[-1], layers[-1]["q"][out] / k0)))
    return dict(S=S, inc=inc, out=out, R=R, T=T, energy=R + T,
                q_inc=layers[0]["q"][inc])
