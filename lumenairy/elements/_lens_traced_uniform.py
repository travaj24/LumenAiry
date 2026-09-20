"""
lumenairy.elements._lens_traced_uniform -- UNIFORM (Airy) dark-side completion
of the multibranch ray-density caustic field (niche N16 / K4).

The K1 multibranch KMAH ray-density sum
(:func:`lumenairy.elements._lens_traced_multibranch.apply_real_lens_traced_multibranch`)
is a purely GEOMETRIC (ART) construction: on the DARK side of a fold caustic no
real ray branch exists, so the sum is identically ZERO there and misses the
exponentially-decaying Airy tail (fold-truth: windowed r2m ~15% low, ~20% of the
caustic energy missing).  This module closes that gap by the Chester-Friedman-
Ursell / Ludwig UNIFORM asymptotic: near a fold the field is one Airy function
``Ai(-k^{2/3} zeta)`` -- oscillatory on the bright side (``zeta > 0``) and an
``Ai(+)`` exponential tail on the DARK side (``zeta < 0``).

Method (the ROBUST fit-and-continue route of plan N16, for a rotationally-
symmetric fold RING -- the caustic_fold_ref regime):

  1. Run the K1 multibranch to get the finite, ludwig-regularized BRIGHT-side
     field (interior + near-caustic).  This module only ADDS the dark tail.
  2. A meridional ray trace (via :mod:`lumenairy.raytrace`) of the same
     prescription to the output plane gives the fold geometry EXACTLY: the
     caustic-ring radius ``r_c`` (the turning point of the ray height
     ``y_obs(h)``), the fold parameter ``zeta(r) = [3/4 (S+ - S-)]^{2/3}``
     (linear in ``r_c - r`` near the fold -- ``zeta = kappa (r_c - r)``, from the
     eikonal difference of the two coalescing branches), and the mean phase
     ``A(r) = (S+ + S-)/2`` (a smooth quadratic).
  3. The two CFU amplitude coefficients ``a0, a1`` are SMOOTH (real-analytic)
     through the caustic but are hard to get accurately from the raw geometric
     ray-tube amplitudes near a sharp/asymmetric fold, so they are FIT (complex
     least squares) to the multibranch BRIGHT field over a band just inside
     ``r_c``, using the exact ``uniform_fold_airy`` CFU kernel
     (:func:`lumenairy.elements.lenses_maslov._fold_airy_eval`) as the basis.
  4. The SAME CFU kernel is then evaluated at ``zeta < 0`` (analytic
     continuation through the caustic) to fill the dark-side pixels with the
     exponential Airy tail -- reusing ``_fold_airy_eval`` verbatim for both
     sides (no divergent reimplementation).

A rotationally-symmetric SINGLE **CUSP** ring (``n_turn == 2`` -- a 3-branch
coalescence at finite radius) is ALSO completed, via the :func:`pearcey`
generalisation (niche R2 / A1): the local ray geometry is mapped to the Pearcey
normal form ``t^4 + x t^2 + y t`` and the cusp-finite Pearcey field replaces the
multibranch in the cusp zone (see the CUSP section below).  Everything else
falls back.

Scope / fallbacks (documented, never inf/nan):
  * A rotationally-symmetric SINGLE fold RING (``n_turn == 1``, the
    caustic_fold_ref class) is completed by the fold-Airy path; a single finite-
    radius CUSP ring (``n_turn == 2``) by the Pearcey path.  A decentered/tilted
    prescription, a non-rotationally-symmetric input, a carrier tilt, a
    near-axial (Bessoid) cusp, three-or-more coalescing rings, or no clean fold
    -> the module DETECTS the case and falls back to the plain multibranch field
    with a one-time warning.
  * If the bright-side fit residual is large (the uniform fold model does not
    describe the field), or the cusp control-solve / linear-map residual is too
    large (not a clean single cusp), it falls back too.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from .. import raytrace as rt

# ``_lens_kernels`` is a leaf (stdlib + numpy only); the helper attributes a
# notice to the caller's frame on every call path.
from ._lens_kernels import caller_stacklevel as _caller_stacklevel
from ._lens_traced_multibranch import (
    _ENERGY_BLOWUP_FACTOR,
    _ENERGY_COLLAPSE_FACTOR,
    _PIXEL_CONTINUITY_MAX,
    _PIXEL_CONTINUITY_MIN,
    _multibranch_render,
    half_pitch_centres,
)
from .lenses_maslov import _fold_airy_eval, pearcey

__all__ = ['apply_real_lens_traced_uniform']

# Airy argument cap for the dark-side fill: ``Ai(50) ~ 1e-73`` (numerically
# zero) so clamping ``-k^{2/3} zeta`` at this bound leaves the physical tail
# untouched while preventing any overflow far out on the grid.
_AIRY_ARG_CAP = 50.0

# Radial extent of the dark-side fill, in Airy lengths ``l_airy`` past the
# caustic ring.  ``Ai(x)`` decays as ``exp(-2/3 x^{3/2})``: it is 1.1e-10 of
# its value at the caustic by 15 l_airy and 4.0e-16 by 20, which is below the
# double-precision resolution of any field this module returns -- so filling
# past that writes zeros in an expensive way.  20 leaves four decades of
# margin over the 15 the physical tail needs.
#
# WP-B7b section 7 proposed clipping this where the extrapolated ``zeta`` no
# longer describes the tail, on the premise that the completion's energy error
# is written into the OUTER part of the annulus.  MEASURED (WP-B7c,
# 2026-09-14, the direct Rayleigh-Sommerfeld oracle, seven planes of the
# N-BAF10 fixture), the premise does not hold.  Cumulative dark-side energy of
# the completed field against the oracle's in the same annulus, by fill depth
# in Airy lengths:
#
#   W/band      1 cell   3 cells   20 cells
#   531.5        2.321     1.995      1.840
#    14.0        2.438     2.169      2.045
#     5.65       1.535     1.383      1.324
#     3.01       1.231     1.134      1.097
#     1.03       0.937     0.892      0.877
#     0.42       0.928     0.946      0.936
#
# -- flat in depth to within 6 % beyond 3 cells, so more than 94 % of the
# excess is written INSIDE 3 Airy lengths, where the tail is physical (``Ai``
# is still 1e-3 of the ring there) and clipping it would trade an energy error
# for a shape error.  The depth is not the lever.  The lever is the
# extrapolated ``zeta`` itself: the effective decay constant read off the
# ORACLE's own dark side, ``kappa_eff``, runs 1.068 / 0.803 / 0.654 / 0.603 /
# 0.558 / 0.489 of the fitted ``kappa`` as W/band runs 0.42 / 1.03 / 3.01 /
# 5.65 / 14.0 / 531.5 -- the linear normal form is good to 7 % only while the
# two-branch band is at least ~2 Airy lengths wide, and 20-50 % slow beyond
# that.  A second fold parameter (or a two-plane fit) is the repair; a shallower
# fill is not.  ``zeta_linear_range`` in the diagnostics reports the band's own
# curvature bound for a caller who wants to see it (see
# ``_trace_meridional_fold``); it is NOT applied to the fill, because on the
# five optics measured it ranges over 6.9 .. 144 Airy lengths and, where it is
# shorter than this constant, everything it would remove is below 2e-6 of the
# ring amplitude.
_AIRY_TAIL_CELLS = 20.0

# How far past the TWO-BRANCH BAND the fold's zeta(r) = kappa (r_c - r) is
# carried before the completion is flagged as extrapolated.
#
# ``kappa`` is fitted on the band of radii reached by BOTH coalescing branches
# -- the only radii where the eikonal difference S+ - S- that DEFINES zeta
# exists.  The bright-side (c0, c1) fit then runs over a band of half-width
# ``W = l_airy`` inside r_c, and the dark fill runs out to
# ``_AIRY_TAIL_CELLS * l_airy`` past it; both evaluate zeta at radii the linear
# fit never saw.  When the two-branch band is much narrower than ``W`` the
# completion is an extrapolation, and the error it makes is in the ENERGY it
# writes into the dark tail, not in its shape.
#
# MEASURED (WP-B7b, 2026-09-14) against a brute-force Rayleigh-Sommerfeld
# oracle on an exact conic raytrace, total power of the completed field
# relative to the oracle's, on a ladder of planes through one caustic
# (plano-convex R = 2.7 mm, t = 1.0 mm, 1.5 mm aperture, n = 1.5168,
# lambda = 1.31 um, N = 512, dx = 3.0 um) plus two more singlets -- N-LAK22
# biconvex R = +/-3.0 mm, t = 0.55 mm, 1.20 mm aperture, lambda = 850 nm,
# N = 640, dx = 2.10 um at its MARGINAL focus (the widest two-branch band
# measured), and N-LAK22 biconvex R = +/-6.0 mm, t = 0.90 mm, 1.10 mm
# aperture, lambda = 1.55 um, N = 320, dx = 3.85 um, z = 4.400 mm (the
# narrowest):
#
#   W / band    0.16    1.9     2.3     3.4     5.4     9.8     453.6
#   power       0.947   0.975   0.971   1.030   1.049   1.125   1.228
#
# i.e. -5.3 % .. +4.9 % up to 5.4 and +12.5 % .. +22.8 % from 9.8 up.  The bar
# sits in that measured gap.  It does NOT change the returned field or the
# route: the completion still beats the multibranch it would otherwise fall
# back to at every one of those planes (fidelity 0.944 / 0.957 / 0.933 /
# 0.969 / 0.972 / 0.962 / 0.930 against 0.882 / 0.805 / 0.856 / 0.853 / 0.884 /
# 0.887 / 0.832), so falling back would be a regression.  It only says so.
#
# The bar is a CONSERVATIVE FLAG, not a calibrated 5 % / 10 % boundary: how
# much energy a given extrapolation costs is optic-dependent.  A second ladder
# through ONE caustic of a single optic (VERIFY-B7b, 2026-09-14; N-BAF10
# biconvex R = +/-2.6 mm, t = 0.70 mm, 0.90 mm aperture, lambda = 1.064 um,
# N = 512, dx = 2.20 um, seven planes z = 1.5654 .. 1.7424 mm, same oracle
# construction) reads
#
#   W / band    0.42    0.74    1.03    3.01    5.65    14.0    27.3    531.5
#   power       0.989   1.019   0.975   0.986   1.011   1.049   1.045   1.041
#   fidelity    0.983   0.986   0.985   0.983   0.983   0.976   0.977   0.978
#
# -- the same monotone drift with the ratio, but saturating near +5 % instead
# of running to +23 %, so on that optic every rung including 531.5 is inside
# the +/-5 % band and the warning above the bar is a false positive for
# absolute energy.  It still separates the two regimes in the right direction
# on both ladders.
#
# RE-DERIVED TWO-SIDED (WP-B7c, 2026-09-14) on THREE optics and 30 fold planes
# against the same direct Rayleigh-Sommerfeld oracle
# (``validation/oracles/caustic_fold_truth.py``; the probes and their JSON are
# in ``validation/probe_multibranch_zeta/``): N-BAF10 biconvex R = +/-2.6 mm,
# t = 0.70 mm, 0.90 mm aperture, lambda = 1.064 um, N = 512, dx = 2.20 um;
# N-BK7 plano-convex R = -2.0 mm with the FLAT side first, t = 0.80 mm,
# 0.90 mm aperture, 780 nm, N = 512, dx = 2.00 um; N-SF11 biconvex
# R = +/-3.4 mm, t = 0.90 mm, 1.20 mm aperture, 1.55 um, N = 512, dx = 3.00 um.
# The bar is confirmed where it stands, but what it separates is not what the
# 5 % / 10 % reading above says:
#
#   BELOW the bar (22 rungs, W/band 0.29 .. 5.65) the energy error is SIGNED
#   and centred -- -2.18 % .. +4.14 %, 12 of the 22 rungs NEGATIVE, mean
#   +0.57 %;
#   ABOVE it (8 rungs, W/band 11.05 .. 3484) it is a one-sided GAIN at every
#   rung -- +3.64 % .. +30.1 %, 8 of 8 POSITIVE, mean +7.88 %.
#
# ONLY THE ABOVE-BAR HALF OF THAT SURVIVES RE-DERIVATION, and the constant is
# documented on that half alone (VERIFY-WP-B7c, 2026-09-14; 30 fold rungs on
# four optics WP-B7c never used, same oracle, completed-field power against
# the oracle's):
#
#   ABOVE the bar (13 rungs, W/band 10.8 .. 1779): +3.92 % .. +12.43 %, 0 of
#   13 negative, mean +8.20 % -- the one-sided gain REPRODUCES, mirroring
#   WP-B7c's 0 of 8;
#   BELOW the bar (17 rungs, W/band 0.64 .. 7.64): -1.02 % .. +28.44 %, 1 of
#   17 negative, mean +7.00 % -- NOT signed, NOT centred, and the LARGEST
#   excursion in either study (+28.4 %, W/band 1.93) is below the bar, not
#   above it.  Excluding two rungs contaminated by the section-1 quadrature
#   defect the below-bar range is -1.02 % .. +17.39 %, 1 of 15.
#
# So: past 8.0 the completion writes a ONE-SIDED energy GAIN, on seven optics
# and two builds.  It does NOT follow that below 8.0 the error is small or
# centred -- it is one-sided there too on four of the seven, and larger.  The
# bar marks where the gain becomes systematic, and nothing about the
# below-bar population.  ``sqrt(5.65 * 11.05) = 7.90`` remains the arithmetic
# that PLACED 8.0 on WP-B7c's three optics, and is recorded as provenance,
# not as a two-sided calibration: the "last signed rung 5.65" that anchors its
# lower end is a property of those three optics only.  For absolute energy on
# EITHER side of the bar, read the ray-to-wave hand-off, which conserved the
# launched power to better than 1 % at every plane in either study.
_ZETA_EXTRAPOLATION_MAX = 8.0

# Refusal bar on the MULTIBRANCH field this completion is built on.
#
# The completion runs ``apply_real_lens_traced_multibranch`` first and keeps its
# bright side verbatim, so a multibranch field that is wrong by decades makes
# the completion wrong by decades -- and NONE of this module's own diagnostics
# can see it: at the planes below the bright-side CFU fit residual reads
# 0.0029-0.0070 (healthy), ``zeta_extrapolation`` reads 0.27-0.38 (the best
# case the docstring's own table records) and ``fell_back`` reads False, on a
# field carrying up to 6097x the launched power.  That is the defect VERIFY-B7b
# recorded as R-5.
#
# WHAT GOES WRONG is in the multibranch rasteriser, not here: its point-sampled
# area quadrature is an unbiased estimator of the launched energy only while
# the mapped triangles are spread over many pixels, and near the AXIAL point
# focus a whole ring collapses onto a handful of them
# (``_lens_traced_multibranch._ENERGY_BLOWUP_FACTOR`` carries the mechanism and
# its three controls).  Bounding the reconstruction would mean a different
# quadrature and would move every multibranch field near a caustic, so this
# module REFUSES instead: falling back is not a remedy, because the fallback
# target IS the blown-up multibranch field (measured: at z = 1770 um on the
# fixture below the module already falls back with reason='zeta_nonlinear' and
# returns a field carrying 6459x the launched power).
#
# MEASURED (WP-B7c, 2026-09-14) against the direct Rayleigh-Sommerfeld oracle
# ``validation/oracles/caustic_fold_truth.py`` on FIVE optics -- N-BAF10
# biconvex R = +/-2.6 mm / 1.064 um / N = 512 / dx = 2.20 um (VERIFY-B7b's own
# fold fixture), N-BK7 plano-convex R = -2.0 mm flat-side-first / 780 nm /
# N = 512 / dx = 2.00 um, N-SF11 biconvex R = +/-3.4 mm / 1.55 um / N = 512 /
# dx = 3.00 um, and WP-B7b's own two N-LAK22 singlets -- over 51 fold planes
# spanning ``zeta_extrapolation`` 0.16 to 3484:
#
#   * all 42 planes the oracle accepts (completed-field fidelity 0.883-0.999)
#     read a bracketed ratio in 0.816 .. 1.246, and only TWO of the 42 exceed
#     1.0 at all (1.001 and 1.246).  Finer scans without oracle scoring extend
#     that population down to 0.763 and add nothing above 1.25;
#   * the 9 planes where the completion is broken (fidelity 0.012 .. 0.350)
#     read 5.848 / 18.86 / 93.5 / 1683 / 2346 / 2801 / 3201 / 3443 / 3545 and
#     up.  The smallest of them, 5.848, is WP-B7b's own fast singlet one
#     micron past its marginal focus (z = 2060 um, fidelity 0.350);
#   * no plane of THAT POPULATION reads between 1.246 and 5.848.
#
# WHAT THAT POPULATION IS, AND WHAT IT IS NOT (VERIFY-WP-B7c, 2026-09-14,
# re-derived on four optics WP-B7c never used -- a cemented doublet, a
# positive meniscus, a convex-first plano-convex at 532 nm and a fast N-LASF9
# biconvex -- 71 planes, both builds).  The five optics above are all
# UNDERCORRECTED biconvex-or-plano singlets, each read on one grid.  Widen the
# population and the two clusters OVERLAP: the largest ACCEPTED reading is
# 0.9804 (meniscus, z = 2201.74 um, fidelity 0.9798) and the smallest BROKEN
# one 1.151 on fold rings (fast singlet, z = 1076 um, fidelity 0.858, power
# 1.284x the oracle's, with ``power_ratio_decision = 'ok'``, ``fell_back =
# False`` and ``zeta_extrapolation = 1.93``) or 1.106 over every plane the
# guard can fire on (doublet, z = 5400 um, fidelity 0.837).  The gap is
# 1.13x-1.17x, not 4.69x, and the interval (1.25, 5.85) is populated
# CONTINUOUSLY once the planes are stepped finely.  The "empty gap" was an
# artefact of plane sampling.
#
# SO THIS BAR IS A FAR-TAIL TRIPWIRE, NOT A CLASSIFIER.  It has no measured
# margin above: the smallest broken reading sits 1.74x INSIDE it.  It is left
# at 2.0 deliberately -- it is the only bar on this quantity in the library
# (``_ENERGY_BLOWUP_FACTOR`` itself, so the completion refuses exactly the
# fields the branch sum has already declared unphysical, and the two cannot
# drift apart), and lowering it toward the measured broken floor of 1.106
# would start refusing fields the oracle accepts at 0.98.  Its margin BELOW
# the accepted population is 2.04x and that is the only margin it has.
# WHAT DECIDES the cases inside it is the second arm,
# ``_MB_PIXEL_CONTINUITY_MAX`` below -- which reads the quadrature's
# convergence rather than its energy against a launch, and which does refuse
# the 1.151 plane above.
#
# The ratio is read through the SAME bracket the multibranch's own gain arm
# uses (``min(power_ratio, power_ratio_triangles)``, i.e. ``p_out / p_in_hi``),
# so the known false-positive class it was introduced for -- an aperture much
# wider than the grid, where the node-count denominator alone reads up to 3.3x
# with the energy conserved to 1 % -- cannot reach this bar either.  The PRICE
# of that bracket is a detection floor, and it is stated here rather than left
# implicit (VERIFY-WP-B7c D4): on the very geometry it exists for, the two
# denominators separate by up to 7.85x (20 planes of the delta audit's D3
# fixture -- 6 mm aperture on a 1.2 mm grid at ray_subsample=8 -- where
# ``power_ratio`` reads 2.02-3.76 while ``power_ratio_triangles`` reads
# 0.45-0.90, and the completion correctly returns).  So on a geometry with
# that spread, the smallest GAIN this arm can see is not 2.0 but 2.0 times the
# spread, i.e. up to ~15.7x.  The bracket is still right -- the alternative is
# spurious refusals on a legitimate geometry -- but a caller reading
# ``multibranch_power_ratio`` against ``multibranch_power_ratio_bracketed``
# can see the spread on their own plane, and the pixel-continuity arm below
# has no such denominator at all.
_MB_POWER_RATIO_MAX = _ENERGY_BLOWUP_FACTOR

# Lower arm of the same reading, REPORTED but not refused.  A multibranch field
# that loses energy is the NORMAL input to this module: the dark-side Airy tail
# it exists to add is exactly what the branch sum drops, and the measured
# legitimate population runs down to 0.78 (fixture V, z = 1754 um, completion
# fidelity 0.991) against the module's own documented 0.887.  There is no
# measured pathological counterpart on the loss side -- the one that exists,
# the total collapse to an identically zero field, is already REFUSED inside
# the multibranch -- so a refusal here would be a bar with a gap on one side
# only.  ``_ENERGY_COLLAPSE_FACTOR`` (0.5, derived there) is reused as the
# reporting threshold so the two modules classify the same field the same way.
_MB_POWER_RATIO_MIN = _ENERGY_COLLAPSE_FACTOR

# The PIXEL-HALVING ARBITER's bar -- the SECOND arm of the same decision, and
# the one that decides the cases the reading above cannot.
#
# WHY A SECOND ARM.  ``_MB_POWER_RATIO_MAX`` reads the branch sum's deposited
# power against the power its launch congruence carries onto the grid.  That
# denominator makes the reading a mixture: a fold plane legitimately LOSES the
# dark-side tail the completion exists to add (the accepted population runs
# down to 0.74), triangles that straddle the grid boundary bias it the other
# way, and light that leaves the window is not a defect.  Re-derived
# independently on four optics the WP-B7c study never used
# (VERIFY-WP-B7c, 2026-09-14), the accepted and broken populations of that
# reading OVERLAP: largest accepted 0.9804 against smallest broken 1.106
# (all planes) / 1.151 (fold rings), a gap of 1.13x under a 2.0 bar, with a
# plane at 1.151 returning a field of oracle fidelity 0.858 and
# ``power_ratio_decision='ok'``.  A bar on that quantity is a far-tail
# tripwire, not a classifier, and there is nowhere for it to move: lowering it
# to 1.10 would start refusing accepted fields at 0.98.
#
# WHAT THIS ARM READS INSTEAD is the quadrature's CONVERGENCE IN THE PIXEL --
# the deposited power of this render over that of the same mapped triangles
# rasterised at half the pitch on the same window
# (``_lens_traced_multibranch._PIXEL_CONTINUITY_MAX`` carries the mechanism).
# Every term that made the reading above a mixture cancels: both renders drop
# the same dark tail, straddle the same boundary and lose the same off-screen
# light.  What is left is the pixel-area scaling of a point-sampled quadrature
# that has stopped being unbiased.
#
# WHERE IT IS READ.  On the field this call RETURNS -- the completed fold
# field where the completion applies, the plain branch sum on every fallback
# path -- not on the branch sum unconditionally.  The half-pitch render the
# branch sum already paid for is completed with the SAME fold parameters
# (``r_c``, ``kappa``, the cone phase and the two fitted CFU coefficients), so
# the comparison isolates the RASTERISATION and adds no second least-squares
# fit's noise to it.  Reading the branch sum instead would refuse fields that
# are right: on VERIFY-B7b's own fixture at z = 1761 um the branch sum reads
# 1.0636 and the completion built on it reads 1.0020, at oracle fidelity
# 0.9878.
#
# RE-MEASURED (WP-B7c round 3, 2026-09-19) against a band-limited ANGULAR
# SPECTRUM at full radius, on **1304 oracle-scored planes over sixteen
# prescriptions** -- round 2's eight, the f/1.2 optic round 2 excluded, the
# five its verification added, and three new (a true even-ASPHERE, a
# CONCAVE-FIRST meniscus, a plano-convex at NA 0.41).  Probes, populations
# and JSON in ``validation/probe_wp_b7c_round3/``; the derivation lives with
# ``_lens_traced_multibranch._PIXEL_CONTINUITY_MAX`` and the tables in
# ``WP-B7c_ROUND3_REPORT.md``:
#
#   * the 250 fold-ring planes RETURNED read 0.8724 .. 1.0595, fidelity
#     0.9421 .. 0.9985;
#   * the 144 REFUSED read 1.0643 .. 3.9988, fidelity 0.0018 .. 0.9520.
#
# Round 2's two statements about this bar do NOT survive that population and
# are corrected there: the two fidelity populations OVERLAP at sixteen optics
# (worst returned 0.9421 against best refused 0.9520), and the bar's margin
# is a property of the z ladder rather than of the quantity -- refining the
# same optics' ladders 30x and then 10x takes the fold-ring gap from 1.371x
# to 1.032x to 1.0045x with no sign of a floor.  The bar is KEPT at 1.06,
# which is the geometric centre of its own gap (1.0600253) on the round-3
# population, and it is derived there from what it COSTS: on the fold ring,
# 1 false refusal and 2 misses at an accept criterion of fidelity 0.95 over
# 394 planes.
#
# WHAT IT COSTS AND WHAT IT DOES NOT.  One extra rasterisation of the SAME
# mapped triangles -- never a second ray trace, a second KMAH pass or a second
# meridional fold trace (measured 1.6x-2.7x of the branch sum alone, and less
# of the completion, which also pays for a 4000-ray meridional trace and a
# least-squares fit).  ``apply_real_lens_traced_multibranch`` does not ask for
# it and is unchanged in cost and in bits.
#
# WHAT IT DOES NOT CATCH.  On FALLBACK planes -- where the module has already
# warned that the completion does not apply and the field returned is the
# bright-side-only branch sum -- the returned and refused fidelity
# populations DO overlap, because what is wrong there is the missing dark
# tail and not the quadrature.  Measured (WP-B7c round 3, 2026-09-19) on the
# **476 fallback planes the shipped bars RETURN**: fidelity 0.5358 .. 0.9999,
# with 263 of them below 0.95 -- round 2 quoted "returned down to 0.764" from
# a population a fifth the size.  This arm is a detector of ONE failure mode,
# and a plane it accepts is not thereby certified.  What DOES order that
# population is the LOSS side of this reading and of the launched-power
# bracket; see ``_lens_traced_multibranch._PIXEL_CONTINUITY_MIN`` for the
# confusion table and ``_PIXEL_CONTINUITY_SCOPES`` below for what a caller is
# told.
_MB_PIXEL_CONTINUITY_MAX = _PIXEL_CONTINUITY_MAX

# Lower arm, REPORTED not refused -- the mirror non-convergence, in which the
# coarse render MISSES deposits the finer one catches.  See the derivation
# with the upper arm.
_MB_PIXEL_CONTINUITY_MIN = _PIXEL_CONTINUITY_MIN

# WHAT THE READING BUYS, PER ROUTE (WP-B7c round 3, E4 / E7).
#
# ``pixel_continuity_of`` says which field the number was measured on.  That
# is not the same question as what the number CERTIFIES, and round 2 published
# only the first: on the Pearcey cusp route the label named the cusp field
# while the number was the branch sum's (E4), and on the FALLBACK route the
# reading was of the returned field but the dominant error there is the dark
# side the completion did not build, which this arm cannot see (E7 -- a plane
# returned at oracle fidelity 0.4639 with ``pixel_continuity`` 1.00201, the
# launched-power bracket 1.005, both decisions ``'ok'`` and no warning).
#
# ``pixel_continuity_scope`` is the second question, as an enum a consumer can
# branch on, with ``pixel_continuity_scope_note`` carrying the same statement
# in words.  MEASURED on the round-3 population (1304 oracle-scored planes, 16
# prescriptions, of which 476 are fallback planes the shipped bars RETURN;
# ``validation/probe_wp_b7c_round3/fallback_win.json``).  On the fallback
# route the reading is of the field this call returns but the dominant error
# is the dark tail the completion did not build, which this arm cannot see.
# Two readings the module ALREADY reports do order that population -- the
# LOSS side of this arm and of the launched-power bracket, whose bars are set
# for the fold-ring route where a dark deficit does not reach the caller (see
# ``_lens_traced_multibranch._PIXEL_CONTINUITY_MIN`` for the confusion table)
# -- and neither refuses there, so what the module can say honestly is that
# the returned field's dark tail is NOT arbitrated, which is what this key
# does.  (RESTATED 2026-09-19, VERIFY-WP-B7c round 3 D-1: the first draft
# cited a 642-plane pass and said nothing orders the fallback route.)
_PIXEL_CONTINUITY_SCOPES = {
    'returned_field': (
        'the reading is of the field this call returns, and the dark side '
        'has been completed, so it arbitrates the quadrature of the whole '
        'returned field'),
    'returned_field_quadrature_only': (
        'the reading is of the field this call returns, but this is a '
        'FALLBACK: the uniform completion declined and the returned field is '
        'the bright-side-only branch sum, so the dark-side tail is absent '
        'and NOT arbitrated.  This arm sees the quadrature and nothing else. '
        'On this route the LOSS sides DO order accuracy, and their bars are '
        'set for the fold-ring route where a dark deficit does not reach the '
        'caller: over 476 returned fallback planes, pixel_continuity below '
        '1/1.06 flags 192 planes and every one scores under oracle fidelity '
        '0.95 with no false alarm, and multibranch_power_ratio_bracketed '
        'below 0.889 flags 254 with none false and only 9 wrong ones missed. '
        'Thirteen wrong planes are left that nothing sees, the worst at '
        'fidelity 0.536 with every reading nominal'),
    'underlying_branch_sum': (
        'the reading is NOT of the field this call returns: it is of the '
        'branch sum that field is built on.  There is no half-pitch Pearcey '
        'cusp field to compare against without a second cusp trace, so the '
        "cusp route carries the branch sum's reading, which the module's "
        "own claim-2 measurement shows can differ from the returned field's "
        "(1.0636 against 1.0020 on VERIFY-B7b's fixture)"),
}

#: what the Pearcey cusp route's reading is OF.  Named rather than written
#: at the call site so the label and the scope cannot drift apart.
_PEARCEY_READING_OF = 'the branch sum the Pearcey cusp field is built on'

# Pearcey series (:func:`pearcey`) converges everywhere but SLOWS / overflows
# for large ``|x|, |y|``; clamp the control coordinates to this box when
# building the CUSP field (far outside the box geometric optics is exact, so
# the clamped value only affects a low-amplitude tail).  ``x`` is the
# defocus-like axis (the cusp opens toward ``x < 0``), ``y`` the transverse one.
_PEARCEY_X_LO, _PEARCEY_X_HI = -12.0, 6.0
_PEARCEY_Y_CAP = 11.0


# ==========================================================================
# Catastrophe classification (niche R5 / roadmap A4).
#
# The number of INTERIOR turning points of the meridional map ``x_out(h)`` (the
# fold caustics crossed along a radial cut) names the catastrophe class in the
# Thom hierarchy:  1 -> FOLD (A_2, the Airy path), 2 -> CUSP (A_3, the Pearcey
# path), 3 -> SWALLOWTAIL (A_4), 4 -> BUTTERFLY (A_5), >=5 -> a higher /
# non-classifiable coalescence.  Only the fold and cusp have canonical
# completions here; the higher catastrophes are RARE in real lenses and their
# canonical integrals are high-effort, so A4 makes the dispatcher DETECT the
# class and ROUTE cleanly to the finite multibranch / GBD fallback with a
# one-time NAMED warning -- keeping 'seamless' true by routing, never by
# emitting inf/nan.
# ==========================================================================
_CATASTROPHE_NAMES = {
    1: 'fold (A2)',
    2: 'cusp (A3)',
    3: 'swallowtail (A4)',
    4: 'butterfly (A5)',
}


def _count_interior_turning_points(x_out):
    """Number of interior turning points (sign changes of ``d x_out / d h``) of a
    monotone-``h`` meridional map ``x_out(h)`` -- each is a fold caustic, and the
    count names the catastrophe class (see :data:`_CATASTROPHE_NAMES`).

    Same intent as the ``diff(sign(diff(x_out)))`` construction in
    :func:`_trace_meridional_fold` / :func:`_trace_meridional_cusp`, but robust
    to an isolated zero-slope SAMPLE at an extremum (which would otherwise
    double-count a ``+ -> 0 -> -`` transition): the zero-sign entries are dropped
    before counting the residual sign changes.  On clean ray-traced maps (no
    exactly-flat slope sample) this equals the live traces' ``turns.size``."""
    xo = np.asarray(x_out, dtype=float)
    if xo.size < 3:
        return 0
    s = np.sign(np.diff(xo))
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(s) != 0))


def _classify_catastrophe(n_turn):
    """Name the catastrophe class from the interior-turning-point count
    ``n_turn``.  ``1`` -> fold, ``2`` -> cusp (both have canonical completions),
    ``3`` -> swallowtail, ``4`` -> butterfly, ``>=5`` -> a non-classifiable
    higher catastrophe.  The swallowtail/butterfly/non-classifiable classes are
    ROUTED (never analytically evaluated) to the finite fallback."""
    name = _CATASTROPHE_NAMES.get(int(n_turn))
    if name is not None:
        return name
    if int(n_turn) >= 5:
        return f'non-classifiable higher catastrophe ({int(n_turn)} folds)'
    return f'degenerate map ({int(n_turn)} folds)'


def _is_rotationally_symmetric(E_in, dx, *, tol=0.12, n_rings=6, n_theta=48):
    """True when ``|E_in|`` is (approximately) rotationally symmetric about the
    grid centre -- the necessary condition for the radial fold-ring completion.

    For ``n_rings`` radii spanning the significant-power support (kept inside
    the inscribed circle so every azimuth is on-grid), ``|E_in|`` is bilinearly
    resampled at ``n_theta`` AZIMUTHS and the per-ring azimuthal coefficient of
    variation (std/mean over theta, at FIXED radius -- so a steep radial profile
    does not trip it) must stay below ``tol``.  Cheap and grid-robust; a false
    negative only steers the caller to the (still-finite) multibranch
    fallback."""
    E_in = np.asarray(E_in)
    N = E_in.shape[0]
    a = np.abs(E_in)
    amax = float(a.max())
    if amax <= 0.0:
        return False
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    sig = a > 0.05 * amax
    if not sig.any():
        return False
    # cap the support radius at the inscribed circle so every sampled ring is
    # fully on-grid (avoids partial-annulus artefacts at the corners)
    r_max_grid = 0.49 * N * dx
    r_sup = float(min(r[sig].max(), r_max_grid))
    if r_sup <= 2.0 * dx:
        return False
    c = 0.5 * N                          # grid-centre index (pixel-centre conv.)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    for rr in np.linspace(0.15 * r_sup, r_sup, n_rings):
        # sample |E| at (rr, theta); grid index = centre + coord/dx
        gi = c + (rr * ct) / dx          # column (x) index
        gj = c + (rr * st) / dx          # row (y) index
        i0 = np.floor(gi).astype(int)
        j0 = np.floor(gj).astype(int)
        if i0.min() < 0 or j0.min() < 0 or i0.max() + 1 >= N or j0.max() + 1 >= N:
            continue
        fi = gi - i0
        fj = gj - j0
        vals = ((1 - fi) * (1 - fj) * a[j0, i0]
                + fi * (1 - fj) * a[j0, i0 + 1]
                + (1 - fi) * fj * a[j0 + 1, i0]
                + fi * fj * a[j0 + 1, i0 + 1])
        mu = float(vals.mean())
        if mu <= 0.02 * amax:
            continue
        if float(vals.std()) / mu > tol:
            return False
    return True


def _trace_meridional_fold(prescription, wavelength, output_plane_distance,
                           output_plane_n, launch_radius, n_fan):
    """Meridional ray trace of ``prescription`` to the output plane; extract the
    single fold-ring geometry.

    Launches a dense on-axis-collimated meridional fan along ``+x`` (``y = 0``)
    through the SAME surfaces the multibranch uses, advances the exit rays a
    distance ``output_plane_distance`` past the exit vertex (index
    ``output_plane_n``), and returns::

        {ok, reason, r_c, kappa, cphi, n_turn, band}

    ``ok`` is True only for exactly ONE interior turning point of ``y_obs(h)``
    (a single fold ring).  ``zeta(r) = kappa (r_c - r)`` (linear fit of
    ``[3/4 |S_A - S_B|]^{2/3}`` over the two-branch band) and
    ``A(r) = polyval(cphi, r - r_c)`` (quadratic mean-eikonal fit).  ``band`` is
    the WIDTH of that two-branch band -- the only radii at which the eikonal
    difference defining ``zeta`` exists, so the interval the linear fit is
    entitled to speak for; the caller compares its own fit / fill widths
    against it (see ``_ZETA_EXTRAPOLATION_MAX``).  ``reason`` names the
    fallback cause when ``ok`` is False."""
    fail = dict(ok=False, r_c=0.0, kappa=0.0, cphi=None, n_turn=0, band=0.0)
    surfaces = rt.surfaces_from_prescription(prescription)
    xs = np.linspace(launch_radius / n_fan, launch_radius, n_fan)
    rays = rt.RayBundle(
        x=xs.copy(), y=np.zeros(n_fan), z=np.zeros(n_fan),
        L=np.zeros(n_fan), M=np.zeros(n_fan), N=np.ones(n_fan),
        wavelength=wavelength, alive=np.ones(n_fan, dtype=bool),
        opd=np.zeros(n_fan))
    tr = rt.trace(rays, surfaces, wavelength)
    # ``rt.trace`` leaves each ray at z = sag(rho) of the LAST surface, so the
    # fold geometry (r_c, kappa, the mean-eikonal fit) would otherwise be
    # resolved on the sag SURFACE rather than on the output PLANE -- the same
    # defect the multibranch rasteriser carried (module: _trace_launch_grid).
    # Transfer to the exit vertex through the exit medium first, then add the
    # free-space leg in ``output_plane_n``.
    ex = tr.at_exit_vertex()
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = output_plane_distance / Nz
    x_out = ex.x + t * ex.L
    opl = ex.opd + float(output_plane_n) * t
    alive = ex.alive & np.isfinite(x_out) & np.isfinite(opl)
    h = xs[alive]
    xo = x_out[alive]
    S = opl[alive]
    if h.size < 64:
        return {**fail, 'reason': 'too_few_rays'}

    # interior turning points of x_out(h) (fold caustics = dx_out/dh sign
    # change).  ``turns`` gives their POSITIONS; the COUNT comes from
    # :func:`_count_interior_turning_points`, which drops exactly-zero-slope
    # samples first.  The raw ``diff(sign(diff))`` count double-counts an
    # isolated flat sample at an extremum (``+ -> 0 -> -`` is two sign
    # changes), which would misclassify a clean FOLD as a cusp and route it
    # away from the Airy completion it qualifies for.  The robust counter has
    # existed since this module was written and was never called.
    dxo = np.diff(xo)
    sgn = np.sign(dxo)
    turns = np.where(np.diff(sgn) != 0)[0] + 1     # index into h of the turn
    n_turn = _count_interior_turning_points(xo)
    if n_turn == 0 or turns.size == 0:
        return {**fail, 'reason': 'no_fold', 'n_turn': 0}
    if n_turn > 1:
        # >1 interior turning point -> cusp / multiple rings (Pearcey regime,
        # out of scope): detect + fall back.
        return {**fail, 'reason': 'cusp_or_multiple', 'n_turn': n_turn}

    i_f = int(turns[0])
    r_c = float(abs(xo[i_f]))
    if r_c <= 0.0:
        return {**fail, 'reason': 'degenerate_rc', 'n_turn': n_turn}

    # two branches around the fold; |x_out| is monotonic on each
    ya = np.abs(xo[:i_f + 1])
    Sa = S[:i_f + 1]
    yb = np.abs(xo[i_f:])
    Sb = S[i_f:]
    from scipy.interpolate import CubicSpline

    def _mono_spline(yv, sv):
        yu, idx = np.unique(yv, return_index=True)
        if yu.size < 4:
            return None
        return CubicSpline(yu, sv[idx])
    spa = _mono_spline(ya, Sa)
    spb = _mono_spline(yb, Sb)
    if spa is None or spb is None:
        return {**fail, 'reason': 'branch_undersampled', 'n_turn': n_turn}

    r_lo = float(max(ya.min(), yb.min()))
    band = r_c - r_lo
    if band <= 0.0:
        return {**fail, 'reason': 'no_two_branch_band', 'n_turn': n_turn}
    rb = np.linspace(r_lo + 0.02 * band, r_c - 0.02 * band, 256)
    dS = np.abs(spa(rb) - spb(rb))
    zeta = (0.75 * dS) ** (2.0 / 3.0)
    # kappa = slope of the linear zeta(r_c - r); intercept must be ~0 (fold)
    u = r_c - rb
    kfit = np.polyfit(u, zeta, 1)
    kappa = float(kfit[0])
    if not np.isfinite(kappa) or kappa <= 0.0:
        return {**fail, 'reason': 'bad_kappa', 'n_turn': n_turn}
    zpred = np.polyval(kfit, u)
    resid = float(np.max(np.abs(zpred - zeta)) / (np.max(zeta) + 1e-300))
    if resid > 0.15:
        return {**fail, 'reason': 'zeta_nonlinear', 'n_turn': n_turn}
    # The band's OWN curvature, as a range over which the linear normal form is
    # entitled to speak.  Re-fitting the same 256 samples with
    # ``zeta = kappa u + q u^2`` gives the radius at which the quadratic term
    # reaches 10 % of the linear one, ``u* = 0.1 kappa / |q|``.  Reported only
    # (``zeta_linear_range``): it is measured ON the band and so is itself an
    # extrapolation, and against the oracle's dark side it is ANTI-conservative
    # -- see the ``_AIRY_TAIL_CELLS`` note, where the fill is deliberately left
    # unclipped.  Costs one extra 2nd-degree polyfit over 256 points per call.
    qfit = np.polyfit(u, zeta, 2)
    q = float(qfit[0])
    u_star = float(0.1 * kappa / abs(q)) if q != 0.0 else float('inf')
    cphi = np.polyfit(rb - r_c, 0.5 * (spa(rb) + spb(rb)), 2)
    return dict(ok=True, reason='fold_ring', r_c=r_c, kappa=kappa,
                cphi=cphi, n_turn=n_turn, band=float(band),
                zeta_linear_resid=resid, zeta_curvature=q,
                zeta_linear_range=u_star)


# ==========================================================================
# CUSP (Pearcey) completion -- A1 / niche R2.
#
# A CUSP is a 3-branch coalescence: the meridional map ``x_out(h)`` has TWO
# interior turning points (``n_turn == 2``), so a band of radii ``r in
# (r1, r2)`` is reached by THREE ray branches whose two folds (at ``r1``, ``r2``)
# sit within a diffraction width of each other -- the fold-Airy of the fold
# completion cannot resolve the pair and the correct local diffraction pattern
# is the PEARCEY (cusp) canonical integral
# ``P(x, y) = int exp(i(t^4 + x t^2 + y t)) dt`` (:func:`pearcey`, REUSED here --
# no reimplementation).  This module maps the local ray geometry to the Pearcey
# normal form, evaluates ``P`` (and its two first derivatives) on the mapped
# control coordinates, and combines them with amplitude coefficients derived
# from the three branches' stationary-phase ray amplitudes -- the exact cusp
# analog of :func:`lumenairy.elements.lenses_maslov.uniform_fold_airy`.
#
# THE MAPPING (validated to machine precision against 1-D quartic-phase
# integrals, and to Icorr ~ 0.97 / windowed r2m ~ few-% against a direct
# Rayleigh-Sommerfeld cusp ground truth):
#
#   * Control coordinates ``(x, y)`` from the three branch PHASES.  The three
#     stationary points ``w_j`` of ``t^4 + x t^2 + y t`` are the roots of
#     ``4 w^3 + 2 x w + y = 0`` and their critical VALUES are
#     ``g(w_j) = (1/2) x w_j^2 + (3/4) y w_j`` with ``sum_j g(w_j) = -(1/2) x^2``.
#     Given the three branch phases ``Phi_j`` (radians = ``k * OPL_j``), solve
#     the 2-DOF system matching the sorted, mean-removed ``g(w_j)`` to the
#     sorted, mean-removed ``Phi_j`` for ``(x, y)`` (a small least squares).  The
#     scale is carried in radians so the Pearcey ``k``-scaling is implicit; the
#     reference phase is ``Phi0 = mean(Phi_j) + x^2 / 6``.
#   * At a FIXED observation plane the map is (to leading order) LINEAR: ``x`` is
#     constant (the plane's distance from the cusp point) and ``y`` is linear in
#     the transverse radius -- so ``x = x0``, ``y = gamma (r - r_sym)``, robustly
#     extrapolatable into the 1-branch regions where the phase solve has no
#     three real branches.
#   * Amplitude coefficients ``(b0, b1, b2)`` of ``(P, dP/dy, dP/dx)`` from the
#     branch ray-tube amplitudes: matching the stationary-phase limit of the
#     uniform form to the geometric ray sum gives, per branch,
#     ``b0 + b1 w_j + b2 w_j^2 = amp_j sqrt(|g''(w_j)| / 2pi)
#        exp(i (sgn(Psi''_j) - sgn(g''(w_j))) pi/4)`` with ``g''(w) = 12 w^2 +
#     2 x`` -- a 3x3 Vandermonde solve (the cusp analog of the fold's ``a0, a1``).
#   * The completed field is ``E = e^{i Phi0} (b0 P - i b1 dP/dy - i b2 dP/dx)``,
#     finite through the caustic (``P`` is entire), with the correct cusp fringe
#     structure and the exponential dark-side tail built in.
#
# SCOPE: rotationally-symmetric SINGLE cusp ring (``n_turn == 2``), the same
# collimated / rot-sym / centred gate as the fold path.  Anything the mapping
# cannot resolve (bad control solve, non-linear map, under-resolved fringe)
# DETECTS + falls back to the plain multibranch field (finite, never inf/nan).
# ==========================================================================


def _pearcey_cubic_roots(x, y):
    """The three real roots of ``4 w^3 + 2 x w + y = 0`` (the stationary points
    of the Pearcey phase ``t^4 + x t^2 + y t``), sorted ascending, or ``None``
    when fewer than three real roots exist (outside the cusp -- 1 branch)."""
    # depressed cubic w^3 + p w + q = 0 with p = x/2, q = y/4
    p = 0.5 * x
    q = 0.25 * y
    if p >= 0.0:
        return None
    disc = (0.5 * q) ** 2 + (p / 3.0) ** 3
    if disc > 0.0:
        return None
    m = 2.0 * np.sqrt(-p / 3.0)
    arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
    th = np.arccos(arg) / 3.0
    roots = np.array([m * np.cos(th - 2.0 * np.pi * kk / 3.0) for kk in range(3)])
    return np.sort(roots)


def _pearcey_crit_values(x, y):
    """Critical values ``g(w_j) = (1/2) x w_j^2 + (3/4) y w_j`` at the three real
    stationary points, or ``None`` when there are not three real roots."""
    w = _pearcey_cubic_roots(x, y)
    if w is None:
        return None
    return 0.5 * x * w ** 2 + 0.75 * y * w


def _solve_pearcey_control(phis, guess=(-2.0, 0.3)):
    """Solve the Pearcey control coordinates ``(x, |y|)`` from three branch
    phases (radians).  Matches the sorted, mean-removed critical values of the
    normal form to the sorted, mean-removed phases.  ``y`` is returned as its
    magnitude (the sorted-phase data cannot distinguish ``y`` from ``-y`` -- the
    sign is fixed downstream by the transverse position).  Returns
    ``(x, abs_y, cost)``; ``cost`` (the residual) flags a poor / non-cusp
    match."""
    from scipy.optimize import least_squares
    d = np.sort(phis)
    d = d - d.mean()

    def resid(v):
        cv = _pearcey_crit_values(v[0], v[1])
        if cv is None:
            return [1e3, 1e3, 1e3]
        c = np.sort(cv)
        return (c - c.mean()) - d

    sol = least_squares(resid, np.asarray(guess, dtype=float),
                        method='lm', xtol=1e-13, ftol=1e-13)
    return float(sol.x[0]), abs(float(sol.x[1])), float(sol.cost)


def _pearcey_cusp_amp_coeffs(x, y, phis, amps, dxdh):
    """The three amplitude coefficients ``(b0, b1, b2)`` of ``(P, dP/dy, dP/dx)``
    from the three branches' ray-tube amplitudes -- the cusp analog of the
    fold's ``(a0, a1)``.

    ``phis, amps, dxdh`` are the branch phases (radians), ray-tube amplitude
    magnitudes and signed ``d x_out / d h`` (its SIGN is the stationary-phase
    Maslov indicator).  Pairs each branch (ordered by phase) with a root
    (ordered by critical value), then solves the 3x3 Vandermonde
    ``[1, w_j, w_j^2] . (b0, b1, b2) = rhs_j`` with
    ``rhs_j = amp_j sqrt(|g''(w_j)| / 2pi) exp(i (sgn(Psi''_j) - sgn(g''_j)) pi/4)``,
    ``g''(w) = 12 w^2 + 2 x``.  Returns the complex ``(b0, b1, b2)`` or ``None``
    (degenerate)."""
    w = _pearcey_cubic_roots(x, y)
    cv = _pearcey_crit_values(x, y)
    if w is None or cv is None:
        return None
    ob = np.argsort(phis)          # branch order by phase
    oc = np.argsort(cv)            # root order by critical value
    wp = w[oc]                     # roots in critical-value order
    amp_p = np.asarray(amps)[ob]   # branch amps in phase order
    dxdh_p = np.asarray(dxdh)[ob]
    gpp = 12.0 * wp ** 2 + 2.0 * x
    if np.any(np.abs(gpp) < 1e-12):
        return None
    rhs = (amp_p * np.sqrt(np.abs(gpp) / (2.0 * np.pi))
           * np.exp(1j * (np.sign(dxdh_p) - np.sign(gpp)) * np.pi / 4.0))
    vander = np.stack([np.ones(3), wp, wp ** 2], axis=1)
    try:
        b = np.linalg.solve(vander, rhs)
    except np.linalg.LinAlgError:
        return None
    return b


def _pearcey_basis(x, y, step=3e-3):
    """``(P, dP/dx, dP/dy)`` at ``(x, y)`` -- REUSING :func:`pearcey` (central
    finite differences for the two derivatives; a reimplementation of the kernel
    would be a defect).  The control coordinates are clamped to the Pearcey
    series' well-behaved box so a far-tail pixel never overflows."""
    xe = float(np.clip(x, _PEARCEY_X_LO, _PEARCEY_X_HI))
    ye = float(np.clip(y, -_PEARCEY_Y_CAP, _PEARCEY_Y_CAP))
    p0 = pearcey(xe, ye)
    px = (pearcey(xe + step, ye) - pearcey(xe - step, ye)) / (2.0 * step)
    py = (pearcey(xe, ye + step) - pearcey(xe, ye - step)) / (2.0 * step)
    return p0, px, py


def _cusp_geometry_from_branches(r_arr, phib, ampb, dxdhb, *,
                                 max_control_cost=5e-4, max_map_resid=0.25):
    """Reduce the three-branch meridional geometry over the cusp band to the
    LINEAR Pearcey control map + smooth amplitude / phase coefficients.

    Parameters (all with the branches in a CONSISTENT per-radius order, e.g.
    by launch height):

    * ``r_arr`` -- ``(M,)`` radii spanning the three-branch band ``(r1, r2)``.
    * ``phib``  -- ``(M, 3)`` branch phases (radians = ``k * OPL``).
    * ``ampb``  -- ``(M, 3)`` branch ray-tube amplitude magnitudes.
    * ``dxdhb`` -- ``(M, 3)`` branch signed ``d x_out / d h``.

    Returns a dict ``{ok, x0, gamma, r_sym, cP, cb, ...}`` (the resolved linear
    map ``x=x0``, ``y=gamma (r - r_sym)``, the quadratic mean-phase poly ``cP``
    and the three linear complex amplitude polys ``cb``), or ``{ok: False,
    reason}`` when the control solve or the linear map is too poor to trust."""
    r_arr = np.asarray(r_arr, dtype=float)
    phib = np.asarray(phib, dtype=float)
    ampb = np.asarray(ampb, dtype=float)
    dxdhb = np.asarray(dxdhb, dtype=float)
    M = r_arr.shape[0]
    x_l, y_l, phi0_l, b_l, cost_l = [], [], [], [], []
    guess = (-2.0, 0.3)
    for i in range(M):
        ph = phib[i]
        x, ay, cost = _solve_pearcey_control(ph, guess=guess)
        guess = (x, ay if ay > 1e-9 else 0.3)
        b = _pearcey_cusp_amp_coeffs(x, ay, ph, ampb[i], dxdhb[i])
        if b is None:
            continue
        x_l.append(x)
        y_l.append(ay)
        phi0_l.append(ph.mean() + x ** 2 / 6.0)
        b_l.append(b)
        cost_l.append(cost)
    if len(x_l) < 8:
        return {'ok': False, 'reason': 'cusp band undersampled'}
    x_a = np.array(x_l)
    y_abs = np.array(y_l)
    phi0_a = np.array(phi0_l)
    b_a = np.array(b_l)
    cost_a = np.array(cost_l)
    # control-solve quality: the phases must actually reduce to the cusp normal
    # form (a large residual => this is not a clean single cusp).
    phi_scale = float(np.median(np.abs(np.diff(np.sort(phib[M // 2]))))) + 1e-30
    if float(np.median(cost_a)) > max_control_cost * phi_scale ** 2 + 1e-12 \
            and float(np.median(np.sqrt(cost_a))) > max_control_cost * phi_scale:
        return {'ok': False, 'reason': 'cusp control-solve residual too large'}
    # symmetric centre = min |y|; linear map y = gamma (r - r_sym), x = const
    ic = int(np.argmin(y_abs))
    r_sym = float(r_arr[ic])
    u = r_arr - r_sym
    y_signed = y_abs * np.sign(u + 1e-30)
    x0 = float(np.median(x_a))
    gam = float(np.linalg.lstsq(u[:, None], y_signed, rcond=None)[0][0])
    if not np.isfinite(gam) or abs(gam) < 1e-30:
        return {'ok': False, 'reason': 'degenerate cusp transverse map'}
    # linear-map fidelity: solved y vs the linear model (relative to the span)
    y_model = gam * u
    map_resid = float(np.max(np.abs(y_model - y_signed))
                      / (np.max(np.abs(y_signed)) + 1e-30))
    if not np.isfinite(map_resid) or map_resid > max_map_resid:
        return {'ok': False, 'reason': 'cusp map not linear (astigmatic/tilted)'}
    cP = np.polyfit(u, phi0_a, 2)
    cb = [np.polyfit(u, b_a[:, j], 1) for j in range(3)]
    return {'ok': True, 'reason': 'cusp_ring', 'x0': x0, 'gamma': gam,
            'r_sym': r_sym, 'cP': cP, 'cb': cb, 'x_var': float(np.std(x_a)),
            'y_max': float(np.max(y_abs)), 'map_resid': map_resid,
            'n_band': int(M), 'u_lo': float(u.min()), 'u_hi': float(u.max())}


def _radial_amp_sampler(E_in, dx):
    """A callable ``r -> |E_in|(r)`` from the rotationally-symmetric input field,
    sampling ``|E_in|`` along the ``+x`` grid axis (pixel-centre convention).

    Origin convention (v5.30, audit E-L10): every grid in this module is
    ``x = (arange(N) - N/2) * dx``, so the axis origin sits at the FLOAT index
    ``N/2`` -- an integer only for even N.  The row / column anchors and the
    radii are therefore derived from that ``x`` vector, not from ``N // 2``.
    For even N this is bit-identical to the old ``c = N // 2`` /
    ``rp = arange(N - c) * dx`` form (``x[N//2] == 0`` exactly, so the row is
    the same and ``hypot(x[c:], 0) == arange(N - c) * dx``).  For ODD N the old
    form mislabelled each sample's radius by up to ~0.7 px: measured on an
    exactly rotationally-symmetric Gaussian (w = 8 dx) the sampled profile was
    off by 5.11e-2 in amplitude -- a best-fit radial shift of -0.44 px -- at
    N = 65 and N = 127, versus 3.5e-3 (pure interpolation error) at N = 64 and
    N = 128.
    """
    a = np.abs(np.asarray(E_in))
    N = a.shape[0]
    x = (np.arange(N) - N / 2.0) * dx
    c = int(np.ceil(N / 2.0))          # first column with x >= 0 (== N//2 even N)
    row = int(np.argmin(np.abs(x)))    # row nearest y = 0        (== N//2 even N)
    rp = np.hypot(x[c:], x[row])       # TRUE radius of each sampled pixel
    ap = a[row, c:].astype(float)

    def sampler(r):
        return np.interp(np.abs(r), rp, ap, left=ap[0], right=0.0)
    return sampler


def _roots_in_segments(h, f):
    """All ``h`` where the sampled ``f(h)`` crosses zero, refined by bisection on
    the bracketing grid cells.  ``h`` monotone ascending."""
    s = np.sign(f)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        a, b = h[i], h[i + 1]
        fa = f[i]
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = np.interp(m, h, f)
            if fa * fm <= 0.0:
                b = m
            else:
                a, fa = m, fm
        out.append(0.5 * (a + b))
    return np.array(out)


def _trace_meridional_cusp(prescription, wavelength, output_plane_distance,
                           output_plane_n, launch_radius, n_fan, E_in, dx,
                           n_band=120):
    """Meridional trace + Pearcey-cusp geometry for a ``n_turn == 2`` (single
    finite-radius CUSP ring) prescription.

    Traces the same collimated meridional fan as :func:`_trace_meridional_fold`,
    requires EXACTLY two interior turning points of the signed ``x_out(h)`` whose
    extrema have the SAME sign (a finite-radius ring, not the on-axis /
    near-axial focus -- that is the Bessoid regime, out of scope), samples the
    three-branch band, and reduces it to the linear Pearcey control map +
    amplitude coefficients via :func:`_cusp_geometry_from_branches`.

    Returns ``{ok, r1, r2, ...geometry...}`` or ``{ok: False, reason}``."""
    fail = dict(ok=False, r1=0.0, r2=0.0)
    surfaces = rt.surfaces_from_prescription(prescription)
    xs = np.linspace(launch_radius / n_fan, launch_radius, n_fan)
    rays = rt.RayBundle(
        x=xs.copy(), y=np.zeros(n_fan), z=np.zeros(n_fan),
        L=np.zeros(n_fan), M=np.zeros(n_fan), N=np.ones(n_fan),
        wavelength=wavelength, alive=np.ones(n_fan, dtype=bool),
        opd=np.zeros(n_fan))
    tr = rt.trace(rays, surfaces, wavelength)
    # Exit-vertex transfer before the free-space leg -- see the note in
    # :func:`_trace_meridional_fold`; the cusp control map is resolved from
    # the same ``(x_out, S)`` pair and inherits the same defect without it.
    ex = tr.at_exit_vertex()
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = output_plane_distance / Nz
    x_out = ex.x + t * ex.L
    opl = ex.opd + float(output_plane_n) * t
    alive = ex.alive & np.isfinite(x_out) & np.isfinite(opl)
    h = xs[alive]
    xo = x_out[alive]
    S = opl[alive]
    if h.size < 128:
        return {**fail, 'reason': 'too_few_rays'}

    dxo = np.diff(xo)
    turns = np.where(np.diff(np.sign(dxo)) != 0)[0] + 1
    if turns.size != 2:
        return {**fail, 'reason': f'n_turn={turns.size} (not a single cusp)'}
    Xa, Xb = xo[int(turns[0])], xo[int(turns[1])]
    if Xa * Xb <= 0.0:
        # the band straddles the axis -> on-axis (Bessoid) cusp, out of scope
        return {**fail, 'reason': 'near-axial cusp (Bessoid, out of scope)'}
    band_lo, band_hi = (Xa, Xb) if Xa < Xb else (Xb, Xa)
    r1 = float(min(abs(Xa), abs(Xb)))
    r2 = float(max(abs(Xa), abs(Xb)))
    if (r2 - r1) <= 0.0 or r1 <= 0.0:
        return {**fail, 'reason': 'degenerate cusp band'}

    k0 = 2.0 * np.pi / wavelength
    amp_of = _radial_amp_sampler(E_in, dx)
    hc = 0.5 * (h[:-1] + h[1:])
    slope = dxo / np.diff(h)          # d x_out / d h (signed), on cell centres

    vs = np.linspace(band_lo + 0.02 * (band_hi - band_lo),
                     band_hi - 0.02 * (band_hi - band_lo), n_band)
    r_arr, phib, ampb, dxdhb = [], [], [], []
    for v in vs:
        hb = _roots_in_segments(h, xo - v)
        if hb.size != 3:
            continue
        rq = abs(v)
        ph = k0 * np.interp(hb, h, S)
        jj = np.interp(hb, hc, slope)
        amp = amp_of(hb) * np.sqrt(np.clip(hb / (rq * np.abs(jj) + 1e-300),
                                           0.0, None))
        r_arr.append(rq)
        phib.append(ph)
        ampb.append(amp)
        dxdhb.append(jj)
    if len(r_arr) < 12:
        return {**fail, 'reason': 'cusp band undersampled (branch tracing)'}
    r_arr = np.array(r_arr)
    order = np.argsort(r_arr)
    geom = _cusp_geometry_from_branches(
        r_arr[order], np.array(phib)[order], np.array(ampb)[order],
        np.array(dxdhb)[order])
    if not geom['ok']:
        return {**fail, 'reason': geom['reason']}
    geom = dict(geom)
    geom.update(r1=r1, r2=r2)
    return geom


def _build_pearcey_cusp_field(E_mb, geom, dx):
    """Build the 2-D Pearcey-cusp completed field from the multibranch base
    ``E_mb`` and the resolved cusp geometry.

    LAMBDA-FREE by construction, which is why this function takes no
    wavelength (it used to accept one and never read it -- exactly the shape
    of a missing ``k``-scaling, so the ambiguity is worth closing explicitly).
    The scaling lives upstream: :func:`_solve_pearcey_control` fits the control
    coordinates to the branch PHASES ``k*OPL`` in radians, so the Pearcey
    ``k``-scaling is already absorbed into ``(x0, gamma, cP, cb)``; the overall
    diffraction prefactor is then fixed by matching to ``E_mb`` in a clean
    single-branch annulus, which carries whatever ``k``-dependence remains.

    The Pearcey envelope depends only on the RADIUS (the control map is
    ``x=x0``, ``y=gamma (r - r_sym)``), so it is evaluated on a dense 1-D radial
    LUT and interpolated onto the grid (fast; mirrors the fold path's radial
    evaluation).  The rapidly-oscillating reference phase ``exp(i Phi0(r))`` is
    applied per-pixel (no LUT aliasing).  The overall (diffraction-prefactor)
    scale is fixed by matching to ``E_mb`` in a clean single-branch annulus just
    inside ``r1``; the completed field REPLACES ``E_mb`` only in the cusp zone
    ``[r1 - margin, r2 + margin]`` (the multibranch stays elsewhere).  Returns
    the completed field (finite everywhere) or ``None`` (guarded degenerate)."""
    N = E_mb.shape[0]
    x0 = geom['x0']
    gam = geom['gamma']
    r_sym = geom['r_sym']
    cP = geom['cP']
    cb = geom['cb']
    r1, r2 = geom['r1'], geom['r2']
    band = r2 - r1
    margin = 2.5 * band
    # the amplitude / mean-phase polynomials are fitted ONLY over the three-
    # branch band; beyond it (the 1-branch regions) the transverse control ``y``
    # keeps growing linearly (physical -- the single-ray asymptotic) but the
    # amplitude coefficients must NOT be extrapolated past the band (a linear
    # fit would blow up).  Clamp ``u`` for the coefficient / phase evaluation to
    # the band range, so they hold their band-edge value in the tails.
    u_lo = geom.get('u_lo', -0.5 * band)
    u_hi = geom.get('u_hi', 0.5 * band)

    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    rg = np.sqrt(X * X + Y * Y)

    def envelope(r):
        u = r - r_sym
        uc = min(max(u, u_lo), u_hi)
        yq = gam * u
        p0, px, py = _pearcey_basis(x0, yq)
        return (np.polyval(cb[0], uc) * p0
                - 1j * np.polyval(cb[1], uc) * py
                - 1j * np.polyval(cb[2], uc) * px)

    # dense radial LUT of the slowly-varying complex envelope over the zone
    r_lo = max(0.0, r1 - margin)
    r_hi = r2 + margin
    n_lut = max(256, int(4.0 * (r_hi - r_lo) / dx))
    r_lut = np.linspace(r_lo, r_hi, n_lut)
    env_lut = np.array([envelope(rr) for rr in r_lut])
    if not np.all(np.isfinite(env_lut)):
        return None

    zone = (rg >= r_lo) & (rg <= r_hi)
    rz = rg[zone]
    env_z = (np.interp(rz, r_lut, env_lut.real)
             + 1j * np.interp(rz, r_lut, env_lut.imag))
    phi0_z = np.polyval(cP, rz - r_sym)
    F = env_z * np.exp(1j * phi0_z)

    # prefactor: match F to E_mb in a clean single-branch annulus inside r1
    ann_lo = max(0.0, r1 - 3.0 * band)
    ann_hi = max(ann_lo + dx, r1 - 0.5 * band)
    ann = (rz >= ann_lo) & (rz <= ann_hi)
    if int(ann.sum()) < 16:
        ann = (rz >= max(0.0, r1 - 6.0 * band)) & (rz <= r1 - 0.5 * band)
    denom = np.vdot(F[ann], F[ann])
    if int(ann.sum()) < 8 or abs(denom) <= 0.0:
        return None
    Emb_ann = np.asarray(E_mb)[zone][ann]
    pref = np.vdot(F[ann], Emb_ann) / denom
    if not np.isfinite(pref):
        return None

    E_out = np.array(E_mb, dtype=np.complex128, copy=True)
    E_out[zone] = pref * F
    if not np.all(np.isfinite(E_out)):
        return None
    return E_out


def apply_real_lens_traced_uniform(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    output_plane_distance: float = 0.0,
    output_plane_n: float = 1.0,
    ray_subsample: int = 2,
    min_area_ratio: float = 1e-6,
    caustic_band: str = 'ludwig',
    input_carrier: Any = None,
    uniform_fit_halfwidth: Any = None,
    n_fan: int = 4000,
    return_diagnostics: bool = False,
) -> Any:
    """Uniform (Airy) dark-side completion of the multibranch ray-density field.

    Runs :func:`apply_real_lens_traced_multibranch` for the finite, ludwig-
    regularized BRIGHT-side field, then -- for a rotationally-symmetric SINGLE
    fold RING -- fills the DARK side (``r > r_c``) with the CFU uniform Airy tail
    (see the module docstring).  Falls back to the plain multibranch field (with
    a one-time warning) for any non-fold / cusp / non-symmetric case.  The
    output is FINITE everywhere (never inf/nan).

    Parameters mirror :func:`apply_real_lens_traced_multibranch`, plus
    ``uniform_fit_halfwidth`` (bright-band fit width [m]; ``None`` -> auto from
    the Airy scale) and ``n_fan`` (meridional-trace ray count).

    Returns the completed ``(N, N)`` complex field (plus a diagnostics dict when
    ``return_diagnostics`` -- with the resolved ``r_c``, ``kappa``, the fitted
    Airy coefficients, the fit residual, the two-branch band ``zeta_band`` and
    the ``zeta_extrapolation`` ratio below, and ``fell_back`` / ``reason``).

    Accuracy envelope (MEASURED, WP-B7b 2026-09-14)
    -----------------------------------------------
    Against a brute-force Rayleigh-Sommerfeld oracle built on an exact conic
    raytrace, on a ladder of planes through one caustic of a plano-convex
    singlet (R = 2.7 mm, t = 1.0 mm, 1.5 mm aperture, n = 1.5168,
    lambda = 1.31 um, N = 512, dx = 3.0 um) and on two more singlets (N-LAK22
    biconvex R = +/-3.0 mm, t = 0.55 mm, 1.20 mm aperture, lambda = 850 nm,
    N = 640, dx = 2.10 um, at its marginal focus; and N-LAK22 biconvex
    R = +/-6.0 mm, t = 0.90 mm, 1.10 mm aperture, lambda = 1.55 um, N = 320,
    dx = 3.85 um, z = 4.400 mm), indexed by
    ``zeta_extrapolation = uniform_fit_halfwidth / zeta_band`` -- how far past
    the two-branch band the fitted ``zeta(r) = kappa (r_c - r)`` is carried:

    ==================  =====  =====  =====  =====  =====  =====  =====
    zeta_extrapolation   0.16    1.9    2.3    3.4    5.4    9.8  453.6
    fidelity            0.944  0.957  0.933  0.969  0.972  0.962  0.930
    power / oracle      0.947  0.975  0.971  1.030  1.049  1.125  1.228
    multibranch fid.    0.882  0.805  0.856  0.853  0.884  0.887  0.832
    ==================  =====  =====  =====  =====  =====  =====  =====

    So the SHAPE is reliable and is the better one at every plane measured (it
    beats the plain multibranch it would fall back to, by 0.06-0.15 of
    fidelity), while the ABSOLUTE ENERGY the dark tail carries degrades with
    the extrapolation.  Past ``_ZETA_EXTRAPOLATION_MAX`` the call WARNS and
    says so; it does not fall back, because falling back is measurably worse.

    Re-measured on three more optics (WP-B7c, 2026-09-14; same oracle, 30 fold
    planes, ``zeta_extrapolation`` 0.27 .. 3484) the ENERGY envelope is
    two-sided in SIGN rather than in magnitude: below the bar the error is
    signed and centred (-2.2 % .. +4.1 %, 12 of 22 rungs negative, mean
    +0.57 %), above it a one-sided GAIN at every rung (+3.6 % .. +30.1 %, 8 of
    8 positive, mean +7.88 %).  The +12.5 % / +22.8 % figures above are WP-B7b's
    three singlets; other optics saturate near +5 %.  For absolute energy /
    encircled energy above the bar, read at a plane whose two branches separate
    further, or use the ray-to-wave hand-off
    ``apply_real_lens_traced(caustic='wave', amplitude_model='ray_density')``,
    which conserved the launched power to better than 1 % at every one of those
    30 planes.

    Which member is CLOSEST is optic-dependent, and the two published readings
    both reproduce (WP-B7c measured five fixtures -- WP-B7b's own two,
    VERIFY-B7b's and two more -- over 42 planes).  On WP-B7b's fast N-LAK22
    singlet the completion is the closer member at 10 of 12 planes (0.9993
    against the hand-off's 0.9974 at its marginal focus); on VERIFY-B7b's
    N-BAF10 biconvex and on two optics of WP-B7c's the hand-off is closer at
    every plane (0.9979 against 0.9915 at ``zeta_extrapolation`` = 1.03).
    Overall 10 of 42.  Both margins are under 0.012 of fidelity, and neither
    ordering generalises -- a beam-width sweep at a fixed optic and plane
    (0.15 .. 0.38 mm, aperture truncation 1.8e-04 .. 0.26 of the rim amplitude)
    does not flip it, so the aperture:beam ratio is not the discriminator
    either.  Read the member ranking on your own prescription before relying on
    it.  The completion beats the plain multibranch at 40 of those 42 planes;
    the two exceptions are the tightest-band planes of WP-B7b's own singlet
    (0.9882 against 0.9889 at ``zeta_extrapolation`` = 173, and 0.9906 against
    0.9910 at 23.1), so even that ordering is not universal.  What IS general
    on every fixture measured is that the hand-off conserves energy where the
    completion above the bar does not.

    The grid matters before any of this does: the fold's Airy layer
    ``l_airy = 1 / (k^(2/3) kappa)`` must be resolved (the gate is
    ``l_airy >= 1.2 dx``) or the call falls back.  On a fast singlet at its
    marginal focus that layer is microns wide, so an aperture-sized grid needs
    several hundred samples across it -- the 640 x 640 / dx = 2.10 um row above
    is the same optic and plane that falls back at dx = 4 um.

    Two REFUSALS (WP-B7c and its round 2)
    -------------------------------------
    This completion keeps the branch sum's bright side verbatim, so a branch
    sum that is wrong by decades makes it wrong by decades -- and none of the
    diagnostics above can see it (at such a plane ``fit_residual``,
    ``zeta_extrapolation`` and ``fell_back`` all read their best values).  Two
    independent readings are therefore taken and the call RAISES rather than
    returning such a field:

    * ``multibranch_power_ratio_bracketed`` -- the branch sum's reconstructed
      grid power over the launch power reaching the grid.  Above
      ``_MB_POWER_RATIO_MAX`` (2.0, the branch sum's own gain tripwire) the
      call refuses.  This arm is a FAR-TAIL tripwire, not a classifier: its
      accepted and broken populations overlap (largest accepted 0.9804 against
      smallest broken 1.106 measured over nine optics), so it has a margin
      below and none above;
    * ``pixel_continuity`` -- the power this call's OWN output deposits, over
      the power the same mapped triangles deposit when rasterised at HALF the
      pitch on the same window from the same launch lattice.  A converged
      point-sampled quadrature deposits the same power at any pitch, so this
      reads 1 on any optic at any grid TO O(1/N) -- measured 0.99941-1.00044
      over 102 planes far from every caustic -- while a quadrature that has
      stopped being unbiased deposits a power proportional to the PIXEL AREA
      and reads ~4 per halving.  Outside ``[1/1.06, 1.06]`` the gain arm
      refuses and the loss arm reports.  Measured on sixteen optics and 394
      oracle-scored fold planes: the 250 returned read 0.8724-1.0595 with
      oracle fidelity 0.9421-0.9985, the 144 refused read 1.0643-3.9988 with
      fidelity 0.0018-0.9520 -- so the two fidelity populations OVERLAP, and
      which of them is "right" is an accept criterion the caller chooses.

    Both readings, their bands and their decisions are in the diagnostics on
    every return path.  REFINING THE GRID IS NOT A WORKAROUND for the second:
    it moves the render toward convergence, but the reading is taken at the
    caller's own pitch and says whether it has arrived.  At a refused plane use
    ``apply_real_lens_traced(caustic='wave', amplitude_model='ray_density')``,
    which scored 0.986-0.999 against the oracle at every plane refused here.

    A plane this passes is not thereby certified: the second reading detects
    ONE failure mode (the quadrature's convergence in the pixel), and on the
    FALLBACK route -- where the field returned is the bright-side-only branch
    sum -- it does not order the returned field's accuracy at all.  That is
    not left for the caller to infer: ``pixel_continuity_of`` names the field
    the number was measured on and ``pixel_continuity_scope`` says what it
    buys on THIS route --

    * ``'returned_field'``            the fold-ring completion: the dark side
                                      is built and the reading is of the whole
                                      returned field;
    * ``'returned_field_quadrature_only'``  a FALLBACK: the reading is of the
                                      returned field, but that field is the
                                      bright-side-only branch sum and the
                                      absent dark tail is NOT arbitrated;
    * ``'underlying_branch_sum'``     the Pearcey cusp route: the number is of
                                      the branch sum the cusp field is built
                                      on and not of the returned field, which
                                      has no half-pitch counterpart to compare
                                      against.

    ``pixel_continuity_scope_note`` carries the same statement in words."""
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced_uniform',
                           input_kind='field')
    E_in = np.asarray(E_in)
    N = E_in.shape[0]
    if E_in.ndim != 2 or E_in.shape[0] != E_in.shape[1]:
        raise ValueError("apply_real_lens_traced_uniform: square 2D field "
                         f"required; got {E_in.shape}.")
    target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128

    # 1. bright-side multibranch (finite through the fold via ludwig)
    # Through the module-private entry rather than the public one, to ask for
    # the PIXEL-HALVING ARBITER's reading as well: the same mapped triangles
    # rasterised onto a half-pitch grid over the same window, which costs one
    # extra rasterisation and no second trace.  The public entry point is
    # unchanged for every other caller, and this call is otherwise identical
    # to the one it replaces.
    E_mb, mb_diag = _multibranch_render(
        E_in, prescription=prescription, wavelength=wavelength, dx=dx,
        output_plane_distance=output_plane_distance,
        output_plane_n=output_plane_n, ray_subsample=ray_subsample,
        min_area_ratio=min_area_ratio, caustic_band=caustic_band,
        input_carrier=input_carrier, return_diagnostics=True,
        pixel_halving_arbiter=True)
    E_mb = np.asarray(E_mb)
    # The half-pitch RENDER is an internal hand-off, not a diagnostic: it is a
    # (2N, 2N) complex image, four times the returned field, and a caller who
    # asked for ``return_diagnostics`` did not ask to be handed that.  Taken
    # out of the dict here -- so it cannot reach the caller through the
    # ``dict(mb_diag)`` copies below -- and dropped as soon as the reading is
    # taken.
    _mb_half_render = mb_diag.pop('pixel_halved_field', None)

    # 1b. the energy DECISION on the field this completion is built on.
    # Read through the multibranch's own gain bracket (the smaller of the two
    # ratios, i.e. the larger denominator) so the wide-aperture geometry
    # artefact that bracket exists for cannot reach the bar.  Recorded in the
    # diagnostics on EVERY return path below -- including the fallbacks, which
    # return the multibranch field itself.
    _mb_pr = mb_diag.get('power_ratio')
    _mb_prt = mb_diag.get('power_ratio_triangles')
    _mb_ratios = [float(v) for v in (_mb_pr, _mb_prt)
                  if v is not None and np.isfinite(v)]
    _mb_bracket = min(_mb_ratios) if _mb_ratios else None
    if _mb_bracket is None:
        _mb_decision = 'no_launched_power'
    elif _mb_bracket > _MB_POWER_RATIO_MAX:
        _mb_decision = 'refused_energy_gain'
    elif _mb_bracket < _MB_POWER_RATIO_MIN:
        _mb_decision = 'energy_loss'
    else:
        _mb_decision = 'ok'
    _mb_energy = {
        'multibranch_power_ratio': (float(_mb_pr) if _mb_pr is not None
                                    else None),
        'multibranch_power_ratio_bracketed': _mb_bracket,
        'multibranch_power_ratio_band': (_MB_POWER_RATIO_MIN,
                                         _MB_POWER_RATIO_MAX),
        'power_ratio_decision': _mb_decision,
    }
    # 1c. the PIXEL-HALVING ARBITER's decision -- the second, independent arm.
    # ``_mb_bracket`` above is the cheap tripwire (it costs nothing, and it
    # catches the far tail); this is the one that decides the cases the
    # tripwire is silent on, because it reads the quadrature's CONVERGENCE
    # rather than its energy against a launch.  See
    # ``_lens_traced_multibranch._PIXEL_CONTINUITY_MAX`` for the mechanism and
    # ``_MB_PIXEL_CONTINUITY_MAX`` below for the bar and its derivation.
    _mb_cont = mb_diag.get('pixel_continuity')
    _mb_energy.update({
        'multibranch_pixel_continuity': (float(_mb_cont)
                                         if _mb_cont is not None else None),
        'pixel_continuity_band': (_MB_PIXEL_CONTINUITY_MIN,
                                  _MB_PIXEL_CONTINUITY_MAX),
        # placed here rather than only in ``_arbitrate`` so the keys exist on
        # every dict this function builds, and a consumer that branches on
        # the scope never has to guess what a missing key meant
        'pixel_continuity_of': None,
        'pixel_continuity_scope': None,
        'pixel_continuity_scope_note': None,
    })

    def _continuity_verdict(c):
        if c is None:
            # The reading is of the RETURNED field.  When it could not be
            # taken -- the fine grid is past the arbiter's entry cap, or the
            # half-pitch render deposited nothing -- the branch sum's own
            # verdict cannot stand in for it: refusing on a number this call
            # is not returning is exactly the mistake round 1 made.  Say it
            # was not measured.
            d = mb_diag.get('pixel_continuity_decision') or 'not_measured'
            return d if d in ('not_requested', 'not_measured')                 else 'not_measured'
        if c > _MB_PIXEL_CONTINUITY_MAX:
            return 'not_converged_gain'
        if c < _MB_PIXEL_CONTINUITY_MIN:
            return 'not_converged_loss'
        return 'ok'

    def _arbitrate(c, what, scope):
        """Record the continuity of the field about to be RETURNED, and
        refuse if the render it came from has not converged in the pixel.

        Read on the RETURNED field, not on the branch sum, because those are
        not the same number where the completion applies: the CFU swap
        rewrites the fold band and the whole dark side, which is exactly where
        a modest branch-sum excess sits.  Measured (WP-B7c round 2,
        2026-09-15) on VERIFY-B7b's own fixture at z = 1761 um, one micron
        before its blow-up window: the BRANCH SUM reads 1.0636 while the
        COMPLETION reads 1.0020, and the completed field's oracle fidelity is
        0.9878 with its power 0.994x the oracle's -- deciding on the branch
        sum's reading there would refuse a field that is right.

        ``what`` names the field the number is a reading OF; ``scope`` says
        what that buys on THIS route, and the two are not the same question
        (WP-B7c round 3, E4 / E7).  See ``_PIXEL_CONTINUITY_SCOPES``.
        """
        if scope not in _PIXEL_CONTINUITY_SCOPES:
            raise AssertionError(
                f'apply_real_lens_traced_uniform: unknown continuity scope '
                f'{scope!r}; expected one of '
                f'{sorted(_PIXEL_CONTINUITY_SCOPES)}')
        d = _continuity_verdict(c)
        _mb_energy['pixel_continuity'] = (float(c) if c is not None else None)
        _mb_energy['pixel_continuity_of'] = what
        _mb_energy['pixel_continuity_scope'] = scope
        _mb_energy['pixel_continuity_scope_note'] = (
            _PIXEL_CONTINUITY_SCOPES[scope])
        _mb_energy['pixel_continuity_decision'] = d
        if d != 'not_converged_gain':
            return
        raise RuntimeError(
            "apply_real_lens_traced_uniform (also reached via "
            "apply_real_lens_traced(caustic='uniform')): the reading this "
            f"call decides on -- taken on {what} -- has NOT CONVERGED in the "
            "output pixel.  "
            "Rasterising the same mapped triangles onto a grid of half the "
            "pitch, over the same physical window and from the same launch "
            f"lattice, gives {1.0 / c:.4g}x this field's power -- a "
            f"continuity ratio of {c:.4g}, above the "
            f"{_MB_PIXEL_CONTINUITY_MAX:g} bar -- with up to "
            f"{int(mb_diag.get('n_branch_max') or 0)} branches on one pixel "
            "and a launched-power ratio of "
            f"{'unavailable' if _mb_bracket is None else f'{_mb_bracket:.4g}x'}"
            f" (INSIDE that arm's {_MB_POWER_RATIO_MAX:g}x bar, which is why "
            "this second reading exists).  The branch sum deposits "
            "dx^2 |E|^2 / ratio wherever a mapped triangle catches a pixel "
            "CENTRE, which estimates the launched energy without bias only "
            "while those triangles are spread over many pixels; where a ring "
            "collapses onto a handful, the written power follows the PIXEL "
            "area instead of the mapped area and falls ~4x per halving.  A "
            "converged render reads ~1 here whatever the pitch.  REFINING "
            "THE GRID IS NOT A WORKAROUND: it moves the render toward "
            "convergence, but this reading is what says whether it has "
            "arrived, and it is taken at the caller's own pitch.  Use the "
            "wave hand-off -- apply_real_lens_traced(caustic='wave', "
            "output_plane_distance=...), which propagates the traced "
            "exit-vertex field with the band-limited angular spectrum and is "
            "exact through folds, cusps and the axial focus alike (measured "
            "0.986-0.999 fidelity against a direct Rayleigh-Sommerfeld "
            "oracle at the very planes refused here) -- or "
            "apply_real_lens_gbd / apply_real_lens_maslov.  Read the ratio "
            "yourself with apply_real_lens_traced_uniform(..., "
            "return_diagnostics=True)['pixel_continuity'].")
    if _mb_decision == 'refused_energy_gain':
        # The MECHANISM sentence is conditioned on this plane's own branch
        # count, not stated unconditionally.  VERIFY-WP-B7c's D3: on a
        # cemented doublet at z = 5.422 / 5.460 mm the refusal is CORRECT
        # (oracle fidelity 0.637 / 0.292) but ``n_branch_max`` reads 1 -- the
        # ray map is single-valued and there is no ring, so a message that
        # printed "up to 1 branches on one pixel" in one clause and "a whole
        # RING of branches coalesces" in the next told the caller something
        # its own reading contradicted, and could not be used to diagnose
        # their plane.  Both narratives below are the SAME quadrature defect;
        # what differs is what compresses the mapped triangles onto too few
        # pixels.
        _nbr = int(mb_diag.get('n_branch_max') or 0)
        _ndeg = int(mb_diag.get('n_triangles_degenerate') or 0)
        _nfin = int(mb_diag.get('n_triangles_finite') or 0)
        if _nbr >= 3:
            _mechanism = (
                "The output plane is at or near the AXIAL point focus, where "
                "a whole RING of branches coalesces -- more than the closest "
                "PAIR the fold-uniform 'ludwig' swap regularizes -- and the "
                "branch sum's point-sampled quadrature stops conserving "
                "energy (measured 5.8x to 6097x of the launched power on "
                "five singlets, with the field's fidelity against a direct "
                "Rayleigh-Sommerfeld oracle falling from 0.99 to 0.01).")
        else:
            _mechanism = (
                f"The ray map here is NOT multi-valued ({_nbr} branch(es) on "
                "the busiest pixel), so no ring of branches is coalescing: "
                "what has failed is the same point-sampled quadrature without "
                "the coalescence.  The map compresses the mapped triangles "
                "onto too few pixels for the point sample to stay an unbiased "
                "estimator of their area "
                f"({_ndeg} of {_nfin} mapped triangles were also skipped as "
                "degenerate here), so the deposited power follows the PIXEL "
                "area instead of the mapped area.  Read "
                "['pixel_continuity'] to see it directly: a converged render "
                "deposits the same power at half the pitch.")
        raise RuntimeError(
            "apply_real_lens_traced_uniform (also reached via "
            "apply_real_lens_traced(caustic='uniform')): the multibranch "
            "bright-side field this completion is built on carries "
            f"{_mb_bracket:.4g}x the launched power that reaches this grid, "
            f"above the {_MB_POWER_RATIO_MAX:g}x refusal bar, with up to "
            f"{_nbr} branches on one pixel.  "
            + _mechanism + "  This is REFUSED rather than passed "
            "through or fallen back, because the fallback target is that same "
            "multibranch field and because none of this function's own "
            "diagnostics can see the defect: the bright-side fit residual, "
            "zeta_extrapolation and fell_back all read their best values "
            "there.  Use the wave hand-off -- apply_real_lens_traced("
            "caustic='wave', output_plane_distance=...), which propagates the "
            "traced exit-vertex field with the band-limited angular spectrum "
            "and is exact through folds, cusps and the axial focus alike "
            "(measured 0.998 fidelity at the very planes refused here) -- or "
            "apply_real_lens_gbd / apply_real_lens_maslov.  Read the ratio "
            "yourself with apply_real_lens_traced_multibranch(..., "
            "return_diagnostics=True)['power_ratio'].")

    def _fallback(reason):
        # Name THIS function first.  A message naming only
        # ``apply_real_lens_traced(caustic='uniform')`` covers one of the
        # two routes here -- a direct
        # ``apply_real_lens_traced_uniform(...)`` caller (a public entry
        # point with no ``caustic`` kwarg at all) would be told to look at
        # a knob they never touched and cannot find in this signature.
        warnings.warn(
            "apply_real_lens_traced_uniform (also reached via "
            "apply_real_lens_traced(caustic='uniform')): the uniform Airy dark-"
            f"side completion does not apply here ({reason}); returning the "
            "plain multibranch field (bright-side only, no dark tail).  The "
            "uniform completion covers a rotationally-symmetric SINGLE fold "
            "RING (collimated / rot-sym input, centred prescription, one "
            "interior caustic); a cusp needs the Pearcey generalisation and a "
            "decentered / astigmatic fold is out of scope -- use "
            "apply_real_lens_gbd / apply_real_lens_fga or single-branch "
            "ray_density + ASM for those.", RuntimeWarning, _caller_stacklevel())
        # On this path the field RETURNED is the branch sum itself, so the
        # arbiter's reading of the branch sum is the reading of the returned
        # field and needs no second completion.
        _arbitrate(_mb_cont, 'the plain multibranch field',
                   'returned_field_quadrature_only')
        out = E_mb.astype(target_cdtype) if E_mb.dtype != target_cdtype else E_mb
        if return_diagnostics:
            d = dict(mb_diag)
            d.update(_mb_energy)
            d.update(fell_back=True, reason=reason, r_c=None, kappa=None,
                     c0=None, c1=None, fit_residual=None, zeta_band=None,
                     zeta_extrapolation=None)
            return out, d
        return out

    # 2. rotational-symmetry / on-axis gate
    from ._lens_traced import _prescription_has_field_frame
    kcx, kcy = mb_diag.get('input_carrier', (0.0, 0.0))
    if _prescription_has_field_frame(prescription):
        return _fallback('decentered/tilted prescription')
    if abs(kcx) > 0.0 or abs(kcy) > 0.0:
        return _fallback('carrier tilt (fold not a centred ring)')
    if not _is_rotationally_symmetric(E_in, dx):
        return _fallback('non-rotationally-symmetric input')

    # 3. meridional fold geometry (r_c, kappa, phi)
    aperture = prescription.get('aperture_diameter')
    if aperture is not None:
        launch_radius = 0.5 * float(aperture) * 0.98
    else:
        launch_radius = 0.5 * N * dx
    fold = _trace_meridional_fold(
        prescription, wavelength, float(output_plane_distance),
        float(output_plane_n), launch_radius, int(n_fan))
    if not fold['ok']:
        # ``n_turn == 2`` is a CUSP (3-branch coalescence); try the Pearcey
        # cusp completion (A1 / niche R2) before falling back to multibranch.
        if fold.get('reason') == 'cusp_or_multiple' and fold.get('n_turn') == 2:
            cusp = _trace_meridional_cusp(
                prescription, wavelength, float(output_plane_distance),
                float(output_plane_n), launch_radius, int(n_fan), E_in, dx)
            if cusp['ok']:
                E_cusp = _build_pearcey_cusp_field(E_mb, cusp, dx)
                if E_cusp is not None and np.all(np.isfinite(E_cusp)):
                    # the Pearcey field is built ON the branch sum, and there
                    # is no half-pitch Pearcey to compare it against without a
                    # second cusp trace, so this path is arbitrated on the
                    # branch sum's own reading
                    _arbitrate(_mb_cont, _PEARCEY_READING_OF,
                               'underlying_branch_sum')
                    E_out = E_cusp.astype(target_cdtype)
                    if return_diagnostics:
                        d = dict(mb_diag)
                        d.update(_mb_energy)
                        d.update(fell_back=False, reason='cusp_ring',
                                 r_c=None, kappa=None, c0=None, c1=None,
                                 fit_residual=None, cusp_r1=cusp['r1'],
                                 cusp_r2=cusp['r2'], cusp_x0=cusp['x0'],
                                 cusp_gamma=cusp['gamma'],
                                 cusp_map_resid=cusp['map_resid'])
                        return E_out, d
                    return E_out
                return _fallback('cusp: non-finite field (guarded)')
            return _fallback('cusp: ' + cusp['reason'])
        # A4 (niche R5): a map with THREE OR MORE interior turning points is a
        # HIGHER catastrophe (swallowtail / butterfly / non-classifiable) -- NOT
        # a clean fold or cusp.  There is no canonical completion here (rare in
        # real lenses, high effort); DETECT the class from the turning-point
        # structure and ROUTE cleanly to the finite multibranch/GBD fallback
        # with a one-time NAMED warning (never inf/nan).
        if fold.get('reason') == 'cusp_or_multiple' \
                and int(fold.get('n_turn') or 0) >= 3:
            n_turn = int(fold['n_turn'])
            cls = _classify_catastrophe(n_turn)
            return _fallback(
                f'{cls}: {n_turn} coalescing fold branches -- a higher '
                f'catastrophe with no canonical analytic completion; routing '
                f'to the finite multibranch field (use apply_real_lens_gbd / '
                f'apply_real_lens_fga for the diffraction-correct field here)')
        return _fallback(fold['reason'])
    r_c = fold['r_c']
    kappa = fold['kappa']
    cphi = fold['cphi']
    k0 = 2.0 * np.pi / wavelength

    # radial coordinate on the wave grid (library centre convention)
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    rgrid = np.sqrt(X * X + Y * Y)

    def _A(rq):
        return np.polyval(cphi, rq - r_c)

    def _zeta(rq):
        return kappa * (r_c - rq)

    # 4. fit the two smooth CFU coefficients (c0 -> Ai term, c1 -> Ai' term) to
    #    the BRIGHT multibranch field over a band just inside r_c, using the
    #    exact CFU kernel as the (linear) basis -- REUSE, no reimplementation.
    # The Airy boundary layer around the caustic has radial scale
    # ``l_airy = 1/(k0^{2/3} kappa)``.  The grid must resolve it (>~1.2 px) for
    # the fit to be meaningful; otherwise fall back (a coarse grid under-samples
    # the fold ring itself).
    l_airy = 1.0 / (k0 ** (2.0 / 3.0) * kappa)
    if l_airy < 1.2 * dx:
        return _fallback('fold Airy scale under-resolved by the grid')
    # Fit over a band ~1 l_airy wide JUST inside r_c: a wider band pulls in the
    # cone-dominated interior and over-fits the Ai' term -> a too-fat tail
    # (validated ~1 l_airy across N=512..1024, dx 1.5-3 um).
    W = float(uniform_fit_halfwidth) if uniform_fit_halfwidth is not None \
        else l_airy
    gap = 0.15 * l_airy
    if r_c - W <= gap:
        return _fallback('fit band collapses (fold too small vs grid)')
    bright = (rgrid >= r_c - W) & (rgrid <= r_c - gap)
    if int(bright.sum()) < 24:
        return _fallback('too few bright-band pixels for the fit')
    # How far past the two-branch band this evaluation carries zeta.  The fit
    # band is W wide and the dark fill reaches _AIRY_TAIL_CELLS * l_airy; both
    # read zeta at radii the kappa fit never saw when the two-branch band is
    # narrower.  Reported always (diagnostics), warned above the measured bar.
    zeta_band = float(fold.get('band') or 0.0)
    zeta_extrap = (float(W / zeta_band) if zeta_band > 0.0 else float('inf'))
    if zeta_extrap > _ZETA_EXTRAPOLATION_MAX:
        warnings.warn(
            "apply_real_lens_traced_uniform (also reached via "
            "apply_real_lens_traced(caustic='uniform')): the fold's "
            f"zeta(r) = kappa (r_c - r) was fitted on a two-branch band only "
            f"{zeta_band * 1e9:.3g} nm wide and is being carried across a "
            f"{W * 1e9:.3g} nm fit band ({zeta_extrap:.0f}x) and a "
            f"{_AIRY_TAIL_CELLS * l_airy * 1e9:.3g} nm dark fill, so the two "
            "CFU coefficients and the tail they continue are an EXTRAPOLATION "
            "of the fold normal form, not a fit to it.  MEASURED against a "
            "brute-force Rayleigh-Sommerfeld oracle on four optics (WP-B7b, "
            "WP-B7c): below this bar the completed field's total power error "
            "is signed and centred on the truth (-2.2 % to +4.1 %, 12 of 22 "
            "planes low); above it, it is a one-sided GAIN at every plane "
            "measured (+3.6 % to +30.1 %, 8 of 8) -- the shape is still the "
            "better one (it beats the plain multibranch at every measured "
            "plane, which is why this is a warning and not a fallback), but "
            "absolute energy / encircled energy in the dark tail is "
            "unreliable here.  Read at a plane where the two branches separate "
            "further (a caustic ring with a wider two-branch band), or use "
            "apply_real_lens_traced(caustic='wave', amplitude_model="
            "'ray_density') -- which conserved the launched power to better "
            "than 1 % at every one of those planes -- or apply_real_lens_gbd / "
            "apply_real_lens_fga for absolute energy.",
            RuntimeWarning, _caller_stacklevel())
    rb = rgrid[bright]
    Eb = E_mb[bright]
    # Basis = the EXACT CFU kernel (reused, no reimplementation), evaluated at
    # (a0, a1) = (1, 0) and (0, 1); linear in the coefficients, so fitting
    # (c0, c1) here and evaluating _fold_airy_eval(..., c0, c1) at zeta<0 below
    # continues the SAME uniform field analytically to the dark side.  The
    # per-pixel rows are weighted by 1/sqrt(r) so each RADIUS carries equal
    # weight (undoing the circumference ~r pixel-count bias, which otherwise
    # over-weights the pixels nearest r_c and over-fits the Ai' term); this is
    # a binning-free, grid-robust radially-uniform least squares.
    wt = 1.0 / np.sqrt(rb)
    basis0 = _fold_airy_eval(k0, _A(rb), _zeta(rb), 1.0, 0.0)
    basis1 = _fold_airy_eval(k0, _A(rb), _zeta(rb), 0.0, 1.0)
    G = np.stack([basis0, basis1], axis=1)
    Gw = G * wt[:, None]
    Ew = Eb * wt
    coef, *_ = np.linalg.lstsq(Gw, Ew, rcond=None)
    c0, c1 = complex(coef[0]), complex(coef[1])
    fit_resid = float(np.linalg.norm(Gw @ coef - Ew)
                      / (np.linalg.norm(Ew) + 1e-300))
    if not (np.isfinite(c0) and np.isfinite(c1)) or fit_resid > 0.5:
        return _fallback(f'uniform fit residual too large ({fit_resid:.2f})')

    # 5. fill the DARK side (r > r_c): analytic continuation to zeta < 0 -> the
    #    exponential Airy tail.  Clamp the Airy argument far out (no overflow;
    #    the physical tail -- arg <~ 15 -- is untouched).
    #
    # Restricted to the annulus the tail actually occupies.  The dark-side
    # amplitude is ``Ai((r - r_c)/l_airy)``, which is 6e-19 at 15 l_airy and is
    # clamped to numerical zero by ``_AIRY_ARG_CAP`` beyond ~50 anyway -- so
    # evaluating the CFU kernel (a scipy ``airy`` plus a complex ``exp``) on
    # EVERY pixel outside r_c produced values the caller cannot distinguish
    # from the zero ``E_mb`` already holds there.  Measured on an f/2
    # plano-convex at lambda = 1 um, output_plane_distance = 1.9 mm: the
    # physically non-zero annulus is r_c < r < r_c + 15 l_airy = 21.9..67 um,
    # ~0.6 % of the pixels at N = 2048, while the fill covered 4 193 535 of
    # 4 194 304 (100.0 %) and cost 25.4 s against the multibranch's 0.85 s
    # (93.1 s against 11.9 s at N = 4096).
    # -k0^{2/3} zeta = k0^{2/3} kappa (r - r_c) >= 0 on the dark side; cap it
    # (belt and braces -- the annulus below already bounds the argument at
    # ``_AIRY_TAIL_CELLS``, well inside the cap).
    zfloor = -_AIRY_ARG_CAP / (k0 ** (2.0 / 3.0))

    def _fill(E_mb_r, rgrid_r):
        """The dark-side fill, as a function of the OUTPUT sampling alone.

        Factored so the SAME fold parameters (``r_c``, ``kappa``, the cone
        phase and the two fitted CFU coefficients) can be applied to the
        half-pitch render for the arbiter below.  Re-using the coarse grid's
        FIT there is deliberate: it isolates the rasterisation difference,
        which is what the control measures, instead of adding a second
        least-squares fit's own noise to the comparison.  Called with
        ``(N, dx)`` for the returned field, byte-identically to the
        straight-line code it replaces.
        """
        out = E_mb_r.astype(np.complex128, copy=True)
        dark_r = (rgrid_r > r_c) & (rgrid_r < r_c
                                    + _AIRY_TAIL_CELLS * l_airy)
        rd_r = rgrid_r[dark_r]
        zd_r = np.maximum(_zeta(rd_r), zfloor)
        out[dark_r] = _fold_airy_eval(k0, _A(rd_r), zd_r, c0, c1)
        return out

    E_out = _fill(E_mb, rgrid)
    if not np.all(np.isfinite(E_out)):
        return _fallback('non-finite dark-side fill (guarded)')

    # 5b. the PIXEL-HALVING ARBITER, on the field this call is about to
    # return.  The half-pitch branch-sum render already exists (the arbiter
    # paid for it inside the branch sum); completing it with the SAME fold
    # parameters costs one masked ``_fold_airy_eval`` over a thin annulus.
    _uni_cont = None
    _E_half = _mb_half_render
    if _E_half is not None:
        _dx_h = 0.5 * dx
        # The SAME lattice the branch sum rasterised the half-pitch render
        # onto, taken from the one definition rather than rebuilt here
        # (WP-B7c round 3, E5).  Rebuilt inline this was a seam the two could
        # drift apart at, and the reading is a ratio of two integrals: if the
        # completion is evaluated at radii the render does not have, the two
        # halves are taken on different geometry.  See
        # ``_lens_traced_multibranch._HALF_PITCH_CENTRE_OFFSET`` for the
        # convention (the fine lattice NESTS on the coarse one) and for what
        # that costs at the window edge.
        _xh = half_pitch_centres(N, dx)
        _Xh, _Yh = np.meshgrid(_xh, _xh)
        _E_half_full = _fill(np.asarray(_E_half),
                             np.sqrt(_Xh * _Xh + _Yh * _Yh))
        _p_half = float(np.sum(np.abs(_E_half_full) ** 2)) * (_dx_h * _dx_h)
        _p_coarse = float(np.sum(np.abs(E_out) ** 2)) * (dx * dx)
        if _p_half > 0.0 and _p_coarse > 0.0:
            _uni_cont = _p_coarse / _p_half
        del _E_half_full
    _mb_half_render = None
    _arbitrate(_uni_cont, 'the completed fold field', 'returned_field')

    E_out = E_out.astype(target_cdtype)

    if return_diagnostics:
        d = dict(mb_diag)
        d.update(_mb_energy)
        d.update(fell_back=False, reason='fold_ring', r_c=r_c, kappa=kappa,
                 c0=c0, c1=c1, fit_residual=fit_resid, fit_halfwidth=W,
                 n_turn=fold['n_turn'], zeta_band=zeta_band,
                 zeta_extrapolation=zeta_extrap,
                 # the linear zeta's own curvature bound and the depth the
                 # dark tail was actually filled to, both in metres, so a
                 # caller can see whether the fill outran the normal form on
                 # THEIR optic (it does on WP-B7b's fixture F: 17.2 l_airy
                 # against the 20 the fill covers)
                 zeta_curvature=fold.get('zeta_curvature'),
                 zeta_linear_range=fold.get('zeta_linear_range'),
                 zeta_linear_resid=fold.get('zeta_linear_resid'),
                 dark_fill_depth=float(_AIRY_TAIL_CELLS * l_airy),
                 l_airy=float(l_airy))
        return E_out, d
    return E_out
