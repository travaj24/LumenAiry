"""
lumenairy.elements._lens_traced_multibranch -- MULTI-BRANCH (multi-arrival)
ray-traced lens field, valid THROUGH focus and caustics.

:func:`apply_real_lens_traced` assumes the entrance->output ray map is
single-valued (one ray per output pixel), so it breaks down through focus /
at caustics where several rays arrive at the same point.  This module
constructs the full MULTI-VALUED geometric field by the wavefront-construction
/ triangulated-ray-map method of the seismology literature:

* Triangulate the regular entrance launch grid; map each triangle by its
  traced output-plane vertex positions; every output pixel covered by a
  mapped triangle receives ONE arrival branch from it (Lambare, Lucio &
  Hanyga 1996, Geophys. J. Int. 125, 584, Secs 5-6; Vinje, Iversen &
  Gjoystdal 1993, Geophysics 58, 1157).
* Per-branch OPL by the second-order "intrapolation" formula
  ``T(x) = sum_i alpha_i [ T_i + 1/2 (x - x_i) . p_i ]`` with ``p_i`` the
  transverse direction cosines (Kraaijpoel 2003, PhD thesis Utrecht,
  eq. 5.7 -- the 1/2 is exact for quadratic T, NOT a typo).
* Per-branch amplitude ``1/sqrt|J|`` from the SIGNED-AREA RATIO of the mapped
  to launch triangle (Chambers & Kendall 2008, Geophys. J. Int. 173, 1030,
  eq. 1 "ratio method" -- more robust near degenerate configurations than the
  differential Jacobian).
* KMAH (Maslov) index per ray: on the homogeneous exit leg the transverse
  Jacobian is LINEAR in distance, so ``det Q(z) = det Q0 + z tr(adj(Q0) Qdot)
  + z^2 det Qdot`` is an exact quadratic whose roots are the two astigmatic
  focal-line crossings -- caustic passages are counted analytically (Cerveny,
  Seismic Ray Theory, CUP 2001 dynamic ray tracing; Klimes 2010, Stud.
  Geophys. Geod. 54, 269, App. B; Cormier, Treatise on Geophysics ch. 3.04
  eq. 23).  Each fold crossing multiplies the branch by ``exp(-i pi/2)``
  (time convention ``exp(-i w t)``, field ``exp(+i k OPL)`` -- Klimes 2010
  Sec. 1 verbatim; = the Gouy phase in optics, Visser & Wolf 2010).
* In-glass caustics: each surface-to-surface leg is a homogeneous straight
  segment, so its ``det Q(z)`` is the same exact quadratic the exit leg uses
  -- the internal focal-line crossings are counted per leg and added to the
  index (0 for an air-focus system).  A runtime parity assertion
  ``sgn det J = (-1)^m`` (Mitrofanov & Priimenko 2013 eq. 1) then guarantees
  the EXACT parity; if the direct in-glass count disagrees with it (an
  internal focus the global-frame leg quadratic under-resolves), the parity
  is still enforced and a warning is raised.
* AT the caustic itself (degenerate mapped triangles) the ART amplitude is
  not evaluated -- the literature-standard choice (Vinje 1993 leaves it
  undefined; Lambare 1996 notes the singularity is inherent to ART); the
  caustic band is O(k^-2/3) wide (~1 output pixel at optical wavelengths,
  Kravtsov & Orlov criterion k|dS| <~ pi), and the plain two-branch sum is
  accurate "up to the first Airy peak" outside it.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from .. import raytrace as rt

# ``_lens_kernels`` is a leaf too: the warning attribution helper walks
# out to the caller's frame so a notice names the user's line.
from ._lens_kernels import caller_stacklevel as _caller_stacklevel

# Configuration objects (audit 2026-09-11 TESTS-ARCH section 14 item 13).
# ``lens_config`` is a LEAF -- it imports nothing from lumenairy at module
# scope -- so this edge is one-way and adds no import cost.
from .lens_config import (
    LensConfig,
    LensGeometry,
    LensNumerics,
    LensResources,
    _wants_config,
    resolve_entry_point_kwargs as _resolve_lens_config,
)

__all__ = ['apply_real_lens_traced_multibranch', 'ludwig_fold']

#: ``scipy.special.airy``, bound on FIRST USE by :func:`_airy`.  It used to be
#: a module-level ``from scipy.special import airy``, which made this file the
#: last module-level ``scipy.special`` importer outside ``backend/scipy.py``:
#: importing it charged ``scipy.special``'s ~540 ms shared prefix
#: (``scipy._lib._array_api`` -> ``array_api_compat.numpy`` -> ``numpy.f2py``
#: -> ``charset_normalizer`` -> ``numpy.testing``) to every ``import
#: lumenairy``, whether or not a caustic band was ever folded.
_scipy_airy = None


def _airy(z):
    """Return ``scipy.special.airy(z)``, importing ``scipy.special`` on first use.

    One module-global read and one ``is None`` test per call once bound --
    strictly cheaper than the in-function ``from scipy.special import airy``
    this replaced (a ``sys.modules`` lookup plus an attribute fetch), and
    cheaper still than paying the import for callers who never fold a band.
    :func:`ludwig_fold` is fully elementwise, so this runs once per fold, not
    once per pixel.
    """
    global _scipy_airy
    if _scipy_airy is None:
        from scipy.special import airy as _a
        _scipy_airy = _a
    return _scipy_airy(z)

# Half-open (top-left) rasterization tie-break tolerance, in BARYCENTRIC units.
# A pixel with |a_i| <= _EDGE_TOL is treated as lying ON mapped edge_i (see the
# rasterizer coverage test).  It sits far above the ~1e-14 boundary noise of an
# exactly-commensurate map (pixel centres land ~1 ULP off the mapped nodes) and
# far below the O(0.1..1) barycentric coordinates of genuine interior pixels.
_EDGE_TOL = 1e-9

# Energy-conservation tripwire (Scope note D5): at a rotationally-symmetric
# AXIAL point focus a whole RING of branches coalesces, and the reconstructed
# grid power blows up ~1e5..1e6x (probe: ~1e6x at the BFL plane).  A
# well-behaved through-focus field stays within ~1.2x of the input aperture
# power, so a warn above this multiple flags the catastrophe with no false
# positives.
#
# WHAT THE BLOW-UP IS (measured, WP-B7c 2026-09-14; the three controls are in
# ``validation/probe_multibranch_zeta/probe2_mechanism.py``).  It is the
# rasteriser's POINT-SAMPLED AREA QUADRATURE losing its unbiasedness, not a
# wrong branch, not an unclipped Jacobian and not the fold-member ordering:
#
#   * a mapped triangle deposits ``|E_in|^2 / ratio`` over EVERY pixel whose
#     CENTRE it covers.  Near a caustic essentially every mapped triangle is
#     sub-pixel (median mapped area 1/430 of a pixel on the fixture below, at
#     every plane including the healthy ones), so the write is a Monte-Carlo
#     estimator of the area integral: a triangle of mapped area ``A`` catches a
#     pixel centre with probability ``A/dx^2`` and then deposits
#     ``dx^2 |E_in|^2 / ratio``, whose expectation is exactly the launched
#     ``A_tri |E_in|^2``.  That is why the healthy planes conserve energy at
#     all.  The estimator is unbiased only while the triangles are SPREAD over
#     many pixels; where a whole RING collapses onto a handful of pixels the
#     variance becomes the mean;
#   * ``caustic_band='plain'`` reproduces the blow-up to 0.1 % (6090 against
#     the 'ludwig' 6097 at z = 1768 um on the fixture below), so the pair swap
#     is not involved;
#   * ``min_area_ratio`` 1e-8 and the 1e-6 default give an IDENTICAL ratio
#     (6096.7), i.e. the clip is inoperative at the default -- the divergent
#     amplitudes sit at area ratios 1e-6..1e-3 (``1/sqrt|J|`` of 32..1000),
#     which it admits.  Raising it to 1e-4 cuts the ratio to 93 and to 1e-3 to
#     0.53, at the cost of skipping 4138 and 52384 of 64336 triangles;
#   * REFINING the launch lattice makes it WORSE, by 7.4x
#     (``ray_subsample`` 2 -> 1: 6097 -> 44878; 4 -> 2: 771 -> 6097), the
#     signature of a quadrature whose written energy scales with the triangle
#     COUNT instead of with their mapped area.  ``n_branch.max()`` on one pixel
#     rises 2 -> 97 -> 565 -> 797 across the same four planes.
#
# Fixture: N-BAF10 biconvex R = +/-2.6 mm, t = 0.70 mm, 0.90 mm aperture,
# lambda = 1.064 um, N = 512, dx = 2.20 um, w0 = 330 um, z = 1750..1790 um
# (VERIFY-B7b section 4.1's own fold fixture); reproduced on two more optics in
# ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7c_REPORT.md``.
# Bounding the reconstruction would mean replacing the point sample by an
# area-weighted splat -- a different quadrature that moves every multibranch
# field near a caustic -- so WP-B7c left the reconstruction alone and put the
# REFUSAL in the uniform completion that builds on it
# (``_lens_traced_uniform._MB_POWER_RATIO_MAX``).
_ENERGY_BLOWUP_FACTOR = 2.0

# Lower arm of the same tripwire.  The upper arm alone could only ever see
# energy GAIN, so the two silent failure modes on the other side were
# invisible: the total collapse at an axial focus (every mapped triangle
# degenerate -> an identically ZERO field, measured P/P_in = 0 at the BFL of an
# f = 25 mm singlet) and the 11 % LOSS a resolved fold shows from the skipped
# degenerate triangles plus the missing dark-side tail (measured P/P_in = 0.887
# at N = 4096 against an independent ray-to-wave ASM oracle's 1.000).  0.5 sits
# below any fold the audit measured (0.887 / 0.888) and far above a collapse.
_ENERGY_COLLAPSE_FACTOR = 0.5

# The upper arm was 10.0, which is above the 4-8x the reconstructed power
# reaches in the ~100 um band BEFORE an axial focus (measured 3.99 at
# z = 24.70 mm, 8.10 at 24.74 mm on a 24.834 mm BFL) -- so that band was
# silent too.  A well-behaved through-focus field stays within ~1.2x of the
# input aperture power and a RESOLVED fold within ~1.18x, so 2.0 keeps the
# legitimate cases quiet while catching the run-up to the catastrophe.

# ==========================================================================
# The PIXEL-HALVING ARBITER (WP-B7c round 2).
#
# WHAT IT MEASURES.  The rasteriser above deposits ``dx^2 |E_in|^2 / ratio``
# on every pixel whose CENTRE a mapped triangle covers.  Near a caustic
# essentially every mapped triangle is sub-pixel, so that write is a
# Monte-Carlo estimator of the area integral: a triangle of mapped area ``A``
# catches a pixel centre with probability ``A/dx^2`` and deposits
# ``dx^2 |E_in|^2 / ratio``, whose EXPECTATION is the launched
# ``A |E_in|^2`` -- independent of ``dx``.  So a render that has converged
# deposits the same total power at any pixel pitch, and one that has not
# deposits a power that scales with the PIXEL AREA.
#
# Re-rasterise the SAME mapped triangles onto a HALF-PITCH grid over the same
# physical window (``_render(2 * N, dx / 2)`` -- the trace, the triangulation
# and the mapped areas are unchanged, so this costs one rasterisation and no
# second trace) and take
#
#     pixel_continuity = p_out(dx) / p_out(dx / 2).
#
# A converged render reads ~1.  A quadrature failure reads ~4 per halving,
# because its deposited power is proportional to the pixel area.  This is the
# control VERIFY-WP-B7c ran by hand on four optics: holding optic, plane,
# launch lattice and window fixed and halving the pixel divides the EXCESS by
# exactly 4 (S 107 -> 27.3 -> 7.40, M 504 -> 127 -> 32.2, Q 11238 -> 2810 ->
# 703, F 2.98 -> 1.42 -> 1.04) while a healthy plane is invariant
# (0.937 -> 0.937 -> 0.938), identically on two builds.
#
# WHY IT IS THE RIGHT QUANTITY TO DECIDE ON, where ``power_ratio`` is not.
# ``power_ratio`` compares the deposited power against the LAUNCHED power, so
# it mixes the quadrature's failure with everything else that legitimately
# moves the ratio -- the dark-side tail the branch sum drops (0.76-0.99 on the
# fold planes measured), the triangles that straddle the grid boundary, light
# that leaves the window.  That is why its accepted and broken populations
# OVERLAP: VERIFY-WP-B7c measured a largest accepted reading of 0.9804 against
# a smallest broken one of 1.106, a gap of 1.13x against a 2.0 bar.  The
# continuity ratio cancels every one of those terms -- both renders drop the
# same tail, straddle the same boundary and lose the same off-screen light --
# and leaves only the pixel-area scaling.
#
# WHERE THE READING IS TAKEN.  This module measures the reading on its OWN
# render, which is the mechanism; the consumer that decides on it takes it on
# the field IT returns (``_lens_traced_uniform._arbitrate``), because the two
# are not the same number where the uniform completion applies -- the CFU swap
# rewrites the fold band and the whole dark side, which is exactly where a
# modest branch-sum excess sits.  The half-pitch RENDER is therefore handed
# out in the diagnostics (``pixel_halved_field``) as well as its power, so the
# consumer can put its own output through the same control.  Measured on
# VERIFY-B7b's own fixture at z = 1761 um: this render reads 1.0636 while the
# completion built on it reads 1.0020, and the completed field's oracle
# fidelity is 0.9878 with its power 0.994x the oracle's.
#
# MEASURED (WP-B7c round 2, 2026-09-15), on EIGHT optics -- VERIFY-B7b's own
# N-BAF10 biconvex, VERIFY-WP-B7c's four (a cemented doublet, a positive
# meniscus, a convex-first plano-convex at 532 nm, a fast N-LASF9 at f/2.0)
# and three this round adds (an AIR-SPACED doublet, a CONIC-surfaced N-LAK22,
# and an N-SF11 at NA 0.33, above the 0.12-0.29 envelope every published
# fixture sits in) -- over 82 oracle-scored fold-ring planes, with the
# populations, the probes and their JSON in
# ``validation/probe_wp_b7c_round2/``:
#
#   * the 67 planes the guard RETURNS read 0.9860 .. 1.0221 and their fields
#     score 0.9593 .. 0.9985 against the oracle;
#   * the 15 it REFUSES read 1.092 .. 3.998 and their fields score
#     0.0173 .. 0.9302.
#
# The two FIDELITY populations do not overlap, and that statement is made with
# no accept bar chosen: the split is the guard's own.  Round 1's reading could
# not do this -- its accepted and broken populations overlap (largest accepted
# 0.9804 against smallest broken 1.106) -- which is why this second arm
# exists.
#
# THE BAR.  The gap between the largest returned reading (1.0221, the fast
# singlet at z = 1073 um, fidelity 0.9742) and the smallest refused one
# (1.092, the same optic at z = 1074 um, fidelity 0.9302) is 1.0683x, whose
# geometric centre is 1.0564 -- 1.06 to three figures.  Margins: 1.037x above
# the largest returned reading and 1.030x below the smallest refused one.
# Neither is decades, and that is stated rather than papered over: this is a
# 7 % gap, measured on eight optics and two builds, and a ninth optic could
# narrow it.  What makes it usable where round 1's was not is that it is a
# gap at all, and that the quantity has a FIXED reference -- a converged
# quadrature reads 1 exactly, at any pitch, on any optic -- so the bar is a
# tolerance on a known value rather than a boundary between two moving
# populations.
#
# Upper arm: above this the render has NOT converged in the pixel and the
# uniform completion built on it REFUSES
# (``_lens_traced_uniform._MB_PIXEL_CONTINUITY_MAX``).
_PIXEL_CONTINUITY_MAX = 1.06

# Lower arm of the same reading, REPORTED and not refused.  A reading BELOW
# its half-pitch value means the coarse render is missing deposits the finer
# one catches -- the mirror of the same non-convergence, and the one that runs
# to the identically-zero field at an axial focus (already refused above).
# Placed symmetrically in the LOG of the ratio, so the band is
# ``[1/_PIXEL_CONTINUITY_MAX, _PIXEL_CONTINUITY_MAX]`` and neither arm is
# arbitrary relative to the other.
#
# IT DOES NOT REFUSE, and the asymmetry is physical rather than timid: the
# completion keeps the branch sum's BRIGHT side verbatim, so a bright-side
# excess reaches the caller's field, while the DARK side is exactly what the
# completion replaces, so a dark-side deficit does not.  Measured (WP-B7c
# round 2, 2026-09-15) on the air-spaced doublet at z = 2903 um: the branch
# sum reads 0.9419 -- 6 % short -- while the completed field's power is
# 0.997x the oracle's and its fidelity 0.9910.  Over the whole round-2
# population no fold-ring plane is refused on this arm and none needs to be;
# the only loss-side readings that correspond to a wrong field are on
# FALLBACK planes, where the module has already warned that the completion
# does not apply, and there the nearest returned reading (0.854, oracle
# fidelity 0.904) and the nearest wrong one (0.726, fidelity 0.837) leave
# 1.18x on ONE optic -- not a population a bar can be derived on.
_PIXEL_CONTINUITY_MIN = 1.0 / _PIXEL_CONTINUITY_MAX

# Entry cap on the ARBITER's own render, in array entries of the FINE grid.
# The fine render allocates one complex128 image of ``(2 N)^2``, and the
# consumer that completes it allocates one more, so the working set is
# ``2 x 16 x 4 N^2`` bytes: 67 MB at N = 512, 218 MB at N = 1304 and 1.07 GB
# at N = 2048 -- the last of which is past the working-set discipline the
# rasterisation batches below are held to.  7e6 entries puts the cap at
# N = 1322, which is chosen rather than N = 1024 for a measured reason: the
# grid refinement VERIFY-WP-B7c's D2 is about (a caller halving dx to resolve
# the Airy layer) takes a 640-grid fixture to 1280, and a cap that stopped
# reporting THERE would reproduce the very defect the arbiter closes -- the
# guard disappearing exactly when the caller refines.
#
# Above the cap the reading is reported as ``None`` with
# ``pixel_continuity_decision='not_measured'`` rather than silently skipped,
# so a consumer can tell "converged" from "not asked".  Every fold fixture in
# WP-B7b / VERIFY-B7b / WP-B7c / this round is N = 256 .. 768.
_ARBITER_MAX_FINE_ENTRIES = 7_000_000

# Entry budget for one rasterisation batch, in array ENTRIES (float64
# equivalents).  The bucket batch used to be unbounded: at N = 4096 with
# default arguments the worst bucket held 1 187 616 triangles at 8x8 = 0.61 GB
# per temporary, ~15-18 of them alive at once -> 7.74 GB traced / 12.2 GB RSS
# for ONE call, which drove this workstation into swap.  4e6 entries is 32 MB
# per float64 temporary, i.e. well under 1 GB for the whole working set, and
# is the same order as the sibling budgets in ``_lens_imap`` and
# ``_lens_traced`` (``_IMAP_FIT_CHUNK_ENTRIES``, ``_CHEB_FIT_CHUNK_ENTRIES``).
# Chunking changes no arithmetic -- only ``np.add.at``'s summation order on a
# multi-branch pixel, which this module already documents as order-dependent.
_RASTER_CHUNK_ENTRIES = 4_000_000

# Above this many DISTINCT exact bounding-box shapes the per-shape Python loop
# starts to cost more than the padding it saves, so shapes are coalesced into
# power-of-two classes instead.  Real maps are far
# below it: a near-identity map has one or two shapes, a compressed near-focus
# map a handful.
_RASTER_MAX_SHAPES = 96


def _raster_batches(wxs, wys):
    """Yield ``(idx, wbx, wby, padded)`` rasterisation batches.

    ``idx`` indexes triangles that share the batch box ``wbx x wby``; every
    batch is capped at :data:`_RASTER_CHUNK_ENTRIES` array entries
    (``len(idx) * wbx * wby``) so the ``(n, wbx, wby)`` temporaries stay
    bounded however many triangles a shape holds.

    ``padded`` is False when the box is each triangle's EXACT integer bounding
    box (so every candidate pixel is inside the clipped grid range and the
    caller needs no range mask), and True on the coalesced fallback: when a map
    produces more than :data:`_RASTER_MAX_SHAPES` distinct shapes, they are
    # grouped by rounding each axis UP to the next power of two, which
    # keeps the Python-level loop short at the cost
    of some padded candidates the caller must mask off.  Either way the
    contribution SET is the same; only padding work differs.
    """
    wxs = np.asarray(wxs, dtype=np.int64)
    wys = np.asarray(wys, dtype=np.int64)
    if wxs.size == 0:
        return
    bx, by, padded = wxs, wys, False
    key = bx * (int(by.max()) + 1) + by
    shapes = np.unique(key)
    if shapes.size > _RASTER_MAX_SHAPES:
        bx = (1 << np.ceil(np.log2(np.maximum(wxs, 1))).astype(np.int64))
        by = (1 << np.ceil(np.log2(np.maximum(wys, 1))).astype(np.int64))
        padded = True
        key = bx * (int(by.max()) + 1) + by
        shapes = np.unique(key)
    for kk in shapes:
        s_all = np.nonzero(key == kk)[0]
        wbx = int(bx[s_all[0]])
        wby = int(by[s_all[0]])
        step = max(1, _RASTER_CHUNK_ENTRIES // max(wbx * wby, 1))
        for b0 in range(0, s_all.size, step):
            yield s_all[b0:b0 + step], wbx, wby, padded


def _top_left(ex, ey):
    """Top-left rule: an on-edge pixel is owned by the triangle whose mapped
    edge vector ``(ex, ey)`` points up (``ey > 0``) or is horizontal pointing
    right (``ey == 0 and ex > 0``).  Two triangles sharing a physical edge see
    it with opposite direction vectors, so exactly one of them owns the pixel.
    """
    return (ey > 0.0) | ((ey == 0.0) & (ex > 0.0))


def ludwig_fold(k, S_plus, S_minus, A_plus, A_minus):
    """Uniform (Ludwig 1966 / Kravtsov 1964) FOLD-caustic field from the two
    coalescing ray branches -- finite exactly where the branch amplitudes
    diverge, in the RAY-NATIVE parametrization (branch eikonals + amplitudes).

    For two branches with eikonals ``S+ >= S-`` [m] and COMPLEX amplitudes
    ``A+-`` (each INCLUDING its own Maslov/SPA phase; '+' = the branch that has
    touched the caustic)::

        phi = (S+ + S-)/2,   rho = [3/4 (S+ - S-)]^(2/3)
        g0  = rho^{1/4}/sqrt2 (A+ - i A-)
        g1  = rho^{-1/4}/sqrt2 (A+ + i A-)
        u   = sqrt(2 pi) k^{1/6} e^{i pi/4} e^{i k phi}
              [ g0 Ai(-k^{2/3} rho) + i k^{-1/3} g1 Ai'(-k^{2/3} rho) ]

    (Ludwig, Comm. Pure Appl. Math. 19, 215 (1966), as reproduced in
    Giannopoulou/Makrakis arXiv:1706.02958 eqs. 2.30-2.39 and Grillo & Cordes
    arXiv:1810.09058 eqs. 31-44.)  The combinations ``(A+ - iA-)rho^{1/4}`` and
    ``(A+ + iA-)rho^{-1/4}`` stay FINITE at the fold although ``A+-`` diverge;
    on the bright side the formula reduces exactly to the plain two-branch sum
    ``A+ e^{ikS+} + A- e^{ikS-}``.  Validated to machine precision (~3e-16)
    against the exact cubic-phase integral for both regimes, including exactly
    on the caustic.  Use inside the Kravtsov-Orlov band ``k|S+ - S-| <~ pi`` to
    replace a coalescing pair in a multi-branch sum (the plain sum is accurate
    outside it, "up to the first Airy peak").  Amplitudes must be COMPLEX
    (real amplitudes without their Maslov phases give wrong interference).

    Fully ELEMENTWISE: pass whole arrays of the four branch quantities and get
    the array of uniform fields back -- so ``scipy.special.airy`` is reached
    ONCE per fold, not once per pixel.  It comes through :func:`_airy`, which
    binds it to a module global on first use: a per-call ``from scipy.special
    import airy`` would cost a ``sys.modules`` lookup on every fold, and the
    module-level import this replaced charged ``scipy.special`` to every
    ``import lumenairy``.
    """
    phi = 0.5 * (S_plus + S_minus)
    rho = (0.75 * (S_plus - S_minus)) ** (2.0 / 3.0)
    r14 = rho ** 0.25
    g0 = r14 / np.sqrt(2.0) * (A_plus - 1j * A_minus)
    g1 = (A_plus + 1j * A_minus) / (r14 * np.sqrt(2.0) + 1e-300)
    ai, aip, _, _ = _airy(-(k ** (2.0 / 3.0)) * rho)
    return (np.sqrt(2 * np.pi) * k ** (1.0 / 6.0) * np.exp(1j * np.pi / 4)
            * np.exp(1j * k * phi)
            * (g0 * ai + 1j * k ** (-1.0 / 3.0) * g1 * aip))


def _trace_launch_grid(prescription, wavelength, launch_radius, n_launch,
                       output_plane_distance, output_plane_n,
                       L0=0.0, M0=0.0):
    """Trace a regular entrance grid to the output plane.

    ``L0``/``M0`` are the launch direction cosines of the input congruence
    (0 = collimated on-axis; a tilted/carrier input launches every ray along
    its input phase plane).  The input eikonal on the ``z = 0`` entrance
    plane, ``T_in = L0 x + M0 y``, is included in the returned OPL so the
    branch phases carry the full input carrier exactly.

    THE OUTPUT PLANE IS A PLANE.  ``rt.trace`` leaves every ray at its
    intersection with the LAST SURFACE, i.e. at ``z = sag(rho)``, so the rays
    must first be transferred to the exit VERTEX plane (``z = 0``) through the
    exit medium -- :meth:`lumenairy.raytrace.TraceResult.at_exit_vertex`, the
    one shared implementation of that signed operator -- before the
    free-space leg ``t = output_plane_distance / N`` is added.  Advancing
    ``image_rays`` by ``d / N`` directly evaluates every ray at
    ``z = sag(rho) + d``: a RAY-DEPENDENT longitudinal position, not a plane,
    so two branches reaching the same ``(x, y)`` from different ``rho`` are
    interfered at different ``z``.  Measured on a plano-convex with
    ``R2 = -25 mm`` over a 20 mm aperture: 477 um of transverse error and
    3501 waves of OPD, at ``output_plane_distance = 0`` (the DEFAULT) exactly
    as much as through focus, because the error is the sag, not the
    propagation.

    Returns dict of (n_launch, n_launch) grids: entrance coords, output-plane
    transverse positions, OPL [m], exit direction cosines, alive mask.
    """
    surfaces = rt.surfaces_from_prescription(prescription)
    xs_in = np.linspace(-launch_radius, launch_radius, n_launch)
    Xi, Yi = np.meshgrid(xs_in, xs_in, indexing='ij')
    n_rays = Xi.size
    N0 = np.sqrt(max(0.0, 1.0 - L0 * L0 - M0 * M0))
    rays = rt.RayBundle(
        x=Xi.ravel().copy(), y=Yi.ravel().copy(), z=np.zeros(n_rays),
        L=np.full(n_rays, float(L0)), M=np.full(n_rays, float(M0)),
        N=np.full(n_rays, N0),
        wavelength=wavelength,
        alive=np.ones(n_rays, dtype=bool),
        opd=np.zeros(n_rays),
    )
    tr = rt.trace(rays, surfaces, wavelength)
    # Signed sag -> exit-vertex transfer (t = -z/N, opd += n_exit*t,
    # (x, y) += (L, M)*t, z = 0), with ``n_exit`` resolved from the
    # prescription's own last medium.  Grazing rays (|N| <= 1e-30, which never
    # reach the plane) are killed rather than teleported.
    ex = tr.at_exit_vertex()
    alive = ex.alive.copy()
    # Per-surface state (x, y, z, outgoing slopes) on the launch grid -- for the
    # in-glass KMAH count (D3): each surface-to-surface leg is a homogeneous
    # straight segment, so det Q(z) is the same exact quadratic the exit leg
    # uses, and its roots are the in-glass focal-line crossings.
    shape = (n_launch, n_launch)
    history = []
    for hb in (tr.ray_history or []):
        hNz = np.where(np.abs(hb.N) > 1e-30, hb.N, 1e-30)
        history.append({
            'x': np.asarray(hb.x, float).reshape(shape),
            'y': np.asarray(hb.y, float).reshape(shape),
            'z': np.asarray(hb.z, float).reshape(shape),
            'sx': (np.asarray(hb.L, float) / hNz).reshape(shape),
            'sy': (np.asarray(hb.M, float) / hNz).reshape(shape),
        })
    # advance the exit rays -- now ON the vertex plane -- to the output plane
    # a distance d past it (free space, index output_plane_n): path length
    # t = d / N_z.  ``ex.z == 0`` for every alive ray after the transfer
    # above, so this leg really is ``(d - z)/N``.
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = output_plane_distance / Nz
    x_out = ex.x + t * ex.L
    y_out = ex.y + t * ex.M
    opl = (ex.opd + float(output_plane_n) * t
           + (L0 * Xi + M0 * Yi).ravel())
    bad = ~alive
    for arr in (x_out, y_out, opl):
        arr[bad] = np.nan
    return {
        'xs_in': xs_in,
        'Xi': Xi, 'Yi': Yi,
        'x_out': x_out.reshape(shape), 'y_out': y_out.reshape(shape),
        'opl': opl.reshape(shape),
        'L': ex.L.reshape(shape), 'M': ex.M.reshape(shape),
        'N': ex.N.reshape(shape),
        'x_exit': ex.x.reshape(shape), 'y_exit': ex.y.reshape(shape),
        'alive': alive.reshape(shape),
        'history': history,
    }


def _count_fold_roots(detQ0, trK, detQd, zmax):
    """Count the roots of the exact quadratic ``det Q(z) = detQ0 + z trK +
    z^2 detQd`` in ``(0, zmax]`` (with multiplicity) per grid node -- the
    astigmatic focal-line crossings on one homogeneous leg.  Shared by the
    exit-leg and the in-glass-leg KMAH counts."""
    A, B, C = detQd, trK, detQ0
    m = np.zeros(np.shape(detQ0), dtype=np.int64)
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = B * B - 4.0 * A * C
        sq = np.sqrt(np.maximum(disc, 0.0))
        quad = np.abs(A) > 1e-300
        for sgn in (-1.0, 1.0):
            z = np.where(quad, (-B + sgn * sq) / (2.0 * A), np.nan)
            m += ((disc >= 0.0) & quad & (z > 0.0)
                  & (z <= zmax)).astype(int)
        # linear degenerate case (det Qdot ~ 0): the single root -C/B
        lin = (~quad) & (np.abs(B) > 1e-300)
        zlin = np.where(lin, -C / np.where(lin, B, 1.0), np.nan)
        m += (lin & (zlin > 0.0) & (zlin <= zmax)).astype(int)
    return m


def _kmah_in_glass(g):
    """In-glass (surface-to-surface) fold-caustic count per launch node (D3).

    The exit-leg count (:func:`_kmah_free_leg`) plus a mod-2 parity closure
    left an EVEN number of internal caustic crossings invisible (a full focus
    between two elements = 2 crossings => a pi Maslov-index error).  Each
    surface-to-surface leg is a homogeneous straight segment, so its transverse
    Jacobian is the same exact quadratic ``det Q(z)`` the exit leg uses; sum its
    roots over every internal leg.  Returns the per-node in-glass count (0 when
    no ray focuses inside the prescription -- so an air-focus system is
    byte-identical to the prior release)."""
    hist = g.get('history') or []
    if len(hist) < 2:
        return np.zeros_like(g['x_exit'], dtype=np.int64)
    h = g['xs_in'][1] - g['xs_in'][0]

    def _grad(F):
        return np.gradient(F, h, h)          # d/d(x_in), d/d(y_in)

    m_ig = np.zeros(g['x_exit'].shape, dtype=np.int64)
    for k in range(len(hist) - 1):
        s0, s1 = hist[k], hist[k + 1]
        a11, a12 = _grad(s0['x'])
        a21, a22 = _grad(s0['y'])
        b11, b12 = _grad(s0['sx'])
        b21, b22 = _grad(s0['sy'])
        detQ0 = a11 * a22 - a12 * a21
        detQd = b11 * b22 - b12 * b21
        trK = a22 * b11 - a12 * b21 - a21 * b12 + a11 * b22
        leg = np.sqrt((s1['x'] - s0['x']) ** 2 + (s1['y'] - s0['y']) ** 2
                      + (s1['z'] - s0['z']) ** 2)
        with np.errstate(invalid='ignore'):
            m_ig += np.where(np.isfinite(leg),
                             _count_fold_roots(detQ0, trK, detQd, leg), 0)
    return m_ig


def _kmah_free_leg(g, d_out):
    """Analytic KMAH count on the homogeneous exit leg (Klimes/Cerveny).

    The transverse Jacobian is linear in the leg distance ``z``:
    ``Q(z) = Q0 + z Qdot`` with ``Q0 = d(x_exit)/d(gamma)`` and
    ``Qdot = d(slope)/d(gamma)`` (slope = (L/N, M/N)); hence
    ``det Q(z) = det Q0 + z tr(adj(Q0) Qdot) + z^2 det Qdot`` -- an exact
    quadratic whose roots in ``(0, d_out]`` are the astigmatic focal-line
    crossings.  Count them (with multiplicity) per launch node.  ``Q0``/
    ``Qdot`` are central finite differences on the launch grid.
    """
    h = g['xs_in'][1] - g['xs_in'][0]

    def _grad(F):
        dF_di, dF_dj = np.gradient(F, h, h)
        return dF_di, dF_dj   # d/d(x_in), d/d(y_in)  (indexing='ij')

    Nz = np.where(np.abs(g['N']) > 1e-30, g['N'], 1e-30)
    sx = g['L'] / Nz
    sy = g['M'] / Nz
    # Q0 (exit positions) and Qdot (slopes) w.r.t. launch coords
    a11, a12 = _grad(g['x_exit'])
    a21, a22 = _grad(g['y_exit'])
    b11, b12 = _grad(sx)
    b21, b22 = _grad(sy)
    detQ0 = a11 * a22 - a12 * a21
    detQd = b11 * b22 - b12 * b21
    # tr(adj(Q0) Qdot) with adj([[a,b],[c,d]]) = [[d,-b],[-c,a]]
    trK = a22 * b11 - a12 * b21 - a21 * b12 + a11 * b22
    # exit-leg roots of detQ0 + z trK + z^2 detQd in (0, d_out]
    m = _count_fold_roots(detQ0, trK, detQd, d_out)
    # IN-GLASS legs (D3): the exit-leg count plus a mod-2 parity closure left an
    # EVEN internal caustic count invisible (a focus between two elements = 2
    # crossings => a pi error).  Add the per-leg quadratic count over every
    # surface-to-surface leg; 0 for an air-focus system (byte-identical to the
    # prior release), so existing validated charts are unchanged.
    m_ig = _kmah_in_glass(g)
    m = m + m_ig
    # EXACT parity guarantee (Mitrofanov & Priimenko eq. 1), retained: sgn
    # det J(out) must be (-1)^m relative to the launch (det=+1).  If the
    # combined exit+in-glass count still has the wrong parity, the in-glass
    # magnitude is off by an odd amount -> restore the exact parity by +1
    # (what the old closure did).  Where that correction fires on a node the
    # in-glass detector ALSO flagged (m_ig>0), the two disagree -> the
    # global-frame in-glass approximation is unreliable there; warn once.
    c11, c12 = _grad(g['x_out'])
    c21, c22 = _grad(g['y_out'])
    detJ_out = c11 * c22 - c12 * c21
    parity_bad = (np.sign(detJ_out) != (-1.0) ** m) & np.isfinite(detJ_out)
    m = m + parity_bad.astype(int)
    if bool(np.any(parity_bad & (m_ig > 0))):
        warnings.warn(
            "apply_real_lens_traced_multibranch: the in-glass KMAH count and "
            "the exact parity closure disagree on some launch nodes (an "
            "internal focus inside the prescription that the global-frame "
            "leg quadratic under/over-resolves); the branch Maslov index may "
            "carry a residual pi error there.  Validate against the GBD or "
            "Maslov propagator for internal-focus prescriptions.",
            _caller_stacklevel())
    # fill nodes whose FD Jacobian was NaN-contaminated by a dead neighbour
    # (their own ray may be alive and used by adjacent triangles): nearest
    # valid neighbour, iteratively (KMAH is piecewise constant per sheet).
    #
    # EDGE-CLAMPED, not wrapped.  ``np.roll`` makes the lattice a torus, so on
    # a vignetted rim the "nearest valid neighbour" of a left-edge node is the
    # RIGHT-edge node -- 2*launch_radius away, on the other side of the pupil
    # and quite possibly on a different sheet, i.e. a different Maslov index.
    # ``_shift_clamped`` repeats the boundary row/column instead, so a fill
    # never crosses the pupil.
    bad = ~(np.isfinite(detQ0) & np.isfinite(detJ_out))
    it = 0
    while bad.any() and it < max(m.shape):
        it += 1
        progressed = False
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            src_m = _shift_clamped(m, sh, ax)
            src_ok = _shift_clamped(~bad, sh, ax)
            take = bad & src_ok
            if take.any():
                m[take] = src_m[take]
                bad[take] = False
                progressed = True
        if not progressed:
            # Every remaining bad node is isolated from any valid one (a fully
            # dead lattice, or a dead block with no live boundary) -- the loop
            # cannot make progress, so stop rather than spin to the cap.
            break
    return m, detJ_out


def _shift_clamped(a, shift, axis):
    """``np.roll(a, shift, axis)`` with the boundary REPEATED, not wrapped.

    Used by the KMAH NaN-fill, where a wrapped neighbour is a node on the
    opposite side of the pupil (up to ``2*launch_radius`` away, potentially on
    a different sheet) rather than an adjacent one.
    """
    if shift == 0:
        return a
    out = np.roll(a, shift, axis=axis)
    sl = [slice(None)] * a.ndim
    edge = [slice(None)] * a.ndim
    if shift > 0:
        sl[axis] = slice(0, shift)
        edge[axis] = slice(shift, shift + 1)
    else:
        sl[axis] = slice(shift, None)
        edge[axis] = slice(shift - 1, shift)
    out[tuple(sl)] = out[tuple(edge)]
    return out


def apply_real_lens_traced_multibranch(
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
    return_diagnostics: bool = False,
    geometry: Optional['LensGeometry'] = None,
    numerics: Optional['LensNumerics'] = None,
    resources: Optional['LensResources'] = None,
    config: Optional['LensConfig'] = None,
) -> Any:
    """Multi-branch ray-traced lens field, valid THROUGH focus / caustics.

    Constructs the coherent multi-arrival geometric field at the output plane
    ``output_plane_distance`` past the prescription's exit vertex::

        E(x) = sum_branches  a_br(x) sqrt(1/|J_br|)
               * exp(i k OPL_br(x) - i (pi/2) m_br)

    by the triangulated-ray-map / wavefront-construction method (see module
    docstring for the full literature provenance).  Where the single-valued
    :func:`apply_real_lens_traced` breaks down (several rays per pixel through
    focus), every arrival is summed with its ``1/sqrt|J|`` amplitude and its
    KMAH (Maslov / Gouy) phase ``exp(-i pi m/2)`` -- fold caustics counted
    ANALYTICALLY on the exit leg via the exact quadratic ``det Q(z)``.

    Scope: slowly-varying input ENVELOPE with an optional linear carrier /
    tilt (``input_carrier``; the launch congruence follows the input phase
    plane and the carrier phase is carried exactly by the branch eikonals).
    ``E_in`` supplies per-ray complex envelope amplitude.  The ART amplitude
    is undefined ON the caustic itself (degenerate triangles are skipped --
    literature-standard); the caustic band is ~1 output pixel wide at
    optical wavelengths.

    ``caustic_band='ludwig'`` (default) applies the literature-standard local
    uniform replacement in the Kravtsov-Orlov band: at pixels where a
    coalescing branch pair has ``k |S+ - S-| <= pi``, the pair's plain sum
    (which diverges toward the fold) is swapped for the Ludwig uniform-Airy
    field :func:`ludwig_fold` -- finite exactly where the branch amplitudes
    diverge, and reducing to the plain sum outside the band by construction
    (Kravtsov 1964; Ludwig 1966; Grillo & Cordes 2018 eq. 47).  All other
    branches at the pixel keep their plain contributions.
    ``caustic_band='plain'`` keeps the raw branch sum everywhere.

    Scope note (D5): the band swaps the CLOSEST-EIKONAL branch pair, which is a
    FOLD-uniform (Airy) replacement.  At an AXIAL point focus a whole RING of
    branches coalesces with ``|S+ - S-| ~ 0`` (the correct uniform function
    there is Bessel/Pearcey, not Airy); the Ludwig swap regularizes only the
    closest PAIR, so the remaining ring branches keep their divergent
    ``1/sqrt|J|`` ART amplitudes and the reconstructed field DOES blow up at a
    perfect on-axis focus -- geometric optics diverges there (probe: ~1e6x the
    input power at the BFL plane).  A two-sided energy tripwire fires when the
    reconstructed grid power departs grossly -- either way -- from the power
    the launch congruence carries onto the grid, and a field that comes back
    identically zero is refused outright; route those planes to the wave
    hand-off (``apply_real_lens_traced(caustic='wave')``) or to the Maslov
    (``'levin'``) / GBD propagator.  A ``det Q-dot``-based fold-vs-point
    discriminator is a possible future refinement.

    Parameters
    ----------
    E_in : (N, N) complex -- input field at the lens entrance plane.
    output_plane_distance : float -- output plane distance past the exit
        vertex [m] (e.g. near the focal distance for through-focus study).
    ray_subsample : int -- launch-grid spacing in units of ``dx`` (2 -> one
        ray per 2 pixels).  Smaller = denser branches = more accurate.
    min_area_ratio : float -- skip triangles whose mapped area is below this
        fraction of their launch area (the degenerate caustic set).
    input_carrier : None | 'auto' | (kx, ky) -- transverse carrier
        wavevector of the input field [rad/m].  ``None`` (default) assumes a
        collimated on-axis congruence (v1 behaviour).  ``'auto'`` estimates
        the mean carrier from ``E_in`` by the lag-1 correlation phase (exact
        for carrier x smooth envelope, subpixel; the carrier must be
        resolvable on the pixel grid).  A ``(kx, ky)`` tuple uses the given
        carrier directly (works even for super-Nyquist carriers, since the
        envelope is sampled carrier-stripped).  The launch congruence is
        tilted along the input phase plane (direction cosines ``kx/k0``,
        ``ky/k0``), the input eikonal is carried exactly by the branch
        phases, and the envelope is bilinearly sampled with the carrier
        removed (no aliasing).  Small-tilt scope: the ART amplitude keeps
        the v1 transverse-area ratio (obliquity ``cos theta`` factors are
        neglected -- <0.1% below ~2.5 deg).
    return_diagnostics : bool -- also return a dict with the per-node KMAH
        map, det J, branch-count image, and the resolved ``input_carrier``.
    geometry, numerics, resources, config : optional
        :class:`~lumenairy.LensGeometry` / :class:`~lumenairy.LensNumerics` /
        :class:`~lumenairy.LensResources`, or the
        :class:`~lumenairy.LensConfig` that holds all three, as an alternative
        to spelling the settings out as keywords.  Purely ADDITIVE; a call
        that passes none of the four runs exactly the code it ran before.
        Two of this signature's keywords are spelled differently here from the
        config field that carries them, because ``apply_real_lens_traced``
        owns the family spelling: ``ray_subsample`` here is
        ``LensNumerics.caustic_ray_subsample`` (both default 2 -- it is the
        CAUSTIC launch spacing, not the traced OPL spacing, which also
        defaults to 8 there), and ``min_area_ratio`` is
        ``LensNumerics.caustic_min_area_ratio``.  ``input_carrier`` is NOT
        ``LensGeometry.carrier`` -- it is a transverse WAVEVECTOR in rad/m,
        not a reference congruence -- and stays keyword-only.  See
        ``docs/lens_configuration.md``.

    Returns
    -------
    (N, N) complex output field (plus diagnostics dict if requested).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced_multibranch',
                           input_kind='field')
    # Config objects, if any, are merged into the keywords and the call is
    # re-entered with them -- so the configured path is the SAME code as the
    # equivalent keyword call, by construction rather than by review.  Four
    # ``is not None`` tests when nothing is configured; nothing else changes.
    if _wants_config(geometry, numerics, resources, config):
        return apply_real_lens_traced_multibranch(
            E_in, **_resolve_lens_config(
                apply_real_lens_traced_multibranch, locals(),
                geometry=geometry, numerics=numerics, resources=resources,
                config=config))
    return _multibranch_render(
        E_in, prescription=prescription, wavelength=wavelength, dx=dx,
        output_plane_distance=output_plane_distance,
        output_plane_n=output_plane_n, ray_subsample=ray_subsample,
        min_area_ratio=min_area_ratio, caustic_band=caustic_band,
        input_carrier=input_carrier, return_diagnostics=return_diagnostics)


def _multibranch_render(
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
    return_diagnostics: bool = False,
    pixel_halving_arbiter: bool = False,
) -> Any:
    """The branch sum itself -- everything past the public entry point's input
    check and its configuration-object re-dispatch.

    Split out so ONE internal consumer can ask for a reading the public
    signature does not carry: ``pixel_halving_arbiter`` re-rasterises the same
    mapped triangles onto a HALF-PITCH grid over the same window and reports
    the continuity of the deposited power between the two renders
    (``_PIXEL_CONTINUITY_MAX``).  Keeping it off the public signature keeps
    ``apply_real_lens_traced_multibranch`` byte- and cost-identical for every
    caller that does not ask, and keeps the extra render at ONE rasterisation
    rather than a second trace -- the trace, the triangulation and the mapped
    areas are the same objects, and only the output sampling changes.

    Every message raised below names the PUBLIC entry point, because that is
    the function the caller called.
    """
    E_in = np.asarray(E_in)
    N = E_in.shape[0]
    if E_in.shape[0] != E_in.shape[1]:
        raise ValueError("apply_real_lens_traced_multibranch: square grid "
                         f"required; got {E_in.shape}.")
    k0 = 2.0 * np.pi / wavelength

    # ---- input carrier / tilt (v2): launch along the input phase plane ---
    if input_carrier is None or (isinstance(input_carrier, str)
                                 and input_carrier == 'none'):
        kcx = kcy = 0.0
    elif isinstance(input_carrier, str) and input_carrier == 'auto':
        # lag-1 correlation phase: exact for carrier x smooth envelope
        # (E_in is indexed [y, x]: axis 1 = x, axis 0 = y)
        kcx = float(np.angle(np.nansum(
            E_in[:, 1:] * np.conj(E_in[:, :-1])))) / dx
        kcy = float(np.angle(np.nansum(
            E_in[1:, :] * np.conj(E_in[:-1, :])))) / dx
    else:
        kcx, kcy = (float(input_carrier[0]), float(input_carrier[1]))
    L0 = kcx / k0
    M0 = kcy / k0
    if L0 * L0 + M0 * M0 >= 1.0:
        raise ValueError(
            "apply_real_lens_traced_multibranch: input_carrier "
            f"(kx={kcx:.4g}, ky={kcy:.4g} rad/m) is evanescent at this "
            "wavelength.")
    if kcx != 0.0 or kcy != 0.0:
        # strip the carrier before envelope sampling (re-added exactly via
        # the launch eikonal T_in = L0 x + M0 y in the branch phases)
        xs_px = (np.arange(N) - N / 2.0) * dx
        XP, YP = np.meshgrid(xs_px, xs_px)          # [y, x]: XP cols = x
        E_work = E_in * np.exp(-1j * (kcx * XP + kcy * YP))
    else:
        E_work = E_in

    aperture = prescription.get('aperture_diameter')
    if aperture is not None:
        # stay strictly inside the aperture rim: rays AT the rim can die
        # (vignetting / missed caps) and NaN-poison the finite-difference
        # Jacobians of their neighbours -- exactly the marginal rays whose
        # caustic crossings matter most.
        launch_radius = 0.5 * float(aperture) * 0.98
    else:
        launch_radius = 0.5 * N * dx
    sub = max(1, int(ray_subsample))
    n_launch = max(9, int(2 * launch_radius / (dx * sub)))
    if n_launch % 2 == 0:
        n_launch += 1

    g = _trace_launch_grid(prescription, wavelength, launch_radius, n_launch,
                           output_plane_distance, output_plane_n,
                           L0=L0, M0=M0)
    m_grid, detJ = _kmah_free_leg(g, output_plane_distance)

    # sample the input field at the launch nodes (bilinear)
    xs_in = g['xs_in']
    fi = np.clip((xs_in / dx) + N / 2.0, 0.0, N - 1.0 - 1e-9)
    i0 = np.floor(fi).astype(int)
    w = fi - i0
    # E_work[y, x] (carrier-stripped envelope) with our launch grids indexed
    # (i=x_in, j=y_in)
    Exx = ((1 - w)[None, :] * ((1 - w)[:, None] * E_work[np.ix_(i0, i0)]
                               + w[:, None] * E_work[np.ix_(i0 + 1, i0)])
           + w[None, :] * ((1 - w)[:, None] * E_work[np.ix_(i0, i0 + 1)]
                           + w[:, None] * E_work[np.ix_(i0 + 1, i0 + 1)]))
    # Exx[j_y, i_x] -> transpose to [i_x, j_y] to match Xi/Yi (indexing='ij')
    E_launch = Exx.T

    # per-node data for the triangulation
    XO = g['x_out']
    YO = g['y_out']
    OPL = g['opl']
    L = g['L']
    M = g['M']
    ok = g['alive'] & np.isfinite(XO) & np.isfinite(OPL)

    h = xs_in[1] - xs_in[0]
    tri_launch_area = 0.5 * h * h

    if caustic_band not in ('ludwig', 'plain'):
        raise ValueError(
            "apply_real_lens_traced_multibranch: caustic_band must be "
            f"'ludwig' or 'plain', got {caustic_band!r}")

    # triangulate each launch cell into 2 triangles (fixed diagonal), map,
    # rasterize output pixels via barycentric coords (Lambare Fig. 5).
    # VECTORIZED: all per-triangle setup on flat arrays, then rasterization
    # batched in power-of-2 bounding-box buckets -- the same math and the
    # same contribution set as a per-triangle loop, only the float summation
    # order inside np.add.at differs.
    ok4 = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    ci, cj = np.nonzero(ok4)
    # two triangle families per cell:
    # A = (i, j), (i+1, j), (i, j+1);  B = (i+1, j), (i+1, j+1), (i, j+1)
    V0i = np.concatenate([ci, ci + 1])
    V0j = np.concatenate([cj, cj])
    V1i = np.concatenate([ci + 1, ci + 1])
    V1j = np.concatenate([cj, cj + 1])
    V2i = np.concatenate([ci, ci])
    V2j = np.concatenate([cj + 1, cj + 1])
    x0 = XO[V0i, V0j]
    y0 = YO[V0i, V0j]
    x1 = XO[V1i, V1j]
    y1 = YO[V1i, V1j]
    x2 = XO[V2i, V2j]
    y2 = YO[V2i, V2j]
    # signed area of the mapped triangle (fold parity + Jacobian)
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    with np.errstate(invalid='ignore'):
        ratio = np.abs(0.5 * area2) / tri_launch_area
        finite_tri = np.isfinite(area2)
        good = finite_tri & (ratio >= min_area_ratio)
    # Degenerate-triangle census (D5 / audit T2).  The ART amplitude
    # 1/sqrt|J| is undefined ON a caustic, so a triangle whose mapped area has
    # collapsed below ``min_area_ratio`` is skipped -- the literature-standard
    # choice.  At an AXIAL point focus EVERY triangle collapses at once, the
    # rasteriser writes nothing, and the returned field is IDENTICALLY ZERO;
    # before this census that outcome reached the caller with no diagnostic at
    # all (the energy tripwire could only see a GAIN).  Count the skips so the
    # total collapse can be named and refused.
    _n_tri = int(area2.size)
    _n_finite = int(finite_tri.sum())
    _n_degenerate = int(_n_finite - int(good.sum()))

    def _render(N_r, dx_r):
        """Rasterise the mapped triangles onto an ``N_r x N_r`` grid of pitch
        ``dx_r`` over the same physical window, and assemble the coherent
        branch sum on it.

        Everything above -- the trace, the triangulation, the mapped areas and
        the degeneracy census -- is a property of the LAUNCH lattice and the
        optic, so it is computed once and closed over here.  Only the OUTPUT
        sampling is a parameter, which is what makes the pixel-halving arbiter
        below cost one rasterisation instead of a second trace.

        Called once with ``(N, dx)`` -- byte-identically to the straight-line
        code it replaces -- and, when the arbiter is requested, once more with
        ``(2 * N, dx / 2)``.
        """
        # per-branch contribution lists (assembled after the loop so the caustic
        # band can swap coalescing PAIRS for the Ludwig uniform-fold field)
        br_idx: list = []      # flat pixel index (x-major)
        br_T: list = []        # branch eikonal OPL [m]
        br_A: list = []        # COMPLEX branch amplitude incl. its Maslov phase
        n_branch = np.zeros((N_r, N_r), dtype=np.int32)
        #: Launch power carried by the triangles that RASTERISE onto this grid --
        #: the second, independent denominator the energy tripwire brackets with
        #: (VERIFY-A3).  Filled in the rasteriser block below.
        _p_in_tri = 0.0
        if good.any():
            xmn = np.minimum(np.minimum(x0, x1), x2)[good]
            xmx = np.maximum(np.maximum(x0, x1), x2)[good]
            ymn = np.minimum(np.minimum(y0, y1), y2)[good]
            ymx = np.maximum(np.maximum(y0, y1), y2)[good]
            pxmin = np.maximum(0, np.floor(xmn / dx_r + N_r / 2.0).astype(np.int64))
            pxmax = np.minimum(N_r - 1,
                               np.ceil(xmx / dx_r + N_r / 2.0).astype(np.int64))
            pymin = np.maximum(0, np.floor(ymn / dx_r + N_r / 2.0).astype(np.int64))
            pymax = np.minimum(N_r - 1,
                               np.ceil(ymx / dx_r + N_r / 2.0).astype(np.int64))
            keep = (pxmax >= pxmin) & (pymax >= pymin)

            def _g(a):
                return a[good][keep]

            x0k, y0k, x1k, y1k, x2k, y2k = (
                _g(x0), _g(y0), _g(x1), _g(y1), _g(x2), _g(y2))
            den = _g(area2)
            amp_J = 1.0 / np.sqrt(_g(ratio))
            # KMAH: vertices of one branch share m; take the majority (median)
            mtri = np.stack([m_grid[V0i, V0j], m_grid[V1i, V1j],
                             m_grid[V2i, V2j]], axis=1).astype(np.int64)
            m_br = np.sort(mtri, axis=1)[:, 1][good][keep]
            ph_br = np.exp(-0.5j * np.pi * m_br)
            T0 = _g(OPL[V0i, V0j])
            T1 = _g(OPL[V1i, V1j])
            T2 = _g(OPL[V2i, V2j])
            # eikonal-gradient transverse components p = n*(L, M) for the
            # intrapolation below.  In an index-n output space the phase slowness
            # is n*(L, M), NOT the bare direction cosines (D4); pn = 1 for a
            # vacuum output plane, so this is byte-identical there.  (The OPL
            # advance already carries the n factor -- output_plane_n * t above.)
            # Named ``p0x``/``p0y`` etc., NOT ``L0``/``M0``: those two names are
            # the INPUT congruence's launch direction cosines, live from the top of
            # this function, and rebinding them here to per-vertex slowness
            # components silently shadowed them for the rest of the body.
            pn = float(output_plane_n)
            p0x = pn * _g(L[V0i, V0j])
            p1x = pn * _g(L[V1i, V1j])
            p2x = pn * _g(L[V2i, V2j])
            p0y = pn * _g(M[V0i, V0j])
            p1y = pn * _g(M[V1i, V1j])
            p2y = pn * _g(M[V2i, V2j])
            E0 = _g(E_launch[V0i, V0j])
            E1 = _g(E_launch[V1i, V1j])
            E2 = _g(E_launch[V2i, V2j])
            # Launch power of the triangles that reach this grid.  Over the
            # INTERIOR of the launch lattice this is algebraically the node sum
            # ``sum |E_launch|^2 h^2`` (each node is shared by six triangles and
            # each triangle averages three nodes, so the two quadratures agree);
            # where a triangle STRADDLES the grid boundary it counts the whole
            # triangle where the node sum counts none of it, so it is an upper
            # bound on the launched power that lands -- which is exactly what the
            # gain arm of the tripwire needs to bracket against (see there).
            _p_in_tri = float(np.sum(
                (np.abs(E0) ** 2 + np.abs(E1) ** 2 + np.abs(E2) ** 2)
                * (1.0 / 3.0))) * tri_launch_area
            pxmin, pxmax = pxmin[keep], pxmax[keep]
            pymin, pymax = pymin[keep], pymax[keep]
            nb_flat = n_branch.reshape(-1)

            # mapped output-plane edge vectors (edge_i opposite vertex_i, on which
            # the barycentric a_i == 0) -- for the half-open top-left tie-break
            # below.  e0 = V2 - V1, e1 = V0 - V2, e2 = V1 - V0.
            e0x, e0y = x2k - x1k, y2k - y1k
            e1x, e1y = x0k - x2k, y0k - y2k
            e2x, e2y = x1k - x0k, y1k - y0k

            # ---- batched rasterisation ------------------------------------
            # Triangles are grouped so one NumPy batch covers many of them, and
            # every intermediate below is ``(n_batch, wx, wy)``.  Two properties
            # of that grouping are load-bearing, and neither was true before:
            #
            # * the box is the triangle's EXACT integer bounding box, not the
            #   next power of two.  At ``output_plane_distance = 0`` -- the
            #   DEFAULT -- the map is near-identity, so a 5x5-pixel triangle used
            #   to be padded to 8x8 and ~60 % of the barycentric arithmetic was
            #   thrown away by ``vmask``.  Measured on an f/2 singlet at N = 2048
            #   with default arguments, the exit-vertex call took 144 s against
            #   0.85 s for the SAME grid near focus, where the triangles compress
            #   into small boxes.  With the exact box there is no padding, so no
            #   ``vmask`` either.
            # * the batch is CHUNKED against a named entry budget.  Nothing used
            #   to bound ``n_batch * wx * wy``: at N = 4096 the worst bucket held
            #   1 187 616 triangles at 8x8, i.e. 0.61 GB per temporary and ~15-18
            #   of them alive at once -- 7.74 GB of traced allocation and 12.2 GB
            #   RSS for one default-argument call, with no warning and no model in
            #   ``lumenairy.memory.estimate_lens_memory``.  The budget caps the
            #   working set at ~``_RASTER_CHUNK_ENTRIES`` float64 per temporary
            #   regardless of grid or prescription.
            #
            # Neither changes the CONTRIBUTION SET -- the same pixels enter with
            # the same values -- only the order in which ``np.add.at`` sums a
            # multi-branch pixel, which this module already documents as
            # order-dependent at the ULP level (see the note above).
            wxs = (pxmax - pxmin + 1).astype(np.int64)
            wys = (pymax - pymin + 1).astype(np.int64)
            for s, wbx, wby, _padded in _raster_batches(wxs, wys):
                gx = pxmin[s, None, None] + np.arange(wbx, dtype=np.int64)[
                    None, :, None]
                gy = pymin[s, None, None] + np.arange(wby, dtype=np.int64)[
                    None, None, :]
                # Only the coalesced (power-of-two) fallback can overshoot the
                # triangle's clipped box; with the exact box every candidate is a
                # valid grid index and no mask is needed.
                vmask = (((gx <= pxmax[s, None, None])
                          & (gy <= pymax[s, None, None])) if _padded else None)
                PX = (gx - N_r / 2.0) * dx_r
                PY = (gy - N_r / 2.0) * dx_r
                X0 = x0k[s, None, None]
                Y0 = y0k[s, None, None]
                # barycentric coordinates
                d00x = (x1k - x0k)[s, None, None]
                d00y = (y1k - y0k)[s, None, None]
                d01x = (x2k - x0k)[s, None, None]
                d01y = (y2k - y0k)[s, None, None]
                wx = PX - X0
                wy = PY - Y0
                dn = den[s, None, None]
                a1 = (wx * d01y - wy * d01x) / dn
                a2 = (d00x * wy - d00y * wx) / dn
                a0 = 1.0 - a1 - a2
                # HALF-OPEN (top-left rule): a pixel strictly inside a triangle has
                # all a_i > 0; a pixel ON edge_i has a_i == 0.  A closed test
                # (a_i >= 0) hands every shared mesh edge (cell diagonals) and
                # shared vertex (up to 6 triangles) to ALL its neighbours, so the
                # coherent np.add.at below double- (or 6x-) counts them -- spurious
                # +50%..+400% energy and 2x-amplitude hot pixels.  Instead award an
                # on-edge pixel to the single triangle whose mapped edge points up
                # (or is horizontal pointing right): two triangles sharing a
                # physical edge see it with opposite direction vectors, so exactly
                # one claims it -- regardless of fold orientation.  At a shared
                # vertex two coords are ~0 at once, so BOTH incident edges must
                # win, which resolves the up-to-6 triangle fan to one owner too.
                # The band |a_i| <= _EDGE_TOL (not exact == 0) is essential: even a
                # bit-exact identity map lands pixel centres ~1 ULP off the mapped
                # nodes, so on-edge a_i evaluate to ~1e-14, never exactly 0.  The
                # band is applied symmetrically (barely-inside +eps and
                # barely-outside -eps neighbours both enter the tie-break), so the
                # single-owner property holds with no gaps.  Interior a_i are
                # O(0.1..1) >> _EDGE_TOL >> the ~1e-14 boundary noise, so genuine
                # multi-branch overlaps (distinct sheets from non-adjacent
                # triangles, a_i strictly interior in both) are untouched.
                e0i = (e0x[s, None, None], e0y[s, None, None])
                e1i = (e1x[s, None, None], e1y[s, None, None])
                e2i = (e2x[s, None, None], e2y[s, None, None])
                inside = (
                    ((a0 > _EDGE_TOL) | ((np.abs(a0) <= _EDGE_TOL) & _top_left(*e0i)))
                    & ((a1 > _EDGE_TOL) | ((np.abs(a1) <= _EDGE_TOL) & _top_left(*e1i)))
                    & ((a2 > _EDGE_TOL) | ((np.abs(a2) <= _EDGE_TOL) & _top_left(*e2i))))
                if vmask is not None:
                    inside &= vmask
                if not inside.any():
                    continue
                # second-order intrapolated OPL (Kraaijpoel eq. 5.7):
                # T = sum_i a_i [T_i + 1/2 (x - x_i).p_i], p = n*(L, M) (D4)
                T = (a0 * (T0[s, None, None]
                           + 0.5 * ((PX - X0) * p0x[s, None, None]
                                    + (PY - Y0) * p0y[s, None, None]))
                     + a1 * (T1[s, None, None]
                             + 0.5 * ((PX - x1k[s, None, None])
                                      * p1x[s, None, None]
                                      + (PY - y1k[s, None, None])
                                      * p1y[s, None, None]))
                     + a2 * (T2[s, None, None]
                             + 0.5 * ((PX - x2k[s, None, None])
                                      * p2x[s, None, None]
                                      + (PY - y2k[s, None, None])
                                      * p2y[s, None, None])))
                Ein_tri = (a0 * E0[s, None, None] + a1 * E1[s, None, None]
                           + a2 * E2[s, None, None])
                # branch record: complex amplitude WITH its Maslov phase, and
                # the eikonal separately (needed by the Ludwig pair-swap)
                flat = (gx * N_r + gy)[inside]
                br_idx.append(flat)
                br_T.append(T[inside])
                br_A.append((Ein_tri * amp_J[s, None, None]
                             * ph_br[s, None, None])[inside])
                np.add.at(nb_flat, flat, 1)

        # ---- assemble the coherent branch sum ------------------------------
        E_flat = np.zeros(N_r * N_r, dtype=np.complex128)
        if br_idx:
            bi = np.concatenate(br_idx)
            bT = np.concatenate(br_T)
            bA = np.concatenate(br_A)
            plain = bA * np.exp(1j * k0 * bT)
            np.add.at(E_flat, bi, plain)
            if caustic_band == 'ludwig':
                # In the Kravtsov-Orlov caustic band (k|S+ - S-| <= pi between a
                # coalescing branch pair) the plain two-branch sum is invalid --
                # swap the pair for the Ludwig uniform fold field (finite exactly
                # where the branch amplitudes diverge; reduces to the plain sum
                # outside the band by construction).  Grillo & Cordes eq. 47:
                # uniform-Airy for the coalescing pair, plain GO for the rest.
                #
                # VECTORISED over pixels.  The per-pixel Python loop this replaces
                # cost 0.76-1.07 ms for EVERY multi-branch pixel (measured +3.22 s
                # over 3 004 pixels at N = 2048 and +9.14 s over 12 020 at
                # N = 4096, i.e. a 1.5 Mpx two-branch ring would have taken ~20
                # minutes).  Same selection rule, same arithmetic: sort the
                # branches of each pixel by eikonal, take the closest ADJACENT
                # pair, and swap it when its split is inside the band.
                order = np.argsort(bi, kind='stable')
                bi_s = bi[order]
                starts = np.flatnonzero(np.r_[True, bi_s[1:] != bi_s[:-1]])
                ends = np.r_[starts[1:], bi_s.size]
                band = np.pi / k0
                multi = (ends - starts) >= 2
                if multi.any():
                    g_start = starts[multi]
                    g_end = ends[multi]
                    n_g = g_start.size
                    n_max = int((g_end - g_start).max())
                    # Ragged -> padded (n_groups, n_max) of the group members'
                    # positions in ``order``; pad slots take the group's own first
                    # member so the sort never sees a sentinel that could become
                    # the minimum gap.
                    col = np.arange(n_max)[None, :]
                    cnt = (g_end - g_start)[:, None]
                    valid = col < cnt
                    pos = np.where(valid, g_start[:, None] + col, g_start[:, None])
                    sel = order[pos]                      # (n_g, n_max)
                    Ts = np.where(valid, bT[sel], np.inf)
                    o2 = np.argsort(Ts, axis=1, kind='stable')
                    rows = np.arange(n_g)[:, None]
                    sel_s = sel[rows, o2]
                    Ts_s = Ts[rows, o2]
                    # gaps between ADJACENT sorted branches; a gap that touches a
                    # padded slot is +inf and can never be the minimum.
                    dT = Ts_s[:, 1:] - Ts_s[:, :-1]
                    dT = np.where(np.isfinite(dT), dT, np.inf)
                    jmin = np.argmin(dT, axis=1)
                    dmin = dT[np.arange(n_g), jmin]
                    hit = np.isfinite(dmin) & (dmin <= band)
                    if hit.any():
                        hr = np.nonzero(hit)[0]
                        jh = jmin[hr]
                        ia = sel_s[hr, jh]            # lower-S branch  (S-)
                        ib = sel_s[hr, jh + 1]        # higher-S branch (S+)
                        Sm = bT[ia]
                        # floor the eikonal split (removable 0/0 in the g1 term
                        # exactly at coalescence)
                        Sp = np.maximum(bT[ib], Sm + 1e-4 * wavelength)
                        uni = ludwig_fold(k0, Sp, Sm, bA[ib], bA[ia])
                        # Each group owns one pixel, and a pixel appears in at most
                        # one group, so these writes do not collide -- but use
                        # ``np.add.at`` anyway so the accumulation rule is the same
                        # one the plain sum used.
                        np.add.at(E_flat, bi_s[g_start[hr]],
                                  uni - (plain[ia] + plain[ib]))
        E_out = E_flat.reshape(N_r, N_r)

        # our accumulation is indexed [x, y]; the library field convention is
        # E[y, x] -- transpose on return.
        E_out = E_out.T
        n_branch = n_branch.T

        return E_out, n_branch, _p_in_tri

    E_out, n_branch, _p_in_tri = _render(N, dx)
    # ---- axial point-focus catastrophe tripwire (D5 / audit T2) ---------
    # Independent energy oracle: at a rotationally-symmetric on-axis focus a
    # RING of branches coalesces where the fold-uniform 'ludwig' swap
    # regularizes only the closest PAIR, so the residual branch amplitudes
    # diverge and the reconstructed grid power blows up ~1e5..1e6x (a
    # well-behaved through-focus field conserves the launched power).  Compare
    # in BOTH directions, since a gain-only test cannot see the two failures on
    # the other side (the total collapse below, and the ~11 % loss a resolved
    # fold shows from the skipped degenerate triangles plus the missing
    # dark-side tail).
    #
    # The reference is the power the launch congruence actually carries ONTO
    # THE GRID, not the input power inside the aperture circle.  Two geometry
    # effects make the aperture-circle sum the wrong normaliser, both of them
    # present in ordinary use and neither of them a physics failure:
    #   * the launch sampler CLAMPS E_in at the grid edge, so an aperture wider
    #     than the E_in grid launches the edge amplitude over the whole rim
    #     annulus -- far more power than that sum counts;
    #   * light that legitimately leaves the N x N output grid is not lost
    #     physics, it is off-screen.
    # Summing |E_launch|^2 over alive launch nodes whose mapped exit point
    # lands inside the grid removes both, and puts numerator and denominator in
    # the same physical units (launch cell area h^2, output pixel area dx^2).
    with np.errstate(invalid='ignore'):
        _in_grid = ok & (np.abs(XO) <= 0.5 * N * dx) & (np.abs(YO)
                                                        <= 0.5 * N * dx)
    p_in = float(np.sum(np.abs(E_launch[_in_grid]) ** 2)) * (h * h)
    p_out = float(np.sum(np.abs(E_out) ** 2)) * (dx * dx)
    # BRACKET the gain arm (VERIFY-A3).  ``p_in`` counts a launch node only
    # when its own mapped point lands on the grid, while ``p_out`` counts
    # every pixel a triangle covers -- including triangles that STRADDLE the
    # grid boundary, whose nodes are outside.  On a coarse launch lattice over
    # a grid much smaller than the beam that mismatch alone reaches 3.3x:
    # measured on the D3 fixture (aperture 25x the grid AREA) at
    # ray_subsample=8, z = 100 / 110 / 120 mm -> 3.26 / 2.56 / 2.07 with
    # n_branch = 1 and ZERO degenerate triangles, i.e. no coalescence at all,
    # three spurious RuntimeWarnings from the very geometry the launched-power
    # normaliser was introduced to quieten.  ``_p_in_tri`` counts the WHOLE
    # launch power of every contributing triangle, so it is an upper bound
    # where ``p_in`` is a lower one; requiring the gain to clear BOTH removes
    # the artefact and costs no detection power, because a real point-focus
    # blow-up is 1e5x (measured 1.8e+05 at 0.98 BFL on the same fixture) and
    # the audit's silent pre-focus band is 4-8x -- decades outside either.
    p_in_hi = max(p_in, _p_in_tri)

    # ---- the PIXEL-HALVING ARBITER (WP-B7c round 2) ---------------------
    # Re-rasterise the SAME mapped triangles onto a half-pitch grid over the
    # same physical window and compare the deposited power.  See
    # ``_PIXEL_CONTINUITY_MAX`` for what the reading means and why it is the
    # quantity a refusal can be decided on where ``power_ratio`` is not.
    # Costs one rasterisation, never a second trace.
    _continuity = None
    _p_out_half = None
    _E_half = None
    _continuity_decision = 'not_requested'
    if pixel_halving_arbiter:
        if 4 * N * N > _ARBITER_MAX_FINE_ENTRIES:
            _continuity_decision = 'not_measured'
        elif p_out <= 0.0:
            # an identically-zero coarse render is refused above; a zero
            # numerator here would make the ratio meaningless rather than
            # large, so say so instead of reporting 0.0
            _continuity_decision = 'not_measured'
        else:
            _E_half, _nb_half, _ = _render(2 * N, 0.5 * dx)
            _p_out_half = float(np.sum(np.abs(_E_half) ** 2)) * (
                0.25 * dx * dx)
            if _p_out_half <= 0.0:
                _continuity_decision = 'not_measured'
            else:
                _continuity = p_out / _p_out_half
                if _continuity > _PIXEL_CONTINUITY_MAX:
                    _continuity_decision = 'not_converged_gain'
                elif _continuity < _PIXEL_CONTINUITY_MIN:
                    _continuity_decision = 'not_converged_loss'
                else:
                    _continuity_decision = 'ok'
            del _nb_half

    # TOTAL COLLAPSE.  This is the only outcome of the four that is not merely
    # inaccurate but EMPTY, and before this census it reached the caller as an
    # identically-zero field with no diagnostic at all (measured P/P_in = 0.0,
    # 0 non-zero pixels, 0 warnings at the 24.83 mm BFL of an f = 25 mm
    # singlet, through the public entry point).  Two mechanisms produce it and
    # the test covers both: triangles skipped as degenerate, and triangles so
    # compressed by the map that none contains a pixel CENTRE.  Refuse --
    # returning zeros invites the caller to read "no light" as a physical
    # result at the very plane the method cannot represent.
    if p_in > 0.0 and p_out <= 0.0:
        raise RuntimeError(
            "apply_real_lens_traced_multibranch: the reconstructed field is "
            f"identically ZERO at this output plane, although "
            f"{_n_finite} launch triangles mapped and the launch congruence "
            f"carries power onto this grid.  {_n_degenerate} of those "
            f"triangles "
            f"({100.0 * _n_degenerate / max(_n_finite, 1):.1f}%) were skipped "
            f"as degenerate (mapped/launch area ratio below "
            f"min_area_ratio={min_area_ratio:g}) and the rest are compressed "
            f"below one pixel, so no pixel centre lies inside any mapped "
            f"triangle.  This is the axial point-focus catastrophe (Scope note "
            f"D5): a whole RING of branches coalesces, which is a higher "
            f"catastrophe than the FOLD the 'ludwig' pair swap can "
            f"regularize, and geometric optics genuinely diverges there.  Use "
            f"the wave hand-off -- apply_real_lens_traced(caustic='wave', "
            f"output_plane_distance=...), which propagates the traced "
            f"exit-vertex field with the band-limited angular spectrum and is "
            f"exact through folds, cusps and the axial focus alike -- or "
            f"apply_real_lens_gbd / apply_real_lens_maslov.  Lowering "
            f"min_area_ratio does NOT fix it: it only lets the divergent "
            f"1/sqrt|J| amplitudes through (measured P/P_in = 3.9e+09 at "
            f"min_area_ratio=1e-12 on the same plane).")
    if _n_finite > 0 and _n_degenerate > 0.25 * _n_finite:
        warnings.warn(
            "apply_real_lens_traced_multibranch: "
            f"{_n_degenerate}/{_n_finite} mapped launch triangles "
            f"({100.0 * _n_degenerate / _n_finite:.1f}%) are degenerate at "
            f"this output plane (area ratio below "
            f"min_area_ratio={min_area_ratio:g}) and were skipped, so the "
            f"reconstructed field is missing their contribution.  The output "
            f"plane is at or near a caustic that the fold-uniform 'ludwig' "
            f"swap cannot regularize; prefer caustic='wave' (the band-limited "
            f"ASM hand-off) or the GBD / Maslov propagators here.",
            RuntimeWarning, _caller_stacklevel())

    if p_in_hi > 0.0 and p_out > _ENERGY_BLOWUP_FACTOR * p_in_hi:
        warnings.warn(
            "apply_real_lens_traced_multibranch: reconstructed grid power is "
            f"{p_out / p_in:.3g}x the launched power that reaches this grid "
            f"(up to "
            f"{int(n_branch.max())} branches coalesce on one pixel).  This is "
            "the axial point-focus catastrophe (Scope note D5): a RING of "
            "branches coalesces where the fold-uniform 'ludwig' swap "
            "regularizes only the closest PAIR, so the residual ART amplitudes "
            "diverge.  The near-focus field is unphysical here -- use "
            "caustic='wave' (the band-limited ASM hand-off), or the Maslov "
            "('levin') or GBD propagator, at an on-axis point focus.",
            RuntimeWarning, _caller_stacklevel())
    elif p_in > 0.0 and 0.0 < p_out < _ENERGY_COLLAPSE_FACTOR * p_in:
        warnings.warn(
            "apply_real_lens_traced_multibranch: reconstructed grid power is "
            f"only {p_out / p_in:.3g}x the launched power that reaches this "
            f"grid ({_n_degenerate}/{max(_n_finite, 1)} mapped triangles were "
            f"skipped as degenerate).  A branch-enumeration field LOSES energy "
            f"where the ray map is under-covered -- the skipped degenerate "
            f"triangles and, on the dark side of a fold, the exponential tail "
            f"no real ray reaches (which this method drops to exactly zero).  "
            f"Use caustic='uniform' for the fold's Airy tail, or caustic="
            f"'wave' (the band-limited ASM hand-off) / GBD / Maslov for an "
            f"energy-conserving answer.",
            RuntimeWarning, _caller_stacklevel())

    if return_diagnostics:
        return E_out, {'kmah': m_grid, 'detJ': detJ, 'n_branch': n_branch,
                       'input_carrier': (kcx, kcy),
                       'n_triangles': _n_tri, 'n_triangles_finite': _n_finite,
                       'n_triangles_degenerate': _n_degenerate,
                       # the largest number of mapped triangles that land on
                       # ONE pixel -- the quadrature statistic the energy
                       # tripwire's mechanism note is written on (it rises
                       # 2 -> 797 across the blow-up window), and the number a
                       # caller needs to tell a two-branch fold from a
                       # collapsing ring without re-deriving ``n_branch``
                       'n_branch_max': int(n_branch.max()),
                       # the two denominators of ``power_ratio`` /
                       # ``power_ratio_triangles``, in absolute units [W], so a
                       # consumer can normalise its OWN field against the same
                       # launched power instead of re-deriving it
                       'launched_power': p_in,
                       'launched_power_triangles': _p_in_tri,
                       # reconstructed grid power / launched power reaching
                       # the grid: 1.0 = energy conserved by the branch sum
                       'power_ratio': (p_out / p_in) if p_in > 0.0
                       else None,
                       # the bracketing upper-bound denominator the
                       # gain arm is decided on (VERIFY-A3): equal to
                       # ``power_ratio`` except where triangles
                       # straddle the grid boundary
                       'power_ratio_triangles': (
                           (p_out / _p_in_tri) if _p_in_tri > 0.0
                           else None),
                       # THE PIXEL-HALVING ARBITER (WP-B7c round 2).  The
                       # deposited power of this render over that of the SAME
                       # triangles rasterised at half the pitch on the same
                       # window: ~1 when the point-sampled quadrature has
                       # converged in the pixel, ~4 per halving when it has
                       # not.  ``None`` unless the reading was asked for (the
                       # decision then says which).
                       'pixel_continuity': _continuity,
                       'pixel_continuity_band': (_PIXEL_CONTINUITY_MIN,
                                                 _PIXEL_CONTINUITY_MAX),
                       'pixel_continuity_decision': _continuity_decision,
                       'pixel_halved_power': _p_out_half,
                       # the half-pitch RENDER itself, so a consumer that
                       # post-processes this field (the uniform completion
                       # does: it keeps the bright side verbatim and rewrites
                       # the fold band and the dark tail) can put ITS OWN
                       # output through the same control instead of deciding
                       # on a reading of an intermediate it does not return.
                       # ``None`` unless the arbiter was asked for.
                       'pixel_halved_field': _E_half,
                       'grid_power': p_out}
    return E_out
