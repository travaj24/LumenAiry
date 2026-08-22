"""
lumenairy.elements._lens_traced -- per-pixel ray-traced
``apply_real_lens_traced`` plus its support helpers.

Hybrid wave + ray-trace propagator: amplitude comes from a single
analytic ``apply_real_lens`` call (split-step ASM through glass),
phase comes from a per-pixel geometric ray-trace OPL evaluated on a
coarse entrance grid and Newton-inverted onto the wave grid via a
Chebyshev tensor-product fit.

Extracted from ``lenses.py`` in v3.5.5 to reduce that module's bloat.
``apply_real_lens_traced`` and ``close_worker_pool`` are re-exported
from :mod:`lumenairy.elements.lenses` for backwards-compatible
imports.

Author: Andrew Traverso
"""

from __future__ import annotations

import importlib.util as _importlib_util
import threading
import time as _time
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

# The inverse-characteristic per-pixel evaluator.  Imported as a MODULE, not
# by name: its flags are read at CALL time (house rule -- see
# ``_traced_flags``), so a monkeypatch of ``_lens_imap.TRACED_INVERSE_MAP``
# reaches this file's read sites.  No import cycle: ``_lens_imap`` reaches back
# into this module only from inside function bodies.
from . import _lens_imap as _IMAP

# Optional CuPy backend (lazy).
CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
cp = None


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


def _is_cupy_array(x):
    if not CUPY_AVAILABLE:
        return False
    if cp is None and not _ensure_cupy_loaded():
        return False
    return isinstance(x, cp.ndarray)


# v5.30 (audit E-L4): the numexpr scaffold that used to sit here
# (``NUMEXPR_AVAILABLE`` / ``_ne`` / ``_ensure_numexpr_loaded`` /
# ``_NUMEXPR_MIN_SIZE``) was DEAD -- 0 readers in this module and no importer
# anywhere (the public ``NUMEXPR_AVAILABLE`` re-export comes from ``lenses.py``,
# and the live numexpr phase-screen gate lives in ``_lens_real.py``).  Deleted:
# it advertised a fused-expression fast path this module never had.

# Optional Numba JIT, LAZILY imported on first kernel use (audit P2-D: the eager
# ``import numba`` cost ~1.8 s of ``import lumenairy`` cold start).  The kernel
# (``_cheb2d_val_grad_numba``) has a pure-NumPy fallback, so numba is pulled in
# only when a caller actually hits the fast path AND numba is installed.
import importlib.util as _ilu

_NUMBA_AVAILABLE = _ilu.find_spec("numba") is not None
_numba = None                         # populated by _load_numba() on first use
_njit = None
_prange = None
_NUMBA_KERNELS: dict = {}             # kernel-name -> compiled fn (or None)


def _load_numba():
    """Import numba + njit/prange on first use; cache the handles.  Returns True
    iff numba is importable (False -> caller takes the pure-NumPy fallback)."""
    global _numba, _njit, _prange
    if _numba is not None:
        return True
    if not _NUMBA_AVAILABLE:
        return False
    import numba as _nb
    from numba import njit as _nj
    from numba import prange as _pr
    _numba, _njit, _prange = _nb, _nj, _pr
    return True


# v5.29.1 (audit E-L22): signature defaults of :func:`apply_real_lens_traced`,
# resolved lazily on first use (the function is defined further down) and
# cached.  Single source of truth for the "did the caller actually change this
# knob?" tests, so the discarded-kwarg diagnostic on the
# ``on_noncollimated='delegate'`` model swap cannot drift from the signature.
_TRACED_KWARG_DEFAULTS_CACHE: Dict[str, Any] = {}
_TRACED_KWARG_DEFAULTS_LOCK = threading.Lock()


def _traced_kwarg_defaults() -> Dict[str, Any]:
    """Return ``{kwarg: signature default}`` for ``apply_real_lens_traced``."""
    if not _TRACED_KWARG_DEFAULTS_CACHE:
        import inspect
        with _TRACED_KWARG_DEFAULTS_LOCK:
            if not _TRACED_KWARG_DEFAULTS_CACHE:
                fill = {
                    _n: _p.default
                    for _n, _p in inspect.signature(
                        apply_real_lens_traced).parameters.items()
                    if _p.default is not inspect.Parameter.empty
                }
                _TRACED_KWARG_DEFAULTS_CACHE.update(fill)
    return _TRACED_KWARG_DEFAULTS_CACHE


def _copy_prescription(prescription: Dict[str, Any]) -> Dict[str, Any]:
    """Return a private copy of ``prescription`` for a prepared object to hold.

    v5.29.1 (audit E-H4): a prepared lens must not alias the caller's dict, or
    an in-place edit silently pairs its cached OPL screen with a DIFFERENT lens
    in the per-call amplitude leg.  ``copy.deepcopy`` is the wanted semantics;
    if some exotic member refuses to deep-copy (a ``sag_callable`` bound to an
    object holding a lock / handle, say) fall back to copying exactly the
    containers the mutation vector goes through -- the top-level dict, the
    ``surfaces`` / ``elements`` lists and their member dicts -- and keep the
    leaf objects shared.  Never raises for a copy failure: an aliased leaf is a
    far smaller problem than refusing to prepare.
    """
    import copy as _copy
    try:
        return _copy.deepcopy(prescription)
    except (TypeError, ValueError, RecursionError, AttributeError, OSError):
        out = dict(prescription)
        for _key in ('surfaces', 'elements'):
            _seq = out.get(_key)
            if isinstance(_seq, (list, tuple)):
                out[_key] = [dict(_s) if isinstance(_s, dict) else _s
                             for _s in _seq]
        for _key in ('thicknesses',):
            _seq = out.get(_key)
            if isinstance(_seq, (list, tuple)):
                out[_key] = list(_seq)
        return out


def _kwarg_differs_from_default(value: Any, default: Any) -> bool:
    """True when ``value`` is not the signature ``default``.

    ndarray-safe: an array argument (e.g. an explicit ``carrier`` wavefront)
    is never equal to any scalar/None default, and ``!=`` on it would return
    an array rather than a bool.
    """
    if value is default:
        return False
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return True
    try:
        return bool(value != default)
    except (ValueError, TypeError):
        return True


# Amplitude-masked-Newton default threshold (fraction of ``amp.max()`` below
# which a coarse pixel's Newton solve is skipped).  Named so the
# ``amplitude_model='ray_density'`` guard can tell "the shipped default" from
# an explicitly-requested mask without a not-passed sentinel (audit E-M5).
_NEWTON_AMP_MASK_REL_DEFAULT = 1e-4


# Newton iter cap default.  Set to 12 (the historical value).
# 3.5.5 dropped this to 8 based on an audit recommendation, but the
# active-mask early-exit already short-circuits converged pixels -- the
# cap only matters for outlier pixels that genuinely need 9-12 iters.
# Truncating those at 8 silently lost accuracy on cemented multi-element
# / strongly-aberrated systems.  3.5.6 reverts to the safe 12.  Override
# via apply_real_lens_traced(newton_max_iters=N) when profiling shows
# Newton dominates.
_NEWTON_MAX_ITERS = 12


# N12 (P11): relative floor on |det J| for the opt-in ``amplitude_model=
# 'ray_density'`` mode.  The ray-density exit amplitude is
# ``|E_in| / sqrt(|det J|)``, which diverges as ``det J -> 0`` at a fold
# caustic.  |det J| is floored at this fraction of the median |det J| over the
# ray-covered region so the amplitude stays finite (never inf/nan); a fold is
# also flagged when |det J| drops below the floor OR det J changes sign between
# adjacent ray cells, which triggers a one-time caustic warning steering the
# caller to GBD/FGA.  1e-3 is well below the ~O(1) det-J variation a smooth
# (non-folding) coma redistribution produces, so it never clips a legitimate
# aberrated spot; it only engages at a genuine fold.
_RAY_DENSITY_CAUSTIC_FLOOR_REL = 1e-3

# N12 (P11): a |det J| dynamic-range (max/min over the ray-covered region) above
# this flags a near-caustic even when the coarse grid does not resolve the exact
# fold curve (a sign change) or drive a sample below the absolute floor -- e.g.
# tracing to a plane at/near a focus, where all rays crowd into a tiny region so
# |det J| spans orders of magnitude and the single-branch ray-density amplitude
# under-resolves the singular spot (energy is NOT conserved there).  A smooth,
# non-folding aberrated map varies |det J| by <~a few x, well below this; a
# genuine caustic spans >>30x.  Conservative (a false positive only steers the
# caller to GBD/FGA, never returns a wrong number).
_RAY_DENSITY_CAUSTIC_MAXMIN = 30.0

# D1 (2026-07-28): the whole-grid det J scan is NOT masked to the beam's own
# support when the ray-fit disc sits off centre.  An earlier D1 revision did
# mask it, on the reading that an off-centre disc leaves the rest of the launch
# domain to polynomial EXTRAPOLATION whose det J is not a property of the
# optics.  That reading was wrong in the only way that matters: the fold the
# scan reported was REAL -- the hard-masked off-centre fit genuinely folded, and
# the same calls returned a spurious lobe at 0.75 of the on-beam peak (see
# ``_FIT_DISC_OUTSIDE_WEIGHT_REL``).  Masking the scan would have converted a
# loud wrong answer into a silent one.  With the fit regularised the fold is
# gone at the source and the unmasked scan is silent on the same cases, so the
# scan stays exactly as it was on every path.

# v5.30 (audit E-M6): post-hoc energy self-check for ``amplitude_model=
# 'ray_density'``.  The mode is documented as "energy-conserving in the
# geometric limit", and it is -- in the LIMIT.  At finite ``ray_subsample`` the
# Jacobian is evaluated on the coarse Newton lattice and the resulting
# amplitude is bilinearly upsampled, so the exit power falls short of the
# aperture-transmitted input power by an amount that shrinks monotonically as
# ``ray_subsample -> 1``, with no diagnostic at all before this check.
#
# Measured over the P2 design-battery envelope (all 24 cells: the six groups of
# the four battery designs x w0 in {0.6, 1.2} mm x aperture:beam in {1.2, 2.5},
# N = 512, ``P_out / P_in_aperture``), full min..max per subsample:
#
#     ray_subsample = 1 : 0.95685 .. 1.00003
#     ray_subsample = 2 : 0.95669 .. 0.99944
#     ray_subsample = 4 : 0.95604 .. 0.99794
#     ray_subsample = 8 : 0.95347 .. 0.99200   (the SHIPPED default)
#
# Two distinct effects are visible in that table and the band has to tolerate
# BOTH:
#
#   * the ``ray_subsample`` discretisation loss -- the well-conditioned cells
#     run 0.9992 at sub=1 and 0.9866-0.9913 at sub=8, i.e. ~1% at the shipped
#     default, converging monotonically (this is the audit's finding);
#   * a subsample-INDEPENDENT physical floor -- the battery's negative
#     corrector (relay g2) at a 1.2x aperture:beam ratio sits at 0.9535 (sub=8)
#     to 0.9569 (sub=1): its diverging exit fan genuinely leaves the output
#     window, so that power is not "lost" by the model, it is off-grid.
#
# So: a sub-aware deficit tolerance ``BASE + SLOPE * (sub - 1)`` = 0.080 /
# 0.090 / 0.110 / 0.150 at sub = 1 / 2 / 4 / 8, i.e. lower bounds 0.920 /
# 0.910 / 0.890 / 0.850 against measured worst cells 0.9569 / 0.9567 / 0.9560 /
# 0.9535 -- clear of every battery cell at every subsample, so the shipped
# defaults never warn spuriously.  What it DOES catch is the order-of-magnitude
# class: measured on a deliberately-broken cell (a strong biconcave, R = -3 /
# +3 mm, on a grid barely covering its 3 mm aperture) the ratio came back
# 1.100 at sub=8 and 1.330 at sub=4 -- energy GAIN, from a fold caustic
# inflating the capped ``1/sqrt(|det J|)`` amplitude.
#
# The GAIN side has no discretisation excuse -- ray-tube transport cannot
# create energy -- so it is a small fixed band (max measured 1.00003).
#
# v5.32 (2026-07-31, docs/audits/C6_FIT_GUARD_DECISION_2026_07_31.md S5.1):
# the GAIN side was MEASURED for tightening and DELIBERATELY LEFT AT 0.050.
# Recorded because the obvious change is wrong and the reason is not obvious.
#
# The band above was calibrated on the P2 battery at N = 512, where the largest
# ratio anywhere is 1.00003 -- which makes 0.050 look like ~1600x of unused
# headroom, and makes 0.005 look free.  Re-measured on the SAME battery at the
# N = 1024 the CI test actually runs (``tests/unit/test_niche_p2_design_battery
# .py``, ``_N = 1024``, ``_RS = 4``), the ratio per element is:
#
#     cell (triplet, w0 1.6 mm, aperture:beam 2.5x)   grp0     grp1     grp2
#       ray_subsample = 8                            0.99731  1.38446  1.30097
#       ray_subsample = 4  (the CI value)            0.99933  0.94477  1.04374
#       ray_subsample = 2                            0.99984  1.00069  1.00793
#       ray_subsample = 1                            0.99996  0.99996  0.99998
#
# So a currently-GREEN battery cell reads **1.04374** at the subsample CI uses,
# and 0.005 -- or any tolerance below 0.044 -- would warn on it.  That ratio is
# a real defect (it converges to 1.00000 as ray_subsample -> 1, and the
# fold-caustic warning already fires on every group of that design), but it is
# the SAME MAGNITUDE as the defects the audit wanted the band to catch
# (1.03317, 1.04593).  A scalar power sum therefore cannot separate the two
# populations on this library at all: the honest conclusion is not "tighten
# it", it is "this observable is exhausted".
#
# Nor would tightening have caught what prompted the review.  The shipped
# on-axis C6 production call reads 1.000741 while carrying a manufactured lobe
# at 83 % of peak; the same defect read 1.001058 before a library edit and
# 0.999371 after, with the lobe unchanged at 3.4e-03 of the input power.
# TOTAL POWER IS NOT SUFFICIENT -- which is what the halo term below is for.
_RD_ENERGY_DEFICIT_BASE = 0.080
_RD_ENERGY_DEFICIT_PER_SUB = 0.010
_RD_ENERGY_GAIN_TOL = 0.050


# ---- v5.32: the HALO-AMPLITUDE term of the ray_density self-check ----------
#
# WHY A SECOND TERM AT ALL.  The energy self-check above is a scalar power sum,
# and the failure mode it was written for -- a fold caustic inflating the
# capped ``1/sqrt(|det J|)`` amplitude -- is only ONE of the two ways the
# ray-density path goes wrong.  The other is D1's: the fitted entrance->exit
# map is Newton-inverted far outside its own data support, the inversion finds
# a SPURIOUS root, and ``ray_density`` hands that root real amplitude.  That
# deposits a lobe at a radius no traced ray can reach.  Measured on design
# 121's last group (docs/audits/ENERGY_CONSERVATION_AUDIT_2026_07_31.md S2.4):
# a library change moved the total-power signature from 1.001058 (visibly
# wrong) to 0.999371 (inside every absolute band) while the lobe stayed put at
# 3.4e-03 of the input power at 77 % of peak.  A criterion on total power alone
# would have called that fixed.
#
# Total power separates a clean field from a ghosted one by the SIZE of the
# ghost -- 4.7e-03 against a 5e-02 tolerance, 11x inside.  The halo amplitude
# separates the same two fields by 1.40e-05 against 8.32e-01, a factor of 6e4.
#
# WHAT IS MEASURED.  Immediately after the launch lattice is traced -- before
# any fit-domain restriction, so this reads the OPTICS and not the model -- the
# exact exit positions of every ALIVE launch ray carrying input amplitude
# >= ``e^-_RD_HALO_AMP_CONTOUR`` of peak are reduced to an amplitude-weighted
# exit centroid and a support radius ``r_hull`` (their largest radius about
# that centroid).  ``e^-9`` is the ``r = 3w`` contour of a Gaussian, whose
# interior holds 1 - 1.5e-08 of the beam power; on a hard-edged or truncated
# input it degenerates to the whole traced pupil, which is the conservative
# thing to do.  Then, on the returned field:
#
#     amax_halo = max |E_out| beyond ``_RD_HALO_RADIUS_FACTOR * r_hull``
#                 of that centroid, over max |E_out|
#
# and the check warns when ``amax_halo > _RD_HALO_AMAX_TOL``.  The radius is
# derived from the prescription and the input, never chosen, and both the hull
# and the halo are measured about the SAME centroid, so the statistic is
# invariant to where the beam sits on the grid.
#
# AMPLITUDE, NOT POWER, IS THE BOUND.  A power fraction is only as sensitive as
# the ghost is large; the amplitude ratio is what separates a LOBE from the
# ray-model's legitimate skirt.  The lobe this catches on design 121 carries
# 0.34 % of the power but stands at 77 % of the peak.  ``g_halo`` (the power
# fraction beyond the same radius) is reported in the message for context and
# is deliberately NOT part of the bar.
#
# CALIBRATION -- ``validation/repro_traced_carrier_121/halo_calibration.py``,
# which forces the tolerance negative so every call prints its own reading and
# then re-scores the captured field at every radius factor from ONE pass.  The
# populations are 180 element calls, 177 of which produce a reading (the other
# three are P2 battery cells the check DECLINES on, per SCOPE (d) below):
#
#   CLEAN     the CI-safe P2 design battery (``tests/unit/
#             test_niche_p2_design_battery.py``: four designs x two beam sizes
#             x two aperture:beam ratios, every group, at ray_subsample 1, 4
#             and 8 -- gated in CI against an exact meridional ray oracle), the
#             synthetic C6 ghost fixtures on their clean branches, and every
#             design-121 element call whose halo is under its own exact-ray
#             ceiling (six DOE orders x six groups x three flag configurations).
#   DEFECTIVE lobes CONFIRMED manufactured against an exact ray trace: design
#             121's on-axis C6 call (77 % of peak), its (-2,0) and (-3,0) C6
#             calls (both above the exact-ray g4 ceiling, and both exactly zero
#             with C6 off), and the fit guard's own regression on two synthetic
#             fixtures (4.6e-02 and 8.8e-01 of peak).
#
# ``amax_halo`` of the WORST clean call and the MILDEST real defect, as the
# radius factor is swept (this is the whole basis for both constants):
#
#     factor    worst CLEAN    mildest DEFECT    separation
#      1.00      2.270e-04       5.727e-03           25x
#      1.10      1.046e-04       5.727e-03           55x
#      1.25      4.622e-05       5.684e-03          123x
#      1.50      1.246e-05       5.684e-03          456x
#      2.00      ~1e-16          MISSED (1.4e-04 / 0.0)
#
# THE FACTOR 1.25.  The hull is a RAY radius and the returned field is a
# ray-density field bilinearly upsampled off a coarse lattice, so its amplitude
# support spills about one coarse cell past the last traced ray; that spill is
# what the clean column above is measuring, and it dies by 1.25.  Beyond 1.50
# the bound starts stepping OVER real defects -- at 2.00 both the (-2,0)-class
# lobe and the synthetic one are missed entirely.  1.25 is the smallest factor
# that clears the upsample spill by more than an order of magnitude while still
# reading every measured defect at full amplitude.
#
# THE TOLERANCE 1.0e-03 then sits inside a gap of 123x, at 21.6x above the
# worst clean reading and 5.7x below the mildest real defect.  Every number in
# this note is a measurement; the full table is in the audit S5.2.
#
# WHAT IT FIRES ON TODAY, stated so nobody is surprised.
#
#   * design 121's post-DOE RELAY, on the last group of orders (0,0), (-2,0)
#     and (-3,0) -- 3 of 36 shipped element calls.  All three are true
#     positives: each carries a lobe above that order's exact-ray halo ceiling
#     and each is exactly zero with ``REMAP_STATIONARY_PHASE_LAUNCH = False``.
#     By (d) it does not reach that chain's fine-grid readout leg.
#   * **niche D6's exact-tilted-leg RETRACE**, in
#     ``test_the_tilted_exact_leg_conserves_power_like_the_paraxial_one`` and
#     ``test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle``:
#     ``amax_halo`` = 6.405e-01 of peak beyond 2.0202 mm against an exact-ray
#     support of 1.6161 mm, ``g_halo`` = 6.449e-04, on a grid reaching 2.4341
#     mm from the centroid -- a FULL annulus, not a corner sliver.  That is a
#     previously unreported manufactured lobe in a green CI fixture, found by
#     this check on its first run; the fold-caustic warning already fires on
#     the same call, so the two diagnostics agree on the mechanism.  Those
#     tests assert spot metrics and stage power and still pass -- the lobe is
#     outside everything they measure, which is the point.  Recorded, NOT
#     silenced.
#
# It does NOT fire on any group of any P2 battery cell at any subsample (54
# calls, 51 readings, worst 1.6e-05, three declined under (d)).
#
# SCOPE, stated rather than papered over.
#
# (a) This bounds the halo at the ELEMENT's own exit plane; where a
#     manufactured lobe ends up after further propagation is not determined by
#     it.
# (b) It cannot see the OTHER half of the defect the audit found -- a
#     configuration returning materially LESS discretisation deficit than the
#     same design's ray lattice produces -- because that needs a same-design
#     reference run the element does not have.
# (c) It is a halo test, not a conservation test: a field can pass this and
#     still fail the power band above, and vice versa.  Both fire
#     independently.
# (d) **IT DECLINES WHEN THE TRACED EXIT SUPPORT DOES NOT FIT THE GRID**, and
#     that is not a hypothetical corner case.  The check requires the bound
#     circle ``_RD_HALO_RADIUS_FACTOR * r_hull`` to lie inside the grid about
#     the traced exit centroid; otherwise all that is left of the annulus is a
#     sliver of corners, and the statistic measured there is unreliable in
#     BOTH directions.  That is measured, not assumed:
#
#       * design 121, PRODUCTION readout leg (the last group re-run on the
#         ``n_fine_cap = 12288`` fine grid): at (0,0) the grid half-width is
#         6.277 mm and ``r_hull`` is 7.136 mm -- there is no annulus at all;
#         at (-2,0) it is 7.771 vs 7.827 mm, and the corners-only reading says
#         4.5e-04 (silent) with the fit guard OFF and 1.4e-03 (warns) with it
#         ON, on two fields that are BOTH defective at 4 mm (1.63e-02 and
#         1.48e-02 of peak).  It got the ordering right and the verdict wrong;
#       * niche D6's exact-tilted-leg retrace (``tests/unit/
#         test_niche_d6_exact_tilted_leg.py``): a 3.4 mm aperture on a 3.6 mm
#         grid, ``r_hull`` 1.619 mm against a 1.799 mm half-width, whose
#         corners-only reading is 0.841 of peak at the grid corner diagonally
#         opposite the beam, with ``P/Pin`` = 1.00057.
#
#     The SAME design-121 groups in the RELAY configuration (the co-moving
#     1024 grid, ``dx`` 33.2 um, supports 2.993 / 4.817 mm against a 17.0 mm
#     half-width) give a full annulus and read 0.770 and 5.7e-03 of peak.  So
#     the check covers the relay and declines on the readout leg, and the D6
#     retrace reading above is recorded as an OPEN observation rather than
#     shipped as a warning -- see the audit S7.  A grid whose extent is
#     comparable to its own exit fan cannot support a halo statement; the
#     alternative -- tightening ``_RD_HALO_AMP_CONTOUR`` until a hull always
#     fits -- buys that coverage with false positives on every legitimate
#     skirt.
#
# Warning-only, like the energy check, and suppressed process-wide by setting
# ``RAY_DENSITY_HALO_CHECK = 'silent'``.
_RD_HALO_AMP_CONTOUR = 9.0
_RD_HALO_RADIUS_FACTOR = 1.25
_RD_HALO_AMAX_TOL = 1.0e-03
#: Policy for the ray-density HALO self-check: ``'warn'`` (default) or
#: ``'silent'``.  Never an error -- the returned field's core metrics are
#: unaffected by a far lobe, so refusing the call would be worse than
#: reporting it.  See ``_RD_HALO_AMAX_TOL``.
RAY_DENSITY_HALO_CHECK = 'warn'


# ---- niche D9: the analytic amplitude leg under a decentred grid ORIGIN -----
# ``apply_real_lens_traced(origin=...)`` decentres the WAVE grid only (see the
# ``origin`` docstring entry).  Everything the traced leg owns -- the ray
# launch, the Newton inverse, the ray-density amplitude, the residual transport
# -- is carried in the element's absolute (optical-axis) frame and therefore
# moves with the origin exactly.  The ANALYTIC amplitude leg does not: it is one
# ``apply_real_lens`` call, which has no origin of its own and builds the
# element's sag, its ``aperture_diameter`` mask, the per-surface
# ``clear_aperture`` masks and the stop symmetrically about ITS OWN grid centre.
#
# Under ``amplitude_model='ray_density'`` + ``preserve_input_phase='remap'``
# that leg enters the returned field through EXACTLY ONE quantity: the ZERO SET
# of ``amp = |apply_real_lens(E_in)|``.  The proof is mechanical --
# ``preserve_input_phase='remap'`` sets ``preserve_input_phase = False``, so the
# assembly builds ``E_out = amp * exp(i k0 opl)`` with ``amp`` REAL and
# non-negative, and the ray-density swap then divides its own modulus out
# (``_unit = E_out / |E_out|``, zero where ``|E_out| == 0``) and multiplies the
# ray-tube magnitude back in.  So a wrong analytic ENVELOPE shape is discarded;
# only a wrong analytic MASK can reach the answer, and only by deleting light.
#
# That is a checkable statement, so it is CHECKED rather than assumed: after the
# swap, the power the analytic zero set removes from the ray-density amplitude
# is measured directly and the call REFUSES above ``_ORIGIN_AMP_SUPPORT_TOL``.
# Set this to ``'warn'`` to accept the deletion, ``'silent'`` to skip the two
# reductions entirely (do not: the failure it catches is a beam quietly clipped
# by a stop the analytic leg placed at the wrong transverse position).
#
# MEASURED (2026-08-04, decentred singlet fixture, chief ray 0.233 mm off axis
# against a 0.25 mm beam): the zero set is EMPTY -- and the coupling therefore
# identically absent -- whenever every analytic mask is followed by a
# propagation, which covers the ordinary case of a prescription carrying only an
# entrance ``aperture_diameter``: the ASM through glass fills the shadow back in
# and ``min |E_analytic|`` reads 6.3e-06, not 0.0, over the whole grid.  What
# does produce exact zeros is a mask that lands on the EXIT PLANE itself -- a
# ``clear_aperture`` on the LAST surface, or a stop there.  On that fixture such
# a mask deletes 1.3e-04 % of the ray-density exit power at
# ``clear_aperture`` = 1.20 mm, 0.13 % at 0.90 mm and 0.58 % at 0.80 mm, i.e.
# real beam, quietly, and the refusal is what stands between that and a
# plausible-looking result.
ORIGIN_AMP_SUPPORT_CHECK = 'error'
#: Fraction of the ray-density exit power the axis-centred analytic amplitude
#: leg's zero set may delete before ``ORIGIN_AMP_SUPPORT_CHECK`` fires.  Not a
#: tuned tolerance -- the intended value is exactly zero, and this is only the
#: allowance for an analytic envelope that has UNDERFLOWED to 0.0 in a far
#: skirt where the ray-density amplitude is itself ~1e-30 of peak.
_ORIGIN_AMP_SUPPORT_TOL = 1.0e-09


# Module-level default for ``apply_real_lens_traced(parallel_amp=...)``.  The
# kwarg default is ``None`` -> resolves to this global, so a process-wide
# ``set_lens_parallel_amp(False)`` (or ``lumenairy.set_low_memory(True)``)
# flips the amp+amp(pw) concurrency off for callers that don't pass the kwarg.
# Shipped default True is byte-identical to the historical behaviour; turning
# it off is the single largest lens-step memory claw-back (~2x working set)
# and is numerically identical (same math, serialised).
_LENS_PARALLEL_AMP_DEFAULT = True

# Row-block budget (in float64 ENTRIES) for the Chebyshev fit's design-matrix
# construction in ``_Cheb2DEvaluator.__init__``.  8e6 entries = 64 MB of
# gather scratch per block, independent of the sample count and the fit order.
# See ``docs/audits/FIX_RUNNER_OOM_2026_08_13.md``; the sibling constant on the
# inverse-map side is ``_lens_imap._IMAP_FIT_CHUNK_ENTRIES``.
_CHEB_FIT_CHUNK_ENTRIES = 8_000_000


def set_lens_parallel_amp(enabled: bool) -> None:
    """Set the process-wide default for ``apply_real_lens_traced``'s
    concurrent amp + amp(pw) execution.  ``False`` halves the lens-step
    peak working set (byte-identical output, ~20% slower lens step)."""
    global _LENS_PARALLEL_AMP_DEFAULT
    _LENS_PARALLEL_AMP_DEFAULT = bool(enabled)


def get_lens_parallel_amp() -> bool:
    """Return the process-wide default for the lens amp/amp(pw) concurrency."""
    return bool(_LENS_PARALLEL_AMP_DEFAULT)


# Helpers shared with lenses.py (aperture warning).
# v5.30 (audit E-L4): ``surface_sag_general`` and its private
# ``_surface_sag_general`` alias were imported/defined here with ZERO readers in
# this module (grep-verified: the only two occurrences were the import and the
# alias itself) and nothing imports either name FROM this module -- the live
# users are ``_lens_real.py`` and ``elements.py``, which import from
# ``lenses.py`` directly.  Both deleted; this module gets its sag from the
# raytrace core via the trace, not from the analytic helper.
# Sibling-module imports.
# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
# Module-level logger for apply_real_lens_traced entry / per-Newton-
# iteration progress.  Default-quiet via the lumenairy root logger's
# NullHandler -- users opt in by attaching a handler to the
# ``lumenairy`` logger.
from .._logging import get_logger
from ..glass import get_glass_index
from ..progress import ProgressScaler, call_progress
from .lenses import _warn_if_aperture_exceeds_grid

logger = get_logger(__name__)

# apply_real_lens (analytic split-step) is the workhorse for the
# amplitude leg of apply_real_lens_traced.  Lives in _lens_real.py
# since v3.5.5; re-imported here for the in-function callbacks.
from ._lens_real import apply_real_lens


def _prescription_has_field_frame(prescription) -> bool:
    """True when any surface carries a P3-style FIELD-FRAME decenter / tilt /
    freeform ``sag_callable`` (the ``apply_real_lens`` displaced-pointwise
    convention).  For such elements the ray trace (``trace`` ->
    ``_intersect_surface`` / ``_refract`` via the shared field-frame
    ``_surface_sag_xy``) carries the transverse ray WALK-OFF -- the true induced
    coma -- into the geometry and OPL, so the traced centroid / sign-mirror /
    tilt are oracle-matched.

    Detection helper only (used by the tests and available for dispatcher
    routing).  It does NOT gate any amplitude change: the P9 field-frame
    amplitude override was REMOVED (2026-07-20) after the adversarial verifier
    proved it was a model-mixing artefact.  The traced hybrid's grid-indexed
    amplitude leg cannot carry the decentered walk-off (an asymmetric ray-
    density redistribution), so its decentered-spot EE is amplitude-limited -- a
    genuine model limit of the P3 single-plane class.  Route decentered-coma EE
    to ``apply_real_lens_gbd`` (N10b), whose beamlets carry the walk-off
    amplitude and BROADEN matching ZOS (1.035 @1.31um) + the geom-spot oracle.
    See docs/audit_real_lens_displaced_2026_07_19.md (P9 / N10a)."""
    for s in (prescription.get('surfaces') or []):
        if not isinstance(s, dict):
            continue
        if s.get('sag_callable') is not None:
            return True
        dec = s.get('decenter') or (0.0, 0.0)
        if float(dec[0]) != 0.0 or float(dec[1]) != 0.0:
            return True
        tl = s.get('tilt') or (0.0, 0.0)
        if float(tl[0]) != 0.0 or float(tl[1]) != 0.0:
            return True
    return False


# --------------------------------------------------------------------------
# Newton-pool PAYLOAD RESIDENCY (FIX_PERF_ROUND2_2026_08_10 item 3;
# AUDIT_TRACED_SPEED_2026_08_09 sec 6.2 / ranked row 10).
#
# THE DEFECT.  ``_invert_newton_parallel`` builds one arg tuple per chunk and
# puts ``_spline_data`` -- the five ray-fit grids plus the pinned backend and
# the built Chebyshev coefficients -- inside EVERY one of them.  The executor
# therefore pickles the same payload once per chunk.  MEASURED (audit sec 6.2,
# design 121's largest fit grid, 531^2 = 11.28 MB): a bare 8-worker round trip
# is 1.5 ms and the same round trip WITH the payload is 173.1 ms, i.e. the
# payload is 99.2 % of the dispatch constant -- and ``FIX_D1_POOL``'s ~0.22 s
# constant is now identified rather than merely observed.
#
# THE FIX, in two independent halves.
#   1. The parent pickles the payload ONCE (``pickle.dumps``) instead of
#      letting the executor do it n_cpu times.  What crosses the wire is a
#      ``bytes`` object, whose own pickling is a memcpy.
#   2. Workers KEEP the last payload they were given, keyed by a content
#      digest of that blob.  A later dispatch whose payload digests the same
#      sends the KEY ALONE -- nothing else crosses.
#
# WHY THE KEY IS A CONTENT DIGEST AND NOT AN IDENTITY OR A COUNTER.  The
# payload dict is REBUILT per ``apply_real_lens_traced`` call but MUTATED per
# dispatch (``cheb_backend`` and ``cheb_fit`` are stamped in immediately before
# every dispatch).  Keying on ``id()`` or on a per-call counter would therefore
# let a worker answer from a payload whose pinned backend or built fit had
# since changed -- a silently different floating-point order, which is the
# exact failure class the backend pin and the shipped-fit change exist to
# remove (v5.32.3 / v5.33.0).  A digest of the wire bytes cannot do that: any
# change to any field changes the key.
#
# WHY RESIDENCY IS NEVER LOAD-BEARING FOR CORRECTNESS.  The parent tracks which
# key it BELIEVES the live workers hold, but ``ProcessPoolExecutor`` gives no
# guarantee that every worker took a chunk from a given dispatch (a fast worker
# can take two while another takes none).  A worker asked for a key it does not
# have raises :class:`NewtonPayloadNotResident`, and the parent re-submits that
# chunk WITH the blob.  The worst case is therefore today's behaviour, and the
# belief can only cost a re-submit, never a wrong answer.
#
# MEMORY.  A worker holds at most ONE payload: the registry is cleared before
# every insert, so the per-worker resident cost is bounded by the largest
# payload (11.28 MB on design 121), which is what a chunk held for its own
# duration anyway.  The pool initializer clears it at worker start.
_WORKER_PAYLOADS: dict = {}


class NewtonPayloadNotResident(RuntimeError):
    """A Newton pool worker was sent a payload KEY it does not hold.

    Expected and benign: the parent's residency belief is an optimisation, not
    a guarantee (see the block above).  ``_invert_newton_parallel`` catches it
    and re-submits that chunk with the payload attached, so the answer is
    unchanged and only a round trip is spent.
    """


def _newton_pool_init():
    """``ProcessPoolExecutor`` initializer: start every worker with an EMPTY
    payload registry.

    A freshly spawned worker imports this module and gets an empty dict
    anyway, so this is belt-and-braces -- it matters only if a worker process
    is ever reused across two pools in one interpreter, where a stale key
    would otherwise survive.  Cheap, and it makes the residency mechanism
    explicit at the pool's construction site rather than implicit in module
    import order."""
    _WORKER_PAYLOADS.clear()


def _newton_worker_payload(key, blob):
    """Worker side of the residency protocol: store-and-return, or look up.

    ``blob is None`` means the parent believes this worker already holds
    ``key``; if it does not, refuse loudly rather than guessing."""
    if blob is not None:
        import pickle
        data = pickle.loads(blob)
        # At most ONE resident payload per worker -- see MEMORY above.
        _WORKER_PAYLOADS.clear()
        _WORKER_PAYLOADS[key] = data
        return data
    try:
        return _WORKER_PAYLOADS[key]
    except KeyError:
        raise NewtonPayloadNotResident(
            f"Newton pool worker was asked for payload key {key!r}, which it "
            f"does not hold (resident: {sorted(_WORKER_PAYLOADS)}).  The "
            f"parent re-submits this chunk with the payload attached; nothing "
            f"about the ANSWER changes.") from None


def _newton_payload_blob(knot_data):
    """Parent side: ``(key, blob)`` for one dispatch.

    ONE ``pickle.dumps`` of the payload, whatever the worker count -- the
    executor then pickles a ``bytes``, not the dict, once per chunk.  The key
    is a 128-bit digest of those exact bytes, so it changes whenever any field
    of the payload changes (see the block above for why that matters)."""
    import hashlib
    import pickle
    blob = pickle.dumps(knot_data, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.blake2b(blob, digest_size=16).hexdigest(), blob


def _newton_invert_chunk(args):
    """Module-level worker for ``apply_real_lens_traced`` Newton inversion.

    Rebuilds the three ``RectBivariateSpline`` objects from their knot
    data in-process (so we avoid pickling the SciPy spline objects,
    which is expensive) and runs the Newton loop on ``(x_chunk,
    y_chunk)`` for up to ``knot_data['newton_max_iters']`` iterations.
    Returns ``(opl, n_unconverged)``: the OPL at the converged entrance
    positions with NaN for any points that landed outside the fit
    domain, plus the number of chunk points still active at the
    iteration cap.

    v5.29.1 (audit E-H2): the cap used to be the module constant
    ``_NEWTON_MAX_ITERS``, which made ``newton_max_iters`` INERT on the
    pool path (then: >=200k points with ``newton_fit='spline'``; since
    v5.30.1 / v5.32.2 the pool serves EITHER fit, above the two-tier
    200k-cold / 8k-warm size gate) -- the caller's
    cap was honoured by the serial closure only, so the OPL (pool) and the
    ray-density amplitude (always serial) could come from DIFFERENT Newton
    solutions.  The resolved cap now travels in the pickled payload;
    payloads written by older callers (no key) keep the historical 12.
    The unconverged count travels back so the pool path can emit the same
    "did not converge ... increase newton_max_iters" warning the serial
    path emits (pre-fix the pool was silent, and the advice in that very
    message did nothing on this path).

    v5.32.3 (FIX_CI_POOL): so does the parent's Chebyshev EVALUATOR BACKEND
    (``cheb_backend``), for the same reason and with the same fallback for
    payloads that predate the key.  A worker that resolved a different branch
    of ``_Cheb2DEvaluator.ev_value_and_grad`` than the parent ran the same
    mathematics in a different floating-point order, which cost the pool its
    bit-identity to serial (MEASURED 5.167e-14 locally, 1.358e-11 on CI).  A
    worker that cannot honour a pinned ``'numba'`` raises
    :class:`NewtonWorkerBackendUnavailable` rather than substituting the other
    order.

    v5.33.0 (FIX_POOL_REBUILD): and so does the parent's BUILT Chebyshev FIT
    (``cheb_fit``), which retires the polynomial re-fit above entirely.
    Rebuilding it here re-ran ``_solve_lstsq_thread_safe`` -- a BLAS reduction
    over a ~78 000-row design matrix -- in a fresh interpreter, and OpenBLAS
    reduces in a thread-count-dependent order, so a worker whose BLAS width
    differed from its parent's recovered DIFFERENT coefficients on identical
    data (MEASURED max|dc| 4.6e-15, which the Newton convergence threshold
    amplifies to 1.370e-11 of the field -- CI's 1.341e-11 / 1.358e-11).  The
    worker now EVALUATES the parent's coefficients and fits nothing; payloads
    with no ``cheb_fit`` key keep the historical rebuild.

    Lives at module scope so ``ProcessPoolExecutor`` can pickle it on
    Windows (spawn) workers.  The caller is ``_invert_newton_parallel``
    inside :func:`apply_real_lens_traced`.

    ARG SHAPE (FIX_PERF_ROUND2_2026_08_10 item 3).  ``args`` is either the
    historical ``(knot_data, x_chunk, y_chunk)`` -- still accepted, and what a
    direct caller or an old pickled payload uses -- or the four-element
    ``(payload_key, blob_or_None, x_chunk, y_chunk)`` the pool now sends, where
    ``blob`` is ``pickle.dumps(knot_data)`` and ``None`` means "you already
    have this key".  See ``_newton_worker_payload``.
    """
    if len(args) == 4:
        (_pkey, _blob, x_chunk, y_chunk) = args
        knot_data = _newton_worker_payload(_pkey, _blob)
        args = (knot_data, x_chunk, y_chunk)
    (knot_data, x_chunk, y_chunk) = args
    # Which fit to rebuild.  Payloads written before the polynomial worker
    # path existed carry no key -- default to 'spline' so they still run.
    _fit = str(knot_data.get('newton_fit', 'spline'))
    if _fit != 'polynomial':
        from scipy.interpolate import RectBivariateSpline
    xs_in = knot_data['xs_in']
    x_out_grid = knot_data['x_out_grid']
    y_out_grid = knot_data['y_out_grid']
    opl_grid = knot_data['opl_grid']
    launch_radius = knot_data['launch_radius']
    dx = knot_data['dx']
    bound = knot_data['bound']
    # Paraxial-magnification initial-guess factors.  See the docstring
    # in ``apply_real_lens_traced`` where these are computed from the
    # central finite-difference slope of the forward map.  Older knot
    # data written by pre-3.1.3 callers won't have these keys -- fall
    # back to the historical 1.10 multiplier so the worker stays
    # backwards compatible.
    inv_M_x = float(knot_data.get('inv_M_x', 1.10))
    inv_M_y = float(knot_data.get('inv_M_y', 1.10))
    # Newton iteration cap, resolved by the caller (caller override >
    # module default).  Pre-5.29.1 payloads have no key -- fall back to
    # the module default so an old pickled payload still runs (audit E-H2).
    max_iters = int(knot_data.get('newton_max_iters', _NEWTON_MAX_ITERS))

    if _fit == 'polynomial':
        _ord = int(knot_data.get('fit_poly_order', 6))
        _wts = knot_data.get('fit_weights', None)
        # The evaluator must EVALUATE in the parent's floating-point order,
        # which the payload pins (v5.32.3).  ``None`` -- payloads written
        # before the pin existed, mirroring the ``newton_fit`` /
        # ``newton_max_iters`` tolerance above -- keeps the historical
        # "resolve it here" behaviour.
        _backend = knot_data.get('cheb_backend', None)
        if _backend is not None and _backend not in ('numba', 'numpy'):
            _backend = None
        if _backend == 'numba' and _get_cheb2d_val_grad_numba() is None:
            # The parent evaluated through the numba kernel and this worker
            # cannot.  Answering with the pure-xp branch would be a DIFFERENT
            # floating-point order, i.e. a silently different result from the
            # serial path this pool promises to be bit-identical to -- so
            # refuse, and let the parent run the chunk itself.
            raise NewtonWorkerBackendUnavailable(
                "the Newton pool payload pins the numba Chebyshev kernel (the "
                "parent evaluated with it), but this worker cannot load it: "
                f"_NUMBA_AVAILABLE={_NUMBA_AVAILABLE!r}.  Refusing the chunk "
                "rather than answering in a different floating-point order.")
        # v5.33.0 (FIX_POOL_REBUILD): EVALUATE THE PARENT'S FIT, do not re-fit.
        # The payload ships the built coefficients, so the worker's polynomial
        # is the parent's by construction -- no least-squares solve here, and
        # therefore no dependence on this interpreter's BLAS thread regime.
        # See the measurement block at ``_cheb_fit_state``: a rebuild under a
        # different BLAS width moved the field by up to 1.370e-11, which is the
        # 1.341e-11 / 1.358e-11 that
        # ``test_pool_result_is_bit_identical_to_serial[polynomial]`` failed by
        # on all four CI python lanes AFTER the backend pin had shipped.
        _fit_state = knot_data.get('cheb_fit', None)
        if _fit_state is not None:
            Sx = _Cheb2DEvaluator.from_state(_fit_state['x_out'], xp=np,
                                             backend=_backend)
            Sy = _Cheb2DEvaluator.from_state(_fit_state['y_out'], xp=np,
                                             backend=_backend)
            So = _Cheb2DEvaluator.from_state(_fit_state['opl'], xp=np,
                                             backend=_backend)
        else:
            # No shipped fit: a payload written before this key existed, the
            # same tolerance ``newton_fit`` / ``newton_max_iters`` /
            # ``cheb_backend`` get above.  Re-fit from the grids, which is what
            # this worker always did -- and which is bit-identical to the
            # parent only while the two share a BLAS regime.
            Sx = _Cheb2DEvaluator(xs_in, xs_in, x_out_grid, order=_ord,
                                  xp=np, weights=_wts, backend=_backend)
            Sy = _Cheb2DEvaluator(xs_in, xs_in, y_out_grid, order=_ord,
                                  xp=np, weights=_wts, backend=_backend)
            So = _Cheb2DEvaluator(xs_in, xs_in, opl_grid, order=_ord,
                                  xp=np, weights=_wts, backend=_backend)
    else:
        Sx = RectBivariateSpline(xs_in, xs_in, x_out_grid, kx=3, ky=3)
        Sy = RectBivariateSpline(xs_in, xs_in, y_out_grid, kx=3, ky=3)
        So = RectBivariateSpline(xs_in, xs_in, opl_grid, kx=3, ky=3)
    # Mirror the serial path's choice EXACTLY: it prefers the combined
    # value+gradient when the fit exposes one (the polynomial evaluator does,
    # RectBivariateSpline does not).  Using a different call here would change
    # the floating-point operation order and break pool/serial bit-identity.
    _has_combined = (hasattr(Sx, 'ev_value_and_grad')
                     and hasattr(Sy, 'ev_value_and_grad'))

    xe = x_chunk.copy() * inv_M_x
    ye = y_chunk.copy() * inv_M_y
    tol = 0.01 * dx
    active = np.ones(xe.size, dtype=bool)
    for _it in range(max_iters):
        if not active.any():
            break
        xa = xe[active]
        ya = ye[active]
        xw = x_chunk[active]
        yw = y_chunk[active]
        if _has_combined:
            fx_val, jxx, jxy = Sx.ev_value_and_grad(xa, ya)
            fy_val, jyx, jyy = Sy.ev_value_and_grad(xa, ya)
            rx = fx_val - xw
            ry = fy_val - yw
        else:
            rx = Sx.ev(xa, ya) - xw
            ry = Sy.ev(xa, ya) - yw
            jxx = Sx.ev(xa, ya, dx=1)
            jxy = Sx.ev(xa, ya, dy=1)
            jyx = Sy.ev(xa, ya, dx=1)
            jyy = Sy.ev(xa, ya, dy=1)
        det = jxx * jyy - jxy * jyx
        safe = np.abs(det) > 1e-12
        inv_det = np.where(safe, 1.0 / det, 0.0)
        dxe = (jyy * rx - jxy * ry) * inv_det
        dye = (-jyx * rx + jxx * ry) * inv_det
        xa_new = np.clip(xa - dxe, -bound, bound)
        ya_new = np.clip(ya - dye, -bound, bound)
        xe[active] = xa_new
        ye[active] = ya_new
        res = np.sqrt(rx * rx + ry * ry)
        converged = res < tol
        idx_active = np.where(active)[0]
        active[idx_active[converged]] = False

    opl_flat = So.ev(xe, ye)
    out_of_domain = (xe * xe + ye * ye > (launch_radius * 0.99) ** 2)
    # (opl, n_unconverged) -- the count lets the parent emit the serial
    # path's unconverged warning for the pool path too (audit E-H2).
    return (np.where(out_of_domain, np.nan, opl_flat), int(active.sum()))


# --------------------------------------------------------------------------
# Persistent ProcessPool for apply_real_lens_traced Newton inversion.
#
# Pre-3.5.5: every apply_real_lens_traced call created+torn-down its own
# pool, paying the Windows-spawn startup cost (~5 s for n_workers=8) once
# per call.  For optimisation runs and tolerancing studies that call
# apply_real_lens_traced 100+ times the cumulative cost was minutes.
#
# 3.5.5+: a module-level pool is lazily created on first parallel-Newton
# call and reused across subsequent calls with the same worker count.
# An atexit handler shuts it down cleanly.  Call ``close_worker_pool()``
# explicitly to free the workers early (e.g. after a final optimisation
# step, before a long-running serial post-process).
# --------------------------------------------------------------------------

# Minimum Newton points before the process pool is used at all; below this the
# inversion runs in-process.  Originally a HEURISTIC sized against Windows spawn
# cost (~200-400 ms per worker).  That rationale weakened once the pool became
# PERSISTENT (v3.5.5): spawn is then paid once per session rather than per call,
# leaving only per-chunk pickling.  Module scope so the crossover can be
# MEASURED rather than assumed.
_POOL_MIN_PIXELS = 200_000
# ... and once that pool EXISTS, the spawn is already paid and only per-chunk
# pickling has to amortise, so the pool pays off far sooner.  MEASURED on a
# 2-group chain, 8 workers (serial -> pooled):
#
#   points     COLD (fresh process)        WARM (pool already alive)
#    16 384    1.707 -> 2.768  0.62x        0.60 -> 0.50   1.20x
#    65 536    4.129 -> 4.797  0.86x        2.88 -> 2.35   1.22x
#   262 144   11.427 -> 9.360  1.22x       10.98 -> 7.17   1.53x
#     1 024         -                       0.04 -> 0.03   1.14x
#
# So the COLD crossover really is ~200k (the shipped value is right, and
# lowering it outright would make one-shot runs SLOWER -- 16k cold is 1.62x
# worse pooled).  Warm, the pool wins at every size measured down to 1k.  A
# multi-group chain calls apply_real_lens_traced once per group, so with a
# single threshold every group of a design-121-class chain at ray_subsample=4
# (65k points) runs serial forever, even though only the FIRST one is cold.
_POOL_MIN_PIXELS_WARM = 8_000

# ---- SECOND-CALL PROMOTION (v5.32.2, adversarial review D1) ---------------
# The two-tier threshold above was UNREACHABLE as first shipped.  ``warm`` was
# derived from ``_PERSISTENT_POOL is not None``, but the pool is created at
# exactly one site -- ``_get_persistent_worker_pool``, called DOWNSTREAM of the
# gate that consults it -- so only a call that had already cleared the 200k
# COLD bar could ever warm the process.  A process whose calls all sit in the
# 8k-200k band therefore never warmed, and every group of the design-121 chain
# the split was written for (65 536 Newton points/group at ray_subsample=4)
# still ran serial.  Measured on a fresh process, N=512/rs=2/3 groups, 4
# workers: pool created False, pool-dispatch events 0.  The mechanism was
# right; its trigger was unreachable.
#
# The reachability fix is a fact about the WORKLOAD, not about the pool:
# remember that this process has already run a pool-SIZED Newton inversion
# serially.  The FIRST sub-cold-bar call runs serial (so a genuine one-shot
# never pays a spawn it cannot amortise -- cold 16k is 1.62x SLOWER pooled);
# the SECOND such call may create the pool, and every call after it is warm.
# A process that reaches that line twice is a chain / sweep, not a one-shot.
#
# ---- ...AND A COST GATE, because POINT COUNT IS NOT COST ------------------
# Reachability alone is not enough, and shipping it alone would have made the
# motivating workload SLOWER.  ``_POOL_MIN_PIXELS_WARM`` is a POINT count, but
# what the pool actually competes against is the WALL TIME of the serial
# Newton step -- and at a fixed 65 536 points that varies by 20x with the fit
# backend, because the polynomial fit's Chebyshev evaluator has a numba
# ``prange`` kernel (``@njit(parallel=True)``, see
# ``_get_cheb2d_val_grad_numba``) and is therefore ALREADY multicore
# in-process, while ``RectBivariateSpline.ev`` is single-threaded and does not
# release the GIL.  Measured here, 65 536 Newton points per call, steady state
# (24 cores, numba 0.65.1 using 24 threads):
#
#   backend                        serial Newton   share of the group's wall
#   polynomial + numba (DEFAULT)      0.048 s               1.5 %
#   spline                            0.553 s              13.1 %
#   polynomial, numba unavailable     0.95  s              17.3 %
#
# Against that, the measured per-DISPATCH pool overhead is ~0.22 s at 8
# workers (payload pickling + IPC round trip), so pooling the DEFAULT path at
# this size is a ~5x net LOSS per call.  End-to-end, 121-shape chain (N=1024,
# ray_subsample=4, polynomial, 8 workers, 3 interleaved reps, quiet box,
# min wall):
#
#    6 groups   serial-Newton 19.363 s   pooled-every-group 20.673 s  (+6.8%)
#   12 groups   serial-Newton 45.806 s   pooled-every-group 48.215 s  (+5.3%)
#
# The gap GROWS with group count, so it is a per-dispatch cost, not a one-off
# spawn that more groups amortise.  In the two regimes where the Newton step
# is genuinely expensive the same pool wins -- numba-unavailable, 6 groups:
# 32.787 s serial vs 27.231 s pooled (1.20x), with the Newton step itself
# going 5.685 s -> 0.803 s.
#
# So the promotion is gated on the MEASURED serial time of the deferred call,
# not on its point count.  This self-calibrates: it needs no backend sniffing,
# no numba probe and no per-machine table, and it automatically follows any
# future change that makes either fit faster or slower.
#
# THRESHOLD.  A dispatch can save at most ``t (1 - 1/n_workers)`` and costs
# ~0.22 s, so at 8 workers the break-even is 0.22/0.875 = 0.25 s of serial
# Newton.  0.35 s carries a 1.4x margin over that while sitting 6.4x above the
# default path's steady-state 0.055 s -- both sides comfortable.  Measured
# per-group deferred Newton times at 65 536 points, same chain, 8 workers:
#
#   polynomial + numba   0.637, 0.055, 0.048, 0.051, 0.055, 0.049   -> serial
#   spline               0.534, 0.545, 0.547, 0.580, 0.612, 0.619   -> pooled
#   polynomial, no numba 1.028, 1.011, 0.978, 1.099, 1.250, 1.173   -> pooled
#
# With very few workers the break-even rises (``1 - 1/n`` shrinks) and this
# single constant is optimistic there; the pool is only a modest win at n=2
# anyway, so it is not worth a second knob.
_POOL_PROMOTE_MIN_SECONDS = 0.35

# ...and note the FIRST column of that table.  0.637 s against a 0.048 s
# steady state: the first Newton inversion in a process pays one-time warm-up
# (numba compiles / loads the cached ``prange`` kernel on first call), which
# is exactly 13x the recurring cost the pool would actually be competing
# against.  Deciding on that sample promotes on an artifact -- measured: it
# pooled all 5 remaining groups of a 6-group 121-shape chain and cost 6.8%.
# So the estimator is the MINIMUM over at least TWO deferred inversions: the
# warm-up sample is outvoted by the first steady-state one, and taking the min
# (rather than the mean) keeps the rule conservative -- it under-promotes on a
# heterogeneous sweep rather than over-promoting.
_POOL_PROMOTE_MIN_SAMPLES = 2

# ---- V5 (2026-08-06): THE EVIDENCE MUST BE KEYED BY WHAT DETERMINES COST --
# As first shipped, that measurement was a BARE WALL TIME recorded against a
# worker count.  It carried no record of WHICH fit backend produced it and no
# point count, and ``_pool_reuse_is_likely`` then applied it to EVERY later
# inversion at that worker count that cleared the warm bar.  Both blind spots
# re-admit exactly the +5-7% regression the gate above exists to reject.
# Measured on the shipped gate, 4 workers, one process per block:
#
#   2 SPLINE groups at  65 536 pts -> 0 dispatches, armed at 0.504 s
#   then 4 POLYNOMIAL groups
#        at the SAME    65 536 pts -> 4 dispatches -- yet that step measures
#                                     0.048 s, 7.3x UNDER the 0.35 s bar and
#                                     ~4.6x cheaper than one dispatch
#
#   2 SPLINE groups at 116 281 pts -> 0 dispatches, armed at 1.036 s
#   then 4 groups at    16 384 pts -> 4 dispatches -- 7.1x fewer points, so
#                                     ~0.15 s, under the bar AND under the
#                                     ~0.22 s dispatch cost
#
# Mixed-backend processes are a SHIPPED idiom rather than a hypothetical:
# spline is used deliberately as the fit-domain-free oracle in ``test_niche_d7``,
# ``c6``, ``c8`` and the C11/C12 validation scripts (see the D5 adjudication at
# ``_fit_domain_basis_ok``), so a spline reading arming the default polynomial
# path is a thing this library's own test suite does.
#
# The pending measurement is therefore keyed by a COST CLASS -- the triple
# (worker count, fit backend, point-count band) -- and re-armed from scratch
# whenever any leg of it changes:
#
#   * WORKER COUNT.  Unchanged rationale: a pool built for a different count
#     is torn down and rebuilt, so it amortises nothing.
#   * FIT BACKEND.  The 20x spread in the table above is measured at FIXED
#     size, so it is a property of the backend, not of the work.  A sample
#     from one backend is not evidence about another.  ``_newton_cost_class``
#     also separates polynomial-with-numba from polynomial-without, which is
#     the other 20x edge of that same table (0.048 s vs 0.95 s).
#   * POINT COUNT.  The warm band spans 8 000 - 200 000 points -- a 25x range
#     -- and the serial step's cost scales with points.  A sample is reused
#     only for a query within ``_POOL_PROMOTE_SIZE_RATIO`` of its OWN size,
#     and inside that band it is scaled linearly to the query size before it
#     is compared against the bar.  The band is what bounds the
#     extrapolation: proportionality is a reasonable local model and a
#     terrible 25x one, so it is only ever applied over a factor of two.
#
# Cost of being wrong is asymmetric, which is why re-arming is the response to
# a mismatch rather than widening the key: a bucket that has to re-measure
# loses only the win on two calls, while a bucket that promotes on someone
# else's measurement taxes every remaining call in the process.
#
# 2.0: at the band edge the scaled estimate is exact only if the step is
# perfectly proportional to point count, which is not claimed -- there is a
# fixed per-call fit/setup cost, so a call twice the size takes somewhat less
# than twice as long and the scaled estimate is OPTIMISTIC there.  The bar's
# own 1.4x margin over the measured break-even absorbs that; what it must not
# absorb, and no longer has to, is the 7.3x and 7.1x errors above.
_POOL_PROMOTE_SIZE_RATIO = 2.0

# Pending promotion state, per COST CLASS (above).
#   _POOL_DEFERRED_NWORKERS  worker count, or None when nothing is pending
#   _POOL_DEFERRED_CLASS     fit-backend cost class, or None
#   _POOL_DEFERRED_SECONDS   deferred serial Newton time of the cheapest
#                            (PER POINT) sample recorded in this bucket
#   _POOL_DEFERRED_POINTS    that same sample's Newton point count.  The pair
#                            (_SECONDS, _POINTS) is ONE measurement, and
#                            _POINTS is also the size band's anchor.
#   _POOL_DEFERRED_COUNT     how many deferred inversions have been timed
_POOL_DEFERRED_NWORKERS = None
_POOL_DEFERRED_CLASS = None
_POOL_DEFERRED_SECONDS = 0.0
_POOL_DEFERRED_POINTS = 0
_POOL_DEFERRED_COUNT = 0


_PERSISTENT_POOL = None
_PERSISTENT_POOL_NWORKERS = None
# The payload key the parent BELIEVES every live worker of the current pool
# holds, or None.  Reset on every pool construction and on close_worker_pool,
# because a new pool means new (empty) workers.  Never load-bearing: a worker
# that disagrees raises NewtonPayloadNotResident and gets the blob.  See the
# PAYLOAD RESIDENCY block above _newton_invert_chunk.
_POOL_RESIDENT_PAYLOAD_KEY = None
# v5.30 (audit E-L2): the lock is built AT MODULE SCOPE.  It used to be a lazy
# ``None`` that ``_get_persistent_worker_pool`` created on first use -- the
# classic broken double-checked-locking shape, since the ``if
# _PERSISTENT_POOL_LOCK is None: ... = threading.Lock()`` guard is itself
# unsynchronised, so two threads racing the first parallel-Newton call could
# each build a lock, each acquire their own, and both construct a pool (the
# second overwriting the first, leaking its workers).  Building it here is
# free (a ``threading.Lock`` costs nothing at import) and makes the guard
# unnecessary.
_PERSISTENT_POOL_LOCK = threading.Lock()
# v5.30 (audit E-L1): one-shot flag for the atexit registration.  The handler
# used to be registered INSIDE the pool-construction block, i.e. once per pool
# creation: measured ``atexit._ncallbacks()`` growing 2 -> 8 across five
# creations (one extra ``close_worker_pool`` callback each time after the
# executor's own).  Every duplicate re-runs a full ``shutdown(wait=True)`` at
# interpreter exit.
_PERSISTENT_POOL_ATEXIT_REGISTERED = False


def _get_persistent_worker_pool(n_workers):
    """Return a (possibly newly-created) shared ProcessPoolExecutor.

    Reuses the same pool across calls when the requested ``n_workers``
    matches the cached pool's size.  Tears down and rebuilds when
    ``n_workers`` changes.  ``close_worker_pool`` is registered with
    ``atexit`` EXACTLY ONCE per process for clean shutdown on
    interpreter exit.
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_NWORKERS
    global _PERSISTENT_POOL_ATEXIT_REGISTERED, _POOL_RESIDENT_PAYLOAD_KEY
    with _PERSISTENT_POOL_LOCK:
        if _PERSISTENT_POOL is not None:
            if _PERSISTENT_POOL_NWORKERS == n_workers:
                return _PERSISTENT_POOL
            # n_workers changed: tear down the existing pool.
            try:
                _PERSISTENT_POOL.shutdown(wait=False)
            except (RuntimeError, OSError, BrokenPipeError):
                # Pool already torn down by atexit / signal handler,
                # or worker pipe broke under shutdown -- safe to
                # discard the reference.
                pass
            _PERSISTENT_POOL = None
        # v4.16.1 (audit M-2): force the ``spawn`` start method.  The
        # default on Linux is ``fork``, which inherits the parent's
        # FFT plan caches and threading state -- both of which are
        # unsafe to share between forked processes (pyFFTW's plan
        # cache holds module-private locks that the forked child
        # cannot release; numpy/MKL spin up a duplicate thread pool
        # that races with the parent).  ``spawn`` is portable across
        # Linux + macOS + Windows and matches the v4.16.0 CHANGELOG
        # claim that the library uses spawn (which was previously
        # only true of the multi-process storage tests, not the
        # library worker pool itself).
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        _spawn_ctx = _mp.get_context('spawn')
        _PERSISTENT_POOL = ProcessPoolExecutor(
            max_workers=int(n_workers),
            mp_context=_spawn_ctx,
            # FIX_PERF_ROUND2_2026_08_10 item 3: every worker starts with an
            # EMPTY payload registry.  See the PAYLOAD RESIDENCY block above
            # ``_newton_invert_chunk``.
            initializer=_newton_pool_init,
        )
        _PERSISTENT_POOL_NWORKERS = int(n_workers)
        # A new pool means new, empty workers: nothing is resident.
        _POOL_RESIDENT_PAYLOAD_KEY = None
        # Register the atexit handler exactly once per PROCESS, not once per
        # pool creation (v5.30, audit E-L1 -- the guard is the module-level
        # flag, checked under the same lock that guards the pool globals).
        if not _PERSISTENT_POOL_ATEXIT_REGISTERED:
            import atexit
            atexit.register(close_worker_pool)
            _PERSISTENT_POOL_ATEXIT_REGISTERED = True
    return _PERSISTENT_POOL


def _newton_cost_class(newton_fit) -> str:
    """Label the machinery that decides what one serial Newton step COSTS.

    Two inversions belong to the same class only if a wall-clock measurement
    of one is evidence about the other.  Measured at a FIXED 65 536 points,
    24 cores (see the block at ``_POOL_PROMOTE_SIZE_RATIO``):

        ``'polynomial+numba'``   0.048 s   (``@njit(parallel=True)`` Chebyshev
                                            evaluator -- already multicore)
        ``'spline'``             0.553 s   (``RectBivariateSpline.ev``, single
                                            threaded, does not release the GIL)
        ``'polynomial'``         0.95  s   (same Chebyshev evaluator, pure-xp
                                            fallback, numba unavailable)

    a 20x spread at identical size, against a ~0.22 s per-dispatch pool
    overhead -- so the backend, not the point count, is what decides whether
    the pool can win.  ``_NUMBA_AVAILABLE`` is read at CALL time rather than
    baked in at import, so a process that neutralises it lands in a different
    bucket instead of inheriting the fast path's measurement.

    An unrecognised ``newton_fit`` returns its own name, so a future backend
    starts in a bucket of its own rather than silently inheriting one.
    """
    fit = str(newton_fit)
    if fit == 'polynomial':
        return 'polynomial+numba' if _NUMBA_AVAILABLE else 'polynomial'
    return fit


def _pool_size_band_ok(n_points, anchor_points) -> bool:
    """True when ``n_points`` is close enough to the point count a recorded
    measurement was taken at for that measurement to be scaled onto it.

    Symmetric ratio bound: within ``_POOL_PROMOTE_SIZE_RATIO`` in EITHER
    direction.  Symmetry matters -- the shipped defect fired in the shrinking
    direction (a 116 281-point sample promoting 16 384-point calls) but the
    growing direction is the one that would over-promote a genuinely cheap
    call the next time a chain steps its ray_subsample down.
    """
    a = float(n_points)
    b = float(anchor_points)
    if not (a > 0.0 and b > 0.0):
        return False
    return max(a, b) <= _POOL_PROMOTE_SIZE_RATIO * min(a, b)


def _note_pool_deferral(n_workers, cost_class, n_points, seconds) -> None:
    """Record that a POOL-SIZED Newton inversion just ran serially: at what
    worker count, on which fit backend, over how many points, and how long it
    took.

    Called from ``_invert_newton_parallel`` on every call that cleared
    ``_POOL_MIN_PIXELS_WARM`` but not ``_POOL_MIN_PIXELS`` (i.e. a call a
    live pool would have served, but that was not worth a cold spawn on its
    own).  ``seconds`` is the measurement the cost gate needs: point count
    alone does not say whether the pool can beat the in-process path, since
    the default polynomial fit evaluates through a numba ``prange`` kernel
    that already uses every core.

    Keeps ONE sample per cost class -- the cheapest PER POINT seen in it --
    together with the point count it was taken at, plus how many samples the
    class has collected.  Cheapest-per-point is the steady-state estimate: the
    first inversion in a process carries one-time numba warm-up (measured
    0.637 s against a 0.048 s steady state), so a rule that trusted the first
    or the largest sample would promote on a compile it is not going to pay
    again.  Keeping the sample's own size alongside it is what lets
    :func:`_pool_reuse_is_likely` scale it onto a differently-sized call
    instead of pretending point count does not enter the cost.

    Recording a DIFFERENT cost class -- a different worker count, a different
    fit backend, or a point count outside ``_POOL_PROMOTE_SIZE_RATIO`` of the
    sample on record -- RESTARTS the measurement rather than stacking, so no
    bucket can ever promote on another bucket's evidence (V5).
    """
    global _POOL_DEFERRED_NWORKERS, _POOL_DEFERRED_SECONDS
    global _POOL_DEFERRED_COUNT, _POOL_DEFERRED_CLASS, _POOL_DEFERRED_POINTS
    n_workers = int(n_workers)
    cost_class = str(cost_class)
    n_points = int(n_points)
    seconds = float(seconds)
    with _PERSISTENT_POOL_LOCK:
        _same_bucket = (_POOL_DEFERRED_NWORKERS == n_workers
                        and _POOL_DEFERRED_CLASS == cost_class
                        and _pool_size_band_ok(n_points,
                                               _POOL_DEFERRED_POINTS))
        if not _same_bucket:
            _POOL_DEFERRED_NWORKERS = n_workers
            _POOL_DEFERRED_CLASS = cost_class
            _POOL_DEFERRED_SECONDS = seconds
            _POOL_DEFERRED_POINTS = n_points
            _POOL_DEFERRED_COUNT = 1
        else:
            # cross-multiplied so the comparison needs no division and cannot
            # divide by a zero point count
            if (seconds * _POOL_DEFERRED_POINTS
                    < _POOL_DEFERRED_SECONDS * n_points):
                _POOL_DEFERRED_SECONDS = seconds
                _POOL_DEFERRED_POINTS = n_points
            _POOL_DEFERRED_COUNT += 1


def _pool_reuse_is_likely(n_workers, cost_class, n_points) -> bool:
    """True when this process has already deferred pool-sized Newton
    inversions IN THIS CALL'S OWN COST CLASS, and those inversions were
    expensive enough that pooling one of this size would pay (see
    :func:`_note_pool_deferral`, :func:`_newton_cost_class` and
    ``_POOL_PROMOTE_MIN_SECONDS``).

    The "has been asked" half is the flag the two-tier threshold needs, as
    distinct from "is alive": ``_PERSISTENT_POOL is not None`` can only become
    true downstream of the gate that reads it, so on its own it can never
    promote a process whose calls all sit below the cold bar.  The COST half
    is what keeps the reachability fix from being a slowdown on the default
    path, where 65 536 Newton points cost 0.048 s against a ~0.22 s
    per-dispatch pool overhead.  The CLASS half (V5) is what stops one of
    those two facts being carried across to a call the measurement says
    nothing about.

    Needs ``_POOL_PROMOTE_MIN_SAMPLES`` measurements in the class before it
    will say yes, so the first (warm-up-inflated) inversion cannot decide on
    its own.
    """
    with _PERSISTENT_POOL_LOCK:
        if (_POOL_DEFERRED_NWORKERS is None
                or _POOL_DEFERRED_NWORKERS != int(n_workers)
                or _POOL_DEFERRED_CLASS != str(cost_class)
                or _POOL_DEFERRED_COUNT < _POOL_PROMOTE_MIN_SAMPLES):
            return False
        if not _pool_size_band_ok(int(n_points), _POOL_DEFERRED_POINTS):
            return False
        # Scale the recorded sample onto THIS call's size.  The band check
        # above bounds that extrapolation to a factor of
        # _POOL_PROMOTE_SIZE_RATIO, so it is a local model, never a 25x one.
        _est = (_POOL_DEFERRED_SECONDS * float(n_points)
                / float(_POOL_DEFERRED_POINTS))
        return bool(_est >= _POOL_PROMOTE_MIN_SECONDS)


def close_worker_pool() -> None:
    """Shut down the module-level worker pool used by
    :func:`apply_real_lens_traced`.

    Safe to call multiple times.  Called automatically at interpreter
    exit; only call explicitly when you want to free the workers
    before a long-running serial step (e.g. plotting, I/O).

    Also clears any pending second-call promotion (see
    ``_POOL_DEFERRED_NWORKERS``), so this is the one entry point that
    returns the process to a genuinely COLD state -- which is what the name
    promises, what the pool tests need between scenarios, and the right
    back-off after the pool-infrastructure fallback in
    ``_invert_newton_parallel`` closes a pool that just broke (a pool that
    failed should not be rebuilt on the very next 65k call).
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_NWORKERS
    global _POOL_DEFERRED_NWORKERS, _POOL_DEFERRED_SECONDS
    global _POOL_DEFERRED_COUNT, _POOL_DEFERRED_CLASS, _POOL_DEFERRED_POINTS
    global _POOL_RESIDENT_PAYLOAD_KEY
    # v5.30 (audit E-L2): take the SAME lock the constructor takes.  This
    # function mutates the pool globals, so running it concurrently with
    # ``_get_persistent_worker_pool`` could shut down a pool that had just
    # been handed to a caller, or clear ``_PERSISTENT_POOL`` between the
    # constructor's assignment and its return.
    with _PERSISTENT_POOL_LOCK:
        # The workers this belief was about are going away.
        _POOL_RESIDENT_PAYLOAD_KEY = None
        _POOL_DEFERRED_NWORKERS = None
        _POOL_DEFERRED_CLASS = None
        _POOL_DEFERRED_SECONDS = 0.0
        _POOL_DEFERRED_POINTS = 0
        _POOL_DEFERRED_COUNT = 0
        if _PERSISTENT_POOL is not None:
            try:
                _PERSISTENT_POOL.shutdown(wait=True)
            except (RuntimeError, OSError, BrokenPipeError):
                # Same shutdown-race tolerance as ``_get_pool``.
                pass
            _PERSISTENT_POOL = None
            _PERSISTENT_POOL_NWORKERS = None
    _reset_newton_pool_resource_state()


# ---------------------------------------------------------------------------
# Newton pool RESOURCE clamp (docs/audits/FIX_POOL_MEMORY_2026_08_06.md)
#
# ``_invert_newton_parallel`` used to submit ``n_workers`` chunks with NO memory
# accounting of any kind, while the fine grid those chunks come from is sized by
# ``carrier._memory_bounded_n_fine`` with a SINGLE-PROCESS cost model.  The other
# process pool in this library already has the clamp -- see
# ``carrier._multi_resolve_workers``, whose comment records the identical failure
# being fixed there -- and this is the same treatment for the Newton pool.
#
# MEASURED on this box (Ryzen 9 5950X, 137.4 GB, python 3.14.6, numpy 2.4.4,
# numba present) by running ``_newton_invert_chunk`` in a FRESH interpreter --
# exactly what a spawn worker is -- and reading ``psutil`` ``peak_pagefile``
# (Windows peak commit charge).  Fit-grid edge 531 (design 121's own), chunk
# swept 4x:
#
#     chunk points     peak commit        marginal
#        2 097 152     2.288 GB              --
#        4 194 304     2.849 GB       267.42 B/pt
#        8 388 608     3.970 GB       267.13 B/pt
#       16 777 216     6.211 GB       267.16 B/pt
#
# i.e. DEAD linear at 267.2 B per Newton point (0.1 % spread over a 8x range),
# on a 1.728 GB intercept.  267 B/point is ~33 float64 temporaries per point,
# which is what the Newton loop actually holds live (xa/ya/xw/yw, the six
# Jacobian entries, rx/ry/det/inv_det/dxe/dye/xa_new/ya_new/res, the numba
# kernel's u_flat/v_flat/f/fx/fy for each of two evaluators, plus the chunk).
#
# The intercept is a per-PROCESS import cost, not physics: bare python commits
# 0.012 GB, ``import numpy`` 0.831 GB, ``import lumenairy.elements._lens_traced``
# 1.65 GB, and the numba Chebyshev kernel's JIT adds ~0.07 GB on first call.
# Eight workers therefore commit ~14 GB before touching a single Newton point.
#
# Fit-grid axis, at a fixed 2 097 152-point chunk (the Chebyshev lstsq builds an
# (n_fit, 28) design matrix and SciPy/LAPACK copies it):
#
#     fit edge   fit points     peak commit    marginal
#         531       281 961       2.288 GB        --
#        1024     1 048 576       2.626 GB    440.3 B/pt
#        2048     4 194 304       5.274 GB    841.8 B/pt
#
# superlinear, so the shipped constant is the LARGER measured slope.
#
# The resulting model over-predicts every measured point by 5-15 %, which is the
# direction a resource clamp has to err in.
# ---------------------------------------------------------------------------

#: Per-worker peak commit at zero work: interpreter + numpy + lumenairy import
#: + the numba Chebyshev JIT.  MEASURED 1.728 GB intercept of the 4-point chunk
#: sweep above.
_NEWTON_WORKER_BASE_BYTES = 1.75e9
#: Per-worker peak commit per NEWTON POINT in the chunk it receives.  MEASURED
#: 267.2 B/pt, 0.1 % spread across a 8x chunk range.
_NEWTON_WORKER_BYTES_PER_POINT = 268.0
#: Per-worker peak commit per point of the pickled ray-fit grid the worker
#: re-fits (``n_launch**2``).  MEASURED 440-842 B/pt; the larger is shipped.
_NEWTON_WORKER_FIT_BYTES_PER_POINT = 850.0
#: Fraction of AVAILABLE memory the whole pool may claim.  Same 0.5 the fine
#: grid's own ceiling uses (``carrier._FINE_GRID_RAM_FRAC``), so the two clamps
#: that meet on the exact final leg speak the same language.
_NEWTON_POOL_RAM_FRAC = 0.5
#: ...on top of which this much is always left for the OS and for the PARENT's
#: own remaining growth (its fine grid, its band assembly).  Same RESERVE idiom
#: ``_multi_resolve_workers`` uses (``congruence_worker_min_free_gb``, 8 GB),
#: scaled to this pool: that one guards ~24 GB chain workers, these are ~2 GB
#: Newton workers, and an 8 GB reserve there would refuse a 2-worker Newton
#: dispatch on any box under ~20 GB -- a clamp that binds where nothing is
#: wrong is how a resource guard gets turned off.
_NEWTON_POOL_MIN_FREE_GB = 2.0

#: ``on_pool_memory`` vocabulary.  ``'warn'`` (default) announces a binding
#: memory cap; ``'silent'`` clamps identically and says nothing.  'ignore' /
#: 'off' are accepted ALIASES for 'silent' -- the same two-house-style
#: collision ``_NONCOL_ALIASES`` and ``_FDB_ALIASES`` resolve the same way
#: ('ignore' is what every ``on_*`` knob in ``propagators/carrier.py`` spells,
#: 'silent' is what this signature's own siblings spell).
#:
#: There is deliberately no ``'error'``: the clamp exists because the box
#: cannot hold the pool, and raising there would turn a run that COMPLETES with
#: a bit-identical answer into one that does not.  ``on_aperture_beam`` is the
#: sibling precedent for a two-value warn/silent knob.
_POOL_MEM_ALIASES = {'ignore': 'silent', 'off': 'silent'}
_POOL_MEM_ACTIONS = ('warn', 'silent')


def _pool_memory_policy(action) -> str:
    """Canonicalise an ``on_pool_memory`` value; raise on anything else.

    Kept beside the constants (and applied at the TOP of
    :func:`_newton_resolve_workers`, not inside the branch that warns) so a
    junk value cannot behave as 'warn' on the boxes where the clamp never binds
    and only surface on the one where it does -- which is precisely the
    ``on_undersample`` shape the D5 ``_KNOWN_UNGATED`` ledger records.
    """
    if isinstance(action, str):
        action = _POOL_MEM_ALIASES.get(action, action)
    if action not in _POOL_MEM_ACTIONS:
        raise ValueError(
            f"apply_real_lens_traced: on_pool_memory={action!r} is not a "
            f"valid policy.  Choose from 'warn' (default) or 'silent' "
            f"('ignore' / 'off' are accepted aliases for 'silent').  The knob "
            f"only decides whether a BINDING Newton-pool memory cap is "
            f"announced; the cap itself always applies, and the pool path is "
            f"bit-identical to serial either way.")
    return action


#: ``__main__`` guard verdicts, keyed by resolved script path (an AST parse per
#: path, not per Newton dispatch), and the one-shot warning ledger.
#:
#: Both are guarded by ``_MAIN_GUARD_LOCK`` and cleared together by
#: ``_reset_newton_pool_resource_state``.  The lock is not decorative:
#: ``apply_real_lens_traced`` runs on a ThreadPoolExecutor whenever
#: ``parallel_amp`` is on, so two threads can reach the resolver at once, and
#: the warning ledger's "have we said this yet" is a read-modify-write whose
#: whole contract is that it fires EXACTLY once.
_MAIN_GUARD_CACHE: Dict[str, bool] = {}
_MAIN_GUARD_WARNED: set = set()
_MAIN_GUARD_LOCK = threading.Lock()

#: Set once a Newton pool worker has REFUSED a chunk because it could not
#: provide the Chebyshev evaluator backend the payload pinned
#: (:class:`NewtonWorkerBackendUnavailable`).  That is a property of this
#: process's workers, not of one call, so every later dispatch in the process
#: goes straight to the (bit-identical) serial path instead of re-paying a
#: round trip to learn the same thing.  Cleared by ``close_worker_pool``, which
#: is the documented "return to a cold state" entry point -- and the right
#: re-arm, since it also tears the workers down.
#:
#: Guarded by ``_MAIN_GUARD_LOCK`` (the same lock as the warn-once ledger it
#: sits beside, and for the same reason: ``parallel_amp`` runs this path on a
#: ThreadPoolExecutor).
_POOL_BACKEND_REFUSED = False


def _note_pool_backend_refusal() -> bool:
    """Latch "the workers cannot honour the pinned evaluator backend" and
    report whether THIS is the first time it has been seen in this process.

    Module level rather than a closure so the read-modify-write happens under
    ``_MAIN_GUARD_LOCK`` in one place (``parallel_amp`` runs the dispatcher on
    a ThreadPoolExecutor, and ``if not flag: flag = True`` is a torn RMW whose
    whole contract is "say it exactly once"), and so the latch is testable
    without driving a real pool.
    """
    global _POOL_BACKEND_REFUSED
    with _MAIN_GUARD_LOCK:
        first = not _POOL_BACKEND_REFUSED
        _POOL_BACKEND_REFUSED = True
    return first


def _reset_newton_pool_resource_state() -> None:
    """Forget the cached ``__main__``-guard verdict and its one-shot warning.

    ``close_worker_pool`` is the documented "return this process to a cold
    state" entry point, so it has to clear this too -- otherwise a test (or a
    driver) that swaps ``sys.modules['__main__']`` would keep answering from
    the previous program's verdict, and the once-per-process warning could
    never fire again for a genuinely new one.

    Also clears the worker-backend refusal latch (``_POOL_BACKEND_REFUSED``):
    the workers that refused are gone by the time this returns, so the next
    dispatch is entitled to ask a fresh set.

    Also the module's ``register_cache_clearer`` entry point, so
    ``clear_all_registered_caches`` (and ``lumenairy_context(
    clear_caches_on_exit=True)``) reach it like any other cache.
    """
    global _POOL_BACKEND_REFUSED
    with _MAIN_GUARD_LOCK:
        _MAIN_GUARD_CACHE.clear()
        _MAIN_GUARD_WARNED.clear()
        _POOL_BACKEND_REFUSED = False


try:
    from .._cache_registry import (
        register_cache_clearer as _register_cache_clearer,
    )
    _register_cache_clearer('lens_traced_main_guard',
                            _reset_newton_pool_resource_state)
except ImportError:  # pragma: no cover - registry always present in-tree
    pass


def _is_main_guard_test(node) -> bool:
    """Is ``node`` the test of an ``if __name__ == '__main__':`` guard?

    Accepts every spelling that actually protects a spawn child: ``==`` either
    way round, ``in ('__main__', '__mp_main__')``, and the ``__mp_main__``
    name multiprocessing itself uses for the re-executed module.
    """
    import ast
    saw_name = False
    saw_main = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == '__name__':
            saw_name = True
        elif isinstance(sub, ast.Constant) and sub.value in ('__main__',
                                                             '__mp_main__'):
            saw_main = True
    return saw_name and saw_main


def _script_has_main_guard(path: str) -> bool:
    """Does the module at ``path`` have a TOP-LEVEL ``__name__`` guard?

    Top-level only, deliberately: a guard nested inside a function does not
    protect the module body, and the failure this predicate exists to catch is
    exactly "the module body re-runs".  A file that cannot be read or parsed
    returns ``True`` (= "cannot prove it is unguarded"), which preserves the
    historical pool behaviour rather than silently serialising a caller we know
    nothing about.
    """
    import ast
    with _MAIN_GUARD_LOCK:
        cached = _MAIN_GUARD_CACHE.get(path)
    if cached is not None:
        return cached
    ok = True
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, ValueError, SyntaxError, MemoryError, RecursionError):
        ok = True
    else:
        ok = any(isinstance(st, ast.If) and _is_main_guard_test(st.test)
                 for st in tree.body)
    # The parse happens OUTSIDE the lock (it is pure, and two threads racing it
    # recompute the same verdict rather than serialising on file I/O); only the
    # publish is guarded.
    with _MAIN_GUARD_LOCK:
        _MAIN_GUARD_CACHE[path] = ok
    return ok


def _spawn_reexecuted_main_script() -> Optional[str]:
    """Path of this process's ``__main__`` when a ``spawn`` worker would
    RE-EXECUTE its whole module body, else ``None``.

    This mirrors ``multiprocessing.spawn.get_preparation_data`` +
    ``_fixup_main_from_{name,path}`` rather than guessing:

      * ``__main__.__spec__.name`` set and equal to ``__main__`` or ending in
        ``.__main__`` (``python -m pytest``, ``pytest``) -- the child returns
        early, NOTHING is re-run;
      * ``__spec__.name`` set otherwise (``python -m yourscript``) -- the child
        ``runpy.run_module``s it;
      * no ``__spec__``, ``__file__`` set (``python yourscript.py``,
        ``runpy.run_path(..., run_name='__main__')``) -- the child
        ``runpy.run_path``s it;
      * frozen / embedded / no ``__file__`` / ``ipython`` -- nothing is re-run.

    A module the child re-runs is only a PROBLEM when its body is unguarded, so
    the guard check is applied here and a guarded script returns ``None``.
    """
    import os
    import sys
    main_module = sys.modules.get('__main__')
    if main_module is None:
        return None
    if getattr(sys, 'frozen', False):
        return None
    spec_name = getattr(getattr(main_module, '__spec__', None), 'name', None)
    if spec_name is not None:
        # _fixup_main_from_name's early return: a package's own __main__ is
        # already import-safe by construction.
        if spec_name == '__main__' or spec_name.endswith('.__main__'):
            return None
    path = getattr(main_module, '__file__', None)
    if not path:
        return None
    try:
        base = os.path.splitext(os.path.basename(path))[0]
    except (TypeError, ValueError):
        return None
    if base == 'ipython':          # multiprocessing's own carve-out
        return None
    if path.lower().endswith('.exe'):
        return None
    return None if _script_has_main_guard(path) else path


def _newton_worker_bytes(chunk_points, fit_points) -> float:
    """Peak commit ONE Newton pool worker takes for a chunk of
    ``chunk_points`` points against a ``fit_points``-point ray-fit grid.

    The measured law is at the top of this section.  Every term is a MEASURED
    slope or intercept, not an allowance.
    """
    return (_NEWTON_WORKER_BASE_BYTES
            + _NEWTON_WORKER_BYTES_PER_POINT * max(float(chunk_points), 0.0)
            + _NEWTON_WORKER_FIT_BYTES_PER_POINT * max(float(fit_points), 0.0))


def _newton_resolve_workers(requested, n_total, fit_points,
                            fn='apply_real_lens_traced', min_pool_points=None,
                            on_pool_memory='warn', _free_b=None) -> int:
    """Clamp the Newton pool's worker count to what this box can hold.

    Two rules, in order, and both of them can only ever LOWER the count -- the
    pool path is bit-identical to the serial path (see
    :func:`_newton_invert_chunk` and
    ``tests/unit/test_niche_newton_pool_both_fits.py``), so this is a pure
    resource decision that cannot move a number.

    1. **The caller's ``__main__`` must not be re-executed.**  ``spawn``
       workers re-run an unguarded ``__main__`` module body IN FULL before
       serving their chunk, so each worker pays the caller's WHOLE program, not
       a 1/K slice of a Newton grid.  MEASURED on design 121's acceptance
       (``validation/repro_traced_carrier_121/focus_scan_121.py``, which has no
       guard): 22.1 GB committed per worker x 8 = ~177 GB, taking a
       227.5 GB-commit box to 205.7 GB and 0.0 GB free, against an intrinsic
       1.9 GB/worker for the chunk it was actually given -- an 11.5x overhead
       that no chunk-sized model can see.  There is no worker count that makes
       that acceptable (the caller's side effects run K extra times too), so
       this rule returns SERIAL, not a smaller pool.
    2. **The pool must fit AVAILABLE memory**, at ``_NEWTON_POOL_RAM_FRAC`` of
       it with a ``_NEWTON_POOL_MIN_FREE_GB`` reserve for the parent's own
       remaining growth -- the parent is mid-``apply_real_lens_traced`` and its
       fine grid / band assembly is still ahead of it.

    Composes with, and is deliberately UPSTREAM of, the V5 cost-class gate in
    ``_invert_newton_parallel``: this bounds the CEILING (how many workers may
    ever run) while the cost gate decides whether to dispatch at all.  Running
    it first is what keeps the promotion evidence keyed by the worker count
    that would actually be used.

    ``min_pool_points`` is the other half of that composition.  Because the
    clamp runs first it also sees calls the SIZE gate is about to answer
    serially at any worker count, and announcing a cap on a dispatch that will
    never happen is noise that trains a reader to ignore the warning -- observed
    on ``test_fga.py``'s 384-point dispatcher probe, where a 24-CPU default and
    a 1125^2 ray-fit grid produced a perfectly correct 24 -> 20 clamp for a
    16-points-per-chunk call that then ran in-process.  Pass the pool's own
    lower size bar and the clamp still CLAMPS below it (the count feeds the
    gate) but stays SILENT.

    ``on_pool_memory`` is rule 2's POLICY knob, routed here from
    :func:`apply_real_lens_traced`'s signature exactly the way ``on_undersample``
    / ``on_aperture_beam`` route their guards (v5.32.3, FIX_CI_POOL).  ``'warn'``
    (default) emits the notice; ``'silent'`` clamps exactly the same way and
    says nothing -- it changes what is REPORTED, never what is run, because the
    clamp is the thing keeping the box alive.  Rule 1 is deliberately NOT
    routed through it: an unguarded ``__main__`` re-runs the caller's side
    effects K extra times, which is a correctness hazard rather than a resource
    notice, and there is no worker count at which it is acceptable.

    ``_free_b`` overrides the live memory read (tests only).
    """
    on_pool_memory = _pool_memory_policy(on_pool_memory)
    requested = int(requested)
    if requested <= 1:
        return 1
    unguarded = _spawn_reexecuted_main_script()
    if unguarded is not None:
        # Claim the "say it once" token under the lock: the whole contract of
        # this warning is that a 6-group chain does not emit six copies of a
        # paragraph, and ``if x not in s: s.add(x)`` is a torn RMW without it.
        with _MAIN_GUARD_LOCK:
            _first = unguarded not in _MAIN_GUARD_WARNED
            _MAIN_GUARD_WARNED.add(unguarded)
        if _first:
            import warnings
            warnings.warn(
                f"{fn}: running the Newton inversion SERIAL instead of on "
                f"{requested} workers, because this process's __main__ "
                f"({unguarded}) has no top-level `if __name__ == "
                f"'__main__':` guard.  multiprocessing's spawn workers "
                f"RE-EXECUTE the whole __main__ module body before serving "
                f"their chunk, so each worker would re-run your entire "
                f"program: MEASURED on design 121, 22.1 GB committed per "
                f"worker (~177 GB across 8) against 1.9 GB of actual chunk, "
                f"which took a 227.5 GB-commit box to 0.0 GB free.  Wrap the "
                f"top-level code of that file in the guard to get the pool "
                f"back -- the serial result is bit-identical either way, so "
                f"nothing but wall time changes here.",
                RuntimeWarning, stacklevel=3)
        return 1
    if _free_b is None:
        try:
            import psutil as _ps

            from ..memory import get_ram_budget
            _free_b = min(int(_ps.virtual_memory().available),
                          int(get_ram_budget()))
        except (ImportError, AttributeError, OSError, ValueError):
            # No memory oracle -> historical behaviour (the caller's
            # n_workers), exactly as _multi_resolve_workers does.
            return requested
    free_b = float(_free_b)
    if not np.isfinite(free_b):
        return requested
    per_worker_b = _newton_worker_bytes(
        float(n_total) / float(requested), fit_points)
    if per_worker_b <= 0:
        return requested
    budget_b = (_NEWTON_POOL_RAM_FRAC * free_b
                - _NEWTON_POOL_MIN_FREE_GB * 1e9)
    allowed = int(max(1, budget_b // per_worker_b))
    if allowed >= requested:
        return requested
    # Re-price at the count we will actually run: fewer workers means BIGGER
    # chunks, so the per-worker cost is not the one priced above.  Shrink until
    # the projection holds (bounded: at most ``requested`` steps).
    while allowed > 1:
        pw = _newton_worker_bytes(float(n_total) / float(allowed), fit_points)
        if pw * allowed <= budget_b:
            break
        allowed -= 1
    if min_pool_points is not None and n_total < int(min_pool_points):
        # Clamped, but this call cannot reach the pool at ANY worker count --
        # see ``min_pool_points`` above.  Return the count, say nothing.
        return allowed
    if on_pool_memory == 'silent':
        # The caller has acknowledged the clamp (``on_pool_memory='silent'``).
        # Same count, same run -- only the notice is suppressed.
        return allowed
    import warnings
    warnings.warn(
        f"{fn}: the Newton process pool asked for {requested} workers, which "
        f"projects to ~{requested * per_worker_b / 1e9:.1f} GB "
        f"({per_worker_b / 1e9:.2f} GB per worker at "
        f"{int(n_total) // int(requested)} Newton points/chunk and a "
        f"{int(fit_points)}-point ray-fit grid), but only "
        f"{free_b / 1e9:.1f} GB is available and this pool may claim "
        f"{_NEWTON_POOL_RAM_FRAC:.0%} of it less a "
        f"{_NEWTON_POOL_MIN_FREE_GB:.0f} GB reserve = "
        f"{budget_b / 1e9:.1f} GB; running {allowed} worker(s) instead.  The "
        f"result is UNCHANGED (the pool path is bit-identical to serial); "
        f"only the wall time is.  Lower n_workers, raise the RAM budget "
        f"(lumenairy.set_max_ram), or free memory to use more.",
        RuntimeWarning, stacklevel=3)
    return allowed


# Sibling-module imports (created separately in this package) ----------------

# Typing: the Maslov section (merged in 3.2.2 from the former
# lens_maslov.py) uses Any / Dict / Optional / Tuple in function
# annotations.
# The Maslov section uses ``time`` for internal progress timing.


# ---------------------------------------------------------------------------
# Optional Numba JIT for the polynomial-evaluator inner loop.
#
# The hot path of _Cheb2DEvaluator.ev_value_and_grad is a doubly-nested
# reduction over (basis_term, output_sample).  Plain NumPy executes it
# as a chain of broadcast multiplies and sum-reductions with a handful
# of allocated temporaries; a @njit kernel collapses that to a single
# tight loop with zero temporaries and thread-parallel output rows.
#
# Guarded import -- fallback to pure-xp path (which is fine on NumPy and
# REQUIRED on CuPy) when numba isn't installed.  The kernel is compiled LAZILY
# on first call via _get_cheb2d_val_grad_numba() so ``import lumenairy`` never
# pays the numba import / compile cost (audit P2-D).
# ---------------------------------------------------------------------------


def _get_cheb2d_val_grad_numba():
    """Compile (once, on first call) and return the Chebyshev value+gradient
    numba kernel, or ``None`` if numba is unavailable."""
    if "cheb2d" in _NUMBA_KERNELS:
        return _NUMBA_KERNELS["cheb2d"]
    if not _load_numba():
        _NUMBA_KERNELS["cheb2d"] = None
        return None

    @_njit(cache=True, parallel=True, fastmath=True)
    def _cheb2d_val_grad_numba(coeffs, K1, K2, u_flat, v_flat, max_order):
        """Combined Chebyshev value + gradient via in-place recurrence.

        Computes f(u, v), df/du, df/dv at every (u_flat[i], v_flat[i])
        sample in parallel.  Chebyshev T and U (second kind) values are
        generated by 3-term recurrence on the stack per sample -- no
        Vandermonde matrices are materialised.  This implements the
        Clenshaw-style "#3" optimisation: O(order) stack work per
        sample instead of an O(N x order) materialised Vandermonde.

        Parameters
        ----------
        coeffs : (M,) float64   -- polynomial coefficients in total-degree order
        K1, K2 : (M,) int64     -- multi-indices (kx, ky) for each term
        u_flat, v_flat : (N,) float64 -- normalised sample coords in [-1, 1]
        max_order : int         -- maximum individual Chebyshev order

        Returns
        -------
        f, fx_u, fy_v : three (N,) float64 arrays: value and du/dv-partials
        in normalised coordinates.  Caller applies chain rule for
        physical derivatives.
        """
        N = u_flat.shape[0]
        M = coeffs.shape[0]
        f = np.zeros(N)
        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in _prange(N):
            u = u_flat[i]
            v = v_flat[i]

            # T_n(u), T_n(v): first kind, by 3-term recurrence
            # T_0 = 1, T_1 = u, T_{n+1} = 2u T_n - T_{n-1}
            Tu = np.empty(max_order + 1)
            Tv = np.empty(max_order + 1)
            Tu[0] = 1.0
            Tv[0] = 1.0
            if max_order >= 1:
                Tu[1] = u
                Tv[1] = v
            for n in range(2, max_order + 1):
                Tu[n] = 2.0 * u * Tu[n - 1] - Tu[n - 2]
                Tv[n] = 2.0 * v * Tv[n - 1] - Tv[n - 2]

            # T'_n(u) = n * U_{n-1}(u); U_0 = 1, U_1 = 2u, U_{n+1} = 2u U_n - U_{n-1}
            # We store dTu[n] = T'_n(u) directly for n = 0..max_order
            dTu = np.zeros(max_order + 1)
            dTv = np.zeros(max_order + 1)
            if max_order >= 1:
                dTu[1] = 1.0          # T'_1 = 1 * U_0 = 1
                dTv[1] = 1.0
                if max_order >= 2:
                    U_prev_u = 1.0    # U_0
                    U_u = 2.0 * u     # U_1
                    U_prev_v = 1.0
                    U_v = 2.0 * v
                    dTu[2] = 2.0 * U_u    # T'_2 = 2 * U_1
                    dTv[2] = 2.0 * U_v
                    for n in range(3, max_order + 1):
                        U_next_u = 2.0 * u * U_u - U_prev_u
                        U_next_v = 2.0 * v * U_v - U_prev_v
                        U_prev_u = U_u
                        U_u = U_next_u
                        U_prev_v = U_v
                        U_v = U_next_v
                        dTu[n] = n * U_u
                        dTv[n] = n * U_v

            # Accumulate coefficient-weighted sum over multi-indices
            acc_f = 0.0
            acc_fx = 0.0
            acc_fy = 0.0
            for m in range(M):
                kx = K1[m]
                ky = K2[m]
                c = coeffs[m]
                tu = Tu[kx]
                tv = Tv[ky]
                acc_f  += c * tu * tv
                acc_fx += c * dTu[kx] * tv
                acc_fy += c * tu * dTv[ky]
            f[i] = acc_f
            fx[i] = acc_fx
            fy[i] = acc_fy
        return f, fx, fy

    _NUMBA_KERNELS["cheb2d"] = _cheb2d_val_grad_numba
    return _cheb2d_val_grad_numba


# ---------------------------------------------------------------------------
# THE CHEBYSHEV EVALUATOR'S BACKEND IS A RESOLVED DECISION, AND IT MUST TRAVEL
# (docs/audits/FIX_CI_POOL_2026_08_06.md; closes FIX_POOL_MEMORY sec 8.1)
#
# ``_Cheb2DEvaluator.ev_value_and_grad`` has two implementations of ONE
# formula: an ``@njit(parallel=True, fastmath=True)`` Chebyshev recurrence and a
# pure-xp Vandermonde contraction.  They agree to ~1e-16 relative, not bit for
# bit -- different summation order over the 28 basis terms.  A Newton pool
# worker rebuilds the evaluator in a FRESH interpreter, so whichever branch IT
# resolves is the branch the pooled OPL comes from, and the pickled payload
# carried ``newton_fit`` / ``fit_poly_order`` / ``fit_weights`` /
# ``newton_max_iters`` but NO backend flag.  A parent whose resolution differs
# from its workers' therefore returned a DIFFERENT answer from the serial path:
# MEASURED here, N=1024 / ray_subsample=2, both directions of the split,
# max|delta| 5.167e-14; on CI (ubuntu, py3.10) 1.358e-11.
#
# This is the same class of gap as audit E-H2's ``newton_max_iters``: a decision
# the parent RESOLVES and the worker re-derives from its own process state.  The
# parent's resolution is genuinely process state and not merely environment --
# ``_NUMBA_KERNELS['cheb2d']`` caches a permanent ``None`` once ``_load_numba``
# has failed once -- so a worker CANNOT infer it, and the flag has to be pinned.
# ---------------------------------------------------------------------------

def _resolved_cheb_backend(newton_fit='polynomial'):
    """Name the Chebyshev evaluator branch THIS process actually takes.

    ``'numba'`` iff numba is importable here AND the kernel compiles/loads;
    ``'numpy'`` otherwise.  Resolves through the same getter
    ``ev_value_and_grad`` uses, so it reports the branch that WILL run rather
    than the branch the environment suggests -- the two differ in a process
    whose first ``_load_numba()`` failed (the ``None`` is cached forever).

    Returns ``None`` for any fit that has no Chebyshev evaluator.
    ``newton_fit='spline'`` rebuilds a ``RectBivariateSpline`` in the worker,
    which has no backend split to pin -- and asking anyway would compile the
    numba kernel in a process that is never going to evaluate it, purely to
    describe a payload.
    """
    if str(newton_fit) != 'polynomial':
        return None
    if not _NUMBA_AVAILABLE:
        return 'numpy'
    return 'numba' if _get_cheb2d_val_grad_numba() is not None else 'numpy'


def _validated_cheb_backend(backend):
    """Return ``backend`` if it names a real Chebyshev evaluation branch.

    ``None`` means "resolve it from this process's numba availability" (the
    historical behaviour).  Anything else is refused rather than silently
    treated as ``None``, because a typo'd pin that quietly meant "auto" would
    reintroduce exactly the split the pin exists to close.
    """
    if backend is not None and backend not in ('numba', 'numpy'):
        raise ValueError(
            f"_Cheb2DEvaluator: backend={backend!r} is not a valid "
            f"evaluation backend.  Choose 'numba', 'numpy', or None "
            f"(resolve from this process's numba availability).")
    return backend


# ---------------------------------------------------------------------------
# ...AND SO IS THE FIT ITSELF (docs/audits/FIX_POOL_REBUILD_2026_08_08.md)
#
# Pinning ``cheb_backend`` made the two sides EVALUATE in the same order.  It
# did not make them evaluate the same POLYNOMIAL.  ``_newton_invert_chunk``
# still called ``_Cheb2DEvaluator(...)``, i.e. it re-ran the least-squares fit
# from the pickled grids, and that fit is a BLAS reduction:
# ``_solve_lstsq_thread_safe`` forms ``G = A^T A`` and ``A^T b`` over a
# ~78 000-row design matrix, which OpenBLAS reduces in a thread-count-dependent
# order.  Two processes on the same data therefore recover coefficients that
# differ in the last bits whenever their BLAS regimes differ -- and a worker's
# regime is NOT guaranteed to be its parent's: ``threadpoolctl``'s cap is
# PROCESS-GLOBAL on OpenBLAS, so a long-lived pytest parent that has been
# through a capped section runs at a different width from a freshly spawned
# worker, which always starts at the environment default.
#
# MEASURED here (Windows 11, py3.14.6, numpy 2.4.4 / scipy-openblas 0.3.31,
# 24 cores), the traced doublet's own 77 841-point OPL fit, coefficients from
# ``A^T A`` at BLAS width W against width 24:
#
#     W = 1     NOT bit-identical, max|dc| 4.596e-15
#     W = 2     NOT bit-identical, max|dc| 4.243e-15
#     W = 4     NOT bit-identical, max|dc| 5.484e-15
#     W = 8     NOT bit-identical, max|dc| 8.766e-16
#     same W    bit-identical, 30/30 rebuilds, and parent == spawned worker
#
# and end-to-end on this file's own contract shape (N=1024, ray_subsample=2,
# 262 144 Newton points, 4 chunks, serial vs pool with the WORKER body under a
# BLAS cap the parent does not have):
#
#     worker BLAS width 1     max|dfield| 1.830e-12
#     worker BLAS width 2     max|dfield| 1.062e-11
#     worker BLAS width 4     max|dfield| 1.370e-11   <- CI: 1.341e-11/1.358e-11
#     worker BLAS width 8     max|dfield| 1.198e-12
#     real spawn pool, same regime as the parent   0.000e+00  (why it passes here)
#
# The amplification is the Newton loop's ``res < tol`` threshold: a coefficient
# perturbation of 1e-15 relative moves a handful of the 262 144 points across
# the convergence line, which changes their iteration count and so their OPL by
# ~1e-18 m; at 1.31 um that is 2*pi/lambda * 1e-18 ~ 1e-11 of a unit-amplitude
# field.  The polynomial fit is the ONLY BLAS in that worker, which is why
# ``[spline]`` -- whose worker rebuilds a single-threaded FITPACK
# ``RectBivariateSpline`` -- passed in every lane it ran in.
#
# The remedy is not to make the least-squares solve reproducible across BLAS
# regimes (it is a library's reduction order, not ours to fix); it is to stop
# solving it twice.  The parent SHIPS its built fit -- 28 coefficients and two
# 28-entry index vectors per evaluator, ~700 bytes against the ~1.9 MB of grids
# the payload already carries -- and the worker EVALUATES it.  Bit-identity
# then holds by construction, for every BLAS regime, present and future.
# ---------------------------------------------------------------------------

def _cheb_fit_state(ev):
    """Picklable state of an ALREADY-BUILT :class:`_Cheb2DEvaluator`.

    Everything the evaluator needs to evaluate, and nothing it needed to FIT:
    no design matrix, no sample grid, no weights.  Reconstructed by
    :meth:`_Cheb2DEvaluator.from_state`.

    ``backend`` is deliberately absent -- it is a fact about the process that
    will evaluate, not about the fit, and it travels separately as the
    payload's ``cheb_backend`` (see :func:`_resolved_cheb_backend`).
    """
    _xp = getattr(ev, 'xp', np)

    def _host(a):
        # CuPy evaluators are unreachable from the pool (the GPU path returns
        # in-process before dispatch), but three 28-element arrays are cheap
        # insurance against that ever changing silently.
        return np.asarray(_xp.asnumpy(a) if _xp is not np else a)

    return {
        'order': int(ev.order),
        'mi': [(int(kx), int(ky)) for (kx, ky) in ev._mi],
        'coeffs': np.asarray(_host(ev.coeffs), dtype=np.float64),
        'K1': np.asarray(_host(ev._K1), dtype=np.int64),
        'K2': np.asarray(_host(ev._K2), dtype=np.int64),
        'xmin': float(ev.xmin), 'xmax': float(ev.xmax),
        'ymin': float(ev.ymin), 'ymax': float(ev.ymax),
    }


def _cheb_fit_payload(Sx, Sy, So, newton_fit='polynomial'):
    """The parent's three BUILT Chebyshev fits, ready to pickle.

    Returns ``None`` for any fit that has no Chebyshev evaluator -- i.e. for
    ``newton_fit='spline'``, whose worker rebuilds a ``RectBivariateSpline``
    through FITPACK.  That rebuild is single-threaded and takes no BLAS path,
    so it has no thread-regime axis to close; ``[spline]`` passed on CI in
    every lane it ran in while ``[polynomial]`` failed in all four, which is
    the measurement behind leaving it alone.
    """
    if str(newton_fit) != 'polynomial':
        return None
    return {'x_out': _cheb_fit_state(Sx),
            'y_out': _cheb_fit_state(Sy),
            'opl': _cheb_fit_state(So)}


class NewtonWorkerBackendUnavailable(RuntimeError):
    """A Newton pool worker cannot provide the evaluator backend the payload
    pins, so it refuses the chunk instead of answering in a different
    floating-point order.

    Raised INSIDE ``_newton_invert_chunk`` and handled by
    ``_invert_newton_parallel``, which falls back to the (bit-identical to the
    parent) in-process serial path and says so once.  A worker cannot conjure
    the parent's backend, and quietly substituting the other one is exactly the
    silent-wrong outcome this pin exists to remove.

    Derives from ``RuntimeError`` on purpose: the pool-infrastructure ``except``
    in ``_invert_newton_parallel`` already catches that and falls back to
    serial, so a refactor that drops the specific handler loses the diagnostic
    but keeps the SAFE behaviour rather than propagating a hard failure.
    """


def _get_array_module(arr):
    """Return the array namespace (numpy or cupy) for ``arr``.

    Enables array-API polymorphism: code that uses only namespace-
    agnostic operations (xp.asarray, xp.sum, xp.meshgrid, ...) runs
    unchanged on NumPy or CuPy arrays.  Gracefully degrades to NumPy
    when CuPy isn't installed.
    """
    try:
        import cupy as _cp
        if isinstance(arr, _cp.ndarray):
            return _cp
    except ImportError:
        pass
    return np


#: niche C13 (2026-08-03): SCREEN the normal-equations least-squares solve for
#: a numerically singular Gram matrix, and where it is singular, RE-SOLVE by a
#: backward-stable QR and keep whichever answer measurably fits the data
#: better.
#:
#: ``False`` restores the pre-C13 solver EXACTLY -- :func:`_solve_lstsq_thread_safe`
#: then returns the Cholesky/LU answer unconditionally, bit for bit -- and it
#: is the fail-before for everything below.
#:
#: THE DEFECT.  :func:`_solve_lstsq_thread_safe`'s own docstring asserted that
#: ``A`` "is a well-conditioned normalised tensor-Chebyshev / monomial
#: Vandermonde (~1.5x oversampled), so squaring the condition number in ``G``
#: is safe".  That is true of the CONCENTRIC, unweighted fits it was written
#: for and FALSE of the weighted ones D1/D7 introduced:
#: ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` = 1e-8 splits the rows into two scales ~1
#: and ~1e-4, the in-disc rows alone do not determine a total-degree-10
#: (66-term) basis over the whole launch box, and the directions they leave
#: over are pinned only through the down-weighted rows.  Measured on niche D4's
#: own chain, ``cond(A)`` = 1.4e10, so ``cond(G)`` ~ 1.9e20 -- past float64,
#: and the Cholesky answer is an arbitrary draw from the numerical null space
#: rather than the least-squares solution.  On that chain the draw's fit
#: residual runs 15-23x above the attainable minimum on one build and 1.05x on
#: the other, from the SAME source at the SAME degree.
#:
#: WHAT IT COST.  ``test_niche_d4_dgrating::test_matches_the_manual_hand_split``
#: read 5.93e-07 on one build and 8.80e-02 on the other, RUN ALONE, in the
#: shipped configuration -- a 148,000x route disagreement decided by BLAS
#: rounding.  The exit field on the losing build is speckled at the pixel scale
#: (roughness ``max|lap E| / peak`` 0.30 against 0.035), which is numerical
#: noise, not an approximation error.  It was attributed to niche C10's degree
#: 6 because degree is what moved it, and degree 6 IS the stimulus -- it
#: perturbs the OPL samples enough to move which null-space draw each build
#: takes -- but the degree is sound and the solve was not.  See
#: ``docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md``.
#:
#: WHY THE DECISION IS A RESIDUAL COMPARISON and not a certificate on one
#: answer.  The obvious test -- the least-squares stationarity residual
#: ``A^T (b - A x)``, which is exactly zero at the solution -- was built,
#: measured, and REJECTED: on these matrices it is computed by cancellation at
#: the ``||A||^2 ||x||`` scale, so its float64 value is rounding noise.  It
#: scores the WRONG answer better than the right one (the QR solution reads
#: 3e-09 where the null-space draw reads 8e-11, with 15x the fit quality).
#: ``||b - A x||`` has no such problem here -- the residual is only ~3 orders
#: below ``||b||`` -- so the two candidates are compared on the quantity the
#: fit is actually defined by.
LSTSQ_CONDITIONING_STEPDOWN = True

#: SCREEN: reciprocal condition number of the diagonally EQUILIBRATED Gram
#: matrix below which the second solve is computed at all.
#:
#: This screen exists for COST, not for correctness -- the residual comparison
#: below decides.  Its only requirement is that it must not skip a solve whose
#: two candidates could differ by more than ``_LSTSQ_RESID_MARGIN``, and it has
#: two decades of room: the normal equations lose ~``cond(G) * eps``, so at the
#: screen (``cond`` = 1e8) the most a skipped solve can be off is ~1e-8, a
#: hundredfold inside the margin.  A build that screened one way and another
#: build the other therefore CANNOT change the answer at the boundary.
#:
#: Measured over all 54 solves niche D4's chain makes: the fits it skips read
#: ``cond`` 1.0e2, and the ones it screens in read 1.7e15 / 5.5e15 / 1.5e16 /
#: 2.8e18 -- thirteen orders of separation, with the screen in the middle of
#: it.
#:
#: It is NOT true that design 121's production route escapes this.  Measured
#: (``c13_solver_census.py focus_scan_121.py``): 32 solves, **24 screen in and
#: 6 REROUTE**, with normal-equations-to-QR fit-residual ratios up to **1072x**
#: on the 28-term coordinate fits.  The production ACCEPTANCE is nevertheless
#: unchanged to every printed digit including the peak (S6.3 of the audit) --
#: the readout re-traces the final leg on a fine grid, where the rerouted fits
#: are not what limits it.  "Unchanged metrics" and "untouched arithmetic" are
#: different claims and only the first one is made.
_LSTSQ_GRAM_RCOND_MIN = 1e-8

#: The QR answer replaces the normal-equations answer only if it fits the data
#: better by MORE than this relative margin.  Ties go to the shipped path, so
#: every solve the normal equations already got right returns its historical
#: bits.
#:
#: The margin is a build-independence device.  ``||b - A x||`` is itself
#: computed to ~1e-13 relative on these matrices, and two builds' copies of the
#: SAME candidate differ by about that much, so any margin far above 1e-13
#: makes the winner build-independent; any margin far below the smallest real
#: gap keeps the fix.  The smallest real gap measured anywhere in this campaign
#: is 4.8e-02 (design-121-sized OPL fit, Windows build), so 1e-6 sits ~7 orders
#: above the noise and ~5 below the signal.
_LSTSQ_RESID_MARGIN = 1e-6


def _gram_rcond(G):
    """Reciprocal 2-norm condition number of ``G`` after DIAGONAL
    EQUILIBRATION (``G -> D G D``, ``D = diag(1/sqrt(G_ii))``).

    Equilibration is what makes this a statement about the fit rather than
    about the units the columns happen to carry: van der Sluis's bound says the
    equilibrated condition number is within ``sqrt(m)`` of the best any column
    scaling could achieve.  ``G`` is the tiny ``M x M`` Gram (``M`` ~ 15-120),
    so the symmetric eigensolve costs microseconds.

    Returns ``0.0`` for a Gram that is not usable at all (non-finite, or an
    eigensolve that will not converge), which routes the caller to the stable
    solver -- the safe direction.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1] or not np.all(np.isfinite(G)):
        return 0.0
    d = np.sqrt(np.abs(np.diag(G)))
    d = np.where(d > 0.0, d, 1.0)
    Gs = G / np.outer(d, d)
    try:
        ev = np.linalg.eigvalsh(Gs)
    except np.linalg.LinAlgError:
        return 0.0
    hi = float(np.max(np.abs(ev)))
    lo = float(np.min(ev))          # SIGNED: a negative eigenvalue means the
    if not (hi > 0.0) or lo <= 0.0:  # Gram lost positive-definiteness outright
        return 0.0
    return lo / hi


def _solve_lstsq_qr(A, b):
    """Backward-stable least squares by Householder QR of ``[A | b]``.

    ``R = qr([A | b])`` has ``R[:m, :m]`` = the R factor of ``A`` and
    ``R[:m, m:]`` = ``(Q^T b)[:m]``, so ONE ``geqrf`` and one triangular solve
    give the answer with no ``Q`` materialised and no second pass over ``A``.
    Measured at the traced fits' worst shape (141471 x 66), best of three,
    RELATIVE to the normal equations because the box was loaded and the ratio
    is what transfers: this route **15x** (Windows) / **19x** (Linux), ``qr``
    with an explicit ``Q`` 32x / 41x, ``gelsd`` 12x / 19x.  Absolute, on that
    run: 0.069 s for the normal equations against 1.06 / 1.31 s here.  The
    condition number is NOT squared, which is the entire point --
    ``cond(A)`` = 1.4e10 is unremarkable for float64 QR and fatal for a Gram
    matrix.  This runs only where the screen fires, and the screen is what
    keeps the cost off the fits that do not need it.

    ``geqrf`` -- NOT ``gelsd``.  B7 banned ``np.linalg.lstsq`` from this module
    because ``gelsd``'s divide-and-conquer SVD spawns an OpenBLAS OpenMP pool
    that deadlocks nested inside JAX's; Householder QR is a blocked BLAS-3
    factorisation and takes no such path.  ``lstsq`` survives only as the
    last-resort branch for an ``A`` whose ``R`` comes out exactly singular.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = A.shape[1]
    B = b.reshape(b.shape[0], -1)
    try:
        if A.shape[0] < m:
            # UNDER-determined: ``R`` has no ``m x m`` leading block and the
            # answer wanted is the minimum-norm one, which is gelsd's.  The
            # traced fits are guarded against this upstream (every caller
            # enforces a samples-per-term floor); this is belt and braces.
            raise ValueError('under-determined')
        from scipy.linalg import qr as _qr
        from scipy.linalg import solve_triangular as _solve_tri
        M = np.empty((A.shape[0], m + B.shape[1]), dtype=np.float64, order='F')
        M[:, :m] = A
        M[:, m:] = B
        R = _qr(M, mode='r', check_finite=False)[0]
        x = _solve_tri(R[:m, :m], R[:m, m:], check_finite=False)
        if np.all(np.isfinite(x)):
            return x.reshape((m,) + b.shape[1:])
    except (ImportError, ValueError, np.linalg.LinAlgError):
        pass
    # exactly rank-deficient R (or no scipy): the minimum-norm solution is the
    # only defensible answer, and this branch is rare enough to pay for gelsd.
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def _lstsq_residual(A, b, x):
    """``||b - A x||_F``, the quantity a least-squares fit is defined to
    minimise, and the only one of the two candidates' scores that survives
    float64 on these matrices (see ``LSTSQ_CONDITIONING_STEPDOWN``)."""
    return float(np.linalg.norm(np.asarray(b) - np.asarray(A) @ np.asarray(x)))


def _solve_lstsq_thread_safe(A, b):
    """Least-squares solve ``A @ x ~= b`` (overdetermined, single or multi RHS)
    via the NORMAL EQUATIONS -- ``G x = A^T b`` with ``G = A^T A`` -- Cholesky
    then LU, with a QR re-solve (:func:`_solve_lstsq_qr`) wherever ``G`` comes
    out numerically singular and the QR answer measurably fits better.

    B7 (jax x OpenBLAS mitigation): the traced Chebyshev/coordinate fits used
    ``np.linalg.lstsq`` (LAPACK ``gelsd``, a divide-and-conquer SVD).  In one
    process alongside JAX, ``gelsd``'s multi-threaded OpenBLAS OpenMP pool nests
    inside JAX's OpenMP runtime and DEADLOCKS on the first large fit -- the CI
    worked around it with ``OMP_NUM_THREADS=1`` pins that cannot be relied on
    outside CI.  The normal equations reduce the factorisation to the tiny
    ``M x M`` Gram matrix (``M`` = number of fit terms, ~28-70), which stays
    below OpenBLAS's threading threshold and never takes the ``gelsd`` SVD path,
    so the deadlock cannot recur.  Neither does the re-solve: it is ``geqrf``.

    CONDITIONING (niche C13, 2026-08-03).  This function used to assert that
    ``A`` "is a well-conditioned normalised tensor-Chebyshev / monomial
    Vandermonde (~1.5x oversampled), so squaring the condition number in ``G``
    is safe".  That holds for the concentric unweighted fits and is FALSE for
    the weighted decentred ones, where ``cond(A)`` = 1.4e10 was measured and
    ``G`` is therefore numerically singular.  Rather than assume either way,
    the Gram is now SCREENED and, where it is singular, both answers are scored
    on the data.  A solve that passes the screen -- or one whose two candidates
    tie -- returns the identical bits it returned before, which is what keeps
    the byte-identity contracts of niches C1/C6/C8/C9 intact.
    See ``LSTSQ_CONDITIONING_STEPDOWN``.

    Returns ``x`` with the same trailing shape as ``b`` (1-D for a single RHS).
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    G = A.T @ A
    rhs = A.T @ b
    x = None
    try:
        from scipy.linalg import cho_factor, cho_solve
        x = cho_solve(cho_factor(G, check_finite=False), rhs,
                      check_finite=False)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # scipy absent, or G not positive-definite (rank-deficient fit).
        try:
            x = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            x = None
    if x is None or not np.all(np.isfinite(x)):
        return _solve_lstsq_qr(A, b)
    if not LSTSQ_CONDITIONING_STEPDOWN:
        return x
    if _gram_rcond(G) >= _LSTSQ_GRAM_RCOND_MIN:
        return x
    x_qr = _solve_lstsq_qr(A, b)
    if not np.all(np.isfinite(x_qr)):
        return x
    r_ne = _lstsq_residual(A, b, x)
    r_qr = _lstsq_residual(A, b, x_qr)
    if r_qr < (1.0 - _LSTSQ_RESID_MARGIN) * r_ne:
        return x_qr
    return x


class _Cheb2DEvaluator:
    """2-D Chebyshev tensor-product polynomial fit with an API compatible
    with a SciPy ``RectBivariateSpline`` for the subset used by
    :func:`apply_real_lens_traced` -- specifically the ``ev(x, y)``,
    ``ev(x, y, dx=1)``, and ``ev(x, y, dy=1)`` methods.

    This is the polynomial equivalent of the default spline interpolation
    used by ``apply_real_lens_traced``'s Newton inversion, enabled when
    ``newton_fit='polynomial'``.  For smooth refractive lenses where the
    entrance->exit coordinate map and the OPL are essentially polynomials
    of total degree up to 6 (all Seidel + higher-order aberrations of
    reasonable orders), this is both faster (closed-form analytic
    derivatives, no Fortran spline calls) and more accurate (no cubic
    truncation error on the coarse grid).

    Architecture
    ------------
    * **Array-API polymorphic**: the ``xp`` constructor kwarg selects
      the array backend (default :mod:`numpy`).  Pass ``xp=cupy`` to
      run the fit and evaluation on the GPU with zero code changes
      here -- every internal operation uses ``self.xp``'s namespace.
    * **Combined value + gradient** (``ev_value_and_grad``): returns
      ``(f, df/dx, df/dy)`` in one shared-basis pass, avoiding the 3x
      redundant Vandermonde builds that the separate ``.ev(dx=1)`` and
      ``.ev(dy=1)`` calls would do.
    * **Optional Numba JIT fastpath**: on the NumPy backend, if
      :mod:`numba` is importable, the combined evaluation drops into a
      ``@njit(parallel=True, fastmath=True)`` kernel that runs the
      Chebyshev recurrence inline per sample (no Vandermonde
      materialised).  Typical 3-10x speedup over the pure-NumPy path.
      Silently skipped on the CuPy backend -- the pure-xp fallback
      runs on GPU instead.

    GPU note
    --------
    To use this class on GPU with CuPy::

        import cupy as cp
        ev = _Cheb2DEvaluator(xs_in_cp, ys_in_cp, values_cp,
                              order=6, xp=cp)
        # Later:
        f, fx, fy = ev.ev_value_and_grad(xa_cp, ya_cp)

    All arrays (inputs, outputs, internal state) stay on the GPU;
    there is no implicit host-device copy.  The Newton loop in
    :func:`apply_real_lens_traced` is unchanged as long as ``xa, ya``
    are CuPy arrays.  A future ``use_gpu=True`` kwarg could dispatch
    this automatically.
    """

    __slots__ = ('order', 'coeffs', 'xmin', 'xmax', 'ymin', 'ymax',
                 '_mi', '_K1', '_K2', 'xp', 'backend')

    def __init__(self, xs_in, ys_in, values, order=6, xp=None, weights=None,
                 backend=None):
        if xp is None:
            xp = _get_array_module(values)
        self.xp = xp
        # v5.32.3 (FIX_CI_POOL): PINNED evaluation backend, or None = "resolve
        # it from this process's own numba availability" (the historical
        # behaviour).  The two backends are the SAME mathematics in a different
        # floating-point ORDER, so an evaluator rebuilt in a Newton pool worker
        # must take the branch the PARENT took or the pool stops being
        # bit-identical to serial.  See ``_resolved_cheb_backend``.
        self.backend = _validated_cheb_backend(backend)
        # The fit itself (a tiny lstsq -- typically a few hundred rows
        # by 28-70 terms) is always performed on CPU via NumPy, even
        # when xp=cupy.  Three reasons:
        #   1. NumPy lstsq is reliable and dependency-free; cupy.linalg.
        #      lstsq needs cuSOLVER which isn't guaranteed to be
        #      present on every cupy install.
        #   2. The fit is O(1) per apply_real_lens_traced call (one-
        #      time cost) and is negligible vs per-pixel Newton work.
        #      Routing it via the CPU has no measurable impact.
        #   3. The payoff from xp=cupy is in the Newton hot loop (N^2
        #      evaluator calls), where it does matter -- only the
        #      fitted coefficients need to live on the device.
        xs_np = np.asarray(xp.asnumpy(xs_in) if xp is not np else xs_in,
                            dtype=np.float64)
        ys_np = np.asarray(xp.asnumpy(ys_in) if xp is not np else ys_in,
                            dtype=np.float64)
        vals_np = np.asarray(xp.asnumpy(values) if xp is not np else values,
                              dtype=np.float64)
        self.order = int(order)
        # Scalars extracted as Python floats so chain-rule multiplies
        # stay backend-agnostic and don't pull host-device copies later.
        self.xmin = float(xs_np.min())
        self.xmax = float(xs_np.max())
        self.ymin = float(ys_np.min())
        self.ymax = float(ys_np.max())
        # Build total-degree multi-indices (kx, ky) with kx + ky <= order
        self._mi = [(kx, ky)
                     for kx in range(order + 1)
                     for ky in range(order + 1 - kx)]
        n_terms = len(self._mi)
        # Fit on CPU using NumPy
        X_np, Y_np = np.meshgrid(xs_np, ys_np, indexing='ij')
        u_np = (2.0 * X_np - (self.xmin + self.xmax)) / (self.xmax - self.xmin)
        v_np = (2.0 * Y_np - (self.ymin + self.ymax)) / (self.ymax - self.ymin)
        K1_np = np.asarray([m[0] for m in self._mi], dtype=np.int64)
        K2_np = np.asarray([m[1] for m in self._mi], dtype=np.int64)
        Tu_np = _cheb_vand_2d(u_np, order, np)
        Tv_np = _cheb_vand_2d(v_np, order, np)
        # FIX_RUNNER_OOM_2026_08_13.  Built in ROW BLOCKS into a preallocated
        # buffer.  The one-liner this replaces,
        # ``(Tu_np[K1_np] * Tv_np[K2_np]).reshape(n_terms, -1).T``, is three
        # full ``(n_terms, n_samples)`` arrays alive at once -- the two fancy-
        # index GATHERS and their product -- so a fit that RETURNS a 1.29 GB
        # design matrix (measured: 2401^2 retained samples at order 6, the
        # ``test_niche_c1_consolidation`` exit-NA case) transiently claimed
        # 4.7 GB, three times over in one call.  That is survivable on a 128 GB
        # dev box and is 2/3 of a CI runner.
        #
        # BIT-IDENTICAL, in values AND in what the SOLVE sees.  Every entry is
        # the same product of the same two Chebyshev values -- elementwise, no
        # reduction, so no summation order to change.  Layout would be
        # load-bearing (``_solve_lstsq_thread_safe`` forms ``A.T @ A``, whose
        # BLAS reduction order depends on it) except that the solver's FIRST
        # line is ``np.ascontiguousarray(A)``: it always squared a C-contiguous
        # copy.  Building C-contiguous here hands it exactly that array and
        # RETIRES the copy -- one 1.29 GB allocation less on the measured case,
        # and the weighted branch's ``np.ascontiguousarray`` below likewise
        # becomes a no-op that scales in place.
        _Tu_f = Tu_np.reshape(order + 1, -1)
        _Tv_f = Tv_np.reshape(order + 1, -1)
        _n_flat = _Tu_f.shape[1]
        A_full = np.empty((_n_flat, n_terms), dtype=np.float64)
        _step = max(1, _CHEB_FIT_CHUNK_ENTRIES // max(1, n_terms))
        for _s in range(0, _n_flat, _step):
            _e = min(_s + _step, _n_flat)
            # SLICE first, gather second: ``_Tu_f[K1_np][:, s:e]`` would
            # materialise the full ``(n_terms, n_samples)`` gather this whole
            # block exists to avoid.
            np.multiply(_Tu_f[:, _s:_e][K1_np].T, _Tv_f[:, _s:_e][K2_np].T,
                        out=A_full[_s:_e])
        del _Tu_f, _Tv_f, Tu_np, Tv_np
        vals_flat = vals_np.ravel()
        finite = np.isfinite(vals_flat)
        if weights is not None:
            # D1 (2026-07-28): WEIGHTED least squares.  ``weights`` is a
            # per-sample amplitude weight with the same shape as ``values``;
            # the residual minimised is ``sum (w_i r_i)^2``.  A hard sample
            # mask is the ``w in {0, 1}`` special case, and it is the RIGHT
            # one only when the retained samples are concentric with this
            # basis's domain -- see ``_FIT_DISC_OUTSIDE_WEIGHT_REL``.
            w_flat = np.asarray(
                xp.asnumpy(weights) if xp is not np else weights,
                dtype=np.float64).ravel()
            if w_flat.shape != vals_flat.shape:
                raise ValueError(
                    "_Cheb2DEvaluator: weights shape "
                    f"{np.shape(weights)} != values shape {vals_np.shape}")
            keep = finite & np.isfinite(w_flat) & (w_flat > 0.0)
            _all = bool(keep.all())
            # one design-matrix copy, scaled in place: the unweighted branch
            # also pays exactly one (``_solve_lstsq_thread_safe`` makes ``A``
            # contiguous), so weighting does not raise peak memory.
            A = np.ascontiguousarray(A_full if _all else A_full[keep, :])
            w_keep = w_flat if _all else w_flat[keep]
            A *= w_keep[:, None]
            rhs = (vals_flat if _all else vals_flat[keep]) * w_keep
        elif finite.all():
            A = A_full
            rhs = vals_flat
        else:
            A = A_full[finite, :]
            rhs = vals_flat[finite]
        # B7: normal-equations solve (thread-safe; never takes gelsd's SVD path
        # that deadlocks against JAX's OpenMP runtime in a shared process).
        c_np = _solve_lstsq_thread_safe(A, rhs)
        # Push coefficients + index arrays onto the target backend
        self.coeffs = xp.asarray(c_np, dtype=xp.float64)
        self._K1 = xp.asarray(K1_np, dtype=xp.int64)
        self._K2 = xp.asarray(K2_np, dtype=xp.int64)

    # ----------------------------------------------------------------
    # v5.33.0 (FIX_POOL_REBUILD): construct from an ALREADY-BUILT fit.
    #
    # ``__init__`` RUNS the least-squares fit.  A Newton pool worker used to
    # call it, which meant every worker re-solved the same normal equations in
    # its own interpreter -- and that solve is a BLAS reduction whose ORDER
    # depends on the BLAS thread regime, so a worker whose regime differed from
    # its parent's recovered coefficients that differ in the last bits.  See
    # ``_cheb_fit_state`` for the measurement.  This entry point takes the
    # parent's coefficients and does no arithmetic at all.
    # ----------------------------------------------------------------
    @classmethod
    def from_state(cls, state, xp=None, backend=None):
        """Rebuild an evaluator from :func:`_cheb_fit_state` output.

        NO fit is performed: the coefficients are the caller's.  This is what
        makes a Newton pool worker's evaluator bit-identical to the parent's by
        CONSTRUCTION rather than by the least-squares solve happening to be
        reproducible across two processes' BLAS.

        ``backend`` is the pinned evaluation branch (see
        ``ev_value_and_grad``); it is a property of where the evaluator RUNS,
        not of the fit, so it is passed here rather than carried in ``state``.
        """
        if xp is None:
            xp = np
        self = cls.__new__(cls)
        self.xp = xp
        self.backend = _validated_cheb_backend(backend)
        self.order = int(state['order'])
        self._mi = [(int(kx), int(ky)) for (kx, ky) in state['mi']]
        self.xmin = float(state['xmin'])
        self.xmax = float(state['xmax'])
        self.ymin = float(state['ymin'])
        self.ymax = float(state['ymax'])
        c_np = np.asarray(state['coeffs'], dtype=np.float64)
        K1_np = np.asarray(state['K1'], dtype=np.int64)
        K2_np = np.asarray(state['K2'], dtype=np.int64)
        # A truncated or mismatched payload must fail HERE, loudly, rather than
        # broadcast its way to a plausible-looking wrong field: the whole point
        # of shipping the fit is that the worker's answer is the parent's.
        if not (c_np.ndim == 1 and c_np.shape == K1_np.shape
                == K2_np.shape == (len(self._mi),)):
            raise ValueError(
                f"_Cheb2DEvaluator.from_state: inconsistent fit state -- "
                f"coeffs {c_np.shape}, K1 {K1_np.shape}, K2 {K2_np.shape}, "
                f"{len(self._mi)} multi-indices; all four must agree.")
        self.coeffs = xp.asarray(c_np, dtype=xp.float64)
        self._K1 = xp.asarray(K1_np, dtype=xp.int64)
        self._K2 = xp.asarray(K2_np, dtype=xp.int64)
        return self

    def _to_u(self, x):
        return (2.0 * x - (self.xmin + self.xmax)) / \
                 (self.xmax - self.xmin)

    def _to_v(self, y):
        return (2.0 * y - (self.ymin + self.ymax)) / \
                 (self.ymax - self.ymin)

    # ----------------------------------------------------------------
    # Backward-compat single-quantity API (RectBivariateSpline.ev()).
    # Supports dx=0/1 and dy=0/1 (up to first derivatives).
    # ----------------------------------------------------------------
    def ev(self, x, y, dx=0, dy=0):
        """Evaluate polynomial (or partial derivative) at (x, y).

        Compatible subset of SciPy RectBivariateSpline.ev: supports
        dx=0/1 and dy=0/1 (up to first derivatives).  When multiple
        derivatives are needed at the same (x, y), prefer
        :meth:`ev_value_and_grad` -- one call returns all three.
        """
        if (dx, dy) in ((0, 0), (1, 0), (0, 1)):
            f, fx, fy = self.ev_value_and_grad(x, y)
            if dx == 0 and dy == 0:
                return f
            if dx == 1 and dy == 0:
                return fx
            return fy
        raise NotImplementedError(
            f"_Cheb2DEvaluator.ev with dx={dx}, dy={dy} not supported; "
            f"only 0th and 1st derivatives in a single axis.")

    # ----------------------------------------------------------------
    # Combined value + gradient (#6) -- primary entry point for the
    # Newton loop in apply_real_lens_traced.  Shares Chebyshev basis
    # work across all three quantities.  Uses the Numba fastpath (#1)
    # when available on the NumPy backend; otherwise a pure-xp
    # implementation that runs on NumPy or CuPy alike.
    # ----------------------------------------------------------------
    def ev_value_and_grad(self, x, y):
        """Evaluate the polynomial and both partial derivatives in one
        pass.

        Returns
        -------
        f, df/dx, df/dy : arrays with the broadcast shape of (x, y)
            Value and physical-space partial derivatives (chain rule
            applied to undo the ``[-1, 1]`` normalisation).
        """
        xp = self.xp
        x = xp.asarray(x, dtype=xp.float64)
        y = xp.asarray(y, dtype=xp.float64)
        u = self._to_u(x)
        v = self._to_v(y)
        sx = 2.0 / (self.xmax - self.xmin)
        sy = 2.0 / (self.ymax - self.ymin)

        # Numba fastpath on the NumPy backend (kernel compiled lazily on first
        # use; None when numba is unavailable -> fall through to the pure-xp path)
        #
        # ``self.backend`` PINS that choice when it is not None (v5.32.3): a
        # Newton pool worker rebuilds this evaluator in a fresh interpreter, and
        # if it resolved a different branch than the parent did the same
        # mathematics would run in a different floating-point order and the pool
        # would stop being bit-identical to serial (FIX_POOL_MEMORY sec 8.1,
        # MEASURED 5.167e-14 on this file's own 262 144-point chain, 1.358e-11
        # on CI's).  ``'numba'`` still resolves through the getter, so a worker
        # that CANNOT provide the kernel gets ``None`` here and the caller --
        # ``_newton_invert_chunk`` -- refuses rather than silently substituting
        # the other order.
        if self.backend == 'numpy':
            _cheb_kernel = None
        elif self.backend == 'numba':
            _cheb_kernel = _get_cheb2d_val_grad_numba() if xp is np else None
        else:
            _cheb_kernel = (_get_cheb2d_val_grad_numba()
                            if xp is np and _NUMBA_AVAILABLE else None)
        if _cheb_kernel is not None:
            u_flat = np.ascontiguousarray(u.ravel(), dtype=np.float64)
            v_flat = np.ascontiguousarray(v.ravel(), dtype=np.float64)
            coeffs = np.ascontiguousarray(self.coeffs, dtype=np.float64)
            K1 = np.ascontiguousarray(self._K1, dtype=np.int64)
            K2 = np.ascontiguousarray(self._K2, dtype=np.int64)
            f_flat, fx_u_flat, fy_v_flat = _cheb_kernel(
                coeffs, K1, K2, u_flat, v_flat, self.order)
            shape = u.shape
            return (f_flat.reshape(shape),
                    fx_u_flat.reshape(shape) * sx,
                    fy_v_flat.reshape(shape) * sy)

        # Pure-xp fallback (always-on; REQUIRED for CuPy backend).
        # Build T and T' Vandermondes once, gather by multi-index, and
        # contract against the coefficient vector with one sum each.
        Tu = _cheb_vand_2d(u, self.order, xp)
        Tv = _cheb_vand_2d(v, self.order, xp)
        dTu = _cheb_deriv_vand_2d(u, self.order, xp)
        dTv = _cheb_deriv_vand_2d(v, self.order, xp)
        # Gather per-basis-term arrays: shape (M, ...u.shape)
        Tu_K = Tu[self._K1]
        Tv_K = Tv[self._K2]
        dTu_K = dTu[self._K1]
        dTv_K = dTv[self._K2]
        # Broadcast coefficients and sum over the basis-term axis.
        c_shape = (len(self._mi),) + (1,) * u.ndim
        c_b = self.coeffs.reshape(c_shape)
        f    = xp.sum(c_b * Tu_K  * Tv_K , axis=0)
        fx_u = xp.sum(c_b * dTu_K * Tv_K , axis=0)
        fy_v = xp.sum(c_b * Tu_K  * dTv_K, axis=0)
        return f, fx_u * sx, fy_v * sy


def _cheb_vand_2d(u, max_k, xp=None):
    """Chebyshev T_k(u) for k=0..max_k as (max_k+1,) + u.shape array.

    Backend-agnostic: pass ``xp=numpy`` (default) or ``xp=cupy`` to run
    on host or device respectively.
    """
    if xp is None:
        xp = _get_array_module(u)
    T = xp.empty((max_k + 1,) + u.shape, dtype=xp.float64)
    T[0] = 1.0
    if max_k >= 1:
        T[1] = u
    for n in range(2, max_k + 1):
        T[n] = 2.0 * u * T[n - 1] - T[n - 2]
    return T


def _cheb_deriv_vand_2d(u, max_k, xp=None):
    """T'_k(u) via T'_n = n U_{n-1}; shape (max_k+1,) + u.shape.

    Backend-agnostic: pass ``xp=numpy`` (default) or ``xp=cupy``.
    """
    if xp is None:
        xp = _get_array_module(u)
    Tp = xp.zeros((max_k + 1,) + u.shape, dtype=xp.float64)
    if max_k < 1:
        return Tp
    U = xp.empty((max_k + 1,) + u.shape, dtype=xp.float64)
    U[0] = 1.0
    if max_k >= 1:
        U[1] = 2.0 * u
    for n in range(2, max_k + 1):
        U[n] = 2.0 * u * U[n - 1] - U[n - 2]
    for n in range(1, max_k + 1):
        Tp[n] = float(n) * U[n - 1]
    return Tp


def _geometric_lens_phase(lens_prescription, wavelength, dx, N):
    """Compute the analytic per-surface sag-phase-screen sum for a lens.

    NOT ORIGIN-AWARE, and deliberately so (niche D9): it builds
    ``x = (arange(N) - N/2) * dx`` and evaluates the element's sag on it, i.e.
    it assumes the grid is centred on the optical axis.  It is reached ONLY from
    the two ``fast_analytic_phase and preserve_input_phase`` branches of
    :func:`apply_real_lens_traced`, and ``preserve_input_phase='remap'`` -- the
    only mode in which a non-zero ``origin`` is accepted -- sets
    ``preserve_input_phase = False`` before either is tested.  So it is dead on
    every path that can carry an origin.  (An engaged carrier also forces
    ``fast_analytic_phase = False`` independently.)

    Returns the *geometric* component of the phase a plane wave would
    acquire after passing through the lens -- equivalent to
    ``np.angle(apply_real_lens(ones, ...))`` except that the ASM
    diffractive correction between surfaces is omitted.

    For smooth refractive lens prescriptions the omitted correction
    scales as ``t * k_perp^2 / (2k)`` where t is glass thickness and
    k_perp is the characteristic spatial-frequency of the sag.  On
    typical F/10+ refractive lenses this is under 10 nm OPL; for
    faster lenses (F/3 or below) validate before trusting.

    Parameters
    ----------
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing [m].
    N : int
        Grid size (N x N square).

    Returns
    -------
    phase : ndarray (N, N) float64
        Analytic geometric phase in radians, wrapped to the [-pi, pi]
        range so it can be used interchangeably with
        ``np.angle(E_analytic_pw)``.
    """
    from .. import raytrace as _rt
    surfaces = _rt.surfaces_from_prescription(lens_prescription)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    k0 = 2.0 * np.pi / wavelength

    # v5.1.0 (default-knob resolver rollout): real-dtype OPL allocator
    # honours ``set_default_real_dtype(...)`` -- this is one of the
    # documented consumer wirings.  Falls back to np.float64 if the
    # propagators module is mid-load (defensive).
    try:
        from ..propagators.propagation import get_default_real_dtype
        _real_dtype = get_default_real_dtype()
    except ImportError:
        _real_dtype = np.float64
    phase = np.zeros((N, N), dtype=_real_dtype)

    # Accumulate per-surface sag phase: phi += -k0 * (n_after - n_before) * sag(x, y)
    # This matches the thin-element OPD used inside apply_real_lens's
    # phase-screen model (the default paraxial formula) -- so dropping
    # the ASM step is the only physics difference.
    for surf in surfaces:
        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        if abs(n2 - n1) < 1e-15:
            continue   # no refraction
        sag = _rt._surface_sag_xy(X, Y, surf)
        phase = phase + (-k0 * (n2 - n1) * sag)

    # Also add the bulk glass piston (constant k*n*t_i in each glass)
    # since the full apply_real_lens includes this via the ASM in-glass
    # propagation.  The piston is a rigid offset but keeping it
    # preserves absolute-phase consistency when this function is used
    # for the phase_analytic_lens reference.
    for surf in surfaces[:-1]:
        n_mid = get_glass_index(surf.glass_after, wavelength)
        phase = phase + k0 * n_mid * float(surf.thickness)

    # Wrap to match np.angle convention
    return np.angle(np.exp(1j * phase))


# F1 (audit): residual transverse-angular-spread (radians) above which the
# plane-wave / carrier-referenced traced correction is flagged as invalid.
# ~0.02 rad (~1 deg) cleanly separates a collimated / carrier-matched beam
# (residual ~ 0) from an unreferenced divergent / emitter-array field
# (residual 0.1-0.2 rad); heuristic, tunable by the caller via the
# on_noncollimated policy.
_NONCOLLIMATED_RESID_THRESH = 0.02

# N5 (2026-07-19): when ``tilt_aware_rays=True`` and no explicit ``carrier`` is
# given, an auto-fit carrier eikonal is threaded through the carrier plumbing so
# the exit wavefront carries the input congruence (matching the carrier path's
# H6 entrance-eikonal fix).  It engages only when the fitted eikonal's peak phase
# over the bright support exceeds this floor, so a (near-)collimated tilt_aware
# input -- which fits ``W == 0`` exactly for a real / globally-phased field --
# keeps the byte-identical plane-wave-reference path.  1e-2 rad (~lambda/628 of
# OPD across the beam) sits far below any divergence that shifts the focus (a
# gently diverging R=10 m beam already fits tens of radians) yet safely above
# float round-off.
_TILT_EIKONAL_MIN_RAD = 1e-2

# F3 (audit 2026-07-21): when BOTH ``tilt_aware_rays=True`` and an EXPLICIT
# engaged carrier are given, route the ray launch + the R7 intra-group fixes
# through the carrier gradient (the ``carrier=R`` default path) instead of the
# per-pixel tilt launch, which DEGRADES a steep spherical carrier (1.72 rad rms
# vs 0.008 rad).  Exposed as a module flag so the regression test can force the
# pre-fix per-pixel-tilt launch (fail-before) via monkeypatch.
_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER = True

# R6 / audit F1 (2026-07-21): the ``carrier='auto'`` least-squares gradient fit
# silently degraded to ~inf (no carrier) on a strongly-diverging / coarsely-
# sampled spherical input -- exactly the input class it exists for.  Root cause:
# the nearest-neighbour phase-increment ``angle(E[i+1] conj(E[i]))`` tilt reading
# ALIASES (wraps past +-pi) at radii where the local carrier tilt exceeds the
# grid Nyquist tilt ``lambda/(2 dx)``.  On the 121's S5-S7 group (R_in=+153 mm,
# w=6 mm, dx=24.6 um) that boundary sits at r ~ 4 mm, well inside the beam, so
# MOST of the bright support fed the fit wrapped (near-zero-mean) tilt samples
# that pulled the fitted 1/R toward 0.  Fix: restrict the fit to the CONNECTED
# un-aliased core -- the region, contiguous with the phase-flat point, where the
# tilt reading stays below this fraction of the Nyquist tilt.  The wrapped rings
# beyond the first Nyquist crossing form SEPARATE connected components (a high-
# tilt annulus disconnects them from the core), so they are excluded and the
# central parabola alone -- which fully determines the spherical R -- drives the
# fit.  Recovers R to <~1% on S5-S7 and is byte-identical on well-sampled inputs
# (whole bright support is one un-aliased component -> same samples as before).
# The recovery is insensitive to this fraction over 0.35-0.7; 0.5 is a safe
# midpoint (masks before ANY grid axis component wraps, since gmag is the vector
# tilt magnitude).
_AUTO_CARRIER_NYQUIST_FRAC = 0.5
# Minimum un-aliased-core sample count below which the core restriction is
# abandoned and the full bright support is used (the historical behaviour): too
# few core samples cannot constrain the low-order fit, so a pathological /
# near-fully-aliased input falls back rather than fitting noise.
_AUTO_CARRIER_MIN_CORE = 64
# The un-aliased-core restriction engages ONLY when at least this fraction of
# the bright support reads as aliased (local tilt >= the Nyquist fraction).  A
# well-sampled input -- flat, mildly diverging, or a MULTI-EMITTER array whose
# beamlets are each Nyquist-sampled -- has ~no aliased samples, so the fit keeps
# the full bright support (byte-identical to the historical single-component-
# agnostic fit; critically, it does NOT collapse a disconnected multi-emitter
# field onto one beamlet's connected component).  The F1 strongly-diverging
# single carrier aliases the great majority of its support, far above this.
_AUTO_CARRIER_ALIAS_FRAC = 0.05

# R7 / audit F2 (2026-07-21): fraction of the entrance LAUNCH RADIUS inside which
# the entrance->exit forward-map and OPL Chebyshev fits are restricted WHEN A
# CARRIER IS SET.  The launch grid is a square spanning +-launch_radius (=1.5x
# the aperture radius); its outer margin + corners carry the most strongly
# aberrated / near-vignetting marginal rays, whose out-of-basis high order
# (r^8+) ALIASES into the low-order (defocus / r^4) coefficients of the GLOBAL
# order-6 tensor-Chebyshev least-squares fit.  On a strongly-focusing thick
# group (the 121 triplets) that spurious defocus is the dominant per-group
# exit-wavefront error (audit F2: 122% of 1/R_out on S18-S20, curvature-
# dominated, hf~0, ray_subsample-invariant -- a fit artefact, NOT undersampling;
# the local bicubic spline is immune).  Restricting the fit to r <= FRAC*
# launch_radius (= 0.9x the aperture radius at FRAC=0.6) drops the contaminating
# marginal rays, so the central coefficients are clean; the fit still spans the
# full launch domain (the beam and its evaluated exit rays sit inside FRAC*
# launch_radius, and the smooth low-order fit extrapolates faithfully to the
# low-amplitude tail beyond it).  INPUT-INDEPENDENT (geometry only), so the
# prepared-screen reuse path (return_screen / apply_real_lens_traced_multi)
# stays valid.  Gated on a carrier being set, so the carrier=None default path
# is byte-identical.  0.5 (= 0.75x the aperture radius) recovers per-group rms
# well under 0.1 rad across the 121's 8 groups including the steepest triplet
# (S25-S27 0.100 rad at FRAC=0.6 -> ~0.03 at 0.5); the recovery is insensitive
# over ~0.45-0.6 (looser leaves the steep converging groups near the 0.1 gate;
# tighter over-trims the diverging groups' exit-ray coverage).
_CARRIER_FIT_RADIUS_FRAC = 0.5
# Minimum in-disc coarse-sample count below which the restriction is abandoned
# (too few samples to constrain the order-6 fit): fall back to the full launch
# grid rather than fit noise.
_CARRIER_FIT_MIN_SAMPLES = 64

# P2 / audit AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4 -- the APERTURE:BEAM
# CLIFF.  The R7 restriction above is gated on a carrier being ENGAGED, so the
# plane-wave-reference path (``carrier=None``, or a carrier too flat to engage --
# e.g. the collimated first group of a chain) fits its OPL / forward map over the
# WHOLE launch square, corners included.  When the physical aperture greatly
# exceeds the beam, those corner rays are marginal rays the beam never occupies,
# and on a FAST surface their out-of-basis high order destroys the global
# low-order fit INSIDE the beam.  Measured on the E4 corrected relay (2026-07-25,
# beam w = 2 mm, fast biconvex f/4.5 first group): exit-wavefront Strehl 0.999
# (4 mm aperture) / 0.998 (6 mm) -> 0.105 (7 mm) -> 0.039 (10 mm).  The cliff is
# a THRESHOLD, not a gradual degradation: it tracks the launch square growing
# past ~2.5x the beam radius, and it is entirely a PHASE (fit) effect --
# independent of ``amplitude_model`` / ``preserve_input_phase`` /
# ``carrier_reference`` (all three flip it identically) and NOT recovered by
# shrinking ``_CARRIER_FIT_RADIUS_FRAC`` (which is inactive on this path).
#
# ``fit_radius_beam_factor`` restricts the ray-FIT domain to a beam-relative
# disc, on BOTH paths, WITHOUT touching ``launch_radius`` / ``bound`` /
# ``out_of_domain`` -- so the Newton inversion still covers the full launch
# domain, the low-order fit still extrapolates over the whole aperture, and NO
# field energy is clipped (the failure mode of clamping the aperture itself,
# which vignettes real halo power -- audit
# AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23 §4c.3).
_FIT_RADIUS_BEAM_FACTOR_DEFAULT = 2.0
# Aperture diameter / beam 1/e^2 diameter above which the cliff is possible and
# the warn-only guard fires (the E4 cliff starts at 1.75x; 1.5x is the last
# measured-clean ratio).
_APERTURE_BEAM_WARN_RATIO = 1.5

# D1 (2026-07-28): RELATIVE Gram-matrix budget given to the samples OUTSIDE an
# OFF-CENTRE ray-fit disc.  Both fit-domain restrictions above (R7 and P2) are
# implemented as a hard sample mask -- ``w in {0, 1}`` -- and that is safe only
# while the retained disc is CONCENTRIC with the tensor-Chebyshev basis's own
# domain (the launch square).  Then the unconstrained directions of the fit
# inherit the map's radial symmetry, the extrapolation outside the disc stays
# MONOTONE, and the Newton inversion cannot find a second root.
#
# A decentred beam (``beam_centre`` / a decentred ``TiltedCarrier``) breaks
# that.  Measured on an 80/-80 N-BK7 singlet, 30 mm aperture, 1 mm beam at
# x_c = 10 mm, order-6 fit, 341 in-disc coarse samples of 55225: the hard-mask
# fit reproduces the traced forward map to 1.3 pm INSIDE the disc but departs
# from it by up to 135 mm outside, and d(x_out)/d(x_in) changes sign -- i.e.
# the fitted map FOLDS.  The Newton inverse then sends far-field exit pixels
# back into the bright beam, and ``amplitude_model='ray_density'`` gives them
# real amplitude: a spurious lobe carrying 6.8e-3 of the input power at 0.75 of
# the on-beam peak, ~20 mm from the beam.  (The concentric on-axis fit of the
# same geometry: no sign change, ghost power 1.0e-8 -- identical to no
# restriction at all.)
#
# The fix keeps every sample in the fit and gives the out-of-disc ones a tiny
# weight instead, sized so their total contribution to the normal matrix is
# ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` of the in-disc one:
#
#     w_out = sqrt(_FIT_DISC_OUTSIDE_WEIGHT_REL * n_in / n_out)
#
# (grid-count-independent, so it does not drift with ``ray_subsample`` / ``N``).
# The well-determined directions -- the ones the disc exists to clean -- keep
# the in-disc solution; the directions the disc leaves FREE are pinned to the
# traced map instead of to fit noise.  Same measurement as above: in-disc
# agreement 0.2 nm (the hard mask's ill-conditioned solve is no better), the
# out-of-disc departure from the traced map 135 mm -> 0.14 mm, no sign change,
# and the returned field back on the local-spline oracle (ghost power 6.8e-3 ->
# 2.5e-8 of Pin, max off-beam amplitude 0.75 -> 3e-4 of peak).
#
# Measured end to end on both geometries, sweeping this constant: ANY nonzero
# value removes the fold; the returned field tracks the spline oracle to
# <= 3.4e-4 of peak over 1e-14..1e-8 and then degrades as the out-of-disc data
# starts to matter (7.6e-3 at 1e-6, 9.1e-2 at 1e-2, 0.97 at 1.0 -- at 1.0 the
# restriction is gone and this IS the aperture:beam cliff the disc exists to
# prevent).  1e-8 sits in the middle of the plateau, ~4 decades clear of the
# fold on one side and ~2 of the cliff on the other.  Engaged ONLY when the
# disc is off centre, so the concentric (default) path is byte-identical.
#
# ENVELOPE OF THAT SWEEP -- narrower than it reads (niche C2, 2026-07-30).
# Every number above comes from ONE regime: the 80/-80 N-BK7 singlet, a 1 mm
# beam in a 30 mm aperture, i.e. aperture:beam = 30:1 and a low exit NA.  It is
# NOT established for a beam that FILLS its aperture at design-121-class NA
# (20.397 mm aperture / 6.251 mm beam = 3.26:1, exit NA 0.405), and an attempt
# to extend it there does not merely give a different answer -- it cannot be
# run at all by this method, because the oracle the sweep is scored against
# stops working first.  On design 121's last group at N=1024 / dx = 33.211 um
# the exit NA is 0.356 against a grid Nyquist direction cosine of 0.0197 (18x
# short), and ``newton_fit='spline'`` -- the fit-domain-free reference this
# note and D7's leans on -- fails to converge for 100.0 % of 65536 pixels and
# returns an ALL-ZERO field (the polynomial path fails for 81.4 % and still
# returns a usable one).  With ``on_undersample='silent'`` that zero is
# returned without a word, so a caller reaching for the spline oracle in this
# regime gets nothing and is not told.
#
# So: treat 1e-14..1e-8 as a plateau MEASURED AT LOW NA with a small beam, and
# 1e-8 as a defensible default rather than a value centred on evidence that
# spans the library's regimes.  Nothing here affects the shipped design-121
# chain, which never reads this element at 33 um: it re-traces the final leg on
# a fine grid (``n_fine_cap`` 12288) and its acceptance is unchanged.  Widening
# the evidence needs an oracle that survives high exit NA on a coarse grid,
# which no shipped estimator currently does.
#
# Keeping the samples also keeps the GRID intact, which matters twice over: the
# paraxial-magnification stencil that seeds Newton reads ``x_out_grid`` AT THE
# AXIS (NaN there -> the 0.91 fallback -> Newton seeded on the wrong branch of
# the folded map, which is what actually populated the spurious lobe), and the
# process-pool knot data / direct-fit exit hull are built from the same arrays.
# Set to ``0.0`` to restore the historical hard NaN mask exactly -- that is the
# fail-before switch the D1 tests use, not a supported configuration.
_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8

# D7 (2026-07-29): POLYNOMIAL ORDER for the OFF-CENTRE ray fit.
#
# Same regime as ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` above (a decentred beam, so
# the ray-fit disc is not concentric with the launch square), and it is the
# ONLY thing that measurably limits that fit.  A disc of radius ``r`` about a
# chief ray ``|c|`` off axis reaches ``|c| + r`` into the aperture instead of
# ``r``, i.e. into strictly more aberrated marginal territory, so the SAME
# total-degree budget buys a worse approximation.  Measured on design 121's
# last group (20.397 mm aperture, w = 3.1255 mm beam,
# ``fit_radius_beam_factor=2``, weighted fit, OPL residual rms over ``r <= w``
# about the beam):
#
#     order        6        8       10       12       14
#     on axis   0.177 nm  0.003    0.000    0.000    0.000
#     |c| = 0.97 w   2.508 nm  0.667    0.121    0.114    0.199
#
# RAY GRID, pinned (it was not stated when this table was first written, and
# the numbers do depend on it): ``ray_subsample=4`` on the N=1024 co-moving
# grid, i.e. a 256^2 coarse launch.  Re-measured there the row reads 0.177 /
# 2.507 / 0.122, which is the table above to 3 digits.  The ARGUMENT does not
# hinge on that choice -- at ``ray_subsample=8`` (128^2) the same row reads
# 0.178 on axis and 2.495 / 0.675 / 0.108 / 0.121 / 0.211 off, so order 6 is
# still ~14x worse off axis and order 10 still recovers ~20x of it.  Note that
# NRAY in ``decentred_fit_defect.py`` (default 161) is NOT this quantity: it
# sizes the FFT estimator's oracle bundle and never feeds the fit.
#
# The on-axis entries at order >= 10 are printed as 0.000 but are better read
# as "negligible" than as exact zeros: on axis the fit takes the concentric
# HARD-MASK branch rather than the weighted one, and a re-fit through the
# weighted path puts them at the 0.02-0.10 nm level.  Nothing in the D7
# argument rests on them -- it rests on the off-axis row.
#
# i.e. order 6 is 14x worse off axis than on, order 10 recovers 20x of that,
# and order 14 starts to LOSE to conditioning (the normal-equations Gram matrix
# runs 1.0e10 -> 1.9e13 across the sweep).  End to end on the niche-D6
# ``K = -n^2`` conic stand-in -- whose truth is analytic AND decentre-invariant,
# so chain/oracle EE ratios are directly comparable on axis and off -- the EE2
# ratio at one full beam radius of decentre reads 0.9498 (order 6), 0.9828
# (order 10), 0.9877 (order 14) against 0.9966 on axis.  10 takes ~87 % of the
# available recovery at 66 basis terms against 120.
#
# Engaged ONLY on the off-centre branch (the concentric / on-axis path keeps
# ``newton_poly_order`` exactly, so the shipped default is byte-identical), and
# only ever RAISES the order -- a caller asking for more still gets more.  Pass
# ``decentred_fit_poly_order=<newton_poly_order>`` to restore the pre-D7
# behaviour exactly; that is the fail-before switch the D7 tests use.
#
# NOT taken, and why (measured, 2026-07-29): re-mapping the tensor-Chebyshev
# BASIS DOMAIN onto the off-centre disc (an affine re-centre + re-scale) is a
# NO-OP for accuracy -- the total-degree space is affine-invariant, so the
# weighted least-squares solution is the same polynomial, and the OPL residual
# over the beam agreed to 4-5 significant figures across launch-square vs
# disc-bbox domains AND normal-equations vs SVD solves (2.5076 nm both ways at
# order 6, 0.1210 vs 0.1210 at order 10).  It is also a LIABILITY: the Newton
# loop evaluates these same fits over the WHOLE launch square, where the
# re-mapped basis runs to ``|u| = 2.4`` and ``max|T_k| = 5.7e8`` at order 12, and
# the two mathematically-identical fits then differ by up to 9.9e-7 m of
# ``x_out`` at the launch corners (5.2e-9 m at order 6).  Conditioning alone is
# not a reason to ship it: cond(Gram) does fall 1.0e10 -> 3.2e4, but float64
# already carries the answer.
_DECENTRED_FIT_POLY_ORDER = 10

# niche C1 item 1 (2026-07-30): WHEN a declared beam centre counts as OFF
# CENTRE at all.
#
# D1 selected the off-centre branch with ``bool(_bcx or _bcy)``, i.e. ANY
# nonzero offset -- including a physically NULL one.  That is a discontinuity,
# not a superset: the branch swaps the historical concentric HARD NaN MASK for
# the weighted restriction (``_FIT_DISC_OUTSIDE_WEIGHT_REL``) AND raises the
# fit order (``_DECENTRED_FIT_POLY_ORDER``), so the returned field jumps at the
# first ulp of decentre.  Measured (synthetic N-BK7 f/6 singlet, N=512,
# dx=30 um, w=1.0 mm, ``fit_radius_beam_factor=2``, ``ray_subsample=8``): the
# two branches differ by ``max|dE| / max|E|`` = **8.32e-6** at a decentre of
# 3e-8 m = **1e-9 pixels**, and that same 8.3e-6 persists flat out to ~0.3 w
# (9.46e-5 at 1.0 w, where the physics finally dominates).  A step of 8.3e-6 at
# 1e-9 px is the whole finding: it is 100x the pipeline's ~1e-7 roundoff floor
# and it is bought by nothing.
#
# WHERE the off-centre branch starts to EARN it, same geometry but w = 1.4 mm
# (fit disc r = 2.8 mm inside a 5.0 mm semi-aperture, so ``|c| + r`` really does
# reach new territory), scored against ``newton_fit='spline'`` -- a LOCAL
# bicubic spline of the traced map, which skips the polynomial fit and its disc
# restriction entirely (the disc block is gated on ``newton_fit != 'spline'``),
# so it is the fit-domain-free reference D1/D7 already use.  ``max|dE|/max|E|``
# over the lit region, concentric arm vs off-centre arm:
#
#     |c|/w      0     0.01   0.02   0.05    0.1    0.2    0.35   0.5    0.75   1.0
#     concentric 3.71e-8 3.74 3.77   3.86   3.39   12.08  36.48  98.24  316.9  1155.5   (x1e-8)
#     off-centre 3.71e-8 1.70 1.02   1.64   1.32    1.32   2.54   2.64   3.24    3.84   (x1e-8)
#
# The concentric arm sits on its own on-axis floor (3.4-3.9e-8) out to
# |c| = 0.1 w and departs from it between 0.1 w and 0.2 w (3.3x the floor at
# 0.2 w, then 2.6x per doubling: 26x at 0.5 w, 312x at 1.0 w).  The off-centre
# arm is FLAT at 1.0-3.8e-8 across the whole sweep -- never worse, decisively
# better from ~0.15 w up.
#
# So the gate is a floor, not a knee-chaser: below
#
#     |c| <= max(_DECENTRE_GATE_PIXELS * dx, _DECENTRE_GATE_W_FRAC * w)
#
# the concentric path is kept BYTE-IDENTICALLY (which includes measuring the
# beam radius about the GRID ORIGIN, as the historical path does), and above it
# the weighted + raised-order path runs exactly as D1/D7 shipped.  0.05 w sits
# 3x below the measured departure (concentric error 1.04x its floor at 0.05 w
# against 3.3x at 0.2 w) and the ``0.5 * dx`` term covers the degenerate corner
# where the beam is sampled so coarsely that 0.05 w is itself sub-pixel.
# Design 121 is nowhere near it: its final-group chief ray is 3.373 mm against
# an entrance beam radius 3.126 mm, i.e. **1.08 w = 21x** the gate.
#
# Setting BOTH constants to 0.0 restores the pre-C1 ``bool(_bcx or _bcy)``
# selector exactly -- that is the fail-before switch the C1 tests use, not a
# supported configuration.
_DECENTRE_GATE_PIXELS = 0.5
_DECENTRE_GATE_W_FRAC = 0.05

#: niche C11 (2026-08-03): CHOOSE the decentred beam's ray-fit branch by
#: MEASURING both candidates instead of predicting the winner from ``|c|/w``.
#:
#: WHAT WAS WRONG.  ``_DECENTRE_GATE_W_FRAC`` above is a floor set to kill a
#: discontinuity at NULL decentre (a branch flip at 1e-9 px), and the note that
#: sets it says so.  It was never the crossover.  Measured on design 121 at the
#: shipped ``_REMAP_RESID_EIKONAL_DEGREE = 6`` (`rc_gate_121.py`, EE3
#: area-exact against the exact-ray CARRY=1 ceiling, per-order residual):
#:
#:     last-group |c|/w   0.000   0.241   0.481   0.723   0.965   1.079
#:     off-centre branch -0.048  +0.028  +0.062  +0.091  +0.141  +0.152
#:     concentric branch -0.048  -0.099  -0.127  +2.033 +67.312 +79.145
#:
#: -- i.e. the branches cross between **0.48 and 0.72 w** on that design, and
#: 0.05 w is 10-14x below it.  On a synthetic f/3 N-BK7 singlet scored against
#: the fit-domain-free ``newton_fit='spline'`` oracle the same crossover is at
#: **0.55 w**, on an f/6 one it is at **0 w**, and on design 121's own six
#: groups it lands anywhere in **0.46-0.69 w**.  A single constant cannot be
#: right for all of them, because the crossover is not a property of the
#: decentre at all -- it is where two DIFFERENT approximations of the same
#: traced map happen to be equally good, and that depends on how much
#: aberration the concentric branch's order-``newton_poly_order`` fit leaves
#: over the beam:
#:
#: * the OFF-CENTRE fit's error is FLAT in decentre (its disc is beam-sized and
#:   beam-centred at every offset; measured 6.2e-7 waves across 0-1.5 w on the
#:   f/3 singlet, 1.8e-9 on the f/6 one, and monotone 0.028 -> 0.152 EE3 points
#:   across design 121's fan);
#: * the CONCENTRIC fit's error GROWS with decentre, because its disc is sized
#:   from the ORIGIN-referenced second moment ``sqrt(2 c^2 + w^2)`` -- an
#:   artefact of measuring about the wrong point, not a physical radius -- so
#:   the same total-degree budget is spread over a disc inflated by
#:   ``sqrt(1 + 2 (c/w)^2)`` (1.22x at 0.5 w, 1.73x at 1.0 w, 2.35x at 1.5 w).
#:   That is the P2 aperture:beam cliff re-entering through the back door; see
#:   ``_FIT_RADIUS_BEAM_FACTOR_DEFAULT``.
#:
#: WHAT REPLACES IT.  At the fit site the rays are ALREADY traced, so both
#: candidates can be BUILT and COMPARED before either is used: fit the OPL each
#: way and score it against the traced samples themselves, weighted by the
#: beam's own intensity ``exp(-2 r^2 / w^2)`` about the measured chief ray --
#: "which polynomial reproduces the traced map where the light is".  Smaller
#: weighted rms wins; an exact tie keeps the CONCENTRIC (historical) one.
#:
#: Cost: ONE extra ``_Cheb2DEvaluator`` OPL fit (the rays, the trace and the
#: two discs all already exist), and one extra second-moment pass over
#: ``|E_in|^2``.  Both are taken ONLY above the C1 null gate.
#:
#: VALIDATED.  On three synthetic geometries (f/6 and f/3 N-BK7 singlets at
#: two beam radii) the arbiter's pick agrees with the spline oracle's verdict
#: on **42 of 42** sweep points, including both sides of the f/3 crossover; on
#: design 121 it flips group by group at 0.46-0.69 w, reproducing the band the
#: chain-level arms bracket, and at (-4,0) group 4 it sees the catastrophe
#: coming as a **18x** margin in the fit residual (0.122 waves concentric
#: against 0.0068 off-centre) before any field is reconstructed.
#:
#: SHIPPED ON since 5.32.1 (2026-08-03), by an EXPLICIT DECISION, and the
#: trade it takes is stated rather than buried.  On design 121 the arbiter
#: improves four of the five tilted orders (by 0.017 / 0.052 / 0.110 / 0.082
#: points), takes the worst-case residual from 0.152 to 0.069 and removes the
#: residual's growth with field angle -- and makes ONE order, (-1,0), worse by
#: 0.026 points against a 0.003-0.015 differential floor.  That per-order
#: "improve or hold" failure is why C11 shipped it OFF: it is a judgement
#: about a design rather than a library fact, and it was left to an explicit
#: decision instead of taken silently in a patch release.  **That decision has
#: now been taken** -- see ``docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md``
#: S10 for the re-measurement of the trade on BOTH BLAS builds.
#:
#: THIS FLAG DECIDES, because :data:`DECENTRED_FIT_PREDICTOR` stayed ``False``.
#: The 5.32.1 flip was ordered for BOTH constants and only this one survived
#: the measurement: the predictor reddened 9 tests in niches D6 and D7 and cost
#: 32 % of the encircled energy on D6's analytic fixture, and every one of
#: those 9 goes green with the predictor off and THIS FLAG STILL ON.  So the
#: arbiter is not merely untouched by that finding -- it is what the finding
#: exonerated.  See ``docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md`` S11.
#:
#: The ladder as shipped:
#:
#:     ARBITER True   -> E_c <= E_o   (niche C11, THE SHIPPED SELECTOR)
#:     ARBITER False  -> the v5.32 gate, bit for bit
#:     PREDICTOR True -> u <= u*      (niche C12, opt-in, and see its note)
#:
#: With BOTH ``False`` the whole C11/C12 layer is BYTE-IDENTICAL to the pure
#: ``_DECENTRE_GATE_W_FRAC`` selector: the arbiter's extra second-moment pass
#: and its two trial fits are not merely unused, they are never reached (the
#: gate site tests these flags before measuring the origin-referenced radius,
#: and the fit site tests them before building any candidate).  That remains
#: the era-pinned fail-before.
#:
#: BELOW the C1 null gate the flag is inert in EVERY state by construction (the
#: arbiter is gated on ``_beam_decentred``), so every C1 byte-identity contract
#: holds either way.
DECENTRED_FIT_ARBITER = True


def _decentred_fit_restriction(disc, weighted, base_order, dec_order):
    """Resolve ONE ray-fit candidate's ``(weights, order)`` from its disc.

    ``weighted`` selects D1's regularised restriction (every sample kept, the
    out-of-disc ones down-weighted to ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` of the
    in-disc Gram contribution) over the historical hard NaN mask, and carries
    D7's order raise with the sample-count step-down that keeps the fit
    determined.  ``weights is None`` means "hard mask at ``base_order``".

    Factored out so the niche-C11 arbiter SCORES exactly the configuration it
    would APPLY -- a candidate scored at one order and applied at another is
    not an arbiter, it is a coin toss.
    """
    if not weighted:
        return None, int(base_order)
    n_in = int(disc.sum())
    n_out = int(disc.size) - n_in
    w_out = (float(np.sqrt(_FIT_DISC_OUTSIDE_WEIGHT_REL * n_in / n_out))
             if n_out > 0 else 1.0)
    # D7: give that off-centre fit the terms its region needs -- but never
    # more terms than the disc can constrain.  The out-of-disc samples carry
    # ~1e-4 of the weight, so the IN-DISC count is what determines the fit;
    # require 3 samples per basis term (order 10 -> 66 terms -> 198 samples)
    # and step the order down until that holds.  Without this the raise could
    # hand an order-10 fit as few as ``_CARRIER_FIT_MIN_SAMPLES`` = 64
    # effective rows for 66 unknowns -- an under-determined normal matrix.
    #
    # This cap is SILENT and it can zero the raise out entirely: the disc
    # holds ~pi (frbf w / (dx rs))^2 samples, so order 10 survives only while
    # frbf*w/(dx*rs) >~ 7.9 coarse pixels.  At the DEFAULT ray_subsample=8
    # both documented configs clear it (223 samples for the synthetic f/6
    # example, 1735 for design 121's last group, against 198) -- but the first
    # clears by only 1.13x and reverts to order 6 at ray_subsample=16, and
    # design 121 reverts at 32.  See ``decentred_fit_poly_order``.
    order = int(dec_order)
    base = int(base_order)
    while order > base and (order + 1) * (order + 2) * 3 // 2 > n_in:
        order -= 1
    return np.where(disc, 1.0, w_out), max(base, order)


def _decentred_fit_score(xs_in, opl_grid, weight, disc, weights, order):
    """Beam-weighted rms of one ray-fit candidate's OPL residual against the
    TRACED samples, in the OPL's own length units.

    The candidate is built exactly as the fit site would build it (hard NaN
    mask when ``weights is None``, D1's weighted restriction otherwise), then
    evaluated on the whole coarse launch lattice and differenced against the
    UNMASKED traced OPL -- so a fit that is clean only where it was allowed to
    look is scored on the beam, not on its own domain.

    ``weight`` is the beam's intensity on that lattice; only the OPL is scored
    because it is the quantity that becomes phase.  Returns ``inf`` when the
    candidate carries no usable weight, so an inadmissible candidate can never
    win.  See :data:`DECENTRED_FIT_ARBITER`.
    """
    vals = opl_grid if weights is not None else np.where(disc, opl_grid, np.nan)
    # the launch lattice, indexing='ij' -- axis 0 is X, axis 1 is Y, matching
    # ``opl_grid`` (see the ``np.meshgrid(..., indexing='ij')`` note at the
    # launch site).  Materialised rather than broadcast: the evaluator's
    # combined value+grad kernel wants two arrays of the SAME shape, and a
    # broadcast pair silently yields a column-rank answer.
    _X, _Y = np.meshgrid(xs_in, xs_in, indexing='ij')
    try:
        ev = _Cheb2DEvaluator(xs_in, xs_in, vals, order=int(order),
                              weights=weights)
        pred = np.asarray(ev.ev(_X, _Y))
    except (np.linalg.LinAlgError, ValueError):
        return float('inf')
    ok = np.isfinite(pred) & np.isfinite(opl_grid)
    w = np.where(ok, weight, 0.0)
    tot = float(w.sum())
    if not (tot > 0.0):
        return float('inf')
    r = np.where(ok, pred - opl_grid, 0.0)
    v = float(np.sqrt(float((w * r * r).sum()) / tot))
    return v if np.isfinite(v) else float('inf')


#: niche C12 (2026-08-03): a FLOOR under the arbiter's beam-intensity score
#: weight, so the score can see the skirt as well as the core.
#:
#: THE BLIND SPOT IT ADDRESSES.  ``_decentred_fit_score`` weights by
#: ``exp(-2 r^2 / w^2)``, which is ~1e-4 at 2 w and ~1e-8 at 3 w, while the
#: Newton inversion evaluates the SAME fits over the whole launch square.  A
#: candidate that is clean on the core and wild on the skirt is therefore
#: scored as if the skirt did not exist -- niche C11 S10 item 1.  With a floor
#: ``F`` the weight becomes ``max(exp(-2 r^2 / w^2), F)``.
#:
#: SHIPPED 0.0 -- INERT, and the reason is a measurement, not caution.  Swept
#: on design 121's own captured arbiter inputs (26 arbitrated element calls
#: across five diffraction orders, ``c12_scorer_sweep.py``):
#:
#:   * ``F >= 1e-8`` flips EVERY call to the off-centre branch, on every order.
#:     The concentric candidate is a HARD NaN MASK, so outside its disc the fit
#:     is pure extrapolation over a launch square 7.5x the beam; any nonzero
#:     floor lets that extrapolation dominate the score.  The selector then
#:     degenerates to "always off-centre", which IS the v5.32 gate on this
#:     design -- every niche-C11 gain is lost and nothing is bought;
#:   * restricting the score to the ILLUMINATED SUPPORT instead (weight 0 where
#:     the beam carries no light) moves no verdict at all: at ``F = 1e-4`` the
#:     26 picks are the shipped 26;
#:   * amplitude weighting (``sqrt`` of the intensity) flips two calls at
#:     (-4,0) and two at (-4,-2) and produces a NON-MONOTONE pattern in
#:     ``|c|/w`` -- a selector that prefers the off-centre branch at 0.05 w and
#:     the concentric one at 0.25 w on the same group.
#:
#: So the (-1,0) counter-movement niche C11 reports is NOT a weighting defect,
#: and this knob exists to record that it was tried and priced.  See S4 of
#: docs/audits/C12_PHYSICS_FIT_SELECTION_2026_08_03.md for the whole sweep and
#: for the separate, stronger reason no per-call selector can fix it.
_DECENTRED_FIT_SCORE_FLOOR = 0.0


def _decentred_fit_score_weight(xs_in, bcx, bcy, w_beam, floor=None):
    """The arbiter's score weight: the beam's own intensity on the launch
    lattice, optionally floored (see ``_DECENTRED_FIT_SCORE_FLOOR``).

    Factored out so the floor has ONE home and so ``floor = 0`` is provably
    the bare Gaussian rather than a Gaussian plus an added zero.
    """
    g = np.exp(-2.0 * (((xs_in[:, None] - bcx) ** 2
                        + (xs_in[None, :] - bcy) ** 2)
                       / (w_beam * w_beam)))
    f = _DECENTRED_FIT_SCORE_FLOOR if floor is None else float(floor)
    return np.maximum(g, f) if f > 0.0 else g


#: niche C12: the total degree at which the traced OPL's own Chebyshev
#: spectrum is measured on the launch box.  It has to exceed
#: ``_DECENTRED_FIT_POLY_ORDER`` for the tail beyond the OFF-CENTRE candidate's
#: own order to be visible at all; 14 gives two even shells of headroom above
#: 10 and costs 120 basis terms against that candidate's 66.  ``0`` disables
#: the spectral half of the predictor (it then decides from the measured
#: candidate residuals alone, which is what it does on an unresolved spectrum
#: anyway -- see :data:`DECENTRED_FIT_PREDICTOR`).
_DECENTRED_FIT_SPECTRUM_ORDER = 14


def _decentred_fit_spectrum(xs_in, opl_grid, q, orders=(), weight=None):
    """The traced OPL's degree-shell spectrum on the LAUNCH BOX, plus the
    spectral TAIL surrogate beyond each requested order.

    ONE unweighted total-degree-``q`` Chebyshev fit of the traced samples.
    ``S[n]`` is the rms of its coefficients of total degree ``n``; because the
    basis is degree-graded and normalised to the box, restricting to a
    concentric sub-domain of relative radius ``s`` scales the degree-``n``
    contribution by ``s^n``.  That is the entire inflation law.

    ``tails[m]`` is the same fit with every coefficient of total degree
    ``<= m`` zeroed.  A least-squares fit of total degree ``m`` reproduces a
    degree-``<= m`` polynomial EXACTLY, so

        (I - Pi_m) W  ==  (I - Pi_m) W_>m

    identically -- each candidate's residual IS the residual of fitting its own
    spectral tail, and the tail does not move when the beam does.

    Returns ``(S, tails, resid)`` where ``resid`` is the box fit's own rms
    residual against the traced samples under the SAME ``weight`` the
    candidates are scored with: the surrogate is only usable while what is
    left beyond ``q`` is small against what the candidates are being ranked
    by.  ``(None, {}, inf)`` when the fit cannot be built.
    """
    try:
        ev = _Cheb2DEvaluator(xs_in, xs_in, opl_grid, order=int(q))
    except (np.linalg.LinAlgError, ValueError):
        return None, {}, float('inf')
    co = np.asarray(ev.coeffs, dtype=np.float64).ravel()
    deg = np.asarray([a + b for a, b in ev._mi], dtype=np.intp)
    S = np.zeros(int(q) + 1)
    for n in range(int(q) + 1):
        sel = deg == n
        if sel.any():
            S[n] = float(np.sqrt(float((co[sel] ** 2).sum())))
    X, Y = np.meshgrid(xs_in, xs_in, indexing='ij')
    full = np.asarray(ev.ev(X, Y))
    ok = np.isfinite(full) & np.isfinite(opl_grid)
    wt = np.where(ok, (1.0 if weight is None else weight), 0.0)
    tot = float(wt.sum())
    if tot > 0.0:
        r = np.where(ok, full - opl_grid, 0.0)
        resid = float(np.sqrt(float((wt * r * r).sum()) / tot))
        if not np.isfinite(resid):
            resid = float('inf')
    else:
        resid = float('inf')
    tails = {}
    for m in orders:
        c2 = co.copy()
        c2[deg <= int(m)] = 0.0
        ev.coeffs = ev.xp.asarray(c2)
        tails[int(m)] = np.asarray(ev.ev(X, Y))
    ev.coeffs = ev.xp.asarray(co)
    return S, tails, resid


def _decentred_fit_spectral_moment(S, m, sigma):
    """``m_eff``: the spectral first moment of the tail beyond order ``m``,
    weighted at the beam-disc scale ``sigma``.

    It is exactly ``d log T / d log rho`` at ``rho = 1`` for
    ``T(s)^2 = sum_{n>m} (S_n s^n)^2``, i.e. the EXPONENT of the concentric
    candidate's disc-inflation law.  Falls back to ``m + 2`` -- the first shell
    a total-degree-``m`` fit cannot reach on a map with even symmetry -- when
    the tail carries no measurable energy.
    """
    if S is None:
        return float(int(m) + 2)
    S = np.asarray(S, dtype=np.float64)
    num = den = 0.0
    for n in range(int(m) + 1, S.size):
        e = (S[n] * sigma ** n) ** 2
        num += n * e
        den += e
    v = (num / den) if den > 0.0 else float(int(m) + 2)
    return v if (np.isfinite(v) and v > 0.0) else float(int(m) + 2)


def _decentred_fit_crossover(u, e_conc, e_off, m_eff):
    """The crossover decentre ``u* = |c*| / w`` in closed form.

    The concentric candidate's disc is sized from the ORIGIN-referenced second
    moment, so it inflates by ``rho = sqrt(1 + 2 u^2)``; the off-centre one's
    disc and the beam translate together, so its residual is flat in ``u``.
    With the tail's own exponent ``m_eff`` the concentric residual runs as
    ``rho^m_eff``, and the two curves cross where

        rho* = rho(u) (E_off / E_conc)^(1/m_eff),   u* = sqrt((rho*^2 - 1) / 2)

    Returns ``0.0`` when the off-centre candidate already wins at zero
    decentre, and ``inf`` when the concentric one cannot lose (an unscoreable
    off-centre candidate).  ``nan`` when the inputs cannot support a crossover
    at all, which callers treat as "no prediction".
    """
    if not (np.isfinite(u) and u >= 0.0 and np.isfinite(m_eff) and m_eff > 0.0):
        return float('nan')
    if not (np.isfinite(e_conc) and e_conc > 0.0):
        return 0.0
    if not np.isfinite(e_off):
        return float('inf')
    if e_off <= 0.0:
        return 0.0
    rho = float(np.sqrt(1.0 + 2.0 * u * u))
    try:
        rstar = rho * float(e_off / e_conc) ** (1.0 / float(m_eff))
    except (OverflowError, ValueError, ZeroDivisionError):
        return float('nan')
    if not np.isfinite(rstar):
        return float('inf')
    return float(np.sqrt(max(rstar * rstar - 1.0, 0.0) / 2.0))


#: niche C12 (2026-08-03): decide the decentred beam's ray-fit branch from the
#: lens's OWN spectral tail and the disc-inflation law, instead of comparing
#: two numbers and taking the smaller.
#:
#: WHAT NICHE C11 LEFT.  ``DECENTRED_FIT_ARBITER`` builds both candidates and
#: keeps the one with the smaller beam-weighted OPL residual.  That is a
#: measurement, not a model: it cannot say WHY the crossover sits at 0.55 w on
#: an f/3 singlet and at 0 w on an f/6 one, and it says nothing about any
#: decentre other than the one in front of it.
#:
#: THE DERIVATION (docs/audits/C12_PHYSICS_FIT_SELECTION_2026_08_03.md S2).
#: The traced OPL is a fixed function of the ENTRANCE position -- moving the
#: beam moves neither it nor the launch grid.  A total-degree-``m`` fit
#: reproduces the degree-``<= m`` part exactly, so each candidate's residual is
#: the residual of fitting ITS OWN spectral tail, identically.  The tail is
#: decentre-free; the whole ``u``-dependence is therefore geometric:
#:
#:   * the CONCENTRIC disc is sized from the ORIGIN-referenced second moment
#:     ``sqrt(2 c^2 + w^2)``, so it inflates by ``rho = sqrt(1 + 2 u^2)`` and
#:     the same total-degree budget is spread over a disc ``rho`` times bigger;
#:   * the OFF-CENTRE disc and the beam translate TOGETHER, so nothing about
#:     that candidate changes with ``u`` -- its residual is flat.
#:
#: Each shell of the tail scales as ``s^n``, so the concentric residual runs as
#: ``rho^m_eff`` with ``m_eff`` the tail's spectral first moment
#: (:func:`_decentred_fit_spectral_moment`), and the crossover follows in
#: closed form (:func:`_decentred_fit_crossover`).  There is no fitted constant
#: anywhere in it: ``S_n`` is the lens's own measured spectrum, ``sigma`` and
#: ``rho`` are geometry, and the orders and ``_FIT_DISC_OUTSIDE_WEIGHT_REL``
#: are library constants.
#:
#: VALIDATED, three designs, no tuning.  Predicted crossover against the
#: fit-domain-free ``newton_fit='spline'`` oracle's own measured one:
#:
#:     f/3  N-BK7 singlet, w = 1.0 mm    u* = 0.525   measured 0.545
#:     f/6  N-BK7 singlet, w = 1.0 mm    u* = 0       measured 0
#:     f/6  N-BK7 singlet, w = 1.4 mm    u* = 0       measured 0
#:
#: -- and the model reproduces each candidate's MEASURED residual to
#: ``K = 0.61-1.00`` on the two geometries where the residual is truncation- or
#: leak-limited, with no constant.  ``m_eff`` reads 8.000 on all three (the
#: first even shell a degree-6 fit cannot reach), and 7.14-8.04 on all 26 of
#: design 121's arbitrated element calls.
#:
#: WHERE THE SPECTRAL HALF FAILS, measured and stated.  The model needs the
#: traced map to be spectrally RESOLVED on the launch box.  Design 121's groups
#: are not: their launch square is 47 mm against a 6.3 mm beam (7.5:1) with
#: ~0.5 % of it carrying no ray at all, and the box expansion does not converge
#: -- the shells sit flat at ~1e-3 out to degree 20 and the order-14 box fit's
#: own residual over the beam is 1e-5 m, four decades ABOVE the candidate
#: residuals it would have to predict.  So the predictor tests that
#: (``resid <= min(E_conc, E_off)``) and falls back to the MEASURED pair when
#: the spectrum is unresolved.  On that fall-back the closed form is
#: algebraically equivalent to niche C11's comparison, and the check below
#: cannot fire; that is stated rather than hidden.
#:
#: THE ARCHITECTURE.  The predictor DECIDES (``u <= u*``); niche C11's raw
#: comparison always runs as a CHECK; a disagreement raises a ``RuntimeWarning``
#: naming both score pairs, ``u`` and ``u*``.  It is never silent.
#:
#: SHIPPED ON since 5.32.1 (2026-08-03).  C12 shipped it OFF, and the reason
#: is in S5 of that audit and still stands as a STATEMENT: on design 121 no
#: per-call selector can meet a per-order "improve or hold" bar.  Group 2 is
#: preferred CONCENTRIC by 4.77x at (-1,0) (``u`` = 0.062) and by 4.24x at
#: (-2,0) (``u`` = 0.125), and the chain wants OFF-CENTRE at the first and
#: CONCENTRIC at the second -- so no rule monotone in the margin, in ``u``, or
#: in ``u/u*`` can produce that pair.  The chain-level EE3 is not separable
#: across the six groups (measured: the mixed patterns ``ccoo`` 0.069 and
#: ``cooo`` 0.067 are WORSE at (-1,0) than either ``ccco`` 0.055 or ``oooo``
#: 0.029), and a fit-site selector only ever chooses per call.
#:
#: STAYS OFF -- and the 5.32.1 flip of this constant was REVERTED on evidence,
#: 2026-08-03.  ``DECENTRED_FIT_ARBITER`` did ship ``True``; this one did not,
#: and the two must not be confused because only ONE of them was implicated.
#:
#: THE MEASUREMENT THAT REVERTED IT (niche C13 S11).  Turning this flag on
#: reddens **9 tests** in ``test_niche_d6_exact_tilted_leg`` and
#: ``test_niche_d7_decentred_fit``, and every one of them goes green again with
#: this flag ``False`` and the ARBITER still ``True`` -- so the arbiter is
#: innocent and the predictor is the whole cause.  On niche D6's ``K = -n^2``
#: conic stand-in, whose truth is ANALYTIC and decentre-invariant (an inline
#: exact conic raytrace sharing no code with the library):
#:
#:     PREDICTOR   EE2/oracle   FWHM/oracle   spot off the Fermat focus
#:     True          0.6670        1.0952            3.96e-07 m
#:     False         0.9819        1.0000            9.87e-09 m
#:
#: -- a 32 % loss of encircled energy and a spot 40x further off the
#: analytically known focus.  The library WARNS on that very call and then
#: applies the losing choice: at ``u`` = 1.0001 the model reads
#: ``u*`` = 1.4161 from ``E_c`` = 1.94e-07 against ``E_o`` = 1.51e-06 and picks
#: CONCENTRIC, while the arbiter's MEASURED residuals on the same call are
#: 8.70e-11 off-centre against 1.51e-06 concentric -- **off-centre is 17,000x
#: better and the model has the ordering inverted**.
#:
#: WHY C12's OWN VALIDATION DID NOT CATCH IT.  S3.2 validated three geometries
#: (f/3 and f/6 singlets at two beam radii) and the model landed on the oracle
#: to 0.03 % on the one with a nonzero crossover.  D6's stand-in is a FOURTH
#: geometry, at ``u`` = 1.0 rather than the ~0.57 those crossovers sat at, and
#: the model does not generalise to it.  Three designs is not enough for a
#: closed form that ships as a default.
#:
#: AND IT BUYS NOTHING ON DESIGN 121.  S3.4 already says the launch-box
#: spectrum there is UNRESOLVED, so the predictor falls back to the measured
#: pair and is algebraically the arbiter.  Measured, not inferred: across four
#: design-121 runs with the flag ON (both builds, ``rc_resdeg_121``,
#: ``focus_scan_121``, ``energy_stage_audit_121``) the predictor/arbiter
#: disagreement warning fires **zero times**, and the per-order table
#: reproduces C11's arbiter column to all four decimals at all six orders.
#: The flag is inert where it was wanted and harmful where it is live.
#:
#: The original argument for shipping it OFF (S5 of the C12 audit) is
#: unchanged and still stands: on design 121 no per-call selector can meet a
#: per-order "improve or hold" bar.  Group 2 is preferred CONCENTRIC by 4.77x
#: at (-1,0) (``u`` = 0.062) and by 4.24x at (-2,0) (``u`` = 0.125), and the
#: chain wants OFF-CENTRE at the first and CONCENTRIC at the second.
#:
#: FALL-BACK LADDER (see :data:`DECENTRED_FIT_ARBITER`): with this ``False``
#: the C11 arbiter decides, which is what 5.32.1 ships; both ``False`` is the
#: v5.32 gate, bit for bit.
DECENTRED_FIT_PREDICTOR = False


def _input_beam_amp_radius(E_in, dx, dy=None, centre=None,
                           origin=(0.0, 0.0)):
    """1/e AMPLITUDE radius of ``E_in`` on the centred wave grid, from the
    intensity second moment (``w = sqrt(2 <r^2>)`` -- the same convention as
    :func:`lumenairy.propagators.carrier._envelope_amp_radius`, so the chain's
    per-stage ``w`` and this element-side measurement agree).

    ``centre=(x0, y0)`` (metres) measures the second moment about that
    transverse point instead of the GRID ORIGIN -- required whenever the beam
    is decentred (the niche-D1 tilted-carrier hand-off puts the beam at its
    physical chief-ray position ``(x_c, y_c)``).  About the origin a beam of
    true radius ``w`` sitting at ``x_c`` reads ``sqrt(2 x_c^2 + w^2)``, so the
    P2 aperture:beam guard that is SIZED from this number silently stops
    guarding as the decentre grows.  ``None`` / ``(0, 0)`` leaves the grid
    arrays untouched (byte-identical default).

    ``origin=(x0, y0)`` (niche D9) states that the grid's CENTRE pixel sits at
    that transverse point in the element's absolute (optical-axis) frame, so it
    is ADDED to the grid axes BEFORE ``centre`` is subtracted -- ``centre`` is
    quoted in the absolute frame (a ``TiltedCarrier``'s chief-ray position),
    while the axes are grid-relative.  Getting this wrong does not raise: the
    moment is then taken about ``centre - origin``, so the P2 aperture:beam
    guard this number sizes silently stops guarding.  ``(0, 0)`` leaves the
    axes untouched (byte-identical default).

    Returns ``0.0`` for an empty / zero / non-finite field (callers treat that
    as "unmeasurable" and skip the beam-relative guard).

    Accumulated in row BANDS so no full-grid ``|E|^2`` temporary is
    materialised (this runs on every traced call at the default
    ``on_aperture_beam='warn'``, where an N=8192 whole-grid float64 temporary
    would cost 0.5 GiB)."""
    E = np.asarray(E_in)
    if E.ndim != 2 or E.size == 0:
        return 0.0
    Ny, Nx = E.shape[-2], E.shape[-1]
    _dy = float(dy) if (dy is not None and np.isfinite(dy) and dy > 0) else dx
    xg = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yg = (np.arange(Ny, dtype=np.float64) - Ny / 2) * _dy
    if origin is not None and (origin[0] or origin[1]):
        # niche D9: grid axis -> ABSOLUTE transverse coordinate, before the
        # (absolute) ``centre`` is subtracted below.
        xg = xg + float(origin[0])
        yg = yg + float(origin[1])
    if centre is not None:
        _cx, _cy = float(centre[0]), float(centre[1])
        if _cx:
            xg = xg - _cx
        if _cy:
            yg = yg - _cy
    x2 = xg * xg
    tot = 0.0
    acc = 0.0
    band = max(1, min(Ny, int(1 << 22) // max(Nx, 1)))
    for r0 in range(0, Ny, band):
        r1 = min(Ny, r0 + band)
        Ib = np.abs(E[r0:r1]) ** 2
        rows = Ib.sum(axis=1)
        tot += float(rows.sum())
        acc += float((Ib @ x2).sum()) + float(rows @ (yg[r0:r1] ** 2))
    if not np.isfinite(tot) or tot <= 0.0 or not np.isfinite(acc):
        return 0.0
    r2 = acc / tot
    if not np.isfinite(r2) or r2 <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * r2))


def _carrier_residual_rms(E_in, W_full, wavelength, dx):
    """RMS transverse angular spread (radians) of ``E_in`` AFTER removing
    the carrier wavefront ``W_full`` (length units; ``None`` -> no carrier).

    This is the discriminator for the F1 collimation guard: a beam that is
    a single smooth carrier plus a small angular residual (an emitter array,
    a diverging source) has a SMALL residual once the carrier is subtracted,
    even though its raw angular spread is large -- so it is well within the
    carrier-referenced traced model's validity.  Uses the wrapping-safe
    nearest-neighbour phase-increment estimator.
    """
    k0 = 2.0 * np.pi / wavelength
    E = np.asarray(E_in)
    if W_full is not None:
        E = E * np.exp(-1j * k0 * np.asarray(W_full))
    mag = np.abs(E)
    mx = mag.max()
    if not np.isfinite(mx) or mx <= 0:
        return 0.0
    mask = mag > 0.05 * mx
    del mag
    if not mask.any():
        return 0.0
    gx = E[:, 1:] * np.conj(E[:, :-1])
    lx = (np.angle(gx) / (k0 * dx))[mask[:, 1:] & mask[:, :-1]]
    del gx
    gy = E[1:, :] * np.conj(E[:-1, :])
    my = (np.angle(gy) / (k0 * dx))[mask[1:, :] & mask[:-1, :]]
    del gy
    if lx.size == 0 or my.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(lx ** 2) + np.mean(my ** 2)))


def _input_tilt_stats(E_in, wavelength, dx):
    """Wrapping-safe transverse tilt statistics of ``E_in`` over its bright
    support: ``(tilt_rms, coherence_ratio)`` (radians / dimensionless), or
    ``None`` when they cannot be formed (empty / degenerate field).

    ``tilt_rms`` is the SAME quantity two collimated-input diagnostics need and
    used to compute independently -- the ``carrier=None`` residual angular
    spread (``_carrier_residual_rms(E_in, None, ...)``) AND the
    ``tilt_aware_rays=False`` launch-warning discriminator -- so this single
    pass replaces the duplicate full-grid ``angle(E[i+1]*conj(E[i]))`` +
    ``np.angle`` computation that ran twice per call (~9.5% of the runtime at
    N=4096).  The arithmetic is byte-identical to both original sites (same
    ``0.05*max`` bright mask, same nearest-neighbour phase increments, same
    ``sqrt(mean(lx^2)+mean(my^2))`` / ``hypot(mean(lx),mean(my))/tilt_rms``)."""
    k0 = 2.0 * np.pi / wavelength
    E = np.asarray(E_in)
    mag = np.abs(E)
    mx = mag.max()
    if not np.isfinite(mx) or mx <= 0:
        return None
    mask = mag > 0.05 * mx
    del mag
    if not mask.any():
        return None
    gx = E[:, 1:] * np.conj(E[:, :-1])
    lx = (np.angle(gx) / (k0 * dx))[mask[:, 1:] & mask[:, :-1]]
    del gx
    gy = E[1:, :] * np.conj(E[:-1, :])
    my = (np.angle(gy) / (k0 * dx))[mask[1:, :] & mask[:-1, :]]
    del gy
    if lx.size == 0 or my.size == 0:
        return None
    tilt_rms = float(np.sqrt(np.mean(lx ** 2) + np.mean(my ** 2)))
    coherent = float(np.hypot(np.mean(lx), np.mean(my)))
    coherence_ratio = coherent / tilt_rms if tilt_rms > 0 else 1.0
    return (tilt_rms, coherence_ratio)


#: Niche C5 (2026-07-30).  ``True`` -- a tilted congruence is referenced to
#: the EXACT eikonal of a displaced point source (see :class:`TiltedCarrier`).
#: ``False`` restores the pre-C5 SPHERE-PLUS-LINEAR-RAMP form
#: ``S_R(rho) + L*u + M*v`` exactly, in the element AND in
#: :func:`lumenairy.propagate_traced_carrier_chain` (which reads this same
#: flag through
#: :func:`~lumenairy.propagators.carrier._exact_tilt_reference`, a helper that
#: imports and returns THIS constant at call time, so the
#: two can never be configured apart -- they only make sense together).  The
#: fail-before switch for niche C5: flip it to reproduce any pre-C5 tilted
#: result bit for bit.  It has NO effect on an untilted congruence -- the two
#: eikonals are the same function when ``L == M == 0`` -- so the on-axis path
#: is byte-identical either way.
TILTED_CARRIER_EXACT_EIKONAL = True


class TiltedCarrier(NamedTuple):
    """The exact reference congruence of a POINT SOURCE whose chief ray passes
    through ``(x0, y0)`` with direction cosines ``(L, M)`` (niche D1, roadmap
    ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1a; eikonal corrected in
    niche C5, 2026-07-30).

    The eikonal (metres; reference phase ``k0 * W``) is

    .. code-block:: text

        u = x - x0,  v = y - y0,  N = sqrt(1 - L^2 - M^2)
        W(x, y) = sign(R) * ( sqrt((u + R L/N)^2 + (v + R M/N)^2 + R^2)
                              - |R|/N )

    -- the same exact on-axis point-source sphere
    ``S_R(rho) = sign(R)(sqrt(rho^2+R^2)-|R|)`` of :func:`_compute_carrier`'s
    scalar branch, transversely re-centred on the SOURCE's own projection
    ``(x0 - R L/N, y0 - R M/N)``, plus a constant.  ``R`` is the signed AXIAL
    distance to that source (``R > 0`` diverging, i.e. behind the plane).
    ``W(x0, y0) == 0`` and ``grad W(x0, y0) == (L, M)`` exactly.
    ``R = +/-inf`` degenerates to the pure tilted PLANE ``W = L*u + M*v`` (the
    post-DOE order at a pupil), which is exactly what a scalar ``carrier=inf``
    cannot express.  ``L == M == 0`` reduces to ``S_R`` term for term, so an
    untilted congruence is unaffected.

    **Why not sphere-plus-ramp.**  Through v5.31 this was
    ``S_R(sqrt(u^2+v^2)) + L u + M v`` -- an on-axis sphere with a linear ramp
    ADDED.  That is not a solution of the eikonal equation; it differs from
    the exact form by

    .. code-block:: text

        -(L u + M v)(u^2 + v^2)/(2 R^2)  -  (L u + M v)^2/(2 R)  +  ...

    i.e. COMA linear in the field angle plus ASTIGMATISM quadratic in it.  On
    design 121's last coarse leg (R = -24.46 mm, tilt 54.9 mrad, beam radius
    3.63 mm, lambda 1.31 um) that reaches 2.5 waves within one beam radius and
    2.0 mrad of launch-direction error.

    It is fatal on the CHAIN side, which defines its envelope as the field
    with this reference divided out and then transports that envelope by a
    Sziklas-Siegman step: a reference that is not a wavefront dumps the term
    above into the "envelope", where a plain dilation ``du -> m du`` cannot
    carry it.  Measured in closed form
    (``validation/repro_traced_carrier_121/probe_leg_exactness.py``): leg
    error **0.136 waves rms** before, **1e-5** after, against an untilted
    control of 1e-5.  End to end that one leg was worth **20.7 EE3 points** at
    DOE order (-4,-2).

    ``R`` is AXIAL, which is what makes the fix local: a free leg still
    advances it by the AXIAL gap ``R -> R + z``, the Sziklas-Siegman
    magnification is still ``(R+z)/R``, and
    :func:`~lumenairy.propagators.carrier._paraxial_group_r_out`'s Moebius law
    still returns the paraxial AXIAL image distance.  NOTHING about the
    transport changes -- only the shape of the reference wavefront.

    Set :data:`TILTED_CARRIER_EXACT_EIKONAL` to ``False`` to restore the
    pre-C5 form everywhere (element and chain together -- a MIXED pair is
    measurably worse than either, since the element would then de-chirp with
    a reference the field is no longer written against).

    Why it exists: a post-DOE fan is K comparable-power beams at well-separated
    angles, and ``apply_real_lens_traced``'s entrance->exit map assumes ONE
    congruence per exit pixel (see the ``carrier`` docstring's validity
    paragraph).  Taken ONE ORDER AT A TIME each beam IS a single clean
    congruence -- but only if the reference can carry its tilt, which a scalar
    conjugate distance cannot.  With this spec each order's residual after the
    carrier is removed is the same small diffraction residual the on-axis case
    has, so the ``_NONCOLLIMATED_RESID_THRESH`` envelope is respected per order
    instead of being violated by the full split angle.

    Closed under paraxial transfer: through an air-to-air ABCD the state
    ``(R, L, x0)`` maps to ``R' = (A R + B)/(C R + D)`` (the wavefront Moebius
    law :func:`lumenairy.propagators.carrier._paraxial_group_r_out` already
    uses) with the CHIEF RAY ``(x0, L)`` transforming as an ordinary paraxial
    ray, ``x0' = A x0 + B L``, ``L' = C x0 + D L`` -- which is what lets
    :func:`lumenairy.propagate_traced_carrier_chain` carry it hand-off by
    hand-off (pass ``r_in=TiltedCarrier(...)``).

    Attributes
    ----------
    R : float
        Signed sphere radius (m); ``> 0`` diverging in front of the plane,
        ``+/-inf`` collimated (pure tilt).
    L, M : float
        Transverse direction cosines of the uniform tilt (rad, paraxially).
    x0, y0 : float, default 0
        Transverse centre of the congruence (m) -- the chief-ray position at
        this plane.  The tilt is referenced to it, so ``W(x0, y0) == 0``.
    """

    R: float
    L: float = 0.0
    M: float = 0.0
    x0: float = 0.0
    y0: float = 0.0

    @property
    def is_tilted(self) -> bool:
        """True when this spec carries anything a scalar ``carrier=R``
        cannot (a tilt or a transverse decentre)."""
        return bool(self.L or self.M or self.x0 or self.y0)


def _tilted_carrier_parts(spec, X, Y):
    """``(W, L_grad, M_grad)`` for a :class:`TiltedCarrier` on the query
    coordinates ``X, Y`` (arrays or scalars).  Analytic -- no finite
    differences, no grid lookup -- so the ray launch, the H6 entrance eikonal
    and the ``exp(i k0 W)`` reference leg all see the SAME congruence to all
    orders (the ndarray-carrier branch can only offer ``np.gradient`` +
    nearest-neighbour, which quantises the launch cosines to the grid)."""
    s = float(spec.R)
    L, M = float(spec.L), float(spec.M)
    u = X - float(spec.x0)
    v = Y - float(spec.y0)
    if s == 0.0:
        raise ValueError("TiltedCarrier: R == 0 (the congruence's own focus)")
    if not np.isfinite(s):
        # collimated limit: the pure tilted PLANE (the |R| -> inf limit of the
        # sphere).  Evaluating the closed form here would give inf - inf = NaN
        # over the whole grid, and an all-NaN eikonal is a SILENT sentinel that
        # disables the engage test (same trap the scalar branch documents).
        W = L * u + M * v
        return W, np.full_like(W, L), np.full_like(W, M)
    sgn = 1.0 if s > 0.0 else -1.0
    if (L == 0.0 and M == 0.0) or not TILTED_CARRIER_EXACT_EIKONAL:
        # Untilted (the two conventions are the same function there) or the
        # C5 fail-before switch: the historical sphere-PLUS-RAMP expression,
        # verbatim, so the on-axis path is byte-identical.
        rho = np.sqrt(u * u + v * v + s * s)
        # RATIONALIZED: rho - |s| == (u^2+v^2) / (rho + |s|).  Algebraically
        # identical; the subtraction loses k0*eps*|s| radians to catastrophic
        # cancellation wherever u^2+v^2 << s^2.  Same fix and same reason as
        # a185cfc in propagators/carrier.py.  The "byte-identical" claim
        # above is against the C5 fail-before ARM of this same function,
        # which is rationalized identically below, so the two conventions
        # still coincide exactly at L = M = 0.
        W = sgn * ((u * u + v * v) / (rho + abs(s))) + L * u + M * v
        return W, sgn * u / rho + L, sgn * v / rho + M
    # niche C5: the EXACT displaced-point-source eikonal.  ``s`` is the AXIAL
    # distance to the source, so the source's transverse projection sits at
    # ``-s (L, M)/N`` and the sphere radius is ``s`` itself.  Evaluated as
    # that same on-axis sphere SHIFTED -- never on a grid centred on the
    # projection, which is metres away for a near-collimated carrier.
    _N = np.sqrt(1.0 - L * L - M * M) if (L * L + M * M) < 1.0 else 0.0
    if _N == 0.0:
        raise ValueError(
            f"TiltedCarrier: L^2 + M^2 = {L * L + M * M:.6g} >= 1, i.e. "
            f"(L, M) is not a propagating direction.  L and M are DIRECTION "
            f"COSINES (sin of the angle), not slopes.")
    uu = u + s * L / _N
    vv = v + s * M / _N
    rho = np.sqrt(uu * uu + vv * vv + s * s)
    # RATIONALIZED -- see the untilted arm above.  The difference of squares
    # collapses ANALYTICALLY here: with _N^2 = 1 - L^2 - M^2 the s^2/_N^2
    # terms cancel against s^2 exactly, leaving u^2+v^2+2s(Lu+Mv)/_N, which
    # carries no large term at all.
    W = sgn * ((u * u + v * v + 2.0 * s * (L * u + M * v) / _N)
               / (rho + abs(s) / _N))
    return W, sgn * uu / rho, sgn * vv / rho


def _compute_carrier(carrier, E_in, wavelength, dx, X, Y, auto_degree=2,
                     origin=(0.0, 0.0), need_W=True):
    """Build the carrier reference wavefront ``W(x, y)`` (length units;
    reference phase = ``k0 * W``) and a callable giving its transverse
    gradient -- the ray direction cosines ``L = dW/dx``, ``M = dW/dy``.

    ``carrier`` accepts:

    * ``float`` -- an on-axis point-source conjugate at signed distance
      ``s`` (metres): the EXACT spherical wavefront
      ``W = sign(s)(sqrt(x^2+y^2+s^2) - |s|)`` (R7 / audit F2; ``s > 0``
      for a diverging source in front of the plane).  The exact sphere --
      not the paraxial ``r^2/(2s)`` -- so the reference leg, ray-launch
      cosines and H6 entrance eikonal match the true congruence on a STEEP
      conjugate (reduces to the parabola to ~1e-10 when ``r/|s| << 1``).
      ``+/-inf`` (a COLLIMATED conjugate -- the chain's documented
      ``r_in=inf`` launch default) returns the plane-wave limit ``W == 0``
      with zero gradient, so the caller keeps the byte-identical
      plane-wave-reference path; the closed form would give an ALL-NaN
      ``W`` there, which silently disabled the engage test instead.
    * ``ndarray`` -- an explicit wavefront (metres), same shape as ``E_in``.
    * ``'auto'`` -- a low-order (``auto_degree``) polynomial fit of the
      smooth carrier, obtained by least-squares matching the polynomial's
      GRADIENT to the wrapping-safe local tilt field of ``E_in`` over its
      bright support (never per-pixel gradients -- that is F4's failure).
      Curl-free by construction (a scalar potential is fit, not L/M
      separately).

    Returns ``(W_full, grad_fn, w_fn)`` where ``W_full`` is an ``(N, N)``
    array, ``grad_fn(xq, yq)`` returns ``(L, M)`` at the query positions,
    and ``w_fn(xq, yq)`` evaluates the carrier eikonal ``W`` (metres) at
    the query positions -- v5.25.1 (hammer H6): the per-ray OPL must be
    referenced to the carrier congruence by ADDING ``W(x_in)`` at the
    entrance plane; omitting it collapsed every diverging-input trace to
    the collimated focal plane.

    ``origin=(x0, y0)`` (niche D9) is the ABSOLUTE transverse position of the
    grid's CENTRE pixel.  ``X`` / ``Y`` already carry it (the caller builds
    them), and ``grad_fn`` / ``w_fn`` are always queried at ABSOLUTE positions
    (ray-launch heights on the axis-centred launch grid), so the two branches
    that go back to grid INDICES -- the ndarray-wavefront lookup and the
    ``'auto'`` fit's sample coordinates -- are the only ones that need it.  A
    :class:`TiltedCarrier` needs nothing: it states its own chief-ray position
    in the same absolute frame and subtracts it from ``X`` / ``Y`` itself, so
    the origin must NOT also be removed from ``carrier.x0`` (that
    double-subtraction is exactly right at the grid centre and wrong in the
    wings -- the failure has no symptom at the beam peak).

    ``need_W=False`` (v5.40) returns ``None`` in place of ``W_full`` and skips
    building it.  ``W_full`` is a FULL ``(N, N)`` float64 grid -- 8.59 GB at
    N = 32768 -- and most callers here discard it immediately; the ones that
    use it keep the default.  With it off, ``X`` and ``Y`` are only ever read
    for their SHAPE, so a caller may pass zero-copy ``np.broadcast_to`` views
    of the two axis vectors and the whole coordinate stack disappears with it.
    ``grad_fn`` / ``w_fn`` are unaffected: they are pointwise closures over
    the same coefficients, so a band evaluated through them is bit-for-bit the
    corresponding slice of the whole-grid answer.
    """
    N = X.shape[0]
    _org_x, _org_y = float(origin[0]), float(origin[1])
    if isinstance(carrier, TiltedCarrier):
        # niche D1: exact sphere + uniform tilt about (x0, y0), evaluated
        # ANALYTICALLY everywhere (grid, ray-launch heights, H6 eikonal).
        W_full = (_tilted_carrier_parts(carrier, X, Y)[0]
                  if need_W else None)

        def grad_fn(xq, yq):
            _, Lq, Mq = _tilted_carrier_parts(carrier, xq, yq)
            return Lq, Mq

        def w_fn(xq, yq):
            Wq, _, _ = _tilted_carrier_parts(carrier, xq, yq)
            return Wq

        return W_full, grad_fn, w_fn

    if isinstance(carrier, np.ndarray):
        W_full = np.asarray(carrier, dtype=np.float64)
        if W_full.shape != X.shape:
            raise ValueError(
                f"carrier ndarray shape {W_full.shape} != field shape "
                f"{X.shape}")
        gWy, gWx = np.gradient(W_full, dx, dx)

        # niche D9: ABSOLUTE query position -> grid index.  ``- 0.0`` is the
        # exact IEEE identity, so the on-axis lookup is unchanged bit for bit.
        def grad_fn(xq, yq):
            fx = np.clip((xq - _org_x) / dx + N / 2.0, 0, N - 1).astype(np.int64)
            fy = np.clip((yq - _org_y) / dx + N / 2.0, 0, N - 1).astype(np.int64)
            return gWx[fy, fx], gWy[fy, fx]

        def w_fn(xq, yq):
            fx = np.clip((xq - _org_x) / dx + N / 2.0, 0, N - 1).astype(np.int64)
            fy = np.clip((yq - _org_y) / dx + N / 2.0, 0, N - 1).astype(np.int64)
            return W_full[fy, fx]

        return W_full, grad_fn, w_fn

    if isinstance(carrier, str):
        if carrier != 'auto':
            raise ValueError(
                f"carrier string must be 'auto', got {carrier!r}")
        # Wrapping-safe local tilt field over the bright support.
        k0 = 2.0 * np.pi / wavelength
        E = np.asarray(E_in)
        mag = np.abs(E)
        mask = mag > 0.05 * mag.max()
        gx = E[:, 1:] * np.conj(E[:, :-1])
        Lx = np.angle(gx) / (k0 * dx)
        del gx
        gy = E[1:, :] * np.conj(E[:-1, :])
        My = np.angle(gy) / (k0 * dx)
        del gy
        # R6 / audit F1: build the CONNECTED un-aliased core mask so the
        # gradient fit sees only samples whose local tilt reading is below the
        # grid Nyquist tilt (i.e. NOT wrapped).  The per-pixel tilt magnitude is
        # the vector norm of the two nearest-neighbour phase increments; where it
        # exceeds ``_AUTO_CARRIER_NYQUIST_FRAC * (lambda/2dx)`` the reading is
        # (approaching) aliased.  The restriction engages only when a non-trivial
        # fraction (``_AUTO_CARRIER_ALIAS_FRAC``) of the bright support is
        # aliased; then connected-component labelling keeps only the component
        # containing the BRIGHTEST pixel (the beam centre): the central parabola
        # whose curvature fixes the spherical R.  Wrapped rings past the first
        # Nyquist crossing are separate components (a high-tilt annulus
        # disconnects them) and are excluded.  The brightest-pixel seed is
        # essential -- the min-tilt point can land on a wrapped-to-zero alias
        # ring on a coarse grid, seeding an off-centre blob that injects a
        # spurious tilt.  On a well-sampled input (~no aliasing) ``core`` stays
        # the full bright support so the fit is byte-identical to before and a
        # disconnected multi-emitter field is NOT collapsed onto one beamlet.
        core = mask
        if mask.any():
            _gphx = np.angle(np.roll(E, -1, axis=1) * np.conj(E)) / (k0 * dx)
            _gphy = np.angle(np.roll(E, -1, axis=0) * np.conj(E)) / (k0 * dx)
            _gmag = np.hypot(_gphx, _gphy)
            del _gphx, _gphy
            _nyq_tilt = wavelength / (2.0 * dx)
            _core_ok = mask & (_gmag < _AUTO_CARRIER_NYQUIST_FRAC * _nyq_tilt)
            del _gmag
            _n_bright = int(mask.sum())
            _n_aliased = _n_bright - int(_core_ok.sum())
            if (_n_aliased > _AUTO_CARRIER_ALIAS_FRAC * max(_n_bright, 1)
                    and _core_ok.any()):
                from scipy.ndimage import label as _ndlabel
                _lbl, _nlbl = _ndlabel(_core_ok)
                if _nlbl > 0:
                    _seed_lbl = int(_lbl.ravel()[int(mag.ravel().argmax())])
                    if _seed_lbl > 0:
                        _cand = _lbl == _seed_lbl
                        if int(_cand.sum()) >= _AUTO_CARRIER_MIN_CORE:
                            core = _cand
                del _lbl
            del _core_ok
        mxx = core[:, 1:] & core[:, :-1]
        myy = core[1:, :] & core[:-1, :]
        # sample coords at the increment midpoints.  niche D9: the two axes
        # separate once the grid centre is off axis (``x0 != y0`` breaks the
        # shared-vector structure, not merely the offset); ``yax is xax`` on
        # axis, so the meshgrid call is byte-identical there.
        xax = (np.arange(N) - N / 2.0) * dx
        yax = xax
        if _org_x or _org_y:
            yax = xax + _org_y
            xax = xax + _org_x
        # v5.40: the sample coordinates are SEPARABLE, and materialising them
        # was the fit's largest fixed cost.  ``np.meshgrid`` built two full
        # (N, N) float64 grids and each midpoint expression built a third
        # before the mask threw all but the bright support away -- four
        # float64 grids, 34 GB at N = 32768, to produce six 1-D vectors.
        #
        # Bit-identical, because the identity is exact rather than close:
        # ``Xg[i, j] == xax[j]`` and ``Yg[i, j] == yax[i]`` for every (i, j),
        # so ``0.5 * (Xg[:, 1:] + Xg[:, :-1])[i, j] == 0.5 * (xax[j+1] +
        # xax[j])`` -- the SAME two IEEE operands in the same order -- and
        # likewise down the columns.  Boolean-masking a zero-strided
        # ``broadcast_to`` view selects only the elements the mask keeps, so
        # the (N, N) intermediate never exists at all.
        _xmid = 0.5 * (xax[1:] + xax[:-1])
        _ymid = 0.5 * (yax[1:] + yax[:-1])
        _shp_L = (yax.size, xax.size - 1)
        _shp_M = (yax.size - 1, xax.size)
        xL = np.broadcast_to(_xmid[None, :], _shp_L)[mxx]
        yL = np.broadcast_to(yax[:, None], _shp_L)[mxx]
        Lv = Lx[mxx]
        xM = np.broadcast_to(xax[None, :], _shp_M)[myy]
        yM = np.broadcast_to(_ymid[:, None], _shp_M)[myy]
        Mv = My[myy]
        # Intensity weights: on a fringed multi-source field the local tilt
        # is noisy, so weight each sample by the local |E| (bright regions
        # -- the imaged carrier -- dominate the low-order fit and the
        # fringe noise averages out).  This is why the fit is robust where
        # per-pixel tilts (F4) fail.
        magI = np.abs(E)
        wL = 0.5 * (magI[:, 1:] + magI[:, :-1])[mxx]
        wM = 0.5 * (magI[1:, :] + magI[:-1, :])[myy]
        del magI
        # Polynomial basis terms b_k(x,y)=x^i y^j (i+j<=deg, i+j>=1); fit the
        # scalar potential W = sum c_k b_k by matching grad(W) to (Lv, Mv).
        terms = [(i, j) for d in range(1, auto_degree + 1)
                 for i in range(d + 1) for j in [d - i]]
        nL, nM = xL.size, xM.size
        A = np.zeros((nL + nM, len(terms)))
        for k, (i, j) in enumerate(terms):
            # d/dx of x^i y^j = i x^(i-1) y^j ; d/dy = j x^i y^(j-1)
            A[:nL, k] = (i * xL ** (i - 1) * yL ** j) if i >= 1 else 0.0
            A[nL:, k] = (j * xM ** i * yM ** (j - 1)) if j >= 1 else 0.0
        rhs = np.concatenate([Lv, Mv])
        w = np.concatenate([wL, wM])
        # v5.40: weight IN PLACE.  ``A`` is the largest array in the fit --
        # ``(2 * n_bright, n_terms)`` float64, which on a design-121 group at
        # N = 4096 with 21% bright support is 2.1 float64 grids -- and
        # ``A = A * w[:, None]`` doubled it for the duration of the multiply.
        # ``np.multiply(A, w[:, None], out=A)`` is the same elementwise
        # product on the same operands, so the coefficients are bit-identical.
        A *= w[:, None]
        rhs = rhs * w
        # B7: normal-equations solve (thread-safe; no gelsd/JAX-OpenMP deadlock).
        coef = _solve_lstsq_thread_safe(A, rhs)

        def _poly_and_grad(xq, yq, want_W=True):
            # v5.40: ``want_W=False`` skips the potential itself.  ``grad_fn``
            # never reads it, and ``Wq`` accumulates into its own array from
            # its own terms, so dropping it cannot move a bit of the gradient.
            # It saves one full-grid float64 array plus its per-term
            # temporaries on every gradient evaluation.
            Wq = np.zeros_like(xq, dtype=np.float64) if want_W else None
            Lq = np.zeros_like(xq, dtype=np.float64)
            Mq = np.zeros_like(xq, dtype=np.float64)
            for k, (i, j) in enumerate(terms):
                if want_W:
                    Wq += coef[k] * xq ** i * yq ** j
                if i >= 1:
                    Lq += coef[k] * i * xq ** (i - 1) * yq ** j
                if j >= 1:
                    Mq += coef[k] * j * xq ** i * yq ** (j - 1)
            return Wq, Lq, Mq

        W_full = _poly_and_grad(X, Y)[0] if need_W else None

        def grad_fn(xq, yq):
            _, Lq, Mq = _poly_and_grad(xq, yq, want_W=False)
            return Lq, Mq

        def w_fn(xq, yq):
            Wq, _, _ = _poly_and_grad(xq, yq)
            return Wq

        return W_full, grad_fn, w_fn

    # scalar conjugate distance -- an on-axis point-source conjugate at signed
    # distance ``s``.  R7 / audit F2 (2026-07-21): use the EXACT spherical
    # wavefront ``W = sign(s)(sqrt(r^2 + s^2) - |s|)``, NOT the paraxial parabola
    # ``r^2/(2s)``.  The exact sphere is what an on-axis point source physically
    # radiates and what the meridional-oracle input eikonal is; the paraxial form
    # drops an ``-r^4/(8 s^3)`` term that, on a STEEP conjugate (``r/|s|`` not
    # small -- e.g. the 121's S25-S27, ``w/R = 0.15``), leaves several radians of
    # spurious r^4 in the exit wavefront because the carrier eikonal (H6), the
    # reference leg ``exp(i k0 W)`` and the ray-launch cosines ``grad W`` no
    # longer match the true diverging/converging sphere the wave model carries.
    # With the exact sphere the three carrier legs agree with the ray trace to
    # all orders (per-group r^4 residual 2.7 rad -> ~0 on S25-S27).  Reduces to
    # the paraxial form to ~1e-10 for a gentle conjugate (``r/|s| << 1``), so
    # every previously-validated gentle-carrier call is unchanged in practice.
    s = float(carrier)
    if s == 0.0:
        raise ValueError("carrier conjugate distance must be non-zero")
    if not np.isfinite(s):
        # COLLIMATED conjugate (``carrier=+/-inf``): the analytic |s| -> inf
        # limit of the sphere is the PLANE WAVE, ``W == 0`` with zero
        # gradient.  Evaluating the closed form would give ``inf - inf =
        # NaN`` over the whole grid, and that all-NaN eikonal is a SILENT
        # SENTINEL, not an error: ``np.nanmax`` on it returns NaN (plus an
        # "All-NaN slice encountered" RuntimeWarning), the engage test
        # ``NaN > _TILT_EIKONAL_MIN_RAD`` is False, and the carrier
        # machinery -- including the R7 fit-domain restriction -- is
        # skipped with no diagnostic (audit
        # AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4 traced the
        # aperture:beam cliff to exactly that path; the warning was its
        # only visible symptom).  ``inf`` is a LEGITIMATE value here --
        # ``propagate_traced_carrier_chain(r_in=inf)`` is the documented
        # collimated-launch default -- so return the correct limit
        # explicitly.  The engage test then reads ``_peakW = 0`` and the
        # call keeps the byte-identical plane-wave-reference path it
        # already took via the NaN, minus the two spurious warnings
        # (verified: ``carrier=inf`` and ``carrier=None`` outputs are
        # byte-equal before and after).
        W_full = (np.zeros_like(np.asarray(X, dtype=np.float64))
                  if need_W else None)

        def grad_fn(xq, yq):
            return np.zeros_like(xq, dtype=np.float64), \
                np.zeros_like(yq, dtype=np.float64)

        def w_fn(xq, yq):
            return np.zeros_like(xq, dtype=np.float64)

        return W_full, grad_fn, w_fn
    _sgn = 1.0 if s > 0.0 else -1.0
    _abs_s = abs(s)
    # RATIONALIZED: sqrt(r^2+s^2) - |s| == r^2 / (sqrt(r^2+s^2) + |s|).  Same
    # fix and same reason as a185cfc in propagators/carrier.py.  This one
    # feeds the ray launch, the H6 entrance eikonal AND the exp(i k0 W)
    # reference leg, so the k0*eps*|s| error it used to carry was COHERENT
    # across all three.
    if need_W:
        _r2_full = X ** 2 + Y ** 2
        W_full = _sgn * (_r2_full / (np.sqrt(_r2_full + s * s) + _abs_s))
        del _r2_full
    else:
        W_full = None

    def grad_fn(xq, yq):
        _rho = np.sqrt(xq * xq + yq * yq + s * s)
        return _sgn * xq / _rho, _sgn * yq / _rho

    def w_fn(xq, yq):
        _r2 = xq * xq + yq * yq
        return _sgn * (_r2 / (np.sqrt(_r2 + s * s) + _abs_s))

    return W_full, grad_fn, w_fn


# ---------------------------------------------------------------------------
# niche C6 (2026-07-30): the stationary-phase launch for
# ``preserve_input_phase='remap'``
# ---------------------------------------------------------------------------
#: ``True`` -- ``preserve_input_phase='remap'`` launches its rays along the
#: TOTAL entrance eikonal ``grad(W + a_fit)`` rather than the carrier's
#: ``grad(W)`` alone, where ``a_fit`` is a smooth curl-free model of the input
#: RESIDUAL eikonal (see :func:`_fit_residual_eikonal`).  ``False`` restores the
#: pre-C6 ``grad(W)``-only launch bit for bit -- the fail-before switch.
#:
#: WHY (the derivation).  Write the input as ``E_in = |E_in| exp(i k0 (W + a))``
#: and let ``V(x, X)`` be the point characteristic of the element (the traced
#: optical path from entrance point ``x`` to exit point ``X``).  Geometrical
#: optics gives the exit eikonal as the STATIONARY value
#:
#:     Psi(X) = stat_x [ W(x) + a(x) + V(x, X) ]  ==  F(x_*) ,
#:
#: and ``dV/dx = -p_in`` (Hamilton), so the stationary point ``x_*`` is the foot
#: of the ray launched along ``grad(W + a)``.  The shipped 'remap' evaluates the
#: SAME function ``F`` at the foot ``x_W`` of the ``grad(W)`` ray instead
#: (``opl_map`` carries ``W + V`` and the transported phasor adds ``a``), and
#: since ``grad F(x_W) = grad a(x_W)``,
#:
#:     Phi_remap - Psi = F(x_W) - F(x_*)
#:                     = 1/2 * grad a^T H^-1 grad a + O(|grad a|^3),
#:     H = Hess_x (W + V) ,
#:
#: i.e. a second-order term QUADRATIC in the input residual's own ray slope.
#: That is the whole of the element's residual model error: on design 121's
#: last group it scales as measured (``grad a`` rms 1.46 -> 2.30 mrad across the
#: niche-C5 fix, exit wavefront error 0.036 -> 0.089 waves, ratio 2.48 against
#: the predicted ``(2.30/1.46)^2 = 2.48``).
#:
#: Launching along ``grad(W + a_fit)`` puts the Newton foot ON the stationary
#: point (exactly, to the extent ``a_fit`` models ``a``), leaving only
#: ``1/2 grad(a - a_fit)^T H^-1 grad(a - a_fit)`` -- quadratic in what the fit
#: MISSES.  The same augmented map also supplies ``det J`` for
#: ``amplitude_model='ray_density'``, which is the matching stationary-phase
#: amplitude, so phase and amplitude stay consistent by construction, and the
#: transported residual phasor becomes the LEFTOVER ``exp(i k0 (a - a_fit))``.
#:
#: MEASURED.  Design 121's post-DOE chain, EE3 % at the group-5 exit against
#: the exact-ray oracle's ceiling (validation/repro_traced_carrier_121/
#: probe_c6_chain.py, the instrument the niche-C5 table used):
#:
#:   order      ceiling   C5 (this OFF)   C6 ON      recovered
#:   (0,0)       90.08        87.99       89.21      58 % of 2.09
#:   (-4,0)      90.78        76.61       88.94      87 % of 14.17
#:   (-4,-2)     89.78        73.66       88.49      92 % of 16.12
#:
#: FIELD-ANGLE SPREAD 14.33 -> 0.72 points, and FWHM 4.008 -> 3.628 um at
#: (-4,-2).  On the element pass ALONE, scored pointwise against the exact-ray
#: oracle, the exit wavefront error goes 0.0659 -> 0.0140 waves at (-4,-2)
#: (Marechal Strehl 0.843 -> 0.992) and 0.0580 -> 0.0144 at (-4,0).  A split
#: into implementation and model legs (probe_c6_split.py) puts the
#: IMPLEMENTATION at 0.0008-0.0042 waves at every setting: what is left is the
#: model's own remainder, not the plumbing.
#:
#: NOT BYTE-IDENTICAL, and deliberately so: the on-axis input residual of a
#: real relay is not zero (``grad a`` rms 0.66 mrad on design 121's last
#: group), so an UNTILTED chain moves too -- that is the whole point, and it is
#: why this ships behind its own switch rather than silently.
REMAP_STATIONARY_PHASE_LAUNCH = True

#: Total polynomial degree of the residual-eikonal potential ``a_fit``.  A
#: SCALAR POTENTIAL is fitted (by matching its gradient to the measured local
#: ray slope), never ``L``/``M`` separately, so the model is curl-free -- a
#: necessary condition for it to be an eikonal at all.  The degree is stepped
#: DOWN automatically when the bright support cannot constrain it (8 kept
#: gradient samples per basis term).
#:
#: WHY 4.  Measured on design 121's last group at order (-4,-2), the worst
#: case -- the element pass scored pointwise against the exact-ray oracle, and
#: the same setting scored END TO END through the whole post-DOE chain:
#:
#:   degree  rms grad(a-a_fit)  element WFE (waves)  ghost power  chain EE3 %
#:     off        --                 0.0659            0.00000       73.66
#:      2       8.83e-4              0.0388            0.00000       81.95
#:      3       2.99e-4              0.0074            0.00000       88.66
#:      4       1.03e-4              0.0140            0.00000       88.49
#:      5       1.01e-4              0.0144            0.00132       88.49*
#:      6       8.5e-5               0.0136            0.01255       88.72
#:
#: (* interpolated; degree 5 was not run end to end.  Oracle ceiling 89.78.)
#:
#: Read it as: everything from degree 3 on lands within 0.25 EE3 points of
#: everything else END TO END, so the choice is made on GENERALITY.  Degree 4
#: is where the residual's own content is -- a carrier-referenced relay carries
#: an r^4-dominant correction (see ``remap_sampling``), and degree 4 spans that
#: EXACTLY: on a synthetic fixture whose residual IS r^4 the corrected launch
#: reads 2e-5 waves against 0.021 uncorrected, a factor of 1000, where degree 3
#: reads 0.014 and removes only a third.  Degree 3 happens to fit design 121's
#: particular group-5 residual better (90 % of its slope) but cannot represent
#: the generic case at all.  Degrees 5-6 buy nothing and start self-caustiking
#: in the 2-3 w skirt.
#:
#: KNOWN COST OF THIS CHOICE, measured and NOT fixed: at degree 4 the ON-AXIS
#: design-121 call gains **0.103 % of the input power** as a far ghost lobe
#: (exit power 0.9959 -> 0.9970), where degree 3 gains none.  It appears only
#: on the CONCENTRIC fit branch (hard NaN mask, ``newton_poly_order=6``); the
#: off-centre branch (weighted, ``_DECENTRED_FIT_POLY_ORDER=10``) is clean at
#: every order.  It does not move the spot (on-axis EE3 89.21 at degree 4
#: against 88.87 at degree 3, and the field's second moment inside r < 1 mm
#: moves by 0.0003 mm) but it IS spurious energy -- 1.03e-03 of Pin as an
#: annulus at 6.298-7.216 mm exit radius carrying 33 % of the peak amplitude --
#: so any halo / second-moment metric taken through this path on axis should be
#: checked against ``REMAP_STATIONARY_PHASE_LAUNCH = False``.
#:
#: MECHANISM (2026-07-31), and it is NOT the one first recorded here.  The
#: original note read this as "the order-6 forward-map fit being unable to
#: carry a degree-4 launch augmentation".  That is REFUTED: raising the order
#: on the hard-mask branch makes it 86x WORSE (``newton_poly_order=10`` gains
#: 8.5 % of the input power).  It is the D1 fold -- see
#: ``REMAP_STATIONARY_PHASE_FIT_GUARD`` for the mechanism, the sweeps and the
#: opt-in remedy.
#:
#: RESOLVED (2026-07-31): the ELEMENT-vs-oracle column's NON-MONOTONICITY in
#: the degree (3 reading 0.0074 against degree 4's 0.0140, while the model's
#: own slope residual keeps improving) is an artefact of the PROBE's 2 %-of-
#: peak amplitude threshold, not of the fit and not -- as was guessed here --
#: of the oracle's band-limited representation of ``a``.  Measured
#: (validation/repro_traced_carrier_121/probe_c6_degree_oracle.py): the
#: oracle's own upsample factor is converged (4 / 8 / 16 move degree 4 by
#: 1.8 % and never reorder anything), the scored patch size is irrelevant, and
#: raising the amplitude threshold to 10 % of peak REVERSES the ordering to
#: 4 < 6 < 3 -- the order ``grad(a - a_fit)`` predicts -- while dropping
#: degree 4's reading 6x, 0.0140 -> 0.0024 waves.  The whole penalty lives in
#: the 2-10 %-of-peak skirt, where degree 4's extra terms are constrained by
#: the core (``_REMAP_RESID_BRIGHT_FRAC`` = 0.05) and then evaluated outside
#: it.  On a synthetic fixture with an ANALYTIC oracle the response is
#: perfectly ordered -- 2.065e-02 (off) -> 1.406e-02 (degrees 2 and 3) ->
#: 2.344e-05 (degrees 4, 5, 6) -- so nothing is wrong with the fit.  Degree 4
#: is the best model over the bright core.
#:
#: ---------------------------------------------------------------------------
#: 2026-08-02 -- RAISED TO 6.  THE ONLY THING KEEPING IT AT 4 WAS A GHOST THAT
#: NICHE C8 NOW BOUNDS.  docs/audits/D121_RESIDUAL_CLOSURE_2026_08_02.md.
#:
#: The fail-before is this constant: setting it back to ``4`` restores the
#: v5.32.0 / niche-C9 behaviour exactly, since it is the ONLY thing that
#: changes (it is read once, at :func:`_fit_residual_eikonal`, and clamped by
#: ``_REMAP_RESID_DEGREE_CAP``).
#:
#: WHAT THE "degrees 5-6 buy nothing and start self-caustiking" line above was
#: measuring, and why it no longer holds.  Both records of that ghost -- the
#: ``ghost power`` column here (1.255e-02 at degree 6, order (-4,-2)) and
#: ``REMAP_STATIONARY_PHASE_FIT_GUARD``'s "degree 6 still reads 9.78e-03" --
#: predate ``REMAP_INVERSE_SUPPORT_BOUND`` (niche C8, 2026-08-01), whose whole
#: job is to stop the library CLAIMING amplitude outside the traced ray
#: support.  The degree-6 ghost is exactly such a claim.  Re-measured on the
#: post-C9 tree through ``energy_stage_audit_121.py`` (unedited), design 121
#: order (-4,-2), ``RN=1024``, ``rs=4``, six post-DOE groups:
#:
#:   degree  C8   P_out/P_in   g4          amax4      r_rms (mm)
#:      4    ON    0.993839   8.653e-09   1.147e-04    0.8373
#:      4    OFF   0.993839   8.653e-09   1.147e-04    0.8373   (C8 inert)
#:      6    ON    0.993843   9.694e-09   1.117e-04    0.8376
#:      6    OFF   1.051890   5.818e-02   9.448e-01    2.3601   <- the ghost
#:
#: i.e. at degree 4 the support bound is INERT and at degree 6 it is decisive:
#: with it off the chain MANUFACTURES 5.2 % of the input power and the exit
#: second moment triples; with it on, degree 6's conservation and halo are
#: within noise of degree 4's on every order measured ((0,0): both 0.000e+00 /
#: 0.000e+00; (-1,0): 1.285e-12 -> 2.663e-11; (-2,0): 7.947e-12 -> 7.659e-11 --
#: all 1e-3 or less of the C3 bound).  **The counter-evidence was real and it
#: is now bounded, so the reason for 4 is spent.**
#:
#: WHAT 6 BUYS, and why it is a FORM argument rather than a resolution one.
#: A carrier-referenced relay's residual eikonal is r^4-DOMINANT with an r^6
#: next term; degree 4 spans the first and degree 6 the second, while degree 5
#: adds only ODD terms a near-radial residual has no use for.  If that is the
#: mechanism the response must read 4 ~ 5 << 6, and it does.  Design 121, EE3
#: (area-exact) against the exact-ray oracle's true ceiling, chain readout
#: split against the exact eikonal -- the residual left, in points:
#:
#:   order     deg 3    deg 4     deg 5    deg 6    recovered
#:   (0,0)     1.577    0.048     0.048   -0.048     +0.096
#:   (-1,0)    1.815    0.934     0.946    0.029     +0.905
#:   (-2,0)    1.461    0.774     0.796    0.063     +0.711
#:   (-3,0)    1.087    0.527     0.554    0.090     +0.438
#:   (-4,0)    0.999    0.305     0.338    0.141     +0.164
#:   (-4,-2)   0.967    0.279     0.386    0.152     +0.127
#:
#: ``deg 5 - deg 4`` is +0.000/+0.012/+0.022/+0.027/+0.032/+0.107 -- nothing,
#: or slightly worse, at every order -- while ``deg 6 - deg 5`` is
#: -0.096/-0.917/-0.732/-0.465/-0.197/-0.234.  The field-angle SPREAD goes
#: 0.886 -> 0.200 points and every order lands within +-0.16 of the exact-ray
#: oracle.  That is the C6 derivation closing on itself: what this launch
#: leaves behind is ``1/2 grad(a - a_fit)^T H^-1 grad(a - a_fit)``, quadratic
#: in what the fit MISSES, and what degree 4 was missing was the r^6 term.  A
#: change that were merely "more resolution" would improve through degree 5;
#: this does not.
#:
#: PRODUCTION ACCEPTANCE IS UNCHANGED.  ``focus_scan_121.py`` (unedited, pure
#: library defaults, N=2048/NFC=8192/WF=4.0), run with the degree pinned both
#: ways in the same session: BEST-FOCUS[peak] ``dz = 0``,
#: **3.350 um / EE3 90.3 / EE6 99.7 / EE12 99.8** in BOTH, with the peak
#: 5.516e+03 -> 5.529e+03 (+0.24 %).  Production re-traces the last group on a
#: fine grid where much of this is inert; the coarse per-group element calls
#: that the diagnostic paraxial route and every per-order oracle comparison go
#: through are where the 0.9 points lived.
#:
#: SCOPE, stated rather than implied.  Measured on ONE design at one
#: wavelength.  What is NOT design-specific is the argument: the ghost that
#: kept the degree at 4 is bounded by a shipped guard, and the term degree 6
#: adds is the next RADIAL order of a residual whose form is set by the
#: carrier reference, not by design 121.  The synthetic r^4 fixture quoted
#: above is unaffected (degrees 4, 5 and 6 all read 2.344e-05 on it).
_REMAP_RESID_EIKONAL_DEGREE = 6
#: Amplitude floor (fraction of peak) for a sample to enter the residual fit.
_REMAP_RESID_BRIGHT_FRAC = 0.05
#: Wrapped nearest-neighbour phase step (rad) above which a residual-gradient
#: sample is REJECTED as (approaching) aliased.  ``pi`` is the fold point; half
#: of it is the usual safe working limit.
_REMAP_RESID_MAX_STEP_RAD = 0.5 * np.pi
#: Minimum kept gradient samples per polynomial term.
_REMAP_RESID_MIN_SAMPLES_PER_TERM = 8
#: Radius of the residual-eikonal FIT disc, in units of the measured beam
#: amplitude radius ``w``.  It is also the RADIAL FREEZE radius (see
#: :class:`_ResidualEikonal`), i.e. the boundary between where the model
#: interpolates its data and where it is continued.
#:
#: It must COVER the beam.  Shrinking it to 1.5 w on design 121's last group
#: takes the exit wavefront error from 0.0141 to 0.313 waves -- WORSE than not
#: correcting at all -- because the 1.5-2 w annulus, where the residual's own
#: slope is largest (it grows as r^3), is then extrapolated rather than fitted.
#: (That figure was taken under the earlier multiplicative-window model, where
#: this radius also set the window; under the radial freeze the failure is
#: milder but the reason is the same and the setting is not worth revisiting.)
#: 2.0 w holds 99.97 % of a Gaussian's power and matches the ray-fit disc the
#: chain already uses (``fit_radius_beam_factor=2.0``); widening it to 2.5 w
#: moves the design-121 result by 0.00001 waves.
#:
#: SAMPLE DISC ONLY (2026-07-31).  This constant used to set the RADIAL FREEZE
#: circle as well.  It no longer does -- see ``_REMAP_RESID_FREEZE_MARGIN`` for
#: the defect that separation fixes.  It still sets the polynomial's own
#: normalisation ``scale`` (the fit is solved in ``r / r_fit``), so the two
#: cannot simply be merged.
_REMAP_RESID_FIT_W = 2.0

#: The RADIAL FREEZE circle (see :class:`_ResidualEikonal`) must sit strictly
#: OUTSIDE the ray-fit disc, by at least this factor.  Existence, not taste:
#: with the two circles coincident the polynomial forward-map backend and the
#: spline backend stop describing the same map.
#:
#: THE DEFECT.  ``newton_fit='polynomial'`` fits the traced entrance->exit map
#: with a GLOBAL total-degree Chebyshev whose data support is the ray-fit disc
#: (D1 weights / D7 order / the concentric hard NaN mask), and EXTRAPOLATES
#: outside it.  That extrapolation is sound only while the map stays smooth
#: there -- which is exactly what ``_FIT_DISC_OUTSIDE_WEIGHT_REL``'s safety
#: argument assumes.  The C6 launch augments every ray by ``grad(a_fit)``, and
#: ``a_fit``'s radial freeze puts a CURVATURE DISCONTINUITY on its own circle:
#: with that circle at ``_REMAP_RESID_FIT_W * w`` and the ray-fit disc at
#: ``fit_radius_beam_factor * w`` -- both 2.0 in the shipped chain -- the map
#: stops being smooth at precisely the radius where the fit's data stops, so
#: the extrapolation is invalid from the first pixel outward.
#:
#: MEASURED, D7's ghost fixture (weak singlet, 12 mm aperture, beam 5.6 mm off
#: axis, ``ray_subsample=2``), the two backends' forward maps scored POINTWISE
#: against the EXACT skew ray trace of the same augmented launch
#: (``lumenairy.raytrace``, Zemax-validated -- neither interpolant):
#:
#:   zone (about the beam)   polynomial      spline
#:   inside the 2 w disc      0.000 um       0.007 um
#:   skirt 2-4 w              5.608 um       0.006 um
#:   entrance aperture rim   15.079 um       0.002 um
#:
#: -- against 0.089 um for the SAME polynomial fit with the C6 launch off, i.e.
#: the augmentation cost the polynomial backend a factor of 170 outside its own
#: disc while the spline backend was unaffected.  The polynomial backend is the
#: wrong one, and the returned fields diverged by 3.7e-03 of peak (426 exit
#: pixels dropped by the entrance-aperture stop, whose hard cut is taken at the
#: FITTED pullback) against 7.8e-09 before C6.
#:
#: IT IS NOT A CONDITIONING LOTTERY.  The response is sharp and it turns at the
#: predicted radius.  Sweeping the freeze circle with the SAMPLE disc held at
#: 2.0 w and the ray-fit disc at 2.0 w (max |E| difference between the two
#: backends, over peak):
#:
#:   freeze  1.00 w   1.50 w   1.75 w   2.00 w | 2.25 w   2.50 w   3.00 w
#:   d       1.7e-03  1.7e-03  3.7e-03  3.7e-03| 2.8e-05  8.6e-06  2.5e-06
#:
#: -- a 130x step across the ray-fit disc radius and nowhere else.  Inside it
#: the fit is forced to represent the kink and rings over its own disc; outside
#: it the fit never sees the kink and its extrapolation is valid over the whole
#: skirt that still carries amplitude.
#:
#: WHY NOT LARGER.  The freeze is what stops a degree-``d`` polynomial fitted
#: to the BRIGHT support (the ``_REMAP_RESID_BRIGHT_FRAC`` floor puts a
#: Gaussian's last sample at 1.73 w) from being extrapolated into a launch
#: deflection it has no data for.  ``_REMAP_RESID_FREEZE_MAX_W`` caps the reach
#: at 3.0 w, i.e. ``(3.0/1.73)^4 = 9x`` amplification of the model's own
#: top-order term, and the margin below is the smallest that clears the ray-fit
#: disc with room for the fit's own extrapolation to be exact ACROSS it.
#:
#: RESIDUAL SCOPE, measured and NOT fixed HERE: a caller whose
#: ``fit_radius_beam_factor`` exceeds ``_REMAP_RESID_FREEZE_MAX_W`` runs the
#: cap into the ray-fit disc and the pathology returns.  The structural cure --
#: bound the Newton inverse to the traced samples' own support -- SHIPPED as
#: niche C8; see ``REMAP_INVERSE_SUPPORT_BOUND``.  It does not repair the fit
#: (the two backends still disagree outside the disc, which is why this margin
#: stays), but it stops the disagreement being handed amplitude.
_REMAP_RESID_FREEZE_MARGIN = 1.25
#: Hard ceiling on the freeze circle, in beam amplitude radii.  See
#: ``_REMAP_RESID_FREEZE_MARGIN``.
_REMAP_RESID_FREEZE_MAX_W = 3.0
#: Ceiling on the residual-eikonal degree, enforced in
#: :func:`_fit_residual_eikonal`.  Existence rather than taste: the radial
#: freeze bounds the model's SLOPE outside the fit disc but a high-degree
#: polynomial can also self-caustic INSIDE it, where the freeze cannot help.
#: See ``_REMAP_RESID_EIKONAL_DEGREE`` for the measured table.
_REMAP_RESID_DEGREE_CAP = 6

#: OPT-IN REMEDY for the C6 ON-AXIS GHOST.  Gives the C6 stationary-phase
#: launch the D1 WEIGHTED ray-fit restriction (and D7's raised order) even when
#: the fit disc is CONCENTRIC.  **Default ``False``: the shipped path is the
#: historical concentric hard NaN mask, byte for byte.**  It is a lever for a
#: caller who has measured a halo on their own design, not a fix -- see SCOPE.
#:
#: THE DEFECT IT ADDRESSES.  ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` states the
#: precondition under which the concentric hard sample mask is safe: *"the
#: unconstrained directions of the fit inherit the map's RADIAL SYMMETRY, the
#: extrapolation outside the disc stays MONOTONE, and the Newton inversion
#: cannot find a second root."*  ``REMAP_STATIONARY_PHASE_LAUNCH`` augments
#: every launch direction by ``grad(a_fit)`` of a general (non-radial)
#: polynomial fitted to the measured input residual, so it DESTROYS that
#: precondition, and the D1 failure mode returns on the one branch D1 left
#: alone: design 121's ON-AXIS last-group call (the only CONCENTRIC one in the
#: fan) returns a spurious ANNULUS at 6.298-7.216 mm exit radius, peaking at
#: 6.675 mm and 33 % of peak amplitude, carrying 1.03e-03 of the input power --
#: while the traced congruence puts the whole ILLUMINATED pupil (launch
#: ``r <= 2 w``) inside 2.36 mm.  It does not move the spot, and the chain's
#: 19.2 um readout tile cannot see it, so it is invisible to every encircled-
#: energy metric; it needs a halo / second-moment metric.
#:
#: MEASURED, design 121's last group, on-axis call, residual degree 4,
#: ``ray_subsample=4``, ``fit_radius_beam_factor=2``
#: (validation/repro_traced_carrier_121/probe_ghost_c6.py + _locate.py).
#: ``g4`` = returned power beyond 4 mm from the traced exit chief ray over
#: input power; ``amax4`` = largest |E| there over the peak:
#:
#:   restriction  order   P/Pin      g4        amax4   r_rms (mm)
#:   C6 OFF, mask   6    0.99590   3.6e-11    1.4e-05    0.8422
#:   mask           6    0.99701   1.03e-03   3.30e-01   0.8638   <- SHIPPED
#:   mask          10    1.08514   8.92e-02   9.30e-01   2.4192
#:   mask          14    0.99773   1.42e-03   4.05e-01   0.8927
#:   weight         6    0.99605   7.45e-05   1.12e-01   0.8388
#:   weight        10    0.99598   0.00e+00   0.00e+00   0.8371   <- this flag
#:
#: This RETRACTS the hypothesis recorded in ``_REMAP_RESID_EIKONAL_DEGREE`` --
#: "the order-6 forward-map fit [is] unable to carry a degree-4 launch
#: augmentation".  More terms on the hard-mask branch makes it 86x WORSE (row
#: 3), which is what an unconstrained extrapolation does and not what an
#: under-resolved fit does.  The mechanism is the fitted entrance->exit map
#: being Newton-inverted far outside its own data support, with
#: ``amplitude_model='ray_density'`` handing the spurious roots real amplitude.
#:
#: SCOPE -- WHY THIS IS OPT-IN AND NOT THE DEFAULT.  Every knob on that fit
#: (restriction method, order, ``fit_radius_beam_factor``,
#: ``_FIT_DISC_OUTSIDE_WEIGHT_REL``, the residual degree) has clean and dirty
#: settings with no monotone structure, and the DIRECTION REVERSES between
#: fixtures.  Measured:
#:
#:   * design 121, on axis: weighted+10 is clean (0.00e+00), mask+6 ghosts;
#:   * design 121, tilted orders: already weighted, so this flag is inert and
#:     the field is BYTE-IDENTICAL either way -- the C6 recovery cannot move;
#:   * design 121, on axis, ``fit_radius_beam_factor`` 1.5 / 2.0 ghost, 2.5+
#:     are clean; on the TILTED order 2.0 is clean and 2.5+ ghost at 1.0e-02;
#:   * ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` on the tilted order: 1e-8 (shipped)
#:     reads 2.1e-08, but 1e-10 reads 2.1e-04 and 1e-6 reads 1.7e-03 -- a
#:     ~1-decade well, not the 4-decades-clear plateau that constant's own note
#:     claims from its low-NA fixture;
#:   * residual degree through this flag: clean at 2, 3, 4, 5; degree 6 (the
#:     ``_REMAP_RESID_DEGREE_CAP``) still reads 9.78e-03;
#:   * and on a SYNTHETIC singlet built at design 121's own scale (N=1024,
#:     dx=33 um, w=3.1 mm, R_c=-21 mm, 20 mm aperture) the hard mask is EXACTLY
#:     clean and this flag INTRODUCES a 4.5e-03 lobe at 61 % of peak
#:     (validation/repro_traced_carrier_121/probe_ghost_synthetic.py).
#:
#: So this is a conditioning lottery, not a mechanism-level cure, and turning
#: it on by default would trade design 121's on-axis ghost for someone else's.
#: It also costs accuracy where the hard-mask fit was exact: on the free-leg
#: fixture of tests/unit/test_niche_c6_stationary_phase_launch.py the C6 exit
#: error goes 2.34e-05 -> 6.93e-04 waves (the residual model's RADIAL FREEZE
#: continues ``a`` linearly in r outside the fit disc, so weighting those
#: samples imports a shape the Chebyshev basis cannot represent).  On design
#: 121's real on-axis call that cost is 0.01574 -> 0.01629 waves (3.5 %).
#:
#: THE STRUCTURAL FIX is not on this axis at all: bound the Newton inverse to
#: the traced samples' own support, or use a caustic-faithful amplitude model
#: (``apply_real_lens_gbd`` / ``apply_real_lens_fga``).  Neither is attempted
#: here.  See docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S3.
#:
#: 2026-08-01: THE FIRST OF THOSE SHIPPED, as ``REMAP_INVERSE_SUPPORT_BOUND``
#: (niche C8), and it makes this flag REDUNDANT on every case measured -- see
#: the closing note at the bottom of this docstring.
#:
#: ---------------------------------------------------------------------------
#: 2026-07-31: MEASURED AT CHAIN LEVEL, AND THE DEFAULT IS CONFIRMED ``False``.
#: docs/audits/C6_FIT_GUARD_DECISION_2026_07_31.md.  The two claims above that
#: were element-level inferences are now chain measurements, and ONE OF THEM
#: WAS WRONG.
#:
#: (1) REACH.  The guard acts on a group whose beam decentre is <= the
#: ``_DECENTRE_GATE_W_FRAC`` gate (0.05 w), NOT on "the on-axis order".
#: Replaying every group of design 121's post-DOE chain on its own captured
#: input, guard off vs on (validation/repro_traced_carrier_121/
#: fitguard_branch_map.py), the decentre in beam radii and the groups the guard
#: MOVES are:
#:
#:   order    grp0    grp1    grp2    grp3    grp4    grp5   groups moved
#:   (0,0)   0.0000  0.0000  0.0000  0.0000  0.0000  0.0000    all six
#:   (-1,0)  0.0128  0.0493  0.0623  0.1530  0.1989  0.2406    0 and 1
#:   (-2,0)  0.0255  0.0986  0.1246  0.3061  0.3978  0.4813    0 only
#:   (-3,0)  0.0383  0.1480  0.1869  0.4593  0.5968  0.7227    0 only
#:   (-4,0)  0.0511  0.1974  0.2493  0.6126  0.7960  0.9647    NONE
#:   (-4,-2) 0.0571  0.2207  0.2788  0.6850  0.8900  1.0791    NONE
#:
#: So "byte-identical on every tilted order" holds only for (-4,0) and
#: (-4,-2), where it is exact (``array_equal`` on all six stages, at
#: ``ray_subsample`` 4 AND 2).  On (-1,0)/(-2,0)/(-3,0) the guard moves the
#: FIRST group by 2.2e-04 of peak and that cascades: the chain exit differs by
#: up to 2.0e-03 of peak.  The conservation and halo metrics do not move with
#: it -- ``P_out/P_ap`` on the last group agrees to all six printed digits on
#: all three orders -- but the field is not byte-identical and code that
#: assumed it was is wrong.
#:
#: (2) IT DOES NOT FIX (-2,0) OR (-3,0), which the energy audit left open.
#: Those orders' last group sits at 0.48 w / 0.72 w, i.e. ALREADY on the
#: off-centre weighted branch, so the guard is structurally inert exactly where
#: their lobe is made.  Measured at the last group (``rs`` = 4 / 2):
#:
#:   (-2,0)  g4     ship 2.270e-07 / 7.08e-06   guard 2.274e-07 / 2.79e-06
#:           amax4  ship 5.727e-03 / 5.02e-02   guard 5.749e-03 / 3.23e-02
#:   (-3,0)  g4     ship 2.234e-07 / 5.69e-07   guard 1.633e-07 / 6.15e-07
#:           amax4  ship 5.873e-03 / 1.39e-02   guard 5.684e-03 / 9.92e-03
#:
#: -- against exact-ray ``g4`` ceilings of 3.0443e-08 and 5.9279e-08 and a
#: halo-amplitude bound of 1.0e-03.  Both configurations fail on both orders at
#: both subsamples, and the guard moves the number by less than the quantity's
#: own conditioning -- at (-3,0) it is 1.37x BETTER on ``g4`` at rs=4 and 1.08x
#: WORSE at rs=2.  Their lobe is a DIFFERENT defect from the concentric one and
#: this flag is not its remedy.  With ``REMAP_STATIONARY_PHASE_LAUNCH = False``
#: both orders are clean (``g4`` 7.75e-09 / 1.92e-08, 0.25x / 0.32x of their
#: ceilings), so the lobe is C6's and it is made on the WEIGHTED branch.
#:
#: (3) WHERE IT DOES WORK IT IS DECISIVE, at both subsamples.  Design 121's
#: on-axis order, last group, ``P_out/P_ap`` and the halo:
#:
#:            rs   guard OFF                    guard ON            C6 OFF
#:   elem(5)   4   0.999371                     0.995976            0.995901
#:   elem(5)   2   1.003696  (ABOVE UNITY)      0.999199            0.999189
#:   g4        4   3.400e-03                    0.000e+00           3.61e-11
#:   g4        2   4.495e-03                    0.000e+00           4.56e-11
#:   amax4     4   7.70e-01                     0.000e+00           1.40e-05
#:   r_rms/mm  4   0.9349                       0.8384              0.8422
#:
#: It restores the discretisation deficit floor (0.98x the C6-off reference at
#: rs=4, 1.01x at rs=2, against 0.15x with the guard off), takes the halo to
#: EXACTLY zero, and puts the second moment back on the exact-ray reference
#: (0.8407 mm).  In the PRODUCTION configuration it costs EE3.
#:
#: (4) WHY IT IS STILL NOT THE DEFAULT.  The synthetic counter-example this
#: note already records was RE-MEASURED and it stands, on the current tree:
#: probe_ghost_synthetic.py's design-121-scale stand-in singlet goes from a
#: hard-mask field that is EXACTLY clean to a weighted one carrying
#: ``P/Pin`` = 1.00697 with 8.7e-03 of the input power beyond the exact-ray
#: exit support at 88 % of peak; the 'medium, finer grid' fixture goes from
#: 6.9e-06 to 4.6e-02 of peak.  Two of six fixtures regress, and neither is
#: exotic.  Design 121 itself shows no such regression anywhere -- but a
#: library default is not answerable to one design.  A caller who has MEASURED
#: a halo on their own design should turn this on and re-measure; the v5.32
#: halo self-check (``_RD_HALO_AMAX_TOL``) reports both directions of the trade
#: automatically, and it fires on design 121's on-axis last group with this
#: flag off and is silent with it on.
#:
#: ---------------------------------------------------------------------------
#: 2026-08-01: SUPERSEDED IN PRACTICE, KEPT ON PURPOSE.
#: ``REMAP_INVERSE_SUPPORT_BOUND`` (niche C8) reaches the same defect
#: structurally, and on every case measured it dominates this flag:
#:
#:   * design 121 on axis -- C8 matches this guard's conservation result to
#:     five decimals (``elem(5)`` 0.995971 vs 0.995976, ``g4`` and ``amax4``
#:     exactly zero on both) while COSTING NO EE, where the guard costs 0.96
#:     EE3 points and 0.59 EE6 points in the production configuration;
#:   * the six synthetic fixtures above -- C8 regresses NONE, this guard
#:     regresses two;
#:   * (-2,0) and (-3,0) -- C8 repairs them, this guard structurally cannot
#:     (their lobe is made on the off-centre branch, where it is inert).
#:
#: It is NOT removed.  It is a different intervention on a different object --
#: this one changes the FIT, C8 bounds what is claimed FROM the fit -- so it
#: remains available for a caller whose defect is the fit's conditioning inside
#: its own support, which no support bound can reach.  Its default stays
#: ``False``, and turning it on with C8 also on is measured, not assumed:
#: docs/audits/C8_INVERSE_SUPPORT_BOUND_2026_08_01.md.
REMAP_STATIONARY_PHASE_FIT_GUARD = False

#: niche C8 -- BOUND THE NEWTON INVERSE TO THE TRACED SAMPLES' OWN SUPPORT.
#: An exit pixel outside the region the traced rays actually REACHED gets zero
#: ray-density amplitude instead of an extrapolated inverse-map value.
#: ``REMAP_INVERSE_SUPPORT_BOUND = False`` restores the pre-C8 library bit for
#: bit (the fail-before switch).
#:
#: THE DEFECT.  ``newton_fit='polynomial'`` fits the traced entrance->exit map
#: with a GLOBAL total-degree Chebyshev and EXTRAPOLATES outside its own data;
#: ``newton_fit='spline'`` extrapolates its bicubic past the last knot.  Either
#: way the Newton loop is then asked to invert a model where nothing was
#: measured, and on a map that has lost its radial symmetry that inverse can
#: land a far exit pixel BACK INSIDE THE BRIGHT BEAM -- whereupon
#: ``_ray_density_amp_grid`` samples ``|E_in|`` there and hands the pixel real
#: amplitude.  The energy is not misplaced, it is MANUFACTURED: no ray of the
#: call reaches that radius.
#:
#: ``REMAP_STATIONARY_PHASE_LAUNCH`` (niche C6) is what destroys the symmetry
#: -- it augments every launch direction by ``grad(a_fit)`` of a non-radial
#: polynomial -- so C6 is where the defect became visible, on design 121's
#: on-axis last group: ``P_out/P_ap`` = 1.000741 in the production path against
#: C6-off's 0.995883, i.e. **+0.486 % of the input power created**, deposited
#: at 4-8 mm at 83 % of peak where the exact ray trace permits 3.6e-10.  But
#: the mechanism is NOT C6's: it is the unbounded inverse, and this bound also
#: repairs two defects C6 has nothing to do with (see (4) and (5) below).
#:
#: THE SUPPORT, and why it is defined this way.  The convex hull of the EXIT
#: landing points of the alive traced rays whose ENTRANCE the stop passes:
#:
#:   * taken BEFORE the fit-domain restriction, so it is the exact traced map
#:     and not something the model fitted (the restriction NaNs samples that
#:     are perfectly good optics, and reading it after would understate the
#:     support and cut real light);
#:   * restricted to the rays the aperture passes because the launch square
#:     spans 1.5x the aperture RADIUS, so a third of it is blocked light --
#:     including it would inflate the support with territory no photon reaches.
#:     This is the SAME criterion ``_ray_density_amp_grid`` already masks on;
#:   * CONVEX because a lens exit region is (the same argument
#:     ``inversion_method='fit'`` has always used for its own hull mask -- this
#:     bound gives the Newton path the containment the direct-fit path had).
#:     Convexity can only make the bound LOOSER, never tighter, so it cannot
#:     manufacture a cut.
#:
#: THE FEATHER.  ``_SUPPORT_BOUND_FEATHER_CELLS`` -- see there for the
#: measurement.  The hull is the hull of SAMPLES; the ray map is continuous
#: between them, so the true support reaches about half an exit-lattice cell
#: further.  The taper is a raised cosine over a band lying entirely OUTSIDE
#: the hull, so every pixel with traced data behind it keeps its full
#: amplitude.
#:
#: SCOPE.  ``amplitude_model='ray_density'`` only.  That is not a hedge: it is
#: the only amplitude in this function DERIVED FROM THE INVERSE MAP.  The
#: ``'screen'`` amplitude comes from ``apply_real_lens``'s analytic transport
#: of the input field and never reads ``(xe, ye)``, so there is nothing there
#: for an extrapolated inverse to corrupt.  The OPL is left alone for the same
#: reason -- where the taper is zero the amplitude is zero and the phase is
#: unobservable, and NaN-ing the OPL would hard-cut a mask that is deliberately
#: smooth.
#:
#: MEASURED -- design 121, the diagnostic per-stage chain (``RN=1024``, six
#: post-DOE groups, ``final_distance=0``, ``final_leg='paraxial'``), last group,
#: against the exact-ray ceilings of ``ENERGY_CONSERVATION_AUDIT_2026_07_31``:
#:
#:   ray_subsample = 4          elem(5)     g4        amax4     r_rms/mm
#:   (0,0)  C6 off             0.995901   3.61e-11   1.40e-05   0.8422
#:   (0,0)  C6 on  (pre-C8)    0.999371   3.40e-03   7.70e-01   0.9349
#:   (0,0)  C6 on  + C8        0.995971   0.00e+00   0.00e+00   0.8385
#:   (-2,0) C6 on  (pre-C8)    0.996043   2.27e-07   5.73e-03   0.8382
#:   (-2,0) C6 on  + C8        0.996043   5.37e-09   1.61e-04   0.8382
#:   (-3,0) C6 on  (pre-C8)    0.995917   2.23e-07   5.87e-03   0.8380
#:   (-3,0) C6 on  + C8        0.995916   1.31e-08   2.05e-04   0.8379
#:
#:   ray_subsample = 2          elem(5)     g4        amax4     end to end
#:   (0,0)  C6 on  (pre-C8)    1.003696   4.50e-03   9.78e-01   1.003186
#:   (0,0)  C6 on  + C8        0.999201   0.00e+00   0.00e+00   0.998693
#:   (-2,0) C6 on  (pre-C8)    0.999195   7.08e-06   5.02e-02   0.998681
#:   (-2,0) C6 on  + C8        0.999188   5.44e-09   1.58e-04   0.998674
#:   (-4,-2) C6 OFF (pre-C8)   0.999196   6.32e-06   7.84e-02   0.998661
#:   (-4,-2) C6 OFF + C8       0.999190   2.63e-08   2.36e-04   0.998655
#:
#: Read the columns in order.
#:
#: (1) It repairs the on-axis order OUTRIGHT, at both subsamples, on every one
#:     of the audit's six bounds -- 2/6 -> 6/6 at ``rs=4`` and 0/6 -> 6/6 at
#:     ``rs=2``.  ``g4`` and ``amax4`` go to EXACTLY zero, not merely under
#:     bound, and the discretisation deficit floor comes back to 0.98x / 0.99x
#:     of the C6-off reference from 0.15x / negative.
#: (2) It matches ``REMAP_STATIONARY_PHASE_FIT_GUARD``'s conservation result on
#:     that order to five decimals (0.995971 against 0.995976) WITHOUT the
#:     guard's cost: on ``probe_ghost_synthetic``'s six fixtures the guard
#:     regresses two (one to ``P/Pin`` = 1.00697 with 3.6e-01 of peak beyond
#:     3 w) and this bound regresses NONE -- every fixture reproduces the hard-
#:     mask branch to all printed digits.
#: (3) It costs no EE.  Design 121 per-order EE3 against the exact-ray oracle
#:     at the chain's group-5 exit is UNCHANGED to 0.01 points on all three
#:     scored orders: (0,0) 89.20, (-4,0) 88.98, (-4,-2) 88.53.
#: (4) It fixes what the fit guard STRUCTURALLY COULD NOT.  (-2,0) and (-3,0)
#:     make their lobe on the OFF-CENTRE weighted branch, where the guard is
#:     inert by construction; both fail C3+C4 with the guard on OR off, at both
#:     subsamples.  Under this bound both pass: ``g4`` 0.18x / 0.22x of their
#:     exact-ray ceilings and ``amax4`` 6x under its bound.
#: (5) It fixes a defect C6 has nothing to do with.  At ``rs=2`` on (-4,-2) it
#:     is C6-OFF that violates the halo criterion (28x / 78x over) -- the
#:     energy audit's "reversal" -- and the bound takes that row to 0.35x of
#:     its ceiling as well.  The mechanism was never C6's; C6 only made it
#:     large enough to see.
#:
#: WHAT IT IS NOT.  It does not repair the FIT: outside the hull the fitted map
#: is still wrong, and inside it the fit's error is untouched (byte-identical
#: fits, byte-identical Newton, byte-identical OPL).  It bounds what the
#: library is willing to CLAIM from that fit.  A defect that deposits energy
#: INSIDE the traced support is invisible to it by construction, exactly as it
#: is to the v5.32 halo self-check.
REMAP_INVERSE_SUPPORT_BOUND = True

#: Feather width of the C8 support bound, in EXIT-LATTICE cells -- the median
#: exit separation of entrance-adjacent traced rays, measured from the samples
#: themselves rather than from a paraxial magnification, so it tracks the
#: resolution at which the support is actually known.  0.0 is a hard binary
#: cut.
#:
#: WHAT IT IS *NOT* FOR.  Protecting legitimate light at the boundary is the
#: PLATEAU's job (see :func:`_support_taper`), and the plateau does it exactly:
#: the power the bound removes from inside its own support is 0.000e+00 at
#: EVERY feather including 0.  So the feather is not a safety margin on the
#: sampling; with the plateau in place it is measurably inert on design 121 --
#: every metric of every order is identical from 0.0 to 4.0 cells.
#:
#: WHAT IT IS FOR: the sharpness of what is left.  On the one fixture measured
#: where the bound truncates a field carrying real amplitude at its boundary --
#: niche D6's decentred fixture, where 1.4e-01 of peak sits just outside the
#: hull -- ``edge jump`` is the largest nearest-neighbour ``|E|`` step, over
#: the peak, within 6 coarse cells of the hull, and ``dP`` is the power removed
#: (100 % of it outside the hull of every alive ray, at every feather):
#:
#:   feather (cells)   OFF       0.00      0.25      0.50      1.00     2.00     4.00
#:   edge jump       1.366e-01  3.07e-02  3.07e-02  3.07e-02  2.40e-02 1.93e-02 1.19e-02
#:   dP / P_in            0     2.62e-03  2.59e-03  2.56e-03  2.48e-03 2.30e-03 2.06e-03
#:
#: Read the OFF column first: **the unbounded field's own largest step in that
#: band is 1.366e-01, so even a HARD cut leaves an edge 4.5x smoother than what
#: it removed.** The "a binary mask may diffract" worry is real in principle
#: and does not arise here, because what the mask deletes is jagged
#: manufactured light and what it leaves is smoother than the original.
#:
#: **1.0 is the smallest feather that measurably improves on the hard cut**
#: (3.07e-02 -> 2.40e-02; 0.25 and 0.5 cells are sub-pixel on this fixture and
#: change nothing), and it gives up 5 % of the manufactured light removed to
#: get it.  Wider is a straight trade against the fix: 4.0 cells buys another
#: 2x of edge for 21 % less manufactured light removed.
#:
#: IT DOES NOT WEAKEN THE FIX.  On design 121's on-axis order the manufactured
#: lobe is removed IDENTICALLY at every feather from 0.0 to 4.0: ``g4`` and
#: ``amax4`` exactly zero, ``P_out/P_ap`` 0.995976, and the same -1.034e-03 of
#: input power removed.  The lobe lies wholly outside the hull, so no width of
#: transition band reaches it.
_SUPPORT_BOUND_FEATHER_CELLS = 1.0


# ===========================================================================
# NICHE C14 (2026-08-03) -- UNIT C: THE TRACED EXIT SUPPORT, AS ONE OBJECT
# ===========================================================================
# WHAT WAS WRONG.  There were THREE notions of "the region the traced rays
# reached", computed from the same arrays, at nearly the same point in
# ``apply_real_lens_traced``, by three different rules and three separate
# copies of the same convex-hull algebra:
#
#   1. the C7 halo radius -- amplitude-weighted centroid + max radius over the
#      samples above the ``e^-_RD_HALO_AMP_CONTOUR`` amplitude contour, times
#      ``_RD_HALO_RADIUS_FACTOR`` at report time;
#   2. the C8 support hull -- convex hull of the alive STOP-PASSING landings,
#      plus a ``sqrt(2) sub dx`` plateau and one exit-lattice cell of feather;
#   3. the direct-fit hull -- the ``inversion_method='fit'`` path's own
#      long-standing exit hull mask over the post-restriction samples.
#
# (2) and (3) are the same idea implemented twice, and the C8 audit says so:
# "This bound gives the Newton path the containment the direct-fit path has had
# all along."  They had two copies of the ConvexHull call, two copies of the
# ``equations -> (A, b)`` unpacking and two different half-plane evaluators
# (one chunked over a BLAS product, one a full-width ``np.all``).
#
# THE MEASURED CONSEQUENCE, and the reason this is not cosmetics.
# ``RECON_PINS_POST_C8_2026_08_01.md`` S7 item 1: on the E-M6 fixture the
# post-C8 field still carries **0.19998 of P_ap outside the exact-ray hull** --
# in the plateau+feather band C8 keeps DELIBERATELY -- and **its global |E|
# maximum sits in that band**.  Neither self-check reports it: the energy
# check's reading (1.01931) is inside its own band, and the C7 halo check looks
# only beyond ``1.25 x r_hull``, which on that fixture is 1.525 mm while the
# taper's outer edge is 1.4996 mm.  So C7's reporting annulus lies ENTIRELY in
# the region C8 has already zeroed: under the bound the halo check cannot fire,
# and the band the bound retains is watched by nobody.  "The two self-checks
# are jointly blind to a residual 20 % of P_ap of manufactured light."
#
# That was found by an adversarial re-check, not by the checks, and it is
# exactly the shape of defect a single object makes cheap to state: with one
# hull, "is the field's maximum inside the support the rays actually reached?"
# is one comparison on one object, where before it was a relationship between
# an ``e^-9`` amplitude contour times 1.25 and a ``sqrt(2) sub dx`` plateau
# plus one lattice cell, computed 40 lines apart from different masks.
#
# WHAT THIS OBJECT DOES *NOT* DO.  It does not merge the three rules.  They are
# not interchangeable and unifying them would be a behaviour change, not a
# refactor: C7's radius is amplitude-weighted ON PURPOSE (a REPORTING radius
# calibrated over 180 element calls, with a measured 123x separation between
# the clean and defective populations at factor 1.25), and C8's is a convex
# hull of stop-passing rays ON PURPOSE (convexity "can only make the bound
# LOOSER, never tighter, so it cannot manufacture a cut").  Merging them would
# re-open a calibration that cost 177 readings.  What is unified is the
# CONSTRUCTION and the CONVENTIONS -- one alive mask, one hull builder, one
# signed-distance rule -- with three NAMED VIEWS on top.  Every view reproduces
# its old arithmetic operand for operand, which is why the extraction is
# byte-identical and is proved so by ``probe_c14_byte_identity.py``.
#: Policy for the niche-C14 SUPPORT-BAND self-check: ``'warn'`` (default) or
#: ``'silent'``.  ``'silent'`` is the FAIL-BEFORE SWITCH: it restores the
#: pre-C14 reporting exactly, because the band check is the only thing C14
#: adds to what the element SAYS (it adds nothing at all to what the element
#: RETURNS -- like C7 and the energy check, it is reporting-only and cannot
#: change a returned bit).
#:
#: WHAT IT WATCHES.  The annulus C8 retains outside the traced hull: the
#: ``sqrt(2) sub dx`` plateau plus ``_SUPPORT_BOUND_FEATHER_CELLS`` of
#: feather, i.e. exactly ``0 < s <= d0 + f`` in the hull's own signed distance.
#: No traced ray of the call reaches there, so any amplitude in it is either
#: legitimate skirt bleeding out of the support or manufactured light -- and
#: the two are separable without a new calibration, which is the point of the
#: criterion below.
#:
#: WHY THE CRITERION IS A RATIO AND NOT A TOLERANCE.  A skirt decays outward.
#: A manufactured lobe does not: on the E-M6 fixture the field's GLOBAL
#: maximum sits in the retained band.  So the test is
#: ``max|E| in the band  >  _SUPPORT_BAND_PEAK_RATIO_TOL * max|E| inside the
#: support`` -- scale-free, unit-free, and needing none of the 180-call
#: calibration ``_RD_HALO_RADIUS_FACTOR`` needed, because at a ratio of 1.0 it
#: asks only "does this field peak somewhere the rays never went?", which no
#: correct field can answer yes to.
#:
#: IT INHERITS C7's DECLINATION and that is deliberate.  The check runs inside
#: the same block as the halo check and behind the same grid-fit test, so on a
#: grid whose extent is comparable to its own exit fan -- design 121's
#: production readout leg -- it declines for the same measured reason C7 does
#: (see SCOPE (d) at ``_RD_HALO_AMAX_TOL``): a statistic read on a sliver of
#: corners is unreliable in BOTH directions.  Closing the blind spot on the
#: readout leg needs a hull that fits the grid, which is a different problem.
SUPPORT_BAND_CHECK = 'warn'
#: Ratio of in-support peak amplitude above which the retained band is
#: reported.  1.0 = "the global maximum may not sit outside the traced
#: support".  Lower it to tighten (0.5 reports a band lobe at half the core
#: peak); raise it above 1.0 only to silence a fixture you have adjudicated.
_SUPPORT_BAND_PEAK_RATIO_TOL = 1.0


class _TracedExitSupport(object):
    """UNIT C: where the traced rays of THIS call actually landed.

    Built ONCE, from the exact traced map, between the alive mask and the
    fit-domain restriction -- the position both the C7 and the C8 blocks
    independently chose, for two reasons their comments state almost verbatim:
    the fit restriction NaNs samples that are still perfectly good optics (so
    reading it later would understate the support and over-fire the check), and
    this is the last point at which ``x_out_grid`` is the exact traced map
    rather than anything the model fitted.

    Three views, three rules, one set of conventions:

    ==================  =========================================  ==========
    view                rule                                       consumer
    ==================  =========================================  ==========
    ``centroid``/       amplitude-weighted centroid and max         C7 halo
    ``radius``          radius over samples above the e^-N          check
                        amplitude contour
    ``hull``            convex hull of the alive STOP-PASSING       C8 bound
                        landings, as half-planes ``(A, b)``         + C14 band
    :meth:`half_planes` the same hull builder, on any point set     direct fit
    ==================  =========================================  ==========

    ``hull`` and the direct-fit path's own hull are built by the SAME
    classmethod and evaluated by the SAME signed-distance rule; they differ
    only in the point set they are given (the fit path hands it the
    post-restriction samples, which is its documented behaviour and is not
    changed here).
    """

    __slots__ = ('centroid', 'radius', 'hull', 'pitch', 'feather', 'n_hull',
                 'hull_c', 'hull_rmax')

    def __init__(self, centroid=None, radius=None, hull=None, pitch=None,
                 feather=None, n_hull=0, hull_c=None, hull_rmax=None):
        self.centroid = centroid
        self.radius = radius
        self.hull = hull            # (A, b) half-planes, or None
        self.pitch = pitch
        self.feather = feather
        self.n_hull = int(n_hull)
        # A point known to be INSIDE the hull, and a radius about it that
        # CONTAINS the hull.  Only the band check uses these, and only to
        # bound the work: see :meth:`retained_band_masks`.
        self.hull_c = hull_c
        self.hull_rmax = hull_rmax

    # -- shared primitives ------------------------------------------------
    @staticmethod
    def half_planes(px, py, strict=False):
        """Convex hull of scattered 2-D points as outward half-planes
        ``(A, b)``, with ``A`` contiguous ``(2, F)`` and ``b`` ``(F,)``, or
        ``None`` when the point set cannot support a hull.

        ``strict=True`` lets the hull failure PROPAGATE instead of declining.
        The two consumers genuinely differ and both are right: the C8 bound is
        an optional containment, so a support it cannot measure means "do not
        bound", while the direct-fit path's hull IS its output domain -- a
        fit with no domain has nothing to return, and it has always raised.
        Sharing the construction must not quietly convert one into the other.

        Qhull normalises ``equations`` to UNIT outward normals, which is what
        makes ``max_f (n_f . p + d_f)`` the exact signed distance to the
        boundary for a point outside it (and ``<= 0`` inside).  Every consumer
        of this class relies on that, so the unpacking lives here once.

        A degenerate support (collinear / duplicated landings) has no hull;
        this DECLINES rather than guessing -- a degenerate bound is exactly the
        regime this must not invent an answer in.  The except tuple is NARROWED
        to what this path can actually raise (the non-ui broad-except budget,
        ``tests/unit/test_audit_except_budget.py``): ``ImportError`` when the
        compiled ``scipy.spatial`` qhull extension is absent from a trimmed
        install, ``RuntimeError`` because ``QhullError`` is a documented
        ``RuntimeError`` subclass (named indirectly: ``scipy.spatial.QhullError``
        is only public from scipy 1.8 and the floor here is 1.7), and
        ``ValueError`` for the input rejections (non-finite / wrong ndim)
        ConvexHull raises before qhull ever runs.
        """
        if strict:
            from scipy.spatial import ConvexHull as _CH
            _eq = _CH(np.column_stack([px, py])).equations
        else:
            try:
                from scipy.spatial import ConvexHull as _CH
                _eq = _CH(np.column_stack([px, py])).equations
            except (ImportError, RuntimeError, ValueError):
                return None
        return (np.ascontiguousarray(_eq[:, :2].T),
                np.ascontiguousarray(_eq[:, 2]))

    @staticmethod
    def signed_distance(A, b, Xg, Yg):
        """``s = max_f (n_f . p + d_f)`` on arbitrary coordinate ARRAYS.

        Chunked (pixels x facets) product: BLAS does the work, and the chunk
        caps the temporary at ~160 MB however many facets the hull has and
        however fine the lattice is.  A per-facet Python loop would be ~50x
        slower on a large lattice."""
        _sh = np.asarray(Xg).shape
        _xg = np.asarray(Xg, dtype=np.float64).ravel()
        _yg = np.asarray(Yg, dtype=np.float64).ravel()
        _nf = int(b.size)
        _s = np.empty(_xg.size, dtype=np.float64)
        _cn = max(1, int(2e7 // max(_nf, 1)))
        for _i in range(0, _xg.size, _cn):
            _j = min(_i + _cn, _xg.size)
            _s[_i:_j] = (np.column_stack([_xg[_i:_j], _yg[_i:_j]]) @ A
                         + b).max(axis=1)
        return _s.reshape(_sh)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_landings(cls, x_out_grid, y_out_grid, amp, xs_in, aperture,
                      dx, sub, want_halo, want_bound):
        """Build the support from the EXACT traced landings.

        ``want_halo`` / ``want_bound`` gate the two expensive views
        independently, so a call that has switched one of them off does exactly
        the work it did before (the two blocks this replaces had separate
        gates, and preserving that preserves both the bits and the runtime).

        The one thing genuinely SHARED is the finiteness mask, which the two
        blocks each computed for themselves."""
        self = cls()
        if not (want_halo or want_bound):
            return self
        alive = np.isfinite(x_out_grid) & np.isfinite(y_out_grid)

        # ---- view 1: the C7 amplitude-weighted reporting radius ----------
        if want_halo:
            _h_ok = alive.copy()
            _h_pk = float(np.max(amp)) if amp.size else 0.0
            if _h_pk > 0.0:
                _h_ok &= (amp >= np.exp(-_RD_HALO_AMP_CONTOUR) * _h_pk)
            if _h_ok.any():
                _h_w = amp[_h_ok].astype(np.float64) ** 2
                _h_wt = float(_h_w.sum())
                if _h_wt > 0.0:
                    _h_x = x_out_grid[_h_ok]
                    _h_y = y_out_grid[_h_ok]
                    self.centroid = (float((_h_x * _h_w).sum() / _h_wt),
                                     float((_h_y * _h_w).sum() / _h_wt))
                    self.radius = float(np.sqrt(
                        (_h_x - self.centroid[0]) ** 2
                        + (_h_y - self.centroid[1]) ** 2).max())
                    del _h_x, _h_y
                del _h_w
            del _h_ok

        # ---- view 2: the C8 stop-passing convex hull ---------------------
        if want_bound:
            _sup_ok = alive.copy()
            if aperture is not None:
                # Only rays the ENTRANCE STOP passes carry energy, and the
                # ray-density amplitude already masks on exactly that criterion
                # (see ``_ray_density_amp_grid``).  Rays launched outside the
                # stop (the launch square reaches 1.5x the aperture RADIUS)
                # land further out but are blocked, so including them would
                # inflate the support with territory no light can reach.
                _sup_ok &= ((xs_in[:, None] ** 2 + xs_in[None, :] ** 2)
                            <= (0.5 * aperture) ** 2)
            if int(_sup_ok.sum()) >= 3:
                _px = x_out_grid[_sup_ok]
                _py = y_out_grid[_sup_ok]
                _hp = cls.half_planes(_px, _py)
                if _hp is not None:
                    # Geometry for the band check's work bound ONLY -- it can
                    # move no bit of the bound or the taper.  ``hull_c`` is the
                    # mean landing (inside a convex hull of the same points by
                    # construction) and ``hull_rmax`` contains every one of
                    # them about it.
                    self.hull_c = (float(_px.mean()), float(_py.mean()))
                    self.hull_rmax = float(np.sqrt(
                        (_px - self.hull_c[0]) ** 2
                        + (_py - self.hull_c[1]) ** 2).max())
                    # The feather is measured in EXIT-LATTICE cells, taken from
                    # the traced samples themselves rather than from a paraxial
                    # magnification: the exit spacing of entrance-adjacent rays
                    # IS the resolution at which the support is known.  (The
                    # separate, non-negotiable allowance for the bilinear
                    # upsample's reach is the PLATEAU, :meth:`taper`.)
                    with np.errstate(invalid='ignore'):
                        _sup_step = np.hypot(np.diff(x_out_grid, axis=0),
                                             np.diff(y_out_grid, axis=0))
                    _sup_step = _sup_step[np.isfinite(_sup_step)]
                    _sup_pitch = (float(np.median(_sup_step))
                                  if _sup_step.size else float(dx * sub))
                    if not (np.isfinite(_sup_pitch) and _sup_pitch > 0.0):
                        _sup_pitch = float(dx * sub)
                    self.hull = _hp
                    self.pitch = _sup_pitch
                    self.feather = (float(_SUPPORT_BOUND_FEATHER_CELLS)
                                    * _sup_pitch)
                    self.n_hull = int(_hp[1].size)
                    del _sup_step
                del _px, _py
            del _sup_ok
        return self

    # -- views ------------------------------------------------------------
    @property
    def bound(self):
        """C8's ``(A, b, feather)`` triple, or ``None`` -- the exact tuple the
        pre-C14 ``_sup_bound`` local carried, so every consumer's truth test
        and unpacking are unchanged."""
        if self.hull is None:
            return None
        return (self.hull[0], self.hull[1], self.feather)

    def taper(self, Xg, Yg, plateau):
        """niche C8: 1 inside the traced exit support, 0 beyond it, raised
        cosine across a feather band of ``_SUPPORT_BOUND_FEATHER_CELLS``
        exit-lattice cells outside the boundary.

        THE PLATEAU ``plateau`` IS NOT TASTE, IT IS THE UPSAMPLE.  This taper
        is evaluated on the COARSE Newton lattice (pitch ``sub * dx`` in the
        exit plane) and the amplitude is then bilinearly interpolated to the
        wave grid, so a coarse node OUTSIDE the hull lends its attenuation to
        wave pixels up to one coarse cell INSIDE it.  ``s`` is 1-Lipschitz, so
        a pixel with ``s <= 0`` interpolates only from nodes with
        ``s <= sqrt(2) * sub * dx``; holding the taper at exactly 1 out to
        there makes the bleed identically zero rather than merely small.
        MEASURED on niche D6's decentred fixture -- power removed from INSIDE
        the bound's own support, over the chain input power:

          feather (cells)     0.0        0.5        1.0        2.0        4.0
          without the plateau 2.211e-04  8.194e-05  3.808e-05  1.207e-05  3.2e-06
          with it             0          0          0          0          0

        i.e. 1.1 % of what the bound removes on that fixture was legitimate
        skirt lost to interpolation, and it is now exactly none.  The cost is
        that the bound sits ``sqrt(2) sub dx`` further out (188 um on design
        121's last group, 3 % of its 6.3 mm hull), which is measured not to
        readmit any of the manufactured lobe there -- and it is that same
        deliberately-retained band the niche-C14 check now watches, because
        nothing watched it before (see ``SUPPORT_BAND_CHECK``).
        """
        _A, _b, _f = self.bound
        _s = self.signed_distance(_A, _b, Xg, Yg) - float(plateau)
        if _f <= 0.0:
            return (_s <= 0.0).astype(np.float64)
        return np.where(_s <= 0.0, 1.0,
                        np.where(_s >= _f, 0.0,
                                 0.5 * (1.0 + np.cos(np.pi * _s
                                                     / max(_f, 1e-300)))))

    def taper_grid(self, xg, yg, plateau):
        """:meth:`taper` on the OUTER PRODUCT of two axes, with the work
        bounded by the same two strict radial screens
        :meth:`retained_band_masks` uses.

        BIT-IDENTICAL to ``taper(X, Y, plateau)`` on the same grid, and the
        argument is the one written out there: every pixel closer to
        ``hull_c`` than the inradius ``r_in`` has ``s < 0``, so its taper is
        exactly 1; every pixel beyond ``hull_rmax + plateau + feather`` has
        ``s > plateau + feather``, so its taper is exactly 0.  Only the ring
        between them reaches the exact half-plane reduction.

        WHY IT EXISTS.  The taper's natural home is the COARSE Newton lattice
        (9 025 points at design 121's retrace), where ``O(pixels x facets)`` is
        free.  The inverse-characteristic path has no coarse lattice, so it
        asks for the taper on the WAVE grid -- 6.71e+07 pixels x ~150 facets is
        a 10^10-MAC BLAS pass, MEASURED at 5.9 s per call on a 1.7e+07-pixel
        grid, which would have eaten half of what that path saves.  The ring is
        a few per cent of the grid on a near-circular exit hull.

        Falls back to the dense form when the screens are unavailable (a
        hand-built object, or a hull whose interior point was not recorded).
        """
        if self.bound is None:
            return None
        _A, _b, _f = self.bound
        xg = np.asarray(xg, dtype=np.float64)
        yg = np.asarray(yg, dtype=np.float64)
        if self.hull_c is None or self.hull_rmax is None:  # pragma: no cover
            return self.taper(np.broadcast_to(xg[None, :],
                                              (yg.size, xg.size)),
                              np.broadcast_to(yg[:, None],
                                              (yg.size, xg.size)), plateau)
        cx, cy = self.hull_c
        pl = float(plateau)
        w = pl + float(_f or 0.0)
        r_in = -float((np.asarray(_A[0]) * cx + np.asarray(_A[1]) * cy
                       + np.asarray(_b)).max())
        out = np.zeros((int(yg.size), int(xg.size)), dtype=np.float64)
        r2 = (xg[None, :] - cx) ** 2 + (yg[:, None] - cy) ** 2
        done = np.zeros(out.shape, dtype=bool)
        if r_in + pl > 0.0:
            done = r2 < (r_in + pl) ** 2
            out[done] = 1.0
        r_out = float(self.hull_rmax) + w
        ring = (~done) & (r2 <= r_out * r_out)
        del r2, done
        if ring.any():
            iy, ix = np.nonzero(ring)
            _s = self.signed_distance(_A, _b, xg[ix], yg[iy]) - pl
            if _f <= 0.0:
                out[iy, ix] = (_s <= 0.0).astype(np.float64)
            else:
                out[iy, ix] = np.where(
                    _s <= 0.0, 1.0,
                    np.where(_s >= _f, 0.0,
                             0.5 * (1.0 + np.cos(np.pi * _s
                                                 / max(_f, 1e-300)))))
        return out

    def retained_band_masks(self, xg, yg, plateau):
        """``(inside, band)`` boolean masks on the wave grid's outer product.

        ``inside`` is the traced support (``s <= 0``); ``band`` is the annulus
        C8 RETAINS outside it (``0 < s <= plateau + feather``), which is
        precisely the region no traced ray reached and the bound does not cut.
        Returns ``(None, None)`` when there is no hull to measure against.

        THE WORK IS BOUNDED EXACTLY, not approximately.  The half-plane
        reduction is ``O(pixels x facets)`` and this runs on the WAVE grid, so
        a naive evaluation would put a ~30x BLAS pass on every ray-density
        call for a diagnostic.  Two radial screens cut it to a thin annulus
        without changing a single verdict, because both are strict:

        * every pixel closer to ``hull_c`` than ``r_in`` -- the distance from
          that interior point to its NEAREST facet -- is strictly inside the
          hull, so ``s < 0`` there and it can only be ``inside``;
        * the hull lies inside the disc of radius ``hull_rmax`` about
          ``hull_c``, and the signed distance to a convex set contained in
          that disc is at least ``|p - c| - hull_rmax``; so every pixel beyond
          ``hull_rmax + w`` has ``s > w`` and can be in NEITHER mask.

        Only the ring between them is handed to the exact test.  On a
        near-circular exit hull that ring is a few per cent of the grid; on a
        strongly elongated one it degrades gracefully to the full disc, which
        is what the naive version would have cost anyway."""
        if self.hull is None:
            return None, None
        A, b = self.hull
        w = float(plateau) + float(self.feather or 0.0)
        inside = np.zeros((int(yg.size), int(xg.size)), dtype=bool)
        band = np.zeros_like(inside)
        if self.hull_c is None or self.hull_rmax is None:  # pragma: no cover
            # Only reachable on a hand-built object; ``from_landings`` sets
            # both whenever it sets ``hull``.  Decline rather than measure a
            # band whose work cannot be bounded.
            return None, None
        cx, cy = self.hull_c
        # Distance from the interior point to the nearest facet: the hull's
        # own inradius about ``hull_c`` (negated because ``s`` is <= 0 inside).
        r_in = -float((np.asarray(A[0]) * cx + np.asarray(A[1]) * cy
                       + np.asarray(b)).max())
        r2 = ((np.asarray(xg, dtype=np.float64)[None, :] - cx) ** 2
              + (np.asarray(yg, dtype=np.float64)[:, None] - cy) ** 2)
        r_out = float(self.hull_rmax) + w
        if r_in > 0.0:
            inside |= (r2 < r_in * r_in)
        ring = (~inside) & (r2 <= r_out * r_out)
        del r2
        if ring.any():
            iy, ix = np.nonzero(ring)
            s = self.signed_distance(A, b,
                                     np.asarray(xg, dtype=np.float64)[ix],
                                     np.asarray(yg, dtype=np.float64)[iy])
            inside[iy, ix] = (s <= 0.0)
            band[iy, ix] = (s > 0.0) & (s <= w)
        return inside, band


class _ResidualEikonal(object):
    """A curl-free polynomial model ``a_fit(x, y)`` of the carrier-de-chirped
    input residual eikonal (METRES), with an exact analytic gradient and a
    BOUNDED continuation outside its fit disc.

    ``value`` and ``grad`` are the potential and its true gradient of ONE
    scalar field -- never a modified gradient -- so the launched bundle is a
    genuine congruence and the traced ``W + a_fit + OPL`` grid stays an eikonal
    (its entrance derivative is ``p_out . J``, exactly as on the shipped
    ``grad(W)`` launch).

    ``scale`` and ``r_fit`` are DIFFERENT radii and both are load-bearing:
    ``scale`` is the sample disc the coefficients were solved in (it fixes the
    polynomial's normalisation, so it cannot move after the fit), while
    ``r_fit`` is the RADIAL FREEZE circle below.  They coincided until
    2026-07-31; see ``_REMAP_RESID_FREEZE_MARGIN`` for why they must not.

    RADIAL FREEZE.  The fit is constrained by the beam and then EVALUATED over
    a launch square that can be 5x wider, where a degree-``d`` monomial grows
    as ``r^d``.  Outside the freeze radius ``r1`` the potential is therefore
    continued LINEARLY IN ``r`` along each ray from the beam centre,

        a(r, th) = P(r1, th) + (r - r1) * dP/dr(r1, th) ,   r > r1

    which is C^1 across ``r1``, adds EXACTLY ZERO radial curvature outside it
    (so the continuation cannot focus, and cannot form the ring caustic that a
    multiplicative window's ``P * grad(window)`` term does), and has a gradient
    bounded for all ``r`` by the model's own first and second derivatives ON
    the circle ``r = r1`` -- i.e. by quantities the data constrains.

    The first implementation used a ``cos^2`` window to zero instead.  It was
    MEASURED and rejected: on design 121's last group the window's own radial
    curvature focused the 2-3 w annulus into a ring caustic whose ray-density
    amplitude reached 87 % of the beam peak, worth 4.8 % of the input power
    (and the failure MOVED WITH the window -- placing it at 3-4 w or 3.5-4.5 w
    reproduced it, 5.2 % and 5.6 %, which is what identified it).
    """

    __slots__ = ('coef', 'terms', 'cx', 'cy', 'scale', 'r_fit', 'diag')

    def __init__(self, coef, terms, cx, cy, scale, r_fit, diag=None):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.terms = tuple(terms)
        self.cx = float(cx)
        self.cy = float(cy)
        self.scale = float(scale)
        self.r_fit = float(r_fit)
        self.diag = dict(diag or {})

    def _poly(self, ex, ey, hess=True):
        """``(P, grad P, Hess P)`` in PHYSICAL coordinates about the centre.

        ``hess=False`` returns ``(P, Pu/s, Pv/s, None, None, None)`` -- the
        VALUE and its gradient only.  The three Hessian slots are ``None``
        rather than a stale interior Hessian so a consumer that reads them
        anyway fails loudly instead of silently using the wrong thing; the
        only caller that asks for it is :meth:`_eval` on its value-only path,
        where the Hessian feeds nothing (see ``value``).

        PERFORMANCE (AUDIT_TRACED_SPEED_2026_08_09 sec 2, item 1;
        FIX_PERF_POLY_LOCALS_2026_08_09).  This method was MEASURED at 57.8 %
        of one design-121 fan order's wall (py-spy cross-check 59.1 %), all of
        it reached through ``_pip_residual_ri -> value -> _eval``.  The cost
        was structural, not algorithmic: the loop recomputed ``u ** i`` and
        ``v ** j`` FROM SCRATCH for every one of the six accumulators of every
        term, and numpy has no fast path for integer exponents above 2, so
        each of those is a libm ``pow()`` per element.  At degree 6 (27 terms)
        that is ~130 whole-array ``pow`` passes where 14 DISTINCT exponents
        exist.

        The fix is to issue one ``np.power`` per distinct exponent and index
        the result.  It is BIT-IDENTICAL, not "identical to round-off": the
        same ``np.power`` calls on the same operands produce the same bits,
        the operand ORDER inside each term is unchanged (``(c*i) * u^p * v^q``
        associates left to right exactly as before), and ``x += y`` rounds
        identically to ``x = x + y``.  Pinned by
        ``test_niche_perf_poly_locals.py``, which keeps a verbatim copy of the
        pre-change implementation as its reference and asserts
        ``np.array_equal`` on the real design-121 term list at degree 6.

        The power table costs ``degree + 1`` arrays per axis where the old
        loop held six accumulators; on the shipped consumer that is a wash,
        because the row band is bounded (``_pip_residual_ri`` bands at 4.19
        Mpt) and the value-only path below drops three accumulators and the
        whole freeze-gradient temporary chain.
        """
        s = self.scale
        u = ex / s
        v = ey / s
        coef = self.coef
        terms = self.terms
        # Which exponents the term list actually indexes.  Built from the
        # SKIPPED-ZERO term set (the loop below skips ``c == 0`` exactly as it
        # always did), so a sparse or low-degree fit builds a smaller table --
        # and ``hess=False`` never builds the exponents only the Hessian reads.
        need_u = set()
        need_v = set()
        for c, (i, j) in zip(coef, terms):
            if c == 0.0:
                continue
            need_u.add(i)
            need_v.add(j)
            if i >= 1:
                need_u.add(i - 1)
            if j >= 1:
                need_v.add(j - 1)
            if hess:
                if i >= 2:
                    need_u.add(i - 2)
                if j >= 2:
                    need_v.add(j - 2)
        # ONE np.power per distinct exponent -- the same call the old loop made
        # per term per accumulator, so the same bits.
        #
        # ...except exponents 0 and 1, which are pure waste and are elided
        # (FIX_PERF_ROUND2_2026_08_10 item 4b).  ``u ** 0`` is a whole extra
        # full-grid array of ONES whose only use is a multiply by exactly 1.0,
        # and ``u ** 1`` is a whole extra full-grid COPY of ``u``.  On the
        # shipped degree-6 list that is 12 of 27 terms carrying a redundant
        # full-array multiply in ``P`` alone, plus two allocations per axis.
        # BIT-IDENTICAL: multiplying a float64 by exactly 1.0 is exact for
        # every finite operand and preserves inf, nan and the sign of zero;
        # ``np.power(x, 1)`` returns ``x``'s bits.  MEASURED 1.18x (512x8192
        # band) and 1.31x (256x16384 band) on the value path, byte-identical
        # against the pre-elision loop.
        UP = {p: (u if p == 1 else u ** p) for p in sorted(need_u) if p}
        VP = {q: (v if q == 1 else v ** q) for q in sorted(need_v) if q}

        def _mul(scale, p, q, _U=UP, _V=VP):
            """``scale * u**p * v**q`` with the exponent-0 factors elided.

            Left-to-right association is preserved exactly: the shipped
            expression was ``((scale) * UP[p]) * VP[q]``, and dropping a factor
            that is identically 1.0 cannot move a bit.  The tables come in as
            DEFAULT ARGUMENTS rather than as closure cells so that the
            end-of-function ``del UP, VP`` still frees them."""
            if p and q:
                return scale * _U[p] * _V[q]
            if p:
                return scale * _U[p]
            if q:
                return scale * _V[q]
            return scale
        # ``np.zeros_like(u)`` on the shipped float64 path; the explicit dtype
        # only matters for a lower-precision ``ex`` (where the old loop's
        # ``P = P + <float64 term>`` upcast on its first iteration anyway, so
        # the values are the same either way -- in-place accumulation just
        # cannot rely on that upcast).
        _dt = np.result_type(u, coef) if len(terms) else np.result_type(u)
        P = np.zeros_like(u, dtype=_dt)
        Pu = np.zeros_like(u, dtype=_dt)
        Pv = np.zeros_like(u, dtype=_dt)
        Puu = Puv = Pvv = None
        if hess:
            Puu = np.zeros_like(u, dtype=_dt)
            Puv = np.zeros_like(u, dtype=_dt)
            Pvv = np.zeros_like(u, dtype=_dt)
        for c, (i, j) in zip(coef, terms):
            if c == 0.0:
                continue
            P += _mul(c, i, j)
            if i >= 1:
                Pu += _mul(c * i, i - 1, j)
            if j >= 1:
                Pv += _mul(c * j, i, j - 1)
            if hess:
                if i >= 2:
                    Puu += _mul(c * i * (i - 1), i - 2, j)
                if i >= 1 and j >= 1:
                    Puv += _mul(c * i * j, i - 1, j - 1)
                if j >= 2:
                    Pvv += _mul(c * j * (j - 1), i, j - 2)
        del UP, VP, _mul
        if not hess:
            return (P, Pu / s, Pv / s, None, None, None)
        return (P, Pu / s, Pv / s,
                Puu / (s * s), Puv / (s * s), Pvv / (s * s))

    def _eval(self, xq, yq, need_grad=True):
        """``(a, da/dx, da/dy)``.  With ``need_grad=False`` the two gradient
        slots are ``None`` and NO Hessian is built -- see ``value``."""
        ex = np.asarray(xq, dtype=np.float64) - self.cx
        ey = np.asarray(yq, dtype=np.float64) - self.cy
        r = np.sqrt(ex * ex + ey * ey)
        r1 = self.r_fit
        out = r > r1
        # evaluate the polynomial at the CLAMPED point (identical to the query
        # point wherever the freeze is inactive, so the interior is exact)
        sc = np.where(out, r1 / np.where(r > 0.0, r, 1.0), 1.0)
        cx_ = ex * sc
        cy_ = ey * sc
        # The Hessian is the FROZEN GRADIENT's business alone (it enters
        # ``ex_x`` / ``ex_y`` and nothing else), so a value-only query never
        # builds it: three of _poly's six accumulators, and the whole
        # tangential-Hessian temporary chain below, are dead on that path.
        P, gx, gy, hxx, hxy, hyy = self._poly(cx_, cy_, hess=need_grad)
        # interior
        a = P
        ax = gx if need_grad else None
        ay = gy if need_grad else None
        if np.any(out):
            rs = np.where(r > 0.0, r, 1.0)
            ux = ex / rs
            uy = ey / rs
            b = gx * ux + gy * uy                    # dP/dr on the circle
            d = r - r1
            a = np.where(out, P + d * b, a)
            if need_grad:
                gtx = gx - b * ux                    # tangential part of gP
                gty = gy - b * uy
                hux = hxx * ux + hxy * uy            # H . u
                huy = hxy * ux + hyy * uy
                uhu = hux * ux + huy * uy
                htx = hux - uhu * ux                 # tangential part of H.u
                hty = huy - uhu * uy
                f = r1 / rs
                ex_x = f * gtx + b * ux + d * (f * htx + gtx / rs)
                ex_y = f * gty + b * uy + d * (f * hty + gty / rs)
                ax = np.where(out, ex_x, ax)
                ay = np.where(out, ex_y, ay)
        return a, ax, ay

    def value(self, xq, yq):
        # HOT PATH (57.8 % of a design-121 fan order -- see ``_poly``).  The
        # value needs ``P`` and, outside the radial freeze, ``dP/dr`` on the
        # circle; it never reads the frozen GRADIENT, so it never needs the
        # Hessian that gradient is built from.  Bit-identical to taking [0] of
        # the full evaluation -- ``a`` is computed by the same expressions from
        # the same operands.
        return self._eval(xq, yq, need_grad=False)[0]

    def grad(self, xq, yq):
        _a, gx, gy = self._eval(xq, yq)
        return gx, gy


def _fit_residual_eikonal(E_in, W_grid, wavelength, dx, dy, centre, w_beam,
                          degree=None, stride=1, ray_fit_radius=None,
                          origin=(0.0, 0.0)):
    """Fit the curl-free potential ``a_fit`` of the input RESIDUAL eikonal.

    The measured quantity is the residual's own transverse RAY SLOPE, taken as
    WRAPPED nearest-neighbour phase increments of the carrier-de-chirped field
    ``E_in * exp(-i k0 W)`` -- no unwrap anywhere (the house 2-D unwrap idiom is
    worth 1.96 rad on a beam skirt, DIAG_LAST_GROUP_DECENTRE S8.2) and no FFT
    derivative (worth 400 urad on a field correct to 0.36 urad, same section).
    Samples are kept only where BOTH pixels of the pair are above
    ``_REMAP_RESID_BRIGHT_FRAC`` of peak, where the wrapped step is below
    ``_REMAP_RESID_MAX_STEP_RAD`` (an aliased reading is not a slope), and
    inside the fit disc ``_REMAP_RESID_FIT_W * w_beam`` about ``centre``.
    Each sample is POWER-weighted (``|E_a| |E_b|``), which is the weight under
    which the exit-field error the model removes is itself measured.

    ``ray_fit_radius`` is the radius of the caller's RAY-fit disc measured
    about the same ``centre`` (``None`` when the caller places no restriction).
    It does not touch the fit; it only pushes the returned model's RADIAL
    FREEZE circle clear of that disc, which is what keeps the polynomial and
    spline ``newton_fit`` backends describing the same map -- see
    ``_REMAP_RESID_FREEZE_MARGIN``.

    ``origin=(x0, y0)`` (niche D9) is the ABSOLUTE transverse position of the
    grid's CENTRE pixel; the returned model is expressed in that ABSOLUTE frame
    (its ``centre`` and the launch heights it is later evaluated at both are),
    so the sample coordinates must carry it.  Omitting it does not raise: the
    fit disc lands ``|origin|`` away from the beam, keeps ~no samples, and this
    returns ``None`` -- which the caller reads as "unmeasurable, keep the
    shipped ``grad(W)`` launch", i.e. SILENT feature disablement.

    Returns a :class:`_ResidualEikonal` or ``None`` (unmeasurable / too few
    samples / degenerate fit), in which case the caller keeps the shipped
    ``grad(W)`` launch.
    """
    E = np.asarray(E_in)
    if E.ndim != 2 or E.size < 9:
        return None
    if not (np.isfinite(w_beam) and w_beam > 0.0):
        return None
    k0 = 2.0 * np.pi / wavelength
    ny, nx = E.shape[-2], E.shape[-1]
    _dy = float(dy) if (dy is not None and np.isfinite(dy) and dy > 0) else dx
    cx, cy = float(centre[0]), float(centre[1])
    r_fit = _REMAP_RESID_FIT_W * float(w_beam)
    # The RADIAL FREEZE circle is NOT the sample disc: it has to clear the
    # caller's ray-fit disc, or the forward-map fit's extrapolation crosses the
    # freeze's curvature discontinuity on its very first pixel outside its own
    # data.  Capped in beam radii so a wide ray-fit disc cannot extrapolate the
    # model into a launch deflection it has no data for.  See
    # ``_REMAP_RESID_FREEZE_MARGIN`` for the mechanism and the sweep.
    r_freeze = r_fit
    if (ray_fit_radius is not None and np.isfinite(ray_fit_radius)
            and ray_fit_radius > 0.0):
        r_freeze = max(r_freeze,
                       min(_REMAP_RESID_FREEZE_MARGIN * float(ray_fit_radius),
                           _REMAP_RESID_FREEZE_MAX_W * float(w_beam)))
    s = max(1, int(stride))
    xax = (np.arange(nx, dtype=np.float64) - nx / 2) * dx
    yax = (np.arange(ny, dtype=np.float64) - ny / 2) * _dy
    if origin is not None and (origin[0] or origin[1]):
        # niche D9: grid axis -> ABSOLUTE transverse coordinate (``centre``,
        # ``r_fit`` and every later query of the model are absolute).
        xax = xax + float(origin[0])
        yax = yax + float(origin[1])
    mag_pk = float(np.abs(E).max()) if E.size else 0.0
    if not (np.isfinite(mag_pk) and mag_pk > 0.0):
        return None
    thr = _REMAP_RESID_BRIGHT_FRAC * mag_pk
    _W = None if W_grid is None else np.asarray(W_grid, dtype=np.float64)

    def _pairs(axis):
        """Adjacent-pixel wrapped residual slope samples along ``axis``.

        DECIMATED by ``s`` in both directions but the differenced pixels are
        always ADJACENT, so the wrapped increment can never alias for a reason
        the full-resolution reading would not also have.
        """
        n_along = nx if axis == 1 else ny
        idx = np.arange(0, n_along - 1, s)
        if idx.size < 2:
            return None
        if axis == 1:
            Ea = E[::s, :][:, idx]
            Eb = E[::s, :][:, idx + 1]
            Wa = None if _W is None else _W[::s, :][:, idx]
            Wb = None if _W is None else _W[::s, :][:, idx + 1]
            xs = 0.5 * (xax[idx] + xax[idx + 1])
            ys = yax[::s]
            Xs = np.broadcast_to(xs[None, :], Ea.shape)
            Ys = np.broadcast_to(ys[:, None], Ea.shape)
            h = dx
        else:
            Ea = E[idx, :][:, ::s]
            Eb = E[idx + 1, :][:, ::s]
            Wa = None if _W is None else _W[idx, :][:, ::s]
            Wb = None if _W is None else _W[idx + 1, :][:, ::s]
            ys = 0.5 * (yax[idx] + yax[idx + 1])
            xs = xax[::s]
            Xs = np.broadcast_to(xs[None, :], Ea.shape)
            Ys = np.broadcast_to(ys[:, None], Ea.shape)
            h = _dy
        z = Eb * np.conj(Ea)
        if Wa is not None:
            z = z * np.exp(-1j * k0 * (Wb - Wa))
        d = np.angle(z)
        aa = np.abs(Ea)
        ab = np.abs(Eb)
        keep = ((aa > thr) & (ab > thr) & np.isfinite(d)
                & (np.abs(d) <= _REMAP_RESID_MAX_STEP_RAD)
                & (((Xs - cx) ** 2 + (Ys - cy) ** 2) <= r_fit * r_fit))
        if not keep.any():
            return None
        return (Xs[keep], Ys[keep], (d[keep] / (k0 * h)),
                (aa[keep] * ab[keep]))

    px = _pairs(1)
    py = _pairs(0)
    if px is None or py is None:
        return None
    deg0 = int(_REMAP_RESID_EIKONAL_DEGREE if degree is None else degree)
    deg0 = min(deg0, int(_REMAP_RESID_DEGREE_CAP))
    if deg0 < 1:
        return None
    n_samp = px[0].size + py[0].size
    # amplitude-weighted rms of the MEASURED slope (the quantity the shipped
    # launch drops); reported for the caller's diagnostics.
    _wall = np.concatenate([px[3], py[3]])
    _gall = np.concatenate([px[2], py[2]])
    _wsum = float(_wall.sum())
    g_rms = (float(np.sqrt((_wall * _gall ** 2).sum() / _wsum))
             if _wsum > 0 else 0.0)
    for deg in range(deg0, 0, -1):
        terms = [(i, d - i) for d in range(1, deg + 1) for i in range(d + 1)]
        if n_samp < _REMAP_RESID_MIN_SAMPLES_PER_TERM * len(terms):
            continue
        nL = px[0].size
        A = np.zeros((nL + py[0].size, len(terms)), dtype=np.float64)
        uL = (px[0] - cx) / r_fit
        vL = (px[1] - cy) / r_fit
        uM = (py[0] - cx) / r_fit
        vM = (py[1] - cy) / r_fit
        for k, (i, j) in enumerate(terms):
            if i >= 1:
                A[:nL, k] = i * uL ** (i - 1) * vL ** j / r_fit
            if j >= 1:
                A[nL:, k] = j * uM ** i * vM ** (j - 1) / r_fit
        rhs = np.concatenate([px[2], py[2]])
        wgt = np.concatenate([px[3], py[3]])
        coef = _solve_lstsq_thread_safe(A * wgt[:, None], rhs * wgt)
        if not np.all(np.isfinite(coef)):
            continue
        resid = rhs - A @ coef
        g_res = (float(np.sqrt((wgt * resid ** 2).sum() / _wsum))
                 if _wsum > 0 else 0.0)
        return _ResidualEikonal(
            coef, terms, cx, cy, r_fit, r_freeze,
            diag={'degree': deg, 'n_terms': len(terms), 'n_samples': n_samp,
                  'grad_a_rms': g_rms, 'grad_a_residual_rms': g_res,
                  'w_beam': float(w_beam), 'centre': (cx, cy),
                  'stride': s, 'r_fit': float(r_fit),
                  'r_freeze': float(r_freeze),
                  'ray_fit_radius': (None if ray_fit_radius is None
                                     else float(ray_fit_radius))})
    return None


def _sample_local_tilts(E_in, wavelength, dx, entrance_x, entrance_y,
                         max_sin=0.5, smooth_sigma_px=4.0,
                         multimode_diagnostic=None, origin=(0.0, 0.0)):
    """Extract ``(L, M)`` direction cosines for each entrance ray from
    the local phase gradient of ``E_in``.

    For a field ``E_in = A(x,y) * exp(i*phi(x,y))``, the local wavevector
    at each pixel is ``k_local = grad(phi)``.  A ray launched from
    that pixel should carry direction cosines ``L = k_x / k0``,
    ``M = k_y / k0``, where ``k0 = 2*pi/wavelength``.

    We compute ``grad(phi)`` via the conjugate-product trick
    ``angle(E_shifted * conj(E))`` so the wrap-to-(-pi, pi] happens
    once per pair without a separate unwrap pass.  Low-amplitude
    pixels (below 0.1 % of peak) and NaN/inf phase are returned as
    zero tilt.  The final cosines are clipped to ``|L|, |M| <=
    max_sin`` for numerical safety.

    Why this function has to be careful
    -----------------------------------
    A single-mode field has ONE well-defined phase gradient at every
    pixel (plane wave, smooth Gaussian, MLA-tilted beamlet).  A
    multi-mode field -- a superposition of several plane-wave
    components like a post-DOE diffraction pattern -- has NO
    well-defined local direction: neighbouring pixels can report wildly
    different ``np.angle(E_shift * conj(E))`` values because the sum of
    components interferes coherently.  Feeding those aliased per-pixel
    directions straight into the entrance->exit spline in
    :func:`apply_real_lens_traced` produces a chaotic map that Newton
    cannot invert, resulting in an all-NaN OPL and a zero output field.

    Fix: **amplitude-weighted Gaussian smoothing of the tilt field**
    before it's returned.  The smoothing is a low-pass on the local
    wavevector, with the physical interpretation that a ray launched
    from an entrance pixel carries the *mean* direction of the wave
    components within a few-wavelength neighbourhood, rather than the
    single-pixel aliased fringe phase.

    *   Single-mode fields: the true tilt is a slowly-varying function
        of position, so a Gaussian of sigma a few pixels leaves it
        essentially unchanged.  MLA-modulated fields keep their
        per-beamlet tilts.
    *   Multi-mode fields: the tilt oscillates pixel-to-pixel with
        mean zero (for a balanced set of orders).  Gaussian smoothing
        pulls the tilt toward that zero mean, naturally degenerating
        to a classical collimated launch for post-DOE inputs.
    *   Amplitude weighting ensures low-amplitude pixels (between
        DOE orders, outside MLA beamlets, etc.) don't drag the
        smoothed tilt toward the noisy phase readings those pixels
        contribute.

    No threshold to tune, no global "reject" decision -- the smoothing
    is the universal fix.

    Parameters
    ----------
    smooth_sigma_px : float, default 4.0
        Gaussian smoothing radius (pixels) applied to the tilt field.
        Set to 0 to disable smoothing (pre-smoothing behaviour, NOT
        recommended for multi-mode inputs).  A few pixels is enough
        to suppress single-pixel aliasing while preserving tilts that
        vary on the scale of typical beam features (MLA beamlet
        diameters, Gaussian waists, etc.).

        PIXEL UNITS -- so the PHYSICAL smoothing length is ``4*dx`` and
        shrinks with the grid pitch.  Flagged as an F-B suspect by audit
        AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22 ("physical smoothing
        length shrinks linearly with dx"); isolated and MEASURED
        2026-07-25 (fixed physical field + fixed physical launch
        positions, pitch swept over 4 octaves, ``lambda = 1.31 um``):

        * SINGLE-MODE input (smooth Gaussian x linear tilt -- the regime
          the traced model is valid in): the returned tilt at the beam
          CORE is ``+0.060000`` at every pitch, i.e. dx-invariant to 6
          digits; the smoothing is the documented no-op there.  The only
          dx dependence is in the far skirt (``r > 2.5 w``, amplitude
          < 6e-4 of peak), where the ``den > den.max()*1e-6`` guard's
          boundary moves with ``sigma*dx``: max ``|dL|`` 0.049 (= the
          whole tilt) at individual skirt launch points, 0.014 with a
          fixed PHYSICAL sigma.
        * MULTI-MODE input (two plane waves, a physical fringe beat):
          strongly pitch-dependent -- rms ``L`` 0.0103 / 0.0073 / 0.0024
          / 0.0013 at ``dx = 4 / 2 / 1 / 0.5 um``.  A fixed physical
          sigma does NOT fix it (0.0169 / 0.0073 / 0.0008 / 0.0003):
          there is no single local ray direction to converge to, and
          BOTH choices simply decay toward the collimated launch this
          docstring already describes as the intended degeneration.

        So the pixel unit is a real grid dependence but NOT an F-B
        driver, and a physical-length rewrite would trade one arbitrary
        length for another without a convergent target.  It is also
        UNREACHABLE from the traced-carrier chain: this function is only
        called for ``tilt_aware_rays=True`` (default False, and
        ``propagate_traced_carrier_chain`` never sets it -- the F3 guard
        additionally reroutes it whenever an explicit carrier is given)
        and from the experimental ``inversion_method='backward'`` OPL.
        Left at 4 px, pinned by ``test_niche_s10_sibling_patterns.py``.
    multimode_diagnostic : dict, optional
        If provided, gets populated with tilt-field statistics before
        and after smoothing (``raw_rms_L``, ``smoothed_rms_L``,
        ``raw_rms_M``, ``smoothed_rms_M``, ``smoothing_ratio``).
        Useful for callers that want to log or verify the smoothing
        is doing what's expected.
    """
    k0 = 2.0 * np.pi / wavelength
    N_y, N_x = E_in.shape

    # Phase gradient: d(phi)/dx ~ angle(E[:, 1:] * conj(E[:, :-1])) / dx
    # Use np.roll so shapes match; the rolled-into-the-boundary pixels
    # get low weights after the amplitude mask.
    E_shift_x = np.roll(E_in, -1, axis=1)
    E_shift_y = np.roll(E_in, -1, axis=0)
    grad_phi_x = np.angle(E_shift_x * np.conj(E_in)) / dx
    grad_phi_y = np.angle(E_shift_y * np.conj(E_in)) / dx

    L_grid = grad_phi_x / k0
    M_grid = grad_phi_y / k0

    # Zero-out noise-floor pixels and boundary wrap
    amp = np.abs(E_in)
    amp_thresh = 1e-3 * float(amp.max()) if amp.size else 0.0
    mask = (amp > amp_thresh) & np.isfinite(L_grid) & np.isfinite(M_grid)
    L_grid = np.where(mask, L_grid, 0.0)
    M_grid = np.where(mask, M_grid, 0.0)

    # Statistics before smoothing -- for diagnostics and as the "raw"
    # baseline the smoothing is operating on.
    raw_rms_L = float(np.sqrt(np.mean(L_grid[mask] ** 2))) if mask.any() else 0.0
    raw_rms_M = float(np.sqrt(np.mean(M_grid[mask] ** 2))) if mask.any() else 0.0

    # ---- Amplitude-weighted Gaussian smoothing ---------------------
    # Low-pass the tilt field with an intensity-weighted kernel:
    #
    #     L_smooth = blur(|E|^2 * L) / blur(|E|^2)
    #     M_smooth = blur(|E|^2 * M) / blur(|E|^2)
    #
    # This averages neighbouring pixels' tilts using their amplitude
    # squared (intensity) as weights.  On a smooth single-mode field
    # this leaves L and M essentially unchanged (neighbours already
    # agree).  On a multi-mode superposition with pixel-to-pixel
    # aliased phase gradients, the oscillations average out and
    # amplitude-weighting discounts the low-amplitude interference
    # nulls where the phase is noisiest.  Low-amplitude regions
    # (between beamlets, outside the main field) where the raw
    # gradient is unreliable naturally decay to zero because both
    # numerator and denominator weight them out.
    if smooth_sigma_px > 0:
        from scipy.ndimage import gaussian_filter
        I = (amp * amp).astype(np.float64)
        sigma = float(smooth_sigma_px)
        num_L = gaussian_filter(I * L_grid, sigma=sigma, mode='nearest')
        num_M = gaussian_filter(I * M_grid, sigma=sigma, mode='nearest')
        den = gaussian_filter(I, sigma=sigma, mode='nearest')
        # Guard against division by zero far from the field support
        safe = den > (den.max() * 1e-6)
        L_grid = np.where(safe, num_L / np.where(safe, den, 1.0), 0.0)
        M_grid = np.where(safe, num_M / np.where(safe, den, 1.0), 0.0)

    smoothed_rms_L = float(np.sqrt(np.mean(L_grid[mask] ** 2))) if mask.any() else 0.0
    smoothed_rms_M = float(np.sqrt(np.mean(M_grid[mask] ** 2))) if mask.any() else 0.0
    if multimode_diagnostic is not None:
        multimode_diagnostic['raw_rms_L'] = raw_rms_L
        multimode_diagnostic['raw_rms_M'] = raw_rms_M
        multimode_diagnostic['smoothed_rms_L'] = smoothed_rms_L
        multimode_diagnostic['smoothed_rms_M'] = smoothed_rms_M
        # Ratio < 1 means smoothing reduced the tilt magnitude (i.e.
        # noise was averaged out); ratio ~= 1 means smoothing was a
        # no-op (field was already smooth).
        raw_mag = np.hypot(raw_rms_L, raw_rms_M)
        smoothed_mag = np.hypot(smoothed_rms_L, smoothed_rms_M)
        multimode_diagnostic['smoothing_ratio'] = (
            smoothed_mag / raw_mag if raw_mag > 0 else 1.0)

    # Clip to physical range -- rays with |sin(theta)| > max_sin are
    # unphysical for most lens designs and will overwhelm the Newton
    # fit domain.  After smoothing this clip typically never triggers,
    # but we keep it as a defence against pathological inputs.
    np.clip(L_grid, -max_sin, max_sin, out=L_grid)
    np.clip(M_grid, -max_sin, max_sin, out=M_grid)

    # Interpolate to launch positions (physical -> pixel index,
    # bilinear sample).  Launch positions outside the E_in grid
    # (|x| > N*dx/2) fall back to zero tilt (edge -- no information).
    from scipy.ndimage import map_coordinates
    # niche D9: ``entrance_x/y`` are ABSOLUTE launch heights while the pixel
    # index is grid-relative, so the grid's centre position is removed first.
    pix_x = (entrance_x - float(origin[0])) / dx + N_x / 2.0
    pix_y = (entrance_y - float(origin[1])) / dx + N_y / 2.0
    coords = np.vstack([pix_y.ravel(), pix_x.ravel()])
    L = map_coordinates(L_grid, coords, order=1,
                        mode='constant', cval=0.0).reshape(entrance_x.shape)
    M = map_coordinates(M_grid, coords, order=1,
                        mode='constant', cval=0.0).reshape(entrance_x.shape)
    return L, M


def _reverse_prescription(prescription):
    """Build a prescription describing the same lens traversed in the
    backward direction.

    Used by the experimental backward-trace OPL inversion in
    :func:`apply_real_lens_traced`.  Reversing amounts to:

    *   Swap surface order.
    *   Negate every radius of curvature (curvature direction flips
        when viewed from the opposite side).  Conic constants and
        even-power aspheric coefficients are invariant under this
        reflection.
    *   Swap ``glass_before`` and ``glass_after`` on each surface.
    *   Reverse the thickness list (the gap AFTER surface i in the
        forward prescription is the gap BEFORE surface (N-1-i) in
        the reversed one, which is the same list read right-to-left).
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription.get('thicknesses', [])
    rev_surfaces = []
    for s in reversed(surfaces):
        rs = dict(s)
        rs['radius'] = -rs['radius']
        if rs.get('radius_y') is not None:
            rs['radius_y'] = -rs['radius_y']
        rs['glass_before'], rs['glass_after'] = (
            rs['glass_after'], rs['glass_before'])
        rev_surfaces.append(rs)
    rev = {
        'surfaces': rev_surfaces,
        'thicknesses': list(reversed(thicknesses)),
    }
    if 'aperture_diameter' in prescription:
        rev['aperture_diameter'] = prescription['aperture_diameter']
    return rev


def _opl_by_backward_trace(E_analytic, lens_prescription, wavelength, dx,
                           N_grid, ray_subsample,
                           tilt_smooth_sigma_px=4.0):
    """Alternative to the Newton-based forward-map inversion in
    :func:`apply_real_lens_traced`.

    NOT ORIGIN-AWARE (niche D9): it builds its own axis-centred grid.  It is
    reached only from ``inversion_method='backward_trace'``, which
    ``amplitude_model='ray_density'`` already refuses, and ``'ray_density'`` is
    a precondition for a non-zero ``origin`` -- so it is unreachable with one.

    **Validation** (2026-04-18):

    *   Single-ray forward-vs-backward OPL on a plano-convex singlet:
        **< 1 pm** (machine-precision agreement) when the exit-vertex
        correction is applied to both ends.
    *   End-to-end ``apply_real_lens_traced`` OPD RMS vs the Newton
        path: **~35-40 nm** on singlets at N=512.  The residual is
        not a bug in the reversal; it comes from using the
        finite-difference phase gradient of ``E_analytic`` as the
        backward-launch direction estimate (Newton uses the
        forward-trace's exact entrance-plane direction).  For
        design-verification work at lambda/10 tolerance this is deep
        in the margin; for sub-nm precision use Newton.

    Measured speed at N=512: ~1.7x faster than Newton on a singlet.
    Scales better to large N because the work is ``O(N^2)`` rather
    than ``O(N^2 * newton_iters)``.

    Algorithm in brief:

    Instead of ray-tracing the entrance grid forward and then
    Newton-inverting the spline of that map to find each exit pixel's
    entrance ray, we trace rays BACKWARD from a coarse subsample of
    the exit grid through the lens to the entrance, accumulating
    OPL along the way.  Fermat's principle makes OPL path-reversible,
    so the backward-trace OPL is numerically the same as the
    forward-trace OPL up to a sign convention.

    The exit-plane ray directions are derived from the local phase
    gradient of ``E_analytic`` (same mechanism as the input-aware
    forward launch, just applied at the exit).  The
    amplitude-weighted Gaussian smoothing keeps this robust on
    multi-mode inputs.

    Advantages over the Newton path (when it works):
        *   No spline fit, no Newton iteration.  The entire
            computation is a single forward pass of ``trace()``
            through a reversed prescription plus interpolation
            of the OPL map to the wave grid.
        *   Embarrassingly parallel in the trace itself (no
            dependencies between rays).

    Disadvantages / caveats:
        *   Accuracy depends on how well the exit-plane direction
            is extracted from ``E_analytic``.  Near a focus the
            true direction varies rapidly and the smoothed
            gradient is less representative; Newton handles this
            via the spline without needing a direction estimate.
        *   Only tested on singlet and doublet geometries so far;
            compound systems with intermediate foci may behave
            unexpectedly.  **Labelled experimental.**
    """
    from ..raytrace import _make_bundle, surfaces_from_prescription, trace

    N = int(N_grid)
    sub = max(1, int(ray_subsample))
    # Coarse exit-plane sampling (same stride pattern as the Newton
    # path's ``X[::sub, ::sub]`` slice so the final interpolation
    # grids line up identically).
    idx_c = np.arange(0, N, sub)
    N_c = idx_c.size
    x_c = (idx_c - N / 2.0) * dx
    Xc, Yc = np.meshgrid(x_c, x_c)

    # Extract exit-plane direction cosines from the phase gradient
    # of E_analytic, smoothed per the 3.1.3 multi-mode fix.
    L_out, M_out = _sample_local_tilts(
        E_analytic, wavelength, dx, Xc, Yc,
        smooth_sigma_px=tilt_smooth_sigma_px)

    # Build the reversed prescription + its surface list.  Note:
    # surfaces_from_prescription uses the per-element semi-diameter
    # plus the prescription-level aperture_diameter for vignetting;
    # both carry through to the reverse automatically.
    rev_rx = _reverse_prescription(lens_prescription)
    rev_surfaces = surfaces_from_prescription(rev_rx)

    # Rays start at the exit vertex plane (z=0) with direction
    # cosines (-L_out, -M_out, +sqrt(1-L^2-M^2)).  The sign flip on
    # (L, M) accounts for tracing in the reversed-axis frame:
    # "forward" here == backward in the original frame.  _make_bundle
    # computes N = +sqrt(1-L^2-M^2) which is the correct "forward"
    # direction in the reversed frame.
    rays = _make_bundle(
        x=Xc.ravel(), y=Yc.ravel(),
        L=-L_out.ravel(), M=-M_out.ravel(),
        wavelength=wavelength,
    )
    result = trace(rays, rev_surfaces, wavelength)
    final = result.image_rays

    # ---- Exit-vertex correction on the backward trace ----
    # trace() leaves rays at z = sag(last_surface_in_reversed_frame)
    # = sag of original S1 (the original entrance-side vertex) in
    # the reversed frame.  Without propagating each ray to z=0 of
    # this reversed-frame last surface (the original entrance
    # vertex plane), we under-count the OPL by the
    # vertex-to-sag leg in the final medium -- exactly the same
    # correction the forward path applies in apply_real_lens_traced
    # at lenses.py:1548-1556.  For on-axis rays this is zero; for
    # marginal rays on a strong-curvature lens it's tens of nm to
    # hundreds of nm.  Missing this is what made the first draft
    # of this function disagree with Newton by ~343 nm RMS.
    rev_surfaces_list = rev_surfaces
    n_exit_backward = get_glass_index(
        rev_surfaces_list[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_to_vertex = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit_backward * t_to_vertex
    # (We don't actually need to update x/y/z since we only
    # consume final.opd downstream, but keep it consistent.)
    final.x = final.x + final.L * t_to_vertex
    final.y = final.y + final.M * t_to_vertex
    final.z = np.zeros_like(final.z)

    # OPL: set NaN for dead rays (TIR / vignetted during the
    # reverse trace) so downstream NaN-propagation matches the
    # Newton path's treatment of out-of-domain points.
    opl_flat = np.where(final.alive, final.opd, np.nan)
    opl_coarse = opl_flat.reshape(Xc.shape)

    # Reference to on-axis so the returned OPL has the same origin
    # as the Newton path.  (Forward Newton does this at the spline
    # fit step via ``opl_grid = opl_grid - opl_grid[i_axis, i_axis]``.)
    i_c = N_c // 2
    ref = opl_coarse[i_c, i_c]
    if np.isfinite(ref):
        opl_coarse = opl_coarse - ref

    # Interpolate coarse OPL to the full wave grid, with the same
    # mode='nearest' + NaN-majority masking the Newton path uses.
    from scipy.ndimage import map_coordinates
    ii, jj = np.indices((N, N), dtype=np.float64)
    # Coarse sample u sits at FINE index u*sub (idx_c = arange(0, N, sub)),
    # so fine pixel ii maps to coarse coordinate ii/sub -- EXACT for any sub.
    # The previous ``ii * N_c / N`` equals ii/sub only when sub divides N;
    # otherwise it is a corner-anchored scale error that displaces the whole
    # map diagonally by (N/2)*(N_c*sub - N)/N pixels (audit
    # AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24: the traced chain's diagonal
    # focus walk -- measured -6.100 um at N=8192/sub=50 vs -6.11 predicted).
    # Bit-identical to the old expression whenever sub | N.
    coords = np.array([ii / sub, jj / sub])
    # v5.17.0 lifetime hygiene (same pattern as the Newton-path upsample):
    # ii/jj are folded into coords -- free them before interpolating, and
    # free coords before the final mask combine.  Byte-identical.
    del ii, jj
    opl_map = map_coordinates(
        np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
        coords, order=1, mode='nearest')
    nan_coarse = np.isnan(opl_coarse).astype(np.float64)
    nan_full = map_coordinates(
        nan_coarse, coords, order=1, mode='nearest')
    del coords
    opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
    return opl_map


def apply_real_lens_traced(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    ray_subsample: int = 8,
    n_workers: Optional[int] = None,
    progress: Optional[Any] = None,
    min_coarse_samples_per_aperture: int = 32,
    on_undersample: str = 'error',
    preserve_input_phase: bool = True,
    remap_sampling: str = 'lattice',
    tilt_aware_rays: bool = False,
    carrier: Optional[Any] = None,
    on_noncollimated: str = 'warn',
    fit_radius_beam_factor: Optional[float] = None,
    on_aperture_beam: str = 'warn',
    on_fit_domain_basis: str = 'warn',
    on_pool_memory: str = 'warn',
    beam_centre: Optional[Any] = None,
    decentred_fit_poly_order: Optional[int] = None,
    parallel_amp: Optional[bool] = None,
    parallel_amp_min_free_gb: float = 48.0,
    newton_amp_mask_rel: float = _NEWTON_AMP_MASK_REL_DEFAULT,
    newton_mask_dilate_coarse_px: int = 2,
    newton_max_iters: Optional[int] = None,
    inversion_method: str = 'newton',
    fast_analytic_phase: bool = False,
    newton_fit: str = 'auto',
    newton_poly_order: int = 6,
    use_gpu: bool = False,
    amp_use_gpu: bool = False,
    wave_propagator: Optional[str] = None,
    sag_dtype: Optional[Any] = None,
    sag_chunk_rows: Optional[int] = None,
    return_screen: bool = False,
    amplitude_model: str = 'screen',
    caustic: Optional[str] = None,
    output_plane_distance: float = 0.0,
    caustic_ray_subsample: int = 2,
    caustic_band: str = 'ludwig',
    caustic_min_area_ratio: float = 1e-6,
    origin: Tuple[float, float] = (0.0, 0.0),
    inverse_map: Optional[bool] = None,
    _exit_na_out: Optional[dict] = None,
    _remap_launch_out: Optional[dict] = None,
    _imap_out: Optional[dict] = None,
) -> np.ndarray:
    """Wave + per-pixel ray-traced phase variant of :func:`apply_real_lens`.

    See Also
    --------
    apply_real_lens :
        Faster (3-10x) analytic split-step model.  Use as the default
        when sub-nm OPD on multi-surface curved-interface systems
        isn't required and a coarser grid is preferable.
    apply_real_lens_maslov :
        Phase-space propagator via Chebyshev polynomial fit of the
        canonical map.  Caustic-safe and differentiable; preferable
        for JAX-autodiff optimisation loops and for output planes at
        or near a caustic.

    Quick decision guide (revised per the 2026-07 wave-lens-models audit)
    --------------------
    * Collimated / MLA-relayed input, thick or cemented optics, sub-nm OPD
      -> ``apply_real_lens_traced`` (this function), ``carrier=None``.
    * SINGLE divergent / converging / tilted source through a multi-element
      train -> ``apply_real_lens_traced(carrier='auto')`` (or a known
      conjugate): the carrier drives the reference residual to ~0.
    * MULTI-source / emitter-array direct imaging (e.g. the no-MLA TX case):
      a SINGLE carrier is insufficient -- each source is its own congruence,
      so a per-lens residual survives (the ``on_noncollimated`` guard keeps
      firing even with ``carrier='auto'``) and the spots stay soft.  Use
      ``apply_real_lens`` (all angles via ASM legs; the validated choice for
      this family -- mind its ``sag*theta^2`` oblique floor on fast /
      asymmetric designs, see its Oblique validity boundary).  A future
      K-carrier decomposition would extend the traced model here.
    * Genuinely multi-congruence fields, planes at/near a caustic, or
      JAX-autodiff design loops -> ``apply_real_lens_maslov`` /
      ``apply_real_lens_maslov_jax`` (``integration_method='local_quadrature'``
      at production NA).
    * Aberration-free paraxial reference / isolating model vs geometry
      -> the thin-lens ABCD equivalent.

    Description
    -----------
    For each pixel of the simulation grid, a geometric ray is launched
    from the entrance plane straight through the prescription using the
    sequential ray tracer in :mod:`lumenairy.raytrace`.  The
    accumulated optical path length (OPL) per pixel is used as the
    exit-plane phase, while the wave's *amplitude* envelope (vignetting,
    diffraction, edge effects) comes from a single ASM propagation of
    the entrance aperture to the exit-vertex plane.

    This eliminates the uniform-glass-slab approximation that limits
    the closed-form thin-element model on cemented doublets and other
    multi-surface curved-interface systems: each pixel sees the
    geometrically-correct glass path for its (x,y) position.  In
    practice the OPD agrees with the geometric ray trace to the
    sampling limit of the grid, at the cost of one ray trace per
    pixel (~3-10x slowdown relative to the analytic phase-screen
    model).

    Critical sampling rule
    ----------------------
    Extracting OPD from a converging wavefront requires

        dx <= lambda * f / aperture

    where ``f`` is the back focal length and ``aperture`` is the pupil
    diameter.  Coarser sampling makes ``np.unwrap`` lose cycles at the
    pupil edge, giving catastrophically wrong OPD values there.  Run
    :func:`lumenairy.analysis.check_opd_sampling` before a
    large ``apply_real_lens_traced`` call to verify.  If a coarser
    grid is required, use :func:`apply_real_lens` (with
    ``seidel_correction=True`` for doublets) instead.

    Limitations
    -----------
    * The DEFAULT (``carrier=None``) references the correction to a
      collimated plane wave (each pixel ray launched parallel to z), valid
      only when the input beam is ~collimated.  For a divergent / converging
      / tilted / emitter-array input, pass ``carrier=`` (a conjugate, an
      explicit wavefront, or ``'auto'``) to reference the beam's own
      congruence -- this generalises the model to those inputs (audit
      S5.1).  Without a carrier, such inputs blur; the ``on_noncollimated``
      guard warns or delegates to :func:`apply_real_lens` when it detects
      this regime.
    * Replaces the wave's exit phase with the geometric OPL; this
      gives correct OPD by construction but bypasses any wave-physics
      phase content that the ASM would have introduced (negligible for
      typical lens systems but worth noting).
    * Fresnel transmission and absorption are NOT applied here -- if
      you need them, run both this function and
      :func:`apply_real_lens` and combine.

    Parameters
    ----------
    E_in : ndarray, complex, shape (N, N)
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
    dx : float
        Grid spacing [m] (square pixels assumed for the traced model).
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.  Accepted for API
        symmetry with the rest of the lens family; the traced ray-
        subsample / interpolation paths currently require ``dy == dx``
        and will raise otherwise.  Use :func:`apply_real_lens` for
        anamorphic grids.
    bandlimit : bool, default True
        Passed to the (single) ASM propagation used for amplitude
        evolution.
    ray_subsample : int, default 8
        Compute the ray-trace OPL on every ``ray_subsample``-th pixel
        and bilinearly interpolate to the full grid.  OPL is a very
        smooth function of pupil position, so the default ``8`` (and
        ``ray_subsample=4``) typically loses < 1 nm of fidelity while
        cutting cost by ``ray_subsample**2``.  Set ``1`` to trace every
        pixel (no subsampling).  Recommended for production use on large
        grids.
    min_coarse_samples_per_aperture : int, default 32
        Guardrail against undersampled Newton inversion.  After
        ``ray_subsample`` is applied, the coarse output grid must have
        at least this many samples spanning the lens aperture,
        otherwise the cubic-spline interpolation of the wavefront will
        alias and the result will be wrong.  When the prescription has
        no ``aperture_diameter``, the effective pupil is the largest
        per-surface ``clear_aperture`` (capped at the launch diameter)
        if any surface carries one, else the launch diameter itself
        (= the grid extent ``N * dx``).  (v5.17.1, audit P3-08:
        previously the check was silently SKIPPED for apertureless
        prescriptions.)

        Empirical scaling on a singlet at lambda = 1.31 um:

        ====================  ==================
        coarse-samples / ap   typical RMS phase
        ====================  ==================
        64                    ~20 nm
        32 (default safe)     ~85 nm
        16                    ~350 nm  (unusable)
        ====================  ==================

        Pass ``0`` to disable the check entirely.
    on_undersample : ``'error'`` (default) / ``'warn'`` / ``'silent'``
        What to do when the coarse-sample count falls below
        ``min_coarse_samples_per_aperture``.  ``'error'`` raises
        ``ValueError`` with the safe ``ray_subsample`` value computed
        for the current grid; ``'warn'`` logs via the ``warnings``
        module and continues; ``'silent'`` is the explicit "I know
        what I'm doing" escape hatch.
    n_workers : int, optional
        Number of worker *processes* for the Newton-inversion step.
        Defaults to
        :func:`lumenairy._backends.available_cpus` -- the
        affinity-aware count of CPUs this process can actually use
        (respects cgroup limits, ``taskset`` masks, Python 3.13+
        ``process_cpu_count``, Windows process affinity).  Pass 1 to
        force the in-process serial path (useful for reproducible
        timings or when called from a parent pool that already
        saturates the machine).

        .. note::
           Engages for BOTH ``newton_fit`` backends on the CPU path
           (``use_gpu=False``) since v5.30.1.  It used to be a silent
           no-op for the default ``newton_fit='polynomial'``, because the
           process worker only knew how to rebuild a SciPy spline
           (``RectBivariateSpline``) and not the polynomial
           ``_Cheb2DEvaluator``; the effect was that the default
           configuration lost the pool speed-up with no diagnostic, so
           whether you got parallel Newton depended on a knob most callers
           never set.  The worker now rebuilds whichever fit the caller
           chose, from the same pickled grids, and mirrors the serial
           path's combined value+gradient evaluation -- so the pool result
           stays bit-identical to serial for both backends.

           Still in-process for ``use_gpu=True``: shipping CuPy device
           arrays through a ``ProcessPoolExecutor`` would host-copy them.

           The size gate is TWO-TIER (v5.30.2) plus a second-call
           promotion (v5.32.2, review D1).  A cold process needs
           ``_POOL_MIN_PIXELS`` (200 000) Newton points before the pool is
           worth a spawn -- at 16 384 a cold pool measured 1.62x SLOWER --
           so a genuine one-shot call stays serial.  Once a pool is live,
           or once this process has already run sub-cold-bar inversions
           serially (i.e. it is a chain or a sweep, not a one-shot) AND
           TIMED them above ``_POOL_PROMOTE_MIN_SECONDS``, the bar drops to
           ``_POOL_MIN_PIXELS_WARM`` (8 000).  A multi-group chain
           therefore runs its first ``_POOL_PROMOTE_MIN_SAMPLES`` groups
           serial and pools every group after them; before the promotion
           existed the warm bar was unreachable from a cold process and
           every group of a design-121-class chain (65 536 points/group at
           ``ray_subsample=4``) ran serial.  Coarse ``ray_subsample`` below
           the warm bar stays serial by design.

           The promotion is COST-gated, not size-gated: at 65 536 points
           the serial Newton step measures 0.048 s on the default
           polynomial+numba path against a ~0.22 s per-dispatch pool
           overhead, so that path deliberately keeps running in-process
           (its Chebyshev evaluator is already an ``@njit(parallel=True)``
           kernel using every core), while spline (0.553 s) and
           polynomial-without-numba (0.95 s) promote and win.  The timing
           evidence is kept per COST CLASS -- worker count, fit backend and
           a point-count band -- so a measurement taken on one backend or
           at one grid size never promotes calls it says nothing about
           (finding V5).
    tilt_aware_rays : bool, default False
        If True, each ray's initial direction ``(L, M)`` is derived
        from the local phase gradient of ``E_in`` at the entrance
        position (the "Tier 1 input-aware ray launch" added in 3.1.2).
        If False (the default), collimated rays are launched
        (L = M = 0 everywhere) and the plane-wave lens-OPL reference
        is used.

        **Why the default flipped from True to False in 3.1.3:**  When
        ``preserve_input_phase=True`` (also the default), the exit
        field is assembled as

            E_out = E_analytic * exp(i * delta_phase)
            delta_phase = k0 * opl_traced - phase_analytic_lens

        where ``phase_analytic_lens`` is the phase produced by running
        :func:`apply_real_lens` on a unit PLANE WAVE -- i.e. a
        plane-wave reference.  For ``delta_phase`` to be a
        mathematically clean "ray-traced minus analytic" correction,
        ``opl_traced`` must use the same reference: a plane-wave
        entrance launch.  With ``tilt_aware_rays=True``, ``opl_traced``
        instead mixes the lens-model correction with per-pixel
        tilt-induced phase shifts that the plane-wave ``phase_analytic_lens``
        does not contain.  The resulting ``delta_phase`` is only
        approximately right for small/uniform input tilts, and breaks
        materially on multi-mode inputs (post-DOE fields, strongly
        off-axis compound beams) where the per-pixel tilts vary
        significantly across the pupil.

        The 3.1.4 default ``tilt_aware_rays=False`` restores the
        reference-consistent plane-wave launch that pre-3.1.2 releases
        used, so ``delta_phase`` remains well-defined for any input the
        wave model can represent.  If you have a specifically small,
        uniform input tilt and want the per-ray OPL variation (e.g.
        rigorous off-axis lens characterisation with a single tilted
        input), pass ``tilt_aware_rays=True`` explicitly and validate
        against the default on your specific case.

        When this flag is True, tilts are clipped to
        ``|sin(theta)| <= 0.5`` (~30 deg) for numerical safety and
        amplitude-weighted-Gaussian-smoothed (``smooth_sigma_px=4``
        by default inside :func:`_sample_local_tilts`) to tame
        multi-mode aliasing; neither applies when the flag is False
        (the default).
    carrier : float | ndarray | 'auto' | None, default None
        Reference congruence for the traced correction (audit S5.1).  The
        default (``None``) references the correction to a PLANE WAVE (unit
        input for ``phase_analytic_lens``; rays launched parallel to z),
        which is valid only when the input beam is ~collimated.  For a
        DIVERGENT / converging / tilted / emitter-array input (e.g. no-MLA
        direct imaging), supply the beam's smooth carrier wavefront so the
        reference matches the beam:

        * ``float`` -- an on-axis point-source conjugate at signed distance
          ``s`` metres (``W = (x^2+y^2)/(2s)``; ``s > 0`` diverging in front).
        * ``ndarray`` -- an explicit wavefront ``W(x, y)`` in metres,
          same shape as ``E_in`` (reference phase = ``k0 * W``).
        * :class:`TiltedCarrier` ``(R, L, M, x0, y0)`` -- the exact sphere
          ``R`` transversely centred at ``(x0, y0)`` PLUS a uniform tilt
          ``(L, M)``, evaluated analytically (niche D1).  This is the
          per-ORDER reference for a post-DOE fan: each order is a single
          clean congruence once its own tilt is carried, so the residual
          stays inside the ``on_noncollimated`` envelope instead of being
          the full split angle.  ``R=inf`` gives the pure tilted plane wave.
          Prefer it over an equivalent ``ndarray``: the ndarray branch can
          only differentiate/sample by ``np.gradient`` + nearest neighbour,
          which quantises the ray-launch cosines to the grid.
        * ``'auto'`` -- fit a low-order polynomial carrier from ``E_in``'s
          intensity-weighted, wrapping-safe local tilt field (never
          per-pixel gradients -- that is the ``tilt_aware_rays`` failure
          mode).  Extracts the smooth COMMON wavefront; the correct choice
          for a single divergent source of unknown conjugate.

        With a carrier the exit reference is well-conditioned (it focuses
        where the real beam does) and the rays launch along the carrier
        normals, so the traced OPL is applied to the small angular RESIDUAL
        only.  ``carrier`` forces ``fast_analytic_phase=False`` (the fast
        geometric reference cannot carry the carrier congruence).

        Validity: a SINGLE carrier only helps when the residual after its
        removal is small.  It is INSUFFICIENT for genuinely multi-congruence
        fields -- an emitter array whose per-source residual (source spread
        / throw) is not small (e.g. the no-MLA TX imaging case; measured
        design-119 per-lens residual ~0.02-0.04 rad even with
        ``carrier='auto'``, so the ``on_noncollimated`` guard keeps firing
        and the spots stay soft), comparable-power beams at well-separated
        angles (post-DOE at large split), or planes at/near an intermediate
        focus.  Use :func:`apply_real_lens` (split-step, all angles) or
        :func:`apply_real_lens_maslov` there.
    on_noncollimated : {'warn', 'delegate', 'off'}, default 'warn'
        Policy when the input's residual angular spread (after removing any
        ``carrier``) exceeds the collimated-reference validity threshold --
        i.e. the plane-wave-referenced correction would blur (the silent
        regression class the audit was written for).  ``'warn'`` emits a
        ``RuntimeWarning`` pointing at ``carrier=`` / :func:`apply_real_lens`;
        ``'delegate'`` transparently falls back to :func:`apply_real_lens`
        (a ``RuntimeWarning`` lists any traced-only physics kwargs the
        analytic model cannot honour); ``'off'`` disables the check (and its
        one-FFT-free cost).  ``'silent'`` and ``'ignore'`` are accepted
        aliases for ``'off'`` (the sibling knobs here spell suppression
        ``'silent'``).  v5.29.1 (audit E-M3): any OTHER value now raises --
        it used to select ``'warn'`` silently, and ``'silent'`` in particular
        therefore warned instead of suppressing.
    inversion_method : {'newton', 'fit', 'backward_trace'}, default 'newton'
        How the entrance->exit map is inverted to get the per-pixel OPL.
        ``'newton'`` (default, fully validated) fits the forward map and runs
        the per-pixel Newton inversion; ``'fit'`` (T-P2) fits the SCATTERED
        inverse map directly (no Newton loop -- cheaper, slightly less
        accurate); ``'backward_trace'`` is the experimental direct backward
        trace through the reversed prescription.  ``amplitude_model=
        'ray_density'`` requires ``'newton'``.  v5.29.1 (audit E-M4): an
        unrecognised value now raises instead of silently running Newton.
    inverse_map : bool, optional
        Per-call override of the module gate
        ``lumenairy.elements._lens_imap.TRACED_INVERSE_MAP``; ``None``
        (default) follows it, and it ships ``True``.

        WHAT IT SELECTS.  At ``ray_subsample > 1`` the shipped path Newtons the
        COARSE lattice ``X[::sub, ::sub]`` and interpolates that answer to the
        wave grid -- the OPL (order 3), its NaN mask, the ray-density
        amplitude, its NaN mask, and the ``remap_sampling='full'`` entrance
        pull-back.  With the map engaged, an exit-coordinate Chebyshev model
        fitted from the landings THIS call already traced supplies all of them
        EXACTLY, per pixel, in one pass.  MEASURED on design 121's last group
        at ``n_fine = 8192``: those six full-grid ``map_coordinates`` calls are
        14.767 s of a 96.9 s element; the map's evaluation is 1.910 s and its
        build ~0.2 s, and it needs no extra rays.

        The map is used only when every guard passes -- one congruence, an
        unfolded Jacobian, a live ray set, a landing hull it can be honest
        about, a degree that reaches the bar, and PARITY with the very Newton
        path it replaces at OFF-LATTICE probe points, against ground truth
        from the element's own trace there.  A guard that fires keeps the
        shipped path unchanged and reports why (``INVERSE_MAP_GUARD``).
        ``False`` is the FAIL-BEFORE, and it is byte-identical to the
        pre-feature library; see that flag's own note for the exact-trace
        oracle and the design-121 banner measurement that decided the
        default.  Ignored at ``ray_subsample == 1`` (there is no coarse lattice
        to replace), with ``inversion_method != 'newton'``, with
        ``use_gpu=True``, and on the row-band assembly path (which exists
        precisely so a full-grid float64 is never materialised).
    newton_max_iters : int, optional
        Newton iteration cap; ``None`` (default) uses the module default
        (12).  Honoured by BOTH the serial and the process-pool inversion
        paths since v5.29.1 (audit E-H2 -- the pool worker previously
        hard-coded 12, making this knob inert whenever the pool engaged; at
        that time that meant >=200k Newton points with ``newton_fit='spline'``
        on the CPU path, whereas the pool now serves EITHER fit above a
        two-tier size gate -- see ``n_workers``).
        When more than 1% of pixels are still unconverged at the cap, both
        paths emit the same ``RuntimeWarning`` (suppressed by
        ``on_undersample='silent'``).
    newton_amp_mask_rel : float, default 1e-4
        Skip the Newton solve on coarse pixels whose analytic amplitude is
        below this fraction of ``amp.max()`` (they contribute ~nothing to the
        final field); ``0.0`` disables the mask and runs Newton on the whole
        coarse grid.  ``amplitude_model='ray_density'`` REQUIRES ``0.0`` (its
        amplitude carries the coma tail where ``|E_analytic|`` is small) --
        since v5.29.1 an explicit non-default value there raises instead of
        being overridden silently (audit E-M5).

    fit_radius_beam_factor : float, optional
        **Aperture:beam cliff guard** (P2, audit
        AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4).  When set, the
        entrance ray samples that enter the forward-map / OPL fits are
        restricted to the BEAM-RELATIVE disc
        ``r <= fit_radius_beam_factor * w_in``, where ``w_in`` is the input
        beam's 1/e amplitude radius measured from ``E_in``
        (``sqrt(2 <r^2>)``).  ``None`` (default) keeps the historical
        aperture-only domain (byte-identical).

        This decouples the ray-fit domain from the VIGNETTING aperture: when
        the physical aperture greatly exceeds the beam, the launch square's
        marginal / corner rays sample surface zones the beam never occupies,
        and on a fast surface their out-of-basis high order corrupts the global
        low-order fit *inside* the beam -- a sharp cliff, not a gradual
        degradation (E4 corrected relay, beam w = 2 mm: exit-wavefront Strehl
        0.998 at a 6 mm aperture -> 0.105 at 7 mm -> 0.039 at 10 mm).

        Only the FIT domain is restricted.  ``launch_radius``, the Newton
        ``bound`` and the out-of-domain NaN threshold are untouched, so the
        Newton inversion still spans the full launch domain, the smooth
        low-order fit still extrapolates over the whole aperture, and **no
        field energy is clipped** -- unlike clamping the aperture itself,
        which vignettes real halo power (audit
        AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23 §4c.3).  The restriction
        is combined (``min``) with the carrier-gated
        ``_CARRIER_FIT_RADIUS_FRAC`` domain when a carrier is engaged, and is
        abandoned (falling back to the unrestricted domain) if the disc would
        hold fewer than 64 coarse samples.

        ``2.0`` is the validated default used by
        :func:`lumenairy.propagate_traced_carrier_chain`; the recovery is flat
        for 1.5-2.5 (measured E4 Strehl at a 2.5x-beam aperture: 0.9988 at 1.5,
        0.9995 at 2.0, 0.9967 at 2.5, 0.9591 at 3.0 -- above ~2.5 the marginal
        rays start coming back in).  Note the resulting screen is
        input-DEPENDENT (it reads ``|E_in|``), so it is not meaningful with
        ``return_screen=True`` / :func:`prepare_real_lens_traced` -- on the
        flat ``ones`` placeholder those use, the measured ``w_in`` spans the
        grid and the restriction is inert.

        Both ``w_in`` and the disc are referenced to ``beam_centre`` (below),
        so a DECENTRED beam is guarded exactly as an on-axis one is.

        **NOT a soft knob on a real carrier chain (2026-07-31).**  The
        "flat for 1.5-2.5" reading above is an E4 Strehl on a low-NA singlet.
        Measured END TO END on design 121's post-DOE chain at order (-4,-2)
        against the landed niche-C6 launch, raising it 2.0 -> 3.0 costs
        **-77.5 EE3 points** (87.771 -> 10.290 %, ``P_tile`` -22.2 points) and
        is the only configuration in that study to raise a fold-caustic
        warning.  At ELEMENT level the same change puts 1.0-1.1e-02 of the
        input power beyond 4 mm of the exit chief ray.  Against pinned HEAD --
        i.e. with the C6 defect open -- the SAME change read **+0.66 EE3
        points**, so this knob's sign is not stable across that fix.  Treat
        2.0 as load-bearing on a carrier chain and re-measure before moving it.
        See docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S1.
    beam_centre : (float, float), optional
        Transverse position ``(x0, y0)`` (metres) of the input beam on the
        wave grid.  ``None`` (default) means: take it from ``carrier`` when
        that is a :class:`TiltedCarrier` (its ``(x0, y0)`` IS the chief-ray
        position at this plane), else the grid origin -- so the on-axis
        default path is byte-identical.

        This is what keeps the ``fit_radius_beam_factor`` cliff guard honest
        off axis.  Both the measured beam radius (an intensity second moment)
        and the ray-FIT disc are otherwise referenced to the grid ORIGIN, so a
        beam of radius ``w`` at ``x_c`` reads ``sqrt(2 x_c^2 + w^2)`` and the
        disc it sizes grows with the DECENTRE instead of following the beam:
        measured (30 mm aperture, ``w`` = 1 mm, ``fit_radius_beam_factor=2``)
        the guard degrades from "disc == the beam" at ``x_c`` = 0 to the FULL
        launch domain -- i.e. inert, the exact regime the guard exists to
        exclude -- by ``x_c`` = 8 mm, silently.  Set this (or pass a
        :class:`TiltedCarrier`) whenever the beam is not on the grid centre.

        Only the ray-FIT domain and the warn-only aperture:beam ratio move
        with it; ``launch_radius``, the Newton ``bound`` and the out-of-domain
        threshold stay aperture-derived, so nothing is vignetted.  The
        carrier-gated ``_CARRIER_FIT_RADIUS_FRAC`` disc stays APERTURE-centred
        (it exists to drop near-vignetting marginal rays, which is a property
        of the aperture, not of the beam); the two discs INTERSECT.

        An off-centre restriction is applied by WEIGHTS, not by dropping the
        out-of-disc ray samples: a hard mask leaves the global fit's remaining
        freedom unconstrained, and off centre the fitted forward map then folds
        and the returned field grows a bright spurious lobe far from the beam.
        See ``_FIT_DISC_OUTSIDE_WEIGHT_REL``.  The concentric (on-axis) path
        keeps the historical hard mask and is byte-identical.

        "Off centre" is a MEASURED threshold, not ``!= 0``: an offset within
        ``max(0.5 * dx, 0.05 * w)`` of the grid centre keeps the concentric
        path byte-identically (a physically null offset must not swap the fit
        branch -- measured, it moved the returned field by 8.3e-6 of peak at
        1e-9 pixels of decentre).  See ``_DECENTRE_GATE_W_FRAC`` for the sweep
        that set both floors.
    on_aperture_beam : {'warn', 'silent'}, default 'warn'
        Policy for the warn-only half of the cliff guard: emit a
        ``RuntimeWarning`` when the physical aperture diameter exceeds
        ``1.5x`` the beam 1/e^2 diameter (the last measured-clean ratio) and no
        ``fit_radius_beam_factor`` restriction is active -- i.e. the call is in
        the regime where a fast surface can silently mis-report.  ``'silent'``
        suppresses it.  Whether the cliff actually bites depends on how
        aberrated the surfaces are out at the aperture edge (a gently-corrected
        group tolerates 6x), so this is a *possible-cliff* flag, not a
        diagnosis.

        FIX D5 (2026-08-06): "no ``fit_radius_beam_factor`` restriction is
        active" now means what it says.  The suppression used to key on the
        knob being PASSED, so on a basis that cannot apply the restriction an
        inert knob silenced the warning about the very failure it had stopped
        preventing.  See ``on_fit_domain_basis``.
    on_fit_domain_basis : {'warn', 'error', 'silent'}, default 'warn'
        What to do when the RESOLVED ``newton_fit`` basis cannot honour a
        fit-domain restriction the caller asked for.  The restriction is
        implemented as least-squares WEIGHTS or as a hard NaN mask, and
        neither has a home on ``newton_fit='spline'``'s
        ``RectBivariateSpline`` (it takes no weights, and one NaN in its data
        array makes ``.ev()`` return NaN at the grid CENTRE) -- so
        ``fit_radius_beam_factor``, ``decentred_fit_poly_order`` and the
        niche-C11/C12 branch selection are INERT there.

        Usually harmless: a local interpolant does not spread marginal-ray
        error into the beam the way a global polynomial does (measured
        pointwise against an exact skew ray trace, 0.006 / 0.002 um in the
        skirt / at the rim against the polynomial's 5.608 / 15.079).  But NOT
        a safe substitute past the aperture:beam cliff -- on the E4 corrected
        relay with the beam held fixed and only the aperture grown, the
        polynomial basis degrades to exit-wavefront Strehl 0.042 and
        ``fit_radius_beam_factor=2.0`` recovers it to 0.999, while the spline
        basis returns an ALL-ZERO exit field the knob cannot rescue.  So the
        inertness is ANNOUNCED rather than assumed benign.  ``'warn'``
        (default) names which knob is inert; ``'error'`` refuses the
        combination; ``'silent'`` acknowledges it -- which is what a caller
        deliberately using spline as a FIT-DOMAIN-FREE reference should pass.
        ``'ignore'`` and ``'off'`` are accepted ALIASES for ``'silent'``
        (``lumenairy.propagators.carrier``'s ``on_*`` knobs spell suppression
        ``'ignore'``, this signature's siblings spell it ``'silent'``).
        Validated at entry since v5.32.2 (finding V4): any other value raises
        ``ValueError``.  Before that gate every unrecognised value -- including
        ``'Error'`` and ``'ignore'`` -- silently selected ``'warn'``, so a
        caller asking for a fatal got a warning and a returned field.
    on_pool_memory : {'warn', 'silent'}, default 'warn'
        Policy for the Newton process pool's MEMORY CAP notice (v5.32.3).
        When the pool this call asked for does not fit the box -- the projected
        per-worker commit against ``_NEWTON_POOL_RAM_FRAC`` of available memory
        less a ``_NEWTON_POOL_MIN_FREE_GB`` reserve -- the clamp lowers
        ``n_workers`` and, by default, emits one ``RuntimeWarning`` naming what
        was asked for, what a worker costs, what the box has and what will run.
        ``'silent'`` clamps identically and says nothing; ``'ignore'`` and
        ``'off'`` are accepted ALIASES for it (same two-house-style collision
        ``on_fit_domain_basis`` resolves the same way).  Validated at entry:
        any other value raises ``ValueError``.

        This knob changes what is REPORTED, never what is run: the pool path is
        bit-identical to serial at every worker count, so a clamped dispatch
        returns the same numbers as an unclamped one and only the wall time
        moves.  There is deliberately no ``'error'`` -- the clamp is what keeps
        an under-provisioned box completing at all.

        Pass ``'silent'`` when a test or a harness asserts on the ABSENCE of
        warnings from a physics guard: the cap is a resource notice whose firing
        depends on the box's free RAM, so leaving it routed through the default
        makes such an assertion pass on a 256 GB workstation and fail on a
        12 GB CI runner (which is exactly how it was found -- FIX_CI_POOL).
        The unguarded-``__main__`` refusal is NOT routed through this knob: it
        reports that spawn workers would re-run the caller's whole program,
        side effects included, which is a correctness hazard rather than a
        resource notice.
    decentred_fit_poly_order : int, optional
        Minimum tensor-Chebyshev total degree for the ray fit WHEN THAT FIT'S
        DISC IS OFF CENTRE (niche D7).  ``None`` (default) uses
        ``_DECENTRED_FIT_POLY_ORDER`` = 10; the effective order is
        ``max(newton_poly_order, this)``, so a caller asking for more still
        gets more, and passing your own ``newton_poly_order`` restores the
        pre-D7 behaviour exactly.

        Why it exists: an off-centre disc of radius ``r`` about a chief ray
        ``|c|`` off axis covers the aperture out to ``|c| + r`` instead of
        ``r``, so the same degree buys a worse fit over strictly more aberrated
        territory.  Measured on design 121's last group at 0.97 beam radii of
        decentre, the OPL residual over the beam is **14x** the on-axis one at
        order 6 (2.508 nm vs 0.177 nm) and recovers 20x at order 10 (0.121 nm);
        end to end on the ``K = -n^2`` conic stand-in the chain/oracle EE2 ratio
        goes 0.9498 -> 0.9828 at one beam radius of decentre.  See
        ``_DECENTRED_FIT_POLY_ORDER`` for the full sweep, the cost, and the
        basis-domain re-map that was measured and REFUSED.

        Ignored (and the on-axis path byte-identical) whenever the disc is
        concentric with the launch square, and on ``newton_fit='spline'``,
        which takes no fit-domain restriction at all.

        CAN GO INERT, SILENTLY.  The raise is capped by the "3 samples per
        basis term" step-down below it: order 10 needs 198 in-disc COARSE ray
        samples, and the loop walks the order back down to
        ``newton_poly_order`` until that holds.  The disc holds about
        ``pi * (fit_radius_beam_factor * w / (dx * ray_subsample)) ** 2``
        samples, so the raise survives only while

            ``fit_radius_beam_factor * w / (dx * ray_subsample) >~ 7.9``

        i.e. while the fit disc spans ~8 coarse pixels in radius.  Both
        documented configurations clear this at the default
        ``ray_subsample=8`` -- but not by much, and ONE step is enough to lose
        it, with no warning and no diagnostic:

          * the synthetic f/6 example above (N=512, dx=30 um, w=1.0 mm,
            ``fit_radius_beam_factor=2``) holds 223 samples against 198, a
            **1.13x** margin; at ``ray_subsample=16`` it holds 56 and the
            order falls straight back to 6, i.e. D7 is fully inert.
          * design 121's last group (N=1024, dx=33.211 um, w=3.1255 mm) holds
            1735 at ``ray_subsample=8`` and 432 at 16, and goes inert at 32.

        So a caller who coarsens the ray grid to buy speed can silently get
        the pre-D7 fit back.  If the off-centre accuracy matters, keep
        ``ray_subsample`` low enough to satisfy the inequality above rather
        than assuming the raise is in force.

    preserve_input_phase : bool or 'remap', default True
        If True, the input field's phase structure (source tilts,
        MLA / DOE phase modulation, off-axis wavefronts, etc.) is
        preserved through the lens and combined with the ray-traced
        OPL correction.  This is the physically-correct behaviour
        and matches what :func:`apply_real_lens` does (with the added
        benefit of corrected geometric OPL).

        If False (legacy behaviour prior to v3.1.2), the output is
        ``|E_analytic| * exp(i*k0*OPL_traced)`` -- the input-field
        phase is discarded entirely and only the lens's ray-traced
        OPL is retained.  Use this mode when you specifically want
        the lens-only OPD response on a synthetic plane wave;
        otherwise keep the default.

        ``'remap'`` (requires ``amplitude_model='ray_density'``; audit
        AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S6.7/S8): the input's
        RESIDUAL phase -- the input field de-chirped by the carrier
        eikonal, ``angle(E_in * exp(-i*k0*W_carrier))``, with the
        identity de-chirp (W=0) when the carrier is absent or too flat
        to engage (a collimated beam's own phase IS its residual) -- is
        transported to the exit GEOMETRICALLY, sampled at each exit
        pixel's Newton-inverted entrance point (the same pullback the
        ray-density amplitude uses for ``|E_in|``), and multiplied
        onto the ``k0*OPL_traced`` exit phase.  Unlike ``True`` it
        never touches the analytic wave pair (whose phase corrupts
        with grid refinement on a carrier-referenced input: 0.015 ->
        0.243 rad/group as dx 20 -> 5 um on the 121 front group), so
        it is dx-independent by construction; unlike ``False`` it
        does not discard the input's genuine residual (a corrected
        relay's inter-group pre-shaping).  For a pure carrier-sphere
        input the residual is ~0 and 'remap' coincides with
        ``False``.  See ``remap_sampling`` for the resolution at which
        that residual is sampled -- it is the accuracy-limiting choice
        of the mode.

        Niche C6 (2026-07-30): the rays are launched along the TOTAL
        entrance eikonal ``grad(W + a_fit)``, not the carrier's
        ``grad(W)`` alone, so the entrance pullback lands on the
        STATIONARY point of ``W + a + V(., X)`` and the second-order
        term ``1/2 grad a^T H^-1 grad a`` this mode used to drop is
        carried.  Worth 14.8 EE3 points on design 121's worst DOE
        order.  See :data:`REMAP_STATIONARY_PHASE_LAUNCH` for the
        derivation, the measurements and the fail-before switch.

        Cost: ``preserve_input_phase=True`` runs the analytic
        apply_real_lens *twice* (once for the input field, once for
        a unit plane-wave reference so we can subtract the analytic
        lens phase before adding the traced one).  This roughly
        doubles the ~40 % amplitude-leg budget.  At large N the
        total overhead is ~20 %.

        Implementation note: the work is dispatched via
        ``concurrent.futures.ProcessPoolExecutor`` rather than threads
        because SciPy's ``RectBivariateSpline.ev`` does not release
        the GIL in current versions, so threading delivers no
        speedup.  Each worker rebuilds the splines locally from
        their knot data (cheap), avoiding the pickle cost of the
        spline objects themselves.  Sequential fallback is used when
        the coarse grid is below ~200 k pixels (pool startup cost
        dominates) or when pool spawn fails.  Measured speedup on
        large grids: ~8x on 16 cores.

    remap_sampling : {'lattice', 'full'}, default 'lattice'
        Resolution at which ``preserve_input_phase='remap'`` samples the
        transported residual phasor.  Ignored unless ``'remap'`` is in
        force.

        ``'lattice'`` (default, pre-S12 behaviour): the residual phasor is
        sampled at the COARSE ray lattice's entrance pullback points
        (pitch ``ray_subsample * dx``) and the resulting phasor is then
        bilinearly upsampled to the wave grid.

        ``'full'``: only the SMOOTH entrance pullback COORDINATES are
        upsampled from the coarse lattice; the residual phasor is then
        sampled at full wave-grid resolution.  Same cost class (two extra
        ``map_coordinates`` calls on N^2 smooth arrays), strictly better
        physics, and an exact no-op at ``ray_subsample == 1``.  Transient
        peak memory is ~5 float64/complex128 arrays of the wave grid while
        the residual map is built (~3 GiB at the N = 8192 fine retrace leg),
        released immediately afterwards.

        Why it matters (audit S12, measured on design-121): the residual a
        carrier-regime chain carries is the design's own correction, which is
        r^4-dominant, so its phase GRADIENT grows as r^3.  On a lattice of
        pitch ``h = ray_subsample*dx`` the phasor therefore exceeds Nyquist
        beyond a finite radius ``r_alias = (pi w^4 / (4 A h))^(1/3)`` (with
        ``A`` the r^4 residual in rad at ``r = w``), and outside that radius
        the transported residual is ALIASED -- the beam skirt receives
        scrambled phase.  On the design-121 final leg (``A = 9.2`` rad,
        ``w = 3.124`` mm, ``h = 50 * 1.524`` um) the prediction is
        ``r_alias = 1.52 w``, and the measured exit residual rms jumps from
        0.052 rad (r < w) and 0.139 rad (1-1.5 w) to 0.971 rad (1.5-2 w) --
        95 % of the whole Strehl-loss variance, from 1.2 % of the power.

        Measured effect of ``'full'`` (design-121, pure chain defaults
        otherwise, N = 2048 / NFC = 8192 / WF = 4.0, through-focus scanned at
        1 um, on-axis): exit-wavefront 1.5-2 w annulus rms **0.971 -> 0.486
        rad**, amplitude-weighted pupil Strehl **0.910 -> 0.931**,
        window-total **99.44 -> 99.79 %**; at the acceptance focus
        (dz = +10 um) FWHM **3.550 -> 3.450 um**, EE3 **88.19 -> 88.83**,
        EE6 **99.28 -> 99.58**; at each metric's own optimum EE3
        89.57 -> 90.01 and EE6 99.26 -> 99.55 against an ideal-field ceiling
        of 90.73 / 99.74 through the same readout.  Unchanged between
        N = 2048 (``ray_subsample=4``) and N = 4096 (``ray_subsample=8``) to
        the digit.  Combined with ``ray_subsample=2`` it reaches EE6 99.73 --
        the ceiling.

        The deeper reason to prefer it is CONVERGENCE, not the point gain:
        the mode is documented as dx-independent by construction, but with
        ``'lattice'`` its result depends on ``ray_subsample`` (measured
        amplitude-weighted rms phase difference vs the full-resolution
        ``ray_subsample=1`` reference: 0.55 / 0.84 / 1.09 rad at
        ``ray_subsample`` 2 / 4 / 8).  With ``'full'`` the same differences
        are 0.0001 / 0.0003 / 0.0059 rad -- a 180-9000x reduction, i.e. the
        transported residual becomes a property of the physics rather than of
        the ray-fit lattice.  ``'lattice'`` is kept as the default only for
        byte-compatibility; ``'full'`` is the correct sampling and the
        recommended setting for any carrier-regime chain.

        **The POINT GAIN is gone once niche C6 lands; the CONVERGENCE argument
        is not (2026-07-31).**  On the worst tilted order of design 121,
        `(-4,-2)`, end to end against the landed C6 launch, ``'lattice'``
        measures **+0.0988 EE3 points** -- i.e. marginally BETTER than
        ``'full'``, against **-17.73 points** for the same substitution on
        pinned HEAD with the C6 defect open.  That is expected: at HEAD the
        launch went along ``grad(W)``, so the residual was being SAMPLED at the
        wrong foot and the sampling resolution mattered enormously; with the
        stationary-phase launch it is sampled at the right one.  Nothing above
        is retracted -- the dx-independence measurement is what this default
        rests on, and it is untouched -- but do NOT expect ``'full'`` to buy
        EE points on a post-C6 chain.  See
        docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S1.

    sag_dtype : {None, np.float32, np.float64}, default None
        v5.17.0 opt-in geometry dtype, forwarded to the internal
        :func:`apply_real_lens` amplitude legs.  ``None`` (default)
        resolves to the process-wide :func:`set_lens_sag_dtype` value
        (float64 by default -- byte-identical to prior releases).
        ``np.float32`` is ACCURACY-RISKY -- validate the prescription
        with :func:`lens_sag_float32_opd_error` first.  See
        :func:`apply_real_lens` for details.
    sag_chunk_rows : int or None, default None
        v5.17.0 row-band (chunked) memory mode: banded per-surface
        phase screens inside the internal :func:`apply_real_lens`
        amplitude legs AND banded OPL-upsample / exit-field assembly
        here (the latter on the ``ray_subsample > 1`` Newton path).
        ``None`` -> AUTO: row-banded (``max(256, N // 16)`` rows per
        band) when ``N >= 4096``, whole-grid below.  ``0`` forces the
        whole-grid path in BOTH stages; a positive int forces that
        band size.  Byte-identical to the whole-grid path, WITH ONE
        NAMED EXCEPTION since v5.35: the band path refuses the
        inverse-characteristic evaluator by construction (see
        ``_imap_domain_gate`` and its note below), so on a call the
        evaluator would otherwise engage -- ``ray_subsample > 1``,
        ``inversion_method='newton'``, not ``use_gpu``, and the
        ``'screen'`` amplitude, since ``'ray_density'`` forces the band
        path off anyway -- banding selects the incumbent coarse-Newton
        inversion instead.  Byte-identity then holds against the
        whole-grid path with ``inverse_map=False``, not against the
        default (measured 2.19e-02 relative on the niche-S10 carrier
        fixture).  Refused, never degraded: that incumbent is the
        pre-v5.35 shipped inversion.  Pass ``sag_chunk_rows=0`` on a
        large-N call to keep the evaluator.
    amplitude_model : {'screen', 'ray_density'}, default 'screen'
        Which model supplies the exit-plane AMPLITUDE (the phase is the
        ray-traced OPL either way).

        * ``'screen'`` (default) -- the historical hybrid amplitude: the
          magnitude of a single analytic :func:`apply_real_lens` call
          (ASM through glass).  **Byte-identical to prior releases.**  This
          amplitude is a single-plane phase-screen leg, so it carries no
          asymmetric ray-density redistribution: on a DECENTERED / tilted
          (generally aberrated) element the induced-coma SPOT is
          amplitude-limited (it does not broaden -- P9 / N10a).  Good for
          wavefront / pointing / on-axis work.
        * ``'ray_density'`` (opt-in, niche N12) -- geometric ray-tube energy
          conservation: with ``J = d(x_out,y_out)/d(x_in,y_in)`` the ray-map
          Jacobian (from the analytic gradient of the entrance->exit fit),
          the exit magnitude is ``|E_in(x_in)| / sqrt(|det J|)`` placed at the
          exit ray position with the traced OPL phase.  This ``1/sqrt(|det J|)``
          IS the asymmetric coma redistribution the screen leg lacks, so the
          decentered / aberrated SPOT broadens (usable for PSF / spot-size /
          EE metrics, not just wavefront).  Energy-conserving in the geometric
          limit (no silent renormalisation).

          **What "in the geometric limit" costs at finite sampling**
          (v5.30, audit E-M6).  The Jacobian is evaluated on the COARSE Newton
          lattice and the resulting magnitude is bilinearly upsampled, so the
          exit power falls short of the aperture-transmitted input power by an
          amount that shrinks monotonically as ``ray_subsample`` approaches 1.
          Measured on a well-conditioned battery cell (2.5x aperture:beam),
          exit power / aperture-transmitted input power::

              ray_subsample = 1 : 0.99981 .. 1.00003
              ray_subsample = 2 : 0.99921 .. 0.99944
              ray_subsample = 4 : 0.99664 .. 0.99794
              ray_subsample = 8 : 0.98664 .. 0.99200   (shipped default)

          i.e. **the shipped default loses about 1% of the power** (up to ~2%
          on faster elements).  That is a sampling artefact, not a physical
          loss -- if absolute throughput matters, drop ``ray_subsample`` and
          watch the ratio converge rather than renormalising.  Separately,
          a *subsample-independent* deficit is physical: on a strongly
          diverging element at a tight aperture:beam the exit fan leaves the
          output window (the battery's negative corrector sits at 0.9535-0.9569
          at every subsample).

          The function runs this ratio as a cheap post-hoc **energy
          self-check** and emits a ``RuntimeWarning`` when it leaves a
          ``ray_subsample``-aware band (deficit tolerance
          ``0.080 + 0.010 * (ray_subsample - 1)``, gain tolerance ``0.050``).
          The band is set from the full 24-cell battery sweep above, so
          correct runs at the shipped defaults stay silent; what speaks up is
          the order-of-magnitude class -- a fold caustic inflating the capped
          ``1/sqrt(|det J|)`` (measured ratio 1.10 at sub=8 / 1.33 at sub=4 on
          a deliberately-broken biconcave), a ray map running off the grid, or
          an ``aperture_diameter`` wider than the traced pupil.

          **Caustic caveat.**  ``det J -> 0`` (or a sign change) at a fold, so
          the single-branch amplitude diverges there.  This mode DETECTS the
          fold (relative floor on ``|det J|`` + adjacent sign change), CAPS the
          amplitude (never returns inf/nan), and emits a one-time
          ``RuntimeWarning`` steering to :func:`apply_real_lens_gbd` /
          :func:`apply_real_lens_fga` -- it does NOT sum the multi-valued ray
          branches (no KMAH/Maslov phase); GBD/FGA remain the caustic reference.

          Requires ``inversion_method='newton'`` and the CPU path
          (``use_gpu=False``); incompatible with ``return_screen=True``.
    caustic : {None, 'single', 'multibranch', 'uniform'}, default None
        Opt-in MULTIBRANCH (KMAH / Maslov) refinement of the ``ray_density``
        amplitude (niche N13 / K1).  ``None`` / ``'single'`` (default) is the
        single-branch behaviour above -- BYTE-IDENTICAL to prior releases.

        ``'uniform'`` (niche N16 / K4; requires ``amplitude_model='ray_density'``
        + the CPU path) adds the Chester-Friedman-Ursell UNIFORM Airy DARK-side
        completion on top of the multibranch bright field so the traced field is
        diffraction-correct THROUGH a fold caustic.  The pure ``'multibranch'``
        geometric sum is identically ZERO on the DARK side of a fold (no real ray
        branch there) and so drops the exponentially-decaying Airy tail; the
        ``'uniform'`` mode meridional-ray-traces the fold to get the caustic
        radius ``r_c``, the fold parameter ``zeta(r) = kappa (r_c - r)`` and the
        mean phase, FITS the two smooth Airy coefficients to the bright field just
        inside ``r_c``, and continues the SAME ``uniform_fold_airy`` CFU kernel to
        ``zeta < 0`` to fill the dark tail -- closing the K1 fold-truth gap
        (windowed r2m -14.8% -> ~2%, energy 0.80 -> ~1.0 vs the direct
        Rayleigh-Sommerfeld ``caustic_fold_ref``).  It applies to a
        rotationally-symmetric SINGLE fold RING (collimated / rot-sym input,
        centred prescription); a decentered / astigmatic fold, a carrier tilt, a
        plane with no fold, or a CUSP / multiple rings (the Pearcey regime) are
        DETECTED and fall back to the plain multibranch field (finite, one-time
        warning).  Bright side ``r < r_c`` is byte-identical to ``'multibranch'``.

        ``'multibranch'`` (requires ``amplitude_model='ray_density'``) is the
        multi-valued generalisation: where the ray map FOLDS (``det J -> 0`` /
        sign change) it gathers ALL real ray branches reaching each output
        pixel, weights each ``|E_in(x_in^b)| / sqrt(|det J_b|)``, applies the
        Maslov phase ``exp(-i (pi/2) KMAH_b)`` (``KMAH_b`` = the number of
        ``det J`` sign changes -- astigmatic focal-line crossings -- along that
        branch's ray, counted ANALYTICALLY from the exact quadratic
        ``det Q(z)``), and SUMS COHERENTLY.  It reuses the existing
        :func:`apply_real_lens_traced_multibranch` branch-finder + det-Q KMAH
        counter (Ludwig uniform-Airy swap in the Kravtsov-Orlov caustic band),
        so the field is FINITE at the fold (never inf/nan / no ``sqrt``-blowup)
        and the sqrt-singularity resolves into the finite fold-diffraction
        profile.  Output is taken at ``output_plane_distance`` past the exit
        vertex, so a through-focus caustic plane is reached DIRECTLY (no
        separate ASM step).

        Scope / honest caveat.  The multibranch field is a GEOMETRIC (ART)
        construction: on the DARK side of a fold no real ray branches exist, so
        it carries no evanescent diffraction tail there.  On a fine, wave-
        resolved grid the single-branch ray-density exit field ASM-propagated
        to the fold plane (a genuine wave propagation) is therefore MORE
        accurate for the full caustic-ring r2m/EE than the pure multibranch
        sum; keep :func:`apply_real_lens_gbd` / :func:`apply_real_lens_fga` (or
        single-branch ``ray_density`` + ASM) as the quantitative caustic
        reference.  Multibranch is the tool when you need the coherent
        multi-arrival field / KMAH branch decomposition AT the caustic plane in
        one call (finite, no blow-up) rather than an aliasing-sensitive wave
        propagation.  See ``docs/plan_kmah_gpu_perf_2026_07_21.md`` (N13) and
        ``tests/unit/test_niche_k1_kmah_caustic.py`` for the measured envelope.
    output_plane_distance : float, default 0.0
        Observation-plane distance [m] past the last surface's exit vertex,
        honoured ONLY by ``caustic='multibranch'`` (the single-branch / screen
        paths always output at the exit vertex; a non-zero value with any other
        mode raises).  ``0.0`` = the exit vertex.
    caustic_ray_subsample : int, default 2
        ``caustic='multibranch'`` launch-grid spacing in units of ``dx`` (one
        ray per ``caustic_ray_subsample`` pixels); smaller = denser ray
        branches = finer caustic resolution.  Distinct from ``ray_subsample``
        (the Newton-inversion coarse grid), which is unused on the multibranch
        path.
    caustic_band : {'ludwig', 'plain'}, default 'ludwig'
        ``caustic='multibranch'`` fold-caustic band model: ``'ludwig'`` swaps a
        coalescing branch pair in the Kravtsov-Orlov band for the uniform
        Airy-fold field (finite at the fold); ``'plain'`` keeps the raw branch
        sum (diverges toward the fold).
    caustic_min_area_ratio : float, default 1e-6
        ``caustic='multibranch'`` degenerate-triangle skip threshold (mapped /
        launch area) -- the caustic set where ART is undefined.
    origin : (float, float), default (0, 0)
        Transverse position ``(x0, y0)`` (m) of the WAVE GRID's CENTRE PIXEL in
        the element's own (optical-axis) coordinate system -- niche D9.  Grid
        index ``(i, j)`` then denotes the physical point
        ``(x0 + (j - N/2) dx, y0 + (i - N/2) dy)``; the element, its aperture
        and the ray launch stay where they are.  The default is short-circuited
        everywhere, so the on-axis path is BYTE-IDENTICAL (pinned by
        ``tests/unit/test_niche_d9_grid_origin.py``).

        **Why.**  A tilted congruence's beam sits at its chief ray, not on the
        axis, so an axis-centred grid has to span BOTH -- a window of
        ``2*max(|x_c|,|y_c|) + window_factor*w`` where the beam itself needs
        only ``window_factor*w``.  On design 121's order (-4,-2) that is
        12.50 -> 18.54 mm, i.e. 1.48x the linear size and 2.2x the memory, on a
        leg already at 17 GB.  Centring the grid on the chief ray collapses the
        first term entirely.  :func:`~lumenairy.propagators.carrier._fine_trace_group_exit`
        is the caller that does this.

        **What moves and what does not.**  Everything the traced leg owns is
        carried in the ABSOLUTE frame and therefore needs the origin exactly
        once, at the grid <-> coordinate boundary: the ``X`` / ``Y`` wave-grid
        meshes, the three entrance-coordinate -> ``E_in``-pixel conversions
        (the ray-density ``|E_in|`` sample and both ``remap_sampling='full'``
        residual samples), the exit-NA guard's launch-height -> pixel map, the
        C6 de-chirp grid, the carrier grid, and the beam-radius / residual-
        eikonal measurements.  What must NOT get it: the ray LAUNCH grid
        ``xs_in`` (already absolute -- and keeping it axis-centred is what
        preserves the odd-``n_launch`` on-axis piston reference
        ``opl_grid -= opl_grid[i_axis, i_axis]``), the Newton ``bound`` and
        out-of-domain disc, the entrance-stop test on ``(x_e, y_e)``, the
        exit-support hull built from traced landings, and a
        :class:`TiltedCarrier`'s ``(x0, y0)`` -- the carrier states the
        CONGRUENCE's position and subtracts it from the (already absolute)
        ``X`` / ``Y`` itself, so removing the origin from it too would
        double-count (a defect that is exactly right at the grid centre and
        wrong in the wings).

        **Restricted to the validated carrier regime.**  ``origin != (0, 0)``
        raises :class:`NotImplementedError` unless
        ``amplitude_model='ray_density'`` AND ``preserve_input_phase='remap'``.
        That is not caution for its own sake: only on that path is the analytic
        ``apply_real_lens`` amplitude leg -- which has NO origin of its own, and
        so places the element's sag and masks about the wrong transverse point
        -- reduced to its ZERO SET (the ray-density swap divides its modulus
        out), and only there can that residual coupling be measured and refused
        (see :data:`ORIGIN_AMP_SUPPORT_CHECK`).  On the ``'screen'`` amplitude
        leg, or with ``preserve_input_phase=True``, the analytic envelope IS the
        answer's magnitude and a decentred grid would silently return an
        on-axis element's diffraction pattern.
    _exit_na_out : dict, optional
        PRIVATE diagnostic sink (niche C1 item 4).  When given, it is filled
        with the exit-NA measurement this function already makes for its own
        Nyquist warning: ``na_exit`` (the largest exit direction-cosine
        magnitude over rays carrying >= e^-4 of the peak input AMPLITUDE),
        ``dx`` / ``na_nyquist`` (this grid's pitch and the NA it can carry),
        ``power_frac_above_nyquist`` (the |E_in|^2-weighted fraction of the
        traced exit power at an NA this grid cannot carry) and ``n_rays``.
        Nothing in this function reads it back, so no default behaviour
        depends on it.  Consumed by
        :func:`lumenairy.propagators.carrier._fine_trace_group_exit`, whose
        ``on_tilt_exact_grid`` refusal must be sourced from the MEASURED exit
        NA rather than from the chain's paraxial ``w_in/|R_out|``.

    _remap_launch_out : dict, optional
        PRIVATE diagnostic sink (niche C6).  When given, it is filled with the
        state of the stationary-phase ray launch: ``engaged`` (whether the
        residual-eikonal model was built and used), ``flag``
        (:data:`REMAP_STATIONARY_PHASE_LAUNCH`), ``remap`` (whether
        ``preserve_input_phase='remap'`` was in force) and, when engaged, the
        fit's own record -- ``degree``, ``n_terms``, ``n_samples``,
        ``grad_a_rms`` (amplitude-weighted rms of the MEASURED input-residual
        ray slope, rad), ``grad_a_residual_rms`` (the part the fit does NOT
        model, which is what the leftover second-order error scales with),
        ``grad_a_fit_max_launch`` (the largest slope the model actually adds
        over the launch square -- the extrapolation diagnostic), ``w_beam``,
        ``centre``, ``r_fit`` (the SAMPLE disc, which is also the fit's own
        normalisation), ``r_freeze`` (the radial-freeze circle, which clears
        the ray-fit disc -- see ``_REMAP_RESID_FREEZE_MARGIN``),
        ``ray_fit_radius`` and ``stride``.  Nothing in this function reads it
        back.

    Returns
    -------
    E_out : ndarray, complex, shape (N, N)
        Field at the exit-vertex plane of the last surface.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced', input_kind='field')
    # ``newton_fit='auto'`` resolves to POLYNOMIAL.  Spline was tried as the CPU
    # default (v5.30.2) on the grounds that the two fits are indistinguishable in
    # accuracy -- differences sit in the 4th-5th significant figure and swap
    # direction with ray_subsample -- while spline parallelises better
    # (1.29-1.31x at 8 workers vs 1.10-1.13x).  REVERTED: the spline path gates
    # OFF the ray-fit-radius restriction (see the ``newton_fit != 'spline'``
    # guards below), i.e. ``fit_radius_beam_factor`` silently stops applying.
    # That restriction is not optional for real designs: design 121's post-DOE
    # groups carry 20-32 mm apertures against a sub-mm beam (~75x), far past the
    # 1.5x aperture:beam cliff at which the traced OPL fit is corrupted by
    # marginal rays the beam never occupies -- measured to return a NON-FINITE
    # exit field without it.  Selecting a fit backend must not silently disable
    # an accuracy guard, so the modest parallel speed-up does not justify it.
    # 15 tests across niche c6 / c11 / c12 (the stationary-phase fit guard, the
    # decentred-fit arbiter and physics fit selection) also exercise machinery
    # that only runs on the polynomial path.
    # Spline remains fully supported as an EXPLICIT choice.
    # The delegate branch reports the CALLER's request, not the resolved
    # value, so an all-default call never reports newton_fit as discarded.
    _newton_fit_requested = newton_fit
    if newton_fit == 'auto':
        newton_fit = 'polynomial'

    # ---- Enum membership guards (v5.29.1; audit E-M3 / E-M4) ------------
    # Both knobs used to be read by EQUALITY against one value, so every
    # other string (a typo, a value borrowed from a sibling knob, a
    # non-string) silently selected the fall-through branch: any junk
    # ``on_noncollimated`` behaved as 'warn' and any junk
    # ``inversion_method`` as 'newton'.  House rule: unknown enum values
    # raise.
    #
    # ``on_noncollimated``: 'silent' / 'ignore' are accepted ALIASES for
    # 'off'.  The sibling knobs in this same signature spell suppression
    # 'silent' (``on_undersample``, ``on_aperture_beam``), so every call
    # site in this repo that wanted the check off wrote
    # ``on_noncollimated='silent'`` -- and got the WARN branch, i.e. the
    # opposite of what it asked for.  Honouring the alias is the fix; the
    # canonical value stays 'off'.
    _NONCOL_ALIASES = {'silent': 'off', 'ignore': 'off'}
    if isinstance(on_noncollimated, str):
        on_noncollimated = _NONCOL_ALIASES.get(on_noncollimated,
                                               on_noncollimated)
    if on_noncollimated not in ('warn', 'delegate', 'off'):
        raise ValueError(
            f"apply_real_lens_traced: on_noncollimated="
            f"{on_noncollimated!r} is not a valid policy.  Choose from "
            f"'warn' (default), 'delegate', 'off' ('silent' / 'ignore' are "
            f"accepted aliases for 'off').  Pre-v5.29.1 any unrecognised "
            f"value silently selected 'warn'.")
    # ``inverse_map`` is a per-call override of the module gate
    # ``_lens_imap.TRACED_INVERSE_MAP``: None follows the flag, True/False
    # force it for this call.  Validated HERE, unconditionally, rather than at
    # the build site -- a junk value must not behave as "follow the flag" on
    # the calls where the map never engages and only surface on the one where
    # it does.
    if inverse_map is not None and not isinstance(inverse_map, (bool, np.bool_)):
        raise ValueError(
            f"apply_real_lens_traced: inverse_map={inverse_map!r} is not a "
            f"valid setting.  Pass None (default: follow "
            f"lumenairy.elements._lens_imap.TRACED_INVERSE_MAP), True to "
            f"force the inverse-characteristic per-pixel evaluator on for "
            f"this call, or False to force the shipped coarse-Newton + "
            f"upsample path.")
    if inversion_method not in ('newton', 'fit', 'backward_trace'):
        raise ValueError(
            f"apply_real_lens_traced: inversion_method="
            f"{inversion_method!r} is not a valid method.  Choose from "
            f"'newton' (default: forward trace + per-pixel Newton "
            f"inversion), 'fit' (scattered Chebyshev inverse-map fit) or "
            f"'backward_trace' (experimental direct backward trace).  "
            f"Pre-v5.29.1 any unrecognised value silently ran Newton.")

    # ``on_fit_domain_basis`` (finding V4, 2026-08-06).  Added by the D5 fix,
    # and the only string mode knob in this file that shipped with NO
    # membership gate: 'Error', 'ignore', None, 1 and '' all fell through to
    # the warn branch at ``_fit_domain_inert``, so a caller who asked for the
    # combination to be FATAL got a RuntimeWarning and a returned field --
    # and, per the D5 measurement, that field can be ALL ZERO past the
    # aperture:beam cliff.  Exactly the defect class the two guards above
    # close, shipped in the commit that closed them.
    #
    # THE VALID SET, decided here rather than left implicit:
    #   'warn'   (default) name which knob is inert and continue
    #   'error'  raise where the announcement would have fired
    #   'silent' acknowledge -- what a caller deliberately using spline as a
    #            FIT-DOMAIN-FREE reference passes
    # 'ignore' and 'off' are accepted ALIASES for 'silent'.  Not indulgence:
    # 'ignore' is the vocabulary EVERY ``on_*`` knob in
    # ``lumenairy/propagators/carrier.py`` uses for suppression
    # (``_check_guard_action``: 'error' / 'warn' / 'ignore'), and this
    # signature's own siblings spell it 'silent' (``on_undersample``,
    # ``on_aperture_beam``), so a caller has two house styles to guess
    # between for one knob.  Before this gate 'ignore' was accepted AND
    # INERT -- the worst of the three possible outcomes.  ``on_noncollimated``
    # above resolves the same collision the same way (``_NONCOL_ALIASES``).
    _FDB_ALIASES = {'ignore': 'silent', 'off': 'silent'}
    if isinstance(on_fit_domain_basis, str):
        on_fit_domain_basis = _FDB_ALIASES.get(on_fit_domain_basis,
                                               on_fit_domain_basis)
    if on_fit_domain_basis not in ('warn', 'error', 'silent'):
        raise ValueError(
            f"apply_real_lens_traced: on_fit_domain_basis="
            f"{on_fit_domain_basis!r} is not a valid policy.  Choose from "
            f"'warn' (default), 'error', 'silent' ('ignore' / 'off' are "
            f"accepted aliases for 'silent').  Pre-v5.32.2 any unrecognised "
            f"value silently selected 'warn', so a caller asking for 'error' "
            f"got a warning and a returned field instead of a refusal.")

    # ``on_pool_memory`` (v5.32.3, FIX_CI_POOL).  Gated HERE, unconditionally,
    # like its siblings above and unlike ``on_undersample`` -- whose validation
    # sits inside the branch that only runs when the undersampling condition
    # trips, so junk behaves as 'warn' on every well-sampled call (the D5
    # ``_KNOWN_UNGATED`` ledger).  The Newton pool's memory cap has exactly
    # that shape: it binds on a 12 GB CI runner and never on a 256 GB
    # workstation, so a gate inside the warning branch would validate the knob
    # on one box and not the other.
    on_pool_memory = _pool_memory_policy(on_pool_memory)

    # ---- N12 (P11): opt-in ray-density (Jacobian) amplitude model -------
    # ``amplitude_model='screen'`` (default) is byte-identical to prior
    # releases -- none of the ray-density code below runs.  ``'ray_density'``
    # replaces the exit magnitude with the geometric ray-tube amplitude
    # ``|E_in| / sqrt(|det J|)`` (see the docstring), keeping the traced OPL
    # phase.  It is confined to the CPU Newton path so the entrance->exit fits
    # + Newton inverse it reuses are available in-process; the fold-detection
    # flag is a 1-element list so the nested amplitude closure can set it.
    if amplitude_model not in ('screen', 'ray_density'):
        raise ValueError(
            f"amplitude_model must be 'screen' or 'ray_density', got "
            f"{amplitude_model!r}.")
    _ray_density = (amplitude_model == 'ray_density')
    _rd_fold_detected = [False]
    # ---- preserve_input_phase='remap' (audit
    # AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S6.7): geometric transport of
    # the carrier-de-chirped input residual phase via the Newton entrance
    # pullback.  Downstream, the assembly must take the ``False`` (pure
    # k0*opl) branch and the ray-density block multiplies the remapped
    # residual phasor in -- so normalise the flag here.
    if not isinstance(preserve_input_phase, bool):
        if preserve_input_phase != 'remap':
            raise ValueError(
                "preserve_input_phase must be True, False or 'remap', got "
                f"{preserve_input_phase!r}.")
        if not _ray_density:
            raise ValueError(
                "preserve_input_phase='remap' requires "
                "amplitude_model='ray_density' (it reuses the ray-density "
                "entrance pullback).")
    _pip_remap = (preserve_input_phase == 'remap')
    if _pip_remap:
        preserve_input_phase = False
    if remap_sampling not in ('lattice', 'full'):
        raise ValueError(
            "apply_real_lens_traced: remap_sampling must be 'lattice' or "
            f"'full', got {remap_sampling!r}.")
    # S12: sample the transported residual PHASOR at full wave-grid resolution
    # (upsample the SMOOTH entrance pullback, not the fast phasor).  Only
    # meaningful when there is a coarse ray lattice to begin with.
    _pip_full = _pip_remap and remap_sampling == 'full'
    _rd_resid_coarse = [None]
    _rd_entrance_coarse = [None]
    _rd_resid_map = None
    # ---- N13 (K1): opt-in MULTIBRANCH (KMAH/Maslov) caustic refinement ----
    # ``caustic=None``/'single' (default) is byte-identical to prior releases.
    # ``'multibranch'`` routes the whole call to the existing
    # ``apply_real_lens_traced_multibranch`` (branch-finder + det-Q KMAH
    # counter); it is the multi-valued generalisation of the ray-density
    # amplitude, so it requires ``amplitude_model='ray_density'`` and the CPU
    # path.  The routing itself happens after the shared square-grid / dy / mirror
    # guards below (so it inherits them), via ``_multibranch``.
    if caustic is not None and caustic not in ('single', 'multibranch',
                                               'uniform'):
        raise ValueError(
            "caustic must be None, 'single', 'multibranch', or 'uniform', got "
            f"{caustic!r}.")
    _multibranch = (caustic == 'multibranch')
    # ---- N16 (K4): opt-in UNIFORM (Airy) dark-side completion --------------
    # ``caustic='uniform'`` runs the multibranch (bright side) and adds the
    # Chester-Friedman-Ursell dark-side Airy tail so the traced field is
    # diffraction-correct THROUGH a fold caustic; it shares the multibranch's
    # ray_density / CPU / output_plane_distance requirements (routed via
    # ``_uniform``).
    _uniform = (caustic == 'uniform')
    _mb_family = _multibranch or _uniform
    if _mb_family:
        _mode_name = 'multibranch' if _multibranch else 'uniform'
        if not _ray_density:
            raise ValueError(
                f"caustic={_mode_name!r} requires amplitude_model='ray_density' "
                "(it is the multi-valued generalisation of the ray-density "
                f"amplitude); got amplitude_model={amplitude_model!r}.")
        if use_gpu or amp_use_gpu:
            raise ValueError(
                f"caustic={_mode_name!r} requires the CPU path "
                "(use_gpu=amp_use_gpu=False): it reuses the CPU ray-trace "
                "branch-finder + analytic det-Q KMAH counter.")
    if float(output_plane_distance) != 0.0 and not _mb_family:
        raise ValueError(
            "output_plane_distance is only honoured by caustic='multibranch' / "
            "'uniform' (the single-branch / screen paths output at the exit "
            f"vertex); got output_plane_distance={output_plane_distance!r} with "
            f"caustic={caustic!r}.")
    if _ray_density:
        if return_screen:
            raise ValueError(
                "amplitude_model='ray_density' is incompatible with "
                "return_screen=True: the ray-density amplitude depends on "
                "|E_in| (and the traced phase), so it cannot be baked into an "
                "input-independent prepared screen.")
        if use_gpu:
            raise ValueError(
                "amplitude_model='ray_density' requires the CPU path "
                "(use_gpu=False): it re-uses the in-process entrance->exit "
                "fits + Newton inverse to build |E_in|/sqrt(|det J|).")
        if inversion_method != 'newton':
            raise ValueError(
                "amplitude_model='ray_density' requires "
                "inversion_method='newton' (it evaluates det J from the "
                f"Newton entrance->exit fits); got {inversion_method!r}.")
        # Full-grid Newton phase (no amp mask) so the reconstructed phasor
        # covers the WHOLE ray-valid region -- the ray-density amplitude has
        # energy (the coma tail) where |E_analytic| is small, which the amp
        # mask would otherwise drop.  Whole-grid final assembly (below) so the
        # magnitude swap sees the fully-built exit field.
        #
        # v5.29.1 (audit E-M5): this override used to be SILENT while every
        # other requirement in this block raises.  Match the block (and the
        # ``_FORCED`` contract in apply_real_lens_traced_multi): a value equal
        # to the forced 0.0 is accepted, anything else raises with the reason.
        # The shipped default ``_NEWTON_AMP_MASK_REL_DEFAULT`` is read as "not
        # requested" (there is no separate not-passed sentinel), so a caller
        # who explicitly passes exactly the default still gets the silent
        # override -- pass 0.0 to state the intent.
        if float(newton_amp_mask_rel) not in (
                0.0, _NEWTON_AMP_MASK_REL_DEFAULT):
            raise ValueError(
                f"apply_real_lens_traced: newton_amp_mask_rel="
                f"{newton_amp_mask_rel!r} conflicts with "
                f"amplitude_model='ray_density', which requires the FULL "
                f"coarse Newton grid (newton_amp_mask_rel=0.0): the "
                f"ray-density amplitude carries energy -- the coma tail -- "
                f"exactly where |E_analytic| is small, so the amplitude mask "
                f"would drop the rays that make this model differ from "
                f"'screen'.  Pre-v5.29.1 the value was overridden SILENTLY.  "
                f"Pass newton_amp_mask_rel=0.0 (or drop the argument), or use "
                f"amplitude_model='screen' if you need the mask.")
        newton_amp_mask_rel = 0.0

    # ---- niche D9: the CHIEF-RAY-CENTRED wave grid ------------------------
    # ``origin`` moves the grid, not the element.  Resolved here -- above every
    # coordinate site and above the multibranch dispatch -- so a configuration
    # this cannot carry is refused before any work is done.  See the ``origin``
    # docstring entry for what does and does not take the shift.
    try:
        _org_x, _org_y = (float(_v) for _v in origin)
    except (TypeError, ValueError):
        raise ValueError(
            "apply_real_lens_traced: origin must be a 2-sequence (x0, y0) in "
            f"metres (got {origin!r}).")
    if not (np.isfinite(_org_x) and np.isfinite(_org_y)):
        raise ValueError(
            f"apply_real_lens_traced: origin must be finite (got {origin!r}).")
    _origin_set = bool(_org_x or _org_y)
    if _origin_set:
        # THE REFUSAL IS THE FEATURE.  A decentred grid is only representable
        # here because the ray-density magnitude swap makes the axis-centred
        # analytic amplitude leg contribute nothing but its zero set; on every
        # other amplitude/phase path that leg IS the answer's magnitude, and it
        # would return an ON-AXIS element's diffraction pattern for an OFF-AXIS
        # beam -- correct-looking, wrong, and with no symptom to catch it by.
        if not (_ray_density and _pip_remap):
            raise NotImplementedError(
                f"apply_real_lens_traced: origin={origin!r} (a grid whose "
                f"centre pixel is off the optical axis) is implemented ONLY "
                f"for amplitude_model='ray_density' with "
                f"preserve_input_phase='remap' -- the carrier-regime "
                f"configuration the chain's exact final leg uses.  This call "
                f"has amplitude_model={amplitude_model!r} and "
                f"preserve_input_phase="
                f"{'remap' if _pip_remap else preserve_input_phase!r}.  The "
                f"reason is the ANALYTIC amplitude leg: apply_real_lens has no "
                f"origin of its own, so it builds the element's sag, its "
                f"aperture_diameter / clear_aperture masks and its stop about "
                f"the GRID centre, which under a decentred origin is not the "
                f"optical axis.  Only under 'ray_density' + 'remap' does that "
                f"leg reduce to its ZERO SET (the exit magnitude is replaced by "
                f"the ray-tube amplitude |E_in|/sqrt|det J|), which is a "
                f"coupling this function can and does MEASURE and refuse -- see "
                f"ORIGIN_AMP_SUPPORT_CHECK.  Under 'screen', or with "
                f"preserve_input_phase=True, the analytic envelope is the "
                f"returned magnitude and there is nothing to check: the call "
                f"would silently return an on-axis element's diffraction "
                f"pattern.  Keep origin=(0, 0) and size the grid to hold both "
                f"the axis and the beam, or switch to the ray-density remap "
                f"configuration.")
        if _mb_family:
            raise NotImplementedError(
                f"apply_real_lens_traced: origin={origin!r} is not supported "
                f"with caustic={caustic!r}: that route hands the whole call to "
                f"the multibranch / uniform branch-finder, which builds its own "
                f"axis-centred grids and knows nothing about the origin.  Pass "
                f"caustic=None for the decentred single-branch path.")
        if on_noncollimated == 'delegate':
            raise NotImplementedError(
                f"apply_real_lens_traced: origin={origin!r} is not supported "
                f"with on_noncollimated='delegate': the fallback returns "
                f"apply_real_lens(E_in) directly, and that model has no origin "
                f"-- it would place the element about the grid centre and "
                f"return an on-axis result without a word.  Use "
                f"on_noncollimated='warn' (the default) or 'off'.")
        if ORIGIN_AMP_SUPPORT_CHECK not in ('error', 'warn', 'silent'):
            raise ValueError(
                "apply_real_lens_traced: ORIGIN_AMP_SUPPORT_CHECK must be "
                f"'error', 'warn' or 'silent' (got "
                f"{ORIGIN_AMP_SUPPORT_CHECK!r}).")

    # v5.1.0 (default-knob resolver rollout): resolve ``wave_propagator``
    # / ``dy`` from the library-wide defaults when callers leave them
    # at the ``None`` sentinel.  Explicit values bypass the resolver.
    if wave_propagator is None:
        from ..propagators.propagation import get_default_wave_propagator
        wave_propagator = get_default_wave_propagator()
    if dy is None:
        from ..propagators.propagation import get_default_dy
        dy = get_default_dy()
        if dy is None:
            dy = dx

    # 4.12.0 (B2-5): explicit mirror-in-surfaces guard.  The shared
    # ``_check_no_silent_fold_drop`` only looks at the prescription's
    # ``elements`` list (the full element sequence, populated by
    # ``load_zemax_zmx``); a hand-built prescription that puts a
    # mirror directly into ``surfaces`` (via ``is_mirror=True`` or
    # ``glass_after='MIRROR'``) would slip past the shared check, and
    # the ray-traced OPL leg would silently treat the mirror as a
    # refractor with the wrong sign.  Fail loudly with a
    # mirror-specific message before the trace begins.
    _surfaces_list = prescription.get('surfaces') or []
    _mirror_surf_idx = []
    for _i, _s in enumerate(_surfaces_list):
        if not isinstance(_s, dict):
            continue
        _gl_after = _s.get('glass_after')
        _is_mirror = bool(_s.get('is_mirror', False)) or (
            isinstance(_gl_after, str)
            and _gl_after.upper() == 'MIRROR'
        )
        if _is_mirror:
            _mirror_surf_idx.append(_i)
    if _mirror_surf_idx:
        raise ValueError(
            f"apply_real_lens_traced: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens_traced "
            f"only walks refracting surfaces.  Running this "
            f"prescription as-is would silently treat the mirror as "
            f"a refractor (wrong sign / wrong focusing phase) and "
            f"propagate along the unfolded-equivalent axis.  Use "
            f"the per-segment trace + apply_mirror pattern for "
            f"folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens_traced (each segment) with "
            f"apply_mirror (each fold).  See Guide-Folded-Designs "
            f"section 'Wave-optics through a fold'.")

    # Folded-design silent-drop guard: same as apply_real_lens.
    from ._lens_real import _check_no_silent_fold_drop
    _check_no_silent_fold_drop(
        prescription, fn_name='apply_real_lens_traced')

    # Internal references keep the legacy local name to avoid a
    # sprawling rename across this 1500-line function body.
    lens_prescription = prescription

    # Local import to avoid a circular dep at module load time
    from ..raytrace import (
        _make_bundle,
        surfaces_from_prescription,
        trace,
    )

    call_progress(progress, 'real_lens_traced', 0.0, 'initialising')

    # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
    # Entry log -- grid size + surface count + Newton iter cap so users
    # who attach a handler can see the call shape at a glance.  The
    # actual Newton-cap value is resolved further down (caller override
    # > module default); we reproduce that resolution here so the entry
    # log reports what the run will actually use.
    _ny_entry, _nx_entry = np.shape(E_in)
    _n_surfaces_entry = len(prescription.get('surfaces') or [])
    _newton_cap_entry = (int(newton_max_iters)
                         if newton_max_iters is not None
                         else _NEWTON_MAX_ITERS)
    logger.info(
        "apply_real_lens_traced: entry N=%dx%d n_surfaces=%d "
        "newton_max_iters=%d ray_subsample=%d",
        int(_ny_entry), int(_nx_entry), int(_n_surfaces_entry),
        int(_newton_cap_entry), int(ray_subsample))

    # Pre-flight grid vs prescription-aperture check.
    try:
        _warn_if_aperture_exceeds_grid(
            lens_prescription, int(np.shape(E_in)[0]), dx,
            source='apply_real_lens_traced')
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture-check failure is informational only.
        pass

    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError("apply_real_lens_traced requires a square grid")
    N = Nx

    if dy is None:
        dy = dx
    # The traced variant's ray-subsample + interpolation paths assume
    # a square, isotropic grid.  Anamorphic (dy != dx) propagation is
    # supported by :func:`apply_real_lens` and
    # :func:`apply_real_lens_maslov`; for the traced model, pass
    # equal dx + dy or fall back to the analytic model.
    if abs(float(dy) - float(dx)) > 1e-15 * max(abs(float(dx)), 1.0):
        raise ValueError(
            "apply_real_lens_traced currently requires square pixels "
            f"(dx == dy); got dx={dx!r}, dy={dy!r}.  Use apply_real_lens "
            "for anamorphic grids.")

    # ---- N13 (K1): dispatch the MULTIBRANCH (KMAH/Maslov) caustic sum ------
    # Route the whole call to the existing multibranch branch-finder (REUSE,
    # do not reimplement).  It gathers all real ray branches per output pixel,
    # weights each ``|E_in| / sqrt(|det J|)``, applies ``exp(-i (pi/2) KMAH)``,
    # and sums coherently -- finite at the fold (Ludwig uniform-Airy), output
    # at ``output_plane_distance`` past the exit vertex.  Bypasses the Newton
    # OPL machinery entirely (a ray-native construction; no phase unwrap, so
    # the ``on_undersample`` OPD-sampling check does not apply).
    if _mb_family:
        _mode_name = 'multibranch' if _multibranch else 'uniform'
        # ``carrier`` -> ``input_carrier``: the multibranch launch is one
        # tilted congruence taking a transverse carrier wavevector (rad/m) or
        # 'auto'; the traced None/'auto' vocabulary maps directly.  A scalar
        # conjugate / explicit-wavefront carrier is not representable as a
        # single launch tilt here.
        if carrier is None:
            _input_carrier = None
        elif isinstance(carrier, str) and carrier == 'auto':
            _input_carrier = 'auto'
        else:
            raise ValueError(
                f"caustic={_mode_name!r} supports carrier=None or "
                "carrier='auto' only (the launch is one tilted congruence); "
                f"got carrier={carrier!r}.  Use the single-branch ray_density "
                "path for a scalar-conjugate / explicit-wavefront carrier.")
        if _uniform:
            # N16 (K4): multibranch bright side + CFU uniform Airy dark tail
            # (rotationally-symmetric fold ring; falls back to plain
            # multibranch for cusp / non-symmetric / non-fold cases).
            from ._lens_traced_uniform import apply_real_lens_traced_uniform
            _mb = np.asarray(apply_real_lens_traced_uniform(
                E_in,
                prescription=prescription,
                wavelength=wavelength,
                dx=dx,
                output_plane_distance=float(output_plane_distance),
                ray_subsample=int(caustic_ray_subsample),
                min_area_ratio=float(caustic_min_area_ratio),
                caustic_band=caustic_band,
                input_carrier=_input_carrier,
            ))
        else:
            from ._lens_traced_multibranch import (
                apply_real_lens_traced_multibranch,
            )
            _mb = np.asarray(apply_real_lens_traced_multibranch(
                E_in,
                prescription=prescription,
                wavelength=wavelength,
                dx=dx,
                output_plane_distance=float(output_plane_distance),
                ray_subsample=int(caustic_ray_subsample),
                min_area_ratio=float(caustic_min_area_ratio),
                caustic_band=caustic_band,
                input_carrier=_input_carrier,
            ))
        _target_cdtype = (E_in.dtype if np.iscomplexobj(E_in)
                          else np.complex128)
        if _mb.dtype != _target_cdtype:
            _mb = _mb.astype(_target_cdtype)
        call_progress(progress, 'real_lens_traced', 1.0, 'done')
        return _mb

    aperture = lens_prescription.get('aperture_diameter')
    thicknesses = lens_prescription['thicknesses']
    float(sum(thicknesses))

    # 4.11.2: warn if the prescription specifies a stop_index other than
    # the entrance (or carries a decentered stop).  ``apply_real_lens``
    # honours ``stop_index`` (the aperture is applied at the indicated
    # surface, possibly off-axis); but ``apply_real_lens_traced``'s
    # ray-tracing path launches rays from the entrance plane and the
    # final exit-aperture mask uses the entrance-aperture diameter.
    # Porting the per-surface stop-application logic into the ray-trace
    # leg is feature-scope; warn so the silent move-to-entrance is
    # visible to callers who have written a stop_index into their
    # prescription.
    _stop_index = lens_prescription.get('stop_index')
    if _stop_index is not None and int(_stop_index) != 0:
        import warnings
        warnings.warn(
            f"apply_real_lens_traced: prescription specifies "
            f"stop_index={_stop_index}, but the ray-traced phase leg "
            "launches rays from the entrance pupil only; the aperture "
            "stop is effectively applied at the entrance (index 0).  "
            "For physically-correct stop behaviour on a non-entrance "
            "stop, use apply_real_lens.",
            RuntimeWarning, stacklevel=2,
        )
    else:
        # Decentered entrance stop: warn similarly -- the inner amp
        # path applies the stop centred at the surface's ``decenter``,
        # but the ray-trace leg's launch geometry is centred on the
        # optical axis.
        _surfs = lens_prescription.get('surfaces') or []
        if _surfs:
            _stop_surf_idx = int(_stop_index) if _stop_index is not None else 0
            if 0 <= _stop_surf_idx < len(_surfs):
                _dec = _surfs[_stop_surf_idx].get('decenter') or (0.0, 0.0)
                if _dec[0] != 0.0 or _dec[1] != 0.0:
                    import warnings
                    warnings.warn(
                        f"apply_real_lens_traced: stop surface "
                        f"{_stop_surf_idx} has decenter={_dec}; the "
                        "ray-traced phase leg uses an on-axis launch "
                        "geometry and will not see the off-axis stop "
                        "correctly.  Use apply_real_lens for "
                        "decentered-stop systems.",
                        RuntimeWarning, stacklevel=2,
                    )

    x = (np.arange(N) - N / 2) * dx
    # ---- niche D9: the wave grid's two PHYSICAL axes ----------------------
    # ``x`` is the x (column) axis and ``_y_ax`` the y (row) axis, both in the
    # element's absolute frame.  On axis they are THE SAME OBJECT, so every
    # consumer below -- ``np.meshgrid(x, _y_ax)``, the row-band aperture term,
    # the halo edge test, the support-band masks -- is byte-identical to the
    # single-vector form it replaces.  Off axis they genuinely separate: with
    # ``x0 != y0`` the shared-vector structure breaks, not merely the offset,
    # which is why the split is made once here rather than at each site.
    _y_ax = x
    if _origin_set:
        _y_ax = (np.arange(N) - N / 2) * dy + _org_y
        x = x + _org_x
    # Opt-in row-band (chunked) FINAL ASSEMBLY: when ``sag_chunk_rows`` is set
    # (and the standard sub>1 Newton path is active), the OPL upsample +
    # delta-phase + exit-field assembly run in row bands, so the full-grid
    # float64 stack (ii/jj indices, the (2,N,N) map_coordinates input,
    # opl_map, nan_full, delta_phase, the complex128-first phase_exp) never
    # materialises -- only (chunk_rows x N) bands.  Values are byte-identical
    # to the whole-grid path (map_coordinates order-1 is pointwise in the
    # output; the phase/mask algebra is elementwise) -- pinned by
    # test_chunked_assembly_byte_identical.  v5.35: that identity is against
    # the whole-grid path AT THE SAME INVERSION.  ``_imap_domain_gate`` below
    # excludes this path from the inverse-characteristic evaluator, so on a
    # call the evaluator would engage, banding also selects the incumbent
    # coarse-Newton inversion -- see the ``sag_chunk_rows`` parameter doc and
    # tests/unit/test_niche_s10_sibling_patterns.py.
    # The full X/Y meshgrids are not
    # built on this path: the Newton coarse grid comes from the 1-D x
    # subsample and the exit-aperture mask is banded.
    # v5.17.0: sag_chunk_rows=None resolves to AUTO (banded when N >= 4096);
    # pass 0 to force the whole-grid path.  The caller's RAW kwarg also flows
    # to the apply_real_lens amp legs so both stages resolve -- and band --
    # consistently.
    # v5.17.1 (audit P2-05): forward the RAW kwarg, not the resolved value.
    # The resolver maps 0 -> None, and apply_real_lens re-resolves None ->
    # AUTO, so forwarding the resolved value silently re-enabled row-banding
    # in the amp legs when the caller passed the documented force-whole-grid
    # sentinel 0.  Both stages resolve the raw value against the same N, so
    # None / positive ints band identically in both stages and 0 now forces
    # whole-grid in BOTH.
    from ._lens_real import _resolve_sag_chunk_rows
    _sag_chunk_rows_raw = sag_chunk_rows
    sag_chunk_rows = _resolve_sag_chunk_rows(sag_chunk_rows, N)
    _chunk_assembly = (
        sag_chunk_rows is not None and int(sag_chunk_rows) > 0
        and max(1, int(ray_subsample)) > 1
        and inversion_method == 'newton'
        and not _ray_density   # ray-density does the magnitude swap on the
                               # whole-grid exit field (below), not per band
    )
    if _chunk_assembly:
        X = Y = None
    else:
        # BROADCAST, not meshgrid (AUDIT_TRACED_MEMORY_2026_08_09 sec 5.2 /
        # row 5; FIX_PERF_POLY_LOCALS_2026_08_09).  ``np.meshgrid`` MATERIALISES
        # two full 2-D float64 arrays -- 2 x 2.147 GB at the n_fine = 16384
        # retrace leg, held for the whole call (the memory census caught both
        # of them live at the peak plateau).  ``np.broadcast_to`` returns
        # zero-copy read-only views with the SAME element values, so every
        # consumer here -- ``X[::sub, ::sub]``, ``X[mask_full]``,
        # ``X ** 2 + Y ** 2``, ``.ravel()`` inside the Newton / fit inverters,
        # ``_compute_carrier`` -- reads exactly the same numbers and the
        # results are bitwise identical (pinned by
        # test_niche_perf_poly_locals.py).  Nothing writes through X or Y: the
        # sub > 1 path already handed those consumers the STRIDED VIEW
        # ``X[::sub, ::sub]``, so a write would have corrupted X long before
        # this change.
        X = np.broadcast_to(x[None, :], (N, N))
        Y = np.broadcast_to(_y_ax[:, None], (N, N))
    # OPL coarse->fine spline order; resolved for real once the carrier engage
    # test has run (see the R7 note at its assignment).  Initialised here so
    # the row-band assembly can never read it unbound.
    _opl_up_order = 1

    # ----- Carrier-referenced correction (audit S5.1) -------------------
    # Traced's default correction is referenced to a PLANE WAVE (unit input
    # for phase_analytic_lens; rays launched parallel to z), valid only when
    # the input congruence is ~collimated.  For a divergent / tilted /
    # emitter-array input, supply the beam's own smooth CARRIER wavefront:
    # the reference then matches the beam (well-conditioned exit reference,
    # fixing N5) and the rays launch along the carrier normals, so the
    # traced correction is applied to the small residual only.  W is in
    # length units (reference phase = k0 * W); grad(W) gives the ray
    # direction cosines.
    _k0 = 2.0 * np.pi / wavelength
    _carrier_W = None
    _carrier_grad = None
    _carrier_W_fn = None
    # N5 (2026-07-19): tilt_aware_rays with NO explicit carrier still needs an
    # entrance-eikonal REFERENCE so the exit wavefront carries the input
    # congruence -- the same physics the carrier path's H6 fix restored.  On the
    # DEFAULT preserve_input_phase=True the plane-wave reference already works (a
    # diverging/tilted tilt_aware input focuses at its true image: E_analytic
    # carries the input eikonal and the plane-wave reference leg does not
    # subtract it, unlike the carrier path's exp(i*k0*W) leg that made the H6
    # collapse surface on the default path).  But preserve_input_phase=False
    # builds the exit phase from opl_traced ALONE, which the ray tracer
    # accumulates only from the entrance plane forward -- dropping k0*W(x_in) and
    # collapsing a diverging/tilted input to the collimated focal plane (the H6
    # class, here confined to the non-default mode).  Fix: auto-fit the input's
    # smooth carrier and thread it through the SAME carrier plumbing (reference
    # leg exp(i*k0*W) + the H6 entrance-eikonal OPL term); the per-pixel tilt
    # LAUNCH below is retained (the tilt_aware branch wins).  A (near-)collimated
    # input fits W == 0 exactly (real / globally-phased field -> zero tilt
    # samples), so it keeps the byte-identical plane-wave path.
    _carrier_src = carrier
    if _carrier_src is None and tilt_aware_rays:
        _carrier_src = 'auto'
    if _carrier_src is not None:
        if X is None:
            # (chunked-assembly path only, which niche D9's origin can never
            # reach -- ray-density forces ``_chunk_assembly`` off -- but the
            # axes are split here too so the two constructions cannot drift.)
            _cx = (np.arange(E_in.shape[0]) - E_in.shape[0] / 2) * dx
            _cy = _cx
            if _origin_set:
                _cy = (np.arange(E_in.shape[0]) - E_in.shape[0] / 2) * dy \
                    + _org_y
                _cx = _cx + _org_x
            _CX, _CY = np.meshgrid(_cx, _cy)
        else:
            _CX, _CY = X, Y
        _cW, _cGrad, _cWfn = _compute_carrier(
            _carrier_src, E_in, wavelength, dx, _CX, _CY,
            origin=(_org_x, _org_y))
        del _CX, _CY
        # Engage the carrier machinery only when the carrier eikonal is
        # NON-TRIVIAL over the bright support.  A (near-)collimated input --
        # whether it reached here as an explicit long-conjugate ``carrier``, a
        # ``carrier='auto'`` fit that returned W ~ 0, or the implicit
        # ``tilt_aware_rays`` auto-carrier -- fits an eikonal below this floor,
        # so it keeps the byte-identical plane-wave-reference path (pin:
        # ``carrier='auto'`` == ``carrier=None`` on a collimated input; R7 also
        # relies on this so its reference-taper / fit-restriction / cubic-upsample
        # never perturb a W~0 carrier).  A carrier that actually shifts the focus
        # is tens+ of radians over the beam, far above the 1e-2-rad floor.
        _mag0 = np.abs(E_in)
        _pk0 = float(_mag0.max()) if _mag0.size else 0.0
        if _pk0 > 0:
            _bright0 = _mag0 > 0.05 * _pk0
            # FREE AT LAST USE (AUDIT_TRACED_MEMORY_2026_08_09 sec 2.3/2.4:
            # ``_mag0`` was caught LIVE in this frame at the peak plateau).
            # It is a full-grid float array -- 2.147 GB at n_fine = 16384 --
            # whose only two readers are the peak above and this bright mask,
            # yet it stayed bound for the whole call, including the ~794 s the
            # element spends below.  Pure lifetime; no value changes.
            del _mag0
            # NaN-safe peak WITHOUT np.nanmax's "All-NaN slice encountered"
            # RuntimeWarning: an all-NaN eikonal (a user-supplied ndarray
            # carrier with NaN holes -- the collimated scalar case is now
            # handled analytically in _compute_carrier) must read 0.0 (=>
            # do not engage) rather than NaN-compare its way there while
            # emitting a warning nobody can act on.  Identical to
            # np.nanmax on any slice that has at least one non-NaN sample.
            if _bright0.any():
                _cWb = np.abs(_cW[_bright0])
                _fin0 = ~np.isnan(_cWb)
                _peakW = float(_cWb[_fin0].max()) if _fin0.any() else 0.0
                del _cWb, _fin0
            else:
                _peakW = 0.0
            del _bright0          # full-grid bool, 0.27 GB at n_fine=16384
        else:
            _peakW = 0.0
            del _mag0
        _engage = (_peakW * _k0) > _TILT_EIKONAL_MIN_RAD
        if _engage:
            _carrier_W, _carrier_grad, _carrier_W_fn = _cW, _cGrad, _cWfn
            # The fast_analytic_phase reference is the lens's on-axis geometric
            # phase (input-independent), which cannot carry the carrier
            # congruence; force the full wave reference when a carrier is set.
            if fast_analytic_phase:
                fast_analytic_phase = False

    # R7 / audit F2 (2026-07-21): the R7 intra-group fidelity fixes (reference
    # taper to |E_in|, carrier-gated fit-domain restriction, cubic OPL upsample)
    # apply to the CARRIER-REFERENCED path -- rays launched along the carrier
    # normals grad(W).
    #
    # F3 (audit 2026-07-21): ``tilt_aware_rays=True`` DEGRADES a steep explicit
    # carrier.  The per-pixel tilt launch (below) is redundant with -- and far
    # less accurate than -- the carrier-gradient launch on a steeply-curved
    # spherical input: measured on the 121 S5-S7 triplet the tilt_aware path
    # reads 1.72 rad rms / ~185% exit-curvature error vs 0.008 rad / <1% for the
    # ``carrier=R`` default, because the smooth carrier already carries the input
    # congruence exactly while the per-pixel tilt reading is noisy AND the R7 fit
    # fixes are inapplicable to it.  When the caller supplies BOTH an EXPLICIT
    # engaged carrier (a float conjugate or an ndarray wavefront -- NOT the
    # implicit / ``'auto'`` fit) AND ``tilt_aware_rays``, route the ray launch +
    # the R7 path through the carrier gradient (i.e. the ``carrier=R`` default
    # path) and warn.  The N5 auto-carrier tilt_aware path (``carrier`` None or
    # ``'auto'``) is UNTOUCHED -- its per-pixel launch stays pinned byte-identical
    # (a collimated input fits W == 0, so the guard never fires there).
    _explicit_carrier = (carrier is not None) and not (
        isinstance(carrier, str) and carrier == 'auto')
    _tilt_aware_launch = tilt_aware_rays
    if (_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER and tilt_aware_rays
            and (_carrier_W is not None) and _explicit_carrier):
        _tilt_aware_launch = False
        import warnings
        warnings.warn(
            "apply_real_lens_traced: tilt_aware_rays=True together with an "
            "explicit carrier= is less accurate than the carrier-referenced "
            "launch alone on a steeply-curved input (audit F3: ~5x the "
            "exit-curvature error on a steep spherical carrier).  The explicit "
            "carrier already carries the input congruence, so the per-pixel "
            "tilt launch is routed to the carrier-gradient (carrier=) path.  "
            "Drop tilt_aware_rays=True when passing an explicit carrier=.",
            RuntimeWarning, stacklevel=2)
    # ``_tilt_aware_launch`` is the EFFECTIVE launch mode after the F3 reroute;
    # the R7 path launches rays along the carrier gradient grad(W) and is guarded
    # OFF only for a genuine per-pixel tilt_aware launch (N5), i.e. when the F3
    # guard did NOT reroute (``_tilt_aware_launch`` stays True).
    _r7_carrier_path = (_carrier_W is not None) and (not _tilt_aware_launch)
    # preserve_input_phase='remap' needs the carrier eikonal to de-chirp the
    # input residual -- require an ENGAGED carrier (explicit or auto-fit that
    # engaged), else the "residual" would be the raw (possibly beyond-Nyquist)
    # input phase and the mode's premise breaks.  When the carrier is absent
    # or too flat to engage (collimated / near-collimated leg: W ~ 0 over the
    # grid), the de-chirp degenerates to the IDENTITY and the "residual" is
    # simply the input's own phase -- which is exactly the slow quantity the
    # pullback should carry there (a collimated beam's phase structure IS its
    # residual).  So 'remap' degrades gracefully to a W=0 de-chirp instead of
    # raising: required for the chain default, whose hand-off carriers pass
    # through near-collimated values (design-121 mid-chain R ~ +7e5 mm).
    _pip_remap_W = _carrier_W if (_pip_remap and _carrier_W is not None) \
        else (0.0 if _pip_remap else None)
    # niche C6: the smooth part of the input residual eikonal, ``a_fit``.  Built
    # further down (it needs the measured beam radius / centre) and left None on
    # every other path, so `` is None`` is the byte-identical gate everywhere it
    # is consulted.  When it is set, the rays launch along ``grad(W + a_fit)``,
    # the H6 entrance-eikonal term carries ``W + a_fit``, and what the residual
    # phasor transports is the LEFTOVER ``exp(i k0 (a - a_fit))`` -- the three
    # halves of one substitution ``W -> W + a_fit`` of the launch congruence.
    _resid_eik = None
    # The carrier-de-chirped input residual UNIT PHASOR on the ENTRANCE wave
    # grid, split into float64 real/imag parts for ``map_coordinates``.  Built
    # lazily and cached: both remap sampling paths need it, and it is a full
    # N^2 pair (the fine retrace leg runs at N = 8192-16384).
    _pip_res_ri = [None]

    def _pip_residual_ri():
        if _pip_res_ri[0] is None:
            _k = 2.0 * np.pi / wavelength
            _r = np.asarray(E_in) * np.exp(-1j * _k * _pip_remap_W)
            if _resid_eik is not None:
                # niche C6: de-chirp by ``a_fit`` too, and do it HERE rather
                # than at the sample points.  The rays already carry ``a_fit``
                # (launch direction + H6 eikonal + traced OPL), so what is left
                # to transport pointwise is ``a - a_fit`` either way -- but
                # this array is what the sampler INTERPOLATES bilinearly, and
                # removing the modelled part first leaves it the SLOW quantity.
                # Same discipline as ``remap_sampling='full'``: never resample
                # the fast phasor.  Row-banded so no second full-grid
                # coordinate pair is materialised (the N = 8192 fine retrace
                # leg would pay 1 GiB for it).
                # niche D9: ``_resid_eik`` is a model in the element's ABSOLUTE
                # frame (it was fitted there), so the grid it is evaluated on
                # must be absolute too.  Mixing the frames here would leave a
                # residual ``a(x) - a_fit(x - origin)`` that breaks the C6
                # cancellation with NO amplitude symptom.
                _ax = (np.arange(N, dtype=np.float64) - N / 2.0) * dx
                _ay = (np.arange(N, dtype=np.float64) - N / 2.0) * dy
                if _origin_set:
                    _ax = _ax + _org_x
                    _ay = _ay + _org_y
                _bd = max(1, int(4194304 // max(N, 1)))
                for _b0 in range(0, N, _bd):
                    _b1 = min(N, _b0 + _bd)
                    _r[_b0:_b1] *= np.exp(-1j * _k * _resid_eik.value(
                        np.broadcast_to(_ax[None, :], (_b1 - _b0, N)),
                        np.broadcast_to(_ay[_b0:_b1, None], (_b1 - _b0, N))))
            _a = np.abs(_r)
            with np.errstate(divide='ignore', invalid='ignore'):
                _r = np.where(_a > 0.0, _r / np.maximum(_a, 1e-300),
                              1.0 + 0.0j)
            _pip_res_ri[0] = (np.real(_r).astype(np.float64),
                              np.imag(_r).astype(np.float64))
            del _r, _a
        return _pip_res_ri[0]

    def _pip_sample_residual(row, col, sh=None):
        """Bilinearly sample the de-chirped input residual unit phasor at the
        entrance pixel coordinates ``(row, col)``.  ``cval`` = identity phase
        (1+0j) outside the input grid -- those rays carry zero amplitude
        anyway.  Returns the RAW sample (``|z| <= 1`` off-node); callers
        renormalise where they historically did, so the legacy sampling path
        stays byte-identical."""
        from scipy.ndimage import map_coordinates as _mc_r
        _rr, _ri = _pip_residual_ri()
        _c = np.vstack([np.asarray(row).ravel(), np.asarray(col).ravel()])
        _sr = _mc_r(_rr, _c, order=1, mode='constant', cval=1.0)
        _si = _mc_r(_ri, _c, order=1, mode='constant', cval=0.0)
        _z = _sr + 1j * _si
        return _z if sh is None else _z.reshape(sh)

    # F1 (audit) collimation guard: measure the residual angular spread
    # (after removing any carrier) and warn / delegate when the input is
    # too far from the reference congruence for the traced correction to be
    # accurate.  With a carrier supplied the residual is small (the carrier
    # absorbs the divergence), so this only fires for an UNREFERENCED
    # non-collimated input -- exactly the silent-blur regression class.
    # C4 (perf): the carrier=None residual IS the raw input tilt RMS -- the SAME
    # quantity the tilt_aware_rays=False launch-warning block below computes.
    # Compute the wrapping-safe tilt stats ONCE here and reuse them there (saves
    # one full-grid phase-increment + np.angle pass, ~9.5% of runtime at N=4k).
    # None until computed; the launch-warning block computes it if we skipped.
    _input_tilt = None
    if on_noncollimated != 'off':
        try:
            if _carrier_W is None:
                _input_tilt = _input_tilt_stats(E_in, wavelength, dx)
                _resid = _input_tilt[0] if _input_tilt is not None else 0.0
            else:
                _resid = _carrier_residual_rms(E_in, _carrier_W, wavelength, dx)
        except (ValueError, RuntimeError, FloatingPointError):
            _resid = 0.0
        if _resid > _NONCOLLIMATED_RESID_THRESH:
            if on_noncollimated == 'delegate':
                # v5.29.1 (audit E-L22): the model swap DROPS every
                # traced-only physics knob -- apply_real_lens has no ray
                # trace, so there is nothing to carry them.  Report the
                # non-default ones instead of discarding them silently (the
                # caller asked for a carrier-referenced ray-density field and
                # would otherwise receive an analytic screen field with no
                # diagnostic).
                # v5.35.0 (BUILD_R1_WIRING S4): the carrier is the ONE
                # traced-only argument the analytic model CAN honour -- it
                # drives the screen-obliquity + R1 corrections, which is the
                # whole reason those exist.  Forward it, but only when the
                # F1 statistic says it describes the field BETTER THAN
                # NOTHING: this branch fires precisely because the residual
                # after removing the carrier is large, and a carrier that
                # points the wrong way is worse than none (the correction's
                # cross term 2 p0 . q flips sign -- the same "wrong by twice
                # the term" signature the refutation used).  ``_resid`` is
                # that residual; the raw input tilt is the same statistic
                # with NO carrier removed, i.e. exactly the q = 0 arm.  So
                # the comparison is measured, not assumed, and it reuses the
                # guard's own estimator rather than inventing a second one.
                _fwd_carrier = None
                if carrier is not None and _carrier_W is not None:
                    if _input_tilt is None:
                        try:
                            _input_tilt = _input_tilt_stats(E_in, wavelength,
                                                            dx)
                        except (ValueError, RuntimeError,
                                FloatingPointError):
                            _input_tilt = None
                    _raw = (_input_tilt[0] if _input_tilt is not None
                            else 0.0)
                    if _resid < _raw:
                        _fwd_carrier = carrier
                _tdef = _traced_kwarg_defaults()
                _dropped = [
                    f'{_k}={_v!r}' for _k, _v in (
                        ('carrier', carrier if _fwd_carrier is None else
                         _tdef.get('carrier')),
                        ('amplitude_model', amplitude_model),
                        ('preserve_input_phase',
                         'remap' if _pip_remap else preserve_input_phase),
                        ('remap_sampling', remap_sampling),
                        ('caustic', caustic),
                        ('output_plane_distance', output_plane_distance),
                        ('tilt_aware_rays', tilt_aware_rays),
                        ('fit_radius_beam_factor', fit_radius_beam_factor),
                        ('inversion_method', inversion_method),
                        ('inverse_map', inverse_map),
                        ('newton_fit', _newton_fit_requested),
                        ('newton_max_iters', newton_max_iters),
                        ('newton_poly_order', newton_poly_order),
                        ('decentred_fit_poly_order', decentred_fit_poly_order),
                        ('ray_subsample', ray_subsample),
                        # the most dangerous drop: apply_real_lens has no
                        # notion of a reusable screen and returns a FIELD
                        ('return_screen', return_screen),
                    ) if _kwarg_differs_from_default(_v, _tdef.get(_k))]
                if _dropped or carrier is not None:
                    import warnings
                    _kept = (
                        "  carrier= IS forwarded (it drives the analytic "
                        "model's screen-obliquity + R1 corrections)."
                        if _fwd_carrier is not None else
                        ("  carrier= is NOT forwarded: removing it does not "
                         "reduce the input's angular spread, so it does not "
                         "describe this field and the angular correction "
                         "would be driven by the wrong ray angle."
                         if carrier is not None else ''))
                    _drop_txt = (
                        f"The analytic model has no ray-trace leg, so these "
                        f"traced-only arguments are DISCARDED: "
                        f"{', '.join(_dropped)}." if _dropped else
                        "The analytic model has no ray-trace leg.")
                    warnings.warn(
                        f"apply_real_lens_traced: on_noncollimated="
                        f"'delegate' is handing this call to "
                        f"apply_real_lens (input residual angular spread "
                        f"{_resid:.3f} rad > "
                        f"{_NONCOLLIMATED_RESID_THRESH} rad).  "
                        f"{_drop_txt}{_kept}  Keep "
                        f"on_noncollimated='warn' if you need them "
                        f"honoured, or call apply_real_lens directly to "
                        f"make the model choice explicit.",
                        RuntimeWarning, stacklevel=2)
                # v5.29.1 (audit E-M2): forward the RAW ``sag_chunk_rows``,
                # matching the four sibling amp-leg call sites -- the
                # resolver maps the documented force-whole-grid sentinel 0 to
                # None, which apply_real_lens then re-resolves to AUTO, so
                # forwarding the RESOLVED value re-enabled row banding
                # against an explicit opt-out.  See the resolver note above.
                return apply_real_lens(
                    E_in, prescription=lens_prescription,
                    wavelength=wavelength, dx=dx, bandlimit=bandlimit,
                    use_gpu=amp_use_gpu, wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype,
                    sag_chunk_rows=_sag_chunk_rows_raw,
                    carrier=_fwd_carrier,
                    progress=progress)
            else:  # 'warn'
                import warnings
                warnings.warn(
                    f"apply_real_lens_traced: input residual angular spread "
                    f"{_resid:.3f} rad exceeds the collimated-reference "
                    f"validity threshold ({_NONCOLLIMATED_RESID_THRESH} "
                    f"rad).  The plane-wave-referenced traced correction "
                    f"will be inaccurate (blurred).  Pass carrier= (a "
                    f"conjugate distance, an explicit wavefront, or 'auto') "
                    f"to reference the beam's own congruence, or use "
                    f"apply_real_lens.  Set on_noncollimated='delegate' to "
                    f"fall back automatically, or 'off' to silence.",
                    RuntimeWarning, stacklevel=2)

    # Reference input for the analytic lens-phase leg: the carrier
    # wavefront exp(i*k0*W) when supplied, else a unit plane wave (legacy
    # default).  ``phase_analytic_lens = angle(apply_real_lens(reference))`` is
    # subtracted from ``angle(E_analytic)`` in the exit assembly to remove the
    # analytic model's lens phase before the exact ray-traced OPL is added; the
    # carrier-referenced sphere makes that reference the beam's own congruence.
    # (R7 / audit F2: with the exact-sphere carrier the untapered reference already
    # cancels cleanly across the beam once the fit-restriction + cubic OPL upsample
    # below fix the ray-traced OPL -- so the reference is left INPUT-INDEPENDENT,
    # which keeps the prepared-screen / multi reuse path -- built on a ``ones``
    # placeholder -- byte-identical to a direct carrier call.)
    def _reference_input():
        if _carrier_W is not None:
            return np.exp(1j * _k0 * _carrier_W).astype(E_in.dtype)
        return np.ones_like(E_in)

    # ----- Step 1: amplitude envelope from the ANALYTIC lens model -----
    #
    # WHY WE CALL apply_real_lens HERE (the "double call"):
    #
    # apply_real_lens_traced is a HYBRID method.  It combines:
    #   (a) AMPLITUDE from wave optics — diffraction, vignetting, and
    #       the physically correct in-glass beam evolution (Fresnel
    #       effects at curved surfaces, edge ripples, aperture clipping)
    #   (b) PHASE from geometric ray tracing — the exact OPL through
    #       every curved glass/air interface, per pixel, via vector
    #       Snell's law at each surface
    #
    # The thin-element model's accuracy limitation is in its PHASE
    # (it approximates curved surfaces as phase screens at a single
    # z-plane), NOT in its amplitude (ASM through a uniform glass slab
    # handles diffraction correctly).  So we:
    #   1. Run apply_real_lens to get the full exit-plane field
    #   2. Keep only |E| (amplitude) — the wave-optics part
    #   3. Replace the phase with the geometrically exact ray-traced
    #      OPL map computed in Step 2 below
    #
    # This gives sub-nanometre OPD agreement with the geometric ray
    # trace (the "truth") while retaining physically correct
    # diffraction effects that pure ray tracing cannot capture.
    #
    # An earlier version used a simple air-ASM for the amplitude,
    # which produced a ~3.5 mm focus offset because air propagation
    # ≠ glass propagation (different wavenumber k = n·k0).  Using
    # apply_real_lens for the amplitude solves this because it
    # propagates through the correct glass/air refractive index
    # sequence.
    # Allocate 40% of the budget to the amplitude (which runs a full
    # apply_real_lens with its own per-surface cost), 50% to the ray
    # trace + Newton inversion, and 10% to the final field assembly.
    # ---------- Parallelism decision for amp and amp(pw) --------------
    # The two apply_real_lens calls (``amp`` on the real input, and
    # ``amp(pw)`` on a unit plane wave to recover the analytic lens
    # OPD) are data-independent and can run concurrently.  We dispatch
    # them on a ThreadPoolExecutor so the non-FFT work (sag, phase
    # screens, numexpr-fused multiplies, glass-interval setup)
    # overlaps.  The pyFFTW plan cache in ``propagation._fft2`` /
    # ``_ifft2`` holds a per-plan lock so the actual FFT execution
    # serialises safely on the shared aligned buffer; overlap is
    # therefore bounded by the FFT share of each call (~45-50 %) but
    # still gives ~40 % wall-time savings on the combined amp step.
    #
    # Memory cost of parallelism: two E fields and two sets of lens
    # intermediates alive simultaneously (~2x the peak of a single
    # call).  The ``parallel_amp_min_free_gb`` guard drops back to
    # sequential execution when available RAM is too tight for this
    # doubled working set -- tuned for the N=32768 complex128 case,
    # where the single-call transient peak is ~25 GB and doubling
    # brings it to ~50 GB.
    # parallel_amp=None (the default) resolves to the module global, letting
    # ``set_lens_parallel_amp(False)`` / ``set_low_memory(True)`` flip the
    # default for callers that don't pass the kwarg.  Explicit True/False win.
    if parallel_amp is None:
        parallel_amp = _LENS_PARALLEL_AMP_DEFAULT
    _use_parallel_amp = (preserve_input_phase and parallel_amp)
    if _use_parallel_amp:
        try:
            import psutil as _psutil

            # v5.17.2 (audit P2-21): honour a pinned set_max_ram() budget --
            # the doubled parallel working set must fit the effective
            # budget, not just physical free RAM (get_ram_budget() equals
            # the psutil read when no override is set).
            from ..memory import get_ram_budget
            _free_gb = min(int(_psutil.virtual_memory().available),
                           get_ram_budget()) / 1e9
            if _free_gb < parallel_amp_min_free_gb:
                _use_parallel_amp = False
        except (ImportError, AttributeError, OSError):
            # psutil missing or virtual_memory query failed --
            # leave parallel_amp enabled but the user can still
            # force off via the kwarg.
            pass

    amp_cb = ProgressScaler(progress, 'real_lens_traced',
                            lo=0.0, hi=0.50 if _use_parallel_amp else 0.40)

    if _use_parallel_amp:
        # Parallel path: run amp and amp(pw) concurrently.  Only the
        # amp call reports progress (0-50%); amp(pw) runs silently to
        # avoid interleaved status lines from two threads.  The ones-
        # like plane wave is materialised outside the thread so the
        # 17 GB allocation happens once, synchronously, with clear
        # OOM semantics.
        from concurrent.futures import ThreadPoolExecutor

        def _amp_call():
            return apply_real_lens(
                E_in, prescription=lens_prescription, wavelength=wavelength, dx=dx,
                bandlimit=bandlimit, use_gpu=amp_use_gpu,
                wave_propagator=wave_propagator,
                sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
                progress=lambda stage, frac, msg='':
                    amp_cb(frac, f'amp: {msg}'))

        if fast_analytic_phase and preserve_input_phase:
            # Skip the full amp(pw) ASM pass; compute the geometric
            # lens phase analytically from per-surface sag.
            E_analytic = _amp_call()
            # np.abs and np.angle work on cupy arrays via __array_function__
            # in recent numpy; but to be explicit, use xp.abs/xp.angle via
            # the module selector below.
            _xp = cp if _is_cupy_array(E_analytic) else np
            amp = _xp.abs(E_analytic)
            phase_analytic_lens = _geometric_lens_phase(
                lens_prescription, wavelength, dx, E_in.shape[0])
            if _xp is cp:
                phase_analytic_lens = cp.asarray(phase_analytic_lens)
        else:
            ones_input = _reference_input()  # carrier wavefront or plane wave

            def _amp_pw_call():
                return apply_real_lens(
                    ones_input, prescription=lens_prescription, wavelength=wavelength, dx=dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
                    progress=None)

            with ThreadPoolExecutor(max_workers=2,
                                    thread_name_prefix='rlt_amp') as _tp:
                fut_amp = _tp.submit(_amp_call)
                fut_pw = _tp.submit(_amp_pw_call)
                E_analytic = fut_amp.result()
                E_analytic_pw = fut_pw.result()
            del ones_input
            _xp = cp if _is_cupy_array(E_analytic) else np
            amp = _xp.abs(E_analytic)
            phase_analytic_lens = _xp.angle(E_analytic_pw)
            del E_analytic_pw  # free ~17 GB at N=32768 before Newton starts
    else:
        # Sequential fallback (preserve_input_phase=False or RAM tight).
        E_analytic = apply_real_lens(
            E_in, prescription=lens_prescription, wavelength=wavelength, dx=dx, bandlimit=bandlimit,
            use_gpu=amp_use_gpu, wave_propagator=wave_propagator,
            sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
            progress=lambda stage, frac, msg='': amp_cb(frac, f'amp: {msg}'))
        _xp = cp if _is_cupy_array(E_analytic) else np
        amp = _xp.abs(E_analytic)
        # When preserving input phase (the physically-correct default),
        # we also need to know the *analytic model's lens-only phase* so
        # we can subtract it out before adding the ray-traced OPL back in.
        # We extract it by running apply_real_lens on a unit plane wave --
        # the result's phase is exactly the analytic lens's OPL
        # (plus small wave-propagation-through-glass effects) applied to
        # a flat input.
        if preserve_input_phase:
            if fast_analytic_phase:
                # Analytic geometric phase: per-surface sag phase
                # screens summed locally, no ASM through glass.  On
                # Design 51 lenses this introduces at most ~10 nm OPL
                # error (L4, F/6.8 doublet) and essentially none on
                # slower singlets -- below the numerical noise floor
                # of the rest of the pipeline.
                phase_analytic_lens = _geometric_lens_phase(
                    lens_prescription, wavelength, dx, E_in.shape[0])
                if _xp is cp:
                    phase_analytic_lens = cp.asarray(phase_analytic_lens)
            else:
                analytic_pw_cb = ProgressScaler(progress, 'real_lens_traced',
                                                 lo=0.40, hi=0.50)
                E_analytic_pw = apply_real_lens(
                    _reference_input(), prescription=lens_prescription, wavelength=wavelength, dx=dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
                    progress=lambda stage, frac, msg='':
                        analytic_pw_cb(frac, f'amp(pw): {msg}'))
                phase_analytic_lens = _xp.angle(E_analytic_pw)
                del E_analytic_pw
        else:
            phase_analytic_lens = None
    # When amp_use_gpu=True the amp pipeline returns CuPy arrays.  The
    # rest of apply_real_lens_traced (ray trace, Newton, final E_out
    # assembly) is CPU-only, so pull the amp outputs back to the host
    # here rather than xp-ifying the final-assembly section.
    if _is_cupy_array(E_analytic):
        E_analytic = cp.asnumpy(E_analytic)
    if _is_cupy_array(amp):
        amp = cp.asnumpy(amp)
    if phase_analytic_lens is not None and _is_cupy_array(phase_analytic_lens):
        phase_analytic_lens = cp.asnumpy(phase_analytic_lens)

    # ---- N10a: NO field-frame amplitude override (removed 2026-07-20) -----
    # The P9 build swapped the amplitude leg to the bare input envelope
    # (``E_out = E_in * exp(i k0 opl)``) for field-frame (decenter/tilt)
    # prescriptions, on the theory that coma is a pure exit-pupil PHASE
    # aberration.  The adversarial verifier REFUTED this: forcing the input-
    # envelope amplitude on CENTERED geometry (a 1e-7 decenter, or an exact-
    # conic ``sag_callable``) already widened the on-axis EE80 by ~8% with ZERO
    # decenter (grid-robust), so the reported "1.097 broadening / within 1.6% of
    # ZOS" was an amplitude-MODEL artefact (decentered override-amp compared to
    # centered analytic-amp), not induced coma.  Held to ONE amplitude model the
    # traced EE80 under decenter is unstable and wavelength/plane-dependent
    # (|E_analytic|: 0.88x @1.31 / 1.09x @0.633 at the paraxial image; |E_in|:
    # 0.99x @1.31 / 0.95x @0.633) -- because the traced hybrid's GRID-INDEXED
    # amplitude cannot carry the transverse walk-off (the coma flare is an
    # asymmetric ray-DENSITY redistribution the Newton-inverted OPL alone does
    # not put into |E|), and the singlet's paraxial plane is strongly defocused.
    # This is a genuine traced-model limit of the same class as the P3 single-
    # plane analytic limit.  The decenter GEOMETRY + OPL the traced model now
    # carries are correct (centroid / sign-mirror / tilt all oracle-matched),
    # but its decentered-spot EE is amplitude-limited, so the amplitude leg is
    # left as the standard self-consistent reconstruction here (no swap).  The
    # accurate decentered-coma EE reference is ``apply_real_lens_gbd`` (N10b):
    # its beamlets carry the walk-off amplitude, so it BROADENS matching ZOS
    # (ratio 1.035 @1.31um) and the geom-spot oracle (~1% on the ratio).  See
    # docs/audit_real_lens_displaced_2026_07_19.md (P9 / N10a) for the full
    # envelope + the routing to GBD.  ``_prescription_has_field_frame`` is kept
    # as the field-frame detector (used by the tests and available for routing).

    call_progress(progress, 'real_lens_traced', 0.40,
                  'ray-tracing exit pupil')

    # ----- Step 2: ray-traced OPL per (subsampled) pixel ---------------
    # Launch a dense grid of rays from the entrance pupil; each ray
    # bends through the lens and lands at a *different* (x_out, y_out)
    # at the exit plane.  We need OPL associated with the exit
    # position (matching the wave's exit-plane grid), not the entrance
    # position, so we scatter-interpolate ``opl(x_out, y_out)`` onto
    # the wave grid.
    #
    # IMPORTANT: build surfaces WITHOUT the prescription aperture so
    # rays launched slightly beyond the entrance pupil are not
    # vignetted -- they may end up landing *inside* the wave grid
    # after refraction-induced inward shift.  But we DO restrict the
    # entrance launch positions to a modest over-margin around the
    # actual aperture so ultra-marginal rays (at huge angles of
    # incidence on the first surface) don't contaminate the OPL
    # function with non-paraxial branches.  The wave amplitude mask is
    # applied separately and zeros any spurious phase outside the
    # physical aperture anyway.
    pres_no_ap = dict(lens_prescription)
    pres_no_ap.pop('aperture_diameter', None)
    surfaces = surfaces_from_prescription(pres_no_ap)

    sub = max(1, int(ray_subsample))
    # Pick the launch radius: aperture (if specified) plus a 50 %
    # over-margin so that the entrance-grid sampling covers all wave-
    # grid exit positions even for fast lenses (rays bend inward so
    # exit positions are closer to axis than entrance).
    if aperture is not None:
        launch_radius = 0.5 * aperture * 1.50
    else:
        # niche D9: the apertureless branch is GRID-EXTENT-derived, and the
        # launch grid stays AXIS-centred (that is what keeps the odd-n_launch
        # on-axis piston reference exact), so a decentred grid whose far corner
        # sits at |origin| + N*dx/2 would be launched over only part of itself
        # -- Newton would then hand those pixels NaN and the beam would be
        # quietly clipped.  Reach far enough to cover the moved grid.  Exactly
        # the historical value at origin=(0, 0).
        launch_radius = 0.5 * N * dx + float(np.hypot(_org_x, _org_y))

    # ----- P2 aperture:beam cliff guard (audit
    # AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4) -------------------
    # Resolve the beam-relative RAY-FIT radius (used at the fit-mask site far
    # below) and, independently, the warn-only aperture >> beam flag.  Only the
    # fit domain is affected -- ``launch_radius`` above, the Newton ``bound``
    # and the out-of-domain NaN threshold all stay aperture-derived, so no
    # field energy is clipped by this guard.  See
    # ``_FIT_RADIUS_BEAM_FACTOR_DEFAULT`` for the mechanism + measurements.
    if on_aperture_beam not in ('warn', 'silent'):
        raise ValueError(
            "apply_real_lens_traced: on_aperture_beam must be 'warn' or "
            f"'silent' (got {on_aperture_beam!r})")
    _frbf = None
    if fit_radius_beam_factor is not None:
        _frbf = float(fit_radius_beam_factor)
        if not (np.isfinite(_frbf) and _frbf > 0.0):
            raise ValueError(
                "apply_real_lens_traced: fit_radius_beam_factor must be a "
                f"finite positive number or None (got "
                f"{fit_radius_beam_factor!r})")
    # niche D1: the beam is not always on the grid centre.  Both halves of the
    # guard (the measured radius AND the disc it sizes) are second moments /
    # discs about a POINT; referencing them to the origin makes the guard read
    # sqrt(2 x_c^2 + w^2) and go inert as the decentre grows (see
    # ``beam_centre``).  A TiltedCarrier already states the chief-ray position,
    # so it supplies the default and the chain needs no extra plumbing.
    # D7: minimum fit order for an OFF-CENTRE disc (resolved here so a bad
    # value raises before any ray work, like every other guard above).
    if decentred_fit_poly_order is None:
        _dec_order = int(_DECENTRED_FIT_POLY_ORDER)
    else:
        try:
            _dec_order = int(decentred_fit_poly_order)
        except (TypeError, ValueError):
            raise ValueError(
                "apply_real_lens_traced: decentred_fit_poly_order must be a "
                f"positive integer or None (got "
                f"{decentred_fit_poly_order!r})")
        if _dec_order < 1 or _dec_order != decentred_fit_poly_order:
            raise ValueError(
                "apply_real_lens_traced: decentred_fit_poly_order must be a "
                f"positive integer or None (got "
                f"{decentred_fit_poly_order!r})")
    _bcx = _bcy = 0.0
    if beam_centre is not None:
        try:
            _bcx, _bcy = (float(_v) for _v in beam_centre)
        except (TypeError, ValueError):
            raise ValueError(
                "apply_real_lens_traced: beam_centre must be a 2-sequence "
                f"(x0, y0) in metres, or None (got {beam_centre!r})")
        if not (np.isfinite(_bcx) and np.isfinite(_bcy)):
            raise ValueError(
                "apply_real_lens_traced: beam_centre must be finite (got "
                f"{beam_centre!r})")
    elif isinstance(carrier, TiltedCarrier):
        _bcx, _bcy = float(carrier.x0), float(carrier.y0)
    # niche C1 item 1: a NULL decentre must NOT flip the fit branch.  Stage
    # one is the grid-pitch floor (an offset the grid cannot even represent);
    # stage two, below, is the beam-relative floor -- it needs the measured
    # radius, so it runs after it.  See ``_DECENTRE_GATE_W_FRAC`` for both
    # thresholds and the sweep that set them.
    _dec_mag = float(np.hypot(_bcx, _bcy))
    _dec_null_floor = _DECENTRE_GATE_PIXELS * min(float(dx), float(dy))
    _beam_decentred = _dec_mag > _dec_null_floor
    # ---- fix D5 (2026-08-06): can the RESOLVED fit basis honour a fit-domain
    # restriction at all?  Named ONCE here; every gate below reads this flag
    # instead of re-spelling ``newton_fit != 'spline'``, so a future basis
    # cannot pick up half of them.
    #
    # WHY THE SPLINE BASIS CANNOT, as a fact about the code rather than a
    # preference.  The restriction has exactly two implementations and neither
    # has a home on ``RectBivariateSpline``:
    #   * D1's REGULARISED form is least-squares WEIGHTS (``_fit_weights``),
    #     consumed only by ``_Cheb2DEvaluator(..., weights=...)``.
    #     ``RectBivariateSpline.__init__(x, y, z, bbox, kx, ky, s, maxit)``
    #     takes no weights.
    #   * the historical form is a hard NaN mask outside the disc.  A single
    #     non-finite sample anywhere in a ``RectBivariateSpline``'s data array
    #     propagates through its banded solve: measured, one NaN corner on a
    #     21x21 lattice makes ``.ev()`` return NaN AT THE GRID CENTRE.  So
    #     masking would not restrict the fit, it would destroy it.
    # And the MECHANISM the restriction exists to control -- a GLOBAL
    # total-degree least-squares fit whose every coefficient sees every
    # sample, extrapolating outside its own data support -- is a property of
    # that basis, not of the data.  The library's own pointwise scoring
    # against an exact skew ray trace (see ``_REMAP_RESID_FREEZE_MARGIN``)
    # measures it: polynomial 5.608 / 15.079 um in the skirt / at the
    # aperture rim, spline 0.006 / 0.002 um.
    # The C11 arbiter and the C12 predictor CHOOSE BETWEEN TWO DISCS, so with
    # no disc there is nothing for them to arbitrate; they are gated on the
    # same flag for that reason and not as a separate judgement.
    #
    # WHAT IS *NOT* CLAIMED: that the spline basis is safe past the
    # aperture:beam cliff.  Measured on the E4 corrected relay (beam w = 2 mm
    # FIXED, only the aperture grown, N=1536): at a 4 mm aperture both bases
    # give exit-wavefront Strehl 0.998; at 6-10 mm the polynomial basis
    # degrades to 0.042 and ``fit_radius_beam_factor=2.0`` recovers it to
    # 0.999, while the spline basis returns an ALL-ZERO exit field and the
    # knob cannot rescue it because this flag excludes it.  That is why the
    # guard below is emitted rather than the inertness being left silent, and
    # it is the same finding that put ``newton_fit='auto'`` back on POLYNOMIAL.
    _fit_domain_basis_ok = (newton_fit != 'spline')
    # ---- FIX FIT-DOMAIN SYMMETRY (2026-08-12): the DOMAIN is not the
    # INTERPOLANT.  ``_fit_domain_basis_ok`` above answers exactly one
    # question -- "can THIS BASIS restrict ITS OWN forward fit to the
    # requested region?" -- and D5's adjudication of it stands unchanged.
    # What it must NOT answer is the different question "is there a requested
    # region at all?", because the region is a property of the BEAM and the
    # traced samples, and OTHER consumers can honour it even where the
    # interpolant cannot.
    #
    # Conflating the two leaked: the inverse-characteristic model
    # (:mod:`lumenairy.elements._lens_imap`) is a GLOBAL total-degree
    # Chebyshev in EXIT coordinates, i.e. exactly the mechanism the
    # restriction exists to control, and it needs the domain for the same
    # reason the forward polynomial fit does (measured: unrestricted, its
    # held-out OPL error inside the beam is 4.5258e-01 waves against
    # 1.9965e-05 restricted -- BUILD_INVERSE_MAP_2026_08_11 S6.5b).  But it
    # inherited its sample set from whatever the FORWARD fit's restriction
    # had left behind, so on the spline basis it silently received the whole
    # launch square while on the polynomial basis it received the disc.  The
    # two bases therefore described DIFFERENT MAPS, which broke the shipped
    # backend-symmetry guard
    # ``test_niche_c6::test_the_two_newton_fit_backends_still_describe_the_same_map``
    # (measured 1.0600e-02 against its 5e-04 bar; on this fixture the
    # polynomial arm ENGAGED the model on 2 809 disc samples while the spline
    # arm handed it 32 761 launch-square samples and G8 refused the build).
    #
    # So the domain is resolved BASIS-INDEPENDENTLY whenever a consumer that
    # can honour it is going to run, and each consumer honours it iff it can:
    #
    #   consumer                         honours the domain?
    #   forward fit, polynomial basis    YES -- NaN mask or D1 weights
    #   forward fit, spline basis        NO  -- D5, unchanged and re-pinned
    #   the inverse-map model            YES -- ALWAYS, on either basis
    #
    # WHY NOT THE OTHER WAY ROUND (i.e. make the spline forward fit honour it
    # too).  ``RectBivariateSpline`` needs a full NaN-free tensor grid, so the
    # only expressible restriction is a rectangular SUB-LATTICE -- and no
    # rectangle equals a disc.  MEASURED on niche C6's own 181 x 181 lattice:
    # the disc holds 2 809 of 32 761 nodes; the largest sub-lattice that stays
    # inside it holds 1 849 and DROPS 960 disc samples, and the best-overlap
    # sub-lattice still misses by 504 nodes (252 kept that the disc excludes,
    # 252 dropped that it includes).  A sub-lattice restriction would
    # therefore leave the two bases handing the model DIFFERENT sample sets
    # again -- it cannot fix this defect even in principle -- while
    # additionally moving every spline consumer.  The alternative "mask and
    # fill" keeps the rectangle but fabricates data outside the disc, which is
    # not a restriction at all.  Both are refuted in
    # FIX_FIT_DOMAIN_SYMMETRY_2026_08_12 S2.
    _imap_domain_gate = (sub > 1 and inversion_method == 'newton'
                         and not _chunk_assembly and not use_gpu)
    #: True when a fit domain must be resolved even though the resolved basis
    #: cannot apply it to its own forward fit.  Scoped to the calls that
    #: actually build the model, so no spline call that does not build one
    #: moves a single bit.
    _fit_domain_for_model = (not _fit_domain_basis_ok
                             and _IMAP.imap_enabled(inverse_map)
                             and _imap_domain_gate)
    #: The domain is REQUESTED (by any consumer) when either can use it.
    _fit_domain_wanted = _fit_domain_basis_ok or _fit_domain_for_model
    _fit_domain_inert = []
    if not _fit_domain_basis_ok:
        if _frbf is not None:
            _fit_domain_inert.append(
                f'fit_radius_beam_factor={_frbf!r} (the beam-relative ray-fit '
                f'disc, and with it the aperture:beam cliff remedy)')
        if decentred_fit_poly_order is not None:
            _fit_domain_inert.append(
                f'decentred_fit_poly_order={decentred_fit_poly_order!r} (the '
                f'niche-D7 order raise, which applies only to the weighted '
                f'off-centre fit)')
        if _beam_decentred and (DECENTRED_FIT_ARBITER
                                or DECENTRED_FIT_PREDICTOR):
            _fit_domain_inert.append(
                f'DECENTRED_FIT_ARBITER={DECENTRED_FIT_ARBITER!r} / '
                f'DECENTRED_FIT_PREDICTOR={DECENTRED_FIT_PREDICTOR!r} (the '
                f'niche-C11/C12 ray-fit branch selection, which chooses '
                f'between two fit DISCS)')
    if _fit_domain_inert and on_fit_domain_basis != 'silent':
        # SCOPE, and it is narrower than it used to be.  The announcement is
        # about the FORWARD FIT only.  When the inverse-characteristic model
        # is going to be built on this call it resolves and applies the SAME
        # domain regardless of basis (fix FIT-DOMAIN SYMMETRY 2026-08-12), so
        # saying "no ray-fit-domain guard at all" would be false there.
        _fdb_scope = (
            "the element's FORWARD fit.  The inverse-characteristic model "
            "IS built on this call and it applies the same domain on either "
            "basis, so the restriction is not wholly inert -- what it does "
            "not reach is the RectBivariateSpline forward fit and the Newton "
            "inversion that reads it"
            if _fit_domain_for_model else
            "this call, which therefore runs with NO ray-fit-domain guard at "
            "all")
        _fdb_msg = (
            f"apply_real_lens_traced: newton_fit={newton_fit!r} cannot honour "
            f"the fit-domain restriction(s) requested here -- "
            + '; '.join(_fit_domain_inert) + ".  They are INERT on this "
            "basis (a local bicubic takes no least-squares weights, and a "
            "NaN-masked data array makes RectBivariateSpline return NaN "
            "everywhere rather than a restricted fit) for " + _fdb_scope +
            ".  That is usually harmless "
            "-- a local interpolant does not spread marginal-ray error into "
            "the beam the way a global polynomial does -- but it is NOT a "
            "safe substitute past the aperture:beam cliff: measured on the "
            "E4 corrected relay (beam fixed, aperture grown) the polynomial "
            "basis degrades to exit-wavefront Strehl 0.042 and "
            "fit_radius_beam_factor=2.0 recovers it to 0.999, while the "
            "spline basis returns an ALL-ZERO exit field that this knob "
            "cannot rescue.  Use newton_fit='polynomial' (what 'auto' "
            "resolves to) if you need the restriction; pass "
            "on_fit_domain_basis='silent' to acknowledge, or 'error' to "
            "make the combination fatal.")
        if on_fit_domain_basis == 'error':
            raise ValueError(_fdb_msg)
        import warnings
        warnings.warn(_fdb_msg, RuntimeWarning, stacklevel=2)
    _beam_fit_radius = None
    _beam_fit_radius_conc = None
    _w_in_beam = 0.0
    _w_in_origin = 0.0
    if _frbf is not None or (on_aperture_beam == 'warn' and aperture is not None):
        _w_in_beam = _input_beam_amp_radius(
            E_in, dx, dy, centre=((_bcx, _bcy) if _beam_decentred else None),
            origin=(_org_x, _org_y))
        if (_beam_decentred and _w_in_beam > 0.0
                and _dec_mag <= _DECENTRE_GATE_W_FRAC * _w_in_beam):
            # A disc this nearly concentric reaches no further into the
            # aperture than the historical one did, so keep the historical
            # path -- INCLUDING its origin-referenced radius, which is what
            # makes the fall-back byte-identical rather than merely close.
            _beam_decentred = False
            _w_in_beam = _input_beam_amp_radius(E_in, dx, dy, centre=None,
                                                origin=(_org_x, _org_y))
        elif (_beam_decentred and _frbf is not None
                and (DECENTRED_FIT_ARBITER or DECENTRED_FIT_PREDICTOR)
                and _fit_domain_wanted):
            # niche C11: the arbiter below scores the HISTORICAL concentric
            # candidate too, and that disc is sized from the ORIGIN-referenced
            # second moment -- the same number the fall-back above re-measures.
            # Taken ONLY when the arbiter can actually run, so nothing below
            # the C1 gate does any extra work.
            _w_in_origin = _input_beam_amp_radius(E_in, dx, dy, centre=None,
                                                  origin=(_org_x, _org_y))
    if _frbf is not None and _w_in_beam > 0.0:
        _beam_fit_radius = min(_frbf * _w_in_beam, launch_radius)
    if _frbf is not None and _w_in_origin > 0.0:
        _beam_fit_radius_conc = min(_frbf * _w_in_origin, launch_radius)
    # fix D5: the cliff guard is skipped when a ray-fit disc will be applied,
    # because the disc IS the remedy.  On a basis that cannot apply it the
    # remedy is not in force, so the guard must stay reachable -- otherwise an
    # INERT ``fit_radius_beam_factor`` silences the warning about the very
    # failure it has stopped preventing.
    if (on_aperture_beam == 'warn' and aperture is not None
            and _w_in_beam > 0.0
            and (_beam_fit_radius is None or not _fit_domain_basis_ok)):
        _ap_beam_ratio = float(aperture) / (2.0 * _w_in_beam)
        if _ap_beam_ratio > _APERTURE_BEAM_WARN_RATIO:
            import warnings
            warnings.warn(
                f"apply_real_lens_traced: the physical aperture "
                f"({float(aperture)*1e3:.3f} mm) is {_ap_beam_ratio:.2f}x the "
                f"beam 1/e^2 diameter ({2e3*_w_in_beam:.3f} mm), above the "
                f"{_APERTURE_BEAM_WARN_RATIO}x aperture:beam ratio beyond "
                f"which the traced OPL fit can be corrupted by marginal rays "
                f"the beam never occupies (audit "
                f"AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4: measured "
                f"exit-wavefront Strehl 0.998 -> 0.039 across this cliff on a "
                f"fast singlet).  Whether it bites depends on how aberrated "
                f"the surfaces are at the aperture edge.  Pass "
                f"fit_radius_beam_factor=2.0 to restrict the ray-fit domain to "
                f"the beam (no energy is vignetted by that -- only the fit "
                f"domain changes), or on_aperture_beam='silent' to "
                f"acknowledge.",
                RuntimeWarning, stacklevel=2)

    # The RAY-fit disc.  Resolved HERE rather than at the fit site below
    # because niche C6's residual-eikonal freeze has to clear it (see
    # ``_REMAP_RESID_FREEZE_MARGIN``) and that model is built above the fit.
    # The restriction itself is applied unchanged further down.
    _fit_r_geom = (_CARRIER_FIT_RADIUS_FRAC * launch_radius
                   if _r7_carrier_path else None)
    _fit_r_max = _fit_r_geom
    if _beam_fit_radius is not None:
        _fit_r_max = (_beam_fit_radius if _fit_r_max is None
                      else min(_fit_r_max, _beam_fit_radius))
    # niche C11: and the same resolution for the CONCENTRIC candidate, whose
    # radius comes from the origin-referenced moment.  ``None`` whenever the
    # arbiter is not going to run (see the gate site above).
    _fit_r_max_conc = None
    if _beam_fit_radius_conc is not None:
        _fit_r_max_conc = (_beam_fit_radius_conc if _fit_r_geom is None
                           else min(_fit_r_geom, _beam_fit_radius_conc))
    # ... and its radius measured about the BEAM, which is what the freeze
    # circle has to clear.  Off centre the disc IS beam-centred with radius
    # ``_beam_fit_radius`` (the geometric intersection below only trims the
    # side away from the beam); concentric, the beam sits at the disc centre
    # and the radius is ``_fit_r_max``.
    _fit_r_about_beam = (_beam_fit_radius
                         if (_beam_fit_radius is not None and _beam_decentred)
                         else _fit_r_max)

    # ---- niche C6: the stationary-phase launch for 'remap' ---------------
    # Fit the smooth part of the input RESIDUAL eikonal and hand it to the ray
    # launch, the H6 entrance-eikonal term and the residual de-chirp together.
    # See :data:`REMAP_STATIONARY_PHASE_LAUNCH` for the derivation.
    #
    # GATE.  Only ``preserve_input_phase='remap'`` transports the residual
    # GEOMETRICALLY, so only there does the launch direction decide where the
    # residual is sampled; on every other mode the residual is carried by the
    # analytic wave pair (``True``) or discarded (``False``) and augmenting the
    # launch would be incoherent.  An ENGAGED carrier is required as well: it is
    # what makes ``a`` a small residual rather than the input's whole phase (the
    # ``carrier``-less de-chirp is the identity), and the ``_r7_carrier_path``
    # test also excludes the per-pixel ``tilt_aware_rays`` launch, which already
    # carries the input's own slope and would double-count it.
    if (_pip_remap and REMAP_STATIONARY_PHASE_LAUNCH and _r7_carrier_path
            and _carrier_grad is not None and _carrier_W_fn is not None):
        _res_w = _w_in_beam
        if not (_res_w > 0.0):
            _res_w = _input_beam_amp_radius(
                E_in, dx, dy,
                centre=((_bcx, _bcy) if _beam_decentred else None),
                origin=(_org_x, _org_y))
        _resid_eik = _fit_residual_eikonal(
            E_in, _carrier_W, wavelength, dx, dy, (_bcx, _bcy), _res_w,
            stride=max(1, int(sub)), ray_fit_radius=_fit_r_about_beam,
            origin=(_org_x, _org_y))
    if _remap_launch_out is not None:
        _remap_launch_out.update(
            {'engaged': _resid_eik is not None,
             'flag': bool(REMAP_STATIONARY_PHASE_LAUNCH),
             'remap': bool(_pip_remap)})
        if _resid_eik is not None:
            _remap_launch_out.update(_resid_eik.diag)

    # ----- Subsampling guardrail --------------------------------------
    # The Newton-inversion step builds a cubic-spline interpolant of the
    # entrance->exit map on a coarse grid and uses bilinear interp to
    # back-fill the full grid.  If the coarse grid is too sparse
    # relative to the lens aperture the interpolant aliases and the
    # whole exit-pupil OPD is wrong (RMS phase err blows up roughly
    # as (samples_per_aperture)^-2 from the benchmark sweep -- 32
    # samples gives ~85 nm at lambda = 1.31 um, 16 samples gives ~350
    # nm and is unusable).
    if min_coarse_samples_per_aperture:
        if aperture is not None:
            ap_diameter = float(aperture)
            _ap_label = 'aperture'
        else:
            # v5.17.1 (audit P3-08): the floor was documented as enforced
            # against the launch radius when no ``aperture_diameter`` is
            # set, but the guard was silently skipped for apertureless
            # prescriptions.  Derive the effective pupil from the largest
            # per-surface ``clear_aperture`` when present (the actual
            # pupil-limiting hardware, capped at the launch diameter the
            # coarse grid actually spans), else the launch diameter itself
            # (= the grid extent), so apertureless prescriptions get the
            # same aliasing protection.
            _cas = [float(s['clear_aperture'])
                    for s in (lens_prescription.get('surfaces') or [])
                    if isinstance(s, dict)
                    and s.get('clear_aperture') is not None]
            if _cas:
                ap_diameter = min(max(_cas), 2.0 * launch_radius)
                _ap_label = 'largest clear_aperture'
            else:
                ap_diameter = 2.0 * launch_radius
                _ap_label = 'launch diameter (grid extent)'
        coarse_dx = dx * sub
        n_coarse_across = ap_diameter / coarse_dx if coarse_dx > 0 else 0
        if n_coarse_across < min_coarse_samples_per_aperture:
            # Compute the largest sub that *would* be safe so the
            # error message gives the user an actionable number.
            safe_sub = max(1, int(np.floor(
                ap_diameter / (dx * min_coarse_samples_per_aperture))))
            msg = (
                f'apply_real_lens_traced: ray_subsample={ray_subsample} '
                f'gives only {n_coarse_across:.1f} coarse samples across '
                f'the {ap_diameter*1e3:.2f}-mm {_ap_label} (threshold '
                f'{min_coarse_samples_per_aperture}).  At this density '
                f'the spline interpolation of the wavefront will alias '
                f'and the OPD will be wrong by ~lambda/4 or more.  '
                f'Drop to ray_subsample <= {safe_sub} (or pass '
                f'min_coarse_samples_per_aperture=0 to override).'
            )
            if on_undersample == 'error':
                raise ValueError(msg)
            elif on_undersample == 'warn':
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            elif on_undersample != 'silent':
                raise ValueError(
                    f"on_undersample must be 'error', 'warn', or "
                    f"'silent' (got {on_undersample!r})")

    # Number of samples across the launch grid (subsampled).  Keep it
    # at least proportional to the grid resolution so the OPL
    # function is well sampled.
    n_launch = max(8, int(2 * launch_radius / (dx * sub)))
    # Ensure odd so there's a sample on the optical axis (entrance
    # centre) -- makes on-axis piston subtraction exact.
    if n_launch % 2 == 0:
        n_launch += 1
    xs_in = np.linspace(-launch_radius, launch_radius, n_launch)
    # Use indexing='ij' so that after reshaping trace results to
    # (n_launch, n_launch), array[i, j] corresponds to entrance
    # (X = xs_in[i], Y = xs_in[j]) -- matching scipy's
    # RectBivariateSpline(x_knots, y_knots, values) convention where
    # values[i, j] is the value at (x_knots[i], y_knots[j]).  With the
    # default 'xy' indexing the reshape transposes x/y, which makes
    # the spline's Jacobian wrong and Newton converges to bogus
    # points for 2D wave pixels off the symmetry axes.
    Xs_in, Ys_in = np.meshgrid(xs_in, xs_in, indexing='ij')
    h_x = Xs_in.ravel()
    h_y = Ys_in.ravel()
    # Tier 1 input-aware ray launch: derive each ray's direction from
    # the local phase gradient of E_in at its entrance position.  For
    # plane-wave inputs this reduces to L = M = 0 (identical to the
    # classical collimated launch); for structured inputs (MLA
    # modulation, off-axis sources, pre-aberrated wavefronts) the
    # rays correctly start at the angle implied by E_in, giving the
    # lens its actual per-ray OPL instead of a plane-wave-reference
    # OPL map.  See :func:`_sample_local_tilts` for the extraction.
    if _tilt_aware_launch:
        L_in, M_in = _sample_local_tilts(E_in, wavelength, dx, Xs_in, Ys_in,
                                         origin=(_org_x, _org_y))
        L_in = L_in.ravel()
        M_in = M_in.ravel()
    elif _carrier_grad is not None:
        # Carrier-referenced launch (audit S5.1): rays follow the carrier
        # normals grad(W) at their entrance positions, so the ray-traced
        # OPL is referenced to the beam's own congruence (matching the
        # exp(i*k0*W) amplitude reference) rather than a plane wave.
        L_in, M_in = _carrier_grad(h_x, h_y)
        L_in = np.asarray(L_in, dtype=np.float64).ravel()
        M_in = np.asarray(M_in, dtype=np.float64).ravel()
        if _resid_eik is not None:
            # niche C6: launch along grad(W + a_fit), the TOTAL entrance
            # eikonal, so the Newton pullback lands on the stationary point of
            # ``W + a + V(., X)`` instead of ``W + V(., X)``.  grad(eikonal) IS
            # the transverse direction cosine (|grad S|^2 + (dS/dz)^2 = 1), so
            # the two gradients ADD with no renormalisation -- the O(|grad a|^2)
            # correction a normalisation would introduce is precisely the order
            # this fix exists to carry, and it belongs in the eikonal, not in a
            # rescaling of it.
            _gLa, _gMa = _resid_eik.grad(h_x, h_y)
            _gLa = np.asarray(_gLa, dtype=np.float64).ravel()
            _gMa = np.asarray(_gMa, dtype=np.float64).ravel()
            _resid_eik.diag['grad_a_fit_max_launch'] = (
                float(np.nanmax(np.hypot(_gLa, _gMa))) if _gLa.size else 0.0)
            if _remap_launch_out is not None:
                _remap_launch_out.update(_resid_eik.diag)
            L_in = L_in + _gLa
            M_in = M_in + _gMa
    else:
        # 4.10: emit a one-time warning when the input field has a
        # measurable transverse tilt and tilt_aware_rays=False.  The
        # plane-wave reference OPD becomes inaccurate when the input
        # tilt is comparable to lambda / aperture.  Estimate the
        # transverse tilt as the RMS of grad(phase) / k0 over the
        # support of |E_in|; cap the check via a try-except so degenerate
        # input fields don't crash apply_real_lens_traced.
        # 4.10: emit a one-time warning when the input field has a measurable
        # transverse tilt and tilt_aware_rays=False -- the plane-wave reference
        # OPD becomes inaccurate when the input tilt is comparable to
        # lambda / aperture.  The tilt statistics (wrapping-safe nearest-
        # neighbour phase increments over the bright support; see
        # :func:`_input_tilt_stats`) are the SAME the noncollimated guard used
        # above, so we REUSE its result here (C4 perf) -- only computing them
        # when that guard was skipped (on_noncollimated='off').  The
        # coherence_ratio distinguishes a genuine single-beam tilt (~1, where
        # tilt_aware_rays=True would help) from a multi-beam / post-DOE
        # interference field (<<1, where it cannot -- F4 audit), so the two
        # branches point the user at the right fix.  Best-effort: a degenerate
        # field yields None / an exception, both silently skipping the warning.
        try:
            if _input_tilt is None:
                _input_tilt = _input_tilt_stats(E_in, wavelength, dx)
            if _input_tilt is not None:
                tilt_rms, coherence_ratio = _input_tilt
                if tilt_rms > 1e-4:
                    import warnings
                    if coherence_ratio >= 0.5:
                        warnings.warn(
                            "apply_real_lens_traced: tilt_aware_rays=False "
                            f"with a non-trivial single-beam input tilt "
                            f"(RMS = {tilt_rms:.2e} rad).  The plane-wave "
                            "reference OPD is off by an amount proportional "
                            "to (tilt * aperture); set tilt_aware_rays=True "
                            "for tilt-sensitive analyses.",
                            RuntimeWarning, stacklevel=3,
                        )
                    else:
                        warnings.warn(
                            "apply_real_lens_traced: tilt_aware_rays=False "
                            f"with a non-trivial input tilt of no single "
                            f"direction (RMS = {tilt_rms:.2e} rad, coherence "
                            f"{coherence_ratio:.2f}, i.e. INCOHERENT) -- a "
                            "divergent, multi-beam, or post-DOE interference "
                            "field.  Do NOT set tilt_aware_rays=True here "
                            "(per-pixel single-direction estimation fails on "
                            "such fields); pass carrier= (a conjugate, a "
                            "wavefront, or 'auto') to reference the beam's "
                            "congruence, or use apply_real_lens.",
                            RuntimeWarning, stacklevel=3,
                        )
        except (ValueError, RuntimeError, ZeroDivisionError, IndexError,
                AttributeError, TypeError):
            # tilt-RMS estimation is best-effort; suppressing the
            # warning when it can't be computed is preferable to
            # blowing up the traced-lens path.
            pass
        L_in = np.zeros_like(h_x)
        M_in = np.zeros_like(h_x)
    rays = _make_bundle(x=h_x, y=h_y, L=L_in, M=M_in,
                        wavelength=wavelength)
    # output_filter='last': only keep the image-plane bundle.  We do
    # not consume any intermediate per-surface state here, so saving
    # ray_history for all surfaces would allocate ~1 GB per surface
    # at N=32768 and ~250 MB per surface at N=4096 (for an
    # apply_real_lens_traced call at ray_subsample=8) for no benefit.
    result = trace(rays, surfaces, wavelength, output_filter='last')
    final = result.image_rays
    if not final.alive.any():
        raise RuntimeError(
            'apply_real_lens_traced: no rays survived the prescription; '
            'check aperture and clear-aperture settings.')

    # ---- EXIT-VERTEX CORRECTION ----------------------------------------
    # trace() leaves rays at the SAG of the last surface, i.e. at
    # z = sag(h) ≠ 0 for curved exit surfaces.  But the wave model's
    # exit field is defined at the flat exit VERTEX plane (z = 0).
    # Without this correction, the OPL comparison between on-axis
    # (z = 0) and off-axis (z = sag < 0 for concave) rays is made
    # at DIFFERENT z-planes, which introduces a systematic defocus
    # error equal to n_exit * sag(h) — enough to shift the implied
    # focal length by tens of percent for cemented doublets with
    # curved rear surfaces.
    #
    # Fix: propagate each ray from its current sag position to z = 0
    # in the exit medium, accumulating the remaining OPL and updating
    # the exit position to the vertex plane.
    #
    # IMPORTANT: use SIGNED t, not abs(t).  For concave rear surfaces
    # (sag < 0, z < 0) the ray must go forward (t > 0) → add OPL.
    # For convex rear surfaces (sag > 0, z > 0) the ray is AHEAD of
    # the vertex and must go backward (t < 0) → subtract OPL.
    # Using abs() forces the wrong sign for convex exits (e.g.
    # negative meniscus lenses), producing ~45x worse OPD.
    n_exit = get_glass_index(surfaces[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_to_vertex = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit * t_to_vertex
    final.x = final.x + final.L * t_to_vertex
    final.y = final.y + final.M * t_to_vertex
    final.z = np.zeros_like(final.z)

    # ---- v5.25.1 (hammer audit H6): carrier entrance eikonal -----------
    # The ray tracer accumulates OPL only from the ENTRANCE plane forward.
    # When a carrier congruence is set, each ray belongs to a wavefront
    # whose phase AT the entrance plane is k0*W(x_in) -- that eikonal must
    # be added so the traced exit wavefront is referenced to the beam's
    # own diverging/converging sphere, CONSISTENT with the
    # exp(i*k0*W) reference leg used by preserve_input_phase.  Omitting it
    # imprinted a spurious -k0*W on the field, cancelling the input
    # divergence the wave model correctly carried: every diverging-input
    # trace collapsed to the COLLIMATED focal plane f and the true image
    # at z_img smeared by NA_exit*(z_img - f) (production exp22: energy
    # over +/-1.8 mm, EE(100um) = 0.9% -- reproduced to the digit; with
    # this term EE(100um) = 0.999 across the R_in = 300/150/100 mm scan
    # and per-group relay chains, no change for collimated input).
    if _carrier_W_fn is not None:
        final.opd = final.opd + _carrier_W_fn(h_x, h_y)
    if _resid_eik is not None:
        # niche C6: the launched congruence's entrance eikonal is W + a_fit, so
        # the H6 term must carry BOTH halves.  Without ``a_fit`` here the OPL
        # grid would be ``W + V`` along ``grad(W + a_fit)`` rays, whose entrance
        # derivative is ``-grad a_fit + p_out . J`` -- i.e. it would carry a
        # copy of ``-a_fit`` for the low-order Chebyshev fit to represent, and
        # the ``+a`` the transported phasor adds back would no longer cancel it.
        # With it the grid's derivative is ``p_out . J`` exactly as on the
        # shipped launch, so the fit sees the same class of function it always
        # did.
        final.opd = final.opd + _resid_eik.value(h_x, h_y)

    # ---- v5.25.0 (hammer audit H3): exit-NA Nyquist guard --------------
    # The docstring's critical-sampling rule (dx <= lambda*f/aperture) was
    # documented but never ENFORCED, and violating it is silent: the exit
    # converging wavefront exceeds grid Nyquist (|sin theta| > lambda/2dx)
    # beyond some radius, the aliased annulus folds to WRONG positions,
    # and r^2-weighted far-halo metrics (r2m) read low while EE50/EE80
    # stay plausible -- measured on the dual-oracle f/5 singlet: r2m 40.9
    # vs 65.0 um at dx = 2.24x the limit, fully recovered (64.77, 99.7%)
    # at dx inside the limit.  Guard: the exact per-ray exit direction
    # cosines are already in hand; compare the beam's exit NA against the
    # grid Nyquist angle.  Amplitude-aware: only rays carrying input
    # amplitude >= e^-4 of peak count (a Gaussian's 99.97%-energy disc),
    # so zero-energy aperture-edge rays cannot over-fire the guard.
    # Policy: warn (RuntimeWarning) unless on_undersample == 'silent'.
    # Deliberately NOT an error even under on_undersample='error': the
    # returned field's core metrics remain valid; only far-halo moments
    # degrade -- erroring would break legitimate coarse-dx workflows.
    # niche D9: ``xs_in`` are ABSOLUTE launch heights (the launch grid stays
    # axis-centred) while these are grid indices, so the grid's own centre
    # position comes off first.  The ``np.clip`` makes a missing origin SILENT
    # -- every launch node would saturate to the same grid corner -- and this is
    # not diagnostic-only: ``_amp`` below feeds the MEASURED ``na_exit``, the
    # ``_exit_na_out`` sink the chain's ``on_tilt_exact_grid`` refusal reads
    # (default 'error'), and the C7 halo self-check's support radius.
    _ray_ix = np.clip(
        np.rint((xs_in - _org_x) / dx + E_in.shape[1] / 2).astype(int),
        0, E_in.shape[1] - 1)
    _dy_eff = dy if dy is not None else dx
    _ray_iy = np.clip(
        np.rint((xs_in - _org_y) / _dy_eff + E_in.shape[0] / 2).astype(int),
        0, E_in.shape[0] - 1)
    # niche C4 (2026-07-30): the ``.T`` is load-bearing.  The launch grid is
    # ``np.meshgrid(xs_in, xs_in, indexing='ij')`` -- x along axis 0 -- so a
    # ray's FLAT index is x-major, and that is the order ``final.L`` /
    # ``final.M`` / ``final.alive`` come back in.  ``np.ix_(_ray_iy, _ray_ix)``
    # builds a ``(y, x)`` block whose ``.ravel()`` is y-MAJOR, so every
    # amplitude was paired with the TRANSPOSED ray.  A rotationally symmetric
    # beam is invariant under that swap, which is why it survived; on an
    # asymmetric one the two readings exchange outright (measured on a biconic:
    # 0.0338 reported against 0.0684 true, and 0.0669 against 0.0340).  Design
    # 121's last group reported ``na_exit`` 0.3633 where the transpose-immune
    # value is 0.2912 -- 25 % overstated -- and ``_exit_na_out`` feeds the
    # chain's ``on_tilt_exact_grid`` routing, so this was not merely cosmetic.
    _amp = np.abs(E_in)[np.ix_(_ray_iy, _ray_ix)].T    # (x, y): x-major ravel
    _sig = (_amp >= np.exp(-4.0) * _amp.max()).ravel() & final.alive
    if _sig.any():
        _na_exit = float(np.sqrt(final.L[_sig] ** 2
                                 + final.M[_sig] ** 2).max())
        _dx_eff = max(dx, _dy_eff)
        # niche C1 item 4: report the MEASURED exit NA (and how much exit
        # power sits above this grid's Nyquist angle) to a caller who asked
        # for it.  ``_exit_na_out`` is private and pure-diagnostic -- nothing
        # here reads it back, so no default behaviour depends on it.  The
        # chain's tilted-leg guard (``on_tilt_exact_grid``) uses it because
        # its own ``na_exit`` is the CHAIN's paraxial ``w_in/|R_out|``, which
        # on design 121 reads 0.4053 against this measurement's 0.4780 -- so
        # the guard used to stay silent on a leg the element itself calls
        # under-sampled.  The power fraction is what makes the refusal
        # calibratable: the marginal ray is at the e^-4 AMPLITUDE contour
        # (r = 2w for a Gaussian), where the content carries ~3e-4 of the
        # power, so a bare NA comparison over-refuses.
        if _exit_na_out is not None:
            _na_all = np.sqrt(final.L ** 2 + final.M ** 2)
            _wgt = (_amp.ravel() ** 2) * final.alive
            _wtot = float(_wgt.sum())
            _na_ny = (wavelength / (2.0 * _dx_eff)) if _dx_eff > 0 else np.inf
            _exit_na_out.update({
                'na_exit': _na_exit,
                'dx': float(_dx_eff),
                'na_nyquist': float(_na_ny),
                'power_frac_above_nyquist': (
                    float(_wgt[_na_all > _na_ny].sum()) / _wtot
                    if _wtot > 0.0 else 0.0),
                'n_rays': int(final.alive.sum())})
        if _na_exit > 0 and _dx_eff > wavelength / (2.0 * _na_exit):
            _dx_need = wavelength / (2.0 * _na_exit)
            if on_undersample != 'silent':
                import warnings
                warnings.warn(
                    f'apply_real_lens_traced: the exit beam converges at '
                    f'NA_exit={_na_exit:.4f}, so the exit wavefront needs '
                    f'dx <= lambda/(2*NA_exit) = {_dx_need*1e6:.2f} um but '
                    f'the grid has dx = {_dx_eff*1e6:.2f} um.  The '
                    f'beyond-Nyquist annulus of the exit phase ALIASES: '
                    f'far-halo energy lands at wrong radii, so r^2-weighted '
                    f'spot metrics (r2m / second moments) read low while '
                    f'EE50/EE80 stay plausible.  Use a finer grid (dx <= '
                    f'{_dx_need*1e6:.2f} um) for halo-faithful results, or '
                    f'pass on_undersample="silent" to suppress.',
                    RuntimeWarning, stacklevel=2)

    # Reshape final.x, final.y, final.opd onto the regular ENTRANCE
    # grid.  Dead rays would break RectBivariateSpline (which requires
    # strictly regular data); vignetting is rare for normal lenses but
    # we guard against it by filling dead entries with NaN and
    # extrapolating with the spline's natural extrapolation (OK inside
    # the entrance disc of interest).
    x_out_grid = final.x.reshape(n_launch, n_launch)
    y_out_grid = final.y.reshape(n_launch, n_launch)
    opl_grid = final.opd.reshape(n_launch, n_launch)
    if not final.alive.all():
        alive_grid = final.alive.reshape(n_launch, n_launch)
        # Fill NaN into dead entries to make spline fitting fail
        # cleanly (rare path -- vignetted prescriptions)
        x_out_grid = np.where(alive_grid, x_out_grid, np.nan)
        y_out_grid = np.where(alive_grid, y_out_grid, np.nan)
        opl_grid = np.where(alive_grid, opl_grid, np.nan)

    # Reference OPL to on-axis (center of the entrance grid is an
    # exact sample because n_launch is odd).
    #
    # ---- FIX_TILT_QUADRATIC_OPL_2026_08_11: this is CONDITIONING, not the
    # ---- physics, so the removed constant is RE-APPLIED at assembly.
    #
    # The subtraction has to happen: ``opl_grid`` is an ABSOLUTE optical path
    # (metres -- 1e-2 for a design-121 group) whose interesting variation
    # across the beam is 1e-8..1e-9, so fitting the raw values with a
    # Chebyshev / spline would spend the whole double-precision mantissa on a
    # constant.  What was WRONG is that the constant was then dropped: every
    # branch below builds the exit phase from ``k0 * opl_map`` alone, so the
    # returned field's absolute phase was referenced to
    #
    #     Lam(0) = W(0, 0) + a_fit(0, 0) + P(0, 0)
    #
    # -- the entrance eikonal of the launched congruence at the LAUNCH-LATTICE
    # AXIS (the H6 / niche-C6 terms added to ``final.opd`` above) plus the
    # geometric path of the ray launched there.  On an UNTILTED, UNDECENTRED
    # congruence the axis IS the chief ray, so this only cost an unobservable
    # global phase -- which is why it survived.  Under a
    # :class:`TiltedCarrier` the axis is NOT the chief ray, and BOTH pieces
    # become functions of the tilt:
    #
    #     W(0, 0)  = -theta^2 * z_c + O(theta^4)        (z_c = the chief ray's
    #                                                    axial lever at entry)
    #     P(0, 0)  =  P_0 + theta^2 * T_g + O(theta^4)  (the axis ray's own
    #                                                    obliquity through the
    #                                                    group)
    #
    # so the element silently subtracted a pure TILT-QUADRATIC piston from the
    # chief ray of every tilted congruence.  Measured on design 121's first
    # post-DOE group (``validation/repro_traced_carrier_121/tqopl_mechanism.py``)
    # ``-k0 * [Lam_theta(0) - Lam_0(0)]`` reproduces the observed chief-ray
    # piston deficit to 1e-5 RELATIVE over a 66x span in tilt -- i.e. it is the
    # whole of the 4.8 %/group deficit
    # ``docs/audits/PROBE_CHAIN_LADDER_PISTON_2026_08_11.md`` S3.7 names, and
    # it does not converge with the grid, the ray lattice, or any fit lever
    # because it is a missing TERM.
    #
    # Precision of the re-application: ``k0 * Lam(0)`` is ~5e+04 rad for a
    # design-121 group, so the phasor carries ~1e-11 rad of round-off -- seven
    # decades under the lambda/100 = 6.3e-02 rad bar the consumers state, and
    # the SMALL ``k0 * opl_map`` keeps its own full precision because the two
    # are combined by a phasor MULTIPLY, never by adding the constant back into
    # the fitted map.
    #
    # A dead axis ray (NaN) already made ``opl_grid`` all-NaN and the returned
    # field identically zero before this change; the piston falls back to 0.0
    # there so the degenerate path keeps its pre-fix behaviour exactly.
    i_axis = n_launch // 2
    _opl_ref = opl_grid[i_axis, i_axis]
    opl_grid = opl_grid - _opl_ref       # UNCHANGED -- byte-identical
    _opl_piston = float(_opl_ref)
    if not np.isfinite(_opl_piston):
        _opl_piston = 0.0

    # ---- UNIT C (niche C14): the traced exit support, built ONCE ----------
    # Taken HERE, between the alive mask and the fit-domain restriction below,
    # for two reasons: the fit restriction NaNs out samples that are still
    # perfectly good optics (so reading it after would understate the hull and
    # over-fire the check), and this is the last point at which ``x_out_grid``
    # is the exact traced map rather than anything the model fitted.  Cost is
    # two reductions over the coarse lattice plus one hull.
    #
    # This ONE construction replaces the two blocks that used to stand here --
    # the v5.32 halo hull (C7) and the niche-C8 support bound -- which read the
    # same arrays, at the same point, through two separately-maintained copies
    # of the same finiteness mask and the same convex-hull algebra.  Each view
    # keeps its own rule and its own gate, so the work done and the bits
    # produced are unchanged; see ``_TracedExitSupport`` for why the rules are
    # deliberately NOT merged, and ``SUPPORT_BAND_CHECK`` for the blind spot
    # that having one object made statable.
    #
    # ``_amp`` is the input amplitude sampled at the launch nodes and is
    # already in the launch grid's (x, y) x-major layout -- the SAME layout
    # ``final.x.reshape(n_launch, n_launch)`` produced above (niche C4's
    # transpose note at ``_amp``).  Pairing it with the wrong one would weight
    # each ray by its transpose's amplitude, which is invisible on a
    # rotationally symmetric beam and wrong on every other.
    _exit_support = _TracedExitSupport.from_landings(
        x_out_grid, y_out_grid, _amp, xs_in, aperture, dx, sub,
        want_halo=bool(_ray_density and RAY_DENSITY_HALO_CHECK != 'silent'),
        want_bound=bool(REMAP_INVERSE_SUPPORT_BOUND and _ray_density))
    # ---- niche C15 (S6.5b, REFUTED -- kept as a note, not as code) --------
    # The obvious cure for the remaining ``newton_fit`` coupling was to build
    # the inverse-characteristic model from THESE arrays -- the pre-restriction
    # landings, the same ones ``from_landings`` is handed -- unweighted, so its
    # sample set is the traced rays and nothing the basis can reach.  It was
    # implemented and MEASURED, and the model's own G8 refused it: on design
    # 121's fixture the unweighted launch-square fit reads 4.5257e-01 waves of
    # held-out OPL error inside the beam against the restricted model's
    # 1.9965e-05, i.e. 4.4 decades worse and 2.3e+04x outside parity.  A
    # degree-14 exit-coordinate fit cannot span a launch square that reaches
    # ~5x past the beam; the fit-domain restriction is not an inconvenience it
    # inherits, it is load-bearing.  See BUILD_INVERSE_MAP_2026_08_11 S6.5b.
    # NOTE the two gates are the pre-C14 ones, unchanged and still separate.
    # In particular ``want_halo`` is NOT widened to cover the band check: the
    # band the C14 check watches is C8's RETAINED band, so it needs the HULL
    # (the ``want_bound`` view) and not the amplitude-weighted reporting
    # radius.  Widening ``want_halo`` would make ``_rd_hull_r`` non-None under
    # ``RAY_DENSITY_HALO_CHECK = 'silent'`` and the C7 warning would fire from
    # a policy that says it must not -- which is what
    # ``test_policy_silent_suppresses`` exists to catch.
    # The three pre-C14 locals, kept by name so every consumer below -- the
    # taper closure, ``_ray_density_amp_grid``'s capture, and the halo report
    # -- reads exactly what it read before.
    _rd_hull_r = _exit_support.radius
    _rd_hull_c = _exit_support.centroid
    _sup_bound = _exit_support.bound

    # R7 / audit F2 (2026-07-21): CARRIER-GATED fit-domain restriction.  When a
    # carrier is set, drop the entrance launch grid's outer margin + corners
    # (the strongly-aberrated / near-vignetting marginal rays) from the fit by
    # NaN-masking them here -- both the polynomial ``_Cheb2DEvaluator`` and the
    # direct-``fit`` inverse map skip NaN samples, so this restricts the fitted
    # region without touching the Newton loop.  Their out-of-basis high-order
    # aberration otherwise ALIASES into the global fit's low-order (defocus)
    # coefficients, the dominant per-group F2 error on strongly-focusing thick
    # groups.  Skipped for the ``spline`` fit (RectBivariateSpline needs a
    # regular NaN-free grid) and when a carrier is absent (byte-identical
    # default).  See ``_CARRIER_FIT_RADIUS_FRAC``.
    #
    # P2 (2026-07-25): ``fit_radius_beam_factor`` adds a BEAM-relative radius to
    # the same mechanism and lifts the carrier gate -- the plane-wave-reference
    # path (no engaged carrier) is exactly where the aperture:beam cliff lives,
    # since it fits over the whole launch square including its corners.  The two
    # radii combine by ``min``.  Still fit-domain-only: the Newton loop, the
    # ``bound`` and the out-of-domain NaN threshold are untouched, so nothing is
    # vignetted.  See ``_FIT_RADIUS_BEAM_FACTOR_DEFAULT``.
    #
    # D1 (2026-07-28): the two radii are discs about DIFFERENT points once the
    # beam is decentred -- the R7 one is aperture-relative (it drops the
    # near-vignetting marginal rays, a property of the aperture), the P2 one
    # follows the BEAM.  ``min`` of two radii is only their intersection when
    # they are concentric, so combine by ``min`` on axis (byte-identical) and
    # by INTERSECTION off axis.  If the intersection is too thin to constrain
    # the fit, the BEAM disc alone wins: keeping the guard active on the rays
    # the beam actually uses beats falling back to the whole launch square,
    # which is exactly the cliff regime.
    #
    # An OFF-CENTRE disc is also restricted DIFFERENTLY: by weights, not by the
    # hard NaN mask, because a hard mask leaves the fit's remaining freedom
    # unconstrained and the resulting map FOLDS -- a spurious bright lobe in
    # the returned field, not merely a fit that is loose where nothing lives.
    # See ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` for the mechanism + measurements.
    # ``_fit_weights is None`` (concentric disc) keeps the historical mask, so
    # the on-axis path is byte-identical.
    #
    # D7 (2026-07-29): an off-centre disc also needs MORE TERMS.  Its radius is
    # the same but it sits over aperture out to ``|c| + r``, where the map is
    # more aberrated, so a total-degree-6 fit is 14x worse there than the
    # concentric one (design 121 last group, measured).  ``_fit_poly_order``
    # raises the order on exactly the weighted branch below and nowhere else,
    # so the concentric path is byte-identical.  See
    # ``_DECENTRED_FIT_POLY_ORDER``.
    _fit_weights = None
    _fit_poly_order = int(newton_poly_order)
    # FIX FIT-DOMAIN SYMMETRY (2026-08-12): the inverse-map model's copy of the
    # SAME restriction, resolved here on either basis and consumed only at the
    # ``build_inverse_map`` call site.  ``None`` everywhere means "the forward
    # arrays already carry it" -- which is the polynomial path, so that path
    # is byte-identical by construction rather than by re-derivation.
    _imap_xo = _imap_yo = _imap_op = None
    _imap_weights = None
    # ``_fit_r_geom`` / ``_fit_r_max`` were resolved at the launch-radius site
    # above (niche C6's freeze circle has to clear them); the restriction they
    # drive is applied here, unchanged.
    if _fit_r_max is not None and _fit_domain_wanted:
        _r2_launch = xs_in[:, None] ** 2 + xs_in[None, :] ** 2
        _fit_why = f"r <= {_fit_r_max * 1e3:.4f} mm"
        _off_branch = (_beam_fit_radius is not None and _beam_decentred)
        if _off_branch:
            # launch grid is indexing='ij': axis 0 is X, axis 1 is Y (see the
            # ``np.meshgrid(..., indexing='ij')`` note at the launch site).
            _fit_disc = (((xs_in[:, None] - _bcx) ** 2
                          + (xs_in[None, :] - _bcy) ** 2)
                         <= _beam_fit_radius ** 2)
            _fit_why = (f"|r - ({_bcx * 1e3:.4f}, {_bcy * 1e3:.4f}) mm| <= "
                        f"{_beam_fit_radius * 1e3:.4f} mm")
            if _fit_r_geom is not None:
                _both = _fit_disc & (_r2_launch <= _fit_r_geom ** 2)
                if int(_both.sum()) >= _CARRIER_FIT_MIN_SAMPLES:
                    _fit_disc = _both
                    _fit_why += (f" AND r <= {_fit_r_geom * 1e3:.4f} mm")
        else:
            _fit_disc = _r2_launch <= _fit_r_max ** 2
        # niche C6 follow-up: the C6 launch augments every ray direction by
        # ``grad(a_fit)`` of a NON-RADIAL polynomial, which destroys the radial
        # symmetry the concentric hard-mask branch's safety argument rests on
        # (see ``_FIT_DISC_OUTSIDE_WEIGHT_REL``), so the D1 fold comes back on
        # axis.  Route a C6-engaged concentric disc through the SAME weighted
        # restriction the off-centre branch already uses.  The DISC itself is
        # unchanged (still ``_r2_launch <= _fit_r_max**2`` -- only the
        # restriction method and the order move), and with the C6 launch
        # disengaged this is exactly the historical path.  See
        # ``REMAP_STATIONARY_PHASE_FIT_GUARD`` for the measurements.
        _c6_fit_guard = (_resid_eik is not None
                         and REMAP_STATIONARY_PHASE_FIT_GUARD)
        # ---- niche C11: ARBITRATE the branch instead of predicting it ----
        # The rays are already traced, so both candidates can be built and
        # scored against the traced OPL before either is used.  Engaged only
        # ABOVE the C1 null gate (``_off_branch``), so every byte-identity
        # contract below that gate is untouched.  See
        # :data:`DECENTRED_FIT_ARBITER`.
        if ((DECENTRED_FIT_ARBITER or DECENTRED_FIT_PREDICTOR)
                and _off_branch
                and _fit_r_max_conc is not None
                and _w_in_beam > 0.0):
            _disc_c = _r2_launch <= _fit_r_max_conc ** 2
            if (int(_disc_c.sum()) >= _CARRIER_FIT_MIN_SAMPLES
                    and int(_fit_disc.sum()) >= _CARRIER_FIT_MIN_SAMPLES):
                _wgt = _decentred_fit_score_weight(
                    xs_in, _bcx, _bcy, _w_in_beam)
                _use_w = _FIT_DISC_OUTSIDE_WEIGHT_REL > 0.0
                _wo, _oo = _decentred_fit_restriction(
                    _fit_disc, _use_w, newton_poly_order, _dec_order)
                _wc, _oc = _decentred_fit_restriction(
                    _disc_c, _use_w and _c6_fit_guard, newton_poly_order,
                    _dec_order)
                _s_off = _decentred_fit_score(
                    xs_in, opl_grid, _wgt, _fit_disc, _wo, _oo)
                _s_conc = _decentred_fit_score(
                    xs_in, opl_grid, _wgt, _disc_c, _wc, _oc)
                # niche C11's verdict: smaller residual wins, an exact tie --
                # including two unscoreable candidates -- keeps the historical
                # branch.  Computed either way, because with the C12 predictor
                # engaged it is the runtime CHECK.
                _keep_conc = bool(_s_conc <= _s_off)
                _why_tag = 'C11 arbiter'
                # ---- niche C12: the PHYSICS PREDICTOR decides ---------------
                # ``u <= u*``, with ``u*`` the closed-form crossover of the
                # lens's own spectral tail under the disc-inflation law.  The
                # spectral half is used only where it is RESOLVED -- what the
                # order-``q`` box fit leaves over must be small against the
                # residuals it would have to rank.  See
                # :data:`DECENTRED_FIT_PREDICTOR`.
                if DECENTRED_FIT_PREDICTOR:
                    _u_dec = (float(_dec_mag / _w_in_beam)
                              if _w_in_beam > 0.0 else float('nan'))
                    _R_box = 0.5 * float(xs_in[-1] - xs_in[0])
                    _sigma = ((float(_beam_fit_radius) / _R_box)
                              if _R_box > 0.0 else 0.0)
                    _q = int(_DECENTRED_FIT_SPECTRUM_ORDER)
                    _S = None
                    _sp_resid = float('inf')
                    _tails = {}
                    if _q > int(_oo):
                        _S, _tails, _sp_resid = _decentred_fit_spectrum(
                            xs_in, opl_grid, _q, (int(_oc), int(_oo)),
                            weight=_wgt)
                    _m_eff = _decentred_fit_spectral_moment(
                        _S, int(_oc), _sigma)
                    # The spectral (model-only) pair, and whether it is usable.
                    # RESOLUTION TEST.  The surrogate differs from the traced
                    # map by whatever lies beyond degree ``q``, and the model's
                    # error is exactly what a degree-``oc`` fit over the
                    # concentric disc CANNOT absorb of that difference --
                    # ``(I - Pi)(W - W_q) == (I - Pi)(W - tails[oc])``, one
                    # more score of the same kind.  The model is used only
                    # while that gap is below the residuals it would rank; on
                    # design 121 it never is (S3.4 of the C12 audit), so the
                    # predictor falls back to the MEASURED pair there.
                    _m_conc, _m_off = _s_conc, _s_off
                    _resolved = False
                    if _tails:
                        _t_conc = _decentred_fit_score(
                            xs_in, _tails[int(_oc)], _wgt, _disc_c, _wc, _oc)
                        _t_off = _decentred_fit_score(
                            xs_in, _tails[int(_oo)], _wgt, _fit_disc, _wo, _oo)
                        _sp_gap = _decentred_fit_score(
                            xs_in, opl_grid - _tails[int(_oc)], _wgt,
                            _disc_c, _wc, _oc)
                        _resolved = bool(np.isfinite(_sp_gap)
                                         and np.isfinite(_sp_resid)
                                         and _sp_gap <= min(_t_conc, _t_off))
                        if _resolved:
                            _m_conc, _m_off = _t_conc, _t_off
                    _u_star = _decentred_fit_crossover(
                        _u_dec, _m_conc, _m_off, _m_eff)
                    if np.isnan(_u_star):
                        _pred_conc = _keep_conc      # no prediction available
                    else:
                        _pred_conc = bool(_u_dec <= _u_star)
                    _why_tag = (f"C12 predictor: u = {_u_dec:.4f} against "
                                f"u* = {_u_star:.4f} (m_eff {_m_eff:.2f}, "
                                f"spectrum "
                                f"{'resolved' if _resolved else 'UNRESOLVED'})"
                                f"; C11 arbiter")
                    if _pred_conc != _keep_conc:
                        import warnings
                        warnings.warn(
                            f"apply_real_lens_traced: the niche-C12 ray-fit "
                            f"PREDICTOR and the niche-C11 ARBITER disagree on "
                            f"this call.  The predictor selects the "
                            f"{'CONCENTRIC' if _pred_conc else 'OFF-CENTRE'} "
                            f"branch from |c|/w = {_u_dec:.4f} against a "
                            f"crossover u* = {_u_star:.4f} (spectral exponent "
                            f"m_eff = {_m_eff:.3f}, spectrum "
                            f"{'resolved' if _resolved else 'UNRESOLVED'} at "
                            f"order {_q}, box-fit residual {_sp_resid:.3e} m, "
                            f"modelled OPL residuals {_m_conc:.3e} m "
                            f"concentric / {_m_off:.3e} m off-centre); the "
                            f"arbiter's own measured residuals are "
                            f"{_s_conc:.3e} m concentric / {_s_off:.3e} m "
                            f"off-centre and select the "
                            f"{'CONCENTRIC' if _keep_conc else 'OFF-CENTRE'} "
                            f"one.  The PREDICTOR's choice is applied.  See "
                            f"DECENTRED_FIT_PREDICTOR; set it False to fall "
                            f"back to the arbiter, or also "
                            f"DECENTRED_FIT_ARBITER False for the v5.32 gate.",
                            RuntimeWarning, stacklevel=2)
                    _keep_conc = _pred_conc
                if _keep_conc:
                    # the historical disc reproduces the traced map better
                    # where the light is -- take it, and with it the radius
                    # and restriction that were scored
                    _off_branch = False
                    _fit_disc = _disc_c
                    _fit_r_max = _fit_r_max_conc
                    _fit_why = (f"r <= {_fit_r_max_conc * 1e3:.4f} mm "
                                f"({_why_tag}: OPL residual "
                                f"{_s_conc:.3e} m concentric against "
                                f"{_s_off:.3e} m off-centre)")
                else:
                    _fit_why += (f" ({_why_tag}: OPL residual {_s_off:.3e} m "
                                 f"off-centre against {_s_conc:.3e} m "
                                 f"concentric)")
        if int(_fit_disc.sum()) >= _CARRIER_FIT_MIN_SAMPLES:
            # The restriction is built ONCE and then routed on
            # ``_fit_domain_basis_ok``: to the forward arrays when this basis
            # can honour it (byte-identical to the shipped path), and
            # otherwise to the inverse-map model's own copies.  Reaching here
            # at all means SOME consumer asked for it (``_fit_domain_wanted``),
            # so exactly one of the two destinations is taken and the domain
            # is never resolved and then dropped.
            if (_beam_fit_radius is not None
                    and (_off_branch or _c6_fit_guard)
                    and _FIT_DISC_OUTSIDE_WEIGHT_REL > 0.0):
                # D1: OFF-CENTRE disc (or, opt-in, a C6-engaged concentric one)
                # -- regularised restriction.  Keep every traced
                # sample (so the paraxial-magnification stencil, the
                # process-pool knot data and the direct-fit exit hull all stay
                # intact) and down-weight the out-of-disc ones to a fixed
                # fraction of the in-disc Gram contribution; and (D7) give that
                # fit the terms its region needs, with the sample-count
                # step-down that keeps it determined.  Both live in
                # ``_decentred_fit_restriction`` so the niche-C11 arbiter above
                # scores exactly what is applied here.
                _w_res, _o_res = _decentred_fit_restriction(
                    _fit_disc, True, newton_poly_order, _dec_order)
                if _fit_domain_basis_ok:
                    _fit_weights, _fit_poly_order = _w_res, _o_res
                else:
                    # the model's copy; the D7 ORDER raise is a property of the
                    # element's own degree-``newton_poly_order`` fit and has no
                    # meaning for a degree-14 exit model, so only the WEIGHTS
                    # travel.
                    _imap_weights = _w_res
            elif _fit_domain_basis_ok:
                x_out_grid = np.where(_fit_disc, x_out_grid, np.nan)
                y_out_grid = np.where(_fit_disc, y_out_grid, np.nan)
                opl_grid = np.where(_fit_disc, opl_grid, np.nan)
            else:
                _imap_xo = np.where(_fit_disc, x_out_grid, np.nan)
                _imap_yo = np.where(_fit_disc, y_out_grid, np.nan)
                _imap_op = np.where(_fit_disc, opl_grid, np.nan)
        elif _beam_fit_radius is not None and on_aperture_beam == 'warn':
            import warnings
            warnings.warn(
                f"apply_real_lens_traced: fit_radius_beam_factor would "
                f"restrict the ray-fit domain to "
                f"{_fit_why}, which holds only "
                f"{int(_fit_disc.sum())} coarse ray samples (< "
                f"{_CARRIER_FIT_MIN_SAMPLES} needed to constrain the "
                f"order-{int(newton_poly_order)} fit).  The restriction is "
                f"ABANDONED and the full launch domain is fitted, so the "
                f"aperture:beam cliff guard is NOT active on this call.  "
                f"Lower ray_subsample (currently {int(ray_subsample)}) or "
                f"raise fit_radius_beam_factor"
                + (f" (the beam is {np.hypot(_bcx, _bcy) * 1e3:.4f} mm off "
                   f"the grid centre, so the disc that must hold it sits off "
                   f"axis too)" if _off_branch else "") + ".",
                RuntimeWarning, stacklevel=2)

    # T-P2 (audit perf): optional DIRECT inverse-map fit.  Instead of Newton-
    # inverting the forward map per output pixel, fit ``opl`` as a smooth
    # function of the EXIT coordinates ``(x_out, y_out)`` by scattered
    # Chebyshev least squares from the already-traced ray samples, then
    # evaluate that polynomial on the exit grid -- one lstsq + one poly eval,
    # no per-pixel Newton.  A GLOBAL Chebyshev fit (vs the pre-3.x griddata
    # scatter this file replaced) avoids the Delaunay-edge spikes noted below,
    # while staying opt-in (``inversion_method='fit'``) so the thoroughly-
    # validated Newton path remains the default.  Output convention is
    # identical: on-axis-referenced OPL in metres, NaN outside the exit
    # sample hull.
    _use_fit = (inversion_method == 'fit')
    if _use_fit:
        from numpy.polynomial.chebyshev import chebvander as _chebvander
        # D7: the direct inverse-map fit reads the SAME off-centre samples
        # through the SAME weights, so it takes the SAME raised order.
        _fo = int(_fit_poly_order)
        _xo_s = x_out_grid.ravel()
        _yo_s = y_out_grid.ravel()
        _op_s = opl_grid.ravel()
        _g = np.isfinite(_xo_s) & np.isfinite(_yo_s) & np.isfinite(_op_s)
        # D1: an OFF-CENTRE fit disc is expressed as weights, not as a NaN
        # mask, so the restriction reaches this path through ``_fit_weights``
        # (None -> unweighted, byte-identical).
        _wfit_s = None if _fit_weights is None else _fit_weights.ravel()[_g]
        _xo_s, _yo_s, _op_s = _xo_s[_g], _yo_s[_g], _op_s[_g]
        _fx_c = 0.5 * (_xo_s.max() + _xo_s.min())
        _fx_h = 0.5 * (_xo_s.max() - _xo_s.min()) or 1.0
        _fy_c = 0.5 * (_yo_s.max() + _yo_s.min())
        _fy_h = 0.5 * (_yo_s.max() - _yo_s.min()) or 1.0
        # total-degree multi-index list, encoded as (P, 2) int for a
        # vectorized column-product Chebyshev design.
        _terms = np.array([[a, b] for a in range(_fo + 1)
                           for b in range(_fo + 1 - a)], dtype=np.intp)

        def _fit_design(ux, uy):
            Vx = _chebvander(ux, _fo)   # (K, _fo+1); col a = T_a(ux)
            Vy = _chebvander(uy, _fo)
            return Vx[:, _terms[:, 0]] * Vy[:, _terms[:, 1]]   # (K, M)

        _Afit = _fit_design((_xo_s - _fx_c) / _fx_h, (_yo_s - _fy_c) / _fy_h)
        # B7: normal-equations solve (thread-safe; no gelsd/JAX-OpenMP deadlock).
        if _wfit_s is None:
            _fit_coef = _solve_lstsq_thread_safe(_Afit, _op_s)
        else:
            _fit_coef = _solve_lstsq_thread_safe(_Afit * _wfit_s[:, None],
                                                 _op_s * _wfit_s)
        # Domain: keep only exit pixels inside the convex hull of the ray
        # landing spots -- a vectorized half-plane test (A.x + b <= 0 for
        # every facet), far cheaper than a Delaunay simplex search over the
        # full output grid.  A lens exit region is convex (a disc), so the
        # hull is the exact coverage boundary.
        #
        # UNIT C (niche C14): this is the THIRD notion of the traced exit
        # support, and the C8 audit's own sentence is that it is the same idea
        # as the second -- "This bound gives the Newton path the containment
        # the direct-fit path has had all along."  It now shares C8's hull
        # BUILDER and its signed-distance RULE, so the two cannot drift in
        # their conventions.  What it does NOT share is the POINT SET: this
        # path hulls the post-restriction samples ``(_xo_s, _yo_s)``, which is
        # its documented, long-standing behaviour and is deliberately
        # unchanged -- unifying the point sets would be a behaviour change, not
        # a refactor.  ``strict=True`` keeps the historical "a fit with no
        # domain raises" contract, with the original qhull exception.
        _hA, _hb = _TracedExitSupport.half_planes(_xo_s, _yo_s, strict=True)

        def _invert_fit(Xw, Yw):
            _sh = np.asarray(Xw).shape
            xw = np.asarray(Xw).ravel()
            yw = np.asarray(Yw).ravel()
            val = _fit_design((xw - _fx_c) / _fx_h,
                              (yw - _fy_c) / _fy_h) @ _fit_coef
            inside = _TracedExitSupport.signed_distance(
                _hA, _hb, xw, yw) <= 1e-12
            return np.where(inside, val, np.nan).reshape(_sh)

    # ----- OPTION B: RectBivariateSpline + Newton-inversion of the
    # entrance->exit mapping ------------------------------------------
    #
    # Because the rays were launched on a regular (xs_in, xs_in) grid,
    # final.x, final.y, final.opl are regular-grid functions of the
    # entrance position.  We build three 2-D splines:
    #
    #     Sx(xe, ye) = x_out at entrance (xe, ye)
    #     Sy(xe, ye) = y_out at entrance (xe, ye)
    #     So(xe, ye) = OPL   at entrance (xe, ye)
    #
    # For each wave-grid exit pixel (Xw, Yw) we find the entrance
    # (xe, ye) that lands there via Newton iteration on the residual
    # r = (Sx(xe,ye) - Xw, Sy(xe,ye) - Yw) = 0.  Then OPL at that
    # wave pixel = So(xe, ye).
    #
    # Advantages over the previous scatter-to-grid (griddata) path:
    #   * C^2 smooth interpolation (no Delaunay-edge spikes).
    #   * RectBivariateSpline.ev() is implemented in Fortran and DOES
    #     release the GIL, so we CAN multi-thread the Newton loop.
    #   * Works correctly even for fast lenses with caustic-like
    #     behaviour near the exit-pupil edge (the mapping is still
    #     single-valued on the entrance grid; inversion is stable).
    # ---- Validate use_gpu combination ---------------------------------
    _newton_xp = np  # default Newton array backend
    if use_gpu:
        if newton_fit != 'polynomial':
            raise ValueError(
                f"use_gpu=True requires newton_fit='polynomial'; "
                f"got newton_fit={newton_fit!r}.  The spline path uses "
                f"SciPy RectBivariateSpline which has no GPU backend.")
        if not CUPY_AVAILABLE:
            raise ImportError(
                "use_gpu=True requires the 'cupy' package.  Install with "
                "'pip install cupy-cuda12x' (NVIDIA, matching your CUDA "
                "version) or 'pip install cupy-rocm-6-1' (AMD ROCm); or set "
                "use_gpu=False to stay on the CPU path.")
        _newton_xp = cp

    if newton_fit == 'polynomial':
        # 2-D Chebyshev tensor-product fit -- closed-form evaluation and
        # analytic derivatives, better accuracy than bicubic spline on
        # smooth refractive-lens data.  Same .ev(...) API so the
        # Newton loop below is untouched.
        #
        # When use_gpu=True, build the evaluator on GPU (all arrays
        # pushed to device via cp.asarray).  The Newton loop below
        # auto-detects the evaluator backend and runs on the matching
        # device.
        _xp = _newton_xp
        _xs_xp = _xp.asarray(xs_in)
        _xout_xp = _xp.asarray(x_out_grid)
        _yout_xp = _xp.asarray(y_out_grid)
        _opl_xp = _xp.asarray(opl_grid)
        Sx = _Cheb2DEvaluator(_xs_xp, _xs_xp, _xout_xp,
                               order=_fit_poly_order, xp=_xp,
                               weights=_fit_weights)
        Sy = _Cheb2DEvaluator(_xs_xp, _xs_xp, _yout_xp,
                               order=_fit_poly_order, xp=_xp,
                               weights=_fit_weights)
        So = _Cheb2DEvaluator(_xs_xp, _xs_xp, _opl_xp,
                               order=_fit_poly_order, xp=_xp,
                               weights=_fit_weights)
    elif newton_fit == 'spline':
        try:
            from scipy.interpolate import RectBivariateSpline
        except ImportError:
            raise ImportError(
                'apply_real_lens_traced requires SciPy for spline '
                'interpolation.')
        Sx = RectBivariateSpline(xs_in, xs_in, x_out_grid, kx=3, ky=3)
        Sy = RectBivariateSpline(xs_in, xs_in, y_out_grid, kx=3, ky=3)
        So = RectBivariateSpline(xs_in, xs_in, opl_grid, kx=3, ky=3)
    else:
        raise ValueError(
            f"newton_fit must be 'spline' or 'polynomial', "
            f"got {newton_fit!r}")

    # N12 (P11): the forward-map fits Sx/Sy expose a combined value+gradient on
    # the polynomial path (``_Cheb2DEvaluator.ev_value_and_grad``); the spline
    # path uses the ``.ev(dx=1)`` / ``.ev(dy=1)`` API.  The ray-density
    # amplitude closure needs the Jacobian d(x_out,y_out)/d(x_in,y_in), so it
    # dispatches on this flag.
    _has_combined_fits = (hasattr(Sx, 'ev_value_and_grad')
                          and hasattr(Sy, 'ev_value_and_grad'))

    # ---- Paraxial magnification from the already-computed forward
    # trace.  Used as the Newton initial guess: (xe, ye) ~ (Xw, Yw) / M.
    #
    # We read the central finite-difference slope of the forward map:
    #     M_x = [x_out(i_c+1, i_c) - x_out(i_c-1, i_c)] / (2 * d_xs_in)
    #     M_y = [y_out(i_c, i_c+1) - y_out(i_c, i_c-1)] / (2 * d_xs_in)
    # where (i_c, i_c) is the on-axis entrance grid point (exact sample
    # because n_launch is odd).  4.11.2: the indices match the meshgrid
    # at the launch step
    #     ``Xs_in, Ys_in = np.meshgrid(xs_in, xs_in, indexing='ij')``
    # which puts x along axis 0 and y along axis 1, so ∂x_out/∂x_in
    # varies the FIRST index, not the second.  Pre-4.11.2 the indices
    # were swapped, computing ∂x_out/∂y_in (~zero by rotational
    # symmetry) instead of ∂x_out/∂x_in.  Newton still converged
    # because the polynomial Jacobian is right, but every pixel started
    # at the clipped-to-boundary initial guess (0.91-fallback) instead
    # of the actual paraxial slope.
    #
    # This stencil is strictly better than the previous hard-coded 1.10
    # multiplier: the old heuristic assumed M ~ 0.91 (converging system
    # "shrinks 10%") which is approximately right for singlets at their
    # exit vertex (M ~ 1) but wildly off for compound systems with real
    # imaging magnification (TX Design 36 full-system inversion would
    # have M = 0.25; using 1.10 as the guess puts Newton 4x from the
    # answer and costs several extra iterations per pixel).  Zero
    # additional compute -- the grid values are already in memory from
    # the forward trace above.
    i_c = n_launch // 2
    d_xs = float(xs_in[1] - xs_in[0])
    try:
        M_x = (float(x_out_grid[i_c + 1, i_c])
               - float(x_out_grid[i_c - 1, i_c])) / (2.0 * d_xs)
        M_y = (float(y_out_grid[i_c, i_c + 1])
               - float(y_out_grid[i_c, i_c - 1])) / (2.0 * d_xs)
    except (IndexError, ValueError):
        M_x = M_y = 0.91  # fallback to pre-3.1.3 heuristic (1/1.10)
    # Guard against NaNs from dead rays at the center (unlikely -- the
    # axial ray always survives in a well-posed prescription) and
    # against extreme values that would blow up the initial guess.
    if not (np.isfinite(M_x) and np.isfinite(M_y)):
        M_x = M_y = 0.91
    M_x = float(np.clip(abs(M_x), 1e-3, 1e3))
    M_y = float(np.clip(abs(M_y), 1e-3, 1e3))

    # Store spline knot data for potential process-pool pickling.
    # Include the inverse magnification so the process-pool path (which
    # rebuilds splines inside each worker) can seed Newton identically.
    _spline_data = {
        'xs_in': xs_in,
        'x_out_grid': x_out_grid,
        'y_out_grid': y_out_grid,
        'opl_grid': opl_grid,
        'launch_radius': launch_radius,
        'dx': dx,
        'bound': launch_radius * 0.999,
        'inv_M_x': 1.0 / M_x,
        'inv_M_y': 1.0 / M_y,
        # The worker rebuilds whichever fit the caller chose (both are
        # supported since v5.30.1), so it needs the polynomial parameters too.
        'newton_fit': newton_fit,
        'fit_poly_order': _fit_poly_order,
        'fit_weights': _fit_weights,
    }

    # Bound for the clipped Newton update (stay inside fitted domain)
    bound = launch_radius * 0.999

    # Newton iter cap: caller override > module default.  See the note
    # at _NEWTON_MAX_ITERS for the 8-vs-12 trade-off.
    MAX_NEWTON_ITERS = (int(newton_max_iters) if newton_max_iters is not None
                        else _NEWTON_MAX_ITERS)
    # v5.29.1 (audit E-H2): carry the RESOLVED cap into the pickled worker
    # payload.  ``_newton_invert_chunk`` used to hard-code
    # ``_NEWTON_MAX_ITERS``, so ``newton_max_iters`` was inert whenever the
    # process pool engaged (then >=200k points, newton_fit='spline', CPU;
    # now either fit, above the two-tier cold/warm gate) -- and
    # the ray-density amplitude leg, which always runs the SERIAL closure,
    # could then be built from a different Newton solution than the OPL.
    _spline_data['newton_max_iters'] = int(MAX_NEWTON_ITERS)

    def _warn_newton_unconverged(n_unconverged, n_total, tol):
        """Emit the Newton-unconverged RuntimeWarning (shared by the serial
        and process-pool inversion paths so both report identically).

        Healthy prescriptions can have a handful of out-of-domain edge pixels
        left active at the iteration cap -- those are benign and don't warrant
        a warning.  Threshold: >1% of total pixels unconverged means a real
        convergence problem.  Honours the same ``on_undersample`` knob the
        rest of the function uses ('silent' suppresses).  Pre-3.5.6
        unconverged pixels were silently kept at their last Newton value; the
        POOL path stayed silent until v5.29.1 (audit E-H2) even though the
        message's own advice is "increase newton_max_iters".
        """
        n_unconverged = int(n_unconverged)
        n_total = max(int(n_total), 1)
        if not (n_unconverged > 0 and n_unconverged > 0.01 * n_total
                and on_undersample != 'silent'):
            return
        import warnings as _warnings
        _warnings.warn(
            f"apply_real_lens_traced Newton inversion: "
            f"{n_unconverged}/{n_total} pixels "
            f"({100.0*n_unconverged/n_total:.1f}%) did not converge "
            f"to tol={tol:.3e} m within {MAX_NEWTON_ITERS} "
            f"iterations.  Affected pixels keep their last Newton "
            f"value, which may carry residual error.  Increase "
            f"newton_max_iters if this matters for your tolerance "
            f"budget.",
            RuntimeWarning, stacklevel=3)

    def _invert_newton(Xw, Yw, sub_progress=None, _want_entrance=False):
        """Run Newton iteration to find (xe, ye) such that (Sx, Sy)
        evaluated at (xe, ye) equals (Xw, Yw).  Returns OPL at the
        converged entrance positions plus a validity mask.

        Fully vectorised over the input arrays -- ``Xw`` and ``Yw``
        may be any shape; result has the same shape.

        ``sub_progress`` is an optional ``ProgressScaler`` (or any
        callable ``f(frac, msg)``) driven once per Newton iteration.

        ``_want_entrance`` (N12/P11, internal): when True, ALSO return the
        converged entrance coordinates ``(xe, ye)`` (same shape as ``Xw``) as a
        3-tuple ``(opl, xe, ye)`` so the ray-density amplitude closure can
        evaluate ``det J`` and ``|E_in|`` at the entrance point.  Default False
        keeps the historical single-array return byte-identical.
        """
        # Detect Newton-loop array backend from the evaluator.  The
        # evaluator's xp is either numpy (CPU) or cupy (GPU when
        # use_gpu=True was set earlier).  Using xp uniformly inside
        # the Newton loop keeps this code device-agnostic -- the only
        # other GPU plumbing needed is pushing xe/ye/active/idx_active
        # to xp and pulling opl_flat back to numpy at the end.
        xp = getattr(Sx, 'xp', np)
        # Push wave-grid coordinates to the Newton backend.  On the
        # CPU path this is a zero-cost view; on GPU it's a H->D copy
        # of order (N_wave^2) floats, incurred once per Newton call.
        x_w_flat = xp.asarray(Xw.ravel())
        y_w_flat = xp.asarray(Yw.ravel())
        n_total = int(x_w_flat.size)
        # Initial guess: entrance ~ exit / M, where M is the paraxial
        # magnification measured from the central finite-difference slope
        # of the forward map (see `inv_M_x` / `inv_M_y` computed above from
        # the already-traced ray grid -- no extra compute).  This is a
        # strictly better guess than the pre-3.1.3 hard-coded 1.10
        # multiplier: for singlets with M ~ 1 the two are nearly identical,
        # but for compound systems or unusual magnifications the measured
        # value avoids putting Newton several iterations away from
        # convergence.
        xe = x_w_flat * _spline_data['inv_M_x']
        ye = y_w_flat * _spline_data['inv_M_y']
        tol = 0.01 * dx
        active = xp.ones(xe.size, dtype=bool)  # pixels still iterating
        if sub_progress is not None:
            sub_progress(0.0, f'newton 0/{MAX_NEWTON_ITERS}: {n_total} pixels')
        # When the fit objects support combined value+gradient
        # (polynomial path via _Cheb2DEvaluator), use it to halve the
        # number of Newton-hot-path evaluator calls per iteration from
        # 6 down to 2, and share Chebyshev basis work across f/fx/fy.
        _has_combined = (hasattr(Sx, 'ev_value_and_grad')
                          and hasattr(Sy, 'ev_value_and_grad'))
        for _it in range(MAX_NEWTON_ITERS):
            if not bool(active.any()):
                if sub_progress is not None:
                    sub_progress(1.0,
                                 f'newton converged after {_it} iters')
                # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration
                # telemetry): emit a "converged" marker so an attached
                # handler sees the early-exit path.
                logger.info(
                    "apply_real_lens_traced: newton iter %d/%d converged "
                    "(all %d pixels)",
                    int(_it), int(MAX_NEWTON_ITERS), int(n_total))
                break
            # Only evaluate splines at active (unconverged) pixels
            xa = xe[active]
            ya = ye[active]
            xw = x_w_flat[active]
            yw = y_w_flat[active]
            if _has_combined:
                fx_val, jxx, jxy = Sx.ev_value_and_grad(xa, ya)
                fy_val, jyx, jyy = Sy.ev_value_and_grad(xa, ya)
                rx = fx_val - xw
                ry = fy_val - yw
            else:
                rx = Sx.ev(xa, ya) - xw
                ry = Sy.ev(xa, ya) - yw
                jxx = Sx.ev(xa, ya, dx=1)
                jxy = Sx.ev(xa, ya, dy=1)
                jyx = Sy.ev(xa, ya, dx=1)
                jyy = Sy.ev(xa, ya, dy=1)
            det = jxx * jyy - jxy * jyx
            safe = xp.abs(det) > 1e-12
            inv_det = xp.where(safe, 1.0 / det, 0.0)
            dxe = (jyy * rx - jxy * ry) * inv_det
            dye = (-jyx * rx + jxx * ry) * inv_det
            xa_new = xp.clip(xa - dxe, -bound, bound)
            ya_new = xp.clip(ya - dye, -bound, bound)
            xe[active] = xa_new
            ye[active] = ya_new
            # Mark converged pixels as inactive
            res = xp.sqrt(rx * rx + ry * ry)
            converged = res < tol
            idx_active = xp.where(active)[0]
            active[idx_active[converged]] = False
            if sub_progress is not None:
                remaining = int(active.sum())
                pct_done = 1.0 - remaining / max(n_total, 1)
                # Emit max(iteration-based, convergence-based) fraction,
                # bounded to <1 so the final "assembling" tick owns 1.0.
                frac = min(max((_it + 1) / MAX_NEWTON_ITERS, pct_done),
                           0.99)
                sub_progress(
                    frac,
                    f'newton {_it + 1}/{MAX_NEWTON_ITERS}: '
                    f'{remaining}/{n_total} pixels unconverged')
            # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration
            # telemetry): per-Newton-iteration log, independent of the
            # sub_progress callback (sub_progress is None on the
            # serial / single-call code paths).  Reports current OPD
            # residual norm + remaining-active-pixel count so an
            # attached handler can track convergence.
            _remaining_log = int(active.sum())
            # v5.4 (audit P3): deduplicate -- reuse res from convergence check above
            try:
                _res_norm = float(res.max()) if res.size > 0 else 0.0
            except (ValueError, TypeError):
                _res_norm = float('nan')
            logger.info(
                "apply_real_lens_traced: newton iter %d/%d "
                "residual_max=%.3e m remaining=%d/%d",
                int(_it + 1), int(MAX_NEWTON_ITERS),
                _res_norm, _remaining_log, int(n_total))
        # Surface unconverged pixels (shared emitter -- see
        # ``_warn_newton_unconverged``; the pool path calls the same helper).
        n_unconverged = int(active.sum()) if hasattr(
            active, 'sum') else 0
        n_total = int(active.size) if hasattr(active, 'size') else 1
        _warn_newton_unconverged(n_unconverged, n_total, tol)
        opl_flat = So.ev(xe, ye)
        out_of_domain = (xe * xe + ye * ye > (launch_radius * 0.99) ** 2)
        opl_flat = xp.where(out_of_domain, xp.nan, opl_flat)
        # If we ran on GPU, pull the result back to the host so the
        # rest of apply_real_lens_traced -- which is CPU-only
        # (amplitude from apply_real_lens, final field assembly) --
        # sees a NumPy array.
        if xp is not np:
            opl_flat = cp.asnumpy(opl_flat)
            if _want_entrance:
                xe = cp.asnumpy(xe)
                ye = cp.asnumpy(ye)
        if _want_entrance:
            # N12 (P11): the ray-density amplitude closure needs the converged
            # entrance coordinates (to evaluate det J and |E_in| there).
            return (opl_flat.reshape(Xw.shape),
                    np.asarray(xe).reshape(Xw.shape),
                    np.asarray(ye).reshape(Xw.shape))
        return opl_flat.reshape(Xw.shape)

    # ----- Coarse-grid Newton + interpolation --------------------------
    # The OPL map is extremely smooth (well-approximated by a
    # low-order polynomial), so evaluating the expensive Newton
    # inversion at every wave-grid pixel is wasteful.  Instead we
    # evaluate on a COARSER output grid and bilinearly interpolate to
    # the full wave grid.  ``ray_subsample`` controls the output
    # sub-sampling factor:
    #
    #   ray_subsample=1  -> Newton at every pixel (exact, slow)
    #   ray_subsample=4  -> Newton at every 4th pixel, interp rest
    #   ray_subsample=8  -> Newton at every 8th pixel (fastest)
    #
    # Parallelism: Newton is embarrassingly parallel (per-pixel
    # independent, immutable splines).  We dispatch to a process pool
    # when the grid is large enough that pool startup + knot-pickling
    # is worth it.  Threads don't help here: scipy's
    # ``RectBivariateSpline.ev`` does not release the GIL in current
    # versions, so threading delivers no speedup.

    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool

    from ..memory import available_cpus

    # Affinity-aware: respect cgroup limits, taskset masks, Python 3.13+
    # process_cpu_count so we don't oversubscribe a restricted machine.
    # If the user pinned half the cores via taskset (or the container
    # has a CPU quota) we'll see the restricted count here, whereas
    # os.cpu_count() would still return the raw logical total.
    _n_cpu_req = n_workers if n_workers is not None else available_cpus()
    _n_cpu_req = max(1, int(_n_cpu_req))
    # Points in the ray-fit grid each worker re-fits from the pickled payload
    # (``n_launch**2``); one of the two terms the per-worker memory model
    # prices.  See ``_newton_worker_bytes``.
    _fit_points = int(np.size(_spline_data['opl_grid']))


    def _invert_newton_parallel(Xw, Yw, sub_progress=None):
        """Dispatch ``_invert_newton`` work across a process pool when
        useful; fall back to the in-process serial path otherwise.

        Preserves the serial path's numerical behaviour exactly (same
        Newton iteration count -- the resolved ``newton_max_iters`` travels
        in ``_spline_data`` since v5.29.1 / audit E-H2 -- same convergence
        tolerance, same out-of-domain NaN policy, and the same
        unconverged-pixel warning; see :func:`_newton_invert_chunk`).
        """
        global _POOL_RESIDENT_PAYLOAD_KEY
        # GPU path must stay in-process: the worker function
        # ``_newton_invert_chunk`` rebuilds SciPy splines per worker
        # (CPU-only), and shipping CuPy device arrays through a
        # ProcessPoolExecutor would host-copy them anyway.  Go direct.
        if use_gpu:
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
        x_w_flat = Xw.ravel()
        y_w_flat = Yw.ravel()
        n_total = x_w_flat.size
        # RESOURCE CEILING, resolved before the COST gate below and composing
        # with it: this bounds how many workers may ever run (measured
        # per-worker commit vs available RAM, plus the unguarded-__main__
        # refusal), the cost gate then decides whether to dispatch at all.
        # Upstream on purpose -- the promotion evidence is keyed by worker
        # count, so it has to be keyed by the count we would actually use.
        n_cpu = _newton_resolve_workers(_n_cpu_req, n_total, _fit_points,
                                        min_pool_points=_POOL_MIN_PIXELS_WARM,
                                        on_pool_memory=on_pool_memory)
        # A worker in THIS process has already refused a chunk because it could
        # not provide the pinned Chebyshev backend (v5.32.3).  That is a
        # property of the workers, not of one call, so stop asking: the serial
        # path is bit-identical and does not pay a round trip to be told again.
        if _POOL_BACKEND_REFUSED:
            n_cpu = 1
        # A pool that is already up has no spawn left to amortise -- see the
        # measured cold/warm tables at _POOL_MIN_PIXELS.
        _pool_is_warm = (_PERSISTENT_POOL is not None
                         and _PERSISTENT_POOL_NWORKERS == n_cpu)
        # ...and a process that has ALREADY deferred a pool-sized inversion,
        # and MEASURED it to be slow enough to be worth pooling, is a chain or
        # a sweep whose remaining calls will repay the spawn.  Without this
        # second arm the warm bar is unreachable from a cold process (review
        # D1): ``_pool_is_warm`` can only become true downstream of this very
        # gate, so a 6-group chain at 65 536 points/group ran every group
        # serial.  Without the MEASUREMENT inside it, fixing that reachability
        # made the same chain 5-7% SLOWER, because at 65 536 points the
        # default fit's numba prange kernel beats an 8-worker dispatch.  See
        # the block at ``_POOL_PROMOTE_MIN_SECONDS``.
        #
        # The measurement is asked for, and recorded against, this call's own
        # COST CLASS -- (worker count, fit backend, point-count band).  A bare
        # wall time keyed on the worker count alone let two expensive SPLINE
        # inversions promote four CHEAP polynomial ones at the same size, and
        # let two 116k inversions promote 16k ones: both re-admit the 5-7%
        # regression above.  See ``_POOL_PROMOTE_SIZE_RATIO`` (finding V5).
        _cost_class = _newton_cost_class(newton_fit)
        _pool_promoted = _pool_reuse_is_likely(n_cpu, _cost_class, n_total)
        _min_px = (_POOL_MIN_PIXELS_WARM if (_pool_is_warm or _pool_promoted)
                   else _POOL_MIN_PIXELS)
        if n_cpu <= 1 or n_total < _min_px:
            # Serial for this call.  If a LIVE pool would have served it, TIME
            # it and remember the deferral, so the next one can be promoted on
            # evidence.  Guarded on the warm bar, not on ``_min_px``: sub-8k
            # calls never justify a pool at any temperature, so they must not
            # arm the promotion (and must not pay for the clock).
            if n_cpu > 1 and n_total >= _POOL_MIN_PIXELS_WARM:
                _t_defer = _time.perf_counter()
                _opl = _invert_newton(Xw, Yw, sub_progress=sub_progress)
                _note_pool_deferral(n_cpu, _cost_class, n_total,
                                    _time.perf_counter() - _t_defer)
                return _opl
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)

        if sub_progress is not None:
            sub_progress(0.0,
                         f'newton pool: {n_total} pts across '
                         f'{n_cpu} workers')
        # PIN THE PARENT'S EVALUATOR BACKEND into the payload (v5.32.3,
        # FIX_CI_POOL).  Resolved HERE rather than where ``_spline_data`` is
        # built, for two reasons: only a dispatch needs it (so a call that never
        # pools does not force a numba compile just to describe itself), and it
        # must be the branch in force at DISPATCH time.  The value is what
        # ``_Cheb2DEvaluator.ev_value_and_grad`` would take in this process, so
        # the worker reproduces the parent's floating-point ORDER instead of
        # re-deriving it from its own numba availability -- without this the
        # pool is only CONDITIONALLY bit-identical to serial (measured
        # 5.167e-14 locally / 1.358e-11 on CI when the two sides split).
        _spline_data['cheb_backend'] = _resolved_cheb_backend(newton_fit)
        # ...AND SHIP THE PARENT'S BUILT FIT (v5.33.0, FIX_POOL_REBUILD).
        # Pinning the backend made the two sides evaluate in the same ORDER; it
        # did not stop the worker RE-FITTING the polynomial from the pickled
        # grids.  That re-fit is a BLAS reduction (``A^T A`` over ~78 000 rows),
        # and OpenBLAS's reduction order depends on the thread width -- which a
        # spawn worker does not inherit, because ``threadpoolctl``'s cap is
        # process-global and a fresh interpreter starts at the environment
        # default.  Measured: a worker at a different BLAS width moved the
        # field by up to 1.370e-11, i.e. exactly the 1.341e-11 / 1.358e-11 that
        # ``test_pool_result_is_bit_identical_to_serial[polynomial]`` kept
        # failing by on CI after the backend pin shipped.  ``Sx``/``Sy``/``So``
        # here are the very objects the SERIAL closure evaluates, so shipping
        # their coefficients makes pool == serial true by construction rather
        # than by two least-squares solves agreeing.  ~700 bytes against the
        # ~1.9 MB of grids the payload already carries; ``None`` on the spline
        # path, whose FITPACK rebuild is single-threaded and BLAS-free.
        _spline_data['cheb_fit'] = _cheb_fit_payload(Sx, Sy, So, newton_fit)
        # PAYLOAD, PICKLED ONCE (FIX_PERF_ROUND2_2026_08_10 item 3).  The
        # payload used to be embedded in every chunk's arg tuple, so the
        # executor pickled it n_cpu times -- 99.2 % of the measured 0.173 s
        # 8-worker dispatch constant.  One ``dumps`` here, and the KEY ALONE
        # when the live workers already hold these exact bytes.  See the
        # PAYLOAD RESIDENCY block above ``_newton_invert_chunk`` for why the
        # key is a content digest and why residency can never be wrong.
        _pkey, _pblob = _newton_payload_blob(_spline_data)
        # Split indices into roughly-equal chunks.  ``np.array_split``
        # handles the n_total % n_cpu != 0 case cleanly.
        chunk_idx = np.array_split(np.arange(n_total), n_cpu)
        _chunks = [(x_w_flat[c].copy(), y_w_flat[c].copy())
                   for c in chunk_idx]
        results = [None] * len(_chunks)

        # Use the module-level persistent ProcessPool to amortise
        # Windows-spawn startup cost across repeated apply_real_lens_traced
        # calls (the dominant overhead for optimisation / tolerancing
        # workflows).  See _get_persistent_worker_pool docstring for
        # details.
        try:
            ex = _get_persistent_worker_pool(n_cpu)
            # Read the belief AFTER the pool call: that call may have torn the
            # old pool down and built a fresh (empty) one, which resets it.
            _send = (None if _POOL_RESIDENT_PAYLOAD_KEY == _pkey else _pblob)
            future_to_idx = {
                ex.submit(_newton_invert_chunk, (_pkey, _send, _xc, _yc)): i
                for i, (_xc, _yc) in enumerate(_chunks)}
            done = 0
            _missed = []
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    results[i] = fut.result()
                except NewtonPayloadNotResident:
                    # This worker never received these bytes (the executor
                    # does not promise a chunk per worker).  Collect and
                    # re-send WITH the payload below -- worst case is the
                    # pre-change behaviour for that one chunk.
                    _missed.append(i)
                done += 1
                if sub_progress is not None:
                    frac = min(done / max(len(_chunks), 1), 0.99)
                    sub_progress(
                        frac,
                        f'newton chunk {done}/{len(_chunks)} done')
            if _missed:
                _retry = {
                    ex.submit(_newton_invert_chunk,
                              (_pkey, _pblob, _chunks[i][0],
                               _chunks[i][1])): i
                    for i in _missed}
                for fut in as_completed(_retry):
                    results[_retry[fut]] = fut.result()
            # Every chunk answered, so at least the workers that ran one hold
            # these bytes.  Optimistic for a worker that ran none -- which the
            # miss path above absorbs on the next dispatch.
            _POOL_RESIDENT_PAYLOAD_KEY = _pkey
        except NewtonWorkerBackendUnavailable:
            # The worker could not provide the Chebyshev backend the payload
            # pinned and refused rather than answering in a different
            # floating-point order (v5.32.3).  Run the chunk here, where the
            # parent's own backend IS the pinned one -- which is what makes
            # this fallback bit-identical rather than merely close -- and say
            # so ONCE per process: a chain would otherwise repeat a paragraph
            # per group about a fact that cannot change under it.
            if _note_pool_backend_refusal():
                import warnings
                warnings.warn(
                    f"apply_real_lens_traced: running the Newton inversion "
                    f"SERIAL instead of on {n_cpu} workers, because this "
                    f"process evaluates the Chebyshev fit through the numba "
                    f"kernel and its spawn workers cannot load numba.  The "
                    f"two evaluate the same polynomial in a different "
                    f"floating-point order, so a worker that substituted the "
                    f"pure-NumPy branch would return a DIFFERENT answer from "
                    f"the serial path (measured 5.2e-14 to 1.4e-11 of the "
                    f"field) -- the pool's whole safety argument is that it is "
                    f"bit-identical.  Install numba where the workers can "
                    f"import it, or pass newton_fit='spline' (whose worker "
                    f"has no backend split), to get the pool back; the serial "
                    f"result is bit-identical either way, so nothing but wall "
                    f"time changes here.",
                    RuntimeWarning, stacklevel=3)
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
        except (BrokenProcessPool, RuntimeError, OSError, EOFError):
            # POOL-INFRASTRUCTURE failures only (v5.30, audit E-L3):
            # ``BrokenProcessPool`` (a worker died / spawn was blocked by
            # Windows antivirus), ``RuntimeError`` (executor already shut
            # down -- e.g. a prior atexit / interpreter-teardown race),
            # ``OSError`` / ``BrokenPipeError`` (its subclass: the worker
            # pipe broke) and ``EOFError`` (truncated pickle stream).  Fall
            # through to the in-process serial path so the caller still gets
            # a result, and DROP the cached executor first -- a pool that
            # just failed is not reusable, and the old code left it cached
            # so every subsequent call paid the same failure again.
            #
            # ``ValueError``, ``ImportError`` and ``MemoryError`` are
            # deliberately NOT caught any more.  Those are raised by
            # ``fut.result()`` on behalf of the WORKER, i.e. they are real
            # faults in ``_newton_invert_chunk`` (a bad spline fit, a missing
            # SciPy, an out-of-memory chunk).  Swallowing them silently re-ran
            # the identical computation serially, where the same fault
            # normally reproduces -- but with a completely different, much
            # more confusing traceback -- or, worse, succeeded serially and
            # hid a genuine parallel-path bug behind a silent 8x slowdown.
            close_worker_pool()
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)

        # v5.29.1 (audit E-H2): the worker returns (opl, n_unconverged); sum
        # the counts and emit the SAME warning the serial path emits (the
        # pool used to be silent, so the one regime whose convergence the
        # message's own advice addresses never reported).
        opl_flat = np.concatenate([r[0] for r in results])
        _warn_newton_unconverged(sum(int(r[1]) for r in results),
                                 int(n_total), 0.01 * dx)
        return opl_flat.reshape(Xw.shape)

    def _support_taper(Xg, Yg, _axes=None):
        """niche C8's exit-support taper on the coarse Newton lattice.

        The rule, the plateau and the whole measurement record now live on
        :meth:`_TracedExitSupport.taper` -- this closure is the call site's
        binding of the PLATEAU, which is the one quantity the object cannot
        know: ``sqrt(2) * sub * dx`` is a property of the lattice this call
        will interpolate FROM, not of the support.  Keeping the plateau here
        and the taper there is what lets the same support object serve the
        coarse-lattice taper and the wave-grid band check without either one
        guessing the other's sampling.

        ``_axes`` (internal) names the two 1-D axes whose outer product
        ``Xg`` / ``Yg`` ARE, which lets the radially-screened
        :meth:`_TracedExitSupport.taper_grid` bound the half-plane reduction.
        Only the inverse-characteristic path passes it: that path asks for the
        taper on the WAVE grid rather than on the coarse lattice, where the
        dense form is a 10^10-MAC BLAS pass (measured 5.9 s at 1.7e+07
        pixels).  The two forms are bit-identical -- the screens are strict --
        and the coarse lattice keeps the dense one because at 9 025 points it
        is free."""
        if _axes is not None:
            _t = _exit_support.taper_grid(_axes[0], _axes[1],
                                          np.sqrt(2.0) * sub * dx)
            if _t is not None:
                return _t
        return _exit_support.taper(Xg, Yg, np.sqrt(2.0) * sub * dx)

    def _ray_density_amp_grid(Xg, Yg, _pre=None):
        """N12 (P11): geometric ray-density exit amplitude ``|E_in(x_in)| /
        sqrt(|det J|)`` on the exit-position grid ``(Xg, Yg)``.

        Uses the SAME entrance->exit fits (``Sx``, ``Sy``) + Newton inverse the
        OPL phase uses, so the amplitude and phase are placed at consistent exit
        positions.  For each exit pixel Newton returns the entrance point
        ``(xe, ye)``; ``det J = d(x_out,y_out)/d(x_in,y_in)`` is the analytic
        gradient of the forward-map fits there (energy-conserving in the
        geometric limit -- the SAME ``1/sqrt(|det J|)`` Jacobian the ring-tube
        oracle uses), and ``|E_in|`` is bilinearly sampled at the entrance.  NaN
        where the ray map is out of domain.  ``|det J|`` is floored at a caustic
        (never inf/nan) and the fold is flagged in ``_rd_fold_detected``.

        ``_pre`` (internal) supplies ``(opl, xe, ye, det_j)`` already solved --
        the inverse-characteristic path's exact per-pixel answer instead of
        Newton on a coarse lattice.  EVERYTHING BELOW THE INVERSION IS SHARED:
        the caustic floor, the fold census, the entrance aperture stop, the
        niche-C8 support taper, the ``remap_sampling`` branches.  That sharing
        is the point -- one physics, two sources for the entrance point -- and
        with ``_pre=None`` this closure is byte-identical to the shipped one.
        """
        if _pre is None:
            opl_f, xe_g, ye_g = _invert_newton(Xg, Yg, _want_entrance=True)
            _det_pre = None
        else:
            opl_f, xe_g, ye_g, _det_pre = _pre
        sh = np.asarray(Xg).shape
        invalid = ~np.isfinite(np.asarray(opl_f))
        xef = np.asarray(xe_g, dtype=np.float64).ravel()
        yef = np.asarray(ye_g, dtype=np.float64).ravel()
        # Forward-map Jacobian J = d(x_out,y_out)/d(x_in,y_in) at the entrance.
        if _det_pre is not None:
            jxx = jxy = jyx = jyy = None
        elif _has_combined_fits:
            _fx, jxx, jxy = Sx.ev_value_and_grad(xef, yef)
            _fy, jyx, jyy = Sy.ev_value_and_grad(xef, yef)
        else:
            jxx = np.asarray(Sx.ev(xef, yef, dx=1))
            jxy = np.asarray(Sx.ev(xef, yef, dy=1))
            jyx = np.asarray(Sy.ev(xef, yef, dx=1))
            jyy = np.asarray(Sy.ev(xef, yef, dy=1))
        det_j = (np.asarray(_det_pre, dtype=np.float64).reshape(sh)
                 if _det_pre is not None
                 else (np.asarray(jxx) * np.asarray(jyy)
                       - np.asarray(jxy) * np.asarray(jyx)).reshape(sh))
        # |E_in| at the entrance (bilinear); rays whose entrance falls outside
        # the input grid contribute zero amplitude.
        from scipy.ndimage import map_coordinates as _mc
        _absin = np.abs(np.asarray(E_in)).astype(np.float64)
        # niche D9, entrance-coordinate -> E_in pixel index, copy 1 of 3 (the
        # other two are the remap_sampling='full' samplers below).  ``xef`` /
        # ``yef`` are ABSOLUTE entrance positions; the index is grid-relative.
        # NOTE the asymmetry of the two lines -- ``_col`` is built from x and
        # ``_row`` from y, and they are consumed in the (row, col) order
        # ``map_coordinates`` wants; a mechanical edit that pairs ``_org_x``
        # with the row would put the shift on the wrong axis silently.
        _col = (xef - _org_x) / dx + N / 2.0
        _row = (yef - _org_y) / dy + N / 2.0
        a_in = _mc(_absin, np.vstack([_row, _col]), order=1,
                   mode='constant', cval=0.0).reshape(sh)
        # preserve_input_phase='remap': sample the carrier-de-chirped input
        # residual UNIT PHASOR at the same entrance points (geometric
        # transport of the residual along its ray).  cval=1 (identity phase)
        # for rays whose entrance falls outside the grid -- their amplitude
        # is already zero above.  De-chirp with the exact carrier eikonal is
        # pointwise-exact even where the raw carrier is beyond-Nyquist.
        if _pip_remap:
            if _pip_full:
                # S12 (remap_sampling='full'): do NOT sample the residual on
                # this (possibly coarse) ray lattice -- keep the SMOOTH
                # entrance pullback and let the caller re-sample the residual
                # at full wave-grid resolution after upsampling these
                # coordinates.  See the ``remap_sampling`` docstring for why:
                # the residual's phase gradient grows as r^3 for r^4 carried
                # content, so on a lattice of pitch ``ray_subsample*dx`` it
                # exceeds Nyquist beyond a finite radius and the phasor
                # aliases in the beam SKIRT.
                # The copy exists because ``xef`` / ``yef`` are views into
                # Newton's own working arrays, which the caller may reuse.  On
                # the ``_pre`` path they are views into the inverse map's
                # full-grid outputs, which live to the end of the call -- so
                # aliasing them is safe and saves 2 x N^2 float64 (1.07 GB at
                # n_fine = 8192, 4.29 GB at 16384).
                _rd_entrance_coarse[0] = (
                    (xef.reshape(sh), yef.reshape(sh)) if _pre is not None
                    else (xef.reshape(sh).copy(), yef.reshape(sh).copy()))
            else:
                _rd_resid_coarse[0] = _pip_sample_residual(_row, _col, sh)
        absdet = np.abs(det_j)
        fin = np.isfinite(absdet) & (~invalid)
        ref = float(np.median(absdet[fin])) if fin.any() else 0.0
        floor = _RAY_DENSITY_CAUSTIC_FLOOR_REL * ref
        # Caustic/fold detection: |det J| driven below the floor (det J -> 0), a
        # large |det J| dynamic range (near a focus/caustic the ray tube
        # collapses so |det J| spans orders of magnitude), or a det J sign
        # change between adjacent (valid) ray cells.
        if fin.any():
            _amin = float(np.min(absdet[fin]))
            _amax = float(np.max(absdet[fin]))
            if floor > 0.0 and _amin < floor:
                _rd_fold_detected[0] = True
            if _amin > 0.0 and _amax / _amin > _RAY_DENSITY_CAUSTIC_MAXMIN:
                _rd_fold_detected[0] = True
        _sd = np.sign(det_j)
        _mh = fin[:, 1:] & fin[:, :-1]
        _mv = fin[1:, :] & fin[:-1, :]
        if (bool(np.any((_sd[:, 1:] * _sd[:, :-1] < 0.0) & _mh))
                or bool(np.any((_sd[1:, :] * _sd[:-1, :] < 0.0) & _mv))):
            _rd_fold_detected[0] = True
        absdet_capped = np.maximum(absdet, floor) if floor > 0.0 else absdet
        with np.errstate(divide='ignore', invalid='ignore'):
            a_rd = a_in / np.sqrt(absdet_capped)
        a_rd = np.where(invalid | (~np.isfinite(a_rd)), np.nan, a_rd)
        # Aperture is a stop at the ENTRANCE: a ray whose entrance falls outside
        # the aperture is physically blocked, so it carries no energy.  Masking
        # on the entrance (vs the final exit-position mask, which for a
        # converging element admits rays whose entrance exceeds the stop) makes
        # the ray-density power exactly the aperture-transmitted input power.
        if aperture is not None:
            r_ent2 = (xef * xef + yef * yef).reshape(sh)
            a_rd = np.where(r_ent2 <= (0.5 * aperture) ** 2, a_rd, np.nan)
        # niche C8: an exit pixel OUTSIDE the region the traced rays actually
        # reached has no data behind it -- the fitted inverse map there is an
        # extrapolation, and on a non-radial map it can fold back into the
        # bright beam and hand that pixel real amplitude.  Taper to zero across
        # the support boundary instead.  See ``REMAP_INVERSE_SUPPORT_BOUND``.
        if _sup_bound is not None:
            a_rd = a_rd * _support_taper(
                Xg, Yg, _axes=((x, _y_ax) if _pre is not None else None))
        return a_rd

    call_progress(progress, 'real_lens_traced', 0.55,
                  'inverting entrance->exit map')
    # Give the Newton loop its own slice of the parent budget
    # (0.55 -> 0.88) so the bar advances through the iterations
    # instead of sitting still between the 0.55 and 0.90 ticks.
    newton_cb = ProgressScaler(progress, 'real_lens_traced',
                               lo=0.55, hi=0.88)

    # ---------- Amplitude-mask the Newton work --------------------
    # Pixels where ``amp`` is well below peak produce a final field
    # of ``|E_analytic| * exp(...)`` that is already ~zero no matter
    # what OPL we compute for them, so running Newton there is
    # wasted effort.  We build a boolean mask on the coarse output
    # grid, dilate by ``newton_mask_dilate_coarse_px`` so bilinear
    # interpolation at the full grid always has real data in its
    # support near mask boundaries, and run Newton only on the
    # masked coarse pixels.  Skipped pixels get ``NaN`` which the
    # existing NaN-propagation logic below treats exactly like the
    # ray-domain-failure NaNs from Newton itself.
    #
    # Controls:
    #   newton_amp_mask_rel=0  disables masking (runs Newton on the
    #                          entire coarse grid, bit-identical to
    #                          pre-mask behaviour).
    #   newton_amp_mask_rel>0  threshold = that fraction of amp.max().
    #   newton_mask_dilate_coarse_px  0 for no dilation, else that
    #                          many iterations of binary_dilation.
    #
    # The mask is SKIPPED if it would capture essentially everything
    # (>95 %) -- in that case the filter overhead isn't worth it --
    # or essentially nothing (<1 %) -- which signals a pathological
    # amp field and we fall back to full-grid Newton rather than
    # returning garbage.
    def _build_newton_mask(amp_grid):
        if newton_amp_mask_rel <= 0.0:
            return None
        amp_max = float(amp_grid.max())
        if amp_max <= 0.0:
            return None
        thresh = amp_max * float(newton_amp_mask_rel)
        m = amp_grid > thresh
        if newton_mask_dilate_coarse_px > 0:
            from scipy.ndimage import binary_dilation
            m = binary_dilation(
                m, iterations=int(newton_mask_dilate_coarse_px))
        frac = float(m.mean())
        if frac > 0.95 or frac < 0.01:
            return None
        return m

    # K3 (N15 perf): the ray-density upsample below reuses the OPL upsample's
    # coarse->full (2, N, N) coordinate stack when their coarse resolutions
    # match (they always do -- both are ``X[::sub, ::sub]``).  Stashed here as
    # ``(_coords, Ns)`` by the OPL sub>1 branch; ``None`` means build a fresh
    # one.  Bounded to this call (freed after the ray-density upsample); no cache.
    _rd_upsample_coords = None

    # ---- THE INVERSE CHARACTERISTIC (per-pixel exact inversion) -----------
    # See :mod:`lumenairy.elements._lens_imap`.  The map is fitted from the
    # landings THIS call already traced, so it describes this congruence
    # exactly -- including the niche-C6 ``a_fit`` launch augmentation, which is
    # 87 % pupil-varying aberration on design 121's last group and which no
    # source-labelled shared map can represent (measured: 48.5 waves against a
    # 1.11e-04-wave bar; the module header carries the table).
    #
    # THE GATE, and every clause of it is a refusal rather than a workaround:
    #   sub > 1                 at sub == 1 there IS no coarse lattice and no
    #                           upsample to replace -- the Newton already runs
    #                           per pixel and the map would only add error;
    #   inversion_method newton the 'fit' path is already a per-pixel exit
    #                           polynomial, and 'backward_trace' has no
    #                           forward fits to build a map from;
    #   not _chunk_assembly     the band path exists to never materialise a
    #                           full-grid float64; handing it one would undo
    #                           the memory fix it is;
    #   not use_gpu             the fits live on the device and the map's
    #                           kernel is a CPU numba/NumPy pair;
    _imap = None
    _imap_rec = {'engaged': False}
    # resolved once, at the fit-domain site above, so the domain the model is
    # given and the gate that decides whether to build it cannot disagree.
    _imap_gate = _imap_domain_gate
    if _IMAP.imap_enabled(inverse_map) and _imap_gate:
        def _imap_incumbent(_xq, _yq):
            """G8's arm B: the INCUMBENT inversion, which is this element's
            own Newton on this element's own forward fits.  Not a
            reproduction of it -- the function itself."""
            _o, _xe, _ye = _invert_newton(_xq, _yq, _want_entrance=True)
            return _xe, _ye, _o

        def _imap_probe_trace(_px, _py):
            """G8's GROUND TRUTH: this element's own trace of this element's
            own launch congruence, at entrance points that are NOT launch
            nodes.

            WHY THE GUARD NEEDS IT.  G8 used to score both arms at held-out
            LAUNCH NODES.  ``newton_fit='spline'`` builds its
            ``RectBivariateSpline`` with the default ``s = 0``, so the
            incumbent reproduces every launch node EXACTLY -- the held-out ones
            included, because they were held out of the MODEL only -- and its
            measured "error" there was the Newton loop's leftover residual
            rather than its accuracy (34.5x better at its own knots than
            between them; ``docs/audits/FIX_G8_PROBE_2026_08_12.md`` S2).  Exit
            pixels are never launch nodes, so the acceptance bar has to be
            measured where they DO fall, against truth neither arm defines.

            Every term is the one the lattice trace used, in the same order and
            from the same objects: the launch directions
            (``grad(W + a_fit)`` under niche C6), the exit-vertex correction,
            the H6 carrier eikonal, the C6 residual eikonal and the SAME
            on-axis reference ``_opl_ref``.  It is pinned to reproduce
            ``x_out_grid`` / ``y_out_grid`` / ``opl_grid`` when handed the
            launch lattice itself.
            """
            _px = np.asarray(_px, dtype=np.float64).ravel()
            _py = np.asarray(_py, dtype=np.float64).ravel()
            if _px.size == 0:
                _e = np.empty(0, dtype=np.float64)
                return _e, _e.copy(), _e.copy()
            if _tilt_aware_launch:
                _pL, _pM = _sample_local_tilts(
                    E_in, wavelength, dx, _px, _py,
                    origin=(_org_x, _org_y))
                _pL = np.asarray(_pL, dtype=np.float64).ravel()
                _pM = np.asarray(_pM, dtype=np.float64).ravel()
            elif _carrier_grad is not None:
                _pL, _pM = _carrier_grad(_px, _py)
                _pL = np.asarray(_pL, dtype=np.float64).ravel()
                _pM = np.asarray(_pM, dtype=np.float64).ravel()
                if _resid_eik is not None:
                    # the C6 launch augmentation.  ``.diag`` is NOT touched
                    # here: the launch diagnostic belongs to the lattice trace,
                    # and a probe that overwrote it would report itself.
                    _aL, _aM = _resid_eik.grad(_px, _py)
                    _pL = _pL + np.asarray(_aL, dtype=np.float64).ravel()
                    _pM = _pM + np.asarray(_aM, dtype=np.float64).ravel()
            else:
                _pL = np.zeros_like(_px)
                _pM = np.zeros_like(_px)
            _pfin = trace(_make_bundle(x=_px.copy(), y=_py.copy(),
                                       L=_pL, M=_pM, wavelength=wavelength),
                          surfaces, wavelength,
                          output_filter='last').image_rays
            with np.errstate(divide='ignore', invalid='ignore'):
                _pt = np.where(_pfin.alive & (np.abs(_pfin.N) > 1e-30),
                               -_pfin.z / _pfin.N, 0.0)
            _pox = _pfin.x + _pfin.L * _pt
            _poy = _pfin.y + _pfin.M * _pt
            _pop = _pfin.opd + n_exit * _pt
            if _carrier_W_fn is not None:
                _pop = _pop + _carrier_W_fn(_px, _py)
            if _resid_eik is not None:
                _pop = _pop + _resid_eik.value(_px, _py)
            # the SAME on-axis reference the lattice OPL carries, so the model
            # channel, the incumbent and this truth are on one convention.
            _pop = _pop - _opl_ref
            if not _pfin.alive.all():
                _pox = np.where(_pfin.alive, _pox, np.nan)
                _poy = np.where(_pfin.alive, _poy, np.nan)
                _pop = np.where(_pfin.alive, _pop, np.nan)
            return _pox, _poy, _pop

        # NO Jacobian is passed.  The model derives its own from TRACED data
        # (``_IMAP_DETJ_SOURCE``), which is what keeps its amplitude
        # independent of ``newton_fit`` -- see that constant's note, and the
        # shipped guard it exists to honour.
        #
        # THE SAMPLE SET IS THE SAME ON EITHER BASIS (fix FIT-DOMAIN SYMMETRY
        # 2026-08-12).  On the polynomial basis the forward arrays already
        # carry the restriction and the ``_imap_*`` copies are ``None``, so
        # this is verbatim the shipped call; on the spline basis -- which
        # cannot restrict its own bicubic -- the copies carry the SAME disc
        # and the SAME D1 weights, resolved from the SAME basis-independent
        # beam radius.  Without this the two backends described different
        # maps and the shipped c6 backend-symmetry guard was right to say so.
        _imap = _IMAP.build_inverse_map(
            xs_in,
            x_out_grid if _imap_xo is None else _imap_xo,
            y_out_grid if _imap_yo is None else _imap_yo,
            opl_grid if _imap_op is None else _imap_op,
            wavelength=wavelength, launch_radius=launch_radius,
            weights=(_fit_weights if _imap_weights is None
                     else _imap_weights),
            parity_invert=_imap_incumbent,
            # G8's ground truth, which is neither arm's: the element's own
            # trace at OFF-LATTICE entrance points.  The guard picks the
            # points; this only supplies the congruence.
            probe_trace=_imap_probe_trace,
            # WHICH incumbent this is.  The model is now identical on either
            # basis, so without this the two bases COLLIDE in the build cache
            # and the second call inherits the first's G8 verdict -- an
            # acceptance decided by call order.  See ``parity_tag``.
            parity_tag=(str(newton_fit), int(_fit_poly_order),
                        _fit_weights is None, int(MAX_NEWTON_ITERS)),
            caller='apply_real_lens_traced', guard_record=_imap_rec)
        if _imap is None:
            _IMAP.report_refusal(_imap_rec, 'apply_real_lens_traced')
        else:
            _imap_rec['engaged'] = True
    if _imap_out is not None:
        _imap_out.update(_imap_rec)
        _imap_out['gate_open'] = bool(_imap_gate)

    # Dispatch the OPL inversion to Newton (default) or the experimental
    # backward-trace alternative.  Both produce a wave-grid OPL map
    # with the same axis convention (on-axis referenced to zero, NaN
    # for out-of-domain / dead-ray pixels).
    if inversion_method == 'backward_trace':
        # Experimental path.  Bypasses the forward ray trace + Newton
        # spline inversion entirely; see _opl_by_backward_trace for
        # the algorithm and caveats.  Kept as an opt-in because the
        # accuracy on focused-beam exit planes has not been as
        # thoroughly validated as the Newton path.
        opl_map = _opl_by_backward_trace(
            E_analytic, lens_prescription, wavelength, dx,
            N_grid=N, ray_subsample=sub)
    elif _imap is not None:
        # ---- the inverse characteristic, evaluated at EVERY exit pixel -----
        # This replaces the whole coarse-lattice chain below it: the Newton on
        # ``X[::sub, ::sub]``, the order-3 OPL upsample, the order-1 NaN pass,
        # and (on the ray-density branch) the two amplitude upsamples and the
        # two entrance-coordinate upsamples.  MEASURED on design 121's last
        # group at n_fine = 8192: those six full-grid ``map_coordinates`` calls
        # are 14.767 s of a 96.9 s element; one 4-channel degree-14 evaluation
        # over the same 6.71e+07 pixels is 1.910 s.
        #
        # The entrance coordinates and ``det J`` are kept for the ray-density
        # branch below, which then does exactly what it always did -- the
        # caustic floor, the fold census, the entrance aperture stop, the C8
        # taper -- on an EXACT per-pixel inversion instead of an interpolated
        # one.
        if X is None:                                    # pragma: no cover
            X = np.broadcast_to(x[None, :], (N, N))
            Y = np.broadcast_to(_y_ax[:, None], (N, N))
        # The Newton loop's own 0.55 -> 0.88 progress slice belongs to a loop
        # that is not running; drive the same slice from here so a caller's
        # bar does not sit still through the one full-grid pass of the call.
        call_progress(progress, 'real_lens_traced', 0.60,
                      'inverse map: evaluating %d exit pixels' % (N * N,))
        _im_xin = np.empty((N, N), dtype=np.float64)
        _im_yin = np.empty((N, N), dtype=np.float64)
        opl_map = np.empty((N, N), dtype=np.float64)
        _im_chans = [_im_xin, _im_yin, opl_map]
        _im_ids = [_IMAP.InverseCharacteristic.CH_X_IN,
                   _IMAP.InverseCharacteristic.CH_Y_IN,
                   _IMAP.InverseCharacteristic.CH_OPL]
        if _ray_density:
            _im_detj = np.empty((N, N), dtype=np.float64)
            _im_chans.append(_im_detj)
            _im_ids.append(_IMAP.InverseCharacteristic.CH_DET_J)
        else:
            _im_detj = None
        _imap.eval_into(X, Y, _im_chans, channels=_im_ids)
        # THE DOMAIN, and it is not optional.  Outside the convex hull of the
        # ray landings there is no ray and the degree-14 model extrapolates --
        # measured at 1.1e+04 waves one plateau out (proto S5.1).  Inside it,
        # the entrance-radius rule is ``_invert_newton``'s own out-of-domain
        # test verbatim, applied to the exact entrance point instead of to a
        # Newton iterate that was clipped to the boundary and then failed it.
        # THE RELAXATION IS THE BAND NICHE C8 RETAINS, and it is load-bearing.
        # The exit-support taper holds the amplitude at exactly 1 out to
        # ``sqrt(2) * sub * dx`` beyond the landing hull and then feathers over
        # one exit-lattice cell, so the element EMITS a ring outside the hull
        # at full amplitude.  ``valid = np.isfinite(opl_map)`` gates the PHASE
        # correction and not the amplitude, so NaN-ing that ring would not
        # truncate it -- it would hand it the IDENTITY phase.  MEASURED on the
        # shipping banner: cutting at ``s <= 0`` moved FWHM 3.350 -> 3.550 um
        # and EE3 90.3 -> 89.7 %.  Model exactly the support the element emits.
        _im_relax = (np.sqrt(2.0) * sub * dx
                     + float(_exit_support.feather
                             if _exit_support.feather is not None
                             else _SUPPORT_BOUND_FEATHER_CELLS * sub * dx))
        _im_ok = _imap.domain_mask(X, Y, _im_xin, _im_yin, axes=(x, _y_ax),
                                   relax=_im_relax)
        if not _im_ok.all():
            _im_bad = ~_im_ok
            opl_map = np.where(_im_ok, opl_map, np.nan)
            # ...and park the out-of-domain ENTRANCE coordinates on the origin.
            # They are extrapolations of a degree-14 polynomial; every consumer
            # of them (the |E_in| sampler, the residual sampler, the entrance
            # aperture stop) is masked to NaN by ``opl_map`` a few lines later,
            # so their VALUE cannot reach the field -- but leaving 1e+04-mm
            # coordinates in an array that gets divided by ``dx`` and handed to
            # ``map_coordinates`` is how a masked-off quantity becomes a
            # not-masked-off performance cliff.  Zero is inside the grid and
            # costs one pass.
            np.copyto(_im_xin, 0.0, where=_im_bad)
            np.copyto(_im_yin, 0.0, where=_im_bad)
            del _im_bad
        _imap_rec['n_out_of_domain'] = int(_im_ok.size - _im_ok.sum())
        if _imap_out is not None:
            _imap_out['n_out_of_domain'] = _imap_rec['n_out_of_domain']
        call_progress(progress, 'real_lens_traced', 0.88,
                      'inverse map: %d pixels outside the landing hull'
                      % (_imap_rec['n_out_of_domain'],))
    elif sub > 1:
        # Evaluate Newton on sub-sampled output grid.  On the chunked-
        # assembly path the full X/Y meshgrids were never built; the coarse
        # grid from the 1-D subsampled vector is element-identical to
        # ``X[::sub, ::sub]`` (meshgrid(x,x) is x[j]/x[i] replicated).
        if X is None:
            Xs, Ys = np.meshgrid(x[::sub], _y_ax[::sub])
        else:
            Xs = X[::sub, ::sub]
            Ys = Y[::sub, ::sub]
        if _use_fit:
            # T-P2: one polynomial evaluation over the whole coarse grid; no
            # amp mask (the fit is cheap everywhere, hull-masked to NaN).
            if preserve_input_phase:
                del amp
            opl_coarse = _invert_fit(Xs, Ys)
        else:
            amp_coarse = amp[::sub, ::sub]
            mask_coarse = _build_newton_mask(amp_coarse)
            if preserve_input_phase:
                # v5.17.1 (audit P3-09): on the sub>1 preserve_input_phase
                # path ``amp`` is never read again (Step 3 combines with
                # E_analytic, not amp) and ``amp_coarse`` is dead after the
                # Newton-mask build -- but amp_coarse is a VIEW, so the
                # full-grid float base (float64 for complex128 fields,
                # ~8.6 GB at N=32768) would otherwise stay resident through
                # the Newton inversion and the entire band assembly.  Free
                # both eagerly -- same lifetime-fix pattern as the v5.16.2
                # eager frees; values/outputs byte-identical.
                del amp_coarse, amp
            if mask_coarse is None:
                opl_coarse = _invert_newton_parallel(
                    Xs, Ys, sub_progress=newton_cb)
            else:
                Xs_masked = Xs[mask_coarse]
                Ys_masked = Ys[mask_coarse]
                opl_1d = _invert_newton_parallel(
                    Xs_masked, Ys_masked, sub_progress=newton_cb)
                opl_coarse = np.full(Xs.shape, np.nan, dtype=opl_1d.dtype)
                opl_coarse[mask_coarse] = opl_1d
        # Bilinearly interpolate to full grid
        from scipy.ndimage import map_coordinates
        Ns = opl_coarse.shape[0]
        # R7 / audit F2: CUBIC (order-3) OPL upsample when a carrier is set --
        # see the whole-grid branch below for the derivation.  Resolved HERE,
        # before the path split, because the row-band assembly needs the SAME
        # order: it used to hard-code order=1, so ``sag_chunk_rows`` (which the
        # v5.17 auto-resolver turns ON by default at N >= 4096) silently
        # downgraded the R7 cubic upsample to linear on every carrier-
        # referenced call, breaking this module's own "row-band assembly is
        # BYTE-IDENTICAL to the whole-grid path" contract.  Measured
        # 2026-07-25 (singlet, carrier=+30 mm, N=512, sub=8, forced band):
        # 4.17e-2 max / 2.92e-2 rms relative field change AT THE BEAM CORE
        # (power unchanged -> pure phase), i.e. ~lambda/216 rms of the very
        # residual R7 exists to remove (~0.37 rad in the outer beam on the
        # 121's steepest triplet).  Banding is over the OUTPUT coordinates
        # while the (small, coarse) INPUT array and its spline prefilter are
        # whole, so order-3 is exactly band-decomposable -- the fix restores
        # bit-equality rather than trading it.
        _opl_up_order = 3 if _r7_carrier_path else 1
        # NaN-PASS GUARD (FIX_PERF_ROUND2_2026_08_10 item 2;
        # AUDIT_TRACED_SPEED_2026_08_09 ranked row 6).  The upsample runs a
        # SECOND full-grid ``map_coordinates`` purely to carry the coarse OPL's
        # NaN mask to the wave grid.  When ``opl_coarse`` carries no NaN at all
        # -- no Newton amp mask, and no ray leaving the fit domain -- that pass
        # interpolates an ALL-ZERO array, so ``nan_full`` is identically 0,
        # ``nan_full > 0.5`` is identically False, and the ``np.where`` that
        # consumes it is the IDENTITY.  Skipping it is therefore bit-identical
        # BY CONSTRUCTION, not to a tolerance.  The test on the small coarse
        # lattice (95^2 - 531^2 here) is free next to the N^2 output it saves.
        #
        # MEASURED on the design-121 fan order at ``n_fine_cap=8192``: the two
        # order-1 NaN passes (this one and the ray-density one below) were
        # 2.43 % of the order's wall; the whole ``map_coordinates`` bucket was
        # 9.87 %, the profile's #2 site.  Whether the guard FIRES is a property
        # of the ray-fit hull, which the audit explicitly left unmeasured
        # (its sec 11 item 5) -- ``FIX_PERF_ROUND2_2026_08_10.md`` sec 3
        # measures it per call site.
        _opl_has_nan = bool(np.isnan(opl_coarse).any())
        if _chunk_assembly:
            # Row-band path: defer the upsample into the Step-3 band loop
            # (map_coordinates is pointwise in the OUTPUT at any order -- the
            # coarse input and its prefilter are whole -- so the banded
            # interpolation is element-identical).  Only the SMALL coarse
            # arrays are kept; the full-grid ii/jj index pair, the
            # (2, N, N) coords stack, opl_map and nan_full never allocate.
            _opl_coarse_clean = np.where(
                np.isnan(opl_coarse), 0.0, opl_coarse)
            # ``None`` when the coarse OPL is NaN-free -- the band loop then
            # skips its own per-band NaN pass, exactly as the whole-grid
            # branch does, and for the same by-construction reason.
            _nan_coarse = (np.isnan(opl_coarse).astype(np.float64)
                           if _opl_has_nan else None)
            opl_map = None
        else:
            # v5.16.2 (memory root-cause): build the (2, N, N) coordinate
            # stack ONCE and free ii/jj before interpolating.  Pre-fix the
            # stack was constructed twice (once per map_coordinates call)
            # with ii/jj held throughout -- ~4 extra full-grid float64
            # (~34 GB at N=32768) at the upsample peak.  Same coords,
            # same map_coordinates inputs -> byte-identical outputs.
            # FIX_PERF_ROUND2_2026_08_10 item 2 (the coords half).  The stack is
            # built STRAIGHT INTO its final buffer instead of through
            # ``np.indices`` + a two-element ``np.array`` list build.  The old
            # form materialised the (2, N, N) index pair (4.295 GB at
            # n_fine = 16384), then TWO more full-grid quotients, then the
            # (2, N, N) result -- a ~12.9 GB transient peak for a 4.295 GB
            # answer.  BIT-IDENTICAL: ``np.indices(..., float64)`` holds exact
            # integer-valued float64s, so ``arange(N, float64) / sub`` is the
            # same IEEE division of the same two operands, broadcast into the
            # same rows/columns (``ii[r, c] = r``, ``jj[r, c] = c``).
            #
            # The audit's OTHER option here -- caching the stack ACROSS calls
            # (its ranked row 7, 1.10-1.19x on a synthetic upsample) -- is
            # deliberately NOT taken: on the real order this whole build
            # MEASURES 0.30 % of the wall (profile leaf ``:9833``), while a
            # live cache would retain 4.295 GB of full-grid float64 at
            # n_fine = 16384 for the rest of the order, against a branch whose
            # companion item just FREED 34.9 GB of exactly this shape.  The
            # measurement is in ``FIX_PERF_ROUND2_2026_08_10.md`` sec 3.2.
            #
            # Coarse sample u sits at FINE index u*sub (the ``X[::sub]``
            # lattice), so the exact mapping is ii/sub for ANY sub.  The
            # previous ``ii * Ns / N`` (Ns = ceil(N/sub)) equals ii/sub only
            # when sub divides N; otherwise it displaced the OPL map
            # diagonally toward the (-x,-y) corner by (N/2)*(Ns*sub-N)/N
            # pixels and radially mis-scaled it -- the traced chain's
            # diagonal focus walk (audit
            # AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24; the F-C fine-retrace
            # rescale routinely produces non-divisor ray_subsample values).
            _idx_ax = np.arange(N, dtype=np.float64) / sub
            _coords = np.empty((2, N, N), dtype=np.float64)
            _coords[0] = _idx_ax[:, None]
            _coords[1] = _idx_ax[None, :]
            del _idx_ax
            # R7 / audit F2 (2026-07-21): CUBIC (order-3) OPL upsample when a
            # carrier is set.  The Newton OPL is solved on the COARSE grid
            # (spacing sub*dx) and interpolated to the wave grid; LINEAR
            # (order-1) upsampling of a rapidly-CURVING OPL leaves a smooth
            # residual ~ f''*(sub*dx)^2/8 that grows toward the beam edge -- for
            # the 121's steepest triplet (S25-S27) ~0.37 rad in the OUTER beam
            # (r > w), invisible to the r<w per-group oracle but SCATTERING
            # energy through the composed chain.  Cubic upsampling drops that
            # residual ~2 orders (error ~ f''''*(sub*dx)^4/384) for the smooth
            # OPL, at negligible cost.  Cubic needs a prefilter so the NaN 0-fill
            # cannot bleed across the ray-domain boundary; ``map_coordinates``
            # with ``prefilter=False`` on a 0-filled array plus the separate
            # order-1 NaN mask (dilated by the > 0.5 threshold below) keeps the
            # boundary crisp.  Gated on a carrier being set (byte-identical
            # carrier=None default).  ``_opl_up_order`` is resolved above the
            # path split so the row-band assembly uses the SAME order.
            opl_map = map_coordinates(
                np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
                _coords, order=_opl_up_order, mode='nearest',
                prefilter=(_opl_up_order > 1))
            # Propagate NaN mask (order-1 keeps the ray-domain boundary crisp;
            # any cubic bleed of the 0-fill into a valid pixel adjacent to NaN
            # is masked out here).  Skipped outright when there is no NaN to
            # propagate -- see the guard's derivation at ``_opl_has_nan``.
            if _opl_has_nan:
                nan_coarse = np.isnan(opl_coarse).astype(np.float64)
                nan_full = map_coordinates(
                    nan_coarse, _coords, order=1, mode='nearest')
            else:
                nan_full = None
            # K3 (N15 perf): hand the coordinate stack to the ray-density
            # upsample (identical ``Ns``) rather than let it rebuild
            # ``np.indices`` + a second (2, N, N) float64 array.  For the
            # screen path (no ray-density), free it now exactly as before.
            if _ray_density:
                _rd_upsample_coords = (_coords, Ns)
            # Drop the LOCAL name unconditionally.  On the ray-density path
            # the stash owns the array from here (both are released at
            # ``_rd_upsample_coords = None`` below); leaving ``_coords`` bound
            # kept the (2, N, N) float64 stack -- 4.295 GB at n_fine = 16384 --
            # alive for the WHOLE remaining call, long after the ray-density
            # upsample had freed its own alias.  The memory census caught it
            # live in this frame at the peak plateau
            # (AUDIT_TRACED_MEMORY_2026_08_09 sec 2.3).  Pure lifetime; the
            # screen path deleted it here already.
            del _coords
            if nan_full is not None:
                opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
            del nan_full
    elif _use_fit:
        # T-P2: full-grid inverse-map fit (no Newton, no amp mask).
        if X is None:
            # Broadcast views, as at the primary X/Y build above -- same
            # elements, no 2 x N^2 float64 materialisation.
            X = np.broadcast_to(x[None, :], (N, N))
            Y = np.broadcast_to(_y_ax[:, None], (N, N))
        opl_map = _invert_fit(X, Y)
    else:
        mask_full = _build_newton_mask(amp)
        if mask_full is None:
            opl_map = _invert_newton_parallel(
                X, Y, sub_progress=newton_cb)
        else:
            X_masked = X[mask_full]
            Y_masked = Y[mask_full]
            opl_1d = _invert_newton_parallel(
                X_masked, Y_masked, sub_progress=newton_cb)
            opl_map = np.full(X.shape, np.nan, dtype=opl_1d.dtype)
            opl_map[mask_full] = opl_1d

    # ---- N12 (P11): ray-density (Jacobian) exit amplitude on the wave grid ---
    # Built on the SAME coarse Newton grid the OPL used (or the full grid at
    # sub=1) and upsampled identically, so the ray-density magnitude and the
    # traced OPL phase share exit positions.  ``_chunk_assembly`` is forced off
    # for ray-density, so X/Y exist here.
    ard_map = None
    if _ray_density:
        if _imap is not None:
            # The inverse characteristic already solved the inversion at every
            # exit pixel, so there is no second Newton, no coarse amplitude
            # lattice and no upsample: the amplitude closure runs ONCE, at full
            # resolution, on the exact entrance points the OPL was taken at.
            # Everything it does below the inversion -- the caustic floor, the
            # fold census, the |E_in| sample, the entrance aperture stop, the
            # niche-C8 support taper -- is the shipped code, unchanged.
            ard_map = _ray_density_amp_grid(
                X, Y, _pre=(opl_map, _im_xin, _im_yin, _im_detj))
            if _pip_remap and _rd_resid_coarse[0] is not None:
                # remap_sampling='lattice': the closure sampled the residual at
                # the entrance points, and those points are now the wave grid's
                # own -- there is nothing coarse left to upsample.
                _rd_resid_map = _rd_resid_coarse[0]
            elif _pip_full and _rd_entrance_coarse[0] is not None:
                # S12 (remap_sampling='full'): sample the residual phasor at
                # full wave-grid resolution, exactly as the upsampled path
                # does -- the ONLY difference is that the entrance pullback it
                # is sampled at is EXACT rather than bilinearly upsampled from
                # a ``sub*dx``-pitch lattice.  The renormalisation is kept
                # because ``_pip_sample_residual`` itself interpolates the unit
                # phasor on the INPUT grid (|z| <= 1 off-node); it is that
                # interpolation, not the pullback's, that it corrects.
                _xe_f, _ye_f = _rd_entrance_coarse[0]
                # niche D9 copy 2 of 3: (row from y, col from x).
                _pz = _pip_sample_residual((_ye_f - _org_y) / dy + N / 2.0,
                                           (_xe_f - _org_x) / dx + N / 2.0,
                                           _xe_f.shape)
                _pa = np.abs(_pz)
                _rd_resid_map = np.where(
                    _pa > 1e-6, _pz / np.maximum(_pa, 1e-300), 1.0 + 0.0j)
                del _pz, _pa
            _im_detj = None
        elif sub > 1:
            Xs_rd = X[::sub, ::sub]
            Ys_rd = Y[::sub, ::sub]
            ard_coarse = _ray_density_amp_grid(Xs_rd, Ys_rd)
            from scipy.ndimage import map_coordinates as _mc_rd
            Ns_rd = ard_coarse.shape[0]
            # K3 (N15 perf): reuse the OPL upsample's coordinate stack when it
            # matches this coarse resolution (it always does -- same
            # ``X[::sub, ::sub]`` grid, so ``ii/sub`` / ``jj/sub`` are the
            # SAME float64 array bit-for-bit); otherwise build a fresh one.
            if (_rd_upsample_coords is not None
                    and _rd_upsample_coords[1] == Ns_rd):
                _coords_rd = _rd_upsample_coords[0]
            else:
                # ii/sub, not ii*Ns/N: exact for any sub (see the OPL
                # upsample above -- same lattice, same walk bug otherwise),
                # and built straight into its buffer for the same reason and
                # with the same bit-identity argument.
                _idx_rd = np.arange(N, dtype=np.float64) / sub
                _coords_rd = np.empty((2, N, N), dtype=np.float64)
                _coords_rd[0] = _idx_rd[:, None]
                _coords_rd[1] = _idx_rd[None, :]
                del _idx_rd
            _a_rd = _mc_rd(np.where(np.isnan(ard_coarse), 0.0, ard_coarse),
                           _coords_rd, order=1, mode='nearest')
            # Same NaN-pass guard as the OPL upsample above, on this array's
            # OWN NaN census (the ray-density amplitude and the OPL come from
            # the same Newton solve but are masked separately, so neither
            # census stands in for the other).  Bit-identical by construction:
            # with no NaN in ``ard_coarse`` the second interpolation returns
            # identically 0 and the ``np.where`` below is the identity.
            _rd_has_nan = bool(np.isnan(ard_coarse).any())
            _nan_rd = (_mc_rd(np.isnan(ard_coarse).astype(np.float64),
                              _coords_rd, order=1, mode='nearest')
                       if _rd_has_nan else None)
            # 'remap': upsample the entrance-pulled residual phasor with the
            # SAME coordinate stack, then renormalise to unit modulus (the
            # bilinear interp of a unit phasor has |z| <= 1; where |z| ~ 0
            # the phase is undefined -- identity there).
            if _pip_remap and _rd_resid_coarse[0] is not None:
                _prc = _rd_resid_coarse[0]
                _pz = (_mc_rd(np.real(_prc), _coords_rd, order=1,
                              mode='nearest')
                       + 1j * _mc_rd(np.imag(_prc), _coords_rd, order=1,
                                     mode='nearest'))
                _pa = np.abs(_pz)
                _rd_resid_map = np.where(
                    _pa > 1e-6, _pz / np.maximum(_pa, 1e-300), 1.0 + 0.0j)
                del _prc, _pz, _pa
            elif _pip_full and _rd_entrance_coarse[0] is not None:
                # S12 (remap_sampling='full'): upsample the SMOOTH entrance
                # pullback coordinates with the SAME coordinate stack the
                # amplitude uses (so phase and amplitude keep sharing exit
                # positions), then sample the residual phasor at FULL wave-grid
                # resolution.  The pullback is a geometric ray map -- smooth,
                # so bilinear upsampling of it is accurate to O(h^2 * |x_e''|);
                # the phasor is the fast quantity and is never resampled off a
                # coarse lattice.
                _xe_c, _ye_c = _rd_entrance_coarse[0]
                _xe_f = _mc_rd(_xe_c, _coords_rd, order=1, mode='nearest')
                _ye_f = _mc_rd(_ye_c, _coords_rd, order=1, mode='nearest')
                # niche D9 copy 2 of 3: (row from y, col from x).
                _pz = _pip_sample_residual((_ye_f - _org_y) / dy + N / 2.0,
                                           (_xe_f - _org_x) / dx + N / 2.0,
                                           _xe_f.shape)
                del _xe_c, _ye_c, _xe_f, _ye_f
                _pa = np.abs(_pz)
                _rd_resid_map = np.where(
                    _pa > 1e-6, _pz / np.maximum(_pa, 1e-300), 1.0 + 0.0j)
                del _pz, _pa
            del _coords_rd
            _rd_upsample_coords = None
            ard_map = (np.where(_nan_rd > 0.5, np.nan, _a_rd)
                       if _nan_rd is not None else _a_rd)
            # Both upsampled halves are consumed by that one ``where``; they
            # were held to the end of the call.  2 x 2.147 GB at
            # n_fine = 16384 (census sec 2.3).  Pure lifetime.  On the skipped
            # branch ``ard_map`` IS ``_a_rd`` (the ``where`` was the identity),
            # so only the name is dropped -- the array is the return path's.
            del _nan_rd, _a_rd
        else:
            ard_map = _ray_density_amp_grid(X, Y)
            if _pip_remap and _rd_resid_coarse[0] is not None:
                _rd_resid_map = _rd_resid_coarse[0]
            elif _pip_full and _rd_entrance_coarse[0] is not None:
                # sub == 1: the "coarse" lattice IS the wave grid, so 'full'
                # and 'lattice' sample the residual at exactly the same
                # coordinates.  Reproduce the legacy expression (including its
                # un-normalised phasor) so remap_sampling is a no-op here.
                _xe_c, _ye_c = _rd_entrance_coarse[0]
                # niche D9 copy 3 of 3: (row from y, col from x).
                _rd_resid_map = _pip_sample_residual(
                    (_ye_c - _org_y) / dy + N / 2.0,
                    (_xe_c - _org_x) / dx + N / 2.0, _xe_c.shape)
                del _xe_c, _ye_c
        # Release the cached entrance-grid residual pair (2 float64 N^2
        # arrays -- 1 GiB at the N = 8192 fine retrace leg): it is only ever
        # needed while the residual map is being built.
        _pip_res_ri[0] = None
        _rd_entrance_coarse[0] = None
        _rd_resid_coarse[0] = None
        if _rd_fold_detected[0]:
            import warnings as _rd_warn
            _rd_warn.warn(
                "apply_real_lens_traced: amplitude_model='ray_density' "
                "detected a fold caustic (det J -> 0 or a sign change) in the "
                "ray map.  The single-branch ray-density amplitude is CAPPED "
                "there (finite, never inf/nan) but is UNRELIABLE near the fold "
                "-- this mode does NOT sum the multi-valued ray branches with "
                "the KMAH/Maslov phase.  Use apply_real_lens_gbd or "
                "apply_real_lens_fga for caustic-faithful amplitude.",
                RuntimeWarning, stacklevel=2)
    if _imap is not None:
        # The inverse map's full-grid channels are consumed by here: the OPL is
        # ``opl_map``, the amplitude is ``ard_map``, and the entrance pullback
        # went into ``_rd_resid_map`` (whose own alias was just cleared).  Drop
        # the local names -- 2 x N^2 float64 is 1.07 GB at n_fine = 8192 and
        # 4.29 GB at 16384, and holding them to the end of the call would put
        # the peak back where the memory census took it out.
        _im_xin = _im_yin = _im_detj = None
        del _im_xin, _im_yin, _im_detj
    call_progress(progress, 'real_lens_traced', 0.90,
                  'assembling exit field')

    # ----- Step 3: combine amplitude with geom phase -------------------
    # When preserve_input_phase=True (default, physically correct):
    #   We KEEP the full complex E_analytic (which already contains the
    #   input field's phase correctly propagated through the glass
    #   split-step) and APPLY A CORRECTION that replaces the analytic
    #   model's lens-only phase with the ray-traced OPL.
    #
    #   delta_phase = k0 * opl_traced - phase_analytic_lens
    #   E_out = E_analytic * exp(i * delta_phase)
    #
    # This preserves any input-field phase structure (source tilts, MLA
    # patterns, off-axis aberrations) that apply_real_lens correctly
    # carried through.  Before this fix, the input phase was silently
    # discarded -- tilted inputs focused on-axis, MLA-modulated inputs
    # came out as a featureless envelope, etc.
    #
    # When preserve_input_phase=False (legacy behaviour):
    #   E_out = |E_analytic| * exp(i * k0 * opl_traced).  Useful for
    #   measuring the lens-only OPD on a plane-wave input, where the
    #   input-phase question is moot.
    k0 = 2.0 * np.pi / wavelength
    # Preserve the caller's complex dtype: apply_real_lens (called
    # above to build E_analytic / amp) already returns a field in
    # E_in.dtype, but the ``* np.exp(1j * ...)`` multiply here would
    # silently upcast to complex128 unless we cast the exp() result.
    target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    # ---- FIX_TILT_QUADRATIC_OPL_2026_08_11: restore the ABSOLUTE OPL --------
    # ``opl_grid`` (and therefore ``opl_map``) was referenced to the axis
    # launch ray purely to condition the fits -- see the long note at
    # ``_opl_piston``.  Put that constant back, as a unit phasor multiplied
    # onto the exit phase, so the returned field carries the element's true
    # accumulated optical path and a chain can compose legs on an ABSOLUTE
    # piston (the contract ``propagate_traced_carrier_chain`` documents in
    # four places and, before this, did not meet).
    #
    # Multiplied onto ``phase_exp`` rather than added into ``opl_map``: the
    # constant is ~5e+04 rad and the map is ~1e-03 rad, so adding would round
    # the map at 1e-11 rad for no reason, and the multiply also threads the
    # piston through the ray-density magnitude swap for free (that swap keeps
    # ``E_out``'s UNIT PHASOR and replaces only its modulus).
    #
    # NOT applied on the experimental ``inversion_method='backward_trace'``
    # route: that path builds its own map in :func:`_opl_by_backward_trace`,
    # with its own on-axis reference and its own reversed sign convention, so
    # the FORWARD trace's constant is not the one it dropped.  It keeps the
    # pre-fix (piston-free) behaviour, which is what its opt-in status is for.
    _opl_piston_phasor = None
    if _opl_piston != 0.0 and inversion_method != 'backward_trace':
        _opl_piston_phasor = np.exp(1j * (k0 * _opl_piston))
    if return_screen:
        # T-P1 (prepared-traced): the entire traced leg -- ray trace, fits,
        # Newton inversion, phase_analytic_lens, opl_map, and the valid /
        # aperture masks -- is input-independent per
        # (prescription, wavelength, dx, N, carrier).  The ONLY
        # input-dependent factor is E_analytic = apply_real_lens(E_in), which
        # the assembly below multiplies in pointwise (with the masks folding
        # in multiplicatively).  Substituting ones for E_analytic here makes
        # the returned E_out equal the reusable "screen"
        # = mask(valid) * mask(aperture) * exp(1j*(k0*opl - phase_analytic)).
        # prepare_real_lens_traced() caches it; each subsequent call is then
        # one apply_real_lens(E_in) + one complex multiply.  Requires
        # preserve_input_phase=True (else the assembly uses |E_analytic|),
        # newton_amp_mask_rel=0 and tilt_aware_rays=False (else the valid
        # region / opl depend on E_in), and carrier != 'auto'; the factory
        # enforces all four.
        E_analytic = np.ones_like(E_analytic)
    if _chunk_assembly and opl_map is None:
        # Row-band assembly: upsample + delta-phase + combine + masks per
        # (chunk_rows x N) band, writing into E_analytic in place (it is not
        # read again after its own band is consumed).  Element-identical to
        # the whole-grid branch below: map_coordinates interpolates each
        # output point independently from the WHOLE coarse grid (true at
        # ANY spline order -- the prefilter runs on the coarse INPUT, which
        # is never banded), and every other op is pointwise; the band
        # aperture term ``x[j]^2 + x[i]^2`` reproduces
        # ``(X**2 + Y**2)[r0:r1]`` exactly.  The OPL upsample uses the SAME
        # ``_opl_up_order`` (cubic under an engaged carrier, R7) the
        # whole-grid path uses -- see the note at its definition.
        from scipy.ndimage import map_coordinates
        cr = int(sag_chunk_rows)
        r_ap_sq = (aperture / 2) ** 2 if aperture is not None else None
        E_out = E_analytic
        for r0 in range(0, N, cr):
            r1 = min(N, r0 + cr)
            ii_b, jj_b = np.indices((r1 - r0, N), dtype=np.float64)
            if r0:
                ii_b += r0
            # ii/sub, not ii*Ns/N: exact for any sub (see the whole-grid
            # OPL upsample -- same lattice, same walk bug otherwise).
            coords_b = np.array([ii_b / sub, jj_b / sub])
            opl_b = map_coordinates(_opl_coarse_clean, coords_b,
                                    order=_opl_up_order, mode='nearest',
                                    prefilter=(_opl_up_order > 1))
            # NaN mask stays order-1 (crisp ray-domain boundary), exactly as
            # the whole-grid path does -- including its NaN-pass guard, so the
            # banded and whole-grid routes stay element-identical to each
            # other as well as to their pre-guard selves.
            if _nan_coarse is not None:
                nan_b = map_coordinates(_nan_coarse, coords_b,
                                        order=1, mode='nearest')
            else:
                nan_b = None
            del ii_b, jj_b, coords_b
            if nan_b is not None:
                opl_b = np.where(nan_b > 0.5, np.nan, opl_b)
            valid_b = np.isfinite(opl_b)
            if preserve_input_phase:
                dp_b = np.where(
                    valid_b, k0 * opl_b - phase_analytic_lens[r0:r1], 0.0)
            else:
                dp_b = np.where(valid_b, k0 * opl_b, 0.0)
            pe_b = np.exp(1j * dp_b)
            if _opl_piston_phasor is not None:
                pe_b *= _opl_piston_phasor      # absolute-OPL piston (in place)
            if pe_b.dtype != target_cdtype:
                pe_b = pe_b.astype(target_cdtype)
            if preserve_input_phase:
                band = E_analytic[r0:r1] * pe_b
            else:
                band = amp[r0:r1] * pe_b
            band = np.where(valid_b, band, target_cdtype.type(0))
            if r_ap_sq is not None:
                h_b = x[None, :] ** 2 + _y_ax[r0:r1, None] ** 2
                band = np.where(h_b <= r_ap_sq, band,
                                target_cdtype.type(0))
            E_out[r0:r1] = band
        if E_out.dtype != target_cdtype:
            E_out = E_out.astype(target_cdtype)
        call_progress(progress, 'real_lens_traced', 1.0, 'done')
        return E_out
    # ---- PRIVATE DIAGNOSTIC PROBE (niche C15's independent oracle) ---------
    # A caller that passes ``_imap_out`` carrying ``probe_rc = (rows, cols)``
    # gets the FINALISED ``opl_map`` -- and ``ard_map`` where one was built --
    # sampled at exactly those pixels, in the same dict.  Same contract as
    # ``_exit_na_out``: private, opt-in, pure-diagnostic, nothing reads it
    # back, and it retains only the M sampled values, never a full-grid array,
    # so a call that does not ask pays one dict lookup and keeps every bit and
    # every byte.
    #
    # WHY IT EXISTS.  Deciding which INVERSION is faithful needs the map each
    # arm actually built, and the returned field carries it only through the
    # amplitude model, the piston phasor and the residual transport -- three
    # stages that are common to both arms and would only add noise to the
    # comparison.  Taken HERE because this is the one point every non-chunked
    # branch converges on with ``opl_map`` final.
    if _imap_out is not None and _imap_out.get('probe_rc') is not None:
        _p_r = np.asarray(_imap_out['probe_rc'][0], dtype=np.intp)
        _p_c = np.asarray(_imap_out['probe_rc'][1], dtype=np.intp)
        _imap_out['probe_opl'] = np.asarray(opl_map)[_p_r, _p_c].copy()
        _imap_out['probe_ard'] = (None if ard_map is None else
                                  np.asarray(ard_map)[_p_r, _p_c].copy())
        _imap_out['probe_opl_piston'] = float(_opl_piston)
        del _p_r, _p_c

    valid = np.isfinite(opl_map)
    # v5.16.2: free each full-grid intermediate as soon as its consumer is
    # built (delta_phase/phase after phase_exp; phase_exp after E_out;
    # opl_map after the phase build).  Pure lifetime fixes -- values and
    # outputs unchanged.
    if preserve_input_phase:
        delta_phase = np.where(valid, k0 * opl_map - phase_analytic_lens, 0.0)
        del opl_map
        phase_exp = np.exp(1j * delta_phase)
        del delta_phase
        if _opl_piston_phasor is not None:
            phase_exp *= _opl_piston_phasor     # absolute-OPL piston (in place)
        if phase_exp.dtype != target_cdtype:
            phase_exp = phase_exp.astype(target_cdtype)
        E_out = E_analytic * phase_exp
        del phase_exp
    else:
        phase = np.where(valid, k0 * opl_map, 0.0)
        del opl_map
        phase_exp = np.exp(1j * phase)
        del phase
        if _opl_piston_phasor is not None:
            phase_exp *= _opl_piston_phasor     # absolute-OPL piston (in place)
        if phase_exp.dtype != target_cdtype:
            phase_exp = phase_exp.astype(target_cdtype)
        E_out = amp * phase_exp
        del phase_exp
    # ...and the analytic field itself.  The assembly above is its LAST reader
    # on BOTH branches (``preserve_input_phase='remap'`` sets the flag False at
    # its normalisation, so the remap path arrives here through the ``amp``
    # branch, and the ray-density swap below divides the screen modulus out and
    # reads ``amp``, never ``E_analytic``).  4.295 GB complex128 at
    # n_fine = 16384, held to the end of the call.  Pure lifetime.
    del E_analytic
    # Zero outside the exit-pupil (ray-coverage) region
    E_out = np.where(valid, E_out, target_cdtype.type(0))
    del valid                  # full-grid bool, 0.268 GB at n_fine=16384
    # And outside the entrance aperture (defensive: in practice the
    # ray-coverage region is a subset of the entrance aperture, so
    # this is a no-op except in pathological configurations)
    if aperture is not None:
        E_out = np.where(X ** 2 + Y ** 2 <= (aperture / 2) ** 2,
                         E_out, target_cdtype.type(0))
    # ---- N12 (P11): swap the exit MAGNITUDE to the ray-density amplitude -----
    # The screen-mode ``E_out`` above carries the correct traced OPL phase and
    # the valid / aperture masks (it is 0 outside the ray-covered pupil).  In
    # ray-density mode we keep that phase (its unit phasor) and replace the
    # magnitude with ``|E_in|/sqrt(|det J|)`` -- the geometric ray-tube energy
    # redistribution the screen amplitude lacks.  The unit phasor is 0 exactly
    # where the screen field is 0 (masked region), so the ray-density field
    # inherits the same support without a separate mask; NaN ray-density values
    # (out-of-domain) contribute 0.
    if _ray_density and ard_map is not None:
        _absE = np.abs(E_out)
        with np.errstate(divide='ignore', invalid='ignore'):
            _unit = np.divide(E_out, _absE,
                              out=np.zeros_like(E_out), where=_absE > 0)
        del _absE              # consumed by the divide; 2.147 GB float64
        _ard = np.where(np.isfinite(ard_map), ard_map, 0.0)
        ard_map = None         # consumed by the where; 2.147 GB float64
        # ---- niche D9: the analytic amplitude leg's ZERO SET, measured -------
        # ``amp`` is the ONE thing an axis-centred ``apply_real_lens`` still
        # contributes here, and it contributes only where it is exactly 0 (the
        # swap above divides its modulus out).  Under a decentred ``origin``
        # that leg placed the element's sag, its aperture / clear_aperture
        # masks and its stop about the wrong transverse point, so its zeros are
        # in the wrong place -- harmlessly, IF they do not overlap the beam.
        # That is measurable, so it is measured rather than assumed.  See
        # ``ORIGIN_AMP_SUPPORT_CHECK``.
        if _origin_set and ORIGIN_AMP_SUPPORT_CHECK != 'silent':
            _oa_w = _ard * _ard
            _oa_tot = float(_oa_w.sum())
            _oa_cut = float(_oa_w[amp <= 0.0].sum()) if _oa_tot > 0.0 else 0.0
            _oa_frac = (_oa_cut / _oa_tot) if _oa_tot > 0.0 else 0.0
            del _oa_w
            if _oa_frac > _ORIGIN_AMP_SUPPORT_TOL:
                _oa_msg = (
                    f"apply_real_lens_traced: origin=({_org_x * 1e3:+.4f}, "
                    f"{_org_y * 1e3:+.4f}) mm decentres the WAVE GRID, but the "
                    f"analytic amplitude leg (apply_real_lens) has no origin "
                    f"and built this element -- its sag, its "
                    f"aperture_diameter / clear_aperture masks and its stop -- "
                    f"about the GRID centre, which is not the optical axis "
                    f"here.  That leg reaches the returned field ONLY through "
                    f"its zero set, and its zeros are deleting "
                    f"{_oa_frac * 100:.6f} % of the ray-density exit power "
                    f"(tolerance {_ORIGIN_AMP_SUPPORT_TOL * 100:.1e} %, i.e. "
                    f"the intended value is zero).  The light removed is real "
                    f"beam clipped by a stop placed at the wrong transverse "
                    f"position, NOT physical vignetting -- the physical "
                    f"entrance stop is applied separately, on absolute "
                    f"entrance coordinates.  Remedies: shrink the grid (or "
                    f"raise window_factor's beam multiple) so the decentred "
                    f"window stays clear of the analytic leg's masks; widen or "
                    f"drop aperture_diameter / the per-surface clear_aperture "
                    f"on the prescription handed to this call (the ray leg "
                    f"already enforces the true stop); or keep origin=(0, 0) "
                    f"and pay for the axis-centred window.  Set "
                    f"lumenairy.elements._lens_traced.ORIGIN_AMP_SUPPORT_CHECK "
                    f"= 'warn' to accept the deletion, 'silent' to stop "
                    f"measuring it.")
                if ORIGIN_AMP_SUPPORT_CHECK == 'error':
                    raise NotImplementedError(_oa_msg)
                import warnings as _oa_warn
                _oa_warn.warn(_oa_msg, RuntimeWarning, stacklevel=2)
        E_out = _ard * _unit
        # The ray-density magnitude and the unit phasor are consumed by that
        # one multiply -- 2.147 + 4.295 GB at n_fine = 16384, both caught live
        # in the census (sec 2.3) while the readout was building its own grids.
        del _ard, _unit
        # preserve_input_phase='remap': multiply the geometrically-transported
        # input-residual phasor onto the k0*opl exit phase (audit S6.7).  Unit
        # modulus by construction, identity where the pullback is undefined.
        if _rd_resid_map is not None:
            E_out = E_out * _rd_resid_map.astype(E_out.dtype, copy=False)
            _rd_resid_map = None       # consumed; 4.295 GB complex128
        # ---- v5.30 (audit E-M6): post-hoc ENERGY SELF-CHECK ------------------
        # Two N^2 reductions, negligible against the trace + Newton stages.
        # Reference = the input power the element ADMITS (inside the entrance
        # aperture), which is what the ray-tube map transports; comparing
        # against the whole grid would flag legitimate vignetting (measured:
        # 0.935 vs 0.990 on the same 1.2x aperture:beam cell).
        _rd_pin = np.abs(np.asarray(E_in, dtype=np.complex128)) ** 2
        if aperture is not None:
            _rd_pin = np.where(X ** 2 + Y ** 2 <= (aperture / 2) ** 2,
                               _rd_pin, 0.0)
        _rd_p_in = float(_rd_pin.sum())
        del _rd_pin
        _rd_p_out = float((np.abs(E_out) ** 2).sum())
        if _rd_p_in > 0.0:
            _rd_ratio = _rd_p_out / _rd_p_in
            _rd_lo = 1.0 - (_RD_ENERGY_DEFICIT_BASE
                            + _RD_ENERGY_DEFICIT_PER_SUB * (sub - 1))
            _rd_hi = 1.0 + _RD_ENERGY_GAIN_TOL
            if not (_rd_lo <= _rd_ratio <= _rd_hi):
                import warnings as _rd_ewarn
                _rd_ewarn.warn(
                    f"apply_real_lens_traced: amplitude_model='ray_density' "
                    f"energy self-check FAILED -- exit power / "
                    f"aperture-transmitted input power = {_rd_ratio:.6f}, "
                    f"outside the documented band "
                    f"[{_rd_lo:.4f}, {_rd_hi:.4f}] for ray_subsample={sub}.  "
                    f"The ray-tube amplitude is energy-conserving only in the "
                    f"GEOMETRIC LIMIT; at finite ray_subsample it loses about "
                    f"1% at the shipped ray_subsample=8 (design-battery "
                    f"envelope 0.9535-0.9920, converging to 0.9569-1.0000 at "
                    f"ray_subsample=1), and this band is set clear of that "
                    f"whole envelope.  A ratio this far off usually means "
                    f"something else: a fold caustic capping |det J| (see the "
                    f"fold-caustic warning -- use apply_real_lens_gbd / "
                    f"apply_real_lens_fga there), a ray map running off the "
                    f"grid, or an aperture_diameter wider than the traced "
                    f"pupil.  Lower ray_subsample to check convergence.",
                    RuntimeWarning, stacklevel=2)
        # ---- v5.32: HALO-AMPLITUDE self-check --------------------------
        # The power sum above cannot see a lobe deposited outside the traced
        # pupil (measured: a defect whose total-power signature vanished
        # while the lobe stayed at 77 % of peak).  This one can.  Radius and
        # centroid come from the EXACT ray trace of this very call; see
        # ``_RD_HALO_AMAX_TOL`` for the derivation and the calibration.
        if _rd_hull_r is not None and _rd_hull_r > 0.0:
            _hb = _RD_HALO_RADIUS_FACTOR * _rd_hull_r
            # The annulus has to be a GENUINE annulus about the traced exit
            # centroid.  Once the bound circle runs off the grid all that is
            # left of it is a sliver of corners, and the statistic measured
            # there is unreliable in BOTH directions -- measured, twice (see
            # SCOPE (d) at ``_RD_HALO_AMAX_TOL``).  Decline rather than report
            # a number that cannot be trusted either way.
            # niche D9: the y half of this edge test compared the y centroid
            # against the X axis's extent.  On a square axis-centred grid the
            # two axes are the SAME vector, so it was invisible; with the grid
            # centre off axis they separate and it would be a real defect.
            # ``_y_ax is x`` on axis, so this is byte-identical there.
            _h_edge = min(float(x[-1]) - _rd_hull_c[0],
                          _rd_hull_c[0] - float(x[0]),
                          float(_y_ax[-1]) - _rd_hull_c[1],
                          _rd_hull_c[1] - float(_y_ax[0]))
            _h_far = (((x[None, :] - _rd_hull_c[0]) ** 2
                       + (_y_ax[:, None] - _rd_hull_c[1]) ** 2) > _hb ** 2
                      if _hb <= _h_edge else np.zeros((1, 1), dtype=bool))
            if _h_far.any():
                _h_abs = np.abs(E_out)
                _h_pkE = float(_h_abs.max())
                if _h_pkE > 0.0:
                    _h_amax = float(_h_abs[_h_far].max()) / _h_pkE
                    _h_gpow = (float((_h_abs[_h_far] ** 2).sum()) / _rd_p_in
                               if _rd_p_in > 0.0 else 0.0)
                    if _h_amax > _RD_HALO_AMAX_TOL:
                        import warnings as _rd_hwarn
                        _rd_hwarn.warn(
                            f"apply_real_lens_traced: amplitude_model="
                            f"'ray_density' HALO self-check FAILED -- "
                            f"amax_halo = {_h_amax:.3e} of peak beyond "
                            f"{_hb * 1e3:.4f} mm "
                            f"({_RD_HALO_RADIUS_FACTOR:g} x the exact-ray "
                            f"exit support radius {_rd_hull_r * 1e3:.4f} mm "
                            f"about the traced exit centroid "
                            f"({_rd_hull_c[0] * 1e3:+.4f}, "
                            f"{_rd_hull_c[1] * 1e3:+.4f}) mm), against a "
                            f"tolerance of {_RD_HALO_AMAX_TOL:.1e}; that "
                            f"halo carries g_halo = {_h_gpow:.3e} of the "
                            f"aperture-transmitted input power, and the grid "
                            f"reaches {_h_edge * 1e3:.4f} mm from that "
                            f"centroid.  NO TRACED "
                            f"RAY OF THIS CALL REACHES THAT RADIUS, so the "
                            f"light there was manufactured, not merely "
                            f"misplaced: the usual cause is the fitted "
                            f"entrance->exit map being Newton-inverted "
                            f"outside its own data support and the "
                            f"ray-density amplitude handing the spurious "
                            f"root real power.  Note the energy self-check "
                            f"CANNOT see this -- a lobe of a few 1e-3 of the "
                            f"input power sits well inside its band.  Try a "
                            f"different fit_radius_beam_factor, "
                            f"newton_fit='spline', or a caustic-faithful "
                            f"propagator (apply_real_lens_gbd / "
                            f"apply_real_lens_fga); set "
                            f"lumenairy.elements._lens_traced."
                            f"RAY_DENSITY_HALO_CHECK = 'silent' to suppress.",
                            RuntimeWarning, stacklevel=2)
                del _h_abs
            del _h_far
        # ---- niche C14: the RETAINED-BAND self-check --------------------
        # The blind spot between the two checks above, closed.  C8 keeps a
        # band outside the traced hull ON PURPOSE (the sqrt(2) sub dx plateau
        # that makes the upsample's bleed identically zero, plus the feather),
        # and C7 looks only beyond 1.25 x r_hull -- which under the bound is
        # territory C8 has already zeroed.  So on the E-M6 fixture 0.19998 of
        # P_ap of manufactured light, carrying the field's GLOBAL maximum, sat
        # in a band that neither check watches and the energy check reads as
        # 1.01931, inside its own band (RECON_PINS_POST_C8_2026_08_01 S7.1).
        #
        # This asks the one question that needs no new calibration: does the
        # field peak somewhere no traced ray of this call reached?  A skirt
        # decays outward and cannot; a manufactured lobe does.  See
        # ``SUPPORT_BAND_CHECK`` for why the criterion is a RATIO.
        if (SUPPORT_BAND_CHECK != 'silent'
                and _exit_support.hull is not None):
            _bd_in, _bd_band = _exit_support.retained_band_masks(
                x, _y_ax, np.sqrt(2.0) * sub * dx)
            if _bd_band is not None and _bd_band.any() and _bd_in.any():
                _bd_abs = np.abs(E_out)
                _bd_core = float(_bd_abs[_bd_in].max())
                _bd_amax = float(_bd_abs[_bd_band].max())
                if (_bd_core > 0.0
                        and _bd_amax > _SUPPORT_BAND_PEAK_RATIO_TOL * _bd_core):
                    _bd_gpow = (float((_bd_abs[_bd_band] ** 2).sum()) / _rd_p_in
                                if _rd_p_in > 0.0 else 0.0)
                    import warnings as _bd_warn
                    _bd_warn.warn(
                        f"apply_real_lens_traced: amplitude_model="
                        f"'ray_density' SUPPORT-BAND self-check FAILED -- the "
                        f"field's maximum in the band the C8 exit-support "
                        f"bound RETAINS outside the traced hull is "
                        f"{_bd_amax:.3e}, which is "
                        f"{_bd_amax / max(_bd_core, 1e-300):.3f}x the maximum "
                        f"INSIDE the traced support ({_bd_core:.3e}), against "
                        f"a ratio tolerance of "
                        f"{_SUPPORT_BAND_PEAK_RATIO_TOL:g}.  That band is the "
                        f"sqrt(2)*ray_subsample*dx plateau plus "
                        f"{_SUPPORT_BOUND_FEATHER_CELLS:g} exit-lattice cells "
                        f"of feather ({(np.sqrt(2.0) * sub * dx + (_exit_support.feather or 0.0)) * 1e3:.4f} mm "
                        f"wide, on a hull of {_exit_support.n_hull} facets), "
                        f"and it carries g_band = {_bd_gpow:.3e} of the "
                        f"aperture-transmitted input power.  NO TRACED RAY OF "
                        f"THIS CALL LANDED THERE: the plateau exists to stop "
                        f"the bilinear upsample eating legitimate skirt, not "
                        f"to host a lobe, and a field whose GLOBAL MAXIMUM "
                        f"lies outside its own traced support has "
                        f"manufactured that light rather than misplaced it.  "
                        f"Note that neither of the other two instruments can "
                        f"see this band: the ray-density power band reads it "
                        f"as a fraction of a per cent, and the halo report "
                        f"covers only radii beyond "
                        f"{_RD_HALO_RADIUS_FACTOR:g} x r_hull, which the "
                        f"bound has already zeroed.  Try a different "
                        f"fit_radius_beam_factor, newton_fit='spline', a "
                        f"lower ray_subsample, or a caustic-faithful "
                        f"propagator (apply_real_lens_gbd / "
                        f"apply_real_lens_fga); set "
                        f"lumenairy.elements._lens_traced."
                        f"SUPPORT_BAND_CHECK = 'silent' to suppress (that is "
                        f"also the pre-C14 fail-before).",
                        RuntimeWarning, stacklevel=2)
                del _bd_abs
            del _bd_in, _bd_band
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    call_progress(progress, 'real_lens_traced', 1.0, 'done')
    return E_out


def _carrier_reuse_key(carrier):
    """Hashable key for a carrier that is SAFE to share a prepared screen
    across emitters, or None if it must get its own trace.  'auto' fits the
    carrier from each field, so every emitter's 'auto' carrier is different ->
    NOT reusable.  An ndarray wavefront is per-emitter data -> not reusable.
    A float conjugate distance or None (plane wave) is a shared geometry ->
    reusable."""
    if carrier is None:
        return ('none',)
    if isinstance(carrier, str):
        return None            # 'auto' (or any string mode): never share
    if isinstance(carrier, np.ndarray):
        return None            # explicit per-field wavefront: never share
    if isinstance(carrier, TiltedCarrier):
        # analytic geometry, no per-field data: shareable like a scalar
        return ('tilted',) + tuple(float(v) for v in carrier)
    try:
        return ('scalar', float(carrier))
    except (TypeError, ValueError):
        return None


def apply_real_lens_traced_multi(
    emitter_fields,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    carriers: Any = 'auto',
    reuse_prepared: bool = True,
    **traced_kwargs,
) -> np.ndarray:
    """Coherently sum the traced lens applied to each emitter's field SEPARATELY.

    The traced model assigns one ray-traced OPL per output pixel (the dominant
    congruence), so it is **not linear**: when several emitter beams overlap on
    a pixel, ``traced(sum_k E_k)`` violates the single-OPL assumption.  Each
    emitter taken ALONE is a single congruence, so ``traced(E_k, carrier_k)`` is
    valid, and this returns their coherent sum::

        E_image = sum_k  apply_real_lens_traced(E_k, carrier=carrier_k, ...)

    which is the tractable form of carrier K-decomposition -- the K congruences
    are the *known* emitters, so no blind congruence segmentation is needed.
    It is exact for a single emitter and reproduces every per-emitter congruence
    correctly.

    **When this actually helps (read before using).**  The value is regime-
    dependent, and this is NOT a universal upgrade:

    * The analytic :func:`apply_real_lens` is **exactly linear**, so for it
      ``analytic(sum E_k) == sum analytic(E_k)`` -- there is *zero* benefit; just
      propagate the combined field once.
    * For a **well-corrected** lens the traced OPD correction is ~0, i.e.
      ``traced ~= analytic``, so ``traced(sum E_k)`` is already essentially exact
      and this per-emitter path only *adds* the per-emitter carrier-fit residual.
    * This mode earns its keep only when you genuinely need the traced ray-OPD
      **refinement** (a lens aberrated enough that analytic is insufficient) AND
      the scene is multi-emitter.  There the traced non-linearity is large
      (>100% between ``traced(sum)`` and this sum on strongly-aberrated lenses),
      and applying traced per emitter is the correct way to keep it valid.
    * Note the no-MLA multi-angle *direct-imaging* case was separately found to
      be modelled correctly by **analytic**, not traced -- so for that geometry
      prefer analytic on the combined field; this mode is for when traced's
      refinement is the thing you specifically want.

    Composes with T-P1: pass a shared explicit ``carrier`` (or ``None``) with
    ``reuse_prepared=True`` to pay the trace/fit/Newton cost once across all
    emitters that share it (see ``reuse_prepared``).

    Parameters
    ----------
    emitter_fields : sequence of complex ndarray
        Each ``E_k`` is the field AT THE LENS-INPUT PLANE from emitter ``k``
        alone (propagate each emitter to the lens plane first).  All must share
        the grid.
    carriers : 'auto' | None | float | ndarray | TiltedCarrier | sequence
        Per-emitter carrier passed to :func:`apply_real_lens_traced`.  A scalar
        / string / single ndarray / single :class:`TiltedCarrier` is broadcast
        to every emitter; a list/tuple of length ``len(emitter_fields)`` is used
        element-wise.  Default ``'auto'`` fits each emitter's own congruence
        (drives its residual angular spread to ~0), which is what a divergent
        point-source array needs.

        A :class:`TiltedCarrier` is a ``NamedTuple``, i.e. a ``tuple``: it is
        matched BEFORE the sequence test, so one shared spec broadcasts instead
        of being unpacked into five per-emitter carriers.  Pass ``[tc] * n``
        (or a list of distinct specs) for the genuine per-emitter case.
    reuse_prepared : bool
        When True and a carrier is a shared geometry (``None`` or a float
        conjugate distance), a :class:`PreparedTracedLens` screen is built once
        per distinct carrier and reused across emitters -- the trace/fit/Newton
        cost is paid once instead of per emitter.  ``'auto'`` and ndarray
        carriers are always full per-emitter passes (their screens differ).

    Returns
    -------
    E_image : complex ndarray
        The coherently-summed output field, dtype following the emitter fields.

    Notes
    -----
    RELATION TO THE v5.29 CHAIN DEFAULTS (audit
    AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 §8.5 / this sibling sweep).  This
    entry point CANNOT run the validated carrier-regime configuration
    ``propagate_traced_carrier_chain`` defaults to, and the reasons are
    structural, not oversights:

    * ``preserve_input_phase`` is FORCED ``True`` (the per-emitter contract:
      each emitter's own phase must ride through).  It used to be popped from
      ``traced_kwargs`` SILENTLY, so ``preserve_input_phase='remap'`` returned a
      field byte-identical to ``True`` (measured: 0.0 difference, while the
      direct element call differs by 7.5e-3) -- a caller could not tell the
      requested mode had been discarded.  A conflicting value now RAISES.
    * ``amplitude_model='ray_density'`` and ``fit_radius_beam_factor`` are
      input-DEPENDENT (the ray-tube amplitude scales ``|E_in|``; the fit radius
      is measured from the beam), so they cannot be baked into a prepared
      screen: they used to fail with an opaque ``TypeError`` from
      :func:`prepare_real_lens_traced` on the DEFAULT ``reuse_prepared=True``
      path while working on ``reuse_prepared=False``.  They are now rejected at
      this entry point with the remedy named, and pass through on the
      per-emitter path.

    Consequence for accuracy: a multi-emitter scene therefore runs the
    ``screen`` amplitude and, with ``reuse_prepared=True``, WITHOUT the P2
    aperture:beam cliff guard.  Pass ``reuse_prepared=False,
    fit_radius_beam_factor=2.0`` (optionally
    ``amplitude_model='ray_density'``) to get the chain-grade per-emitter
    treatment at the cost of one full traced pass per emitter.
    """
    fields = list(emitter_fields)
    if not fields:
        raise ValueError("apply_real_lens_traced_multi: emitter_fields is empty.")
    n = len(fields)
    shape0 = np.asarray(fields[0]).shape
    for k, E in enumerate(fields):
        if np.asarray(E).shape != shape0:
            raise ValueError(
                f"apply_real_lens_traced_multi: emitter_fields[{k}] shape "
                f"{np.asarray(E).shape} != emitter_fields[0] {shape0}.")

    # D1 (2026-07-28): a :class:`TiltedCarrier` IS a tuple (NamedTuple), so the
    # sequence test below would take its five FIELDS as five per-emitter
    # carriers.  Measured with n = 5: the five emitters were traced with scalar
    # conjugate distances (R, L, M, x0, y0) metres, silently -- 2.61x low total
    # power, relative field error 1.74, no carrier-related error or warning; for
    # n != 5 it raised "carriers list length 5 != number of emitters", naming a
    # list the caller never wrote.  A single TiltedCarrier is ONE carrier and
    # broadcasts like a scalar.  (``propagate_traced_carrier_chain`` was never
    # affected -- its ``_parse_chain_carrier`` isinstance-checks TiltedCarrier
    # first -- so this hole was in the ``_multi`` dispatcher alone.)
    if isinstance(carriers, TiltedCarrier):
        carr_list = [carriers] * n
    elif isinstance(carriers, (list, tuple)):
        if len(carriers) != n:
            raise ValueError(
                f"apply_real_lens_traced_multi: carriers list length "
                f"{len(carriers)} != number of emitters {n}.")
        carr_list = list(carriers)
    else:
        carr_list = [carriers] * n     # scalar / str / single ndarray broadcast

    # Each per-emitter pass runs full-grid Newton (no amp mask) so an emitter's
    # OWN dim regions are never clipped -- they may still contribute where a
    # later emitter is bright, and the reuse path (prepared screen) already
    # forces this.  tilt_aware_rays is off (the carrier carries the tilt) and
    # the phase is preserved.  These values are FIXED by the per-emitter
    # contract.  Pre-v5.29 they were popped SILENTLY, so a caller asking for
    # e.g. ``preserve_input_phase='remap'`` got ``True`` with no diagnostic
    # (measured byte-identical to True while the direct element call differs by
    # 7.5e-3).  Now: a value EQUAL to the forced one is accepted (no behaviour
    # change), anything else raises with the reason.
    _FORCED = {'newton_amp_mask_rel': 0.0, 'tilt_aware_rays': False,
               'preserve_input_phase': True, 'return_screen': False,
               'parallel_amp': False}
    _FORCED_WHY = {
        'newton_amp_mask_rel':
            'each emitter must be Newton-inverted on the FULL coarse grid (its '
            'own dim regions can still contribute where another emitter is '
            'bright)',
        'tilt_aware_rays':
            'the per-emitter carrier already carries the tilt; the per-pixel '
            'tilt launch would fight it (audit F3)',
        'preserve_input_phase':
            "each emitter's own input phase must ride through the traced leg "
            "for the coherent sum to mean anything -- 'remap'/False would "
            'discard or re-transport it per emitter',
        'return_screen':
            'this function returns a summed FIELD, not a reusable screen (use '
            'prepare_real_lens_traced directly for that)',
        'parallel_amp':
            'the per-emitter loop is already the concurrency axis',
    }
    for _k, _forced in _FORCED.items():
        if _k not in traced_kwargs:
            continue
        _got = traced_kwargs.pop(_k)
        if _got is _forced or _got == _forced:
            continue
        raise ValueError(
            f"apply_real_lens_traced_multi: {_k}={_got!r} conflicts with the "
            f"per-emitter contract, which fixes {_k}={_forced!r} -- "
            f"{_FORCED_WHY[_k]}.  Pre-v5.29 this argument was dropped "
            f"SILENTLY, so the call returned a field for {_k}={_forced!r} "
            f"while appearing to honour the request.  Drop the argument, or "
            f"call apply_real_lens_traced per emitter yourself if you need "
            f"this mode.")

    # Keys the PREPARED-screen path structurally cannot express (they are
    # input-dependent, so a screen built on the ``ones`` placeholder would be
    # wrong): reject them at the entry point with the remedy, instead of the
    # opaque ``TypeError: prepare_real_lens_traced() got an unexpected keyword
    # argument`` they used to raise from three frames down -- and only on the
    # DEFAULT reuse path, so the same call worked or crashed depending on the
    # carrier kind.
    _NO_SCREEN = {
        'amplitude_model':
            "the ray-density amplitude scales |E_in|, so it cannot be baked "
            "into an input-independent screen",
        'fit_radius_beam_factor':
            "the fit radius is measured from the input beam, and the prepared "
            "screen is built on a flat ``ones`` placeholder whose 'beam' is "
            "meaningless",
        'on_aperture_beam':
            "the aperture:beam ratio is measured from the input beam, which "
            "the prepared screen's ``ones`` placeholder does not have",
        'beam_centre':
            "it only positions the beam-relative fit disc, which the prepared "
            "screen has no beam to build (a decentred TiltedCarrier carrier "
            "already states the chief-ray position for the per-emitter path)",
        'decentred_fit_poly_order':
            "it only orders the OFF-CENTRE ray fit, and the prepared screen "
            "has no beam to place a disc around (niche D7)",
        'caustic':
            "the multibranch / uniform caustic modes require the ray-density "
            "amplitude, hence the per-emitter path",
    }
    # Only when a carrier ACTUALLY routes to a prepared screen: ``'auto'`` /
    # ndarray carriers always take the per-emitter path, where these modes work.
    if reuse_prepared and any(_carrier_reuse_key(c) is not None
                              for c in carr_list):
        _bad = sorted(k for k in _NO_SCREEN if k in traced_kwargs)
        if _bad:
            raise ValueError(
                "apply_real_lens_traced_multi: "
                + '; '.join(f"{k}={traced_kwargs[k]!r} cannot be used with "
                            f"reuse_prepared=True ({_NO_SCREEN[k]})"
                            for k in _bad)
                + ".  Pass reuse_prepared=False to run a full traced pass per "
                  "emitter (these modes are honoured there), or drop the "
                  "argument to keep the shared prepared screen.")

    N = int(shape0[0])
    prepared_cache = {}
    E_out = None
    for E_k, carrier_k in zip(fields, carr_list):
        E_k = np.asarray(E_k)
        key = _carrier_reuse_key(carrier_k) if reuse_prepared else None
        if key is not None:
            prep = prepared_cache.get(key)
            if prep is None:
                prep = prepare_real_lens_traced(
                    prescription=prescription, wavelength=wavelength, dx=dx,
                    N=N, carrier=carrier_k, **traced_kwargs)
                prepared_cache[key] = prep
            contrib = prep(E_k)
        else:
            contrib = apply_real_lens_traced(
                E_k, prescription=prescription, wavelength=wavelength, dx=dx,
                carrier=carrier_k, newton_amp_mask_rel=0.0,
                tilt_aware_rays=False, preserve_input_phase=True,
                parallel_amp=False, **traced_kwargs)
        E_out = contrib if E_out is None else E_out + contrib
    return E_out


def _flattop_partition_1d(u, cuts, halfwidth):
    """Flat-top cos^2-edge partition of unity over axis ``u`` split at ``cuts``.

    ``len(cuts)+1`` weight arrays (same shape as ``u``), each ~1 in its bin
    interior with a ``cos^2``/``sin^2`` transition of half-width ``halfwidth``
    centred on each cut, so adjacent bins hand off as ``cos^2 + sin^2 = 1`` and
    the whole set **sums to 1**.  Unlike a uniform partition, the transitions
    sit AT the cuts (spectral gaps between congruences), so each whole beam
    lands in one bin instead of being fragmented across bins -- which is what
    makes the per-segment traced pass valid (one congruence per segment).
    Requires ``halfwidth`` below half the smallest cut spacing for exact unity.
    """
    cuts = sorted(float(c) for c in cuts)
    K = len(cuts) + 1
    if K == 1:
        return [np.ones(np.shape(u), dtype=float)]
    hw = max(float(halfwidth), 1e-30)
    W = []
    for k in range(K):
        w = np.ones(np.shape(u), dtype=float)
        if k > 0:                       # rising edge at cuts[k-1]
            s = np.clip((u - (cuts[k - 1] - hw)) / (2.0 * hw), 0.0, 1.0)
            w = w * np.sin(np.pi * s / 2.0) ** 2
        if k < K - 1:                   # falling edge at cuts[k]
            s = np.clip((u - (cuts[k] - hw)) / (2.0 * hw), 0.0, 1.0)
            w = w * np.cos(np.pi * s / 2.0) ** 2
        W.append(w)
    return W


def _occupied_freq_support(power_1d, freqs, frac):
    """Frequency bounds ``(lo, hi)`` of the marginal-power support capturing
    ``frac`` of the total (the highest-power bins)."""
    total = float(power_1d.sum())
    if total <= 0.0:
        return float(freqs[0]), float(freqs[-1])
    order = np.argsort(power_1d)[::-1]
    cum = np.cumsum(power_1d[order]) / total
    klast = int(np.searchsorted(cum, frac)) + 1
    keep = order[:max(1, klast)]
    return float(freqs[keep].min()), float(freqs[keep].max())


def _spectral_gap_cuts(marginal, freqs, lo, hi, valley_frac, peak_frac):
    """Cut frequencies at deep valleys of the 1-D marginal angular power --
    the gaps that SEPARATE distinct beams (congruences).  A valley qualifies as
    a cut only if its power is below ``valley_frac`` of the marginal peak AND it
    is flanked (within the occupied ``[lo, hi]``) by peaks above ``peak_frac``,
    so a single (unimodal) congruence yields NO cuts (one segment = plain
    traced) and only genuinely separated beams are split."""
    p = np.asarray(marginal, dtype=float)
    pk = float(p.max())
    if pk <= 0.0:
        return []
    p = p / pk
    inband = (freqs >= lo) & (freqs <= hi)
    idx = np.where(inband)[0]
    cuts = []
    for i in idx:
        if i <= 0 or i >= len(p) - 1:
            continue
        if p[i] <= p[i - 1] and p[i] < p[i + 1] and p[i] < valley_frac:
            if p[:i].max() > peak_frac and p[i + 1:].max() > peak_frac:
                cuts.append(float(freqs[i]))
    # merge cuts that are closer than a few samples (same valley)
    if cuts:
        merged = [cuts[0]]
        df = float(abs(freqs[1] - freqs[0])) * 3.0
        for c in cuts[1:]:
            if c - merged[-1] > df:
                merged.append(c)
        cuts = merged
    return cuts


def _segment_field_by_angle(E, dx, dy, segments_x, segments_y,
                            power_frac, valley_frac, min_segment_power,
                            max_segments):
    """Partition ``E`` into angular sub-fields at the spectral GAPS between
    beams, so each sub-field is a single congruence.  With
    ``min_segment_power <= 0`` the segments sum to ``E`` EXACTLY.  Returns a
    single segment (the input) when the spectrum is unimodal (nothing to
    separate)."""
    E = np.asarray(E)
    Ny, Nx = E.shape[-2], E.shape[-1]
    F = np.fft.fftshift(np.fft.fft2(E))
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))
    P = np.abs(F) ** 2
    lox, hix = _occupied_freq_support(P.sum(axis=0), fx, power_frac)
    loy, hiy = _occupied_freq_support(P.sum(axis=1), fy, power_frac)

    if segments_x == 'auto':
        cutx = _spectral_gap_cuts(P.sum(axis=0), fx, lox, hix, valley_frac, 0.25)
    else:
        nseg = max(1, int(segments_x))
        cutx = ([] if nseg == 1
                else list(np.linspace(lox, hix, nseg + 1)[1:-1]))
    if segments_y == 'auto':
        cuty = _spectral_gap_cuts(P.sum(axis=1), fy, loy, hiy, valley_frac, 0.25)
    else:
        nseg = max(1, int(segments_y))
        cuty = ([] if nseg == 1
                else list(np.linspace(loy, hiy, nseg + 1)[1:-1]))

    # cap total segments: drop the SHALLOWEST cuts first (D15) -- the valley
    # with the highest marginal power is the weakest separation, so removing it
    # keeps the deepest (best-separating) gaps.  (Previously popped the
    # last-listed = highest-frequency cut, contradicting the comment.)
    mx, my = P.sum(axis=0), P.sum(axis=1)

    def _valley_power(cut, marg, freqs):
        return float(marg[int(np.argmin(np.abs(freqs - cut)))])

    while (len(cutx) + 1) * (len(cuty) + 1) > max_segments:
        cand = ([("x", i, _valley_power(c, mx, fx))
                 for i, c in enumerate(cutx)]
                + [("y", i, _valley_power(c, my, fy))
                   for i, c in enumerate(cuty)])
        if not cand:
            break
        ax, idx, _ = max(cand, key=lambda t: t[2])   # shallowest = most power
        (cutx if ax == "x" else cuty).pop(idx)

    def _hw(cuts, lo, hi):
        # transition half-width: below half the smallest cut spacing (and to
        # the band edges) so the partition stays a partition of unity.
        edges = [lo] + sorted(cuts) + [hi]
        gaps = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        # narrow transition (sharp separation, since the cut sits in a near-
        # zero-power gap so Gibbs ringing is negligible), but < half the
        # smallest spacing so the partition stays a partition of unity.
        return 0.2 * min(gaps) if gaps else (hi - lo)

    hwx = _hw(cutx, lox, hix)
    hwy = _hw(cuty, loy, hiy)
    FX, FY = np.meshgrid(fx, fy)
    Wx = _flattop_partition_1d(FX, cutx, hwx)
    Wy = _flattop_partition_1d(FY, cuty, hwy)
    tot_power = float(np.sum(np.abs(E) ** 2)) + 1e-300
    segments = []
    for wi in Wx:
        for wj in Wy:
            Ej = np.fft.ifft2(np.fft.ifftshift((wi * wj) * F)).astype(E.dtype)
            if float(np.sum(np.abs(Ej) ** 2)) / tot_power > min_segment_power:
                segments.append(Ej)
    if not segments:
        segments = [E.copy()]
    return segments


def apply_real_lens_traced_segmented(
    E_in,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    n_segments: Any = 'auto',
    valley_frac: float = 0.15,
    power_frac: float = 0.995,
    min_segment_power: float = 1e-3,
    max_segments: int = 32,
    carriers: Any = 'auto',
    return_segments: bool = False,
    **traced_kwargs,
):
    """Traced lens on a single, possibly MULTI-congruence field via blind
    angular segmentation.

    :func:`apply_real_lens_traced` assumes ONE ray congruence per output pixel,
    so it is invalid for a field that superposes several beams / an extended
    multi-angle source.  :func:`apply_real_lens_traced_multi` handles that when
    the emitters are *already separated*; this handles the case where you only
    have the **combined** field, by splitting its angular spectrum at the deep
    VALLEYS between beams (the gaps that separate distinct congruences), so each
    segment captures one whole beam -- single-congruence -> traced-valid.  The
    segments sum to the input EXACTLY when ``min_segment_power=0``; the
    per-segment traced results are coherently summed via
    :func:`apply_real_lens_traced_multi`.

    Splitting at the spectral GAP (not at uniform bin edges) is essential:
    traced is non-linear, so fragmenting ONE beam across bins would *add* error;
    splitting only at true gaps keeps each congruence intact.  A unimodal
    spectrum (a single congruence) yields one segment == plain traced, so this
    is safe to call unconditionally.

    Parameters
    ----------
    n_segments : 'auto' | int | (int, int)
        Segment count.  ``'auto'`` splits at detected spectral valleys (0 cuts
        for a unimodal field); an int forces that many uniform bins-per-axis; a
        pair is ``(n_x, n_y)``.  Total is capped at ``max_segments``.
    valley_frac : float
        A spectral valley counts as a beam-separating gap only if its marginal
        power is below this fraction of the peak (with a real peak on each side).
    min_segment_power : float
        Drop segments carrying less than this fraction of the input power (saves
        traced passes on empty bins); ``0`` keeps the exact partition.
    return_segments : bool
        If True, return the list of segment fields instead of applying the lens
        (for inspection / the partition-sums-to-input check).

    Returns
    -------
    complex ndarray, or list of complex ndarray if ``return_segments``.

    Notes
    -----
    ``traced_kwargs`` reach a DIFFERENT consumer depending on the segment
    count: one segment goes straight to :func:`apply_real_lens_traced` (so the
    v5.29 modes ``amplitude_model='ray_density'`` /
    ``preserve_input_phase='remap'`` are honoured), while two or more go
    through :func:`apply_real_lens_traced_multi`, whose per-emitter contract
    fixes ``preserve_input_phase=True`` and (on the default
    ``reuse_prepared=True``) cannot express the input-dependent modes at all --
    see that function's Notes.  Both paths now REPORT the restriction instead
    of silently dropping the argument, but the count-dependent reach is real:
    if you need those modes on a multi-segment field, pass
    ``reuse_prepared=False`` and drop ``preserve_input_phase``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced_segmented',
                           input_kind='field')
    if dy is None:
        dy = dx
    if isinstance(n_segments, (tuple, list)) and len(n_segments) == 2:
        sx, sy = n_segments
    elif n_segments == 'auto':
        sx = sy = 'auto'
    else:
        sx = sy = int(n_segments)
    segments = _segment_field_by_angle(
        E_in, dx, dy, sx, sy, power_frac, valley_frac,
        min_segment_power, max_segments)
    if return_segments:
        return segments
    if len(segments) == 1:
        # single congruence -> the plain traced path (no per-segment overhead)
        return apply_real_lens_traced(
            segments[0], prescription=prescription, wavelength=wavelength,
            dx=dx, carrier=carriers, **traced_kwargs)
    return apply_real_lens_traced_multi(
        segments, prescription=prescription, wavelength=wavelength, dx=dx,
        carriers=carriers, **traced_kwargs)


class PreparedTracedLens:
    """A traced lens with its input-independent phase ``screen`` precomputed.

    Built by :func:`prepare_real_lens_traced`.  The entire traced leg (ray
    trace, Chebyshev/spline fits, Newton inversion, ``phase_analytic_lens``,
    the ``opl`` map and the valid / aperture masks) depends only on
    ``(prescription, wavelength, dx, N, carrier)``, so it is computed once and
    stored as ``screen``.  Each call is then just the input-dependent analytic
    leg plus one complex multiply::

        E_out = apply_real_lens(E_in, ...) * screen

    which drops the trace/fit/Newton stages from optimizer / tolerancing /
    multi-field loops entirely (>=2x per call).  Mirrors the library's
    ``PreparedRCWA2D`` / ``PreparedPMM2D`` precedent.

    A prepared lens FREEZES the settings that were live when it was prepared:
    ``wave_propagator`` / ``sag_dtype`` hold the values
    :func:`prepare_real_lens_traced` resolved from the process-wide defaults,
    and ``prescription`` is a deep copy of the caller's dict.  Flipping a
    global default or editing the prescription in place afterwards therefore
    does NOT move a prepared lens -- rebuild it (audit E-H4; see that
    function's docstring for the pre-v5.29.1 desynchronisation).

    Memory footprint
    ----------------
    The retained payload is the ``screen`` -- a single ``(N, N)`` complex128
    array of ``N*N*16`` bytes (**64 MB at N=2048, 256 MB at N=4096**); the
    other slots are tiny scalars / the prescription dict.  A prepared lens is
    a user-held object (not a module cache), so it is freed by normal garbage
    collection when it goes out of scope.  In a long-running optimizer /
    tolerancing loop that builds many prepared screens, call
    :meth:`release` to drop the screen deterministically (or reuse one prepared
    object).  There is no library-wide registry entry for these -- their
    lifetime is the caller's to manage.
    """

    __slots__ = ('screen', 'prescription', 'wavelength', 'dx', 'bandlimit',
                 'amp_use_gpu', 'wave_propagator', 'sag_dtype',
                 'sag_chunk_rows', 'N')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def release(self) -> None:
        """Free the precomputed ``screen`` (the ``N*N*16``-byte complex128
        array: 64 MB at N=2048, 256 MB at N=4096).

        After release the prepared lens can no longer be called (a subsequent
        ``prepared(E_in)`` raises).  Use in long-running optimizer / tolerancing
        loops to drop a prepared screen you are finished with, without waiting
        for garbage collection.  Idempotent.  ``clear`` is an alias.
        """
        self.screen = None

    clear = release

    def __call__(self, E_in: np.ndarray) -> np.ndarray:
        """Apply the prepared traced lens to ``E_in`` (shape must match N)."""
        if self.screen is None:
            raise RuntimeError(
                "PreparedTracedLens: the screen has been released (.release()/"
                ".clear() was called); rebuild it with prepare_real_lens_traced.")
        E_in = np.asarray(E_in)
        if E_in.shape != self.screen.shape:
            raise ValueError(
                f"PreparedTracedLens: E_in shape {E_in.shape} != prepared "
                f"grid {self.screen.shape}.")
        # Reproduce E_analytic EXACTLY as apply_real_lens_traced's internal
        # amp leg builds it (same 8 kwargs; note use_gpu=amp_use_gpu and the
        # raw sag_chunk_rows; dy is intentionally not forwarded there either).
        E_analytic = apply_real_lens(
            E_in, prescription=self.prescription, wavelength=self.wavelength,
            dx=self.dx, bandlimit=self.bandlimit, use_gpu=self.amp_use_gpu,
            wave_propagator=self.wave_propagator, sag_dtype=self.sag_dtype,
            sag_chunk_rows=self.sag_chunk_rows)
        out = E_analytic * self.screen
        tcd = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
        return out.astype(tcd) if out.dtype != tcd else out


def prepare_real_lens_traced(
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    N: int,
    carrier: Optional[Any] = None,
    bandlimit: bool = True,
    ray_subsample: int = 8,
    min_coarse_samples_per_aperture: int = 32,
    on_undersample: str = 'error',
    on_noncollimated: str = 'warn',
    inversion_method: str = 'newton',
    newton_fit: str = 'auto',
    newton_poly_order: int = 6,
    newton_max_iters: Optional[int] = None,
    amp_use_gpu: bool = False,
    use_gpu: bool = False,
    wave_propagator: Optional[str] = None,
    sag_dtype: Optional[Any] = None,
    sag_chunk_rows: Optional[int] = None,
    n_workers: Optional[int] = None,
    progress: Optional[Any] = None,
    amplitude_model: str = 'screen',
    fit_radius_beam_factor: Optional[float] = None,
    inverse_map: Optional[bool] = None,
) -> PreparedTracedLens:
    """Precompute the input-independent traced-lens screen for reuse (T-P1).

    The returned :class:`PreparedTracedLens` caches the whole trace/fit/Newton
    result, so every subsequent ``prepared(E_in)`` costs one analytic
    ``apply_real_lens`` + one complex multiply -- ideal for optimizer,
    tolerancing and multi-field loops that hold ``(prescription, wavelength,
    dx, N, carrier)`` fixed.

    The screen is exactly input-independent only for
    ``carrier in {None, <explicit wavefront / conjugate distance>}`` (NOT
    ``'auto'``, which fits the carrier from the field), so ``'auto'`` is
    rejected.  With H6 (v5.25.1) the carrier's entrance eikonal ``k0*W`` is
    baked into the cached ``opl`` map, so an explicit scalar-conjugate /
    ndarray carrier's screen focuses a diverging (or converging) input at the
    correct conjugate -- the 121-class per-group workflow, where every group
    has a KNOWN conjugate shared across many emitter fields, pays the
    trace/fit/Newton cost once and reuses the screen per field.
    ``tilt_aware_rays`` is forced False and the amplitude Newton mask is
    disabled (full coarse grid) so the cached ``valid`` region does not depend
    on any particular input; this makes the first (prepare) call a touch more
    expensive, amortized on the first reuse.  The screen is stored at float64
    complex precision; per-call output is cast back to the input dtype.

    ``on_noncollimated`` is honoured only for ``carrier=None`` (the plane-wave
    reference, where the ``ones`` placeholder the screen is built on is
    genuinely collimated).  For an explicit carrier the guard is forced
    ``'off'`` internally: the placeholder is a flat ``ones`` field, not the
    beam, so a scalar/ndarray carrier makes it LOOK strongly non-collimated
    (its residual is the whole carrier tilt) even though the actual reuse
    fields carry exactly that congruence -- the guard would either warn
    spuriously or, under ``'delegate'``, silently hand off to
    ``apply_real_lens`` (which ignores ``return_screen``) and cache a garbage
    screen.  The residual guard is the per-field caller's responsibility.

    WHAT A SCREEN CANNOT HOLD (v5.29; audit
    AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 §8.5 + this sibling sweep).  The
    validated carrier-regime options ``propagate_traced_carrier_chain`` now
    defaults to are all INPUT-DEPENDENT, so none of them is screen-able:
    ``amplitude_model='ray_density'`` scales ``|E_in|`` by the ray-tube
    Jacobian; ``preserve_input_phase='remap'`` transports the input's own
    residual phase; ``fit_radius_beam_factor`` sizes the ray-fit disc from the
    measured beam radius, which on the flat ``ones`` placeholder this screen is
    built on is meaningless.  ``amplitude_model`` / ``fit_radius_beam_factor``
    are accepted here ONLY at their screen-compatible values and rejected with
    an explanation otherwise (they used to raise a bare ``TypeError`` from the
    keyword binding, which is what surfaced when
    :func:`apply_real_lens_traced_multi` forwarded them).  A prepared screen is
    therefore a ``screen``-amplitude, aperture-fit-domain object: correct, but
    not the chain-grade model.

    WHAT A PREPARED LENS FREEZES (v5.29.1; audit E-H4).  **A prepared object
    freezes the settings that were live when it was prepared.**  Concretely:
    ``wave_propagator`` and ``sag_dtype`` are RESOLVED here against the
    process-wide defaults (:func:`set_default_wave_propagator` /
    :func:`set_lens_sag_dtype`) and the resolved values are stored on the
    returned object, and the ``prescription`` is DEEP-COPIED.  So flipping a
    global default -- or mutating the prescription dict in place -- after
    preparing leaves the prepared lens unchanged; rebuild it to pick up the
    new settings.  Pre-v5.29.1 the unresolved ``None`` sentinels were stored,
    so the frozen screen (prepare-time defaults) and the per-call analytic
    amplitude leg (call-time defaults) silently desynchronised on a global
    flip (measured 49.6 on a singlet), and an in-place prescription edit --
    the optimizer / tolerancing pattern this class advertises -- produced a
    stale-OPL x new-amplitude hybrid (measured 0.71 from a correct rebuild).
    """
    if isinstance(carrier, str) and carrier == 'auto':
        raise ValueError(
            "prepare_real_lens_traced cannot cache carrier='auto' (the "
            "carrier is fit from the field -> input-dependent).  Pass an "
            "explicit carrier (conjugate distance or wavefront ndarray) or "
            "None (plane-wave reference).")
    if amplitude_model != 'screen':
        raise ValueError(
            f"prepare_real_lens_traced cannot cache amplitude_model="
            f"{amplitude_model!r}: the ray-density exit amplitude is "
            f"|E_in|/sqrt(|det J|), i.e. INPUT-DEPENDENT, so it cannot be "
            f"baked into a reusable screen (apply_real_lens_traced rejects "
            f"amplitude_model='ray_density' with return_screen=True for the "
            f"same reason).  Call apply_real_lens_traced per field instead "
            f"(apply_real_lens_traced_multi(..., reuse_prepared=False) for the "
            f"multi-emitter case).")
    if fit_radius_beam_factor is not None:
        raise ValueError(
            f"prepare_real_lens_traced cannot cache fit_radius_beam_factor="
            f"{fit_radius_beam_factor!r}: the P2 aperture:beam cliff guard "
            f"sizes the ray-FIT disc from the measured input beam radius, and "
            f"this screen is built on a flat ``ones`` placeholder whose beam "
            f"radius is the grid, not the beam.  Call apply_real_lens_traced "
            f"per field (apply_real_lens_traced_multi(..., "
            f"reuse_prepared=False)) if the guard is needed -- a prepared "
            f"screen always uses the aperture-derived fit domain.")
    # The screen is built on a ``ones`` PLACEHOLDER (return_screen=True makes
    # it input-independent), so the collimation guard cannot judge it against a
    # carrier: force it off whenever a carrier is set.  For carrier=None the
    # placeholder IS the plane-wave reference, so the caller's value applies
    # (a correct, silent no-op there).
    _screen_noncol = 'off' if carrier is not None else on_noncollimated
    # v5.29.1 (audit E-H4): resolve the process-wide defaults NOW and store the
    # resolved values, so the frozen screen and every later per-call amplitude
    # leg use the SAME propagator / geometry dtype no matter what the caller
    # flips afterwards.  See "WHAT A PREPARED LENS FREEZES" above.
    if wave_propagator is None:
        from ..propagators.propagation import get_default_wave_propagator
        wave_propagator = get_default_wave_propagator()
    if sag_dtype is None:
        from ._lens_real import get_lens_sag_dtype
        sag_dtype = get_lens_sag_dtype()
    # Deep-copy the prescription: the stored dict must not alias the caller's,
    # or an in-place edit (the advertised optimizer / tolerancing loop) silently
    # pairs the cached OPL screen with a DIFFERENT lens in the amplitude leg.
    # Cheap next to the trace/fit/Newton this function runs; plain functions in
    # ``sag_callable`` survive (copy treats them as atomic).
    prescription = _copy_prescription(prescription)
    ones = np.ones((int(N), int(N)), dtype=np.complex128)
    screen = apply_real_lens_traced(
        ones, prescription=prescription, wavelength=wavelength, dx=dx,
        bandlimit=bandlimit, ray_subsample=ray_subsample,
        min_coarse_samples_per_aperture=min_coarse_samples_per_aperture,
        on_undersample=on_undersample, preserve_input_phase=True,
        tilt_aware_rays=False, carrier=carrier,
        on_noncollimated=_screen_noncol, parallel_amp=False,
        newton_amp_mask_rel=0.0, inversion_method=inversion_method,
        fast_analytic_phase=False, newton_fit=newton_fit,
        newton_poly_order=newton_poly_order, newton_max_iters=newton_max_iters,
        use_gpu=use_gpu, amp_use_gpu=amp_use_gpu,
        wave_propagator=wave_propagator, sag_dtype=sag_dtype,
        sag_chunk_rows=sag_chunk_rows, n_workers=n_workers, progress=progress,
        inverse_map=inverse_map, return_screen=True)
    return PreparedTracedLens(
        screen=screen, prescription=prescription, wavelength=wavelength,
        dx=dx, bandlimit=bandlimit, amp_use_gpu=amp_use_gpu,
        wave_propagator=wave_propagator, sag_dtype=sag_dtype,
        sag_chunk_rows=sag_chunk_rows, N=int(N))


__all__ = [
    'apply_real_lens_traced',
    'apply_real_lens_traced_multi',
    'prepare_real_lens_traced',
    'PreparedTracedLens',
    'close_worker_pool',
    'set_lens_parallel_amp',
    'get_lens_parallel_amp',
]
