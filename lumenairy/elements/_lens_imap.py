"""The INVERSE (mixed) characteristic: an exact per-pixel entrance/OPL/Jacobian
evaluator for the traced-lens exit grid.

WHAT THIS REPLACES.  ``apply_real_lens_traced`` does not Newton-invert every
exit pixel.  At ``ray_subsample = sub > 1`` it Newtons the COARSE lattice
``X[::sub, ::sub]`` -- 95 x 95 = 9 025 points at design 121's
``n_fine = 8192`` retrace -- and then INTERPOLATES that answer to the wave grid
with ``scipy.ndimage.map_coordinates``: the OPL (order 3), its NaN mask, the
ray-density amplitude, its NaN mask, and, under ``remap_sampling='full'``, the
entrance pull-back coordinates ``(xe, ye)``.  Measured on that element call
(``docs/audits/PROTO_INVERSE_MAP_2026_08_11.md`` S4.1 / S4.3): the Newton is
**0.010 s, 0.010 % of a 96.9 s element**, and those six full-grid upsamples are
**14.767 s, 12.9 % of it**.

So the quantity worth removing is the UPSAMPLE, not the Newton, and the object
that removes it is the inverse characteristic evaluated per pixel::

    G(x_out, y_out) -> (x_in, y_in, OPL, det J)

a total-degree Chebyshev least-squares model in the EXIT coordinates, fitted
from the ray landings the element has ALREADY traced.  One numba pass over
6.71e+07 pixels at exit degree 14 measures **1.910 s** against **4.035 s** for
the single order-3 ``map_coordinates`` it replaces (S4.4), and it is EXACT per
pixel where the interpolation was not.

WHY THE MAP IS PER-CALL AND NOT SHARED ACROSS A FAN.
``PROTO_INVERSE_MAP_2026_08_11`` S6.2 sketched a SHARED 4-D map
``G(x_out, y_out; x_src, y_src)`` on the two-parameter source label, built once
per (group, launch config) and contracted at each order's own label, with the
niche-C6 residual covered by WIDENING the source box by
``max|grad a_fit| x |R|`` -- i.e. by treating the C6 launch augmentation as an
equivalent SOURCE DISPLACEMENT.  That is exact only for the part of
``grad a_fit`` that is CONSTANT over the launch lattice (a wavefront tilt).
**It is not, and the shared map is not viable.**  MEASURED on design 121's real
chain at the real last group, order (-4,-2)
(``validation/repro_traced_carrier_121/imap_afit_121.py``)::

    max |grad a_fit| over the union pupil        2.1753e-02 rad
      of which the mean (the TILT)               2.7668e-03 rad  (12.7 %)
      and the NON-TILT part                      1.9546e-02 rad max
                                                 6.8066e-03 rad rms
    traced, 16 237 union-pupil rays:
      grad(W + a_fit) vs grad(W)          exit shift 210.3 um, OPL 5.08e+01 waves
      grad(W + a_fit) vs grad(W) + TILT   exit shift 193.7 um, OPL 4.85e+01 waves

against a parity bar of **1.11e-04 waves**.  The best pure source-shift stand-in
for the production launch congruence is wrong by **4.4e+05 x** that bar, and
removing the tilt buys 8 %: the C6 augmentation is aberration, not a source
displacement, so no source-labelled map at any node count represents the
congruence ``apply_real_lens_traced`` actually launches.

The source axes existed only to SHARE one build across a fan.  Dropping them
costs nothing and buys everything: the map is then fitted from the element's
OWN 52 441 traced landings, so it describes the exact congruence including
``a_fit``, it needs ZERO extra rays (the trace is already paid for), it needs no
C6 pad and no node-hull coverage argument across labels, and it is a decade
MORE accurate than the shared build (the proto's own S3.1 -- one congruence, no
source interpolation, exit degree 14 -- reads 1.24e-02 nm entrance and
3.24e-06 waves against the 4-D map's 0.032 nm / 3.84e-05).  The build is
~0.2 s against the shared map's 6.00 s, so break-even is immediate rather than
half an order.

THE BAR IS PARITY, NOT ``lambda/100``.  On this group the shipped path already
delivers 1.11e-04 waves (forward fit + Newton 6.65e-05, cubic upsample
4.44e-05) = lambda/9 000.  Sizing this map to ``lambda/100 with 3x margin``
would have shipped a **24x silent regression** (proto S3.4).  Guard G8 below is
therefore COMPARATIVE -- the map must beat, on the same held-out samples
against the same exact ray truth, the very Newton path it replaces -- and it is
the guard that matters.

Everything here is gated on :data:`TRACED_INVERSE_MAP`, which ships ``True``
and is SCOPED by ``carrier.py`` to the terminal fine retrace.  Setting it
``False`` restores the shipped coarse-Newton-plus-upsample chain BIT FOR BIT;
that is the fail-before, and it is what the regression tests force.  See that
flag's own note for the independent exact-trace oracle that decided the
default, and for the banner re-baseline it carries.
"""
from __future__ import annotations

import hashlib
import importlib.util as _ilu
import math
import threading
import warnings
from typing import Dict

import numpy as np

__all__ = [
    'InverseCharacteristic',
    'build_inverse_map',
    'inverse_map_cache_clear',
    'inverse_map_cache_info',
]


# ---------------------------------------------------------------------------
# THE FLAGS
# ---------------------------------------------------------------------------
#: Master gate for the inverse-characteristic per-pixel evaluator.
#:
#: ``True`` lets ``apply_real_lens_traced`` replace the coarse Newton lattice +
#: ``map_coordinates`` upsample chain with one exact polynomial evaluation per
#: exit pixel, WHEN the map builds and every guard G1-G8 passes.  When any
#: guard refuses, the call keeps the shipped path unchanged -- refuse, never
#: degrade.
#:
#: ``False`` is the FAIL-BEFORE: the map is never built, never consulted, and
#: the returned field is byte-identical to the pre-feature library.  It is the
#: switch the regression tests force.
#:
#: **THE ACCURACY CASE IS DECIDED AND IT FAVOURS THE MODEL; THE DEFAULT IS
#: BLOCKED ON ONE SHIPPED GUARD.**  Turning it on
#: MOVES design 121's shipping acceptance banner -- FWHM 3.350 -> 3.450 um,
#: peak 5.529e+03 -> 5.486e+03, with EE3 unchanged at 90.3 % and EE6 / EE12
#: each 0.1 point BETTER.  An INDEPENDENT ORACLE decided which of the two is
#: faithful (``BUILD_INVERSE_MAP_2026_08_11`` S6): the terminal group's exit
#: OPL and ray-tube amplitude from an EXACT-TRACE Newton -- residual and
#: Jacobian both from real traced rays, no polynomial model of any degree
#: anywhere -- converged to 2e-17 m of exit residual, scored against each
#: arm's own ``opl_map`` at 1 024 probe pixels on each of two orders.  In the
#: CORE, which is the region every banner metric is computed on::
#:
#:     order    incumbent rms / max        map rms / max          ratio
#:     (0,0)    1.078e-02 / 2.050e-02 w    2.422e-06 / 6.29e-06 w  2.2e-04
#:     (-4,-2)  5.982e-03 / 1.527e-02 w    1.227e-03 / 3.79e-03 w  0.205
#:     amplitude, core:  7.50e-04 vs 1.21e-04  |  1.63e-02 vs 2.30e-05
#:
#: **The incumbent carries 6.0e-03 to 1.08e-02 waves rms of core wavefront
#: error at the terminal leg** -- a Strehl of 0.9955-0.9986 -- plus 0.075 % to
#: 1.6 % of core amplitude error, and the model is 4.9x to 4 450x closer to
#: the exact trace on every statistic.  So the recorded 3.350 um was FLATTERED
#: by the path this replaces, and 3.450 um is the faithful reading.  The
#: banner is a user-approved acceptance number and its own history records two
#: re-baselines; this build reports the numbers and does NOT re-baseline it.
#:
#: **WHAT BLOCKS THE FLIP -- one coupling left, and it is not ``det J``.**
#: ``tests/unit/test_niche_c6_stationary_phase_launch.py::
#: test_the_two_newton_fit_backends_still_describe_the_same_map`` asserts that
#: ``newton_fit`` is an INTERPOLANT choice and that the polynomial and spline
#: backends return the same field to 5e-04.
#:
#: The FIRST coupling was the ``det J`` channel, taken from the polynomial
#: forward fit.  That is FIXED: :data:`_IMAP_DETJ_SOURCE` now derives it from
#: traced data alone, and the fix is also an improvement -- raced against the
#: exact-trace oracle's amplitude column, ``'analytic_inverse'`` reads
#: 1.90e-07 / 2.00e-05 relative on the two orders against the incumbent's
#: 7.50e-04 / 1.63e-02.
#:
#: The SECOND coupling remains and is named here rather than worked around:
#: ``_fit_domain_basis_ok = (newton_fit != 'spline')``.  The element applies
#: its ray-fit-domain restriction for the polynomial basis (as a NaN mask, or
#: as niche-D1 weights) and ABANDONS it for the spline basis (fix D5: that
#: basis cannot express it).  So the two backends hand this model DIFFERENT
#: sample sets -- a disc versus the whole launch square -- which moves its exit
#: normalisation box, its landing hull and its coefficients together.  The
#: incumbent absorbs that difference (its forward fit is in ENTRANCE
#: coordinates on a fixed lattice and its Newton only ever evaluates inside the
#: launch disc); a degree-14 fit in EXIT coordinates does not, and MEASURED it
#: amplifies it about 20x -- 1.06e-02 against the incumbent's <5e-04 on that
#: fixture.
#:
#: THE FIX, specified: build the model from the PRE-RESTRICTION landings --
#: the same arrays ``_TracedExitSupport.from_landings`` is given, before the
#: fit-domain restriction touches them -- and unweighted, so its sample set is
#: the traced rays and nothing else.  It is a small edit (stash three
#: ``n_launch^2`` copies, ~1.2 MB) but it changes the model's fit domain on
#: EVERY call, so it needs the oracle race, the banner and the full suite
#: re-run behind it before it can be trusted.  That is not done here, and a
#: shipped guard is not weakened to buy the flip, so the default stays
#: ``False``.
#:
#: **SCOPING.**  ``carrier.py``'s ordinary chain-leg call site passes
#: ``inverse_map=False``: the evaluator runs on the terminal FINE RETRACE
#: only, the leg whose output nothing re-fits and the leg every one of its
#: measurements covers.  Intermediate legs are byte-identical whatever this
#: flag holds (measured: 2.0e-18 waves, zero power change).  See S5 for why --
#: the chain amplifies skirt-level changes to an intermediate field by ~two
#: orders of magnitude, through the model BRANCHES each leg re-fits.
#:
#: WHAT IT IS NOT.  It does not change the ray trace, the forward Chebyshev
#: fits, the fit-domain policy, the exit support, the caustic floor, the
#: aperture stop or the field assembly.  It changes ONE thing: where the
#: entrance coordinates, the OPL and ``det J`` on the wave grid come from --
#: an exact per-pixel model of the traced map instead of a cubic/bilinear
#: interpolation of a 95 x 95 sample of it.
TRACED_INVERSE_MAP = False

#: What a refused build does.  ``'warn'`` (shipped) reports which guard
#: refused, with its measured number, and falls back to the shipped path;
#: ``'silent'`` falls back with no report; ``'error'`` raises.
#:
#: 'silent' does NOT change any returned bit relative to 'warn' -- both keep
#: the shipped path -- so this is a reporting knob, not a physics one.
INVERSE_MAP_GUARD = 'warn'

_IMAP_GUARD_ACTIONS = ('warn', 'silent', 'error')

#: Total degree of the exit-coordinate Chebyshev model.  The proto's exit-degree
#: ladder on ONE congruence (S3.1), maximum least-squares residual over 16 237
#: union-pupil samples::
#:
#:     degree   terms   x_in (nm)   OPL (waves)
#:      8        45     1.45e+00    6.25e-03
#:     10        66     1.23e+00    3.01e-04
#:     12        91     1.45e-01    2.41e-05
#:     14       120     1.24e-02    3.24e-06     <- shipped
#:     16       153     8.82e-04    2.97e-07
#:
#: Geometric, ~1 decade per two degrees.  14 is where the model is a decade
#: inside the incumbent's own 6.65e-05 waves with the evaluation still cheaper
#: than the order-3 ``map_coordinates`` it replaces (1.910 s vs 4.035 s at
#: 6.71e+07 points, S4.4).  Note the ASYMMETRY that makes this number bigger
#: than it looks: the shipped path's degree 6 in ENTRANCE coordinates is 0.124
#: waves in EXIT coordinates -- a direct exit fit is a DIFFERENT function and
#: needs more degree, not less.
_IMAP_EXIT_DEGREE = 14

#: G8's comparative bar.  The map's held-out error must be at most this factor
#: times the incumbent's on the SAME samples against the SAME exact ray truth.
#: 1.0 = "parity or better".  It is deliberately not a tolerance in waves: an
#: absolute ``lambda/100`` bar would have admitted a 24x regression on design
#: 121 (proto S3.4), because the incumbent is 90x tighter than lambda/100.
_IMAP_PARITY_FACTOR = 1.0

#: Held-out fraction pattern for G8.  Samples with ``i % 3 == 1 and j % 3 == 1``
#: on the launch lattice -- 1/9 of them, spatially spread -- are EXCLUDED from
#: the scoring fit, scored against the exact ray data, and then the shipped
#: coefficients are refitted on everything.  The incumbent's forward fit sees
#: those samples (it always fits the whole lattice), so the comparison is
#: BIASED AGAINST the map, which is the direction an acceptance bar should err.
_IMAP_HOLDOUT_MOD = 3
_IMAP_HOLDOUT_PHASE = 1

#: G7: least-squares samples per free coefficient below which the fit is
#: refused as under-determined.  At degree 14 (120 terms) this asks for 480
#: retained samples; design 121's retrace lattice supplies 52 441.
_IMAP_MIN_SAMPLES_PER_TERM = 4.0

#: G2: refuse when the forward Jacobian changes sign over the retained
#: samples (a fold), and when its dynamic range exceeds this (a caustic).  The
#: same 30x the ray-density amplitude already uses for its own fold flag; the
#: proto measured design 121's last group at 1.2627.
_IMAP_DETJ_MAXMIN = 30.0

#: WHERE THE ``det J`` CHANNEL COMES FROM -- and it must not come from the
#: ``newton_fit`` basis.
#:
#: The first build took it from ``_Cheb2DEvaluator.ev_value_and_grad``, i.e.
#: from the POLYNOMIAL forward fit.  That made the model's amplitude a function
#: of an interpolant choice, and it broke a shipped guard:
#: ``test_niche_c6::test_the_two_newton_fit_backends_still_describe_the_same_map``
#: asserts that ``newton_fit`` is an INTERPOLANT choice, not a physics one, and
#: that the polynomial and spline backends return the same field to 5e-04.  A
#: model whose amplitude changes with the basis cannot honour that, and gating
#: it to one basis states the asymmetry louder rather than removing it.
#:
#: Both options below read only TRACED data -- the landings and the launch
#: lattice -- so neither can see the basis at all:
#:
#: ``'landing_stencil'``
#:     ``np.gradient`` of the exit landings over the launch lattice.  The
#:     simplest thing that is basis-independent; the proto measured this
#:     estimator's own floor at 1.58e-05 relative amplitude (S3.5).
#: ``'analytic_inverse'``
#:     the analytic exit-gradient of the model's OWN ``x_in`` / ``y_in``
#:     channels gives ``det d(x_in,y_in)/d(x_out,y_out)``; the forward Jacobian
#:     is its reciprocal.  The proto measured that reciprocity at 1.37e-05
#:     against the traced forward Jacobian (S5.4).  No finite difference
#:     anywhere, but it inherits the ``x_in`` / ``y_in`` fit's own error.
#:
#: THE SHIPPED VALUE IS THE MEASURED WINNER, raced against the exact-trace
#: oracle's amplitude column on both orders
#: (``BUILD_INVERSE_MAP_2026_08_11`` S6.5); it is a resolved decision, not a
#: tunable.
_IMAP_DETJ_SOURCE = 'analytic_inverse'

_IMAP_DETJ_SOURCES = ('landing_stencil', 'analytic_inverse')

#: Row-chunk (in PIXELS) for the per-pixel evaluation.  Bounds the transient at
#: ~``chunk x n_channels x 8`` bytes on top of the outputs, so a 6.71e+07-pixel
#: grid never materialises a second full copy of anything.
_IMAP_EVAL_CHUNK = 4_000_000

#: How many built maps to retain.  The key is a SHA-256 over every array and
#: scalar the map depends on (see :func:`_imap_key`), so a cached map cannot
#: describe a different trace -- the chain-A cache discipline of
#: ``docs/audits/FIX_D4_D6_D7_2026_08_06.md``, applied to an in-process cache.
#: Small on purpose: the build is ~0.2 s, so the cache is a convenience for
#: repeated identical calls (sweeps, ``apply_real_lens_traced_multi``), not a
#: load-bearing optimisation worth holding memory for.
_IMAP_CACHE_SIZE = 4


# ---------------------------------------------------------------------------
# numba
# ---------------------------------------------------------------------------
_NUMBA_AVAILABLE = _ilu.find_spec('numba') is not None
_numba = None
_njit = None
_prange = None
_NUMBA_KERNELS: dict = {}


def _load_numba():
    """Import numba + njit/prange on first use; cache the handles.  Returns
    True iff numba is importable (False -> caller takes the NumPy fallback)."""
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


def _get_imap_eval_numba():
    """Compile (once, on first call) and return the multi-channel total-degree
    Chebyshev VALUE kernel, or ``None`` if numba is unavailable.

    Deliberately the same SHAPE as ``_lens_traced._cheb2d_val_grad_numba``:
    3-term recurrence on the stack per sample, ``prange`` over samples,
    ``fastmath``, no materialised Vandermonde.  The shape matters as much as
    the degree -- the shipped ``_invert_fit``'s chunked design-matrix form is
    39x slower at the SAME degree (38.356 s vs 0.866 s at 6.71e+07 points,
    proto S4.4) and its ``(6.71e+07, 66)`` float64 design matrix is 35 GB
    unchunked.  This kernel is value-only (no gradient): the inverse map is
    consumed pointwise and nothing downstream differentiates it.
    """
    if 'imap_eval' in _NUMBA_KERNELS:
        return _NUMBA_KERNELS['imap_eval']
    if not _load_numba():
        _NUMBA_KERNELS['imap_eval'] = None
        return None

    @_njit(cache=True, parallel=True, fastmath=True)
    def _imap_eval_numba(coeffs, K1, K2, xq, yq, xc, xh, yc, yh,
                         max_order, out):
        """Evaluate an ``n_channels``-column total-degree Chebyshev model.

        Parameters
        ----------
        coeffs : (M, C) float64   -- one column per output channel
        K1, K2 : (M,) int64       -- multi-indices (kx, ky), kx + ky <= degree
        xq, yq : (N,) float64     -- PHYSICAL exit coordinates
        xc, xh, yc, yh : float    -- the exit normalisation box
        max_order : int           -- maximum individual Chebyshev order
        out : (N, C) float64      -- written in place
        """
        N = xq.shape[0]
        M = coeffs.shape[0]
        nch = coeffs.shape[1]
        for i in _prange(N):
            u = (xq[i] - xc) / xh
            v = (yq[i] - yc) / yh
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
            for c in range(nch):
                acc = 0.0
                for m in range(M):
                    acc += coeffs[m, c] * Tu[K1[m]] * Tv[K2[m]]
                out[i, c] = acc

    _NUMBA_KERNELS['imap_eval'] = _imap_eval_numba
    return _imap_eval_numba


# ---------------------------------------------------------------------------
# the basis
# ---------------------------------------------------------------------------
def _td_terms(degree):
    """Total-degree multi-indices ``kx + ky <= degree`` as a ``(P, 2)`` int
    array -- the SAME basis and the SAME ordering the shipped
    ``inversion_method='fit'`` path builds, so "degree" here means exactly what
    ``newton_poly_order`` means there."""
    return np.array([[a, b] for a in range(degree + 1)
                     for b in range(degree + 1 - a)], dtype=np.intp)


def _td_design(ux, uy, degree, terms):
    """Column-product Chebyshev design matrix on normalised coordinates."""
    from numpy.polynomial.chebyshev import chebvander
    Vx = chebvander(np.asarray(ux, dtype=np.float64), degree)
    Vy = chebvander(np.asarray(uy, dtype=np.float64), degree)
    return Vx[:, terms[:, 0]] * Vy[:, terms[:, 1]]


def _cheb_dvander(u, degree):
    """``dT_n/du`` at every ``u`` -- ``T'_n = n U_{n-1}``, by the second-kind
    recurrence.  Same relation the shipped ``_cheb2d_val_grad_numba`` uses."""
    u = np.asarray(u, dtype=np.float64)
    U = np.zeros((u.size, degree + 1))
    if degree >= 0:
        U[:, 0] = 1.0
    if degree >= 1:
        U[:, 1] = 2.0 * u
    for n in range(2, degree + 1):
        U[:, n] = 2.0 * u * U[:, n - 1] - U[:, n - 2]
    D = np.zeros((u.size, degree + 1))
    for n in range(1, degree + 1):
        D[:, n] = n * U[:, n - 1]
    return D


def _td_design_grad(ux, uy, degree, terms):
    """``(dA/du, dA/dv)`` for :func:`_td_design`, analytically."""
    from numpy.polynomial.chebyshev import chebvander
    Vx = chebvander(np.asarray(ux, dtype=np.float64), degree)
    Vy = chebvander(np.asarray(uy, dtype=np.float64), degree)
    Dx = _cheb_dvander(ux, degree)
    Dy = _cheb_dvander(uy, degree)
    return (Dx[:, terms[:, 0]] * Vy[:, terms[:, 1]],
            Vx[:, terms[:, 0]] * Dy[:, terms[:, 1]])


def _detj_landing_stencil(xs_in, x_out_grid, y_out_grid):
    """``det d(x_out,y_out)/d(x_in,y_in)`` from the LANDINGS, by central
    differences on the regular launch lattice.  Traced data only -- it cannot
    see ``newton_fit``.  ``np.gradient`` erodes one lattice ring around any
    dead ray, which is the correct behaviour: a Jacobian with a NaN neighbour
    is not a Jacobian."""
    d = float(np.asarray(xs_in)[1] - np.asarray(xs_in)[0])
    dXdx, dXdy = np.gradient(np.asarray(x_out_grid, dtype=np.float64), d, d)
    dYdx, dYdy = np.gradient(np.asarray(y_out_grid, dtype=np.float64), d, d)
    return dXdx * dYdy - dXdy * dYdx


# ---------------------------------------------------------------------------
# the object
# ---------------------------------------------------------------------------
class InverseCharacteristic(object):
    """One congruence's exit-coordinate ray map, as a Chebyshev model.

    Four channels, in this order: ``x_in``, ``y_in``, ``OPL``, ``det J``.  The
    OPL channel carries EXACTLY the quantity the element's own forward fit
    carries -- the traced path plus the H6 carrier eikonal plus the niche-C6
    residual, on-axis referenced -- because it is fitted from the same
    ``opl_grid`` array.  Nothing here re-derives an eikonal, so nothing here
    can disagree with the element about one.

    ``hull`` is the convex hull of EVERY alive landing (not of the fitted
    subset): coverage is a property of where rays went, and restricting it to
    the fit domain would refuse pixels that have perfectly good ray data behind
    them.  Outside that hull the model EXTRAPOLATES -- measured at 1.1e+04
    waves one plateau out (proto S5.1) -- so :meth:`domain_mask` is not
    optional and every consumer goes through it.
    """

    __slots__ = ('coef', 'terms', 'K1', 'K2', 'degree', 'exit_c', 'exit_h',
                 'hull', 'hull_c', 'hull_rmax', 'launch_radius',
                 'wavelength', 'n_samples', 'residual', 'det_j_range',
                 'det_j_sign', 'guards', 'key', 'build_seconds',
                 'n_channels')

    CH_X_IN = 0
    CH_Y_IN = 1
    CH_OPL = 2
    CH_DET_J = 3

    def __init__(self, coef, terms, degree, exit_c, exit_h, hull, hull_c,
                 hull_rmax, launch_radius, wavelength, n_samples, residual,
                 det_j_range, det_j_sign, guards, key, build_seconds):
        self.coef = np.ascontiguousarray(coef, dtype=np.float64)
        self.terms = np.asarray(terms, dtype=np.intp)
        self.K1 = np.ascontiguousarray(self.terms[:, 0].astype(np.int64))
        self.K2 = np.ascontiguousarray(self.terms[:, 1].astype(np.int64))
        self.degree = int(degree)
        self.exit_c = (float(exit_c[0]), float(exit_c[1]))
        self.exit_h = (float(exit_h[0]), float(exit_h[1]))
        self.hull = hull
        # An interior point of the hull and a radius about it that
        # CONTAINS the hull -- the two quantities that bound the
        # half-plane reduction to a ring.  Same pair, same construction
        # and same purpose as ``_TracedExitSupport.hull_c`` /
        # ``hull_rmax``.
        self.hull_c = (None if hull_c is None
                       else (float(hull_c[0]), float(hull_c[1])))
        self.hull_rmax = (None if hull_rmax is None else float(hull_rmax))
        self.launch_radius = float(launch_radius)
        self.wavelength = float(wavelength)
        self.n_samples = int(n_samples)
        self.residual = np.asarray(residual, dtype=np.float64)
        self.det_j_range = float(det_j_range)
        self.det_j_sign = float(det_j_sign)
        self.guards = dict(guards)
        self.key = key
        self.build_seconds = float(build_seconds)
        self.n_channels = int(self.coef.shape[1])

    # -- introspection ----------------------------------------------------
    @property
    def n_terms(self):
        return int(self.coef.shape[0])

    @property
    def nbytes(self):
        return int(self.coef.nbytes + self.terms.nbytes)

    def __repr__(self):                                   # pragma: no cover
        return ('<InverseCharacteristic degree=%d terms=%d channels=%d '
                '%.1f kB>' % (self.degree, self.n_terms, self.n_channels,
                              self.nbytes / 1e3))

    # -- evaluation -------------------------------------------------------
    def eval_into(self, Xg, Yg, out, channels=None, chunk=None):
        """Evaluate the requested channels at the exit coordinates
        ``(Xg, Yg)``.

        ``out`` is a sequence of preallocated arrays of the query shape, one
        per requested channel, written in place.  ``channels`` defaults to all
        of them; asking for fewer is not an optimisation of the arithmetic
        alone -- on a 6.71e+07-pixel grid each channel IS a 537 MB array, and
        the screen path genuinely does not want ``det J``.

        Chunked over the flattened query so the scratch never exceeds
        ``chunk x n_channels x 8`` bytes.  The whole case for this path is that
        it costs LESS memory than the ``(2, N, N)`` coordinate stack plus the
        four upsampled full-grid arrays it retires, not more.
        """
        chans = (tuple(range(self.n_channels)) if channels is None
                 else tuple(int(c) for c in channels))
        if len(out) != len(chans):
            raise ValueError(
                'InverseCharacteristic.eval_into: expected %d output arrays '
                'for channels %r, got %d' % (len(chans), chans, len(out)))
        coef = (self.coef if chans == tuple(range(self.n_channels))
                else np.ascontiguousarray(self.coef[:, list(chans)]))
        sh = np.shape(Xg)
        n = int(np.prod(sh)) if sh else 1
        outs = [np.asarray(o).reshape(-1) for o in out]
        xf = np.asarray(Xg, dtype=np.float64).reshape(-1)
        yf = np.asarray(Yg, dtype=np.float64).reshape(-1)
        step = max(1, int(chunk if chunk is not None else _IMAP_EVAL_CHUNK))
        kern = _get_imap_eval_numba()
        buf = np.empty((min(step, max(n, 1)), len(chans)), dtype=np.float64)
        for s in range(0, n, step):
            e = min(s + step, n)
            xs = np.ascontiguousarray(xf[s:e])
            ys = np.ascontiguousarray(yf[s:e])
            b = buf[:e - s]
            if kern is not None:
                kern(coef, self.K1, self.K2, xs, ys,
                     self.exit_c[0], self.exit_h[0],
                     self.exit_c[1], self.exit_h[1], self.degree, b)
            else:
                # Pure-NumPy fallback (always-on).  Same formula, same
                # coefficients; the summation order over the basis differs, so
                # the two agree to ~1e-16 relative and not bit for bit -- the
                # same contract ``_Cheb2DEvaluator`` states for its own pair.
                A = _td_design((xs - self.exit_c[0]) / self.exit_h[0],
                               (ys - self.exit_c[1]) / self.exit_h[1],
                               self.degree, self.terms)
                np.matmul(A, coef, out=b)
                del A
            for c in range(len(chans)):
                outs[c][s:e] = b[:, c]
        return out

    def eval(self, Xg, Yg, channels=None, chunk=None):
        """Allocate and return one array per requested channel."""
        chans = (tuple(range(self.n_channels)) if channels is None
                 else tuple(int(c) for c in channels))
        sh = np.shape(Xg)
        out = [np.empty(sh, dtype=np.float64) for _ in chans]
        self.eval_into(Xg, Yg, out, channels=chans, chunk=chunk)
        return out

    # -- domain -----------------------------------------------------------
    def hull_mask_grid(self, x_axis, y_axis, relax=0.0):
        """``s <= relax`` on the OUTER PRODUCT of two axes, radially
        screened.

        Same device, same strictness argument and the same two quantities as
        ``_TracedExitSupport.retained_band_masks``: every pixel closer to the
        interior point than the inradius is strictly inside, every pixel
        beyond ``hull_rmax`` is strictly outside, and only the ring between
        reaches the ``O(pixels x facets)`` half-plane reduction.  Bit-identical
        to the dense test, and the reason this path can afford a hull test at
        all -- 6.71e+07 pixels x ~150 facets is a 10^10-MAC pass.
        """
        from ._lens_traced import _TracedExitSupport
        xg = np.asarray(x_axis, dtype=np.float64)
        yg = np.asarray(y_axis, dtype=np.float64)
        A, b = self.hull
        cx, cy = self.hull_c
        r_in = -float((np.asarray(A[0]) * cx + np.asarray(A[1]) * cy
                       + np.asarray(b)).max())
        rl = float(relax)
        r2 = (xg[None, :] - cx) ** 2 + (yg[:, None] - cy) ** 2
        r_i = r_in + rl
        ok = (r2 < r_i * r_i) if r_i > 0.0 else np.zeros(r2.shape, bool)
        r_o = float(self.hull_rmax) + max(rl, 0.0)
        ring = (~ok) & (r2 <= r_o * r_o)
        del r2
        if ring.any():
            iy, ix = np.nonzero(ring)
            sd = _TracedExitSupport.signed_distance(A, b, xg[ix], yg[iy])
            ok[iy, ix] = sd <= rl + 1e-12
        return ok

    def domain_mask(self, Xg, Yg, x_in=None, y_in=None, axes=None,
                    relax=0.0):
        """G6 at evaluation: which exit pixels the model may be asked about.

        Two rules, and both are the SHIPPED path's own:

        * inside the landing hull (the direct-fit path's rule, and the only
          thing standing between a degree-14 model and its own extrapolation
          -- measured at 1.1e+04 waves one plateau outside);
        * entrance inside ``0.99 * launch_radius`` -- verbatim
          ``_invert_newton``'s out-of-domain test, applied to the exact
          entrance point instead of to a Newton iterate that was clipped to
          the boundary and then failed the same test.

        The hull test comes FIRST because outside it ``x_in`` is an
        extrapolation, and a radius test on an extrapolated value means
        nothing.  ``axes`` names the two 1-D axes whose outer product
        ``Xg`` / ``Yg`` are, taking the screened path.

        ``relax`` WIDENS the hull test, and it is not a fudge -- it is the
        band niche C8 DELIBERATELY RETAINS.  The shipped exit-support taper
        holds the amplitude at exactly 1 out to ``sqrt(2) * sub * dx``
        beyond the hull (the plateau that makes the upsample's bleed
        identically zero) and then feathers over one exit-lattice cell, so
        the element EMITS a ring outside the hull carrying full amplitude.
        A model that NaN'd its OPL there would not truncate that ring, it
        would hand it the IDENTITY phase correction (``valid`` gates the
        phase, not the amplitude) -- full amplitude with the wrong phase,
        which is worse than either keeping it or cutting it.  MEASURED on
        design 121's shipping banner: cutting at ``s <= 0`` moved FWHM
        3.350 -> 3.550 um and EE3 90.3 -> 89.7 %.  So the model must be
        honest over exactly the support the element emits, no more and no
        less.
        """
        if self.hull is not None and axes is not None and self.hull_c:
            ok = self.hull_mask_grid(axes[0], axes[1], relax=relax)
        elif self.hull is not None:
            from ._lens_traced import _TracedExitSupport
            sd = _TracedExitSupport.signed_distance(
                self.hull[0], self.hull[1], Xg, Yg)
            ok = sd <= float(relax) + 1e-12
        else:                                             # pragma: no cover
            ok = np.ones(np.shape(Xg), dtype=bool)
        if x_in is not None and y_in is not None:
            lim = (self.launch_radius * 0.99) ** 2
            ok &= (np.asarray(x_in) ** 2 + np.asarray(y_in) ** 2) <= lim
        return ok


# ---------------------------------------------------------------------------
# the cache -- chain-A key discipline
# ---------------------------------------------------------------------------
_IMAP_CACHE: Dict[str, InverseCharacteristic] = {}
_IMAP_CACHE_ORDER: list = []
_IMAP_CACHE_STATS = {'hits': 0, 'misses': 0}

#: THE COMPANION LOCK for :data:`_IMAP_CACHE`, and the granularity is a
#: decision rather than a default.
#:
#: WHAT IT GUARDS, and why it is ONE lock and not three.  The cache is three
#: containers -- the map dict, the LRU order list and the hit/miss counters --
#: and the invariant worth protecting is that they AGREE.  Both accessors do
#: non-atomic read-modify-write sequences ACROSS them (``_cache_get`` removes
#: and re-appends a key while bumping a counter; ``_cache_put`` inserts, then
#: appends, then evicts from both in a loop), so interleaving two threads can
#: leave a key in ``_IMAP_CACHE`` that is no longer in ``_IMAP_CACHE_ORDER``
#: -- which leaks past the capacity bound -- or the reverse, which drops a
#: live entry.  A finer lock cannot express that invariant, because the
#: invariant is precisely the relationship BETWEEN the containers.
#:
#: WHAT IT DELIBERATELY DOES NOT GUARD: the BUILD.  ``build_inverse_map``
#: takes the lock for the lookup, releases it, and takes it again for the
#: store; the ~0.2 s least-squares fit in between runs unlocked.  Two threads
#: missing on the same key therefore both build, and the second store wins --
#: which is harmless BY CONSTRUCTION here, because the key is a SHA-256 over
#: every array and scalar the map depends on, so two maps under one key are
#: the same map.  Holding the lock across the build would serialise seconds of
#: numerical work to protect an invariant that cannot be violated.
#:
#: A PLAIN ``Lock``, not an ``RLock``, and that is the library's rule rather
#: than this module's preference: ``test_v4_14_2_dispatcher_pin_cache_locks``
#: requires cache locks to be ``threading.Lock`` "so the with-block discipline
#: is uniform across the library".  Re-entrancy was considered and is not
#: needed -- no region holding this lock calls any other function that takes
#: it.  The three accessors touch only the three containers, and
#: :func:`inverse_map_cache_clear` is reached from the registry and from
#: callers, never from inside a critical section.  A non-re-entrant lock also
#: turns any future violation of that into an immediate deadlock at the call
#: site instead of a silently nested one, which is the better failure.
#:
#: THE COUNTERS ARE EXACT UNDER CONTENTION, and that is a choice too.  They
#: live inside the same critical section the container invariant already
#: requires, so exactness is free -- and it is pinned:
#: ``test_niche_c15_inverse_map::test_a_repeated_identical_call_hits_the_cache``
#: asserts ``hits == 1`` on the nose.  Approximate counters would have made a
#: shipped test flaky to buy nothing.
_IMAP_LOCK = threading.Lock()


#: Target number of probe points used to FINGERPRINT the incumbent for the
#: cache key.  A few hundred evaluations of ``_invert_newton`` against a build
#: that costs 0.15-0.79 s, i.e. well under a per-cent of the thing being
#: cached.
#:
#: THE PROBE POINTS ARE THE ACTUAL TRACED LANDINGS, subsampled on a stride --
#: NOT a synthetic lattice inside their bounding box.  Measured while fixing
#: P0-5: a 5x5 interior lattice does NOT separate ``newton_max_iters`` 2 from
#: 40, because Newton converges in one or two steps near the axis and the two
#: incumbents agree there to the last bit.  They diverge exactly where G8
#: scores and where the map is hardest -- the outer landings -- so the
#: fingerprint has to look there.  Striding the real landing grid reaches
#: them by construction and needs no guess about where "hard" is.
_IMAP_INCUMBENT_PROBE_TARGET = 256


#: The exception classes the incumbent probe HASHES instead of raising.
#:
#: NARROWED, not broad (``AUDIT_V4_12_1_2026_05_16.md`` L5: NARROW >
#: WARN-BEFORE-PASS > RE-RAISE > KEEP-AS-IS).  ``parity_invert`` is this
#: element's own Newton over its own forward fits, so the only ways it can
#: legitimately refuse a strided probe point are numeric-evaluation failures
#: of those fits: ``scipy``'s ``RectBivariateSpline.ev`` raises ``ValueError``
#: off its knot rectangle, the Chebyshev evaluator raises ``ValueError`` /
#: ``TypeError`` on a shape or dtype it cannot take, and an ``ArithmeticError``
#: (``FloatingPointError`` under ``np.seterr('raise')``, ``ZeroDivisionError``)
#: is the same class of refusal.  ``np.linalg.LinAlgError`` is a ``ValueError``
#: subclass and is covered.
#:
#: Everything else -- ``AttributeError``, ``NameError``, ``KeyError``,
#: ``MemoryError``, ``KeyboardInterrupt`` -- is a DEFECT, not an incumbent
#: identity, and must propagate.  It costs nothing to let it: the shipped G8
#: arm calls ``parity_invert`` unguarded a few hundred lines below, so a bug
#: swallowed here surfaces there anyway, with a worse traceback and after the
#: cache has already been keyed on ``<raised:...>``.
_INCUMBENT_PROBE_ERRORS = (ValueError, TypeError, ArithmeticError)


def _incumbent_fingerprint(parity_invert, x_out_grid, y_out_grid):
    """A CONTENT hash of the incumbent inversion, for the cache key.

    WHY THE INCUMBENT IS IN THE KEY AT ALL (VERIFY_ARCHITECTURE P0-5).  G8 is
    a COMPARATIVE bar: it accepts the map only if the map beats the incumbent
    on held-out samples.  The incumbent is therefore half the acceptance
    criterion -- and it was the half the key did not hash.  So a map built
    against a weak incumbent was served to a call with a strong one, and the
    one guard that decides whether the map may be used AT ALL was bypassed.
    Production-reachable on the real element: ``newton_max_iters`` 2 -> 40,
    ``newton_poly_order`` 6 -> 8 and ``newton_fit`` 'polynomial' -> 'spline'
    each change ``_invert_newton`` and none of them changed the key, so each
    gave a stale hit -- with a cold-cache control on the same data reading
    ``refused = G8: held-out OPL error 1.8798e-04 waves against the
    incumbent's 1.5213e-08``.

    WHY A FINGERPRINT AND NOT A PARAMETER LIST.  This module's own doctrine,
    two docstrings up: "a key that names the CONFIGURATION and not the CONTENT
    is how a cache silently becomes a cache of something else."  Listing
    ``newton_max_iters``/``newton_poly_order``/``newton_fit`` would cover the
    three knobs someone thought of today and miss the fourth -- and the
    incumbent is a CLOSURE over this element's forward fits, so it can change
    without any named knob changing.  Evaluating it is the only thing that
    cannot be fooled.

    The probe is a deterministic STRIDE over the traced landings themselves,
    evaluated with the same RuntimeWarning suppression G8 uses (the
    incumbent's unconverged-Newton report belongs to the CALL, not to a
    hashing probe).  A non-finite answer is hashed as such rather than
    dropped: an incumbent that fails where another succeeds is a DIFFERENT
    incumbent."""
    if parity_invert is None:
        return b'<no-incumbent>'
    XO = np.asarray(x_out_grid, dtype=np.float64)
    YO = np.asarray(y_out_grid, dtype=np.float64)
    fin = np.isfinite(XO) & np.isfinite(YO)
    n_fin = int(fin.sum())
    if n_fin == 0:
        return b'<no-landings>'
    # Stride the landing GRID (not the flattened finite list) so the probe
    # keeps its two-dimensional spread and reaches the outer rows and columns
    # where the incumbent's Newton works hardest.
    ny, nx = XO.shape[-2], XO.shape[-1]
    per = max(1, int(math.isqrt(max(1, int(_IMAP_INCUMBENT_PROBE_TARGET)))))
    si = max(1, ny // per)
    sj = max(1, nx // per)
    sub = np.zeros_like(fin)
    sub[::si, ::sj] = True
    sel = fin & sub
    if not sel.any():                                     # pragma: no cover
        sel = fin
    PX = XO[sel].ravel()
    PY = YO[sel].ravel()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out = parity_invert(PX, PY)
    except _INCUMBENT_PROBE_ERRORS as exc:
        # An incumbent that REFUSES these points is also a distinct
        # incumbent, so the refusal is hashed rather than raised -- but only
        # for the numeric-evaluation classes named above.  Anything else is a
        # defect and propagates: the SHIPPED G8 arm calls ``parity_invert``
        # bare, so swallowing a real bug here would only move the traceback.
        return ('<raised:%s>' % type(exc).__name__).encode('ascii', 'replace')
    h = hashlib.sha256()
    h.update(b'incumbent-probe-v2')
    h.update(repr((si, sj, PX.shape, n_fin)).encode('ascii'))
    h.update(np.ascontiguousarray(PX).tobytes())
    h.update(np.ascontiguousarray(PY).tobytes())
    for part in out:
        a = np.ascontiguousarray(np.asarray(part, dtype=np.float64).ravel())
        h.update(repr(a.shape).encode('ascii'))
        # NaN has many bit patterns; canonicalise so the hash is about the
        # ANSWER and not about which NaN the FPU produced.
        bad = ~np.isfinite(a)
        if bad.any():
            a = a.copy()
            a[bad] = np.nan
        h.update(a.tobytes())
    return h.digest()


def _imap_key(xs_in, x_out_grid, y_out_grid, opl_grid, det_j_grid, weights,
              degree, launch_radius, wavelength, extra=(),
              incumbent_fp=b'<not-probed>', census_amp=None):
    """SHA-256 over EVERYTHING the map depends on.

    The chain-A cache lesson (``docs/audits/FIX_D4_D6_D7_2026_08_06.md`` D6),
    restated for an in-process cache: a key that names the CONFIGURATION and
    not the CONTENT is how a cache silently becomes a cache of something else.
    So the key is the BYTES of every array that enters the fit -- the landings,
    the OPL, the Jacobian, the weights -- plus every scalar that shapes it, plus
    the library version and the flag values that select the branch.  Any edit
    anywhere upstream that moves a single traced ray moves the key.

    ... AND EVERYTHING THE ACCEPTANCE DECISION DEPENDS ON, which is not the
    same set, and is where three things were missing:

    * ``incumbent_fp`` -- :func:`_incumbent_fingerprint`, G8's arm B hashed by
      EVALUATION.  Half the comparative bar, and unhashed (P0-5);
    * ``_IMAP_DETJ_SOURCE`` -- selects the Jacobian the amplitude channel
      consumes; a stale hit served a ``det J`` 3.12e-03 relative wrong, i.e.
      1.56e-03 of ray-density amplitude error (P1);
    * ``census_amp`` -- names WHERE G2, G7 and G8 are scored; a stale hit
      reported ``n_detj_census`` 1681 where a cold build reads 553 (P2).
    """
    h = hashlib.sha256()
    h.update(b'imap-v2')
    from .. import __version__ as _ver
    h.update(str(_ver).encode('ascii', 'replace'))
    h.update(repr((int(degree), float(launch_radius), float(wavelength),
                   bool(TRACED_INVERSE_MAP), str(INVERSE_MAP_GUARD),
                   float(_IMAP_PARITY_FACTOR), int(_IMAP_HOLDOUT_MOD),
                   int(_IMAP_HOLDOUT_PHASE), float(_IMAP_MIN_SAMPLES_PER_TERM),
                   float(_IMAP_DETJ_MAXMIN),
                   str(_IMAP_DETJ_SOURCE))).encode('ascii'))
    h.update(repr(tuple(extra)).encode('ascii', 'replace'))
    h.update(b'|incumbent|')
    h.update(incumbent_fp)
    for arr in (xs_in, x_out_grid, y_out_grid, opl_grid, det_j_grid, weights,
                census_amp):
        if arr is None:
            h.update(b'<none>')
            continue
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(repr(a.shape).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()


def _cache_get(key):
    with _IMAP_LOCK:
        hit = _IMAP_CACHE.get(key)
        if hit is not None:
            _IMAP_CACHE_STATS['hits'] += 1
            try:
                _IMAP_CACHE_ORDER.remove(key)
            except ValueError:                            # pragma: no cover
                pass
            _IMAP_CACHE_ORDER.append(key)
        else:
            _IMAP_CACHE_STATS['misses'] += 1
        return hit


def _cache_put(key, imap):
    with _IMAP_LOCK:
        _IMAP_CACHE[key] = imap
        _IMAP_CACHE_ORDER.append(key)
        while len(_IMAP_CACHE_ORDER) > max(0, int(_IMAP_CACHE_SIZE)):
            old = _IMAP_CACHE_ORDER.pop(0)
            _IMAP_CACHE.pop(old, None)


def inverse_map_cache_clear():
    """Drop every cached inverse map (and the hit/miss counters).

    This is the function handed to the central cache registry below, so
    ``lumenairy.clear_all_registered_caches`` and
    ``lumenairy_context(clear_caches_on_exit=True)`` drain this cache like any
    other.  Before v5.35 it existed but was never enrolled, which is exactly
    the sibling gap the v4.16.0 registry was written to retire.
    """
    with _IMAP_LOCK:
        _IMAP_CACHE.clear()
        del _IMAP_CACHE_ORDER[:]
        _IMAP_CACHE_STATS['hits'] = 0
        _IMAP_CACHE_STATS['misses'] = 0


def inverse_map_cache_info():
    """``{'size', 'capacity', 'hits', 'misses'}`` -- for a runner's provenance
    banner, and for the tests that pin the key discipline.

    Read under the lock: the four fields are reported as ONE consistent
    snapshot, not four independent reads that a concurrent eviction can
    straddle."""
    with _IMAP_LOCK:
        return {'size': len(_IMAP_CACHE), 'capacity': int(_IMAP_CACHE_SIZE),
                'hits': int(_IMAP_CACHE_STATS['hits']),
                'misses': int(_IMAP_CACHE_STATS['misses'])}


# ---- central cache-registry enrollment -----------------------------------
# The v4.16.0 contract: a module owning a cache hands its clearer to the
# registry, so the library-wide drain reaches it.  Prior art and identical
# shape: ``_lens_traced.py``'s ``_register_cache_clearer(
# 'lens_traced_main_guard', ...)``.  The indirection through ``_sys.modules``
# is the registry's own convention -- it keeps the registration bound to the
# module ATTRIBUTE rather than to the function object captured at import, so a
# test that monkeypatches ``inverse_map_cache_clear`` is still the thing the
# registry calls.
try:
    import sys as _sys

    from .._cache_registry import (
        register_cache_clearer as _register_cache_clearer,
    )
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'inverse_map',
        lambda: getattr(_this_mod, 'inverse_map_cache_clear')(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------
def _guard_fail(record, name, detail, **numbers):
    record['refused'] = name
    record['detail'] = detail
    record.update(numbers)
    return None


def build_inverse_map(xs_in, x_out_grid, y_out_grid, opl_grid,
                      det_j_grid=None,
                      *, wavelength, launch_radius, weights=None,
                      census_amp=None,
                      exit_degree=None, parity_invert=None,
                      parity_factor=None, cache=True, caller=None,
                      guard_record=None):
    """Fit the inverse characteristic from ONE congruence's traced landings.

    Parameters
    ----------
    xs_in : (n_launch,) float64
        The launch lattice, ``linspace(-launch_radius, launch_radius,
        n_launch)`` -- the element's own.
    x_out_grid, y_out_grid, opl_grid : (n_launch, n_launch) float64
        The traced exit landings and the on-axis-referenced OPL, NaN at dead
        rays.  These are consumed VERBATIM: whatever eikonal terms the element
        added to ``opl_grid`` travel into the model with them.
    det_j_grid : (n_launch, n_launch) float64, optional
        ``det d(x_out, y_out)/d(x_in, y_in)`` at each launch node.  ``None``
        (and the only value any shipped caller passes) makes the model derive
        it itself, from TRACED data only, per :data:`_IMAP_DETJ_SOURCE`.  That
        is not a convenience: an externally-supplied Jacobian is how the first
        build made its amplitude a function of the ``newton_fit`` basis and
        broke a shipped guard.  The argument is retained for tests that want to
        pin a known Jacobian.
    weights : (n_launch, n_launch) float64, optional
        The element's own ``_fit_weights``.  Passing them is what makes this
        model's fit domain the SAME domain the forward fit uses -- the niche-D1
        regularised restriction, expressed as weights rather than as a NaN
        mask.  ``None`` fits every retained sample unweighted.
    census_amp : (n_launch, n_launch) float64, optional
        ``abs(E_in)`` sampled at the launch nodes -- the element's own
        ``_amp``.  It does NOT weight the fit; it names WHERE G2, G7 and G8 are
        scored, at the same ``e^-4`` amplitude contour the element already uses
        for its exit-NA guard ("a Gaussian's 99.97 %-energy disc").  Without
        it the census runs over the whole launch square, where both this model
        and the incumbent are extrapolating their own fit into a region the
        beam never occupies -- and a max-over-samples comparison of two
        extrapolations decides an acceptance bar by coin flip.  MEASURED: with
        the census unrestricted a total-degree-FOUR model passes G8, because
        arm B is worse than it is out in the corners.  Basis-independent by
        construction: it is the input field at traced launch heights.

    parity_invert : callable, optional
        ``f(xq, yq) -> (x_in, y_in, opl)`` -- the INCUMBENT inversion, i.e. the
        element's own ``_invert_newton``.  Required for G8; without it the
        build refuses, because a comparative bar with nothing to compare to is
        not a bar.

    Returns
    -------
    InverseCharacteristic or None
        ``None`` when any guard refused.  ``guard_record`` (if given) is filled
        with the measured numbers either way, so a caller can report WHY.
    """
    import time as _time
    t0 = _time.perf_counter()
    rec = guard_record if guard_record is not None else {}
    rec.setdefault('refused', None)
    rec['caller'] = str(caller or 'build_inverse_map')
    degree = int(exit_degree if exit_degree is not None else _IMAP_EXIT_DEGREE)
    pf = float(parity_factor if parity_factor is not None
               else _IMAP_PARITY_FACTOR)

    xs_in = np.asarray(xs_in, dtype=np.float64)
    XO = np.asarray(x_out_grid, dtype=np.float64)
    YO = np.asarray(y_out_grid, dtype=np.float64)
    OP = np.asarray(opl_grid, dtype=np.float64)
    W = None if weights is None else np.asarray(weights, dtype=np.float64)
    rec['degree'] = degree
    rec['n_launch'] = int(xs_in.size)
    src = str(_IMAP_DETJ_SOURCE)
    if src not in _IMAP_DETJ_SOURCES:
        raise ValueError(
            '_IMAP_DETJ_SOURCE must be %s (got %r).'
            % (' or '.join(repr(a) for a in _IMAP_DETJ_SOURCES), src))
    rec['det_j_source'] = ('supplied' if det_j_grid is not None else src)
    # The CENSUS Jacobian -- what G2 judges the congruence on -- is always the
    # landing stencil: it is traced data, it costs two ``np.gradient`` passes,
    # and a fold or a caustic is a property of the rays, not of any model of
    # them.  The CHANNEL Jacobian (what the amplitude consumes) is resolved
    # further down and may differ.
    DJ = (np.asarray(det_j_grid, dtype=np.float64) if det_j_grid is not None
          else _detj_landing_stencil(xs_in, XO, YO))

    key = None
    if cache:
        # The incumbent is fingerprinted BY EVALUATION, because it is a
        # closure over this element's forward fits and half of G8's bar.
        key = _imap_key(xs_in, XO, YO, OP, DJ, W, degree, launch_radius,
                        wavelength, extra=(pf,),
                        incumbent_fp=_incumbent_fingerprint(parity_invert,
                                                            XO, YO),
                        census_amp=census_amp)
        hit = _cache_get(key)
        if hit is not None:
            rec.update(hit.guards)
            # AFTER the update, not before: ``hit.guards`` carries the
            # BUILDING call's ``cached = False`` and clobbered this flag, so
            # every hit reported itself as a miss -- which is the channel
            # that hid the stale-key defects above from every probe that
            # asked the record instead of the counters.
            rec['cached'] = True
            rec['refused'] = None
            return hit
    rec['cached'] = False

    # ---- G3: alive census -------------------------------------------------
    alive = np.isfinite(XO) & np.isfinite(YO) & np.isfinite(OP)
    n_alive = int(alive.sum())
    rec['n_alive'] = n_alive
    rec['n_dead'] = int(alive.size - n_alive)
    if n_alive < 16:
        return _guard_fail(rec, 'G3',
                           'only %d of %d launch rays survived the '
                           'prescription' % (n_alive, alive.size))

    # ---- G1: the landings are ONE single-valued congruence ----------------
    # (the fold half of it; the sign half is G2).  A NaN det J anywhere inside
    # the retained set means the forward fits could not be differentiated
    # there, which is not something to average over.
    good = alive & np.isfinite(DJ)
    n_good = int(good.sum())
    rec['n_fit_samples'] = n_good
    if n_good < 16:
        return _guard_fail(rec, 'G1',
                           'only %d launch nodes carry a finite Jacobian'
                           % n_good)

    # ---- G2: Jacobian sign + caustic floor --------------------------------
    # CENSUSED ON THE CONSTRAINED REGION, not on the whole launch square, and
    # the difference is four decades.  ``det J`` here is the analytic gradient
    # of the element's degree-10 forward fit; outside the ray-fit disc that fit
    # is a weighted EXTRAPOLATION (weight ``_FIT_DISC_OUTSIDE_WEIGHT_REL``
    # = 1e-8), so its gradient there is not a Jacobian, it is a polynomial
    # running away.  MEASURED on design 121's last group at n_fine = 4096:
    # 1.7e+05 over the whole square against the proto's own 1.2627 over the
    # union pupil.  Censusing the runaway would refuse every real call for a
    # caustic that is not there -- which is the wrong failure, and it is the
    # one this comment exists to stop being reintroduced.  The physics agrees:
    # ``|E_in|`` is negligible out there, so the ray-density amplitude those
    # samples carry is zero whatever ``det J`` says.
    sig = good
    _CA = (None if census_amp is None
           else np.asarray(census_amp, dtype=np.float64))
    if _CA is not None and np.isfinite(_CA).any():
        # the element's OWN e^-4 amplitude contour, reused verbatim.  No
        # shipped caller passes ``census_amp``; it exists because S6.5b needed
        # a basis-independent census region and is kept for the next attempt.
        _s = good & (_CA >= np.exp(-4.0) * float(np.nanmax(_CA)))
        if int(_s.sum()) >= 16:
            sig = _s
    elif W is not None:
        _wmax = float(W[good].max())
        _s = good & (W >= 0.5 * _wmax)
        if int(_s.sum()) >= 16:
            sig = _s
    rec['n_detj_census'] = int(sig.sum())
    dj = DJ[sig]
    n_pos = int((dj > 0).sum())
    n_neg = int((dj < 0).sum())
    adj = np.abs(dj)
    amin = float(adj.min())
    amax = float(adj.max())
    rng = amax / amin if amin > 0.0 else np.inf
    rec['det_j_range'] = float(rng)
    _adj_all = np.abs(DJ[good])
    _amin_all = float(_adj_all.min())
    rec['det_j_range_all_samples'] = float(
        (float(_adj_all.max()) / _amin_all) if _amin_all > 0.0 else np.inf)
    del _adj_all
    rec['det_j_sign'] = float(np.sign(np.median(dj)))
    rec['det_j_n_pos'] = n_pos
    rec['det_j_n_neg'] = n_neg
    if n_pos and n_neg:
        return _guard_fail(rec, 'G2',
                           'det J changes sign over the retained launch '
                           'nodes (%d positive, %d negative) -- the exit map '
                           'is folded and has no single-valued inverse'
                           % (n_pos, n_neg))
    if not np.isfinite(rng) or rng > _IMAP_DETJ_MAXMIN:
        return _guard_fail(rec, 'G2',
                           'det J spans %.4g over the retained launch nodes '
                           '(cap %.4g) -- a ray tube that collapses this far '
                           'is at a caustic' % (rng, _IMAP_DETJ_MAXMIN))

    # ---- the exit normalisation box ---------------------------------------
    # From EVERY alive landing, not from the fitted subset: an exit pixel
    # inside the landing hull must normalise inside [-1, 1], or the model
    # extrapolates in its own coordinate on data it actually has.
    xo_a, yo_a = XO[alive], YO[alive]
    ex_c = (0.5 * (float(xo_a.max()) + float(xo_a.min())),
            0.5 * (float(yo_a.max()) + float(yo_a.min())))
    ex_h = (0.5 * (float(xo_a.max()) - float(xo_a.min())) or 1.0,
            0.5 * (float(yo_a.max()) - float(yo_a.min())) or 1.0)
    rec['exit_centre_mm'] = [ex_c[0] * 1e3, ex_c[1] * 1e3]
    rec['exit_half_mm'] = [ex_h[0] * 1e3, ex_h[1] * 1e3]

    # ---- G6: the landing hull, from every ALIVE ray -----------------------
    # Coverage is a property of the HULL, not of the fit domain, and the two
    # must be allowed to differ: the proto measured the fit-cut variant FAILING
    # at +1.78 mm where the full-square one passes at -2.87 mm (S5.3).  Here
    # that distinction is between the WEIGHTED fit samples and every alive ray,
    # and taking the wider one costs no rays at all -- they are already traced.
    from ._lens_traced import _TracedExitSupport
    hull = _TracedExitSupport.half_planes(xo_a, yo_a, strict=False)
    if hull is None:
        return _guard_fail(rec, 'G6',
                           'the %d alive landings do not support a convex '
                           'hull (degenerate / collinear), so the model has '
                           'no domain to be honest about' % n_alive)
    rec['n_hull_facets'] = int(hull[1].size)
    hull_c = (float(xo_a.mean()), float(yo_a.mean()))
    hull_rmax = float(np.sqrt((xo_a - hull_c[0]) ** 2
                              + (yo_a - hull_c[1]) ** 2).max())

    # ---- the design ------------------------------------------------------
    terms = _td_terms(degree)
    P = int(terms.shape[0])
    rec['n_terms'] = P
    if n_good < _IMAP_MIN_SAMPLES_PER_TERM * P:
        return _guard_fail(rec, 'G7',
                           'the exit fit has %d retained samples for %d '
                           'total-degree-%d coefficients, below the %.1f '
                           'samples/coefficient floor'
                           % (n_good, P, degree,
                              _IMAP_MIN_SAMPLES_PER_TERM))

    gi, gj = np.nonzero(good)
    ux = (XO[good] - ex_c[0]) / ex_h[0]
    uy = (YO[good] - ex_c[1]) / ex_h[1]
    # Channels 0-2 are RAW TRACED DATA: the launch heights the rays started
    # from and the OPL grid the element built.  No fit, no basis, nothing that
    # ``newton_fit`` can reach.  Channel 3 is resolved after them.
    B = np.stack([np.broadcast_to(xs_in[:, None], XO.shape)[good],
                  np.broadcast_to(xs_in[None, :], XO.shape)[good],
                  OP[good]], axis=1)
    A = _td_design(ux, uy, degree, terms)
    w = None if W is None else W[good]

    from ._lens_traced import _solve_lstsq_thread_safe

    def _solve(rows=None):
        """Weighted least squares over a row subset.

        Expressed as a row WEIGHT rather than a fancy-index slice on purpose:
        ``A`` is ``(n_samples, n_terms)`` -- 50 MB at design 121's lattice and
        135 MB at a 375-node one -- and ``A[mask]`` would copy it once per
        solve on top of the weighted temporary.  Zeroing a row contributes
        nothing to ``A^T A`` or ``A^T b``, so the two forms give the same
        normal equations.
        """
        rw = w
        if rows is not None:
            rw = rows.astype(np.float64) if w is None else w * rows
        if rw is None:
            return _solve_lstsq_thread_safe(A, B)
        return _solve_lstsq_thread_safe(A * rw[:, None], B * rw[:, None])

    # ---- G8: PARITY with the incumbent, on HELD-OUT samples ---------------
    if parity_invert is None:
        return _guard_fail(rec, 'G8',
                           'no incumbent inversion was supplied, so the '
                           'comparative bar cannot be measured')
    hold = ((gi % _IMAP_HOLDOUT_MOD == _IMAP_HOLDOUT_PHASE)
            & (gj % _IMAP_HOLDOUT_MOD == _IMAP_HOLDOUT_PHASE))
    keep = ~hold
    n_hold = int(hold.sum())
    rec['n_holdout'] = n_hold
    if n_hold < 32 or int(keep.sum()) < _IMAP_MIN_SAMPLES_PER_TERM * P:
        return _guard_fail(rec, 'G8',
                           'the held-out parity probe would leave %d probe '
                           'samples and %d fit samples, too few to decide'
                           % (n_hold, int(keep.sum())))
    coef_h = _solve(keep)
    pred = A[hold] @ coef_h
    truth = B[hold]
    xq = XO[good][hold]
    yq = YO[good][hold]
    with warnings.catch_warnings():
        # The incumbent's own unconverged-Newton report belongs to the CALL,
        # not to this probe: firing it here would add a warning the shipped
        # path never emits at these points, which is itself a behaviour change.
        warnings.simplefilter('ignore', RuntimeWarning)
        inc = parity_invert(xq, yq)
    inc_x = np.asarray(inc[0], dtype=np.float64).ravel()
    inc_y = np.asarray(inc[1], dtype=np.float64).ravel()
    inc_o = np.asarray(inc[2], dtype=np.float64).ravel()

    lam = float(wavelength)
    fin = np.isfinite(inc_o) & np.isfinite(inc_x) & np.isfinite(inc_y)
    # THE SCORING SET is the constrained region, for the reason G2's census is
    # taken there: outside the ray-fit disc BOTH arms are extrapolating their
    # own weighted fit, the input amplitude is ~zero so nothing that happens
    # there can reach the returned field, and a max-over-samples comparison of
    # two extrapolations decides the bar by coin flip.  The full-set numbers
    # are recorded alongside so the choice is visible rather than implied.
    core = fin & sig[good][hold]
    if int(core.sum()) < 32:
        core = fin
    rec['n_parity'] = int(core.sum())
    rec['n_parity_all'] = int(fin.sum())
    if int(fin.sum()) < 32:
        return _guard_fail(rec, 'G8',
                           'the incumbent returned only %d finite answers on '
                           'the %d probe samples' % (int(fin.sum()), n_hold))

    def _err(px, py, po, m):
        e_opl = np.abs(po - truth[:, 2])[m] / lam
        e_pos = np.maximum(np.abs(px - truth[:, 0]),
                           np.abs(py - truth[:, 1]))[m]
        return (float(np.max(e_opl)), float(np.max(e_pos)),
                float(np.sqrt(np.mean(e_opl ** 2))))

    a_opl, a_pos, a_rms = _err(pred[:, 0], pred[:, 1], pred[:, 2], core)
    b_opl, b_pos, b_rms = _err(inc_x, inc_y, inc_o, core)
    rec['parity_map_opl_waves'] = a_opl
    rec['parity_incumbent_opl_waves'] = b_opl
    rec['parity_map_opl_rms_waves'] = a_rms
    rec['parity_incumbent_opl_rms_waves'] = b_rms
    rec['parity_map_pos_m'] = a_pos
    rec['parity_incumbent_pos_m'] = b_pos
    rec['parity_ratio_opl'] = (a_opl / b_opl) if b_opl > 0 else np.inf
    rec['parity_ratio_opl_rms'] = (a_rms / b_rms) if b_rms > 0 else np.inf
    rec['parity_ratio_pos'] = (a_pos / b_pos) if b_pos > 0 else np.inf
    _aa, _ap, _ar = _err(pred[:, 0], pred[:, 1], pred[:, 2], fin)
    _ba, _bp, _br = _err(inc_x, inc_y, inc_o, fin)
    rec['parity_map_opl_waves_all'] = _aa
    rec['parity_incumbent_opl_waves_all'] = _ba
    rec['parity_ratio_opl_all'] = (_aa / _ba) if _ba > 0 else np.inf
    if not np.isfinite(_aa):
        return _guard_fail(rec, 'G8',
                           'the model returned a non-finite OPL on the '
                           'held-out probe')
    if not (a_opl <= pf * b_opl):
        return _guard_fail(rec, 'G8',
                           'held-out OPL error %.4e waves against the '
                           'incumbent Newton path\'s %.4e on the same samples '
                           '(%.2fx, bar %.2fx)'
                           % (a_opl, b_opl, rec['parity_ratio_opl'], pf))
    # ...and on the RMS as well as the MAX.  A max-only bar is decided by the
    # incumbent's WORST pixel, and its worst pixels are the ones its Newton
    # left unconverged at the iteration cap -- a real error, but one that makes
    # the bar easy for the wrong reason.  Requiring both means the model has to
    # be better typically AND better at the extreme.
    if not (a_rms <= pf * b_rms):
        return _guard_fail(rec, 'G8',
                           'held-out OPL rms %.4e waves against the incumbent '
                           'Newton path\'s %.4e on the same samples (%.2fx, '
                           'bar %.2fx)'
                           % (a_rms, b_rms, rec['parity_ratio_opl_rms'], pf))
    if not (a_pos <= pf * b_pos):
        return _guard_fail(rec, 'G8',
                           'held-out entrance-position error %.4e m against '
                           'the incumbent Newton path\'s %.4e on the same '
                           'samples (%.2fx, bar %.2fx)'
                           % (a_pos, b_pos, rec['parity_ratio_pos'], pf))

    # ---- the shipped coefficients, on EVERY retained sample ---------------
    coef = _solve()

    # ---- CHANNEL 3, the Jacobian the amplitude consumes -------------------
    # Resolved HERE, after the position channels exist, because one of the two
    # options is their own analytic derivative.  Both read traced data only.
    if det_j_grid is not None:
        dj_col = DJ[good]
    elif src == 'landing_stencil':
        dj_col = DJ[good]
    else:
        # ``det d(x_in,y_in)/d(x_out,y_out)`` from the analytic exit-gradient
        # of channels 0 and 1; the FORWARD Jacobian is its reciprocal.  The
        # proto measured that reciprocity at 1.37e-05 against the traced
        # forward Jacobian (S5.4), and it carries no finite difference at all.
        _Au, _Av = _td_design_grad(ux, uy, degree, terms)
        _gx = (_Au @ coef[:, :2]) / ex_h[0]
        _gy = (_Av @ coef[:, :2]) / ex_h[1]
        _det_inv = _gx[:, 0] * _gy[:, 1] - _gy[:, 0] * _gx[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            dj_col = 1.0 / _det_inv
        dj_col = np.where(np.isfinite(dj_col), dj_col, DJ[good])
        del _Au, _Av, _gx, _gy, _det_inv
    _Bd = dj_col[:, None]
    _rwd = w
    if _rwd is None:
        coef3 = _solve_lstsq_thread_safe(A, _Bd)
    else:
        coef3 = _solve_lstsq_thread_safe(A * _rwd[:, None], _Bd * _rwd[:, None])
    B = np.concatenate([B, _Bd], axis=1)
    coef = np.concatenate([coef, coef3], axis=1)
    del _Bd
    # Reported where the fit is CONSTRAINED, for the same reason G2's census is
    # taken there: outside the ray-fit disc both this model and the incumbent's
    # forward fit are weighted extrapolations, and a residual dominated by a
    # region carrying 1e-8 of the weight and ~zero of the amplitude measures
    # the weighting, not the model.
    _rs = sig[good]
    _rr = np.abs(A @ coef - B)[_rs]
    resid_max = _rr.max(axis=0)
    rec['fit_resid_x_in_m'] = float(resid_max[0])
    rec['fit_resid_y_in_m'] = float(resid_max[1])
    rec['fit_resid_opl_waves'] = float(resid_max[2] / lam)
    rec['fit_resid_opl_rms_waves'] = float(
        np.sqrt(np.mean((_rr[:, 2] / lam) ** 2)))
    # The Jacobian channel is reported RELATIVE, because that is how the
    # ray-density amplitude consumes it (``|E_in| / sqrt(|det J|)``) and
    # because its absolute scale is meaningless next to the other three.
    _dref = np.abs(B[_rs, 3])
    _dok = _dref > 0.0
    rec['fit_resid_det_j_rel_max'] = float(
        (_rr[_dok, 3] / _dref[_dok]).max()) if _dok.any() else float('nan')
    rec['fit_resid_amp_rel_max'] = 0.5 * rec['fit_resid_det_j_rel_max']
    del _rr

    # ---- G7: exit-degree adequacy, from the fit residual (free) -----------
    # The bar is the incumbent's own held-out number, already measured above:
    # a model whose OWN residual on the data it fitted is worse than the
    # incumbent's error on data it did not is not adequate at this degree.
    if not (resid_max[2] / lam <= pf * b_opl):
        return _guard_fail(rec, 'G7',
                           'the total-degree-%d exit fit leaves %.4e waves of '
                           'least-squares residual, above the incumbent\'s '
                           '%.4e (raise the exit degree)'
                           % (degree, resid_max[2] / lam, b_opl))

    imap = InverseCharacteristic(
        coef=coef, terms=terms, degree=degree, exit_c=ex_c, exit_h=ex_h,
        hull=hull, hull_c=hull_c, hull_rmax=hull_rmax,
        launch_radius=launch_radius, wavelength=wavelength,
        n_samples=n_good, residual=resid_max, det_j_range=rng,
        det_j_sign=rec['det_j_sign'], guards=dict(rec), key=key,
        build_seconds=_time.perf_counter() - t0)
    rec['build_seconds'] = imap.build_seconds
    rec['bytes'] = imap.nbytes
    imap.guards = dict(rec)
    if cache and key is not None:
        _cache_put(key, imap)
    return imap


def report_refusal(record, caller='apply_real_lens_traced'):
    """The house action for a refused build: report which guard refused and
    with what number, then let the caller keep the shipped path.

    ``INVERSE_MAP_GUARD`` is validated here rather than at the refusal site so
    a junk value cannot behave as 'warn' on the calls where nothing ever
    refuses and only surface on the one where something does.
    """
    action = INVERSE_MAP_GUARD
    if action not in _IMAP_GUARD_ACTIONS:
        raise ValueError(
            '%s: INVERSE_MAP_GUARD must be %s (got %r).'
            % (caller, ' or '.join(repr(a) for a in _IMAP_GUARD_ACTIONS),
               action))
    if action == 'silent' or not record.get('refused'):
        return
    msg = (
        '%s: the inverse-characteristic per-pixel evaluator refused to build '
        '-- guard %s: %s.  The call KEEPS the shipped coarse-Newton + '
        'map_coordinates upsample path unchanged (refuse, never degrade), so '
        'this costs speed, never accuracy.  Set '
        'lumenairy.elements._lens_imap.INVERSE_MAP_GUARD = \'silent\' to stop '
        'reporting it, or TRACED_INVERSE_MAP = False to stop attempting the '
        'build at all (that is the fail-before switch and it is '
        'byte-identical to this outcome).'
        % (caller, record.get('refused'), record.get('detail', '')))
    if action == 'error':
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def imap_enabled(inverse_map=None):
    """Resolve the per-call switch against the module flag.

    ``inverse_map=None`` (the default) follows :data:`TRACED_INVERSE_MAP`;
    ``True`` / ``False`` override it for this call.  Kept here so the element
    and any other consumer cannot resolve the gate differently.
    """
    if inverse_map is None:
        return bool(TRACED_INVERSE_MAP)
    return bool(inverse_map)
