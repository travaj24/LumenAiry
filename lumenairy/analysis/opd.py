"""
Wavefront / OPD analysis + sampling diagnostics + depth-of-focus.

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

Contents:

* Sampling diagnostics: :func:`check_sampling_conditions`,
  :func:`check_opd_sampling`.
* Mode subtraction: :func:`remove_wavefront_modes`.
* OPD statistics: :func:`opd_pv_rms`, :func:`wave_opd_1d`,
  :func:`wave_opd_2d`.
* Depth of focus: :func:`depth_of_focus`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    'check_sampling_conditions',
    'check_opd_sampling',
    'remove_wavefront_modes',
    'opd_pv_rms',
    'wave_opd_1d',
    'wave_opd_2d',
    'unwrap_phase_2d',
    'depth_of_focus',
]

_TWO_PI = 2.0 * np.pi

# Residue tolerance for the 2-D unwrap self-check, in waves.
#
# Derivation.  On a residue-free field the returned map reproduces every
# in-mask wrapped neighbour difference EXACTLY in exact arithmetic, so the
# residue is pure float64 accumulation: a path of L samples carrying a
# total phase P accumulates ~L * eps * P.  The worst grid this library
# ships (N = 4096 at ~1e5 rad of pupil phase) gives
# 4096 * 2.2e-16 * 1e5 = 9e-8 rad = 1.4e-8 waves.  The defect this bar
# exists to catch is an integer-wave slip, i.e. >= 1.0 waves.  0.01 waves
# sits ~6 decades above the float floor and 2 decades below the smallest
# real failure.
_UNWRAP_RESIDUE_TOL_WAVES = 0.01

# Sample cap for the opt-in quality-guided unwrap.  Its union-find merge is a
# Python-level loop over ~2 links per sample; measured ~4 us/link on this
# class of machine, i.e. ~9 s at 1.05e6 samples (a 1024x1024 pupil).  Above
# that the caller wants ``method='itoh'``.
_RELIABILITY_MAX_SAMPLES = 1_100_000


def _wrap_to_pi(d: np.ndarray) -> np.ndarray:
    """Principal value of an angle difference, in ``[-pi, pi]``.

    Antisymmetric (``_wrap_to_pi(-d) == -_wrap_to_pi(d)``), which is what
    makes the path integral below path-consistent.
    """
    return d - _TWO_PI * np.round(d / _TWO_PI)


def _unwrap_2d_itoh(
    phase: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Masked 2-D Itoh path-integral unwrap on a spanning forest.

    The wrapped phase is integrated along horizontal runs of ``valid``
    samples (one compiled ``cumsum``), and the runs are then linked to one
    another through their vertical neighbours by whole multiples of
    ``2*pi``.  Because the mask is applied BEFORE the integration, no
    integration path ever crosses the zero-amplitude exterior -- which is
    the failure mode of a plain ``unwrap(axis=1)`` + ``unwrap(axis=0)``
    (the column pass re-anchors independently in every column and each
    column picks up its own ``2*pi*k``).

    For a residue-free field (phase gradient below ``pi`` per sample
    everywhere inside the mask) the result is independent of the path and
    therefore exact; the caller checks that with
    :func:`_unwrap_residue_waves`.

    Parameters
    ----------
    phase : ndarray, float, shape (Ny, Nx)
        Wrapped phase, e.g. ``np.angle(field)``.
    valid : ndarray, bool, shape (Ny, Nx)
        Samples that take part in the unwrap.

    Returns
    -------
    unwrapped : ndarray, float
        Unwrapped phase.  Meaningless where ``valid`` is False.
    comp_of_pixel : ndarray, int
        Connected-component label of each sample (``-1`` where invalid).
        Separate components have no phase relationship to one another.
    n_components : int
    residue_waves : float
        Largest in-mask neighbour link the result could not satisfy, in
        waves.  ``0`` means a single-valued map exists for this data.
    """
    Ny, Nx = phase.shape

    # ---- 1. integrate along the horizontal runs -------------------------
    # ``cum`` is the running sum of the WRAPPED column-to-column steps, with
    # the steps that would cross out of the mask zeroed so no run can pull
    # phase through the zero-amplitude exterior.  Built in place: the
    # difference, the wrap and the mask all write into one N^2 buffer.
    dh = np.zeros((Ny, Nx), dtype=np.float64)
    if Nx > 1:
        np.subtract(phase[:, 1:], phase[:, :-1], out=dh[:, 1:])
        d = dh[:, 1:]
        d -= _TWO_PI * np.round(d / _TWO_PI)
        d *= (valid[:, 1:] & valid[:, :-1])
    cum = np.cumsum(dh, axis=1)

    # A run starts at the first valid sample after an invalid one (or at
    # column 0).  Runs never span rows, so a row-major cumulative count of
    # the starts is a dense global run id.
    start = valid.copy()
    if Nx > 1:
        start[:, 1:] &= ~valid[:, :-1]
    flat_start = np.flatnonzero(start.ravel())
    n_runs = int(flat_start.size)
    if n_runs == 0:
        return (np.zeros_like(phase),
                np.full(phase.shape, -1, dtype=np.int32), 0, 0.0)
    run_id = (np.cumsum(start.ravel()) - 1).astype(np.int32).reshape(Ny, Nx)
    del start
    # Anchor every run on the principal value of its own first sample, so
    # ``unwrapped == phase`` (mod 2*pi) holds exactly at every valid sample.
    base = phase.ravel()[flat_start] - cum.ravel()[flat_start]
    del dh

    # ---- 2. link the runs vertically ------------------------------------
    # A vertical neighbour pair (i, c) -> (i+1, c) inside the mask requires
    #     out[i+1, c] + 2*pi*k_b  ==  out[i, c] + 2*pi*k_a + wrap(dphi)
    # so the integer wave offset between the two runs is fixed:
    #     k_b - k_a = ((out[i, c] + wrap(dphi)) - out[i+1, c]) / (2*pi),
    # with ``out = cum + base[run_id]``.  Everything here is evaluated on
    # the LINK SAMPLES only -- the full-grid ``out`` is not needed until
    # the very end, and neither is the wrapped vertical difference.
    pairs_a = np.empty(0, dtype=np.int64)
    pairs_b = np.empty(0, dtype=np.int64)
    pairs_k = np.empty(0, dtype=np.int64)
    a = b = k = None
    off = None
    if Ny > 1:
        vlink = valid[1:, :] & valid[:-1, :]
        if vlink.any():
            a = run_id[:-1, :][vlink]
            b = run_id[1:, :][vlink]
            dv = _wrap_to_pi((phase[1:, :] - phase[:-1, :])[vlink])
            off = (((cum[:-1, :][vlink] + base[a] + dv)
                    - (cum[1:, :][vlink] + base[b])) / _TWO_PI)
            k = np.round(off).astype(np.int64)
            del dv, vlink
            # Boolean indexing yields the links in row-major order, and a
            # run occupies one row only, so every distinct (run above, run
            # below) pair appears as ONE contiguous block.  Collapsing the
            # O(valid-sample) link list to its blocks is therefore an O(V)
            # scan, no sort: one representative per run pair, taken at the
            # middle of the overlap (away from both mask edges).  Any pair
            # whose samples disagree is a residue, which the caller's
            # residue self-check reports.
            if a.size:
                cut = np.flatnonzero((a[1:] != a[:-1]) | (b[1:] != b[:-1]))
                lo = np.concatenate(([0], cut + 1))
                hi = np.concatenate((cut + 1, [a.size]))
                mid = (lo + hi - 1) // 2
                pairs_a = a[lo].astype(np.int64)
                pairs_b = b[lo].astype(np.int64)
                pairs_k = k[mid]

    # ---- 3. propagate the offsets over the run graph (BFS per component) -
    adj: list = [[] for _ in range(n_runs)]
    for p, q, d in zip(pairs_a.tolist(), pairs_b.tolist(), pairs_k.tolist()):
        adj[p].append((q, d))
        adj[q].append((p, -d))

    k_of_run = np.zeros(n_runs, dtype=np.int64)
    comp_of_run = np.full(n_runs, -1, dtype=np.int32)
    n_components = 0
    for seed in range(n_runs):
        if comp_of_run[seed] >= 0:
            continue
        comp_of_run[seed] = n_components
        stack = [seed]
        while stack:
            p = stack.pop()
            kp = int(k_of_run[p])
            for q, d in adj[p]:
                if comp_of_run[q] < 0:
                    comp_of_run[q] = n_components
                    k_of_run[q] = kp + d
                    stack.append(q)
        n_components += 1

    # ---- 4. residue ------------------------------------------------------
    # Every HORIZONTAL in-mask link is satisfied by construction (``cum`` IS
    # the running sum of the wrapped steps), to the float64 accumulation of
    # the cumsum -- ~Nx * eps * |phase|, 1e-12 waves on the largest grids
    # this library runs.  So the only links that can disagree are the
    # vertical ones, and their disagreement is exactly the difference
    # between the offset each link asks for and the offset its two runs
    # were given.  That is the residue, measured on the supplied data, and
    # it costs one pass over the link list instead of a second full-grid
    # difference-and-wrap.
    residue_waves = 0.0
    if off is not None and off.size:
        residue_waves = float(np.max(np.abs(
            off - (k_of_run[b] - k_of_run[a]))))

    # One gather for the whole per-run constant: the run anchor and its
    # whole-wave offset fold together before the broadcast.
    out = cum
    out += (base + _TWO_PI * k_of_run)[run_id]
    comp_of_pixel = np.where(valid, comp_of_run[run_id], np.int32(-1))
    return out, comp_of_pixel, int(n_components), residue_waves


def _unwrap_2d_reliability(
    phase: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Quality-guided masked 2-D unwrap (Herraez, Burton, Lalor & Gdeisat,
    *Appl. Opt.* **41** (2002) 7437).

    Neighbour links are merged in order of decreasing reliability
    (reciprocal of the local wrapped second difference), so integration
    paths run through the smooth interior first and the noisy / aliased
    samples are unwrapped last, against an already-resolved neighbourhood.
    On a residue-free field this returns exactly what
    :func:`_unwrap_2d_itoh` returns; on a field that carries residues it
    confines the damage instead of dragging a whole row or column with it.

    Returns the same 4-tuple as :func:`_unwrap_2d_itoh`.
    """
    Ny, Nx = phase.shape
    npix = Ny * Nx
    vflat = np.flatnonzero(valid.ravel())
    n_valid = int(vflat.size)
    if n_valid == 0:
        return (np.zeros_like(phase),
                np.full(phase.shape, -1, dtype=np.int32), 0, 0.0)
    if n_valid > _RELIABILITY_MAX_SAMPLES:
        raise ValueError(
            f"unwrap_phase_2d: method='reliability' sorts and merges every "
            f"neighbour link in a Python union-find pass, which is O(1 min) "
            f"at {_RELIABILITY_MAX_SAMPLES} samples; this mask has "
            f"{n_valid}.  Use method='itoh' (the default, exact for "
            f"residue-free wavefronts and fully vectorised), or unwrap a "
            f"decimated / cropped pupil.")
    # Compact pixel ids so the union-find state is O(valid), not O(Ny*Nx).
    cmap = np.full(npix, -1, dtype=np.int64)
    cmap[vflat] = np.arange(n_valid, dtype=np.int64)

    # ---- reliability = 1 / second-difference magnitude ------------------
    # Herraez eq. (2): H, V and the two diagonal wrapped second
    # differences.  Edge samples keep the large-D (low-reliability) default
    # so the interior is always unwrapped first.
    ph = np.where(valid, phase, 0.0)
    D = np.full((Ny, Nx), np.inf, dtype=np.float64)
    if Ny > 2 and Nx > 2:
        c = ph[1:-1, 1:-1]
        def _sd(pa, pb):
            return _wrap_to_pi(pa - c) + _wrap_to_pi(pb - c)
        H = _sd(ph[1:-1, :-2], ph[1:-1, 2:])
        V = _sd(ph[:-2, 1:-1], ph[2:, 1:-1])
        D1 = _sd(ph[:-2, :-2], ph[2:, 2:])
        D2 = _sd(ph[:-2, 2:], ph[2:, :-2])
        core = np.sqrt(H * H + V * V + D1 * D1 + D2 * D2)
        # An all-valid 3x3 neighbourhood is a precondition for the second
        # difference to mean anything.
        vc = valid[1:-1, 1:-1].copy()
        for sl in ((slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                   (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None)),
                   (slice(None, -2), slice(None, -2)), (slice(2, None), slice(2, None)),
                   (slice(None, -2), slice(2, None)), (slice(2, None), slice(None, -2))):
            vc &= valid[sl]
        D[1:-1, 1:-1] = np.where(vc, core, np.inf)
    rel = np.where(np.isfinite(D) & (D > 0.0), 1.0 / D, 0.0)

    # ---- edge list: (reliability, p, q, k_q - k_p) ----------------------
    # The constraint on an edge is
    #     (phase[q] + 2*pi*k_q) - (phase[p] + 2*pi*k_p) == wrap(phase[q] - phase[p])
    # so  k_q - k_p == (wrap(d) - d) / (2*pi)  with  d = phase[q] - phase[p].
    idx = cmap.reshape(Ny, Nx)
    e_p, e_q, e_w, e_k = [], [], [], []
    for ax in (1, 0):
        if phase.shape[ax] < 2:
            continue
        if ax == 1:
            sl_lo, sl_hi = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
        else:
            sl_lo, sl_hi = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
        m = valid[sl_lo] & valid[sl_hi]
        if not m.any():
            continue
        d = (phase[sl_hi] - phase[sl_lo])[m]
        e_p.append(idx[sl_lo][m])
        e_q.append(idx[sl_hi][m])
        e_w.append((rel[sl_lo] + rel[sl_hi])[m])
        e_k.append(np.round((_wrap_to_pi(d) - d) / _TWO_PI).astype(np.int64))
    if not e_p:
        ep = eq = ek = np.empty(0, dtype=np.int64)
    else:
        ep = np.concatenate(e_p)
        eq = np.concatenate(e_q)
        ek = np.concatenate(e_k)
        ew = np.concatenate(e_w)
        order = np.argsort(-ew, kind='stable')
        ep, eq, ek = ep[order], eq[order], ek[order]

    # ---- weighted union-find over the valid samples ---------------------
    # parent/offset: k[p] == k[root(p)] + offset-to-root(p), in whole waves.
    par_l = list(range(n_valid))
    off_l = [0] * n_valid
    siz_l = [1] * n_valid

    def _find(p: int):
        acc = 0
        path = []
        while par_l[p] != p:
            path.append(p)
            acc += off_l[p]
            p = par_l[p]
        run = acc
        for node in path:                       # path compression
            nxt = off_l[node]
            par_l[node] = p
            off_l[node] = run
            run -= nxt
        return p, acc

    for p, q, kk in zip(ep.tolist(), eq.tolist(), ek.tolist()):
        rp, op = _find(p)
        rq, oq = _find(q)
        if rp == rq:
            continue
        # k_q - k_p == kk  ->  k_rq = k_rp + (op + kk - oq)
        delta = op + kk - oq
        if siz_l[rp] >= siz_l[rq]:
            par_l[rq] = rp
            off_l[rq] = delta
            siz_l[rp] += siz_l[rq]
        else:
            par_l[rp] = rq
            off_l[rp] = -delta
            siz_l[rq] += siz_l[rp]

    roots = np.empty(n_valid, dtype=np.int64)
    kvals = np.empty(n_valid, dtype=np.int64)
    for p in range(n_valid):
        r, o = _find(p)
        roots[p] = r
        kvals[p] = o
    uroots, inv = np.unique(roots, return_inverse=True)
    comp_flat = np.full(npix, -1, dtype=np.int32)
    comp_flat[vflat] = inv.astype(np.int32)
    kflat = np.zeros(npix, dtype=np.int64)
    kflat[vflat] = kvals
    out = phase + _TWO_PI * kflat.reshape(Ny, Nx)
    # The quality-guided spanning tree mixes horizontal and vertical edges,
    # so unlike the row-run forest it has no cheap closed form for the
    # residue; measure it directly.
    resid = _unwrap_residue_waves(phase, out, valid)
    return out, comp_flat.reshape(Ny, Nx), int(uroots.size), resid


def _unwrap_residue_waves(
    phase: np.ndarray,
    unwrapped: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Largest neighbour-link inconsistency of an unwrap, in waves.

    A successful unwrap reproduces the wrapped difference of every
    in-mask neighbour pair exactly; anything left over is a residue, i.e.
    a place where no unwrap can succeed (aliasing, a vortex, noise).  This
    is the quantity the Nyquist warning is gated on -- unlike a sampling
    estimate it is measured on the data actually supplied.
    """
    worst = 0.0
    Ny, Nx = phase.shape
    if Nx > 1:
        m = valid[:, 1:] & valid[:, :-1]
        if m.any():
            r = ((unwrapped[:, 1:] - unwrapped[:, :-1])
                 - _wrap_to_pi(phase[:, 1:] - phase[:, :-1]))[m]
            worst = max(worst, float(np.max(np.abs(r))))
    if Ny > 1:
        m = valid[1:, :] & valid[:-1, :]
        if m.any():
            r = ((unwrapped[1:, :] - unwrapped[:-1, :])
                 - _wrap_to_pi(phase[1:, :] - phase[:-1, :]))[m]
            worst = max(worst, float(np.max(np.abs(r))))
    return worst / _TWO_PI


def unwrap_phase_2d(
    phase: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = 'itoh',
) -> np.ndarray:
    """Unwrap a 2-D wrapped phase map over an arbitrary mask.

    Parameters
    ----------
    phase : ndarray, float, shape (Ny, Nx)
        Wrapped phase [rad], e.g. ``np.angle(field)``.
    mask : ndarray of bool, optional
        Samples to unwrap.  Defaults to every sample.  Samples outside the
        mask never appear on an integration path, so a zero-amplitude
        exterior (where ``np.angle(0) == 0``) cannot inject phase.
    method : ``'itoh'`` or ``'reliability'``
        ``'itoh'`` (default) integrates the wrapped gradient along a
        spanning forest of horizontal runs -- exact and fully vectorised
        for residue-free data.  ``'reliability'`` is the quality-guided
        Herraez (2002) unwrap: same answer on residue-free data, more
        graceful on noisy data, but ~50x slower (a Python union-find pass).

    Returns
    -------
    unwrapped : ndarray, float
        Unwrapped phase [rad], equal to ``phase`` modulo ``2*pi`` at every
        in-mask sample.  Out-of-mask samples are returned as ``phase``.

    Notes
    -----
    Disconnected mask components carry no phase relationship to one
    another; each is anchored on its own sample nearest the array centre.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError(
            f"unwrap_phase_2d: phase must be a 2-D array; got shape "
            f"{phase.shape}.")
    if mask is None:
        valid = np.ones(phase.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != phase.shape:
            raise ValueError(
                f"unwrap_phase_2d: mask shape {valid.shape} does not match "
                f"phase shape {phase.shape}.")
    out, comp, n_comp, _resid = _dispatch_unwrap_2d(phase, valid, method)
    out = _anchor_unwrap_components(phase, out, valid, comp, n_comp)
    return np.where(valid, out, phase)


def _dispatch_unwrap_2d(
    phase: np.ndarray,
    valid: np.ndarray,
    method: str,
    fn_name: str = 'unwrap_phase_2d',
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Route to the requested 2-D unwrap kernel (shared entry validation).

    Returns ``(unwrapped, comp_of_pixel, n_components, residue_waves)``.
    """
    if method == 'itoh':
        return _unwrap_2d_itoh(phase, valid)
    if method == 'reliability':
        return _unwrap_2d_reliability(phase, valid)
    raise ValueError(
        f"{fn_name}: unwrap method must be 'itoh' or 'reliability'; "
        f"got {method!r}.")


def _anchor_unwrap_components(
    phase: np.ndarray,
    unwrapped: np.ndarray,
    valid: np.ndarray,
    comp_of_pixel: np.ndarray,
    n_components: int,
    r2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pin each component's piston to the principal value at its centre.

    An unwrap fixes the phase only up to one additive ``2*pi*k`` per
    connected component.  Anchoring on the sample nearest the array centre
    -- rather than on whatever sample the integration happened to start
    from -- is what makes a known-defocus pupil come back with the right
    INTEGER WAVE COUNT: a converging wavefront is stationary at the pupil
    centre, so the principal value there is the true absolute phase.
    """
    if n_components <= 0:
        return unwrapped
    Ny, Nx = phase.shape
    if r2 is None:
        yy = (np.arange(Ny) - (Ny - 1) / 2.0) ** 2
        xx = (np.arange(Nx) - (Nx - 1) / 2.0) ** 2
        r2 = yy[:, None] + xx[None, :]
    r2 = np.asarray(r2, dtype=np.float64)
    if n_components == 1:
        # The overwhelmingly common case (one connected pupil): the anchor
        # is just the in-mask sample nearest the origin, no sort needed.
        anchor = int(np.argmin(np.where(valid, r2, np.inf)))
        shift = _TWO_PI * round(
            float(unwrapped.ravel()[anchor] - phase.ravel()[anchor])
            / _TWO_PI)
        if shift:
            unwrapped = unwrapped - shift
        return unwrapped
    r2 = r2.ravel()
    vflat = np.flatnonzero(valid.ravel())
    compv = comp_of_pixel.ravel()[vflat]
    order = np.lexsort((r2[vflat], compv))
    cs = compv[order]
    uc, first = np.unique(cs, return_index=True)
    anchors = vflat[order[first]]
    shift = np.zeros(n_components, dtype=np.float64)
    shift[uc] = _TWO_PI * np.round(
        (unwrapped.ravel()[anchors] - phase.ravel()[anchors]) / _TWO_PI)
    sh = np.where(valid, shift[np.clip(comp_of_pixel, 0, None)], 0.0)
    return unwrapped - sh


def check_sampling_conditions(
    N: int,
    dx: float,
    z: float,
    wavelength: float,
    feature_size: Optional[float] = None,
    NA: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Check whether grid parameters satisfy ASM sampling conditions.

    Evaluates the Nyquist criterion and the Fresnel aliasing condition
    for a given propagation geometry, and returns actionable diagnostics.

    Parameters
    ----------
    N : int
        Grid size (assumes a square N x N grid).
    dx : float
        Grid spacing [m].
    z : float
        Propagation distance [m].
    wavelength : float
        Optical wavelength [m].
    feature_size : float, optional
        Minimum feature size to resolve [m].  Required for the Fresnel
        aliasing check; if omitted that check is skipped.
    NA : float, optional
        4.10: when provided, the Nyquist criterion is relaxed to
        ``dx < wavelength / (2 * NA)``, which is what's actually needed
        to resolve the propagating cone within the specified NA.
        The strict ``dx < wavelength/2`` criterion (i.e. NA = 1) is
        only required if you also intend to resolve the full
        evanescent spectrum.
    verbose : bool, default True
        If ``True``, print a human-readable diagnostic summary.

    Returns
    -------
    dict
        ``'nyquist_ok'`` : bool
            Whether the Nyquist condition is satisfied (NA-aware if NA
            is supplied).
        ``'fresnel_ok'`` : bool
            Whether the Fresnel aliasing condition is satisfied.
        ``'d_min'`` : float
            Minimum resolvable feature size [m] for the current grid.
        ``'recommendations'`` : list of str
            Suggestions for fixing any violated conditions.  Empty when
            all conditions are met.
    """
    L = N * dx  # Grid extent

    # Condition 1: Nyquist.  Strict form dx < lambda/2 is for the full
    # angular spectrum (including evanescents); for a beam with max
    # NA, dx < lambda/(2*NA) is sufficient.  Default to the strict
    # form (NA = 1) for backward compatibility.
    if NA is None or NA <= 0:
        nyquist_limit = wavelength / 2
    else:
        nyquist_limit = wavelength / (2.0 * float(NA))
    nyquist_ok = dx < nyquist_limit

    # Condition 2: Fresnel aliasing (d_min = 2*lambda*z/L)
    d_min = 2 * wavelength * abs(z) / L

    if feature_size is not None:
        fresnel_ok = d_min < feature_size
    else:
        fresnel_ok = True  # Can't check without feature size

    recommendations = []
    if not nyquist_ok:
        recommendations.append(f"Decrease dx below {nyquist_limit * 1e6:.3f} um")
    if not fresnel_ok:
        required_L = 2 * wavelength * abs(z) / feature_size
        required_N = int(np.ceil(required_L / dx))
        recommendations.append(
            f"Increase grid extent to L > {required_L * 1e3:.2f} mm (N > {required_N})"
        )

    if verbose:
        print("ASM Sampling Conditions Check")
        print("=" * 40)
        print(f"Grid: {N}x{N}, dx = {dx * 1e6:.3f} um")
        print(f"Extent: L = {L * 1e3:.3f} mm")
        print(f"Propagation: z = {z * 1e3:.3f} mm")
        print(f"Wavelength: {wavelength * 1e9:.1f} nm")
        print()
        print(f"Nyquist (dx < λ/2 = {nyquist_limit * 1e6:.3f} um): "
              f"{'OK' if nyquist_ok else 'FAIL'}")
        print(f"Minimum resolvable feature: d_min = {d_min * 1e6:.2f} um")
        if feature_size is not None:
            print(f"Target feature size: {feature_size * 1e6:.2f} um")
            print(f"Fresnel aliasing: "
                  f"{'OK' if fresnel_ok else 'FAIL - increase grid extent'}")
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

    return {
        'nyquist_ok': nyquist_ok,
        'fresnel_ok': fresnel_ok,
        'd_min': d_min,
        'recommendations': recommendations,
    }


def depth_of_focus(
    wavelength: float,
    f_number: float,
    *,
    formula: str = 'rayleigh',
) -> float:
    """One-sided depth of focus [m] for a diffraction-limited system.

    Returns the ``+/-`` half-range: the axial shift from best focus at which
    the marginal-ray defocus wavefront error reaches the criterion.  The
    marginal-ray defocus OPD for a shift ``dz`` is ``W = dz * NA**2 / 2``, so
    the ``lambda/4`` (Rayleigh) criterion gives the one-sided
    ``dz = lambda / (2 * NA**2) = 2 * f_number**2 * wavelength`` (v5.21.1 AN-1
    fix: the prior ``4 * f#**2 * wavelength`` was the *total* range mislabelled
    as the half-range, i.e. 2x too large as a one-sided tolerance).

    Two standard formulas are supported; with ``NA = 1 / (2 * f_number)`` they
    evaluate to the SAME number, because the Marechal ``S > 0.8`` defocus
    bound coincides with the Rayleigh ``lambda/4`` peak (the classic
    ``lambda/4`` PV <-> ``lambda/14`` RMS coincidence):

    * ``'rayleigh'`` (default): ``+/- 2 * f_number**2 * wavelength``
      (``= lambda / (2 * NA**2)``), the classical quarter-wave OPD limit at
      the marginal ray.
    * ``'marechal'``: the Strehl ``> 0.8`` defocus bound, numerically equal
      to the Rayleigh value here.

    The two named entries are retained because optical-design practice
    distinguishes them by *derivation* (Rayleigh: OPD margin; Marechal:
    Strehl criterion), and downstream tools may want to annotate the choice.
    The TOTAL axial tolerance (both sides of focus) is
    ``2 * depth_of_focus(...) = 4 * f_number**2 * wavelength = lambda / NA**2``.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength [m].
    f_number : float
        System f-number ``f / D``.  Must be > 0.
    formula : {'rayleigh', 'marechal'}, default 'rayleigh'
        Which DOF expression to evaluate.

    Returns
    -------
    dof : float
        One-sided (half-range) depth of focus [m].

    Examples
    --------
    >>> from lumenairy.analysis import depth_of_focus
    >>> # f/2 at 550 nm, one-sided Rayleigh: 2 * 4 * 550e-9 = 4.4 um
    >>> float(depth_of_focus(550e-9, 2.0))
    4.4e-06
    >>> # Same system, Marechal coincides with Rayleigh here
    >>> float(depth_of_focus(550e-9, 2.0, formula='marechal'))
    4.4e-06
    """
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            f"depth_of_focus: wavelength must be positive and finite; "
            f"got {wavelength!r}.")
    if not np.isfinite(f_number) or f_number <= 0:
        raise ValueError(
            f"depth_of_focus: f_number must be positive and finite; "
            f"got {f_number!r}.")

    f = float(f_number)
    wl = float(wavelength)
    # One-sided (half-range) DOF = lambda/(2 NA**2) = 2 f#**2 lambda for BOTH
    # criteria (the Marechal S>0.8 defocus bound coincides with Rayleigh
    # lambda/4 here).  With NA = 1/(2 f#): lambda/(2 NA**2) = 2 f#**2 lambda.
    if formula in ('rayleigh', 'marechal'):
        return 2.0 * f * f * wl
    raise ValueError(
        f"depth_of_focus: formula must be 'rayleigh' or 'marechal'; "
        f"got {formula!r}.")


def check_opd_sampling(
    dx: float,
    wavelength: float,
    aperture: float,
    focal_length: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check whether grid sampling is adequate for clean OPD extraction
    from a converging wavefront.

    A converging wavefront of focal length ``f`` has a radial phase
    gradient ``k * r / f`` at pupil height ``r``.  At the pupil edge
    ``r = aperture / 2`` this gradient is maximal, so the phase change
    per grid sample is

        dphi = k * (aperture / 2) / f * dx
             = pi * aperture * dx / (wavelength * f)

    ``np.unwrap`` correctly tracks cycles as long as ``|dphi| < pi``
    at every sample, giving the Nyquist sampling rule

        dx <= lambda * f / aperture

    Violating this rule causes ``np.unwrap`` to skip cycles near the
    pupil edge, producing catastrophically wrong OPD values there (the
    classic symptom is a quadratic residual that blows up beyond some
    radius while the inner pupil looks clean).  See
    ``validation/real_lens_opd`` for an empirical illustration.

    Parameters
    ----------
    dx : float
        Grid spacing [m].
    wavelength : float
        Vacuum wavelength [m].
    aperture : float
        Clear aperture diameter [m].
    focal_length : float
        Effective focal length [m] of the optic producing the
        converging wavefront.  For a lens prescription, use the
        paraxial back focal length (BFL) from
        :func:`lumenairy.raytrace.system_abcd`.
    verbose : bool, default True
        Print a human-readable diagnostic.

    Returns
    -------
    result : dict
        ``'ok'`` : bool -- whether sampling is safely above Nyquist.
        ``'margin'`` : float -- ``dx_max / dx`` where dx_max is the
            Nyquist sampling limit.  Margin >= 2 is safe, 1 < margin
            < 2 is marginal, < 1 is failing.
        ``'dx_max'`` : float -- Nyquist-limited maximum dx [m].
        ``'phase_per_sample'`` : float -- radians of phase change per
            sample at the pupil edge (Nyquist limit is pi).
        ``'recommendations'`` : list of str -- suggestions to fix
            marginal or failing sampling.
    """
    f = float(abs(focal_length))
    ap = float(aperture)
    # Phase gradient at pupil edge = k * (ap/2) / f
    # Phase change per sample = gradient * dx
    phase_per_sample = (2 * np.pi / wavelength) * (ap / 2.0) / f * dx

    # Nyquist limit: max dx such that phase_per_sample <= pi
    dx_max = wavelength * f / ap
    margin = dx_max / dx
    ok = margin >= 2.0

    recommendations = []
    if not ok:
        required_dx = 0.5 * dx_max  # 2x safety margin
        recommendations.append(
            f'Reduce dx to <= {required_dx*1e6:.3f} um '
            f'(currently {dx*1e6:.3f} um).')
        recommendations.append(
            f'Or reduce aperture below '
            f'{(wavelength * f / (2 * dx)) * 1e3:.3f} mm at current dx.')
        recommendations.append(
            'Or use f_ref in wave_opd_1d/2d to subtract the reference '
            'sphere before unwrapping.')

    if verbose:
        print('--- OPD sampling check ---')
        print(f'  dx                          = {dx*1e6:.3f} um')
        print(f'  wavelength                  = {wavelength*1e9:.1f} nm')
        print(f'  aperture                    = {ap*1e3:.3f} mm')
        print(f'  focal length                = {f*1e3:.3f} mm')
        print(f'  phase change per sample     = {phase_per_sample:.3f} rad '
              f'(Nyquist limit = pi = {np.pi:.3f})')
        print(f'  Nyquist dx_max              = {dx_max*1e6:.3f} um')
        print(f'  margin (dx_max/dx)          = {margin:.2f} '
              f'({"SAFE" if margin >= 2 else ("MARGINAL" if margin >= 1 else "FAIL")})')
        if recommendations:
            print('  Recommendations:')
            for rec in recommendations:
                print(f'    - {rec}')

    return {
        'ok': ok,
        'margin': float(margin),
        'dx_max': float(dx_max),
        'phase_per_sample': float(phase_per_sample),
        'recommendations': recommendations,
    }


def remove_wavefront_modes(
    x: np.ndarray,
    opd: np.ndarray,
    modes: str = 'piston,tilt,defocus',
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Least-squares subtract low-order 1-D wavefront modes from an OPD
    profile.

    Useful for isolating high-order aberrations from an OPD cut.
    Operates on a 1-D OPD profile ``opd(x)`` where ``x`` is a pupil
    coordinate.

    Parameters
    ----------
    x : ndarray
        Pupil coordinate [m], 1-D.
    opd : ndarray
        Optical-path-difference values at ``x``, same length.  May contain
        ``NaN`` for out-of-aperture samples; those are ignored in the fit.
    modes : str
        Comma-separated subset of ``'piston'``, ``'tilt'``, ``'defocus'``.
        Pass ``''`` or ``None`` to fit nothing (returns input unchanged).
    weights : ndarray, optional
        Per-sample non-negative weights (e.g. pupil intensity ``|E|^2``).
        When supplied, the fit minimises ``sum(w_i * (opd_i - fit_i)^2)``
        so that the piston / tilt / defocus split honours where the
        light actually is rather than treating every grid point equally.
        Critical for vignetted, annular, or sparsely-illuminated pupils
        where unweighted fits leak high-order content into the low-order
        coefficients.  Default ``None`` reproduces the legacy uniform
        behaviour bit-for-bit.

    Returns
    -------
    opd_residual : ndarray
        ``opd`` minus the fitted modes.
    coeffs : dict
        Fit coefficients for each included mode.  Keys match the names
        passed in ``modes``.  Units: piston [m]; tilt [dimensionless
        slope]; defocus [1/m] (coefficient of x**2).

    Notes
    -----
    "Piston" is a constant phase offset -- physically irrelevant because
    detectors only see intensity.  "Tilt" is a linear phase ramp -- it
    just shifts the image laterally.  "Defocus" is a quadratic ``x**2``
    term -- it moves the focal plane axially.  Remove one, several, or
    all of these to isolate the "interesting" aberration content.
    """
    x = np.asarray(x)
    opd = np.asarray(opd)

    if not modes:
        return opd.copy(), {}
    mode_set = set(m.strip() for m in modes.split(',') if m.strip())

    cols, names = [], []
    if 'piston' in mode_set:
        cols.append(np.ones_like(x))
        names.append('piston')
    if 'tilt' in mode_set:
        cols.append(x)
        names.append('tilt')
    if 'defocus' in mode_set:
        cols.append(x ** 2)
        names.append('defocus')

    if not cols:
        return opd.copy(), {}

    A = np.column_stack(cols)
    mask = np.isfinite(opd)
    if not mask.any():
        return opd.copy(), {}

    if weights is None:
        coeffs, *_ = np.linalg.lstsq(A[mask], opd[mask], rcond=None)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != opd.shape:
            raise ValueError(
                f"weights shape {w.shape} != opd shape {opd.shape}")
        # Drop non-finite / non-positive weights from the fit.
        wmask = mask & np.isfinite(w) & (w > 0)
        if not wmask.any():
            return opd.copy(), {}
        sw = np.sqrt(w[wmask])
        coeffs, *_ = np.linalg.lstsq(
            A[wmask] * sw[:, None], opd[wmask] * sw, rcond=None)
    fit = A @ coeffs
    return opd - fit, dict(zip(names, coeffs.tolist()))


def opd_pv_rms(opd: np.ndarray) -> Tuple[float, float]:
    """Peak-valley and RMS of a 1-D or 2-D OPD array.

    Parameters
    ----------
    opd : ndarray
        OPD values.  ``NaN`` entries are ignored.

    Returns
    -------
    pv : float
        Peak-valley (max - min), in the same units as ``opd``.
    rms : float
        RMS deviation from the mean, in the same units as ``opd``.
    """
    arr = np.asarray(opd)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan'), float('nan')
    pv = float(finite.max() - finite.min())
    rms = float(np.sqrt(np.mean((finite - finite.mean()) ** 2)))
    return pv, rms


def wave_opd_1d(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    axis: str = 'x',
    aperture: Optional[float] = None,
    dy: Optional[float] = None,
    focal_length: Optional[float] = None,
    f_ref: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a 1-D OPD profile along the central row or column of a
    complex field.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric field on a regular grid.
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].  Used to convert unwrapped phase to OPL.
    axis : ``'x'`` or ``'y'``
        Which pupil cut to extract.  ``'x'`` takes the row NEAREST
        ``y = 0`` (row index ``Ny // 2``); ``'y'`` takes the column
        nearest ``x = 0`` (column index ``Nx // 2``).

        S11-6e (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1, the
        ``N // 2``-vs-``N / 2`` label mismatch): the row/column INDEX is
        the floor ``N // 2`` while the returned ``coord`` axis is
        centred with the float ``N / 2``, so the extracted cut sits at
        ``(N // 2 - N / 2) * d`` -- exactly ``0`` for EVEN ``N``, but
        ``-d / 2`` for ODD ``N``.  That is not a bug in the row choice:
        the centred grid ``(arange(N) - N / 2) * d`` has NO sample at
        exactly 0 when ``N`` is odd, so ``N // 2`` is one of the two
        nearest samples.  The docstring (which used to claim the cut is
        at exactly ``y = 0`` / ``x = 0``) is what was wrong; the code and
        the returned ``coord`` are unchanged.
    aperture : float, optional
        Clear-aperture diameter [m].  If given, the returned profile is
        cropped to |pupil coordinate| <= 0.5 * aperture.  Only the
        out-of-aperture samples at the two ENDS of the cut are removed
        before unwrapping (AN-2): INTERIOR zero-amplitude samples -- e.g.
        an annular / centrally-obscured pupil -- are NOT excluded and stay
        in the unwrap chain (with ``angle(0) = 0`` phase), which can inject
        a 2*pi slip across the far side of the gap.  For obscured pupils
        unwrap each connected region separately.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    focal_length : float, optional
        v5.4.6 (audit F-26): if given, used to emit the Nyquist
        sampling warning (``dx < lambda * focal_length / aperture``) so
        an under-sampled pupil cut is flagged before unwrapping.
    f_ref : float, optional
        v5.4.6 (audit F-26): reference-sphere focal length [m].  When
        provided, the quadratic (defocus) reference phase of a sphere
        converging to ``f_ref`` is subtracted before unwrapping, so the
        returned OPD is the residual wavefront error relative to that
        reference sphere rather than the full focusing wavefront.

    Returns
    -------
    coord : ndarray
        Pupil coordinate [m] for each returned sample.
    opd : ndarray
        Optical path length [m], ``+phase / k0`` with ``np.unwrap``
        applied along the cut.

    Notes
    -----
    * The sign convention assumes a forward-propagating wave, for which
      the phase at a given height equals ``+k * OPL``.
    * Unwrapping along a single row requires ``dx`` fine enough that
      the phase change between adjacent samples is below ``pi``.  For a
      lens of focal length ``f``, the worst case is at the pupil edge:
      ``dx < lambda * f / pupil_diameter``.
    """
    if dy is None:
        dy = dx

    Ny, Nx = E.shape
    k0 = 2 * np.pi / wavelength

    # Emit a Nyquist sampling warning if focal_length is known and
    # sampling is marginal / failing.
    if focal_length is not None and aperture is not None:
        samp = check_opd_sampling(
            dx, wavelength, aperture, focal_length, verbose=False)
        if not samp['ok']:
            import warnings as _w
            _w.warn(
                f'wave_opd_1d: Nyquist sampling is '
                f'{"failing" if samp["margin"] < 1 else "marginal"} '
                f'(margin = {samp["margin"]:.2f}).  Phase unwrap may '
                f'lose cycles near the pupil edge, producing '
                f'catastrophically wrong OPD values there.  '
                f'Recommended: {samp["recommendations"][0] if samp["recommendations"] else "see check_opd_sampling"}',
                RuntimeWarning, stacklevel=2)

    if axis == 'x':
        row = E[Ny // 2, :]
        coord = (np.arange(Nx) - Nx / 2) * dx
    elif axis == 'y':
        row = E[:, Nx // 2]
        coord = (np.arange(Ny) - Ny / 2) * dy
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # Optional reference-sphere subtraction: for strongly-converging
    # wavefronts we can divide out ``exp(-i*k0*coord**2 / (2*f_ref))``
    # before unwrap so the residual phase is small and unwrap is
    # robust regardless of sampling.  Caller must add the reference
    # phase back to the returned OPD.
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        ref_phase = -k0 * coord ** 2 / (2.0 * f_ref)
        row = row * np.exp(-1j * ref_phase)  # conjugate ref sphere

    valid = np.abs(row) > 0
    if aperture is not None:
        valid = valid & (np.abs(coord) <= 0.5 * aperture)

    if not valid.any():
        raise ValueError("No valid samples along the selected cut.")

    idx = np.where(valid)[0]
    i0, i1 = idx[0], idx[-1]
    row_crop = row[i0:i1 + 1]
    coord_crop = coord[i0:i1 + 1]

    phase = np.unwrap(np.angle(row_crop))
    opd = phase / k0

    # Add back the reference sphere so the returned OPD is absolute
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        opd = opd + (-coord_crop ** 2 / (2.0 * f_ref))
    return coord_crop, opd


def wave_opd_2d(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    aperture: Optional[float] = None,
    dy: Optional[float] = None,
    f_ref: Optional[float] = None,
    focal_length: Optional[float] = None,
    unwrap: str = 'itoh',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2-D OPD map from a complex field over its pupil.

    The unwrap runs on the PUPIL MASK only -- the zero-amplitude exterior
    (where ``np.angle(0) == 0``) is excluded before any integration path is
    laid down -- and every integration path stays inside the mask, so the
    result is independent of the path and exact whenever the wavefront is
    residue-free (below one half wave of phase change per sample).  See
    :func:`unwrap_phase_2d`.

    For converging wavefronts with many fringes, a reference spherical
    wave of focal length ``f_ref`` can be divided out before unwrapping so
    that the remaining phase is small; that is a conditioning knob, not a
    correctness requirement.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric field on a regular grid.
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    aperture : float, optional
        Clear-aperture diameter [m].  Samples outside the aperture
        (and any with |E| == 0) are set to ``NaN`` in the returned map.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    f_ref : float, optional
        If given, divide ``E`` by ``exp(-1j * k0 * r**2 / (2 * f_ref))``
        before unwrap.  The returned map is then the OPD *deviation* from
        that reference sphere.  Supply the paraxial focal length to
        flatten the converging wavefront before unwrap.
    unwrap : ``'itoh'`` or ``'reliability'``, default ``'itoh'``
        2-D unwrap kernel, see :func:`unwrap_phase_2d`.  ``'itoh'`` is
        exact for residue-free wavefronts and vectorised; ``'reliability'``
        is the quality-guided Herraez (2002) unwrap for noisy maps.

    Returns
    -------
    X, Y : ndarray
        Pupil coordinate grids [m], same shape as ``opd_map``.
    opd_map : ndarray
        2-D OPD in meters.  ``NaN`` outside the aperture.

    Notes
    -----
    * **Piston anchor.**  An unwrap fixes the phase only up to one additive
      whole wave per connected pupil region.  The map is anchored on the
      principal value of the valid sample NEAREST ``x = y = 0``, so a
      CENTRED pupil carrying a known defocus comes back with the right
      integer wave count -- a converging wavefront is stationary at the
      pupil centre, so the principal value there is the true absolute
      phase.  For a pupil that does NOT straddle the grid origin the
      anchor lands on its rim instead, where the wavefront is not
      stationary, and the returned piston is then an arbitrary whole
      number of waves (measured on a 1.2-waves-rms coma pupil decentred
      by (+60, -40) and (+110, +90) um: +1.0000 and +2.0000 waves, against
      0.0000 centred).  It is always an EXACT whole wave -- the map stays
      congruent to the wrapped phase everywhere -- so the shape, PV, RMS
      and every Zernike coefficient above piston are unaffected; only an
      absolute-piston reading is.  Disconnected pupil regions are anchored
      independently and carry no phase relationship to one another.
    * **Limit.**  No unwrap can recover a wavefront whose true phase moves
      by more than ``pi`` between neighbouring samples -- the wrapped data
      no longer determines the branch.  For a lens of focal length ``f``
      that is ``dx < lambda * f / aperture`` at the pupil edge (see
      :func:`check_opd_sampling`).  Two DIFFERENT diagnostics cover the
      two ways that bites, because neither covers both:

      - *Residues.*  A vortex, amplitude noise, or an ASYMMETRICALLY
        aliased wavefront leaves neighbour links that no single-valued
        map can satisfy at once.  That is measured here, on the supplied
        field, and warned about with its size in waves (measured: 1 wave
        for a charge-1 vortex, 1-7 waves for 5-30 waves of aliased coma,
        11 waves for uniformly random phase).
      - *Sampling.*  A RADIALLY SYMMETRIC aliased wavefront -- plain
        defocus is the case that matters -- wraps onto the exactly
        self-consistent phase of a LOWER-frequency wavefront, so it has
        no residue at all (measured 0.0000 waves at 16.5, 32.9 and 98.8
        rad per sample) and no unwrap of any kind can tell the two
        apart.  Only the sampling criterion catches it: pass
        ``focal_length`` to have that checked, and ``f_ref`` to remove
        the reference sphere so the residual is sampled at all.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-v4.15.5 an MCF / 3-D
    # ensemble input failed at ``E.shape`` unpacking with
    # ``ValueError: too many values to unpack`` (3-D) or
    # ``AttributeError`` (MCF) -- routes both to the canonical
    # v4.16 message via the V6 walker.  Input kind: 'field'.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'wave_opd_2d', input_kind='field')
    if dy is None:
        dy = dx

    Ny, Nx = E.shape
    k0 = 2 * np.pi / wavelength

    # Emit a Nyquist sampling warning if focal_length is known and
    # sampling is marginal / failing (see wave_opd_1d for rationale).
    if focal_length is not None and aperture is not None and f_ref is None:
        samp = check_opd_sampling(
            dx, wavelength, aperture, focal_length, verbose=False)
        if not samp['ok']:
            import warnings as _w
            _w.warn(
                f'wave_opd_2d: Nyquist sampling is '
                f'{"failing" if samp["margin"] < 1 else "marginal"} '
                f'(margin = {samp["margin"]:.2f}).  2-D unwrap may '
                f'lose cycles near the pupil edge.  '
                f'Recommended: pass f_ref={focal_length:.4g} to divide '
                f'out the reference sphere before unwrap, or {samp["recommendations"][0] if samp["recommendations"] else "reduce aperture / dx"}',
                RuntimeWarning, stacklevel=2)

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    field = E.copy()
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        # Remove ideal converging reference sphere.  A lens of focal
        # length f imparts phase exp(-i k0 r^2 / (2 f)); dividing by
        # that is the same as multiplying by the conjugate.
        field = field * np.exp(+1j * k0 * (X ** 2 + Y ** 2) / (2.0 * f_ref))

    valid = np.abs(field) > 0
    if aperture is not None:
        valid = valid & (X ** 2 + Y ** 2 <= (0.5 * aperture) ** 2)

    phase = np.angle(field)

    # Masked 2-D unwrap.  ``valid`` is applied BEFORE the integration, so
    # no path crosses the zero-amplitude exterior; the runs of the mask are
    # then linked to one another by whole waves.  The residue self-check
    # below measures, on the supplied data, whether a single-valued map
    # exists at all -- the wrapped phase of an under-sampled or
    # vortex-carrying field leaves neighbour links that cannot all be
    # satisfied, and that is the only regime in which the returned map can
    # still be wrong.
    phase_unwrapped, comp, n_comp, residue_waves = _dispatch_unwrap_2d(
        phase, valid, unwrap, fn_name='wave_opd_2d')
    phase_unwrapped = _anchor_unwrap_components(
        phase, phase_unwrapped, valid, comp, n_comp, r2=X ** 2 + Y ** 2)
    if residue_waves > _UNWRAP_RESIDUE_TOL_WAVES:
        import warnings as _w
        _w.warn(
            f'wave_opd_2d: the wrapped phase carries residues -- the '
            f'largest neighbour link the unwrap could not satisfy is '
            f'{residue_waves:.3f} waves, so no single-valued OPD map '
            f'exists for this field and the returned map slips by whole '
            f'waves somewhere.  Causes: the pupil phase moves by more than '
            f'pi between samples (need dx < lambda * f / aperture, see '
            f'check_opd_sampling -- pass f_ref to divide out the reference '
            f'sphere first), amplitude noise, or an optical vortex.',
            RuntimeWarning, stacklevel=2)
    if n_comp > 1:
        import warnings as _w
        _w.warn(
            f'wave_opd_2d: the pupil support has {n_comp} disconnected '
            f'regions; each is unwrapped and anchored on its own, so their '
            f'relative piston is undetermined (any whole number of waves). '
            f'Fit / compare each region separately.',
            RuntimeWarning, stacklevel=2)

    opd = phase_unwrapped / k0
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        # Add the reference sphere back so the returned OPD is
        # ABSOLUTE (matching wave_opd_1d's convention), not a
        # deviation.  This makes f_ref purely a numerical
        # conditioning knob, not a physical reinterpretation.
        opd = opd + (-(X ** 2 + Y ** 2) / (2.0 * f_ref))

    opd = np.where(valid, opd, np.nan)
    return X, Y, opd
