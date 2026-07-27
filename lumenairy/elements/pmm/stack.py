"""PMM unified multi-layer API: PMMStack (mixes vertical + slanted
layers via the div-conforming covariant-metric generator)."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np

# Backend detection for the JAX (differentiable) dispatch in pmm_efficiency_1d.
# Mirrors rcwa's pattern: a JAX input routes to the self-contained jnp twin,
# while a NumPy input falls through to the original (byte-identical) code.
# Reused for the slanted-grating solver: the slant breaks the +/-q field
# symmetry (like a full-3x3 tensor layer), so it needs the GENERALIZED
# (explicit forward/backward) S-matrix.  rcwa does NOT import pmm, so this
# top-level import introduces no cycle.
from ...backend import is_jax_array
from ..rcwa import _interface_smatrix_general
from ..rcwa._core import (
    _blas_threads_quiet,
    _require_propagating_incidence,
)
from ._core import (
    _C,
    _COV_MIN_SLANT_RAD,
    _assemble_jones_farfield,
    _build_nodal_metric_segments,
    _build_sem_tensor_segments,
    _cov_layer_4n,
    _cov_split,
    _freeze_cached,
    _half_M_sym_metric,
    _interface_smatrix,
    _kz_forward,
    _layer_modes_metric,
    _n_propagating_orders,
    _pmm_union_grid,
    _propagation_smatrix,
    _propagation_star,
    _propagation_star_general,
    _redheffer_star,
    _resolve_incidence_checked,
    _resolve_order_count,
    _sem_fourier_projection,
    _sem_modes_tensor,
    _sem_modes_uniform,
    _t3_slant,
    _tensor3_dict,
    _uniform_geo_eig,
)


def _warn_stack_energy(R_eff, T_eff):
    """Energy tripwire for a stack solve -- the PMM counterpart of RCWA's
    ``_check_energy`` (replicated, not imported: pmm does not depend on rcwa
    internals).  Three severities, matching that model:

    * NON-FINITE total -> **raise**.  A NaN/inf half-space index or
      permittivity used to reach the far field and return ``tot = [nan, nan]``
      completely silently (audit M3 2026-07-25): a one-sided ``>`` comparison
      is NaN-blind, so the tripwire never fired.
    * NEGATIVE total -> **raise**.  ``R``/``T`` are flux ratios of non-negative
      numerators, so a negative total means the NORMALISATION is non-physical
      (a gain / non-propagating incidence medium flipping ``kz_inc`` past the
      entry guards -- the audit-P1 class RCWA also raises on).
    * ``R+T > 1`` -> **warn** (unchanged).  A passive structure cannot
      reflect+transmit more than the incident power regardless of loss, so this
      is the pure instability signature of a near-singular interface mode-match
      in the Redheffer cascade (the measure-zero quasi-resonance; the
      many-interface tapered z-staircase can hit it at large ``n_slices``).
      Kept a WARNING (not a raise) so it never breaks an existing working
      solve; reduce ``n_slices`` / raise ``degree`` to clear it.
    """
    R = np.asarray(R_eff)
    T = np.asarray(T_eff)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    if tot.size and not np.all(np.isfinite(tot)):
        raise ValueError(
            "PMMStack.solve: non-finite total efficiency (R+T = "
            f"{np.array2string(np.asarray(tot), precision=4)}); a NaN/inf "
            "material index or permittivity reached the solve (check "
            "n_substrate / n_superstrate and the layer eps values).")
    worst = float(np.max(tot)) if tot.size else 0.0
    least = float(np.min(tot)) if tot.size else 0.0
    if least < -1e-9:
        raise ValueError(
            f"PMMStack.solve: NEGATIVE total efficiency (min R+T = "
            f"{least:.3e}); the efficiency normalisation is non-physical -- a "
            "gain or non-propagating incidence medium slipped past the entry "
            "guards (kz_inc < 0 negates every order).")
    if worst > 1.0 + 1e-2:
        import warnings
        warnings.warn(
            f"PMMStack.solve: energy not conserved (max R+T = {worst:.3g} > 1) "
            "-- a near-singular interface mode-match in the cascade (e.g. too "
            "many tapered slices). The result is unreliable; reduce n_slices or "
            "raise degree.", stacklevel=3)


class PMMStack:
    """Multilayer 1-D grating stack solved by the Polynomial Modal Method -- the
    spectral-element counterpart of :class:`~lumenairy.elements.rcwa.RCWAStack`.

    Compose anisotropic (or isotropic) 1-D patterned layers and uniform spacers
    between a superstrate and substrate, set the incident plane wave, and solve
    once for the diffraction efficiencies of both incident polarizations plus the
    zeroth-order ``2x2`` Jones reflection.  The whole stack is solved on the
    UNION of every layer's walls (one shared nodal grid), so each layer converges
    spectrally in ``degree`` with no Fourier truncation in-plane.

    Example
    -------
    >>> st = PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=20)
    >>> st.add_layer(0.2e-6, eps=2.1)                       # uniform spacer
    >>> st.add_layer(0.3e-6, segments=[(0.5, lc), (0.5, 1.0)])  # patterned
    >>> orders, R, T, jones = st.set_source(0.55e-6, angle=0.2).solve()

    Parameters
    ----------
    period : float
        Grating period (metres).
    n_substrate, n_superstrate : complex, optional
        Transmission / incidence half-space indices.
    degree : int, optional
        Polynomial degree per spectral element (the spectral knob).  Default 16.
    elements_per_region, grade, far_field_orders : as in
        :func:`pmm_jones_1d_segments`.
    factorization : {'auto', 'convection', 'covariant'}, optional
        Slant treatment (as in :func:`pmm_jones_1d_slanted`).  ``'auto'``
        (default) uses the SPECTRAL covariant oblique-coordinate generator for
        an IN-PLANE stack whose layers share a single non-zero slant, and the
        (algebraic-but-fully-general) convection generator otherwise --
        vertical, MIXED-slant, or slanted OUT-OF-PLANE stacks (the covariant
        oblique frame is per-slant, so a mixed-slant cascade falls back to
        convection; a slanted out-of-plane stack falls back because the
        covariant layout's DISCONTINUOUS off-plane TM channel carries a known
        ~0.1 per-order factorization defect -- see the limitation note on
        :func:`pmm_jones_1d_slanted`).  ``'covariant'`` forces the spectral
        path (raises on mixed/zero slant; carries out-of-plane with that
        documented limitation); ``'convection'`` forces the general path.

    Notes
    -----
    Anisotropic / Jones throughout (scalar layers are promoted to isotropic
    tensors), FULL ``(3,3)`` tensors IN-PLANE OR OUT-OF-PLANE
    (``eps_xz/yz/zx/zy``), normal or oblique incidence, NumPy (not JAX).  Layers
    may be VERTICAL or SLANTED (``add_layer(..., slant_angle=...)``), in-plane or
    out-of-plane, and freely mixed: an all-vertical-in-plane stack uses the
    symmetric ``+/-q`` cascade (bit-identical to the prior release), and any
    slanted OR out-of-plane layer promotes the whole stack to the general
    forward/backward S-matrix (solved by the div-conforming metric generator,
    which carries the slant fold AND the out-of-plane ezz-Schur).  A slanted
    out-of-plane layer reaches the same ~1e-4 wall-normal per-order floor as the
    single-layer solver (validated vs an RCWA tensor z-staircase; energy
    conserves).  The modal forward set uses the z-Poynting-flux
    selector (as the multi-region single-layer solver), so the many-element shared
    grid stays resonance-free.
    """

    def __init__(self, period, *, n_substrate=1.0, n_superstrate=1.0,
                 degree=16, elements_per_region=1, grade=True,
                 far_field_orders=21, n_orders=None, factorization="auto",
                 min_feature=None):
        far_field_orders = _resolve_order_count(far_field_orders, n_orders)
        if int(degree) < 2:
            raise ValueError("PMMStack: degree must be >= 2.")
        if factorization not in ("auto", "convection", "covariant"):
            raise ValueError(
                "PMMStack: factorization must be 'auto', 'convection' or "
                f"'covariant', got {factorization!r}.")
        self.period = float(period)
        # A JAX half-space index stays RAW so its gradient flows (the
        # differentiable dispatch in solve()); _C() would sever the trace.
        self.n_sub = (n_substrate if (callable(n_substrate)
                                      or is_jax_array(n_substrate))
                      else _C(n_substrate))
        self.n_sup = (n_superstrate if (callable(n_superstrate)
                                        or is_jax_array(n_superstrate))
                      else _C(n_superstrate))
        self.degree = int(degree)
        self.n_el = int(elements_per_region)
        self.grade = bool(grade)
        self.factorization = factorization
        self.ffo = int(far_field_orders)
        # PHYSICAL wall-snap threshold for the shared union grid (item 3a):
        # ABSOLUTE metres; default period*1e-5 -- far above float noise, far
        # below intentional features, and cross-layer-pairs-only either way.
        # ALSO THE COST KNOB for dense staircases (application feedback
        # 2026-06-10): snapping colliding cross-layer walls shrinks the
        # union grid -- measured 5.7x (321 s -> 56 s) on an ns8 coated taper
        # at min_feature=1.5e-9, at a ~1.6% geometry-perturbation cost
        # (+-0.75 nm wall moves).
        self.min_feature = (float(period) * 1e-5 if min_feature is None
                            else float(min_feature))
        self._layers = []                          # (thickness, segments)
        self._taper_recipes = []                   # (i0, count, method, kwargs)
        self._src = None
        self._modal = None      # per-order amplitudes of the last solve (B)

    def _as_tensor(self, eps):
        if is_jax_array(eps):
            # Keep the trace alive: promote in jnp (np.asarray on a tracer
            # raises; on a concrete jax array it would sever the gradient).
            import jax.numpy as jnp
            M = jnp.asarray(eps, dtype=jnp.complex128)
            if M.ndim == 0:
                M = M * jnp.eye(3, dtype=jnp.complex128)
            if M.shape[-2:] != (3, 3):
                raise ValueError(
                    "PMMStack.add_layer: each eps must be a scalar or a "
                    "(3, 3) permittivity tensor.")
            return M
        M = np.asarray(eps, dtype=_C)
        if M.ndim == 0:
            M = M * np.eye(3, dtype=_C)
        if M.shape[-2:] != (3, 3):
            raise ValueError(
                "PMMStack.add_layer: each eps must be a scalar or a (3, 3) "
                "permittivity tensor.")
        return M

    @staticmethod
    def _is_oop(M):
        """True if the (3,3) tensor has out-of-plane coupling (eps_xz/yz/zx/zy)."""
        scale = max(float(np.max(np.abs(M))), 1.0)
        return float(np.max(np.abs(M[[0, 1, 2, 2], [2, 2, 0, 1]]))) > 1e-9 * scale

    def add_layer(self, thickness, *, segments=None, eps=None, slant_angle=0.0):
        """Append a layer.  Give exactly one of ``eps`` (uniform: scalar or
        ``(3,3)`` tensor) or ``segments`` (a list of ``(width_fraction, eps)`` --
        each ``eps`` scalar or ``(3,3)``; the widths are FRACTIONS of the period,
        each ``> 0``, and must sum to 1 within ``1e-6`` -- ENFORCED, as in
        :func:`pmm_jones_1d_segments`; absolute (metre-valued) widths raise).
        ``slant_angle``
        (radians) tilts the layer's straight side-walls from the vertical (``0``
        = vertical); a slanted layer is solved by the div-conforming covariant-
        metric generator and cascaded via the general fwd/back S-matrix, so a
        stack may MIX vertical and slanted layers.  Returns ``self``."""
        self._internal = None   # supersedes any retained internals (audit P1-04)
        self._modal = None      # ... and retained per-order amplitudes (B)
        if (segments is None) == (eps is None):
            raise ValueError(
                "PMMStack.add_layer: give exactly one of `segments` or `eps`.")
        # DISPERSIVE materials (device-geometry roadmap item 5, 2026-06-10):
        # any eps slot may be a ``wl -> value`` callable; such a stack is
        # solved with solve_vs_wavelength (solve() raises -- no single
        # wavelength is defined).  Material KEYS (strings, item 8) are also
        # stored raw and resolved via solve(materials={...}).
        if eps is not None:
            segs = [(1.0, eps if (callable(eps) or isinstance(eps, str))
                     else self._as_tensor(eps))]
        else:
            if len(segments) < 1:
                raise ValueError("PMMStack.add_layer: empty segments.")
            segs = [(float(w), e if (callable(e) or isinstance(e, str))
                     else self._as_tensor(e)) for w, e in segments]
            # Width-FRACTION validation (audit M2 2026-07-25) -- the same check
            # the 1-D sibling ``pmm_jones_1d_segments`` runs in
            # ``_segment_walls`` (sum to 1 within 1e-6, every width > 0),
            # replicated here because ``_pmm_union_grid`` normalises with
            # ``cw[-1] = 1.0`` and therefore SILENTLY CLIPS a bad list: widths
            # summing to 1.4 solved bit-identically to the clipped structure,
            # and METRE-valued widths (the common misuse -- absolute widths
            # instead of fractions) solved as a physically unrelated,
            # energy-clean near-uniform slab.  Energy conservation provably
            # cannot detect either, so the input must be rejected here.
            _w = np.asarray([w for w, _e in segs], dtype=float)
            if not np.all(np.isfinite(_w)) or np.any(_w <= 0.0):
                raise ValueError(
                    "PMMStack.add_layer: every segment width FRACTION must be "
                    f"finite and > 0 (got {list(_w)}).")
            _s = float(_w.sum())
            if abs(_s - 1.0) > 1e-6:
                raise ValueError(
                    "PMMStack.add_layer: segment width FRACTIONS must sum to 1 "
                    f"(got {_s:.6g}).  They are fractions of the period, NOT "
                    "absolute widths -- a metre-valued list such as "
                    "[(0.25e-6, eps), ...] on a 0.5e-6 period must be written "
                    "[(0.5, eps), ...].  (Unvalidated widths were silently "
                    "CLIPPED to sum 1 and solved as a different structure that "
                    "still conserved energy.)")
        # Thickness validation (audit P3-30) -- mirror PMM2DStack.add_layer:
        # finite and > 0.  A CONCRETE JAX thickness is checked; a TRACED one
        # (under jit/grad) has no value to range-check and skips the guard.
        if is_jax_array(thickness):
            try:
                t_c = float(np.asarray(thickness))
            except Exception:
                t_c = None                 # tracer: validated only if concrete
            if t_c is not None and (t_c <= 0.0 or not np.isfinite(t_c)):
                raise ValueError("PMMStack.add_layer: thickness must be > 0")
        else:
            if float(thickness) <= 0.0 or not np.isfinite(float(thickness)):
                raise ValueError("PMMStack.add_layer: thickness must be > 0")
            thickness = float(thickness)
        self._layers.append((thickness, segs, float(slant_angle)))
        return self

    @staticmethod
    def _ridge_slice_segments(duty, centre, eps_ridge, eps_groove,
                              tol=1e-12):
        """One vertical slice of a SINGLE-ridge grating as the consecutive
        ``(width_fraction, eps)`` segment list, laid out from ``u = 0``.

        ``duty`` is the ridge fraction (assumed strictly inside ``(0, 1)`` --
        the callers short-circuit the vanished-ridge / vanished-groove ends)
        and ``centre`` its centre in period fractions (it may sit anywhere on
        the real line: the ridge interval is reduced mod 1).  The ridge is the
        HALF-OPEN period-1 interval ``[centre - duty/2, centre + duty/2)``,
        which is EXACTLY the convention
        :meth:`RCWAStack.add_tapered_grating` rasterizes, so the two packages
        build the same staircase geometry -- laterally EXACT here, lattice-
        quantised there.

        Two layouts, both WRAP-aware:

        * ``[groove | ridge | groove]`` -- the ridge lies inside the cell;
        * ``[ridge_head | groove | ridge_tail]`` -- the ridge crosses ``u = 1``.

        The widths are built so the list sums to 1 to within one ULP even
        though only ``duty`` and the total groove ``1 - duty`` are computed
        directly (:meth:`add_layer` enforces ``sum == 1`` within ``1e-6`` and
        every width ``> 0``): the centred case reproduces the historical
        ``[edge, duty, edge]`` triple BIT-for-BIT because
        ``(1 - duty)/2 == 0.5 - duty/2`` exactly in binary floating point
        (halving is exact).  Slivers narrower than ``tol`` period fractions
        (``1e-18`` m on a micron pitch -- far below the ``min_feature``
        union-grid snap) are DROPPED rather than handed to the nodal grid as a
        degenerate element."""
        groove = 1.0 - duty                     # total groove fraction
        lo = float(centre - 0.5 * duty) % 1.0   # ridge lower wall, in [0, 1)
        if lo + duty > 1.0 + tol:               # ... the ridge wraps u = 1
            tail = 1.0 - lo                     # ridge part at u in [lo, 1)
            raw = ((duty - tail, eps_ridge), (groove, eps_groove),
                   (tail, eps_ridge))
        else:
            raw = ((lo, eps_groove), (duty, eps_ridge),
                   (groove - lo, eps_groove))
        return [(w, e) for w, e in raw if w > tol]

    def add_tapered_grating(self, thickness, *, eps_ridge, eps_groove,
                            duty_bottom, duty_top=None, n_slices=8,
                            rule="midpoint", shear=0.0):
        """Append a 1-D grating with SLANTED / TRAPEZOIDAL sidewalls as an
        auto-sliced z-staircase of thin VERTICAL PMM layers -- the spectral-
        element counterpart of :meth:`RCWAStack.add_tapered_grating`.

        The ridge's duty cycle varies linearly with depth from
        ``duty_top`` (top, ``zeta = 0``) to ``duty_bottom`` (bottom, ``zeta = 1``);
        each slice is a vertical binary grating whose walls are resolved EXACTLY
        by the nodal grid (no Fourier/Gibbs floor in x -- the PMM advantage), so
        the only approximation is the z-staircase of the taper (a true trapezoid
        is the ``n_slices -> infinity`` limit).  ``duty_top == duty_bottom`` gives
        the usual vertical binary grating.  ``shear`` additionally walks the
        ridge CENTRE across the cell with depth (a parallelogram / sheared
        profile), wrap-aware.

        COST NOTE: every slice's two walls enter the stack's SHARED union grid, so
        the global node count -- and thus each layer's eig -- grows with
        ``n_slices``.  This is laterally exact and beats an RCWA z-staircase per
        slice (no Fourier floor), but is practical for MODEST ``n_slices``; for a
        scalable no-floor taper prefer a single covariant taper-metric layer (a
        roadmap item).  A wavelength sweep should use
        :meth:`solve_vs_wavelength`, which assembles the (large) shared grid ONCE.

        MEASURED BUDGET (audit W9).  Cost is ``~O(n_slices^3.4)`` end to end
        (``P = 1 um``, ``wl = 633 nm``, ``d = 300 nm``, ``eps_ridge = 4``,
        ``duty 0.30 -> 0.62``, ``degree = 12``: ``n_slices`` 4/8/16 ->
        0.30/2.97/34.1 s), because the union grid grows with the slice count and
        every layer's eig grows with it.  The z-staircase itself converges
        ``O(1/n_slices^2)`` with either rule -- CONFIRMED on the cross-package
        oracle (the RCWA twin at ``raster='area'``, whose realised duty is exact
        so the ONLY z error is the staircase; reference ``n_slices = 768``): the
        max error over ``(R, T)`` of orders ``-1, 0, +1`` and both polarizations
        falls 3.82e-03 / 1.27e-03 / 3.34e-04 / 8.47e-05 / 2.18e-05 / 5.70e-06 at
        ``n_slices`` 8 / 16 / 32 / 64 / 128 / 256 -- a factor 3.9 per doubling,
        and every one of the 12 components shares it (3.5-4.6 at 32 -> 64).

        DO NOT bother RICHARDSON-extrapolating in ``n_slices``.  It was measured
        across 8 designs (duty ranges 0.10-0.85, lossy, oblique 0.17/0.45,
        ``eps_ridge`` 4/9, deep + sheared, with and without shear) against the
        EQUAL-COST comparator ``f(2n)`` -- ``cost(n) + cost(2n) ~ 1.1 cost(2n)``
        here -- and the ``(n, 2n)``, ``p = 2`` two-point Richardson gains only
        3.0x / 7.4x / 14.5x on the clean designs and 1.0x / 0.95x / 1.17x
        (i.e. NOTHING) on the steep-duty, narrow-duty, high-contrast and deep
        designs at ``n = 8/16/32``.  Multi-exponent (``1/n + 1/n^2``) and
        3-point FITTED-exponent variants are outright WORSE (fitted-p at
        ``n = 4, 5, 6`` returned 9.2e-03 against ``f(6) = 4.4e-03``: the
        per-component exponent fit amplifies solver noise).  The staircase error
        is not a single clean power law per component below ``n ~ 16``.

        Parameters
        ----------
        thickness : float
            Grating thickness (metres).
        eps_ridge, eps_groove : complex or (3, 3)
            Ridge / groove permittivity (scalar or full tensor; PUBLIC
            ``Im(eps) > 0``).
        duty_bottom : float
            Ridge fraction at the bottom of the grating, in ``[0, 1]``.
        duty_top : float, optional
            Ridge fraction at the top; defaults to ``duty_bottom`` (vertical).
        n_slices : int, optional
            Staircase slice count (the convergence knob; default 8).
        rule : {'midpoint', 'trapezoid'}, optional
            Sample each slice's duty at its centre (``'midpoint'``, default) or
            average its two edges (``'trapezoid'``).  For the LINEAR duty ramp
            this builder lays down these are the SAME RULE -- ``0.5*(k/n +
            (k+1)/n) == (k + 0.5)/n`` exactly -- so ``'trapezoid'`` is a no-op
            up to last-bit rounding (audit W9; MEASURED bit-identical for 5 of
            the 7 slices at ``n_slices = 7`` and 1 ULP apart for the other 2).
            It is kept
            for signature parity with :meth:`add_graded_layer`, where the
            profile is arbitrary and the two rules genuinely differ.
        shear : float, optional
            SHEARED (parallelogram) sidewalls -- the EXACT convention of
            :meth:`RCWAStack.add_tapered_grating` (audit GAP1, v5.14.1),
            mirrored here so the two packages build the IDENTICAL staircase:
            the ridge CENTRE is ``0.5 + shear * (zeta - 0.5)`` in period
            fractions, i.e. it shifts by ``shear`` PERIODS from top to bottom
            about the mid-depth plane (so
            ``shear = depth * tan(wall_angle) / period`` renders a wall tilt).
            ``zeta`` is sampled per slice by the SAME ``rule`` as the duty.
            Wrap-aware (the ridge may cross the cell edge, giving the
            ``[ridge_head, groove, ridge_tail]`` layout).  Default 0 (centred,
            BIT-identical to the pre-``shear`` builder).

        Notes
        -----
        CROSS-PACKAGE STAIRCASE PARITY (audit W9).  At the SAME
        ``(n_slices, duty_bottom, duty_top, shear)`` the RCWA sibling's
        rasterized staircase converges to this laterally-exact one as its
        ``n_x`` grows (``O(1/n_x)``, the ``raster='hard'`` quantisation).
        MEASURED on ``P = 1 um``, ``wl = 633 nm``, ``d = 300 nm``,
        ``eps_ridge = 4``, ``n_sub = 1.5``, ``theta = 0.17`` rad,
        ``duty 0.30 -> 0.62``, ``shear = 0.35``, ``n_slices = 6``,
        ``degree = 12`` / ``nox = 9``; error is ``max|x_RCWA - x_PMM|`` over
        the propagating ``(R, T)`` of both polarizations::

            n_x |  err     | ratio
            512 | 1.53e-03 |
           2048 | 2.60e-04 | 5.9x
           8192 | 8.98e-05 | 2.9x

        The two builders' realised staircase GEOMETRY agrees to the raster
        lattice exactly: rasterizing these segments on the RCWA pixel-centre
        lattice reproduces its ``eps_cell`` for every slice at
        ``n_x = 64..4096``, ``shear = -1.7 .. 1.7`` (measured 0 mismatched
        pixels), including the wrapping cases.
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_tapered_grating: n_slices must be >= 1, got {n_slices}.")
        if rule not in ("midpoint", "trapezoid"):
            raise ValueError(
                f"add_tapered_grating: rule must be 'midpoint' or 'trapezoid', "
                f"got {rule!r}.")
        dt = float(duty_bottom if duty_top is None else duty_top)
        db = float(duty_bottom)
        for d in (db, dt):
            if not (0.0 <= d <= 1.0):
                raise ValueError(
                    f"add_tapered_grating: duty cycles must be in [0, 1], got "
                    f"duty_top={dt}, duty_bottom={db}.")
        sh = float(shear)
        if not np.isfinite(sh):
            raise ValueError(
                f"add_tapered_grating: shear must be finite, got {shear!r}.")
        self._taper_recipes.append((len(self._layers), n,
                                    "add_tapered_grating",
                                    dict(thickness=thickness,
                                         eps_ridge=eps_ridge,
                                         eps_groove=eps_groove,
                                         duty_bottom=duty_bottom,
                                         duty_top=duty_top, rule=rule,
                                         shear=shear)))
        dz = float(thickness) / n
        for k in range(n):
            if rule == "midpoint":
                zeta = (k + 0.5) / n
            else:
                zeta = 0.5 * (k / n + (k + 1) / n)
            duty = dt + (db - dt) * zeta
            tol = 1e-9
            if duty <= tol:                       # ridge vanished -> all groove
                self.add_layer(dz, eps=eps_groove)
            elif duty >= 1.0 - tol:               # groove vanished -> all ridge
                self.add_layer(dz, eps=eps_ridge)
            else:                                 # ridge between grooves; the
                # RCWA centre convention (bit-identical [edge, duty, edge] at
                # shear = 0 -- see _ridge_slice_segments)
                self.add_layer(dz, segments=self._ridge_slice_segments(
                    duty, 0.5 + sh * (zeta - 0.5), eps_ridge, eps_groove))
        return self

    def add_sheared_grating(self, thickness, *, eps_ridge, eps_groove, duty,
                            shear, centre=0.5):
        """Append a PURE-SHEAR (parallelogram) binary grating as ONE EXACT
        SLANTED layer -- NO z-staircase at all (audit W9).

        ``add_tapered_grating(d, ..., duty_top=duty_bottom, shear=s)`` builds
        exactly this geometry as an ``n_slices`` staircase; but a parallelogram
        is the ONE taper the shipped covariant / convection machinery solves
        EXACTLY, because ``u = x - z tan(phi)`` keeps the modal coefficients
        z-independent (the symmetric trapezoid does not -- see the note on
        :meth:`add_tapered_grating`).  Same ``shear`` units, same centre law:
        the ridge centre is ``centre + shear * (zeta - 0.5)`` in period
        fractions, so this and the staircase describe the SAME structure and
        the two are interchangeable at the call site.

        Parameters
        ----------
        thickness : float
            Grating thickness (metres).
        eps_ridge, eps_groove : complex or (3, 3)
            Ridge / groove permittivity (PUBLIC ``Im(eps) > 0``).
        duty : float
            Ridge fraction, strictly inside ``(0, 1)``.
        shear : float
            Lateral walk of the ridge centre from top to bottom, in PERIODS
            (the same knob as :meth:`add_tapered_grating`'s).  Realised as
            ``slant_angle = arctan(shear * period / thickness)``; ``0`` gives a
            plain VERTICAL layer.  Wrap-aware.
        centre : float, optional
            Ridge centre at MID-DEPTH (``zeta = 0.5``) in period fractions;
            default ``0.5``.  Lateral position is a symmetry of an otherwise
            uniform stack, so it is (numerically) unobservable in the
            efficiencies and the zeroth-order Jones -- MEASURED 8.8e-06 between
            ``centre - shear/2`` and ``centre + shear/2`` at degree 16, i.e.
            300x below this path's own per-order floor below.

        Notes
        -----
        MEASURED BUDGET (audit W9), against the ``n_slices`` staircase of the
        IDENTICAL geometry (``P = 1 um``, ``wl = 633 nm``, ``d = 300 nm``,
        ``eps_ridge = 4``, ``duty = 0.45``, ``shear = 0.35`` -- a 49.4 deg wall
        tilt, ``theta = 0.17`` rad, ``n_sub = 1.5``, ``degree = 10``; error is
        ``max|x - x_staircase(n_slices=20)|`` over ``(R, T)`` of orders
        ``|m| <= 2``, both polarizations)::

            route                       | err       | time    | R+T - 1
            add_tapered_grating ns=4    | 2.59e-02  |  0.25 s | 2e-11
            add_tapered_grating ns=8    | 6.93e-03  |  2.27 s | 3e-11
            add_tapered_grating ns=12   | 2.63e-03  |  7.53 s | 4e-11
            add_tapered_grating ns=16   | 9.01e-04  | 23.04 s | 5e-11
            add_sheared_grating deg=8   | 3.16e-03  |  0.03 s | 6.9e-05
            add_sheared_grating deg=12  | 2.93e-03  |  0.05 s | 1.9e-05
            add_sheared_grating deg=16  | 2.91e-03  |  0.13 s | 3.3e-06
            add_sheared_grating deg=24  | 2.93e-03  |  0.24 s | 9.9e-07

        READ THIS TABLE BOTH WAYS.  At a matched budget the exact layer WINS
        BIG -- 2.93e-03 in 0.05 s against the staircase's 2.59e-02 in 0.25 s
        (8.9x more accurate at a fifth of the cost), and it reaches the
        staircase's ``n_slices = 12`` answer ~150x faster.  But it PLATEAUS: at
        ``degree >= 12`` refining does NOT help, because the residual is the
        slant solver's OWN wall-normal per-order floor (see the limitation note
        on :func:`~lumenairy.elements.pmm.pmm_jones_1d_slanted`), and the
        staircase overtakes it by ``n_slices = 16``.  That floor GROWS WITH THE
        WALL TILT (``sec^2`` conditioning) -- MEASURED plateau, same probe::

            wall tilt      | plateau  | matches the staircase at
            45.0 deg, obl. | 1.76e-03 | n_slices ~ 11  (19.2 s -> 0.10 s)
            49.4 deg, th=0 | 1.89e-03 | n_slices ~ 12  ( 7.4 s -> 0.05 s)
            49.4 deg, obl. | 2.93e-03 | n_slices ~ 12  ( 7.5 s -> 0.05 s)
            69.4 deg, obl. | 1.16e-02 | n_slices ~  8  ( 1.9 s -> 0.06 s)

        (the 45 deg row is a different design -- ``duty = 0.60``,
        ``d = 200 nm``, ``shear = 0.20``, ``theta = 0.30`` -- so the four rows
        are four designs, not one lucky probe)

        so use this method for design SWEEPS and for accuracy targets at or
        above its plateau for your tilt (~2e-03 at 50 deg, ~1e-02 at 70 deg),
        and switch to the staircase below that.  Energy closure is this path's
        own too: MEASURED ``|R+T-1|`` 6.9e-05 (degree 8) falling to ~1e-06
        (degree 20-24), against the staircase's ~5e-11.

        RESTRICTIONS inherited from the slant path: a slanted layer promotes
        the whole stack to the generalized forward/backward S-matrix, and
        :meth:`solve_vs_wavelength`, :meth:`prepare`,
        ``solve(retain_internal=True)`` (so :meth:`internal_field` /
        :meth:`layer_absorption`), the JAX (differentiable) dispatch and
        conical incidence (``phi != 0``) all reject slanted stacks -- they
        raise ``NotImplementedError``, loudly (VERIFIED here for
        ``solve_vs_wavelength``, ``prepare``, ``retain_internal`` and conical;
        the JAX guard is the shared all-vertical one already pinned by
        ``test_audit_w3_pmm_jax_guards``).
        ``stabilize='slices'`` has nothing to re-slice here (no staircase
        recipe) and warns instead.  Where you need any of those, build the same
        geometry with ``add_tapered_grating(..., duty_top=duty_bottom,
        shear=...)`` and pay the staircase.  Returns ``self``."""
        d = float(duty)
        if not (0.0 < d < 1.0):
            raise ValueError(
                f"add_sheared_grating: duty must be strictly inside (0, 1), "
                f"got {duty}.  A vanished ridge or groove is a uniform layer "
                f"-- use add_layer(thickness, eps=...).")
        sh = float(shear)
        if not np.isfinite(sh):
            raise ValueError(
                f"add_sheared_grating: shear must be finite, got {shear!r}.")
        # The slant frame is anchored at the layer TOP (u = x - z tan(phi)), so
        # the u-frame ridge centre is the zeta = 0 one; the lab centre is then
        # ``centre + shear * (zeta - 0.5)`` -- add_tapered_grating's law.
        segs = self._ridge_slice_segments(d, float(centre) - 0.5 * sh,
                                          eps_ridge, eps_groove)
        phi = float(np.arctan(sh * self.period / float(thickness)))
        return self.add_layer(thickness, segments=segs, slant_angle=phi)

    @staticmethod
    def _ridges_to_segments(period, ridges, eps_groove):
        """Convert absolute ``(center, width, eps)`` ridges (metres) into the
        consecutive ``(width_fraction, eps)`` segment list of one vertical
        slice.  Wrap-aware (a ridge may cross the period edge); raises on
        overlapping ridges (ambiguous geometry must be explicit)."""
        intervals = []                      # (lo, hi, eps) in fractions
        for c, w, eps in ridges:
            if w <= 0.0:
                continue
            if w >= period:
                raise ValueError(
                    "add_tapered_ridges: a ridge width must be < period.")
            lo = ((c - 0.5 * w) / period) % 1.0
            hi = lo + w / period
            if hi <= 1.0 + 1e-15:
                intervals.append((lo, min(hi, 1.0), eps))
            else:                           # wraps the period edge
                intervals.append((lo, 1.0, eps))
                intervals.append((0.0, hi - 1.0, eps))
        ivs = sorted(intervals)
        for (l1, h1, _e1), (l2, _h2, _e2) in zip(ivs, ivs[1:]):
            if l2 < h1 - 1e-12:
                raise ValueError(
                    f"add_tapered_ridges: ridges overlap (intervals "
                    f"[{l1:.6g}, {h1:.6g}] and starting {l2:.6g} in period "
                    f"fractions); merge or separate them explicitly.")
        walls = sorted({0.0, 1.0} | {x for lo, hi, _e in ivs
                                     for x in (lo, hi)})
        segs = []
        for a, b in zip(walls, walls[1:]):
            if b - a < 1e-15:
                continue
            mid = 0.5 * (a + b)
            eps = eps_groove
            for lo, hi, e in ivs:
                if lo <= mid < hi:
                    eps = e
                    break
            segs.append((b - a, eps))
        return segs

    def add_tapered_ridges(self, thickness, *, ridges, eps_groove,
                           n_slices=8, rule="midpoint", shear=0.0):
        """Append a MULTI-RIDGE tapered grating as an auto-sliced z-staircase
        (device-geometry roadmap item 1, 2026-06-10) -- the N-feature
        generalization of :meth:`add_tapered_grating`.

        ``ridges`` is a list of ``(center, w_top, w_bottom, eps)`` tuples in
        ABSOLUTE metres: each ridge narrows (or widens) linearly from
        ``w_top`` (at the top, ``zeta = 0``) to ``w_bottom`` (bottom,
        ``zeta = 1``) ABOUT ITS OWN FIXED CENTER -- the load-bearing detail: a
        width-sequence construction that left-anchors each ridge drifts its
        center as it tapers (a real ~3%-of-out-coupling bug in the device
        port this roadmap item came from).  ``eps_groove`` fills the
        remainder.  Wrap-aware; overlapping ridges raise.  The single
        centred-ridge case reproduces :meth:`add_tapered_grating`.

        Each ``eps`` may be scalar, ``(3, 3)``, a ``wl -> value`` callable
        (solve with :meth:`solve_vs_wavelength`) or a material KEY string
        (resolve with ``materials={...}``).

        ``shear`` (audit W9) walks EVERY ridge centre together, in the same
        units as :meth:`add_tapered_grating`'s: ``shear`` PERIODS from top to
        bottom about the mid-depth plane, i.e. each centre becomes
        ``center + shear * period * (zeta - 0.5)`` (so the shear is a rigid
        lateral drift of the whole tooth pattern -- the fabrication
        undercut/lean, not a change of pitch).  ``zeta`` is sampled per slice
        by the same ``rule`` as the widths, wrap-aware, and the overlap guard
        still applies at every slice.  Default 0 -- BIT-identical to the
        pre-``shear`` builder.  A single ridge at ``center = period/2``
        reproduces :meth:`add_tapered_grating` at the same ``shear``
        (MEASURED agreement of the realised segment widths: ``<= 1.2e-16``).
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_tapered_ridges: n_slices must be >= 1, got {n_slices}.")
        if rule not in ("midpoint", "trapezoid"):
            raise ValueError(
                f"add_tapered_ridges: rule must be 'midpoint' or 'trapezoid', "
                f"got {rule!r}.")
        sh = float(shear)
        if not np.isfinite(sh):
            raise ValueError(
                f"add_tapered_ridges: shear must be finite, got {shear!r}.")
        rid = [(float(c), float(wt), float(wb), e)
               for c, wt, wb, e in ridges]
        self._taper_recipes.append((len(self._layers), n,
                                    "add_tapered_ridges",
                                    dict(thickness=thickness, ridges=ridges,
                                         eps_groove=eps_groove, rule=rule,
                                         shear=shear)))
        dz = float(thickness) / n
        for k in range(n):
            if rule == "midpoint":
                zeta = (k + 0.5) / n
            else:
                zeta = 0.5 * (k / n + (k + 1) / n)
            dc = sh * (zeta - 0.5) * self.period     # exactly 0.0 at shear = 0
            slice_ridges = [(c + dc, wt + (wb - wt) * zeta, e)
                            for c, wt, wb, e in rid]
            segs = self._ridges_to_segments(self.period, slice_ridges,
                                            eps_groove)
            self.add_layer(dz, segments=segs)
        return self

    def plot_geometry(self, ax=None, material_names=None):
        """Draw the stack cross-section EXACTLY as the solver sees it (device-
        geometry roadmap item 7, 2026-06-10): one rectangle per
        (layer x segment) from the stored analytic walls (crisp nm features,
        no pixelation), substrate/superstrate bands, and a legend keyed by
        permittivity identity.  "The picture IS the model": the renderer reads
        the same ``_layers`` the solve consumes, so figure and physics cannot
        silently diverge.

        ``material_names`` optionally maps a complex eps (or a material key
        string) to a display name; a stack built via
        ``SegmentStackGeometry.to_pmm_stack(materials=...)`` already carries
        its key names, so legends read ``"LC"``/``"Ta"`` automatically
        (application feedback 2026-06-10).

        Returns
        -------
        matplotlib Axes -- use ``ax.figure.savefig(...)`` to save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 4))
        names = dict(getattr(self, "_material_names", {}) or {})
        names.update(material_names or {})

        def _key(e):
            if isinstance(e, str):
                return e
            if callable(e):
                return getattr(e, "__name__", "dispersive")
            M = np.asarray(e)
            return complex(M[0, 0]) if M.ndim == 2 else complex(M)

        def _label(k):
            if k in names:
                return str(names[k])
            if isinstance(k, str):
                return k
            return (f"eps={k.real:.3g}" if abs(k.imag) < 1e-12
                    else f"eps={k.real:.3g}{k.imag:+.3g}j")

        seen = {}
        cmap = plt.get_cmap("tab10")
        total = sum(L[0] for L in self._layers)
        seg_labels = getattr(self, "_segment_labels", None)
        z = 0.0
        for li, (t, segs, _slant) in enumerate(self._layers):
            x = 0.0
            for j, (w, e) in enumerate(segs):
                # prefer the material KEY label (a SegmentStackGeometry
                # export) so twin keys with the same eps stay distinct
                if (seg_labels and li < len(seg_labels)
                        and j < len(seg_labels[li])):
                    k = seg_labels[li][j]
                else:
                    k = _key(e)
                if k not in seen:
                    seen[k] = cmap(len(seen) % 10)
                ax.add_patch(Rectangle((x * self.period, -z - t),
                                       w * self.period, t,
                                       facecolor=seen[k], edgecolor="none"))
                x += w
            z += t
        band = 0.12 * total if total > 0 else 1.0
        for nval, z0, lab in ((self.n_sup, band, "superstrate"),
                              (self.n_sub, -total - band, "substrate")):
            ax.add_patch(Rectangle((0.0, z0 - (band if z0 > 0 else 0.0)),
                                   self.period, band, facecolor="0.9",
                                   edgecolor="none"))
            ax.text(0.02 * self.period, z0 - 0.5 * band, lab, fontsize=8,
                    va="center")
        handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in seen.values()]
        ax.legend(handles, [_label(k) for k in seen], fontsize=7,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_xlim(0.0, self.period)
        ax.set_ylim(-total - band, band)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m] (0 = top)")
        ax.set_title("PMMStack geometry (exact walls)")
        return ax

    def set_source(self, wavelength, *, angle=0.0, theta=None, phi=0.0):
        """Set the incident plane wave (vacuum wavelength [m], polar incidence
        ``angle`` [rad]).  ``theta`` is accepted as a cross-suite alias for
        ``angle`` (matching ``RCWAStack.set_source``'s polar angle); ``theta``
        WINS when both are supplied -- the SAME rule as ``RCWAStack.set_source``
        and the 1-D entry points, so ``set_source(angle=A, theta=T)`` resolves
        to ``T`` in every suite.

        ``phi`` [rad] is the CONICAL (out-of-plane) azimuth (audit conical Path
        B phase 4).  ``phi = 0`` is the classical x-z mount (the historical
        default, unchanged).  A ``phi != 0`` solve runs the native ``O(N)``
        conical reduction (``n_y = 0`` degenerate; the coupled 2N cascade), and
        is restricted to ALL-VERTICAL scalar / in-plane stacks with NumPy inputs
        (slant, out-of-plane tensors, JAX, ``stabilize`` and ``retain_internal``
        raise under ``phi != 0`` -- use ``PMM2DStack`` with y-invariant cells for
        those).  Returns ``self``."""
        self._internal = None   # supersedes any retained internals (audit P1-04)
        self._modal = None      # ... and retained per-order amplitudes (B)
        # Route through the CHECKED resolver (audit S1-7): a back-side angle
        # (|angle| >= pi/2, e.g. 2.5 rad) would otherwise be stored verbatim and
        # silently solve the supplementary front-side geometry (byte-identical
        # to pi - angle) -- the 1-D entry points already reject it, so match them.
        angle = _resolve_incidence_checked("PMMStack.set_source", angle, theta)
        self._src = dict(
            wl=wavelength if is_jax_array(wavelength) else float(wavelength),
            angle=angle if is_jax_array(angle) else float(angle),
            phi=phi if is_jax_array(phi) else float(phi))
        return self

    def _resliced_clone(self, dn):
        """A copy of this stack with every recorded taper builder re-sliced at
        ``n_slices + dn`` (hand-added layers copied verbatim) -- the
        slices-consensus probe of ``solve(stabilize='slices')``.  Returns
        ``None`` if no taper recipe is recorded."""
        if not self._taper_recipes:
            return None
        clone = PMMStack(self.period, n_substrate=self.n_sub,
                         n_superstrate=self.n_sup, degree=self.degree,
                         elements_per_region=self.n_el, grade=self.grade,
                         far_field_orders=self.ffo,
                         factorization=self.factorization,
                         min_feature=self.min_feature)
        recipes = sorted(self._taper_recipes, key=lambda r: r[0])
        i = 0
        ri = 0
        while i < len(self._layers):
            if ri < len(recipes) and recipes[ri][0] == i:
                i0, cnt, method, kw = recipes[ri]
                getattr(clone, method)(
                    n_slices=max(1, cnt + dn), **kw)
                i += cnt
                ri += 1
            else:
                t, segs, slant = self._layers[i]
                clone._layers.append((t, segs, slant))
                i += 1
        return clone

    def _holds_traced(self):
        """True when any stored input is a JAX array (the differentiable
        dispatch in solve(); the assemble-once paths are NumPy-only)."""
        return (is_jax_array(self.n_sub) or is_jax_array(self.n_sup)
                or (self._src is not None
                    and (is_jax_array(self._src["wl"])
                         or is_jax_array(self._src["angle"])))
                or any(is_jax_array(t) for t, _s, _a in self._layers)
                or any(is_jax_array(e) for _t, segs, _a in self._layers
                       for _w, e in segs))

    def _solve_conical(self, wl, angle, phi):
        """Native ``O(N)`` conical (``phi != 0``) multilayer solve (Path B phase
        4) -- the ``n_y = 0`` degenerate reduction.

        Every layer's coupled ``2N`` modes come from the SAME 2-D machinery the
        native conical single layer uses (:func:`_tensor_layer_modes` with
        ``oy = [0]``); segments are stored as ``(3, 3)`` tensors, so a scalar
        layer is the isotropic diagonal that reduces exactly to
        :func:`pmm_jones_1d_conical`.  Each patterned layer carries its OWN
        spectral-element grid + projector (no union grid -- the projection
        decouples layers, as in :class:`PMM2DStack`).  Vertical layers keep the
        ``+/-lam`` symmetric cascade; an out-of-plane layer promotes the whole
        stack to the generalized fwd/back S-matrix.  Closed with the shared
        conical far field.  Restricted to all-vertical NumPy stacks (the caller
        gates slant / JAX / stabilize / retain_internal)."""
        from ..rcwa._core import (
            _grazing_safe_wavelength,
            _interface_smatrix_general,
            _modes_to_M,
            _propagation_smatrix_general,
            _require_propagating_incidence,
        )
        from .conical import _conical_jones_farfield
        from .twod import (
            _build_axis,
            _homogeneous_modes,
            _interface_smatrix,
            _layer_modes_projected,
            _propagation_smatrix,
            _redheffer_star,
            _scalar_projected_ops,
        )
        from .twod_jones import _tensor_layer_modes

        # ---- PATTERNED layers -> the pure-nodal conical cascade -------------
        # (AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12): the Fourier-
        # projected layer-mode route below carries a RESOLUTION-INDEPENDENT
        # systematic error for ANY patterned layer (scalar or tensor -- the
        # defect is in the operator projection, not the factorization rule),
        # so its ky0=0 degenerate limit never reduced to the classical solve
        # (~3e-3 Jones / ~3.5 deg retardance, flat under refinement).  Route
        # every stack containing a patterned layer through the classical
        # nodal machinery generalized to ky0 (exact reduction at ky0=0).
        # IN-PLANE tensors only: a patterned stack carrying ANY out-of-plane
        # tensor is rejected loudly (the old path was silently wrong for it);
        # all-UNIFORM stacks (incl. out-of-plane tensors) keep the exact
        # Fourier path below (Berreman-validated).
        if any(len(segs) > 1 for (_t, segs, _sl) in self._layers):
            if any(self._is_oop(e) for (_t, segs, _sl) in self._layers
                   for (_w, e) in segs):
                raise NotImplementedError(
                    "PMMStack.solve (conical, phi != 0): a stack containing "
                    "a PATTERNED layer together with an out-of-plane tensor "
                    "(eps_xz/yz/zx/zy) is not supported -- the previous "
                    "projected route returned silently-wrong retardance for "
                    "patterned cells (AUDIT_PMM_CONICAL_PATTERNED_TENSOR_"
                    "BUG_2026_07_12) and the nodal conical cascade is "
                    "in-plane only.  Use the classical mount (phi=0) or "
                    "PMM2DStackPure.")
            from .conical import _conical_nodal_solve
            layer_specs = [
                (float(thk), [float(w) for w, _ in segs],
                 [np.asarray(e, dtype=_C) for _w, e in segs])
                for (thk, segs, _sl) in self._layers]
            o2, R_eff, T_eff, jones, modal = _conical_nodal_solve(
                self.period, layer_specs, _C(self.n_sup) ** 2,
                _C(self.n_sub) ** 2, wl, angle, phi, self.degree, self.n_el,
                self.grade, max(3, (self.ffo - 1) // 2),
                min_feature=self.min_feature,
                label="PMMStack.solve (conical)", return_modal=True)
            self._modal = modal          # AUDIT_DYNAMETA_CONSUMER_API_GAPS B
            _warn_stack_energy(R_eff, T_eff)
            # stack contract: the 1-D (m,) order array (pmm_jones_1d_segments
            # parity; y is degenerate so n_y == 0 for every order).
            return o2[:, 0], R_eff, T_eff, jones

        def _scalar_eps(M):
            """The scalar eps of an isotropic-diagonal (3, 3) tile entry, else
            None (a genuine tensor -> the tensor path)."""
            M = np.asarray(M, dtype=_C)
            d = np.diag(M)
            scale = max(float(np.max(np.abs(M))), 1.0)
            off = float(np.max(np.abs(M - np.diag(d))))
            if (off < 1e-12 * scale and abs(d[0] - d[1]) < 1e-12 * scale
                    and abs(d[0] - d[2]) < 1e-12 * scale):
                return _C(d[0])
            return None

        P = self.period
        degree = self.degree
        eps_sup = np.conj(_C(self.n_sup) ** 2)         # internal exp(+iwt) gauge
        eps_sub = np.conj(_C(self.n_sub) ** 2)
        # Order count: requested from far_field_orders, then CAPPED to the
        # coarsest PATTERNED layer's x-axis nodal capacity (a uniform layer is
        # exact in the Fourier basis -> no nodal need).  Mirrors the classical
        # path's n_glob cap + _validate_cell_orders (audit review): without it a
        # far_field_orders raised past the spectral-element grid drives a rank-
        # deficient projection that silently returns energy-violating numbers.
        n_orders = max(3, (self.ffo - 1) // 2)
        cap = None
        for (_thk, _segs, _sl) in self._layers:
            if len(_segs) > 1:                          # patterned -> nodal grid
                c = (len(_segs) * self.n_el * degree - 1) // 2
                cap = c if cap is None else min(cap, c)
        if cap is not None:
            n_orders = max(1, min(n_orders, cap))
        # The grid must at least resolve the propagating orders (classical
        # parity: degree too low for the physics is an error, not silent garbage).
        _idx = [float(np.real(np.sqrt(eps_sup))), float(np.real(np.sqrt(eps_sub)))]
        for _t, _segs, _s in self._layers:
            for _w, _e in _segs:
                _d = np.diag(np.asarray(_e, dtype=_C))
                _idx.append(float(np.real(np.sqrt(np.max(np.abs(_d))))))
        m_prop = _n_propagating_orders(P, wl, max(_idx))
        if 2 * m_prop + 1 > 2 * n_orders + 1:
            raise ValueError(
                "PMMStack.solve (conical, phi != 0): the spectral-element grid "
                f"resolves only {2 * n_orders + 1} orders but {2 * m_prop + 1} "
                "propagate; raise degree / elements_per_region (or lower "
                "far_field_orders).")
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(angle) * np.cos(phi)
        ky0 = nre * np.sin(angle) * np.sin(phi)
        # Suite-standard incidence guard (D1): the phi != 0 dispatch returns
        # before the classical solve body, so the guards there are bypassed --
        # reject a gain / evanescent superstrate here too, matching every other
        # PMM/RCWA entry.
        _require_propagating_incidence("PMMStack.solve (conical)", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)

        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.array([0])
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        # Grazing-safe wavelength (D1): nudge off any exact Rayleigh cutoff of
        # the half-spaces OR a layer constituent (diagonal permittivities) so a
        # grazing order can't crash the interface / generalized S-matrix; the
        # nudge is ~1e-7 relative, far below the resolvability check above.
        eps_reals = [eps_sup, eps_sub]
        for _t, _segs, _s in self._layers:
            for _w, _e in _segs:
                eps_reals.extend(complex(v) for v in
                                 np.diag(np.asarray(_e, dtype=_C)))
        wl = _grazing_safe_wavelength(float(wl), kx0, ky0, order_x, order_y,
                                      P, P, eps_reals)
        k0 = 2.0 * np.pi / wl
        kxv = kx0 + order_x * (wl / P)
        kyv = ky0 + order_y * (wl / P)                 # == ky0 (constant)
        ez_rule = "li" if self.factorization != "convection" else "laurent"

        Wsup, Vsup, _ls, _kzr = _homogeneous_modes(kxv, kyv, eps_sup)
        Wsub, Vsub, _lb, _kzt = _homogeneous_modes(kxv, kyv, eps_sub)

        # per-layer coupled modes (each layer's own grid; oy=[0] -> O(N)).  A
        # scalar (isotropic) layer takes the SCALAR projected path (the Li
        # inverse rule on the wall-normal component -> fast TM convergence,
        # matching pmm_jones_1d_conical); a genuine tensor layer the tensor path.
        modes = []
        for (thk, segs, _slant) in self._layers:
            widths = np.array([float(w) for w, _ in segs], dtype=float)
            scalars = [_scalar_eps(e) for _, e in segs]
            is_scalar = all(s is not None for s in scalars)
            if len(segs) == 1:                         # uniform layer
                x_walls = []
            else:
                cw = np.cumsum(widths)
                cw = cw / cw[-1]
                x_walls = [float(c * P) for c in cw[:-1]]
            if is_scalar and len(segs) == 1:           # uniform isotropic slab
                Wl, Vl, lam, _ = _homogeneous_modes(kxv, kyv, np.conj(scalars[0]))
                modes.append(("sym", Wl, Vl, lam, thk))
                continue
            ax = _build_axis(P, x_walls, degree, self.n_el, self.grade)
            ay = _build_axis(P, [], degree, self.n_el, self.grade)
            if is_scalar:                              # patterned isotropic
                eps_tile = np.conj(np.array([[s] for s in scalars]))  # (nseg, 1)
                lops = _scalar_projected_ops(ax, ay, eps_tile, ox, oy, P, P)
                GxF = lops["Gx0F"] / k0 + kx0 * lops["IpxF"]
                GyF = lops["Gy0F"] / k0 + ky0 * lops["IpyF"]
                Wl, Vl, lam = _layer_modes_projected(
                    GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
                    formulation=ez_rule, EpnxF=lops["EpnxF"],
                    EpnyF=lops["EpnyF"])
                modes.append(("sym", Wl, Vl, lam, thk))
                continue
            tile = np.stack([np.asarray(e, dtype=_C) for _, e in segs])
            tile_i = np.conj(tile)[:, None, :, :]      # (nseg, 1, 3, 3) internal
            out = _tensor_layer_modes(ax, ay, x_walls, [], tile_i, k0, kx0, ky0,
                                      ox, oy, kxv, kyv, ez_rule)
            entry = (("gen",) + out if len(out) == 6 else ("sym",) + tuple(out))
            modes.append(entry + (thk,))

        any_oop = any(m[0] == "gen" for m in modes)
        if not any_oop:
            nlay = len(modes)
            ifc = [_interface_smatrix(Wsup, Vsup, modes[0][1], modes[0][2])]
            for i in range(1, nlay):
                ifc.append(_interface_smatrix(modes[i - 1][1], modes[i - 1][2],
                                              modes[i][1], modes[i][2]))
            ifc.append(_interface_smatrix(modes[-1][1], modes[-1][2],
                                          Wsub, Vsub))
            S = ifc[0]
            for i, (_k, _W, _V, lam, t) in enumerate(modes):
                S = _redheffer_star(S, _propagation_smatrix(lam, k0 * t))
                S = _redheffer_star(S, ifc[i + 1])
        else:
            def _blocks(m):
                if m[0] == "sym":
                    _k, W, V, lam, t = m
                    return _modes_to_M(W, V, W, -V), lam, -lam, t
                _k, Wf, Vf, lf, Wb, Vb, lb, t = m
                return _modes_to_M(Wf, Vf, Wb, Vb), lf, lb, t

            M_prev = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
            S = None
            for m in modes:
                Ml, lf, lb, t = _blocks(m)
                Si = _interface_smatrix_general(M_prev, Ml)
                S = Si if S is None else _redheffer_star(S, Si)
                S = _redheffer_star(
                    S, _propagation_smatrix_general(lf, lb, k0 * t))
                M_prev = Ml
            Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
            S = _redheffer_star(S, _interface_smatrix_general(M_prev, Msub))
        S11, _S12, S21, _S22 = S

        _o2d, R_eff, T_eff, jones, modal = _conical_jones_farfield(
            S11, S21, order_x, order_y, kxv, kyv, kx0, ky0, eps_sup, eps_sub,
            return_modal=True)
        modal["wavelength"] = float(wl)          # the grazing-safe-nudged wl
        self._modal = modal                      # B: public-gauge amplitudes
        # Energy tripwire (audit review): the conical path must warn on R+T > 1
        # like every classical PMMStack return, so an under-resolved solve is not
        # silently trusted.  Orders are the 1-D (m,) array (the classical /
        # pmm_jones_1d_segments contract; y is degenerate so n_y == 0 for all).
        _warn_stack_energy(R_eff, T_eff)
        return order_x, R_eff, T_eff, jones

    def solve(self, *, stabilize=None, retain_internal=False):
        """Solve the stack.  Returns ``(orders, R_eff, T_eff, jones_reflection)``
        as :func:`pmm_jones_1d_segments` (``R_eff`` / ``T_eff`` are ``(2, M)``:
        row 0 = incident ``E_x``, row 1 = incident ``E_y``; ``jones`` is the
        zeroth-order ``2x2`` reflection).

        ``stabilize='slices'`` (device-geometry roadmap item 3b, 2026-06-10;
        opt-in, ~3x cost): the n_slices-consensus tripwire for PASSIVE-BUT-
        WRONG staircases.  An over-sliced taper whose walls collide can return
        an energy-clean yet completely wrong answer (the audited ns8/deg14
        row: out-coupling 27% instead of 78% with the R+T>1 tripwire silent);
        raising ``degree`` restores passivity, NOT accuracy.  The check
        re-solves every recorded taper builder at ``n_slices`` +/- 1 and
        WARNS when the zeroth-order Jones moves beyond the stack consensus
        tolerance -- energy cannot catch this class, slice-consensus can.

        DIFFERENTIABLE (JAX): passing any input as a jnp array -- a segment
        eps (scalar or in-plane ``(3,3)``), a layer ``thickness``, the
        ``wavelength``, the ``angle`` or a half-space index -- routes the
        whole solve to the jnp twin (``pmm/_jax_stack.py``): same physics,
        same order set, forward-identical to NumPy at ~1e-15, with gradients
        flowing to every traced input (AD-vs-FD validated ~1e-8).  The
        surface is ALL-VERTICAL IN-PLANE stacks with STATIC widths/walls;
        slant, out-of-plane tensors, ``stabilize``, ``retain_internal`` and
        the assemble-once sweep/prepare paths raise.  x64 required;
        ``jnp.linalg.eig`` is CPU-only."""
        # Invalidate retained internals BEFORE any dispatch/early return
        # (audit P1-04): every solve() supersedes the retained state, so
        # internal_field/layer_absorption can only serve the LAST solve --
        # a retain_internal=True one.  Stale fields from a previous
        # source/geometry must never be served silently.  Same contract for
        # the retained per-order amplitudes (per_order_amplitudes /
        # jones_transmission serve the LAST solve only).
        self._internal = None
        self._modal = None
        if self._src is None:
            raise ValueError("PMMStack.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("PMMStack.solve: add at least one layer.")
        # Validate ``stabilize`` EAGERLY, before ANY dispatch/early return
        # (audit P3-31): the covariant (uniform-slant) dispatch used to return
        # before the late vertical-path check, silently accepting garbage.
        if stabilize not in (None, False, "slices"):
            raise ValueError(
                f"PMMStack.solve: stabilize must be None or 'slices', got "
                f"{stabilize!r}.")
        if (any(callable(e) for L in self._layers for _w, e in L[1])
                or callable(self.n_sub) or callable(self.n_sup)):
            raise ValueError(
                "PMMStack.solve: the stack holds DISPERSIVE (wl -> value) "
                "materials; use solve_vs_wavelength(wavelengths), which "
                "materialises every callable per wavelength.")
        _unresolved = sorted({e for L in self._layers for _w, e in L[1]
                              if isinstance(e, str)})
        if _unresolved:
            raise ValueError(
                f"PMMStack.solve: unresolved material keys {_unresolved}; "
                f"pass materials={{...}} via prepare()/solve(materials=...) "
                f"or use concrete permittivities.")
        # ---- native conical (out-of-plane, phi != 0) dispatch --------------
        # Path B phase 4: the O(N) n_y = 0 reduction.  Restricted to all-vertical
        # NumPy scalar / in-plane stacks; the JAX / slant / stabilize /
        # retain_internal combinations raise (PMM2DStack with y-invariant cells
        # is the general route).  An out-of-plane tensor layer is permitted (it
        # promotes to the generalized cascade) but inherits the shared-generator
        # OOP-at-conical residual documented on pmm_jones_1d_conical_tensor.
        phi = float(self._src.get("phi", 0.0))
        if abs(phi) > 1e-12:
            if self._holds_traced():
                raise NotImplementedError(
                    "PMMStack.solve: conical incidence (phi != 0) is NumPy-only "
                    "(the JAX surface is the classical phi = 0 mount).")
            if stabilize not in (None, False):
                raise NotImplementedError(
                    "PMMStack.solve(stabilize=...): not available at conical "
                    "incidence (phi != 0).")
            if retain_internal:
                raise NotImplementedError(
                    "PMMStack.solve(retain_internal=True): not available at "
                    "conical incidence (phi != 0).")
            if any(abs(L[2]) > 1e-12 for L in self._layers):
                raise NotImplementedError(
                    "PMMStack: conical incidence (phi != 0) is not available for "
                    "SLANTED layers; use PMM2DStack with y-invariant cells.")
            return self._solve_conical(self._src["wl"], self._src["angle"], phi)

        # ---- differentiable (JAX) dispatch ---------------------------------
        # Any traced input (a layer eps / thickness, a half-space index, the
        # wavelength or the angle) routes the whole solve to the jnp twin.
        # The twin covers the ALL-VERTICAL IN-PLANE surface only -- the same
        # surface as the single-layer _pmm_jones_1d_jax -- so everything else
        # raises HERE (loudly, with the numpy alternative named).
        if self._holds_traced():
            if retain_internal:
                raise NotImplementedError(
                    "PMMStack.solve(retain_internal=True): not available on "
                    "the JAX (differentiable) path; use NumPy inputs for "
                    "internal fields / absorption.")
            if stabilize not in (None, False):
                raise NotImplementedError(
                    "PMMStack.solve(stabilize=...): the slices-consensus "
                    "check is host-side; use NumPy inputs.")
            if any(abs(L[2]) > 1e-12 for L in self._layers):
                raise NotImplementedError(
                    "PMMStack: slanted layers are not differentiable (the "
                    "covariant/metric generators are NumPy-only); use NumPy "
                    "inputs for slanted stacks.")
            for _t, _segs, _sl in self._layers:
                for _w, _M in _segs:
                    try:
                        _oop = self._is_oop(np.asarray(_M))
                    except Exception:
                        _oop = False    # traced tensor: in-plane contract
                    if _oop:
                        raise NotImplementedError(
                            "PMMStack: out-of-plane tensor layers are not "
                            "differentiable (the generalized fwd/back "
                            "cascade is NumPy-only); the JAX surface is "
                            "in-plane.  NB a TRACED (3,3) tensor cannot be "
                            "inspected -- its xz/yz/zx/zy entries are "
                            "ignored.")
            from ._jax_stack import _pmm_stack_solve_jax
            return _pmm_stack_solve_jax(self)

        wl, angle = self._src["wl"], self._src["angle"]
        k0 = 2.0 * np.pi / wl
        kx0 = float(np.real(self.n_sup)) * np.sin(angle) * k0
        eps_sup, eps_sub = self.n_sup ** 2, self.n_sub ** 2
        # Suite-standard incidence guard (audit M3 2026-07-25): the CLASSICAL
        # (phi = 0) cascade was the one PMM solve path without it -- a gain
        # superstrate (public Im(n_sup) < 0, even -1e-9j) flips _kz_forward to
        # its Re < 0 root, so kz_inc < 0 SILENTLY negated every efficiency
        # (measured tot = [-0.95, -0.82] with no warning), and a metallic /
        # evanescent incidence medium returned tot ~ 55.  The conical dispatch
        # already guards here (its own call below); this covers the covariant,
        # convection and all-vertical branches from one place.  ``eps_sup`` is
        # PUBLIC here -> conj to the internal loss gauge the guard expects.
        _require_propagating_incidence("PMMStack.solve", np.conj(_C(eps_sup)),
                                       (kx0 / k0) ** 2)

        # ---- factorization dispatch: covariant (SPECTRAL slant) vs convection --
        # 'auto' uses the covariant oblique-coordinate generator (spectral TM)
        # for an IN-PLANE uniform-slant stack, and the convection metric
        # generator otherwise (all-vertical, mixed-slant, or slanted
        # OUT-OF-PLANE -- see the routing note below).  'covariant' forces the
        # spectral path (raises on mixed/zero slant; carries out-of-plane with
        # the documented discontinuous-cell TM limitation); 'convection' forces
        # the algebraic-but-fully-general path.
        _oop = any(self._is_oop(M) for L in self._layers for _w, M in L[1])
        _slants = [abs(L[2]) for L in self._layers]
        _signed = [L[2] for L in self._layers]
        # The covariant oblique frame is per-slant (the shear u = x - tanφ z), so
        # the spectral covariant cascade requires a UNIFORM slant across all layers
        # (and the homogeneous half-spaces are solved in that same frame).  A
        # MIXED-slant stack (e.g. a vertical spacer + a slanted grating) would need
        # inter-layer lateral-shift corrections and falls back to convection.
        # Gate uniformity on the SIGNED slant (not abs): the shear u = x - tanφ z is
        # MIRROR-sheared for +φ vs -φ, so an equal-magnitude opposite-sign stack
        # would cascade -φ layer modes against half-spaces fixed in the +φ frame --
        # incompatible gauges, a silently-wrong S-matrix.  Opposite-sign / mixed
        # slants fall back to convection ('auto') or raise ('covariant').
        _uniform_slant = (max(_slants) >= _COV_MIN_SLANT_RAD
                          and (max(_signed) - min(_signed)) <= 1e-12)
        _fac = self.factorization
        if _fac == "auto":
            # OUT-OF-PLANE slanted stacks route to CONVECTION (2026-07-14):
            # the covariant layout's discontinuous off-plane TM channel has a
            # ~0.1 per-order factorization defect under the corrected
            # factor-i physics (the pre-fix covariant-for-OOP routing was
            # validated against engines sharing the same defect; convection
            # and the RCWA tensor staircase now agree at ~4e-3 while
            # covariant is the outlier).  Explicit 'covariant' still solves
            # OOP (documented limitation; AUDIT_OOP_GENERATOR_FACTOR_I).
            _fac = ("covariant" if (_uniform_slant and not _oop)
                    else "convection")
        if _fac == "covariant":
            if not _uniform_slant:
                raise NotImplementedError(
                    "PMMStack: factorization='covariant' requires a UNIFORM "
                    "non-zero slant across all layers (the covariant oblique "
                    "frame is per-slant); use 'convection' or 'auto' for "
                    "vertical / mixed-slant stacks.")
            # kwarg handling BEFORE the covariant early return (audit P3-31):
            # retain_internal raises (the covariant cascade keeps no partial
            # cascades -- same contract as the general slanted path below);
            # stabilize='slices' is HONOURED via the shared consensus check
            # (which itself warns when no taper builder is recorded).
            if retain_internal:
                raise NotImplementedError(
                    "PMMStack.solve(retain_internal=True): all-vertical "
                    "in-plane stacks only (the symmetric cascade).")
            res = self._solve_covariant(wl, angle, k0)
            if stabilize == "slices":
                self._slices_consensus_check(res[3])
            return res

        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers],
                                       self.min_feature / self.period)
        nU = len(uwidths)
        layer_mats = [
            _build_sem_tensor_segments(
                self.period, uwidths, [_tensor3_dict(e) for e in eps_u],
                self.degree, self.n_el, self.grade)
            for eps_u in layer_eps_u]
        t_sup = _tensor3_dict(eps_sup * np.eye(3))
        t_sub = _tensor3_dict(eps_sub * np.eye(3))
        mats_sup = _build_sem_tensor_segments(
            self.period, uwidths, [t_sup] * nU, self.degree, self.n_el, self.grade)
        mats_sub = _build_sem_tensor_segments(
            self.period, uwidths, [t_sub] * nU, self.degree, self.n_el, self.grade)
        n_glob = mats_sup["n_glob"]

        _geo = _uniform_geo_eig(mats_sup, k0, kx0)
        Wsup, Vsup, _l, _g = _sem_modes_uniform(mats_sup, k0, kx0, eps_sup,
                                                _geo)
        Wsub, Vsub, _l, _g = _sem_modes_uniform(mats_sub, k0, kx0, eps_sub,
                                                _geo)

        # Redheffer recursion: sup -> [interface, propagation]*layers -> sub.
        # A layer needs the GENERAL fwd/back cascade if it is SLANTED or carries
        # full-3x3 OUT-OF-PLANE coupling (both break the +/-q symmetry and are
        # solved by the metric generator); an all-vertical-in-plane stack keeps the
        # symmetric path (bit-identical to the prior release).
        oop_layer = [any(self._is_oop(M) for _w, M in L[1]) for L in self._layers]
        if not any(abs(L[2]) > 1e-12 or oo
                   for L, oo in zip(self._layers, oop_layer)):
            # ALL-VERTICAL IN-PLANE: the symmetric (+/-q) S-matrix path --
            # bit-identical to the prior release.  Dedup the per-layer modal eig
            # (the dominant cost) across IDENTICAL layers -- a periodic / Bragg
            # (ABAB...) stack recomputes the same eig P times; content-key it on
            # the layer's eps bytes (a 1:1 fingerprint of ``m`` since geometry /
            # degree / grade are shared).  Byte-identical (deterministic eig,
            # same arrays a plain loop would build; mirrors rcwa/stack.py).
            _eig_memo = {}
            lmodes = []
            for eps_u, m in zip(layer_eps_u, layer_mats):
                ck = tuple(np.asarray(e, dtype=_C).tobytes() for e in eps_u)
                hit = _eig_memo.get(ck)
                if hit is None:
                    hit = _sem_modes_tensor(m, k0, kx0, True)
                    _eig_memo[ck] = hit
                lmodes.append(hit)
            nlay = len(lmodes)
            ifc = [_interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])]
            for i in range(1, nlay):
                ifc.append(_interface_smatrix(
                    lmodes[i - 1][0], lmodes[i - 1][1],
                    lmodes[i][0], lmodes[i][1]))
            ifc.append(_interface_smatrix(lmodes[-1][0], lmodes[-1][1],
                                          Wsub, Vsub))
            S = ifc[0]
            for i in range(nlay):
                S = _propagation_star(S, lmodes[i][2], k0 * self._layers[i][0])
                S = _redheffer_star(S, ifc[i + 1])
            if retain_internal:
                # partial cascades for internal-amplitude recovery (device-
                # geometry roadmap item 6): S_above[i] = sup -> TOP of layer i
                # (through the interface INTO it, before its propagation);
                # S_below_bot[i] = BOTTOM of layer i -> sub (without its own
                # propagation), so backward amplitudes reference the layer
                # BOTTOM and both exponentials DECAY (no overflow through a
                # deep lossy layer) -- the RCWAStack _internal_partials
                # pattern.
                S_above = [None] * nlay
                S_above[0] = ifc[0]
                for i in range(1, nlay):
                    S_above[i] = _redheffer_star(
                        _redheffer_star(S_above[i - 1], _propagation_smatrix(
                            lmodes[i - 1][2],
                            k0 * self._layers[i - 1][0])), ifc[i])
                # LINEAR reverse recurrence (audit P2-12): the Redheffer star
                # is associative, so S_below_bot[i] = (ifc[i+1] * prop_{i+1})
                # * S_below_bot[i+1] -- O(nlay) stars instead of the former
                # per-layer full-chain rebuild (O(nlay^2)); the RCWAStack
                # _internal_partials pattern.  Same operands, different
                # association -- results agree to float round-off (~1e-16).
                S_below_bot = [None] * nlay
                S_below_bot[nlay - 1] = ifc[nlay]
                for i in range(nlay - 2, -1, -1):
                    S_below_bot[i] = _redheffer_star(
                        _redheffer_star(ifc[i + 1], _propagation_smatrix(
                            lmodes[i + 1][2],
                            k0 * self._layers[i + 1][0])),
                        S_below_bot[i + 1])
                # MEMORY: the recurrences are complete, so drop the (2N,2N)
                # blocks no reader touches -- _internal_amplitudes reads only
                # S_above[.][2:4] and S_below_bot[.][0].  8 -> 3 retained
                # blocks/layer (byte-identical; retained arrays unchanged).
                S_above = [(None, None, Sa[2], Sa[3]) for Sa in S_above]
                S_below_bot = [(Sbb[0], None, None, None)
                               for Sbb in S_below_bot]
                # per-union-cell material LABELS (application feedback
                # 2026-06-10): when the layers carry key names (a
                # SegmentStackGeometry export), map them onto the union grid
                # by the same midpoint rule the eps assignment uses, so
                # layer_absorption can attribute by KEY (twin keys with the
                # same eps split the loss map as they split the geometry).
                labels_u = None
                seg_labels = getattr(self, "_segment_labels", None)
                if seg_labels and len(seg_labels) >= len(self._layers):
                    uwalls = np.concatenate([[0.0], np.cumsum(uwidths)])
                    mids = 0.5 * (uwalls[:-1] + uwalls[1:])
                    labels_u = []
                    for li, (_t, segs, _sl) in enumerate(self._layers):
                        labs = seg_labels[li]
                        cw = np.concatenate(
                            [[0.0], np.cumsum([w for w, _ in segs])])
                        cw[-1] = 1.0
                        labels_u.append([
                            labs[min(max(int(np.searchsorted(
                                cw, m, side="right") - 1), 0),
                                len(labs) - 1)] for m in mids])
                self._internal = dict(
                    lmodes=lmodes, S_above=S_above, S_below_bot=S_below_bot,
                    layer_mats=layer_mats, layer_eps_u=layer_eps_u,
                    uwidths=uwidths, mats_sup=mats_sup, k0=k0, kx0=kx0,
                    n_glob=n_glob, labels_u=labels_u)
        else:
            if retain_internal:
                raise NotImplementedError(
                    "PMMStack.solve(retain_internal=True): all-vertical "
                    "in-plane stacks only (the symmetric cascade).")
            # MIXED vertical / SLANTED / OUT-OF-PLANE: every layer carries EXPLICIT
            # forward/backward modes cascaded with the GENERAL fwd/back S-matrix.  A
            # SLANTED or OUT-OF-PLANE layer uses the div-conforming metric generator
            # (_layer_modes_metric carries both the slant fold AND the out-of-plane
            # pointwise ezz-Schur); a plain vertical-in-plane layer reuses its
            # symmetric modes as the degenerate [[W,W],[V,-V]] forward/backward set.
            Ms = _half_M_sym_metric(Wsup, Vsup)
            Mb = _half_M_sym_metric(Wsub, Vsub)
            Mls, lamfs, lambs = [], [], []
            for i, (_thk, _segs, slant) in enumerate(self._layers):
                if abs(slant) > 1e-12 or oop_layer[i]:
                    mm = _build_nodal_metric_segments(
                        self.period, uwidths,
                        [_tensor3_dict(e) for e in layer_eps_u[i]],
                        self.degree, self.n_el, self.grade)
                    Wf, Vf, lamf, _qf, Wb, Vb, lamb, _qb = _layer_modes_metric(
                        mm, k0, slant, kx0)
                else:
                    Wl, Vl, lam_l, _q = _sem_modes_tensor(
                        layer_mats[i], k0, kx0, True)
                    Wf, Wb, Vf, Vb = Wl, Wl, Vl, -Vl
                    lamf, lamb = lam_l, -lam_l
                Mls.append(np.block([[Wf, Wb], [Vf, Vb]]))
                lamfs.append(lamf)
                lambs.append(lamb)
            S = _interface_smatrix_general(Ms, Mls[0])
            for i in range(len(self._layers)):
                S = _propagation_star_general(
                    S, lamfs[i], lambs[i], k0 * self._layers[i][0])
                nextM = (Mb if i == len(self._layers) - 1 else Mls[i + 1])
                S = _redheffer_star(S, _interface_smatrix_general(Mls[i], nextM))
        S11, _S12, S21, _S22 = S

        # far-field projection (mirrors _pmm_jones_solve_core)
        # max over BOTH in-plane diagonal components -- TM sees exx, TE sees eyy, so
        # exx alone under-resolves a high-eyy stack and can miss a propagating order
        # (audit P2; consistent with every single-layer solver).
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(np.sqrt(np.asarray(e, _C)[1, 1]))
                       for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(self.n_sup), np.real(self.n_sub)])
        m_prop = _n_propagating_orders(self.period, wl, n_max)
        n_proj = max(self.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        if 2 * m_prop + 1 > n_proj:               # parity with the single-layer cores
            raise ValueError(
                f"PMMStack.solve: degree={self.degree} too low to resolve the "
                f"{2 * m_prop + 1} propagating orders (n_glob={n_glob}); raise "
                f"degree or elements_per_region.")
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        G = 2.0 * np.pi / self.period
        kx = (kx0 + orders * G) / k0
        N = len(orders)
        Tp = _sem_fourier_projection(orders, self.period, mats_sup)

        def _proj(Wm):
            return np.vstack([Tp @ Wm[:n_glob, :], Tp @ Wm[n_glob:, :]])
        Hsup, Hsub = _proj(Wsup), _proj(Wsub)
        kz_sup = _kz_forward(eps_sup, kx)
        kz_sub = _kz_forward(eps_sub, kx)
        kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
        kx0n = kx0 / k0
        R_eff, T_eff, jones, modal = _assemble_jones_farfield(
            Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc, kx0n, N,
            return_modal=True)
        modal["wavelength"] = float(wl)
        self._modal = modal          # AUDIT_DYNAMETA_CONSUMER_API_GAPS B
        if retain_internal and getattr(self, "_internal", None) is not None:
            m0 = np.where(orders == 0)[0][0]
            cinc_cols = []
            for col in range(2):
                rhs = np.zeros(2 * N, dtype=_C)
                rhs[(col * N) + m0] = 1.0
                ci, *_ = np.linalg.lstsq(Hsup, rhs, rcond=None)
                cinc_cols.append(ci)
            self._internal["cinc"] = np.stack(cinc_cols, axis=1)
            self._internal["R_tot"] = R_eff.sum(axis=1)
            self._internal["T_tot"] = T_eff.sum(axis=1)
        _warn_stack_energy(R_eff, T_eff)
        if stabilize == "slices":
            self._slices_consensus_check(jones)
        elif stabilize not in (None, False):
            raise ValueError(
                f"PMMStack.solve: stabilize must be None or 'slices', got "
                f"{stabilize!r}.")
        return orders, R_eff, T_eff, jones

    def per_order_amplitudes(self, port="reflection"):
        """Per-order complex tangential field amplitudes (PUBLIC ``exp(-iwt)``
        convention) and transverse k-vectors of the LAST :meth:`solve` -- the
        exact :meth:`~lumenairy.elements.rcwa.RCWAResult.per_order_amplitudes`
        contract, mirrored for the PMM family
        (AUDIT_DYNAMETA_CONSUMER_API_GAPS B).

        Returns a dict with ``Ex`` / ``Ey`` each ``(2, N)`` (row 0 = response
        to incident lab ``E_x``, row 1 to incident ``E_y``), the per-order
        ``kx`` / ``ky`` / ``kz`` normalised by ``k0``, the 1-D ``orders``
        array, and the ``wavelength``.  ``port`` selects the reflection or
        transmission side.

        These are raw TANGENTIAL amplitudes: recovering an order's efficiency
        needs the flux weight ``Re(kz_m / kz_inc)``, the longitudinal
        ``|Ez|^2`` with ``Ez = -(kx Ex + ky Ey)/kz``, and the incident
        ``|E|^2`` -- the recipe documented on the RCWA twin.  With them, the
        response to any incident SUPERPOSITION (e.g. the rotated conical
        s-hat) is the same linear combination of the two rows PER ORDER --
        the cross terms that per-order POWERS cannot provide.

        Retained by NumPy solves that close through the shared Rayleigh far
        field: the classical mount (all-vertical in-plane, convection-slant
        and generalized out-of-plane cascades) and the native conical path
        (patterned nodal + uniform Fourier).  The covariant (uniform-slant)
        cascade and the JAX twin do not retain amplitudes."""
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"PMMStack.per_order_amplitudes: port must be 'reflection' "
                f"or 'transmission', got {port!r}.")
        m = self._modal
        if m is None:
            raise ValueError(
                "PMMStack.per_order_amplitudes: no per-order amplitudes "
                "retained -- run a NumPy solve() first (any add_layer / "
                "set_source / re-solve supersedes them; the covariant "
                "uniform-slant cascade and the JAX path do not retain "
                "amplitudes).")
        ex, ey = ("rx", "ry") if port == "reflection" else ("tx", "ty")
        kz = m["kz_ref"] if port == "reflection" else m["kz_trn"]
        return dict(orders=m["orders"].copy(), Ex=m[ex].copy(),
                    Ey=m[ey].copy(), kx=m["kx"].copy(), ky=m["ky"].copy(),
                    kz=np.asarray(kz).copy(), wavelength=m["wavelength"],
                    # incidence terms for the jones_field_from_orders bridge
                    # (the transmission-port kz is the substrate kz) -- S5-5.
                    kz_inc=m["kz_inc"], kx0=m["kx0"], ky0=m["ky0"])

    def jones_transmission(self):
        """The ``(2, 2)`` zeroth-order TRANSMISSION Jones of the last
        :meth:`solve` (columns = incident lab ``E_x`` / ``E_y``, rows =
        ``[E_x; E_y]``, PUBLIC ``exp(-iwt)`` convention) -- the phase-bearing
        modulator observable, previously unavailable from any PMM-family
        solve (AUDIT_DYNAMETA_CONSUMER_API_GAPS B minimal cut).  Same
        availability as :meth:`per_order_amplitudes`."""
        m = self._modal
        if m is None:
            raise ValueError(
                "PMMStack.jones_transmission: no per-order amplitudes "
                "retained -- run a NumPy solve() first (the covariant "
                "uniform-slant cascade and the JAX path do not retain "
                "amplitudes).")
        p0 = int(m["p0"])
        return np.stack([m["tx"][:, p0], m["ty"][:, p0]], axis=0)

    def _slices_consensus_check(self, jones, tol=0.02):
        """Re-solve the recorded taper builders at n_slices +/- 1 and warn if
        the zeroth-order Jones moves beyond ``tol`` -- the passive-but-wrong
        staircase detector (energy cannot see this failure class)."""
        import warnings as _warnings
        if not self._taper_recipes:
            _warnings.warn(
                "PMMStack.solve(stabilize='slices'): no taper builder "
                "recorded on this stack (hand-added layers cannot be "
                "re-sliced); the consensus check was skipped.", stacklevel=3)
            return
        deltas = []
        for dn in (-1, +1):
            clone = self._resliced_clone(dn)
            if clone is None or any(c + dn < 1 for _i, c, _m, _k in
                                    self._taper_recipes):
                continue
            clone._src = dict(self._src)
            try:
                _o, _R, _T, j2 = clone.solve()
            except Exception:               # a probe that itself blows up is
                deltas.append(np.inf)       # maximal disagreement
                continue
            deltas.append(float(np.max(np.abs(np.asarray(j2)
                                              - np.asarray(jones)))))
        if deltas and max(deltas) > tol:
            _warnings.warn(
                f"PMMStack.solve(stabilize='slices'): the zeroth-order Jones "
                f"moved {max(deltas):.3g} (> {tol}) when the taper staircase "
                f"was re-sliced at n_slices +/- 1 -- the PASSIVE-BUT-WRONG "
                f"staircase signature (colliding walls; energy stays clean "
                f"while the answer is wrong).  Reduce n_slices toward the "
                f"converged plateau, raise min_feature, or check the "
                f"staircase against a coarser-sliced reference.",
                stacklevel=3)

    @staticmethod
    def _seg_at(e, w):
        """Materialise one segment eps at wavelength ``w`` (callable -> value)."""
        return e(w) if callable(e) else e

    def _internal_amplitudes(self):
        """Per-layer modal amplitudes from the retained partial cascades:
        ``(c_fwd_top, c_bwd_bot)`` per layer, each ``(n_modes, 2)`` (one
        column per incident polarization).  Forward referenced at the layer
        TOP, backward at the layer BOTTOM (both propagators decay)."""
        if getattr(self, "_internal", None) is None or \
                "cinc" not in self._internal:
            raise ValueError(
                "PMMStack: no internal data retained; the MOST RECENT "
                "solve must use solve(retain_internal=True) (any re-solve "
                "invalidates previously retained internals).")
        d = self._internal
        out = []
        k0 = d["k0"]
        for i, (Wl, Vl, lam, _q) in enumerate(d["lmodes"]):
            Sa = d["S_above"][i]
            Sb = d["S_below_bot"][i]
            t = self._layers[i][0]
            X = np.exp(-lam * k0 * t)
            n = lam.shape[0]
            # c+ at layer TOP; the below-cascade sees the layer-bottom plane,
            # so its reflection acts on X c+ and returns the BOTTOM-referenced
            # backward amplitude; the above-cascade's S22 sees that backward
            # wave back at the TOP after another X decay.
            A22, A21 = Sa[3], Sa[2]
            B11 = Sb[0]
            M = np.eye(n, dtype=_C) - (A22 * X[None, :]) @ (B11 * X[None, :])
            c_fwd = np.linalg.solve(M, A21 @ d["cinc"])
            c_bwd = B11 @ (X[:, None] * c_fwd)
            out.append((c_fwd, c_bwd))
        return out

    def _flux_at(self, i, z_frac):
        """z-Poynting flux (modal quadratic form, both polarizations) at
        fractional depth ``z_frac`` inside layer ``i`` -- the same
        ``Im(Ex^T S0 conj(Hy) - Ey^T S0 conj(Hx))`` form the modal forward
        selector uses, evaluated for the TOTAL internal field."""
        d = self._internal
        amps = self._internal_amplitudes()
        Wl, Vl, lam, _q = d["lmodes"][i]
        c_fwd, c_bwd = amps[i]
        k0, t = d["k0"], self._layers[i][0]
        n_glob = d["n_glob"]
        S0 = d["mats_sup"]["S0"]
        P = np.exp(-lam * k0 * (z_frac * t))[:, None]
        Q = np.exp(-lam * k0 * ((1.0 - z_frac) * t))[:, None]
        E = Wl @ (P * c_fwd) + Wl @ (Q * c_bwd)
        H = Vl @ (P * c_fwd) - Vl @ (Q * c_bwd)
        Ex, Ey = E[:n_glob], E[n_glob:]
        Hx, Hy = H[:n_glob], H[n_glob:]
        return np.array([
            float(np.imag(Ex[:, c] @ (S0 @ np.conj(Hy[:, c])))
                  - np.imag(Ey[:, c] @ (S0 @ np.conj(Hx[:, c]))))
            for c in range(2)])

    def internal_field(self, z, *, component="all", incident=None, pol=None,
                       nx=None, layer=None):
        """Reconstruct the E and/or H field INSIDE the stack -- the PMM
        counterpart of :meth:`RCWAResult.internal_field`, on the NODAL grid
        (spectral-element exact in x: no Fourier/Gibbs reconstruction floor).

        Requires that the MOST RECENT ``solve`` used
        ``retain_internal=True`` (all-vertical in-plane stacks); any
        re-solve invalidates previously retained internals.

        Parameters
        ----------
        z : float or array-like
            Depth(s) [m] from the stack TOP (the superstrate interface).
            With ``layer=i``, ``z`` is the LOCAL depth inside layer ``i``.
        component : {'E', 'H', 'all'}, optional
            Which field(s) to return (default all six components).
        incident : (complex, complex), optional
            Incident Jones vector ``(E_x, E_y)``; default x-polarized.
            Mutually exclusive with ``pol``.
        pol : {0, 1}, optional
            Legacy selector (0 = incident ``E_x``, 1 = ``E_y``).
        nx : int, optional
            Resample onto a UNIFORM x grid of ``nx`` points (barycentric
            evaluation of the spectral interpolant).  Default: the exact
            GLL nodal grid (non-uniform; returned in ``'x'``).
        layer : int, optional
            Force a layer index (``z`` then local to that layer).

        Returns
        -------
        dict
            ``{'Ex','Ey','Ez','Hx','Hy','Hz'}`` (per ``component``) plus
            ``'x'``, ``'z'``, ``'layer'``.  Scalar ``z`` gives 1-D ``(n_x,)``
            arrays and an int layer; array ``z`` gives ``(nz, n_x)`` arrays
            and an ``(nz,)`` layer index array.  Fields are the FULL Bloch
            fields (envelope x carrier ``exp(i kx0 x)``) in the PUBLIC
            ``exp(-i w t)`` convention, ``H`` in the RCWA-co-registered
            ``-i eta0`` scale (CHANGED in v5.14.3 from the modal-convention
            envelope the day-one v5.14.2 method returned).  ``E_x`` is the
            wall-NORMAL component: at a segment wall the C0 nodal basis
            stores the (convergent) compromise value; ``Ez``/``Hz`` come
            from per-element spectral derivatives, averaged at shared nodes.
        """
        d = getattr(self, "_internal", None)
        if d is None or "cinc" not in d:
            raise ValueError(
                "PMMStack.internal_field: no internal data retained; the "
                "MOST RECENT solve must use solve(retain_internal=True) "
                "(any re-solve invalidates previously retained internals).")
        names = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz"),
                 "all": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        if component not in names:
            raise ValueError(
                f"PMMStack.internal_field: component must be 'E', 'H' or "
                f"'all', got {component!r}.")
        want = names[component]
        if incident is not None and pol is not None:
            raise ValueError(
                "PMMStack.internal_field: give incident= OR pol=, not both.")
        if incident is None:
            p = 0 if pol is None else pol
            if p not in (0, 1):
                raise ValueError(
                    "PMMStack.internal_field: pol must be 0 or 1.")
            ex0, ey0 = (1.0, 0.0) if p == 0 else (0.0, 1.0)
        else:
            ex0, ey0 = complex(incident[0]), complex(incident[1])

        amps = self._internal_amplitudes()
        k0, kx0, n_glob = d["k0"], d["kx0"], d["n_glob"]
        thick = [L[0] for L in self._layers]
        z_top = np.concatenate([[0.0], np.cumsum(thick)])
        zs = np.atleast_1d(np.asarray(z, dtype=float))
        scalar_in = np.ndim(z) == 0

        # ---- shared nodal geometry (one union grid for every layer) -------
        from ._core import _gll_nodes_weights, _lagrange_derivative_matrix
        mats0 = d["layer_mats"][0]
        deg = mats0["degree"]
        ref_nodes, _w = _gll_nodes_weights(deg)
        Dref = _lagrange_derivative_matrix(ref_nodes)
        eb0 = mats0["elem_bnds"]
        l2g = mats0["l2g"]
        x_glob = np.zeros(n_glob)
        for e, (xl, xr, _t) in enumerate(eb0):
            x_glob[l2g[e]] = 0.5 * (xl + xr) + 0.5 * (xr - xl) * ref_nodes
        order = np.argsort(x_glob)
        x_sorted = x_glob[order]

        def _ddx(env, ez_of_elem=None, mats_i=None):
            """Per-element spectral d/dx of a nodal envelope, averaged at
            shared (wall) nodes; optionally scaled per element by a tensor
            function (the 1/ezz weight for Ez)."""
            out = np.zeros(n_glob, dtype=_C)
            cnt = np.zeros(n_glob)
            for e, (xl, xr, tns) in enumerate((mats_i or mats0)["elem_bnds"]):
                J = 0.5 * (xr - xl)
                idx = l2g[e]
                dloc = (Dref @ env[idx]) / J
                if ez_of_elem is not None:
                    dloc = dloc * ez_of_elem(tns)
                out[idx] += dloc
                cnt[idx] += 1.0
            return out / cnt

        # uniform resampling operator (barycentric Lagrange, per element)
        if nx is not None:
            nxi = int(nx)
            xu = np.arange(nxi) * (self.period / nxi)
            wbary = np.ones(deg + 1)
            for jj in range(deg + 1):
                for kk in range(deg + 1):
                    if kk != jj:
                        wbary[jj] /= (ref_nodes[jj] - ref_nodes[kk])
            rows = []
            cols = []
            vals = []
            for e, (xl, xr, _t) in enumerate(eb0):
                inside = np.where((xu >= xl - 1e-15) & (xu <= xr + 1e-15))[0]
                if inside.size == 0:
                    continue
                xi = (2.0 * (xu[inside] - xl) / (xr - xl)) - 1.0
                for r, xv in zip(inside, xi):
                    dif = xv - ref_nodes
                    hit = np.argmin(np.abs(dif))
                    if abs(dif[hit]) < 1e-14:
                        rows.append(r)
                        cols.append(l2g[e][hit])
                        vals.append(1.0)
                    else:
                        num = wbary / dif
                        for a in range(deg + 1):
                            rows.append(r)
                            cols.append(l2g[e][a])
                            vals.append(num[a] / num.sum())
            R = np.zeros((nxi, n_glob))
            for r, c, v in zip(rows, cols, vals):
                R[r, c] += v
            # duplicate hits at element boundaries average out via overwrite
            # safety: renormalize rows that accumulated two boundary copies
            rs = R.sum(axis=1)
            R[rs > 1.5] *= 0.5
            x_out = xu
        else:
            R = None
            x_out = x_sorted

        n_x = len(x_out)
        out = {c: np.empty((zs.shape[0], n_x), dtype=_C) for c in want}
        layers_out = np.empty(zs.shape[0], dtype=int)
        carrier = (np.exp(1j * kx0 * x_out) if abs(kx0) > 0
                   else np.ones(n_x))

        cache = {}
        for zi, zz in enumerate(zs):
            if layer is None:
                i = int(np.clip(np.searchsorted(z_top, zz, side="right") - 1,
                                0, len(thick) - 1))
                zloc = zz - z_top[i]
            else:
                i, zloc = int(layer), float(zz)
            layers_out[zi] = i
            t_i = thick[i]
            Wl, Vl, lam, _q = d["lmodes"][i]
            if i not in cache:
                c_fwd, c_bwd = amps[i]
                cache[i] = (ex0 * c_fwd[:, 0] + ey0 * c_fwd[:, 1],
                            ex0 * c_bwd[:, 0] + ey0 * c_bwd[:, 1])
            cF, cB = cache[i]
            P = np.exp(-lam * k0 * zloc)
            Q = np.exp(-lam * k0 * max(t_i - zloc, 0.0))
            s = Wl @ (P * cF) + Wl @ (Q * cB)
            u = Vl @ (P * cF) - Vl @ (Q * cB)
            Ex_e, Ey_e = s[:n_glob], s[n_glob:]
            # +i (not RCWA's -i): the PMM modal V partner is built with the
            # OPPOSITE sign convention (Q = [[...],[-Cxx,...]]); pinned by
            # the RCWA co-registration oracle on a uniform film (Hy == +Ex
            # for the forward vacuum wave) -- see test_internal_field
            Hx_e, Hy_e = 1j * u[:n_glob], 1j * u[n_glob:]
            h = dict(Ex=Ex_e, Ey=Ey_e, Hx=Hx_e, Hy=Hy_e)
            if "Ez" in want:
                mats_i = d["layer_mats"][i]
                dHy = _ddx(Hy_e, mats_i=mats_i) + 1j * kx0 * Hy_e
                ez_n = np.zeros(n_glob, dtype=_C)
                cnt = np.zeros(n_glob)
                for e, (xl, xr, tns) in enumerate(mats_i["elem_bnds"]):
                    ez_n[l2g[e]] += tns["ezz"]
                    cnt[l2g[e]] += 1.0
                ez_n /= cnt
                h["Ez"] = (1j / k0) * dHy / ez_n
            if "Hz" in want:
                dEy = _ddx(Ey_e) + 1j * kx0 * Ey_e
                h["Hz"] = -(1j / k0) * dEy
            for c in want:
                env = h[c]
                row = (R @ env if R is not None else env[order])
                out[c][zi] = row * carrier
        result = {}
        for c in want:
            result[c] = out[c][0] if scalar_in else out[c]
        result["x"] = x_out
        result["z"] = float(zs[0]) if scalar_in else zs
        result["layer"] = int(layers_out[0]) if scalar_in else layers_out
        return result

    def layer_absorption(self, *, by_material=False):
        """Per-layer absorbed power fraction (device-geometry roadmap item 6,
        2026-06-10) -- the PMM counterpart of
        :meth:`RCWAResult.layer_absorption`.

        Requires that the MOST RECENT ``solve`` used
        ``retain_internal=True`` (all-vertical in-plane stacks); any
        re-solve invalidates previously retained internals.  Returns an
        ``(n_layers, 2)`` array (one column per
        incident polarization, the :meth:`solve` row order) computed from the
        INTERNAL z-Poynting flux difference across each layer, normalized so
        the totals close against the far field: the test invariant is
        ``sum_i A_i ~= 1 - sum(R) - sum(T)`` -- an honest cross-check, since
        the fluxes come from the internal modal amplitudes and R/T from the
        Rayleigh far field.

        ``by_material=True`` additionally returns a dict mapping each
        distinct material -- by KEY NAME (``"Cu"``, ``"Ta"``) when the stack
        was built via ``SegmentStackGeometry.to_pmm_stack`` (so twin keys
        with the SAME eps, the under-tooth-liner trick, split the loss map
        as they split the geometry; application feedback 2026-06-10), else
        by distinct permittivity (complex key, e.g. the Ta vs the Cu value) to
        its ``(2,)`` absorbed fraction, attributed by the per-element lossy
        volume integral ``Im(E* . eps . E)`` within each layer (GLL
        quadrature in x, closed-form in z) over ALL THREE diagonal channels
        ``Im(exx)|Ex|^2 + Im(eyy)|Ey|^2 + Im(ezz)|Ez|^2`` (audit P2-13:
        ``Ez`` is reconstructed per element as in :meth:`internal_field`, so
        an ``ezz``-only-lossy segment is attributed, not dropped); the
        per-material SPLIT is renormalized to the exact per-layer flux
        total.
        """
        d = getattr(self, "_internal", None)
        if d is None or "cinc" not in d:
            raise ValueError(
                "PMMStack.layer_absorption: no internal data retained; the "
                "MOST RECENT solve must use solve(retain_internal=True) "
                "(any re-solve invalidates previously retained internals).")
        nlay = len(self._layers)
        F_top = np.array([self._flux_at(i, 0.0) for i in range(nlay)])
        F_bot = np.array([self._flux_at(i, 1.0) for i in range(nlay)])
        # normalize: the top-of-stack internal flux equals (1 - sum R) of the
        # incident flux, so F_inc = F_top[0] / (1 - sum R) per polarization.
        one_minus_R = 1.0 - d["R_tot"]
        F_inc = np.where(np.abs(one_minus_R) > 1e-15,
                         F_top[0] / one_minus_R, np.inf)
        A = (F_top - F_bot) / F_inc[None, :]
        if not by_material:
            return A
        # ---- per-material attribution within each layer --------------------
        # Split each layer's exact (flux-based) absorption across its
        # distinct permittivities by the lossy volume density
        # Im(exx)|Ex|^2 + Im(eyy)|Ey|^2 + Im(ezz)|Ez|^2 (audit P2-13: the
        # ezz channel used to be omitted, so a segment whose ONLY loss is
        # Im(ezz) -- a uniaxial absorber with the lossy axis along z -- was
        # dropped from the dict entirely), GLL-quadratured in x and Gauss-
        # sampled in z, then renormalized to the flux total (the split is a
        # RATIO, so the common normalization cancels).  Ez is reconstructed
        # per element from Ampere's law, Ez = (i/k0) (dHy/dx + i kx0 Hy)/ezz
        # -- the internal_field construction, with the ELEMENT's own ezz.
        amps = self._internal_amplitudes()
        k0, kx0 = d["k0"], d["kx0"]
        nz = 24
        zg, zw = np.polynomial.legendre.leggauss(nz)
        zg = 0.5 * (zg + 1.0)                  # fractional depth in (0, 1)
        zw = 0.5 * zw
        out = {}
        for i in range(nlay):
            Wl, Vl, lam, _q = d["lmodes"][i]
            c_fwd, c_bwd = amps[i]
            t = self._layers[i][0]
            mats_i = d["layer_mats"][i]
            n_glob = d["n_glob"]
            elem_bnds = mats_i["elem_bnds"]
            l2g = mats_i["l2g"]
            from ._core import _gll_nodes_weights, _lagrange_derivative_matrix
            _rn, ref_w = _gll_nodes_weights(mats_i["degree"])
            Dref = _lagrange_derivative_matrix(_rn)
            keyed = {}
            ez_terms = []   # (key, idx, 1/J, wel, ezz, Im(ezz)) per element
            labels_u = d.get("labels_u")
            ucum = np.concatenate([[0.0], np.cumsum(d["uwidths"])]) \
                * self.period
            for e, (xl, xr, tns) in enumerate(elem_bnds):
                if labels_u is not None:
                    # attribute by material KEY: locate the element's union
                    # cell by midpoint (robust to element ordering/grading)
                    mid = 0.5 * (xl + xr)
                    cell = min(max(int(np.searchsorted(
                        ucum, mid, side="right") - 1), 0),
                        len(d["uwidths"]) - 1)
                    key = labels_u[i][cell]
                else:
                    key = complex(tns["exx"])
                im_ezz = float(np.imag(tns["ezz"]))
                if abs(np.imag(tns["exx"])) < 1e-15 and                         abs(np.imag(tns["eyy"])) < 1e-15 and abs(im_ezz) < 1e-15:
                    continue                   # lossless segment
                J = 0.5 * (xr - xl)
                wel = ref_w * J
                if key not in keyed:
                    keyed[key] = (np.zeros(n_glob), np.zeros(n_glob))
                wx, wy = keyed[key]
                np.add.at(wx, l2g[e], wel * float(np.imag(tns["exx"])))
                np.add.at(wy, l2g[e], wel * float(np.imag(tns["eyy"])))
                if abs(im_ezz) >= 1e-15:
                    ez_terms.append((key, l2g[e], 1.0 / J, wel,
                                     complex(tns["ezz"]), im_ezz))
            if not keyed:
                continue                       # nothing lossy in this layer
            raw = {k: np.zeros(2) for k in keyed}
            for zf, w_z in zip(zg, zw):
                P = np.exp(-lam * k0 * (zf * t))[:, None]
                Q = np.exp(-lam * k0 * ((1.0 - zf) * t))[:, None]
                E = Wl @ (P * c_fwd) + Wl @ (Q * c_bwd)
                Ex, Ey = E[:n_glob], E[n_glob:]
                for key, (wx, wy) in keyed.items():
                    raw[key] += w_z * (
                        (wx[:, None] * np.abs(Ex) ** 2).sum(axis=0)
                        + (wy[:, None] * np.abs(Ey) ** 2).sum(axis=0))
                if ez_terms:
                    # Ez channel (audit P2-13): Hy from the modal V partner
                    # (the internal_field +i convention; a global phase, so
                    # |Ez| is unaffected), then the per-element spectral
                    # derivative -- Ez jumps with ezz at material walls, so
                    # NO wall averaging (Ez is wall-tangential here, but the
                    # per-element form is the discontinuity-safe one).
                    U = Vl @ (P * c_fwd) - Vl @ (Q * c_bwd)
                    Hy = 1j * U[n_glob:]
                    for key, idx, invJ, wel, ezz, im_ezz in ez_terms:
                        dHy = (Dref @ Hy[idx]) * invJ + 1j * kx0 * Hy[idx]
                        Ez_e = (1j / k0) * dHy / ezz
                        raw[key] += (w_z * im_ezz) * (
                            wel[:, None] * np.abs(Ez_e) ** 2).sum(axis=0)
            tot = np.zeros(2)
            for v in raw.values():
                tot = tot + v
            safe = np.where(tot > 0, tot, 1.0)
            for key, v in raw.items():
                share = np.where(tot > 0, v / safe, 0.0)
                out.setdefault(key, np.zeros(2))
                out[key] = out[key] + share * A[i]
        return A, out

    def solve_vs_wavelength(self, wavelengths, *, angle=0.0, theta=None,
                            jones=False, max_workers=None, blas_per_worker=1):
        """Diffraction efficiencies across a wavelength sweep on ONE call,
        reusing the geometry-only spectral-element assembly.

        The shared union grid and the per-layer SEM operators
        (``_build_sem_tensor_segments``) + the Fourier far-field projector are
        wavelength-INDEPENDENT and are assembled ONCE; only the per-wavelength
        modal eigs + S-matrix cascade rerun.  The per-layer generalized eig
        dominates the cost, and the per-wavelength solves are INDEPENDENT and
        release the GIL inside LAPACK, so they run on a bounded THREAD POOL
        (``max_workers=None`` picks ``min(cpu, n_wl)``; each worker pins its BLAS
        pool to ``blas_per_worker`` so the product stays within the core count).
        Results are stored BY INDEX -> BYTE-IDENTICAL to a serial sweep regardless
        of worker count (``max_workers=1`` forces serial).  Within each wavelength
        the per-layer eig is DEDUPED across identical layers (an ABAB Bragg stack
        eigs each distinct layer once, mirroring :meth:`solve`).  Bit-identical to
        per-wavelength :meth:`solve` on the propagating orders.

        ALL-VERTICAL IN-PLANE stacks only (the symmetric ``+/-q`` cascade -- which
        is what the tapered/vertical builders produce); for SLANTED or
        OUT-OF-PLANE stacks call :meth:`solve` per wavelength.  NON-DISPERSIVE
        indices assumed across the sweep.  A FIXED diffraction-order set (covering
        the shortest wavelength's propagating orders) is used so the result is a
        dense array.

        Parameters
        ----------
        wavelengths : array-like of float
            Vacuum wavelengths [m].
        angle / theta : float, optional
            Incidence angle [rad] (``theta`` is the cross-suite alias, and wins
            when both are given) -- FIXED across the sweep.
        max_workers : int, optional
            Thread-pool size for the per-wavelength solves (default
            ``min(cpu, n_wl)``; ``1`` = serial).  Output is byte-identical to
            serial for any value.
        blas_per_worker : int, optional
            BLAS threads each worker pins (default 1, so ``max_workers *
            blas_per_worker`` stays within the core count).

        DISPERSIVE materials (device-geometry roadmap item 5, 2026-06-10):
        any segment eps / region index added as a ``wl -> value`` callable is
        materialised per wavelength here (the union-grid WIDTHS are geometry
        and stay hoisted; only the dispersive layers' operators reassemble).

        Returns
        -------
        orders : (N,) int ndarray
            The (wavelength-independent) retained diffraction orders.
        R, T : (n_wavelengths, 2, N) float ndarray
            Reflected / transmitted efficiency, ``[wavelength][incident pol][order]``
            (pol 0 = incident ``E_x``, pol 1 = incident ``E_y``).
        jones : (n_wavelengths, 2, 2) complex ndarray
            Zeroth-order Jones reflection per wavelength -- returned as a
            FOURTH element only when ``jones=True`` (default ``False`` keeps
            the released 3-tuple).
        """
        self._internal = None   # supersedes any retained internals (audit P1-04)
        self._modal = None      # ... and retained per-order amplitudes (B)
        if self._holds_traced():
            raise NotImplementedError(
                "PMMStack.solve_vs_wavelength: the assemble-once sweep "
                "is NumPy-only; for traced (JAX) inputs call solve() "
                "per wavelength (it dispatches to the differentiable "
                "twin) or vmap over wavelength.")
        angle = _resolve_incidence_checked(
            "PMMStack.solve_vs_wavelength", angle, theta)   # reject back-side (S1-7)
        if not self._layers:
            raise ValueError("PMMStack.solve_vs_wavelength: add at least one "
                             "layer.")
        wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        if wl.size == 0:
            raise ValueError("PMMStack.solve_vs_wavelength: wavelengths is empty.")
        if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
            raise ValueError("PMMStack.solve_vs_wavelength: every wavelength "
                             "must be a finite value > 0 [m].")
        # dispersive detection: callables anywhere -> per-wavelength
        # materialisation of the affected layers / regions
        disp_layer = [any(callable(e) for _w, e in L[1]) for L in self._layers]
        disp_region = callable(self.n_sup) or callable(self.n_sub)
        if any(isinstance(e, str) for L in self._layers for _w, e in L[1]):
            raise ValueError(
                "PMMStack.solve_vs_wavelength: unresolved material keys; "
                "resolve them via prepare()/solve(materials=...) first.")

        def _tens(e, w):
            return self._as_tensor(self._seg_at(e, float(w)))

        oop = any(self._is_oop(_tens(M, wl[0]))
                  for L in self._layers for _w, M in L[1])
        if oop or any(abs(L[2]) > 1e-12 for L in self._layers):
            raise NotImplementedError(
                "PMMStack.solve_vs_wavelength: all-vertical in-plane stacks only "
                "(the symmetric cascade); call solve() per wavelength for slanted "
                "/ out-of-plane stacks.")

        # ---- GEOMETRY-ONLY assembly (reused across the whole sweep) ----
        # The union WIDTHS depend only on segment widths (never on eps), so the
        # grid is hoisted even for a dispersive stack; only dispersive layers'
        # SEM operators reassemble per wavelength below.
        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers],
                                       self.min_feature / self.period)
        nU = len(uwidths)
        layer_mats = [
            (None if disp_layer[i] else _build_sem_tensor_segments(
                self.period, uwidths,
                [_tensor3_dict(self._as_tensor(e)) for e in eps_u],
                self.degree, self.n_el, self.grade))
            for i, eps_u in enumerate(layer_eps_u)]

        def _region_mats(w):
            nsup = self._seg_at(self.n_sup, w)
            nsub = self._seg_at(self.n_sub, w)
            t_sup = _tensor3_dict(_C(nsup) ** 2 * np.eye(3))
            t_sub = _tensor3_dict(_C(nsub) ** 2 * np.eye(3))
            m_sup = _build_sem_tensor_segments(
                self.period, uwidths, [t_sup] * nU, self.degree, self.n_el,
                self.grade)
            m_sub = _build_sem_tensor_segments(
                self.period, uwidths, [t_sub] * nU, self.degree, self.n_el,
                self.grade)
            return _C(nsup), _C(nsub), m_sup, m_sub

        n_sup0, n_sub0, mats_sup, mats_sub = _region_mats(float(wl[0]))
        n_glob = mats_sup["n_glob"]

        # FIXED order set: cover the SHORTEST wavelength's propagating orders,
        # scanning DISPERSIVE eps at every sweep point.
        _nmx = [np.real(n_sup0), np.real(n_sub0)]
        for w in ([float(wl[0])] if not (any(disp_layer) or disp_region)
                  else [float(x) for x in wl]):
            for eps_u in layer_eps_u:
                for e in eps_u:
                    M3 = _tens(e, w)
                    _nmx.append(np.real(np.sqrt(M3[0, 0])))
                    _nmx.append(np.real(np.sqrt(M3[1, 1])))
            if disp_region:
                _nmx.append(np.real(_C(self._seg_at(self.n_sup, w))))
                _nmx.append(np.real(_C(self._seg_at(self.n_sub, w))))
        n_max = max(_nmx)
        m_prop = _n_propagating_orders(self.period, float(np.min(wl)), n_max)
        n_proj = max(self.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        if 2 * m_prop + 1 > n_proj:
            raise ValueError(
                f"PMMStack.solve_vs_wavelength: degree={self.degree} too low to "
                f"resolve the {2 * m_prop + 1} propagating orders at the shortest "
                f"wavelength (n_glob={n_glob}); raise degree / elements_per_region.")
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        N = len(orders)
        G = 2.0 * np.pi / self.period
        Tp = _sem_fourier_projection(orders, self.period, mats_sup)

        def _proj(Wm):
            return np.vstack([Tp @ Wm[:n_glob, :], Tp @ Wm[n_glob:, :]])

        R_all = np.empty((wl.size, 2, N), dtype=float)
        T_all = np.empty((wl.size, 2, N), dtype=float)
        J_all = np.empty((wl.size, 2, 2), dtype=complex)
        depths = [float(self._layers[i][0]) for i in range(len(layer_mats))]

        def _solve_one(iw, w):
            # Independent per-wavelength solve (writes R_all[iw]... by index in
            # the main thread -> byte-identical to serial).  DISPERSIVE
            # half-spaces materialise their OWN mats locally (no shared-variable
            # race); the union-grid projector Tp is geometry-only and stays
            # hoisted.  The wl-independent geometric-eig cache is lock-guarded.
            w = float(w)
            k0 = 2.0 * np.pi / w
            if disp_region:
                n_sup_w, n_sub_w, msup, msub = _region_mats(w)
            else:
                n_sup_w, n_sub_w, msup, msub = n_sup0, n_sub0, mats_sup, mats_sub
            eps_sup, eps_sub = n_sup_w ** 2, n_sub_w ** 2
            kx0 = float(np.real(n_sup_w)) * np.sin(angle) * k0
            # Same guard as PMMStack.solve (audit M3): a DISPERSIVE half-space
            # can turn gain / non-propagating at one wavelength only, so it is
            # checked per point (eps_sup is PUBLIC here -> conj).
            _require_propagating_incidence(
                "PMMStack.solve_vs_wavelength", np.conj(_C(eps_sup)),
                (kx0 / k0) ** 2)
            with _blas_threads_quiet(blas_per_worker):
                _geo = _uniform_geo_eig(msup, k0, kx0)
                Wsup, Vsup, _l, _g = _sem_modes_uniform(msup, k0, kx0,
                                                        eps_sup, _geo)
                Wsub, Vsub, _l2, _g2 = _sem_modes_uniform(msub, k0, kx0,
                                                          eps_sub, _geo)
                # DBR / identical-layer dedup: eig each DISTINCT layer once per
                # wavelength (an ABAB Bragg stack recomputes the same eig P times
                # otherwise -- solve() already dedups, the sweep did not).  Key
                # on the materialised per-region eps bytes (mirrors solve():937).
                _eig_memo = {}
                lmodes = []
                for i, m in enumerate(layer_mats):
                    ck = tuple(np.asarray(_tens(e, w), dtype=_C).tobytes()
                               for e in layer_eps_u[i])
                    hit = _eig_memo.get(ck)
                    if hit is None:
                        mm = (m if m is not None else _build_sem_tensor_segments(
                            self.period, uwidths,
                            [_tensor3_dict(_tens(e, w)) for e in layer_eps_u[i]],
                            self.degree, self.n_el, self.grade))
                        hit = _sem_modes_tensor(mm, k0, kx0, True)
                        _eig_memo[ck] = hit
                    lmodes.append(hit)
                S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
                for i, (Wl_, Vl_, lam_l, _q) in enumerate(lmodes):
                    S = _propagation_star(S, lam_l, k0 * depths[i])
                    nW, nV = ((Wsub, Vsub) if i == len(lmodes) - 1
                              else (lmodes[i + 1][0], lmodes[i + 1][1]))
                    S = _redheffer_star(S, _interface_smatrix(Wl_, Vl_, nW, nV))
            S11, _S12, S21, _S22 = S
            kx = (kx0 + orders * G) / k0
            Hsup, Hsub = _proj(Wsup), _proj(Wsub)
            kz_sup = _kz_forward(eps_sup, kx)
            kz_sub = _kz_forward(eps_sub, kx)
            kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
            R, T, jr = _assemble_jones_farfield(
                Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc,
                kx0 / k0, N)
            return iw, R, T, np.asarray(jr)

        def _store(iw, R, T, jr):
            _warn_stack_energy(R, T)
            R_all[iw], T_all[iw], J_all[iw] = R, T, jr

        _mw = (min(os.cpu_count() or 1, int(wl.size)) if max_workers is None
               else max(1, int(max_workers)))
        if _mw == 1 or wl.size == 1:
            for iw, w in enumerate(wl):
                _store(*_solve_one(iw, w))
        else:
            from concurrent.futures import ThreadPoolExecutor
            # ThreadPoolExecutor.map yields IN INPUT ORDER, so _store runs in
            # wavelength order (deterministic warnings) and the result is
            # byte-identical to the serial path regardless of worker count.
            with ThreadPoolExecutor(max_workers=_mw) as _ex:
                for res in _ex.map(lambda p: _solve_one(*p),
                                   list(enumerate(wl))):
                    _store(*res)
        if jones:
            return orders, R_all, T_all, J_all
        return orders, R_all, T_all

    def prepare(self):
        """Hoist the stack for MATERIAL sweeps at fixed geometry (device-
        geometry roadmap item 8, 2026-06-10): segments may carry material KEY
        strings (``(width, "LC")``); the returned prepared object re-eigs
        ONLY the layers whose keys are overridden, caching every other
        layer's eig per ``(wavelength, angle)``.

        >>> st.add_layer(t, segments=[(0.5, "LC"), (0.5, 1.0)])
        >>> prep = st.prepare()
        >>> for tensor in director_sweep:
        ...     o, R, T, J = prep.solve(wavelength=wl,
        ...                             materials={"LC": tensor})

        Honest-perf note: the saving is the key-free layers' eigs + all
        assembly (for an LC device ~40% of a solve), not an order of
        magnitude.  All-vertical in-plane stacks (the symmetric cascade).
        """
        if self._holds_traced():
            raise NotImplementedError(
                "PMMStack.prepare: the prepared (assemble-once) path "
                "is NumPy-only; traced (JAX) inputs dispatch through "
                "solve().")
        return _PreparedPMMStack(self)

    def _solve_covariant(self, wl, angle, k0):
        """SPECTRAL multi-layer solve via the Li covariant oblique-coordinate
        generator (in-plane OR out-of-plane).  Parallels the general fwd/back
        cascade of
        :meth:`solve` but uses :func:`_cov_layer_4n` modes + COVARIANT
        homogeneous half-spaces on the shared union grid, so slanted layers
        converge SPECTRALLY (vertical-grade) instead of the convection
        generator's algebraic ~1e-4 floor.  Internally exp(+iwt): eps are
        conjugated in and the Jones conjugated out; the union widths/tensors are
        PRE-REVERSED to cancel the _segment_elem_bnds [::-1] (so orders land in
        the user's input frame, matching the convection path)."""
        period, degree, n_el, grade = self.period, self.degree, self.n_el, \
            self.grade
        kx0 = float(np.real(np.conj(self.n_sup))) * np.sin(angle) * k0
        eps_sup = np.conj(self.n_sup ** 2)
        eps_sub = np.conj(self.n_sub ** 2)
        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers],
                                       self.min_feature / self.period)
        uw = list(uwidths)[::-1]
        nU = len(uwidths)

        def grid(eps_list):
            seg = [_t3_slant(np.conj(np.asarray(e, _C))) for e in eps_list][::-1]
            return _build_nodal_metric_segments(period, uw, seg, degree, n_el,
                                                grade)
        # the homogeneous half-spaces are solved in the SAME oblique frame as the
        # layers (uniform slant), so their modes share the layer convention; pass
        # PUBLIC eps (grid conjugates to the internal exp(+iwt) convention).
        slant = self._layers[0][2]
        layer_mats = [grid(layer_eps_u[i]) for i in range(len(self._layers))]
        mats_s = grid([self.n_sup ** 2 * np.eye(3) for _ in range(nU)])
        mats_b = grid([self.n_sub ** 2 * np.eye(3) for _ in range(nU)])
        n_glob = mats_s["n_glob"]

        # DIV-CONFORMING Ez closure if ANY layer is out-of-plane (machine-precision
        # OOP), applied uniformly to all layers + both half-spaces so the closure
        # matches across every interface; in-plane stacks keep the modal closure.
        divconf = any(self._is_oop(M) for _L in self._layers for _w, M in _L[1])
        Ws, Vs, kzs, fws = _cov_layer_4n(mats_s, k0, slant, kx0, divconf)
        Wb, Vb, kzb, fwb = _cov_layer_4n(mats_b, k0, slant, kx0, divconf)
        fs, _bs = _cov_split(Ws, Vs, kzs, fws)
        fb, _bb = _cov_split(Wb, Vb, kzb, fwb)

        def _msym(W, V, f):
            return np.block([[W[:, f], W[:, f]], [V[:, f], -V[:, f]]])
        Mls, lamf_l, lamb_l = [], [], []
        for i, (_thk, _segs, slant) in enumerate(self._layers):
            Wl, Vl, kzl, fwl = _cov_layer_4n(layer_mats[i], k0, slant, kx0,
                                             divconf)
            fl, bl = _cov_split(Wl, Vl, kzl, fwl)
            Mls.append(np.block([[Wl[:, fl], Wl[:, bl]],
                                 [Vl[:, fl], Vl[:, bl]]]))
            lamf_l.append(-1j * kzl[fl])
            lamb_l.append(-1j * kzl[bl])
        S = _interface_smatrix_general(_msym(Ws, Vs, fs), Mls[0])
        for i in range(len(self._layers)):
            S = _propagation_star_general(
                S, lamf_l[i], lamb_l[i], k0 * self._layers[i][0])
            nextM = (_msym(Wb, Vb, fb) if i == len(self._layers) - 1
                     else Mls[i + 1])
            S = _redheffer_star(S, _interface_smatrix_general(Mls[i], nextM))
        S11, _S12, S21, _S22 = S

        # max over BOTH in-plane diagonal components -- TM sees exx, TE sees eyy, so
        # exx alone under-resolves a high-eyy stack and can miss a propagating order
        # (audit P2; consistent with every single-layer solver).
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(np.sqrt(np.asarray(e, _C)[1, 1]))
                       for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(self.n_sup), np.real(self.n_sub)])
        m_prop = _n_propagating_orders(period, wl, n_max)
        n_proj = max(self.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        if 2 * m_prop + 1 > n_proj:               # parity with the single-layer cores
            raise ValueError(
                f"PMMStack.solve (covariant): degree={self.degree} too low to "
                f"resolve the {2 * m_prop + 1} propagating orders "
                f"(n_glob={n_glob}); raise degree or elements_per_region.")
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        N = len(orders)
        kx = kx0 / k0 + orders * (2.0 * np.pi / period) / k0
        Tp = _sem_fourier_projection(orders, period, mats_s)
        # W7 F-C (2026-07-26), the Berreman-F-1 twin.  ``_kz_forward`` is a
        # PUBLIC-gauge helper (``Im(kz) >= 0`` for ``exp(-iwt)``), but
        # ``eps_sup``/``eps_sub`` are INTERNAL exp(+iwt) here (conjugated at the
        # top of this method) -- so a LOSSY half-space arrived double-
        # conjugated, ``sqrt`` landed in the 4th quadrant, the ``Im < 0`` flip
        # sent ``Re(kz) < 0``, and the ``Re(kz) > 0`` propagating mask inside
        # _assemble_jones_farfield SILENTLY ZEROED T.  Measured pre-fix on a
        # HOMOGENEOUS eps=2.25 slab (where the slant is a physical no-op, so
        # the vertical cascade is the exact oracle), P=0.30 um, 0.22 um deep,
        # wl 0.55 um, slant 0.35 rad, theta=0.3:
        #     n_sub 1.5+0.01j  ->  T = [0, 0]   (oracle [0.96586, 0.95613])
        #     n_sub 1.5+0.30j  ->  T = [0, 0]   (oracle [0.98578, 0.98047])
        #     n_sub 0.2+3.5j   ->  T = [0, 0]   (oracle [0.08641, 0.08135])
        # with ZERO warnings -- ``_warn_stack_energy`` only sees super-unity /
        # negative totals, and 0.014 is "passive".  An absorbing SUPERSTRATE
        # was worse: ``kz_inc = -1.14651`` tripped the "non-propagating
        # incidence" raise on a perfectly propagating medium.  Un-conjugating
        # restores the public gauge (identity for a real eps -> every lossless
        # solve is BYTE-UNCHANGED); this is exactly the ``kz_ord`` bridge in
        # ``_core._pmm_jones_oblique_core`` and rcwa's ``_forward_flux_kz``.
        # (The MODAL kz inside the cascade keeps the internal convention -- that
        # path is already correct.)
        kz_sup = _kz_forward(np.conj(eps_sup), kx)
        kz_sub = _kz_forward(np.conj(eps_sub), kx)
        kz_inc = float(np.real(
            _kz_forward(np.conj(eps_sup), np.array([kx0 / k0]))[0]))
        Hsup = np.vstack([Tp @ Ws[:n_glob, fs], Tp @ Ws[n_glob:, fs]])
        Hsub = np.vstack([Tp @ Wb[:n_glob, fb], Tp @ Wb[n_glob:, fb]])
        R, T, jones = _assemble_jones_farfield(
            Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc,
            kx0 / k0, N)
        _warn_stack_energy(R, T)
        return orders, R, T, np.conj(jones)        # conj: bridge +iwt -> public


__all__ = [
    "PMMStack",
]



class _PreparedPMMStack:
    """Material-slot-swappable prepared PMM stack (see :meth:`PMMStack.prepare`).

    Caches per-layer modal eigs keyed by ``(layer, wavelength, angle,
    resolved-eps-content)`` so a material sweep (e.g. an LC director tuning
    curve) re-eigs only the layers whose material keys changed; key-free
    layers and the half-spaces eig once per ``(wavelength, angle)``.

    Both caches are per-INSTANCE LRU-bounded ``OrderedDict``\\ s (audit
    P3-32): unbounded dicts retained one modal set per historical
    (wavelength, angle, material-content) point forever -- a 1000-tensor LC
    director sweep on a long-lived prepared object accumulated GBs the
    documented re-eig contract never needs.  Sizing: a keyed ``_eig_cache``
    entry is ~3 dense ``(2*n_glob)^2`` complex arrays (~11.5 MB at
    production ``n_glob~300``), and one solve holds at most ``n_layers``
    entries live at once, so 64 slots (worst case ~0.7 GB) guarantee a
    single solve never self-evicts while typical few-state material scans
    still hit; a ``_mats_cache`` entry (SEM operators + two half-space mode
    sets per ``(wavelength, angle)``) is used once per solve, so 8 slots
    cover alternating wavelength/angle revisits.  Eviction only drops the
    dict's reference -- returned mode arrays held by callers are never
    mutated -- and a post-eviction recompute is byte-identical (same LAPACK
    eig on the same input bytes).  ``clear_cache`` drains both on demand.

    These instance caches are NOT enrolled in ``lumenairy._cache_registry``
    (deliberate): the registry's contract (v4.16.0) covers PROCESS-GLOBAL
    module caches, and the v4.16.1 enrollment meta-pin walker accordingly
    discovers only module-level ``_CACHE`` assignments -- an instance
    attribute's lifetime is its owning object's, so dropping the prepared
    object (or calling :meth:`clear_cache`) is the release path."""

    _EIG_CACHE_SIZE = 64
    _MATS_CACHE_SIZE = 8

    def __init__(self, stack):
        if any(abs(L[2]) > 1e-12 for L in stack._layers):
            raise NotImplementedError(
                "PMMStack.prepare: all-vertical stacks only.")
        if any(callable(e) for L in stack._layers for _w, e in L[1]):
            raise NotImplementedError(
                "PMMStack.prepare: dispersive (wl -> value) materials are the "
                "solve_vs_wavelength path; prepare() swaps material KEYS.")
        # OOP guard (audit S1-1): the prepared eig is the in-plane 2n solver
        # (_build_sem_tensor_segments reads only exx/exy/eyx/eyy/ezz), so it
        # silently drops eps_xz/yz/zx/zy.  Reject CONCRETE out-of-plane tensor
        # layers here (mirrors the solve_vs_wavelength guard); a material KEY
        # that resolves to an OOP tensor is unknown until solve() and is
        # caught there.
        if any(not isinstance(e, str) and stack._is_oop(e)
               for L in stack._layers for _w, e in L[1]):
            raise NotImplementedError(
                "PMMStack.prepare: out-of-plane tensor layers "
                "(eps_xz/yz/zx/zy) are not supported on the prepared path "
                "(the assemble-once eig is the in-plane 2n solver); call "
                "PMMStack.solve() per point.")
        self._st = stack
        self._uwidths, self._layer_eps_u = _pmm_union_grid(
            [L[1] for L in stack._layers], stack.min_feature / stack.period)
        self._layer_keys = [sorted({e for e in eps_u if isinstance(e, str)})
                            for eps_u in self._layer_eps_u]
        self._eig_cache = OrderedDict()
        self._mats_cache = OrderedDict()
        # Guards get / move_to_end / setitem / popitem on both caches
        # (mirrors the _WRAPPER_MERIT_CACHE_LOCK precedent); the eig /
        # assembly builds themselves run outside the lock.
        self._cache_lock = threading.Lock()

    def clear_cache(self):
        """Drop every cached modal eig + per-``(wavelength, angle)`` operator
        set (audit P3-32).  The next :meth:`solve` recomputes them
        byte-identically; call between sweep stages to release memory
        without discarding the prepared stack."""
        with self._cache_lock:
            self._eig_cache.clear()
            self._mats_cache.clear()

    def _resolve(self, eps_u, materials, wavelength):
        out = []
        for e in eps_u:
            if isinstance(e, str):
                if e not in materials:
                    raise ValueError(
                        f"PMMStack.prepare/solve: material key {e!r} has no "
                        f"entry in materials={{...}}.")
                v = materials[e]
                v = v(wavelength) if callable(v) else v
                out.append(self._st._as_tensor(v))
            else:
                out.append(e)
        return out

    def _layer_eig(self, i, materials, k0, kx0, wavelength, angle):
        st = self._st
        keys = self._layer_keys[i]
        if keys:
            resolved = self._resolve(self._layer_eps_u[i], materials,
                                     wavelength)
            content = tuple(np.asarray(t, dtype=_C).tobytes()
                            for t in resolved)
            ck = (i, wavelength, angle, content)
        else:
            resolved = None
            ck = (i, wavelength, angle)
        with self._cache_lock:
            hit = self._eig_cache.get(ck)
            if hit is not None:
                self._eig_cache.move_to_end(ck)   # LRU: refresh recency
        if hit is not None:
            return hit
        if resolved is None:
            resolved = [st._as_tensor(e) for e in self._layer_eps_u[i]]
        mats = _build_sem_tensor_segments(
            st.period, self._uwidths, [_tensor3_dict(e) for e in resolved],
            st.degree, st.n_el, st.grade)
        modes = _sem_modes_tensor(mats, k0, kx0, True)
        with self._cache_lock:
            self._eig_cache[ck] = _freeze_cached(modes)   # W7 A13
            while len(self._eig_cache) > self._EIG_CACHE_SIZE:
                self._eig_cache.popitem(last=False)
        return modes

    def solve(self, *, wavelength, materials=None, angle=0.0, theta=None):
        """Solve at ``wavelength`` with the material keys resolved from
        ``materials`` (values: scalar / ``(3, 3)`` / ``wl -> value``
        callables).  Returns ``(orders, R, T, jones)`` exactly as
        :meth:`PMMStack.solve`; equal to the rebuild path to ~1e-12."""
        st = self._st
        st._internal = None  # prepared solve supersedes retained internals (audit P1-04)
        materials = dict(materials or {})
        angle = _resolve_incidence_checked(
            "PMMStack.solve", angle, theta)                 # reject back-side (S1-7)
        wl = float(wavelength)
        # OOP guard (audit S1-1): a material key may resolve to an out-of-plane
        # tensor, which the in-plane prepared eig would silently drop (concrete
        # OOP layers were already rejected in __init__).  Only KEYED layers can
        # newly resolve to OOP, so check just those.
        for i, keys in enumerate(self._layer_keys):
            if keys and any(st._is_oop(st._as_tensor(e)) for e in
                            self._resolve(self._layer_eps_u[i], materials, wl)):
                raise NotImplementedError(
                    "PMMStack.prepare/solve: a material key resolved to an "
                    "out-of-plane tensor (eps_xz/yz/zx/zy); the prepared path "
                    "is in-plane only -- call PMMStack.solve() per material "
                    "point.")
        k0 = 2.0 * np.pi / wl
        kx0 = float(np.real(st.n_sup)) * np.sin(angle) * k0
        eps_sup, eps_sub = st.n_sup ** 2, st.n_sub ** 2
        # Same guard as PMMStack.solve (audit M3 -- the prepared path is a
        # third classical cascade entry; eps_sup is PUBLIC here -> conj).
        _require_propagating_incidence("PMMStack.prepare().solve",
                                       np.conj(_C(eps_sup)), (kx0 / k0) ** 2)
        nU = len(self._uwidths)
        rk = (wl, angle)
        with self._cache_lock:
            entry = self._mats_cache.get(rk)
            if entry is not None:
                self._mats_cache.move_to_end(rk)   # LRU: refresh recency
        if entry is None:
            t_sup = _tensor3_dict(eps_sup * np.eye(3))
            t_sub = _tensor3_dict(eps_sub * np.eye(3))
            m_sup = _build_sem_tensor_segments(
                st.period, self._uwidths, [t_sup] * nU, st.degree, st.n_el,
                st.grade)
            m_sub = _build_sem_tensor_segments(
                st.period, self._uwidths, [t_sub] * nU, st.degree, st.n_el,
                st.grade)
            _geo = _uniform_geo_eig(m_sup, k0, kx0)
            entry = (
                m_sup, m_sub,
                _sem_modes_uniform(m_sup, k0, kx0, eps_sup, _geo),
                _sem_modes_uniform(m_sub, k0, kx0, eps_sub, _geo))
            with self._cache_lock:
                self._mats_cache[rk] = _freeze_cached(entry)  # W7 A13
                while len(self._mats_cache) > self._MATS_CACHE_SIZE:
                    self._mats_cache.popitem(last=False)
        mats_sup, _mats_sub, sup_modes, sub_modes = entry
        Wsup, Vsup = sup_modes[0], sup_modes[1]
        Wsub, Vsub = sub_modes[0], sub_modes[1]
        n_glob = mats_sup["n_glob"]

        lmodes = [self._layer_eig(i, materials, k0, kx0, wl, angle)
                  for i in range(len(st._layers))]
        # out-of-plane resolved tensors are not supported on this path
        S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
        for i, (Wl_, Vl_, lam_l, _q) in enumerate(lmodes):
            S = _propagation_star(S, lam_l, k0 * st._layers[i][0])
            nW, nV = ((Wsub, Vsub) if i == len(lmodes) - 1
                      else (lmodes[i + 1][0], lmodes[i + 1][1]))
            S = _redheffer_star(S, _interface_smatrix(Wl_, Vl_, nW, nV))
        S11, _S12, S21, _S22 = S

        # far field (mirrors PMMStack.solve)
        resolved_all = [self._resolve(eps_u, materials, wl)
                        if ks else [st._as_tensor(e) for e in eps_u]
                        for eps_u, ks in zip(self._layer_eps_u,
                                             self._layer_keys)]
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in resolved_all for e in eps_u]
                    + [np.real(np.sqrt(np.asarray(e, _C)[1, 1]))
                       for eps_u in resolved_all for e in eps_u]
                    + [np.real(st.n_sup), np.real(st.n_sub)])
        m_prop = _n_propagating_orders(st.period, wl, n_max)
        n_proj = max(st.ffo, 2 * m_prop + 5)
        cap = n_glob if n_glob % 2 else n_glob - 1
        n_proj = min(n_proj, cap)
        if n_proj % 2 == 0:
            n_proj -= 1
        half = (n_proj - 1) // 2
        orders = np.arange(-half, half + 1)
        G = 2.0 * np.pi / st.period
        kx = (kx0 + orders * G) / k0
        N = len(orders)
        Tp = _sem_fourier_projection(orders, st.period, mats_sup)

        def _proj(Wm):
            return np.vstack([Tp @ Wm[:n_glob, :], Tp @ Wm[n_glob:, :]])
        Hsup, Hsub = _proj(Wsup), _proj(Wsub)
        kz_sup = _kz_forward(eps_sup, kx)
        kz_sub = _kz_forward(eps_sub, kx)
        kz_inc = float(np.real(_kz_forward(eps_sup,
                                           np.array([kx0 / k0]))[0]))
        R_eff, T_eff, jones = _assemble_jones_farfield(
            Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc,
            kx0 / k0, N)
        _warn_stack_energy(R_eff, T_eff)
        return orders, R_eff, T_eff, jones
