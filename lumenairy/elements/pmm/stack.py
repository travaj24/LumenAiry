"""PMM unified multi-layer API: PMMStack (mixes vertical + slanted
layers via the div-conforming covariant-metric generator)."""
from __future__ import annotations

import numpy as np

# Backend detection for the JAX (differentiable) dispatch in pmm_efficiency_1d.
# Mirrors rcwa's pattern: a JAX input routes to the self-contained jnp twin,
# while a NumPy input falls through to the original (byte-identical) code.
# Reused for the slanted-grating solver: the slant breaks the +/-q field
# symmetry (like a full-3x3 tensor layer), so it needs the GENERALIZED
# (explicit forward/backward) S-matrix.  rcwa does NOT import pmm, so this
# top-level import introduces no cycle.
from ..rcwa import _interface_smatrix_general, _propagation_smatrix_general
from ._core import (
    _C,
    _COV_MIN_SLANT_RAD,
    _assemble_jones_farfield,
    _build_nodal_metric_segments,
    _build_sem_tensor_segments,
    _cov_layer_4n,
    _cov_split,
    _half_M_sym_metric,
    _interface_smatrix,
    _kz_forward,
    _layer_modes_metric,
    _n_propagating_orders,
    _pmm_union_grid,
    _propagation_smatrix,
    _redheffer_star,
    _resolve_incidence,
    _resolve_order_count,
    _sem_fourier_projection,
    _sem_modes_tensor,
    _t3_slant,
    _tensor3_dict,
)


def _warn_stack_energy(R_eff, T_eff):
    """Warn if a solve returns non-physical gain (``R+T > 1`` per incident
    polarization) -- the signature of a near-singular interface mode-match in the
    Redheffer cascade (the measure-zero quasi-resonance RCWA guards with
    ``_check_energy``; the many-interface tapered z-staircase can hit it at large
    ``n_slices``).  A passive structure cannot reflect+transmit more than the
    incident power regardless of loss, so this is a pure instability tripwire
    (lossy media give ``R+T < 1`` and never trip).  A WARNING (not a raise) so it
    never breaks an existing working solve; reduce ``n_slices`` / raise ``degree``
    to clear it."""
    R = np.asarray(R_eff)
    T = np.asarray(T_eff)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    worst = float(np.max(tot)) if tot.size else 0.0
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
        (default) uses the SPECTRAL covariant oblique-coordinate generator for a
        stack whose layers share a single non-zero slant -- IN-PLANE OR
        OUT-OF-PLANE (the full-3x3 coupling enters via the Li Eq.12 ezz-Schur
        composites + cos*Dop cross blocks) -- and the (algebraic-but-fully-
        general) convection generator otherwise -- vertical or MIXED-slant stacks
        (the covariant oblique frame is per-slant, so a mixed-slant cascade falls
        back to convection).  ``'covariant'`` forces the spectral path (raises on
        mixed/zero slant); ``'convection'`` forces the general path.

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
        self.n_sub = n_substrate if callable(n_substrate) else _C(n_substrate)
        self.n_sup = (n_superstrate if callable(n_superstrate)
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

    def _as_tensor(self, eps):
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
        each ``eps`` scalar or ``(3,3)``; widths sum to 1).  ``slant_angle``
        (radians) tilts the layer's straight side-walls from the vertical (``0``
        = vertical); a slanted layer is solved by the div-conforming covariant-
        metric generator and cascaded via the general fwd/back S-matrix, so a
        stack may MIX vertical and slanted layers.  Returns ``self``."""
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
        self._layers.append((float(thickness), segs, float(slant_angle)))
        return self

    def add_tapered_grating(self, thickness, *, eps_ridge, eps_groove,
                            duty_bottom, duty_top=None, n_slices=8,
                            rule="midpoint"):
        """Append a 1-D grating with SLANTED / TRAPEZOIDAL sidewalls as an
        auto-sliced z-staircase of thin VERTICAL PMM layers -- the spectral-
        element counterpart of :meth:`RCWAStack.add_tapered_grating`.

        The centred ridge's duty cycle varies linearly with depth from
        ``duty_top`` (top, ``zeta = 0``) to ``duty_bottom`` (bottom, ``zeta = 1``);
        each slice is a vertical binary grating whose walls are resolved EXACTLY
        by the nodal grid (no Fourier/Gibbs floor in x -- the PMM advantage), so
        the only approximation is the z-staircase of the taper (a true trapezoid
        is the ``n_slices -> infinity`` limit).  ``duty_top == duty_bottom`` gives
        the usual vertical binary grating.

        COST NOTE: every slice's two walls enter the stack's SHARED union grid, so
        the global node count -- and thus each layer's eig -- grows with
        ``n_slices``.  This is laterally exact and beats an RCWA z-staircase per
        slice (no Fourier floor), but is practical for MODEST ``n_slices``; for a
        scalable no-floor taper prefer a single covariant taper-metric layer (a
        roadmap item).  The ``z``-staircase converges as ``O(1/n_slices^2)`` with
        the default centre (``'midpoint'``) rule.  A wavelength sweep should use
        :meth:`solve_vs_wavelength`, which assembles the (large) shared grid ONCE.

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
            Sample each slice's duty at its centre (``'midpoint'``, default,
            ``O(1/n^2)``) or average its two edges (``'trapezoid'``).
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
        self._taper_recipes.append((len(self._layers), n,
                                    "add_tapered_grating",
                                    dict(thickness=thickness,
                                         eps_ridge=eps_ridge,
                                         eps_groove=eps_groove,
                                         duty_bottom=duty_bottom,
                                         duty_top=duty_top, rule=rule)))
        dz = float(thickness) / n
        for k in range(n):
            if rule == "midpoint":
                duty = dt + (db - dt) * ((k + 0.5) / n)
            else:
                duty = dt + (db - dt) * 0.5 * (k / n + (k + 1) / n)
            tol = 1e-9
            if duty <= tol:                       # ridge vanished -> all groove
                self.add_layer(dz, eps=eps_groove)
            elif duty >= 1.0 - tol:               # groove vanished -> all ridge
                self.add_layer(dz, eps=eps_ridge)
            else:                                 # centred ridge between grooves
                edge = 0.5 * (1.0 - duty)
                self.add_layer(dz, segments=[(edge, eps_groove),
                                             (duty, eps_ridge),
                                             (edge, eps_groove)])
        return self

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
                           n_slices=8, rule="midpoint"):
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
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_tapered_ridges: n_slices must be >= 1, got {n_slices}.")
        if rule not in ("midpoint", "trapezoid"):
            raise ValueError(
                f"add_tapered_ridges: rule must be 'midpoint' or 'trapezoid', "
                f"got {rule!r}.")
        rid = [(float(c), float(wt), float(wb), e)
               for c, wt, wb, e in ridges]
        self._taper_recipes.append((len(self._layers), n,
                                    "add_tapered_ridges",
                                    dict(thickness=thickness, ridges=ridges,
                                         eps_groove=eps_groove, rule=rule)))
        dz = float(thickness) / n
        for k in range(n):
            if rule == "midpoint":
                zeta = (k + 0.5) / n
            else:
                zeta = 0.5 * (k / n + (k + 1) / n)
            slice_ridges = [(c, wt + (wb - wt) * zeta, e)
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

    def set_source(self, wavelength, *, angle=0.0, theta=None):
        """Set the incident plane wave (vacuum wavelength [m], incidence
        ``angle`` [rad] in the x-z plane).  ``theta`` is accepted as a cross-suite
        alias for ``angle`` (matching ``RCWAStack.set_source``'s polar angle, with
        the 1-D classical mount's azimuth ``phi = 0``).  ``theta`` WINS when both
        are supplied -- the SAME rule as ``RCWAStack.set_source`` and the 1-D entry
        points, so ``set_source(angle=A, theta=T)`` resolves to ``T`` in every
        suite.  Returns ``self``."""
        angle = _resolve_incidence(angle, theta)
        self._src = dict(wl=float(wavelength), angle=float(angle))
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
        tolerance -- energy cannot catch this class, slice-consensus can."""
        if self._src is None:
            raise ValueError("PMMStack.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("PMMStack.solve: add at least one layer.")
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
        wl, angle = self._src["wl"], self._src["angle"]
        k0 = 2.0 * np.pi / wl
        kx0 = float(np.real(self.n_sup)) * np.sin(angle) * k0
        eps_sup, eps_sub = self.n_sup ** 2, self.n_sub ** 2

        # ---- factorization dispatch: covariant (SPECTRAL slant) vs convection --
        # 'auto' uses the covariant oblique-coordinate generator (spectral TM) for
        # ANY slanted stack -- in-plane OR out-of-plane (the full-3x3 coupling
        # enters via the Li Eq.12 ezz-Schur composites + cos*Dop cross blocks) --
        # and the convection metric generator otherwise (all-vertical).  The
        # covariant cascade still requires a UNIFORM slant.  'covariant' forces it
        # (raises on
        # out-of-plane); 'convection' forces the algebraic-but-fully-general path.
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
            _fac = "covariant" if _uniform_slant else "convection"
        if _fac == "covariant":
            if not _uniform_slant:
                raise NotImplementedError(
                    "PMMStack: factorization='covariant' requires a UNIFORM "
                    "non-zero slant across all layers (the covariant oblique "
                    "frame is per-slant); use 'convection' or 'auto' for "
                    "vertical / mixed-slant stacks.")
            return self._solve_covariant(wl, angle, k0)

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

        Wsup, Vsup, _l, _g = _sem_modes_tensor(mats_sup, k0, kx0, True)
        Wsub, Vsub, _l, _g = _sem_modes_tensor(mats_sub, k0, kx0, True)

        # Redheffer recursion: sup -> [interface, propagation]*layers -> sub.
        # A layer needs the GENERAL fwd/back cascade if it is SLANTED or carries
        # full-3x3 OUT-OF-PLANE coupling (both break the +/-q symmetry and are
        # solved by the metric generator); an all-vertical-in-plane stack keeps the
        # symmetric path (bit-identical to the prior release).
        oop_layer = [any(self._is_oop(M) for _w, M in L[1]) for L in self._layers]
        if not any(abs(L[2]) > 1e-12 or oo
                   for L, oo in zip(self._layers, oop_layer)):
            # ALL-VERTICAL IN-PLANE: the symmetric (+/-q) S-matrix path --
            # bit-identical to the prior release.
            lmodes = [_sem_modes_tensor(m, k0, kx0, True) for m in layer_mats]
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
                S = _redheffer_star(S, _propagation_smatrix(
                    lmodes[i][2], k0 * self._layers[i][0]))
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
                S_below_bot = [None] * nlay
                for i in range(nlay - 1, -1, -1):
                    below = ifc[i + 1]
                    for j in range(i + 1, nlay):
                        below = _redheffer_star(below, _propagation_smatrix(
                            lmodes[j][2], k0 * self._layers[j][0]))
                        below = _redheffer_star(below, ifc[j + 1])
                    S_below_bot[i] = below
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
                S = _redheffer_star(S, _propagation_smatrix_general(
                    lamfs[i], lambs[i], k0 * self._layers[i][0]))
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
        R_eff, T_eff, jones = _assemble_jones_farfield(
            Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc, kx0n, N)
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
                "PMMStack: no internal data retained; call "
                "solve(retain_internal=True) first.")
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

    def layer_absorption(self, *, by_material=False):
        """Per-layer absorbed power fraction (device-geometry roadmap item 6,
        2026-06-10) -- the PMM counterpart of
        :meth:`RCWAResult.layer_absorption`.

        Requires a prior ``solve(retain_internal=True)`` (all-vertical
        in-plane stacks).  Returns an ``(n_layers, 2)`` array (one column per
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
        quadrature in x, closed-form in z; the in-plane field components --
        the wall-normal ``E_z`` channel of an isotropic lossy segment is
        carried by the in-plane form's flux normalization within each layer,
        so the per-material SPLIT is renormalized to the exact per-layer
        flux total).
        """
        d = getattr(self, "_internal", None)
        if d is None or "cinc" not in d:
            raise ValueError(
                "PMMStack.layer_absorption: call solve(retain_internal=True) "
                "first.")
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
        # Im(exx)|Ex|^2 + Im(eyy)|Ey|^2, GLL-quadratured in x and Gauss-
        # sampled in z, then renormalized to the flux total (the split is a
        # RATIO, so the common normalization cancels).
        amps = self._internal_amplitudes()
        k0 = d["k0"]
        nz = 24
        zg, zw = np.polynomial.legendre.leggauss(nz)
        zg = 0.5 * (zg + 1.0)                  # fractional depth in (0, 1)
        zw = 0.5 * zw
        out = {}
        for i in range(nlay):
            Wl, _Vl, lam, _q = d["lmodes"][i]
            c_fwd, c_bwd = amps[i]
            t = self._layers[i][0]
            mats_i = d["layer_mats"][i]
            n_glob = d["n_glob"]
            elem_bnds = mats_i["elem_bnds"]
            l2g = mats_i["l2g"]
            from ._core import _gll_nodes_weights
            _rn, ref_w = _gll_nodes_weights(mats_i["degree"])
            keyed = {}
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
                if abs(np.imag(tns["exx"])) < 1e-15 and                         abs(np.imag(tns["eyy"])) < 1e-15:
                    continue                   # lossless segment
                J = 0.5 * (xr - xl)
                wel = ref_w * J
                if key not in keyed:
                    keyed[key] = (np.zeros(n_glob), np.zeros(n_glob))
                wx, wy = keyed[key]
                np.add.at(wx, l2g[e], wel * float(np.imag(tns["exx"])))
                np.add.at(wy, l2g[e], wel * float(np.imag(tns["eyy"])))
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
                            jones=False):
        """Diffraction efficiencies across a wavelength sweep on ONE call,
        reusing the geometry-only spectral-element assembly.

        The shared union grid and the per-layer SEM operators
        (``_build_sem_tensor_segments``) + the Fourier far-field projector are
        wavelength-INDEPENDENT and are assembled ONCE; only the per-wavelength
        modal eigs + S-matrix cascade rerun.  HONEST PERF NOTE: the per-layer
        generalized eig dominates the cost (the SEM assembly is a tiny fraction of
        it), so this is essentially a CONVENIENCE + correctness wrapper returning a
        dense ``(n_wl, 2, N)`` array, NOT a speedup -- it runs at roughly the same
        wall-clock as a per-wavelength :meth:`solve` loop (the cost is eig-bound,
        like the 1-D solvers).  Bit-identical to per-wavelength :meth:`solve` on
        the propagating orders.

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
        angle = _resolve_incidence(angle, theta)
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
        for iw, w in enumerate(wl):
            w = float(w)
            k0 = 2.0 * np.pi / w
            if disp_region and iw > 0:
                n_sup_w, n_sub_w, mats_sup, mats_sub = _region_mats(w)
            elif iw == 0 or not disp_region:
                n_sup_w, n_sub_w = n_sup0, n_sub0
            eps_sup, eps_sub = n_sup_w ** 2, n_sub_w ** 2
            kx0 = float(np.real(n_sup_w)) * np.sin(angle) * k0
            mats_w = [
                (m if m is not None else _build_sem_tensor_segments(
                    self.period, uwidths,
                    [_tensor3_dict(_tens(e, w)) for e in layer_eps_u[i]],
                    self.degree, self.n_el, self.grade))
                for i, m in enumerate(layer_mats)]
            Wsup, Vsup, _l, _g = _sem_modes_tensor(mats_sup, k0, kx0, True)
            Wsub, Vsub, _l, _g = _sem_modes_tensor(mats_sub, k0, kx0, True)
            lmodes = [_sem_modes_tensor(m, k0, kx0, True) for m in mats_w]
            S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
            for i, (Wl_, Vl_, lam_l, _q) in enumerate(lmodes):
                S = _redheffer_star(S, _propagation_smatrix(
                    lam_l, k0 * self._layers[i][0]))
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
            _warn_stack_energy(R, T)
            R_all[iw] = R
            T_all[iw] = T
            J_all[iw] = np.asarray(jr)
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
            S = _redheffer_star(S, _propagation_smatrix_general(
                lamf_l[i], lamb_l[i], k0 * self._layers[i][0]))
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
        kz_sup = _kz_forward(eps_sup, kx)
        kz_sub = _kz_forward(eps_sub, kx)
        kz_inc = float(np.real(_kz_forward(eps_sup, np.array([kx0 / k0]))[0]))
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
    layers and the half-spaces eig once per ``(wavelength, angle)``."""

    def __init__(self, stack):
        if any(abs(L[2]) > 1e-12 for L in stack._layers):
            raise NotImplementedError(
                "PMMStack.prepare: all-vertical stacks only.")
        if any(callable(e) for L in stack._layers for _w, e in L[1]):
            raise NotImplementedError(
                "PMMStack.prepare: dispersive (wl -> value) materials are the "
                "solve_vs_wavelength path; prepare() swaps material KEYS.")
        self._st = stack
        self._uwidths, self._layer_eps_u = _pmm_union_grid(
            [L[1] for L in stack._layers], stack.min_feature / stack.period)
        self._layer_keys = [sorted({e for e in eps_u if isinstance(e, str)})
                            for eps_u in self._layer_eps_u]
        self._eig_cache = {}
        self._mats_cache = {}

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
        if ck in self._eig_cache:
            return self._eig_cache[ck]
        if resolved is None:
            resolved = [st._as_tensor(e) for e in self._layer_eps_u[i]]
        mats = _build_sem_tensor_segments(
            st.period, self._uwidths, [_tensor3_dict(e) for e in resolved],
            st.degree, st.n_el, st.grade)
        modes = _sem_modes_tensor(mats, k0, kx0, True)
        self._eig_cache[ck] = modes
        return modes

    def solve(self, *, wavelength, materials=None, angle=0.0, theta=None):
        """Solve at ``wavelength`` with the material keys resolved from
        ``materials`` (values: scalar / ``(3, 3)`` / ``wl -> value``
        callables).  Returns ``(orders, R, T, jones)`` exactly as
        :meth:`PMMStack.solve`; equal to the rebuild path to ~1e-12."""
        st = self._st
        materials = dict(materials or {})
        angle = _resolve_incidence(angle, theta)
        wl = float(wavelength)
        k0 = 2.0 * np.pi / wl
        kx0 = float(np.real(st.n_sup)) * np.sin(angle) * k0
        eps_sup, eps_sub = st.n_sup ** 2, st.n_sub ** 2
        nU = len(self._uwidths)
        rk = (wl, angle)
        if rk not in self._mats_cache:
            t_sup = _tensor3_dict(eps_sup * np.eye(3))
            t_sub = _tensor3_dict(eps_sub * np.eye(3))
            m_sup = _build_sem_tensor_segments(
                st.period, self._uwidths, [t_sup] * nU, st.degree, st.n_el,
                st.grade)
            m_sub = _build_sem_tensor_segments(
                st.period, self._uwidths, [t_sub] * nU, st.degree, st.n_el,
                st.grade)
            self._mats_cache[rk] = (
                m_sup, m_sub,
                _sem_modes_tensor(m_sup, k0, kx0, True),
                _sem_modes_tensor(m_sub, k0, kx0, True))
        mats_sup, _mats_sub, sup_modes, sub_modes = self._mats_cache[rk]
        Wsup, Vsup = sup_modes[0], sup_modes[1]
        Wsub, Vsub = sub_modes[0], sub_modes[1]
        n_glob = mats_sup["n_glob"]

        lmodes = [self._layer_eig(i, materials, k0, kx0, wl, angle)
                  for i in range(len(st._layers))]
        # out-of-plane resolved tensors are not supported on this path
        S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
        for i, (Wl_, Vl_, lam_l, _q) in enumerate(lmodes):
            S = _redheffer_star(S, _propagation_smatrix(
                lam_l, k0 * st._layers[i][0]))
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
