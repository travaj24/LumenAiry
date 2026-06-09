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
                 far_field_orders=21, n_orders=None, factorization="auto"):
        far_field_orders = _resolve_order_count(far_field_orders, n_orders)
        if int(degree) < 2:
            raise ValueError("PMMStack: degree must be >= 2.")
        if factorization not in ("auto", "convection", "covariant"):
            raise ValueError(
                "PMMStack: factorization must be 'auto', 'convection' or "
                f"'covariant', got {factorization!r}.")
        self.period = float(period)
        self.n_sub = _C(n_substrate)
        self.n_sup = _C(n_superstrate)
        self.degree = int(degree)
        self.n_el = int(elements_per_region)
        self.grade = bool(grade)
        self.factorization = factorization
        self.ffo = int(far_field_orders)
        self._layers = []                          # (thickness, segments)
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
        if eps is not None:
            segs = [(1.0, self._as_tensor(eps))]
        else:
            if len(segments) < 1:
                raise ValueError("PMMStack.add_layer: empty segments.")
            segs = [(float(w), self._as_tensor(e)) for w, e in segments]
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

    def solve(self):
        """Solve the stack.  Returns ``(orders, R_eff, T_eff, jones_reflection)``
        as :func:`pmm_jones_1d_segments` (``R_eff`` / ``T_eff`` are ``(2, M)``:
        row 0 = incident ``E_x``, row 1 = incident ``E_y``; ``jones`` is the
        zeroth-order ``2x2`` reflection)."""
        if self._src is None:
            raise ValueError("PMMStack.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("PMMStack.solve: add at least one layer.")
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

        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers])
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
            S = _interface_smatrix(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
            for i, (Wl, Vl, lam_l, _q) in enumerate(lmodes):
                S = _redheffer_star(S, _propagation_smatrix(
                    lam_l, k0 * self._layers[i][0]))
                nW, nV = ((Wsub, Vsub) if i == len(lmodes) - 1
                          else (lmodes[i + 1][0], lmodes[i + 1][1]))
                S = _redheffer_star(S, _interface_smatrix(Wl, Vl, nW, nV))
        else:
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
        _warn_stack_energy(R_eff, T_eff)
        return orders, R_eff, T_eff, jones

    def solve_vs_wavelength(self, wavelengths, *, angle=0.0, theta=None):
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

        Returns
        -------
        orders : (N,) int ndarray
            The (wavelength-independent) retained diffraction orders.
        R, T : (n_wavelengths, 2, N) float ndarray
            Reflected / transmitted efficiency, ``[wavelength][incident pol][order]``
            (pol 0 = incident ``E_x``, pol 1 = incident ``E_y``).
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
        oop = any(self._is_oop(M) for L in self._layers for _w, M in L[1])
        if oop or any(abs(L[2]) > 1e-12 for L in self._layers):
            raise NotImplementedError(
                "PMMStack.solve_vs_wavelength: all-vertical in-plane stacks only "
                "(the symmetric cascade); call solve() per wavelength for slanted "
                "/ out-of-plane stacks.")

        # ---- GEOMETRY-ONLY assembly (reused across the whole sweep) ----
        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers])
        nU = len(uwidths)
        layer_mats = [
            _build_sem_tensor_segments(
                self.period, uwidths, [_tensor3_dict(e) for e in eps_u],
                self.degree, self.n_el, self.grade)
            for eps_u in layer_eps_u]
        t_sup = _tensor3_dict(self.n_sup ** 2 * np.eye(3))
        t_sub = _tensor3_dict(self.n_sub ** 2 * np.eye(3))
        mats_sup = _build_sem_tensor_segments(
            self.period, uwidths, [t_sup] * nU, self.degree, self.n_el, self.grade)
        mats_sub = _build_sem_tensor_segments(
            self.period, uwidths, [t_sub] * nU, self.degree, self.n_el, self.grade)
        n_glob = mats_sup["n_glob"]
        eps_sup, eps_sub = self.n_sup ** 2, self.n_sub ** 2

        # FIXED order set: cover the SHORTEST wavelength's propagating orders.
        n_max = max([np.real(np.sqrt(np.asarray(e, _C)[0, 0]))
                     for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(np.sqrt(np.asarray(e, _C)[1, 1]))
                       for eps_u in layer_eps_u for e in eps_u]
                    + [np.real(self.n_sup), np.real(self.n_sub)])
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
        for iw, w in enumerate(wl):
            k0 = 2.0 * np.pi / float(w)
            kx0 = float(np.real(self.n_sup)) * np.sin(angle) * k0
            Wsup, Vsup, _l, _g = _sem_modes_tensor(mats_sup, k0, kx0, True)
            Wsub, Vsub, _l, _g = _sem_modes_tensor(mats_sub, k0, kx0, True)
            lmodes = [_sem_modes_tensor(m, k0, kx0, True) for m in layer_mats]
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
            R, T, _j = _assemble_jones_farfield(
                Hsup, Hsub, S11, S21, orders, kx, kz_sup, kz_sub, kz_inc,
                kx0 / k0, N)
            _warn_stack_energy(R, T)
            R_all[iw] = R
            T_all[iw] = T
        return orders, R_all, T_all

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
        uwidths, layer_eps_u = _pmm_union_grid([L[1] for L in self._layers])
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
