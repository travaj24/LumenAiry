"""
lumenairy.optimize.jax_merits -- JAX-traceable merit variants.

v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
Hosts :class:`JaxMeritTerm` (a differentiable merit shell wrapping an
arbitrary JAX-traceable callable) and :func:`make_lg_aberration_merit_jax`
(the JAX twin of :class:`LGAberrationMerit`).  Re-exported by
``optimize/core.py`` for bit-for-bit public-API preservation.

These merits route through :func:`design_optimize`'s ``jac='auto'`` path
when constructed with ``build_args=callable(x)`` -- the analytic
gradient computed via :func:`jax.grad` is combined with finite-
difference gradients for the remaining non-JAX merit terms.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from .context import MeritTerm


class JaxMeritTerm(MeritTerm):
    """Differentiable merit term backed by a JAX-traceable callable.

    Has two operating modes:

    1. **Forward-only** (the default).  ``fn(ctx) -> JAX scalar``
       runs once per merit evaluation; the gradient w.r.t. design
       parameters comes from SciPy's finite-difference Jacobian.
       Useful for quick experimentation.

    2. **Differentiable in x** -- pass ``build_args=callable(x)``
       returning a tuple of JAX-traceable inputs to ``fn``.  In this
       mode :meth:`gradient_at_x` returns ``d(weight * |fn|) / dx``
       via :func:`jax.grad`, and :func:`design_optimize` will pick
       these gradients up automatically when ``jac='auto'`` (default
       there).  This closes the loop on JAX-differentiable merits:
       analytic gradients flow into SciPy without finite differences.

    Parameters
    ----------
    fn : callable
        Either ``fn(ctx) -> JAX scalar`` (mode 1) or
        ``fn(*args) -> JAX scalar`` (mode 2; args provided by
        ``build_args``).
    weight : float
    needs_wave : bool, default False
        Set True if the inner JAX function reads ``ctx.E_exit``.
        Forces the wave leg to run during :meth:`evaluate`.
    name : str, optional
    real_part : bool, default False
        If True, return ``real(fn(...))`` instead of ``|fn(...)|``.
        Useful when the JAX evaluator already returns a real scalar
        (e.g. RMS spot size).
    build_args : callable, optional
        ``build_args(x: ndarray) -> tuple of JAX scalars`` mapping a
        parameter vector to the positional arguments of ``fn``.
        Required for analytic gradient propagation through
        :func:`design_optimize`.

    Notes
    -----
    The forward-mode ``evaluate(ctx)`` works the same in both modes.
    When ``build_args`` is provided and ``ctx.x`` exists,
    ``evaluate`` calls ``fn(*build_args(ctx.x))`` (matching the
    gradient path).  Without ``ctx.x``, it falls back to the legacy
    ``fn(ctx)`` form -- letting the same JaxMeritTerm be used both
    inside ``design_optimize`` and in standalone evaluations.
    """

    name = 'JaxMerit'

    def __init__(self, fn: Callable, weight: float = 1.0,
                 needs_wave: bool = False,
                 name: Optional[str] = None,
                 real_part: bool = False,
                 build_args: Optional[Callable] = None,
                 needs_ray: bool = True) -> None:
        self.fn = fn
        self.weight = float(weight)
        self.needs_wave = bool(needs_wave)
        # needs_ray=False lets a rigorous-element merit (RCWA / metasurface /
        # coatings, with no imaging prescription) skip the ray-leg in
        # design_optimize -- no dummy lens prescription needed.
        self.needs_ray = bool(needs_ray)
        self.real_part = bool(real_part)
        self.build_args = build_args
        if name is not None:
            self.name = name

    def supports_jax_grad(self) -> bool:
        """True if this merit can produce an analytic gradient via
        :func:`jax.grad` on the parameter vector x."""
        return self.build_args is not None

    def _reduce(self, val):
        """Reduce a complex / array JAX value to a real scalar
        (matching :meth:`evaluate`'s output)."""
        import jax.numpy as jnp
        if self.real_part:
            return jnp.real(val)
        return jnp.abs(val)

    def evaluate(self, ctx: Any) -> float:
        if self.build_args is not None and getattr(ctx, 'x', None) is not None:
            val = self.fn(*self.build_args(ctx.x))
        else:
            val = self.fn(ctx)
        # Bridge JAX -> NumPy at the merit boundary.
        try:
            arr = np.asarray(val)
        except (TypeError, ValueError):
            # val is a non-array-like scalar (e.g. a Python float
            # returned from a non-JAX merit); fall through to the
            # scalar coercion path below.
            return self.weight * float(val)
        if self.real_part:
            return self.weight * float(arr.real)
        return self.weight * float(abs(arr))

    def gradient_at_x(self, x):
        """Analytic gradient of ``weight * reduction(fn(...))`` wrt x.

        Returns a NumPy array of shape ``(len(x),)``.  Requires
        ``build_args`` to have been supplied at construction.
        """
        if self.build_args is None:
            raise RuntimeError(
                "JaxMeritTerm.gradient_at_x requires build_args to be "
                "set; either pass build_args=... at construction or "
                "fall back to finite differences.")
        import jax
        import jax.numpy as jnp
        weight = self.weight

        def _scalar(x_jax):
            args = self.build_args(x_jax)
            val = self.fn(*args)
            return weight * self._reduce(val)

        # Use JAX's default float dtype -- requesting float64 raises
        # a UserWarning when jax_enable_x64 isn't set, which is the
        # common case for a fresh ``import jax``.  OPT-nit
        # (AUDIT_OPTIMIZE_MERITS): consequently, for a GENERIC JaxMeritTerm
        # with x64 OFF the gradient is single precision while the forward
        # ``evaluate`` bridges through ``float()`` (double) -- a precision
        # mismatch inherent to JAX-without-x64.  ``make_lg_aberration_merit_jax``
        # enables x64 process-wide, so ITS gradient is float64 here; callers
        # wanting double-precision gradients from other JaxMeritTerms should
        # enable x64 themselves (``jax.config.update('jax_enable_x64', True)``).
        x_jax = jnp.asarray(x)
        g = jax.grad(_scalar)(x_jax)
        return np.asarray(g)


def _is_jax_tracer(a: Any) -> bool:
    """True iff ``a`` is a *live* JAX tracer (a ``jax.grad`` / ``jax.jit``
    ``Tracer`` -- a differentiable value flowing through a trace).

    Concrete values (Python / NumPy scalars, materialised ``jax.Array``)
    return ``False``: they can be safely ``float()``-ed by the static
    downstream code.  Used by :func:`make_lg_aberration_merit_jax` to
    reject differentiable ``build_args`` inputs that are consumed as
    static Python floats (see S3-6).
    """
    try:
        import jax
        return isinstance(a, jax.core.Tracer)
    except (ImportError, AttributeError):     # no JAX / old JAX layout
        return False


def make_lg_aberration_merit_jax(prescription: Dict[str, Any],
                                 wavelength: float,
                                 targets: Dict[Any, float],
                                 build_args: Callable,
                                 *,
                                 field_points: Optional[Sequence[Any]] = None,
                                 image_points: Optional[Sequence[Any]] = None,
                                 w_s: float = 50e-6,
                                 w_p: float = 0.05,
                                 poly_order: int = 4,
                                 n_field: int = 8, n_pupil: int = 8,
                                 weight: float = 1.0,
                                 name: str = 'LGAberrationJax') -> "JaxMeritTerm":
    """Build a JAX-grad-compatible LG-aberration merit term.

    Wraps :func:`fit_canonical_polynomials_jax` +
    :func:`aberration_tensor_lg00_jax` into a :class:`JaxMeritTerm`
    so :func:`design_optimize` picks up its analytic gradient via
    the existing ``jac='auto'`` routing.  The merit returns the
    weighted sum of ``|L_{(p, ell), (0,0)}|^2`` across all targets
    and field points, just like :class:`LGAberrationMerit`.

    *Currently supported as differentiable inputs* (map into these via
    ``build_args``):

    * ``w_s``, ``w_p`` (source / pupil Gaussian waists) -- the only two
      ``build_args`` slots that carry a ``jax.grad`` tracer end to end
      (both FD-verified to ~1e-6).

    *Accepted only as STATIC overrides* -- pass a plain Python / NumPy
    float, or ``None`` for the nominal.  S3-6 (AUDIT_V5_24_2): routing a
    *differentiable* ``x`` into any of the four slots below raises
    :class:`NotImplementedError` at trace time.  They are consumed as
    static Python floats deep in :func:`fit_canonical_polynomials_jax`
    (``float(pupil_box_half)``, ``-float(object_distance)``,
    ``opd / float(wavelength)``) and in :func:`trace_jax`'s glass
    dispersion, so a tracer there would otherwise crash ``jax.grad``
    with an opaque ``TracerArrayConversionError`` /
    ``ConcretizationTypeError``:

    * ``wavelength`` (chromatic optimisation -- needs a JAX-traceable
      glass dispersion; roadmap)
    * ``source_box_half``, ``pupil_box_half`` (fit-domain knobs)
    * ``object_distance`` (source-plane positioning)

    To optimise wavelength / box-size / object-distance, use the
    finite-difference path (``design_optimize(..., jac='fd')`` or a
    non-JAX merit term).

    *NOT supported as inputs at all:* radii, thicknesses, conics,
    aspheric coefficients.  These are read by :func:`trace_jax` as
    static Python floats; making the trace differentiable in
    prescription parameters is on the roadmap (a "trace_jax with
    JAX-array surfaces" extension) but not in this release.  Until
    then, design-parameter optimisation uses the existing FD path,
    which is still adequate for systems with O(30-50) free vars.

    Parameters
    ----------
    prescription : dict
        Static lumenairy prescription (radii, conics, thicknesses,
        glass).  Treated as a closure constant by JAX.
    wavelength : float
        Wavelength [m] -- can be made differentiable via
        ``build_args``.
    targets : dict[(int, int), float]
        LG channels to penalise; same format as
        :class:`LGAberrationMerit`.  **The JAX path supports only the
        ``(0, 0)`` channel** (``aberration_tensor_lg00_jax`` computes the
        (0,0) Strehl amplitude only) -- a ``{(0, 0): w}`` target minimises
        the Strehl DEFICIT ``w*(1 - |Strehl|^2)`` (i.e. maximises the on-axis
        Strehl / sharpens focus).  For general aberration-channel control
        (e.g. primary spherical ``(2, 0)``) use the NumPy sibling
        :class:`LGAberrationMerit` (finite-difference path); a non-(0,0)
        target raises ``NotImplementedError`` here.
    build_args : callable
        ``build_args(x: ndarray) -> tuple`` mapping the parameter
        vector to differentiable scalars passed positionally to
        the inner JAX function.  The inner function's signature is
        ``fn(wavelength, source_box_half, pupil_box_half,
        object_distance, w_s, w_p)`` -- match the order of returned
        scalars to those positions, or pass ``None`` placeholders
        for static values.

    Other parameters mirror :class:`LGAberrationMerit`.

    Returns
    -------
    merit : JaxMeritTerm
        Plug into :class:`CompositeMerit` like any other merit
        term.  ``design_optimize(jac='auto', ...)`` will use
        ``merit.gradient_at_x`` automatically.

    Examples
    --------
    Optimise the source-Gaussian waist ``w_s`` to MAXIMISE the on-axis
    Strehl (minimise the Strehl deficit ``1 - |Strehl|^2``)::

        def build_args(x):
            return (None, None, None, None, x[0], None)

        merit = make_lg_aberration_merit_jax(
            prescription, wavelength=1.31e-6,
            targets={(0, 0): 1.0},   # (0,0) is the only JAX-supported channel
            build_args=build_args,
            field_points=[(0.0, 0.0)],
        )
        # x[0] = w_s, optimise via design_optimize (which MINIMISES the
        # merit sum -> drives |Strehl|^2 -> 1).  For general aberration
        # channels (e.g. (2, 0) spherical) use the NumPy LGAberrationMerit.

    Notes
    -----
    OPT-nit (AUDIT_OPTIMIZE_MERITS): constructing this merit enables JAX
    double precision PROCESS-WIDE (``jax.config.update('jax_enable_x64',
    True)``) -- the fit needs float64 for a meaningful gradient.  A caller
    relying on JAX's default float32 elsewhere in the same process is
    silently switched to float64 by this call.
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError(
            "make_lg_aberration_merit_jax requires JAX.  "
            "Install with `pip install jax`.")
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from ..propagators.asymptotic import (
        aberration_tensor_lg00_jax,
        fit_canonical_polynomials_jax,
    )
    if field_points is None:
        field_points = [(0.0, 0.0)]

    targets_kv = [(tuple(k), float(v)) for k, v in targets.items()]
    # v4.13.2 (C-P0-1): aberration_tensor_lg00_jax only computes the
    # (0, 0) -> (0, 0) -> (0, 0) coefficient.  General (p, ell)
    # targets need a full aberration_tensor_jax that does not yet
    # exist.  Reject non-(0, 0) targets at construction time with a
    # clear migration message so the silent ``pass``-only loop bug
    # (the merit returning the same piston sum regardless of the
    # ``targets`` dict) cannot recur.
    non_piston = [k for k, _ in targets_kv if k != (0, 0)]
    if non_piston:
        raise NotImplementedError(
            f"make_lg_aberration_merit_jax: targets other than "
            f"(0, 0) are not supported in the JAX path "
            f"(got {non_piston}).  aberration_tensor_lg00_jax "
            f"computes the (0, 0) Strehl amplitude only.  Use the "
            f"NumPy sibling :class:`LGAberrationMerit` for general "
            f"(p, ell) targets, or restrict ``targets`` to "
            f"``{{(0, 0): wgt}}``.")
    # Piston weight: default to 1.0 if (0, 0) is absent so the merit
    # still has well-defined semantics (matches the LGAberrationMerit
    # behaviour of always including piston in output_modes).
    piston_weight = 1.0
    for k, w in targets_kv:
        if k == (0, 0):
            piston_weight = float(w)
            break
    nominal_w_s = float(w_s)
    nominal_w_p = float(w_p)
    nominal_lambda = float(wavelength)
    nominal_obj_d = float(prescription.get('object_distance', 0.0) or 0.0)
    # Use these defaults for any None placeholders in build_args.
    nominal = (nominal_lambda, 50e-6, 0.05, nominal_obj_d,
                nominal_w_s, nominal_w_p)

    # Import the IFT-based envelope solver so v_star is JAX-traceable
    # (required positional argument of aberration_tensor_lg00_jax).
    from ..propagators.asymptotic import (
        solve_envelope_stationary_jax_ift,
    )

    def fn(*args):
        # Resolve None placeholders to nominals.
        wl, sbh, pbh, obj_d, w_s_local, w_p_local = (
            (a if a is not None else default)
            for a, default in zip(args, nominal))
        # S3-6 (AUDIT_V5_24_2): only w_s (slot 4) / w_p (slot 5) are
        # differentiable.  The wavelength / source_box_half /
        # pupil_box_half / object_distance slots are consumed as static
        # Python floats deep in the trace (fit_canonical_polynomials_jax:
        # float(pupil_box_half), -float(object_distance),
        # opd/float(wavelength); trace_jax glass dispersion), so a
        # build_args that routes a *differentiable* x into any of them
        # crashes jax.grad with an opaque TracerArrayConversionError /
        # ConcretizationTypeError.  Detect the live tracer at trace time
        # and raise ONE actionable error pointing at the FD path, instead
        # of four confusing deep failures.  A *static* float override in
        # these slots is fine and passes through untouched -- only a
        # differentiable tracer is rejected.
        _live = [nm for nm, v in (('wavelength', wl),
                                  ('source_box_half', sbh),
                                  ('pupil_box_half', pbh),
                                  ('object_distance', obj_d))
                 if _is_jax_tracer(v)]
        if _live:
            raise NotImplementedError(
                f"make_lg_aberration_merit_jax: build_args routes a "
                f"differentiable x into {_live}, but only ``w_s`` "
                f"(slot 4) and ``w_p`` (slot 5) are differentiable in "
                f"this release.  {_live} are read as static Python "
                f"floats by fit_canonical_polynomials_jax / trace_jax "
                f"and cannot carry a jax.grad tracer (that is the deep "
                f"TracerArrayConversionError / ConcretizationTypeError "
                f"you would otherwise hit).  Either pass these as static "
                f"values (a Python/NumPy float, or None for the nominal) "
                f"in build_args, or optimise them via the "
                f"finite-difference path (design_optimize(..., "
                f"jac='fd'), or a non-JAX merit term).")
        fit = fit_canonical_polynomials_jax(
            prescription, float(wl) if not hasattr(wl, 'shape') else wl,
            source_box_half=sbh, pupil_box_half=pbh,
            n_field=n_field, n_pupil=n_pupil,
            poly_order=poly_order,
            object_distance=float(obj_d) if not hasattr(obj_d, 'shape') else obj_d,
        )
        total = jnp.array(0.0)
        for ifp, src in enumerate(field_points):
            if image_points is not None:
                s2_img = image_points[ifp]
            else:
                s2_img = (fit.s2x_centre, fit.s2y_centre)
            # v_star is required positionally; compute via the
            # IFT-based solver so gradients still flow back through
            # build_args inputs (w_s, w_p, s2, source_point).
            v_star = solve_envelope_stationary_jax_ift(
                fit, s2_img, tuple(src),
                w_s=w_s_local, w_p=w_p_local,
                v2_centre=(fit.v2x_centre, fit.v2y_centre))
            res = aberration_tensor_lg00_jax(
                fit, s2_img, v_star,
                source_point=tuple(src),
                w_s=w_s_local, w_p=w_p_local,
                v2_centre=(fit.v2x_centre, fit.v2y_centre))
            # res is a complex scalar (the L_{(0,0),(0,0)} element) -- the
            # leading STREHL AMPLITUDE: |res|^2 -> 1 for a perfect system and
            # -> 0 as aberration grows.  OPT-1 (AUDIT_OPTIMIZE_MERITS): the
            # merit must therefore be the Strehl DEFICIT ``1 - |res|^2`` (0
            # for a perfect system, growing with aberration), NOT ``|res|^2``
            # -- ``design_optimize`` MINIMISES the weighted merit sum, so the
            # old ``|res|^2`` drove the design toward |Strehl|=0 (MAXIMUM
            # aberration), the opposite of every other merit and of
            # optimize_traced_geometry's peak-intensity objective.  Weighted
            # by piston_weight so ``targets={(0, 0): w}`` scales it linearly.
            total = total + piston_weight * (1.0 - jnp.abs(res) ** 2)
        return total

    return JaxMeritTerm(
        fn=fn, weight=weight, name=name,
        real_part=True,         # fn already returns real
        build_args=build_args,
    )


def _asm_jax(E, z, wavelength, dx):
    """Minimal JAX angular-spectrum free-space propagator (exact scalar
    Helmholtz), for use inside a differentiable design merit."""
    import jax.numpy as jnp
    N = E.shape[-1]
    fx = jnp.fft.fftfreq(N, dx)
    FX, FY = jnp.meshgrid(fx, fx)
    k = 2.0 * jnp.pi / wavelength
    kz = jnp.sqrt((k * k - (2 * jnp.pi * FX) ** 2
                   - (2 * jnp.pi * FY) ** 2).astype(jnp.complex128))
    return jnp.fft.ifft2(jnp.fft.fft2(E) * jnp.exp(1j * kz * z))


def optimize_traced_geometry(
    E_in, prescription, *, wavelength, dx,
    focal_distance=None,
    optimize=('radii',),
    merit=None,
    n_steps=40, lr=None,
    cheb_order=10, newton_iters=12, ray_subsample=8,
    verbose=False,
):
    """Turnkey gradient-based lens-design optimisation of prescription GEOMETRY
    (radii / conics / thicknesses) via the differentiable ray-traced OPD.

    Wraps :func:`~lumenairy.elements._lens_jax.apply_real_lens_traced_jax`'s
    differentiable-geometry path (``jax.grad`` w.r.t. ``radii`` / ``conics`` /
    ``thicknesses``) in an Adam loop.  The **default merit sharpens the focus**:
    propagate the lens output to ``focal_distance`` with an exact angular-
    spectrum step and maximise the peak intensity there (a Strehl-like objective)
    -- so it rewards the *correct* converging wavefront, not a flat one.  Pass a
    custom ``merit(E_out_jax) -> real scalar to MINIMISE`` for any other goal.

    Parameters
    ----------
    E_in : (N, N) complex -- input field at the lens plane (e.g. a collimated
        aperture field to be focused).
    prescription : dict -- lumenairy prescription; its ``surfaces`` radii /
        conics and ``thicknesses`` are the geometry being optimised.
    optimize : subset of ``('radii', 'conics', 'thicknesses')`` -- which
        geometry to free.  Infinite (flat) radii are held fixed.
    focal_distance : float -- required for the default focus merit (the plane,
        past the lens output, where the focus forms).
    merit : callable, optional -- ``E_out -> scalar``; overrides the default.
    n_steps, lr : Adam iteration count and step (``lr`` auto-scales to the
        parameter magnitude when ``None``).

    Returns
    -------
    dict -- ``{'prescription': optimised dict, 'loss_history': [...],
    'loss': final, 'params': ndarray}``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'optimize_traced_geometry')
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError("optimize_traced_geometry requires JAX.")
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from ..elements._lens_jax import apply_real_lens_traced_jax
    if merit is None and focal_distance is None:
        raise ValueError(
            "optimize_traced_geometry: pass focal_distance for the default "
            "focus-sharpening merit, or supply a custom merit=.")

    surfaces = prescription.get('surfaces', [])
    radii0 = jnp.array([float(s.get('radius', np.inf)) for s in surfaces])
    conics0 = jnp.array([float(s.get('conic', 0.0)) for s in surfaces])
    thick0 = jnp.array([float(t) for t in prescription.get('thicknesses', [])])
    E0 = jnp.asarray(E_in)

    specs = []
    if 'radii' in optimize:
        specs.append(('radii', [i for i in range(len(surfaces))
                                 if np.isfinite(float(radii0[i]))]))
    if 'conics' in optimize:
        specs.append(('conics', list(range(len(surfaces)))))
    if 'thicknesses' in optimize:
        specs.append(('thicknesses', list(range(int(thick0.shape[0])))))

    base_of = {'radii': radii0, 'conics': conics0, 'thicknesses': thick0}
    p0 = jnp.array([float(base_of[name][i]) for name, idx in specs for i in idx])

    def unpack(p):
        radii, conics, thick = radii0, conics0, thick0
        off = 0
        for name, idx in specs:
            vals = p[off:off + len(idx)]
            off += len(idx)
            ia = jnp.array(idx)
            if name == 'radii':
                radii = radii.at[ia].set(vals)
            elif name == 'conics':
                conics = conics.at[ia].set(vals)
            else:
                thick = thick.at[ia].set(vals)
        return radii, conics, thick

    def _merit(E_out):
        if merit is not None:
            return merit(E_out)
        Efoc = _asm_jax(E_out, focal_distance, wavelength, dx)
        return -jnp.max(jnp.abs(Efoc) ** 2) / (jnp.sum(jnp.abs(E0) ** 2) + 1e-30)

    def loss(p):
        radii, conics, thick = unpack(p)
        E_out = apply_real_lens_traced_jax(
            E0, prescription=prescription, wavelength=wavelength, dx=dx,
            radii=radii, conics=conics, thicknesses=thick,
            cheb_order=cheb_order, newton_iters=newton_iters,
            ray_subsample=ray_subsample)
        return _merit(E_out)

    grad_loss = jax.grad(loss)
    if lr is None:
        lr = 0.02 * float(jnp.mean(jnp.abs(p0)) + 1e-30)
    m = jnp.zeros_like(p0)
    v = jnp.zeros_like(p0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    p = p0
    hist = [float(loss(p))]
    for step in range(n_steps):
        g = grad_loss(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mhat = m / (1 - b1 ** (step + 1))
        vhat = v / (1 - b2 ** (step + 1))
        p = p - lr * mhat / (jnp.sqrt(vhat) + eps)
        hist.append(float(loss(p)))
        if verbose:
            print(f"  step {step + 1}/{n_steps}: loss={hist[-1]:.6e}")

    radii, conics, thick = unpack(p)
    opt = dict(prescription)
    opt_surfaces = []
    for i, s in enumerate(surfaces):
        s2 = dict(s)
        if np.isfinite(float(radii0[i])) and 'radii' in optimize:
            s2['radius'] = float(radii[i])
        if 'conics' in optimize:
            s2['conic'] = float(conics[i])
        opt_surfaces.append(s2)
    opt['surfaces'] = opt_surfaces
    if 'thicknesses' in optimize:
        opt['thicknesses'] = [float(t) for t in thick]
    return {'prescription': opt, 'loss_history': hist, 'loss': hist[-1],
            'params': np.asarray(p)}
