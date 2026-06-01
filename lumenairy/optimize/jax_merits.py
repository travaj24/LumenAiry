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
        # common case for a fresh ``import jax``.
        x_jax = jnp.asarray(x)
        g = jax.grad(_scalar)(x_jax)
        return np.asarray(g)


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

    *Currently supported as differentiable inputs* (anything you map
    into via ``build_args``):

    * ``wavelength`` (chromatic optimisation)
    * ``source_box_half``, ``pupil_box_half`` (fit-domain knobs;
      useful for fit-quality optimisation but rarely a design free
      var)
    * ``object_distance`` (source-plane positioning)
    * ``w_s``, ``w_p`` (source / pupil Gaussian waists)

    *NOT currently supported as differentiable inputs:* radii,
    thicknesses, conics, aspheric coefficients.  These are read by
    :func:`trace_jax` as static Python floats; making the trace
    differentiable in prescription parameters is on the roadmap (a
    "trace_jax with JAX-array surfaces" extension) but not in this
    release.  Until then, design-parameter optimisation uses the
    existing FD path, which is still adequate for systems with
    O(30-50) free vars.

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
        :class:`LGAberrationMerit`.
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
    Optimise the source-Gaussian waist ``w_s`` to minimise primary
    spherical aberration::

        def build_args(x):
            return (None, None, None, None, x[0], None)

        merit = make_lg_aberration_merit_jax(
            prescription, wavelength=1.31e-6,
            targets={(2, 0): 1.0},
            build_args=build_args,
            field_points=[(0.0, 0.0)],
        )
        # x[0] = w_s, optimise via design_optimize.
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
            # res is a complex scalar (the L_{(0,0),(0,0)} element);
            # weight by the user-supplied piston weight so changing
            # ``targets={(0, 0): w}`` actually scales the merit
            # linearly in w (the pre-fix code ignored ``wgt`` and
            # always returned the same |L|^2 sum).
            total = total + piston_weight * (jnp.abs(res) ** 2)
        return total

    return JaxMeritTerm(
        fn=fn, weight=weight, name=name,
        real_part=True,         # fn already returns real
        build_args=build_args,
    )
