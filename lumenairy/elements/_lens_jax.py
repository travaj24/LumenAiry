"""
lumenairy.elements._lens_jax -- JAX-traceable real-lens propagators.

Two ``jax.grad``-friendly variants of the analytic / ray-traced
real-lens path:

* ``apply_real_lens_traced_jax`` -- per-pixel ray-traced phase screen
  via ``trace_jax`` + Chebyshev-fit Newton inversion.
* ``apply_real_lens_maslov_jax`` -- traced + det(J)-sign-flip caustic
  phase counter.  NOT algorithmically equivalent to the NumPy
  ``apply_real_lens_maslov`` (audit P2-03) -- see its docstring for
  the even-multiplicity (point-focus Gouy) limitation.

Plus the shared Chebyshev tensor-product fit + evaluation helpers
that both functions use, and a ``custom_jvp``-wrapped amplitude
callback that lets ``jax.grad`` flow through the NumPy
``apply_real_lens`` analytic leg.

Extracted from ``lenses.py`` in v3.5.5.  Re-exported from
:mod:`lumenairy.elements.lenses` for backwards-compatible imports.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _jax_available():
    """Lazy JAX_AVAILABLE check.  Imported lazily to keep this module
    importable without :mod:`lumenairy.backend` being eager.
    """
    from ..backend import JAX_AVAILABLE as _JA
    return _JA


# Sibling-module imports.  apply_real_lens (analytic split-step) is
# the workhorse for the amplitude leg of the JAX traced/Maslov
# variants; called via jax.pure_callback wrapped in jax.custom_jvp
# so jax.grad through E_in works.
from ..glass import get_glass_index
from ._lens_real import apply_real_lens

# ----------------------------------------------------------------------
# JAX helpers shared by apply_real_lens_traced_jax and
# apply_real_lens_maslov_jax: Chebyshev tensor-product fit + evaluation,
# Newton inversion of a 2-D smooth map.
# ----------------------------------------------------------------------

def _cheb_T_recurrence_jax(u, max_k):
    """Build T_0..T_{max_k} for argument u via the Chebyshev recurrence.

    Returns a list of arrays (length max_k+1), each with the shape of
    ``u``.  ``max_k`` is a Python int (static at trace time).
    """
    import jax.numpy as jnp
    if max_k == 0:
        return [jnp.ones_like(u)]
    T = [jnp.ones_like(u), u]
    for n in range(2, max_k + 1):
        T.append(2.0 * u * T[-1] - T[-2])
    return T


def _cheb_Tprime_recurrence_jax(u, max_k):
    """Build T_0..T_{max_k} and their derivatives T'_0..T'_{max_k}."""
    import jax.numpy as jnp
    T = [jnp.ones_like(u)]
    Tp = [jnp.zeros_like(u)]
    if max_k >= 1:
        T.append(u)
        Tp.append(jnp.ones_like(u))
    for n in range(2, max_k + 1):
        T.append(2.0 * u * T[-1] - T[-2])
        Tp.append(2.0 * T[-2] + 2.0 * u * Tp[-1] - Tp[-2])
    return T, Tp


def _cheb_total_degree_indices(order):
    """Return (K1, K2) lists of multi-indices with kx + ky <= order."""
    K1 = []
    K2 = []
    for kx in range(order + 1):
        for ky in range(order + 1 - kx):
            K1.append(kx)
            K2.append(ky)
    return K1, K2


def _cheb_fit_2d_jax(xs, ys, vals, order):
    """Total-degree Chebyshev tensor-product fit.

    Mirrors ``_Cheb2DEvaluator``'s lstsq fit but in JAX.  Returns
    ``coeffs`` of shape (n_terms,), the multi-index arrays ``K1, K2``,
    and the (xmin, xmax, ymin, ymax) scaling bounds.
    """
    import jax.numpy as jnp
    xmin = jnp.min(xs)
    xmax = jnp.max(xs)
    ymin = jnp.min(ys)
    ymax = jnp.max(ys)
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')
    u = (2.0 * X - (xmin + xmax)) / (xmax - xmin)
    v = (2.0 * Y - (ymin + ymax)) / (ymax - ymin)
    Tu = jnp.stack(_cheb_T_recurrence_jax(u, order))   # (order+1, n_xs, n_ys)
    Tv = jnp.stack(_cheb_T_recurrence_jax(v, order))
    K1_list, K2_list = _cheb_total_degree_indices(order)
    K1 = jnp.asarray(K1_list, dtype=jnp.int32)
    K2 = jnp.asarray(K2_list, dtype=jnp.int32)
    # A_full[i] = T_{K1[i]}(u) * T_{K2[i]}(v), shape (n_terms, n_xs, n_ys).
    A_full = Tu[K1] * Tv[K2]
    n_terms = K1.shape[0]
    A = A_full.reshape(n_terms, -1).T  # (n_pts, n_terms)
    rhs = vals.ravel()
    # Drop NaN rows so the fit is robust to vignetting (rare).
    finite = jnp.isfinite(rhs)
    rhs_safe = jnp.where(finite, rhs, 0.0)
    weights = finite.astype(rhs.dtype)
    Aw = A * weights[:, None]
    bw = rhs_safe * weights
    coeffs, *_ = jnp.linalg.lstsq(Aw, bw, rcond=None)
    return coeffs, K1, K2, xmin, xmax, ymin, ymax


def _cheb_eval_value_grad_jax(coeffs, K1, K2, xmin, xmax, ymin, ymax,
                              x, y, order):
    """Evaluate the polynomial fit and its physical-space gradient at (x, y).

    Returns ``(f, df/dx, df/dy)`` with the broadcast shape of ``(x, y)``.
    """
    import jax.numpy as jnp
    u = (2.0 * x - (xmin + xmax)) / (xmax - xmin)
    v = (2.0 * y - (ymin + ymax)) / (ymax - ymin)
    Tu, Tup = _cheb_Tprime_recurrence_jax(u, order)
    Tv, Tvp = _cheb_Tprime_recurrence_jax(v, order)
    Tu = jnp.stack(Tu)
    Tup = jnp.stack(Tup)   # (order+1, *u.shape)
    Tv = jnp.stack(Tv)
    Tvp = jnp.stack(Tvp)
    # Reshape coeffs so it broadcasts over the trailing dims of u/v.
    extra_dims = u.ndim
    coeffs_b = coeffs.reshape((-1,) + (1,) * extra_dims)
    Tu_K1 = Tu[K1]    # (n_terms, *u.shape)
    Tv_K2 = Tv[K2]
    Tup_K1 = Tup[K1]
    Tvp_K2 = Tvp[K2]
    cu = coeffs_b * Tu_K1
    f = jnp.sum(cu * Tv_K2, axis=0)
    fx_norm = jnp.sum(coeffs_b * Tup_K1 * Tv_K2, axis=0)
    fy_norm = jnp.sum(cu * Tvp_K2, axis=0)
    # Chain rule: u = (2x - (xmin+xmax)) / (xmax - xmin), so du/dx = 2/(xmax-xmin).
    fx = fx_norm * (2.0 / (xmax - xmin))
    fy = fy_norm * (2.0 / (ymax - ymin))
    return f, fx, fy


def _newton_invert_2d_jax(Xw, Yw,
                           Sx_coeffs, Sy_coeffs,
                           Sx_K1, Sx_K2, Sy_K1, Sy_K2,
                           xmin, xmax, ymin, ymax, order,
                           inv_M_x, inv_M_y, bound,
                           n_iter=12):
    """Newton-invert the smooth map (xe, ye) -> (Sx(xe,ye), Sy(xe,ye)).

    Returns the converged entrance positions ``(xe, ye)`` such that
    ``(Sx, Sy)(xe, ye) ~ (Xw, Yw)``.  Fully traceable via JAX; uses
    ``lax.fori_loop`` for a fixed iteration count.
    """
    import jax
    import jax.numpy as jnp
    xe0 = Xw * inv_M_x
    ye0 = Yw * inv_M_y

    def body(_, carry):
        xe, ye = carry
        fx, jxx, jxy = _cheb_eval_value_grad_jax(
            Sx_coeffs, Sx_K1, Sx_K2, xmin, xmax, ymin, ymax,
            xe, ye, order)
        fy, jyx, jyy = _cheb_eval_value_grad_jax(
            Sy_coeffs, Sy_K1, Sy_K2, xmin, xmax, ymin, ymax,
            xe, ye, order)
        rx = fx - Xw
        ry = fy - Yw
        det = jxx * jyy - jxy * jyx
        det_safe = jnp.where(jnp.abs(det) > 1e-18, det, 1e-18)
        dxe = (jyy * rx - jxy * ry) / det_safe
        dye = (-jyx * rx + jxx * ry) / det_safe
        xe_new = jnp.clip(xe - dxe, -bound, bound)
        ye_new = jnp.clip(ye - dye, -bound, bound)
        return xe_new, ye_new

    xe, ye = jax.lax.fori_loop(0, int(n_iter), body, (xe0, ye0))
    return xe, ye


def _amp_callback_numpy(E_in_np, lens_prescription, wavelength, dx,
                       preserve_input_phase, bandlimit):
    """Pure-NumPy callback: returns (amp_complex, phase_analytic_lens).

    Lives outside the JAX trace -- treated as a non-differentiable
    closure (gradients do not flow through the prescription via this
    leg, only through the OPL leg).  Returns two real-valued arrays
    when ``preserve_input_phase=False`` (in which case ``amp_complex``
    is the analytic complex field whose magnitude we use, and
    ``phase_analytic_lens`` is unused).
    """
    E_analytic = apply_real_lens(
        E_in_np, prescription=lens_prescription, wavelength=wavelength, dx=dx, bandlimit=bandlimit)
    if preserve_input_phase:
        # Need the analytic lens phase on a unit plane wave for the
        # delta-phase correction.
        ones = np.ones_like(E_in_np)
        E_lens_pw = apply_real_lens(
            ones, prescription=lens_prescription, wavelength=wavelength, dx=dx, bandlimit=bandlimit)
        phase_pw = np.angle(E_lens_pw)
    else:
        phase_pw = np.zeros_like(np.asarray(E_in_np).real, dtype=np.float64)
    return E_analytic, phase_pw


def _amp_callback_jax_linear(E_in, lens_prescription, wavelength, dx,
                              preserve_input_phase, bandlimit):
    """Linear pure-callback wrapper with a custom JVP rule.

    ``apply_real_lens`` is a linear operator on the input field, so
    differentiating ``E_analytic = T(E_in)`` w.r.t. E_in just means
    re-applying ``T`` to the tangent.  The plane-wave reference phase
    (used only when ``preserve_input_phase=True``) does not depend on
    E_in and has zero tangent.  Implements this as a
    ``jax.custom_jvp`` closure so ``jax.grad`` flows through the
    amplitude leg.
    """
    if not _jax_available():
        raise ImportError("JAX is not installed.")
    import jax
    import jax.numpy as jnp

    # v4.13.0 (audit L2): unify on the library-wide default dtype rather
    # than reading ``jax.config.jax_enable_x64``.  Pre-fix users who set
    # ``set_default_complex_dtype(np.complex128)`` could still hit
    # complex64 here when ``jax_enable_x64`` was left at its default
    # False -- inconsistent with the NumPy-side ``apply_real_lens``.
    from ..propagators.propagation import (
        _resolve_jax_complex_dtype,
        _resolve_jax_real_dtype,
        get_default_complex_dtype,
    )
    cdtype = _resolve_jax_complex_dtype()
    rdtype = _resolve_jax_real_dtype()
    target_c = np.dtype(get_default_complex_dtype()).type
    target_r = (np.float32
                 if np.dtype(get_default_complex_dtype()) == np.dtype(np.complex64)
                 else np.float64)

    def _cb_full(E_in_arr):
        E_in_np = np.asarray(E_in_arr)
        E_an_np, phase_pw_np = _amp_callback_numpy(
            E_in_np, lens_prescription, wavelength, float(dx),
            preserve_input_phase, bandlimit)
        return (np.asarray(E_an_np, dtype=target_c),
                np.asarray(phase_pw_np, dtype=target_r))

    def _cb_amp_only(E_in_arr):
        # Tangent re-application: only need the analytic lens output
        # (the plane-wave reference phase is independent of E_in, so
        # its tangent is zero -- skip the second apply_real_lens call).
        E_in_np = np.asarray(E_in_arr)
        E_an_np = apply_real_lens(
            E_in_np, prescription=lens_prescription, wavelength=wavelength, dx=float(dx),
            bandlimit=bandlimit)
        return np.asarray(E_an_np, dtype=target_c)

    @jax.custom_jvp
    def _amp(E_in_):
        return jax.pure_callback(
            _cb_full,
            (jax.ShapeDtypeStruct(E_in_.shape, cdtype),
             jax.ShapeDtypeStruct(E_in_.shape, rdtype)),
            E_in_,
        )

    @_amp.defjvp
    def _amp_jvp(primals, tangents):
        E_in_, = primals
        dE_in_, = tangents
        primal_out = _amp(E_in_)
        # apply_real_lens is linear in E_in: T(E + dE) = T(E) + T(dE).
        # The tangent of the analytic field is T(dE_in).  The tangent
        # of the plane-wave reference phase is zero (independent of
        # E_in).
        dE_an = jax.pure_callback(
            _cb_amp_only,
            jax.ShapeDtypeStruct(dE_in_.shape, cdtype),
            dE_in_,
        )
        zeros_phase = jnp.zeros(dE_in_.shape, dtype=rdtype)
        return primal_out, (dE_an, zeros_phase)

    return _amp(E_in)


# v5.17 audit (P3-04): a second, identical ``_jax_available`` definition
# lived here and silently shadowed the module-top one; removed -- the
# single definition near the top of the module is now the only binding.


def apply_real_lens_traced_jax(
    E_in: Any,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    ray_subsample: int = 8,
    cheb_order: int = 10,
    newton_iters: int = 12,
    amplitude: str = 'input',
    bandlimit: bool = True,
    radii: Any = None,
    conics: Any = None,
    thicknesses: Any = None,
) -> Any:
    """JAX-traceable per-pixel ray-traced apply_real_lens.

    Builds a phase screen from the per-pixel ray-traced OPL through
    ``lens_prescription`` and applies it to ``E_in``::

        E_out(x, y) = E_in(x, y) * mask(x, y) * exp(i k0 OPD(x, y))

    where ``OPD`` is computed by launching collimated rays on a coarse
    entrance grid (via :func:`trace_jax`), referencing OPL to the
    on-axis launch, fitting a Chebyshev tensor-product polynomial to
    the entrance->exit coordinate map and the OPL, then inverting the
    forward map at each wave-grid pixel via Newton's method.  ``mask``
    is the system aperture (or full grid if no ``aperture_diameter``)
    intersected with the ray-coverage region the Newton inversion
    converged on.

    Differs from :func:`apply_real_lens_traced` (the NumPy version) in
    that the amplitude evolution does NOT come from a separate
    wave-optics propagation through the analytic lens model; the lens
    is treated as a thin OPD phase screen on top of ``E_in``.  This is
    the same approximation used by :func:`apply_thin_lens` (just with
    a ray-traced rather than analytic OPL) and the right model for
    gradient-based lens design where the OPD is the dominant signal.
    For diffraction-amplitude-accurate forward simulation, use the
    NumPy version.

    Parameters
    ----------
    E_in : (N, N) complex array
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
    dx : float
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.  Accepted for API
        symmetry with the rest of the lens family; the JAX traced
        ray-subsample / Chebyshev fit / Newton inversion paths
        currently require ``dy == dx`` and will raise otherwise.  Use
        :func:`apply_real_lens` (NumPy) for anamorphic grids -- it
        honours ``dy != dx`` end-to-end.  Added in v4.13.2 (audit
        P1-NEW-E) to close the JAX-side gap left by the v4.13.0 L3
        sweep.
    ray_subsample : int, default 8
        Coarse-grid subsampling factor for the entrance ray launch.
        Identical meaning to the NumPy version.
    cheb_order : int, default 10
        Total-degree of the 2-D Chebyshev fit used for the entrance->exit
        map and the OPL surface.  Order 10 (66 terms) gets sub-nm fit
        residual on typical refractive lenses; bump higher for
        cemented multi-element systems if needed.
    newton_iters : int, default 12
        Fixed Newton iteration count.  12 reaches sub-nm OPD residual
        for typical refractive lenses.
    amplitude : ``'input'`` (default) / ``'analytic'``
        ``'input'`` returns ``E_in * mask * exp(i k0 OPD)`` -- the
        thin-OPD-screen treatment.  Pure JAX, full ``jax.grad`` flow
        through both ``E_in`` and the OPD phase.
        ``'analytic'`` matches the NumPy version's amplitude leg by
        running :func:`apply_real_lens` via a callback (linearised
        via ``jax.custom_jvp``).  Gradient flows through ``E_in`` but
        the path is slower and only available when JAX is running on
        the host (``jax.pure_callback`` does not run inside
        ``jax.jit(..., backend='gpu')``).
    bandlimit : bool, default True
        Forwarded to the ASM step inside :func:`apply_real_lens` when
        ``amplitude='analytic'`` (ignored otherwise).

    Returns
    -------
    E_out : (N, N) complex JAX array

    ``radii`` / ``conics`` / ``thicknesses`` (v5.21, optional JAX arrays)
    enable ``jax.grad`` w.r.t. the prescription's GEOMETRY -- the lever for
    gradient-based lens *design* on the accurate ray-traced OPD (not just the
    field).  When any is given the trace runs through
    :func:`~lumenairy.raytrace.jax_trace.trace_jax_with_params` (which exposes
    radius / conic / thickness as differentiable pytree leaves) and the
    Newton-inversion initial guess is taken tracer-safe, so the OPD -- and hence
    ``E_out`` -- is differentiable end-to-end in those parameters.  ``None``
    (default) keeps the fully static, byte-identical trace.  Requires
    ``amplitude='input'`` (the OPD-screen path); the analytic-amplitude callback
    does not carry a prescription gradient.

    Notes
    -----
    * With ``radii`` / ``conics`` / ``thicknesses`` left ``None`` the
      prescription dict is treated as a static argument by JAX (gradients w.r.t.
      geometry are off) -- differentiate w.r.t. ``E_in``.  Pass those arrays to
      turn on the geometry gradient (see above).
    * Multi-process Newton parallelism, GPU CuPy paths, and tilt-aware
      rays from the NumPy version are not replicated; JAX's vmap+JIT
      replaces the first, JAX's GPU backend the second, and the
      tilt-aware launch falls naturally out of the OPD-screen model
      (``E_in``'s phase is preserved through the multiply).
    * The v5.17.0 row-band memory mode (``sag_chunk_rows``, the NumPy
      default at N >= 4096) is NOT replicated either: this function
      allocates fully monolithic ``(N, N)`` wave grids and phase
      screens, plus an ``(cheb_order + 1, N, N)`` Chebyshev stack
      inside the Newton loop, and reverse-mode autodiff additionally
      retains forward residuals for the tape.
      :func:`lumenairy.memory.estimate_lens_memory` does NOT model
      this JAX path.  For memory-bound large grids use the NumPy
      :func:`apply_real_lens_traced` (row-banded by default), or
      budget for the full monolithic cost here.  (The
      ``amplitude='analytic'`` callback leg runs the NumPy
      :func:`apply_real_lens` and does inherit its row-band default.)
    """
    if not _jax_available():
        raise ImportError(
            "JAX is not installed; install with `pip install jax`")
    import jax.numpy as jnp

    from ..raytrace.jax_trace import make_jax_ray_state, trace_jax

    # v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard
    # from ``apply_real_lens_traced``.  Pre-fix a hand-built prescription
    # with ``surfaces[i]['is_mirror']=True`` would slip past, and the
    # ray-traced OPD leg would silently treat the mirror as a refractor
    # with the wrong sign.
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
            f"apply_real_lens_traced_jax: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens_traced_jax "
            f"only walks refracting surfaces.  Running this "
            f"prescription as-is would silently treat the mirror as "
            f"a refractor (wrong sign / wrong focusing phase) and "
            f"propagate along the unfolded-equivalent axis.  Use "
            f"the per-segment trace + apply_mirror pattern for "
            f"folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens_traced_jax (each segment) with "
            f"apply_mirror (each fold).  See Guide-Folded-Designs "
            f"section 'Wave-optics through a fold'.")

    # Local alias preserves the legacy name used throughout this
    # function body (sprawling references; cheaper to rebind than
    # rename everywhere).
    lens_prescription = prescription

    E_in = jnp.asarray(E_in)
    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError("apply_real_lens_traced_jax requires a square grid")
    N = int(Nx)

    # v4.13.2 (audit P1-NEW-E): the JAX twin's ray-subsample +
    # Chebyshev tensor-product fit + Newton inversion paths all
    # assume an isotropic square grid (dx == dy).  Enforce the same
    # contract the NumPy ``apply_real_lens_traced`` documents so
    # anamorphic dy is rejected loudly rather than silently squared
    # to dx.
    if dy is None:
        dy = dx
    if abs(float(dy) - float(dx)) > 1e-15 * max(abs(float(dx)), 1.0):
        raise ValueError(
            "apply_real_lens_traced_jax: dy != dx not supported on the "
            f"JAX path (got dx={dx!r}, dy={dy!r}); use the NumPy variant "
            "apply_real_lens for anamorphic grids.")

    aperture = lens_prescription.get('aperture_diameter')
    pres_no_ap = dict(lens_prescription)
    pres_no_ap.pop('aperture_diameter', None)

    # ---- Entrance launch grid ---------------------------------------
    if aperture is not None:
        # Stay just inside the physical aperture: rays launched beyond
        # the aperture often miss the surface caps (glass would have
        # negative edge thickness) which the JAX trace cannot detect
        # the way the NumPy trace does.  This loses the marginal-pixel
        # over-margin that the NumPy version's 1.5x launch radius
        # provided, but those pixels are outside the aperture mask
        # anyway and are zeroed before return.
        launch_radius = 0.5 * float(aperture) * 1.02
    else:
        launch_radius = 0.5 * N * float(dx)
    sub = max(1, int(ray_subsample))
    n_launch = max(8, int(2 * launch_radius / (float(dx) * sub)))
    if n_launch % 2 == 0:
        n_launch += 1
    xs_in = jnp.linspace(-launch_radius, launch_radius, n_launch)
    Xs_in, Ys_in = jnp.meshgrid(xs_in, xs_in, indexing='ij')
    h_x = Xs_in.ravel()
    h_y = Ys_in.ravel()

    state = make_jax_ray_state(
        x=h_x, y=h_y, z=jnp.zeros_like(h_x),
        L=jnp.zeros_like(h_x), M=jnp.zeros_like(h_x),
        N=jnp.ones_like(h_x),
    )
    # v5.21: differentiable-geometry path -- when radii/conics/thicknesses are
    # supplied as JAX arrays, trace through trace_jax_with_params so jax.grad
    # flows into the prescription's geometry (the OPD -> E_out gradient used for
    # gradient-based lens design).  None -> the static, byte-identical trace.
    _diff_geom = (radii is not None or conics is not None
                  or thicknesses is not None)
    if _diff_geom:
        if amplitude != 'input':
            raise ValueError(
                "apply_real_lens_traced_jax: differentiable radii/conics/"
                "thicknesses require amplitude='input' (the OPD-screen path); "
                "the analytic-amplitude callback carries no prescription "
                "gradient.")
        from ..raytrace.jax_trace import trace_jax_with_params
        final = trace_jax_with_params(
            state, pres_no_ap, wavelength,
            radii=radii, conics=conics, thicknesses=thicknesses)
    else:
        final = trace_jax(state, pres_no_ap, wavelength)

    # ---- Exit-vertex correction -------------------------------------
    surfaces_raw = pres_no_ap.get('surfaces', [])
    n_exit = float(get_glass_index(
        surfaces_raw[-1].get('glass_after', 'air'), wavelength))
    eps = 1e-30
    safe_N = jnp.where(jnp.abs(final.N) > eps, final.N, eps)
    t_to_vertex = jnp.where(final.alive, -final.z / safe_N, 0.0)
    final_x = final.x + final.L * t_to_vertex
    final_y = final.y + final.M * t_to_vertex
    final_opd = final.opd + n_exit * t_to_vertex

    x_out_grid = final_x.reshape(n_launch, n_launch)
    y_out_grid = final_y.reshape(n_launch, n_launch)
    opl_grid = final_opd.reshape(n_launch, n_launch)
    # Reference OPL to on-axis (n_launch is odd by construction).
    i_axis = n_launch // 2
    opl_grid = opl_grid - opl_grid[i_axis, i_axis]

    # Replace dead-ray entries with NaN so the fit's robust-to-NaN path
    # drops them.
    alive_grid = final.alive.reshape(n_launch, n_launch)
    x_out_grid = jnp.where(alive_grid, x_out_grid, jnp.nan)
    y_out_grid = jnp.where(alive_grid, y_out_grid, jnp.nan)
    opl_grid = jnp.where(alive_grid, opl_grid, jnp.nan)

    # ---- Chebyshev fits ---------------------------------------------
    Sx_coeffs, K1, K2, xmin, xmax, ymin, ymax = _cheb_fit_2d_jax(
        xs_in, xs_in, x_out_grid, cheb_order)
    Sy_coeffs, _, _, _, _, _, _ = _cheb_fit_2d_jax(
        xs_in, xs_in, y_out_grid, cheb_order)
    So_coeffs, _, _, _, _, _, _ = _cheb_fit_2d_jax(
        xs_in, xs_in, opl_grid, cheb_order)

    # ---- Newton inversion of (Sx, Sy) on the wave grid --------------
    x_wave = (jnp.arange(N) - N / 2) * float(dx)
    # v5.4.6 (audit F-7): 'xy' indexing so the wave-grid axis order matches
    # the library's image-like (y, x) field layout (E_in axis 0 = y) and the
    # NumPy traced reference.  With 'ij' the phase screen is transposed vs
    # E_in for non-symmetric prescriptions -- latent today (the asymmetric
    # JAX trace builder raises NotImplementedError) but a live autodiff bug
    # once biconic/decenter JAX trace lands.  Strict no-op for the
    # rotationally symmetric square grids the current suite exercises.
    Xw, Yw = jnp.meshgrid(x_wave, x_wave, indexing='xy')

    # Initial-guess paraxial magnification: use the central finite-
    # difference slope of the forward map as in the NumPy version.
    di = max(1, n_launch // 8)
    dx_in = 2.0 * float(launch_radius) / (n_launch - 1)
    if _diff_geom:
        # tracer-safe: keep the initial guess as JAX scalars and stop_gradient
        # (the Newton root is independent of the starting point, so the
        # initial-guess magnification needs no gradient -- and x_out_grid is a
        # tracer here, so float() would raise a ConcretizationError).
        import jax as _jax
        _dxox = (x_out_grid[i_axis + di, i_axis]
                 - x_out_grid[i_axis - di, i_axis]) / (2.0 * di * dx_in)
        _dyoy = (y_out_grid[i_axis, i_axis + di]
                 - y_out_grid[i_axis, i_axis - di]) / (2.0 * di * dx_in)
        inv_M_x = _jax.lax.stop_gradient(
            jnp.where(jnp.abs(_dxox) > 1e-9, 1.0 / _dxox, 1.10))
        inv_M_y = _jax.lax.stop_gradient(
            jnp.where(jnp.abs(_dyoy) > 1e-9, 1.0 / _dyoy, 1.10))
    else:
        dx_out_x = float(x_out_grid[i_axis + di, i_axis] -
                          x_out_grid[i_axis - di, i_axis]) / (2.0 * di * dx_in)
        dy_out_y = float(y_out_grid[i_axis, i_axis + di] -
                          y_out_grid[i_axis, i_axis - di]) / (2.0 * di * dx_in)
        inv_M_x = 1.0 / dx_out_x if abs(dx_out_x) > 1e-9 else 1.10
        inv_M_y = 1.0 / dy_out_y if abs(dy_out_y) > 1e-9 else 1.10
    bound = float(launch_radius) * 0.999

    xe, ye = _newton_invert_2d_jax(
        Xw, Yw, Sx_coeffs, Sy_coeffs, K1, K2, K1, K2,
        xmin, xmax, ymin, ymax, cheb_order,
        inv_M_x, inv_M_y, bound, n_iter=newton_iters,
    )
    opl_map, _, _ = _cheb_eval_value_grad_jax(
        So_coeffs, K1, K2, xmin, xmax, ymin, ymax,
        xe, ye, cheb_order)

    # Mark out-of-domain pixels (Newton landed outside fit support)
    inside = (xe * xe + ye * ye) < (float(launch_radius) * 0.99) ** 2
    opl_map = jnp.where(inside, opl_map, jnp.nan)

    # ---- Combine OPL with amplitude ---------------------------------
    # v4.13.0 (audit L2): unify on the library-wide default dtype
    # (``set_default_complex_dtype``) rather than reading the JAX
    # global ``jax_enable_x64``.  Two different knobs for the same
    # decision is exactly the divergence audit L2 flagged.
    # v4.14.0: pass ``E_in.dtype`` so the input dtype is honoured.
    # Pre-v4.14 the resolver returned the library default, silently
    # upcasting a complex64 input to complex128 whenever
    # ``jax_enable_x64=True``.  Caught by parametrized dispatcher pin
    # in v4.14.0 Agent 6.
    from ..propagators.propagation import (
        _resolve_jax_complex_dtype,
        _resolve_jax_real_dtype,
    )
    cdtype = _resolve_jax_complex_dtype(E_in.dtype)
    rdtype = _resolve_jax_real_dtype(E_in.dtype)
    k0 = 2.0 * jnp.pi / float(wavelength)
    valid = jnp.isfinite(opl_map)
    phase = jnp.where(valid, k0 * opl_map, 0.0)
    phase_screen = jnp.exp(1j * phase).astype(cdtype)

    if amplitude == 'input':
        E_out = E_in.astype(cdtype) * phase_screen
    elif amplitude == 'analytic':
        E_analytic, _ = _amp_callback_jax_linear(
            E_in, lens_prescription, wavelength, dx,
            preserve_input_phase=False, bandlimit=bandlimit,
        )
        E_out = (jnp.abs(E_analytic).astype(rdtype) *
                 phase_screen).astype(cdtype)
    else:
        raise ValueError(
            f"amplitude must be 'input' or 'analytic', got {amplitude!r}")

    E_out = jnp.where(valid, E_out, jnp.zeros_like(E_out))
    if aperture is not None:
        ap = float(aperture) / 2.0
        E_out = jnp.where(Xw * Xw + Yw * Yw <= ap * ap,
                          E_out, jnp.zeros_like(E_out))
    return E_out


def apply_real_lens_maslov_jax(
    E_in: Any,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    ray_subsample: int = 8,
    cheb_order: int = 10,
    newton_iters: int = 12,
    amplitude: str = 'input',
    bandlimit: bool = True,
) -> Any:
    """JAX-traceable ray-traced lens propagator with a det(J)-sign-flip
    caustic phase counter.

    Same structure as :func:`apply_real_lens_traced_jax` -- a thin-OPD
    geometric phase screen built from a Chebyshev fit of the traced
    ray map -- plus a Maslov-style correction: when the forward map's
    Jacobian determinant changes sign along the straight path from the
    axis to each exit pixel, a -pi/2 phase shift is accumulated per
    crossing.

    .. note::
       Despite the historical name this is **not** the phase-space
       *diffraction integral* of the NumPy :func:`apply_real_lens_maslov`
       (whose amplitude is the integral itself); it is a thin-OPD geometric
       phase *screen* plus a Maslov / Gouy phase term.  What it now gets right
       (v5.21) is the **caustic phase**: the term is
       ``-pi/2 * (number of negative eigenvalues of the forward ray-map
       Jacobian J at the pixel)`` -- the Morse / Maslov index at the
       observation point.  This counts **even-multiplicity caustics** an axial
       point focus flips BOTH eigenvalues -> index 2 -> the ``-pi`` Gouy shift)
       as well as off-axis fold caustics from spherical aberration (one
       eigenvalue negative -> ``-pi/2``), superseding the old ``det(J)``
       sign-flip radial scan that could not see an axial focus at all and
       dropped the ``-pi`` past focus.  The **amplitude** is still the input /
       traced screen, not the diffraction integral, so for quantitatively
       correct focal-region *intensity* use the NumPy phase-space integrator;
       this path is for JAX autodiff with the correct focal-region *phase*.

    Parameters and gradient flow are identical to
    :func:`apply_real_lens_traced_jax`.  Differentiable through
    ``E_in``; lens prescription is treated as a static closure.
    The Notes there apply here too -- in particular, the v5.17.0
    row-band memory mode (``sag_chunk_rows``) is NOT replicated:
    allocation is fully monolithic ``(N, N)`` (plus the
    ``(cheb_order + 1, N, N)`` Chebyshev stack), so for memory-bound
    large grids use the NumPy path or budget for the monolithic cost.

    The correction actually implemented is:

        m(xe, ye) = #{det(J) sign flips along the straight path
                       from on-axis to (xe, ye), 32 samples}
        phase_maslov = -pi/2 * m(xe, ye)

    evaluated with a ``jax.lax.fori_loop`` over the Chebyshev-fitted
    forward map for JAX traceability.
    """
    if not _jax_available():
        raise ImportError(
            "JAX is not installed; install with `pip install jax`")
    import jax.numpy as jnp

    from ..raytrace.jax_trace import make_jax_ray_state, trace_jax

    # v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard
    # from ``apply_real_lens_traced``.
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
            f"apply_real_lens_maslov_jax: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens_maslov_jax "
            f"only walks refracting surfaces.  Running this "
            f"prescription as-is would silently treat the mirror as "
            f"a refractor (wrong sign / wrong focusing phase) and "
            f"propagate along the unfolded-equivalent axis.  Use "
            f"the per-segment trace + apply_mirror pattern for "
            f"folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens_maslov_jax (each segment) with "
            f"apply_mirror (each fold).  See Guide-Folded-Designs "
            f"section 'Wave-optics through a fold'.")

    # Local alias preserves the legacy name used in this function body.
    lens_prescription = prescription

    E_in = jnp.asarray(E_in)
    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError("apply_real_lens_maslov_jax requires a square grid")
    N = int(Nx)

    # v4.13.2 (audit P1-NEW-E): same square-grid constraint as the
    # traced twin -- the Chebyshev fit + Newton inversion + Maslov
    # radial sampler all assume isotropic dx == dy.
    if dy is None:
        dy = dx
    if abs(float(dy) - float(dx)) > 1e-15 * max(abs(float(dx)), 1.0):
        raise ValueError(
            "apply_real_lens_maslov_jax: dy != dx not supported on the "
            f"JAX path (got dx={dx!r}, dy={dy!r}); use the NumPy variant "
            "apply_real_lens for anamorphic grids.")

    aperture = lens_prescription.get('aperture_diameter')
    pres_no_ap = dict(lens_prescription)
    pres_no_ap.pop('aperture_diameter', None)

    if aperture is not None:
        # Stay just inside the physical aperture: rays launched beyond
        # the aperture often miss the surface caps (glass would have
        # negative edge thickness) which the JAX trace cannot detect
        # the way the NumPy trace does.  This loses the marginal-pixel
        # over-margin that the NumPy version's 1.5x launch radius
        # provided, but those pixels are outside the aperture mask
        # anyway and are zeroed before return.
        launch_radius = 0.5 * float(aperture) * 1.02
    else:
        launch_radius = 0.5 * N * float(dx)
    sub = max(1, int(ray_subsample))
    n_launch = max(8, int(2 * launch_radius / (float(dx) * sub)))
    if n_launch % 2 == 0:
        n_launch += 1
    xs_in = jnp.linspace(-launch_radius, launch_radius, n_launch)
    Xs_in, Ys_in = jnp.meshgrid(xs_in, xs_in, indexing='ij')
    h_x = Xs_in.ravel()
    h_y = Ys_in.ravel()

    state = make_jax_ray_state(
        x=h_x, y=h_y, z=jnp.zeros_like(h_x),
        L=jnp.zeros_like(h_x), M=jnp.zeros_like(h_x),
        N=jnp.ones_like(h_x),
    )
    final = trace_jax(state, pres_no_ap, wavelength)

    surfaces_raw = pres_no_ap.get('surfaces', [])
    n_exit = float(get_glass_index(
        surfaces_raw[-1].get('glass_after', 'air'), wavelength))
    eps = 1e-30
    safe_N = jnp.where(jnp.abs(final.N) > eps, final.N, eps)
    t_to_vertex = jnp.where(final.alive, -final.z / safe_N, 0.0)
    final_x = final.x + final.L * t_to_vertex
    final_y = final.y + final.M * t_to_vertex
    final_opd = final.opd + n_exit * t_to_vertex

    x_out_grid = final_x.reshape(n_launch, n_launch)
    y_out_grid = final_y.reshape(n_launch, n_launch)
    opl_grid = final_opd.reshape(n_launch, n_launch)
    i_axis = n_launch // 2
    opl_grid = opl_grid - opl_grid[i_axis, i_axis]

    alive_grid = final.alive.reshape(n_launch, n_launch)
    x_out_grid = jnp.where(alive_grid, x_out_grid, jnp.nan)
    y_out_grid = jnp.where(alive_grid, y_out_grid, jnp.nan)
    opl_grid = jnp.where(alive_grid, opl_grid, jnp.nan)

    Sx_coeffs, K1, K2, xmin, xmax, ymin, ymax = _cheb_fit_2d_jax(
        xs_in, xs_in, x_out_grid, cheb_order)
    Sy_coeffs, *_ = _cheb_fit_2d_jax(
        xs_in, xs_in, y_out_grid, cheb_order)
    So_coeffs, *_ = _cheb_fit_2d_jax(
        xs_in, xs_in, opl_grid, cheb_order)

    x_wave = (jnp.arange(N) - N / 2) * float(dx)
    # v5.4.6 (audit F-7): 'xy' indexing so the wave-grid axis order matches
    # the library's image-like (y, x) field layout (E_in axis 0 = y) and the
    # NumPy traced reference.  With 'ij' the phase screen is transposed vs
    # E_in for non-symmetric prescriptions -- latent today (the asymmetric
    # JAX trace builder raises NotImplementedError) but a live autodiff bug
    # once biconic/decenter JAX trace lands.  Strict no-op for the
    # rotationally symmetric square grids the current suite exercises.
    Xw, Yw = jnp.meshgrid(x_wave, x_wave, indexing='xy')

    di = max(1, n_launch // 8)
    dx_in = 2.0 * float(launch_radius) / (n_launch - 1)
    dx_out_x = float(x_out_grid[i_axis + di, i_axis] -
                      x_out_grid[i_axis - di, i_axis]) / (2.0 * di * dx_in)
    dy_out_y = float(y_out_grid[i_axis, i_axis + di] -
                      y_out_grid[i_axis, i_axis - di]) / (2.0 * di * dx_in)
    inv_M_x = 1.0 / dx_out_x if abs(dx_out_x) > 1e-9 else 1.10
    inv_M_y = 1.0 / dy_out_y if abs(dy_out_y) > 1e-9 else 1.10
    bound = float(launch_radius) * 0.999

    xe, ye = _newton_invert_2d_jax(
        Xw, Yw, Sx_coeffs, Sy_coeffs, K1, K2, K1, K2,
        xmin, xmax, ymin, ymax, cheb_order,
        inv_M_x, inv_M_y, bound, n_iter=newton_iters,
    )
    opl_map, _, _ = _cheb_eval_value_grad_jax(
        So_coeffs, K1, K2, xmin, xmax, ymin, ymax,
        xe, ye, cheb_order)

    inside = (xe * xe + ye * ye) < (float(launch_radius) * 0.99) ** 2
    opl_map = jnp.where(inside, opl_map, jnp.nan)

    # ---- Maslov index (Morse index) at each pixel ------------------
    # v5.21: the caustic phase is -pi/2 * (number of negative eigenvalues of the
    # forward ray-map Jacobian J at the pixel) -- the Morse / Maslov index at
    # the OBSERVATION point.  This supersedes the previous det(J)-sign-flip
    # radial scan, which (a) could not see an AXIAL focus at all (it is not a
    # radial feature: every pixel is 'past' the on-axis caustic, so a scan from
    # the axis crosses nothing) and (b) missed EVEN-multiplicity caustics (an
    # axial point focus flips BOTH eigenvalues, so det(J) does not change sign
    # -- the old counter stayed 0 and dropped the -pi Gouy shift).  Counting the
    # negative real eigenvalues directly gives index 2 past an axial focus
    # (-> -pi) and index 1 past an off-axis fold caustic (spherical aberration,
    # one eigenvalue negative -> -pi/2), matching the NumPy phase-space Maslov.
    _, jxx, jxy = _cheb_eval_value_grad_jax(
        Sx_coeffs, K1, K2, xmin, xmax, ymin, ymax, xe, ye, cheb_order)
    _, jyx, jyy = _cheb_eval_value_grad_jax(
        Sy_coeffs, K1, K2, xmin, xmax, ymin, ymax, xe, ye, cheb_order)
    _tr = jxx + jyy
    _det = jxx * jyy - jxy * jyx
    _disc = _tr * _tr - 4.0 * _det
    _sq = jnp.sqrt(jnp.maximum(_disc, 0.0))
    _lam1 = 0.5 * (_tr + _sq)
    _lam2 = 0.5 * (_tr - _sq)
    _real = _disc >= 0.0            # complex eigenvalues -> rotating, no caustic
    n_neg = ((_real & (_lam1 < 0.0)).astype(opl_map.dtype)
             + (_real & (_lam2 < 0.0)).astype(opl_map.dtype))
    phase_maslov = -0.5 * jnp.pi * n_neg

    # ---- Combine OPL + Maslov phase with amplitude ------------------
    # v4.13.0 (audit L2): unify on the library-wide default dtype.
    # v4.14.0: pass ``E_in.dtype`` so the input dtype is honoured
    # (caught by parametrized dispatcher pin in v4.14.0 Agent 6).
    from ..propagators.propagation import (
        _resolve_jax_complex_dtype,
        _resolve_jax_real_dtype,
    )
    cdtype = _resolve_jax_complex_dtype(E_in.dtype)
    rdtype = _resolve_jax_real_dtype(E_in.dtype)
    k0 = 2.0 * jnp.pi / float(wavelength)
    valid = jnp.isfinite(opl_map)
    phase = jnp.where(valid, k0 * opl_map + phase_maslov, 0.0)
    phase_screen = jnp.exp(1j * phase).astype(cdtype)

    if amplitude == 'input':
        E_out = E_in.astype(cdtype) * phase_screen
    elif amplitude == 'analytic':
        E_analytic, _ = _amp_callback_jax_linear(
            E_in, lens_prescription, wavelength, dx,
            preserve_input_phase=False, bandlimit=bandlimit,
        )
        E_out = (jnp.abs(E_analytic).astype(rdtype) *
                 phase_screen).astype(cdtype)
    else:
        raise ValueError(
            f"amplitude must be 'input' or 'analytic', got {amplitude!r}")

    E_out = jnp.where(valid, E_out, jnp.zeros_like(E_out))
    if aperture is not None:
        ap = float(aperture) / 2.0
        E_out = jnp.where(Xw * Xw + Yw * Yw <= ap * ap,
                          E_out, jnp.zeros_like(E_out))
    return E_out



__all__ = [
    'apply_real_lens_traced_jax',
    'apply_real_lens_maslov_jax',
]
