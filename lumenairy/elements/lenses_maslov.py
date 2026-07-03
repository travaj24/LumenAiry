"""Maslov-method (phase-space asymptotic) propagator through a thick-lens
prescription.

Originally inlined in :mod:`lumenairy.elements.lenses` (and before that
in a now-removed top-level ``lens_maslov.py`` module).  Split out into
its own file in v3.5.5 to reduce ``lenses.py`` bloat.  Imports remain
backwards-compatible -- ``apply_real_lens_maslov`` is still re-exported
from :mod:`lumenairy.elements.lenses` for callers that import it from
there.

This module owns the Maslov NumPy implementation (the JAX variant
``apply_real_lens_maslov_jax`` lives in
:mod:`lumenairy.elements._lens_jax` because it shares the ``_cheb_*``
Chebyshev evaluators there with ``apply_real_lens_traced_jax``; it is
re-exported from :mod:`lumenairy.elements.lenses`).

Author: Andrew Traverso
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .. import raytrace as rt
from .._math.chebyshev import (
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,
)
from .._math.chebyshev import (
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,
)

# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# The three Chebyshev Vandermonde helpers moved from
# ``lumenairy.elements.lenses`` to ``lumenairy._math.chebyshev``.  We
# import the new public names and bind them to the legacy
# underscore-prefixed locals so the rest of this module's call sites
# (~10 references) keep working unchanged.
from .._math.chebyshev import (
    chebyshev_vandermonde as _chebyshev_vandermonde,
)
from ..progress import call_progress

# Other shared helpers still live in lenses.py.
from .lenses import (
    NUMEXPR_AVAILABLE,
    _ensure_numexpr_loaded,
    _fit_normaliser,
    _multi_indices_total_degree,
    _warn_if_aperture_exceeds_grid,
)


def apply_real_lens_maslov(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    ray_field_samples: int = 16,
    ray_pupil_samples: int = 16,
    poly_order: int = 4,
    n_v2: int = 32,
    output_subsample: int = 1,
    extract_linear_phase: bool = True,
    chunk_v2: int = 64,
    use_numexpr: Optional[bool] = None,
    integration_method: str = 'quadrature',
    stationary_newton_iter: int = 12,
    stationary_newton_tol: float = 1e-10,
    local_n_samples: int = 8,
    local_window_sigma: float = 3.0,
    collimated_input: bool = False,
    input_na: Optional[float] = None,
    normalize_output: str = 'power',
    verbose: bool = False,
    progress: Optional[Any] = None,
) -> np.ndarray:
    """
    Phase-space / Maslov propagator through a thick-lens prescription.

    See Also
    --------
    apply_real_lens :
        Analytic split-step thin-element model.  Default fast path
        when the output plane is well away from any caustic and
        autodiff gradients aren't required.
    apply_real_lens_traced :
        Per-pixel ray-traced OPL + wave-optics amplitude envelope.
        Achieves sub-nm OPD on cemented doublets, but is **not**
        differentiable (uses Newton inversion of the
        entrance->exit map) and breaks down at caustics where the
        per-pixel ray map becomes multi-valued.
    apply_real_lens_maslov_jax :
        JAX-traced twin of this function for autodiff /
        gradient-based design optimisation.

    Quick decision guide
    --------------------
    * Default / fast wave model -> ``apply_real_lens``.
    * Sub-nm OPD on cemented doublets / multi-surface curved interfaces
      -> ``apply_real_lens_traced``.
    * Inside a JAX-autodiff design optimisation, or near a caustic
      -> ``apply_real_lens_maslov`` (this function) /
      ``apply_real_lens_maslov_jax``.

    Description
    -----------
    Traces a Chebyshev-node grid of rays from the entrance plane of
    ``lens_prescription`` to the exit plane, fits a 4-variable
    Chebyshev tensor-product polynomial to ``s1(s2, v2)`` and
    ``OPD(s2, v2)``, then evaluates the Maslov integral

        E(s2) = integral E_in(s1(s2, v2)) * exp(2 pi i OPD(s2, v2))
                          * |det(ds1/dv2)|  d^2 v2

    at each output pixel.  See the v3.4.x release notes (or the
    ``Phase-Space Asymptotic Propagator`` wiki page) for the full
    physics derivation and quadrature/stationary-phase trade-offs.

    Parameters mirror the inline-in-lenses.py predecessor exactly so
    no caller-side changes are required.

    The ``dy`` parameter is accepted for API symmetry with the rest
    of the lens family; the Maslov propagator's Chebyshev-tensor-
    product fit assumes square pixels and will raise if
    ``dy != dx``.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_maslov')

    # v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard
    # from ``apply_real_lens_traced``.  Pre-fix a hand-built prescription
    # with ``surfaces[i]['is_mirror']=True`` (or ``glass_after='MIRROR'``)
    # would slip past the shared ``_check_no_silent_fold_drop`` (which
    # only inspects ``prescription['elements']``), and the Maslov leg
    # would silently treat the mirror as a refractor with the wrong
    # sign.  Fail loudly with the same mirror-specific message as
    # ``apply_real_lens_traced``.
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
            f"apply_real_lens_maslov: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens_maslov "
            f"only walks refracting surfaces.  Running this "
            f"prescription as-is would silently treat the mirror as "
            f"a refractor (wrong sign / wrong focusing phase) and "
            f"propagate along the unfolded-equivalent axis.  Use "
            f"the per-segment trace + apply_mirror pattern for "
            f"folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens_maslov (each segment) with "
            f"apply_mirror (each fold).  See Guide-Folded-Designs "
            f"section 'Wave-optics through a fold'.")

    # Folded-design silent-drop guard: same as apply_real_lens.
    from ._lens_real import _check_no_silent_fold_drop
    _check_no_silent_fold_drop(
        prescription, fn_name='apply_real_lens_maslov')

    # Internal references keep the legacy local name to avoid a
    # sprawling rename across the function body.
    lens_prescription = prescription

    # Local references to numexpr (if available) -- the parent module
    # (lenses.py) holds the lazy module slot.
    from . import lenses as _lenses_module
    t0 = time.perf_counter()
    E_in = np.asarray(E_in)
    if E_in.ndim != 2 or E_in.shape[0] != E_in.shape[1]:
        raise ValueError(
            f"E_in must be square 2D, got shape {E_in.shape}")
    N = E_in.shape[0]

    if dy is None:
        dy = dx
    if abs(float(dy) - float(dx)) > 1e-15 * max(abs(float(dx)), 1.0):
        raise ValueError(
            "apply_real_lens_maslov currently requires square pixels "
            f"(dx == dy); got dx={dx!r}, dy={dy!r}.  Use apply_real_lens "
            "for anamorphic grids.")

    # Pre-flight grid vs prescription-aperture check.
    try:
        _warn_if_aperture_exceeds_grid(
            lens_prescription, N, dx, source='apply_real_lens_maslov')
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture-check failure is informational only; the
        # propagator still runs.
        pass

    def _progress(phase, frac, note=''):
        dt = time.perf_counter() - t0
        if progress is not None:
            # F3 (audit): emit the suite-standard (stage, fraction,
            # message) signature via call_progress instead of the old
            # bespoke keyword/4-positional protocol, which raised
            # TypeError on a standard (label, frac[, msg]) callback and
            # crashed the propagator mid-lens.  ``phase`` becomes the
            # stage label; the note + elapsed time fold into the message
            # so no information is lost.  call_progress swallows broken-
            # callback exceptions so a progress bar can never crash the run.
            msg = f'{note} ({dt:.1f}s)' if note else f'({dt:.1f}s)'
            call_progress(progress, phase, float(frac), msg)
        if verbose:
            print(f"  maslov {phase:>10s}  {frac*100:5.1f}%  "
                  f"({dt:6.1f}s) {note}", flush=True)

    # -----------------------------------------------------------------
    # Step 1: Trace rays on a Chebyshev-node (h, p) grid
    # -----------------------------------------------------------------
    _progress('trace', 0.0, 'building ray bundle')

    surfaces = rt.surfaces_from_prescription(lens_prescription)
    if not surfaces:
        raise ValueError("Lens prescription has no surfaces.")

    # 4.11.2: warn if a non-entrance or decentered stop is configured.
    # ``apply_real_lens`` honours ``stop_index`` and per-surface
    # ``decenter`` on the stop; the Maslov path traces a Chebyshev-node
    # ray bundle launched on a centred (h, p) grid scaled by the
    # entrance aperture, so a non-zero stop_index is silently moved to
    # the entrance.
    _stop_index = lens_prescription.get('stop_index')
    if _stop_index is not None and int(_stop_index) != 0:
        import warnings
        warnings.warn(
            f"apply_real_lens_maslov: prescription specifies "
            f"stop_index={_stop_index}, but the Maslov ray bundle is "
            "launched on a centred (h, p) Chebyshev grid scaled by the "
            "entrance aperture; the aperture stop is effectively "
            "applied at the entrance (index 0).  For physically-correct "
            "stop behaviour on a non-entrance stop, use apply_real_lens.",
            RuntimeWarning, stacklevel=2,
        )
    else:
        _surfs_chk = lens_prescription.get('surfaces') or []
        if _surfs_chk:
            _stop_surf_idx = int(_stop_index) if _stop_index is not None else 0
            if 0 <= _stop_surf_idx < len(_surfs_chk):
                _dec = _surfs_chk[_stop_surf_idx].get('decenter') or (0.0, 0.0)
                if _dec[0] != 0.0 or _dec[1] != 0.0:
                    import warnings
                    warnings.warn(
                        f"apply_real_lens_maslov: stop surface "
                        f"{_stop_surf_idx} has decenter={_dec}; the "
                        "Maslov ray bundle is launched on a centred "
                        "(h, p) grid and will not see the off-axis stop "
                        "correctly.  Use apply_real_lens for "
                        "decentered-stop systems.",
                        RuntimeWarning, stacklevel=2,
                    )

    aperture_m = lens_prescription.get('aperture_diameter', None)
    if aperture_m is None:
        sds = [s.semi_diameter for s in surfaces if np.isfinite(s.semi_diameter)]
        if sds:
            aperture_m = 2.0 * min(sds)
        else:
            aperture_m = N * dx * 0.5
    r_aperture = 0.5 * aperture_m

    def cheb_nodes(n):
        i = np.arange(n)
        return np.cos((i + 0.5) * np.pi / n)

    hx = cheb_nodes(ray_field_samples)
    hy = cheb_nodes(ray_field_samples)
    px = cheb_nodes(ray_pupil_samples)
    py = cheb_nodes(ray_pupil_samples)

    HX, HY, PX, PY = np.meshgrid(hx, hy, px, py, indexing='ij')
    HX = HX.ravel()
    HY = HY.ravel()
    PX = PX.ravel()
    PY = PY.ravel()

    keep = (PX**2 + PY**2) <= 1.0
    HX, HY, PX, PY = HX[keep], HY[keep], PX[keep], PY[keep]
    n_rays = len(HX)
    if n_rays < 1.5 * _count_multi_indices_4d(poly_order):
        raise ValueError(
            f"Only {n_rays} rays survived pupil masking; need at least "
            f"~{int(1.5 * _count_multi_indices_4d(poly_order))} "
            f"for a well-conditioned order-{poly_order} fit.")

    s1x = HX * r_aperture
    s1y = HY * r_aperture

    # N3 (audit): the pupil-direction chart must span BOTH the lens
    # acceptance NA and the INPUT field's angular content.  Sizing from
    # the lens EFL alone (the pre-fix na_proxy) drops any divergent /
    # tilted input source off the traced ray chart, so its wide-angle
    # rays are extrapolated or clip at |u_v2| = 1 -- silently dim / wrong
    # output at ANY resolution.  Split the sizing into a lens term and an
    # input term.
    if collimated_input:
        na_lens = 1e-5
    else:
        try:
            _M, _efl, _bfl, _ffl = rt.system_abcd_prescription(
                lens_prescription, wavelength)
            efl_abs = float(abs(_efl))
            if np.isfinite(efl_abs) and efl_abs > 0:
                na_lens = r_aperture / max(efl_abs, r_aperture * 10)
            else:
                lens_total_thickness = sum(s.thickness for s in surfaces)
                na_lens = r_aperture / max(lens_total_thickness,
                                           r_aperture * 10)
        except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                np.linalg.LinAlgError, IndexError, TypeError):
            # system_abcd_prescription failure -- fall back to a
            # thickness-based NA proxy (geometric heuristic).
            lens_total_thickness = sum(s.thickness for s in surfaces)
            na_lens = r_aperture / max(lens_total_thickness,
                                       r_aperture * 10)

    # Divergence NA of the input field: measured from the second moment
    # of its angular spectrum (a single FFT; direction cosine v =
    # wavelength * fx in the paraxial regime), unless the caller supplies
    # input_na explicitly (or the field is declared collimated).
    _na_meas = 0.0
    if not collimated_input:
        _F = np.fft.fft2(E_in)
        _P = np.abs(_F) ** 2
        del _F
        _fx = np.fft.fftfreq(N, d=dx)
        _FX, _FY = np.meshgrid(_fx, _fx, indexing='xy')
        _Ptot = float(_P.sum())
        if _Ptot > 0.0:
            _v2 = (wavelength ** 2) * (_FX ** 2 + _FY ** 2)
            _rms = float(np.sqrt(float((_v2 * _P).sum()) / _Ptot))
            _na_meas = 3.0 * _rms   # ~3-sigma coverage of the spectrum
            del _v2
        del _P, _FX, _FY, _fx
    if input_na is not None:
        na_input = float(input_na)
        # Explicit input_na must be a finite, non-negative direction cosine.
        # A NaN slips past the na_proxy>=1 clamp below (NaN comparisons are
        # False), reaching the trace as N_dir=NaN and dying with a
        # misleading "0 rays survived" TIR message (adversarial review) --
        # fail fast here with the real cause instead.
        if not (np.isfinite(na_input) and na_input >= 0.0):
            raise ValueError(
                f"apply_real_lens_maslov: input_na must be a finite, "
                f"non-negative number (an input-side NA / direction cosine); "
                f"got {input_na!r}.  Omit input_na to auto-size the pupil "
                f"chart from the field's angular spectrum.")
        # Coverage guard: warn if the caller under-specified input_na
        # relative to the measured angular spread (the field will clip).
        if (not collimated_input) and na_input < 0.7 * _na_meas:
            import warnings
            warnings.warn(
                f"apply_real_lens_maslov: input_na={na_input:.4f} is well "
                f"below the measured input angular spread "
                f"(~{_na_meas:.4f}); the pupil chart may not cover the "
                f"field and wide-angle content will be lost.  Omit "
                f"input_na to auto-size from the field.",
                RuntimeWarning, stacklevel=2)
    elif collimated_input:
        na_input = 0.0
    else:
        na_input = _na_meas

    # Chart spans the lens acceptance plus the input divergence.
    na_proxy = na_lens + na_input

    # Clamp to a physical direction cosine (< 1).  A speckled / hard-aperture
    # input can have a 3-sigma angular estimate na_input > 1 (measured
    # 1.3-4.1 on white-noise fields, adversarial review P2); leaving
    # na_proxy > 1 forces every pupil ray to v1x^2+v1y^2 > 1, so
    # N_dir = sqrt(max(1 - v1x^2 - v1y^2, 0)) = 0 and the whole chart is
    # grazing -> the wide-angle content it was meant to capture is dropped.
    # Cap just below unity and tell the caller the estimate is being
    # trusted only up to the horizon.  Use ``not (na_proxy < 1.0)`` rather
    # than ``na_proxy >= 1.0`` so a non-finite proxy (e.g. an inf leaking in
    # from na_lens) is also caught -- NaN would already have been rejected
    # for explicit input_na above, but this keeps N_dir strictly real.
    if not (na_proxy < 1.0):
        import warnings
        warnings.warn(
            f"apply_real_lens_maslov: NA proxy {na_proxy:.3f} (lens "
            f"{na_lens:.3f} + input {na_input:.3f}) exceeds 1; the input "
            f"angular-spread estimate is likely inflated by high-frequency / "
            f"aperture-edge content.  Clamping the pupil chart to NA=0.999 "
            f"(the physical horizon).  Pass input_na explicitly to size the "
            f"chart deliberately.",
            RuntimeWarning, stacklevel=2)
        na_proxy = 0.999

    if verbose:
        print(f"  NA_proxy = {na_proxy:.5f}  (lens {na_lens:.5f} + "
              f"input {na_input:.5f}; collimated_input={collimated_input})")

    v1x = PX * na_proxy
    v1y = PY * na_proxy
    N_dir = np.sqrt(np.maximum(1.0 - v1x**2 - v1y**2, 0.0))
    _progress('trace', 0.05, f'{n_rays} rays prepared')

    rays = rt.RayBundle(
        x=s1x.copy(), y=s1y.copy(), z=np.zeros_like(s1x),
        L=v1x.copy(), M=v1y.copy(), N=N_dir,
        wavelength=wavelength,
        alive=np.ones(n_rays, dtype=bool),
        opd=np.zeros(n_rays),
    )

    tr = rt.trace(rays, surfaces, wavelength)
    exit_rays = tr.image_rays
    alive = exit_rays.alive
    if alive.sum() < 1.5 * _count_multi_indices_4d(poly_order):
        raise ValueError(
            f"Only {alive.sum()}/{n_rays} rays survived the trace; "
            f"likely aperture / TIR issue.  Check prescription.")

    s2x = exit_rays.x[alive]
    s2y = exit_rays.y[alive]
    v2x = exit_rays.L[alive]
    v2y = exit_rays.M[alive]
    opd_m = exit_rays.opd[alive] - rays.opd[alive]
    opd_w = opd_m / wavelength
    s1x_live = s1x[alive]
    s1y_live = s1y[alive]
    _progress('trace', 0.15, f'{alive.sum()} alive rays; '
              f'OPD p-v = {opd_w.max()-opd_w.min():.3f} waves')

    # -----------------------------------------------------------------
    # Step 2: Normalise (s2, v2) to [-1, 1]^4 and fit Chebyshev polys
    # -----------------------------------------------------------------
    _progress('fit', 0.15, 'normalising inputs')
    s2x_c, s2x_h = _fit_normaliser(s2x)
    s2y_c, s2y_h = _fit_normaliser(s2y)
    v2x_c, v2x_h = _fit_normaliser(v2x)
    v2y_c, v2y_h = _fit_normaliser(v2y)

    u_s2x = (s2x - s2x_c) / s2x_h
    u_s2y = (s2y - s2y_c) / s2y_h
    u_v2x = (v2x - v2x_c) / v2x_h
    u_v2y = (v2y - v2y_c) / v2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, opd_w, rcond=None)
        opd_linear = X5 @ linear_coeffs
        opd_residual = opd_w - opd_linear
    else:
        opd_residual = opd_w.copy()

    # N4 (audit): the fitted linear OPD term was subtracted for fit
    # conditioning but never re-applied -- silently dropping output tilt
    # and shifting the stationary point for decentered / tilted / off-axis
    # systems (benign piston for a centered lens).  Re-apply it EXACTLY by
    # splitting it: the s2 part (c0 + c1*u_s2x + c2*u_s2y) is constant in
    # the pupil-momentum integration variable v2, so it factors out of the
    # canonical integral and is re-applied as an output post-multiply after
    # dispatch; the v2 part (c3*u_v2x + c4*u_v2y) lives inside the integral
    # (it shifts the stationary point) and is threaded into every
    # integrator's OPD + saddle-point gradient.  linear_coeffs are in WAVES
    # (same units as opd), so they add directly with no scaling.
    if linear_coeffs is None:
        linear_coeffs = np.zeros(5, dtype=np.float64)
    _lin = np.asarray(linear_coeffs, dtype=np.float64)
    _lin_v3 = float(_lin[3])
    _lin_v4 = float(_lin[4])

    mi = _multi_indices_total_degree(4, poly_order)
    M = len(mi)
    _progress('fit', 0.25, f'building design matrix ({n_rays} x {M})')
    T1 = _chebyshev_vandermonde(u_s2x, poly_order)
    T2 = _chebyshev_vandermonde(u_s2y, poly_order)
    T3 = _chebyshev_vandermonde(u_v2x, poly_order)
    T4 = _chebyshev_vandermonde(u_v2y, poly_order)
    A = np.empty((len(u_s2x), M), dtype=np.float64)
    for j, (k1, k2, k3, k4) in enumerate(mi):
        A[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]

    _progress('fit', 0.35, 'solving lstsq for OPD')
    coef_opd, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    _progress('fit', 0.45, 'solving lstsq for s1x')
    coef_s1x, *_ = np.linalg.lstsq(A, s1x_live, rcond=None)
    _progress('fit', 0.55, 'solving lstsq for s1y')
    coef_s1y, *_ = np.linalg.lstsq(A, s1y_live, rcond=None)

    opd_pred = A @ coef_opd
    s1x_pred = A @ coef_s1x
    s1y_pred = A @ coef_s1y
    res_opd = np.sqrt(np.mean((opd_residual - opd_pred)**2))
    res_s1x = np.sqrt(np.mean((s1x_live - s1x_pred)**2)) * 1e6
    res_s1y = np.sqrt(np.mean((s1y_live - s1y_pred)**2)) * 1e6

    _progress('fit', 0.60,
              f'RMS OPD residual = {res_opd:.2e} waves; '
              f's1x RMS = {res_s1x:.2e} um, s1y RMS = {res_s1y:.2e} um')

    # -----------------------------------------------------------------
    # Step 3: Build output grids
    # -----------------------------------------------------------------
    _progress('grid', 0.60, 'setting up output and v2 grids')
    if output_subsample < 1:
        output_subsample = 1
    N_out_coarse = N // output_subsample

    out_axis = (np.arange(N_out_coarse) - N_out_coarse / 2) * \
               (dx * output_subsample)
    s2x_grid, s2y_grid = np.meshgrid(out_axis, out_axis, indexing='xy')

    u_s2x_out = (s2x_grid - s2x_c) / s2x_h
    u_s2y_out = (s2y_grid - s2y_c) / s2y_h
    inbox = (np.abs(u_s2x_out) <= 1.0) & (np.abs(u_s2y_out) <= 1.0)

    u_v2x_samples = np.linspace(-1.0, 1.0, n_v2)
    u_v2y_samples = np.linspace(-1.0, 1.0, n_v2)
    du = u_v2x_samples[1] - u_v2x_samples[0]

    def tukey(n, alpha=0.2):
        u = np.linspace(-1, 1, n)
        abs_u = np.abs(u)
        w = np.ones_like(u)
        taper_start = 1.0 - alpha
        tmask = abs_u > taper_start
        w[tmask] = 0.5 * (1 + np.cos(np.pi * (abs_u[tmask] - taper_start) / alpha))
        return w
    tuk_x = tukey(n_v2)
    tuk_y = tukey(n_v2)
    tuk_2d = tuk_x[None, :] * tuk_y[:, None]

    # v5.2.1: ``v2x_samples`` / ``v2y_samples`` were computed but never
    # used -- downstream code reads ``u_v2x_samples`` / ``u_v2y_samples``
    # (the unitless Chebyshev-node coords) instead.  Removed dead assigns.


    def sample_E_bilinear(s1x_q: np.ndarray, s1y_q: np.ndarray) -> np.ndarray:
        in_axis = (np.arange(N) - N / 2) * dx
        fx = (s1x_q - in_axis[0]) / dx
        fy = (s1y_q - in_axis[0]) / dx
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        wx = fx - ix
        wy = fy - iy
        ok = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
        ix_c = np.clip(ix, 0, N - 2)
        iy_c = np.clip(iy, 0, N - 2)
        e00 = E_in[iy_c, ix_c]
        e10 = E_in[iy_c, ix_c + 1]
        e01 = E_in[iy_c + 1, ix_c]
        e11 = E_in[iy_c + 1, ix_c + 1]
        val = ((1 - wx) * (1 - wy) * e00
               + wx * (1 - wy) * e10
               + (1 - wx) * wy * e01
               + wx * wy * e11)
        # v4.14.1 (audit P2-6): dtype-aware out-of-bounds sentinel so
        # a complex64 E_in stays complex64 through the bilinear sample
        # (was silently upcasting via the ``0.0 + 0.0j`` complex128
        # literal).  Matches the v4.13.2 canonical pattern.
        val = np.where(ok, val, np.zeros((), dtype=val.dtype))
        return val

    # -----------------------------------------------------------------
    # Step 4: Integrate
    # -----------------------------------------------------------------
    if integration_method not in ('quadrature', 'stationary_phase',
                                    'local_quadrature'):
        raise ValueError(
            f"integration_method must be one of 'quadrature', "
            f"'stationary_phase', 'local_quadrature', "
            f"got {integration_method!r}")

    _progress('integrate', 0.60,
              f'method={integration_method}')

    K1_arr = np.array([k[0] for k in mi], dtype=np.int64)
    K2_arr = np.array([k[1] for k in mi], dtype=np.int64)
    K3_arr = np.array([k[2] for k in mi], dtype=np.int64)
    K4_arr = np.array([k[3] for k in mi], dtype=np.int64)

    inbox_flat = inbox.ravel()

    if integration_method == 'stationary_phase':
        E_out_coarse = _integrate_stationary_phase(
            coef_opd, coef_s1x, coef_s1y, mi,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, N_out_coarse,
            u_s2x_out, u_s2y_out, inbox_flat,
            v2x_c, v2y_c, v2x_h, v2y_h,
            sample_E_bilinear,
            stationary_newton_iter, stationary_newton_tol,
            _progress, verbose,
            out_dtype=E_in.dtype,
            lin_v3=_lin_v3, lin_v4=_lin_v4,
        )
    elif integration_method == 'local_quadrature':
        E_out_coarse = _integrate_local_quadrature(
            coef_opd, coef_s1x, coef_s1y, mi,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, N_out_coarse,
            u_s2x_out, u_s2y_out, inbox_flat,
            v2x_c, v2y_c, v2x_h, v2y_h,
            sample_E_bilinear,
            stationary_newton_iter, stationary_newton_tol,
            local_n_samples, local_window_sigma,
            _progress, verbose,
            out_dtype=E_in.dtype,
            lin_v3=_lin_v3, lin_v4=_lin_v4,
        )
    else:
        # N2 (audit): estimate the v2 oscillation count of the integrand
        # phase 2*pi*OPD(s2,v2) from the fitted coefficients.  Chebyshev
        # polynomials are bounded by 1 on [-1, 1], so the sum of
        # |coef_opd| over v2-dependent terms (k3>0 or k4>0) upper-bounds
        # the OPD excursion in WAVES = cycles along v2.  Uniform n_v2-point
        # quadrature needs a few samples per cycle; when under-resolved the
        # result speckles regardless of grid/memory (no output-resolution
        # fix helps) -- warn and point at the asymptotic evaluators, which
        # are the correct choice at production NA.
        _v2_mask = np.array(
            [1.0 if (k[2] > 0 or k[3] > 0) else 0.0 for k in mi],
            dtype=np.float64)
        _v2_osc = float(np.sum(np.abs(coef_opd) * _v2_mask))
        if n_v2 < 4.0 * _v2_osc:
            import warnings
            warnings.warn(
                f"apply_real_lens_maslov: integration_method='quadrature' "
                f"with n_v2={n_v2} is under-resolved for this chart "
                f"(~{_v2_osc:.0f} v2 oscillations; want n_v2 >~ "
                f"{int(4 * _v2_osc)}).  Uniform quadrature will speckle "
                f"regardless of output resolution or memory.  Increase "
                f"n_v2, or use integration_method='local_quadrature' / "
                f"'stationary_phase' (the correct evaluators at "
                f"production NA).",
                RuntimeWarning, stacklevel=2)
        # N1 (audit): the (N_out^2, M) Chebyshev design matrix G is used
        # ONLY by the quadrature integrator (its G @ H GEMMs).  The
        # stationary_phase / local_quadrature integrators evaluate the
        # Chebyshev basis per pixel-chunk and never touch G, so building
        # it unconditionally forced a 451 GB allocation at N=16384 /
        # output_subsample=1 on integrators that never read it.  Build it
        # here, in the quadrature branch only.
        _progress('integrate', 0.61,
                  f'precomputing (s2)-basis on {N_out_coarse}^2 output grid')
        Tx_1d = _chebyshev_vandermonde(
            (out_axis - s2x_c) / s2x_h, poly_order)
        Ty_1d = _chebyshev_vandermonde(
            (out_axis - s2y_c) / s2y_h, poly_order)
        M = len(mi)
        G = np.empty((N_out_coarse * N_out_coarse, M), dtype=np.float64)
        for m, (k1, k2, _, _) in enumerate(mi):
            G[:, m] = np.outer(Ty_1d[k2], Tx_1d[k1]).ravel()
        _progress('integrate', 0.63,
                  f'G matrix {G.shape} = {G.nbytes/1e6:.1f} MB')
        E_out_coarse = _integrate_quadrature(
            coef_opd, coef_s1x, coef_s1y, mi,
            K1_arr, K2_arr, K3_arr, K4_arr,
            poly_order, G, N_out_coarse,
            u_v2x_samples, u_v2y_samples, tuk_2d, du,
            v2x_h, v2y_h, chunk_v2, inbox_flat,
            sample_E_bilinear,
            use_numexpr, _progress,
            _lenses_module,
            out_dtype=E_in.dtype,
            lin_v3=_lin_v3, lin_v4=_lin_v4,
        )

    # -----------------------------------------------------------------
    # Step 5: Upsample to the full grid if output_subsample > 1
    # -----------------------------------------------------------------
    if output_subsample > 1:
        _progress('upsample', 0.95,
                  f'interpolating {N_out_coarse}^2 -> {N}^2 (cubic)')
        from scipy.ndimage import zoom
        zoom_factor = float(N) / float(N_out_coarse)
        amp = np.abs(E_out_coarse)
        amp_z = zoom(amp, zoom_factor, order=3, mode='nearest')
        # Phase upsampling: pre-3.5.6 used line-by-line np.unwrap then
        # cubic zoom of the unwrapped phase.  Line-by-line unwrap is
        # fragile near caustics / focal saddles where the phase wraps
        # along both axes; the resulting cubic-interpolated phase had
        # ~4% RMS errors from line-mismatched seams.
        #
        # 3.5.6 fix: interpolate the COMPLEX exp(i*phase) directly via
        # cubic zoom of its real and imaginary parts, then take
        # ``angle()``.  This avoids any 2-D phase-unwrap step
        # (and therefore any unwrap-induced seams) at the cost of
        # only being well-behaved when the local phase variation
        # between adjacent coarse pixels is < pi -- which is the same
        # condition the original line-unwrap silently relied on.
        # For Maslov outputs that satisfy that bound (typical
        # refractive systems with output_subsample <= 8), the new
        # path agrees with the OLD output to ~0.3% RMS while
        # eliminating the caustic-seam artifact.
        phase_c = np.angle(E_out_coarse)
        cos_z = zoom(np.cos(phase_c), zoom_factor, order=3, mode='nearest')
        sin_z = zoom(np.sin(phase_c), zoom_factor, order=3, mode='nearest')
        E_out_re = amp_z * cos_z
        E_out_im = amp_z * sin_z

        def _fit(a):
            if a.shape == (N, N):
                return a
            out = np.zeros((N, N), dtype=a.dtype)
            rows = min(a.shape[0], N)
            cols = min(a.shape[1], N)
            out[:rows, :cols] = a[:rows, :cols]
            return out
        # v4.14.0: ``1j * float64`` returns complex128; cast back to
        # E_in.dtype so complex64 inputs are preserved through the
        # final re-fit step.
        E_out = (_fit(E_out_re) + 1j * _fit(E_out_im)).astype(E_in.dtype)
    else:
        E_out = E_out_coarse

    # N4 (audit) re-apply the s2 part of the fitted linear OPD that was
    # subtracted before fitting.  Split by cost + Nyquist-safety:
    #
    #  * Piston (_lin[0]) is a GLOBAL phase -> apply as a scalar.  It is
    #    grid-invariant, so this avoids building an N x N temporary just to
    #    add a constant (the piston is ~10^3 waves and is the ONLY term
    #    that is ever appreciable here -- see below).
    #  * The s2-slope terms (_lin[1], _lin[2]) are ~0 for a rotationally-
    #    symmetric prescription (the OPL is then even in output position:
    #    measured |_lin[1]| ~ 1e-10 for a symmetric singlet, < 0.04 waves
    #    for a 0.04 rad tilted input; literal decenter/tilt dict keys are
    #    dropped by the centred trace).  But a FREEFORM surface
    #    (xy_polynomial / zernike odd terms are honored by the trace) makes
    #    them genuinely large -- a wedge/prism deviates the beam, giving a
    #    real output-position OPL slope of up to ~10^4 waves (adversarial
    #    review; verified prism |_lin[1]| = 15.6 waves >> coarse Nyquist,
    #    still subsample-invariant here).  So this branch is load-bearing,
    #    not defensive: the slope MUST be applied on the FINE (post-upsample)
    #    grid, because a slope above the coarse Nyquist (c1 > N_out_coarse/4)
    #    aliases / flips under the cubic phase-zoom if applied on the coarse
    #    field first.  The fine-pixel coordinate is reproduced by zooming the
    #    coarse output axis with the SAME zoom call, so the tilt lands
    #    exactly where the zoomed content lives (convention-independent;
    #    avoids the grid_mode=False edge-stretch of a nominal fine axis).
    #    The abs()>1e-6 gate skips the N x N coordinate build for the common
    #    symmetric case (where the slope is a negligible ~1e-10 waves and
    #    the meshgrid would otherwise cost ~17 GB at N=32768).
    #
    # NB when a large real output tilt ALSO has an in-integral (pupil, v2)
    # component -- e.g. a strongly-powered freeform lens rather than a flat
    # wedge -- that component lives INSIDE the canonical integral (via the
    # _lin_v3/_v4 terms) and is coarse-resolved, so it aliases for output
    # tilts above the coarse Nyquist regardless of where this post-multiply
    # runs.  That is the N2 under-resolution regime (warned separately);
    # reduce output_subsample.
    if _lin[0]:
        E_out = (E_out * np.exp(2j * np.pi * _lin[0])).astype(E_in.dtype)
    if abs(_lin[1]) > 1e-6 or abs(_lin[2]) > 1e-6:
        if output_subsample > 1:
            from scipy.ndimage import zoom as _zoom1d
            out_axis_f = _zoom1d(out_axis, float(N) / float(N_out_coarse),
                                 order=1, mode='nearest')
            if out_axis_f.shape[0] != N:   # non-divisible safety (matches _fit)
                _tmp = np.zeros(N, dtype=out_axis_f.dtype)
                _n = min(out_axis_f.shape[0], N)
                _tmp[:_n] = out_axis_f[:_n]
                out_axis_f = _tmp
        else:
            out_axis_f = out_axis          # coarse grid == fine grid
        _s2x_f, _s2y_f = np.meshgrid(out_axis_f, out_axis_f, indexing='xy')
        _u_s2x_f = (_s2x_f - s2x_c) / s2x_h
        _u_s2y_f = (_s2y_f - s2y_c) / s2y_h
        E_out = (E_out * np.exp(
            2j * np.pi * (_lin[1] * _u_s2x_f
                          + _lin[2] * _u_s2y_f))).astype(E_in.dtype)
        del _s2x_f, _s2y_f, _u_s2x_f, _u_s2y_f

    # -----------------------------------------------------------------
    # Step 6: Absolute-amplitude normalization.
    # -----------------------------------------------------------------
    if normalize_output == 'power':
        p_in = float((np.abs(E_in)**2).sum())
        p_out = float((np.abs(E_out)**2).sum())
        if p_out > 0 and p_in > 0:
            scale = np.sqrt(p_in / p_out)
            E_out = E_out * scale
    elif normalize_output == 'peak':
        a_in = float(np.abs(E_in).max())
        a_out = float(np.abs(E_out).max())
        if a_out > 0 and a_in > 0:
            E_out = E_out * (a_in / a_out)
    elif normalize_output == 'none':
        pass
    elif isinstance(normalize_output, (int, float, complex)):
        E_out = E_out * normalize_output
    else:
        raise ValueError(f"normalize_output={normalize_output!r}; "
                          f"expected 'power', 'peak', 'none', or scalar")

    # v4.14.0: final dtype cast back to E_in.dtype.  The normalization
    # multiplies above promote complex64 -> complex128 because the
    # scalar scale factor is a python float (float64).  Cast once at
    # the end to preserve the input-dtype contract.
    if E_out.dtype != E_in.dtype:
        E_out = E_out.astype(E_in.dtype)

    _progress('done', 1.0,
              f'total {time.perf_counter()-t0:.1f}s')
    return E_out


def _count_multi_indices_4d(max_order: int) -> int:
    """Number of 4-variable multi-indices with total degree <= max_order
    (== C(n+4, 4) for n = max_order)."""
    from math import comb
    return comb(max_order + 4, 4)


# ---------------------------------------------------------------------------
# Integration method helpers
# ---------------------------------------------------------------------------

def _integrate_quadrature(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, G, N_out_coarse,
    u_v2x_samples, u_v2y_samples, tuk_2d, du,
    v2x_h, v2y_h, chunk_v2, inbox_flat,
    sample_E_bilinear,
    use_numexpr, _progress,
    _lenses_module,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Uniform Tukey-windowed quadrature on the (v2x, v2y) grid.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.
    """
    n_v2 = len(u_v2x_samples)
    n_v2_total = n_v2 * n_v2

    Tu3_all  = _chebyshev_vandermonde(u_v2x_samples, poly_order)
    Tu4_all  = _chebyshev_vandermonde(u_v2y_samples, poly_order)
    dTu3_all = _chebyshev_derivative_vandermonde(u_v2x_samples, poly_order)
    dTu4_all = _chebyshev_derivative_vandermonde(u_v2y_samples, poly_order)

    iy_grid, ix_grid = np.meshgrid(np.arange(n_v2), np.arange(n_v2),
                                     indexing='ij')
    v2x_idx = ix_grid.ravel()
    v2y_idx = iy_grid.ravel()

    T3bj  = Tu3_all [K3_arr[:, None], v2x_idx[None, :]]
    T4bj  = Tu4_all [K4_arr[:, None], v2y_idx[None, :]]
    dT3bj = dTu3_all[K3_arr[:, None], v2x_idx[None, :]]
    dT4bj = dTu4_all[K4_arr[:, None], v2y_idx[None, :]]
    T3_T4  = T3bj * T4bj
    dT3_T4 = dT3bj * T4bj
    T3_dT4 = T3bj * dT4bj

    H_opd      = coef_opd[:, None] * T3_T4
    H_s1x      = coef_s1x[:, None] * T3_T4
    H_s1y      = coef_s1y[:, None] * T3_T4
    H_ds1x_du3 = coef_s1x[:, None] * dT3_T4
    H_ds1x_du4 = coef_s1x[:, None] * T3_dT4
    H_ds1y_du3 = coef_s1y[:, None] * dT3_T4
    H_ds1y_du4 = coef_s1y[:, None] * T3_dT4

    weight_per_sample = tuk_2d.ravel() * du * du * (v2x_h * v2y_h)

    # N4: linear-in-v2 OPD term (c3*u_v2x + c4*u_v2y), one value per v2
    # sample; added to the residual-fit opd_c in the chunk loop below.
    lin_v = (lin_v3 * u_v2x_samples[v2x_idx]
             + lin_v4 * u_v2y_samples[v2y_idx])

    if use_numexpr is None:
        use_numexpr = NUMEXPR_AVAILABLE
    use_numexpr = (bool(use_numexpr) and NUMEXPR_AVAILABLE
                    and _ensure_numexpr_loaded())
    _progress('integrate', 0.65,
              f'quadrature: {n_v2_total} v2 samples, chunk={chunk_v2}, '
              f'numexpr={use_numexpr}')

    if chunk_v2 <= 0:
        chunk_v2 = n_v2_total
    chunk_v2 = min(chunk_v2, n_v2_total)

    E_out_flat = np.zeros(N_out_coarse * N_out_coarse, dtype=out_dtype)
    t_int_start = time.perf_counter()

    for c_start in range(0, n_v2_total, chunk_v2):
        c_end = min(c_start + chunk_v2, n_v2_total)

        opd_c      = G @ H_opd     [:, c_start:c_end]
        opd_c      = opd_c + lin_v[None, c_start:c_end]
        s1x_c      = G @ H_s1x     [:, c_start:c_end]
        s1y_c      = G @ H_s1y     [:, c_start:c_end]
        ds1x_du3_c = G @ H_ds1x_du3[:, c_start:c_end]
        ds1x_du4_c = G @ H_ds1x_du4[:, c_start:c_end]
        ds1y_du3_c = G @ H_ds1y_du3[:, c_start:c_end]
        ds1y_du4_c = G @ H_ds1y_du4[:, c_start:c_end]

        det_J_c = (ds1x_du3_c * ds1y_du4_c
                   - ds1x_du4_c * ds1y_du3_c)
        abs_J_c = np.abs(det_J_c) / (v2x_h * v2y_h)

        Eobj_c = sample_E_bilinear(s1x_c, s1y_c)
        weights_c = weight_per_sample[c_start:c_end]

        if use_numexpr:
            # v5.2.1: numexpr's ``evaluate(expr)`` reads variable names
            # from the caller's stack frame via introspection, which
            # makes ``twopi`` / ``cos_term`` / etc. invisible to static
            # analysis (ruff F841).  Pass an explicit ``local_dict=``
            # so the locals appear in the surrounding code's AST.
            # Matches the canonical pattern at ``_lens_real.py:882``.
            _ne = _lenses_module._ne
            twopi = 2.0 * np.pi
            cos_term = _ne.evaluate(
                "cos(twopi * opd_c)",
                local_dict={'twopi': twopi, 'opd_c': opd_c})
            sin_term = _ne.evaluate(
                "sin(twopi * opd_c)",
                local_dict={'twopi': twopi, 'opd_c': opd_c})
            Er = Eobj_c.real
            Ei = Eobj_c.imag
            contrib_r = _ne.evaluate(
                "(Er*cos_term - Ei*sin_term) * abs_J_c * weights_c",
                local_dict={'Er': Er, 'Ei': Ei,
                            'cos_term': cos_term, 'sin_term': sin_term,
                            'abs_J_c': abs_J_c, 'weights_c': weights_c})
            contrib_i = _ne.evaluate(
                "(Ei*cos_term + Er*sin_term) * abs_J_c * weights_c",
                local_dict={'Er': Er, 'Ei': Ei,
                            'cos_term': cos_term, 'sin_term': sin_term,
                            'abs_J_c': abs_J_c, 'weights_c': weights_c})
            contrib_sum = contrib_r.sum(axis=1) + 1j * contrib_i.sum(axis=1)
        else:
            contrib_c = (Eobj_c
                          * np.exp(2j * np.pi * opd_c)
                          * abs_J_c
                          * weights_c)
            contrib_sum = contrib_c.sum(axis=1)

        E_out_flat[inbox_flat] += contrib_sum[inbox_flat]

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'quadrature: {n_v2_total} v2 samples in {t_int:.1f}s '
              f'({"numexpr" if use_numexpr else "numpy"}, '
              f'chunk={chunk_v2})')

    return E_out_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_stationary_phase(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_c, v2y_c, v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Leading-order stationary-phase (Gaussian-moment) evaluation.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.
    """
    t_int_start = time.perf_counter()
    _progress('integrate', 0.65,
              f'stationary-phase Newton ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse

    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)

    def _opd_and_derivs(coef, u1, u2, u3, u4):
        T1 = _chebyshev_vandermonde(u1, poly_order)
        T2 = _chebyshev_vandermonde(u2, poly_order)
        T3 = _chebyshev_vandermonde(u3, poly_order)
        T4 = _chebyshev_vandermonde(u4, poly_order)
        dT3 = _chebyshev_derivative_vandermonde(u3, poly_order)
        dT4 = _chebyshev_derivative_vandermonde(u4, poly_order)
        d2T3 = _chebyshev_second_derivative_vandermonde(u3, poly_order)
        d2T4 = _chebyshev_second_derivative_vandermonde(u4, poly_order)
        T1b = T1[K1_arr]
        T2b = T2[K2_arr]
        T3b = T3[K3_arr]
        T4b = T4[K4_arr]
        dT3b = dT3[K3_arr]
        dT4b = dT4[K4_arr]
        d2T3b = d2T3[K3_arr]
        d2T4b = d2T4[K4_arr]
        T12 = T1b * T2b
        c = coef[:, None]
        f        = np.sum(c * T12 * T3b  * T4b , axis=0)
        df_du3   = np.sum(c * T12 * dT3b * T4b , axis=0)
        df_du4   = np.sum(c * T12 * T3b  * dT4b, axis=0)
        d2f_33   = np.sum(c * T12 * d2T3b* T4b , axis=0)
        d2f_44   = np.sum(c * T12 * T3b  * d2T4b, axis=0)
        d2f_34   = np.sum(c * T12 * dT3b * dT4b, axis=0)
        return f, df_du3, df_du4, d2f_33, d2f_34, d2f_44

    converged_mask = np.zeros(N_px, dtype=bool)
    converged_mask[~inbox_flat] = True

    for it in range(newton_iter):
        if converged_mask.all():
            break
        active = ~converged_mask
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = _opd_and_derivs(
            coef_opd, u1, u2, u3, u4)
        # N4: the linear-in-v2 OPD term (c3*u_v2x + c4*u_v2y) has constant
        # v2-gradient (c3, c4) and zero Hessian, so it shifts the saddle
        # point but not its curvature.  Add it to the gradient here.
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                             np.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_limit = 0.5
        step_size = np.sqrt(dv3**2 + dv4**2)
        damp = np.where(step_size > step_limit,
                         step_limit / np.maximum(step_size, 1e-30),
                         1.0)
        dv3 *= damp
        dv4 *= damp
        u_v2x_new = u_v2x[active] + dv3
        u_v2y_new = u_v2y[active] + dv4
        u_v2x_new = np.clip(u_v2x_new, -1.0, 1.0)
        u_v2y_new = np.clip(u_v2y_new, -1.0, 1.0)
        u_v2x[active] = u_v2x_new
        u_v2y[active] = u_v2y_new
        grad_mag = np.sqrt(g3**2 + g4**2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged_mask |= newly
        if verbose and (it == 0 or it == newton_iter - 1 or
                         it % max(1, newton_iter // 4) == 0):
            n_conv = converged_mask.sum()
            _progress('integrate', 0.65 + 0.15 * it / newton_iter,
                      f'Newton iter {it+1}/{newton_iter}, '
                      f'{n_conv}/{N_px} pixels converged '
                      f'(max grad {grad_mag.max():.2e})')

    _progress('integrate', 0.85, 'evaluating saddle-point formula')

    opd_star, g3, g4, H33, H34, H44 = _opd_and_derivs(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    # N4: add the linear-in-v2 OPD contribution at the (shifted) saddle.
    opd_star = opd_star + lin_v3 * u_v2x + lin_v4 * u_v2y
    s1x_star, ds1x_du3, ds1x_du4, _, _, _ = _opd_and_derivs(
        coef_s1x, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    s1y_star, ds1y_du3, ds1y_du4, _, _, _ = _opd_and_derivs(
        coef_s1y, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)

    det_J_norm = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
    abs_J = np.abs(det_J_norm) / (v2x_h * v2y_h)

    H33_phys = H33 / (v2x_h * v2x_h)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h * v2y_h)
    det_H_phys = H33_phys * H44_phys - H34_phys * H34_phys
    trace_H = H33_phys + H44_phys
    sig = np.where(det_H_phys > 0,
                    np.where(trace_H > 0, 2, -2),
                    0)
    amp_sp = 1.0 / np.sqrt(np.maximum(np.abs(det_H_phys), 1e-300))
    phase_sp = np.exp(1j * (np.pi / 4.0) * sig)

    Eobj_star = sample_E_bilinear(s1x_star, s1y_star)

    # v4.14.0: cast to ``out_dtype`` (=E_in.dtype from caller) so a
    # complex64 input doesn't get silently upcast to complex128 by
    # the float64-phase * complex128-exp multiply.
    E_flat = (Eobj_star
              * np.exp(2j * np.pi * opd_star)
              * abs_J
              * amp_sp
              * phase_sp).astype(out_dtype)

    not_conv = ~converged_mask
    if not_conv.any():
        E_flat[not_conv] = 0.0
        if verbose:
            _progress('integrate', 0.92,
                      f'{not_conv.sum()}/{N_px} pixels did not converge, '
                      f'zeroed')

    E_flat[~inbox_flat] = 0.0

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'stationary_phase: {N_px} pixels in {t_int:.1f}s')

    return E_flat.reshape(N_out_coarse, N_out_coarse)


def _integrate_local_quadrature(
    coef_opd, coef_s1x, coef_s1y, mi,
    K1_arr, K2_arr, K3_arr, K4_arr,
    poly_order, N_out_coarse,
    u_s2x_out, u_s2y_out, inbox_flat,
    v2x_c, v2y_c, v2x_h, v2y_h,
    sample_E_bilinear,
    newton_iter, newton_tol,
    n_samples, window_sigma,
    _progress, verbose,
    out_dtype=np.complex128,
    lin_v3=0.0, lin_v4=0.0,
):
    """Hybrid stationary-phase + local quadrature.

    v4.14.0: ``out_dtype`` defaults to ``np.complex128`` for back-
    compat; callers pass ``E_in.dtype`` to preserve complex64 inputs.
    """
    t_int_start = time.perf_counter()
    _progress('integrate', 0.60,
              f'local_quadrature: Newton phase ({newton_iter} max iters)')

    N_px = N_out_coarse * N_out_coarse
    u_s2x_flat = u_s2x_out.ravel()
    u_s2y_flat = u_s2y_out.ravel()

    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)

    def _opd_and_derivs(coef, u1, u2, u3, u4):
        T1 = _chebyshev_vandermonde(u1, poly_order)
        T2 = _chebyshev_vandermonde(u2, poly_order)
        T3 = _chebyshev_vandermonde(u3, poly_order)
        T4 = _chebyshev_vandermonde(u4, poly_order)
        dT3 = _chebyshev_derivative_vandermonde(u3, poly_order)
        dT4 = _chebyshev_derivative_vandermonde(u4, poly_order)
        d2T3 = _chebyshev_second_derivative_vandermonde(u3, poly_order)
        d2T4 = _chebyshev_second_derivative_vandermonde(u4, poly_order)
        T1b = T1[K1_arr]
        T2b = T2[K2_arr]
        T3b = T3[K3_arr]
        T4b = T4[K4_arr]
        dT3b = dT3[K3_arr]
        dT4b = dT4[K4_arr]
        d2T3b = d2T3[K3_arr]
        d2T4b = d2T4[K4_arr]
        T12 = T1b * T2b
        c = coef[:, None]
        f        = np.sum(c * T12 * T3b  * T4b , axis=0)
        df_du3   = np.sum(c * T12 * dT3b * T4b , axis=0)
        df_du4   = np.sum(c * T12 * T3b  * dT4b, axis=0)
        d2f_33   = np.sum(c * T12 * d2T3b* T4b , axis=0)
        d2f_44   = np.sum(c * T12 * T3b  * d2T4b, axis=0)
        d2f_34   = np.sum(c * T12 * dT3b * dT4b, axis=0)
        return f, df_du3, df_du4, d2f_33, d2f_34, d2f_44

    converged = np.zeros(N_px, dtype=bool)
    converged[~inbox_flat] = True
    for it in range(newton_iter):
        if converged.all():
            break
        active = ~converged
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = _opd_and_derivs(coef_opd, u1, u2, u3, u4)
        # N4: linear-in-v2 OPD term shifts the saddle gradient (c3, c4).
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                             np.sign(det_H) * 1e-30 + 1e-30, det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_size = np.sqrt(dv3 ** 2 + dv4 ** 2)
        damp = np.where(step_size > 0.5,
                         0.5 / np.maximum(step_size, 1e-30), 1.0)
        dv3 *= damp
        dv4 *= damp
        u_v2x[active] = np.clip(u_v2x[active] + dv3, -1.0, 1.0)
        u_v2y[active] = np.clip(u_v2y[active] + dv4, -1.0, 1.0)
        grad_mag = np.sqrt(g3 ** 2 + g4 ** 2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged |= newly

    _progress('integrate', 0.72, 'computing Hessian eigen-scales')
    _, _, _, H33, H34, H44 = _opd_and_derivs(
        coef_opd, u_s2x_flat, u_s2y_flat, u_v2x, u_v2y)
    H33_phys = H33 / (v2x_h ** 2)
    H34_phys = H34 / (v2x_h * v2y_h)
    H44_phys = H44 / (v2y_h ** 2)
    tau = H33_phys + H44_phys
    detH = H33_phys * H44_phys - H34_phys ** 2
    disc = np.maximum(tau ** 2 / 4.0 - detH, 0.0)
    sqrt_disc = np.sqrt(disc)
    lam1 = tau / 2.0 + sqrt_disc
    lam2 = tau / 2.0 - sqrt_disc
    sigma1_phys = 1.0 / np.sqrt(np.maximum(np.abs(lam1), 1e-30) * np.pi)
    sigma2_phys = 1.0 / np.sqrt(np.maximum(np.abs(lam2), 1e-30) * np.pi)
    sigma1_norm = sigma1_phys / v2x_h
    sigma2_norm = sigma2_phys / v2y_h

    _progress('integrate', 0.75,
              f'local uniform sampling: {n_samples}x{n_samples} pts, '
              f'window={window_sigma}sigma')
    lin = np.linspace(-window_sigma, window_sigma, n_samples)
    dxi = lin[1] - lin[0]
    Xlin, Ylin = np.meshgrid(lin, lin, indexing='xy')
    Xlin_flat = Xlin.ravel()
    Ylin_flat = Ylin.ravel()

    u_v2x_samp = (u_v2x[:, None]
                   + (sigma1_norm[:, None]) * Xlin_flat[None, :])
    u_v2y_samp = (u_v2y[:, None]
                   + (sigma2_norm[:, None]) * Ylin_flat[None, :])
    u_v2x_samp = np.clip(u_v2x_samp, -1.0, 1.0)
    u_v2y_samp = np.clip(u_v2y_samp, -1.0, 1.0)

    n_s2 = n_samples * n_samples
    u_s2x_tile = np.broadcast_to(u_s2x_flat[:, None], (N_px, n_s2))
    u_s2y_tile = np.broadcast_to(u_s2y_flat[:, None], (N_px, n_s2))

    _progress('integrate', 0.78,
              f'evaluating integrand on {N_px*n_s2:,} (pixel,sample) pairs')

    E_flat = np.zeros(N_px, dtype=out_dtype)
    w2d_phys = (sigma1_phys * sigma2_phys) * (dxi ** 2)

    PX_CHUNK = max(1, min(N_px, 1024 * 64 // max(1, n_s2 // 16)))
    for p_start in range(0, N_px, PX_CHUNK):
        p_end = min(p_start + PX_CHUNK, N_px)
        u3 = u_v2x_samp[p_start:p_end].ravel()
        u4 = u_v2y_samp[p_start:p_end].ravel()
        u1 = u_s2x_tile[p_start:p_end].ravel()
        u2 = u_s2y_tile[p_start:p_end].ravel()
        opd_v, _, _, _, _, _        = _opd_and_derivs(coef_opd, u1, u2, u3, u4)
        # N4: linear-in-v2 OPD contribution at each window sample.
        opd_v = opd_v + lin_v3 * u3 + lin_v4 * u4
        s1x_v, ds1x_du3, ds1x_du4, *_ = _opd_and_derivs(coef_s1x, u1, u2, u3, u4)
        s1y_v, ds1y_du3, ds1y_du4, *_ = _opd_and_derivs(coef_s1y, u1, u2, u3, u4)
        det_J = ds1x_du3 * ds1y_du4 - ds1x_du4 * ds1y_du3
        abs_J = np.abs(det_J) / (v2x_h * v2y_h)

        Eobj_v = sample_E_bilinear(s1x_v, s1y_v)

        contrib = (Eobj_v
                    * np.exp(2j * np.pi * opd_v)
                    * abs_J)
        contrib_r = contrib.reshape(p_end - p_start, n_s2)
        E_flat[p_start:p_end] = contrib_r.sum(axis=1) * \
                                  w2d_phys[p_start:p_end]
        if verbose and (p_start % (PX_CHUNK * 8) == 0):
            _progress('integrate',
                      0.78 + 0.15 * (p_end / N_px),
                      f'pixel chunk {p_end}/{N_px}')

    E_flat[~converged] = 0.0
    E_flat[~inbox_flat] = 0.0

    t_int = time.perf_counter() - t_int_start
    _progress('integrate', 0.95,
              f'local_quadrature: {N_px} pixels, '
              f'{n_s2} samples/pixel, {t_int:.1f}s')

    return E_flat.reshape(N_out_coarse, N_out_coarse)


__all__ = [
    'apply_real_lens_maslov',
]
