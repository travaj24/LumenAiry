"""
lumenairy.propagators.hf -- Huygens-Fresnel propagators.

This module is two unrelated halves, and it matters which one you want
(audit K15/K24):

1. **A Van-Vleck-corrected direct quadrature**,
   :func:`propagate_huygens_fresnel_with_opl_callable`, which evaluates

       E_out(s2) = integral E_in(s1) sqrt(|det d2 Phi / d s1 d s2|)
                   * exp(2 pi i Phi(s1, s2)) d^2 s1

   for an arbitrary user-supplied optical-path callable ``Phi``.  The Van
   Vleck density is the one genuine Van Vleck factor in this file; it
   makes the bare HF integrand energy-conserving on non-conjugate output
   planes and keeps it finite at the focus.  Verified: for the exact
   spherical OPL the code's own cross-Hessian stencil reproduces
   ``sqrt|det| = cos(theta)/(lambda r)`` to 1.1e-5, and combined with the
   ``-1j`` Maslov prefactor the kernel is EXACTLY
   ``(1/(i lambda)) cos(theta) e^{ikr}/r`` -- Rayleigh-Sommerfeld I
   without the ``(1 - 1/(ikr))`` near-field term.  It is
   ``O(N_in^2 * N_out^2)``; see that function's ``chunk_output`` note.

2. **A free-space / prescription front end.**
   :func:`propagate_huygens_fresnel` and
   :func:`propagate_huygens_fresnel_freespace` are the canonical-order
   entry points, and they are a thin delegation to
   :func:`~lumenairy.propagators.rs.rayleigh_sommerfeld_propagate` --
   an FFT convolution with the exact RS-I kernel, with NO Van Vleck
   factor and no quadrature of their own (none is needed: for a
   shift-invariant free-space ``Phi`` the FFT route is the same physics
   and ~5e4x faster at N = 256 at the same ~1e-3 accuracy).
   :func:`propagate_huygens_fresnel_through_prescription` dispatches to
   the asymptotic family, not to the quadrature above.

For a free-space plane-to-plane hop prefer
:func:`~lumenairy.propagators.asm.angular_spectrum_propagate` or the RS
kernel directly; this module's quadrature earns its keep only when
``Phi`` is genuinely NOT shift-invariant.

See ``REFERENCES.txt`` Sections A and B for the foundational
publications.

Author: Andrew Traverso
"""

# Version history for this module: ``docs/history/lumenairy.propagators.hf.md``.

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array

# Target size of ONE ``(n_chunk, Ny_in, Nx_in)`` float64 working array in
# the HF OPL quadrature (audit K22).  The batched evaluation does exactly
# the same flops as the per-output-pixel one, so its only lever is
# Python-level dispatch -- which only pays while the working arrays stay
# in cache.  128 KB reproduces the measured per-size optimum; see the
# ladder in :func:`propagate_huygens_fresnel_with_opl_callable`.
_HF_CHUNK_TARGET_BYTES = 128 * 1024


# v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): output-grid kwarg semantics
# disambiguation; see :mod:`lumenairy.propagators.gbd` for full
# rationale.
def _resolve_output_shape(
    output_shape: Optional[Tuple[int, int]],
    output_grid: Optional[Any],
    *,
    fn_name: str,
    default_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve the (Ny, Nx) output shape from the v5.2 ``output_shape``
    kwarg and the deprecated ``output_grid`` legacy kwarg."""
    if output_shape is not None and output_grid is not None:
        raise ValueError(
            f"{fn_name}: both ``output_shape`` and ``output_grid`` were "
            f"provided.  Pass only ``output_shape=(Ny, Nx)`` (v5.2+) or "
            f"the dispatcher's ``output_grid=(N_out, dx_out)`` form via "
            f"``propagate(method=...)``.")
    if output_shape is not None:
        return (int(output_shape[0]), int(output_shape[1]))
    if output_grid is not None:
        warnings.warn(
            f"{fn_name}: the ``output_grid`` kwarg now (v5.2+) means "
            f"the dispatcher's ``(N_out, dx_out)`` grid spec; on "
            f"sub-propagators it has been renamed to ``output_shape`` "
            f"for the ``(Ny, Nx)`` shape-only meaning.  Pass "
            f"``output_shape=(Ny, Nx)`` to silence this warning, or "
            f"call via ``propagate(method='hf', output_grid=(N_out, "
            f"dx_out), ...)`` if you actually want grid resampling.",
            DeprecationWarning, stacklevel=3,
        )
        return (int(output_grid[0]), int(output_grid[1]))
    return default_shape


def _resample_preserving_window_power(
    E_native: np.ndarray,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    fn_name: str,
    order: int = 3,
) -> Tuple[np.ndarray, float]:
    """Resample onto an ``(N_out, N_out)`` grid at ``dx_out`` and restore
    the L2 energy the TARGET WINDOW actually holds.

    The naive Parseval renormalisation ``sqrt(p_in / p_out)`` conflates
    two different things: the small interpolation drift of a bicubic
    ``map_coordinates`` (which SHOULD be corrected) and a genuine
    physical CROP when the requested window is smaller than the source's
    (which must NOT be).  Renormalising to the full source power in the
    crop case FABRICATES energy: measured on
    ``propagate_huygens_fresnel_freespace(E, 1e-3, 633e-9, 2e-6,
    output_dx=0.5e-6)`` at N = 64, the requested +-16 um window genuinely
    contains 67.27 % of the native-grid power and the returned array
    carried 100.00 % -- amplitudes inflated 1.219x, intensities 1.486x.

    This helper measures the reference power on the SOURCE grid,
    restricted to the area the output pixels tile, so the correction is
    the interpolation drift alone.  When the target window covers the
    whole source (the upsampling / same-extent case) the restriction is
    the identity and the behaviour is unchanged.

    A crop that discards more than a part in 1e6 of the power is
    reported as a ``RuntimeWarning``: the caller asked for a window, and
    losing a third of the beam to it is a modelling fact, not a
    numerical detail.

    Parameters
    ----------
    E_native : ndarray, complex
        Field on the source grid (NumPy; ``resample_field`` is
        scipy-backed and host-only).
    dx_in, dx_out : float
        Source and target pitch [m].
    N_out : int
        Target grid size (square).
    fn_name : str
        Caller name for the warning text (CONVENTIONS section 2).
    order : int, default 3
        Interpolation order handed to :func:`resample_field`.

    Returns
    -------
    E_out : ndarray
        Resampled field, in ``E_native``'s dtype.
    dx_out : float
        The target pitch (echoing the ``resample_field`` contract).
    """
    from .mft import resample_field

    E_resampled, dx_resampled = resample_field(
        E_native, dx_in, dx_out, int(N_out), order=order)

    Ny_in, Nx_in = int(E_native.shape[-2]), int(E_native.shape[-1])
    dx_in_f = float(dx_in)
    dx_out_f = float(dx_out)
    n_out = int(N_out)

    # The output pixels are pixel-centred at ``(j - N_out/2)*dx_out`` and
    # each tiles ``[x - dx_out/2, x + dx_out/2)``, so the window they
    # cover is the half-open interval below (asymmetric by half a pixel,
    # which is the library-wide ``arange(N) - N/2`` convention).
    lo = (0.0 - n_out / 2.0) * dx_out_f - 0.5 * dx_out_f
    hi = (n_out - 1.0 - n_out / 2.0) * dx_out_f + 0.5 * dx_out_f
    x_src = (np.arange(Nx_in, dtype=np.float64) - Nx_in / 2.0) * dx_in_f
    y_src = (np.arange(Ny_in, dtype=np.float64) - Ny_in / 2.0) * dx_in_f
    in_x = (x_src >= lo) & (x_src <= hi)
    in_y = (y_src >= lo) & (y_src <= hi)

    amp2 = np.abs(np.asarray(E_native)) ** 2
    p_in = float(np.sum(amp2)) * (dx_in_f ** 2)
    if in_x.all() and in_y.all():
        p_window = p_in
    else:
        p_window = float(np.sum(amp2[np.ix_(in_y, in_x)])) * (dx_in_f ** 2)
    p_out = float(np.sum(np.abs(np.asarray(E_resampled)) ** 2)) * (
        float(dx_resampled) ** 2)

    if p_in > 0.0 and p_window < p_in * (1.0 - 1e-6):
        warnings.warn(
            f"{fn_name}: the requested output window "
            f"{n_out}x{n_out} @ dx={dx_out_f:.4e} m (extent "
            f"{n_out * dx_out_f:.4e} m) is smaller than the field's "
            f"({Ny_in}x{Nx_in} @ dx={dx_in_f:.4e} m, extent "
            f"{Nx_in * dx_in_f:.4e} m) and CROPS it: only "
            f"{100.0 * p_window / p_in:.2f}% of the power falls inside.  "
            f"The returned field carries that fraction (the Parseval "
            f"renormalisation restores the interpolation drift only, not "
            f"the cropped light).  Request a window that spans the field, "
            f"or treat the loss as the aperture it is.",
            RuntimeWarning, stacklevel=3)

    if p_out > 0.0 and p_window > 0.0:
        E_resampled = E_resampled * float(np.sqrt(p_window / p_out))
    # Preserve dtype (resample_field promotes complex64 -> complex128
    # via map_coordinates' float64 output).
    if E_resampled.dtype != E_native.dtype:
        E_resampled = E_resampled.astype(E_native.dtype)
    return E_resampled, dx_resampled


def propagate_huygens_fresnel(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    **kwargs: Any,
) -> np.ndarray:
    """Canonical-order Huygens-Fresnel free-space propagation.

    Argument order ``(E_in, z, wavelength, dx)`` matches
    :func:`angular_spectrum_propagate`, :func:`propagate_gbd`, and
    :func:`propagate_hfpi`.  This is the recommended entry point for
    new code; the trio ``propagate_huygens_fresnel_freespace`` /
    ``_with_opl_callable`` / ``_through_prescription`` is retained
    for specialised use cases.

    Internally delegates to
    :func:`propagate_huygens_fresnel_freespace`.
    """
    return propagate_huygens_fresnel_freespace(
        E_in, z, wavelength, dx, **kwargs)


def propagate_huygens_fresnel_freespace(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    *,
    dy: Optional[float] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Free-space Huygens-Fresnel propagation.

    This IS :func:`lumenairy.propagation.rayleigh_sommerfeld_propagate`
    -- a thin delegation, re-exported here for API consistency with the
    other ``hf.*`` entry points -- plus the optional output-grid
    resample below.

    The kernel applied is the RS-I Green's function
    ``(z/(2 pi r^2))(1/r - ik) exp(ikr)``, whose leading term is
    ``cos(theta)/(i lambda r)``.  There is no Van Vleck factor on this
    path (audit K15).

    v5.3 (AUDIT_V5_2_5 P1-1 closure): ``output_shape`` and
    ``output_dx`` kwargs are accepted and honored via a post-kernel
    ``resample_field`` step (shared with the v5.2.3 MHS
    substantive-resampling path in
    :func:`~lumenairy.propagators.mhs.prescription_subdomain`, via
    :func:`_resample_preserving_window_power`).  The underlying
    ``rayleigh_sommerfeld_propagate`` kernel returns on the input
    grid; the resample step bridges to the caller-requested output
    grid.  v5.2.5 routed these kwargs from the dispatcher into this
    function but the v5.2.5 pass-through to
    ``rayleigh_sommerfeld_propagate`` raised ``TypeError`` because
    the RS kernel does not accept either kwarg.  v5.3 fixes the
    pass-through by handling the resample here instead of
    forwarding to the kernel.

    Return type
    -----------
    When neither ``output_shape`` nor ``output_dx`` is given (the
    common pass-through case), returns the bare ``ndarray`` -- same
    contract as the underlying RS kernel.

    When ``output_shape`` or ``output_dx`` IS given (the v5.3
    resample path), returns a ``(E_out, dx_out)`` 2-tuple matching
    the ``resample_field`` contract -- the call has changed the
    grid spacing and the caller needs to know the new pitch.
    """
    from .propagation import rayleigh_sommerfeld_propagate
    E_native = rayleigh_sommerfeld_propagate(
        E_in, z, wavelength, dx, dy=dy, **kwargs,
    )
    if output_shape is None and output_dx is None:
        return E_native

    # Resample to the requested output grid (matches MHS pattern).
    target_dx = float(output_dx) if output_dx is not None else float(dx)
    if output_shape is None:
        # Same shape as input; only the pitch changed.
        N_out = E_native.shape[-1]
    else:
        if len(output_shape) != 2:
            raise ValueError(
                f"propagate_huygens_fresnel_freespace: output_shape "
                f"must be a (Ny, Nx) tuple of two ints; got "
                f"{output_shape!r}.")
        Ny, Nx = int(output_shape[0]), int(output_shape[1])
        if Ny != Nx:
            raise ValueError(
                f"propagate_huygens_fresnel_freespace: non-square "
                f"output_shape ({Ny}, {Nx}) not supported -- the "
                f"underlying resample_field assumes a square "
                f"target grid.  Either request a square shape or "
                f"call ``rayleigh_sommerfeld_propagate`` + your own "
                f"resampler directly.")
        N_out = Ny

    # v5.4 (audit P2): same-shape short-circuit -- mirrors the strict
    # absolute test at ``mhs.py``'s maslov branch.  If the input grid
    # already matches the requested target grid (same N and same dx),
    # skip ``resample_field`` entirely: ``map_coordinates`` introduces a
    # small power drift at the edges even when the grids nominally match,
    # and the short-circuit guarantees a bit-for-bit native-kernel return.
    #
    # K21 (audit 2026-09-11): the gate was
    # ``np.isclose(dx, target_dx, rtol=1e-12)``, which still carries
    # numpy's default ``atol=1e-8`` -- 10 nm in this library's METRES.
    # A 0.5 % pitch change at 1 um and a 10 % change at 100 nm both
    # compared equal, so the un-resampled field came back LABELLED with
    # the requested pitch: the "wrong sampling metadata" class the
    # dispatcher raises for elsewhere.  A pure relative test has no
    # absolute floor to trip over.
    N_in = int(E_native.shape[-1])
    if N_in == int(N_out) and abs(float(dx) - float(target_dx)) <= (
            1e-12 * abs(float(dx))):
        return E_native, target_dx

    # K11: restore the interpolation drift, not the cropped light.
    return _resample_preserving_window_power(
        E_native, dx, target_dx, N_out,
        fn_name='propagate_huygens_fresnel_freespace')


def propagate_huygens_fresnel_with_opl_callable(
    E_in: np.ndarray,
    *,
    opl_fn: Callable,
    output_grid_x: np.ndarray,
    output_grid_y: np.ndarray,
    input_grid_dx: float,
    apply_van_vleck: bool = True,
    finite_diff_step: float = 1e-6,
    chunk_output: Optional[int] = None,
) -> np.ndarray:
    """Evaluate the HF integral for a user-supplied OPL callable
    ``Phi(s1, s2)``.

    Computes::

        E(s2) = sum over input pixels of
                E_in(s1) * sqrt(|det d2 Phi / d s1 d s2|)
                * exp(2 pi i Phi(s1, s2)) * (d s1)^2

    where the cross-Hessian determinant is evaluated by central
    differences on the supplied callable.

    Units contract -- ``opl_fn`` MUST return WAVES
    -------------------------------------------------
    ``opl_fn(s1x, s1y, s2x, s2y)`` takes input-plane coordinates as
    arrays (broadcast over the whole input grid) and output-plane
    coordinates as scalars -- or, since v5.46, as ``(n, 1, 1)`` arrays
    that broadcast a whole BATCH of output pixels against the input grid
    (see ``chunk_output``).  Write ``opl_fn`` as pure array expressions
    of its four arguments and both forms work unchanged; a callable that
    cannot take the batched form is detected by a one-shot probe and
    served by the historical per-pixel path with a ``RuntimeWarning``.
    All coordinates are in **metres**, and the return is the
    optical path ``Phi`` in **WAVES** (cycles, i.e. OPL_metres /
    wavelength).  The kernel applied here is ``exp(2j*pi*Phi)``, so a
    callable that returns metres is wrong by the factor ``1/wavelength``
    (~1e6 at visible / near-IR wavelengths) -- it produces an almost
    phase-free integrand and a silently wrong field.  Convert inside the
    callable::

        def opl_fn(s1x, s1y, s2x, s2y):          # WAVES, not metres
            r = np.sqrt((s1x - s2x)**2 + (s1y - s2y)**2 + z*z)
            return r / wavelength

    The Van Vleck factor inherits that convention: with ``Phi`` in waves
    the cross-Hessian entries scale as ``1/(wavelength*z)`` and its
    determinant as ``(1/(wavelength*z))**2`` (e.g. exactly
    ``(2.0e7)**2`` for the Fresnel OPL at ``z=50 mm``,
    ``wavelength=1 um``).  The sibling
    :func:`propagate_hf_chebyshev_quadrature` and the
    ``fit_hf_polynomials`` / ``fit_canonical_polynomials`` containers it
    consumes use the same waves convention (``phi = opd / wavelength``).

    Parameters
    ----------
    finite_diff_step : float, default ``1e-6`` (metres)
        Central-difference step ``h`` used for the Van Vleck
        cross-Hessian ``d2 Phi / d s1 d s2``.  The stencil is
        second-order accurate, so its error scales as ``h^2`` in the
        truncation term and as ``eps/h^2`` in the round-off term; for a
        waves-valued ``Phi`` of order ``z/wavelength`` the round-off
        term dominates below ~1e-7 m.

        Measured on an EXACT-QUADRATIC (Fresnel) OPL oracle with
        ``z=50 mm``, ``wavelength=1 um``: the recovered ``sqrt|det|``
        amplitude is in error by -9.05e-2 at ``h=1e-9`` (the pre-v5.30
        default -- essentially all round-off), -1.06e-5 at 1e-7,
        -2.53e-8 at the 1e-6 default, and -1.6e-9 at 1e-5.  End-to-end
        against exact Fresnel quadrature on the same discretisation the
        amplitude error falls from 1.56e-2 to 8.3e-9.

        K24 (audit 2026-09-11): those numbers are specific to a
        quadratic ``Phi``, where the 4th-order truncation term vanishes
        IDENTICALLY and only round-off survives -- so they flatter the
        default.  On the exact SPHERICAL OPL
        (``Phi = sqrt(u^2+v^2+z^2)/lambda``, z = 50 mm, lambda = 1 um,
        u = 2 mm, v = 1 mm) against the closed form
        ``sqrt|det| = cos(theta)/(lambda r)``:

        =========  =====================
        h [m]      rel. err of sqrt|det|
        =========  =====================
        1e-9       +2.2605e-01
        1e-8       +1.5337e-03
        1e-7       +2.5030e-05
        1e-6 (default)  +2.3578e-07
        1e-5 (optimum)  -3.7685e-08
        1e-4       -3.9720e-06
        1e-3       -3.9690e-04
        =========  =====================

        i.e. the optimum moves a decade and the default is ~9x off it.
        Both are ~4 decades below the quadrature's own ~1e-3
        discretisation floor, so this is a documentation point, not a
        numerical one -- but do not read "-2.53e-8" as the accuracy you
        get on a non-quadratic ``Phi``.

        ``h`` is an absolute step in metres: if you
        work at a wildly different length scale (e.g. mm-scale grids or
        a ``Phi`` with structure finer than a micron) scale it with your
        transverse feature size -- a good rule of thumb is
        ``h ~ sqrt(eps_rel) * L`` with ``L`` the scale over which
        ``d2 Phi / d s1 d s2`` varies.

    .. versionchanged:: 5.30
        ``finite_diff_step`` default 1e-9 -> 1e-6 m (audit P3,
        ``AUDIT_ADVERSARIAL_CODEBASE_2026_07_25``): at 1e-9 the
        cross-Hessian stencil was almost pure round-off (9.05% low
        amplitude at the origin, up to 1.56e-2 spatially-varying
        end-to-end error vs exact Fresnel quadrature).  Callers who
        passed ``finite_diff_step`` explicitly are unaffected.

    .. versionchanged:: 5.30
        ``wavelength`` (audit P7) is **REMOVED**.  It was a required
        keyword that the body never read; v5.30 first made it optional +
        ``DeprecationWarning``, and the W5 shim-removal wave deletes it in
        the same release rather than carrying an inert keyword to v5.32.
        The OPL callable's return is in WAVES (see the units contract
        above), so no wavelength is needed here.  Migration: drop the
        kwarg; if your ``opl_fn`` returns METRES, divide by the wavelength
        inside ``opl_fn`` -- passing it here never did that.

    chunk_output : int, optional
        Number of OUTPUT pixels evaluated per vectorised batch.  ``None``
        (default) sizes the batch so one ``(n_chunk, Ny_in, Nx_in)``
        float64 working array is about 128 KB; pass an explicit int to
        pin it, or ``1`` to force strictly-per-pixel evaluation.  The
        returned field is **bit-identical** for every value (verified at
        three ``(N_in, N_out)`` pairs with and without Van Vleck).

        .. versionchanged:: 5.46
            **Un-deprecated and given the meaning its name always
            promised** (audit K22), and honestly sized.  ``opl_fn`` is
            evaluated over the whole input grid, 17 times per output
            pixel (``Phi`` plus the 16 cross-Hessian stencil corners), so
            the cost is ``O(N_in^2 * N_out^2)`` -- measured here 0.40 /
            1.45 / 9.56 ms per output pixel at ``N_in`` = 64 / 128 / 256,
            i.e. ~10 minutes for a full 256x256 output.  Batching the
            output pixels was expected to amortise those 17 evaluations;
            in fact it only removes Python-level DISPATCH, which is a
            small share of a memory-bandwidth-bound computation.
            Measured ladder (medians of 5 interleaved runs, ms/px):

            ========  =====  =====  =====  =====  ======  ======
            N_in      c=1    c=2    c=4    c=8    c=16    c=32
            ========  =====  =====  =====  =====  ======  ======
            64        0.401  0.360  0.323  0.328   0.449   1.634
            128       1.450  1.465  2.099  7.077   6.422   6.419
            256       9.563  30.45  28.02  29.50  30.342  28.815
            ========  =====  =====  =====  =====  ======  ======

            so the best available gain is **1.24x at N_in = 64**, and a
            batch whose working array leaves the L2 cache is 3-5x
            slower.  The auto rule targets 128 KB, which selects the
            measured optimum at each of those three sizes.  If you need a
            full-plane HF output, use a Fourier route instead: for free
            space :func:`propagate_huygens_fresnel_freespace` is ~5e4x
            faster at the same ~1e-3 accuracy, and band-limited ASM is
            equal or better on a Gaussian while delivering the whole
            plane.  This quadrature earns its keep only for a
            genuinely non-shift-invariant ``Phi``.

    .. versionchanged:: 5.17
        ``chunk_output`` (audit P3-57) was deprecated as a no-op: the
        outer "chunk" loop only partitioned an identical per-pixel inner
        loop, so no value changed the result or the runtime.  v5.46
        restores it as a genuine batch size (above).
    """
    # v5.30 (audit P7, W5 removal): ``wavelength`` was a REQUIRED keyword
    # that the body never read -- the OPL callable returns waves, so the
    # kernel ``exp(2j*pi*Phi)`` is already dimensionless.  It is now GONE
    # rather than inert: consuming it (dividing an assumed-metres Phi by
    # wavelength) would have silently broken every existing
    # waves-returning callable by a factor of ~1e6, so there was never a
    # future in which the keyword acquired a meaning.
    if chunk_output is not None:
        if int(chunk_output) != chunk_output or int(chunk_output) < 1:
            raise ValueError(
                f"propagate_huygens_fresnel_with_opl_callable: "
                f"chunk_output must be a positive integer number of output "
                f"pixels per batch (or None to size it from the RAM "
                f"budget); got {chunk_output!r}.")
        chunk_output = int(chunk_output)
    xp = array_namespace(E_in)

    Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
    # v4.12.0 (B1-10): switch from cell-centred `(arange(N) - N/2 + 0.5)*dx`
    # to pixel-centred `(arange(N) - N/2)*dx`, matching the library-wide
    # convention (ASM, Fresnel, RS, sources, ``apply_fresnel_curvature``).
    # The OPL callable is evaluated on input-plane coordinates so they
    # must match the grid that ``E_in`` was sampled on by upstream
    # propagators / source builders.
    s1_x = (xp.arange(Nx_in, dtype=xp.float64) - Nx_in / 2) * input_grid_dx
    s1_y = (xp.arange(Ny_in, dtype=xp.float64) - Ny_in / 2) * input_grid_dx
    S1X, S1Y = xp.meshgrid(s1_x, s1_y, indexing='xy')

    Ny_out = output_grid_y.shape[0]
    Nx_out = output_grid_x.shape[0]
    # 4.10: force a complex dtype so a real-valued E_in (e.g. a pure
    # intensity mask) doesn't silently strip the imaginary part of the
    # HF kernel during the multiply.  Pre-4.10 produced a real-valued
    # "field" with the imaginary half summed into nothing.
    if xp.iscomplexobj(E_in):
        out_dtype = E_in.dtype
    elif E_in.dtype == xp.float64:
        out_dtype = xp.complex128
    else:
        out_dtype = xp.complex64
    out = xp.zeros((Ny_out, Nx_out), dtype=out_dtype)
    pixel_area = input_grid_dx * input_grid_dx
    h = float(finite_diff_step)

    n_out = Ny_out * Nx_out
    flat_x = xp.reshape(xp.broadcast_to(output_grid_x[None, :],
                                        (Ny_out, Nx_out)), (-1,))
    flat_y = xp.reshape(xp.broadcast_to(output_grid_y[:, None],
                                        (Ny_out, Nx_out)), (-1,))

    # K22: size the output batch by CACHE, not by RAM.  With the output
    # coordinates broadcast to ``(n_chunk, 1, 1)`` every intermediate is
    # ``(n_chunk, Ny_in, Nx_in)``, and the batched form does exactly the
    # same flops as the per-pixel loop -- it only trades Python-level
    # dispatch for larger temporaries.  Measured on this workstation
    # (medians of 5 interleaved runs, ms per output pixel, Van Vleck on,
    # complex128, exact spherical OPL):
    #
    #   N_in=64  c=1 0.401  c=2 0.360  c=4 0.323  c=8 0.328  c=16 0.449  c=32 1.634
    #   N_in=128 c=1 1.450  c=2 1.465  c=4 2.099  c=8 7.077  c=16 6.422
    #   N_in=256 c=1 9.563  c=2 30.449 c=4 28.018 c=8 29.497
    #
    # i.e. the dispatch saving is real but small (best 1.24x at N_in=64)
    # and is swamped as soon as one working array leaves the L2 cache --
    # 3-5x SLOWER at N_in >= 128 with a large batch.  Target a ~128 KB
    # working array, which reproduces the measured optimum at all three
    # sizes (N_in=64 -> 4, N_in=128 -> 1, N_in=256 -> 1).
    if chunk_output is None:
        _grid_bytes = max(int(Ny_in) * int(Nx_in) * 8, 1)
        n_chunk = max(1, _HF_CHUNK_TARGET_BYTES // _grid_bytes)
        n_chunk = min(n_chunk, n_out)
    else:
        n_chunk = min(int(chunk_output), n_out)

    # The batched form hands ``opl_fn`` output coordinates of shape
    # ``(n, 1, 1)`` instead of Python scalars.  Every numpy-expression
    # callable broadcasts that for free, but the pre-v5.46 contract said
    # "scalars", so a callable that calls ``math.sqrt`` / ``float()`` on
    # them, or returns something that does not broadcast, must still
    # work.  Probe once with a batch of one and fall back to the
    # historical strictly-per-pixel evaluation if it does not.
    _batched = n_chunk > 1
    if _batched:
        _probe_err = None
        try:
            _probe = opl_fn(S1X, S1Y,
                            xp.reshape(flat_x[0:1], (-1, 1, 1)),
                            xp.reshape(flat_y[0:1], (-1, 1, 1)))
            _batched = tuple(np.shape(_probe)) == (1, Ny_in, Nx_in)
            del _probe
        except (TypeError, ValueError, IndexError, AttributeError,
                ZeroDivisionError, OverflowError, NotImplementedError) as _exc:
            # The realistic ways a caller's ``opl_fn`` can reject an
            # (n, 1, 1) output coordinate: ``float()``/``math.*`` on it
            # (TypeError), a shape mismatch (ValueError), indexing it
            # (IndexError), reaching for a scalar attribute
            # (AttributeError), or an arithmetic path that only works
            # element-wise.  Anything else is a real bug in the callable
            # and should surface, not be absorbed into a silent fallback.
            _batched = False
            _probe_err = _exc
        if not _batched:
            n_chunk = 1
            warnings.warn(
                f"propagate_huygens_fresnel_with_opl_callable: opl_fn did "
                f"not broadcast output coordinates of shape (n, 1, 1) over "
                f"the input grid "
                f"({'raised ' + type(_probe_err).__name__ + ': ' + str(_probe_err) if _probe_err is not None else 'returned a non-broadcast shape'}), "
                f"so evaluation falls back to one output pixel at a time -- "
                f"17 full-input-grid opl_fn calls per output pixel, i.e. "
                f"O(N_in^2 * N_out^2) (measured 109 ms/px at N_in=256).  "
                f"Write opl_fn as pure array expressions of its four "
                f"arguments (no ``float()`` / ``math.*`` on s2x, s2y) to get "
                f"the vectorised path, or pass chunk_output=1 to silence "
                f"this.", RuntimeWarning, stacklevel=2)

    _is_jax = is_jax_array(E_in)
    out_flat = None if _is_jax else out.reshape(-1)
    # JAX: ``out.at[iy, ix].set(...)`` allocates a fresh (Ny_out, Nx_out)
    # array per write, so accumulate the batches in a Python list and
    # assemble once (audit K22 / the PROP-HF P3 on the JAX branch).
    _jax_parts = [] if _is_jax else None

    for k0 in range(0, n_out, n_chunk):
        k1 = min(n_out, k0 + n_chunk)
        # Shape (n, 1, 1) so every ``opl_fn`` call broadcasts the whole
        # input grid against the whole output batch at once: the same
        # flops as the per-pixel loop, but 17 Python-level dispatches and
        # temporary allocations per BATCH instead of per pixel.
        if _batched:
            s2x = xp.reshape(flat_x[k0:k1], (-1, 1, 1))
            s2y = xp.reshape(flat_y[k0:k1], (-1, 1, 1))
        else:
            # Historical contract: Python scalars, one output pixel.
            s2x = (float(flat_x[k0]) if hasattr(flat_x[k0], '__float__')
                   else flat_x[k0])
            s2y = (float(flat_y[k0]) if hasattr(flat_y[k0], '__float__')
                   else flat_y[k0])

        phi = opl_fn(S1X, S1Y, s2x, s2y)

        if apply_van_vleck:
            pxx = (
                opl_fn(S1X + h, S1Y, s2x + h, s2y)
                - opl_fn(S1X + h, S1Y, s2x - h, s2y)
                - opl_fn(S1X - h, S1Y, s2x + h, s2y)
                + opl_fn(S1X - h, S1Y, s2x - h, s2y)
            ) / (4 * h * h)
            pyy = (
                opl_fn(S1X, S1Y + h, s2x, s2y + h)
                - opl_fn(S1X, S1Y + h, s2x, s2y - h)
                - opl_fn(S1X, S1Y - h, s2x, s2y + h)
                + opl_fn(S1X, S1Y - h, s2x, s2y - h)
            ) / (4 * h * h)
            pxy = (
                opl_fn(S1X + h, S1Y, s2x, s2y + h)
                - opl_fn(S1X + h, S1Y, s2x, s2y - h)
                - opl_fn(S1X - h, S1Y, s2x, s2y + h)
                + opl_fn(S1X - h, S1Y, s2x, s2y - h)
            ) / (4 * h * h)
            pyx = (
                opl_fn(S1X, S1Y + h, s2x + h, s2y)
                - opl_fn(S1X, S1Y + h, s2x - h, s2y)
                - opl_fn(S1X, S1Y - h, s2x + h, s2y)
                + opl_fn(S1X, S1Y - h, s2x - h, s2y)
            ) / (4 * h * h)
            det = pxx * pyy - pxy * pyx
            density = xp.sqrt(xp.abs(det))
        else:
            density = 1.0

        # 4.10: cast to the complex output dtype, not E_in.dtype
        # (which may be real -- see comment above the out-array
        # allocation).  Pre-4.10 a real E_in stripped the imag
        # part of the kernel before the multiply.
        #
        # K22: for a complex64 caller, fold ``Phi`` modulo one cycle in
        # float64 BEFORE the float32 cast and build the exponential
        # directly in single precision -- the same mod-2*pi mitigation
        # the ASM transfer function uses.  ``Phi`` is in WAVES and is of
        # order ``z/wavelength`` (~1e5 at z = 50 mm, 1 um), which float32
        # cannot carry to sub-cycle accuracy; reducing first makes the
        # single-precision path both cheaper (no complex128 grid built
        # and thrown away) and accurate.  complex128 is unchanged.
        if np.dtype(out_dtype) == np.complex64:
            kernel = xp.exp(
                (2j * float(np.pi))
                * (phi - xp.floor(phi)).astype(np.float32))
        else:
            kernel = xp.exp(2j * float(np.pi) * phi).astype(out_dtype)
        integrand = E_in * density * kernel
        # Reduce over the INPUT-grid axes only, leaving one value per
        # output pixel in the batch (a 0-d value on the scalar path).
        out_values = xp.sum(integrand, axis=(-2, -1)) * pixel_area
        if _is_jax:
            _jax_parts.append(xp.reshape(out_values, (-1,)))
        else:
            out_flat[k0:k1] = out_values

    if _is_jax:
        out = xp.reshape(xp.concatenate(_jax_parts), (Ny_out, Nx_out))

    # 4.11.2: apply the Van Vleck-Morette asymptotic prefactor
    # (2π)^(-d/2)·i^(-d/2) for d=2, which is -i/(2π).  The 2π part is
    # absorbed in the Phi convention (phase = exp(2πi Phi)), leaving
    # the global ``-1j`` Maslov factor.  The sibling
    # :func:`propagate_hf_chebyshev_quadrature` already applies this
    # (v4.10 C-AS-2 fix); without it the OPL-callable variant is 90°
    # out of phase with the Fresnel kernel ``1/(iλz) = -i/(λz)``,
    # producing incoherent superposition when stacking with
    # ASM/Fresnel outputs.
    out = out * (-1j)
    return out


# ============================================================================
# Prescription-aware HF (Van-Vleck-corrected, via fit_canonical_polynomials)
# ============================================================================

def propagate_huygens_fresnel_through_prescription(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    method: str = 'asymptotic',
    source_lg_p_max: int = 3,
    source_lg_ell_max: int = 3,
    source_lg_amp_threshold: float = 1e-6,
) -> np.ndarray:
    """End-to-end Van-Vleck-corrected HF through a sequential
    prescription.

    Two evaluation modes:

    * ``method='asymptotic'`` (default) -- evaluates the HF
      integral in the leading-order saddle-point (Van Vleck)
      asymptotic limit by routing to
      :func:`lumenairy.propagators.asymptotic.propagate_modal_asymptotic`.
      Closed-form, fast (~milliseconds), accurate for most
      well-conditioned refractive systems and for output planes
      that are not inside a fold caustic.

    * ``method='direct'`` -- direct 2-D quadrature of the HF integral
      using the Chebyshev polynomial fit of ``Phi(s2, v2)`` and
      ``s1(s2, v2)`` from
      :func:`lumenairy.propagators.asymptotic.fit_canonical_polynomials`.
      Includes the Van Vleck density factor
      ``sqrt(|det d2 Phi / d s1 d s2|)``.  Slower (~seconds at
      moderate output-grid sizes) but does not assume the saddle-
      point approximation.

    Parameters
    ----------
    E_in : array (Ny, Nx) complex
        Source-plane field.
    dx : float
        Source-grid pitch (m).
    prescription : dict
    wavelength : float
    output_shape : (int, int), optional
        Output-grid (Ny, Nx) shape.  Defaults to ``E_in.shape``.  v5.2
        rename of the legacy ``output_grid`` kwarg (AUDIT_V4_13_1 Part 2
        P1-A); the dispatcher's :func:`propagate(output_grid=...)`
        contract carries the ``(N_out, dx_out)`` semantics instead.
    output_grid : (int, int), optional
        Deprecated v5.2 alias for ``output_shape``.  Emits a
        ``DeprecationWarning``.
    output_dx, output_centre : grid geometry
    source_box_half, pupil_box_half : float
        Half-widths of the source / pupil sampling boxes for the
        polynomial fit.
    n_field, n_pupil : int
        Per-axis Chebyshev-node grid sizes for the fit.
    poly_order : int
        Total-degree truncation of the Chebyshev fit.
    method : str
        ``'asymptotic'`` or ``'direct'``.

    Returns
    -------
    array (Ny, Nx) complex
        Output-plane complex field.

    .. versionchanged:: 5.2
        ``output_grid`` -> ``output_shape`` rename (AUDIT_V4_13_1 Part 2
        P1-A).
    """
    import numpy as _np

    from .asymptotic import (
        fit_canonical_polynomials,
        propagate_modal_asymptotic,
    )

    # v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): ``output_grid`` is now
    # the deprecated spelling of ``output_shape``; see module helper.
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_huygens_fresnel_through_prescription',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx

    if method == 'asymptotic':
        # Build the canonical polynomial fit.
        fit = fit_canonical_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field,
            n_pupil=n_pupil,
            poly_order=poly_order,
        )

        # Evaluate the modal asymptotic propagator on an output
        # grid.  4.11.2: build the source LG-mode amplitudes by
        # projecting ``E_in`` onto the LG basis (truncated at
        # ``source_lg_p_max`` / ``source_lg_ell_max``).  Pre-4.11.2
        # replaced ``E_in`` with a unit fundamental Gaussian and
        # produced a Gaussian output regardless of the input
        # field's structure -- a structured source (e.g. a vortex
        # beam, off-axis Gaussian, Airy pattern) was silently
        # discarded.
        from ..analysis.core import beam_d4sigma
        from .asymptotic import decompose_lg
        cx, cy = output_centre
        # v4.12.0 (B1-10): pixel-centred grid (drop the `+0.5`),
        # matches ASM/Fresnel/RS/sources so subsequent through-focus
        # scans and overlays stay coherent across propagator families.
        out_x = (_np.arange(Nx) - Nx / 2) * output_dx + cx
        out_y = (_np.arange(Ny) - Ny / 2) * output_dx + cy
        OX, OY = _np.meshgrid(out_x, out_y, indexing='xy')

        # Estimate source waist from input field.  HF-1: beam_d4sigma returns
        # a (d4x, d4y) TUPLE, so the prior float(d4) raised TypeError on EVERY
        # call and control always fell into the except-branch fallback -- the
        # data-driven estimate never ran.  Unpack the x-width.
        #
        # ``decompose_lg`` / ``propagate_modal_asymptotic`` take ``w_s`` as the
        # LG-basis 1/e^2 radius (envelope ``exp(-r^2/w_s^2)``).  For a Gaussian
        # the D4sigma diameter equals ``2 * w`` (1/e^2 radius), so the matching
        # waist is ``d4x / 2`` -- NOT the audit's ``0.25 * d4x`` (which is the
        # second-moment sigma == w/2, half the true waist: it under-fills the
        # fundamental LG mode and pushes energy into higher orders).  Since the
        # ``/ 4`` value was never actually reached (the TypeError always fired)
        # there is no behaviour to preserve.  The fallback is likewise promoted
        # from the raw second-moment sigma to ``2 * sigma`` so it, too, is the
        # 1/e^2 radius and stays consistent with the primary estimate.
        try:
            d4x, _d4y = beam_d4sigma(E_in, dx=dx)
            w_s = float(d4x) / 2.0
        except (TypeError, ValueError, RuntimeError, ZeroDivisionError):
            # beam_d4sigma can raise: TypeError on non-array E_in,
            # ValueError on empty / wrong-rank inputs, RuntimeError
            # when the moments diverge, ZeroDivisionError on a zero
            # total-power normalisation.  Fall back to twice the explicit
            # second-moment sigma (== the 1/e^2 radius for a Gaussian),
            # which is numerically stable for any finite |E|^2 distribution.
            #
            # v5.30 (audit P14): the x-axis grid MUST come from the INPUT
            # field's own dimensions.  Pre-v5.30 this line used ``Nx`` --
            # the OUTPUT grid width resolved above -- against
            # ``|E_in|**2``, so whenever ``output_shape != E_in.shape``
            # the fallback died with an uncaught, unrelated-looking
            # ``ValueError: operands could not be broadcast together with
            # shapes (32,32) (64,)`` instead of returning a waist.  (The
            # pitch was already the input ``dx``, so mixing in the output
            # count was wrong even when the two happened to be equal.)
            _Nx_src = E_in.shape[-1]
            _I_src = _np.abs(E_in) ** 2
            _x_src = (_np.arange(_Nx_src) - _Nx_src / 2) * dx
            w_s = 2.0 * float(_np.sqrt(
                _np.sum(_I_src * _x_src ** 2)
                / max(_np.sum(_I_src), 1e-30)))
        if w_s <= 0 or not _np.isfinite(w_s):
            w_s = source_box_half / 2

        # Build input-plane coordinates for LG decomposition.
        # v4.12.0 (B1-10): pixel-centred grid (drop the `+0.5`).  The
        # input field ``E_in`` was sampled by upstream propagators /
        # source builders on the library-standard `(arange(N) - N/2)*dx`
        # grid; the LG decomposition must use the same coordinates so
        # the projected mode amplitudes correctly represent ``E_in``.
        Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
        in_x = (_np.arange(Nx_in) - Nx_in / 2) * dx
        in_y = (_np.arange(Ny_in) - Ny_in / 2) * dx
        IX, IY = _np.meshgrid(in_x, in_y, indexing='xy')
        # Decompose E_in onto LG modes at the source plane.
        try:
            E_in_np = _np.asarray(E_in)
            source_lg = decompose_lg(
                E_in_np, IX, IY, w_s,
                source_lg_p_max, source_lg_ell_max,
                cx=0.0, cy=0.0,
            )
        except (TypeError, ValueError, RuntimeError) as _exc:
            # decompose_lg fails on TypeError (non-array E_in or bad
            # dtypes), ValueError (shape mismatch with IX/IY or w_s<=0),
            # and RuntimeError (singular projection matrix on a
            # degenerate field).  Surface the failure so the silent
            # plane-wave fallback below doesn't hide a real upstream
            # bug -- the asymptotic propagator is essentially useless
            # without a valid LG decomposition.
            import warnings as _w
            _w.warn(
                f"propagate_hf: source LG decomposition failed "
                f"({type(_exc).__name__}: {_exc}); falling back to a "
                f"single (p=0, l=0) plane-wave mode.  This may "
                f"indicate a bug; please report at "
                f"https://github.com/travaj24/LumenAiry/issues",
                RuntimeWarning, stacklevel=2)
            source_lg = {(0, 0): 1.0 + 0.0j}
        # Drop amplitudes below threshold (sparsity speedup).
        max_amp = max((abs(a) for a in source_lg.values()), default=1.0)
        if max_amp > 0:
            source_lg = {k: v for k, v in source_lg.items()
                         if abs(v) >= source_lg_amp_threshold * max_amp}
        if not source_lg:
            source_lg = {(0, 0): 1.0 + 0.0j}
        # Pupil defaults to plane-wave (single-mode).
        pupil_lg = {(0, 0): 1.0}

        # Source point at origin (LG basis is centred there).
        source_point = (0.0, 0.0)
        w_p = pupil_box_half
        v2_centre = (0.0, 0.0)

        # propagate_modal_asymptotic expects per-pixel s2 grids.
        # OX, OY were built above on the output_grid_xy / output_dx
        # spec and are the right grids to sample on.
        return propagate_modal_asymptotic(
            fit,
            source_amplitudes=source_lg,
            pupil_amplitudes=pupil_lg,
            source_point=source_point,
            w_s=w_s,
            w_p=w_p,
            v2_centre=v2_centre,
            s2_grid_x=OX,
            s2_grid_y=OY,
        )

    if method == 'direct':
        from .asymptotic import (
            fit_hf_polynomials,
            propagate_hf_chebyshev_quadrature,
        )
        # Build the HF-form polynomial fit Phi(s1, s2).
        hf_fit = fit_hf_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field,
            n_pupil=n_pupil,
            poly_order=poly_order,
        )

        # Build input and output grids.
        # v4.12.0 (B1-10): pixel-centred grid (drop the `+0.5`),
        # matches ASM/Fresnel/RS/sources.
        Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
        in_x = (_np.arange(Nx_in) - Nx_in / 2) * dx
        in_y = (_np.arange(Ny_in) - Ny_in / 2) * dx
        cx, cy = output_centre
        out_x = (_np.arange(Nx) - Nx / 2) * output_dx + cx
        out_y = (_np.arange(Ny) - Ny / 2) * output_dx + cy

        return propagate_hf_chebyshev_quadrature(
            hf_fit, E_in,
            input_grid_x=in_x, input_grid_y=in_y,
            output_grid_x=out_x, output_grid_y=out_y,
            apply_van_vleck=True,
        )

    raise ValueError(
        f"propagate_huygens_fresnel_through_prescription: method must be "
        f"'asymptotic' or 'direct', got {method!r}.")


__all__ = [
    # K24 (audit 2026-09-11): ``propagate_huygens_fresnel`` -- the entry
    # point this module's own docstring calls "the recommended entry
    # point for new code" -- was absent from ``__all__``, so
    # ``from lumenairy.propagators.hf import *`` did not give you the
    # function the module tells you to use.  (It was always reachable as
    # ``lumenairy.propagate_huygens_fresnel`` because the package
    # ``__init__`` imports it by name; an ``__all__`` integrity gap, not
    # a breakage.)
    'propagate_huygens_fresnel',
    'propagate_huygens_fresnel_freespace',
    'propagate_huygens_fresnel_with_opl_callable',
    'propagate_huygens_fresnel_through_prescription',
]
