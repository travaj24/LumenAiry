"""
Rayleigh-Sommerfeld propagator
==============================

v5.1.0 Agent C split: extracted from ``propagation.py``.  Contains the
RS Green's-function convolution implementation (first RS solution,
optional Matsushima bandlimit on the padded FFT'd kernel).

Public surface: :func:`rayleigh_sommerfeld_propagate` (re-exported by
``propagation.py`` and at the top level) and
:func:`rs_alias_free_distance`, the sampling distance its
``kernel='auto'`` branches on -- the number a caller needs in order to
choose ``z``, ``N`` or ``dx`` for an RS step, so it is not an internal.

Author:  Andrew Traverso
"""

# Version history for this module: ``docs/history/lumenairy.propagators.rs.md``.

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backend import array_namespace
from . import fft_infra as _state
from .fft_infra import (
    CUPY_AVAILABLE,
    _fft2,
    _h_cache_lookup,
    _h_cache_store,
    _ifft2,
    _is_cupy_array,
    _validate_propagator_inputs,
)

__all__ = [
    'rayleigh_sommerfeld_propagate',
    'rs_alias_free_distance',
]

#: Outer band of the padded window the wrap-around guard measures, as a
#: fraction ``1/_RS_WRAP_RING_BAND`` of the padded extent on each side
#: (verify V6).  1/8 keeps the reduction to one unpadded grid's worth of
#: elements while still sitting far enough from the input's own support
#: that a contained field contributes nothing.
_RS_WRAP_RING_BAND = 8

#: Fraction of the input power that may reach that outer band before the
#: ``kernel='transfer'`` branch warns about wrap-around (verify V6).  See
#: :func:`_warn_rs_transfer_wraparound` for the calibration.
_RS_WRAP_RING_FRACTION = 0.02

#: Points per band the wrap-around detector samples (verify V6).  The
#: ring fraction is a smooth spatial statistic, so a strided estimate
#: converges as 1/sqrt(n); 4096 points give ~1.6 % relative accuracy on
#: a noise-like field, against a threshold with 12 decades of margin on
#: the quiet side.  Makes the diagnostic O(1) in grid size.
_RS_WRAP_SAMPLE_BUDGET = 4096

#: Gauss-Legendre nodes per pixel AXIS in the ``kernel='spatial-integrated'``
#: build (audit K9 second half; Shen & Wang, reference [2]).  The pixel
#: integral of ``h`` is a smooth oscillatory integral whose phase sweeps at
#: most ``pi`` radians across one pixel anywhere the spatial kernel is legal
#: (that bound IS :func:`rs_alias_free_distance`), so a tensor
#: Gauss-Legendre rule converges geometrically in the node count.  Measured
#: relative L2 of the whole padded kernel against a 14-node build, at the
#: WORST legal geometry ``z = 2*N*dx**2/lambda`` where the phase sweep is at
#: its ``pi`` bound (2026-09-13, lambda = 633 nm):
#:
#: ========================  ========  ========  ========  ========  ========
#: grid                      n_g = 3   n_g = 4   n_g = 5   **n_g = 6**  n_g = 7
#: ========================  ========  ========  ========  ========  ========
#: N = 64,  dx = 2 um        1.96e-4   1.85e-6   1.12e-8   4.73e-11  1.81e-13
#: N = 128, dx = 1 um        1.48e-4   1.28e-6   7.06e-9   2.72e-11  1.03e-13
#: N = 128, dx = 2 um        1.96e-4   1.84e-6   1.11e-8   4.64e-11  2.56e-13
#: N = 256, dx = 1 um        1.48e-4   1.27e-6   7.04e-9   2.71e-11  1.57e-13
#: ========================  ========  ========  ========  ========  ========
#:
#: 6 puts the quadrature floor at 4.8e-11 or better -- seven decades below
#: the 1.1e-3 .. 4.7e-3 the kernel CHOICE is worth on a cell-constant
#: input -- for a kernel build measured at 5.5x / 8.7x / 10.9x the
#: point-sampled one at N = 128 / 256 / 512 (the folded build below; the
#: H cache pays it once per geometry, so a repeat call at the same
#: geometry costs the same as ``'spatial'``).
_RS_PIXEL_QUAD_NODES = 6


def _warn_rs_transfer_wraparound(E_conv, p_in, Ny2, Nx2, z, dx, dy,
                                 wavelength):
    """Warn when light has reached the rim of the padded window on the
    ``kernel='transfer'`` branch (verify V6).

    Multiplying by ``H`` is a CIRCULAR convolution on the ``2N`` padded
    grid, so whatever leaves the padded window re-enters on the opposite
    side instead of being discarded.  The ``'spatial'`` branch has the
    complementary behaviour -- it truncates ``h`` at the rim -- which is
    why ``kernel='auto'`` prefers it above the alias threshold.  Below
    that threshold ``'auto'`` has no alternative to offer, so the honest
    thing is to say when the assumption is breaking.

    The detector is the power in the outermost ``1/_RS_WRAP_RING_BAND``
    of the padded window after the multiply, as a fraction of the input
    power (the transfer kernel is unitary on the propagating set, so the
    padded total equals the input power to round-off; the two agreed to
    four digits on every fixture below).

    **Calibration.**  Exposed corner -- a band-limited random-phase
    screen at ``dx ~ lambda`` whose angular content fills 90 % of the
    grid's own representable range, against an 8x zero-padded linear
    convolution with the same transfer function:

    ==================  ==========  =========  ==================
    grid                z/z_crit    ring/P_in  relL2 vs the 8x pad
    ==================  ==========  =========  ==================
    N=64,  dx=1.0 lam   0.2         0.0003     5.8e-4
    N=64,  dx=1.0 lam   0.5         0.0722     2.4e-3
    N=64,  dx=1.0 lam   0.9         0.3035     3.3e-2
    N=128, dx=1.0 lam   0.5         0.0712     1.1e-3
    N=128, dx=1.0 lam   0.9         0.3012     2.3e-2
    N=64,  dx=0.6 lam   0.9         0.3609     1.7e-1
    N=128, dx=2.0 lam   0.9         0.2786     7.2e-3
    ==================  ==========  =========  ==================

    Counter-fixture -- a properly sampled Gaussian (``w0`` = 4-6 um,
    ``dx`` = 0.5-2 um) at the same three ``z/z_crit`` on four grids:
    ring/P_in is **5.5e-22 down to 9.3e-28**, worst case **8.1e-14**,
    with relL2 1e-14 .. 6e-11.  The 2 % threshold therefore sits **12
    decades above** anything a contained field produces and **0.5 to 1.3
    decades below** every case where the wrap is material.  It is not
    reachable at all for a properly sampled beam: leaving the padded
    window before ``z = 2 N dx^2 / lambda`` requires
    ``tan(theta) > lambda/(2 dx)``, i.e. exceeding the grid's own maximum
    representable angle.

    The warning states what was MEASURED (power has reached the rim), not
    a prediction of the error: the ring fraction and the error are
    monotonically related on any one grid but the constant differs
    between grids, so no error bound is claimed.

    **Cost.**  The ring and the reference power are both estimated on a
    STRIDED subsample with a fixed budget (``_RS_WRAP_SAMPLE_BUDGET``
    points per band), so the diagnostic is O(1) in grid size.  A full
    reduction over the ring measured +91 % of the call at N = 256 and
    +15 % at N = 1024 -- unacceptable for a diagnostic; the sampled form
    measures below 2 % at every size (table in the changelog).  The ring
    fraction is a smooth spatial statistic, so a stride estimates it to
    ~1/sqrt(n_samples) -- better than 2 % at the 4096-point budget,
    against a threshold with 12 decades of margin on the quiet side.
    """
    xp = array_namespace(E_conv)
    w = max(1, int(Ny2) // _RS_WRAP_RING_BAND)
    v = max(1, int(Nx2) // _RS_WRAP_RING_BAND)
    if p_in <= 0.0:
        return
    # The padded array holds the field CENTRED (the input was placed at
    # [N//2 : N//2+N]) and ``H`` is in natural order, so the convolution
    # stays in that layout: the outer band of the array IS the outer ring
    # of the physical window.  No fftshift.
    #
    # Stride both axes so each band contributes about the sample budget.
    st_y = max(1, int(np.sqrt(max(w, 1) * int(Nx2)
                             / _RS_WRAP_SAMPLE_BUDGET)))
    st_x = st_y

    def _p(block):
        # ``vdot`` on the flattened block avoids the |.|**2 temporary the
        # naive form allocates (two full arrays per band).
        flat = xp.reshape(block, (-1,))
        return float(xp.real(xp.vdot(flat, flat)))

    n_ring = 0
    ring = 0.0
    for blk in (E_conv[:w:st_y, ::st_x], E_conv[Ny2 - w::st_y, ::st_x],
                E_conv[w:Ny2 - w:st_y, :v:st_x],
                E_conv[w:Ny2 - w:st_y, Nx2 - v::st_x]):
        ring += _p(blk)
        n_ring += int(blk.size)
    if n_ring == 0:
        return
    # Scale the sampled sum back to a full-band sum: the ring holds
    # 2*w*Nx2 + 2*v*(Ny2 - 2w) elements.
    n_full = 2 * w * int(Nx2) + 2 * v * (int(Ny2) - 2 * w)
    frac = (ring * (n_full / n_ring)) / p_in
    if not (frac > _RS_WRAP_RING_FRACTION):
        return
    import warnings
    warnings.warn(
        f"rayleigh_sommerfeld_propagate: {100.0 * frac:.1f}% of the power "
        f"has reached the outer {100.0 / _RS_WRAP_RING_BAND:.0f}% of the "
        f"padded window on the kernel='transfer' branch, which is a "
        f"CIRCULAR convolution on that window -- light leaving it "
        f"re-enters on the opposite side instead of being discarded, so "
        f"the returned field carries wrap-around (measured relative L2 up "
        f"to 1.7e-1 against an 8x-padded linear convolution at a ring "
        f"fraction of 0.36).  This is the 'grid pitch at the wavelength "
        f"scale, content at the Nyquist edge' corner: at z = {z:.4g} m "
        f"with dx = {dx:.4e} m, dy = {dy:.4e} m and "
        f"wavelength = {wavelength:.4e} m the field genuinely spreads "
        f"past the padded window.  Enlarge the grid (N), coarsen dx, or "
        f"propagate in shorter steps; kernel='spatial' is NOT an "
        f"alternative here (it aliases below "
        f"z = 2*N*dx**2/wavelength and refuses).",
        RuntimeWarning, stacklevel=3)



def rs_alias_free_distance(N: int, dx: float, wavelength: float) -> float:
    """Smallest ``z`` at which the POINT-SAMPLED RS Green's function is
    adequately sampled on a grid of ``N`` points at pitch ``dx``.

    The spatial kernel ``h(x, y, z)`` is sampled on the zero-padded
    ``2N`` grid, whose half-extent is ``rho_max = N*dx``.  Its local
    spatial frequency at radius ``rho`` is ``sin(theta)/lambda`` with
    ``sin(theta) = rho / sqrt(rho^2 + z^2)``, so the sampled phase step
    is ``k*sin(theta)*dx``.  Requiring that to stay under the ``pi``
    per-pixel Nyquist limit at the padded rim gives

        2*N*dx^2 / (lambda * sqrt((N*dx)^2 + z^2))  <  1 ,

    whose large-``z`` form is the quoted

        z  >  2*N*dx^2 / lambda .

    Below that distance the point-sampled kernel aliases and the
    convolution CREATES energy (measured ``P_out/P_in`` up to 25.7x);
    see :func:`rayleigh_sommerfeld_propagate`'s ``kernel`` parameter.

    Parameters
    ----------
    N : int
        Number of samples along the axis (the UNPADDED grid).
    dx : float
        Sample pitch [m].
    wavelength : float
        Wavelength in the propagation medium [m].

    Returns
    -------
    z_min : float
        ``2*N*dx**2 / wavelength`` [m].

    Examples
    --------
    >>> from lumenairy.propagators.rs import rs_alias_free_distance
    >>> round(rs_alias_free_distance(64, 2e-6, 632.8e-9), 9)
    0.000809102

    See Also
    --------
    rayleigh_sommerfeld_propagate : whose ``kernel='auto'`` branches on
        exactly this distance (``'transfer'`` below it, ``'spatial'`` at
        and above it), and whose ``kernel='spatial'`` refuses below it.
    """
    return 2.0 * float(N) * float(dx) ** 2 / float(wavelength)


#: Pre-v5.46 spelling of :func:`rs_alias_free_distance`.  The function is
#: the branch point of ``rayleigh_sommerfeld_propagate(kernel='auto')``
#: and the number a caller needs in order to choose ``z``, ``N`` or ``dx``
#: for an RS step, so it is public API now; the private name stays bound
#: to the SAME object (``is``-identical, not a wrapper) because callers
#: -- the audit's own regression files among them -- import it.
_rs_alias_free_distance = rs_alias_free_distance


def _rs_pixel_integrated_kernel(Ny2, Nx2, dy, dx, z, k, xp,
                                n_nodes=_RS_PIXEL_QUAD_NODES):
    """The RS-I impulse response INTEGRATED over each pixel of the padded
    grid (audit K9 second half; Shen & Wang, reference [2]).

    Returns the array whose ``[m, n]`` entry is

        Int_{y_m - dy/2}^{y_m + dy/2} Int_{x_n - dx/2}^{x_n + dx/2}
            h(x, y, z) dx dy ,

    i.e. the same quantity the point-sampled build approximates by the
    one-point midpoint rule ``h(x_n, y_m, z) * dx * dy``.  Discretely
    convolving an input array with THIS kernel is the exact
    Rayleigh-Sommerfeld integral of the field that is CONSTANT on each
    input pixel and equal to that pixel's array value -- exact, not
    approximated, because the pixel integral is what the sum then
    contains.  See :func:`rayleigh_sommerfeld_propagate`'s ``kernel``
    parameter for when that is the field you want and when it is not.

    ``h`` depends on ``x`` and ``y`` only through ``x**2 + y**2``, so the
    pixel integral is even in each axis and only the quadrant
    ``|x|, |y| >= 0`` is built: ``Ny//2 + 1`` by ``Nx//2 + 1`` values, a
    quarter of the padded grid, then gathered out to the full array by the
    index map ``|m - Ny2//2|``.  ``x**2`` is bit-identical for ``x`` and
    ``-x``, so the gathered array carries the same values the unfolded
    build produces (measured relative L2 2.1e-16 on three grids; not
    bit-identical because the node sum runs in the mirrored order).

    Parameters
    ----------
    Ny2, Nx2 : int
        Padded grid shape (``2*Ny``, ``2*Nx``).
    dy, dx : float
        Pixel pitch [m].
    z : float
        Propagation distance [m], > 0.
    k : float
        ``2*pi/wavelength`` in the propagation medium [1/m].
    xp : module
        Array namespace (NumPy / CuPy / ``jax.numpy``).
    n_nodes : int, optional
        Gauss-Legendre nodes per pixel axis; see
        :data:`_RS_PIXEL_QUAD_NODES` for the measured floor.

    Returns
    -------
    h_int : ndarray, complex, shape ``(Ny2, Nx2)``
        Already carries the pixel area -- do NOT multiply by ``dx*dy``.
    """
    t_np, w_np = np.polynomial.legendre.leggauss(int(n_nodes))
    ny, nx = int(Ny2) // 2, int(Nx2) // 2
    x_q = xp.arange(nx + 1, dtype=xp.float64) * dx
    y_q = xp.arange(ny + 1, dtype=xp.float64) * dy
    acc = None
    z2 = float(z) * float(z)
    for a in range(int(n_nodes)):
        y_a = y_q + (0.5 * float(dy) * float(t_np[a]))
        for b in range(int(n_nodes)):
            x_b = x_q + (0.5 * float(dx) * float(t_np[b]))
            X, Y = xp.meshgrid(x_b, y_a, indexing='xy')
            r = xp.sqrt(X ** 2 + Y ** 2 + z2)
            term = ((z / (2 * np.pi * r ** 2)) * xp.exp(1j * k * r)
                    * (1.0 / r - 1j * k)) * (float(w_np[a]) * float(w_np[b]))
            acc = term if acc is None else acc + term
    acc = acc * (0.25 * float(dx) * float(dy))
    iy = xp.abs(xp.arange(int(Ny2)) - ny)
    ix = xp.abs(xp.arange(int(Nx2)) - nx)
    return acc[iy[:, None], ix[None, :]]


def rayleigh_sommerfeld_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
    kernel: str = 'auto',
) -> np.ndarray:
    """
    Propagate an optical field using the Rayleigh-Sommerfeld convolution.

    This computes the first Rayleigh-Sommerfeld solution as a true
    LINEAR convolution of the input field with the free-space impulse
    response, on a ``2N x 2N`` zero-padded grid so no wrap-around
    contaminates the result.  That zero padding -- not the kernel
    construction -- is what distinguishes RS from
    :func:`~lumenairy.propagators.asm.angular_spectrum_propagate`, whose
    single-grid FFT is a CIRCULAR convolution (measured: RS 4.4e-8 vs
    ASM 7.6e-1 relative L2 against an exact Hankel oracle at z = 3 mm,
    N = 128, dx = 1 um, Gaussian w0 = 6 um).

    The impulse response is (Goodman *Introduction to Fourier
    Optics*, 3rd ed., eq. 3-43):

        h(x, y, z) = (1 / 2pi) * (z / r^2) * (1/r - ik) * exp(ikr)

    where ``r = sqrt(x^2 + y^2 + z^2)`` and ``k = 2*pi / lambda``.

    Its exact Fourier transform -- the RS-I TRANSFER function -- is

        H(fx, fy) = exp(i*k*z*sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2))

    on the propagating set, and this is what ``kernel='transfer'``
    evaluates directly (see ``kernel`` below).

    The convolution is computed as::

        E_out = IFFT{ FFT{E_in} * H }

    using zero-padded arrays (2N x 2N) to avoid circular convolution
    artifacts.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.
    z : float
        Propagation distance [m].  Positive = forward.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.
    bandlimit : bool, default False
        Apply a Matsushima-style frequency cutoff
        ``|f| < L2 / (2*lambda*|z|)`` to the kernel, where
        ``L2 = 2*N*dx`` is the PADDED extent.  v5.30 (audit P12): that
        expression is the **z -> infinity asymptote** of Matsushima &
        Shimobaba's exact local-frequency limit
        ``1/(lambda*sqrt((2z/L2)^2 + 1))``, not the exact limit -- it is
        strictly the larger of the two, so it never over-filters (see
        :func:`~lumenairy.propagators.fft_infra._get_or_make_bandlimit`
        for the derivation and the measured over-width table).

        .. warning::
           ``bandlimit`` is **not** a near-field remedy and the default
           ``False`` is the accurate setting for this propagator.  The
           zero padding already removes the wrap-around the Matsushima
           criterion exists to suppress, so the mask can only DISCARD
           valid content: measured relative L2 vs an exact Hankel oracle
           at N = 128, dx = 1 um, z = 3 mm, Gaussian w0 = 6 um is 4.4e-8
           (``False``) against 1.8e-2 (``True``) -- five decades worse.
           In the near field
           (``z < 2*N*dx**2/wavelength``) the cutoff exceeds the grid
           Nyquist under exactly the same algebraic condition that made
           the old spatial kernel alias, so there it is all-pass and does
           nothing at all.
    use_gpu : bool, default False
        Use CuPy GPU acceleration if available.
    verbose : bool, default False
        Print diagnostic info.
    kernel : {'auto', 'transfer', 'spatial', 'spatial-integrated'}, default 'auto'
        Which discretisation of the (single) RS-I operator to use on the
        padded grid.  ``'transfer'`` and ``'spatial'`` build the same
        physics and fail in opposite regimes, so the default routes
        between them; ``'spatial-integrated'`` is a different reading of
        what the input SAMPLES mean (see below).

        * ``'auto'`` (default, v5.46; audit K9) -- ``'transfer'`` when
          ``z < 2*N*dx**2/wavelength`` (see
          :func:`rs_alias_free_distance`), ``'spatial'`` at and above
          that distance.  Every call at or above the threshold is
          therefore **bit-identical to the pre-v5.46 output**; only the
          regime the audit measured as broken is re-routed.
        * ``'transfer'`` -- evaluate the exact RS-I transfer function
          ``exp(i*k*z*sqrt(1 - (lambda*f)^2))`` analytically in the
          FREQUENCY domain, with the evanescent set
          (``(lambda*f)^2 >= 1``) zeroed as everywhere else in this
          library.  This is the closed-form Fourier transform of the
          Goodman 3-43 impulse response above -- the SAME operator,
          without discretising a chirp the grid cannot carry -- so it
          never aliases and conserves energy exactly.  Its own failure
          mode is the complement: multiplying by ``H`` is a CIRCULAR
          convolution on the padded grid, so light that leaves the
          padded window wraps back in instead of being discarded.
        * ``'spatial'`` -- point-sample ``h(x, y, z)`` on the padded grid
          and FFT it (the pre-v5.46 path).  Truncating ``h`` at the
          padded rim makes this the better choice once the beam
          overfills the window -- measured relative L2 4.4e-8 against an
          exact Hankel oracle at z = 3 mm (N = 128, dx = 1 um, Gaussian
          w0 = 6 um, 63 % of the power inside the window) where
          ``'transfer'`` reads 1.9e-2 and single-grid ASM 7.6e-1.
          It RAISES for ``z < 2*N*dx**2/wavelength``, where it aliases.
        * ``'spatial-integrated'`` -- the same convolution with ``h``
          INTEGRATED over each pixel instead of sampled at its centre
          (Shen & Wang, reference [2]; audit K9).  Same support, same
          truncation at the padded rim, same alias refusal; only the
          quadrature of ``h`` changes.  **It is not a more accurate
          version of ``'spatial'`` -- it answers a different question**,
          and which one is right is a property of your input array:

          - ``'spatial'`` reads ``E_in`` as POINT SAMPLES of a smooth,
            adequately sampled field.  The sum is then the trapezoidal
            rule for the RS integral, which for such a field is
            spectrally accurate.
          - ``'spatial-integrated'`` reads ``E_in`` as CELL VALUES of a
            field that is constant across each pixel -- a binary mask, a
            pixelated DOE or SLM map, any input whose staircase IS the
            physical object.  The convolution is then that field's exact
            RS integral.

          Measured (lambda = 633 nm, circular aperture a = 100 um,
          window 512 um, z = 16 mm -- above the alias threshold of every
          grid quoted, so both kernels are legal on all of them).
          Against a super-sampled continuum RS-I double quadrature of the
          STAIRCASE aperture (a midpoint rule, ``S`` sub-samples per
          pixel axis, evaluated by direct summation at four output
          points; its own floor read off from ``S`` against ``2S``):

          ================  ===  ============  ==================  =========
          grid              S    oracle floor  'spatial-integrated'  'spatial'
          ================  ===  ============  ==================  =========
          N = 64, dx = 8um  16   1.3794e-5     4.5979e-6           4.7464e-3
          N = 64, dx = 8um  32   3.4485e-6     1.1495e-6           4.7499e-3
          N = 128, dx = 4um 16   3.3558e-6     1.1186e-6           1.1465e-3
          N = 128, dx = 4um 32   8.3895e-7     2.7965e-7           1.1473e-3
          ================  ===  ============  ==================  =========

          ``'spatial-integrated'`` sits BELOW the oracle's own floor and
          divides by four every time ``S`` doubles -- that is the ORACLE
          converging onto it, which is what "exact" looks like when the
          only available reference is itself approximate.  ``'spatial'``
          does not move with ``S`` at all and stands 1024x / 4103x away.

          Against an adequately sampled SMOOTH input the ranking
          reverses, by four to five decades.  Relative L2 vs an exact
          Hankel angular-spectrum quadrature of a Gaussian ``w0``:

          ==========================  ========  ==========  ==================
          grid                        z         'spatial'   'spatial-integrated'
          ==========================  ========  ==========  ==================
          N = 128, dx = 1um, w0 = 6um   3 mm    4.42e-8     1.24e-3
          N = 128, dx = 1um, w0 = 6um   0.5 mm  6.63e-8     3.27e-3
          N = 128, dx = 2um, w0 = 12um  1.7 mm  3.23e-8     3.27e-3
          N = 64,  dx = 2um, w0 = 6um   0.9 mm  6.46e-8     1.29e-2
          N = 256, dx = 0.5um, w0 = 6um 0.25 mm 6.35e-8     8.18e-4
          N = 256, dx = 1um, w0 = 12um  1 mm    3.34e-8     8.18e-4
          ==========================  ========  ==========  ==================

          The gap is exactly the difference between the Gaussian and its
          own staircase, which is the statement above in numbers.
          ``'spatial'`` therefore stays the default and ``'auto'`` never
          selects this kernel.

          **What it does not fix.**  The audit's roughly-first-order
          convergence on a hard-aperture input is the APERTURE, not the
          kernel.  On-axis relative error against the closed form
          ``U = e^{ikz} - (z/r_a) e^{ik r_a}``, same fixture, with the
          aperture as a pixel-centre indicator ("stair") and as its exact
          pixel-area average ("grey"):

          ======  ========  ==========  ==========  ==========  ==========
          N       dx [um]   stair+pt    stair+int   grey+pt     grey+int
          ======  ========  ==========  ==========  ==========  ==========
          128     4.000     8.3008e-3   8.4184e-3   1.4045e-3   2.6828e-3
          256     2.000     3.3548e-3   3.3707e-3   3.4263e-4   6.6308e-4
          512     1.000     3.4207e-4   3.5238e-4   8.4251e-5   1.6429e-4
          1024    0.500     5.2718e-4   5.2758e-4   2.0677e-5   4.0724e-5
          ======  ========  ==========  ==========  ==========  ==========

          measured order between successive rows: 1.307 / 3.294 / -0.624
          (stair+pt), 1.321 / 3.258 / -0.582 (stair+int), **2.035 / 2.024
          / 2.027** (grey+pt) and **2.016 / 2.013 / 2.012** (grey+int).
          The lever that restores second order is the INPUT's edge (area
          averaging), for either kernel -- 25x at N = 1024 -- and the
          kernel choice then moves the constant by ~2x, in the
          point-sampled kernel's favour.  The library builds that input
          already:
          :func:`~lumenairy.elements.elements.apply_aperture` with
          ``edge='gray'`` gives each rim pixel its supersampled open-area
          fraction.  Reach for it before reaching for this kernel.

          **Cost.**  The pixel integral is a
          :data:`_RS_PIXEL_QUAD_NODES`-node tensor Gauss-Legendre rule per
          pixel, built on one quadrant and mirrored.  Kernel build:
          18.5 / 146 / 782 ms against 3.4 / 16.7 / 72.0 ms for the point
          sample at N = 128 / 256 / 512, i.e. 5.5x / 8.7x / 10.9x
          (medians of five interleaved runs).  End to end that is 2.4x
          and 4.2x a whole ``'spatial'`` call at N = 128 / 256 on a COLD
          H cache, and 0.65x / 0.93x -- the same call -- on a warm one,
          because the cache pays the build once per geometry.

        **Why the routing exists.**  The point-sampled kernel's phase
        gradient ``k*sin(theta)*dx`` exceeds the ``pi``/pixel Nyquist
        limit whenever ``z < 2*N*dx**2/wavelength``, and nothing in the
        kernel build can check it.  Measured against an exact Hankel
        angular-spectrum oracle (Gaussian w0 = 6 um, lambda = 633 nm,
        z = 50 um), ``'auto'`` reads relative L2 5.3e-8 / 6.1e-8 / 5.3e-8
        with ``P_out/P_in`` 1.000000 on (N = 64, dx = 2 um),
        (128, 1 um) and (128, 2 um).  At and above the threshold the
        default is BYTE-IDENTICAL to ``'spatial'``; below it the two agree
        to the FFT's own floor wherever the spatial kernel was still
        adequately sampled (relative L2 1.7e-13 at z = 200 um and 2.1e-13
        at z = 300 um on the N = 128 / dx = 1 um probe, each ~2e-13 from
        an 8x-zero-padded reference).

        The two branches are continuous across the switch, and converge
        onto each other as the grid is refined.  Measured at
        ``z = z_crit`` on the Gaussian probe, ``relL2(transfer,
        spatial)`` against ``relL2(*, oracle)``:

        ===============================  =========  ==================
        grid                             the step   each arm's error
        ===============================  =========  ==================
        N = 64,  dx = 0.50 um            6.3e-5     2.98e-4 / 2.93e-4
        N = 128, dx = 0.40 um            1.0e-9     6.589e-8 / 6.588e-8
        N = 256, dx = 0.25 um            3.8e-14    6.072e-8 / 6.072e-8
        ===============================  =========  ==================

        i.e. the step is always well below the method's own error at
        that grid -- there is no discontinuity a caller can observe.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated field (same shape as input).

    Notes
    -----
    **When to use RS instead of ASM:**

    - Long propagation distances on a grid the beam has spread across:
      RS's zero padding is a linear convolution, so it does not wrap
      energy around the grid the way single-grid ASM does (measured
      4.4e-8 vs 7.6e-1 relative L2 at z = 3 mm, N = 128, dx = 1 um,
      Gaussian w0 = 6 um; 63 % of the power is still inside the window).
    - Validation / cross-check against ASM results.
    - Situations where the exact Green's function is preferred over
      the plane-wave decomposition.

    RS is NOT a remedy for ASM's band limiting in the near field: on the
    N = 128 / dx = 1 um / w0 = 6 um probe at z = 50 um, ASM with its
    default ``bandlimit=True`` measures relative L2 6.1e-8 against the
    exact Hankel oracle -- i.e. ASM is already exact there.

    **Computational cost:** ~4x ASM due to zero-padding (2N FFTs
    instead of N FFTs).

    **Memory:** ~6x input array size (padded E, padded H, FFTs).

    **Agreement with ASM.**  With ``kernel='transfer'`` the two build the
    SAME transfer function; they differ only in the convolution support
    (RS zero-pads to 2N, ASM does not) and in the band limit each
    applies.  They therefore agree to FFT round-off only where neither
    the ASM wrap-around nor a band limit bites; elsewhere the difference
    is a real, quantified modelling difference, not round-off.

    **H caching:** the kernel is cached on the NumPy backend keyed on the
    padded geometry ``(2*Ny, 2*Nx, dy, dx, wavelength, z, bandlimit,
    dtype)`` plus a tag.  Repeat calls at the same geometry skip the
    kernel build (~30-40% of total RS time on 2k+ grids).  The
    ``'spatial'`` kernel is tagged ``'RS'``; the ``'transfer'`` kernel
    goes through :func:`angular_spectrum_propagate`'s own builder and so
    shares its ``'ASM'`` entry at the padded geometry -- legitimately,
    because it is the same array.  Both obey the byte budgets configured
    via :func:`set_asm_cache_size`.  CuPy and JAX arrays are kept out of
    the cache (host-side dict can't safely retain device pointers /
    traced objects); rebuild every call.

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.),
        Section 3.5: Rayleigh-Sommerfeld Diffraction Theory.
    [2] Shen, F. and Wang, A. (2006). "Fast-Fourier-transform based
        numerical integration method for the Rayleigh-Sommerfeld
        diffraction formula." Appl. Opt. 45(6): 1102-1110.  Prescribes
        INTEGRATING the impulse response over each pixel instead of
        point-sampling it; that is ``kernel='spatial-integrated'``, whose
        parameter entry above carries the measured comparison.  It is
        the exact operator for a cell-constant input and NOT a drop-in
        accuracy upgrade for a sampled smooth one, so it is opt-in.
    [3] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space
        propagation in far and near fields." Opt. Express 17(22):
        19662-19673.  NOTE (v5.30, audit P12): ``bandlimit=True`` applies
        the ``z -> infinity`` asymptote of this paper's local-frequency
        limit, not the exact expression (never over-filters).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import rayleigh_sommerfeld_propagate
    >>>
    >>> N = 512; dx = 1e-6; wv = 0.633e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> E_in = (np.sqrt(X**2 + Y**2) < 50e-6).astype(complex)  # circular aperture
    >>>
    >>> E_out = rayleigh_sommerfeld_propagate(E_in, z=1e-3, wavelength=wv, dx=dx)
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'rayleigh_sommerfeld_propagate',
                           input_kind='field')

    # 4.12.0 (audit round-4 B1-3): RS is forward-only.  Pre-4.12 the
    # function accepted z <= 0 silently and computed a 180-degrees-
    # wrong-phase kernel for the back-propagation case.  Match the
    # existing Fresnel / Fraunhofer / SAS guards: hard error with
    # guidance to use ASM / ASM-MFT for back-propagation.
    if z <= 0:
        raise ValueError(
            f"rayleigh_sommerfeld_propagate: z must be > 0 (got "
            f"{z!r}).  RS is forward-only; use "
            f"angular_spectrum_propagate or "
            f"angular_spectrum_propagate_mft for back-propagation "
            f"(those handle the z < 0 case correctly).")
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='rayleigh_sommerfeld_propagate')
    if kernel not in ('auto', 'transfer', 'spatial', 'spatial-integrated'):
        raise ValueError(
            f"rayleigh_sommerfeld_propagate: kernel must be 'auto' (default: "
            f"the exact RS-I transfer function where the point-sampled "
            f"Green's function would alias, i.e. z < 2*N*dx**2/wavelength, "
            f"and the spatial kernel above that), 'transfer' (always the "
            f"transfer function), 'spatial' (always the point-sampled "
            f"Green's function, for an E_in that is point samples of a "
            f"smooth field) or 'spatial-integrated' (the Green's function "
            f"integrated over each pixel, for an E_in whose staircase IS "
            f"the object -- a binary mask, a pixelated DOE or SLM map); "
            f"got {kernel!r}.")

    # -- array library selection -----------------------------------------------
    from ..backend import is_jax_array
    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = _state.cp
        if not _is_cupy_array(E_in):
            E_in = _state.cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    # Target complex dtype for h, H, and the padded buffer.  Inferred
    # from E_in so the caller controls precision via input dtype.
    # Non-complex input falls back to DEFAULT_COMPLEX_DTYPE to match
    # the standardisation used by angular_spectrum_propagate and the
    # MFT propagator family.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(_state.DEFAULT_COMPLEX_DTYPE)

    k = 2 * np.pi / wavelength

    # -- zero-pad to avoid circular convolution --------------------------------
    Ny2 = 2 * Ny
    Nx2 = 2 * Nx

    # -- K9: the sampling limit of the SPATIAL kernel --------------------------
    # The point-sampled Green's function aliases below this distance (see
    # rs_alias_free_distance for the derivation).  Anamorphic pitch: the
    # tighter of the two axes governs.
    z_alias = max(rs_alias_free_distance(Nx, dx, wavelength),
                  rs_alias_free_distance(Ny, dy, wavelength))

    if kernel == 'auto':
        # Route by which discretisation of the SAME operator is sound here.
        # Below z_alias the spatial kernel under-samples its own chirp and
        # creates energy; above it, the spatial kernel is the better of the
        # two because truncating h at the padded rim DISCARDS the light that
        # leaves the window, where the frequency-domain build (a circular
        # convolution on the padded grid) WRAPS it back in.
        kernel_used = 'transfer' if z < z_alias else 'spatial'
    else:
        kernel_used = kernel

    if kernel_used == 'transfer':
        # K9 (P0): build the EXACT RS-I transfer function analytically in
        # the frequency domain on the padded grid instead of FFT-ing a
        # point-sampled h.  H(f) = exp(i k z sqrt(1 - (lam f)^2)) is the
        # closed-form Fourier transform of the Goodman 3-43 impulse
        # response, so this is the same operator evaluated without a
        # discretisation that Nyquist cannot support.
        #
        # ``_get_asm_H_natural`` already IS that builder: it returns the
        # transfer function on the requested grid in natural (un-shifted)
        # FFT layout, with the evanescent zeroing, the shared complex64
        # mod-2*pi mitigation, chunked construction, the Matsushima band
        # limit and the H cache.  Calling it on the PADDED (2Ny, 2Nx)
        # geometry gives the zero-padded -- i.e. linear-convolution --
        # form, which is what distinguishes RS from plain ASM.  The cache
        # entry is legitimately shared with an ASM call at the same padded
        # geometry: it is the same array.
        from .asm import _get_asm_H_natural
        H = _get_asm_H_natural(Ny2, Nx2, dy, dx, wavelength, z,
                               bandlimit, target_cdtype, xp,
                               is_jax=is_jax, verbose=False)
        if verbose:
            print(f"  RS propagation: z = {z*1e3:.3f} mm  "
                  f"(kernel={kernel!r} -> 'transfer', exact RS-I transfer "
                  f"function)")
            print(f"  Grid: {Ny}x{Nx} -> padded {Ny2}x{Nx2}")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
            print(f"  Spatial-kernel alias threshold 2*N*dx^2/lambda = "
                  f"{z_alias*1e6:.1f} um")
        h_key = None
    else:
        h_key = None
        H = None

    # Both spatial builds discretise h on the padded grid and alias under
    # the same condition: the pixel integral narrows the kernel's spectrum
    # by a sinc but does not band-limit it, so the replicas the sampling
    # folds in are attenuated, not removed.  One guard, both tokens.
    if kernel_used in ('spatial', 'spatial-integrated') and z < z_alias:
        raise ValueError(
            f"rayleigh_sommerfeld_propagate: kernel={kernel_used!r} "
            f"discretises "
            f"the Rayleigh-Sommerfeld Green's function, which ALIASES for "
            f"z < 2*N*dx**2/wavelength = {z_alias:.6g} m (got z={z:.6g} m).  "
            f"In that regime the convolution creates energy (measured "
            f"P_out/P_in up to 25.7x with a relative L2 of 4.95 against an "
            f"exact Hankel oracle), and bandlimit=True does not help -- its "
            f"cutoff exceeds the grid Nyquist under the same condition.  Use "
            f"kernel='transfer' (the exact RS-I transfer function; measured "
            f"relative L2 5.3e-8 and P_out/P_in 1.000000 on the same grids) "
            f"or the default kernel='auto' (which selects it here), or "
            f"coarsen the problem so that z >= {z_alias:.6g} m "
            f"(e.g. dx <= {(wavelength*z/(2*max(Ny, Nx)))**0.5:.6g} m at "
            f"N={max(Ny, Nx)}).")

    if kernel_used in ('spatial', 'spatial-integrated'):
        # H cache (NumPy backend only)
        # Geometry signature.  Hits return the previously-built H without
        # re-running the kernel construction or its FFT (~30-40% of total
        # RS time on 2k+ grids, and 5-9x the point build for the
        # pixel-integrated one).  The 'RS' / 'RS_INT' tags keep the two
        # spatial kernels disjoint from each other and from the ASM
        # transfer-function entries that kernel='transfer' shares.
        if xp is np:
            h_key = (int(Ny2), int(Nx2), float(dy), float(dx),
                     float(wavelength), float(z), bool(bandlimit),
                     np.dtype(target_cdtype).str,
                     'RS' if kernel_used == 'spatial' else 'RS_INT')
            H = _h_cache_lookup(h_key)

    if H is None:
        # -- build the RS impulse response h(x, y, z) on the padded grid -------
        # h = -(1/2π) ∂/∂z[exp(ikr)/r]
        #   = (z / (2π r²)) · (1/r − ik) · exp(ikr)         (Goodman 3-43)
        # Pre-4.10 implementation flipped this to (ik − 1/r), producing
        # −h_correct.  Output amplitudes look fine for |E|² consumers but
        # any coherent sum of RS with ASM/Fresnel was 180° out of phase.
        if kernel_used == 'spatial-integrated':
            # The pixel INTEGRAL of the same h, which already carries the
            # pixel area (reference [2]; see the ``kernel`` parameter for
            # what it changes and for whom).
            h = _rs_pixel_integrated_kernel(Ny2, Nx2, dy, dx, z, k, xp)
        else:
            x = (xp.arange(Nx2) - Nx2 / 2) * dx
            y = (xp.arange(Ny2) - Ny2 / 2) * dy
            X, Y = xp.meshgrid(x, y, indexing='xy')
            r = xp.sqrt(X ** 2 + Y ** 2 + z ** 2)
            h = ((z / (2 * np.pi * r ** 2)) * xp.exp(1j * k * r)
                 * (1.0 / r - 1j * k))
            h = h * (dx * dy)
        if h.dtype != target_cdtype:
            h = h.astype(target_cdtype)

        if verbose:
            print(f"  RS propagation: z = {z*1e3:.3f} mm  (H cache miss)")
            print(f"  Grid: {Ny}x{Nx} -> padded {Ny2}x{Nx2}")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
            try:
                print(f"  Kernel max |h|: {float(xp.max(xp.abs(h))):.4e}")
            except (TypeError, ValueError, RuntimeError) as _exc:
                # xp.max + float() can fail under JAX tracing where h is
                # an abstract array (TypeError on the float()
                # conversion).  Cosmetic print failure -- demote to a
                # brief diagnostic so debug runs surface the cause.
                print(f"  Kernel max |h|: <unavailable: "
                      f"{type(_exc).__name__}>")

        # -- FFT the kernel ----------------------------------------------------
        # The result is cached via _h_cache_store below and reused
        # across many subsequent _fft2/_ifft2 calls; under the 4.12
        # double-buffer contract on _fft2, we must take an explicit
        # copy here so the cached H survives the third subsequent
        # call at this shape (which would recycle the slot).  The
        # bandlimit branch below already produces a fresh array via
        # H * mask_c, but the non-bandlimit path would otherwise
        # alias the plan workspace; copy unconditionally for clarity.
        if is_jax:
            H = xp.fft.fft2(xp.fft.ifftshift(h))
        elif xp is np:
            H = _fft2(np.fft.ifftshift(h)).copy()
        else:
            H = xp.fft.fft2(xp.fft.ifftshift(h))

        # -- Matsushima-style bandlimit on the padded H ------------------------
        # Cutoff matches the ASM derivation but uses the padded extent
        # (Lx2 = 2*Nx*dx) since the FFT length is what determines the
        # discrete frequency support.  The mask is built in centred
        # order then ifftshifted to align with H's DC-at-corner layout.
        #
        # v5.30 (audit P12): the cutoff ``L / (2*lambda*|z|)`` is the
        # z -> infinity ASYMPTOTE of Matsushima & Shimobaba's exact
        # local-frequency limit, not that limit itself -- see
        # :func:`lumenairy.propagators.fft_infra._get_or_make_bandlimit`
        # for the derivation and the measured over-width table.  It is an
        # upper bound, so it never over-filters.
        #
        # No ``and z != 0`` conjunct here: RS is forward-only and the guard
        # above hard-raises for ``z <= 0`` (measured: ``z=0`` and
        # ``z=-1e-3`` both ValueError), so ``z`` is always > 0 and the test
        # could never be False (audit P13).  ASM / ASM-MFT keep THEIRS --
        # those DO accept ``z == 0`` as the exact identity (audit S2-11)
        # and rely on it.
        if bandlimit:
            fx = (np.arange(Nx2) - Nx2 / 2) / (Nx2 * dx)
            fy = (np.arange(Ny2) - Ny2 / 2) / (Ny2 * dy)
            Lx2 = Nx2 * dx
            Ly2 = Ny2 * dy
            fx_max = Lx2 / (2 * wavelength * abs(z))
            fy_max = Ly2 / (2 * wavelength * abs(z))
            bl_x = np.abs(fx) < fx_max
            bl_y = np.abs(fy) < fy_max
            mask = (bl_y[:, None] & bl_x[None, :])
            mask = np.fft.ifftshift(mask)
            mask_c = mask.astype(target_cdtype)
            if xp is np:
                H = H * mask_c
            else:
                H = H * xp.asarray(mask_c)
            if verbose:
                kept_frac = float(np.mean(mask))
                print(f"  Bandlimit: keeping {kept_frac*100:.1f}% of "
                      f"padded spectrum")

        # Store under the NumPy key only.  See _h_cache_store for the
        # byte-budget eviction policy.  Cached H is used read-only.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  RS propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    # -- build the padded input field -----------------------------------------
    if is_jax:
        # JAX is functional / immutable -- can't write into a pre-allocated
        # array.  Build the padded array via jnp.zeros + at[].set.
        E_padded = xp.zeros((Ny2, Nx2), dtype=target_cdtype)
        y0 = Ny // 2
        x0 = Nx // 2
        E_padded = E_padded.at[y0:y0 + Ny, x0:x0 + Nx].set(E_in)
    else:
        E_padded = xp.zeros((Ny2, Nx2), dtype=target_cdtype)
        y0 = Ny // 2
        x0 = Nx // 2
        E_padded[y0:y0 + Ny, x0:x0 + Nx] = E_in

    # -- convolve via FFT ------------------------------------------------------
    if is_jax:
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)
    elif xp is np:
        E_fft = _fft2(E_padded)
        E_conv = _ifft2(E_fft * H)
    else:
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)

    # V6 (verify pass, 2026-09-12): the transfer branch's own failure
    # mode -- a circular convolution on the padded window -- is
    # reachable below the alias threshold for a field whose angular
    # content fills the grid.  Say so; the values are unchanged.
    if kernel_used == 'transfer':
        # Reference power, sampled on a stride with the same budget so
        # the whole diagnostic is O(1) in grid size.  The transfer kernel
        # is unitary on the propagating set, so the padded total equals
        # this to round-off (measured agreeing to four digits on every
        # calibration fixture).
        _st = max(1, int(np.sqrt(float(Ny) * float(Nx)
                                 / _RS_WRAP_SAMPLE_BUDGET)))
        _sub = E_in[::_st, ::_st]
        _flat = xp.reshape(_sub, (-1,))
        _p_in = float(xp.real(xp.vdot(_flat, _flat))) * (
            (float(Ny) * float(Nx)) / max(int(_sub.size), 1))
        _warn_rs_transfer_wraparound(
            E_conv, _p_in, Ny2, Nx2,
            float(z), float(dx), float(dy), float(wavelength))

    # -- extract the valid region (same location as input was placed) ----------
    # v5.4.6 (audit F-3): ``.copy()`` is REQUIRED.  For the NumPy/CuPy path
    # ``_ifft2`` returns a view into the cache-owned pyFFTW inverse
    # ping-pong buffer, which the double-buffer contract guarantees only
    # until the NEXT same-key ``_ifft2`` call.  Returning a bare slice
    # (a view) of that buffer means a subsequent RS propagation at the
    # same grid silently overwrites a previously-returned field -- a
    # data-corruption hazard on multi-distance RS sweeps.  Copy detaches
    # the output from the reused buffer.
    E_out = E_conv[y0:y0 + Ny, x0:x0 + Nx].copy()

    return E_out
