"""BOR-PMM Milestone 5 (far-field core): Fourier-Bessel / discrete Hankel
decomposition of an axisymmetric near-field onto cylindrical "orders".

A transverse field of azimuthal order ``m`` on a bounded radius ``[0, R]``
expands in the Fourier-Bessel series

    f(r) = sum_n c_n J_m(alpha_n r / R),     alpha_n = j_{m,n}  (n-th zero of J_m)
    c_n  = (2 / (R^2 J_{m+1}(alpha_n)^2)) * INT_0^R f(r) J_m(alpha_n r/R) r dr

Each term is an outgoing cylindrical wave of transverse wavenumber
``kt_n = alpha_n / R``; in an open half-space of index ``sqrt(eps)`` it radiates
at polar angle ``sin(theta_n) = kt_n / (sqrt(eps) k0)`` (propagating iff
``kt_n < sqrt(eps) k0``).  This is the cylindrical analog of the planar grating's
Fourier-order decomposition; for a circular grating of radial period ``Lambda``
the populated ``kt_n`` cluster near ``kt_inc + 2 pi p / Lambda`` (the p-th
diffraction order), reducing to the planar grating equation as ``R -> inf``.

POWER NORMALIZATION (validated, Parseval to ~1e-10):
    INT_0^R |f|^2 r dr = sum_n |c_n|^2 * N_n,   N_n = R^2 J_{m+1}(alpha_n)^2 / 2
so a per-order power fraction is ``|c_n|^2 N_n / INT|f|^2 r dr`` -- the basis for
diffraction efficiencies (combined with the modal z-flux for the propagating set).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import jn_zeros, jv


def fourier_bessel(f, r, h, m, nmax, *, wq=None, R=None):
    """Decompose ``f`` (sampled on the cell-centered grid ``r``, spacing ``h``)
    into ``nmax`` Fourier-Bessel coefficients of order ``m``.

    Returns ``(c, kt, norm)``: coefficients ``c_n``, transverse wavenumbers
    ``kt_n = alpha_n / R``, and the squared-norms ``N_n`` (for Parseval / power).

    NON-UNIFORM GRIDS (the SEM basis).  Pass ``wq`` -- per-sample quadrature
    weights that ALREADY CONTAIN the ``r dr`` measure (the SEM native
    ``wq_node`` / ``wq_face`` vectors) -- and the domain edge ``R`` (defaults
    to ``r[-1]`` when ``wq`` is given; the cell-centered ``+ h/2`` convention
    is a uniform-grid artefact).  ``h`` is then ignored for the integral and
    used only as a fallback for the Nyquist estimate, which on a non-uniform
    grid is taken from the LARGEST sample spacing (the resolution bottleneck).

    SAMPLING (audit W6-B5).  The coefficient integral is a midpoint rule on the
    given grid, so a requested order is only meaningful while its kernel is
    resolved: ``kt_n = alpha_n / R <= pi / h`` (the grid Nyquist).  Beyond that
    the ``J_m`` kernel aliases and the coefficients become noise whose Parseval
    sum OVER-counts the field power -- measured 3.0x at ``nmax = 250`` on an
    ``N = 100`` grid, with the round-trip reconstruction 100% wrong, and
    ``order_power_fractions`` renormalizes ``frac`` by that inflated total so
    nothing in the returned dict betrays it.  A ``UserWarning`` now fires
    instead of silence.
    """
    if wq is None:
        R = r[-1] + h / 2.0                   # domain edge (cell-centered grid)
        h_eff = h
    else:
        wq = np.asarray(wq)
        if wq.shape != np.shape(r):
            raise ValueError(
                f"fourier_bessel: wq shape {wq.shape} != r shape "
                f"{np.shape(r)}")
        if R is None:
            R = float(np.asarray(r)[-1])
        h_eff = float(np.max(np.diff(np.asarray(r, float))))
    alpha = jn_zeros(m, nmax)
    n_alias = int(np.sum(alpha / R > np.pi / h_eff))
    if n_alias:
        warnings.warn(
            f"fourier_bessel: {n_alias} of the {nmax} requested orders have "
            f"kt = alpha_n/R above the grid Nyquist pi/h = "
            f"{np.pi / h_eff:.4g} "
            f"(kt_max = {alpha[-1] / R:.4g}); their J_m kernels alias on this "
            f"{len(r)}-point grid, so those coefficients are noise and the "
            f"Parseval power sum over-counts.  Use nmax <~ {len(r)} for this "
            f"grid, or refine the grid.", stacklevel=2)
    c = np.zeros(nmax, dtype=complex)
    norm = np.zeros(nmax)
    for n in range(nmax):
        Jn = jv(m, alpha[n] * r / R)
        w_rdr = (r * h) if wq is None else wq        # r dr quadrature weights
        c[n] = np.sum(f * Jn * w_rdr) * 2.0 / (R ** 2
                                               * jv(m + 1, alpha[n]) ** 2)
        norm[n] = R ** 2 * jv(m + 1, alpha[n]) ** 2 / 2.0
    return c, alpha / R, norm


def far_field_angles(kt, eps, k0):
    """Polar angles ``theta_n`` (radians) for transverse wavenumbers ``kt`` in a
    medium of permittivity ``eps``; NaN for evanescent (non-radiating) orders.

    ``eps`` may be complex; the angle is taken in the REAL refractive index
    ``Re sqrt(eps)`` -- the same convention ``BORStack.solve`` uses for its
    ``angles`` (``eps_sup.real``).  Audit W6-B10: a complex ``eps`` used to make
    ``s`` complex, so the propagating mask fell back to numpy's LEXICOGRAPHIC
    complex comparison and ``theta`` was filled from a complex ``arcsin`` whose
    imaginary part was dropped with only a ``ComplexWarning`` -- silently wrong
    angles for any lossy half-space (``order_power_fractions`` passes ``eps``
    straight through).
    """
    with np.errstate(invalid="ignore"):
        n_med = np.sqrt(np.asarray(eps, dtype=complex)).real
    if not np.all(np.isfinite(n_med)) or np.any(n_med <= 0.0):
        raise ValueError(
            "far_field_angles: Re sqrt(eps) must be finite and > 0 (got eps=%r)"
            % (eps,))
    # kt is a transverse wavenumber MAGNITUDE (alpha_n / R) -- real by
    # construction; np.real is a no-op on a real array.
    s = np.real(np.asarray(kt)) / (n_med * float(k0))
    theta = np.full_like(s, np.nan, dtype=float)
    prop = s <= 1.0
    theta[prop] = np.arcsin(s[prop])
    return theta, prop


def order_power_fractions(f, r, h, m, eps, k0, nmax, *, wq=None, R=None):
    """Per-cylindrical-order power fractions of a near-field ``f`` of order ``m``,
    with the propagating mask and far-field angles.  (Power-normalized via the
    Parseval relation; lossless fractions over the propagating set + evanescent
    tail sum to 1.)

    ``total`` is the power carried by the RETAINED ``nmax`` orders, i.e.
    ``sum |c_n|^2 N_n`` -- it equals ``INT_0^R |f|^2 r dr`` only once the series
    has converged (measured deficit 9.6e-3 at nmax = 5 vs 4.5e-9 at nmax = 10 on
    a smooth Gaussian).  ``frac`` is normalized BY ``total``, so it sums to 1 by
    construction and cannot itself reveal a truncation (or an aliasing) loss --
    compare ``total`` against ``sum(|f|**2 * r * h)`` if that matters, and see
    the ``fourier_bessel`` Nyquist note (audit W6-B5).
    """
    c, kt, norm = fourier_bessel(f, r, h, m, nmax, wq=wq, R=R)
    total = np.sum(np.abs(c) ** 2 * norm)
    frac = np.abs(c) ** 2 * norm / max(total, 1e-300)
    theta, prop = far_field_angles(kt, eps, k0)
    return dict(c=c, kt=kt, frac=frac, theta=theta, prop=prop, total=total)
