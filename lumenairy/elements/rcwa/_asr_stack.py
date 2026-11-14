"""Adaptive spatial resolution (ASR) for the 1-D :class:`RCWAStack`.

What it does
------------
The Fourier modal method resolves a lamellar layer with a uniform set of
harmonics ``exp(i k_n x)``.  The TM field of a metal (or any high-contrast)
strip is singular at the strip corners, so that basis converges only
algebraically (about ``N^-1.1`` for the inverse rule on a metal strip).  ASR
(Granet 1999) solves the problem in a stretched lateral coordinate ``u``,
``x = x(u)``, whose Jacobian ``f = dx/du`` is small at the walls: the uniform
``u`` harmonics are then dense near the walls in ``x``.  Here the walls of
EVERY patterned layer of the stack sit on coordinate lines ("matched
coordinates"), so each layer's permittivity is a clean step in ``u``.

The formulation, and why each piece is the way it is
----------------------------------------------------
1. ONE COORDINATE MAP FOR THE WHOLE STACK, built from the union of the
   transition points of all patterned layers (Essig and Busch 2010 apply one
   transformation throughout the whole structure; Weiss et al. 2009 likewise).
   Per-layer maps (Vallius and Honkanen 2002) need a change of basis at every
   interface between layers with different maps; on a thin-layer stack the
   strip and the layers below it couple through high-order evanescent fields,
   which is exactly the content such a projection truncates.  With one map
   the interfaces need no projection at all.  A laterally uniform layer is
   no longer diagonal in the ``u`` basis, but its modes are one Hermitian
   eigendecomposition shared by every uniform layer and both half-spaces
   (item 4), so the cost is one ``N x N`` ``eigh`` per solve.

2. COVARIANT FORM (Weiss et al. 2009, eqs. 39-40; transformation optics).  In
   ``u`` the isotropic medium becomes ``eps' = eps diag(1/f, f, f)`` and
   ``mu' = diag(1/f, f, f)``.  Li's (1996) rules then read: the wall-normal
   ``eps'_uu = eps/f`` takes the inverse rule ``[[f/eps]]^-1``; the
   wall-tangential ``eps'_yy`` and the ``E_z`` elimination take the direct
   rule on ``eps f`` (``[[eps f]]``, ``[[eps f]]^-1``); ``mu'_yy = f`` is
   ``[[f]]`` and ``mu'_uu^-1``, ``mu'_zz^-1`` are ``[[f]]^-1``.  Only ``f``
   (never ``1/f``) enters, which is what lets the map reach ``f = 0`` at the
   walls.  The layer operators are those of the library's anisotropic-diagonal
   Rumpf form (:func:`~._core._layer_P_matrix` with ``mu`` restored).  The
   Fourier coefficients of ``f`` times a piecewise-constant value are exact
   (closed form, :meth:`_AsrMap.fourier_coeffs`) -- no sampling.

3. THE MAP.  On each sub-interval of length ``L`` between two consecutive
   transition points (``s = (u - a)/L``)::

       f(s) = 1 - eta * sum_k a_k cos(2 pi k s),   x(s) = a + L int_0^s f

   Profile ``'sine'`` (``a = (1,)``) is Granet's map; profile ``'sine4'``
   (``a = (4/3, -1/3)``, i.e. ``f = (8/3) sin^4(pi s)`` at ``eta = 1``) has
   ``f, f', f'', f'''`` all zero at the walls, so near a wall
   ``x - a ~ s^5``.  A corner field ``~ r^tau`` then becomes ``~ s^(5 tau)``
   in ``u``: the singularity that limits every uniform basis is removed
   rather than merely resolved more finely.  Any ``eta < 1`` leaves a linear
   zone ``x - a ~ (1 - eta) s`` at the wall in which the singularity keeps
   its exponent, which is why the default reaches ``eta = 1`` (from
   ``n_orders`` 15 on: :data:`_ASR_AUTO_N0`).  Measured on the
   resonant metal-strip stack of ``tests/unit/test_rcwa_stack_asr.py``:
   ``'sine'`` at ``eta = 1`` scatters between 1e-6 and 1e-4 over
   ``n_orders`` 40 .. 100, ``'sine4'`` reaches 2e-6 at 40 and agrees with
   itself to 1.3e-8 between 150 and 200 (``docs/rcwa_stack_asr.md``).

4. HALF-SPACES: AN ACCURATE OUTGOING DIRICHLET-TO-NEUMANN MAP.  This is the
   cause of the accuracy floor of the single-layer ASR in
   :func:`~.oned.rcwa_efficiency_1d` (``asr_eta``, about 1e-4 in TM) and the
   central fix here.  A half-space enters the S-matrix only through the
   relation between the tangential H and E of OUTGOING waves.  Two obvious
   choices are both wrong at the truncation edge:

   * the truncated plane-wave bridge ``G`` (columns = ``u`` coefficients of
     ``exp(i k_m x(u))``, applied as ``G^-1``): the high-order plane waves are
     not representable in ``N`` ``u`` harmonics, ``G`` becomes ill-conditioned
     and the result plateaus (measured 1e-5 .. 2e-5 on a dielectric grating
     whose plain answer converges to 5e-7);
   * the half-space's own truncated eigenmodes (Essig and Busch replace only
     the propagating ones): their high-order eigenvalues are distorted, and in
     TM the resulting outgoing admittance costs an ``O(N^-1)`` error (measured:
     ASR slower than the plain method on a dielectric grating).

   The outgoing admittance is instead computed in a ``u`` basis TWICE as long
   (:data:`_ASR_DTN_OVERSAMPLE`) from that basis's eigenmodes and RESTRICTED
   to the ``N`` retained harmonics (a Galerkin projection of an accurate
   operator, instead of an operator of a projection).  It agrees with the
   admittance built from exact plane waves on a fine ``x`` grid to round-off
   (oversampling 2 and 3 give identical results), and needs no ``x`` grid,
   which the ``eta = 1`` map (``1/f`` singular at the walls) would make
   expensive.  The half-space amplitudes are then the ``u`` coefficients of
   ``(E_u, E_y)``; the Rayleigh amplitude of order ``m`` is their exact
   projection ``x``-coefficient (rows of ``G^H``), and the incident field is
   the exact ``u`` representation of the incident plane wave.  Finite uniform
   LAYERS keep their own truncated modes: both propagation directions are
   present, so their basis is complete, and replacing them by exact projected
   operators was measured to change nothing.

5. BALANCED FIRST-ORDER EIGENSOLVE.  With ``f -> 0`` at the walls the local
   wavenumber ``k/f`` of the ``u`` basis is huge near a wall: the largest
   modal eigenvalue reaches 5e7 at ``n_orders`` 100.  The usual squared
   operator ``P Q`` then has norm ~1e12-1e15, and an eigensolver's absolute
   error ``eps_mach ||P Q||`` swamps the ``O(1)`` eigenvalues that carry the
   answer (measured: erratic 1e-5 .. 4e-4 spikes at isolated ``n_orders``).
   The layer is instead solved as the FIRST-order ``[[0, P'], [Q', 0]]``
   system in the frame of the uniform-medium eigenvectors, with a diagonal
   per-mode scaling that balances each ``2 x 2`` block (``d = (1+h^2)^(1/2)``
   on the TE partner, its inverse on the TM partner).  Its norm is ~ the
   largest modal eigenvalue, not its square, and the spikes are gone.

Scope: 1-D stacks; uniform layers and ``'li'`` analytic-shape layers
(``shapes=`` / ``segments=``); classical and conical incidence; NumPy only.
Everything else is refused with a message (see :func:`_asr_layer_specs`).

References: G. Granet, JOSA A 16, 2510 (1999); T. Vallius and M. Honkanen,
Opt. Express 10, 24 (2002); L. Li, JOSA A 13, 1870 (1996); G. Granet and
B. Guizal, JOSA A 13, 1019 (1996); P. Lalanne and G. M. Morris, JOSA A 13, 779
(1996); T. Weiss et al., Opt. Express 17, 8051 (2009); S. Essig and K. Busch,
Opt. Express 18, 23258 (2010).

Conventions: the solve runs in the library's INTERNAL frame, in which a
public profile ``eps(x)`` is represented by ``conj(eps(-x))`` (see
:func:`~._core._internal_cell`); the map is built for that mirrored profile.
Lengths inside :class:`_AsrMap` are in units of the period.
"""
from __future__ import annotations

import numpy as np

from ._core import (
    _C,
    _interface_smatrix,
    _inv_lam,
    _propagation_star,
    _redheffer_star,
    _sqrt_decay,
    _toeplitz_1d,
)

_TWO_PI = 2.0 * np.pi

#: Map profiles: the cosine coefficients ``a_k`` of
#: ``f(s) = 1 - eta * sum_k a_k cos(2 pi k s)`` on each sub-interval.  Both
#: sum to 1, so ``f = 1 - eta`` at the walls and ``f`` averages to 1 on every
#: sub-interval (the walls stay where they are and the Floquet lattice is kept).
_ASR_PROFILES = {
    "sine": (1.0,),
    "sine4": (4.0 / 3.0, -1.0 / 3.0),
}
_ASR_DEFAULT_PROFILE = "sine4"
_ASR_DEFAULT_ETA = "auto"

#: The ``eta='auto'`` rule: ``eta = clip((n_orders - N0) / (N1 - N0), 0, 1)``.
#: The full stretch clusters the harmonics so hard that the middle of each
#: segment is resolved ``1 + 5/3 = 2.7`` times more coarsely than by the plain
#: basis, which costs accuracy while ``n_orders`` is small.  Measured (TE and
#: TM, ``n_orders`` 4..20, ``eta`` 0 / 0.25 / 0.5 / 0.75 / 1) on a lossless
#: lamellar grating (eps 4 / 1, period 1000 nm, 800 nm, depth 300 nm), on the
#: resonant metal-strip stack and on the same stack at period 1200 nm: with
#: this rule ASR is at least as accurate as the plain basis at every
#: ``n_orders`` >= 6 of all six series except the metal stack's TM at 8, where
#: both are more than 5 % off; ``eta = 1`` everywhere would have lost to the
#: plain basis below ``n_orders`` 13 on the dielectric grating (TE 2.1e-3
#: against 2.9e-4 at 10).  At ``n_orders <= N0`` the rule gives ``eta = 0``
#: and the solve is the standard one.
_ASR_AUTO_N0 = 5
_ASR_AUTO_N1 = 15

#: Length factor of the auxiliary ``u`` basis the half-space admittance is
#: built in (item 4 of the module docstring).  2 and 3 give results equal to
#: round-off on every fixture measured.
_ASR_DTN_OVERSAMPLE = 2

#: Transition points closer than this (period units) are merged into one
#: coordinate line.  The merge moves a wall by at most this much.
_ASR_MERGE_TOL = 1e-9

#: Refuse when the Toeplitz matrix of the Jacobian is this ill-conditioned:
#: its inverse square root builds every uniform-medium mode.  Measured cond
#: on the 370 nm metal-strip stack with ``'sine4'`` at ``eta = 1``: 2.7e3 /
#: 1.7e7 / 2.7e8 at ``n_orders`` 50 / 150 / 300 (it grows as ``N^4``).
_ASR_COND_F_MAX = 1e13



def _normalize_asr(asr, fn_name="RCWAStack"):
    """``asr`` argument -> ``None`` (off) or the hashable spec
    ``(profile, eta)``.

    Accepted: ``None`` / ``False`` (off); ``True`` (``'sine4'``,
    ``eta='auto'``); a number ``eta`` in ``(0, 1]`` or ``'auto'`` (default
    profile); a dict with optional keys ``'eta'`` and ``'profile'``
    (``'sine4'`` or ``'sine'``)."""
    if asr is None or asr is False:
        return None
    if asr is True:
        return (_ASR_DEFAULT_PROFILE, _ASR_DEFAULT_ETA)
    profile, eta = _ASR_DEFAULT_PROFILE, _ASR_DEFAULT_ETA
    if isinstance(asr, dict):
        unknown = sorted(set(asr) - {"eta", "profile"})
        if unknown:
            raise ValueError(
                f"{fn_name}: asr dict takes the keys 'eta' and 'profile', got "
                f"{unknown[0]!r}.")
        profile = asr.get("profile", profile)
        eta = asr.get("eta", eta)
    elif isinstance(asr, (bool, np.bool_)):
        return (_ASR_DEFAULT_PROFILE, _ASR_DEFAULT_ETA) if asr else None
    else:
        eta = asr
    if profile not in _ASR_PROFILES:
        raise ValueError(
            f"{fn_name}: asr profile must be one of {sorted(_ASR_PROFILES)}, "
            f"got {profile!r}.")
    if isinstance(eta, str) and eta == "auto":
        return (profile, "auto")
    try:
        eta = float(eta)
    except (TypeError, ValueError):
        raise ValueError(
            f"{fn_name}: asr must be None/False, True, a stretch eta in "
            f"(0, 1] or 'auto', or a dict(eta=..., profile=...); got "
            f"{asr!r}.") from None
    if not (0.0 < eta <= 1.0):
        raise ValueError(
            f"{fn_name}: the asr stretch eta must be in (0, 1] (eta > 1 makes "
            f"the Jacobian f = dx/du negative; eta = 0 is no stretch -- pass "
            f"asr=None); got {eta}.")
    return (profile, eta)


def _resolved_eta(eta, n_orders):
    """The stretch a solve at ``n_orders`` uses (``'auto'``: the rule of
    :data:`_ASR_AUTO_N0` / :data:`_ASR_AUTO_N1`)."""
    if eta != "auto":
        return float(eta)
    t = (int(n_orders) - _ASR_AUTO_N0) / float(_ASR_AUTO_N1 - _ASR_AUTO_N0)
    return float(min(1.0, max(0.0, t)))


class _AsrMap:
    """The piecewise coordinate map ``x(u)`` of one stack (period units).

    ``breakpoints``: the transition points, sorted in ``[0, 1)``; sub-interval
    ``j`` is ``[b_j, b_{j+1})`` (the last one wraps to ``b_0 + 1``).  The
    ``u``-partition equals the ``x``-partition, so a wall at ``x = b_j`` sits
    at ``u = b_j``.  With no breakpoint the map is the identity.

    THE coordinate-map implementation of the package: the single-layer ASR in
    :mod:`.oned` samples its map through :meth:`samples` too."""

    def __init__(self, breakpoints, eta, profile="sine"):
        b = np.sort(np.asarray(breakpoints, dtype=float).reshape(-1) % 1.0)
        self.b = b
        self.K = int(b.size)
        self.L = (np.diff(np.append(b, b[0] + 1.0)) if self.K
                  else np.zeros(0))
        self.eta = float(eta)
        self.profile = profile
        self.a = tuple(float(v) for v in _ASR_PROFILES[profile])

    # -- the profile on one sub-interval ----------------------------------
    def f_of_s(self, s):
        """Jacobian ``f`` at local coordinate ``s`` in ``[0, 1]``."""
        s = np.asarray(s, dtype=float)
        out = np.ones_like(s)
        for k, ak in enumerate(self.a, start=1):
            out = out - self.eta * ak * np.cos(_TWO_PI * k * s)
        return out

    def xi_of_s(self, s):
        """``int_0^s f``: the local position, ``0`` at ``s = 0``, ``1`` at
        ``s = 1``."""
        s = np.asarray(s, dtype=float)
        out = s.copy()
        for k, ak in enumerate(self.a, start=1):
            out = out - self.eta * ak * np.sin(_TWO_PI * k * s) / (_TWO_PI * k)
        return out

    @property
    def f_max(self):
        """Upper bound of ``f`` (sizes the sampling grids)."""
        return 1.0 + self.eta * float(np.sum(np.abs(self.a)))

    # -- point evaluation --------------------------------------------------
    def _locate(self, v):
        """Sub-interval index and the wrapped value of ``v`` (either axis --
        the two partitions coincide)."""
        b0 = self.b[0]
        vv = b0 + np.mod(np.asarray(v, dtype=float) - b0, 1.0)
        edges = np.append(self.b, b0 + 1.0)
        j = np.clip(np.searchsorted(edges, vv, side="right") - 1, 0,
                    self.K - 1)
        return j, vv

    def samples(self, u):
        """``(x(u), f(u))`` for any real ``u`` (``x(u + 1) = x(u) + 1``)."""
        u = np.asarray(u, dtype=float)
        if self.K == 0:
            return u.copy(), np.ones_like(u)
        j, uu = self._locate(u)
        s = (uu - self.b[j]) / self.L[j]
        x = self.b[j] + self.L[j] * self.xi_of_s(s)
        return x + (u - uu), self.f_of_s(s)

    def inverse(self, x):
        """``(u(x), f(u(x)))``: Newton on the monotone ``xi(s) = t``, safeguarded
        by bisection bounds (``f`` may vanish at the walls)."""
        x = np.asarray(x, dtype=float)
        if self.K == 0:
            return x.copy(), np.ones_like(x)
        j, xx = self._locate(x)
        t = (xx - self.b[j]) / self.L[j]
        lo, hi = np.zeros_like(t), np.ones_like(t)
        s = t.copy()
        for _ in range(200):
            g = self.xi_of_s(s) - t
            lo = np.where(g < 0, s, lo)
            hi = np.where(g > 0, s, hi)
            df = self.f_of_s(s)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.where(df > 1e-300, g / df, 0.0)
            s_new = s - step
            bad = ~((s_new > lo) & (s_new < hi))
            s_new = np.where(bad, 0.5 * (lo + hi), s_new)
            done = np.max(np.abs(s_new - s)) < 1e-15
            s = s_new
            if done:
                break
        u = self.b[j] + self.L[j] * s
        return u + (x - xx), self.f_of_s(s)

    # -- exact Fourier coefficients ---------------------------------------
    def fourier_coeffs(self, values, n_coeffs):
        """EXACT ``c_p = int_0^1 g(u) exp(-i 2 pi p u) du``,
        ``p = -(n_coeffs-1) .. (n_coeffs-1)`` (the :func:`~._core._toeplitz_1d`
        layout), of ``g = values[j] * f(u)`` on sub-interval ``j``.

        Closed form: ``int_a^{a+L} exp(-i 2 pi q u) du
        = L exp(-i pi q (2a + L)) sinc(q L)``, and each cosine of the profile
        shifts ``q`` by ``+/- k/L``."""
        p = np.arange(-(n_coeffs - 1), n_coeffs).astype(float)
        out = np.zeros(p.shape, dtype=_C)
        for j in range(self.K):
            a, L = self.b[j], self.L[j]

            def seg(q, a=a, L=L):
                return L * np.exp(-1j * np.pi * q * (2.0 * a + L)) * np.sinc(q * L)
            term = seg(p).astype(_C)
            for k, ak in enumerate(self.a, start=1):
                ph = np.exp(-1j * _TWO_PI * k * a / L)
                term = term - 0.5 * self.eta * ak * (
                    ph * seg(p - k / L) + np.conj(ph) * seg(p + k / L))
            out = out + _C(values[j]) * term
        return out

    def plane_wave_columns(self, betas, beta0, n_orders):
        """``G[n, m] = int_0^1 exp(i 2 pi beta_m x(u)) exp(-i 2 pi beta_n u) du``
        and ``Gf`` (the same with a factor ``f(u)``): the ``u`` coefficients
        ``n = -n_orders..n_orders`` of the plane wave ``exp(i k_m x)``
        (``beta = k P / (2 pi)``, ``beta_n = beta0 + n``).  ``Gf`` is the
        covariant ``E_u = f E_x`` of a unit ``E_x``.

        The integrand ``exp(i 2 pi (beta_m x(u) - beta0 u))`` is periodic and
        smooth on each sub-interval (``x`` is ``C^2`` across a wall for
        ``'sine'``, ``C^4`` for ``'sine4'``), so the uniform-grid rule (one FFT
        per column) converges as a high power of the grid; the grid holds at
        least 8 samples per period of the fastest local oscillation."""
        betas = np.asarray(betas, dtype=float)
        nmax = float(np.max(np.abs(betas))) if betas.size else 0.0
        need = 8.0 * (self.f_max * nmax + abs(beta0) + n_orders + 8)
        n_s = int(2 ** int(np.ceil(np.log2(max(1024.0, need)))))
        u = np.arange(n_s) / n_s
        x, f = self.samples(u)
        idx = np.arange(-n_orders, n_orders + 1) % n_s
        G = np.empty((2 * n_orders + 1, betas.size), dtype=_C)
        Gf = np.empty_like(G)
        for c0 in range(0, betas.size, 64):
            bm = betas[c0:c0 + 64]
            ph = np.exp(1j * _TWO_PI * (np.outer(x, bm) - beta0 * u[:, None]))
            G[:, c0:c0 + 64] = (np.fft.fft(ph, axis=0) / n_s)[idx]
            Gf[:, c0:c0 + 64] = (np.fft.fft(ph * f[:, None], axis=0) / n_s)[idx]
        return G, Gf


# ===========================================================================
# Stack -> map + per-layer specs
# ===========================================================================

def _merge_points(points, tol=_ASR_MERGE_TOL):
    """Sorted unique points in ``[0, 1)`` with near-duplicates (also across the
    wrap) merged."""
    p = np.sort(np.mod(np.asarray(points, dtype=float), 1.0))
    out = []
    for v in p:
        if not out or v - out[-1] > tol:
            out.append(float(v))
    if len(out) > 1 and (out[0] + 1.0) - out[-1] <= tol:
        out.pop()
    return out


def _shapes_public_profile(eps_bg, shapes, period):
    """Public lamellar profile of a y-invariant shapes layer: its wall
    positions (period units) and a point evaluator ``value(x)``.  A
    full-period rectangle makes the layer uniform (no walls)."""
    edges, rects = [], []
    for sh in shapes:
        cx = float(sh["center"][0]) / period
        wx = float(sh["size"][0]) / period
        if wx >= 1.0 - 1e-12:
            return [], (lambda x, e=_C(sh["eps"]): np.full(np.shape(x), e))
        lo = np.mod(cx - 0.5 * wx, 1.0)
        edges += [lo, np.mod(lo + wx, 1.0)]
        rects.append((lo, wx, _C(sh["eps"])))

    def value(x):
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, _C(eps_bg), dtype=_C)
        for lo, wx, e in rects:
            out = np.where(np.mod(x - lo, 1.0) < wx, e, out)
        return out
    return edges, value


def _asr_layer_specs(stack, layers):
    """Validate the stack for ASR and return ``(breakpoints, specs)`` in the
    INTERNAL frame: ``breakpoints`` (period units) of the shared map and per
    layer ``('u', eps)`` (uniform) or ``('p', value(x))`` (patterned; the
    evaluator returns INTERNAL permittivities at internal positions).

    Refused, each with a ``ValueError`` naming the layer: 2-D stacks;
    ``eps_cell`` rasters (a raster has no exact wall positions to put on
    coordinate lines -- use ``segments=`` or ``shapes=``); tensor layers;
    shapes layers solved with ``'laurent'`` (ASR is built on the inverse rule,
    and a 1-D shapes layer resolves to ``'laurent'`` only when it is
    y-varying or has a zero background, neither of which ASR can represent);
    JAX-traced values."""
    from ...backend import is_jax_array
    if not stack.is_1d:
        raise ValueError(
            "RCWAStack(asr=...): adaptive spatial resolution is implemented "
            "for 1-D stacks only (period_y / n_orders_y unset).")
    P = stack.period_x
    pts, specs = [], []
    for i, L in enumerate(layers):
        if is_jax_array(L.data) or is_jax_array(L.thickness):
            raise ValueError(
                "RCWAStack(asr=...): JAX-traced layer values are not supported "
                "by adaptive spatial resolution (NumPy only); solve without "
                "asr for a differentiable stack.")
        if L.kind == "uniform":
            specs.append(("u", complex(np.conj(_C(L.data)))))
            continue
        if L.kind == "shapes" and getattr(L, "formulation", "laurent") == "li":
            eps_bg, shapes = L.data
            edges, value = _shapes_public_profile(eps_bg, shapes, P)
            if not edges:
                specs.append(("u", complex(np.conj(value(0.0)))))
                continue
            pts += [np.mod(-e, 1.0) for e in edges]          # point mirror
            specs.append(("p", (lambda xi, v=value:
                                np.conj(v(np.mod(-np.asarray(xi), 1.0))))))
            continue
        what = {"iso": "an eps_cell raster (no exact wall positions to put on "
                       "coordinate lines; pass the layer as segments= or "
                       "shapes=)",
                "tensor": "an anisotropic tensor layer",
                "shapes": "a shapes layer solved with formulation='laurent' "
                          "(ASR is built on the inverse rule; a 1-D shapes "
                          "layer resolves to 'laurent' only when it varies "
                          "along y or has a zero background)"}.get(L.kind,
                                                                  L.kind)
        raise ValueError(
            f"RCWAStack(asr=...): layer {i} is {what}, which adaptive spatial "
            f"resolution does not support.  Supported: uniform layers and "
            f"'li' shapes= / segments= layers.")
    return _merge_points(pts), specs


# ===========================================================================
# The solve
# ===========================================================================

def _hermitian_parts(F):
    """``(F^{1/2}, F^{-1/2}, F^{-1}, cond)`` of the Hermitian positive
    definite Toeplitz ``F`` (one ``eigh``)."""
    F = 0.5 * (F + F.conj().T)
    d, U = np.linalg.eigh(F)
    if not (d[0] > 0.0):
        raise np.linalg.LinAlgError(
            "RCWAStack(asr=...): the Toeplitz matrix of the coordinate "
            f"Jacobian is not positive definite (smallest eigenvalue {d[0]:.3e}"
            "); lower the asr stretch eta.")
    Uh = U.conj().T
    return ((U * np.sqrt(d)) @ Uh, (U / np.sqrt(d)) @ Uh, (U / d) @ Uh,
            float(d[-1] / d[0]))


def _uniform_basis(F, kx):
    """The shared uniform-medium basis: ``H = F^{-1/2} K F^{-1/2}`` (Hermitian;
    the discrete ``-i d/dx`` of the ``u`` basis), ``H = Z diag(h) Z^H``.
    Returns ``(Fs, Fis, Fi, h, Z, condF)``."""
    Fs, Fis, Fi, condF = _hermitian_parts(F)
    H = (Fis * kx[None, :]) @ Fis
    h, Z = np.linalg.eigh(0.5 * (H + H.conj().T))
    return Fs, Fis, Fi, h, Z, condF


def _uniform_admittance(eps, h, ky):
    """The four per-mode admittance diagonals of a uniform medium in the
    mode basis (the plain-RCWA ``Q diag(1/lam)`` with ``k_x -> h``), and
    ``lam``.  Ordered ``(xx, xy, yx, yy)``: ``U_u`` from the ``x``/``y``
    modes, ``U_y`` from the ``x``/``y`` modes."""
    lam = _sqrt_decay(h ** 2 + ky ** 2 - eps + 0j)
    il = _inv_lam(lam)
    return (h * ky * il, (eps - h ** 2) * il, (ky ** 2 - eps) * il,
            -ky * h * il), lam


def _uniform_modes(eps, Wx, Wy, h, ky):
    """Modes of a uniform layer: ``W = blockdiag(F^{1/2} Z, F^{-1/2} Z)``
    (``E_u`` of the x-modes, ``E_y`` of the y-modes), ``V = W``-columns times
    the admittance diagonals, ``lam`` doubled.  Exact eigenvectors of the
    layer's ``P Q`` in the truncated basis (``K F^{-1} K F^{-1} F^{1/2} Z =
    F^{1/2} Z diag(h^2)``), in classical and conical incidence."""
    (axx, axy, ayx, ayy), lam = _uniform_admittance(eps, h, ky)
    N = Wx.shape[0]
    Z0 = np.zeros((N, N), dtype=_C)
    W = np.block([[Wx, Z0], [Z0, Wy]])
    V = np.block([[Wx * axx[None, :], Wx * axy[None, :]],
                  [Wy * ayx[None, :], Wy * ayy[None, :]]])
    return W, V, np.concatenate([lam, lam])


def _forward_half(mu):
    """Indices of the forward-branch half of the ``+/-`` pairs of a
    first-order generator's eigenvalues ``mu``: the ``n/2`` whose ``mu`` is
    closest to ``+sqrt_decay(mu^2)`` (always exactly half, even where a pair
    is near cut-off), and that ``lam``."""
    lam = _sqrt_decay(mu ** 2)
    den = np.abs(lam) ** 2
    score = np.real(mu * np.conj(lam)) / np.where(den > 0, den, 1.0)
    keep = np.sort(np.argsort(-score, kind="stable")[: mu.size // 2])
    return keep, lam[keep]


def _patterned_modes(mp, vals, F, Fs, Fis, Fi, h, Z, kx, ky, M):
    """Modes ``(W, V, lam, EF)`` of a patterned layer whose INTERNAL
    permittivity on sub-interval ``j`` is ``vals[j]`` (items 2 and 5 of the
    module docstring).  ``EF = [[eps f]]`` is returned for the ``E_z``
    recovery of the internal field."""
    N = 2 * M + 1
    nco = N
    vals = np.asarray(vals, dtype=_C)
    EF = _toeplitz_1d(mp.fourier_coeffs(vals, nco), M)
    EFi = np.linalg.inv(EF)
    Cn = np.linalg.inv(_toeplitz_1d(mp.fourier_coeffs(1.0 / vals, nco), M))
    I = np.eye(N, dtype=_C)
    KF = kx[:, None] * (Fis @ Z)            # K F^{-1/2} Z
    hd = np.diag(h.astype(_C))
    # Blocks of P' = T^-1 P T and Q' = T^-1 Q T, T = blockdiag(F^{1/2} Z,
    # F^{-1/2} Z) for both (E_u, E_y) and (U_u, U_y).  Derivation: the module
    # docstring and tests/unit/test_rcwa_stack_asr.py (planar / conical
    # agreement).
    FsZ, FisZ = Fs @ Z, Fis @ Z
    A_tm = I - KF.conj().T @ EFi @ KF                       # P'_{u,y}
    B_tm = -(FsZ.conj().T @ Cn @ FsZ)                       # Q'_{y,u}
    B_te = FisZ.conj().T @ EF @ FisZ - hd * hd              # Q'_{u,y}
    d_te = np.sqrt(1.0 + h ** 2)                            # U_u partner
    d_tm = 1.0 / d_te                                       # U_y partner
    Wx = FsZ
    Wy = FisZ
    if ky == 0.0:
        out_W = np.zeros((2 * N, 2 * N), dtype=_C)
        out_V = np.zeros_like(out_W)
        out_l = np.zeros(2 * N, dtype=_C)
        Zn = np.zeros((N, N), dtype=_C)
        # TM: (E_u, U_y); TE: (E_y, U_u) with P'_{y,u} = -I.
        for blk, (Ab, Bb, d, We, Wh) in enumerate((
                (A_tm, B_tm, d_tm, Wx, Wy), (-I, B_te, d_te, Wy, Wx))):
            gen = np.block([[Zn, Ab * d[None, :]], [Bb / d[:, None], Zn]])
            mu, vec = np.linalg.eig(gen)
            keep, lam = _forward_half(mu)
            Wb = We @ vec[:N, keep]
            Vb = Wh @ (d[:, None] * vec[N:, keep])
            if blk == 0:
                out_W[:N, :N], out_V[N:, :N], out_l[:N] = Wb, Vb, lam
            else:
                out_W[N:, N:], out_V[:N, N:], out_l[N:] = Wb, Vb, lam
        return out_W, out_V, out_l, EF, Cn
    # Conical: the full 4N first-order system (Ky = ky I).
    KyEFi = ky * EFi
    P00 = FisZ.conj().T @ (kx[:, None] * KyEFi) @ FsZ
    P01 = A_tm
    P10 = FsZ.conj().T @ (ky * KyEFi - Fi) @ FsZ
    P11 = -(FsZ.conj().T @ (KyEFi * kx[None, :]) @ FisZ)
    Q00 = ky * hd
    Q01 = B_te
    Q10 = (ky ** 2) * I + B_tm
    Q11 = -ky * hd
    D = np.concatenate([d_te, d_tm])
    Pp = np.block([[P00, P01], [P10, P11]])
    Qp = np.block([[Q00, Q01], [Q10, Q11]])
    Z2 = np.zeros((2 * N, 2 * N), dtype=_C)
    gen = np.block([[Z2, Pp * D[None, :]], [Qp / D[:, None], Z2]])
    mu, vec = np.linalg.eig(gen)
    keep, lam = _forward_half(mu)
    e = vec[: 2 * N, keep]
    eta_ = D[:, None] * vec[2 * N:, keep]
    W = np.vstack([Wx @ e[:N], Wy @ e[N:]])
    V = np.vstack([Wx @ eta_[:N], Wy @ eta_[N:]])
    return W, V, lam, EF, Cn


def _half_space_basis(mp, kx0, wl, period, M, Mb):
    """The auxiliary ``2 Mb + 1`` uniform-medium basis both half-spaces share
    (item 4 of the module docstring): its modal ``h`` (ascending, so big index
    ``Mb + m`` is order ``m``), the retained rows of the mode columns
    ``F^{1/2} Z`` / ``F^{-1/2} Z``, and the matching rows of ``W^-1``
    (``Z^H F^{-1/2}`` / ``Z^H F^{1/2}``: the two families are mutually dual)."""
    Nb = 2 * Mb + 1
    kxb = kx0 + np.arange(-Mb, Mb + 1) * (wl / period)
    Fb = _toeplitz_1d(mp.fourier_coeffs(np.ones(mp.K), Nb), Mb)
    Fs, Fis, _Fi, _c = _hermitian_parts(Fb)
    H = (Fis * kxb[None, :]) @ Fis
    h, Z = np.linalg.eigh(0.5 * (H + H.conj().T))
    FY, Y = Fs @ Z, Fis @ Z
    sl = slice(Mb - M, Mb + M + 1)
    return dict(h=h, kxb=kxb, Mb=Mb, FYr=FY[sl], Yr=Y[sl],
                Aix=Y.conj().T[:, sl], Aiy=FY.conj().T[:, sl])


def _half_space_admittance(basis, eps, ky):
    """Outgoing admittance ``Y`` (``U = Y E`` for the ``(E_u, E_y)`` /
    ``(U_u, U_y)`` ``u`` coefficients) of a uniform half-space: the auxiliary
    basis's exact modal admittance, restricted to the retained harmonics."""
    (axx, axy, ayx, ayy), _lam = _uniform_admittance(eps, basis["h"], ky)
    A_, B_ = basis["FYr"], basis["Yr"]
    Ai, Bi = basis["Aix"], basis["Aiy"]
    return np.block([[A_ @ (axx[:, None] * Ai), A_ @ (axy[:, None] * Bi)],
                     [B_ @ (ayx[:, None] * Ai), B_ @ (ayy[:, None] * Bi)]])


def _public_inv_eps(kind, data):
    """``x -> 1/eps_pub(x)`` (``x`` in public period units) of one layer, for
    recovering the discontinuous ``E_x = D_x / eps`` of the internal field.  On
    a wall it returns the mean of the two one-sided values (the convention a
    Fourier series takes at a jump)."""
    if kind == "u":
        e = complex(np.conj(data))
        return lambda x, e=e: np.full(np.shape(x), 1.0 / e, dtype=_C)

    def inv(x, v=data):
        x = np.asarray(x, dtype=float)
        d = 1e-12
        lo = 1.0 / np.conj(v(np.mod(-(x - d), 1.0)))
        hi = 1.0 / np.conj(v(np.mod(-(x + d), 1.0)))
        return 0.5 * (lo + hi)
    return inv


def _asr_solve(stack, layers, *, spec, kx0, ky0, wl, k0, eps_sup, eps_sub,
               retain_internal=False):
    """The ASR stack solve in the internal frame.

    Returns ``None`` when no layer is patterned (nothing to adapt) or the
    resolved stretch is 0 (``eta='auto'`` at low ``n_orders``) -- the caller
    then runs the standard solve -- else ``(rt, internal)``: ``rt`` the list of
    ``(r, t)`` Rayleigh-amplitude ``2N`` vectors (``[E_x; E_y]`` per order,
    internal frame) for incident ``E_x`` and ``E_y`` -- the form the stack's
    efficiency tail consumes -- and ``internal`` the retained data for
    :meth:`RCWAResult.internal_field` / ``layer_absorption`` (or ``None``)."""
    profile, eta = spec
    pts, specs = _asr_layer_specs(stack, layers)
    eta = _resolved_eta(eta, stack.nox)
    if not pts or eta == 0.0:
        return None
    mp = _AsrMap(pts, eta, profile)
    M = int(stack.nox)
    N = 2 * M + 1
    P = stack.period_x
    orders = np.arange(-M, M + 1)
    kx = (kx0 + orders * (wl / P)).astype(float)
    ky = float(ky0)
    F = _toeplitz_1d(mp.fourier_coeffs(np.ones(mp.K), N), M)
    Fs, Fis, Fi, h, Z, condF = _uniform_basis(F, kx)
    if condF > _ASR_COND_F_MAX:
        raise np.linalg.LinAlgError(
            f"RCWAStack(asr=...): the coordinate Jacobian's Toeplitz matrix "
            f"has condition number {condF:.2e} (> {_ASR_COND_F_MAX:.0e}) at "
            f"n_orders={M}; its inverse square root builds every uniform-"
            f"medium mode.  Lower n_orders or the asr stretch eta.")
    Wx, Wy = Fs @ Z, Fis @ Z
    mids = mp.b + 0.5 * mp.L                 # sub-interval midpoints (x = u)
    cache, modes, inv_eps = {}, [], []
    for kind, data in specs:
        if kind == "u":
            key = ("u", data)
        else:
            vals = np.asarray(data(mids), dtype=_C)
            key = ("p",) + tuple(complex(v) for v in vals)
        if key not in cache:
            if kind == "u":
                W, V, lam = _uniform_modes(data, Wx, Wy, h, ky)
                EF, DX = data * F, data * Fi
            else:
                W, V, lam, EF, DX = _patterned_modes(mp, vals, F, Fs, Fis, Fi,
                                                     h, Z, kx, ky, M)
            cache[key] = (W, V, lam, EF, DX)
        modes.append(cache[key])
        inv_eps.append(_public_inv_eps(kind, data))
    Mb = _ASR_DTN_OVERSAMPLE * M
    I2 = np.eye(2 * N, dtype=_C)
    hb = _half_space_basis(mp, kx0, wl, P, M, Mb)
    Vr = _half_space_admittance(hb, eps_sup, ky)
    Vt = _half_space_admittance(hb, eps_sub, ky)
    thick = [float(L.thickness) for L in layers]
    S = _interface_smatrix(I2, Vr, modes[0][0], modes[0][1])
    S = _propagation_star(S, modes[0][2], k0 * thick[0])
    for i in range(1, len(modes)):
        S = _redheffer_star(S, _interface_smatrix(modes[i - 1][0],
                                                  modes[i - 1][1],
                                                  modes[i][0], modes[i][1]))
        S = _propagation_star(S, modes[i][2], k0 * thick[i])
    S = _redheffer_star(S, _interface_smatrix(modes[-1][0], modes[-1][1],
                                              I2, Vt))
    beta = kx * (P / wl)
    G, Gf = mp.plane_wave_columns(beta, float(kx0 * P / wl), M)
    # Incident wave: its exact u representation; Rayleigh amplitudes: the
    # exact x-Fourier coefficient of the truncated field (rows of G^H / Gf^H:
    # E_x(x) = E_u/f, so int E_x e^{-ik_m x} dx = int E_u e^{-ik_m x(u)} du).
    p0 = M
    zN = np.zeros(N, dtype=_C)
    inc = (np.concatenate([Gf[:, p0], zN]), np.concatenate([zN, G[:, p0]]))
    Px, Py = G.conj().T, Gf.conj().T
    rt = []
    for cinc in inc:
        r = S[0] @ cinc
        t = S[2] @ cinc
        rt.append((np.concatenate([Px @ r[:N], Py @ r[N:]]),
                   np.concatenate([Px @ t[:N], Py @ t[N:]])))
    internal = None
    if retain_internal:
        S_above, S_below, S_below_bot = stack._internal_partials(
            modes, I2, Vr, I2, Vt, k0)
        internal = dict(
            W=[m[0] for m in modes], V=[m[1] for m in modes],
            lam=[m[2] for m in modes], EPS=[m[3] for m in modes],
            thick=thick, S_above=S_above, S_below=S_below,
            S_below_bot=S_below_bot,
            asr=dict(map=mp, Fi=Fi, inc=inc, beta=beta,
                     DX=[m[4] for m in modes], inv_eps=inv_eps))
    return rt, internal


# ===========================================================================
# Internal-field helpers (RCWAResult)
# ===========================================================================

def _asr_carriers(asr_info, xg, yg, period, ky, wavelength):
    """Public-frame carriers ``exp(i 2 pi beta_n u_pub(x)) exp(i ky k0 y)`` of
    the ``u`` harmonics on the field grid, ``(N, ny, nx)``.

    The solve's internal frame is the point mirror of the public one, so a
    public position ``x`` is the internal ``-x``: ``u_pub(x) = -U(-x)`` with
    ``U`` the internal map's inverse."""
    mp = asr_info["map"]
    beta = np.asarray(asr_info["beta"], dtype=float)
    u_int, _f = mp.inverse(-np.asarray(xg, dtype=float) / period)
    u_pub = -u_int                                         # period units
    k0 = _TWO_PI / wavelength
    cx = np.exp(1j * _TWO_PI * np.outer(beta, u_pub))      # (N, nx)
    cy = np.exp(1j * k0 * float(np.real(ky)) * np.asarray(yg, dtype=float))
    return cx[:, None, :] * cy[None, :, None]


def _asr_face_fluxes(result, info, i, inc):
    """``(Sz_top, Sz_bottom)`` of layer ``i`` for the public incident Jones
    ``inc``, from the ``u`` coefficients (Parseval; the covariant pair gives
    ``int (E_x H_y* - E_y H_x*) dx = int (E_u U_y* - E_y U_u*) du``), in the
    units :meth:`RCWAResult.layer_absorption` uses."""
    a = info["asr"]
    cinc = (np.conj(inc[0]) * a["inc"][0]
            + np.conj(inc[1]) * a["inc"][1]).astype(_C)
    cplus, cminus_bot = result._internal_cpm(info, i, cinc)
    W = np.asarray(info["W"][i])
    V = np.asarray(info["V"][i])
    lam = np.asarray(info["lam"][i])
    N = info["N"]
    k0, L = info["k0"], float(info["thick"][i])
    out = []
    for z in (0.0, L):
        fwd = cplus * np.exp(-lam * k0 * z)
        bwd = cminus_bot * np.exp(-lam * k0 * (L - z))
        s = W @ (fwd + bwd)
        u = -1j * (V @ (fwd - bwd))
        # internal -> public is a conjugation, which leaves Re(.) unchanged
        out.append(float(np.real(np.sum(s[:N] * np.conj(u[N:])
                                        - s[N:] * np.conj(u[:N])))))
    return out[0], out[1]
