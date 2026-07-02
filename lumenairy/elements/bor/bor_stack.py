"""BOR-PMM: the ``BORStack`` public-API prototype (axisymmetric / body-of-
revolution stack), mirroring ``lumenairy.elements.pmm.PMMStack`` but in
cylindrical coordinates.

A stack of z-layers, each with a radial permittivity profile ``eps(r)`` at one
azimuthal order ``m`` (fields ~ ``exp(i m phi + i q z)``).  Built on the
spurious-free Yee div-conforming radial solver (``staggered=True``), so the
modal basis is complete + spurious-free and the cascade conserves energy to
machine precision.

Example
-------
    s = BORStack(Rbig=4.0, m=1, N=300, n_superstrate=1.4142, n_substrate=1.4142)
    s.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))   # period, duty, n_ridge, n_groove
    s.set_source(wavelength=2*pi/2.0)
    res = s.solve()        # res['R'], res['T'] per propagating order, res['angles']
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .zcascade import interface_smatrix, layer_modes, propagation_smatrix, redheffer_star

# v5.17.1 (audit P3-12): bound on the per-instance modal-basis LRU.  Each
# entry holds one layer's 2N x 2N modal matrices (W, V ~ 5.8 MB each at the
# default N=300), so 16 slots cap the cache at ~190 MB worst-case while
# still covering a full stack of distinct profiles plus a couple of swept
# k0 points.  Wave-4 LRU conventions: OrderedDict + move_to_end on hit +
# popitem(last=False) eviction.
_MODAL_CACHE_SIZE = 16


class BORStack:
    def __init__(self, Rbig, m, *, n_substrate=1.0, n_superstrate=1.0, N=300):
        # Input validation (audit P3-10) -- mirror the PMMStack/PMM2DStack
        # builder guards: a bad domain/grid propagates to plausible-looking
        # garbage (negative-radius grid, 1-point operators) deep in the build.
        Rbig = float(Rbig)
        if not np.isfinite(Rbig) or Rbig <= 0.0:
            raise ValueError(
                f"BORStack: Rbig (domain radius) must be > 0, got {Rbig}.")
        if not float(m).is_integer():
            raise ValueError(
                f"BORStack: m (azimuthal order) must be an integer, got {m!r}.")
        if not float(N).is_integer() or int(N) < 2:
            raise ValueError(
                f"BORStack: N (radial grid points) must be an integer >= 2, "
                f"got {N!r}.")
        self.Rbig = Rbig
        self.m = int(m)
        self.N = int(N)
        self.eps_sub = complex(n_substrate) ** 2
        self.eps_sup = complex(n_superstrate) ** 2
        self._layers = []   # list of (thickness, eps_profile callable, profile key)
        self.k0 = None
        # v5.17.1 (audit P3-12): per-instance modal-basis LRU keyed on
        # (profile fingerprint, k0).  Dedups identical layers WITHIN a solve
        # (an ABAB... periodic stack pays one eig per DISTINCT profile, not
        # per repetition) and reuses eigs ACROSS set_source+solve sweep
        # calls -- the BOR analog of PMMStack.prepare()'s _eig_cache.
        # Byte-identical R/T: a cache hit returns the exact dict a recompute
        # would have produced (same assembled matrices -> same LAPACK bits).
        self._modal_cache = OrderedDict()

    # ---- geometry ------------------------------------------------------- #
    def add_layer(self, thickness, *, eps_profile=None, rings=None,
                  eps=None):
        """Add a z-layer of given ``thickness``.

        Exactly one of: ``eps_profile`` (callable r-array->eps), ``rings``
        (``(period, duty, n_ridge, n_groove)`` concentric binary grating), or
        ``eps`` (uniform scalar permittivity).

        Layers with the SAME profile fingerprint (identical ``rings`` /
        ``eps`` parameters, or the same ``eps_profile`` callable object)
        share one eigensolve per ``k0`` via the modal LRU (audit P3-12)."""
        if rings is not None:
            period, duty, n_r, n_g = rings
            er, eg = complex(n_r) ** 2, complex(n_g) ** 2

            def prof(r, period=period, duty=duty, er=er, eg=eg):
                return np.where((r % period) < duty * period, er, eg
                                ).astype(complex)
            fn = prof
            key = ("rings", float(period), float(duty), er, eg)
        elif eps_profile is not None:
            def prof(r, ep=eps_profile):
                return np.asarray(ep(r), dtype=complex)
            fn = prof
            # Fingerprint by the USER callable's identity (the key tuple
            # keeps it alive, so the id can't be recycled); repeated
            # add_layer with the same callable dedups.  Unhashable callables
            # fall back to the per-call wrapper (unique -> never shared).
            try:
                hash(eps_profile)
                key = ("profile", eps_profile)
            except TypeError:
                key = ("profile", fn)
        elif eps is not None:
            def prof(r, e=complex(eps)):
                return np.full_like(r, e, dtype=complex)
            fn = prof
            key = ("eps", complex(eps))
        else:
            raise ValueError("add_layer needs eps_profile, rings, or eps")
        # Thickness validation (audit P3-10) -- mirror PMM2DStack.add_layer:
        # a NEGATIVE thickness flips the propagation exponent exp(iqL) so
        # forward-oriented evanescent modes GROW, silently destabilizing the
        # Redheffer cascade instead of raising.
        thickness = float(thickness)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("BORStack.add_layer: thickness must be > 0")
        self._layers.append((thickness, fn, key))
        return self

    def set_source(self, wavelength=None, *, k0=None):
        # Source validation (audit P3-10): wavelength <= 0 gives k0 = inf /
        # negative, and solve() then silently returns EMPTY R/T.
        if k0 is None:
            if wavelength is None:
                raise ValueError(
                    "BORStack.set_source: give wavelength or k0.")
            wavelength = float(wavelength)
            if not np.isfinite(wavelength) or wavelength <= 0.0:
                raise ValueError(
                    f"BORStack.set_source: wavelength must be > 0, got "
                    f"{wavelength}.")
            self.k0 = 2 * np.pi / wavelength
        else:
            k0 = float(k0)
            if not np.isfinite(k0) or k0 <= 0.0:
                raise ValueError(
                    f"BORStack.set_source: k0 must be > 0, got {k0}.")
            self.k0 = k0
        return self

    # ---- solve ---------------------------------------------------------- #
    def _build(self, profile_key, eps_fn):
        """Modal basis of one layer, memoized on (profile fingerprint, k0)
        in the per-instance LRU (audit P3-12).  Identical profiles share one
        dense eigensolve within a solve() AND across set_source+solve sweep
        calls; a hit returns the SAME dict a recompute would (solve() never
        mutates it)."""
        ck = (profile_key, self.k0, self.m, self.Rbig, self.N)
        hit = self._modal_cache.get(ck)
        if hit is not None:
            self._modal_cache.move_to_end(ck)     # LRU touch
            return hit
        L = layer_modes(self.m, self.Rbig, self.N, eps_fn, self.k0,
                        staggered=True)
        self._modal_cache[ck] = L
        if len(self._modal_cache) > _MODAL_CACHE_SIZE:
            self._modal_cache.popitem(last=False)
        return L

    def solve(self):
        """Cascade the stack and return per-propagating-order R/T efficiencies.

        Returns a dict: ``q`` (incident propagating axial wavenumbers),
        ``gamma`` (their transverse wavenumbers), ``angles`` (incident-mode
        polar angle in the SUPERSTRATE, rad -- computed from ``eps_sup`` and
        the superstrate modal ``q``; for ``n_substrate != n_superstrate`` the
        transmitted orders' substrate-side angles differ by Snell refraction,
        audit P3-11), ``R``/``T`` (reflected/transmitted power fraction summed
        over propagating orders), ``energy`` (R+T per incident order), ``S``
        (the global S-matrix).

        The staggered basis is spurious-free, so NO reldiv filter is needed (the
        propagating criterion alone is exact) -- which lets us skip the two
        extra half-space eigensolves the reldiv tags would have cost.  Layers
        with identical profile fingerprints (incl. super/substrate) share one
        eig via the modal LRU (audit P3-12)."""
        if self.k0 is None:
            raise RuntimeError("call set_source(...) before solve()")
        k0 = self.k0
        sup = self._build(("eps", self.eps_sup),
                          lambda r: np.full_like(r, self.eps_sup, dtype=complex))
        sub = (sup if self.eps_sub == self.eps_sup
               else self._build(("eps", self.eps_sub),
                                lambda r: np.full_like(r, self.eps_sub,
                                                       dtype=complex)))
        mids = [(thk, self._build(key, fn)) for thk, fn, key in self._layers]
        # cascade: sup -> [mid prop mid] ... -> sub
        S = interface_smatrix(sup["W"], sup["V"], mids[0][1]["W"], mids[0][1]["V"]) \
            if mids else interface_smatrix(sup["W"], sup["V"], sub["W"], sub["V"])
        for i, (thk, L) in enumerate(mids):
            S = redheffer_star(S, propagation_smatrix(L["q"], thk))
            nxt = mids[i + 1][1] if i + 1 < len(mids) else sub
            S = redheffer_star(S, interface_smatrix(L["W"], L["V"],
                                                    nxt["W"], nxt["V"]))
        S11, S12, S21, S22 = S
        q = sup["q"]

        def prop(L, eps):
            # Classify in the DIMENSIONLESS axial index q/k0 (audit P2-06):
            # absolute thresholds on q (units 1/length) silently returned
            # empty R/T for small-k0 unit systems (e.g. nm-scale).  The
            # constants are the validated k0=2.0 thresholds divided by 2,
            # so behavior at the validated scale is bit-identical.
            qn = L["q"] / k0
            return np.where((np.abs(qn.imag) < 5e-5) & (qn.real > 0.05)
                            & (np.sqrt(eps).real - qn.real > -5e-10))[0]
        inc = prop(sup, self.eps_sup)
        out = prop(sub, self.eps_sub)
        R = np.array([np.sum([abs(S11[jp, j]) ** 2 for jp in inc]) for j in inc])
        T = np.array([np.sum([abs(S21[jp, j]) ** 2 for jp in out]) for j in inc])
        qi = q[inc].real
        gamma = np.sqrt(np.maximum(self.eps_sup.real * k0 ** 2 - qi ** 2, 0.0))
        angles = np.arcsin(np.clip(gamma / (np.sqrt(self.eps_sup.real) * k0),
                                   0, 1))
        return dict(q=qi, gamma=gamma, angles=angles, R=R, T=T, energy=R + T,
                    inc=inc, out=out, S=S)
