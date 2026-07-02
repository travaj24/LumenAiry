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

import numpy as np

from .zcascade import interface_smatrix, layer_modes, propagation_smatrix, redheffer_star


class BORStack:
    def __init__(self, Rbig, m, *, n_substrate=1.0, n_superstrate=1.0, N=300):
        self.Rbig = float(Rbig)
        self.m = int(m)
        self.N = int(N)
        self.eps_sub = complex(n_substrate) ** 2
        self.eps_sup = complex(n_superstrate) ** 2
        self._layers = []          # list of (thickness, eps_profile callable)
        self.k0 = None

    # ---- geometry ------------------------------------------------------- #
    def add_layer(self, thickness, *, eps_profile=None, rings=None,
                  eps=None):
        """Add a z-layer of given ``thickness``.

        Exactly one of: ``eps_profile`` (callable r-array->eps), ``rings``
        (``(period, duty, n_ridge, n_groove)`` concentric binary grating), or
        ``eps`` (uniform scalar permittivity)."""
        if rings is not None:
            period, duty, n_r, n_g = rings
            er, eg = complex(n_r) ** 2, complex(n_g) ** 2

            def prof(r, period=period, duty=duty, er=er, eg=eg):
                return np.where((r % period) < duty * period, er, eg
                                ).astype(complex)
            fn = prof
        elif eps_profile is not None:
            def prof(r, ep=eps_profile):
                return np.asarray(ep(r), dtype=complex)
            fn = prof
        elif eps is not None:
            def prof(r, e=complex(eps)):
                return np.full_like(r, e, dtype=complex)
            fn = prof
        else:
            raise ValueError("add_layer needs eps_profile, rings, or eps")
        self._layers.append((float(thickness), fn))
        return self

    def set_source(self, wavelength=None, *, k0=None):
        self.k0 = (2 * np.pi / wavelength) if k0 is None else float(k0)
        return self

    # ---- solve ---------------------------------------------------------- #
    def _build(self, eps_fn):
        return layer_modes(self.m, self.Rbig, self.N, eps_fn, self.k0,
                           staggered=True)

    def solve(self):
        """Cascade the stack and return per-propagating-order R/T efficiencies.

        Returns a dict: ``q`` (incident propagating axial wavenumbers),
        ``gamma`` (their transverse wavenumbers), ``angles`` (far-field polar
        angle in the substrate, rad), ``R``/``T`` (reflected/transmitted power
        fraction summed over propagating orders), ``energy`` (R+T per incident
        order), ``S`` (the global S-matrix).

        The staggered basis is spurious-free, so NO reldiv filter is needed (the
        propagating criterion alone is exact) -- which lets us skip the two
        extra half-space eigensolves the reldiv tags would have cost.  Identical
        super/substrate reuse the same eig."""
        if self.k0 is None:
            raise RuntimeError("call set_source(...) before solve()")
        k0 = self.k0
        sup = self._build(lambda r: np.full_like(r, self.eps_sup, dtype=complex))
        sub = (sup if self.eps_sub == self.eps_sup
               else self._build(lambda r: np.full_like(r, self.eps_sub,
                                                        dtype=complex)))
        mids = [(thk, self._build(fn)) for thk, fn in self._layers]
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
