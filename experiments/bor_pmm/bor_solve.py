"""BOR-PMM Milestone 5: the high-level axisymmetric stack solver (the prototype
of the eventual ``BORStack`` public API).

Pipeline: per-layer M2 radial vector modes (PEC-wall clean half-spaces, M5a) ->
flux-normalized basis -> M4 z-cascade S-matrix -> physical-mode R/T efficiencies
+ cylindrical far-field orders (M5b Fourier-Bessel / vortex Hankel).

ACCURACY (honest): the FD vector discretization emits spurious divergence-
violating modes whose real-q members leak ~1-2% (max ~4%) of the energy into
unphysical channels -- a floor that does NOT decrease with N (measured 3.8e-2 at
N=200 AND N=400).  So this prototype is correct to ~1-2%, not to machine
precision.  The clean fix (no spurious modes) is the div-conforming SEM
re-discretization -- which is also the gate for the production library port and
the full multi-order GATE 4 (the Cartesian-limit diffraction test).
"""
from __future__ import annotations

import numpy as np
from coupled_radial_eigensolver import radial_coupled_modes
from zcascade import interface_smatrix, layer_modes, propagation_smatrix, redheffer_star


def _flux(L, j):
    N = L["N"]
    Er, Ephi = L["W"][:N, j], L["W"][N:, j]
    hr, hphi = L["V"][:N, j], L["V"][N:, j]
    return np.real(np.sum((Er * np.conj(hphi) - Ephi * np.conj(hr)) * L["wq"]))


def _flux_normalize(L):
    """Scale each mode column to unit |z-flux| (propagating) / unit field-norm
    (evanescent), so a flux-normalized S-matrix has ``|S|^2`` = power fraction."""
    L = dict(L)
    W, V = L["W"].copy(), L["V"].copy()
    for j in range(W.shape[1]):
        P = _flux(L, j)
        s = (1.0 / np.sqrt(abs(P)) if abs(P) > 1e-10
             else 1.0 / np.sqrt(np.sum(np.abs(W[:, j]) ** 2) + 1e-300))
        W[:, j] *= s
        V[:, j] *= s
    L["W"], L["V"] = W, V
    L["flux"] = np.array([_flux(L, j) for j in range(W.shape[1])])
    return L


def build_layer(m, Rbig, N, eps_profile, k0, *, wall="pec", thickness=None):
    """A flux-normalized layer with its physical-mode flag (reldiv-tagged)."""
    L = _flux_normalize(layer_modes(m, Rbig, N, eps_profile, k0, wall=wall))
    L["reldiv"] = np.array([md["reldiv"] for md in
                            radial_coupled_modes(m, Rbig, N, eps_profile, k0,
                                                 wall=wall)])
    L["thickness"] = thickness
    return L


def _physical_propagating(L, reldiv_tol=0.5):
    return (np.abs(L["q"].imag) < 1e-4) & (L["q"].real > 0.1) & \
           (L["reldiv"] < reldiv_tol)


def solve(layers, k0):
    """Cascade a list of ``build_layer`` layers (first/last = semi-infinite
    super/substrate) and return per-incident-mode R/T efficiencies over the
    physical propagating channels, plus the global S-matrix.

    Returns a dict: ``S`` (S-matrix), ``inc`` (superstrate physical-prop indices),
    ``out`` (substrate ...), ``R``/``T`` (arrays over ``inc``: total reflected /
    transmitted power fraction), ``energy`` (R+T per incident mode).
    """
    S = interface_smatrix(layers[0]["W"], layers[0]["V"],
                          layers[1]["W"], layers[1]["V"])
    for i in range(1, len(layers) - 1):
        S = redheffer_star(S, propagation_smatrix(layers[i]["q"],
                                                  layers[i]["thickness"]))
        S = redheffer_star(S, interface_smatrix(layers[i]["W"], layers[i]["V"],
                                                layers[i + 1]["W"],
                                                layers[i + 1]["V"]))
    S11, S12, S21, S22 = S
    inc = np.where(_physical_propagating(layers[0]))[0]
    out = np.where(_physical_propagating(layers[-1]))[0]
    R = np.array([np.sum([abs(S11[jp, j]) ** 2 for jp in inc]) for j in inc])
    T = np.array([np.sum([abs(S21[jp, j]) ** 2 for jp in out]) for j in inc])
    return dict(S=S, inc=inc, out=out, R=R, T=T, energy=R + T,
                q_inc=layers[0]["q"][inc])
