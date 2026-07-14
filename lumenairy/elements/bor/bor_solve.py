"""BOR-PMM Milestone 5: the high-level axisymmetric stack solver (the prototype
of the eventual ``BORStack`` public API).

Pipeline: per-layer radial vector modes (closed-wall clean half-spaces, M5a) ->
flux-normalized basis -> M4 z-cascade S-matrix -> physical-mode R/T efficiencies
+ cylindrical far-field orders (M5b Fourier-Bessel / vortex Hankel).

BASIS (follow-up to AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13):
``build_layer`` now defaults to the Yee div-conforming STAGGERED basis -- the
same spurious-free discretization production ``BORStack`` uses -- so the
cascade conserves energy to machine precision.  The historical NODAL FD basis
(``basis='nodal'``) is retained for its legacy gates but is catastrophically
unreliable on large cells: its spurious divergence-violating mode sea (~40-50%
of the basis at Rbig ~ 12 lambda) carries zero z-flux, so each spurious mode's
forward/backward orientation is decided by the SIGN OF NOISE.  When adjacent
layers share most of their cross-section, near-identical spurious modes can
orient OPPOSITELY, making a layer-a "forward" combination equal a layer-b
"backward" combination -- which renders the interface transmission block
``a + b`` numerically singular (measured cond ~ 2.6e15 at Rbig = 12 lambda)
and blows the cascade energy up to ~1e29 (small cells only leak the documented
~1-4% floor, which does NOT decrease with N).
"""
from __future__ import annotations

import numpy as np

from .coupled_radial_eigensolver import radial_coupled_modes
from .zcascade import interface_smatrix, layer_modes, propagation_smatrix, redheffer_star


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
    N, wq = L["N"], np.real(L["wq"])
    for j in range(W.shape[1]):
        P = _flux(L, j)
        # flux threshold RELATIVE to the mode's own r*dr field norm (same
        # measure as the flux -> unit-invariant; absolute 1e-10 silently
        # mis-normalized meter-scale inputs)
        fnrm = np.sum((np.abs(W[:N, j]) ** 2 + np.abs(W[N:, j]) ** 2) * wq)
        s = (1.0 / np.sqrt(abs(P)) if abs(P) > 1e-10 * fnrm
             else 1.0 / np.sqrt(np.sum(np.abs(W[:, j]) ** 2) + 1e-300))
        W[:, j] *= s
        V[:, j] *= s
    L["W"], L["V"] = W, V
    L["flux"] = np.array([_flux(L, j) for j in range(W.shape[1])])
    return L


def build_layer(m, Rbig, N, eps_profile, k0, *, wall="pec", thickness=None,
                basis="staggered"):
    """A flux-normalized layer with its physical-mode flag (reldiv-tagged).

    ``basis='staggered'`` (default) uses the Yee div-conforming discretization
    (spurious-free; the production ``BORStack`` basis) -- the cascade then
    conserves energy to machine precision at any cell size.  ``basis='nodal'``
    keeps the historical FD basis (see the module docstring for why it blows
    up on large cells); its spurious modes are tagged by ``reldiv`` for the
    ``_physical_propagating`` filter.  The staggered wall is the closed
    Dirichlet wall, so ``wall`` must stay ``'pec'`` there.
    """
    if basis == "staggered":
        if wall != "pec":
            raise ValueError("basis='staggered' builds in the closed Dirichlet "
                             "wall; wall must be 'pec' (got %r)" % (wall,))
        # already flux-normalized per column inside _layer_modes_staggered;
        # re-running _flux_normalize here would apply the single-grid ``wq``
        # measure to the two-grid basis (the audit-P3-14 half-cell error).
        L = dict(layer_modes(m, Rbig, N, eps_profile, k0, staggered=True))
        W, V = L["W"], L["V"]
        wq_f, wq_n = L["wq_face"], L["wq_node"]
        L["flux"] = np.real(
            np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
            - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
        # div-conforming by construction: no spurious sea to tag.
        L["reldiv"] = np.zeros(W.shape[1])
    elif basis == "nodal":
        L = _flux_normalize(layer_modes(m, Rbig, N, eps_profile, k0, wall=wall))
        L["reldiv"] = np.array([md["reldiv"] for md in
                                radial_coupled_modes(m, Rbig, N, eps_profile,
                                                     k0, wall=wall)])
    else:
        raise ValueError("basis must be 'staggered' or 'nodal' (got %r)"
                         % (basis,))
    L["thickness"] = thickness
    return L


def _physical_propagating(L, k0, reldiv_tol=0.5):
    # Dimensionless q/k0 classifier (audit P2-06): absolute thresholds on q
    # (units 1/length) silently emptied the propagating set for small-k0 unit
    # systems.
    #
    # AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13: the P2-06 constant
    # (0.05) was an ANGULAR cutoff that dropped genuinely propagating
    # near-grazing orders (energy leak 2.28e-2 on the ring-grating
    # reproducer).  The real-axis floor guards ONLY the q ~ 0 degenerate
    # point (1e-6); kept modes sit >= 4 decades above the flux normalizer's
    # field-norm fallback (P/fnrm = qn for the limiting family), so kept
    # implies flux-normalized.  Twins: bor_stack.solve's prop() and
    # _jax_bor._mask -- keep all three in lockstep.
    qn = L["q"] / k0
    return (np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6) & \
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
    inc = np.where(_physical_propagating(layers[0], k0))[0]
    out = np.where(_physical_propagating(layers[-1], k0))[0]
    R = np.array([np.sum([abs(S11[jp, j]) ** 2 for jp in inc]) for j in inc])
    T = np.array([np.sum([abs(S21[jp, j]) ** 2 for jp in out]) for j in inc])
    return dict(S=S, inc=inc, out=out, R=R, T=T, energy=R + T,
                q_inc=layers[0]["q"][inc])
