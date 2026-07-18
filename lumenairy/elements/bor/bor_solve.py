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

import warnings

import numpy as np

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
    ``_physical_propagating`` filter.  It now WARNS (audit S1-15) when the cell
    radius exceeds a few vacuum wavelengths -- the regime where the spurious-mode
    sea silently drives the cascade energy up to ~1e29.  The staggered wall is
    the closed Dirichlet wall, so ``wall`` must stay ``'pec'`` there.
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
        # Large-cell blow-up guard (audit S1-15): the nodal FD basis grows a
        # divergence-violating spurious-mode sea (~40-50% of the basis at
        # Rbig ~ 12 vacuum wavelengths) whose zero-z-flux modes are oriented by
        # the sign of noise, driving the interface transmission block singular
        # (cond ~ 2.6e15) and blowing the cascade energy up to ~1e29 -- silently.
        # Warn past a few vacuum wavelengths (small cells only leak the
        # documented ~1-4% floor); the staggered default has no such sea.
        rbig_lambda = float(np.real(Rbig)) * float(np.real(k0)) / (2.0 * np.pi)
        if rbig_lambda > 4.0:
            warnings.warn(
                "build_layer(basis='nodal'): the cell radius Rbig is "
                f"{rbig_lambda:.1f} vacuum wavelengths; the nodal FD basis "
                "develops a spurious divergence-violating mode sea on large "
                "cells that can render the interface transmission block singular "
                "and blow the cascade energy up to ~1e29.  Use the default "
                "basis='staggered' (div-conforming Yee), which conserves energy "
                "to machine precision at any cell size.",
                stacklevel=2)
        # S1-18: harvest the divergence tag from the SAME dense eig
        # ``layer_modes`` already runs (``with_reldiv=True``) instead of a
        # second byte-identical ``radial_coupled_modes`` eigensolve.  The two
        # nodal paths assemble byte-identical K/B, so ``reldiv`` is unchanged.
        _Lm = layer_modes(m, Rbig, N, eps_profile, k0, wall=wall,
                          with_reldiv=True)
        _reldiv = _Lm["reldiv"]
        L = _flux_normalize(_Lm)
        L["reldiv"] = _reldiv
    else:
        raise ValueError("basis must be 'staggered' or 'nodal' (got %r)"
                         % (basis,))
    L["thickness"] = thickness
    # S1-16: store the layer's index ceiling (the eps of maximum real part over
    # the radial profile) as a per-layer REFERENCE -- it is the axial-index
    # bound q/k0 <= sqrt(eps) that the staggered twins (bor_stack.solve's prop()
    # and _jax_bor._mask) enforce.  It is recorded here for cross-checks but is
    # DELIBERATELY NOT applied by the nodal ``_physical_propagating`` (whose
    # unique leg is reldiv): forcing the ceiling onto the nodal FD basis
    # over-filters and degrades its ~4% energy floor.  For the homogeneous
    # super/substrate (the only layers _physical_propagating ever classifies)
    # this is exactly that medium's eps.
    _eps_arr = np.asarray(eps_profile(L["r"]) if callable(eps_profile)
                          else eps_profile, dtype=complex).ravel()
    L["eps_ceiling"] = complex(_eps_arr[int(np.argmax(_eps_arr.real))])
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
    # implies flux-normalized.
    #
    # S1-16 (audit AUDIT_V5_24_2): the three BOR mode classifiers share a
    # {imag, real-floor, index-ceiling} CORE.  This one previously carried
    # the reldiv leg but NOT the index ceiling, while the staggered twins
    # (bor_stack.solve's prop() and _jax_bor._mask) carried the ceiling but
    # NOT reldiv -- so the "keep all three in lockstep" comment was false.
    # The reldiv leg is UNIQUE to this classifier on purpose: it filters the
    # divergence-violating spurious sea of the optional NODAL basis (staggered
    # sets reldiv == 0, so the leg is a no-op there); the twins are
    # staggered-only (div-conforming, spurious-free) and deliberately skip
    # the reldiv eigensolve.  The index ceiling (q/k0 <= sqrt(eps)) that the
    # staggered twins carry is DELIBERATELY NOT replicated on the nodal basis:
    # applying it here over-filters the reldiv-screened FD mode set and
    # degrades the documented ~4% nodal energy floor (measured 4% -> 10.7% on
    # test_structured_stack_energy_floor_nodal).  So S1-16 is resolved by making
    # this comment TRUE -- the three classifiers share the {imag, real-floor}
    # core and each carries ONE basis-specific leg (nodal: reldiv; staggered
    # twins: index-ceiling) -- rather than forcing a numeric lockstep the bases
    # do not physically share.
    qn = L["q"] / k0
    keep = ((np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6)
            & (L["reldiv"] < reldiv_tol))
    return keep


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
