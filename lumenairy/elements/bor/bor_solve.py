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

from ._orient import channel_core, flux_is_strong
from .zcascade import interface_smatrix, layer_modes, propagation_smatrix, redheffer_star


#: FAIL-BEFORE SWITCH for the nodal passivity refusal (5.45.1).  ``False``
#: restores the pre-fix behaviour bit for bit: the ``Rbig/lambda`` warning
#: below, and the wrong number returned.  A switch, not a policy -- it changes
#: nothing on any solve that does not trip the screen.
BOR_NODAL_PASSIVITY_GUARD = True

#: THE REFUSAL BAR on ``max(R + T) - 1``, for a stack the solver can PROVE
#: passive (every layer lossless, the incidence medium propagating).  On such a
#: stack ``R + T <= 1`` is a theorem, so any super-unity is numerical damage
#: and its SIZE is the detector.
#:
#: NOT a conditioning conjunction.  The scoping measured the nodal
#: ``inv(a + b)``'s equilibrated residual at 5.998e-13 against the Cartesian
#: guard's ``_INV_RESID_REFUSE = 1e-8`` -- four to five decades BELOW the bar --
#: so ``rcwa/_core._guarded_inverse`` as written refuses 0 of 6 broken rows on
#: every kernel.  The nodal ``a + b`` is not singular; it AMPLIFIES.  What
#: separates the two populations is passivity
#: (``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`` section 3.3).
#:
#: MEASURED POPULATIONS, 132 solves over four families -- the scoping's
#: five-layer ring stack at Rbig = 1 .. 16 vacuum wavelengths, an N-refinement
#: sweep, every nodal fixture the shipped suite builds, and a 48-row small-cell
#: census over m = 0,1,2 x N = 120/200 x Rbig/lambda = 0.5 .. 2.0
#: (``validation/probe_fix_bor_guards/s3_nodal_passivity.py``, with the guard
#: DISARMED so the population is the numbers the solver RETURNED):
#:
#:   STAGGERED, every row, every family     worst ``R+T-1`` = 3.7406e-12 (WIN)
#:                                                            1.9959e-11 (WSL)
#:   NODAL, the family that is accurate     worst ``R+T-1`` = 4.4336e-09
#:     (UNIFORM layers on a small cell)                       (BOTH builds)
#:   -------------------------------------  -------------------------------
#:   NODAL, everything else                 ``R+T-1`` = 2.8819e-02 .. 6899.3
#:
#: **A 6.81-decade gap with NOTHING in it, identical on both builds.**  1e-3
#: sits 5.35 decades above the healthy ceiling and 1.46 decades below the
#: mildest broken row -- two-sided, with well over a decade on the tight side.
#:
#: The bar is deliberately NOT the 1-D peer's ``_STACK_SUPERUNITY_BAR = 1e-2``:
#: that value carries only 0.46 decades here, because the mildest broken row
#: measured is the shipped ``test_structured_stack_energy_floor_nodal`` fixture
#: at 2.8819e-02 -- only 2.9x it.  That fixture is the reason the census had to
#: be measured BEFORE the bar was chosen: it was shipped as a demonstration of
#: the nodal basis's "~1-4% floor", and a 2.9% energy violation on a stack of
#: LOSSLESS media, where ``R + T <= 1`` is a theorem, is not a floor.  It is
#: now refused, and the switch below restores it.
#:
#: KERNEL-STABLE BY CONSTRUCTION.  Every nodal row reads the same
#: ``max(R + T)`` to four significant figures on all four thread counts per
#: build and on both builds: the nodal blow-up is a DETERMINISTIC
#: discretisation defect, not an arithmetic one, so a passivity screen against
#: it does not move with the kernel the way a Class-C energy screen does.
_BOR_NODAL_SUPERUNITY_BAR = 1.0e-3

#: The WARNING edge.  Between this and the refusal bar the answer is returned
#: with a ``UserWarning`` quoting the measured violation -- so a mildly damaged
#: nodal solve is never SILENT, which is what the retired ``Rbig/lambda > 4``
#: proxy allowed.  3.71 decades above the healthy ceiling measured above.
_BOR_NODAL_SUPERUNITY_WARN = 1.0e-6

#: How close to real ``eps`` a layer must be for the stack to count as PROVABLY
#: passive.  Relative to the profile's own ``max|Re eps|`` so it is
#: unit-invariant; 1e-12 admits only round-off, not a deliberately lossy layer
#: (whose ``R + T < 1`` is legitimate and which the screen must never judge).
_BOR_LOSSLESS_REL_IM = 1.0e-12


class BORNodalPassivityError(ValueError):
    """Raised by :func:`solve` when the LEGACY NODAL cascade returns
    non-physical ``R + T`` on a stack that is provably passive.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers are
    unaffected.  The message names the basis, quotes the measured violation,
    and names the remedy -- ``basis='staggered'``, which is
    :func:`build_layer`'s default and conserves energy to machine precision at
    any cell size.
    """


def _stack_is_provably_passive(layers):
    """``True`` when every layer's permittivity profile is lossless to
    round-off.

    The screen is ONE-SIDED (super-unity only) because ``R + T <= 1`` is what
    passivity gives: a stack with an absorbing substrate reads BELOW unity
    legitimately.  That is why a lossy layer disarms the screen entirely rather
    than widening it -- on a lossy stack there is no theorem to violate.
    """
    for L in layers:
        rel = L.get("max_rel_im_eps")
        if rel is None or not (float(rel) <= _BOR_LOSSLESS_REL_IM):
            return False
    return True


def _check_nodal_passivity(layers, energy):
    """Refuse (or warn about) a non-physical ``R + T`` from the legacy nodal
    cascade.  No-op on the staggered basis, on a lossy stack, and when
    :data:`BOR_NODAL_PASSIVITY_GUARD` is ``False``."""
    if not BOR_NODAL_PASSIVITY_GUARD:
        return
    if not any(L.get("basis") == "nodal" for L in layers):
        return
    e = np.asarray(energy, dtype=float)
    if e.size == 0 or not _stack_is_provably_passive(layers):
        return
    worst = float(np.max(e)) - 1.0
    if worst <= _BOR_NODAL_SUPERUNITY_WARN:
        return
    nlam = [float(np.real(L.get("Rbig_over_lambda", float("nan"))))
            for L in layers if L.get("Rbig_over_lambda") is not None]
    where = (" (cell radius %.2f vacuum wavelengths)" % (nlam[0],)
             if nlam else "")
    if worst > _BOR_NODAL_SUPERUNITY_BAR:
        raise BORNodalPassivityError(
            "bor_solve.solve(basis='nodal'): the cascade returned "
            "max(R + T) = %.6g on a PROVABLY PASSIVE lossless stack%s, where "
            "R + T <= 1 is a theorem -- a violation of %.4g against the bar "
            "%.0e.  This is the legacy NODAL FD basis's spurious "
            "divergence-violating mode sea: those modes carry zero z-flux, so "
            "their forward/backward orientation is decided by the sign of "
            "noise, adjacent layers orient near-identical spurious modes "
            "OPPOSITELY, and the interface transmission block acquires a null "
            "vector.  Measured over 93 solves, the nodal basis reads 2.88e-02 "
            "to 6899 here while its STAGGERED twin reads 1 + 3.7e-12 on every "
            "one of the same rows.  REMEDY: basis='staggered' (the "
            "build_layer default, div-conforming Yee), which conserves energy "
            "to machine precision at any cell size.  To restore the previous "
            "behaviour and receive this number instead of this error, set "
            "lumenairy.elements.bor.bor_solve.BOR_NODAL_PASSIVITY_GUARD = "
            "False."
            % (worst + 1.0, where, worst, _BOR_NODAL_SUPERUNITY_BAR))
    warnings.warn(
        "bor_solve.solve(basis='nodal'): max(R + T) = %.9g on a provably "
        "passive lossless stack%s -- a violation of %.3g, above the %.0e "
        "warning edge but below the %.0e refusal bar.  The legacy nodal FD "
        "basis leaks energy through its divergence-violating spurious mode "
        "sea; basis='staggered' (the default) reads 1 + 3.7e-12 on the same "
        "geometry." % (worst + 1.0, where, worst,
                       _BOR_NODAL_SUPERUNITY_WARN,
                       _BOR_NODAL_SUPERUNITY_BAR),
        stacklevel=3)


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
        s = (1.0 / np.sqrt(abs(P)) if flux_is_strong(P, fnrm, xp=np)
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

    ``thickness`` is validated (audit W6-B11): the ``BORStack.add_layer`` sibling
    has guarded it since P3-10 ("a NEGATIVE thickness flips the propagation
    exponent exp(iqL) so forward-oriented evanescent modes GROW, silently
    destabilizing the Redheffer cascade instead of raising"), but this
    lower-level twin accepted ``thickness=-0.5`` and cascaded it silently.
    """
    if thickness is not None:
        thickness = float(thickness)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "build_layer: thickness must be > 0 and finite (or None for a "
                "semi-infinite half-space), got %r" % (thickness,))
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
        # 5.45.1: this ``Rbig/lambda > 4`` test is RETIRED AS A DECISION and
        # kept only as an early, cheap hint.  It is a PROXY, and the scoping
        # measured how badly it misses its own population: at 1, 2 and 4
        # vacuum wavelengths the same five-layer stack reads max(R + T) =
        # 3.05, 114.4 and 37.91 and this warning does not fire, while a
        # UNIFORM nodal stack at 12 wavelengths -- which it does warn about --
        # reads 1.035.  The DECISION now lives in solve(), on the measured
        # passivity violation itself (_check_nodal_passivity).  Nothing keys
        # on the number below.
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
    # The two facts the 5.45.1 passivity screen needs and could not otherwise
    # recover from the returned dict: WHICH basis built this layer, and whether
    # its permittivity is lossless (so ``R + T <= 1`` is a theorem rather than
    # merely a hope).  Both are recorded here because this is where the
    # profile is in scope; neither is read by anything else.
    L["basis"] = basis
    _den = float(np.max(np.abs(_eps_arr.real))) if _eps_arr.size else 0.0
    L["max_rel_im_eps"] = (float(np.max(np.abs(_eps_arr.imag))) / _den
                           if _den > 0.0 else float("inf"))
    L["Rbig_over_lambda"] = (float(np.real(Rbig)) * float(np.real(k0))
                             / (2.0 * np.pi))
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
    keep = channel_core(qn, xp=np) & (L["reldiv"] < reldiv_tol)
    return keep


def solve(layers, k0):
    """Cascade a list of ``build_layer`` layers (first/last = semi-infinite
    super/substrate) and return per-incident-mode R/T efficiencies over the
    physical propagating channels, plus the global S-matrix.

    Returns a dict: ``S`` (S-matrix), ``inc`` (superstrate physical-prop indices),
    ``out`` (substrate ...), ``R``/``T`` (arrays over ``inc``: total reflected /
    transmitted power fraction), ``energy`` (R+T per incident mode).

    Audit W6-B11: a MIDDLE layer left at the ``build_layer`` default
    ``thickness=None`` used to die inside ``propagation_smatrix`` with a bare
    ``TypeError: unsupported operand type(s) for *: 'complex' and 'NoneType'``.
    """
    if len(layers) < 2:
        raise ValueError(
            "bor_solve.solve needs at least the two semi-infinite half-spaces "
            "(got %d layer(s))" % (len(layers),))
    for i in range(1, len(layers) - 1):
        if layers[i].get("thickness") is None:
            raise ValueError(
                "bor_solve.solve: middle layer %d has no thickness -- pass "
                "thickness=... to build_layer for every layer between the "
                "half-spaces." % (i,))
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
    # PASSIVITY SCREEN (5.45.1).  On the legacy nodal basis this cascade
    # returned R + T from 2.9e-02 to 6899 on provably passive lossless stacks,
    # and below four vacuum wavelengths it returned it with NO warning at all.
    # See _BOR_NODAL_SUPERUNITY_BAR for the two populations and the 8.17-decade
    # gap between them.
    _check_nodal_passivity(layers, R + T)
    return dict(S=S, inc=inc, out=out, R=R, T=T, energy=R + T,
                q_inc=layers[0]["q"][inc])
