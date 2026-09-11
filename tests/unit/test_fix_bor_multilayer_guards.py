"""The BOR multilayer guards, 5.45.1 -- one orientation kernel, a spectrum-scaled
classifier band, a passivity refusal on the legacy nodal cascade, and a
manufactured-element contract on the SEM mesh.

WHAT A BOR SOLVE IS.  ``BORStack(Rbig, m, ..., basis='fd'|'sem')`` solves an
axisymmetric (body-of-revolution) stack at one azimuthal order ``m``: layers
stacked in ``z``, each a set of concentric rings in ``r``, closed at
``r = Rbig`` by a PEC wall.  Fields go as ``exp(i m phi + i q z)``; ``q`` is the
axial wavenumber and ``qn = q / k0`` the dimensionless axial index.  Two radial
bases ship -- ``'fd'``, a Yee div-conforming staggered finite-difference basis
on ONE uniform radial grid shared by every layer, and ``'sem'``, per-layer
spectral-element meshes aligned with the ring walls and coupled by cross-tested
Galerkin mortars.  A third, legacy NODAL FD basis remains reachable through
``bor_solve.build_layer(basis='nodal')``.

WHAT THIS FILE PINS, and the scoping report each bar comes from
(``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md``; the build's own
evidence is ``docs/audits/BUILD_BOR_MULTILAYER_GUARDS_2026_09_12.md``):

* **The one orientation kernel** (scoping 2.1, 7.1).  Which of ``+q`` and
  ``-q`` is the FORWARD mode was decided by five independent copies of one
  rule.  There is now exactly one, ``xp=``-parametrized, and a grep over the
  package pins that there is exactly one.
* **The classifier band** (scoping 2.3-2.6).  The copies scaled the band by the
  MODE'S OWN ``|Re q|``, so near a radial cutoff -- where that magnitude has
  collapsed -- the orientation was decided by the eigensolver's backward error,
  and the R/T channel count moved with the BLAS kernel (21 of 24 rungs) and the
  thread count (35 of 39).  The band is now relative to the SPECTRUM's top,
  floored at ``k0``.
* **The nodal passivity refusal** (scoping 3.3, 7.3).  ``bor_solve.solve`` on
  ``basis='nodal'`` returned ``R + T`` up to 966.7 on a provably passive
  lossless stack, unwarned below four vacuum wavelengths.
* **The SEM manufactured-element contract** (scoping 4, 7.2).  Two neighbouring
  layers whose ring walls differ by ``delta`` manufacture an element of width
  ``delta`` in BOTH meshes, which injects spurious axial wavenumbers millions
  of times the physical index ceiling.

Every bar is RE-MEASURED on the running build here rather than pinned from the
report, because a bar that only holds on the machine it was derived on is not a
bar (``docs/TESTING_STANDARDS.md``).
"""
from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

from lumenairy.elements.bor import BORStack
from lumenairy.elements.bor import _orient as _or

_PKG = pathlib.Path(_or.__file__).resolve().parents[3] / "lumenairy"


# =========================================================================== #
#  STEP 1 -- exactly one implementation of each consolidated decision          #
# =========================================================================== #
def _defs_of(name):
    """Every ``def <name>`` in the package, as ``path:line``.

    Grep-based on purpose, in the shape of the round-2 ``_sqrt_decay`` pin: an
    identity check on imported names cannot see a copy that is never imported,
    and ``pmm/twod_staggered.py``'s DEAD copy of ``_sqrt_decay`` -- which still
    carried the pin round 1 had removed -- was exactly that.
    """
    pat = re.compile(r"\s*def\s+%s\b" % (re.escape(name),))
    out = []
    for path in sorted(_PKG.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if pat.match(line):
                out.append("%s:%d" % (path.relative_to(_PKG).as_posix(), i))
    return out


@pytest.mark.parametrize("name,fn", [
    ("forward_orient", _or.forward_orient),
    ("orient_band_scale", _or.orient_band_scale),
    ("flux_is_strong", _or.flux_is_strong),
    ("channel_core", _or.channel_core),
])
def test_exactly_one_definition_of_each_bor_orientation_helper(name, fn):
    """Five bodies of one decision is the shape that bred the six-copy
    factor-i defect (audit S1-8) and the six-copy branch-cut defect (round 2).
    The decision is that there is exactly ONE of each."""
    assert _defs_of(name) == [
        "elements/bor/_orient.py:%d" % (fn.__code__.co_firstlineno,)], (
        "expected exactly one definition of %s, in elements/bor/_orient.py; "
        "found: %s" % (name, ", ".join(_defs_of(name))))


def test_no_bor_module_carries_a_private_copy_of_the_orientation_rule():
    """The rule's fingerprint is the pair ``(|Im q| < band * scale)`` followed
    by a ``where(flip, -q, q)``.  A re-introduced copy would show as a literal
    ``1e-300`` scale floor or a bare ``1e-9 *`` band outside ``_orient.py``:
    both were the shipped copies' spelling, and neither has any other use in
    the BOR package."""
    bad = []
    for path in sorted((_PKG / "elements" / "bor").rglob("*.py")):
        if path.name == "_orient.py":
            continue
        for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace")
                .splitlines(), 1):
            code = line.split("#", 1)[0]
            if "1e-300" in code and "maximum" in code:
                bad.append("%s:%d  %s" % (path.name, i, line.strip()))
            if re.search(r"1e-9\s*\*\s*(np|jnp)\.maximum", code):
                bad.append("%s:%d  %s" % (path.name, i, line.strip()))
    assert not bad, ("a private copy of the forward-orientation classifier "
                     "survives:\n  " + "\n  ".join(bad))


def test_every_former_copy_site_imports_the_shared_kernel():
    """The five sites the 5.45.1 consolidation deleted.  ``zcascade.py`` and
    ``sem_radial.py`` carry the eager NumPy paths; ``_jax_bor.py`` and
    ``_jax_sem.py`` the traced twins, which must pass ``xp=jnp`` EXPLICITLY so
    the traced body is the same object the eager path runs."""
    src = {n: (_PKG / "elements" / "bor" / n).read_text(encoding="utf-8")
           for n in ("zcascade.py", "sem_radial.py", "_jax_bor.py",
                     "_jax_sem.py", "bor_solve.py", "bor_stack.py")}
    for n in ("zcascade.py", "sem_radial.py", "_jax_bor.py", "_jax_sem.py"):
        assert "forward_orient(" in src[n], n
        assert "from ._orient import" in src[n], n
    for n in ("_jax_bor.py", "_jax_sem.py"):
        assert "forward_orient(q, Pz, k0, xp=jnp)" in src[n], (
            "%s must pass xp=jnp explicitly" % (n,))
    # the two companion decisions
    for n in ("zcascade.py", "sem_radial.py", "bor_solve.py", "_jax_bor.py",
              "_jax_sem.py"):
        assert "flux_is_strong(" in src[n], n
    for n in ("bor_solve.py", "bor_stack.py", "_jax_bor.py", "_jax_sem.py"):
        assert "channel_core(" in src[n], n


def test_the_shared_kernel_is_xp_parametrized_and_band_is_live():
    """``band=`` must reach the comparison, not be decorative: a band of zero
    calls NOTHING propagating, a band of infinity calls EVERYTHING
    propagating, and the two must disagree on a spectrum that contains both."""
    q = np.array([1.0 + 0.0j, 0.5 + 0.3j, 2.0 + 1e-18j])
    flux = np.array([-1.0, -1.0, -1.0])
    all_evan = _or.forward_orient(q, flux, 1.0, xp=np, band=0.0)
    all_prop = _or.forward_orient(q, flux, 1.0, xp=np, band=np.inf)
    # band=0: nothing is propagating -> every mode oriented by Im q > 0
    assert np.all(np.imag(all_evan) >= 0.0)
    # band=inf: everything is propagating and every flux is negative -> flipped
    assert np.allclose(all_prop, -q)
    assert not np.allclose(all_evan, all_prop)


def test_the_kernel_accepts_an_explicit_namespace_and_does_not_sniff():
    """``xp=np`` must be honoured even when the arrays would have selected it
    anyway, and the result must not depend on which way it was reached."""
    q = np.array([1.0 + 1e-20j, -0.7 + 0.0j])
    flux = np.array([1.0, 1.0])
    assert np.array_equal(_or.forward_orient(q, flux, 2.0, xp=np),
                          _or.forward_orient(q, flux, 2.0))


# =========================================================================== #
#  STEP 2 -- the classifier band, and the near-cutoff population it fixes      #
# =========================================================================== #
_RBIG = 24.0
_NFD = 120
_NREF = 1.41
_EPS = _NREF ** 2


def _fd_modes(m, k0, eps=_EPS, N=_NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, _RBIG, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def _flux_and_norm(L):
    W, V = L["W"], L["V"]
    wq_f = np.real(np.asarray(L["wq_face"]))
    wq_n = np.real(np.asarray(L["wq_node"]))
    N = len(wq_f)
    flux = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
                   - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
    fnrm = (np.sum(np.abs(W[:N]) ** 2 * wq_f[:, None], axis=0)
            + np.sum(np.abs(W[N:]) ** 2 * wq_n[:, None], axis=0))
    return flux, fnrm


def _sigma(L, k0):
    """The band's discriminating ratio ``|Im q| / max(max|q|, k0)``."""
    q = np.asarray(L["q"])
    scale = max(float(np.max(np.abs(q))) if q.size else 0.0, float(k0))
    return q, np.abs(q.imag) / scale


def _gamma_of(m, idx=2):
    """The cutoff wavenumber of one named radial order.

    The PEC-walled cylindrical spectrum is DISCRETE, so no ordinary ``k0``
    sweep reaches a cutoff: the ladder must solve for ``gamma_j`` first and
    then approach it geometrically with ``k0 = gamma_j / (n sqrt(1 - delta))``,
    which puts the order at ``qn = n sqrt(delta)`` exactly.
    """
    L = _fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * _EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def _cutoff_stack(m, k0):
    """A lossless three-layer stack whose superstrate is COINCIDENT with its
    first layer -- the coincidence that makes the mis-oriented near-cutoff
    order visible in the closure."""
    s = BORStack(_RBIG, m, n_substrate=_NREF, n_superstrate=_NREF, N=_NFD,
                 basis="fd")
    s.add_layer(0.4, eps=_EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=_EPS)
    s.set_source(k0=float(k0))
    return s.solve()


def test_near_cutoff_closure():
    """THE DEFECT IN ONE NUMBER.  At ``qn ~ 2.5e-03`` -- an order a quarter of
    a percent above its own radial cutoff, on a PROVABLY LOSSLESS stack -- the
    shipped per-mode band read ``|R + T - 1| = 1.2167e-04``.  The bar is 1e-08,
    four decades tighter, which is where the spectrum-scaled band puts it
    (measured 1.9655e-07 as the WORST rung of the whole 39-rung ladder; this
    single rung is far better).
    """
    m = 0
    g = _gamma_of(m)
    dl = 3.1622776601683795e-06          # qn = 1.41 * sqrt(delta) ~ 2.5e-03
    k0 = g / (_NREF * np.sqrt(1.0 - dl))
    res = _cutoff_stack(m, k0)
    e = np.asarray(res["energy"])
    assert e.size, "the cutoff rung returned no channels at all"
    closure = float(np.max(np.abs(e - 1.0)))
    assert closure < 1e-8, (
        "lossless closure %.4e at qn ~ 2.5e-03 (bar 1e-8; the shipped "
        "per-mode band read 1.2167e-04 here)" % (closure,))


def test_near_cutoff_channel_count_is_stable_over_the_ladder():
    """The shipped band did not merely degrade the closure -- it changed HOW
    MANY diffraction channels the solve reported, and which number came back
    depended on the BLAS kernel (21 of 24 rungs) and on the thread count (35 of
    39).  A channel count that moves with the arithmetic is a defect, not
    noise.  Over the whole near-cutoff ladder the count must now be ONE
    number."""
    m = 0
    g = _gamma_of(m)
    counts = set()
    worst = 0.0
    for e_ in range(8, 21):
        dl = 10.0 ** (-e_ / 2.0)
        k0 = g / (_NREF * np.sqrt(1.0 - dl))
        res = _cutoff_stack(m, k0)
        counts.add(int(np.size(res["R"])))
        en = np.asarray(res["energy"])
        if en.size:
            worst = max(worst, float(np.max(np.abs(en - 1.0))))
    assert len(counts) == 1, (
        "the R/T channel count moves over the near-cutoff ladder: %s "
        "(the shipped band read 2 on some rungs and 3 on others)"
        % (sorted(counts),))
    assert worst < 1e-6, (
        "worst lossless closure over the ladder %.4e (bar 1e-6; the shipped "
        "band read 1.2167e-04)" % (worst,))


def test_no_forward_mode_of_a_lossless_layer_carries_backward_flux():
    """The orientation contract itself, over ordinary geometry: a mode the
    solver ships FORWARD and calls PROPAGATING must carry power in ``+z``.

    Only modes whose flux is SIGNAL are checked -- a mode whose ``|P|/fnrm``
    is at the normalizer's own fallback floor has no flux to have a sign."""
    bad = []
    for eps in (1.41 ** 2, 2.00 ** 2):
        for m in (0, 1, 2):
            for k0 in (0.8, 2.0, 3.5):
                L = _fd_modes(m, k0, eps=eps)
                q, sig = _sigma(L, k0)
                flux, fnrm = _flux_and_norm(L)
                rel = np.abs(flux) / np.maximum(fnrm, 1e-300)
                prop = sig <= _or._BOR_CUT_BAND_REL
                signal = rel > 1e-10
                for j in np.where(prop & signal & (flux < 0.0))[0]:
                    bad.append("eps=%g m=%d k0=%g mode=%d flux=%.3e"
                               % (eps, m, k0, j, flux[j]))
    assert not bad, ("forward-oriented propagating modes carrying BACKWARD "
                     "z-flux:\n  " + "\n  ".join(bad[:10]))


def test_band_two_sided_population():
    """BOTH sides of the bar, RE-MEASURED on the running build.

    The band must REACH the noise side (or a propagating mode is mistaken for
    an evanescent one and oriented by its own backward error) and must NOT
    reach the signal side (or a genuinely lossy mode is oriented by flux).
    The margins asserted are inside each measured population by a stated
    factor, so the gate fails on a real change rather than on a rounding one.
    """
    band = _or._BOR_CUT_BAND_REL

    # --- NOISE, ordinary geometry.  Measured worst 1.3024e-15 = 6.89 decades
    #     of room; assert 3 decades, which no arithmetic difference can eat.
    worst_ord = 0.0
    n_ord = 0
    for eps in (1.41 ** 2, 1.50 ** 2, 2.00 ** 2):
        for m in (0, 1, 2):
            for k0 in (0.8, 2.0, 3.5):
                L = _fd_modes(m, k0, eps=eps)
                q, sig = _sigma(L, k0)
                phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
                if phys.any():
                    worst_ord = max(worst_ord, float(np.max(sig[phys])))
                    n_ord += 1
    assert n_ord >= 27
    assert worst_ord < band / 1e3, (
        "NOISE side (ordinary): worst sigma %.4e against band %.0e -- less "
        "than 3 decades of room (measured 1.3024e-15, 6.89 decades)"
        % (worst_ord, band))

    # --- NOISE at a deep cutoff: the BINDING side.  Measured worst 8.7301e-10
    #     = 1.06 decades.  Assert the band still covers it, with a 2x margin:
    #     this population IS backward error, so it grows with ||K|| and a much
    #     finer radial grid would eat the decade.
    worst_cut = 0.0
    n_cut = 0
    for m in (0, 1, 2):
        g = _gamma_of(m)
        for e_ in range(4, 27, 4):
            dl = 10.0 ** (-e_)
            k0 = g / (_NREF * np.sqrt(1.0 - dl))
            L = _fd_modes(m, k0)
            q, sig = _sigma(L, k0)
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            if phys.any():
                worst_cut = max(worst_cut, float(np.max(sig[phys])))
                n_cut += 1
    assert n_cut >= 18
    assert worst_cut < band / 2.0, (
        "NOISE side (deep cutoff, the binding one): worst sigma %.4e against "
        "band %.0e (measured 8.7301e-10, 1.06 decades of room)"
        % (worst_cut, band))

    # --- SIGNAL: genuinely lossy media must stay OUT of the band.
    for imn, floor in ((1e-3, 1e2), (1e-6, 2.0)):
        smallest = np.inf
        for m in (0, 1, 2):
            for k0 in (2.0, 3.5):
                L = _fd_modes(m, k0, eps=(_NREF + 1j * imn) ** 2)
                q, sig = _sigma(L, k0)
                phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
                if phys.any():
                    smallest = min(smallest, float(np.min(sig[phys])))
        assert smallest > floor * band, (
            "SIGNAL side at Im(n)=%g: smallest sigma %.4e is only %.3gx the "
            "band %.0e (measured 9.4570e-05 and 9.4570e-08, i.e. 3.98 and "
            "0.98 decades)" % (imn, smallest, smallest / band, band))


def test_widening_the_band_is_harmless_because_the_two_rules_agree_there():
    """The band now calls media with ``Im(n)`` between ~4e-07 and ~1e-09
    "propagating" where the shipped one did not, and orients them by FLUX
    rather than by DECAY.  That is only acceptable if the two verdicts agree
    there -- which is a measurement, not an argument."""
    total = 0
    disagree = 0
    relmin = np.inf
    for imn in (1e-6, 1e-8, 1e-10):
        for m in (0, 1, 2):
            L = _fd_modes(m, 2.0, eps=(_NREF + 1j * imn) ** 2)
            q = np.asarray(L["q"])
            flux, fnrm = _flux_and_norm(L)
            rel = np.abs(flux) / np.maximum(fnrm, 1e-300)
            for j in np.where(np.abs(q.real) > 10.0 * np.abs(q.imag))[0]:
                total += 1
                relmin = min(relmin, float(rel[j]))
                if (flux[j] >= 0.0) != (q[j].imag > 0.0):
                    disagree += 1
    assert total >= 300
    assert disagree == 0, (
        "%d of %d weakly-lossy propagating modes have a FLUX verdict that "
        "disagrees with their DECAY verdict -- the widening is NOT harmless"
        % (disagree, total))
    assert relmin > 1e-3, (
        "the flux that governs these modes is only %.3e of the field norm; "
        "the normalizer's own noise floor is 1e-10, so a flux this small "
        "would be a decision made on noise" % (relmin,))


# =========================================================================== #
#  STEP 3 -- the passivity refusal on the legacy nodal cascade                 #
# =========================================================================== #
def _nuni(v):
    return lambda r: np.full_like(r, v, dtype=complex)


def _nring(period, lo, hi, duty=0.5):
    """The shipped suite's own radial-grating profile
    (``test_bor_solve._ring``), verbatim."""
    def f(r):
        e = np.full_like(r, lo, dtype=complex)
        e[(r % period) < duty * period] = hi
        return e
    return f


def _nodal_five_layer(basis, n_lambda, N=140, m=1, k0=2.0):
    from lumenairy.elements.bor.bor_solve import build_layer
    R = n_lambda * 2.0 * np.pi / k0
    return [build_layer(m, R, N, _nuni(2.0), k0, basis=basis),
            build_layer(m, R, N, _nring(0.8, 2.0, 6.0), k0, thickness=0.5,
                        basis=basis),
            build_layer(m, R, N, _nuni(2.0), k0, thickness=0.3, basis=basis),
            build_layer(m, R, N, _nring(1.2, 6.0, 2.0), k0, thickness=0.4,
                        basis=basis),
            build_layer(m, R, N, _nuni(2.0), k0, basis=basis)]


@pytest.mark.parametrize("n_lambda", [1, 2, 4, 8, 14])
def test_the_legacy_nodal_cascade_is_refused_at_every_cell_radius(n_lambda):
    """THE DEFECT: ``bor_solve.solve`` on ``basis='nodal'`` returned ``R + T``
    from 3.05 to 6899 on a PROVABLY PASSIVE lossless stack -- and below four
    vacuum wavelengths it returned it with NO warning at all, because the only
    guard was a ``Rbig/lambda > 4`` PROXY that misses the small end of its own
    population entirely (measured: 3.05 at 1 wavelength, 114.4 at 2 and 37.9 at
    4, all unwarned).

    The refusal must fire at EVERY radius, including the three the proxy
    missed."""
    import warnings as _w

    from lumenairy.elements.bor.bor_solve import BORNodalPassivityError, solve
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        layers = _nodal_five_layer("nodal", n_lambda)
        with pytest.raises(BORNodalPassivityError) as ei:
            solve(layers, 2.0)
    msg = str(ei.value)
    assert "basis='nodal'" in msg
    assert "staggered" in msg
    assert "BOR_NODAL_PASSIVITY_GUARD" in msg


@pytest.mark.parametrize("n_lambda", [1, 2, 4, 8, 14])
def test_the_staggered_twin_returns_and_closes_energy_at_the_same_radii(
        n_lambda):
    """The screen must be INVISIBLE on the production basis.  Every one of the
    five refused rows has a staggered twin that returns and reads
    ``R + T = 1`` to twelve decimal places -- which is what makes the refusal
    an attribution and not a blanket."""
    from lumenairy.elements.bor.bor_solve import solve
    res = solve(_nodal_five_layer("staggered", n_lambda), 2.0)
    e = np.asarray(res["energy"])
    assert e.size
    assert float(np.max(np.abs(e - 1.0))) < 1e-9, (
        "the STAGGERED twin at %d wavelengths reads max|R+T-1| = %.4e"
        % (n_lambda, float(np.max(np.abs(e - 1.0)))))


def test_the_nodal_passivity_bar_is_two_sided_on_this_build():
    """RE-MEASURED, not pinned.  The refusal bar must sit above every healthy
    row and below every broken one ON THE RUNNING BUILD.

    Healthy here means the nodal basis's one genuinely accurate family --
    UNIFORM layers on a small cell -- plus the staggered twin of everything.
    Measured over 132 solves with the guard disarmed: staggered 3.7406e-12
    (WIN) / 1.9959e-11 (WSL), nodal-uniform-small 4.4336e-09 on BOTH builds,
    and the mildest broken row 2.8819e-02 on both.  A 6.81-decade gap with
    nothing in it, and the bar's two margins (5.35 decades / 1.46 decades)
    come out identical on the two builds.  The assertions below demand two
    decades above the healthy side and one below the broken side, which is
    inside both measured populations.
    """
    import warnings as _w

    from lumenairy.elements.bor import bor_solve as _bs
    from lumenairy.elements.bor.bor_solve import build_layer, solve
    bar = _bs._BOR_NODAL_SUPERUNITY_BAR

    # --- HEALTHY: uniform nodal layers on a small cell ----------------------
    worst_ok = 0.0
    n_ok = 0
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        for m in (0, 1, 2):
            for rl in (0.5, 1.0, 2.0):
                R = rl * 2.0 * np.pi / 2.0
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    layers = [build_layer(m, R, 120, _nuni(2.0), 2.0,
                                          basis="nodal"),
                              build_layer(m, R, 120, _nuni(2.5), 2.0,
                                          thickness=0.5, basis="nodal"),
                              build_layer(m, R, 120, _nuni(2.0), 2.0,
                                          basis="nodal")]
                    e = np.asarray(solve(layers, 2.0)["energy"])
                if e.size:
                    worst_ok = max(worst_ok, float(np.max(e)) - 1.0)
                    n_ok += 1
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev
    assert n_ok >= 9
    assert worst_ok < bar / 1e2, (
        "HEALTHY side: a uniform small-cell nodal stack reads R+T-1 = %.4e "
        "against the %.0e bar -- less than two decades of room (measured "
        "4.4336e-09, 5.35 decades)" % (worst_ok, bar))

    # --- BROKEN: the mildest row the refusal must still catch ---------------
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            layers = [build_layer(1, 4.0, 200, _nuni(2.0), 2.0, basis="nodal"),
                      build_layer(1, 4.0, 200, _nring(0.8, 2.0, 6.0), 2.0,
                                  thickness=0.5, basis="nodal"),
                      build_layer(1, 4.0, 200, _nuni(2.0), 2.0, basis="nodal")]
            mild = float(np.max(np.asarray(solve(layers, 2.0)["energy"]))) - 1.0
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev
    assert mild > 10.0 * bar, (
        "BROKEN side: the mildest broken row reads R+T-1 = %.4e, only %.3gx "
        "the %.0e bar (measured 2.8819e-02, 1.46 decades)"
        % (mild, mild / bar, bar))


def test_a_lossy_nodal_stack_is_never_judged_by_the_passivity_screen():
    """``R + T <= 1`` is a theorem only for a PASSIVE stack, and the screen is
    one-sided for the same reason: an absorbing substrate reads BELOW unity
    legitimately.  A layer with real loss therefore DISARMS the screen
    entirely -- there is no theorem left to violate -- rather than widening
    it."""
    import warnings as _w

    from lumenairy.elements.bor.bor_solve import build_layer, solve
    R = 2.0 * 2.0 * np.pi / 2.0
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        layers = [build_layer(1, R, 140, _nuni(2.0), 2.0, basis="nodal"),
                  build_layer(1, R, 140, _nring(0.8, 2.0, 6.0 + 0.3j), 2.0,
                              thickness=0.5, basis="nodal"),
                  build_layer(1, R, 140, _nuni(2.0), 2.0, thickness=0.3,
                              basis="nodal"),
                  build_layer(1, R, 140, _nring(1.2, 6.0, 2.0), 2.0,
                              thickness=0.4, basis="nodal"),
                  build_layer(1, R, 140, _nuni(2.0), 2.0, basis="nodal")]
        res = solve(layers, 2.0)                  # must NOT raise
    assert np.asarray(res["energy"]).size


def test_the_rbig_over_lambda_proxy_no_longer_decides_anything():
    """The old guard was a PROXY -- a ``UserWarning`` past four vacuum
    wavelengths -- and the measurement shows how badly it misses its own
    population: the same five-layer stack reads 3.05, 114.4 and 37.9 at 1, 2
    and 4 wavelengths with the proxy silent, while a UNIFORM nodal stack at 12
    wavelengths, which the proxy DOES warn about, reads 1.035.

    The proxy's text survives as an early hint; nothing keys on it.  This test
    pins that: a cell BELOW the proxy's threshold is still refused."""
    import warnings as _w

    from lumenairy.elements.bor.bor_solve import BORNodalPassivityError, solve
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        layers = _nodal_five_layer("nodal", 1)            # 1 wavelength
        assert not any("vacuum wavelengths" in str(x.message) for x in w), (
            "the Rbig/lambda proxy fired at ONE wavelength; the fixture is "
            "supposed to be below its threshold")
        with pytest.raises(BORNodalPassivityError):
            solve(layers, 2.0)


# =========================================================================== #
#  STEP 4 -- the SEM manufactured-element contract                             #
# =========================================================================== #
def _sem_stack(Rbig=24.0, m=1, k0=2.0, degree=8, N=160,
               elements_per_segment=1, grade=False, basis="sem"):
    st = BORStack(Rbig, m, basis=basis, degree=degree, N=N,
                  n_superstrate=1.0, n_substrate=1.5,
                  elements_per_segment=elements_per_segment, grade=grade)
    st.set_source(k0=k0)
    return st


def _measure(st):
    """Solve with the contract DISARMED and return its own measurement.

    The census must read the numbers the SOLVER computes, not a parallel
    re-implementation of them -- otherwise it certifies the wrong thing."""
    from lumenairy.elements.bor import _sem_contract as _sc
    prev = _sc.BOR_SEM_MESH_GUARD
    _sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev
    return recs, res


def _verdicts(st):
    from lumenairy.elements.bor import _sem_contract as _sc
    recs, _res = _measure(st)
    return sorted({_sc.verdict(r) for r in recs}), recs


def _taper(n_slices, degree=8, k0=2.0, r_top=8.0, r_bot=2.0, H=1.2):
    st = _sem_stack(degree=degree, k0=k0)
    for i in range(n_slices):
        r = r_top + (r_bot - r_top) * (i + 0.5) / n_slices
        st.add_layer(H / n_slices, segments=[(r, 6.0), (24.0, 2.0)])
    return st


def _ordinary_families(degrees=(6, 8, 12), group="all"):
    """Every BOR SEM geometry family the shipped test suite builds, as
    ``(label, stack)``.  Uniform, uniform-anisotropic, multi-segment,
    coincident walls, ring gratings at three periods and duties, hp-refined and
    graded meshes, a wall-free spacer between two ring layers, m = 0/2/5,
    k0 = 0.8/3.5/8, an nm-unit fixture whose Rbig and wavelength differ from
    the others by six orders of magnitude, and a lossy metal ring."""
    out = []
    base = group in ("all", "base")
    mesh = group in ("all", "mesh")
    sweep = group in ("all", "sweep")
    for deg in degrees:
      if base:
        st = _sem_stack(degree=deg)
        st.add_layer(0.5, eps=2.25)
        out.append(("uniform|d%d" % deg, st))
        st = _sem_stack(degree=deg)
        st.add_layer(0.5, eps_tensor=(2.25, 2.25, 3.24))
        out.append(("uniaxial|d%d" % deg, st))
        st = _sem_stack(degree=deg)
        st.add_layer(0.5, segments=[(6.0, 6.0), (12.0, 2.25), (24.0, 2.0)])
        out.append(("segments3|d%d" % deg, st))
        st = _sem_stack(degree=deg)
        st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
        st.add_layer(0.4, segments=[(6.0, 2.25), (24.0, 4.0)])
        out.append(("coincident_walls|d%d" % deg, st))
      if mesh:
        for period, duty in ((3.0, 0.5), (1.5, 0.3), (0.8, 0.5)):
            st = _sem_stack(degree=deg)
            st.add_layer(0.5, rings=(period, duty, 2.449, 1.414))
            out.append(("grating_p%g_d%g|d%d" % (period, duty, deg), st))
        st = _sem_stack(degree=deg, elements_per_segment=3)
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        out.append(("hp3|d%d" % deg, st))
        st = _sem_stack(degree=deg, elements_per_segment=3, grade=True)
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        out.append(("hp3_graded|d%d" % deg, st))
        st = _sem_stack(degree=deg)
        st.add_layer(0.4, segments=[(6.0, 4.0), (24.0, 2.25)])
        st.add_layer(0.3, eps=2.25)
        st.add_layer(0.4, segments=[(9.0, 2.25), (24.0, 4.0)])
        out.append(("spacer_between_rings|d%d" % deg, st))
      if sweep:
        for mm in (0, 2, 5):
            st = _sem_stack(m=mm, degree=deg)
            st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
            out.append(("ring_m%d|d%d" % (mm, deg), st))
        for kk in (0.8, 3.5, 8.0):
            st = _sem_stack(k0=kk, degree=deg)
            st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
            out.append(("ring_k%g|d%d" % (kk, deg), st))
        sc = 1e-6
        st = _sem_stack(Rbig=20.0 * sc, k0=2.0 * np.pi / (1.55 * sc),
                        degree=deg)
        st.add_layer(0.5 * sc, rings=(3.0 * sc, 0.5, 2.45, 1.41))
        out.append(("nm_units|d%d" % deg, st))
        st = _sem_stack(degree=deg)
        st.add_layer(0.5, segments=[(6.0, -20.0 + 2.0j), (24.0, 2.25)])
        out.append(("lossy_metal_ring|d%d" % deg, st))
    return out


@pytest.mark.parametrize("degree", [6, 8, 12])
@pytest.mark.parametrize("group", ["base", "mesh", "sweep"])
def test_ordinary_geometry_census(group, degree):
    """THE BINDING CONSTRAINT ON THE WARN EDGE, and it is re-measured on the
    running build rather than pinned from a report.

    The 2-D Cartesian peer's round-4 correction was exactly this: a census
    margin measured on four geometry families is a SAMPLE property, not a
    library one ("the census margin is sample-scoped, and it is 1.67x, not
    3.6x").  So every BOR SEM geometry family the shipped suite builds must
    land OUTSIDE the degradation band on whatever build runs this, and with a
    stated margin.
    """
    from lumenairy.elements.bor import _sem_contract as _sc
    bad = []
    narrowest = np.inf
    worst_q = 0.0
    fams = _ordinary_families((degree,), group=group)
    for label, st in fams:
        try:
            v, recs = _verdicts(st)
        except Exception as exc:                       # noqa: BLE001
            bad.append("%s: raised %s" % (label, type(exc).__name__))
            continue
        for r in recs:
            if np.isfinite(r["w_min_union_frac"]):
                narrowest = min(narrowest, r["w_min_union_frac"])
            if np.isfinite(r["q_excess"]):
                worst_q = max(worst_q, r["q_excess"])
        if v != ["ok"]:
            bad.append("%s -> %s (narrowest union cell %.4e of Rbig, "
                       "q_excess %.4g)"
                       % (label, v,
                          min(r["w_min_union_frac"] for r in recs),
                          max(r["q_excess"] for r in recs)))
    assert len(fams) >= 4, "the census shrank to %d families" % (len(fams),)
    assert not bad, ("ORDINARY geometry tripped the manufactured-element "
                     "contract:\n  " + "\n  ".join(bad))
    # ...and with margin, not merely on the right side of the edge.
    assert narrowest > 3.0 * _sc._BOR_SLIVER_BAND_FRAC, (
        "the narrowest ORDINARY manufactured cell is %.4e of Rbig against the "
        "%.0e warn edge -- only %.3gx of margin"
        % (narrowest, _sc._BOR_SLIVER_BAND_FRAC,
           narrowest / _sc._BOR_SLIVER_BAND_FRAC))
    assert worst_q < _sc._BOR_Q_EXCESS / 10.0, (
        "ORDINARY geometry reaches |q|max/(n_max k0) = %.4g against the %.0e "
        "screen -- less than one decade of margin" % (worst_q,
                                                      _sc._BOR_Q_EXCESS))


@pytest.mark.parametrize("n_slices,degree,k0", [
    (8, 8, 2.0), (16, 8, 2.0), (32, 8, 2.0), (64, 8, 2.0),
    # The deep arms are swept at a lower degree and k0 purely for runtime (a
    # 256-slice taper is one SEM modal eigensolve per slice: 170 s at
    # degree 8 / k0 = 2.0, 36 s here).  Neither knob touches the GEOMETRIC
    # conjunct -- w/Rbig is (r_top - r_bot) / (N Rbig) exactly -- and a LOWER
    # k0 RAISES |q|max / (n_max k0), so these arms are the DEMANDING ones for
    # the spectral screen, not the lenient ones.
    (128, 6, 0.8), (256, 6, 0.8),
])
def test_the_taper_staircase_is_ordinary_at_every_slice_count(
        n_slices, degree, k0):
    """THE FAMILY THAT WALKS INTO THE CONTRACT, and the one that BINDS the warn
    edge.  A cone sliced into layers makes ADJACENT slices' walls differ by
    ``(r_top - r_bot) / n_slices``, so the manufactured cell HALVES with every
    doubling of the slice count while ``|q|max / ceiling`` DOUBLES.

    The census runs to 256 slices and not to the scoping's 64 because a margin
    measured at 64 says nothing about 256 -- and it did not: the scoping's
    candidate warn edge of 1e-3 carried 3.9x at 64 slices and is REFUTED at
    256, where the manufactured cell is 9.766e-04, i.e. 0.977x, INSIDE the band
    it would have warned on.  That measurement moved the edge a decade.
    """
    from lumenairy.elements.bor import _sem_contract as _sc
    v, recs = _verdicts(_taper(n_slices, degree=degree, k0=k0))
    w = min(r["w_min_union_frac"] for r in recs)
    q = max(r["q_excess"] for r in recs if np.isfinite(r["q_excess"]))
    assert v == ["ok"], (
        "a %d-slice taper is ORDINARY geometry but the contract says %s "
        "(narrowest manufactured cell %.4e of Rbig, |q|max/ceiling %.4g)"
        % (n_slices, v, w, q))
    assert w > 3.0 * _sc._BOR_SLIVER_BAND_FRAC, (
        "a %d-slice taper leaves only %.3gx of margin above the %.0e warn edge "
        "(measured 9.766e-04, 9.77x, at 256 slices)"
        % (n_slices, w / _sc._BOR_SLIVER_BAND_FRAC, _sc._BOR_SLIVER_BAND_FRAC))
    assert q < _sc._BOR_Q_EXCESS / 3.0, (
        "a %d-slice taper reaches |q|max/(n_max k0) = %.4g against the %.0e "
        "screen -- only %.3gx (the widened census's worst ordinary reading is "
        "1134.2, i.e. 8.82x)"
        % (n_slices, q, _sc._BOR_Q_EXCESS, _sc._BOR_Q_EXCESS / q))


@pytest.mark.parametrize("degree", [6, 8, 12])
def test_the_delta_ladder_is_decided_the_same_way_at_every_degree(degree):
    """THE LADDER, decided.  Two ADJACENT ring layers whose walls differ by
    ``delta``: the ``+-1`` enrichment window unions both wall sets, so a cell
    of exactly ``delta`` appears in BOTH meshes and no single layer asked for
    it.

    The verdict per rung must be the SAME at every degree and -- checked
    separately, under both kernel and thread ladders on both builds -- the same
    on every arm.  That is possible only because both conjuncts are
    kernel-stable: the geometry is kernel-EXACT (spread 1.0000x) and the
    spectral excess kernel-stable (1.1895x), where the energy violation moves
    70.90x with the kernel alone and could not be used.
    """
    from lumenairy.elements.bor._sem_contract import BORSemMeshError
    expect = {1e-1: "ok", 1e-2: "ok", 1e-3: "ok",
              1e-4: "warn", 1e-5: "warn",
              1e-6: "refuse", 1e-7: "refuse"}
    from lumenairy.elements.bor import _sem_contract as _sc
    for dl, want in expect.items():
        d = dl * 24.0
        st = _sem_stack(degree=degree, N=200)
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
        # ONE solve per rung: the measurement is attached BEFORE the contract
        # is enforced, so the ARMED run yields both the verdict and the
        # caller-visible outcome.
        err = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                st.solve()
            except BORSemMeshError as exc:
                err = exc
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
        assert recs, "no mesh report at delta/Rbig = %.0e" % (dl,)
        v = sorted({_sc.verdict(r) for r in recs})
        got = ("refuse" if "refuse" in v
               else "warn" if any(x.startswith("warn") for x in v) else "ok")
        assert got == want, (
            "delta/Rbig = %.0e at degree %d: contract says %r, expected %r "
            "(%s)" % (dl, degree, got, want, v))
        warned = any("MANUFACTURED" in str(x.message) for x in w)
        if want == "refuse":
            assert err is not None, (
                "delta/Rbig = %.0e at degree %d was classified 'refuse' but "
                "the solve RETURNED" % (dl, degree))
            msg = str(err)
            assert "MANUFACTURED" in msg
            assert "BOR_SEM_MESH_GUARD" in msg
            assert "basis='fd'" in msg
        else:
            assert err is None, (
                "delta/Rbig = %.0e at degree %d raised but should not"
                % (dl, degree))
            assert warned == (want == "warn"), (
                "delta/Rbig = %.0e at degree %d: warned=%s, expected %s"
                % (dl, degree, warned, want == "warn"))


@pytest.mark.parametrize("delta_frac", [1e-2, 1e-4, 1e-6, 1e-7])
def test_the_separated_control_is_clean_at_every_rung(delta_frac):
    """THE ATTRIBUTION CONTROL.  The same two walls, the same ``delta``, but
    THREE layers apart with TWO wall-free spacers between them -- so the ``+-1``
    window never spans both wall sets while the geometry is unchanged.

    ONE spacer is not enough and the scoping's first attempt at this control
    was invalid for exactly that reason: because ``win[i] = walls[i-1] |
    walls[i] | walls[i+1]``, a wall-free layer BETWEEN the two ring layers
    inherits both of their walls and carries the sliver itself.
    """
    d = delta_frac * 24.0
    st = _sem_stack(degree=8, N=200)
    st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
    st.add_layer(0.3, eps=2.0)
    st.add_layer(0.3, eps=2.0)
    st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
    v, recs = _verdicts(st)
    assert v == ["ok"], (
        "the SEPARATED control at delta/Rbig = %.0e is not clean: %s"
        % (delta_frac, v))
    assert all(not np.isfinite(r["w_min_union_frac"]) for r in recs), (
        "the separated control manufactured a cross-layer cell after all: %s"
        % ([r["w_min_union_frac"] for r in recs],))


def test_a_wall_free_spacer_inherits_both_neighbours_walls():
    """PINS THE WINDOW'S REACH, so a future change to ``win[i]`` is caught.

    ``win[i] = walls[i-1] | walls[i] | walls[i+1]`` means a layer with NO walls
    of its own, placed between two ring layers, gets BOTH of their wall sets.
    That is why a spacer is not a refuge and why the contract is applied to the
    post-window breakpoint set rather than to the caller's wall list."""
    d = 1e-6 * 24.0
    st = _sem_stack(degree=8, N=200)
    st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
    st.add_layer(0.3, eps=2.0)                       # ONE wall-free spacer
    st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
    _v, recs = _verdicts(st)
    spacer = recs[1]
    assert np.isfinite(spacer["w_min_union_frac"]), (
        "the wall-free spacer shows no manufactured cell -- the +-1 window's "
        "reach has changed and this contract's scope with it")
    assert spacer["w_min_union_frac"] < 1e-5, (
        "the wall-free spacer's narrowest manufactured cell is %.4e of Rbig; "
        "it inherits BOTH neighbours' walls, so it should carry the full "
        "1e-6 sliver" % (spacer["w_min_union_frac"],))


@pytest.mark.parametrize("w_frac", [1e-5, 1e-6, 1e-7])
def test_a_within_layer_liner_is_warned_and_never_refused(w_frac):
    """A narrow element the CALLER asked for -- an annular liner inside ONE
    layer's own segment list -- is not something the library manufactured, so
    it is never REFUSED.  It is warned about once it drives the spectrum past
    the same screen, because it is the same damage: the scoping measured the
    degree ladder INVERTING from ``w/Rbig = 1e-06`` (degree 16 becomes 144x
    WORSE than degree 6) while the energy closure stays at its healthy
    baseline."""
    w = w_frac * 24.0
    st = _sem_stack(degree=8, N=200)
    st.add_layer(0.5, segments=[(6.0, 6.0), (6.0 + w, 2.0), (24.0, 2.0)])
    v, recs = _verdicts(st)
    assert "refuse" not in v, (
        "a liner the CALLER prescribed was REFUSED at w/Rbig = %.0e -- the "
        "contract must only refuse what the enrichment window manufactured"
        % (w_frac,))
    assert all(not np.isfinite(r["w_min_union_frac"]) for r in recs), (
        "the within-layer liner was attributed to the union: %s"
        % ([r["w_min_union_frac"] for r in recs],))
    from lumenairy.elements.bor import _sem_contract as _sc
    q = max(r["q_excess"] for r in recs if np.isfinite(r["q_excess"]))
    narrow = min(r["w_min_frac"] for r in recs)
    # The OWN-geometry warning is the same CONJUNCTION as the refusal, minus
    # the union attribution: the cell must be below the resolvable width AND
    # the spectrum must show it.  At w/Rbig = 1e-5 the spectrum is already hot
    # (q_excess ~ 2.5e4) but the cell is still an order of magnitude above the
    # measured onset, and the measured per-order error there is 1.8e-05 -- so
    # the contract is deliberately silent, and this gate says so rather than
    # demanding a warning the bars do not license.
    if q > _sc._BOR_Q_EXCESS and narrow < _sc._BOR_MIN_ELEM_FRAC:
        assert "warn_own" in v, (
            "w/Rbig = %.0e drives |q|max/(n_max k0) to %.4g with a cell at "
            "%.3e of Rbig -- past BOTH screens -- and said nothing"
            % (w_frac, q, narrow))
    else:
        assert v == ["ok"], (
            "w/Rbig = %.0e (cell %.3e of Rbig, q_excess %.4g) is inside both "
            "bars but the contract said %s" % (w_frac, narrow, q, v))


def test_the_fd_basis_is_structurally_immune_and_is_not_contracted():
    """``basis='fd'`` puts every layer on ONE uniform radial grid that does not
    know where the walls are, so no cell is ever manufactured by a wall
    coincidence.  The scoping measured the FD per-order R at
    ``delta/Rbig <= 1e-3`` to be BIT-IDENTICAL to the ``delta = 0`` answer --
    the uniform grid cannot resolve the shift at all.

    No contract is applied there, and none should be."""
    from lumenairy.elements.bor import _sem_contract as _sc

    def fd(delta):
        st = _sem_stack(degree=8, N=200, basis="fd")
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        st.add_layer(0.5, segments=[(6.0 + delta, 2.0), (24.0, 6.0)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = st.solve()
        return (np.asarray(res["R"]),
                [str(x.message) for x in w],
                getattr(st, "_sem_mesh_report", None))

    base, wb, rep_b = fd(0.0)
    assert rep_b is None, "the FD path ran the SEM mesh contract"
    assert not any("MANUFACTURED" in m for m in wb)
    for dl in (1e-4, 1e-6, 1e-7):
        r, w, rep = fd(dl * 24.0)
        assert rep is None
        assert not any("MANUFACTURED" in m for m in w), (
            "the FD basis warned at delta/Rbig = %.0e" % (dl,))
        assert r.shape == base.shape
        assert np.array_equal(r, base), (
            "the FD answer MOVED at delta/Rbig = %.0e (worst |dR| = %.4e); it "
            "is supposed to be structurally blind to the wall shift"
            % (dl, float(np.max(np.abs(r - base)))))
    # the guard's own constants must be unreachable from the FD path
    assert _sc.BOR_SEM_MESH_GUARD is True          # armed, and still immune


def test_no_class_c_decision_is_keyed_on_the_energy_violation():
    """THE RULE THIS CONTRACT IS BUILT AROUND.  On this ladder the closure's
    spread across three OpenBLAS kernels is 70.90x, straddling the 1-D
    ``_SLIVER_TRIGGER_BAR`` of 1e-3 -- the same solve would be arbitrated on
    one kernel and pass silently on another.  Independently the damage is
    ENERGY-INVISIBLE over four decades of the ladder.

    So neither conjunct may read ``R``, ``T`` or the closure.  Checked by
    inspecting what the contract module can even see: its measurement takes a
    breakpoint array, a wall list, a spectrum and an index, and nothing else.
    """
    import inspect

    from lumenairy.elements.bor import _sem_contract as _sc
    params = set(inspect.signature(_sc.measure_layer).parameters)
    assert params == {"bnd", "layer_index", "walls", "Rbig", "q", "n_max",
                      "k0"}, params
    src = inspect.getsource(_sc.measure_layer) + inspect.getsource(_sc.verdict)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    body = code.split('"""')
    body = "".join(body[::2])          # drop docstrings
    for token in ("energy", "closure", "R + T", "sum(R)", "superunity",
                  "super_unity"):
        assert token not in body, (
            "the Class-C decision reads %r -- the energy violation moves "
            "70.90x with the BLAS kernel alone and must not decide anything"
            % (token,))


def test_the_contract_switch_restores_the_previous_behaviour():
    """``BOR_SEM_MESH_GUARD = False`` is a FAIL-BEFORE SWITCH: the refused
    solve returns its pre-5.45.1 number, and no warning is emitted anywhere."""
    from lumenairy.elements.bor import _sem_contract as _sc
    d = 1e-7 * 24.0

    def build():
        st = _sem_stack(degree=8, N=200)
        st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
        st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
        return st

    with pytest.raises(_sc.BORSemMeshError):
        build().solve()
    prev = _sc.BOR_SEM_MESH_GUARD
    _sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = build().solve()
        assert np.asarray(res["R"]).size
        assert not any("MANUFACTURED" in str(x.message) for x in w)
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev


def test_the_contract_is_bit_identical_where_it_does_not_fire():
    """Everywhere the contract is silent the answer must be EXACTLY what it was
    -- the contract computes a measurement and either raises, warns or returns;
    it never touches the mesh, the modes or the cascade."""
    from lumenairy.elements.bor import _sem_contract as _sc

    def solve_R(armed):
        prev = _sc.BOR_SEM_MESH_GUARD
        _sc.BOR_SEM_MESH_GUARD = armed
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = _sem_stack(degree=8)
                st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
                st.add_layer(0.4, segments=[(9.0, 2.25), (24.0, 4.0)])
                return np.asarray(st.solve()["R"])
        finally:
            _sc.BOR_SEM_MESH_GUARD = prev

    a, b = solve_R(True), solve_R(False)
    assert a.shape == b.shape
    assert np.array_equal(a, b), (
        "arming the contract moved an ordinary answer by %.4e"
        % (float(np.max(np.abs(a - b))),))
