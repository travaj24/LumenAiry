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

