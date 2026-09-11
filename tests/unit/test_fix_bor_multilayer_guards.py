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
