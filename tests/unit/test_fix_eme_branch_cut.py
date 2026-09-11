"""The EME branch cut, 5.45.1 -- the exact-zero pin that made a strip mode's
DIRECTION a function of the build.

WHAT AN EME LAYER SOLVE IS.  ``lumenairy.elements.eme.eme_2d`` expands a 2-D
layer in the eigenmodes of its x-cross-section: each "strip" (a y-slice of
constant ``eps(x)``) contributes a set of eigenvalues ``lam``, from which the
LATERAL wavenumber follows as ``ky = sqrt(lam - qz2)``.  ``layer_modes`` then
scans a window of ``qz2`` for the values at which the cascaded strip S-matrix
has a lateral resonance, and the roots of that scan ARE the layer's modes.

WHAT WAS WRONG.  Of the two square roots, the FORWARD one must be the DECAYING
one (``Im(ky) >= 0``) so the propagator ``exp(i ky h)`` never grows.  Both
scalar sites spelled that as an EXACT-ZERO pin::

    ky = np.where(ky.imag < 0.0, -ky, ky)

For a PROPAGATING strip mode of a lossless layer ``ky^2`` is exactly real
positive, so ``Im(ky)`` is not physics -- it is the eigensolver's backward
error, which grows with ``||A|| ~ 4 Nx^2 / Lx^2``.  Negating on the sign of
that error does not perturb the mode: ``-ky`` is the mode's own BACKWARD
partner, so the forward set silently acquires a backward member.

MEASURED (``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`` section 6):
the same layer geometry returns **62** modes when ``eps`` is real (``eigh``
gives an exactly-zero imaginary part and the pin cannot fire) against **69 on
Windows and 71 on WSL** when an INFINITESIMAL ``i 1e-30`` is added to one
region -- a 2.55 % / 3.64 % gap, with the mode lists themselves differing
between the two builds.  A physical perturbation of 1e-30 cannot move a mode
count; a backward error of 1e-16 deciding a direction can.

THE FIX.  ``lumenairy/elements/eme/_branch.py`` is the one definition:
off the cut the decaying root as before, ON the cut the CONJUGATE (which keeps
``Re >= 0`` -- the module's own vector sibling's tie-break -- and puts
``Im >= 0``, so the ``|exp(i ky h)| <= 1`` guarantee survives).  The band is
the one ``eme_2d_vector._strip_split_forward`` has always carried.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from lumenairy.elements.eme import _branch as _eb
from lumenairy.elements.eme import eme_2d

_PKG = pathlib.Path(_eb.__file__).resolve().parents[3] / "lumenairy"

# The scoping fixture, verbatim: two strips, a high/low split in x, a lateral
# Bloch phase, and a qz2 window wide enough to hold tens of modes.
_NX = 96
_LX = _LY = 1.0
_K0 = 20.0 * np.pi
_KY0 = 0.37
# The window and scan are NARROWED from the scoping's (26055.8, 35530.6) /
# n_scan = 300, purely for runtime -- that pair costs 167 s.  The narrowed pair
# is still DECISIVE, which was checked by RESTORING the pre-5.45.1 exact-zero
# pin and re-running: it returns 16 and 18 modes for the two arms, against 16
# and 16 with the band.  Measured 25 s.
_WINDOW = (26055.8, 28500.0)
_NSCAN = 60


def _strips(imag):
    """The two strips, with ``imag`` added to the HIGH region only.

    ``imag = 0`` takes ``strip_x_modes`` down its ``eigh`` path, where
    ``Im(lam)`` is exactly zero and the pin could not fire; any non-zero
    ``imag`` routes it to ``scipy.linalg.eig``, whose ``Im(lam)`` is the
    backward error.  The two must return the SAME modes: 1e-30 is thirteen
    decades below anything physical.
    """
    hi = 12.0 + 1j * imag
    e1 = np.full(_NX, 2.25 + 0j)
    e1[_NX // 2:] = hi
    e2 = np.full(_NX, 2.25 + 0j)
    e2[_NX // 4:3 * _NX // 4] = hi
    return [(e1, 0.5 * _LY), (e2, 0.5 * _LY)]


def _modes(imag):
    return np.asarray(eme_2d.layer_modes(
        _strips(imag), _LX, _NX, _LY, _K0, _WINDOW,
        kx0=0.0, ky0=_KY0, n_scan=_NSCAN))


def test_no_strip_mode_flips_under_an_infinitesimal_loss():
    """THE MECHANISM, measured directly and cheaply -- one strip eigensolve per
    arm rather than a 80-point dispersion scan.

    ``strip_x_modes`` takes ``eigh`` for a real ``eps`` (``Im(lam)`` exactly
    zero, so the pin could not fire) and ``scipy.linalg.eig`` for any complex
    one (``Im(lam)`` = the backward error).  Adding ``i 1e-30`` is a physical
    no-op, so the forward ``ky`` SET must be the same either way.  With the
    exact-zero pin it was not: modes came back negated, which is to say
    replaced by their own BACKWARD partners."""
    def fwd(imag):
        # The two eigensolve branches return lam in DIFFERENT ORDERS -- eigh
        # ascending, eig unsorted, which the module's own comment records --
        # so the comparison is of the SET: sort lam first, then apply the rule
        # elementwise to the sorted spectrum.
        lam = np.asarray(eme_2d.strip_x_modes(
            _strips(imag)[0][0], _LX, _NX, _K0)[0], dtype=complex)
        lam = lam[np.lexsort((lam.imag, lam.real))]
        return eme_2d._ky_forward(lam, 0.0)

    real, tiny = fwd(0.0), fwd(1e-30)
    assert real.shape == tiny.shape
    flipped = int(np.sum(np.abs(real - tiny) > 1e-6 * np.maximum(
        np.abs(real), 1.0)))
    assert flipped == 0, (
        "%d of %d strip modes changed root under Im(eps) = 1e-30 -- the "
        "forward set is being chosen by the eigensolver's backward error"
        % (flipped, real.size))
    # and the rule's own contract on that set
    assert np.all(tiny.imag >= 0.0), "a forward root must decay"


def test_mode_count_is_build_independent_under_an_infinitesimal_loss():
    """THE DEFECT IN ONE NUMBER.  Adding ``i 1e-30`` to one region's
    permittivity is a physical no-op -- it is thirteen decades below the
    smallest loss any material has -- but it switches the strip eigensolve from
    ``eigh`` to ``eig``, and with the exact-zero pin in place that changed the
    layer's mode COUNT from 62 to 69 (Windows) or 71 (WSL) on the scoping's
    full window, and from 16 to 18 on the narrowed one this gate runs.

    The mode count and the mode positions must now be the same either way."""
    real = _modes(0.0)
    tiny = _modes(1e-30)
    assert real.size == tiny.size, (
        "an infinitesimal Im(eps) = 1e-30 moved the layer mode count from %d "
        "to %d -- the count is being decided by the eigensolver's backward "
        "error, not by the physics" % (real.size, tiny.size))
    assert real.size >= 10, ("the fixture found only %d modes; the window was "
                             "chosen to hold enough that a single flipped "
                             "strip mode changes the count" % (real.size,))
    # The mode POSITIONS must also agree.  qz2 here is of order 3e4, and the
    # scan's own resolution is (hi - lo) / n_scan ~ 31, so a shared root is
    # the same root to far better than that.
    assert np.allclose(np.sort(real), np.sort(tiny), rtol=0, atol=1e-6), (
        "the mode POSITIONS moved under Im(eps) = 1e-30: worst |d qz2| = %.4e"
        % (float(np.max(np.abs(np.sort(real) - np.sort(tiny))))
           if real.size == tiny.size else float("nan"),))


def test_no_forward_root_of_a_propagating_strip_mode_is_negated():
    """The mechanism, one level down.  A strip mode whose ``ky^2`` is real
    positive up to backward error must come back with ``Re(ky) > 0``: the
    NEGATIVE-real-part root is that mode's BACKWARD partner, and handing it to
    the forward set is what conditions the mode match."""
    lam = np.array([100.0 - 1e-13j, 400.0 + 3e-14j, 2500.0 - 5e-12j])
    ky = eme_2d._ky_forward(lam, 0.0)
    assert np.all(ky.real > 0.0), (
        "a propagating strip mode came back on the Re < 0 root: %s" % (ky,))
    assert np.all(ky.imag >= 0.0), (
        "a forward root must decay (Im >= 0) so exp(i ky h) cannot grow: %s"
        % (ky,))


def test_a_genuinely_lossy_root_is_still_put_on_the_decaying_branch():
    """The band is a WIDENING of the on-cut set, not a change of the rule: a
    mode with real loss -- an imaginary part far above the band -- must still
    be flipped to the decaying branch, exactly as before."""
    z = np.array([2.0 - 0.5j, 3.0 + 0.5j])
    out = _eb.forward_decaying_root(z, xp=np)
    assert np.allclose(out, np.array([-2.0 + 0.5j, 3.0 + 0.5j]))


def test_the_band_is_relative_to_the_spectrum_and_floored_at_k0():
    """ROUND 2 (D13).  ``ky`` in the EME modules is NOT dimensionless -- it
    carries units of inverse length, exactly like the BOR peer's ``q`` -- so
    the band's floor must carry them too.  It is ``|k0|``, which is what
    ``elements/bor/_orient.orient_band_scale`` floors at and for the same
    reason (audit P2-06).

    As first built this gate asserted the floor was a literal 1.0, which made
    the band -- and therefore the BRANCH DECISION -- depend on whether the
    caller wrote a cell in micrometres or nanometres.

    THE PROPERTY, STATED AS A DECISION AND NOT AS A READING: rescaling the
    whole problem (lengths x s, wavenumbers / s) must scale the band by exactly
    1/s.  A dimensioned literal anywhere in the expression breaks that, and
    nothing else does.
    """
    # the floor is k0 when the spectrum is below it, the spectrum when above
    assert _eb.cut_band(np.array([1e-6 + 0j]), k0=2.0,
                        xp=np) == pytest.approx(2e-9)
    assert _eb.cut_band(np.array([1e4 + 0j]), k0=2.0,
                        xp=np) == pytest.approx(1e-5)
    # with no k0 in scope the scale is the spectrum alone -- also unit-free,
    # and in particular NOT a literal 1.0
    assert _eb.cut_band(np.array([1e-6 + 0j]),
                        xp=np) == pytest.approx(1e-15)

    # THE INVARIANCE, over twelve decades of unit scaling in both directions
    z = np.array([3.0 - 1e-4j, 1e-3 + 0j, -7.5 + 2e-2j])
    k0 = 4.05
    base = float(_eb.cut_band(z, k0=k0, xp=np))
    for s in (1e-6, 1e-3, 1e-1, 1e1, 1e3, 1e6):
        got = float(_eb.cut_band(z / s, k0=k0 / s, xp=np))
        assert got == pytest.approx(base / s, rel=1e-12), (
            "the band is not unit-invariant: scaling every wavenumber by "
            "1/%g changed it by %.6gx instead of %gx"
            % (s, got / base, 1.0 / s))
        got = float(_eb.cut_band(z / s, xp=np))
        assert got == pytest.approx(float(_eb.cut_band(z, xp=np)) / s,
                                    rel=1e-12)

    # and the band parameter is LIVE, not decorative
    z = np.array([5.0 - 1e-3j])
    assert _eb.forward_decaying_root(z, k0=1.0, xp=np,
                                     band=1e-9)[0].real < 0.0
    assert _eb.forward_decaying_root(z, k0=1.0, xp=np,
                                     band=1e-2)[0].real > 0.0


def test_exactly_one_definition_of_each_eme_branch_helper():
    """The three EME sites -- the scalar driver, the diffraction driver and the
    vector solver -- must read ONE definition.  Two of them carried the
    exact-zero pin while the third had the correct band for as long as it has
    existed, which is precisely what a shared definition prevents."""
    for name, fn in (("forward_decaying_root", _eb.forward_decaying_root),
                     ("cut_band", _eb.cut_band)):
        defs = []
        pat = re.compile(r"\s*def\s+%s\b" % (re.escape(name),))
        for path in sorted(_PKG.rglob("*.py")):
            for i, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace")
                    .splitlines(), 1):
                if pat.match(line):
                    defs.append("%s:%d"
                                % (path.relative_to(_PKG).as_posix(), i))
        assert defs == ["elements/eme/_branch.py:%d"
                        % (fn.__code__.co_firstlineno,)], (
            "expected exactly one definition of %s; found %s"
            % (name, defs))


def test_no_exact_zero_branch_pin_survives_in_the_eme_package():
    """Parsed with ``ast`` rather than grepped, so the comments and docstrings
    that QUOTE the removed pin -- which are how the fix records what it removed
    -- are invisible to it while a reintroduced line of code is not.

    SCOPE: an ORDERING comparison of a value's ``.imag`` against an exact
    zero.  That is the defect's exact shape -- the imaginary part of an
    eigensolve output is the backward error, so no branch may be selected by
    its sign alone.

    Two things are deliberately NOT flagged.  ``eps.imag != 0`` on a
    CALLER-SUPPLIED permittivity asks "did the user hand me a lossy material?",
    an exact question about an exact input.  And a ``.real`` tie-break INSIDE a
    band -- ``eme_2d_vector._strip_split_forward``'s ``elif v.real > 0`` after
    its ``|Im| > tol`` tests, and ``_sqrt_decay``'s own ``Re(lam) >= 0``
    convention -- is how an on-cut pair is separated once the band has already
    established that the imaginary part carries no information.  The band is
    what makes that legitimate; the pin was illegitimate because there was no
    band.
    """
    import ast

    def is_zero(node):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and node.value == 0):
            return True
        return (isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub) and is_zero(node.operand))

    def is_imag(node):
        return isinstance(node, ast.Attribute) and node.attr == "imag"

    bad = []
    for path in sorted((_PKG / "elements" / "eme").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8",
                                            errors="replace"))
        except SyntaxError:                              # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            ops = {type(o) for o in node.ops}
            if not (ops & {ast.Lt, ast.Gt, ast.LtE, ast.GtE}):
                continue
            left, right = node.left, node.comparators[0]
            if ((is_imag(left) and is_zero(right))
                    or (is_zero(left) and is_imag(right))):
                bad.append("%s:%d" % (path.name, node.lineno))
    assert not bad, ("an exact-zero branch pin survives in the EME package: "
                     + ", ".join(bad))
