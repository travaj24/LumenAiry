"""THE ONE relative band, and the two forward-branch selectors, that every
modal engine in the library resolves its square-root sign ambiguity with.

WHAT THE SHARED DECISION IS.  A modal solve returns a SQUARED axial or lateral
wavenumber (``lam^2``, ``q^2``, ``ky^2``, ``qz^2``); the square root has two
roots, and which one is the FORWARD member of the pair is the whole bookkeeping
of an S-matrix cascade.  For a PROPAGATING mode of a lossless layer the exact
root sits exactly ON the branch cut, so the component that would decide the
sign is not physics -- it is the eigensolver's backward error,
``~ eps_mach * ||A||``, which moves with the BLAS micro-kernel, the thread
width and the platform.  Every engine therefore decides with a BAND rather than
a sign test: a component within ``band * scale`` of zero is "on the cut" and is
resolved by a rule that a rounding-level value cannot defeat.

WHAT IS SHARED HERE AND WHAT IS NOT.  :func:`band_mask` is the comparison, and
it is the same object at all four sites::

    lumenairy/elements/rcwa/_core.py       _sqrt_decay        |Re r| band
    lumenairy/elements/eme/_branch.py      forward_decaying_root, cut_band
    lumenairy/elements/pmm/_core.py        _forward_branch_flip  |Im q| band
    lumenairy/elements/bor/_orient.py      forward_orient        |Im q| band

The SCALE each band is taken relative to stays at the call site, because the
four scales are genuinely different quantities and folding them together would
move numbers rather than deduplicate a decision: the Cartesian engines floor
the spectrum's top at a dimensionless 1.0, while the EME and BOR engines carry
a DIMENSIONED wavenumber and floor at ``|k0|`` (a dimensionless literal there
makes the branch decision depend on the caller's unit system).  Each site's own
docstring carries the populations its scale and band were measured against.
This is the same split :func:`lumenairy.elements.bor._orient.flux_is_strong`
and ``channel_core`` use: share the comparison, keep each engine's own leg.

THERE ARE TWO SELECTORS, NOT ONE, AND THEY ARE NOT INTERCHANGEABLE.
:func:`negate_forward` selects with ``where(flip, -z, z)``;
:func:`signed_forward` multiplies by a real ``+/-1``.  On real arrays the two
agree bit for bit.  On COMPLEX arrays they do not, and the difference is not
academic -- measured over a 12-value census of the values these bands actually
see (signed zeros, subnormals, unit reals, infinities, NaN)::

    z            where(flip, -z, z)      z * where(flip, -1.0, 1.0)
    1+0j         -1-0j                   -1+0j        sign of the zero imag part
    -0j          -0+0j                   0j           sign of the zero real part
    inf+0j       -inf-0j                 -inf+nanj    inf * 0 in the cross term
    nan+0j       nan-0j                  nan+nanj     same cross term

A complex multiply forms ``(a*1 - b*0, a*0 + b*1)``, so it rewrites the sign of
a zero imaginary part and turns an infinite real part's cross term into a NaN,
where negation touches only the sign bits.  Each engine keeps the selector it
was measured with: the RCWA layer root uses :func:`signed_forward` because
multiplying by a real constant is HOLOMORPHIC and its JAX twins differentiate
through the flip (its own docstring carries the gradient measurement that
rejected ``conj``), and the PMM and BOR selectors use :func:`negate_forward`.
Swapping one for the other is a numerical change, so it needs a measurement --
which is why they are two named functions here rather than one with a flag.
"""
from __future__ import annotations

from .backend.array import array_namespace

__all__ = ["band_mask", "negate_forward", "signed_forward"]


def band_mask(r, *, scale, band, xp=None):
    """``|r| <= band * scale`` -- THE ONE on-cut band comparison.

    ``r`` is the component whose smallness means "on the cut": the REAL part of
    the root for an engine whose forward branch is ``Re >= 0`` (RCWA), the
    IMAGINARY part for one whose forward branch is ``Im >= 0`` (EME, PMM, BOR).

    ``scale`` is the caller's own derived scale -- the spectrum's largest
    element, floored at whatever that engine's units make the right floor --
    and ``band`` the relative half-width it carries.  A caller that already
    holds the PRODUCT (``eme._branch.cut_band`` returns it, because its two
    one-sided legs need it too) passes it as ``scale`` with ``band=1.0``;
    ``x * 1.0`` is exact for every float, so that spelling is bit-identical to
    comparing against the product directly.

    Comparison is ``<=``, so a component of exactly zero is always ON the cut.
    That matters: it is the case a lossless real ``lam^2`` produces in exact
    arithmetic, and the case a hand-written fixture produces.

    ``xp`` is the array namespace; ``None`` detects it from ``r``.  Every
    in-tree caller passes it explicitly, so the traced JAX twins and the eager
    NumPy paths run this same object.  Nothing here reads a traced value as a
    Python bool, so the body is ``jit``- and ``grad``-safe.
    """
    if xp is None:
        xp = array_namespace(r)
    return xp.abs(r) <= band * scale


def negate_forward(z, flip, *, xp=None):
    """``-z`` where ``flip``, ``z`` elsewhere -- selection by NEGATION.

    Negation flips only sign bits, so the zero parts, infinities and NaNs of a
    complex root survive it unchanged.  See the module docstring for the census
    that separates this from :func:`signed_forward`; the two are not
    interchangeable on complex input.
    """
    if xp is None:
        xp = array_namespace(z)
    return xp.where(flip, -z, z)


def signed_forward(z, flip, *, xp=None):
    """``z`` times a real ``-1.0`` where ``flip``, ``+1.0`` elsewhere --
    selection by a REAL SIGN.

    Multiplying by a real constant whose sign is a piecewise-constant boolean
    predicate is holomorphic on each piece, so reverse-mode AD carries the
    analytic cotangent of the branch actually taken.  That is why the RCWA
    layer root selects this way; see the module docstring for what a complex
    multiply costs relative to :func:`negate_forward`, and
    ``rcwa/_core._sqrt_decay`` for the gradient measurement behind the choice.
    """
    if xp is None:
        xp = array_namespace(z)
    return z * xp.where(flip, -1.0, 1.0)
