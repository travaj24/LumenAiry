"""THE ONE branch-cut decision of the EME (eigenmode-expansion) engines.

WHAT THE DECISION IS.  An EME strip solve returns a squared lateral wavenumber
``ky^2`` (``qz^2`` in the diffraction driver) and takes its square root.  Of the
two roots, the FORWARD one must be the DECAYING one -- ``Im(ky) >= 0``, so the
propagator ``exp(i ky h)`` never grows over a forward thickness and the
S-matrix cascade is unconditionally stable.  ``numpy.sqrt``'s principal branch
guarantees only ``Re >= 0``, so a rule is needed.

WHAT WAS WRONG (5.45.1).  ``eme_2d._ky_forward`` and
``eme_diffraction.mode_match`` both spelled that rule as an EXACT-ZERO pin::

    ky = np.where(ky.imag < 0.0, -ky, ky)

For a PROPAGATING strip mode of a lossless layer ``ky^2`` is exactly real
positive, so ``Im(ky)`` is not physics -- it is the eigensolver's backward
error, ``~ eps_mach * ||A||`` with ``||A|| ~ 4 Nx^2 / Lx^2``.  The pin then
decides the mode's DIRECTION on the last bit of that error, and flipping a
propagating mode to ``-ky`` does not merely perturb it: ``-ky`` is that mode's
BACKWARD partner, so the forward set silently acquires a backward member and
the mode-match ``a + b`` conditions on it.

Measured (``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`` section 6,
``validation/probe_scope_bor_guards/e_eme_flip*.json``): ``eme_2d.layer_modes``
at ``Nx = 96``, ``Lx = Ly = 1``, ``k0 = 20 pi``, ``ky0 = 0.37``, window
``(26055.8, 35530.6)`` returns **62 modes** with a real ``eps`` (where
``eigh``'s ``Im(lam)`` is exactly zero and the pin cannot fire) against **69 on
Windows and 71 on WSL** with the same ``eps`` plus an INFINITESIMAL ``i 1e-30``
on the high region -- a gap of 2.55 % / 3.64 %, **with the mode lists differing
between builds** (first mode 26137.93 WIN against 26065.96 WSL).  The deciding
``|Im ky| / |ky|`` is **1.34e-16** against a physical **1.81e-29**: thirteen
decades of backward error deciding a direction.  Flip counts are
build-dependent (29 WIN / 30 WSL at Nx = 96; 32 / 22 at Nx = 128; 51 / 44 at
Nx = 128, k0 = 40 pi) and the onset is a resolution, not a material: 0 flipped
propagating modes at ``Nx <= 64``, flips from ``Nx = 96`` up as ``||A||``
grows.  An answer that changes with the build is a defect, not noise
(``docs/TESTING_STANDARDS.md``).

THE FIX, AND WHERE IT COMES FROM.  The module's OWN vector sibling
``eme_2d_vector._strip_split_forward`` has always carried the correct shape --
a band RELATIVE to the spectrum's top, floored at 1.0 -- and it is the shape
``rcwa/_core._sqrt_decay`` took in round 2 of the Cartesian branch-cut fix and
``elements/bor/_orient.py`` took in this one.  This module makes it ONE
definition that all three EME sites read, so the scalar driver, the diffraction
driver and the vector solver cannot drift apart again.
"""
from __future__ import annotations

from ...backend.array import array_namespace

#: THE ON-CUT BAND, relative to the spectrum's top and floored at 1.0.  ``ky``
#: here is DIMENSIONLESS (the EME modules work in ``k0``-normalized units), so
#: 1.0 is the correct floor and no ``k0`` enters -- unlike the BOR peer
#: (``elements/bor/_orient._BOR_CUT_BAND_REL``), whose ``q`` carries units of
#: inverse length and is therefore floored at ``k0``.
#:
#: The value is the one ``eme_2d_vector._strip_split_forward`` has used since
#: it was written.  Measured separation on the fixture above: the on-cut
#: population's ``|Im ky| / max(1, max|ky|)`` reaches 1.34e-16 and the
#: physically lossy population starts at 1.81e-29 * (a lossy layer's own
#: ratio) -- the band at 1e-9 sits SEVEN decades above the backward error it
#: must absorb, and a mode with real loss large enough to matter for the
#: cascade sits far above it.
_EME_CUT_BAND_REL = 1e-9


def cut_band(z, *, xp=None, band: float = _EME_CUT_BAND_REL):
    """The on-cut band for a spectrum ``z``: ``band * max(1.0, max|z|)``.

    Floored at 1.0 so a sub-unit spectrum still gets an absolute band, which is
    what keeps a nearly-degenerate strip (every ``|ky|`` small) from having its
    band collapse onto its own backward error -- the failure mode the BOR peer
    measured at a radial cutoff.
    """
    if xp is None:
        xp = array_namespace(z)
    if getattr(z, "size", 1) == 0:
        return band
    return band * xp.maximum(xp.max(xp.abs(z)), 1.0)


def forward_decaying_root(z, *, xp=None, band: float = _EME_CUT_BAND_REL):
    """Put a spectrum of square roots ``z`` on the FORWARD (decaying) branch.

    OFF the cut (``|Im z| > band * scale``) the mode is genuinely evanescent or
    genuinely lossy and the forward root is the DECAYING one, so a root with
    ``Im z < 0`` is negated -- the shipped rule, unchanged, because there the
    sign of ``Im z`` is physics.

    ON the cut (``|Im z| <= band * scale``) the mode is PROPAGATING, the exact
    root is real, and ``Im z`` is the eigensolver's backward error.  Negating
    would hand back the mode's BACKWARD partner.  The root is CONJUGATED
    instead: that keeps ``Re z >= 0`` -- which is what makes it the forward
    member of the pair, and is the convention
    ``eme_2d_vector._strip_split_forward`` resolves its own on-cut ties by --
    while also putting ``Im z >= 0``, so the ``|exp(i z h)| <= 1`` guarantee the
    cascade rests on survives.  The two differ by ``2 |Im z| ~ 1e-16`` in a
    quantity of size ``|z|``; the negation differed by ``2 |z|``.

    This is exactly the shape ``rcwa/_core._sqrt_decay`` took in round 2 of the
    Cartesian branch-cut fix, transposed from that function's ``Re >= 0``
    convention to this one's ``Im >= 0``.
    """
    if xp is None:
        xp = array_namespace(z)
    tol = cut_band(z, xp=xp, band=band)
    neg = xp.imag(z) < 0.0
    on_cut = xp.abs(xp.imag(z)) <= tol
    return xp.where(neg, xp.where(on_cut, xp.conj(z), -z), z)
