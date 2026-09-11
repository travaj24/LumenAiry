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
a band RELATIVE to the spectrum's top -- and it is the shape
``rcwa/_core._sqrt_decay`` took in round 2 of the Cartesian branch-cut fix and
``elements/bor/_orient.py`` took in this one.  This module makes it ONE
definition that all three EME sites read, so the scalar driver, the diffraction
driver and the vector solver cannot drift apart again.

ROUND 2 (D13) -- THE FLOOR WAS UNIT-DEPENDENT AND IS NOW ``k0``.  As first
built, :func:`cut_band` floored the spectrum scale at a LITERAL 1.0, justified
in its own comment by "``ky`` here is DIMENSIONLESS (the EME modules work in
``k0``-normalized units)".  That is false.  ``strip_x_modes`` assembles
``d2/dx2 + eps k0^2`` on a spacing ``Lx / Nx``, so ``lam`` carries
1/length^2 and ``ky`` carries 1/length; ``mode_match`` forms
``exp(i qz depth)``, which is dimensionless only because ``qz`` is 1/length.
``k0`` is a FREE ARGUMENT carrying units, not a normalisation.  A literal 1.0
therefore engages whenever ``max|ky| < 1`` in the caller's units -- the ordinary
case for a sub-micron cell written in nanometres -- and the BRANCH DECISION
moves with the unit system, which the pre-5.45.1 exact-zero pin (having no
scale at all) did not.

MEASURED (``validation/probe_fix_bor_round2/r3_eme_units.py``, both builds):
one 1 um cell at lambda = 1550 nm, ``Nx`` = 96, ``eps_hi = 12 - 1e-6j`` (weak
gain), written in um and in nm.  With the literal floor, ``max|ky|`` reads
191.919 in um and 0.191919 in nm, the nm band is pinned at the floor
(1.000e-09 against the um band's 1.919e-07, i.e. 5.2x WIDER in physical terms),
the orientation census differs (93 negated + 3 CONJUGATED in nm against 96
negated in um) and ``mode_match`` inherits it: ``T00`` = 0.738986606108 in um
against 0.738986551923 in nm, a difference of 5.4e-08 where the metre arm and
the real-``eps`` arms agree to 1e-15.

The floor is now ``|k0|``, which is EXACTLY what ``elements/bor/_orient.
orient_band_scale`` does and for exactly the same reason (audit P2-06:
"absolute thresholds on ``q`` silently returned empty R/T for small-``k0`` unit
systems").  ``k0`` scales as 1/length, so ``max(max|z|, |k0|)`` scales as
1/length, so the ratio test is unit-invariant.  Sites that do not have ``k0``
in scope pass nothing and get ``max|z|`` alone, which is unit-invariant too --
what is NOT allowed, anywhere, is a dimensioned literal.
"""
from __future__ import annotations

from ...backend.array import array_namespace

#: THE ON-CUT BAND, relative to the spectrum's top and floored at ``|k0|``.
#: 1e-9 is the same factor ``eme_2d_vector._strip_split_forward`` has carried
#: since it was written; what ROUND 2 changed is the SCALE it multiplies (see
#: the module docstring's D13 paragraph and :func:`cut_band`).
#:
#: Measured separation on the fixture in the module docstring: the on-cut
#: population's ``|Im ky| / max(max|ky|, |k0|)`` reaches 1.34e-16 and the
#: physically lossy population starts at 1.81e-29 * (a lossy layer's own
#: ratio) -- the band at 1e-9 sits SEVEN decades above the backward error it
#: must absorb, and a mode with real loss large enough to matter for the
#: cascade sits far above it.
_EME_CUT_BAND_REL = 1e-9


def cut_band(z, *, k0=None, xp=None, band: float = _EME_CUT_BAND_REL):
    """The on-cut band for a spectrum ``z``: ``band * max(max|z|, |k0|)``.

    ``z`` carries units of inverse length (``ky``, ``qz``), so the scale the
    band is taken relative to must carry them too, or the DECISION moves with
    the caller's unit system.  Two scales do:

    * ``max|z|`` -- the spectrum's own top, which is what decides on every
      ordinary strip, and
    * ``|k0|`` -- the problem's own inverse length, which FLOORS it.

    The floor matters where the spectrum's top is not a trustworthy scale: a
    nearly-degenerate strip (every ``|ky|`` small), or ``mode_match`` handed a
    caller-supplied ``qz2`` whose modes nearly cancel.  ``k0`` is the floor the
    BOR peer (``elements/bor/_orient.orient_band_scale``) uses for exactly this
    reason -- audit P2-06, "absolute thresholds on ``q`` silently returned
    empty R/T for small-``k0`` unit systems".

    ``k0`` is optional because three of this module's readers are public
    functions whose signatures predate the band (``eme_2d.cell_smatrix``,
    ``dispersion``, ``mode_field``).  Passing nothing gives ``max|z|`` alone,
    which is unit-invariant too -- it scales as 1/length exactly like
    ``|Im z|``, so the ratio test is unchanged by a change of units.  What is
    NOT allowed on any path, and what ROUND 2 removed, is a DIMENSIONED LITERAL
    (the floor was 1.0): that made the band 5.2x wider in physical terms for a
    1 um cell written in nanometres, and moved 3 of 96 roots onto a different
    branch.  See the module docstring.
    """
    if xp is None:
        xp = array_namespace(z)
    floor = None if k0 is None else xp.abs(k0)
    if getattr(z, "size", 1) == 0:
        return band if floor is None else band * floor
    top = xp.max(xp.abs(z))
    return band * (top if floor is None else xp.maximum(top, floor))


def forward_decaying_root(z, *, k0=None, xp=None,
                          band: float = _EME_CUT_BAND_REL):
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
    tol = cut_band(z, k0=k0, xp=xp, band=band)
    neg = xp.imag(z) < 0.0
    on_cut = xp.abs(xp.imag(z)) <= tol
    return xp.where(neg, xp.where(on_cut, xp.conj(z), -z), z)
