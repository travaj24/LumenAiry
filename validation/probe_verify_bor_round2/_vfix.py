"""INDEPENDENT fixtures for the round-2 BOR/EME verification.

Written from the physics and the public API, NOT imported from
``validation/probe_fix_bor_round2`` -- the point of this directory is that the
populations of section 4 of the round-2 report are re-derived on a battery this
verification chose.

Every geometry here is a PEC-walled body of revolution: a radial permittivity
profile on ``[0, Rbig]``, one azimuthal order ``m``, a vacuum wavenumber ``k0``.
``rbl`` is the cell radius in VACUUM WAVELENGTHS (``Rbig k0 / 2 pi``), which is
the axis the nodal basis's spurious sea grows along.
"""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def rbig_of(rbl, k0):
    """The cell radius that makes the cell ``rbl`` vacuum wavelengths wide."""
    return float(rbl) * TWOPI / float(k0)


# --------------------------------------------------------------------- #
#  Radial permittivity profiles.  Each returns a callable of r.
#  ``im_rel`` is Im(eps)/Re(eps) applied to the region named by ``where``.
# --------------------------------------------------------------------- #
def uniform(eps, im_rel=0.0):
    """A homogeneous medium.  The family whose channel count has a closed form
    (Bessel zeros) and whose nodal cascade the 5.45.1 census calls accurate."""
    e = complex(eps) * (1.0 + 1j * float(im_rel))

    def f(r):
        return np.full(np.shape(r), e, dtype=complex)
    return f


def ring(eps_lo, eps_hi, Rbig, n_rings=4, duty=0.5, im_rel=0.0, im_on="hi"):
    """A concentric ring grating: ``n_rings`` periods across the radius, the
    high region occupying ``duty`` of each.  This is the family the nodal
    basis's divergence-violating sea actually damages."""
    lo, hi = complex(eps_lo), complex(eps_hi)
    if im_on == "hi":
        hi = hi * (1.0 + 1j * float(im_rel))
    elif im_on == "lo":
        lo = lo * (1.0 + 1j * float(im_rel))
    else:                                   # both
        hi = hi * (1.0 + 1j * float(im_rel))
        lo = lo * (1.0 + 1j * float(im_rel))
    per = float(Rbig) / float(n_rings)

    def f(r):
        r = np.asarray(r, dtype=float)
        frac = np.mod(r, per) / per
        return np.where(frac < duty, hi, lo).astype(complex)
    return f


def segments(eps_list, Rbig, im_rel=0.0, im_on=-1):
    """Equal-width radial segments of the listed permittivities.  ``im_on`` is
    the index of the segment that carries the loss (``None`` = all)."""
    es = [complex(e) for e in eps_list]
    if im_on is None:
        es = [e * (1.0 + 1j * float(im_rel)) for e in es]
    else:
        es[im_on] = es[im_on] * (1.0 + 1j * float(im_rel))
    w = float(Rbig) / len(es)

    def f(r):
        r = np.asarray(r, dtype=float)
        idx = np.clip((r / w).astype(int), 0, len(es) - 1)
        return np.asarray(es, dtype=complex)[idx]
    return f
