"""TASK E fixtures -- built HERE, from a description, not copied from the
build's own probes.  Imported by every ``ve_*`` probe so the PRE tree and the
POST tree are driven by ONE description of each fixture.

Nothing in this module imports lumenairy: the fixture INPUTS must be identical
bits on both trees, so they are pure numpy.
"""
from __future__ import annotations

import numpy as np

PI = np.pi


# --------------------------------------------------------------------------- #
#  eps(x) profiles                                                             #
# --------------------------------------------------------------------------- #
def eps_split(Nx, lo=2.25, hi=12.0, imag=0.0, where="hi"):
    """Half-and-half x profile; ``imag`` added to the HIGH region only (or to
    both, or to the LOW region), so a real-eps arm takes eigh and any complex
    arm takes scipy eig."""
    e = np.full(Nx, complex(lo))
    e[Nx // 2:] = complex(hi)
    if imag != 0.0:
        if where in ("hi", "both"):
            e[Nx // 2:] += 1j * imag
        if where in ("lo", "both"):
            e[:Nx // 2] += 1j * imag
    return e


def eps_centre(Nx, lo=2.25, hi=12.0, imag=0.0):
    """A centred high block (the second strip of the build's own fixture)."""
    e = np.full(Nx, complex(lo))
    e[Nx // 4:3 * Nx // 4] = complex(hi)
    if imag != 0.0:
        e[Nx // 4:3 * Nx // 4] += 1j * imag
    return e


def eps_three(Nx, imag=0.0):
    """A three-region profile, low contrast -- a DIFFERENT spectrum shape from
    the build's fixture (its max|ky| is set by the grid, its band edges are
    not where the build's are)."""
    e = np.full(Nx, complex(1.0))
    e[Nx // 5:2 * Nx // 5] = complex(4.0)
    e[3 * Nx // 5:] = complex(6.25)
    if imag != 0.0:
        e[3 * Nx // 5:] += 1j * imag
    return e


def strips_two(Nx, Ly, imag=0.0):
    """The build's own two-strip layer (their fixture, restated here so my
    numbers are taken on my own construction of it)."""
    return [(eps_split(Nx, imag=imag), 0.5 * Ly),
            (eps_centre(Nx, imag=imag), 0.5 * Ly)]


def strips_three(Nx, Ly, imag=0.0):
    """MY OWN three-strip layer: different contrast, different heights, a
    region of eps = 1 so the spectrum straddles the light line."""
    return [(eps_three(Nx, imag=imag), 0.25 * Ly),
            (eps_split(Nx, lo=1.0, hi=6.25, imag=imag), 0.5 * Ly),
            (eps_centre(Nx, lo=1.0, hi=4.0, imag=imag), 0.25 * Ly)]


# --------------------------------------------------------------------------- #
#  THE STRIP BATTERY -- (name, kwargs) for _ky_forward / strip_x_modes         #
# --------------------------------------------------------------------------- #
def strip_fixtures():
    """>= 10 strip-level EME fixtures.

    Each entry: (name, dict(profile=, Nx=, Lx=, k0=, kx0=, imag=, qz2=)).
    ``imag`` spans a LOSSLESS real eps, three INFINITESIMAL losses
    (1e-30/1e-20/1e-12 -- physical no-ops that route the eigensolve from eigh
    to eig), two GENUINE losses (1e-3, 1e-1) and two GAINS (-1e-6, -1e-3).
    ``qz2`` is the scan variable: 0.0, a value inside the guided band and a
    value chosen to sit ON a band edge (ky^2 ~ 0), which is where the root's
    own magnitude collapses and any band relative to the spectrum's TOP is
    widest relative to the mode.
    """
    F = []
    k0a, k0b = 20.0 * PI, 40.0 * PI
    for Nx in (48, 96, 128):
        for imag in (0.0, 1e-30, 1e-20, 1e-12, 1e-3, 1e-1, -1e-6, -1e-3):
            F.append(("split_Nx%d_k20pi_im%g_qz0" % (Nx, imag),
                      dict(profile="split", Nx=Nx, Lx=1.0, k0=k0a, kx0=0.0,
                           imag=imag, qz2=0.0)))
    for imag in (0.0, 1e-30, 1e-12, 1e-3, -1e-3):
        F.append(("split_Nx128_k40pi_im%g_qz0" % imag,
                  dict(profile="split", Nx=128, Lx=1.0, k0=k0b, kx0=0.0,
                       imag=imag, qz2=0.0)))
        F.append(("split_Nx96_k20pi_im%g_qz26e3" % imag,
                  dict(profile="split", Nx=96, Lx=1.0, k0=k0a, kx0=0.0,
                       imag=imag, qz2=26055.8)))
        F.append(("centre_Nx96_kx037_im%g_qz0" % imag,
                  dict(profile="centre", Nx=96, Lx=1.0, k0=k0a, kx0=0.37,
                       imag=imag, qz2=0.0)))
        F.append(("three_Nx96_k20pi_im%g_qz0" % imag,
                  dict(profile="three", Nx=96, Lx=1.0, k0=k0a, kx0=0.0,
                       imag=imag, qz2=0.0)))
        # a SMALL unit system: Lx = 1e-3, k0 = 2e4 pi -- same physics, the
        # spectrum's top is 1e3 x larger
        F.append(("split_UNITmm_im%g_qz0" % imag,
                  dict(profile="split", Nx=96, Lx=1e-3, k0=2e4 * PI, kx0=0.0,
                       imag=imag, qz2=0.0)))
        # a LARGE unit system: Lx = 1e3, k0 = 2e-2 pi -- the spectrum's top is
        # 1e3 x SMALLER, so cut_band's literal 1.0 floor can engage
        F.append(("split_UNITkm_im%g_qz0" % imag,
                  dict(profile="split", Nx=96, Lx=1e3, k0=2e-2 * PI, kx0=0.0,
                       imag=imag, qz2=0.0)))
    return F


def build_eps(spec):
    p = spec["profile"]
    if p == "split":
        return eps_split(spec["Nx"], imag=spec["imag"])
    if p == "centre":
        return eps_centre(spec["Nx"], imag=spec["imag"])
    if p == "three":
        return eps_three(spec["Nx"], imag=spec["imag"])
    raise SystemExit("unknown profile %r" % (p,))
