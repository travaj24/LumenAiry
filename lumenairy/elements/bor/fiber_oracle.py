"""Open-cladding (step-index FIBER) guided-mode dispersion oracle for BOR-PMM
Milestone 2 -- the CLEAN per-quantity gate for the coupled radial eigensolve.

Core ``eps1`` on ``[0, a]``, infinite cladding ``eps2 < eps1``.  A guided mode
has ``sqrt(eps2) k0 < q < sqrt(eps1) k0``: oscillatory ``J_m`` in the core,
EXPONENTIALLY DECAYING ``K_m`` (modified Bessel) in the cladding -- so the mode
is truly bound and INSENSITIVE to any outer truncation (unlike the PEC-wall
oracle, which also returns wall-sensitive cladding-box modes).  This is the
standard fiber HE/EH/TE/TM dispersion relation.

Per mode: core ``E_z = A J_m(g1 r)``, ``h_z = B J_m(g1 r)`` with
``g1^2 = eps1 k0^2 - q^2 > 0``; cladding ``E_z = C K_m(kap r)``,
``h_z = D K_m(kap r)`` with ``kap^2 = q^2 - eps2 k0^2 > 0``.  Matching
``E_z, h_z, E_phi, h_phi`` continuous at ``r = a`` gives a ``4x4`` system; the
nontrivial-solution condition ``det = 0`` is the dispersion relation.

Transverse fields (normalized ``h = sqrt(mu0/eps0) H``, ``curl E = i k0 h``):
``E_phi = (i/g^2)[(i m q/r) E_z - k0 dh_z/dr]``,
``h_phi = (i/g^2)[(i m q/r) h_z + k0 eps dE_z/dr]`` -- with ``g^2 = -kap^2`` in
the cladding.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, jvp, kv, kvp


def fiber_det(q, m, a, eps1, eps2, k0):
    g1 = np.sqrt(complex(eps1) * k0 ** 2 - q ** 2)        # core (real>0 guided)
    kap = np.sqrt(q ** 2 - complex(eps2) * k0 ** 2)       # cladding decay
    M = np.zeros((4, 4), dtype=complex)
    Jm, Jmp = jv(m, g1 * a), g1 * jvp(m, g1 * a)
    Km, Kmp = kv(m, kap * a), kap * kvp(m, kap * a)
    imq = 1j * m * q / a
    c1 = 1.0 / g1 ** 2
    c2 = 1.0 / (-kap ** 2)                                 # g^2 = -kap^2
    # (1) E_z continuous:  A Jm - C Km = 0
    M[0] = [Jm, 0, -Km, 0]
    # (2) h_z continuous:  B Jm - D Km = 0
    M[1] = [0, Jm, 0, -Km]
    # (3) E_phi continuous: (1/g^2)[(imq) E_z - k0 dh_z/dr]
    M[2] = [c1 * imq * Jm, -c1 * k0 * Jmp,
            -c2 * imq * Km, c2 * k0 * Kmp]
    # (4) h_phi continuous: (1/g^2)[(imq) h_z + k0 eps dE_z/dr]
    M[3] = [c1 * k0 * eps1 * Jmp, c1 * imq * Jm,
            -c2 * k0 * eps2 * Kmp, -c2 * imq * Km]
    return np.linalg.det(M)


def fiber_modes(m, a, eps1, eps2, k0, n_scan=6000):
    """Guided-mode propagation constants q (descending = most-guided first)."""
    lo = np.sqrt(eps2) * k0 * (1 + 1e-7)
    hi = np.sqrt(eps1) * k0 * (1 - 1e-7)
    qs = np.linspace(lo, hi, n_scan)
    d = np.array([fiber_det(q, m, a, eps1, eps2, k0).real for q in qs])
    roots = []
    for i in range(len(qs) - 1):
        if d[i] == 0.0 or d[i] * d[i + 1] < 0:
            a0, b0 = qs[i], qs[i + 1]
            for _ in range(80):
                mid = 0.5 * (a0 + b0)
                if d[i] * fiber_det(mid, m, a, eps1, eps2, k0).real <= 0:
                    b0 = mid
                else:
                    a0 = mid
            roots.append(0.5 * (a0 + b0))
    return np.array(sorted(set(np.round(roots, 9)), reverse=True))
