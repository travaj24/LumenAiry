"""Analytic step-index circular-waveguide dispersion oracle for BOR-PMM
Milestone 2 (the COUPLED radial eigensolve with the E_r inverse rule).

A two-region cylinder -- core ``eps1`` on ``[0, a]``, cladding ``eps2`` on
``[a, R]`` -- with a PEC wall at ``r = R``, azimuthal order ``m``.  Per mode the
longitudinal fields are

    region 1 (axis):  E_z = A J_m(g1 r),               h_z = B J_m(g1 r)
    region 2:         E_z = C J_m(g2 r) + D Y_m(g2 r),  h_z = E J_m(g2 r) + F Y_m(g2 r)

with ``g_i^2 = eps_i k0^2 - q^2``.  Transverse fields (normalized ``h = sqrt(
mu0/eps0) H``, ``curl E = i k0 h``):

    E_phi = (i/g^2)[ (i m q / r) E_z - k0 dh_z/dr ]
    h_phi = (i/g^2)[ (i m q / r) h_z + k0 eps dE_z/dr ]

Matching: ``E_z, h_z, E_phi, h_phi`` continuous at ``r = a`` (4 conditions; the
NORMAL ``E_r`` jumps -- ``D_r = eps E_r`` continuous -- which is exactly the
inverse-rule the PMM operator must reproduce), and PEC ``E_z = E_phi = 0`` at
``r = R`` (2 conditions).  The ``6x6`` system ``M(q) [A..F]^T = 0`` has a
nontrivial solution only where ``det M(q) = 0`` -- the dispersion relation.

This is a PER-QUANTITY oracle (it pins the actual mode q's, mode by mode), the
oracle the critique requires for the energy-invisible inverse rule.  Sanity:
``eps1 = eps2`` collapses it to the homogeneous spectrum ``q^2 = eps k0^2 -
(j_{m,n}/R)^2`` (TM) and ``... (j'_{m,n}/R)^2`` (TE).
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, jvp, yv, yvp


def _g(eps, k0, q):
    return np.sqrt(complex(eps) * k0 ** 2 - q ** 2)


def stepindex_det(q, m, a, R, eps1, eps2, k0):
    """det of the 6x6 boundary-matching matrix at propagation constant q."""
    g1 = _g(eps1, k0, q)
    g2 = _g(eps2, k0, q)
    # unknown order: [A, B, C, D, E, F]
    Mtx = np.zeros((6, 6), dtype=complex)
    # (1) E_z continuous at a:  A J_m(g1 a) - C J_m(g2 a) - D Y_m(g2 a) = 0
    Mtx[0] = [jv(m, g1 * a), 0, -jv(m, g2 * a), -yv(m, g2 * a), 0, 0]
    # (2) h_z continuous at a:  B J_m(g1 a) - E J_m(g2 a) - F Y_m(g2 a) = 0
    Mtx[1] = [0, jv(m, g1 * a), 0, 0, -jv(m, g2 * a), -yv(m, g2 * a)]
    # (3) E_phi continuous at a (drop common i; per-region 1/g^2 factor):
    #     E_phi ~ (1/g^2)[ (i m q / a) E_z - k0 dh_z/dr ]
    c1 = 1.0 / g1 ** 2
    c2 = 1.0 / g2 ** 2
    imq_a = 1j * m * q / a
    Mtx[2] = [
        c1 * imq_a * jv(m, g1 * a),                       # A (E_z1)
        -c1 * k0 * g1 * jvp(m, g1 * a),                   # B (h_z1)
        -c2 * imq_a * jv(m, g2 * a),                      # C (E_z2 J)
        -c2 * imq_a * yv(m, g2 * a),                      # D (E_z2 Y)
        c2 * k0 * g2 * jvp(m, g2 * a),                    # E (h_z2 J)
        c2 * k0 * g2 * yvp(m, g2 * a),                    # F (h_z2 Y)
    ]
    # (4) h_phi continuous at a:
    #     h_phi ~ (1/g^2)[ (i m q / a) h_z + k0 eps dE_z/dr ]
    Mtx[3] = [
        c1 * k0 * eps1 * g1 * jvp(m, g1 * a),             # A (E_z1)
        c1 * imq_a * jv(m, g1 * a),                       # B (h_z1)
        -c2 * k0 * eps2 * g2 * jvp(m, g2 * a),            # C (E_z2 J)
        -c2 * k0 * eps2 * g2 * yvp(m, g2 * a),            # D (E_z2 Y)
        -c2 * imq_a * jv(m, g2 * a),                      # E (h_z2 J)
        -c2 * imq_a * yv(m, g2 * a),                      # F (h_z2 Y)
    ]
    # (5) E_z = 0 at R:  C J_m(g2 R) + D Y_m(g2 R) = 0
    Mtx[4] = [0, 0, jv(m, g2 * R), yv(m, g2 * R), 0, 0]
    # (6) E_phi = 0 at R: (i m q/R)(E_z2) - k0 g2 (h_z2') = 0; with (5) -> h_z2'=0
    imq_R = 1j * m * q / R
    Mtx[5] = [
        0, 0,
        imq_R * jv(m, g2 * R), imq_R * yv(m, g2 * R),
        -k0 * g2 * jvp(m, g2 * R), -k0 * g2 * yvp(m, g2 * R),
    ]
    return np.linalg.det(Mtx)


def stepindex_modes(m, a, R, eps1, eps2, k0, n_modes=6, n_scan=4000):
    """Propagation constants q (>0) of the bound modes, by scanning det(q) for
    sign changes of its real part and bisecting.  Returns sorted q (descending,
    i.e. most-guided first)."""
    q_hi = np.sqrt(max(eps1, eps2)) * k0 * (1 - 1e-9)
    qs = np.linspace(1e-6, q_hi, n_scan)
    dets = np.array([stepindex_det(q, m, a, R, eps1, eps2, k0).real
                     for q in qs])
    roots = []
    for i in range(len(qs) - 1):
        if dets[i] == 0.0 or dets[i] * dets[i + 1] < 0:
            lo, hi = qs[i], qs[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                dm = stepindex_det(mid, m, a, R, eps1, eps2, k0).real
                if dets[i] * dm <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
    roots = np.array(sorted(set(np.round(roots, 10)), reverse=True))
    return roots[:n_modes]
