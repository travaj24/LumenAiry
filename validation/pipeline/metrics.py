# validation/pipeline/metrics.py -- the per-frame scoring vocabulary.
#
# VERBATIM IN BEHAVIOUR from ``sumap_score_121.spot`` / ``.compare`` (which
# are themselves ``fan_multi_121.py``'s own per-frame spot block), and that is
# a requirement, not a convenience: the acceptance of this pipeline is that it
# reproduces PROBE_SUM_AT_APERTURE's printed numbers, and a metric that
# differed in its ring binning or in its core mask would make "reproduces" a
# claim about two different quantities.  The one thing added is that the EE
# radii are a parameter instead of three literals, because a generic driver
# cannot hard-code a design's spot size.
#
# cp1252-safe ASCII only.
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def spot(F, dx_out, ee_radii=(3e-6, 6e-6, 12e-6)) -> Dict[str, float]:
    """FWHM, encircled energy, window power and peak of one frame.

    The FWHM is read off a RADIAL PROFILE about the peak pixel, azimuthally
    averaged in one-pixel rings -- not from a Gaussian fit and not from a
    row cut -- because that is what the campaign's published columns are.
    ``nb`` is capped at 400 rings for the same reason (a 1024-tile at 0.2 um
    would otherwise bin out to 102 um, far past anything the profile means).
    """
    F = np.asarray(F)
    nt = F.shape[-1]
    dxo = float(dx_out)
    I = np.abs(F) ** 2
    tot = float(I.sum())
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    rr = np.hypot((np.arange(I.shape[1]) - ix)[None, :] * dxo,
                  (np.arange(I.shape[0]) - iy)[:, None] * dxo)
    nb = min(nt // 2, 400)
    ring = np.clip((rr / dxo).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cn = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(cn[:nb], 1)) / I[iy, ix]
    rb = (np.arange(nb) + 0.5) * dxo
    idx = np.where(prof < 0.5)[0]
    out = {'fwhm': float(2 * rb[idx[0]]) if len(idx) else float('nan'),
           'power': tot * dxo * dxo, 'peak': float(I[iy, ix]),
           'peak_ij': [int(iy), int(ix)]}
    for r in ee_radii:
        out[f'ee{_ee_label(r)}'] = (float(I[rr <= float(r)].sum()) / tot
                                    if tot > 0.0 else float('nan'))
    return out


def _ee_label(r):
    """``3e-6 -> '3'``, ``12e-6 -> '12'``, ``2.5e-6 -> '2p5'`` -- a stable,
    filesystem- and JSON-safe key so a report column name is a property of the
    radius and not of the order it was listed in."""
    um = float(r) * 1e6
    if abs(um - round(um)) < 1e-9:
        return str(int(round(um)))
    return ('%g' % um).replace('.', 'p')


def compare(A, B) -> Dict[str, float]:
    """Field-to-field comparison: relative L2, PISTON, and core phase rms.

    ``piston`` is ``arg(<A, B>)`` -- the intensity-weighted global phase
    between the two frames -- and the core phase rms is measured with that
    piston REMOVED over the region above 10 % of peak amplitude.  Splitting
    them is the whole reason this function exists rather than a bare relative
    L2: a chain-to-chain piston shows up in one column and a shape error in
    the other, and PROBE_SUM_AT_APERTURE S1.2 is the case where the two
    diverge by six decades (a library edit moved the global phase 2.88 rad
    while every intensity metric held to 0.001 EE points)."""
    A = np.asarray(A)
    B = np.asarray(B)
    nA = float(np.linalg.norm(A))
    ip = complex(np.vdot(A, B))
    piston = float(np.angle(ip))
    a = np.abs(A)
    msk = a > 0.1 * a.max()
    d = np.angle(B[msk] * np.conj(A[msk])) - piston
    d = (d + np.pi) % (2 * np.pi) - np.pi
    w = a[msk] ** 2
    return {'rel_l2': float(np.linalg.norm(B - A) / nA), 'piston': piston,
            'phase_rms_core': float(np.sqrt((w * d * d).sum() / w.sum())),
            'amp_ratio': float(np.linalg.norm(B) / nA)}
