"""PHASE A prototype -- native 2-D slant metric for the 2-D PMM Jones solver.

READ-ONLY with respect to ``lumenairy/``: the slant is injected by
monkeypatching ``twod_jones._layer_eigenmodes_tensor`` so nothing under the
package is modified.  See docs/audits/BUILD_PMM2D_SLANT_METRIC_2026_08_16.md
section 1 for the derivation this implements.

The formulation, in one line: in the sheared frame ``u = x - tx z``,
``v = y - ty z`` the structure is z-invariant, ``det J = 1`` exactly, and the
whole slant is a CONVECTION term ``c*(tx*Kx + ty*Ky)`` added to each of the
four diagonal field blocks of the 4N first-order generator -- the exact 2-D
generalization of the shipped 1-D ``L[_sl,_sl] += tan_conv * Dopx``
(_core.py:5959-5962).
"""
from __future__ import annotations

import numpy as np

import lumenairy
assert "lum_sl" in lumenairy.__file__, lumenairy.__file__

from lumenairy.elements.pmm import twod_jones as TJ
from lumenairy.elements.rcwa._core import _block, _select_forward_flux

_C = np.complex128

# module-global slant, read by the patched eigensolver
_SLANT = {"tx": 0.0, "ty": 0.0, "c": -1j}
_ORIG = TJ._layer_eigenmodes_tensor


def _slanted_eigenmodes(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ,
                        EZX=None, EZY=None, EXZ=None, EYZ=None):
    """4N generator with the slant convection added to all four field blocks.

    Reduces EXACTLY to the shipped ``_layer_eigenmodes_tensor`` when
    tx = ty = 0 (the convection term is then identically zero and the
    generator is the library's own A=B=0 4N generator).
    """
    tx, ty, c = _SLANT["tx"], _SLANT["ty"], _SLANT["c"]
    if tx == 0.0 and ty == 0.0:
        return _ORIG(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ, EZX, EZY, EXZ, EYZ)

    Kx = np.asarray(Kx).astype(_C)
    Ky = np.asarray(Ky).astype(_C)
    N = Kx.shape[0]
    I = np.eye(N, dtype=_C)
    Z = np.zeros((N, N), dtype=_C)
    Ez_inv = np.linalg.inv(EZZ)

    # --- the library's own P and Q blocks, verbatim -------------------------
    P = _block(np, [
        [Kx @ Ez_inv @ Ky,        I - Kx @ Ez_inv @ Kx],
        [Ky @ Ez_inv @ Ky - I,    -Ky @ Ez_inv @ Kx],
    ])
    Q = _block(np, [
        [Cyx + Kx @ Ky,        Cyy - Kx @ Kx],
        [Ky @ Ky - Cxx,        -(Cxy + Ky @ Kx)],
    ])

    # --- out-of-plane cross blocks (kept for completeness; None -> 0) -------
    def _z(t):
        return Z if t is None else np.asarray(t).astype(_C)
    EZXa, EZYa, EXZa, EYZa = _z(EZX), _z(EZY), _z(EXZ), _z(EYZ)
    A = _block(np, [
        [-1j * (Kx @ Ez_inv @ EZXa),   -1j * (Kx @ Ez_inv @ EZYa)],
        [-1j * (Ky @ Ez_inv @ EZXa),   -1j * (Ky @ Ez_inv @ EZYa)],
    ])
    B = _block(np, [
        [-1j * (EYZa @ Ez_inv @ Ky),    1j * (EYZa @ Ez_inv @ Kx)],
        [1j * (EXZa @ Ez_inv @ Ky),    -1j * (EXZa @ Ez_inv @ Kx)],
    ])

    G = _block(np, [[A, P], [Q, B]])

    # --- THE SLANT: convection on all four diagonal field blocks ------------
    Kt = tx * Kx + ty * Ky
    G = G + c * np.kron(np.eye(4, dtype=_C), Kt)

    gam, Vfull = np.linalg.eig(G)
    fidx = _select_forward_flux(gam, Vfull, N)
    fset = set(np.asarray(fidx).tolist())
    bidx = np.array(sorted(set(range(4 * N)) - fset))
    lam, lam_b = gam[fidx], gam[bidx]
    Vf, Vb = Vfull[:, fidx], Vfull[:, bidx]
    return (Vf[:2 * N, :], Vf[2 * N:, :], lam,
            Vb[:2 * N, :], Vb[2 * N:, :], lam_b)


TJ._layer_eigenmodes_tensor = _slanted_eigenmodes


# ---------------------------------------------------------------------------
# The even-parity fold MUST be disabled for a slanted layer.
#
# At normal incidence (kt < 1e-12) pmm_jones_2d takes the F2 even-parity fold
# (twod_jones.py:614-627), which calls _tensor_layer_modes(return_ops=True) and
# _symmetric_cascade_rt -- it NEVER reaches _layer_eigenmodes_tensor.  A shear
# destroys the x -> -x flip symmetry the fold assumes, so if the fold is left
# on, a slanted layer SILENTLY RETURNS THE VERTICAL ANSWER: no warning, energy
# conserved, wrong by 9.2e-02 (10 deg) to 2.5e-01 (35 deg) and NOT converging
# with n_orders.  MEASURED 2026-08-16; disabling it restores tracking of the
# vertical control (9.1e-03 vs ctrl 8.8e-03 at n_orders=7).
#
# Production equivalent: return None from return_ops when slanted, exactly as
# the out-of-plane path already does (twod_jones.py:312-318).
# ---------------------------------------------------------------------------
_ORIG_TLM = TJ._tensor_layer_modes


def _slant_aware_tensor_layer_modes(*a, return_ops=False, **kw):
    if return_ops and (_SLANT["tx"] != 0.0 or _SLANT["ty"] != 0.0):
        return None                     # -> caller falls back to the full solve
    return _ORIG_TLM(*a, return_ops=return_ops, **kw)


TJ._tensor_layer_modes = _slant_aware_tensor_layer_modes


def set_slant(tx=0.0, ty=0.0, c=-1j):
    _SLANT.update(tx=float(tx), ty=float(ty), c=c)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def binary_cell(nx, duty, eps_r, eps_g, shift_px=0, ny=1):
    """Binary x-grating (uniform in y) as a (nx, ny, 3, 3) tensor cell.

    The ridge is an INTEGER number of pixels wide and rolled by an INTEGER
    number of pixels, so every wall lands exactly on a pixel boundary and the
    cell is represented EXACTLY (no pixelation error) by _cell_to_walls_tile.
    """
    nr = int(round(duty * nx))
    line = np.full(nx, eps_g, dtype=_C)
    line[:nr] = eps_r
    line = np.roll(line, int(shift_px))
    cell = np.zeros((nx, ny, 3, 3), dtype=_C)
    for i in range(3):
        cell[:, :, i, i] = line[:, None]
    return cell
