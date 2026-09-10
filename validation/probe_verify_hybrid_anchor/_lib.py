"""Shared fixtures / metrics for the INDEPENDENT verification of the hybrid
2-D PMM frame-anchor fix (VERIFY_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11).

Written from the geometry, not copied from the fix's own probes: different
period, wavelength, cell, slant magnitude and staircase rungs, so a shared
mistake cannot survive in both.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import warnings

import numpy as np

import lumenairy


# ------------------------------------------------------------------ arm
def arm():
    """Which BUILD is imported -- decided from ``lumenairy.__file__``, never
    from a flag."""
    p = os.path.abspath(lumenairy.__file__)
    low = p.replace("\\", "/").lower()
    if "/lum_vhyb/" in low:
        name = "fix"
    elif "/lum_v5440/" in low:
        name = "v5440"
    else:
        name = "unknown"
    import numpy as _np
    import scipy as _sp
    build = "win" if sys.platform.startswith("win") else "wsl"
    return dict(arm=name, build=build, tag="%s.%s" % (name, build),
                lumenairy=p, version=lumenairy.__version__,
                python=sys.version.split()[0], platform=platform.platform(),
                numpy=_np.__version__, scipy=_sp.__version__,
                threads={k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS")})


def dump(name, payload):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    a = arm()
    payload = dict(_arm=a, **payload)
    fn = os.path.join(d, "%s.%s.json" % (name, a["tag"]))
    with open(fn, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print("[wrote] " + fn)
    return fn


# ------------------------------------------------------------------ cells
# a 6-column x-ASYMMETRIC, y-varying base pattern; deliberately NOT the fix's
# (which is 6 x on a 1.44 ground at eps 1.15 .. 3.24).
BASE = np.array([
    [2.10, 2.10, 1.30, 1.30],
    [3.05, 2.60, 1.30, 1.72],
    [1.30, 1.30, 1.30, 1.30],
    [1.30, 2.44, 2.44, 1.30],
    [1.30, 1.30, 1.95, 1.95],
    [1.30, 1.30, 1.30, 1.30],
], dtype=float)          # (6 x, 4 y)

# a LOWER cell for the SCOPE / composition rows -- 12 columns so a quarter
# walk is NOT a symmetry of it.
LOWER = np.array([
    [2.90, 2.90, 1.20, 1.20],
    [1.20, 1.20, 1.20, 1.20],
    [1.20, 2.05, 1.20, 1.20],
    [2.50, 1.20, 1.20, 2.50],
    [1.20, 1.20, 1.75, 1.20],
    [1.20, 1.20, 1.20, 1.20],
    [1.60, 1.60, 1.60, 1.20],
    [1.20, 1.20, 1.20, 1.20],
    [1.20, 3.10, 1.20, 1.20],
    [1.20, 1.20, 1.20, 2.20],
    [1.20, 1.20, 1.20, 1.20],
    [2.75, 1.20, 1.20, 1.20],
], dtype=float)          # (12 x, 4 y)

PX = 0.90e-6             # period_x  (the fix's probes use 1.0 / 1.2 um)
PY = 0.90e-6
WL = 0.62e-6             # (the fix's probes use 0.68 um)
NSUP = 1.0
NSUB = 1.5
DTHICK = 0.45e-6

MOUNTS = {                # (theta, phi) in radians
    "normal": (0.0, 0.0),
    "oblique25": (math.radians(25.0), 0.0),
    "conical25_40": (math.radians(25.0), math.radians(40.0)),
}


def upsample(base, ux, uy=1):
    return np.repeat(np.repeat(np.asarray(base), ux, axis=0), uy, axis=1)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def staircase_upsample_factor(base_cols, frac_denom, rungs):
    """Smallest ``U`` for which every MIDPOINT roll of every rung ``K`` in
    ``rungs`` is an EXACT integer number of pixels, for a walk of
    ``period / frac_denom``.

    Midpoint shift of slice ``k`` of ``K`` = ``(k + 1/2)/K * period/D``;
    in pixels of a ``base_cols * U`` cell that is
    ``(2k + 1) * base_cols * U / (2 * D * K)``.
    """
    U = 1
    for K in rungs:
        den = 2 * frac_denom * K
        g = math.gcd(base_cols, den)
        U = _lcm(U, den // g)
    return U


def staircase_cells(base, U, frac_denom, K, sign=+1):
    """The ``K`` VERTICAL slices of the sheared solid, each the top cell rolled
    to its own MIDPOINT depth.  Walk = ``period / frac_denom`` over the whole
    layer; slice ``k`` is rolled by ``sign * (2k+1) * cols /(2 D K)`` pixels (a
    positive ``slant`` translates the cross-section toward ``+x`` with depth,
    so the cell at depth ``z`` is the top cell shifted by ``+t z``, i.e.
    ``np.roll(cell, +pixels, axis=0)``)."""
    cell = upsample(base, U)
    cols = cell.shape[0]
    out = []
    for k in range(K):
        num = (2 * k + 1) * cols
        den = 2 * frac_denom * K
        if num % den:
            raise AssertionError("non-integer roll: %d/%d" % (num, den))
        out.append(np.roll(cell, sign * (num // den), axis=0))
    return out


# ------------------------------------------------------------------ solves
def hybrid(n_orders=5, **kw):
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    return PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_orders=n_orders, **kw)


def solve_slanted(mount, *, base=BASE, U=1, tx=None, ty=0.0, d=DTHICK,
                  n_orders=5, wl=WL, extra=None, pre=None, **kw):
    """``pre`` layers, then ONE slanted patterned layer, then ``extra``."""
    st = hybrid(n_orders=n_orders, **kw)
    for lay in (pre or []):
        st.add_layer(**lay)
    cell = upsample(base, U)
    st.add_layer(d, eps_cell=cell, slant=(tx, ty))
    for lay in (extra or []):
        st.add_layer(**lay)
    th, ph = MOUNTS[mount]
    st.set_source(wl, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = st.solve()
    return st, out


def solve_staircase(mount, *, base=BASE, U, frac_denom, K, d=DTHICK,
                    n_orders=5, wl=WL, extra=None, pre=None, sign=1, **kw):
    """The SAME solid as :func:`solve_slanted` but as ``K`` VERTICAL slices --
    lab-referenced by construction, so it is an ORACLE the anchor cannot have
    been fitted to."""
    st = hybrid(n_orders=n_orders, **kw)
    for lay in (pre or []):
        st.add_layer(**lay)
    for c in staircase_cells(base, U, frac_denom, K, sign=sign):
        st.add_layer(d / K, eps_cell=c)
    for lay in (extra or []):
        st.add_layer(**lay)
    th, ph = MOUNTS[mount]
    st.set_source(wl, theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = st.solve()
    return st, out


# ------------------------------------------------------------------ metrics
def order_index(orders):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(orders))}


def amp_residual(a, b):
    """Relative Frobenius residual of the (2, N) x/y amplitudes on the orders
    BOTH carry, normalized by the reference ``b``."""
    ia, ib = order_index(a["orders"]), order_index(b["orders"])
    keys = sorted(set(ia) & set(ib))
    A = np.concatenate([a["Ex"][:, [ia[k] for k in keys]],
                        a["Ey"][:, [ia[k] for k in keys]]], axis=1)
    B = np.concatenate([b["Ex"][:, [ib[k] for k in keys]],
                        b["Ey"][:, [ib[k] for k in keys]]], axis=1)
    return float(np.linalg.norm(A - B) / np.linalg.norm(B)), len(keys)


def jones_residual(J, Jref):
    J = np.asarray(J)
    Jref = np.asarray(Jref)
    return float(np.linalg.norm(J - Jref) / np.linalg.norm(Jref))


def rephase(a, w, k0):
    """A COPY of the per-order dict with ``exp(i k0 alpha_m . w)`` applied.
    ``w = -W`` undoes the shipped anchor; ``w = -2W`` is the conjugate arm."""
    ph = np.exp(1j * k0 * (np.asarray(a["kx"]) * w[0]
                           + np.asarray(a["ky"]) * w[1]))
    out = dict(a)
    out["Ex"] = a["Ex"] * ph
    out["Ey"] = a["Ey"] * ph
    return out


def best_global_phase(a, b):
    """``min_phi || e^{i phi} A - B || / ||B||`` and the minimizing phase."""
    ia, ib = order_index(a["orders"]), order_index(b["orders"])
    keys = sorted(set(ia) & set(ib))
    A = np.concatenate([a["Ex"][:, [ia[k] for k in keys]],
                        a["Ey"][:, [ia[k] for k in keys]]], axis=1)
    B = np.concatenate([b["Ex"][:, [ib[k] for k in keys]],
                        b["Ey"][:, [ib[k] for k in keys]]], axis=1)
    phi = float(np.angle(np.vdot(A, B)))
    return float(np.linalg.norm(np.exp(1j * phi) * A - B)
                 / np.linalg.norm(B)), phi


def sha(x):
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode()
                          + str(a.shape).encode()).hexdigest()[:16]


def k0_of(wl=WL):
    return 2.0 * np.pi / wl
