"""v1: the phase-budget warning defect -- confirm or refute, and derive the
threshold the warning SHOULD carry.

The guard under test is ``lumenairy/propagators/_bluestein.py:542-555``::

    phase_budget = float(alpha_max) * float(N_max) ** 2
    if phase_budget > 1e15:
        warnings.warn(...)

Sweep the budget, measure BOTH routes against an exact-summation reference,
and record whether a warning fired.

Usage::  python v1_phase.py <tree> <out.json>
"""
from __future__ import annotations

import math
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                # noqa: E402

TREE = sys.argv[1]
OUT = sys.argv[2]
anchor(TREE)

from lumenairy.propagators._bluestein import (                # noqa: E402
    _bluestein_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2     # noqa: E402


def rand(ny, nx, seed=77):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def exact_reference(E, alpha, my, mx, sign):
    """Correctly-rounded summation (``math.fsum``) of the same terms.

    The phase is reduced modulo one turn EXACTLY (``t - rint(t)`` with
    ``|t| <= 2**52``), so the reference has no phase-precision loss of its
    own and the sweep measures the routes alone.
    """
    ny, nx = E.shape
    n_x = np.arange(nx, dtype=np.float64)
    n_y = np.arange(ny, dtype=np.float64)
    out = np.empty((my, mx), np.complex128)
    for ky in range(my):
        ty = alpha * float(ky) * n_y
        wy = np.exp(1j * sign * 2.0 * np.pi * (ty - np.rint(ty)))
        for kx in range(mx):
            tx = alpha * float(kx) * n_x
            wx = np.exp(1j * sign * 2.0 * np.pi * (tx - np.rint(tx)))
            t = (E * (wy[:, None] * wx[None, :])).ravel()
            out[ky, kx] = complex(math.fsum(t.real.tolist()),
                                  math.fsum(t.imag.tolist()))
    return out


BUDGETS = [1e6, 3.16e6, 1e7, 3.16e7, 1e8, 3.16e8, 1e9, 3.16e9, 1e10,
           3.16e10, 1e11, 3.16e11, 1e12, 3.16e12, 1e13, 3.16e13, 1e14,
           3.16e14, 1e15, 3.16e15, 1e16, 3.16e16, 1e17]
GEOMS = ((24, 12), (48, 24))
rows = []
for (N, M) in GEOMS:
    E = rand(N, N, seed=77)
    for budget in BUDGETS:
        alpha = budget / float(N) ** 2
        ref = exact_reference(E, alpha, M, M, -1)
        rn = np.linalg.norm(ref)
        out = {'N': N, 'M': M, 'budget': budget, 'alpha': alpha}
        for name, kw in (('chirpz', dict(separable=False, method='auto')),
                         ('separable', dict(separable=True, method='auto')),
                         ('dense', dict(method='direct'))):
            _clear_h_fft_cache()
            with warnings.catch_warnings(record=True) as cw:
                warnings.simplefilter('always')
                F = _bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                                  fft2=_fft2, ifft2=_ifft2, **kw)
            out[f'{name}_relL2'] = float(np.linalg.norm(F - ref) / rn)
            out[f'{name}_warned'] = bool(
                any('chirp phase argument' in str(w.message) for w in cw))
        rows.append(out)
        print(f"N={N} budget={budget:.2e} chirp={out['chirpz_relL2']:.3e} "
              f"warned={out['chirpz_warned']} "
              f"dense={out['dense_relL2']:.3e}", file=sys.stderr)

# Where does the chirp route's error first become visible?
thr = {}
for (N, M) in GEOMS:
    sub = [r for r in rows if r['N'] == N]
    for label, lim in (('1e-12', 1e-12), ('1e-10', 1e-10), ('1e-8', 1e-8),
                       ('1e-6', 1e-6), ('1e-4', 1e-4), ('1e-2', 1e-2)):
        hit = next((r['budget'] for r in sub if r['chirpz_relL2'] > lim), None)
        thr[f'N{N}.first_budget_with_relL2_gt_{label}'] = hit
    thr[f'N{N}.first_budget_that_warns'] = next(
        (r['budget'] for r in sub if r['chirpz_warned']), None)
    thr[f'N{N}.relL2_at_1e12'] = next(
        (r['chirpz_relL2'] for r in sub if r['budget'] == 1e12), None)
    thr[f'N{N}.relL2_at_1e15'] = next(
        (r['chirpz_relL2'] for r in sub if r['budget'] == 1e15), None)
    thr[f'N{N}.relL2_at_1e17'] = next(
        (r['chirpz_relL2'] for r in sub if r['budget'] == 1e17), None)

write_json(dict(build=build_tag(), tree=TREE, rows=rows, thresholds=thr,
                guard_source='lumenairy/propagators/_bluestein.py:542-555'),
           OUT)
