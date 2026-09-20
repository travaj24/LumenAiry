"""v3 -- item (6): is a mutation arm LIVE?

Usage: v3_probe_mut_evidence.py <out.json> <tree-root>

Runs the author's own fixture through the tree named on the command line and
digests the returned complex fields, so "the test file still passes" can be
told apart from "the mutation changed nothing".  ``vlib.anchor`` is pointed at
the tree under test (a mutated COPY in the scratchpad, or the worktree).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

ROOT = os.path.abspath(sys.argv[2])
sys.path.insert(0, ROOT)
L = vlib.anchor(os.path.join(ROOT, 'lumenairy'))

from lumenairy.propagators.carrier import (  # noqa: E402
    _collins_transport, propagate_carrier_referenced)

W0, THETA, F, N, DX = 15.915e-6, 20.0e-3, 20.0e-3, 512, 8e-6
LAM = float(np.pi * W0 * THETA)
ZR = float(np.pi * W0 ** 2 / LAM)
K = 2.0 * np.pi / LAM


def grid(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


q_in = complex(-F, -ZR)
R_IN = 1.0 / float(np.real(1.0 / q_in))
W_IN = float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q_in))))
g = grid(N, DX)
ENV = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / W_IN ** 2)).astype(
    np.complex128)


def q_field(xo, z):
    q2 = q_in + z
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    return (np.exp(1j * K * z) / (1.0 + z / q_in)
            * np.exp(1j * K * r2 / (2.0 * q2)))


def pitch_for(w, R):
    return float(min(6.0 * w / N, w / 8.0,
                     (LAM * abs(R) / (4.0 * w)) if np.isfinite(R) else w / 8.0))


out = {'build': vlib.build_tag(), 'tree': ROOT, 'rows': {}}
for d in (1e-6, 1e-4, 5e-3):
    z = F - d
    q_out = q_in + z
    w_out = float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q_out))))
    re = float(np.real(1.0 / q_out))
    R_out = float('inf') if re == 0.0 else 1.0 / re
    dxo = pitch_for(w_out, R_out)
    row = {}
    fields = {}
    for kern in ('fresnel', 'auto', 'exact'):
        st = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                E = np.asarray(_collins_transport(
                    ENV, R_IN, z, LAM, DX, DX, dx_out=dxo, dy_out=dxo,
                    N_out_x=N, N_out_y=N, R_ref=float('inf'),
                    gap_kernel=kern, on_collins_sampling='ignore',
                    stats_out=st))
            fields[kern] = E
            xo = grid(N, dxo)
            T = q_field(xo, z)
            row[kern + '_rel'] = float(np.linalg.norm(E - T)
                                       / np.linalg.norm(T))
            row[kern + '_digest'] = vlib.digest(E)
            row[kern + '_kernel_resolved'] = st.get('kernel')
            row[kern + '_k4'] = float(st.get('k4', np.nan))
        except Exception as exc:                                # noqa: BLE001
            row[kern] = 'RAISED %s: %s' % (type(exc).__name__, str(exc)[:140])
    if 'exact' in fields and 'fresnel' in fields:
        df = fields['exact'] - fields['fresnel']
        row['exact_minus_fresnel_relL2'] = float(
            np.linalg.norm(df) / np.linalg.norm(fields['fresnel']))
        # The SIGN of the correction: the mean phase of exact/fresnel,
        # intensity-weighted.  A sign flip changes this and nothing else.
        w = np.abs(fields['fresnel']) ** 2
        ph = np.angle(fields['exact'] / fields['fresnel'])
        row['mean_signed_correction_phase'] = float((w * ph).sum() / w.sum())
    # the public entry point too
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = propagate_carrier_referenced(ENV, R_IN, z, LAM, DX,
                                         gap_kernel='fresnel')
    row['sziklas_env_digest'] = vlib.digest(np.asarray(p.env))
    row['sziklas_R'] = float(p.R)
    row['sziklas_dx'] = float(p.dx)
    out['rows']['%g' % d] = row

vlib.write_json(out, sys.argv[1])
