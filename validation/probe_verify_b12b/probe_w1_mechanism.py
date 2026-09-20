"""VERIFY-WP-B12b probe W1 -- the defect and the repair at the BEAMLET level,
run inside whichever tree the interpreter resolves.

The arm (``pre`` = the 1218b24f library that still carries the in-line
conic-sag copy, ``post`` = the branch under test) is DETECTED from the library
source at run time, never passed on the command line, and written into the
JSON together with the resolved ``lumenairy.__file__``.

What it measures, with ``z_image = 0`` so the image-side leg is the identity
and every reading is the reference plane and nothing else:

* the returned base-ray POSITION against my own 3-D tracer's exit-VERTEX
  state.  ``post`` must reproduce it to the trace's own floor; ``pre`` misses
  it by ``(sag_true - sag_inline) * u``;
* the returned base-ray PHASE, pre against post, against the predicted
  ``k0 * (n_exit * sign(N) * sag_true - sag_inline) * sec`` -- the whole
  optical-path defect, including the MIRROR row where ``sign(N) = -1`` turns
  the correction into its own double;
* the four separate failure modes, each on the fixture that isolates it.

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import inspect
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vb12b_common import (  # noqa: E402
    assert_tree,
    dump,
    env_block,
    fixtures,
    gbd_surfaces,
    inline_conic_sag,
    to_vertex,
    trace3d,
)

_NR, _NAZ = 21, 8


def _fan(semi):
    r = np.linspace(semi / (2 * _NR), semi * 0.98, _NR)
    az = (np.arange(_NAZ) + 0.5) * (np.pi / _NAZ)
    R, A = np.meshgrid(r, az, indexing='ij')
    return (R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()


def _bundle(h, y, lam, w0b):
    from lumenairy.propagators.gbd import BeamletBundle
    n = h.size
    z_R = np.pi * w0b ** 2 / lam
    return BeamletBundle(
        positions=np.stack([h, y, np.zeros_like(h)], axis=-1),
        directions=np.stack([np.zeros_like(h), np.zeros_like(h),
                             np.ones_like(h)], axis=-1),
        Q=np.full(n, -1j / z_R, dtype=np.complex128),
        amplitude=np.ones(n, dtype=np.complex128),
        waist0=np.full(n, float(w0b)))


def main():
    assert_tree()
    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets as F,
    )
    src = inspect.getsource(F)
    arm = 'pre' if '_Rl = float(' in src else 'post'
    print(f'ARM DETECTED FROM THE LIBRARY: {arm}')

    out = dict(env=env_block(), arm=arm, rows=[])
    for key, fx in fixtures().items():
        presc = fx.prescription()
        surfs = gbd_surfaces(presc)
        h, y = _fan(fx.semi)
        w0b = 4.0 * fx.dx
        b = _bundle(h, y, fx.lam, w0b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = F(b, presc, fx.lam, z_image=0.0)
        pos = np.asarray(res.positions)
        amp = np.asarray(res.amplitude)
        n_ret = pos.shape[0]

        # -- my own 3-D trace of the SAME base rays -------------------------
        osurfs = fx.oracle_surfaces()
        z0 = np.zeros_like(h)
        st = trace3d(h, y, z0, z0, z0, np.ones_like(h), z0, osurfs)
        mv = to_vertex(st, fx.n_exit(), osurfs[-1]['zv'])
        ux_t = st['L'] / st['N']
        uy_t = st['M'] / st['N']
        sec = np.sqrt(1.0 + ux_t ** 2 + uy_t ** 2)
        sag_true = st['z'] - osurfs[-1]['zv']
        sag_inline = inline_conic_sag(surfs[-1], st['x'], st['y'])
        nz = float(np.sign(np.nanmedian(st['N'])))
        n_exit = fx.n_exit()
        # the optical-path error the in-line copy makes, from MY tracer
        opl_err = (n_exit * nz * sag_true - sag_inline) * sec
        pos_err_pred = np.hypot((sag_true - sag_inline) * ux_t,
                                (sag_true - sag_inline) * uy_t)

        row = dict(
            key=key, note=fx.note, arm=arm, n_rays=int(h.size),
            n_returned=int(n_ret), lam=fx.lam, semi=fx.semi,
            n_exit=n_exit, exit_sign=nz,
            sag_true_max=float(np.nanmax(np.abs(sag_true))),
            sag_true_waves=float(np.nanmax(np.abs(sag_true)) / fx.lam),
            sag_inline_err_m=float(np.nanmax(np.abs(sag_true - sag_inline))),
            sag_inline_err_waves=float(
                np.nanmax(np.abs(sag_true - sag_inline)) / fx.lam),
            sag_inline_err_frac=float(
                np.nanmax(np.abs(sag_true - sag_inline))
                / max(float(np.nanmax(np.abs(sag_true))), 1e-300)),
            inline_opl_err_waves=float(np.nanmax(np.abs(opl_err)) / fx.lam),
            inline_opl_err_m=float(np.nanmax(np.abs(opl_err))),
            pred_pos_err_m=float(np.nanmax(np.abs(pos_err_pred))),
        )
        if n_ret == h.size:
            dxv = pos[:, 0] - mv['x']
            dyv = pos[:, 1] - mv['y']
            row['pos_vs_oracle_vertex_x'] = float(np.nanmax(np.abs(dxv)))
            row['pos_vs_oracle_vertex_y'] = float(np.nanmax(np.abs(dyv)))
            row['pos_vs_oracle_max'] = float(np.nanmax(np.hypot(dxv, dyv)))
            row['amp_re'] = amp.real.tolist()
            row['amp_im'] = amp.imag.tolist()
            row['pos_x'] = pos[:, 0].tolist()
            row['pos_y'] = pos[:, 1].tolist()
            row['pred_opl_err'] = opl_err.tolist()
            row['sec'] = sec.tolist()
            row['ux'] = ux_t.tolist()
            row['r_in'] = np.hypot(h, y).tolist()
        else:
            row['pos_vs_oracle_max'] = None
            print(f'  !! {key}: {n_ret} of {h.size} beamlets returned')
        out['rows'].append(row)
        print(f"{key:15s} sag {row['sag_true_waves']:7.3f} wv | inline sag err "
              f"{row['sag_inline_err_waves']:8.4f} wv "
              f"({row['sag_inline_err_frac']:.3f} of sag) | inline OPL err "
              f"{row['inline_opl_err_waves']:8.4f} wv | pos vs my oracle "
              f"{row.get('pos_vs_oracle_max')}")

    dump(out, f'probe_w1_mechanism_{arm}')


if __name__ == '__main__':
    main()
