"""WAVE5-E item E1 -- the BENEFIT side of the fft_infra ping-pong decision.

Companion to ``probe_e1_copy_cost.py`` (the cost side).  Three readings:

  1. **Ownership.**  What ``_fft2``/``_ifft2`` hand back in each mode --
     ``owndata`` and whether ``base`` is the plan workspace.  This is the only
     thing the ping-pong actually changes about the object.

  2. **The library's own outputs, mode against mode.**  Every in-library site
     that multiplies a dispatcher result spells it ``_fft2(...) * H`` with a
     NAMED right operand (``asm.py:919/922/1391``, ``carrier.py:1401/7126``,
     ``fresnel.py:216``), so neither operand is an unreferenced temporary in
     ping-pong mode and the LEFT one is in single-buffer mode.  VERIFY-B14's
     ``probe_v7d_elision`` measured ``left == named`` on both builds, so the
     prediction is that the library's own outputs are byte-identical across
     the switch.  Measured here entry point by entry point.

  3. **The caller-visible A/B**, i.e. the spelling that DOES move: a caller
     writing ``_ifft2(_fft2(E) * np.exp(1j*P))`` -- right operand a fresh
     temporary -- gets the right-elided product in ping-pong mode and the
     both-elided (== named) product in single-buffer mode.  That is the
     divergence remedy (a) would remove, and its size is measured here.

Usage:  python probe_e1_decision.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


def _md5(a):
    if isinstance(a, tuple):
        a = a[0]
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'e1_decision.json'
    import lumenairy as la
    from lumenairy.propagators import (
        carrier as C,  # noqa: N806
        fft_infra as fi,
    )

    res = {'lumenairy_file': la.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'platform': sys.platform,
           'pyfftw_available': bool(fi.PYFFTW_AVAILABLE),
           'use_pyfftw': bool(fi.USE_PYFFTW),
           'fftw_min_size': int(fi.FFTW_MIN_SIZE),
           'ownership': [], 'entry_points': [], 'caller_ab': []}
    print('lumenairy.__file__ =', la.__file__, flush=True)

    wl, dx, z = 1.31e-6, 1.0e-6, 2.0e-3

    # ---- 1. ownership -----------------------------------------------------
    for n in (128, 256, 512, 1024):
        rng = np.random.default_rng(3)
        E = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        row = {'n': n}
        for mode, dbl in (('pingpong', True), ('copy', False)):
            fi.set_fft_double_buffer(dbl)
            la.clear_asm_caches()
            out = fi._fft2(E)
            row['%s_owndata' % mode] = bool(out.flags.owndata)
            row['%s_has_base' % mode] = out.base is not None
        res['ownership'].append(row)
        print('n=%-5d ownership: pingpong owndata=%-5s  copy owndata=%-5s'
              % (n, row['pingpong_owndata'], row['copy_owndata']), flush=True)

    # ---- 2. the library's own entry points, mode against mode -------------
    for n in (256, 512, 1024):
        rng = np.random.default_rng(11)
        E = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        entries = {
            'angular_spectrum_propagate':
                lambda: la.angular_spectrum_propagate(E, z, wl, dx),
            'angular_spectrum_propagate_bl_off':
                lambda: la.angular_spectrum_propagate(E, z, wl, dx,
                                                      bandlimit=False),
            'fresnel_propagate':
                lambda: la.fresnel_propagate(E, z, wl, dx),
            'carrier_exact_envelope_tf_step':
                lambda: C._exact_envelope_tf_step(E, 5e-3, 1.55e-6, 2e-6,
                                                  2e-6, tilt=(0.03, -0.02)),
        }
        for name, fn in entries.items():
            md5s = {}
            for mode, dbl in (('pingpong', True), ('copy', False)):
                fi.set_fft_double_buffer(dbl)
                la.clear_asm_caches()
                md5s[mode] = _md5(fn())
            same = md5s['pingpong'] == md5s['copy']
            res['entry_points'].append(
                {'n': n, 'entry_point': name, 'md5_pingpong': md5s['pingpong'],
                 'md5_copy': md5s['copy'], 'byte_identical': bool(same)})
            print('n=%-5d %-34s byte-identical across the switch: %s'
                  % (n, name, same), flush=True)

    # ---- 3. the caller spelling that DOES move ----------------------------
    for n in (128, 256, 512, 1024):
        rng = np.random.default_rng(7)
        E = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        P = rng.standard_normal((n, n)) * 1e3
        outs = {}
        for mode, dbl in (('pingpong', True), ('copy', False)):
            fi.set_fft_double_buffer(dbl)
            la.clear_asm_caches()
            fi._fft2(E)                     # build the plan first
            outs[mode] = np.array(
                fi._ifft2(fi._fft2(E) * np.exp(1j * P)), copy=True)
        d = float(np.max(np.abs(outs['pingpong'] - outs['copy'])))
        sc = float(np.max(np.abs(outs['copy']))) or 1.0
        ndiff = int(np.count_nonzero(
            outs['pingpong'].view(np.float64) != outs['copy'].view(np.float64)))
        res['caller_ab'].append(
            {'n': n, 'absmax': d, 'rel': d / sc, 'ndiff': ndiff,
             'nvals': int(outs['copy'].size * 2),
             'byte_identical': bool(np.array_equal(outs['pingpong'],
                                                   outs['copy']))})
        print('n=%-5d caller spelling _ifft2(_fft2(E)*np.exp(1j*P)): '
              'identical=%-5s rel=%.3e ndiff=%d/%d'
              % (n, res['caller_ab'][-1]['byte_identical'], d / sc, ndiff,
                 int(outs['copy'].size * 2)), flush=True)

    res['library_outputs_all_byte_identical'] = all(
        r['byte_identical'] for r in res['entry_points'])
    res['caller_spelling_moves'] = any(
        not r['byte_identical'] for r in res['caller_ab'])
    print('LIBRARY OUTPUTS BYTE-IDENTICAL ACROSS THE SWITCH:',
          res['library_outputs_all_byte_identical'], flush=True)
    print('CALLER SPELLING MOVES ON THIS BUILD:',
          res['caller_spelling_moves'], flush=True)

    fi.set_fft_double_buffer(True)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
