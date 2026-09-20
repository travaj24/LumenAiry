"""VERIFY-WAVE5-E / E1(b,c): does any library output move across the ping-pong?

(b) FOUR entry points x THREE shapes: run each with the ping-pong ON and OFF in
    ONE process and compare the returned bytes.  Byte-identical => the scoped
    contract holds for every library consumer.
(c) The CALLER spelling the library never uses --
    ``_ifft2(_fft2(E) * np.exp(1j*P))``, right operand a fresh temporary --
    against its named twin, and against ``np.multiply(a, b, out=b)``.
"""
import hashlib
import json
import os
import sys

import numpy as np

import lumenairy
from lumenairy.propagators import asm as _asm, carrier as _car, fft_infra as _fi, fresnel as _fres

NS = [int(v) for v in os.environ.get('VE1_NS', '256,512,1024').split(',')]


def _field(n):
    rng = np.random.default_rng(20260919 + n)
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / 0.2).astype(np.complex128)
    E *= np.exp(1j * 3.0 * rng.standard_normal((n, n)))
    return np.ascontiguousarray(E)


def _dig(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _first(r):
    return r[0] if isinstance(r, tuple) else r


def entry_points(E, n):
    lam, dx = 633e-9, 2e-6
    z = 0.01
    zf = 2.0 * n * dx * dx / lam
    return {
        'angular_spectrum_propagate':
            lambda: _first(_asm.angular_spectrum_propagate(E, z, lam, dx)),
        'angular_spectrum_propagate(bandlimit=False)':
            lambda: _first(_asm.angular_spectrum_propagate(
                E, z, lam, dx, bandlimit=False)),
        'fresnel_propagate':
            lambda: _first(_fres.fresnel_propagate(E, zf, lam, dx)),
        'carrier._exact_envelope_tf_step':
            lambda: _first(_car._exact_envelope_tf_step(E, z, lam, dx, dx)),
    }


def caller_spelling(E, n):
    rng = np.random.default_rng(7 + n)
    P = rng.standard_normal((n, n))
    out = {}
    a = _fi._fft2(E)
    out['owndata'] = bool(np.asarray(a).flags.owndata)
    h_named = np.exp(1j * P)
    named = _fi._ifft2(a * h_named).copy()
    a = _fi._fft2(E)
    elided = _fi._ifft2(a * np.exp(1j * P)).copy()
    a = _fi._fft2(E)
    h2 = np.exp(1j * P)
    np.multiply(a, h2, out=h2)
    explicit = _fi._ifft2(h2).copy()
    d = np.abs(elided - named)
    den = np.abs(named)
    scale = float(den.max()) or 1.0
    out['rel_elided_vs_named'] = float(d.max() / scale)
    out['n_differ_doubles'] = int(np.count_nonzero(
        elided.view(np.float64) != named.view(np.float64)))
    out['n_doubles'] = int(named.size * 2)
    de = np.abs(explicit - named)
    out['rel_explicit_vs_named'] = float(de.max() / scale)
    out['explicit_matches_named'] = bool(_dig(explicit) == _dig(named))
    out['elided_matches_named'] = bool(_dig(elided) == _dig(named))
    return out


def main():
    res = dict(lumenairy_file=lumenairy.__file__, python=sys.version.split()[0],
               platform=sys.platform, numpy=np.__version__,
               fftw_min_size=_fi.FFTW_MIN_SIZE, cells=[], caller=[])
    for n in NS:
        E = _field(n)
        digs = {}
        for mode in (True, False):
            _fi.set_fft_double_buffer(mode)
            for name, fn in entry_points(E, n).items():
                digs.setdefault(name, {})['on' if mode else 'off'] = _dig(fn())
        _fi.set_fft_double_buffer(True)
        for name, d in digs.items():
            res['cells'].append(dict(n=n, entry=name, digest_on=d['on'],
                                     digest_off=d['off'],
                                     identical=d['on'] == d['off']))
        c = caller_spelling(E, n)
        c['n'] = n
        res['caller'].append(c)
    res['n_cells'] = len(res['cells'])
    res['n_identical'] = sum(c['identical'] for c in res['cells'])
    print(json.dumps(res, indent=1))


main()
