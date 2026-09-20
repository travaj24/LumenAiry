"""Round 3 -- the MFT routes' VALUES, archive-to-archive.

Round 3 changes a ``RuntimeWarning``'s text, three docstring paragraphs and a
comment.  A warning text change is a DIAGNOSTIC change, not a byte move, and
the distinction only means something if it is measured: this probe digests the
returned VALUE and the emitted WARNINGS as SEPARATE keys, so the value keys can
be required to be identical archive-to-archive while the warning keys are
allowed to differ exactly where the text was rewritten.

    PYTHONPATH=<tree> python r3_routes_bitid.py <tree> OUT.json

The tree is named twice on purpose -- ``sys.path[0]`` for a script is the
SCRIPT'S directory, so without the explicit anchor a run from a worktree can
silently bind an installed ``lumenairy`` from site-packages.

Sections:

``M`` the three public MFT propagators over shapes, pitches, zooms, off-axis
    centres and dtypes, at every one of the four ``method`` values.
``R`` ``resample_field`` on both legs.
``B`` the two Bluestein primitives directly, both signs, both ``separable``
    settings, all four ``method`` values, and the off-centre convention.
``P`` the phase-budget regime: budgets straddling ``_PHASE_BUDGET_MAX`` on
    both primitives and all four methods -- the keys where the WARNING text
    differs and the VALUE must not.
``G`` the guard rows (refused sign, refused N_out, refused method).
"""
import hashlib
import json
import os
import sys
import warnings

import numpy as np

WL = 633e-9


def anchor(tree):
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    try:
        same = os.path.commonpath([got, want]) == want
    except ValueError:                      # different drives on Windows
        same = False
    if not same:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def _sha(*chunks):
    m = hashlib.sha256()
    for c in chunks:
        m.update(c if isinstance(c, bytes) else str(c).encode('utf-8'))
        m.update(b'\x00')
    return m.hexdigest()


class Probe:
    """``{key: digest}`` with the VALUE and the WARNINGS kept apart."""

    def __init__(self):
        self.value = {}
        self.warn = {}

    def call(self, key, fn, *a, **kw):
        if key in self.value:
            raise SystemExit(f"duplicate key {key!r}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                v = fn(*a, **kw)
                if isinstance(v, tuple):
                    parts = [b'tuple', str(len(v))]
                    for item in v:
                        parts += _value_chunks(item)
                else:
                    parts = _value_chunks(v)
            except BaseException as exc:            # noqa: BLE001 -- recorded
                parts = [b'raised', type(exc).__name__, str(exc)]
        self.value[key] = _sha(*parts)
        self.warn[key] = _sha(*[f"{w.category.__name__}:{w.message}"
                                for w in caught])


def _value_chunks(v):
    if isinstance(v, np.ndarray):
        return [b'nd', str(v.dtype), repr(v.shape),
                np.ascontiguousarray(v).tobytes()]
    if isinstance(v, (int, float, complex, str, bool)) or v is None:
        return [b'sc', type(v).__name__, repr(v)]
    return [b'ob', type(v).__name__, repr(v)]


def _field(n, dx, w, dtype=np.complex128, seed=7):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    rng = np.random.default_rng(seed)
    speckle = 1.0 + 0.12 * (rng.standard_normal((n, n))
                            + 1j * rng.standard_normal((n, n)))
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w)) * speckle).astype(dtype)


METHODS = ('auto', 'bluestein', 'separable', 'direct')


def main(tree, out_path):
    lumenairy = anchor(tree)
    from lumenairy.propagators._bluestein import (
        _PHASE_BUDGET_MAX,
        _bluestein_2d,
        _bluestein_centred_2d,
        _clear_h_fft_cache,
    )
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    from lumenairy.propagators.mft import (
        angular_spectrum_propagate_mft,
        fraunhofer_propagate_mft,
        fresnel_propagate_mft,
        resample_field,
    )

    p = Probe()

    # ----- M: the three public propagators, every method --------------------
    props = (('fresnel', fresnel_propagate_mft),
             ('fraunhofer', fraunhofer_propagate_mft),
             ('asm', angular_spectrum_propagate_mft))
    for n, dx, w in ((64, 8e-6, 60e-6), (65, 8e-6, 60e-6), (96, 6e-6, 90e-6)):
        E = _field(n, dx, w)
        for z in (5e-3, 4e-1):
            for zoom, n_out in ((1.0, n), (0.25, 32), (4.0, 24)):
                dx_out = zoom * WL * z / (n * dx)
                for name, fn in props:
                    for meth in METHODS:
                        _clear_h_fft_cache()
                        p.call(f"M-{name}-n{n}-z{z}-zoom{zoom}-{meth}", fn,
                               E, z, WL, dx, dx_out, n_out, method=meth)
    E = _field(64, 8e-6, 60e-6)
    E32 = _field(64, 8e-6, 60e-6, dtype=np.complex64)
    for meth in METHODS:
        for cx, cy in ((1.2e-4, 0.0), (-3e-5, 7e-5)):
            _clear_h_fft_cache()
            p.call(f"M-centre-{cx}-{cy}-{meth}", fresnel_propagate_mft,
                   E, 2e-2, WL, 8e-6, 2e-6, 48, centre_out=(cx, cy),
                   method=meth)
        _clear_h_fft_cache()
        p.call(f"M-aniso-{meth}", fresnel_propagate_mft, E, 2e-2, WL, 8e-6,
               2e-6, 48, dy_in=6e-6, dy_out=3e-6, method=meth)
        _clear_h_fft_cache()
        p.call(f"M-c64-{meth}", fresnel_propagate_mft, E32, 2e-2, WL, 8e-6,
               2e-6, 48, method=meth)
        _clear_h_fft_cache()
        p.call(f"M-asmback-{meth}", angular_spectrum_propagate_mft, E, -2e-2,
               WL, 8e-6, 2e-6, 48, method=meth)
        _clear_h_fft_cache()
        p.call(f"M-nobl-{meth}", angular_spectrum_propagate_mft, E, 2e-2, WL,
               8e-6, 2e-6, 48, bandlimit=False, method=meth)

    # ----- R: resample_field ------------------------------------------------
    for meth in ('chirpz', 'spline'):
        for nf in (0.5, 1.0, 2.0):
            _clear_h_fft_cache()
            p.call(f"R-{meth}-{nf}", resample_field, E, 8e-6, 8e-6 * nf,
                   method=meth)

    # ----- B: the two primitives -------------------------------------------
    rng = np.random.default_rng(31415)
    for (ny, nx, my, mx) in ((24, 20, 13, 11), (32, 32, 32, 32),
                             (17, 9, 5, 23)):
        Eb = (rng.standard_normal((ny, nx))
              + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        for sign in (-1, +1):
            for meth in METHODS:
                _clear_h_fft_cache()
                p.call(f"B-2d-{ny}x{nx}-{my}x{mx}-s{sign}-{meth}",
                       _bluestein_2d, Eb, 0.011, 0.013, my, mx, sign=sign,
                       xp=np, fft2=_fft2, ifft2=_ifft2, method=meth)
                _clear_h_fft_cache()
                p.call(f"B-c2d-{ny}x{nx}-{my}x{mx}-s{sign}-{meth}",
                       _bluestein_centred_2d, Eb, 0.011, 0.013, my, mx,
                       sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2,
                       method=meth)
                _clear_h_fft_cache()
                p.call(f"B-c2d-off-{ny}x{nx}-{my}x{mx}-s{sign}-{meth}",
                       _bluestein_centred_2d, Eb, 0.011, 0.013, my, mx,
                       n_centre_in_x=0.0, n_centre_in_y=float(ny // 2),
                       k_centre_out_x=mx / 2.0 - 1.7, k_centre_out_y=0.0,
                       sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2,
                       method=meth)
            for sep in (False, True):
                _clear_h_fft_cache()
                p.call(f"B-2d-sep{sep}-{ny}x{nx}-{my}x{mx}-s{sign}",
                       _bluestein_2d, Eb, 0.011, 0.013, my, mx, sign=sign,
                       xp=np, fft2=_fft2, ifft2=_ifft2, separable=sep)

    # ----- P: the phase-budget regime, where the WARNING text moved --------
    for (nin, nout) in ((24, 12), (32, 16)):
        Ep = (rng.standard_normal((nin, nin))
              + 1j * rng.standard_normal((nin, nin))).astype(np.complex128)
        for budget in (1e8, _PHASE_BUDGET_MAX * 0.9, _PHASE_BUDGET_MAX * 2.2,
                       1e12, 1e15):
            alpha = budget / float(nin) ** 2
            for meth in METHODS:
                for prim, fn in (('plain', _bluestein_2d),
                                 ('centred', _bluestein_centred_2d)):
                    _clear_h_fft_cache()
                    p.call(f"P-{prim}-{nin}x{nout}-b{budget:.3e}-{meth}", fn,
                           Ep, alpha, alpha, nout, nout, sign=-1, xp=np,
                           fft2=_fft2, ifft2=_ifft2, method=meth)

    # ----- G: the guard rows ------------------------------------------------
    Eg = _field(32, 8e-6, 60e-6)
    p.call("G-sign-zero", _bluestein_2d, Eg, 0.011, 0.013, 8, 8, sign=0,
           xp=np, fft2=_fft2, ifft2=_ifft2)
    p.call("G-nout-zero", _bluestein_2d, Eg, 0.011, 0.013, 0, 8, sign=-1,
           xp=np, fft2=_fft2, ifft2=_ifft2)
    p.call("G-bad-method", _bluestein_2d, Eg, 0.011, 0.013, 8, 8, sign=-1,
           xp=np, fft2=_fft2, ifft2=_ifft2, method='bogus')
    p.call("G-bad-method-centred", _bluestein_centred_2d, Eg, 0.011, 0.013,
           8, 8, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2, method='bogus')
    p.call("G-public-bad-method", fresnel_propagate_mft, Eg, 1e-2, WL, 8e-6,
           2e-6, 16, method='bogus')
    p.call("G-dxout-zero", fresnel_propagate_mft, Eg, 1e-2, WL, 8e-6, 0.0, 16)
    p.call("G-backward", fresnel_propagate_mft, Eg, -1e-2, WL, 8e-6, 2e-6, 16)

    out = {'lumenairy_file': lumenairy.__file__,
           'lumenairy_version': lumenairy.__version__,
           'platform': sys.platform, 'python': sys.version.split()[0],
           'n': len(p.value), 'value': p.value, 'warn': p.warn}
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"[r3_routes_bitid] {len(p.value)} keys  "
          f"lumenairy={lumenairy.__file__} -> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
