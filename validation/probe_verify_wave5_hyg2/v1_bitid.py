"""v1: INDEPENDENT bit-identity fixture set for commit 2796d551 (H2-1).

Built from scratch for this verification -- it shares no key with the author's
179-key ``probe_wave5_hyg2/probe_mft_bitid.py``.

Usage::

    python v1_bitid.py <tree> <out.json>

``<tree>`` is the directory that must own ``lumenairy.__file__``.  Keys are
prefixed:

* ``LEG::`` -- the pre-5.48 API only, NO ``method=`` anywhere.  These MUST be
  bit-identical base vs branch; a single differing key refutes the claim.
* ``NEW::`` -- calls that use ``method=``.  These are EXPECTED to differ
  (the base tree has no such keyword); they are carried so the comparison
  shows the new surface is genuinely new rather than silently absent.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from vlib import Probe, anchor, build_tag                     # noqa: E402

TREE = sys.argv[1]
OUT = sys.argv[2]
anchor(TREE)

from lumenairy.propagators._bluestein import (                # noqa: E402
    _bluestein_2d, _bluestein_centred_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2     # noqa: E402
from lumenairy.propagators.mft import (                       # noqa: E402
    angular_spectrum_propagate_mft, fraunhofer_propagate_mft,
    fresnel_propagate_mft)

WL = 633e-9
P = Probe()


def field(n, dx, w, seed, cdtype=np.complex128):
    rng = np.random.default_rng(seed)
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    spk = 1.0 + 0.17 * (rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w)) * spk).astype(cdtype)


def rand(ny, nx, seed, cdtype=np.complex128):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(cdtype)


PROPS = (('fres', fresnel_propagate_mft),
         ('frau', fraunhofer_propagate_mft),
         ('asm', angular_spectrum_propagate_mft))

# ---------------------------------------------------------------- LEGACY ---
# 3 propagators x (4 shapes x 2 distances) with a per-row zoom factor.
for tag, fn in PROPS:
    for si, (n, dx, w) in enumerate(((48, 9e-6, 70e-6),
                                     (64, 8e-6, 60e-6),
                                     (65, 8e-6, 60e-6),
                                     (96, 5e-6, 44e-6))):
        E = field(n, dx, w, seed=1000 + si)
        for zi, z in enumerate((6e-3, 3.1e-2)):
            nat = WL * z / (n * dx)
            for zoom in (0.37, 1.0, 4.1)[si % 3:si % 3 + 1]:
                P.call(f"LEG::prop.{tag}.n{n}.z{zi}.zoom{zoom}",
                       fn, E, z, WL, dx, nat * zoom, n)

# off-axis centres, anisotropic pitch, complex64, back-propagation
E64 = field(64, 8e-6, 60e-6, seed=7)
nat64 = WL * 2e-2 / (64 * 8e-6)
for tag, fn in PROPS:
    P.call(f"LEG::offaxis.{tag}", fn, E64, 2e-2, WL, 8e-6, nat64, 40,
           centre_out=(11 * nat64, -6.5 * nat64))
    P.call(f"LEG::aniso.{tag}", fn, E64, 2e-2, WL, 8e-6, nat64, 48,
           dy_in=1.1 * 8e-6, dy_out=0.83 * nat64)
    P.call(f"LEG::c64.{tag}", fn, field(64, 8e-6, 60e-6, seed=7,
                                        cdtype=np.complex64),
           2e-2, WL, 8e-6, nat64, 32)
    P.call(f"LEG::backprop.{tag}", fn, E64, -2e-2, WL, 8e-6, nat64, 32)

P.call("LEG::asm.nobandlimit", angular_spectrum_propagate_mft,
       E64, 2e-2, WL, 8e-6, nat64, 64, bandlimit=False)
P.call("LEG::asm.separableflag", angular_spectrum_propagate_mft,
       E64, 2e-2, WL, 8e-6, nat64, 64, _bluestein_separable=True)
P.call("LEG::asm.separableflag.offaxis", angular_spectrum_propagate_mft,
       E64, 2e-2, WL, 8e-6, nat64, 40, _bluestein_separable=True,
       centre_out=(9 * nat64, 3 * nat64))

# ---- the two primitives directly (legacy signature) ----
for si, (ny, nx, my, mx) in enumerate(((16, 16, 8, 8), (32, 24, 16, 20),
                                       (17, 9, 5, 23))):
    E = rand(ny, nx, seed=2000 + si)
    for sign in (-1, +1):
        for sep in (False, True):
            _clear_h_fft_cache()
            P.call(f"LEG::b2d.s{si}.sg{sign}.sep{int(sep)}",
                   _bluestein_2d, E, 1 / 64.0, 1 / 48.0, my, mx,
                   sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2,
                   separable=sep)
    _clear_h_fft_cache()
    P.call(f"LEG::bc2d.s{si}.default",
           _bluestein_centred_2d, E, 1 / 64.0, 1 / 48.0, my, mx,
           sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
    _clear_h_fft_cache()
    P.call(f"LEG::bc2d.s{si}.zero",
           _bluestein_centred_2d, E, 1 / 64.0, 1 / 48.0, my, mx,
           n_centre_in_x=0.0, n_centre_in_y=0.0,
           k_centre_out_x=0.0, k_centre_out_y=0.0,
           sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
    _clear_h_fft_cache()
    P.call(f"LEG::bc2d.s{si}.subpixel",
           _bluestein_centred_2d, E, 1 / 64.0, 1 / 48.0, my, mx,
           n_centre_in_x=nx // 2, n_centre_in_y=ny // 2,
           k_centre_out_x=mx / 2.0 - 1.3, k_centre_out_y=my / 2.0 + 0.4,
           sign=+1, xp=np, fft2=_fft2, ifft2=_ifft2, separable=True)

_clear_h_fft_cache()
P.call("LEG::b2d.c64", _bluestein_2d, rand(16, 16, 5, np.complex64),
       0.01, 0.01, 8, 8, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)

# ---- guard rows: raise / warn (legacy) ----
Eg = rand(8, 8, 9)
P.call("LEG::guard.sign0", _bluestein_2d, Eg, 0.01, 0.01, 4, 4,
       sign=0, xp=np, fft2=_fft2, ifft2=_ifft2)
P.call("LEG::guard.sign2", _bluestein_2d, Eg, 0.01, 0.01, 4, 4,
       sign=2, xp=np, fft2=_fft2, ifft2=_ifft2)
P.call("LEG::guard.nout0", _bluestein_2d, Eg, 0.01, 0.01, 0, 4,
       sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
P.call("LEG::guard.nout_neg_centred", _bluestein_centred_2d, Eg,
       0.01, 0.01, 4, -3, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
_clear_h_fft_cache()
P.call("LEG::guard.phasebudget_warn", _bluestein_2d, rand(24, 24, 77),
       1e17 / 24.0 ** 2, 1e17 / 24.0 ** 2, 12, 12,
       sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
P.call("LEG::guard.prop.zzero", fresnel_propagate_mft,
       E64, 0.0, WL, 8e-6, nat64, 32)

# Twins of the NEW::prop.*.auto rows below, written WITHOUT the keyword, so
# the base tree's answer can be compared to the branch's method='auto' answer.
for tag, fn in PROPS:
    P.call(f"LEG::autotwin.{tag}", fn, E64, 2e-2, WL, 8e-6, nat64, 64)

# ------------------------------------------------------------------- NEW ---
for tag, fn in PROPS:
    P.call(f"NEW::prop.{tag}.auto", fn, E64, 2e-2, WL, 8e-6, nat64, 64,
           method='auto')
    P.call(f"NEW::prop.{tag}.direct", fn, E64, 2e-2, WL, 8e-6, nat64, 64,
           method='direct')
    P.call(f"NEW::prop.{tag}.bogus", fn, E64, 2e-2, WL, 8e-6, nat64, 64,
           method='dense')
for m in ('auto', 'bluestein', 'separable', 'direct', 'Direct', ''):
    _clear_h_fft_cache()
    P.call(f"NEW::b2d.m.{m or 'EMPTY'}", _bluestein_2d, rand(20, 24, 31),
           1 / 48.0, 1 / 48.0, 12, 10, sign=-1, xp=np, fft2=_fft2,
           ifft2=_ifft2, method=m)
    _clear_h_fft_cache()
    P.call(f"NEW::bc2d.m.{m or 'EMPTY'}", _bluestein_centred_2d,
           rand(20, 24, 31), 1 / 48.0, 1 / 48.0, 12, 10, sign=-1, xp=np,
           fft2=_fft2, ifft2=_ifft2, method=m)

P.out['__build__'] = build_tag()
P.write(OUT)
