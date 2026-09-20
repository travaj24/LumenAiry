"""H2-1 (item 14) -- the MFT default path, archive-to-archive.

Run as a CHILD process bound to ONE tree:

    python probe_mft_bitid.py <tree> <out.json>

The claim being proved: adding ``method='direct'`` to the three public MFT
propagators and a ``method=`` selector to the two Bluestein primitives moves
NOTHING on the shipped default.  The probe therefore drives every fixture the
default path can reach and digests the WHOLE record -- the returned array's
bytes and dtype and class, plus every warning in emission order -- so a
refactor that reorders two warnings or changes a dtype is caught as surely as
one that moves a bit.

Sections:

``M`` fresnel / fraunhofer / angular-spectrum MFT over a grid of shapes,
    pitches, zoom factors, off-axis centres and dtypes, plus the guard rows
    (a refused ``dx_out``, a back-propagating ``z``, an over-wide window that
    warns about replicas).
``R`` ``resample_field`` on both its legs, since the chirp-Z leg calls the same
    centred primitive.
``B`` the two Bluestein primitives directly, on both settings of ``separable``
    and on the two signs -- the arm where the ``method`` selector was inserted.
``X`` the carrier readout that ships with the separable route ON
    (``carrier_referenced_exact_focus_readout``), because that is the one
    in-library consumer of ``separable=True``.

The ``method='direct'`` arm is deliberately NOT in this probe: it does not
exist on the base tree, so it can have no bit-identity claim.  Its accuracy is
``probe_mft_direct.py``'s ``tolerance`` section.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import hlib  # noqa: E402

import numpy as np  # noqa: E402

hlib.anchor(_TREE)

import lumenairy as la  # noqa: E402
from lumenairy.propagators._bluestein import (  # noqa: E402
    _bluestein_2d, _bluestein_centred_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2  # noqa: E402
from lumenairy.propagators.mft import (  # noqa: E402
    angular_spectrum_propagate_mft, fraunhofer_propagate_mft,
    fresnel_propagate_mft, resample_field)

WL = 633e-9


def _field(n, dx, w, dtype=np.complex128, seed=7):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    rng = np.random.default_rng(seed)
    speckle = 1.0 + 0.12 * (rng.standard_normal((n, n))
                            + 1j * rng.standard_normal((n, n)))
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w)) * speckle).astype(dtype)


def section_mft(p):
    cases = []
    for n, dx, w in ((64, 8e-6, 60e-6), (65, 8e-6, 60e-6),
                     (128, 4e-6, 40e-6), (96, 6e-6, 90e-6)):
        E = _field(n, dx, w)
        for z in (5e-3, 5e-2, 0.4):
            for zoom, n_out in ((1.0, n), (0.25, 64), (4.0, 32)):
                dx_out = zoom * WL * z / (n * dx)
                cases.append((f"n{n}_z{z}_zoom{zoom}", E, z, dx, dx_out,
                              n_out))
    for tag, E, z, dx, dx_out, n_out in cases:
        p.call(f"M-fresnel-{tag}", fresnel_propagate_mft,
               E, z, WL, dx, dx_out, n_out)
        p.call(f"M-fraunhofer-{tag}", fraunhofer_propagate_mft,
               E, z, WL, dx, dx_out, n_out)
        p.call(f"M-asm-{tag}", angular_spectrum_propagate_mft,
               E, z, WL, dx, dx_out, n_out)
    # off-axis centres, anisotropic pitches, dtypes
    E = _field(64, 8e-6, 60e-6)
    E32 = _field(64, 8e-6, 60e-6, dtype=np.complex64)
    for cx, cy in ((0.0, 0.0), (1.2e-4, 0.0), (-3e-5, 7e-5)):
        p.call(f"M-fresnel-centre-{cx}-{cy}", fresnel_propagate_mft,
               E, 2e-2, WL, 8e-6, 2e-6, 48, centre_out=(cx, cy))
        p.call(f"M-asm-centre-{cx}-{cy}", angular_spectrum_propagate_mft,
               E, 2e-2, WL, 8e-6, 2e-6, 48, centre_out=(cx, cy))
    p.call("M-fresnel-aniso", fresnel_propagate_mft, E, 2e-2, WL, 8e-6, 2e-6,
           48, dy_in=6e-6, dy_out=3e-6)
    p.call("M-asm-aniso", angular_spectrum_propagate_mft, E, 2e-2, WL, 8e-6,
           2e-6, 48, dy_in=6e-6, dy_out=3e-6)
    p.call("M-fresnel-c64", fresnel_propagate_mft, E32, 2e-2, WL, 8e-6, 2e-6,
           48)
    p.call("M-asm-c64", angular_spectrum_propagate_mft, E32, 2e-2, WL, 8e-6,
           2e-6, 48)
    p.call("M-asm-back", angular_spectrum_propagate_mft, E, -2e-2, WL, 8e-6,
           2e-6, 48)
    p.call("M-asm-nobandlimit", angular_spectrum_propagate_mft, E, 2e-2, WL,
           8e-6, 2e-6, 48, bandlimit=False)
    p.call("M-asm-separable-private", angular_spectrum_propagate_mft, E, 2e-2,
           WL, 8e-6, 2e-6, 48, _bluestein_separable=True)
    # guard rows: each must RAISE or WARN, and the record carries which
    p.call("M-guard-fresnel-backward", fresnel_propagate_mft, E, -1e-2, WL,
           8e-6, 2e-6, 48)
    p.call("M-guard-fraunhofer-backward", fraunhofer_propagate_mft, E, -1e-2,
           WL, 8e-6, 2e-6, 48)
    p.call("M-guard-dxout-zero", fresnel_propagate_mft, E, 1e-2, WL, 8e-6,
           0.0, 48)
    p.call("M-guard-nout-zero", fresnel_propagate_mft, E, 1e-2, WL, 8e-6,
           2e-6, 0)
    p.call("M-guard-window-replica", fresnel_propagate_mft, E, 1e-2, WL,
           8e-6, 4e-5, 512)


def section_resample(p):
    E = _field(64, 8e-6, 60e-6)
    for meth in ('chirpz', 'spline'):
        for nf in (0.5, 1.0, 2.0):
            p.call(f"R-{meth}-{nf}", resample_field, E, 8e-6, 8e-6 * nf,
                   method=meth)


def section_bluestein(p):
    rng = np.random.default_rng(31415)
    for (ny, nx, my, mx) in ((24, 20, 13, 11), (32, 32, 32, 32),
                             (17, 9, 5, 23)):
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        for sign in (-1, +1):
            for sep in (False, True):
                _clear_h_fft_cache()
                p.call(f"B-2d-{ny}x{nx}-{my}x{mx}-s{sign}-sep{sep}",
                       _bluestein_2d, E, 0.011, 0.013, my, mx,
                       sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2,
                       separable=sep)
                _clear_h_fft_cache()
                p.call(f"B-c2d-{ny}x{nx}-{my}x{mx}-s{sign}-sep{sep}",
                       _bluestein_centred_2d, E, 0.011, 0.013, my, mx,
                       sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2,
                       separable=sep)
                _clear_h_fft_cache()
                p.call(f"B-c2d-off-{ny}x{nx}-{my}x{mx}-s{sign}-sep{sep}",
                       _bluestein_centred_2d, E, 0.011, 0.013, my, mx,
                       n_centre_in_x=0.0, n_centre_in_y=float(ny // 2),
                       k_centre_out_x=mx / 2.0 - 1.7,
                       k_centre_out_y=0.0,
                       sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2,
                       separable=sep)
        # guard rows
        p.call(f"B-guard-sign-{ny}x{nx}", _bluestein_2d, E, 0.011, 0.013,
               my, mx, sign=0, xp=np, fft2=_fft2, ifft2=_ifft2)
        p.call(f"B-guard-nout-{ny}x{nx}", _bluestein_2d, E, 0.011, 0.013,
               0, mx, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
        p.call(f"B-guard-phase-{ny}x{nx}", _bluestein_2d, E, 1e13, 1e13,
               my, mx, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)


def section_exact_readout(p):
    big = _field(128, 4e-6, 120e-6)
    p.call("X-exact-focus-readout",
           la.carrier_referenced_exact_focus_readout, big, -0.05, 4.9e-2,
           wavelength=WL, dx=4e-6, dx_out=4e-6 * 40, N_out=256,
           on_readout_window='warn', on_replica='warn',
           on_n_fine_cap='warn', n_fine_cap=64)
    p.call("X-focus-readout", la.carrier_referenced_focus_readout,
           _field(64, 8e-6, 60e-6), -0.05, 5e-3, wavelength=WL, dx=8e-6,
           dx_out=8e-6, N_out=32)


def main():
    out_path = sys.argv[2]
    p = hlib.Probe()
    section_mft(p)
    section_resample(p)
    section_bluestein(p)
    section_exact_readout(p)
    p.write(out_path)
    print(f"[probe_mft_bitid] {len(p.out)} keys, build={hlib.build_tag()}")


if __name__ == '__main__':
    main()
