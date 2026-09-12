"""PROP-HF p7: mhs.prescription_subdomain(method='hf'/'hfpi'/'gbd') --
mhs.py:534 advertises these as the non-resampling alternative to maslov.
Does the field that comes back out of _prop survive as a FIELD?
Also: what the dispatcher does with method='hf' + output_grid."""
import sys
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la  # noqa: E402
from lumenairy.propagators.dispatch import propagate  # noqa: E402
from lumenairy.propagators.mhs import (  # noqa: E402
    HuygensSurface, MhsPipeline, prescription_subdomain, asm_subdomain)

LAM = 633e-9
N, dx = 32, 20e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x, indexing='xy')
E = np.exp(-(X ** 2 + Y ** 2) / (200e-6) ** 2).astype(np.complex128)
presc = la.make_singlet(R1=50e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
                        aperture=10e-3)

print("=" * 74)
print("(1) dispatcher: propagate(method='hf') return shapes")
print("=" * 74)
for kw, lab in (
        ({}, "no grid kwargs"),
        ({'output_grid': {'N': N, 'dx': dx}}, "output_grid same grid"),
        ({'output_grid': {'N': N, 'dx': dx}, 'output_dx': dx},
         "output_grid + output_dx (what mhs sends)"),
        ({'output_grid': {'N': 16, 'dx': 2 * dx}, 'output_dx': 2 * dx},
         "genuine regrid")):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            o = propagate(E, z=1e-3, wavelength=LAM, dx=dx, method='hf',
                          return_result=False, **kw)
        d = (f"tuple len={len(o)} -> ({type(o[0]).__name__}{getattr(o[0],'shape','')},"
             f" {o[1]!r})" if isinstance(o, tuple)
             else f"{type(o).__name__} {getattr(o, 'shape', '')}")
        print(f"  {lab:42s} -> {d}")
    except Exception as e:
        print(f"  {lab:42s} -> {type(e).__name__}: {str(e)[:80]}")

print()
print("=" * 74)
print("(2) mhs.prescription_subdomain: the advertised gbd/hfpi/hf alternatives")
print("=" * 74)
s_in = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='in')
s_out = HuygensSurface(z=4e-3, Ny=N, Nx=N, dx=dx, label='out')
for method, extra in (('maslov', {}), ('gbd', dict(sample_step=8, chunk_beamlets=256)),
                      ('hf', {}), ('hfpi', {})):
    try:
        sub = prescription_subdomain(s_in, s_out, presc, wavelength=LAM,
                                     method=method, **extra)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = sub.propagator(E, s_in, s_out, **sub.kwargs)
        d = (f"TUPLE len={len(out)} (pipeline would feed this tuple to the next "
             f"subdomain)" if isinstance(out, tuple)
             else f"{type(out).__name__} {getattr(out, 'shape', '')}")
        print(f"  method={method:8s} -> {d}")
    except Exception as e:
        print(f"  method={method:8s} -> {type(e).__name__}: {str(e)[:110]}")

print()
print("=" * 74)
print("(3) a 2-subdomain pipeline that actually chains one of them")
print("=" * 74)
for method in ('maslov', 'gbd', 'hf'):
    try:
        s2 = HuygensSurface(z=10e-3, Ny=N, Nx=N, dx=dx, label='far')
        extra = dict(sample_step=8, chunk_beamlets=256) if method == 'gbd' else {}
        pipe = MhsPipeline([
            prescription_subdomain(s_in, s_out, presc, wavelength=LAM,
                                   method=method, **extra),
            asm_subdomain(s_out, s2, wavelength=LAM)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fin = pipe.run(E, return_intermediate=False)
        print(f"  chain with method={method:8s} -> OK, {type(fin).__name__} "
              f"{getattr(fin,'shape','')}")
    except Exception as e:
        print(f"  chain with method={method:8s} -> {type(e).__name__}: "
              f"{str(e)[:130]}")
