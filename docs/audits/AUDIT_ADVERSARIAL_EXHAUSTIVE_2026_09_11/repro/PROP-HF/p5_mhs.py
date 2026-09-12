"""PROP-HF p5: MHS pipeline semantics -- what it is, what it caches, what
'replay' means, subdomain contracts, thread safety."""
import sys
import os
import tempfile
import threading
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la  # noqa: E402
from lumenairy.propagators.mhs import (  # noqa: E402
    HuygensSurface, MhsSubdomain, MhsPipeline, asm_subdomain,
    aperture_subdomain, gbd_freespace_subdomain, prescription_subdomain)

LAM = 633e-9


def sec(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


N, dx = 64, 5e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x, indexing='xy')
E = np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2).astype(np.complex128)

sec("(1) is MHS a propagator or a composer?  ASM x2 vs one ASM of the sum z")
s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='src')
s1 = HuygensSurface(z=3e-3, Ny=N, Nx=N, dx=dx, label='mid')
s2 = HuygensSurface(z=6e-3, Ny=N, Nx=N, dx=dx, label='out')
pipe = MhsPipeline([asm_subdomain(s0, s1, wavelength=LAM),
                    asm_subdomain(s1, s2, wavelength=LAM)])
hist = pipe.run(E)
E_mhs = hist[-1][1]
E_direct = la.angular_spectrum_propagate(E, 6e-3, LAM, dx)
print(f"  L2(MHS 2-hop, single ASM 6 mm) = "
      f"{np.linalg.norm(E_mhs-E_direct)/np.linalg.norm(E_direct):.3e}"
      f"   (bandlimit=True on both legs)")
pipe_nb = MhsPipeline([asm_subdomain(s0, s1, wavelength=LAM, bandlimit=False),
                       asm_subdomain(s1, s2, wavelength=LAM, bandlimit=False)])
E_nb = pipe_nb.run(E)[-1][1]
E_d_nb = la.angular_spectrum_propagate(E, 6e-3, LAM, dx, bandlimit=False)
print(f"  same with bandlimit=False      = "
      f"{np.linalg.norm(E_nb-E_d_nb)/np.linalg.norm(E_d_nb):.3e}")
p_in = float(np.sum(np.abs(E) ** 2))
print(f"  power  in={p_in:.6f}  mid={float(np.sum(np.abs(hist[1][1])**2)):.6f}"
      f"  out={float(np.sum(np.abs(E_mhs)**2)):.6f}")

sec("(2) run() return-type matrix")
for ri, rr in ((True, False), (False, False), (True, True), (False, True)):
    o = pipe.run(E, return_intermediate=ri, return_result=rr)
    kind = type(o).__name__
    extra = ''
    if kind == 'list':
        extra = f' len={len(o)} elem=({type(o[0][0]).__name__}, ndarray)'
    elif kind == 'PropagationResult':
        extra = f' history={len(o.history)} labels={o.labels()}'
    else:
        extra = f' {o.shape}'
    print(f"  return_intermediate={ri!s:5s} return_result={rr!s:5s} -> {kind}{extra}")

sec("(3) does the pipeline cache anything?  (call twice, count propagator calls)")
calls = {'n': 0}


def counting(E_, in_s, out_s, **kw):
    calls['n'] += 1
    return la.angular_spectrum_propagate(E_, out_s.z - in_s.z, LAM, in_s.dx)


p2 = MhsPipeline([MhsSubdomain(counting, s0, s1), MhsSubdomain(counting, s1, s2)])
p2.run(E); p2.run(E)
print(f"  propagator invocations over 2 identical run() calls: {calls['n']}"
      f"  (4 => no memoisation at all)")

sec("(4) 'replay' = re-read a store; is the store keyed / deduplicated?")
try:
    import h5py  # noqa: F401
    have = True
except ImportError:
    have = False
print(f"  h5py available: {have}")
if have:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'r.h5')
    pipe.run(E, return_intermediate=False, store=path, label_prefix='run')
    r1 = la.replay_run(path, label_prefix='run')
    print(f"  after 1 run: {len(r1.history)} planes, labels={r1.labels()}")
    pipe.run(E * 0.5, return_intermediate=False, store=path, label_prefix='run')
    r2 = la.replay_run(path, label_prefix='run')
    print(f"  after 2 runs into the SAME store+prefix: {len(r2.history)} planes")
    print(f"     labels = {r2.labels()}")
    print(f"     duplicate labels? {len(set(r2.labels())) != len(r2.labels())}")
    print(f"     replay .field == last run's exit field? "
          f"{np.allclose(r2.field, pipe.run(E*0.5, return_intermediate=False))}")
    try:
        os.remove(path); os.rmdir(tmp)
    except OSError:
        pass

sec("(5) thread safety of run()")
res = {}


def worker(i):
    res[i] = pipe.run(E * (i + 1.0), return_intermediate=False)


ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in ths]
[t.join() for t in ths]
base = pipe.run(E, return_intermediate=False)
ok = all(np.allclose(res[i], base * (i + 1.0)) for i in res)
print(f"  4 concurrent run() on one pipeline object give independent results: {ok}")

sec("(6) subdomain builders: contracts")
gsub = gbd_freespace_subdomain(s0, s1, wavelength=LAM, sample_step=8,
                               chunk_beamlets=256)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    g = gsub.propagator(E, s0, s1, **gsub.kwargs)
print(f"  gbd_freespace_subdomain returns {type(g).__name__}"
      f"{' len=' + str(len(g)) if isinstance(g, tuple) else ' ' + str(g.shape)}"
      f"   <- MhsPipeline.run feeds this straight into the next subdomain")
ap = aperture_subdomain(s1, 100e-6)
print(f"  aperture_subdomain out_surface is in_surface: "
      f"{ap.out_surface is ap.in_surface}")
print(f"  _validate accepts [asm(s0->s1), aperture(s1->s1), asm(s1->s2)]: ", end='')
try:
    MhsPipeline([asm_subdomain(s0, s1, wavelength=LAM), ap,
                 asm_subdomain(s1, s2, wavelength=LAM)])
    print("yes")
except ValueError as e:
    print(f"NO -- {e}")

sec("(7) _validate only compares z/Ny/Nx/dx -- not centre")
sa = HuygensSurface(z=3e-3, Ny=N, Nx=N, dx=dx, centre=(0.0, 0.0))
sb = HuygensSurface(z=3e-3, Ny=N, Nx=N, dx=dx, centre=(1e-3, 0.0))
try:
    MhsPipeline([MhsSubdomain(counting, s0, sa), MhsSubdomain(counting, sb, s2)])
    print("  surfaces with DIFFERENT centres accepted as the same plane: YES"
          " (1 mm transverse jump goes unnoticed)")
except ValueError as e:
    print(f"  rejected: {e}")

sec("(8) from_prescription default method vs prescription_subdomain default")
import inspect  # noqa: E402
print("  MhsPipeline.from_prescription method default =",
      inspect.signature(MhsPipeline.from_prescription).parameters['method'].default)
print("  prescription_subdomain     method default =",
      inspect.signature(prescription_subdomain).parameters['method'].default)
