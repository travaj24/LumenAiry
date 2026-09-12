import os, sys, time, warnings, math, shutil
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE/stor"
os.makedirs(TMP, exist_ok=True)
import numpy as np
from lumenairy.io import storage as S

PROBE = {
    'none': None, 'true': True, 'int': 7, 'float': 1.5, 'str': 'hi',
    'nan': float('nan'), 'inf': float('inf'), 'ninf': float('-inf'),
    'complex': 1+2j, 'bytes': b'\x00\x01ab',
    'tuple': (1, 2.5, 'x'), 'list': [1, 2, 3], 'emptylist': [],
    'hetlist': [1, 'a', None],
    'ndarray_f': np.arange(6.0).reshape(2, 3),
    'ndarray_c': np.array([1+1j, 2-2j], dtype=np.complex64),
    'nested': {'a': {'b': [1, 2]}, 'c': 3},
    'np_scalar': np.float32(2.5),
    'obj': object(),
}
def cmp(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return (type(a) is type(b) and a.dtype == b.dtype
                    and np.array_equal(a, b, equal_nan=np.isrealobj(a)))
        except Exception:
            return False
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    if type(a) is not type(b):
        return False
    try: return a == b
    except Exception: return False

def report(tag, back):
    bad = []
    for k, v in PROBE.items():
        if k == 'obj':
            ok = isinstance(back.get(k), str)
        else:
            ok = cmp(v, back.get(k, '<<MISSING>>'))
        if not ok:
            bad.append(f"{k}: wrote {type(v).__name__} -> got {type(back.get(k)).__name__} {str(back.get(k))[:40]!r}")
    print(f"--- {tag}: {len(PROBE)-len(bad)}/{len(PROBE)} faithful")
    for b in bad: print("      ", b)

E64 = (np.random.rand(64, 64) + 1j*np.random.rand(64, 64)).astype(np.complex64)
E128 = E64.astype(np.complex128)

# HDF5 append_plane metadata fidelity + dtype
p = os.path.join(TMP, 'a.h5')
if os.path.exists(p): os.remove(p)
S.append_plane_h5(p, E64, dx=1e-6, label='L0', metadata=PROBE, preserve_dtype=True)
pl, fm = S.load_planes(p)
print("HDF5 preserve_dtype=True dtype:", pl[0]['field'].dtype, " max|d| =", np.abs(pl[0]['field']-E64).max())
report("HDF5 append_plane metadata", pl[0])

# HDF5 save_field_h5
p2 = os.path.join(TMP, 'f.h5')
S.save_field_h5(p2, E64, dx=1e-6, metadata=PROBE, preserve_dtype=True)
Eb, mb = S.load_field_h5(p2)
print("HDF5 save_field dtype:", Eb.dtype)
report("HDF5 save_field metadata", mb)

# HDF5 sim metadata
S.write_sim_metadata(p2, PROBE)
report("HDF5 sim_metadata", S.read_sim_metadata(p2))

# ---- ZARR ----
try:
    import zarr
    print("zarr version:", zarr.__version__)
    S.set_storage_backend('zarr')
    z = os.path.join(TMP, 'a.zarr')
    if os.path.isdir(z): shutil.rmtree(z)
    try:
        S.append_plane(z, E64, dx=1e-6, label='L0', metadata=PROBE, preserve_dtype=True)
        plz, fmz = S.load_planes(z)
        print("ZARR preserve_dtype=True dtype:", plz[0]['field'].dtype)
        report("ZARR append_plane metadata", plz[0])
    except Exception as e:
        print("ZARR append_plane RAISED:", type(e).__name__, str(e)[:200])
    z2 = os.path.join(TMP, 'm.zarr')
    if os.path.isdir(z2): shutil.rmtree(z2)
    S.write_sim_metadata(z2, PROBE)
    report("ZARR sim_metadata", S.read_sim_metadata(z2))
    S.set_storage_backend('hdf5')
except ImportError as e:
    print("zarr unavailable:", e)

# ---- timings: 4096^2 complex field ----
N = 4096
Ebig = (np.random.rand(N, N) + 1j*np.random.rand(N, N)).astype(np.complex128)
for comp, opts in (('gzip', 4), (None, None)):
    q = os.path.join(TMP, f'big_{comp}.h5')
    if os.path.exists(q): os.remove(q)
    t0 = time.perf_counter()
    S.save_field_h5(q, Ebig, dx=1e-6, compression=comp, compression_opts=opts)
    t1 = time.perf_counter()
    sz = os.path.getsize(q)/2**20
    t2 = time.perf_counter(); _ = S.load_field_h5(q); t3 = time.perf_counter()
    print(f"save_field_h5 {N}^2 c128 compression={comp}: write {t1-t0:.2f}s  size {sz:.1f} MiB "
          f"(raw {Ebig.nbytes/2**20:.0f} MiB)  read {t3-t2:.2f}s")
