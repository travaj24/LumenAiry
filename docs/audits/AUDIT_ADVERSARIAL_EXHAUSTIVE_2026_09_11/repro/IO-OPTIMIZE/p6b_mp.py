import os, sys, shutil
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE/stor"
os.makedirs(TMP, exist_ok=True)
import numpy as np

def worker(args):
    path, wid, n = args
    sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
    from lumenairy.io import storage as S
    E = np.full((8, 8), wid + 0j, dtype=np.complex128)
    ok = 0
    for k in range(n):
        try:
            S.append_plane_h5(path, E, dx=1e-6, label=f'w{wid}-{k}',
                              compression=None, lock_timeout=120.0)
            ok += 1
        except Exception as e:
            return (wid, ok, f"{type(e).__name__}: {e}")
    return (wid, ok, None)

if __name__ == '__main__':
    import multiprocessing as mp
    from lumenairy.io import storage as S
    p = os.path.join(TMP, 'mp.h5')
    for f in (p, p + '.lock'):
        if os.path.exists(f):
            try: os.remove(f)
            except OSError: pass
    NW, NA = 2, 50
    with mp.Pool(NW) as pool:
        res = pool.map(worker, [(p, w, NA) for w in range(NW)])
    print("workers:", res)
    planes, meta = S.list_planes(p)
    labels = [pl.get('label') for pl in planes]
    print(f"n_planes attr-driven list: {len(planes)}  expected {NW*NA}")
    print("unique labels:", len(set(labels)), " duplicates:", len(labels)-len(set(labels)))
    missing = set(f'w{w}-{k}' for w in range(NW) for k in range(NA)) - set(labels)
    print("MISSING labels:", len(missing), sorted(missing)[:10])
