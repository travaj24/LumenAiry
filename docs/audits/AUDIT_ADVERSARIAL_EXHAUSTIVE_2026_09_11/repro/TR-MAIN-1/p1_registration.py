"""Probe 1: coarse-lattice registration + upsample fidelity, replicating the
index arithmetic of _lens_traced.py lines 8312-8324, 11274-11456."""
import numpy as np
from scipy.ndimage import map_coordinates

WL = 1.31e-6

def build(N, sub, dx, org=(0.0, 0.0)):
    x = (np.arange(N) - N / 2) * dx
    y = x
    if org[0] or org[1]:
        y = (np.arange(N) - N / 2) * dx + org[1]
        x = x + org[0]
    Xs = np.broadcast_to(x[None, :], (N, N))[::sub, ::sub]
    Ys = np.broadcast_to(y[:, None], (N, N))[::sub, ::sub]
    return x, y, Xs, Ys

def upsample(coarse, N, sub, order):
    idx = np.arange(N, dtype=np.float64) / sub
    coords = np.empty((2, N, N))
    coords[0] = idx[:, None]
    coords[1] = idx[None, :]
    return map_coordinates(coarse, coords, order=order, mode='nearest',
                           prefilter=(order > 1))

print("=== (a) coarse samples coincide with wave-grid pixel centres ===")
for N in (256, 255):
    for sub in (1, 2, 4, 8):
        dx = 2e-6
        x, y, Xs, Ys = build(N, sub, dx)
        # coarse sample (i,j) should equal wave pixel (i*sub, j*sub)
        err = 0.0
        for i in range(Xs.shape[0]):
            err = max(err, abs(Xs[i, 0] - x[0]), abs(Ys[i, 0] - y[i * sub]))
        print(f"  N={N} sub={sub}: Ns={Xs.shape[0]} ceil(N/sub)={-(-N//sub)} "
              f"max |coarse - wave centre| = {err:.3e} m")

print("\n=== (b) symmetry about the optical axis ===")
for N in (256, 255):
    for sub in (1, 2, 3, 4, 8):
        dx = 2e-6
        x, _, Xs, _ = build(N, sub, dx)
        c = Xs[0, :] if False else x[::sub]
        has_axis = np.any(np.abs(c) < 1e-18)
        sym = np.allclose(np.sort(c), -np.sort(-c)[::-1])  # trivially true
        # symmetric means set == -set
        symset = np.allclose(np.sort(c), np.sort(-c))
        print(f"  N={N} sub={sub}: axis sample present={has_axis} "
              f"lattice symmetric about 0={symset}  "
              f"[min={c.min()*1e3:+.4f} mm max={c.max()*1e3:+.4f} mm]")

print("\n=== (c) upsample of a PURE TILT (1 mrad) and PURE DEFOCUS (f=100mm) ===")
k0 = 2 * np.pi / WL
for N in (256, 512):
    for sub in (1, 2, 4, 8):
        dx = 2e-6
        x, y, Xs, Ys = build(N, sub, dx)
        X = np.broadcast_to(x[None, :], (N, N))
        Y = np.broadcast_to(y[:, None], (N, N))
        # tilt OPL
        tilt = 1e-3
        o_c = tilt * Xs
        o_t = tilt * X
        for order in (1, 3):
            up = upsample(o_c, N, sub, order)
            e = np.abs(up - o_t)
            # interior = pixels whose coords stay inside the coarse lattice
            Ns = o_c.shape[0]
            lim = (Ns - 1) * sub
            e_int = e[:lim + 1, :lim + 1].max()
            print(f"  N={N} sub={sub} order={order} TILT   : "
                  f"max err all={e.max()*1e9:9.4f} nm  interior={e_int*1e9:9.4f} nm")
        f = 100e-3
        o_c = (Xs**2 + Ys**2) / (2 * f)
        o_t = (X**2 + Y**2) / (2 * f)
        for order in (1, 3):
            up = upsample(o_c, N, sub, order)
            e = np.abs(up - o_t)
            Ns = o_c.shape[0]
            lim = (Ns - 1) * sub
            e_int = e[:lim + 1, :lim + 1].max()
            print(f"  N={N} sub={sub} order={order} DEFOCUS: "
                  f"max err all={e.max()*1e9:9.4f} nm  interior={e_int*1e9:9.4f} nm"
                  f"   (pred f''(sub dx)^2/8 = {(sub*dx)**2/(8*f)*1e9:.4f} nm)")

print("\n=== (d) origin != 0 ===")
N, sub, dx = 256, 4, 2e-6
org = (37e-6, -11e-6)
x, y, Xs, Ys = build(N, sub, dx, org)
print(f"  coarse x[0]={Xs[0,0]:+.6e} expected {(0-N/2)*dx+org[0]:+.6e}")
print(f"  coarse y[0]={Ys[0,0]:+.6e} expected {(0-N/2)*dx+org[1]:+.6e}")
print(f"  axis (x=0) present in coarse lattice: {np.any(np.abs(x[::sub])<1e-15)}")

print("\n=== (e) how many fine pixels fall OUTSIDE the coarse lattice hull ===")
for N in (256, 512, 1024):
    for sub in (2, 4, 8):
        Ns = -(-N // sub)
        lim = (Ns - 1) * sub          # last fine index covered
        n_out = N - 1 - lim
        print(f"  N={N} sub={sub}: coarse covers fine idx 0..{lim}; "
              f"{n_out} trailing rows/cols are EXTRAPOLATED (mode='nearest')"
              f"  -> {100*(1-((lim+1)/N)**2):.3f}% of pixels")
