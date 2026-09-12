"""Probe 2: does carrier_referenced_reconstruct warn on an undersampled carrier?
   Probe A: near-focus LANDING returns an aliased envelope (scalar path)."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope, carrier_referenced_fit_radius)

wl = 1.31e-6; k = 2*np.pi/wl
N = 256; dx = 2e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x, indexing='xy')
w = 100e-6
env = np.exp(-(X**2+Y**2)/w**2).astype(np.complex128)

print("--- probe 2: reconstruct with a grossly undersampled carrier ---")
for R in (1.0, 1e-2, 1e-3, 2e-4):
    # per-pixel carrier phase step at the beam edge:
    h = k*dx*w/abs(R)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        E = carrier_referenced_reconstruct(env, R, wl, dx)
    print(f"R={R:9.1e} m  edge phase step h={h:8.3f} rad/px (Nyquist=pi)  "
          f"warnings={len(W)}  -> {[str(x.message)[:60] for x in W]}")

print()
print("--- probe A: scalar near-focus LANDING: is the returned envelope aliased? ---")
def gauss(x, w0, z, wl):
    zR = np.pi*w0**2/wl
    X, Y = np.meshgrid(x, x, indexing='xy'); r2 = X**2+Y**2
    w = w0*np.sqrt(1+(z/zR)**2); invR = 0.0 if z == 0 else z/(z**2+zR**2)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(2*np.pi/wl*(z + r2*invR/2.0) - np.arctan2(z, zR)))

w0 = 4e-6; zR = np.pi*w0**2/wl; z0 = -30e-3
w_in = w0*np.sqrt(1+(z0/zR)**2); R0 = z0*(1+(zR/z0)**2)
N = 2048; dx = (2*5.0*w_in)/N
xg = (np.arange(N)-N/2)*dx
E0 = gauss(xg, w0, z0, wl)
env0 = carrier_referenced_envelope(E0, R0, wl, dx)
# land 100 um SHORT of the waist -> inside the bridge zone (delta = 6 zR = 230 um)
z = 30e-3 - 100e-6
env1, R1, dx1 = propagate_carrier_referenced(env0, R0, z, wl, dx)
print(f"landed R_out = {R1:.6e} m (should be {R0+z:.6e}), dx_out={dx1*1e9:.3f} nm, N*dx={N*dx1*1e6:.3f} um")
# measure the envelope's per-pixel phase step
g = np.abs(np.angle(env1[:,1:]*np.conj(env1[:,:-1])))
mag = np.abs(env1); br = mag > 0.05*mag.max()
mk = br[:,1:]&br[:,:-1]
print(f"envelope max |dphi/px| over bright support = {g[mk].max():.3f} rad/px  (Nyquist=pi); "
      f"frac >= pi/2: {(g[mk]>=np.pi/2).mean():.3f}")
# expected carrier fringe on this grid
print(f"expected carrier step at beam edge: k*dx1*w_out/|R1| with w_out ~ {w0*np.sqrt(1+(100e-6/zR)**2)*1e6:.2f} um "
      f"-> {2*np.pi/wl*dx1*w0*np.sqrt(1+(100e-6/zR)**2)/abs(R1):.3f} rad/px")
# Round trip: reconstruct on the SAME grid should be exact
Er = carrier_referenced_reconstruct(env1, R1, wl, dx1)
Ea = gauss((np.arange(N)-N/2)*dx1, w0, z0+z, wl)
print(f"reconstruct vs analytic relL2 = {np.linalg.norm(Er-Ea)/np.linalg.norm(Ea):.3e}  (round trip OK?)")
# but CONTINUE propagating from the returned (env, R) -- this is what a chain does
env2, R2, dx2 = propagate_carrier_referenced(env1, R1, 1e-3, wl, dx1)
E2 = carrier_referenced_reconstruct(env2, R2, wl, dx2)
Ea2 = gauss((np.arange(N)-N/2)*dx2, w0, z0+z+1e-3, wl)
print(f"CONTINUED +1 mm: dx2={dx2*1e6:.4f} um relL2 vs analytic = "
      f"{np.linalg.norm(E2-Ea2)/np.linalg.norm(Ea2):.3e}  peak ratio="
      f"{(np.abs(E2)**2).max()/(np.abs(Ea2)**2).max():.4f}")
# control: do the same continuation from the ANALYTIC field re-enveloped properly
envc = carrier_referenced_envelope(Ea, R0+z, wl, dx1)
envc2, Rc2, dxc2 = propagate_carrier_referenced(envc, R0+z, 1e-3, wl, dx1)
Ec2 = carrier_referenced_reconstruct(envc2, Rc2, wl, dxc2)
print(f"   control (same R, analytic input): relL2={np.linalg.norm(Ec2-Ea2)/np.linalg.norm(Ea2):.3e}")
