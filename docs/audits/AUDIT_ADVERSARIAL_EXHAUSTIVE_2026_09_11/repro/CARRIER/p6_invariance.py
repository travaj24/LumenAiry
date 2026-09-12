"""Probe 6a: CARRIER-CHOICE INVARIANCE of the readout.

The same physical field, referenced to two different carriers, must produce the
same field at the target plane.  A common output grid is supplied by the
Bluestein focus readout.
"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    carrier_referenced_focus_readout, carrier_referenced_envelope,
    carrier_referenced_reconstruct, _default_focus_standoff)

wl = 1.31e-6; k = 2*np.pi/wl
N = 1024; w_in = 1.0e-3; NA = 0.05
R0 = -w_in/NA                    # converging, focus at 20 mm
ext = 4.0
dx = 2*ext*w_in/N
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x, indexing='xy'); r2 = X**2+Y**2
E_phys = np.exp(-r2/w_in**2)*np.exp(1j*k*r2/(2*R0))     # the physical field
z = -R0
w0 = wl*abs(R0)/(np.pi*w_in); dx_out = w0/8.0; N_out = 96
print(f"R0={R0*1e3:.3f}mm z={z*1e3:.3f}mm w0={w0*1e6:.4f}um dx={dx*1e6:.4f}um "
      f"carrier step={k*dx*w_in/abs(R0):.3f} rad/px")

ref = None
for R in (R0, 0.98*R0, 0.90*R0, 1.15*R0):
    env = carrier_referenced_envelope(E_phys, R, wl, dx)
    so = _default_focus_standoff(env, R, z, wl, dx)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        F = carrier_referenced_focus_readout(env, R, z, wl, dx,
                                             dx_out=dx_out, N_out=N_out,
                                             on_replica='ignore')
        nw = len(W)
    if ref is None:
        ref = F; R_ref = R
        print(f"  R={R*1e3:9.4f}mm  standoff={so*1e6:9.3f}um  (reference)  "
              f"peak={np.abs(F).max()**2:.6g}")
    else:
        pist = np.angle(np.vdot(ref, F))
        rel = np.linalg.norm(F*np.exp(-1j*pist)-ref)/np.linalg.norm(ref)
        relraw = np.linalg.norm(F-ref)/np.linalg.norm(ref)
        print(f"  R={R*1e3:9.4f}mm  standoff={so*1e6:9.3f}um  relL2 vs ref={relraw:.3e} "
              f"(piston-free {rel:.3e}, piston {pist:+.4e} rad)  peak ratio="
              f"{(np.abs(F).max()/np.abs(ref).max())**2:.6f}  nwarn={nw}")

print("\n=== 6b: standoff-choice invariance (same carrier, different legs) ===")
env = carrier_referenced_envelope(E_phys, R0, wl, dx)
ref = None
for so in (None, 200e-6, 500e-6, 1e-3, 3e-3):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = carrier_referenced_focus_readout(env, R0, z, wl, dx, dx_out=dx_out,
                                             N_out=N_out, standoff=so,
                                             on_replica='ignore')
    if ref is None:
        ref = F; print(f"  standoff=default -> reference, peak={np.abs(F).max()**2:.6g}")
    else:
        pist = np.angle(np.vdot(ref, F))
        print(f"  standoff={so*1e6:8.1f}um  relL2={np.linalg.norm(F-ref)/np.linalg.norm(ref):.3e}"
              f"  piston-free={np.linalg.norm(F*np.exp(-1j*pist)-ref)/np.linalg.norm(ref):.3e}"
              f"  piston={pist:+.3e}")
