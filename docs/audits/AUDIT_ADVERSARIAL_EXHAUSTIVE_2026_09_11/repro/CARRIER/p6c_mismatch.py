"""Probe 6c: mismatched carrier -> is the failure GUARDED by the shipped defaults?"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    carrier_referenced_focus_readout, carrier_referenced_envelope,
    carrier_referenced_reconstruct, propagate_carrier_referenced,
    _default_focus_standoff, _envelope_amp_radius)

wl = 1.31e-6; k = 2*np.pi/wl
N = 1024; w_in = 1.0e-3; NA = 0.05
R0 = -w_in/NA; ext = 4.0; dx = 2*ext*w_in/N
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x, indexing='xy'); r2 = X**2+Y**2
E_phys = np.exp(-r2/w_in**2)*np.exp(1j*k*r2/(2*R0))
z = -R0
w0 = wl*abs(R0)/(np.pi*w_in); dx_out = w0/8.0; N_out = 64
print(f"true focus at z={z*1e3:.3f} mm, w0={w0*1e6:.4f} um; window N_out*dx_out="
      f"{N_out*dx_out*1e6:.2f} um\n")
ref = None
for fr in (1.00, 0.99, 0.98, 0.95, 0.90):
    R = fr*R0
    env = carrier_referenced_envelope(E_phys, R, wl, dx)
    so = _default_focus_standoff(env, R, z, wl, dx)
    z_stop = z - so
    cr = propagate_carrier_referenced(env, R, z_stop, wl, dx)
    w_true_stop = _envelope_amp_radius(carrier_referenced_reconstruct(cr.env, cr.R, wl, cr.dx),
                                       cr.dx, cr.dx)
    half_stop = 0.5*N*cr.dx
    period = N*cr.dx
    status = 'ok'
    try:
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter('always')
            F = carrier_referenced_focus_readout(env, R, z, wl, dx, dx_out=dx_out,
                                                 N_out=N_out)   # DEFAULT on_replica='error'
            msgs = [str(m.message)[:60] for m in W]
    except Exception as e:
        F = None; status = f"REFUSED: {type(e).__name__}"; msgs = []
    line = (f"R/R0={fr:5.2f}  standoff={so*1e6:9.2f}um  stop dx={cr.dx*1e9:8.2f}nm "
            f"half={half_stop*1e6:8.2f}um  beam@stop={w_true_stop*1e6:8.2f}um "
            f"half/beam={half_stop/max(w_true_stop,1e-30):6.2f}  period={period*1e6:8.2f}um")
    if F is None:
        print(line + f"  -> {status}")
    else:
        if ref is None: ref = F
        print(line + f"  -> ran, peak ratio vs R0 ref={(np.abs(F).max()/np.abs(ref).max())**2:.6f}"
              f"  warns={len(msgs)} {msgs}")
