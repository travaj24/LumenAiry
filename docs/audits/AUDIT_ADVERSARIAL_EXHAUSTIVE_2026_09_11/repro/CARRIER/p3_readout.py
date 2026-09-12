"""Probe 3: carrier_referenced_focus_readout vs an independent oracle.

Oracle = direct Fresnel/Debye integral of the converging Gaussian at the focus,
plus a brute-force chirp-z (matrix DFT) of the reconstructed input field on a
fine grid.  f/10-ish beam.
"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope, carrier_referenced_focus_readout,
    _default_focus_standoff)

wl = 1.31e-6; k = 2*np.pi/wl

def gauss_env(x, w):
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X**2+Y**2)/w**2).astype(np.complex128)

def analytic_focus(xo, w_in, R, wl):
    """Analytic Gaussian at the geometric focus of a converging Gaussian whose
    1/e amplitude radius is w_in and (parabolic) radius R<0.  Waist w0=wl|R|/(pi w_in)
    sits at z=-R (to leading order; exact for a pure Gaussian with R the
    Gaussian radius). We instead use the exact Gaussian ABCD from (w_in,R)."""
    zR_in = None
    # q at input: 1/q = 1/R - i wl/(pi w^2)
    invq = 1.0/R - 1j*wl/(np.pi*w_in**2)
    q = 1.0/invq
    z = -R  # propagate to geometric focus
    q2 = q + z
    invq2 = 1.0/q2
    Rz = 1.0/np.real(invq2) if np.real(invq2) != 0 else np.inf
    wz = np.sqrt(-wl/(np.pi*np.imag(invq2)))
    X, Y = np.meshgrid(xo, xo, indexing='xy'); r2 = X**2+Y**2
    amp = (w_in/wz)*np.exp(-r2/wz**2)
    ph = k*z + (k*r2/(2*Rz) if np.isfinite(Rz) else 0.0)
    gouy = np.angle(q/q2)   # (q/q2) carries 1/(1+z/q); its phase is the Gouy term
    return amp*np.exp(1j*ph)*np.exp(1j*np.angle(q/q2)), wz, Rz

for NA, N, ext in ((0.05, 512, 4.0), (0.05, 512, 2.0), (0.10, 1024, 4.0)):
    w_in = 1.0e-3
    R = -w_in/NA
    dx = 2*ext*w_in/N
    x = (np.arange(N)-N/2)*dx
    env = gauss_env(x, w_in)
    z = -R  # land on the geometric focus
    Eref, wz, Rz = analytic_focus(np.arange(1)*0.0, w_in, R, wl)
    w0 = wz
    dx_out = w0/8.0
    N_out = 128
    so = _default_focus_standoff(env, R, z, wl, dx)
    zR = np.pi*w0**2/wl
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        F = carrier_referenced_focus_readout(env, R, z, wl, dx,
                                             dx_out=dx_out, N_out=N_out,
                                             on_replica='warn')
        nw = [str(x.message)[:70] for x in W]
    xo = (np.arange(N_out)-N_out/2)*dx_out
    T, _, _ = analytic_focus(xo, w_in, R, wl)
    # normalise piston out
    pist = np.angle(np.vdot(T, F))
    rel = np.linalg.norm(F*np.exp(-1j*pist)-T)/np.linalg.norm(T)
    relA = np.linalg.norm(np.abs(F)-np.abs(T))/np.linalg.norm(np.abs(T))
    IF = np.abs(F)**2; IT = np.abs(T)**2
    # encircled energy at 2 w0
    XO, YO = np.meshgrid(xo, xo, indexing='xy'); rr = np.hypot(XO, YO)
    eeF = IF[rr<=2*w0].sum()/IF.sum(); eeT = IT[rr<=2*w0].sum()/IT.sum()
    iy, ix = np.unravel_index(np.argmax(IF), IF.shape)
    print(f"NA={NA} N={N} ext={ext}  w0={w0*1e6:.4f}um zR={zR*1e6:.2f}um standoff={so*1e6:.3f}um "
          f"(f={so/zR:.3f} zR)")
    print(f"    input carrier step at edge = {k*dx*w_in/abs(R):.3f} rad/px "
          f"| stop-plane step = {k*dx*w_in/abs(R)*(so/abs(R)):.4f} rad/px")
    print(f"    relL2(piston-free)={rel:.3e}  |amp| relL2={relA:.3e}  peak ratio={IF.max()/IT.max():.5f} "
          f"peak@({ix-N_out//2},{iy-N_out//2})px  EE(2w0) {eeF:.5f} vs {eeT:.5f}")
    if nw: print("    warnings:", nw)
