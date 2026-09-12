"""HF / RS / ASM cross-comparison against an exact Hankel-transform
angular-spectrum oracle for a Gaussian beam."""
import sys, time, warnings
import numpy as np
from scipy.special import j0
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
from lumenairy.propagators.hf import propagate_huygens_fresnel

lam = 633e-9
k = 2*np.pi/lam
N, dx, w0 = 128, 1e-6, 10e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
Rr = np.hypot(X, Y)
E0 = np.exp(-(Rr/w0)**2).astype(np.complex128)
zR = np.pi*w0**2/lam
print(f"N={N} dx={dx*1e6:.1f}um w0={w0*1e6:.1f}um lam={lam*1e9:.0f}nm  "
      f"z_R={zR*1e6:.1f}um  window=+-{N*dx/2*1e6:.0f}um")


def hankel_oracle(r, z, nf=6000, chunk=400):
    """Exact scalar angular-spectrum propagation of exp(-r^2/w0^2),
    E(r,z) = 2 pi \\int Ehat(f) J0(2 pi f r) exp(i k z sqrt(1-(lam f)^2)) f df,
    Ehat(f) = pi w0^2 exp(-(pi w0 f)^2)."""
    fmax = 8.0/(np.pi*w0)            # spectrum is exp(-(pi w0 f)^2): dead by 8
    fmax = min(fmax, 0.999999/lam)
    fn, fw = np.polynomial.legendre.leggauss(nf)
    fv = 0.5*fmax*(fn+1.0); fwt = 0.5*fmax*fw
    Eh = np.pi*w0**2*np.exp(-(np.pi*w0*fv)**2)
    prop = np.exp(1j*k*z*np.sqrt(np.maximum(1.0-(lam*fv)**2, 0.0)))
    wgt = 2*np.pi*Eh*prop*fv*fwt                       # (nf,)
    r = np.ravel(np.atleast_1d(r))
    out = np.empty(r.shape, dtype=complex)
    for i0 in range(0, r.size, chunk):
        rr = r[i0:i0+chunk]
        out[i0:i0+chunk] = j0(2*np.pi*np.outer(rr, fv)) @ wgt
    return out


def relL2(A, B):
    return float(np.linalg.norm(A-B)/np.linalg.norm(B))


runif = np.unique(np.round(Rr.ravel(), 12))
for z in (100e-6, 1e-3):
    print(f"\n----- z = {z*1e6:.0f} um  ({z/zR:.2f} z_R) -----")
    t0 = time.perf_counter()
    ref_r = hankel_oracle(runif, z)
    lut = dict(zip(runif, ref_r))
    REF = np.vectorize(lut.get)(np.round(Rr, 12)).astype(complex)
    t_ref = time.perf_counter()-t0

    res = {}
    t0 = time.perf_counter()
    res['ASM bandlimit=False'] = angular_spectrum_propagate(
        E0, z=z, wavelength=lam, dx=dx, bandlimit=False)
    t_asm = time.perf_counter()-t0
    res['ASM bandlimit=True'] = angular_spectrum_propagate(
        E0, z=z, wavelength=lam, dx=dx, bandlimit=True)
    t0 = time.perf_counter()
    res['RS  bandlimit=False'] = rayleigh_sommerfeld_propagate(
        E0, z=z, wavelength=lam, dx=dx, bandlimit=False)
    t_rs = time.perf_counter()-t0
    res['RS  bandlimit=True'] = rayleigh_sommerfeld_propagate(
        E0, z=z, wavelength=lam, dx=dx, bandlimit=True)
    t0 = time.perf_counter()
    res['hf.propagate_huygens_fresnel'] = propagate_huygens_fresnel(
        E0, z, lam, dx)
    t_hf = time.perf_counter()-t0

    print(f"  (oracle {t_ref:.2f}s, ASM {t_asm*1e3:.1f}ms, "
          f"RS {t_rs*1e3:.1f}ms, hf {t_hf*1e3:.1f}ms)")
    p_in = float(np.sum(np.abs(E0)**2))
    for nm, E in res.items():
        pw = float(np.sum(np.abs(E)**2))/p_in
        print(f"  {nm:<30s} relL2 vs oracle = {relL2(E, REF):.4e}   "
              f"power/P_in = {pw:.6f}")
    print(f"  {'hf vs RS(bandlimit=False)':<30s} relL2 = "
          f"{relL2(res['hf.propagate_huygens_fresnel'], res['RS  bandlimit=False']):.3e}"
          f"   (hf delegates to RS)")
    print(f"  {'RS(F) vs ASM(F)':<30s} relL2 = "
          f"{relL2(res['RS  bandlimit=False'], res['ASM bandlimit=False']):.4e}")
    # how badly is the RS kernel aliased?
    Lh = N*dx                       # padded half-extent
    sth = Lh/np.hypot(Lh, z)
    print(f"  RS kernel max phase step at padded-grid edge: "
          f"{k*sth*dx:.2f} rad/px  (Nyquist limit pi = 3.14)")
