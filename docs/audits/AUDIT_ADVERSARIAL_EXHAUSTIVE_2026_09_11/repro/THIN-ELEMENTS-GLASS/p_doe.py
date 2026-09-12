import numpy as np, sys, warnings, time
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.doe import (create_kinoform, create_diffractive_lens,
    create_fresnel_zone_plate, create_microlens_array, create_periodic_phase_mask)
from lumenairy.elements.elements import generate_turbulence_screen

print("=== create_kinoform: 1st-order efficiency vs sinc^2(1/L) ===")
# 1-D blazed grating equivalent: quantize a linear ramp
for L in (2,4,8,16):
    M=4096; per=M//8
    phi = np.mod(-2*np.pi*np.arange(M)/per, 2*np.pi)
    step=2*np.pi/L
    for name, q in (('floor', np.floor(phi/step)*step), ('round', np.round(phi/step)*step % (2*np.pi))):
        t=np.exp(1j*q); F=np.fft.fft(t)/M
        eta = abs(F[M//per])**2
        if name=='floor':
            print(f"  L={L:2d}  eta_1(floor)={eta:.6f}", end='')
        else:
            print(f"  eta_1(round)={eta:.6f}   sinc^2(1/L)={np.sinc(1.0/L)**2:.6f}")

print("\n=== create_fresnel_zone_plate: zone radii (paraxial vs exact) ===")
f=10e-3; wl=1.0e-6; N=1024; dx=1.0e-6
T = create_fresnel_zone_plate(N, dx, f, wl, binary=True)
x=(np.arange(N)-N/2)*dx
row = np.abs(T[N//2,:])
# find first transition on the +x side
idx = np.where(np.diff(row[N//2:])!=0)[0]
r_meas = x[N//2+idx[0]+1]
print(f"  1st zone boundary measured {r_meas*1e6:.4f} um ; paraxial sqrt(1*lam*f)={np.sqrt(wl*f)*1e6:.4f} um ;"
      f" exact sqrt(m lam f + m^2 lam^2/4)={np.sqrt(wl*f+wl**2/4)*1e6:.4f} um")
m=np.arange(1,60)
rp=np.sqrt(m*wl*f); rex=np.sqrt(m*wl*f+m**2*wl**2/4)
print(f"  at m=50: paraxial r={rp[49]*1e6:.3f} um exact r={rex[49]*1e6:.3f} um  rel diff={100*(rex[49]/rp[49]-1):.3f}% (NA={rp[49]/f:.3f})")

print("\n=== create_microlens_array ===")
N=2048; dx=2e-6; n_l=8; pitch=100e-6; fl=2e-3; wl=1e-6
t0=time.perf_counter(); M1 = create_microlens_array(N,dx,n_l,pitch,fl,wl); t1=time.perf_counter()
print(f"  N={N} n_lenslets={n_l}: {t1-t0:.4f}s ; |T| unity everywhere: {np.allclose(np.abs(M1),1.0)}")
# fractional pitch: pitch not an integer number of pixels
pitch2 = 101.3e-6
M2 = create_microlens_array(N,dx,n_l,pitch2,fl,wl)
print(f"  fractional pitch {pitch2*1e6} um / dx {dx*1e6} um = {pitch2/dx:.3f} px : runs OK, |T| unity {np.allclose(np.abs(M2),1.0)}")
# lenslet-center phase gradient must be ~0 at each center (no steer)
x=(np.arange(N)-N/2)*dx
ph=np.angle(M1); k=2*np.pi/wl
# check center row through the lenslet at xc = (0 - 3.5)*pitch etc
for j in (0, 3, 7):
    xc=(j-(n_l-1)/2)*pitch
    i=int(round(xc/dx + N/2))
    g=(np.unwrap(ph[N//2, i-2:i+3])[-1]-np.unwrap(ph[N//2, i-2:i+3])[0])/(4*dx)
    print(f"    lenslet j={j} xc={xc*1e6:+8.1f}um  local dphi/dx = {g:+.4g} rad/m -> steer {g/k*1e3:+.4f} mrad")
t0=time.perf_counter(); M3 = create_microlens_array(4096,1e-6,64,60e-6,2e-3,1e-6); t1=time.perf_counter()
print(f"  N=4096, 64x64 lenslets: {t1-t0:.4f}s (fully vectorised, no per-lenslet loop)")

print("\n=== create_periodic_phase_mask: tiling exactness / occupancy ===")
cell = np.where(np.arange(8) < 4, 0.0, np.pi)
cell2 = np.tile(cell, (8,1))
N=256; cps=2e-6
Mk = create_periodic_phase_mask(N, cps, cell2, cps)
idx = np.round(np.mod((np.arange(N)-N/2)*cps, 8*cps)/cps).astype(int) % 8
occ = np.bincount(idx, minlength=8)
print("  per-cell-pixel occupancy:", occ, " (uniform 32 expected)")
F=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Mk)))/N
P=np.abs(F)**2; tot=P.sum()
fx=np.fft.fftshift(np.fft.fftfreq(N,cps))
on = np.zeros_like(P, dtype=bool)
for i,fxx in enumerate(fx):
    for j,fyy in enumerate(fx):
        if abs(np.mod(fxx*8*cps+0.5,1)-0.5)<1e-9 and abs(np.mod(fyy*8*cps+0.5,1)-0.5)<1e-9: on[j,i]=True
print(f"  power off the order lattice: {100*(1-P[on].sum()/tot):.4f} %")

print("\n=== generate_turbulence_screen: structure function vs 6.88 (r/r0)^(5/3) ===")
r0=0.1; N=512; dx=0.005
accum=None
K=20
for s in range(K):
    ph=generate_turbulence_screen(N,dx,r0,seed=s)
    d=[]
    for sep in [1,2,4,8,16,32,64]:
        d.append(np.mean((ph[:, sep:]-ph[:, :-sep])**2))
    d=np.array(d)
    accum = d if accum is None else accum+d
D=accum/K
sep=np.array([1,2,4,8,16,32,64])*dx
Dth=6.88*(sep/r0)**(5/3)
print(f"  {'r[m]':>8s} {'D_meas':>10s} {'D_theory':>10s} {'ratio':>8s}")
for r,dm,dt in zip(sep,D,Dth):
    print(f"  {r:8.4f} {dm:10.4f} {dt:10.4f} {dm/dt:8.4f}")
