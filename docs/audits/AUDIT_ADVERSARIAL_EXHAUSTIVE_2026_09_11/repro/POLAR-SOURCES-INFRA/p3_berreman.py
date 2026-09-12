import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.coatings import coating_reflectance
from lumenairy.elements import polarization as P

wl=633e-9
def tmm_amp(nlist, dlist, n1, n3, th1, pol):
    """Independent characteristic-matrix TMM (standard Fresnel r_p sign)."""
    s1=n1*np.sin(th1)
    def ct(n):
        c=np.sqrt(1-(s1/n)**2+0j)
        return c if (n*c).imag>=0 else -c
    # build via recursive Fresnel (Rouard) for independence
    ns=[n1]+list(nlist)+[n3]; cs=[ct(n) for n in ns]
    ds=[0.0]+list(dlist)+[0.0]
    def rij(i,j):
        if pol=='s': return (ns[i]*cs[i]-ns[j]*cs[j])/(ns[i]*cs[i]+ns[j]*cs[j])
        return (ns[j]*cs[i]-ns[i]*cs[j])/(ns[j]*cs[i]+ns[i]*cs[j])
    def tij(i,j):
        if pol=='s': return 2*ns[i]*cs[i]/(ns[i]*cs[i]+ns[j]*cs[j])
        return 2*ns[i]*cs[i]/(ns[j]*cs[i]+ns[i]*cs[j])
    # transfer-matrix product
    M=np.eye(2,dtype=complex)
    for i in range(len(ns)-1):
        b = 2*np.pi/wl*ns[i]*ds[i]*cs[i]
        Pm=np.array([[np.exp(-1j*b),0],[0,np.exp(1j*b)]])
        r,t=rij(i,i+1),tij(i,i+1)
        Im=np.array([[1,r],[r,1]],dtype=complex)/t
        M=M@Pm@Im
    r=M[1,0]/M[0,0]; t=1/M[0,0]
    return r,t

print("=== A. Berreman isotropic reduction vs independent TMM (amplitude AND phase) ===")
for thd in (0.0, 20.0, 45.0, 70.0):
    th=np.radians(thd)
    layers=[(1.46**2, 120e-9),(2.1**2, 90e-9)]
    R,T,Jr,Jt = berreman_jones_1d(layers, n_substrate=1.52, n_superstrate=1.0,
                                  wavelength=wl, angle=th, phi=0.0)
    # at phi=0: Ey = s(TE), Ex = p(TM) up to sign
    rs_b = Jr[1,1]; rp_b = Jr[0,0]
    ts_b = Jt[1,1]; tp_b = Jt[0,0]
    rs_o,ts_o = tmm_amp([1.46,2.1],[120e-9,90e-9],1.0,1.52,th,'s')
    rp_o,tp_o = tmm_amp([1.46,2.1],[120e-9,90e-9],1.0,1.52,th,'p')
    print(f"  {thd:5.1f}deg  |rs| lib/oracle {abs(rs_b):.10f}/{abs(rs_o):.10f}  dphase={np.angle(rs_b/rs_o):+.2e}")
    print(f"          |rp| lib/oracle {abs(rp_b):.10f}/{abs(rp_o):.10f}  dphase(rp)={np.angle(rp_b/rp_o):+.3e}  dphase(-rp)={np.angle(rp_b/(-rp_o)):+.3e}")
    print(f"          R lib={R}  R_oracle=({abs(rs_o)**2:.8f} s, {abs(rp_o)**2:.8f} p)  R+T={R+T}")

print("\n=== B. uniaxial QWP slab: Berreman Jones vs apply_waveplate Stokes ===")
no,ne=1.5,1.6
d=wl/(4*(ne-no))
# fast axis at +45deg in the xy-plane: eps = R_z(45) diag(ne^2, no^2, no^2) R_z(-45)
# -> 'fast' = the LOW index axis; put no along the 45deg direction.
def rotz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
Rz=rotz(np.pi/4)
eps = Rz @ np.diag([no**2, ne**2, no**2]) @ Rz.T   # ordinary(no) along the rotated x = +45deg (fast)
R,T,Jr,Jt = berreman_jones_1d([(eps, d)], n_substrate=1.0, n_superstrate=1.0,
                              wavelength=wl, angle=0.0)
Ein=np.array([1.0,0.0],dtype=complex)
Eout=Jt@Ein
S3_b = -2*np.imag(Eout[0]*np.conj(Eout[1]))/ (abs(Eout[0])**2+abs(Eout[1])**2)
print("  Berreman jones_t =\n", np.round(Jt,6))
print("  Berreman on x-pol: Ey/Ex =", Eout[1]/Eout[0], " S3_norm =", S3_b)
jf=P.JonesField(np.ones((1,1),complex), np.zeros((1,1),complex), 1e-6)
P.apply_quarter_wave_plate(jf, angle=np.pi/4)
print("  apply_waveplate QWP@+45: Ey/Ex =", jf.Ey[0,0]/jf.Ex[0,0],
      " S3 =", P.stokes_parameters(jf)['S3'][0,0]/P.stokes_parameters(jf)['S0'][0,0])
print("  --> AGREE" if abs(S3_b - P.stokes_parameters(jf)['S3'][0,0]/P.stokes_parameters(jf)['S0'][0,0])<2e-2 else "  --> DISAGREE")

print("\n=== C. lossy BIAXIAL layer: energy R+T+A <= 1 and mode-sorting robustness ===")
for kap in (0.0, 0.02, 0.2, 1.0):
    epsb = np.diag([(1.7+1j*kap)**2, (1.5+0.5j*kap)**2, (1.9+2j*kap)**2]).astype(complex)
    # rotate about x and z to make it fully general
    def rotx(a):
        c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    E = rotz(0.7)@rotx(0.4)@epsb@rotx(-0.4)@rotz(-0.7)
    for thd in (0.0, 35.0, 65.0):
        try:
            R,T,Jr,Jt=berreman_jones_1d([(E, 400e-9)], n_substrate=1.5,
                                        n_superstrate=1.0, wavelength=wl,
                                        angle=np.radians(thd), phi=0.3)
            print(f"  kappa={kap:4.2f} th={thd:4.1f}: R={R} T={T} R+T={R+T}  ok={np.all(R+T<=1.0000001)}")
        except Exception as ex:
            print(f"  kappa={kap:4.2f} th={thd:4.1f}: RAISED {type(ex).__name__}: {str(ex)[:90]}")

print("\n=== D. thick / lossy layer overflow ===")
for t in (1e-6, 1e-4, 1e-2, 1.0):
    try:
        R,T,Jr,Jt=berreman_jones_1d([((1.5+0.01j)**2, t)], n_substrate=1.5,
                                    n_superstrate=1.0, wavelength=wl, angle=0.3)
        print(f"  d={t:9.2e} m: R={R[0]:.6e} T={T[0]:.6e} finite={np.all(np.isfinite(Jt))}")
    except Exception as ex:
        print(f"  d={t:9.2e} m: RAISED {type(ex).__name__}: {str(ex)[:80]}")

print("\n=== E. gyrotropic (Hermitian, lossless) tensor: energy check ===")
g=0.05
epsg=np.array([[2.25, 1j*g, 0],[-1j*g, 2.25, 0],[0,0,2.25]],dtype=complex)
for thd in (0.0, 40.0):
    R,T,Jr,Jt=berreman_jones_1d([(epsg, 500e-9)], n_substrate=1.0, n_superstrate=1.0,
                                wavelength=wl, angle=np.radians(thd))
    print(f"  th={thd}: R={R} T={T} R+T={R+T}")
