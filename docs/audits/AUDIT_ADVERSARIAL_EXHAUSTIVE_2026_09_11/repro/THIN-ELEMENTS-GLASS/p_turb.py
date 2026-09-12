import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.elements import generate_turbulence_screen

N, dx, r0 = 512, 0.005, 0.1
L=N*dx; df=1.0/L
fx=(np.arange(N)-N//2)*df
FX,FY=np.meshgrid(fx,fx); f_sq=FX**2+FY**2
f_safe=np.where(f_sq>0,f_sq,1.0)
psd=0.023*r0**(-5/3)*f_safe**(-11/6)
psd[N//2,N//2]=0.0

def D_lattice(rvec, factor):
    """Exact expected structure function of the lattice sum with amplitude sqrt(factor*psd)*df."""
    A2 = factor*psd*df**2
    return float(np.sum(A2*2*(1-np.cos(2*np.pi*(FX*rvec[0]+FY*rvec[1])))))
def var_lattice(factor):
    return float(np.sum(factor*psd*df**2))

seps=[1,2,4,8,16,32,64]
K=60
acc=np.zeros(len(seps)); varacc=0.0
for s in range(K):
    ph=generate_turbulence_screen(N,dx,r0,seed=1000+s)
    varacc += ph.var()
    for i,sp in enumerate(seps):
        acc[i]+=np.mean((ph[:,sp:]-ph[:,:-sp])**2)
D=acc/K; V=varacc/K
print(f"{'r[m]':>8s} {'D_meas':>10s} {'D_lat(x1)':>10s} {'D_lat(x2)':>10s} {'meas/x1':>8s} {'meas/x2':>8s} {'D_kolm':>10s} {'meas/kolm':>9s}")
for sp,dm in zip(seps,D):
    r=sp*dx
    d1=D_lattice((r,0.0),1.0); d2=D_lattice((r,0.0),2.0)
    dk=6.88*(r/r0)**(5/3)
    print(f"{r:8.4f} {dm:10.4f} {d1:10.4f} {d2:10.4f} {dm/d1:8.4f} {dm/d2:8.4f} {dk:10.4f} {dm/dk:9.4f}")
print(f"\nscreen variance: measured {V:.4f} ; lattice sum(psd*df^2) = {var_lattice(1.0):.4f} ;  x2 = {var_lattice(2.0):.4f}")
print(f"  -> measured/lattice(x1) = {V/var_lattice(1.0):.4f}   measured/lattice(x2) = {V/var_lattice(2.0):.4f}")
print("\nSchmidt (Numerical Simulation of Optical Wave Propagation, ft_phase_screen):")
print("   cn = (randn + i randn) * sqrt(PSD_phi) * del_f ;  phz = real(ift2(cn,1))   [NO sqrt(2)]")
