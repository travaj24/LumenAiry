import numpy as np, sys, cmath
sys.path.insert(0, r"D:/Metacept/Neurophos/Free_Space_Optics")
from lumenairy.elements.coatings import coating_reflectance

def airy(n1,n2,n3,d,wl,th1,pol):
    """Analytic single-layer Airy formula, standard Fresnel convention."""
    s1 = n1*np.sin(th1)
    c1 = np.cos(th1)
    c2 = np.sqrt(1-(s1/n2)**2+0j); c2 = c2 if (n2*c2).imag>=0 else -c2
    c3 = np.sqrt(1-(s1/n3)**2+0j); c3 = c3 if (n3*c3).imag>=0 else -c3
    if pol=='s':
        r12=(n1*c1-n2*c2)/(n1*c1+n2*c2); r23=(n2*c2-n3*c3)/(n2*c2+n3*c3)
        t12=2*n1*c1/(n1*c1+n2*c2);      t23=2*n2*c2/(n2*c2+n3*c3)
    else:  # standard Fresnel p
        r12=(n2*c1-n1*c2)/(n2*c1+n1*c2); r23=(n3*c2-n2*c3)/(n3*c2+n2*c3)
        t12=2*n1*c1/(n2*c1+n1*c2);      t23=2*n2*c2/(n3*c2+n2*c3)
    b = 2*np.pi/wl*n2*d*c2
    r=(r12+r23*np.exp(2j*b))/(1+r12*r23*np.exp(2j*b))
    t=(t12*t23*np.exp(1j*b))/(1+r12*r23*np.exp(2j*b))
    if pol=='s':
        T=np.real(n3*c3)/np.real(n1*c1)*abs(t)**2
    else:
        T=np.real(np.conj(n3)*c3)/np.real(np.conj(n1)*c1)*abs(t)**2
    return r,abs(r)**2,T

wl=550e-9
print("=== A. quarter-wave MgF2 on glass (n=1.38, d=lam/4n, n_sub=1.52) ===")
n2,nsub=1.38,1.52; d=wl/(4*n2)
for thd in (0.0,45.0,70.0):
    th=np.radians(thd)
    for pol in ('s','p'):
        R,T,ph = coating_reflectance([(n2,d)], wl, angle=th, n_substrate=nsub, n_ambient=1.0, polarization=pol)
        r_a,R_a,T_a = airy(1.0,n2,nsub,d,wl,th,pol)
        print(f"  {thd:4.0f}deg {pol}: R_lib={R:.8f} R_airy={R_a:.8f} dR={abs(R-R_a):.2e} |"
              f" T_lib={T:.8f} T_airy={T_a:.8f} dT={abs(T-T_a):.2e} |"
              f" phase_lib={ph:+.6f} phase_airy={np.angle(r_a):+.6f} dphi={abs(ph-np.angle(r_a)):.2e}"
              f" dphi_negp={abs(ph-np.angle(-r_a)):.2e}")

print("\n=== B. 50 nm Au film (n=0.27+2.78j @550nm) on glass ===")
nau=0.27+2.78j; d=50e-9
for thd in (0.0,45.0,70.0):
    th=np.radians(thd)
    for pol in ('s','p'):
        R,T,ph = coating_reflectance([(nau,d)], wl, angle=th, n_substrate=1.52, n_ambient=1.0, polarization=pol)
        r_a,R_a,T_a = airy(1.0,nau,1.52,d,wl,th,pol)
        print(f"  {thd:4.0f}deg {pol}: R_lib={R:.8f} R_airy={R_a:.8f} dR={abs(R-R_a):.2e} |"
              f" T_lib={T:.8f} T_airy={T_a:.8f} dT={abs(T-T_a):.2e} | R+T={R+T:.6f}")

print("\n=== C. bare interface: r_s vs r_p sign at normal incidence (convention) ===")
for pol in ('s','p'):
    R,T,ph = coating_reflectance([], wl, angle=0.0, n_substrate=1.52, n_ambient=1.0, polarization=pol)
    print(f"  pol={pol}: R={R:.6f} phase_r={ph:+.6f}  (r={np.sqrt(R)*np.exp(1j*ph):+.6f})")
print("  standard-Fresnel oracle: r_s=%.6f  r_p=%.6f" % ((1-1.52)/(1+1.52), (1.52-1)/(1.52+1)))

print("\n=== D. polarization alias handling ('te'/'tm'/'S'/junk) ===")
base={}
for pol in ('s','p'):
    base[pol]=coating_reflectance([(n2,d)], wl, angle=np.radians(60), n_substrate=nsub, polarization=pol)[0]
for pol in ('s','p','te','tm','S','P','avg','banana',''):
    try:
        R=coating_reflectance([(n2,d)], wl, angle=np.radians(60), n_substrate=nsub, polarization=pol)[0]
        tag = 's-branch' if abs(R-base['s'])<1e-14 else ('p-branch' if abs(R-base['p'])<1e-14 else '???')
        print(f"  polarization={pol!r:10s} -> R={R:.8f}  [{tag}]")
    except Exception as ex:
        print(f"  polarization={pol!r:10s} -> raised {type(ex).__name__}: {ex}")

print("\n=== E. TIR: glass->air past critical angle (n_amb=1.52, n_sub=1.0, no layers) ===")
for thd in (30.0, 41.14, 45.0, 60.0):
    for pol in ('s','p'):
        R,T,ph=coating_reflectance([], wl, angle=np.radians(thd), n_substrate=1.0, n_ambient=1.52, polarization=pol)
        print(f"  {thd:6.2f}deg {pol}: R={R:.10f} T={T:.3e} R+T={R+T:.10f}")

print("\n=== F. absorbing substrate energy closure (R + T + A) ===")
# 100nm SiO2 on Si (n=4.0+0.05j)
R,T,_=coating_reflectance([(1.46,100e-9)], wl, angle=np.radians(40), n_substrate=4.0+0.05j, polarization='s')
print("  s: R=%.6f T=%.6f R+T=%.6f (A_substrate excluded by definition)"%(R,T,R+T))
R,T,_=coating_reflectance([(1.46,100e-9)], wl, angle=np.radians(40), n_substrate=4.0+0.05j, polarization='p')
print("  p: R=%.6f T=%.6f R+T=%.6f"%(R,T,R+T))
