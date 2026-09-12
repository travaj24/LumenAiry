from common import *
import lumenairy as lm
import lumenairy.elements.polarization as P
from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure, pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_1d, rcwa_jones_2d

wl, Lam = 1.0e-6, 0.2e-6
nSi, f = 3.48, 0.5
eps_r, eps_g = nSi**2, 1.0
eps_par  = f*eps_r + (1-f)*eps_g
eps_perp = 1.0/(f/eps_r + (1-f)/eps_g)
n_par, n_perp = np.sqrt(eps_par), np.sqrt(eps_perp)
dep = wl/4.0/(n_par-n_perp)
def wrap(d): return (d + np.pi) % (2*np.pi) - np.pi
S=20
cell = np.full((S,S), complex(eps_g)); cell[:int(round(f*S)),:] = eps_r

print(f"Lambda/lambda=0.2  Si/air 50% duty  d={dep*1e9:.2f} nm  n_par={n_par:.5f} n_perp={n_perp:.5f}")
print("retardance := wrap(arg(Jyy_t) - arg(Jxx_t))   [+pi/2 = QWP, slow axis = y]")
print()
# oracle
out = rcwa_jones_1d(Lam, eps_r*np.eye(3), eps_g*np.eye(3), 1.0, 1.0, dep, f, wl,
                    angle=0.0, n_orders=60, formulation="li", return_jones_transmission=True)
Jt = out[-1]
ret_ref = wrap(np.angle(Jt[1,1]) - np.angle(Jt[0,0]))
print(f"  rcwa_jones_1d n=60 li: J[0,0]={Jt[0,0]:.8f} J[1,1]={Jt[1,1]:.8f}")
print(f"     wrap(arg(J11)-arg(J00)) = {ret_ref:+.6f} rad = {np.rad2deg(ret_ref):+.4f} deg", flush=True)
def slab_t(n0,n1,n2,d,w):
    k0=2*np.pi/w; r01=(n0-n1)/(n0+n1); r12=(n1-n2)/(n1+n2)
    t01=2*n0/(n0+n1); t12=2*n1/(n1+n2); ph=np.exp(1j*k0*n1*d)
    return t01*t12*ph/(1+r01*r12*ph*ph)
ret_emt = wrap(np.angle(slab_t(1,n_par,1,dep,wl)) - np.angle(slab_t(1,n_perp,1,dep,wl)))
print(f"  EMT 0th-order slab   : {ret_emt:+.6f} rad = {np.rad2deg(ret_emt):+.4f} deg  "
      f"(err vs rigorous {np.rad2deg(wrap(ret_emt-ret_ref)):+.4f} deg)", flush=True)
for nn in (5,9,15):
    st = PMM2DStackHybrid(Lam, Lam, n_superstrate=1.0, n_substrate=1.0,
                          degree=11, n_orders=nn, formulation="li")
    st.add_layer(dep, eps_cell=cell); st.set_source(wl, theta=0.0, phi=0.0); st.solve()
    J = st.jones_transmission(); r = wrap(np.angle(J[1,1]) - np.angle(J[0,0]))
    print(f"  PMM2DStackHybrid n_orders={nn:3d}: {r:+.6f} rad = {np.rad2deg(r):+.4f} deg  "
          f"err={np.rad2deg(wrap(r-ret_ref)):+.4f} deg  |Jxy|={abs(J[0,1]):.1e}", flush=True)
for M in (5,7,9):
    try:
        sp = PMM2DStackPure(Lam, Lam, n_superstrate=1.0, n_substrate=1.0, n_modes=M, n_orders=2)
        sp.add_layer(dep, eps_cell=cell); sp.set_source(wl, theta=0.0, phi=0.0); sp.solve()
        J = sp.jones_transmission(); r = wrap(np.angle(J[1,1]) - np.angle(J[0,0]))
        print(f"  PMM2DStackPure   n_modes={M:3d}: {r:+.6f} rad = {np.rad2deg(r):+.4f} deg  "
              f"err={np.rad2deg(wrap(r-ret_ref)):+.4f} deg", flush=True)
    except Exception as e:
        print("  PMM2DStackPure M=%d: %s %s" % (M, type(e).__name__, str(e)[:160]), flush=True)
print()
print("== REFLECTION-Jones retardance: pmm_jones_2d vs rcwa_jones_2d (same quantity) ==")
c3 = np.zeros((S,S,3,3),dtype=complex); c3[...]=eps_g*np.eye(3); c3[:int(round(f*S)),:]=eps_r*np.eye(3)
S2=80; c3b=np.zeros((S2,S2,3,3),dtype=complex); c3b[...]=eps_g*np.eye(3); c3b[:int(round(f*S2)),:]=eps_r*np.eye(3)
o,R,T,Jr = rcwa_jones_2d(Lam,Lam,c3b,1.0,1.0,dep,wl,n_orders_x=9,n_orders_y=9,formulation="li")
ref_r = wrap(np.angle(Jr[1,1])-np.angle(Jr[0,0]))
print(f"  rcwa_jones_2d li n=9 : {ref_r:+.6f} rad ({np.rad2deg(ref_r):+.4f} deg)  Jxx={Jr[0,0]:.6f}", flush=True)
for form in ("laurent","li","fff_nv"):
    try:
        o,R,T,J = pmm_jones_2d(Lam,Lam,c3,1.0,1.0,dep,wl,degree=11,n_orders=9,formulation=form)
        r = wrap(np.angle(J[1,1])-np.angle(J[0,0]))
        print(f"  pmm_jones_2d {form:8s}: {r:+.6f} rad ({np.rad2deg(r):+.4f} deg) err={np.rad2deg(wrap(r-ref_r)):+.4f} deg  Jxx={J[0,0]:.6f}", flush=True)
    except Exception as e:
        print(f"  pmm_jones_2d {form}: {type(e).__name__} {str(e)[:120]}", flush=True)
print()
print("== Stokes closure (CONVENTIONS: QWP fast axis at +45 on x-pol -> S3 = -1) ==")
def stokes(E):
    Ex,Ey=E; return np.array([abs(Ex)**2+abs(Ey)**2, abs(Ex)**2-abs(Ey)**2,
                              2*np.real(Ex*np.conj(Ey)), -2*np.imag(Ex*np.conj(Ey))])
c,s = np.cos(np.pi/4), np.sin(np.pi/4)
Rp = np.array([[c,-s],[s,c]])
Jideal = np.diag([1.0, np.exp(1j*np.pi/2)])       # slow = y picks up exp(+i*ret)
Jlab = Rp @ Jideal @ Rp.T                          # rotate so the FAST axis (x) sits at +45
Ein = np.array([1.0+0j,0.0+0j])
print("  analytic ideal QWP (slow=y, exp(+i pi/2)), fast axis rotated to +45: S =", stokes(Jlab@Ein))
import inspect
print("  apply_quarter_wave_plate signature:", inspect.signature(P.apply_quarter_wave_plate))
print("  apply_waveplate signature:", inspect.signature(P.apply_waveplate))
try:
    jf = P.create_linear_polarized(1, 1, 1e-6, angle=0.0)
    jf2 = P.apply_quarter_wave_plate(jf, np.pi/4)
    sp_ = P.stokes_parameters(jf2)
    print("  lumenairy QWP(fast/axis=+45) on x-pol -> stokes_parameters:", [np.ravel(np.asarray(v))[0] for v in (sp_ if isinstance(sp_,(list,tuple)) else [sp_])][:4])
except Exception as e:
    print("  lumenairy closure:", type(e).__name__, str(e)[:300])
