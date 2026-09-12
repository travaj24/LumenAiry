from common import *
import lumenairy as lm
import lumenairy.elements.polarization as P
print("polarization exports:", [n for n in dir(P) if not n.startswith('_')][:40], flush=True)

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure
from lumenairy.elements.rcwa import rcwa_jones_1d, rcwa_jones_2d

# ---------------- form-birefringent QWP, Lambda/lambda = 0.2 ----------------
wl   = 1.0e-6
Lam  = 0.2e-6                     # Lambda/lambda = 0.2
nSi, nAir = 3.48, 1.0
f    = 0.5                        # duty (ridge fraction along x)
eps_r, eps_g = nSi**2, nAir**2
# 0th-order EMT (Rytov): E ALONG the grooves (y) sees eps_par ; across (x) eps_perp
eps_par  = f*eps_r + (1-f)*eps_g          # "TE" of the grating (E || grooves, along y)
eps_perp = 1.0/(f/eps_r + (1-f)/eps_g)    # "TM" (E across, along x)
n_par, n_perp = np.sqrt(eps_par), np.sqrt(eps_perp)
print(f"EMT: n_par(y, along grooves)={n_par:.6f}  n_perp(x)={n_perp:.6f}  dn={n_par-n_perp:.6f}")
d_qwp = wl/4.0/(n_par-n_perp)
print(f"QWP thickness (EMT, retardance = pi/2): d = {d_qwp*1e9:.2f} nm", flush=True)
dep = d_qwp

# ---- independent oracle: anisotropic-slab TMM (normal incidence, decoupled) ----
def slab_t(n0, n1, n2, d, wl):
    k0=2*np.pi/wl
    r01=(n0-n1)/(n0+n1); r12=(n1-n2)/(n1+n2)
    t01=2*n0/(n0+n1);    t12=2*n1/(n1+n2)
    ph=np.exp(1j*k0*n1*d)
    return t01*t12*ph/(1+r01*r12*ph*ph)
t_par  = slab_t(1.0, n_par,  1.0, dep, wl)
t_perp = slab_t(1.0, n_perp, 1.0, dep, wl)
d_emt = np.angle(t_par) - np.angle(t_perp)       # arg(Jyy) - arg(Jxx)
print(f"EMT slab: arg(t_y)-arg(t_x) = {d_emt:+.6f} rad = {np.rad2deg(d_emt):+.3f} deg "
      f"(target +90 deg if slow axis = y picks up exp(+i*retardance))", flush=True)

# ---- rcwa_jones_1d TRANSMISSION Jones (te/tm == y/x at phi=0) ----
def iso(e): return e*np.eye(3, dtype=complex)
out = rcwa_jones_1d(Lam, iso(eps_r), iso(eps_g), 1.0, 1.0, dep, f, wl,
                    angle=0.0, n_orders=40, formulation="li",
                    return_jones_transmission=True)
Jt = out[-1]
print("rcwa_jones_1d transmission Jones =\n", Jt, flush=True)
# CONVENTIONS 7.1: 1-D solvers return te/tm.  Determine ordering empirically by |t|.
for lbl, idx in (("[0,0]",(0,0)), ("[1,1]",(1,1))):
    print(f"   J{lbl} = {Jt[idx]:.8f}  |.|={abs(Jt[idx]):.6f} arg={np.angle(Jt[idx]):+.6f}")
d_rcwa1d_00m11 = np.angle(Jt[0,0]) - np.angle(Jt[1,1])
print(f"   arg(J00)-arg(J11) = {d_rcwa1d_00m11:+.6f} rad = {np.rad2deg(d_rcwa1d_00m11):+.3f} deg", flush=True)

# ---- PMM2DStackHybrid transmission Jones (lab x,y) ----
S=20
cell = np.full((S,S), complex(eps_g)); cell[:int(round(f*S)),:] = eps_r   # x-patterned
for nn in (9, 15, 21):
    st = PMM2DStackHybrid(Lam, Lam, n_superstrate=1.0, n_substrate=1.0,
                          degree=11, n_orders=nn, formulation="li")
    st.add_layer(dep, eps_cell=cell); st.set_source(wl, theta=0.0, phi=0.0)
    st.solve()
    Jh = st.jones_transmission()
    dphi = np.angle(Jh[1,1]) - np.angle(Jh[0,0])
    print(f"  PMM2DStackHybrid n_orders={nn:3d}: Jxx={Jh[0,0]:.8f} Jyy={Jh[1,1]:.8f}  "
          f"arg(Jyy)-arg(Jxx)={dphi:+.6f} rad ({np.rad2deg(dphi):+.3f} deg) "
          f"|Jxy|={abs(Jh[0,1]):.2e}", flush=True)

# ---- PMM2DStackPure (no-floor) ----
try:
    for M in (6, 9):
        sp = PMM2DStackPure(Lam, Lam, n_superstrate=1.0, n_substrate=1.0,
                            n_modes=M, n_orders=5)
        sp.add_layer(dep, eps_cell=cell); sp.set_source(wl, theta=0.0, phi=0.0)
        sp.solve()
        Jp = sp.jones_transmission()
        dphi = np.angle(Jp[1,1]) - np.angle(Jp[0,0])
        print(f"  PMM2DStackPure   n_modes={M:3d}: Jxx={Jp[0,0]:.8f} Jyy={Jp[1,1]:.8f}  "
              f"arg(Jyy)-arg(Jxx)={dphi:+.6f} rad ({np.rad2deg(dphi):+.3f} deg)", flush=True)
except Exception as e:
    print("  PMM2DStackPure:", type(e).__name__, str(e)[:200], flush=True)

# ---- close the loop with the Stokes convention ----
print()
print("== Stokes closure: CONVENTIONS says a QWP with FAST axis at +45 deg on x-pol -> S3 = -1 ==")
def stokes(E):
    Ex, Ey = E
    return np.array([abs(Ex)**2+abs(Ey)**2, abs(Ex)**2-abs(Ey)**2,
                     2*np.real(Ex*np.conj(Ey)), -2*np.imag(Ex*np.conj(Ey))])
# Our grating: slow = y (n_par), fast = x.  Rotate so FAST axis is at +45 deg.
c,s = np.cos(np.pi/4), np.sin(np.pi/4)
Rp = np.array([[c,-s],[s,c]])            # rotate element frame -> lab by +45
Jideal = np.diag([1.0, np.exp(1j*np.pi/2)])   # slow=y picks up exp(+i*pi/2)
Jlab = Rp @ Jideal @ Rp.T
Ein = np.array([1.0+0j, 0.0+0j])
print("  ideal QWP (fast=x at 0, rotated +45): S =", stokes(Jlab@Ein))
try:
    apw = getattr(P, "apply_waveplate", None) or getattr(lm, "apply_waveplate", None)
    if apw is not None:
        import inspect; print("  apply_waveplate signature:", inspect.signature(apw))
        Eo = apw(Ein, np.pi/2, np.pi/4)
        print("  lumenairy apply_waveplate(x-pol, pi/2, +45deg) -> S =", stokes(np.asarray(Eo).ravel()[:2]))
except Exception as e:
    print("  apply_waveplate:", type(e).__name__, str(e)[:200])
