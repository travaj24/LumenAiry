"""Probe 18: (a) n_v2 auto-resolution estimator bound; (b) GBD FFT reconstruct
memory/time; (c) Tukey weight normalisation."""
import numpy as np, sys, time, tracemalloc, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter("ignore")
import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.elements.lenses import _multi_indices_total_degree, _fit_normaliser
from lumenairy.elements.lenses_maslov import _tukey_taper
from lumenairy._math.chebyshev import chebyshev_vandermonde as CV

print("### (c) quadrature weight normalisation: sum(w * du) over [-1,1] ###")
for n in (32, 64, 128, 256):
    u = np.linspace(-1, 1, n); du = u[1]-u[0]
    w = _tukey_taper(u, 0.2)
    exact = 2.0*(1.0 - 0.2/2.0)          # int of the Tukey window over [-1,1]
    print(f"  n_v2={n:4d}: sum(w)*du = {w.sum()*du:.8f}   exact int = {exact:.8f}"
          f"   rel err = {(w.sum()*du-exact)/exact:+.3e}   (plain rect sum(du)={n*du:.6f} vs 2)")

print("\n### (a) n_v2 estimator: sum|c| (code) vs the total-variation bound sum|c|*max(k3,k4) ###")
lam=1.0e-6
for R1,R2s,ap,nm in ((6e-3,-6e-3,1.5e-3,'f=6mm biconvex, 1.5mm ap'),
                     (2e-3,-2e-3,0.3e-3,'f=2mm biconvex, 0.3mm ap')):
    presc = la.make_singlet(R1,R2s,0.7e-3,'N-BK7',aperture=ap)
    su = rt.surfaces_from_prescription(presc); r_ap=0.5*ap; na=1e-5
    def cn(n): i=np.arange(n); return np.cos((i+0.5)*np.pi/n)
    h,p = cn(16),cn(16)
    HX,HY,PX,PY = (a.ravel() for a in np.meshgrid(h,h,p,p,indexing='ij'))
    kp=(PX**2+PY**2)<=1.0; HX,HY,PX,PY=HX[kp],HY[kp],PX[kp],PY[kp]
    s1x,s1y,v1x,v1y = HX*r_ap,HY*r_ap,PX*na,PY*na
    Nd=np.sqrt(np.maximum(1-v1x**2-v1y**2,0))
    rb=rt.RayBundle(x=s1x.copy(),y=s1y.copy(),z=np.zeros_like(s1x),L=v1x.copy(),
                    M=v1y.copy(),N=Nd,wavelength=lam,alive=np.ones(len(s1x),bool),
                    opd=np.zeros(len(s1x)))
    e=rt.trace(rb,su,lam).image_rays; al=e.alive
    s2x,s2y,v2x,v2y,opd=e.x[al],e.y[al],e.L[al],e.M[al],e.opd[al]/lam
    cs=[_fit_normaliser(a) for a in (s2x,s2y,v2x,v2y)]
    u=[(a-c)/hh for a,(c,hh) in zip((s2x,s2y,v2x,v2y),cs)]
    X5=np.column_stack([np.ones_like(u[0]),u[0],u[1],u[2],u[3]])
    lc,*_=np.linalg.lstsq(X5,opd,rcond=None); res=opd-X5@lc
    for order in (4,6,8):
        mi=_multi_indices_total_degree(4,order); T=[CV(uu,order) for uu in u]
        A=np.empty((u[0].size,len(mi)))
        for j,k in enumerate(mi): A[:,j]=T[0][k[0]]*T[1][k[1]]*T[2][k[2]]*T[3][k[3]]
        c,*_=np.linalg.lstsq(A,res,rcond=None)
        m=np.array([1.0 if (k[2]>0 or k[3]>0) else 0.0 for k in mi])
        deg=np.array([max(k[2],k[3]) for k in mi],float)
        code=float(np.sum(np.abs(c)*m)); tv=float(np.sum(np.abs(c)*m*deg))
        print(f"  {nm:26s} order={order}: sum|c| = {code:8.3f} -> n_v2={int(4*code)+1:5d}"
              f" |  sum|c|*max(k3,k4) = {tv:8.3f} -> n_v2={int(4*tv)+1:5d}"
              f"  (ratio {tv/max(code,1e-30):.2f}x)")

print("\n### (b) GBD reconstruct: FFT-convolution path memory / time ###")
import lumenairy.propagators.gbd as G
lam=1.0e-6
for N,dx in ((256,2e-6),(512,2e-6)):
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
    E0=np.exp(-(X**2+Y**2)/(N*dx/8)**2).astype(complex)
    b=G.decompose_field_to_beamlets(E0,dx,wavelength=lam,waist_factor=2.0,sample_step=2)
    b=G.propagate_beamlets_freespace(b,1e-3,lam)
    for tag,fn in (('fft-conv (window=5)', lambda: G.reconstruct_field_from_beamlets(
                        b,Ny=N,Nx=N,dx=dx,wavelength=lam,window=5.0)),
                   ('windowed scatter   ', lambda: G._reconstruct_windowed(
                        b,Ny=N,Nx=N,dx=dx,dy=dx,centre=(0.,0.),wavelength=lam,n_sigma=5.0))):
        tracemalloc.start(); t=time.perf_counter(); r=fn(); dt=time.perf_counter()-t
        cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f"  N={N:4d} {tag}: {dt*1e3:8.1f} ms  peak alloc {peak/1e6:8.1f} MB"
              f"  (grid itself {N*N*16/1e6:.1f} MB)")
