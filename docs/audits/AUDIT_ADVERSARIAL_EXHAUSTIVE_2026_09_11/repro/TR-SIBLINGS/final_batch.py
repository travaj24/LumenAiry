import sys, time, warnings, numpy as np, cProfile, pstats, io
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
import lumenairy
from lumenairy.glass import get_glass_index
from lumenairy.elements import _lens_imap as IM
from lumenairy.elements import _lens_traced as LT

lam=0.5876e-6; ng=float(get_glass_index('N-BK7',lam))
R1,R2,d = 25e-3, float('inf'), 3e-3; LR=3.0e-3; n_launch=129
xs=np.linspace(-LR,LR,n_launch); Xi,Yi=np.meshgrid(xs,xs,indexing='ij')
rho=np.hypot(Xi,Yi)
def trace2d(xin,yin,zout=0.0):
    rr=np.hypot(xin,yin)
    xm,opl,L,Nz=trace_singlet(np.maximum(rr,1e-15),R1,R2,d,ng,zout)
    sc=np.where(rr>0, xm/np.maximum(rr,1e-300),0.0)
    x0,o0,_,_=trace_singlet(np.array([1e-12]),R1,R2,d,ng,zout)
    return xin*sc, yin*sc, opl-o0[0]
XO,YO,OP=trace2d(Xi,Yi); amp=np.exp(-(rho/2.0e-3)**2)
from scipy.interpolate import RectBivariateSpline
sx=RectBivariateSpline(xs,xs,XO); sy=RectBivariateSpline(xs,xs,YO); so=RectBivariateSpline(xs,xs,OP)
def parity_invert(xq,yq,iters=8):
    xq=np.atleast_1d(np.asarray(xq,float)); yq=np.atleast_1d(np.asarray(yq,float))
    xe=xq.copy(); ye=yq.copy()
    for _ in range(iters):
        fx=sx.ev(xe,ye); fy=sy.ev(xe,ye)
        jxx=sx.ev(xe,ye,dx=1); jxy=sx.ev(xe,ye,dy=1)
        jyx=sy.ev(xe,ye,dx=1); jyy=sy.ev(xe,ye,dy=1)
        det=jxx*jyy-jxy*jyx; det=np.where(np.abs(det)>1e-18,det,1e-18)
        rx_,ry_=fx-xq,fy-yq
        xe=np.clip(xe-(jyy*rx_-jxy*ry_)/det,-LR,LR)
        ye=np.clip(ye-(-jyx*rx_+jxx*ry_)/det,-LR,LR)
    return xe,ye,so.ev(xe,ye)
def probe_trace(px,py): return trace2d(np.asarray(px),np.asarray(py))
kw=dict(wavelength=lam,launch_radius=LR,census_amp=amp,parity_invert=parity_invert,
        parity_tag=('spline',8),probe_trace=probe_trace)

print('== D) how big can the un-keyed solver-flag difference get? (_DET_REFINE_STEPS) ==')
IM.inverse_map_cache_clear()
old_d, old_r = LT.DETERMINISTIC_TRACED_FIT, LT._DET_REFINE_STEPS
try:
    LT.DETERMINISTIC_TRACED_FIT=False
    a=IM.build_inverse_map(xs,XO,YO,OP,cache=False,**kw)
    LT.DETERMINISTIC_TRACED_FIT=True; LT._DET_REFINE_STEPS=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b=IM.build_inverse_map(xs,XO,YO,OP,cache=False,**kw)
    LT._DET_REFINE_STEPS=1
    c=IM.build_inverse_map(xs,XO,YO,OP,cache=False,**kw)
finally:
    LT.DETERMINISTIC_TRACED_FIT, LT._DET_REFINE_STEPS = old_d, old_r
def cmp(u,v,t):
    if u is None or v is None: print('   %-46s (a build refused)'%t); return
    print('   %-46s max|dcoef| %.4g rel %.4g'%(t,np.max(np.abs(u.coef-v.coef)),
          np.max(np.abs(u.coef-v.coef))/np.max(np.abs(u.coef))))
cmp(a,c,'BLAS(det=False) vs deterministic(refine=1)')
cmp(a,b,'BLAS(det=False) vs deterministic(refine=0)')
# what does that do to the OPL channel?
if a is not None and b is not None:
    rng=np.random.default_rng(1); px=rng.uniform(-0.7*LR,0.7*LR,3000); py=rng.uniform(-0.7*LR,0.7*LR,3000)
    tx,ty,top=trace2d(px,py)
    oa=a.eval(tx,ty,channels=(2,))[0]; ob=b.eval(tx,ty,channels=(2,))[0]
    print('   OPL rms error vs exact trace: BLAS %.4g waves, deterministic(refine=0) %.4g waves'
          %(np.sqrt(np.mean((oa-top)**2))/lam, np.sqrt(np.mean((ob-top)**2))/lam))

print()
print('== E) InverseCharacteristic.eval throughput (numba warm) ==')
IM.inverse_map_cache_clear()
m=IM.build_inverse_map(xs,XO,YO,OP,**kw)
n=2_000_000
qx=np.random.default_rng(0).uniform(-1e-4,1e-4,n); qy=np.random.default_rng(1).uniform(-1e-4,1e-4,n)
out=[np.empty(n) for _ in range(3)]
m.eval_into(qx,qy,out,channels=(0,1,2))                     # warm-up / compile
t0=time.time(); m.eval_into(qx,qy,out,channels=(0,1,2)); t1=time.time()-t0
print('   numba kernel: %.3f s for %d pts x 3 ch = %.1f ns/pt  (kernel=%s)'
      %(t1,n,t1/n*1e9, IM._get_imap_eval_numba() is not None))
import scipy.ndimage as ndi
src=OP.copy(); coords=np.stack([ (qx/ (2*LR/(n_launch-1)))+n_launch//2, (qy/(2*LR/(n_launch-1)))+n_launch//2])
t0=time.time(); ndi.map_coordinates(src,coords,order=3,mode='nearest'); t2=time.time()-t0
print('   scipy map_coordinates order 3, 1 channel: %.3f s (%.1f ns/pt)'%(t2,t2/n*1e9))

print()
print('== F) where does apply_real_lens_traced_uniform spend its time? ==')
from lumenairy.elements._lens_traced_uniform import apply_real_lens_traced_uniform as UNI
from lumenairy.elements._lens_traced_multibranch import apply_real_lens_traced_multibranch as MB
lam2=1.0e-6
rxf=lumenairy.make_singlet(R1=1.2e-3,R2=float('inf'),d=0.4e-3,glass='N-BK7',
                           aperture=2*0.6e-3/0.98)
for Ngrid in (1024, 2048):
    dxg=1.4e-6*2048/Ngrid
    xg=(np.arange(Ngrid)-Ngrid/2)*dxg; Xg,Yg=np.meshgrid(xg,xg); rg=np.hypot(Xg,Yg)
    tt=np.clip((0.56e-3+30e-6-rg)/30e-6,0,1); Ein=(0.5*(1-np.cos(np.pi*tt))).astype(np.complex128)
    kwf=dict(prescription=rxf,wavelength=lam2,dx=dxg,output_plane_distance=1.9e-3,ray_subsample=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0=time.time(); MB(Ein,**kwf); tmb=time.time()-t0
        t0=time.time(); Eu,du=UNI(Ein,**kwf,return_diagnostics=True); tun=time.time()-t0
    ndark=int((rg>(du.get('r_c') or 0)).sum())
    print('   N=%d dx=%.3g um : multibranch %.2f s, uniform %.2f s (+%.2f s) ; '
          'dark pixels filled = %d of %d (%.1f%%), r_c=%.3g um, l_airy=%.4g um'
          %(Ngrid,dxg*1e6,tmb,tun,tun-tmb,ndark,rg.size,100*ndark/rg.size,
            (du.get('r_c') or 0)*1e6,
            1e6/((2*np.pi/lam2)**(2/3)*(du.get('kappa') or np.inf))))
