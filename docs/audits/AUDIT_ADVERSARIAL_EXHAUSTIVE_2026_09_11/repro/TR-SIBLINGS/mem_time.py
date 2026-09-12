import time, tracemalloc, warnings, numpy as np, lumenairy
from lumenairy.elements._lens_traced_multibranch import apply_real_lens_traced_multibranch as MB
lam=1.0e-6
rxf=lumenairy.make_singlet(R1=1.2e-3,R2=float('inf'),d=0.4e-3,glass='N-BK7',aperture=2*0.6e-3/0.98)
def run(N,dx,dout,sub,band='ludwig'):
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r=np.hypot(X,Y)
    t=np.clip((0.56e-3+30e-6-r)/30e-6,0,1); E=(0.5*(1-np.cos(np.pi*t))).astype(np.complex128)
    tracemalloc.start()
    base=tracemalloc.get_traced_memory()[0]
    t0=time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eo,dg=MB(E,prescription=rxf,wavelength=lam,dx=dx,output_plane_distance=dout,
                 ray_subsample=sub,caustic_band=band,return_diagnostics=True)
    el=time.time()-t0
    cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    nb=dg['n_branch']
    n2=int((nb>=2).sum())
    p=float(np.sum(np.abs(Eo)**2))/float(np.sum(np.abs(E)**2))
    print('  N=%-5d sub=%d d_out=%-7.4g band=%-7s  %6.2f s  peak %.2f GB  pixels>=2 branches %d  P_out/P_in %.4f'
          %(N,sub,dout,band,el,(peak-base)/1e9,n2,p))
    return el,n2
print('== memory + time of the triangle rasteriser ==')
run(2048,0.7e-6,0.0,4)
run(2048,0.7e-6,0.0,2)
run(4096,0.35e-6,0.0,4)
print('== ludwig pair-swap Python loop cost (same call, band ludwig vs plain) ==')
for (N,dx,dout,sub) in ((2048,0.7e-6,1.9e-3,2),(4096,0.35e-6,1.9e-3,2)):
    a,_=run(N,dx,dout,sub,'plain'); b,n2=run(N,dx,dout,sub,'ludwig')
    print('   -> ludwig overhead %.2f s over %d multi-branch pixels = %.1f us/pixel'
          %(b-a,n2,(b-a)/max(n2,1)*1e6))
