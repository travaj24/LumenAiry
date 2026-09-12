"""Rasterizer bucket-memory probe: the multibranch batches triangles into
power-of-two bounding-box buckets and materialises ~18 (n_tri, 2^c, 2^c)
float64 temporaries per bucket.  Measure the worst bucket on real cases."""
import numpy as np, lumenairy, tracemalloc, time, warnings
from lumenairy.elements._lens_traced_multibranch import _trace_launch_grid

def bucket_report(rx, lam, dx, N, d_out, sub, tag):
    ap = rx.get('aperture_diameter')
    lr = 0.5*float(ap)*0.98 if ap else 0.5*N*dx
    n_launch = max(9, int(2*lr/(dx*sub)));  n_launch += (n_launch % 2 == 0)
    g = _trace_launch_grid(rx, lam, lr, n_launch, d_out, 1.0)
    XO, YO, ok = g['x_out'], g['y_out'], g['alive'] & np.isfinite(g['x_out'])
    ok4 = ok[:-1,:-1]&ok[1:,:-1]&ok[:-1,1:]&ok[1:,1:]
    ci,cj = np.nonzero(ok4)
    V0i=np.r_[ci,ci+1]; V0j=np.r_[cj,cj]; V1i=np.r_[ci+1,ci+1]; V1j=np.r_[cj,cj+1]
    V2i=np.r_[ci,ci];   V2j=np.r_[cj+1,cj+1]
    x0,y0=XO[V0i,V0j],YO[V0i,V0j]; x1,y1=XO[V1i,V1j],YO[V1i,V1j]; x2,y2=XO[V2i,V2j],YO[V2i,V2j]
    h=g['xs_in'][1]-g['xs_in'][0]; tla=0.5*h*h
    area2=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
    with np.errstate(invalid='ignore'):
        ratio=np.abs(0.5*area2)/tla; good=np.isfinite(area2)&(ratio>=1e-6)
    xmn=np.minimum(np.minimum(x0,x1),x2)[good]; xmx=np.maximum(np.maximum(x0,x1),x2)[good]
    ymn=np.minimum(np.minimum(y0,y1),y2)[good]; ymx=np.maximum(np.maximum(y0,y1),y2)[good]
    pxmin=np.maximum(0,np.floor(xmn/dx+N/2).astype(np.int64)); pxmax=np.minimum(N-1,np.ceil(xmx/dx+N/2).astype(np.int64))
    pymin=np.maximum(0,np.floor(ymn/dx+N/2).astype(np.int64)); pymax=np.minimum(N-1,np.ceil(ymx/dx+N/2).astype(np.int64))
    keep=(pxmax>=pxmin)&(pymax>=pymin)
    wmax=np.maximum(pxmax-pxmin,pymax-pymin)[keep]+1
    cls=np.ceil(np.log2(wmax)).astype(np.int64)
    worst=0; wc=None
    for c in np.unique(cls):
        n=int((cls==c).sum()); wb=int(2**c); b=n*wb*wb*8
        if b>worst: worst, wc = b, (c,n,wb)
    print('%-42s n_tri=%d  worst bucket c=%d n=%d wb=%d -> %.3g GB per (n,wb,wb) float64 '
          'array  (~18 live => %.3g GB)'
          %(tag, keep.sum(), wc[0], wc[1], wc[2], worst/1e9, 18*worst/1e9))

lam=1.0e-6
rxf=lumenairy.make_singlet(R1=1.2e-3,R2=float('inf'),d=0.4e-3,glass='N-BK7',aperture=2*0.6e-3/0.98)
bucket_report(rxf,lam,0.35e-6,4096,1.9e-3,4,'fold case N=4096 sub=4')
bucket_report(rxf,lam,0.35e-6,4096,1.9e-3,2,'fold case N=4096 sub=2')
bucket_report(rxf,lam,0.35e-6,4096,0.0,4,'AT the exit vertex N=4096 sub=4')
lam2=0.5876e-6
rx2=lumenairy.make_singlet(R1=25e-3,R2=float('inf'),d=3e-3,glass='N-BK7',aperture=10e-3)
bucket_report(rx2,lam2,5e-6,4096,0.0,2,'10mm singlet at pupil N=4096 sub=2')
