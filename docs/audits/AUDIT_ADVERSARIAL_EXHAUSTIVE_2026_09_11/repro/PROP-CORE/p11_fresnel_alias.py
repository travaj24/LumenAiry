import numpy as np, sys
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.fresnel import fresnel_propagate
lam=0.633e-6; N=128; dx=2e-6; k=2*np.pi/lam
z_crit=N*dx*dx/lam
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
# smooth field so the ONLY sampling issue is the chirp
w0=40e-6
E=np.exp(-(X**2+Y**2)/w0**2).astype(complex)

def fresnel_exact_1pt(x2,y2,z,ov=16):
    # fine-quadrature Fresnel integral: oversample the SAME continuous aperture field
    xf=(np.arange(N*ov)-N*ov/2)*(dx/ov)
    Xf,Yf=np.meshgrid(xf,xf,indexing='xy')
    Ef=np.exp(-(Xf**2+Yf**2)/w0**2)
    ker=np.exp(1j*k/(2*z)*((Xf-x2)**2+(Yf-y2)**2))
    return np.exp(1j*k*z)/(1j*lam*z)*np.sum(Ef*ker)*(dx/ov)**2

print(f"  N={N} dx={dx*1e6}um lam={lam*1e9}nm z_crit={z_crit*1e3:.4f} mm ; single-FFT Fresnel has NO sampling guard")
for f in (0.25, 0.5, 1.0, 2.0, 4.0):
    z=f*z_crit
    Ef,dxo,_=fresnel_propagate(E,z,lam,dx)
    errs=[]
    for (jy,jx) in [(N//2,N//2),(N//2,N//2+8),(N//2+5,N//2+11)]:
        x2=(jx-N/2)*dxo; y2=(jy-N/2)*dxo
        ex=fresnel_exact_1pt(x2,y2,z,ov=8)
        errs.append(abs(Ef[jy,jx]-ex)/max(abs(ex),1e-300))
    print(f"   z={f:4.2f}*z_crit (dx_out={dxo*1e6:7.3f}um): rel err vs 8x-oversampled Fresnel integral at 3 pts = "
          + ", ".join(f"{e:.2e}" for e in errs))
