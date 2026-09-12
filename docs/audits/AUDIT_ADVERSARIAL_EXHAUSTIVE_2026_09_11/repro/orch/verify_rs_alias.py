import warnings, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.propagators.propagation import rayleigh_sommerfeld_propagate, angular_spectrum_propagate
wl=633e-9; w0=6e-6
for N, dx, z in [(64,0.5e-6,50e-6),(64,2e-6,50e-6),(128,1e-6,50e-6),(128,1e-6,300e-6),(128,1e-6,1e-3)]:
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); E=np.exp(-(X**2+Y**2)/w0**2).astype(complex)
    Ers=rayleigh_sommerfeld_propagate(E, z, wl, dx); Easm=angular_spectrum_propagate(E, z, wl, dx, bandlimit=False)
    P0=np.sum(np.abs(E)**2)
    print(f"N={N:4d} dx={dx*1e6:.1f}um z={z*1e6:6.0f}um  z_crit=2Ndx^2/lam={2*N*dx**2/wl*1e6:8.1f}um  RS P/P0={np.sum(np.abs(Ers)**2)/P0:7.3f}  ASM P/P0={np.sum(np.abs(Easm)**2)/P0:.4f}  relL2(RS,ASM)={np.linalg.norm(Ers-Easm)/np.linalg.norm(Easm):.2e}")
