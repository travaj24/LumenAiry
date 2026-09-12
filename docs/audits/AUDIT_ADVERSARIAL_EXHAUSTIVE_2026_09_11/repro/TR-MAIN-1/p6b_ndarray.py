"""Probe 6b: the carrier=ndarray branch's grid quantisation -- scaling with dx."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import _compute_carrier
WL = common.WL; k0 = 2*np.pi/WL
S = 200e-3; AP = 6e-3

# (a) direct probe of _compute_carrier: ndarray vs the analytic float branch
for N in (256, 512, 1024):
    dx = 1.35*AP/N
    x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
    W = np.sqrt(X**2+Y**2+S*S) - S
    Wf_a, g_a, w_a = _compute_carrier(S, np.ones((N,N)), WL, dx, X, Y)
    Wf_n, g_n, w_n = _compute_carrier(W, np.ones((N,N)), WL, dx, X, Y)
    # query at LAUNCH-grid nodes (a different lattice from the wave grid)
    lr = 0.75*AP
    nl = max(8, int(2*lr/(dx*8)));  nl += (nl % 2 == 0)
    xs = np.linspace(-lr, lr, nl)
    Xq, Yq = np.meshgrid(xs, xs, indexing='ij')
    hx, hy = Xq.ravel(), Yq.ravel()
    inside = (np.abs(hx) < 0.5*N*dx-dx) & (np.abs(hy) < 0.5*N*dx-dx)
    dW = (w_n(hx, hy) - w_a(hx, hy))[inside]
    La, Ma = g_a(hx, hy); Ln, Mn = g_n(hx, hy)
    dL = (Ln-La)[inside]
    print(f"N={N} dx={dx*1e6:6.3f} um  n_launch={nl}: "
          f"eikonal W error: rms={dW.std()*1e9:8.4f} nm max={np.abs(dW).max()*1e9:8.4f} nm "
          f"(= {k0*np.abs(dW).max():.4f} rad) | cosine L error rms={dL.std():.3e} "
          f"max={np.abs(dL).max():.3e}")
    print(f"      predicted half-pixel eikonal error |gradW|*dx/2 = "
          f"{(0.5*AP/np.sqrt((0.5*AP)**2+S*S))*dx/2*1e9:.4f} nm")
