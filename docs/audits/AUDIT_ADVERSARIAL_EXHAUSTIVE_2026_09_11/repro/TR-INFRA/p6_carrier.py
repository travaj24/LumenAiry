"""TR-INFRA probe 6: _compute_carrier -- 'auto' fit conditioning + sign
convention + scalar branch + TiltedCarrier normalisation."""
import warnings
import numpy as np
from lumenairy.elements import _lens_traced as T

lam = 1.31e-6; k0 = 2*np.pi/lam

def grid(N, dx):
    x = (np.arange(N) - N/2)*dx
    X, Y = np.meshgrid(x, x, indexing='xy')   # matches _geometric_lens_phase
    return x, X, Y

print("=== 6a  'auto' carrier: recovery of a KNOWN spherical wavefront ===")
N = 512; dx = 8e-6
x, X, Y = grid(N, dx)
w0 = 0.6e-3
for Rc in (0.20, 0.05, -0.05):
    # exact point-source sphere at signed distance Rc
    sgn = np.sign(Rc); r2 = X**2+Y**2
    W_true = sgn*(r2/(np.sqrt(r2+Rc*Rc)+abs(Rc)))
    E = np.exp(-(r2)/w0**2).astype(np.complex128)*np.exp(1j*k0*W_true)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        W, gfn, wfn = T._compute_carrier('auto', E, lam, dx, X, Y, auto_degree=2)
    # fitted curvature: W ~ r^2/(2 R_fit) -> compare gradients over the beam
    Lq, Mq = gfn(X, Y)
    m = (r2 <= (1.5*w0)**2)
    # true gradient
    rho = np.sqrt(r2+Rc*Rc); Lt = sgn*X/rho; Mt = sgn*Y/rho
    print(f"  R={Rc:+.3f} m : max|L_fit-L_true| over beam = {np.abs(Lq-Lt)[m].max():.3e}"
          f"   rms = {np.sqrt(np.mean((Lq-Lt)[m]**2)):.3e}   (peak |L_true| = {np.abs(Lt)[m].max():.3e})")
    # W sign check: does the fitted W reproduce the field phase (exp(+i k0 W))?
    resid = np.angle(E*np.exp(-1j*k0*W))[m]
    print(f"            residual phase after removing fitted W: rms={np.sqrt(np.mean(resid**2)):.3e} rad,"
          f" peak={np.abs(resid).max():.3e} rad")

print("\n=== 6b  'auto' fit design-matrix conditioning (RAW metre monomials) ===")
# replicate the A matrix the 'auto' branch builds, for several auto_degree
def auto_A(N, dx, deg, w0):
    x = (np.arange(N)-N/2)*dx
    Xg, Yg = np.meshgrid(x, x, indexing='xy')
    r2 = Xg**2+Yg**2
    m = np.exp(-r2/w0**2) > 0.05
    xL = Xg[m]; yL = Yg[m]
    terms=[(i,j) for d in range(1,deg+1) for i in range(d+1) for j in [d-i]]
    n=xL.size
    A=np.zeros((2*n,len(terms)))
    for k,(i,j) in enumerate(terms):
        A[:n,k]=(i*xL**(i-1)*yL**j) if i>=1 else 0.0
        A[n:,k]=(j*xL**i*yL**(j-1)) if j>=1 else 0.0
    return A, terms
for deg in (2,3,4,6):
    A,terms = auto_A(256, 8e-6, deg, 0.6e-3)
    G=A.T@A
    print(f"  auto_degree={deg} M={len(terms):2d}: cond(A)={np.linalg.cond(A):.4e}"
          f"  cond(G)={np.linalg.cond(G):.4e}  _gram_rcond={T._gram_rcond(G):.4e}"
          f"  -> C13 screen {'FIRES' if T._gram_rcond(G)<T._LSTSQ_GRAM_RCOND_MIN else 'passes'}")

print("\n=== 6c  TiltedCarrier: W(x0,y0)==0, grad W(x0,y0)==(L,M), |dir| ===")
for spec in (T.TiltedCarrier(R=0.05, L=0.05, M=-0.03, x0=1e-3, y0=-2e-3),
             T.TiltedCarrier(R=-0.05, L=0.2, M=0.1),
             T.TiltedCarrier(R=np.inf, L=0.05, M=0.02, x0=3e-4)):
    Wq, Lq, Mq = T._tilted_carrier_parts(spec, np.array([spec.x0]), np.array([spec.y0]))
    n2 = Lq**2+Mq**2
    print(f"  R={spec.R:+.4g} L={spec.L} M={spec.M}: W(x0,y0)={float(Wq):.3e}  "
          f"grad=({float(Lq):.10f},{float(Mq):.10f}) vs ({spec.L},{spec.M})  "
          f"|grad|^2={float(n2):.6f}")
    # eikonal equation check |grad W|^2 + N^2 = 1  (exact point-source eikonal)
    xs=np.linspace(-2e-3,2e-3,5); Xg,Yg=np.meshgrid(xs,xs)
    Wg,Lg,Mg=T._tilted_carrier_parts(spec,Xg,Yg)
    print(f"     max ||grad W| - 1| over a 4 mm patch = {np.abs(np.hypot(Lg,Mg)).max():.6f}"
          f"  (must be <1; eikonal |grad W|^2 <= 1)")
