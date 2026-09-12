"""TR-INFRA probe 1: _Cheb2DEvaluator conditioning, derivatives, extrapolation,
numba-vs-numpy agreement."""
import numpy as np
from numpy.polynomial import chebyshev as C
from lumenairy.elements import _lens_traced as T

np.set_printoptions(precision=4, suppress=False)

# ---- 1a: conditioning of the design matrix on DISC-masked samples ----------
def build_A(xs, order, disc_frac=None, weights_mode=None):
    """Replicate _Cheb2DEvaluator.__init__'s design matrix."""
    xmin, xmax = xs.min(), xs.max()
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    u = (2.0*X - (xmin+xmax))/(xmax-xmin)
    v = (2.0*Y - (xmin+xmax))/(xmax-xmin)
    mi = [(kx, ky) for kx in range(order+1) for ky in range(order+1-kx)]
    K1 = np.array([m[0] for m in mi]); K2 = np.array([m[1] for m in mi])
    Tu = T._cheb_vand_2d(u, order, np); Tv = T._cheb_vand_2d(v, order, np)
    A = (Tu[K1]*Tv[K2]).reshape(len(mi), -1).T
    return A, mi

print("=== 1a  cond(A) and cond(G=A^T A) on a launch square vs a disc mask ===")
n = 129
R = 1.0
xs = np.linspace(-R, R, n)
X, Y = np.meshgrid(xs, xs, indexing='ij')
for frac in (None, 0.5, 0.3):
    for order in (6, 8, 10, 12):
        A, mi = build_A(xs, order)
        if frac is None:
            Am = A; lab = 'full square'
        else:
            disc = (X**2+Y**2) <= (frac*R)**2
            Am = A[disc.ravel(), :]; lab = 'hard disc r<=%.2fR (n_in=%d)' % (frac, disc.sum())
        cA = np.linalg.cond(Am)
        G = Am.T@Am
        cG = np.linalg.cond(G)
        rc = T._gram_rcond(G)
        print(f"  order {order:2d} M={len(mi):3d} {lab:35s} cond(A)={cA:9.3e} cond(G)={cG:9.3e} gram_rcond={rc:9.3e}")
    print()

print("=== 1a'  D1 WEIGHTED restriction (out-of-disc w = sqrt(1e-8 n_in/n_out)) ===")
for frac in (0.5, 0.3):
    for order in (6, 10):
        A, mi = build_A(xs, order)
        disc = (X**2+Y**2) <= (frac*R)**2
        w, ordr = T._decentred_fit_restriction(disc, True, 6, order)
        Aw = A * w.ravel()[:, None]
        cA = np.linalg.cond(Aw); G = Aw.T@Aw
        print(f"  order {ordr:2d} disc {frac}: cond(A_w)={cA:9.3e} cond(G)={np.linalg.cond(G):9.3e} "
              f"gram_rcond={T._gram_rcond(G):9.3e}  w_out={w.min():.3e}")
