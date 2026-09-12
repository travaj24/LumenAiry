"""RW test 5: verify the proposed one-line fix (fft2 -> ifft2*N^2)."""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.vector_diffraction as vd

lam = 633e-9; k = 2*np.pi/lam
NA, f, Np, dxp, Nf = 0.3, 2e-3, 1024, 1.5e-6, 1024
a = f*NA; W = 0.35; g = 0.45


def pupil_fn(rho, phi):
    return np.exp(-(rho/g)**2)*np.exp(2j*np.pi*W*(3*rho**3-2*rho)*np.cos(phi))


x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
R = np.hypot(X, Y); PH = np.arctan2(Y, X)
pupil = np.where(R <= a, pupil_fn(R/a, PH), 0.0).astype(np.complex128)

Ex, Ey, Ez, xf, yf = vd.richards_wolf_focus(pupil, lam, NA, f, dxp,
                                            N_focal=Nf, polarization='x')
dxf = lam*f/(Nf*dxp); c = Nf//2

# --- monkeypatch np.fft.fft2 *inside the module namespace* is not possible
# (it calls np.fft.fft2 directly), so emulate the fixed kernel here by
# re-deriving the field from the module output: ifft2*N^2 vs fft2 differ
# exactly by (x,y) -> (-x,-y) on a centred grid, so the fix is equivalent
# to reversing both focal axes with the fftshift index convention.
def flip_centred(A):
    """Map A[i,j] (focal pt (xf[j], yf[i])) -> value at (-xf[j], -yf[i])."""
    N = A.shape[0]
    return np.roll(A[::-1, ::-1], shift=(1, 1), axis=(0, 1))


Exf, Eyf, Ezf = flip_centred(Ex), flip_centred(Ey), flip_centred(Ez)


def oracle(xP, yP, zP, nth=600, nph=1024):
    th_max = np.arcsin(NA)
    tn, tw = np.polynomial.legendre.leggauss(nth)
    th = 0.5*th_max*(tn+1.0); thw = 0.5*th_max*tw
    ph = np.arange(nph)*(2*np.pi/nph); phw = 2*np.pi/nph
    TH, PHI = np.meshgrid(th, ph, indexing='ij')
    Wt = thw[:, None]*phw
    st, ct = np.sin(TH), np.cos(TH); cp, sp = np.cos(PHI), np.sin(PHI)
    A = pupil_fn(f*st/a, PHI)
    base = A*np.sqrt(ct)*np.exp(1j*k*(xP*st*cp+yP*st*sp+zP*ct))*st*Wt
    pre = (-1j*k*f/(2*np.pi))*np.exp(1j*k*f)
    return np.array([pre*np.sum(base*(cp**2*ct+sp**2)),
                     pre*np.sum(base*(cp*sp*(ct-1.0))),
                     pre*np.sum(base*(-cp*st))])


print("after reversing both focal axes (== replacing fft2 by ifft2*N^2):")
for (dy_, dx_) in [(0, 1), (0, 2), (1, 2), (-2, 3), (2, -1), (0, -2)]:
    ov = oracle(dx_*dxf, dy_*dxf, 0.0)
    mv = np.array([Exf[c+dy_, c+dx_], Eyf[c+dy_, c+dx_], Ezf[c+dy_, c+dx_]])
    print(f"  ({dy_:+d},{dx_:+d})  relL2 = "
          f"{np.linalg.norm(mv-ov)/np.linalg.norm(ov):.3e}")

# direct check that ifft2*N^2 == the axis-reversal, on a random array
rng = np.random.default_rng(0)
A = rng.normal(size=(64, 64)) + 1j*rng.normal(size=(64, 64))
fa = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))
ia = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A)))*A.size
print(f"\n||ifft2*N^2 - flip(fft2)||/||.|| = "
      f"{np.linalg.norm(ia-flip_centred(fa))/np.linalg.norm(fa):.3e}")
