"""T1: Wick moments vs brute-force; LG/HG orthonormality Gram."""
import sys, math
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.asymptotic import (
    gaussian_moment_2d, gaussian_moment_table_2d, lg_polynomial, hg_polynomial,
    evaluate_lg_mode, evaluate_hg_mode, decompose_lg, decompose_hg,
)

np.set_printoptions(precision=4, suppress=False, linewidth=150)

# ---------- 1. Wick moment vs brute-force 2-D quadrature -------------
rng = np.random.default_rng(7)
M = np.array([[2.3, 0.4], [0.4, 1.7]], dtype=complex) + 1j*np.array([[0.5, -0.2], [-0.2, 0.3]])
M = 0.5*(M + M.T)   # complex symmetric
Sigma = 0.5*np.linalg.inv(M)

# brute force: integral eta^a eta^b exp(-eta^T M eta) / (pi/sqrt(det M))
n = 4001
L = 14.0
e = np.linspace(-L, L, n)
EX, EY = np.meshgrid(e, e, indexing='xy')
quad = EX*EX*M[0,0] + 2*EX*EY*M[0,1] + EY*EY*M[1,1]
W = np.exp(-quad)
dA = (e[1]-e[0])**2
Z = np.pi/np.sqrt(np.linalg.det(M))
print("norm check: brute Z =", (W.sum()*dA), " analytic =", Z,
      " rel", abs(W.sum()*dA - Z)/abs(Z))
worst = 0.0
for a in range(0, 7):
    for b in range(0, 7-a):
        num = (EX**a * EY**b * W).sum()*dA / Z
        ana = gaussian_moment_2d(a, b, Sigma)
        den = max(abs(ana), 1e-30)
        rel = abs(num - ana)/den
        if (a+b) % 2 == 0 and abs(ana) > 1e-12:
            worst = max(worst, rel)
        if rel > 1e-6 and abs(ana) > 1e-12:
            print(f"  MISMATCH a={a} b={b}: brute {num:.8e} wick {ana:.8e} rel {rel:.2e}")
print("Wick vs brute worst relative (even orders, |ana|>1e-12):", worst)

# ---------- 2. LG orthonormality Gram matrix -------------------------
w = 1.0
Ngrid = 1201
ext = 7.0*w
x = np.linspace(-ext, ext, Ngrid)
y = np.linspace(-ext, ext, Ngrid)
X, Y = np.meshgrid(x, y, indexing='xy')
da = (x[1]-x[0])*(y[1]-y[0])
modes = [(p, l) for p in range(0, 3) for l in range(-2, 3)]
S = np.array([evaluate_lg_mode(p, l, w, X, Y) for (p, l) in modes])
G = np.einsum('mij,nij->mn', np.conj(S), S)*da
I = np.eye(len(modes))
print("\nLG Gram max |G - I| =", np.max(np.abs(G - I)))
d = np.abs(np.diag(G) - 1.0)
print("LG worst |<k,k>-1| =", d.max(), "at", modes[int(np.argmax(d))])
off = np.abs(G - np.diag(np.diag(G)))
print("LG worst off-diagonal =", off.max())
iu = np.unravel_index(np.argmax(off), off.shape)
print("   at pair", modes[iu[0]], modes[iu[1]], "value", G[iu])

# ---------- 3. HG orthonormality -------------------------------------
hmodes = [(m, n_) for m in range(4) for n_ in range(4)]
SH = np.array([evaluate_hg_mode(m, n_, w, w, X, Y) for (m, n_) in hmodes])
GH = np.einsum('mij,nij->mn', np.conj(SH), SH)*da
print("\nHG Gram max |G - I| =", np.max(np.abs(GH - np.eye(len(hmodes)))))

# ---------- 4. decompose_lg reconstruction ---------------------------
# random combination of modes -> recover amplitudes
amps = {k: complex(rng.normal(), rng.normal()) for k in modes}
F = np.zeros_like(X, dtype=complex)
for k, a in amps.items():
    F += a*evaluate_lg_mode(k[0], k[1], w, X, Y)
rec = decompose_lg(F, x, y, w, p_max=2, ell_max=2)
err = max(abs(rec[k]-amps[k]) for k in modes)
print("\ndecompose_lg recovery worst abs error:", err)
