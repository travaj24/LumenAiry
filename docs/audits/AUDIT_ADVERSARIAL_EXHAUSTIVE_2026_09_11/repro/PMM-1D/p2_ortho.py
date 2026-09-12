"""PROBE 2: mode normalisation / orthogonality / biorthogonality.

For the SEM pencil A x = q^2 B x:
  TE:  A = Peps - Lop/k0^2,  B = S0
  TM:  A = S0   - Lop/k0^2,  B = Pinv
Lossless + real kx0 -> A, B Hermitian (C real antisymmetric => -2i kx0 C is
Hermitian), so eigenvectors are B-orthogonal in the CONJUGATED sense.
Lossy -> neither Hermitian nor complex-symmetric at kx0 != 0; the correct
pairing is with the ADJOINT problem A^T = A(-kx0).
"""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc


def build(eps_r, eps_g, duty, period, wl, kx0, pol, degree):
    mats = pc._build_sem(period, duty * period, eps_r, eps_g, degree, 1, 1,
                         False)
    k0 = 2 * np.pi / wl
    k02 = k0 * k0
    if pol == "te":
        Lop = mats["L"]
        if kx0:
            Lop = (Lop - 1j * kx0 * (mats["C"] - mats["C"].T)
                   + kx0 * kx0 * mats["S0"])
        A, B = mats["Peps"] - Lop / k02, mats["S0"]
    else:
        Lop = mats["Linv"]
        if kx0:
            Lop = (Lop - 1j * kx0 * (mats["Cinv"] - mats["Cinv"].T)
                   + kx0 * kx0 * mats["Pinv"])
        A, B = mats["S0"] - Lop / k02, mats["Pinv"]
    W, lam, q, invop = pc._sem_modes(mats, k0, pol, kx0, False)
    return mats, A, B, W, q


def rep(label, M):
    n = M.shape[0]
    D = np.abs(np.diag(M))
    O = np.abs(M - np.diag(np.diag(M)))
    print(f"    {label}: max|offdiag|/max|diag| = "
          f"{O.max()/max(D.max(),1e-300):.3e}")


wl = 1.0e-6
period = 1.0e-6
for label, e1, e2, angd in (
        ("LOSSLESS Si/air, normal", 3.48**2, 1.0, 0.0),
        ("LOSSLESS Si/air, 25 deg", 3.48**2, 1.0, 25.0),
        ("LOSSY  Au-ish/air, normal", (0.55 + 11.5j) ** 2, 1.0, 0.0),
        ("LOSSY  Au-ish/air, 25 deg", (0.55 + 11.5j) ** 2, 1.0, 25.0)):
    print("=" * 74)
    print(label)
    k0 = 2 * np.pi / wl
    kx0 = np.sin(np.deg2rad(angd)) * k0
    for pol in ("te", "tm"):
        mats, A, B, W, q = build(e1, e2, 0.5, period, wl, kx0, pol, 16)
        print(f"  {pol}: A Hermitian? {np.max(np.abs(A - A.conj().T)):.3e}  "
              f"A symmetric? {np.max(np.abs(A - A.T)):.3e}  "
              f"|Im q^2|max = {np.max(np.abs((q**2).imag)):.3e}")
        # normalise columns
        Wn = W / np.linalg.norm(W, axis=0)
        rep("W^H B W (Hermitian)", Wn.conj().T @ B @ Wn)
        rep("W^T B W (unconj.)  ", Wn.T @ B @ Wn)
        # adjoint problem at -kx0
        _, A2, B2, W2, q2 = build(e1, e2, 0.5, period, wl, -kx0, pol, 16)
        # match modes by q^2
        idx = []
        for v in q ** 2:
            idx.append(int(np.argmin(np.abs(q2 ** 2 - v))))
        W2m = W2[:, idx] / np.linalg.norm(W2[:, idx], axis=0)
        rep("W2(-kx0)^T B W (adj)", W2m.T @ B @ Wn)
