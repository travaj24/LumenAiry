"""An independent (eps, mu) Berreman 4x4 oracle, written for Phase E1 (the
shipped ``berreman_jones_1d`` takes a permittivity only).  Validated in
``e3m_mu.py oracle`` against ``berreman_jones_1d`` with mu = I (R, T and both
Jones, out-of-plane tensors, normal / oblique / conical) and against the
analytic isotropic (eps, mu) Airy slab.
"""
import numpy as np
from scipy.linalg import expm


def delta(eps, mu, Kx, Ky):
    """The 4 x 4 Berreman matrix of a homogeneous (eps, mu) medium for
    Psi = (Ex, Ey, Hx, Hy) (H in units of 1 / Z0): dPsi/dz = i k0 Delta Psi.
    Built column by column from curl E = i k0 mu H, curl H = -i k0 eps E
    (PUBLIC exp(-i w t)) with d/dx = i k0 Kx, d/dy = i k0 Ky, the two z
    rows eliminated."""
    eps = np.asarray(eps, dtype=complex)
    mu = np.asarray(mu, dtype=complex)
    D = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        Ex, Ey, Hx, Hy = np.eye(4)[j]
        Ez = (-(Kx * Hy - Ky * Hx) - eps[2, 0] * Ex - eps[2, 1] * Ey) \
            / eps[2, 2]
        Hz = (Kx * Ey - Ky * Ex - mu[2, 0] * Hx - mu[2, 1] * Hy) / mu[2, 2]
        mH = mu @ np.array([Hx, Hy, Hz])
        eE = eps @ np.array([Ex, Ey, Ez])
        D[0, j] = mH[1] + Kx * Ez          # dEx/dz / (i k0)
        D[1, j] = Ky * Ez - mH[0]          # dEy/dz
        D[2, j] = Kx * Hz - eE[1]          # dHx/dz
        D[3, j] = Ky * Hz + eE[0]          # dHy/dz
    return D


def _flux(v):
    return float(np.real(v[0] * np.conj(v[3]) - v[1] * np.conj(v[2])))


def split(D):
    """Forward (+z) and backward modes of a half-space's Delta: forward =
    Im q > 0, or real q with positive z-flux."""
    q, W = np.linalg.eig(D)
    fwd = [i for i in range(4) if (q[i].imag > 1e-12) or
           (abs(q[i].imag) <= 1e-12 and _flux(W[:, i]) > 0)]
    bwd = [i for i in range(4) if i not in fwd]
    assert len(fwd) == 2, q
    return W[:, fwd], W[:, bwd]


def berreman_mu(eps, mu, d, n_sub, n_sup, wl, theta=0.0, phi=0.0):
    """R, T (per incident lab Ex / Ey, flux-normalised) and the reflection /
    transmission Jones (rows = out [Ex; Ey], columns = in [Ex; Ey]) of ONE
    uniform (eps, mu) layer of thickness d between isotropic half-spaces
    (superstrate on the incidence side): the transfer matrix
    expm(i k0 Delta d)."""
    k0 = 2 * np.pi / wl
    Kx = n_sup * np.sin(theta) * np.cos(phi)
    Ky = n_sup * np.sin(theta) * np.sin(phi)
    I3 = np.eye(3, dtype=complex)
    Wf1, Wb1 = split(delta(n_sup ** 2 * I3, I3, Kx, Ky))
    Wf3, _Wb3 = split(delta(n_sub ** 2 * I3, I3, Kx, Ky))
    Pm = expm(1j * k0 * delta(eps, mu, Kx, Ky) * d)
    A = np.concatenate([Pm @ Wb1, -Wf3], axis=1)
    Jr, Jt = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    R, T = np.zeros(2), np.zeros(2)
    for col in range(2):
        a = np.linalg.solve(Wf1[:2], np.eye(2)[col])
        x = np.linalg.solve(A, -Pm @ (Wf1 @ a))
        vin, vr, vt = Wf1 @ a, Wb1 @ x[:2], Wf3 @ x[2:]
        Jr[:, col], Jt[:, col] = vr[:2], vt[:2]
        R[col] = -_flux(vr) / _flux(vin)
        T[col] = _flux(vt) / _flux(vin)
    return R, T, Jr, Jt
