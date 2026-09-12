"""Independent analytic oracles for the PMM-1D audit.

Conventions: exp(-i w t), forward exp(+i kz z), Im(n) > 0 absorbs.
"""
import numpy as np


def tmm_slab(n0, nl, ns, d, wl, theta0, pol):
    """Exact single-slab Fresnel/TMM.  Returns (r, t_amp, R, T).

    n0 = superstrate index, nl = layer index, ns = substrate index,
    d = layer thickness, wl = vacuum wavelength, theta0 = incidence angle (rad),
    pol in {'s','p'}.  Amplitude reference planes: both interfaces at z=0 and
    z=d, r referenced at the TOP interface (z=0), t referenced at the BOTTOM
    interface (z=d) -- i.e. the standard convention with no extra phase.
    """
    n0 = complex(n0); nl = complex(nl); ns = complex(ns)
    k0 = 2 * np.pi / wl
    kx = n0 * np.sin(theta0)

    def kz(n):
        v = np.sqrt(complex(n * n - kx * kx))
        return v if v.imag >= 0 else -v

    kz0, kzl, kzs = kz(n0), kz(nl), kz(ns)

    # tilted admittances
    if pol == 's':
        e0, el, es = kz0, kzl, kzs
    else:
        e0, el, es = kz0 / n0**2, kzl / nl**2, kzs / ns**2

    def fresnel(ea, eb):
        return (ea - eb) / (ea + eb), 2 * ea / (ea + eb)

    r01, t01 = fresnel(e0, el)
    r1s, t1s = fresnel(el, es)
    ph = np.exp(1j * kzl * k0 * d)
    r = (r01 + r1s * ph * ph) / (1 + r01 * r1s * ph * ph)
    t = (t01 * t1s * ph) / (1 + r01 * r1s * ph * ph)
    # power (amplitudes are Ey for s, Hy for p -- but with the admittance
    # definition above, both r/t are for the TANGENTIAL E for s and for
    # tangential H for p ... use the generic flux ratio):
    R = abs(r) ** 2
    if pol == 's':
        T = np.real(kzs) / np.real(kz0) * abs(t) ** 2
    else:
        T = np.real(kzs / ns**2) / np.real(kz0 / n0**2) * abs(t) ** 2
    return r, t, R, T


def tmm_stack(n_list, d_list, wl, theta0, pol):
    """N-layer TMM.  n_list = [n0, n1, ..., nN, nsub]; d_list = [d1..dN]."""
    n_list = [complex(x) for x in n_list]
    k0 = 2 * np.pi / wl
    kx = n_list[0] * np.sin(theta0)

    def kz(n):
        v = np.sqrt(complex(n * n - kx * kx))
        return v if v.imag >= 0 else -v

    kzs = [kz(n) for n in n_list]
    if pol == 's':
        eta = kzs
    else:
        eta = [kzs[i] / n_list[i] ** 2 for i in range(len(n_list))]

    # transfer-matrix over interfaces + phases
    M = np.eye(2, dtype=complex)
    for i in range(len(n_list) - 1):
        ea, eb = eta[i], eta[i + 1]
        rr = (ea - eb) / (ea + eb)
        tt = 2 * ea / (ea + eb)
        I = np.array([[1, rr], [rr, 1]], dtype=complex) / tt
        M = M @ I
        if i < len(d_list):
            ph = np.exp(-1j * kzs[i + 1] * k0 * d_list[i])
            P = np.array([[ph, 0], [0, 1 / ph]], dtype=complex)
            M = M @ P
    r = M[1, 0] / M[0, 0]
    t = 1.0 / M[0, 0]
    R = abs(r) ** 2
    T = np.real(eta[-1]) / np.real(eta[0]) * abs(t) ** 2
    return r, t, R, T


def lamellar_modes_te(eps1, eps2, f, period, wl, kx0, nmodes=40):
    """TRUE-MODE (Botten 1981) TE dispersion relation for a binary lamellar
    grating, solved by dense root scanning on the complex plane.

    Region 1 occupies [0, f*period) with eps1, region 2 the rest with eps2.
    Mode: E_y = A cos(k1 (x - x1c)) + ... ; the standard determinant:

      cos(kx0*period) = cos(k1 d1) cos(k2 d2)
                        - 0.5 (k1/k2 + k2/k1) sin(k1 d1) sin(k2 d2)

    with k_j^2 = k0^2 eps_j - gamma^2.  Returns gamma^2/k0^2 roots.
    """
    k0 = 2 * np.pi / wl
    d1 = f * period
    d2 = period - d1

    def F(u):           # u = (gamma/k0)^2
        k1 = np.sqrt(complex(k0 * k0 * (eps1 - u)))
        k2 = np.sqrt(complex(k0 * k0 * (eps2 - u)))
        # guard k=0
        if abs(k1) < 1e-30:
            k1 = 1e-30
        if abs(k2) < 1e-30:
            k2 = 1e-30
        return (np.cos(k1 * d1) * np.cos(k2 * d2)
                - 0.5 * (k1 / k2 + k2 / k1) * np.sin(k1 * d1) * np.sin(k2 * d2)
                - np.cos(kx0 * period))
    return F


def lamellar_modes_tm(eps1, eps2, f, period, wl, kx0):
    """TM (H_y) lamellar dispersion relation: same with the 1/eps weighting,
    i.e. k1/k2 -> (k1 eps2)/(k2 eps1)."""
    k0 = 2 * np.pi / wl
    d1 = f * period
    d2 = period - d1

    def F(u):
        k1 = np.sqrt(complex(k0 * k0 * (eps1 - u)))
        k2 = np.sqrt(complex(k0 * k0 * (eps2 - u)))
        if abs(k1) < 1e-30:
            k1 = 1e-30
        if abs(k2) < 1e-30:
            k2 = 1e-30
        a = (k1 * eps2) / (k2 * eps1)
        return (np.cos(k1 * d1) * np.cos(k2 * d2)
                - 0.5 * (a + 1.0 / a) * np.sin(k1 * d1) * np.sin(k2 * d2)
                - np.cos(kx0 * period))
    return F
