"""Independent 1-D RCWA oracle, written from first principles.
Convention: exp(-i omega t), forward wave exp(+i kz z), Im(kz)>=0.
Direct 4N boundary matching (NOT an S-matrix) for a single binary layer.
Li 1996 inverse rule for TM, Laurent for TE.
"""
import numpy as np

def _coeffs(c_lo, c_hi, duty, nk):
    """exact Fourier coeffs c_k, k=-(nk-1)..(nk-1) of profile
    = c_hi on [0,duty), c_lo on [duty,1)."""
    k = np.arange(-(nk-1), nk).astype(float)
    ramp = duty*np.sinc(k*duty)*np.exp(-1j*np.pi*k*duty)
    dc = (k == 0).astype(complex)
    return c_lo*dc + (c_hi-c_lo)*ramp

def _toep(c, M):
    N = 2*M+1
    ctr = (c.shape[0]-1)//2
    i = np.arange(N)
    return c[ctr + (i[:,None]-i[None,:])]

def _sq(x):
    r = np.sqrt(np.asarray(x, dtype=complex))
    return np.where(r.imag < 0, -r, r)      # Im>=0

def oracle_1d(period, n_ridge, n_groove, n_sub, n_sup, depth, duty, wl,
              angle=0.0, pol="te", M=11):
    k0 = 2*np.pi/wl
    N = 2*M+1
    m = np.arange(-M, M+1)
    eps_r = complex(n_ridge)**2; eps_g = complex(n_groove)**2
    eps_I = complex(n_sup)**2;   eps_II = complex(n_sub)**2
    a0 = np.real(complex(n_sup))*np.sin(angle)
    alpha = a0 + m*(wl/period)                 # kx/k0
    Kx = np.diag(alpha.astype(complex))
    gI  = _sq(eps_I  - alpha**2)
    gII = _sq(eps_II - alpha**2)
    g0  = _sq(eps_I - a0**2).real
    nk = 2*M+1
    E   = _toep(_coeffs(eps_g, eps_r, duty, nk), M)          # [[eps]]
    Ei  = _toep(_coeffs(1/eps_g, 1/eps_r, duty, nk), M)      # [[1/eps]]
    I = np.eye(N, dtype=complex)
    delta = (m == 0).astype(complex)
    if pol == "te":
        A = Kx@Kx - E
        q2, W = np.linalg.eig(A)
        gam = _sq(-q2)
        G = np.diag(gam)
        Vm = W@G
        # region "V" operators: V = (i/k0) dE/dz  -> forward: -gam, backward:+gam
        VI, VII = np.diag(gI), np.diag(gII)
        src_V = -g0*delta
    else:
        B = np.linalg.inv(Ei)@(Kx@np.linalg.inv(E)@Kx - I)
        q2, W = np.linalg.eig(B)
        gam = _sq(-q2)
        G = np.diag(gam)
        Vm = Ei@W@G
        VI, VII = np.diag(gI/eps_I), np.diag(gII/eps_II)
        src_V = -(g0/eps_I)*delta
    X = np.diag(np.exp(1j*k0*gam*depth))
    Z = np.zeros((N, N), dtype=complex)
    # unknowns [r, a, b, t]
    Amat = np.block([
        [ -I,   W,      W@X,   Z   ],   # z=0 field:  delta + r = W(a+Xb)
        [ -VI, -Vm,     Vm@X,  Z   ],   # z=0 V:  src_V + VI r = Vm(-a+Xb)
        [  Z,   W@X,    W,    -I   ],   # z=d field:  W(Xa+b) = t
        [  Z,  -Vm@X,   Vm,    VII ],   # z=d V:  Vm(-Xa+b) = -VII t
    ])
    rhs = np.concatenate([delta, src_V, np.zeros(N), np.zeros(N)])
    sol = np.linalg.solve(Amat, rhs)
    r = sol[:N]; t = sol[3*N:]
    if pol == "te":
        DEr = np.real(gI/g0)*np.abs(r)**2
        DEt = np.real(gII/g0)*np.abs(t)**2
    else:
        DEr = np.real(gI/eps_I)/ (g0/eps_I.real) * np.abs(r)**2
        DEt = np.real(gII/eps_II)/(g0/eps_I.real) * np.abs(t)**2
    DEr = np.where(np.real(gI) > 1e-14, DEr, 0.0)
    DEt = np.where(np.real(gII) > 1e-14, DEt, 0.0)
    return m, np.real(DEr), np.real(DEt), r, t
