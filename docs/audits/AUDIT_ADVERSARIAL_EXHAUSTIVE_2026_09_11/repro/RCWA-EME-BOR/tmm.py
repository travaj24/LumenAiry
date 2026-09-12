"""Independent TMM, exp(-i omega t), forward exp(+i kz z), Im(kz)>=0.
Returns complex r,t for BOTH the standard Fresnel s/p amplitude convention
(r_p(normal) = -r_s(normal)) and the TANGENTIAL-component ratios
(Ex_out/Ex_in, Ey_out/Ey_in) which is what an RCWA Jones matrix reports.
Reference planes: r at the TOP of the stack (z=0), t at the BOTTOM (z=sum d).
"""
import numpy as np

def _sq(x):
    r = np.sqrt(np.asarray(x, dtype=complex))
    return np.where(r.imag < 0, -r, r)

def tmm(n_list, d_list, wl, theta0=0.0, pol="s"):
    n = np.asarray(n_list, dtype=complex)
    k0 = 2*np.pi/wl
    inv = n[0]*np.sin(theta0)                     # Snell invariant
    kz = k0*_sq(n**2 - inv**2)                    # per layer
    # interface Fresnel (a -> b), standard convention
    def rt(a, b):
        if pol == "s":
            r = (kz[a]-kz[b])/(kz[a]+kz[b])
            t = 2*kz[a]/(kz[a]+kz[b])
        else:
            r = (n[b]**2*kz[a]-n[a]**2*kz[b])/(n[b]**2*kz[a]+n[a]**2*kz[b])
            t = 2*n[a]*n[b]*kz[a]/(n[b]**2*kz[a]+n[a]**2*kz[b])
        return r, t
    L = len(n)
    # transfer-matrix recursion (Born&Wolf / Byrnes tmm)
    M = np.eye(2, dtype=complex)
    for j in range(L-1):
        r, t = rt(j, j+1)
        I = np.array([[1, r], [r, 1]], dtype=complex)/t
        M = M @ I
        if j+1 < L-1:
            delta = kz[j+1]*d_list[j]             # d_list indexes inner layers
            P = np.array([[np.exp(-1j*delta), 0], [0, np.exp(1j*delta)]])
            M = M @ P
    r_tot = M[1, 0]/M[0, 0]
    t_tot = 1.0/M[0, 0]
    return r_tot, t_tot, kz, n

def tmm_jones(n_list, d_list, wl, theta0=0.0):
    """Tangential-field ratios: (rxx, ryy, txx, tyy) = (Ex_r/Ex_i, Ey_r/Ey_i,
    Ex_t/Ex_i, Ey_t/Ey_i) with the s/p Fresnel results converted."""
    rs, ts, kz, n = tmm(n_list, d_list, wl, theta0, "s")
    rp, tp, _, _ = tmm(n_list, d_list, wl, theta0, "p")
    # angles
    inv = n[0]*np.sin(theta0)
    cos0 = kz[0]/(2*np.pi/wl*n[0]); cosN = kz[-1]/(2*np.pi/wl*n[-1])
    # p basis: incident (cos0,0,-sin0); reflected (cos0,0,+sin0);
    #          transmitted (cosN,0,-sinN)
    rxx = rp                    # cos0/cos0
    txx = tp*cosN/cos0
    ryy = rs
    tyy = ts
    return dict(rs=rs, ts=ts, rp=rp, tp=tp, rxx=rxx, ryy=ryy, txx=txx, tyy=tyy)
