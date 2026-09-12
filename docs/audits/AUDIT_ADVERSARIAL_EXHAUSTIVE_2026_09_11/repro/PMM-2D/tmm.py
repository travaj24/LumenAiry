import numpy as np

def tmm_single_layer(n0, n1, n2, d, wl, theta):
    """Independent 3-medium TMM. Returns (r_x, r_y) = reflected Cartesian
    amplitude ratios at phi=0 with unit TANGENTIAL incident E component:
      r_y  == classic Fresnel r_s
      r_x  == E_x^ref / E_x^inc  (TM; equals r_s at normal incidence)
    exp(-i w t), forward exp(+i k z)."""
    k0 = 2*np.pi/wl
    n0, n1, n2 = complex(n0), complex(n1), complex(n2)
    kt = n0*np.sin(theta)
    def kz(n):
        v = np.sqrt(complex(n**2 - kt**2))
        return v if v.imag >= 0 else -v
    kz0, kz1, kz2 = kz(n0), kz(n1), kz(n2)
    # s
    rs01 = (kz0-kz1)/(kz0+kz1); rs12 = (kz1-kz2)/(kz1+kz2)
    # x (TM): Z = kz/eps ; r_x(ij) = (Zj - Zi)/(Zj + Zi)
    Z0, Z1, Z2 = kz0/n0**2, kz1/n1**2, kz2/n2**2
    rx01 = (Z1-Z0)/(Z1+Z0); rx12 = (Z2-Z1)/(Z2+Z1)
    ph = np.exp(2j*k0*kz1*d)
    r_s = (rs01 + rs12*ph)/(1 + rs01*rs12*ph)
    r_x = (rx01 + rx12*ph)/(1 + rx01*rx12*ph)
    return r_x, r_s
