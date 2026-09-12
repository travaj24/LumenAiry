"""Shared fixtures for TR-MAIN-2 probes."""
import numpy as np
import sys, os
REPO = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WL = 1.31e-6

def register_glass():
    import lumenairy as la
    from lumenairy import glass as _g
    try:
        _g.GLASS_REGISTRY['_TRM2_GLASS'] = (lambda wl: 1.5168)
    except Exception as e:
        print('register_glass:', type(e).__name__, e)

def singlet_f5(ap=24e-3, t=5e-3, R=51.68e-3):
    return {
        'wavelength': WL,
        'aperture_diameter': ap,
        'surfaces': [
            {'radius': R, 'thickness': t,
             'glass_before': 'air', 'glass_after': '_TRM2_GLASS',
             'semi_diameter': ap/2},
            {'radius': -R, 'thickness': 0.0,
             'glass_before': '_TRM2_GLASS', 'glass_after': 'air',
             'semi_diameter': ap/2},
        ],
        'thicknesses': [t],
        'stop_index': 0,
    }

def gauss(N, dx, w0, dtype=np.complex128, tilt=(0.0, 0.0), wl=WL):
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X**2 + Y**2)/w0**2)
    if tilt != (0.0, 0.0):
        k = 2*np.pi/wl
        E = E * np.exp(1j*k*(tilt[0]*X + tilt[1]*Y))
    return E.astype(dtype)

def tophat(N, dx, r, dtype=np.complex128):
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    return ((X**2 + Y**2) <= r**2).astype(dtype)

def small_singlet(ap=2.0e-3, t=1.0e-3, R=25.84e-3):
    """f ~ 25 mm biconvex, 2 mm aperture, f/12.5 -> NA 0.04."""
    return {
        'wavelength': WL,
        'aperture_diameter': ap,
        'surfaces': [
            {'radius': R, 'thickness': t,
             'glass_before': 'air', 'glass_after': '_TRM2_GLASS',
             'semi_diameter': ap/2},
            {'radius': -R, 'thickness': 0.0,
             'glass_before': '_TRM2_GLASS', 'glass_after': 'air',
             'semi_diameter': ap/2},
        ],
        'thicknesses': [t],
        'stop_index': 0,
    }
