"""Probe 19: routing decisions only (no propagation) for apply_real_lens_auto /
apply_real_lens_universal over 4 regimes."""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter("ignore")
import lumenairy as la
from lumenairy.propagators.fga import (_universal_route, _caustic_zone, _system_na,
                                       _tilt_dispersion, _sag_screen_aberration_rad,
                                       _default_p_max, _ABERRATION_MAX_RAD,
                                       _SEIDEL_SA_MAX_RAD)
lam = 1.0e-6; N = 192; dx = 2.0e-6
x = (np.arange(N)-N/2)*dx; X,Y = np.meshgrid(x,x); R2 = X*X+Y*Y
k = 2*np.pi/lam
slow = la.make_singlet(100e-3,-100e-3,3e-3,'N-BK7',aperture=0.20e-3)      # F/250
fast = la.make_singlet(1.2e-3,-1.2e-3,0.8e-3,'N-BK7',aperture=0.30e-3)    # fast
def gauss(w, tilt=0.0, curv=0.0, dec=0.0):
    E = np.exp(-((X-dec)**2+Y**2)/w**2).astype(complex)
    if tilt: E = E*np.exp(1j*k*tilt*X)
    if curv: E = E*np.exp(1j*k*R2/(2*curv))
    return E
import lumenairy.propagators.fga as F
for nm, presc, E, opd in (
        ("collimated / slow lens, exit plane", slow, gauss(80e-6), 0.0),
        ("collimated / fast lens, AT FOCUS  ", fast, gauss(100e-6), 1.027e-3),
        ("tilted input / fast lens, at focus", fast, gauss(100e-6, tilt=0.05), 1.027e-3),
        ("decentred beam / fast lens, focus ", fast, gauss(60e-6, dec=80e-6), 1.027e-3),
        ("multi-valued (2 tilted beams)     ", fast, gauss(100e-6,tilt=0.05)+gauss(100e-6,tilt=-0.05), 1.027e-3)):
    na = _system_na(presc, lam)
    zone = _caustic_zone(E, dx, presc, lam)
    ab  = _sag_screen_aberration_rad(E, dx, dx, presc, lam)
    td  = _tilt_dispersion(E, dx, dx, lam, na)
    u = _universal_route(E, presc, lam, dx, dx, opd, 0.12, 3.0, None, 0.06,
                         _ABERRATION_MAX_RAD, _SEIDEL_SA_MAX_RAD)
    # apply_real_lens_auto's 2-way choice, reproduced without propagating
    ch2 = 'gbd'
    if zone is not None:
        na2 = max(_default_p_max(presc, lam)/1.6, 1e-3)
        pad = 3.0*lam/(na2*na2)
        if (zone[0]-pad) <= opd <= (zone[1]+pad): ch2 = 'fga'
    zs = "None" if zone is None else f"[{zone[0]*1e3:.3f},{zone[1]*1e3:.3f}]mm"
    print(f"{nm}: NA={na:.4f} aberr={ab:7.1f}rad tiltdisp={td:.4f} caustic={zs}"
          f"  -> universal='{u}'  auto(2way)='{ch2}'")
