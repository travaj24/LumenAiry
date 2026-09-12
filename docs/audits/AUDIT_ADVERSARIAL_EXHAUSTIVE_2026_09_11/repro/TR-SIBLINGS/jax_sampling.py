"""apply_real_lens_traced_jax OPD accuracy vs an INDEPENDENT exact ray trace,
as a function of the launch-lattice size implied by (dx, ray_subsample) at the
library-default cheb_order=10.  No guard warns when the fit is starved."""
import sys, warnings, numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
from lumenairy.glass import get_glass_index
from lumenairy.elements._lens_jax import apply_real_lens_traced_jax as TJ

lam = 0.5876e-6; k0 = 2*np.pi/lam
ng = float(get_glass_index('N-BK7', lam))
R1, R2, d, AP = 25e-3, float('inf'), 3e-3, 6.0e-3
rx = {'name': 's', 'aperture_diameter': AP,
      'surfaces': [{'radius': R1, 'conic': 0.0, 'aspheric_coeffs': None, 'radius_y': None,
                    'conic_y': None, 'aspheric_coeffs_y': None,
                    'glass_before': 'air', 'glass_after': 'N-BK7'},
                   {'radius': R2, 'conic': 0.0, 'aspheric_coeffs': None, 'radius_y': None,
                    'conic_y': None, 'aspheric_coeffs_y': None,
                    'glass_before': 'N-BK7', 'glass_after': 'air'}],
      'thicknesses': [d]}
# exact per-pixel OPD oracle: for a rot-sym collimated input the exit-vertex
# landing radius rho and the OPL are a 1-D map; invert it exactly.
hh = np.linspace(0.0, 0.5*AP*1.02, 400001); hh[0] = 1e-12
xo, opl, L, Nz = trace_singlet(hh, R1, R2, d, ng, 0.0)
rho = np.abs(xo)
o0 = np.interp(0.0, rho, opl-opl[0])
def exact_opd(r):
    return np.interp(r, rho, opl-opl[0], left=0.0, right=np.nan)

print('%-9s %-9s %-9s %-12s %-14s %s' % ('N','dx[um]','sub','n_launch','samples/term','OPD rms [waves]  max [waves]'))
for (N, dxv, sub) in ((128, 60e-6, 8), (128, 30e-6, 8), (256, 30e-6, 8), (256, 15e-6, 8),
                      (256, 30e-6, 4), (256, 30e-6, 2), (256, 30e-6, 1)):
    x = (np.arange(N)-N/2)*dxv; X, Y = np.meshgrid(x, x); r = np.hypot(X, Y)
    E = np.ones((N, N), dtype=np.complex128)
    lr = 0.5*AP*1.02
    nl = max(8, int(2*lr/(dxv*sub)));  nl += (nl % 2 == 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eo = np.asarray(TJ(E, prescription=rx, wavelength=lam, dx=dxv,
                           ray_subsample=sub, cheb_order=10, newton_iters=14))
    m = (np.abs(Eo) > 0) & (r < 0.45*AP)
    ph = np.angle(Eo[m])
    ref = np.angle(np.exp(1j*k0*exact_opd(r[m])))
    dphi = np.angle(np.exp(1j*(ph-ref)))
    dphi = np.angle(np.exp(1j*(dphi-np.angle(np.sum(np.exp(1j*dphi))))))
    print('%-9d %-9.4g %-9d %-12d %-14.1f %.5g   %.5g'
          % (N, dxv*1e6, sub, nl, nl*nl/66.0,
             np.sqrt(np.mean(dphi**2))/(2*np.pi), np.abs(dphi).max()/(2*np.pi)))
