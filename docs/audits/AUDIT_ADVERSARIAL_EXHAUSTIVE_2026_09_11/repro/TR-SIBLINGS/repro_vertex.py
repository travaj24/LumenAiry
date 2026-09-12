"""P0 repro: _lens_traced_multibranch._trace_launch_grid omits the
EXIT-VERTEX correction that _lens_traced.py (:9583) and _lens_jax.py (:559)
both apply.  The exit rays are left at the last surface's SAG point, so the
"output plane" is really the curved surface z = sag(rho) + output_plane_distance.
"""
import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
import lumenairy
from lumenairy import raytrace as rt
from lumenairy.elements._lens_traced_multibranch import _trace_launch_grid
from lumenairy.glass import get_glass_index

lam = 0.5876e-6
k0 = 2*np.pi/lam

def report(R1, R2, d, ap, d_out, tag):
    rx = lumenairy.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=ap)
    ng = float(get_glass_index('N-BK7', lam))
    lr = 0.5*ap*0.98
    n_launch = 401
    g = _trace_launch_grid(rx, lam, lr, n_launch, d_out, 1.0)
    # meridional row j = centre (y_in = 0): Xi[i,j]=xs_in[i], Yi[i,j]=xs_in[j]
    jc = n_launch//2
    assert abs(g['Yi'][0, jc]) < 1e-15
    rho = g['Xi'][:, jc]
    xo_lib = g['x_out'][:, jc]
    opl_lib = g['opl'][:, jc]
    alive = g['alive'][:, jc]
    # ORACLE with the exit-vertex correction: trace to z = d + d_out measured
    # from the ENTRANCE plane, i.e. d_out past the exit VERTEX.
    xo_ref, opl_ref, L, Nz = trace_singlet(rho, R1, R2, d, ng, d_out)
    m = alive & np.isfinite(xo_lib) & (rho > 0)
    dx_err = np.abs(xo_lib[m]-xo_ref[m])
    # OPL on-axis referenced (that is how the phase is used)
    i0 = np.argmin(np.abs(rho))
    dop = (opl_lib-opl_lib[i0]) - (opl_ref-opl_ref[i0])
    print('%-42s  |dx_out| max %10.4g um   OPD err max %10.4g waves'
          % (tag, dx_err.max()*1e6, np.nanmax(np.abs(dop[m]))/lam))
    return rho, xo_lib, xo_ref, opl_lib, opl_ref, m

print('== plano-convex, FLAT rear (sag == 0): no error expected ==')
report(25e-3, float('inf'), 3e-3, 10e-3, 0.0,   'R2=inf  d_out=0')
report(25e-3, float('inf'), 3e-3, 10e-3, 45e-3, 'R2=inf  d_out=45mm')
print()
print('== curved rear surface (sag != 0) ==')
for (R1,R2,d,ap,dd,tag) in [
    (float('inf'), -25e-3, 5e-3, 20e-3, 0.0,   'R1=inf R2=-25mm ap20 d_out=0'),
    (float('inf'), -25e-3, 5e-3, 20e-3, 40e-3, 'R1=inf R2=-25mm ap20 d_out=40mm'),
    (float('inf'), -100e-3, 5e-3, 20e-3, 0.0,  'R1=inf R2=-100mm ap20 d_out=0'),
    (float('inf'), -100e-3, 5e-3, 20e-3, 190e-3,'R1=inf R2=-100mm ap20 d_out=190mm'),
    (50e-3, -50e-3, 4e-3, 12e-3, 0.0,          'biconvex R=+-50mm ap12 d_out=0'),
    (50e-3, -50e-3, 4e-3, 12e-3, 48e-3,        'biconvex R=+-50mm ap12 d_out=48mm'),
]:
    report(R1,R2,d,ap,dd,tag)
