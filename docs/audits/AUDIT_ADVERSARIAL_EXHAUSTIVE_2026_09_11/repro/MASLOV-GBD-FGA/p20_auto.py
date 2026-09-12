"""Probe 20: which integrator does integration_method='auto' (the DEFAULT) pick,
and how far is local_quadrature from the exact quadrature on a small ROI?"""
import numpy as np, sys, warnings, io, contextlib, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter("ignore")
import lumenairy as la
lam, N, dx = 1.0e-6, 48, 4.0e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); R2=X*X+Y*Y
for nm, presc, ap in (("f=6mm biconvex, 0.15mm ap", la.make_singlet(6e-3,-6e-3,0.7e-3,'N-BK7',aperture=0.15e-3), 0.15e-3),
                      ("f=2mm biconvex, 0.15mm ap", la.make_singlet(2e-3,-2e-3,0.7e-3,'N-BK7',aperture=0.15e-3), 0.15e-3)):
    E0=((R2<=(0.5*ap)**2)*np.exp(-R2/(0.06e-3)**2)).astype(complex)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        Ea=np.asarray(la.apply_real_lens_maslov(E0.copy(),prescription=presc,wavelength=lam,
            dx=dx, collimated_input=True, verbose=True, normalize_output='none',
            ray_field_samples=10, ray_pupil_samples=10, poly_order=4))
    line=[l for l in buf.getvalue().splitlines() if 'auto ->' in l]
    print(f"{nm}: {line[0].strip() if line else 'auto line not printed'}")
    t=time.perf_counter()
    Eq=np.asarray(la.apply_real_lens_maslov(E0.copy(),prescription=presc,wavelength=lam,
        dx=dx, collimated_input=True, normalize_output='none',
        integration_method='quadrature', n_v2=1400,
        ray_field_samples=10, ray_pupil_samples=10, poly_order=4))
    tq=time.perf_counter()-t
    m=np.abs(Eq)>0.05*np.abs(Eq).max()
    rel=np.linalg.norm(Ea[m]-Eq[m])/np.linalg.norm(Eq[m])
    relA=np.linalg.norm(np.abs(Ea[m])-np.abs(Eq[m]))/np.linalg.norm(np.abs(Eq[m]))
    ov = abs(np.vdot(Eq[m],Ea[m]))**2/(np.vdot(Eq[m],Eq[m]).real*np.vdot(Ea[m],Ea[m]).real)
    print(f"    default 'auto' vs resolved quadrature(n_v2=1400, {tq:.0f}s): "
          f"relL2={rel:.3e}  rel|E|={relA:.3e}  overlap={ov:.4f}  "
          f"mean|E| ratio={np.mean(np.abs(Ea[m]))/np.mean(np.abs(Eq[m])):.4f}")
