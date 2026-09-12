"""Probe 16: self-consistency -- appending a ZERO-THICKNESS FLAT dummy surface
after the last (curved) surface moves the Maslov exit chart from the curved
surface to the vertex plane.  If the reference surface were already the vertex
plane, the two results would be identical."""
import numpy as np, sys, warnings, copy
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy import raytrace as rt
warnings.simplefilter("ignore")
lam, N, dx = 1.0e-6, 96, 4.0e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); R2=X**2+Y**2
ap=0.30e-3
E0=((R2<=(0.5*ap)**2)*np.exp(-R2/(0.22e-3)**2)).astype(complex)
presc = la.make_singlet(np.inf, -2.0e-3, 0.7e-3, 'N-BK7', aperture=ap)
p2 = copy.deepcopy(presc)
p2['surfaces'].append({'radius': np.inf, 'conic': 0.0,
                       'glass_before': 'air', 'glass_after': 'air'})
p2['thicknesses'] = list(p2['thicknesses']) + [0.0]

# where do the exit rays sit for each?
for nm, P in (('original', presc), ('+dummy flat', p2)):
    su = rt.surfaces_from_prescription(P)
    xs = np.linspace(0, 0.14e-3, 5); ys=np.zeros_like(xs)
    r = rt.RayBundle(x=xs.copy(),y=ys.copy(),z=np.zeros_like(xs),L=np.zeros_like(xs),
                     M=np.zeros_like(xs),N=np.ones_like(xs),wavelength=lam,
                     alive=np.ones(5,bool),opd=np.zeros(5))
    e = rt.trace(r, su, lam).image_rays
    print(f"{nm:12s}: exit z = {np.array2string(e.z, precision=3)}  "
          f"opd = {np.array2string(e.opd/lam, precision=2)} waves")

kw = dict(wavelength=lam, dx=dx, normalize_output='power', collimated_input=True,
          integration_method='quadrature', n_v2=96, poly_order=5,
          ray_field_samples=12, ray_pupil_samples=12)
Ea = np.asarray(la.apply_real_lens_maslov(E0.copy(), prescription=presc, **kw))
Eb = np.asarray(la.apply_real_lens_maslov(E0.copy(), prescription=p2, **kw))
m = (np.abs(Ea)>0.05*np.abs(Ea).max()) & (np.abs(Eb)>0)
d = np.angle(Eb[m]*np.conj(Ea[m]))
d = np.angle(np.exp(1j*(d-np.angle(np.vdot(Ea[m],Eb[m])))))
amp = np.abs(Eb[m])/np.abs(Ea[m])
R = -2.0e-3
sg = (R2[m]/R)/(1+np.sqrt(np.maximum(1-R2[m]/R**2,0)))
print(f"\nphase(+dummy) - phase(original): wrapped RMS = {np.sqrt(np.mean(d**2)):.4f} rad, "
      f"PV = {np.ptp(d):.4f} rad")
print(f"|E| ratio: mean={amp.mean():.4f} spread={amp.std()/amp.mean():.4f}")
print(f"predicted -k*sag over the mask: RMS = {np.sqrt(np.mean((2*np.pi/lam*sg)**2)):.2f} rad "
      f"({np.sqrt(np.mean(sg**2))/lam:.2f} waves), PV = {np.ptp(2*np.pi/lam*sg):.2f} rad")
# radial profile of the two phases (unwrapped along a radius)
rr = np.sqrt(R2)[N//2, N//2:]
pa = np.unwrap(np.angle(Ea[N//2, N//2:])); pb = np.unwrap(np.angle(Eb[N//2, N//2:]))
ok = np.abs(Ea[N//2,N//2:])>0.05*np.abs(Ea).max()
dd = (pb-pa)[ok]; dd = dd-dd[0]
sgr = ((rr**2/R)/(1+np.sqrt(np.maximum(1-rr**2/R**2,0))))[ok]
pred = (-2*np.pi/lam*sgr); pred = pred-pred[0]
print("\nradial cut: r[um], phase(+dummy)-phase(orig) [waves], -sag/lam [waves]")
for i in range(0, ok.sum(), max(1, ok.sum()//8)):
    print(f"   r={rr[ok][i]*1e6:7.1f}  meas={dd[i]/(2*np.pi):+8.3f}  pred={pred[i]/(2*np.pi):+8.3f}")
