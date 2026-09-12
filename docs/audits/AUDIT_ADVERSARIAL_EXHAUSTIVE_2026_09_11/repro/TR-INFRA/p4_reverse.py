"""TR-INFRA probe 4: _reverse_prescription correctness."""
import copy
import numpy as np
from lumenairy.elements import _lens_traced as T
from lumenairy.io.prescriptions_builders import make_singlet
from lumenairy.raytrace import surfaces_from_prescription, trace, _make_bundle
from lumenairy.glass import get_glass_index

lam = 587.6e-9

def show(p, tag):
    print(f"  {tag}: thicknesses={p['thicknesses']}")
    for i,s in enumerate(p['surfaces']):
        print(f"      S{i}: R={s['radius']:+.6g} conic={s.get('conic')} "
              f"asph={s.get('aspheric_coeffs')} gb={s['glass_before']} ga={s['glass_after']}")

print("=== 4a  ROUND TRIP reverse(reverse(P)) == P ? ===")
P = make_singlet(0.050, -0.050, 0.005, 'N-BK7', aperture=0.020)
P['thicknesses'] = [0.005, 0.100]      # n-thickness convention (VALID per validate_prescription)
show(P,'P      ')
R1 = T._reverse_prescription(P)
show(R1,'rev(P) ')
R2 = T._reverse_prescription(R1)
show(R2,'rev^2  ')
same = (R2['thicknesses']==P['thicknesses'] and
        all(R2['surfaces'][i]['radius']==P['surfaces'][i]['radius'] for i in range(2)))
print(f"  round trip identical: {same}")

print("\n=== 4b  THICKNESS off-by-one under the n-thickness convention ===")
sf = surfaces_from_prescription(P); sr = surfaces_from_prescription(R1)
print(f"  forward  surface thicknesses: {[s.thickness for s in sf]}  "
      f"(glass gap = {sf[0].thickness}, BFD = {sf[1].thickness})")
print(f"  reversed surface thicknesses: {[s.thickness for s in sr]}  "
      f"-> reversed GLASS gap = {sr[0].thickness} (should be {sf[0].thickness})")

print("\n  ...and under the (n-1)-thickness convention:")
P2 = make_singlet(0.050,-0.050,0.005,'N-BK7',aperture=0.020)    # thicknesses=[0.005]
R2b = T._reverse_prescription(P2)
sr2 = surfaces_from_prescription(R2b)
print(f"  forward {[s.thickness for s in surfaces_from_prescription(P2)]}"
      f"  reversed {[s.thickness for s in sr2]}  -> correct")

print("\n=== 4c  PHYSICS: forward OPL through P vs backward OPL through rev(P) ===")
def fwd_opl(P, h):
    surfs = surfaces_from_prescription(P)
    rays = _make_bundle(x=np.array([h]), y=np.array([0.0]),
                        L=np.array([0.0]), M=np.array([0.0]), wavelength=lam)
    r = trace(rays, surfs, lam).image_rays
    # transfer to the last vertex plane (z=0 of the last surface's frame)
    n_ex = get_glass_index(surfs[-1].glass_after, lam)
    t = np.where(r.alive & (np.abs(r.N)>1e-30), -r.z/r.N, 0.0)
    return float(r.opd[0] + n_ex*t[0]), bool(r.alive[0]), float(r.x[0]+r.L[0]*t[0]), float(r.L[0])

for label, Pp in (('n-1 thicknesses', P2), ('n thicknesses  ', P)):
    Rp = T._reverse_prescription(Pp)
    for h in (0.0, 0.002, 0.004):
        o_f, af, xf, Lf = fwd_opl(Pp, h)
        # Backward: launch from the exit vertex plane at the exit height, with
        # the REVERSED direction, through rev(P).
        surfs_r = surfaces_from_prescription(Rp)
        rays = _make_bundle(x=np.array([xf]), y=np.array([0.0]),
                            L=np.array([-Lf]), M=np.array([0.0]), wavelength=lam)
        rr = trace(rays, surfs_r, lam).image_rays
        n_ex = get_glass_index(surfs_r[-1].glass_after, lam)
        t = np.where(rr.alive & (np.abs(rr.N)>1e-30), -rr.z/rr.N, 0.0)
        o_b = float(rr.opd[0] + n_ex*t[0]); xb = float(rr.x[0]+rr.L[0]*t[0])
        print(f"  {label} h={h*1e3:5.2f} mm: OPL_fwd={o_f:.12e}  OPL_bwd={o_b:.12e}"
              f"  d={o_f-o_b:+.3e}  x_back={xb:+.6e} (launch h={h:+.6e}, err={xb-h:+.2e})"
              f" alive={af}/{bool(rr.alive[0])}")

print("\n=== 4d  ASPHERIC COEFFICIENT SIGN under reversal ===")
Pa = make_singlet(0.050, np.inf, 0.005, 'N-BK7', aperture=0.020)
Pa['surfaces'][0]['aspheric_coeffs'] = {4: 1.0e3}
Ra = T._reverse_prescription(Pa)
print(f"  forward S0 asph = {Pa['surfaces'][0]['aspheric_coeffs']}")
print(f"  reversed S1 asph = {Ra['surfaces'][-1]['aspheric_coeffs']}  (should be {{4: -1000.0}})")
# what the sag ACTUALLY is
from lumenairy.raytrace.surface import _surface_sag_xy
sf = surfaces_from_prescription(Pa); sr = surfaces_from_prescription(Ra)
h = np.array([0.005])
sag_f = _surface_sag_xy(h, np.zeros(1), sf[0])
sag_r = _surface_sag_xy(h, np.zeros(1), sr[-1])
print(f"  sag(forward S0, h=5mm) = {sag_f[0]:+.9e} m")
print(f"  sag(reversed S1,h=5mm) = {sag_r[0]:+.9e} m   (should be {-sag_f[0]:+.9e})")
print(f"  ERROR = {sag_r[0]+sag_f[0]:+.6e} m  ({(sag_r[0]+sag_f[0])/lam:+.2f} waves at 588 nm)")

print("\n=== 4e  which surface keys does the reversal silently pass through? ===")
S = dict(Pa['surfaces'][0])
S.update({'conic': -1.0, 'aspheric_coeffs': {4: 1e3, 6: 2e7},
          'radius_y': 0.03, 'conic_y': -0.5, 'aspheric_coeffs_y': {4: -5e2},
          'decenter': (1e-4, 2e-4), 'tilt': (1e-3, -2e-3),
          'freeform': {'freeform_type':'xy_polynomial','xy_coeffs':{(2,0):1e-2}},
          'sag_callable': (lambda x,y: 0.0*x)})
Pk = {'surfaces': [S, dict(Pa['surfaces'][1])], 'thicknesses': [0.005],
      'aperture_diameter': 0.02, 'stop_index': 0, 'name':'x'}
Rk = T._reverse_prescription(Pk)
rs = Rk['surfaces'][-1]
for key in ('radius','conic','aspheric_coeffs','radius_y','conic_y',
            'aspheric_coeffs_y','decenter','tilt','freeform','sag_callable'):
    a = S.get(key); b = rs.get(key)
    flag = 'NEGATED' if key in ('radius','radius_y') else ('unchanged' if a==b or (callable(a) and a is b) else 'CHANGED')
    print(f"   {key:20s} fwd={str(a)[:46]:48s} rev={str(b)[:46]:48s} -> {flag}")
print(f"   top-level keys kept: {sorted(Rk.keys())}  (dropped: "
      f"{sorted(set(Pk)-set(Rk))})")
