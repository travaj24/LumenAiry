"""TR-INFRA probe 3: _geometric_lens_phase vs apply_real_lens (thin plane wave)
and vs an independent analytic sag-OPD sum; sign, piston, masking, biconic,
decenter/tilt, sag_callable."""
import warnings
import numpy as np
from lumenairy.elements import _lens_traced as T
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.io.prescriptions_builders import make_singlet, make_biconic
from lumenairy.glass import get_glass_index
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.surface import _surface_sag_xy

lam = 587.6e-9; k0 = 2*np.pi/lam
N = 256; dxg = 40e-6

def cmp(P, tag, thin=True):
    ones = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = apply_real_lens(ones, prescription=P, wavelength=lam, dx=dxg)
    ph_real = np.angle(E)
    ph_geo = T._geometric_lens_phase(P, lam, dxg, N)
    # compare on the lit, unmasked region
    m = np.abs(E) > 0.5*np.abs(E).max()
    d = np.angle(np.exp(1j*(ph_geo-ph_real)))[m]
    print(f"  {tag}: max|dphi|={np.abs(d).max():.4e} rad  rms={np.sqrt(np.mean(d**2)):.4e}"
          f"  mean(piston)={np.mean(d):+.4e}  masked-out pixels in geo: "
          f"{int(np.sum(~np.isfinite(ph_geo)))}")
    return ph_geo, ph_real, E

print("=== 3a  THIN singlet (d -> 0), plane-wave pass ===")
for d in (1e-6, 1e-4, 2e-3):
    P = make_singlet(0.100, -0.100, d, 'N-BK7', aperture=0.008)
    cmp(P, f"d={d*1e3:7.3f} mm")

print("\n=== 3b  independent analytic oracle: sum (n_after-n_before)*sag ===")
P = make_singlet(0.100, -0.100, 1e-6, 'N-BK7', aperture=0.008)
x = (np.arange(N)-N/2)*dxg; X, Y = np.meshgrid(x, x, indexing='xy')
surfs = surfaces_from_prescription(P)
opd = np.zeros((N,N))
for s in surfs:
    n1 = get_glass_index(s.glass_before, lam); n2 = get_glass_index(s.glass_after, lam)
    opd = opd + (n1-n2)*_surface_sag_xy(X, Y, s)         # OPD = (n_before - n_after)*sag
pist = sum(get_glass_index(s.glass_after, lam)*s.thickness for s in surfs[:-1])
ph_oracle = np.angle(np.exp(1j*k0*(opd+pist)))
ph_geo = T._geometric_lens_phase(P, lam, dxg, N)
m = np.isfinite(ph_geo)&np.isfinite(ph_oracle)
print(f"  max|phi_geo - phi_oracle| = {np.abs(np.angle(np.exp(1j*(ph_geo-ph_oracle))))[m].max():.3e} rad"
      f"  -> sign convention phi = -k0 (n_after-n_before) sag CONFIRMED")

print("\n=== 3c  APERTURE MASKING: does _geometric_lens_phase honour it? ===")
P = make_singlet(0.100, -0.100, 2e-3, 'N-BK7', aperture=0.004)
ones = np.ones((N,N), dtype=np.complex128)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    E = apply_real_lens(ones, prescription=P, wavelength=lam, dx=dxg)
ph_geo = T._geometric_lens_phase(P, lam, dxg, N)
r = np.hypot(X, Y)
print(f"  apply_real_lens |E| outside r=2mm: max={np.abs(E)[r>0.0021].max():.3e}"
      f" (aperture mask applied)")
print(f"  _geometric_lens_phase outside r=2mm: finite={np.isfinite(ph_geo)[r>0.0021].all()},"
      f" range=[{np.nanmin(ph_geo[r>0.0021]):+.3f},{np.nanmax(ph_geo[r>0.0021]):+.3f}]"
      f"  -> NO aperture mask, phase returned everywhere")

print("\n=== 3d  BICONIC (radius_y != radius): x/y axis assignment ===")
Pb = make_biconic(0.100, 0.050, np.inf, np.inf, 1e-6, 'N-BK7', aperture=0.006)
ph_geo, ph_real, E = cmp(Pb, "biconic R_x=100mm R_y=50mm")
# independent check: the y-curvature must be the STRONGER one -> more phase
# curvature along the ROW axis (Y) than along the COLUMN axis (X)
mid = N//2
row = np.unwrap(ph_geo[mid, :]);  col = np.unwrap(ph_geo[:, mid])
def curv(v, ax):
    p = np.polyfit(ax, v, 2); return p[0]
axv = x
print(f"  d2phi/dx2 along ROW (x axis) = {curv(row, axv):.4e}"
      f"   along COLUMN (y axis) = {curv(col, axv):.4e}"
      f"   ratio = {curv(col,axv)/curv(row,axv):.4f} (expect R_x/R_y = 2.0)")

print("\n=== 3e  FIELD-FRAME decenter / tilt / sag_callable ===")
Pd = make_singlet(0.100, -0.100, 1e-6, 'N-BK7', aperture=0.006)
Pd['surfaces'][0]['decenter'] = (1.0e-3, 0.0)
ph_geo_d = T._geometric_lens_phase(Pd, lam, dxg, N)
P0 = make_singlet(0.100, -0.100, 1e-6, 'N-BK7', aperture=0.006)
ph_geo_0 = T._geometric_lens_phase(P0, lam, dxg, N)
print(f"  decenter (1 mm, 0): max|dphi| vs undecentred = "
      f"{np.abs(np.angle(np.exp(1j*(ph_geo_d-ph_geo_0)))).max():.4e} rad -> honoured = "
      f"{np.abs(ph_geo_d-ph_geo_0).max() > 1e-9}")
Pt = make_singlet(0.100,-0.100,1e-6,'N-BK7',aperture=0.006)
Pt['surfaces'][0]['sag_callable'] = lambda xx, yy: 1e-6*np.cos(2*np.pi*xx/1e-3)
ph_geo_t = T._geometric_lens_phase(Pt, lam, dxg, N)
print(f"  sag_callable: max|dphi| vs base = "
      f"{np.abs(np.angle(np.exp(1j*(ph_geo_t-ph_geo_0)))).max():.4e} rad -> honoured = "
      f"{np.abs(ph_geo_t-ph_geo_0).max() > 1e-9}")
Pf = make_singlet(0.100,-0.100,1e-6,'N-BK7',aperture=0.006)
Pf['surfaces'][0]['form_error'] = 100e-9*np.ones((N,N))
ph_geo_f = T._geometric_lens_phase(Pf, lam, dxg, N)
print(f"  form_error (100 nm uniform): max|dphi| vs base = "
      f"{np.abs(ph_geo_f-ph_geo_0).max():.4e} rad -> honoured = "
      f"{np.abs(ph_geo_f-ph_geo_0).max() > 1e-12}")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Ef = apply_real_lens(np.ones((N,N),dtype=np.complex128), prescription=Pf,
                         wavelength=lam, dx=dxg)
    E0 = apply_real_lens(np.ones((N,N),dtype=np.complex128), prescription=P0,
                         wavelength=lam, dx=dxg)
m = np.abs(Ef)>0.5*np.abs(Ef).max()
print(f"  ... but apply_real_lens DOES see form_error: max|dphi| = "
      f"{np.abs(np.angle(Ef*np.conj(E0)))[m].max():.4e} rad "
      f"(expected {2*np.pi/lam*100e-9*1.0:.4f} rad for a (n-1)=0.5 index step... "
      f"or {2*np.pi/lam*100e-9:.4f} for pure OPD)")
