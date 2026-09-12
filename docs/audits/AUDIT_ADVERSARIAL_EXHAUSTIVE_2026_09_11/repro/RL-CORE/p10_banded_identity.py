"""Probe 10/11/13/14: banded byte-identity, complex64 dtype, memmap store,
prepared-lens agreement, thicknesses validation."""
import sys, warnings, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import (apply_real_lens, prepare_real_lens,
                                           set_lens_sag_dtype)

lam = 632.8e-9
OK, BAD = 'OK ', '**DIFF**'


def eq(a, b):
    return np.array_equal(a.view(np.uint8), b.view(np.uint8))


def rx_plain(ap=4e-3):
    return dict(surfaces=[
        dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
        dict(radius=-40e-3, conic=-0.5,
             aspheric_coeffs={4: 1e-5, 6: -2e-7},
             glass_before='N-BK7', glass_after='N-SF11'),
        dict(radius=-200e-3, glass_before='N-SF11', glass_after='AIR')],
        thicknesses=[3e-3, 2e-3], aperture_diameter=ap)


def rx_dec(ap=4e-3):
    r = rx_plain(ap)
    r['surfaces'][1] = dict(r['surfaces'][1]); r['surfaces'][1]['decenter'] = (50e-6, -30e-6)
    return r


def rx_ca(ap=4e-3):
    r = rx_plain(ap)
    r['surfaces'][1] = dict(r['surfaces'][1]); r['surfaces'][1]['clear_aperture'] = 3.4e-3
    return r


def rx_stop(ap=4e-3):
    r = rx_plain(ap); r['stop_index'] = 1
    return r


def rx_mixed(ap=4e-3):
    """band-eligible surface 0, whole-grid surface 1 (decenter), band 2."""
    r = rx_plain(ap)
    r['surfaces'] = [dict(s) for s in r['surfaces']]
    r['surfaces'][1]['decenter'] = (40e-6, 20e-6)
    r['surfaces'][2]['clear_aperture'] = 3.6e-3
    return r


N, dx = 1024, 5e-6
E0 = np.ones((N, N), dtype=np.complex128)
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
Eg = np.exp(-(X ** 2 + Y ** 2) / (1.5e-3) ** 2).astype(np.complex128)

print("=== 10. row-banded vs whole-grid BYTE identity (N=1024) ===")
for nm, rx in (('plain conic+asph 3-surf', rx_plain()),
               ('decentered mid surface', rx_dec()),
               ('clear_aperture mid', rx_ca()),
               ('stop at mid surface', rx_stop()),
               ('mixed band/whole-grid', rx_mixed())):
    ref = apply_real_lens(Eg.copy(), prescription=rx, wavelength=lam, dx=dx,
                          sag_chunk_rows=0)
    for cr in (None, 1, 7, 64, 256, 1024, 4096):
        out = apply_real_lens(Eg.copy(), prescription=rx, wavelength=lam,
                              dx=dx, sag_chunk_rows=cr)
        tag = OK if eq(ref, out) else BAD
        if tag == BAD:
            d = np.abs(ref - out).max()
            print(f"  {nm:<26} cr={str(cr):<5} {tag}  max|d|={d:.3e}")
        else:
            print(f"  {nm:<26} cr={str(cr):<5} {tag}")

print()
print("=== 10b. banded + fresnel / slant (the second banded path) ===")
rx = rx_plain()
for kw in (dict(fresnel=True), dict(slant_correction=True),
           dict(fresnel=True, slant_correction=True)):
    ref = apply_real_lens(Eg.copy(), prescription=rx, wavelength=lam, dx=dx,
                          sag_chunk_rows=0, **kw)
    for cr in (1, 33, 256):
        out = apply_real_lens(Eg.copy(), prescription=rx, wavelength=lam,
                              dx=dx, sag_chunk_rows=cr, **kw)
        print(f"  {str(kw):<48} cr={cr:<5}"
              f"{OK if eq(ref, out) else BAD}")

print()
print("=== 11. complex64 vs complex128 (3-surface, 5 mm total thickness) ===")
rx = dict(surfaces=[
    dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
    dict(radius=-40e-3, glass_before='N-BK7', glass_after='N-SF11'),
    dict(radius=-200e-3, glass_before='N-SF11', glass_after='AIR')],
    thicknesses=[3e-3, 2e-3], aperture_diameter=4e-3)
Ns, dxs = 512, 8e-6
xs = (np.arange(Ns) - Ns / 2) * dxs
Xs, Ys = np.meshgrid(xs, xs)
Eg2 = np.exp(-(Xs ** 2 + Ys ** 2) / (1.2e-3) ** 2)
E128 = apply_real_lens(Eg2.astype(np.complex128), prescription=rx,
                       wavelength=lam, dx=dxs)
E64 = apply_real_lens(Eg2.astype(np.complex64), prescription=rx,
                      wavelength=lam, dx=dxs)
print(f"  dtype out: c128 -> {E128.dtype}, c64 -> {E64.dtype}")
m = np.abs(E128).max()
print(f"  max|E64 - E128| / max|E128| = {np.abs(E64-E128).max()/m:.4e}")
ph = np.angle(E64 * np.conj(E128))
print(f"  phase error: rms {np.sqrt(np.mean(ph**2)):.4e} rad, "
      f"max {np.abs(ph).max():.4e} rad")
print(f"  (piston n*k0*t for 3mm BK7 = {2*np.pi/lam*1.515*3e-3:.4e} rad; "
      f"eps_f32*that = {np.finfo(np.float32).eps*2*np.pi/lam*1.515*3e-3:.3e} rad)")

print()
print("=== 13. accumulator_store='memmap' vs 'ram' (tangent_facet) ===")
rxs = dict(surfaces=[
    dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
    dict(radius=-50e-3, glass_before='N-BK7', glass_after='AIR')],
    thicknesses=[3e-3], aperture_diameter=4e-3)
try:
    a = apply_real_lens(Eg2.astype(np.complex128), prescription=rxs,
                        wavelength=lam, dx=dxs, surface_model='tangent_facet',
                        carrier=0.5, accumulator_store='ram')
    b = apply_real_lens(Eg2.astype(np.complex128), prescription=rxs,
                        wavelength=lam, dx=dxs, surface_model='tangent_facet',
                        carrier=0.5, accumulator_store='memmap')
    print(f"  ram vs memmap byte-identical: {eq(a, b)}")
except Exception as e:
    print(f"  tangent_facet run failed: {type(e).__name__}: {e}")

import tempfile, glob
sd = tempfile.mkdtemp(prefix='rlcore_scratch_')
try:
    apply_real_lens(Eg2.astype(np.complex128), prescription=rxs,
                    wavelength=lam, dx=dxs, surface_model='tangent_facet',
                    carrier=0.5, accumulator_store='memmap', scratch_dir=sd)
    print(f"  scratch_dir leftovers after normal return: "
          f"{glob.glob(os.path.join(sd, '*'))}")
except Exception as e:
    print("  memmap w/ scratch_dir:", e)
# exception mid-run: force a failure via a bad glass on the LAST surface
rxbad = dict(surfaces=[dict(s) for s in rxs['surfaces']],
             thicknesses=list(rxs['thicknesses']),
             aperture_diameter=4e-3)
rxbad['surfaces'][1]['glass_after'] = 'NO_SUCH_GLASS_XYZ'
try:
    apply_real_lens(Eg2.astype(np.complex128), prescription=rxbad,
                    wavelength=lam, dx=dxs, surface_model='tangent_facet',
                    carrier=0.5, accumulator_store='memmap', scratch_dir=sd)
except Exception as e:
    print(f"  mid-run exception ({type(e).__name__}); leftovers: "
          f"{glob.glob(os.path.join(sd, '*'))}")

print()
print("=== 14. prepare_real_lens vs apply_real_lens ===")
P = prepare_real_lens(prescription=rx, wavelength=lam, dx=dxs, N=Ns)
Ep = P(Eg2.astype(np.complex128))
Ea = apply_real_lens(Eg2.astype(np.complex128), prescription=rx,
                     wavelength=lam, dx=dxs)
print(f"  byte-identical: {eq(Ep, Ea)}  max|d| = {np.abs(Ep-Ea).max():.3e} "
      f"(peak {np.abs(Ea).max():.3e})")
Ep64 = P(Eg2.astype(np.complex64))
Ea64 = apply_real_lens(Eg2.astype(np.complex64), prescription=rx,
                       wavelength=lam, dx=dxs)
print(f"  complex64: byte-identical {eq(Ep64, Ea64)} "
      f"max|d| = {np.abs(Ep64-Ea64).max():.3e}")

print()
print("=== 14b. prepared lens: stale w.r.t. the GLASS CATALOGUE? ===")
import lumenairy.glass as G
G.GLASS_REGISTRY['AUDITGLS'] = lambda wl: 1.60
rxg = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                          glass_after='AUDITGLS'),
                     dict(radius=float('inf'), glass_before='AUDITGLS',
                          glass_after='AIR')],
           thicknesses=[3e-3], aperture_diameter=4e-3)
P2 = prepare_real_lens(prescription=rxg, wavelength=lam, dx=dxs, N=Ns)
A1 = P2(Eg2.astype(np.complex128))
G.GLASS_REGISTRY['AUDITGLS'] = lambda wl: 1.90
G._clear_glass_caches()
A2 = P2(Eg2.astype(np.complex128))
B2 = apply_real_lens(Eg2.astype(np.complex128), prescription=rxg,
                     wavelength=lam, dx=dxs)
print(f"  prepared before/after catalogue change identical: {eq(A1, A2)} "
      f"(documented freeze)")
print(f"  prepared vs apply AFTER change: max|d| = {np.abs(A2-B2).max():.3e} "
      f"(peak {np.abs(B2).max():.3e})")

print()
print("=== 14c. prescription MUTATED between prepare and call ===")
rxm = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                     dict(radius=float('inf'), glass_before='N-BK7',
                          glass_after='AIR')],
           thicknesses=[3e-3], aperture_diameter=4e-3)
P3 = prepare_real_lens(prescription=rxm, wavelength=lam, dx=dxs, N=Ns)
rxm['surfaces'][0]['radius'] = 25e-3
C1 = P3(Eg2.astype(np.complex128))
C2 = apply_real_lens(Eg2.astype(np.complex128), prescription=rxm,
                     wavelength=lam, dx=dxs)
print(f"  prepared(after mutation) vs apply: max|d| = {np.abs(C1-C2).max():.3e}")

print()
print("=== 2. thicknesses length validation ===")
bad = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                     dict(radius=float('inf'), glass_before='N-BK7',
                          glass_after='AIR')],
           thicknesses=[3e-3, 9e-3])   # one too many
try:
    apply_real_lens(np.ones((64, 64), complex), prescription=bad,
                    wavelength=lam, dx=1e-5)
    print("  extra thickness: accepted silently!")
except AssertionError as e:
    print(f"  extra thickness -> AssertionError: {str(e)[:70]}")
except Exception as e:
    print(f"  extra thickness -> {type(e).__name__}: {str(e)[:70]}")
