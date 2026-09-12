"""ANALYSIS probe 7: performance / memory measurements."""
import sys, time, tracemalloc, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.opd import wave_opd_2d
from lumenairy.analysis.beam_stats import beam_d4sigma, beam_centroid, M2
from lumenairy.analysis.through_focus import single_plane_metrics, through_focus_scan
from lumenairy.analysis.polychromatic import radial_power_bands
from lumenairy.analysis.psf_mtf_otf import (encircled_energy_curve, compute_psf,
                                            encircled_energy_radius)
from lumenairy.analysis.zernike import (zernike_basis_matrix, zernike_decompose,
                                        clear_zernike_basis_cache)

def tt(fn, n=5):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter()-t)/n

lam = 633e-9
N = 1024
dx = 1e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
R2 = X**2+Y**2
ap = 800e-6
E = (R2 <= (ap/2)**2)*np.exp(-1j*2*np.pi/lam*R2/(2*0.05))

print("=== 1. wave_opd_2d peak extra memory (N=1024 complex128 = 16.8 MB/grid) ===")
tracemalloc.start()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    _ = wave_opd_2d(E, dx, lam, aperture=ap, f_ref=0.05)
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
one = N*N*16/1e6
print(f"  peak traced alloc = {peak/1e6:.1f} MB  = {peak/1e6/one:.1f} x one complex128 grid "
      f"({one:.1f} MB)")
print(f"  time = {tt(lambda: wave_opd_2d(E, dx, lam, aperture=ap, f_ref=0.05), 3)*1e3:.1f} ms")

print()
print("=== 2. through_focus_scan: propagation vs metric cost ===")
Nz = 21
z = np.linspace(0.045, 0.055, Nz)
t0 = time.perf_counter()
scan = through_focus_scan(E, dx, lam, z, verbose=False)
t_full = time.perf_counter()-t0
t_metric = tt(lambda: single_plane_metrics(E, dx, lam), 5)
print(f"  full {Nz}-plane scan       = {t_full*1e3:8.1f} ms  ({t_full/Nz*1e3:.2f} ms/plane)")
print(f"  single_plane_metrics alone = {t_metric*1e3:8.2f} ms/plane "
      f"({100*t_metric*Nz/t_full:.0f} % of the scan)")
t_cent = tt(lambda: beam_centroid(E, dx), 10)
t_d4 = tt(lambda: beam_d4sigma(E, dx), 10)
print(f"     beam_centroid  = {t_cent*1e3:.2f} ms   beam_d4sigma = {t_d4*1e3:.2f} ms "
      f"(centroid is recomputed inside d4sigma: {100*t_cent/t_metric:.0f}% redundant)")
# with bucket
t_metric_b = tt(lambda: single_plane_metrics(E, dx, lam, bucket_radius=20e-6), 5)
print(f"  with bucket_radius          = {t_metric_b*1e3:8.2f} ms/plane")

print()
print("=== 3. encircled_energy_curve / radius cost (argsort of N^2) ===")
for Ng in (512, 1024, 2048):
    xg = (np.arange(Ng)-Ng/2)*dx
    Xg, Yg = np.meshgrid(xg, xg)
    Eg = np.exp(-(Xg**2+Yg**2)/(50e-6)**2).astype(complex)
    t = tt(lambda: encircled_energy_curve(Eg, dx, n_radii=64), 3)
    t2 = tt(lambda: encircled_energy_radius(Eg, dx), 3)
    print(f"  N={Ng}: curve={t*1e3:7.1f} ms   radius={t2*1e3:7.1f} ms "
          f"(each rebuilds the full sort)")

print()
print("=== 4. radial_power_bands scaling with n_radii ===")
for nr in (1, 8, 64):
    radii = np.linspace(5e-6, 200e-6, nr)
    t = tt(lambda: radial_power_bands(E, dx, radii), 3)
    print(f"  n_radii={nr:3d}: {t*1e3:7.2f} ms  ({t/nr*1e3:.3f} ms per radius)")

print()
print("=== 5. Zernike basis cache effectiveness ===")
opd = np.where(R2 <= (ap/2)**2, R2*1e-3, np.nan)
clear_zernike_basis_cache()
t_cold = tt(lambda: (clear_zernike_basis_cache(),
                     zernike_decompose(opd, dx, ap, n_modes=21)), 3)
t_warm = tt(lambda: zernike_decompose(opd, dx, ap, n_modes=21), 5)
print(f"  cold (cache cleared each call) = {t_cold*1e3:7.1f} ms")
print(f"  warm                            = {t_warm*1e3:7.1f} ms   "
      f"speedup {t_cold/t_warm:.1f}x")
t_basis = tt(lambda: (clear_zernike_basis_cache(),
                      zernike_basis_matrix(21, X, Y, ap/2)), 3)
print(f"  basis build alone               = {t_basis*1e3:7.1f} ms")
# return_residual rebuilds the basis a 2nd time (cache hit) -- check
t_res = tt(lambda: zernike_decompose(opd, dx, ap, n_modes=21, return_residual=True), 5)
print(f"  warm with return_residual       = {t_res*1e3:7.1f} ms")

print()
print("=== 6. compute_psf oversample memory/time ===")
pup = ((R2 <= (ap/2)**2)).astype(complex)
for ov in (1, 2, 4):
    tracemalloc.start()
    psf, dxp = compute_psf(pup, lam, 0.05, dx, oversample=ov)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    t = tt(lambda: compute_psf(pup, lam, 0.05, dx, oversample=ov), 3)
    print(f"  oversample={ov}: N_psf={psf.shape[0]:5d} time={t*1e3:8.1f} ms "
          f"peak alloc={peak/1e6:8.1f} MB")

print()
print("=== 7. DeformableMirror IF basis memory ===")
from lumenairy.analysis.ao import DeformableMirror
for (na, Ng) in ((9, 256), (16, 256), (16, 512)):
    tracemalloc.start()
    dm = DeformableMirror(n_actuators=na, pitch=2e-3, dx=1e-4, N=Ng)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    cached = dm._IF_basis is not None
    print(f"  n_act={na:2d} N={Ng}: cached={cached} peak alloc={peak/1e6:8.1f} MB "
          f"(n_act^2*N^2*8 = {na*na*Ng*Ng*8/1e6:.1f} MB)")
