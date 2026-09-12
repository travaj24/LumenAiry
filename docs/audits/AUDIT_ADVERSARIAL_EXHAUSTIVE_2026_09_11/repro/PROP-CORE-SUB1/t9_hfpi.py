"""HFPI / vectorial-HFPI probes."""
import sys, warnings, time
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.hfpi as H
import lumenairy.propagators.vectorial_hfpi as VH
from lumenairy.propagators.asm import angular_spectrum_propagate

lam = 633e-9; k = 2*np.pi/lam
N, dx, w0 = 64, 2e-6, 12e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
E0 = np.exp(-(np.hypot(X, Y)/w0)**2).astype(np.complex128)

print("="*72)
print("(1) cone_half_angle is NOT reachable from propagate_hfpi* -- the")
print("    under-sampling guard's own remedy is not in the signature")
print("="*72)
for fn, kw in ((H.propagate_hfpi,
                dict(z=1e-3, wavelength=lam, dx=dx, aperture_radius=30e-6,
                     z_aperture_to_output=1e-3, n_paths=1000,
                     cone_half_angle=0.05)),
               (H.propagate_hfpi_freespace_aperture,
                dict(dx=dx, z_to_aperture=1e-3, aperture_radius=30e-6,
                     z_aperture_to_output=1e-3, wavelength=lam, n_paths=1000,
                     cone_half_angle=0.05)),
               (VH.propagate_vector_hfpi_freespace_aperture,
                dict(Ey_in=E0, dx=dx, z_to_aperture=1e-3,
                     aperture_radius=30e-6, z_aperture_to_output=1e-3,
                     wavelength=lam, n_paths=1000, cone_half_angle=0.05))):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn(E0, **kw)
        print(f"  {fn.__name__:<42s} accepts cone_half_angle")
    except TypeError as e:
        print(f"  {fn.__name__:<42s} TypeError: {e}")
import inspect
print("  propagate_hfpi_freespace_aperture params:",
      [p for p in inspect.signature(H.propagate_hfpi_freespace_aperture)
       .parameters])

print()
print("="*72)
print("(2) apply_aperture_diffraction default wavelength=0.0 silently")
print("    drops the 1/(i lam) Kirchhoff prefactor (magnitude AND 90 deg)")
print("="*72)
p = H.init_paths_from_field(E0, dx, n_paths=64, wavelength=lam, rng=1)
a = H.apply_aperture_diffraction(p, 30e-6, wavelength=lam, rng=2)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    b = H.apply_aperture_diffraction(p, 30e-6, rng=2)   # wavelength omitted
    print(f"  warnings raised when wavelength omitted: {len(w)}")
r = np.mean(a.weights[np.abs(b.weights) > 0]/b.weights[np.abs(b.weights) > 0])
print(f"  weights(with lam)/weights(default 0.0) = {r:.6e}")
print(f"  |ratio| = {abs(r):.4e}  (= 1/lam = {1/lam:.4e});  "
      f"arg = {np.angle(r, deg=True):.1f} deg")
print(f"  signature default: "
      f"{inspect.signature(H.apply_aperture_diffraction).parameters['wavelength']}")
print(f"  vectorial twin:    "
      f"{inspect.signature(VH.apply_vector_aperture_diffraction).parameters['wavelength']}")

print()
print("="*72)
print("(3) init_paths_stratified: n_paths is an upper bound in the docstring")
print("    but a LOWER bound in the code when strata counts are given")
print("="*72)
for (nxy, ndir, npaths) in (((32, 32), (32, 32), 1000), ((8, 8), (8, 8), 100)):
    n_total = nxy[0]*nxy[1]*ndir[0]*ndir[1]
    n_per = max(1, npaths//n_total)
    print(f"  request n_paths={npaths:6d} strata {nxy}x{ndir}: "
          f"n_total={n_total:9d} -> n_paths_actual={n_per*n_total:9d}  "
          f"({n_per*n_total/npaths:.0f}x requested)")
t0 = time.perf_counter()
pb = H.init_paths_stratified(E0, dx, n_paths=1000, wavelength=lam,
                             rng=0, n_strata_xy=(32, 32), n_strata_dir=(32, 32))
print(f"  MEASURED len(bundle) for n_paths=1000, 32^4 strata: {len(pb)} "
      f"({time.perf_counter()-t0:.2f}s, "
      f"{pb.positions.nbytes/1e6:.0f} MB just for positions)")

print()
print("="*72)
print("(4) HFPI spatial profile vs ASM (missing 1/r + binning Jacobian)")
print("="*72)
z1, z2 = 2e-3, 2e-3
Eref = angular_spectrum_propagate(E0, z=z1+z2, wavelength=lam, dx=dx,
                                  bandlimit=False)
for npaths in (2_000_000, 8_000_000):
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Eh = H.propagate_hfpi(E0, z1, lam, dx, aperture_radius=1e9,
                              z_aperture_to_output=z2, n_paths=npaths, rng=7)
        warned = [str(x.message)[:60] for x in w
                  if 'UNDER-SAMPLED' in str(x.message)]
    Ih = np.abs(Eh)**2; Ir = np.abs(Eref)**2
    occ = np.count_nonzero(Ih)/Ih.size
    # best-fit global scale, then residual
    s = np.sum(Ih*Ir)/max(np.sum(Ih*Ih), 1e-300)
    print(f"  n_paths={npaths:9,d}  {time.perf_counter()-t0:6.1f}s  "
          f"occupancy={occ:6.3f}  undersampled-warn={bool(warned)}")
    print(f"     |E| ratio HFPI/ASM at peak = "
          f"{np.abs(Eh).max()/np.abs(Eref).max():.4e}")
    print(f"     intensity shape relL2 after best global rescale = "
          f"{np.linalg.norm(s*Ih-Ir)/np.linalg.norm(Ir):.4f}")
    # radial bias: ratio of (HFPI/ASM) intensity on-axis vs at r=25um
    c = N//2
    Rr = np.hypot(X, Y)
    m0 = Rr < 6e-6; m1 = (Rr > 20e-6) & (Rr < 30e-6)
    r0 = Ih[m0].sum()/max(Ir[m0].sum(), 1e-300)
    r1 = Ih[m1].sum()/max(Ir[m1].sum(), 1e-300)
    print(f"     radial bias  (HFPI/ASM)|r<6um / (HFPI/ASM)|20-30um = "
          f"{r0/r1:.3f}")

print()
print("="*72)
print("(5) vectorial HFPI has no under-sampling guard at all")
print("="*72)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    Ex, Ey = VH.propagate_vector_hfpi_freespace_aperture(
        E0, np.zeros_like(E0), dx, z_to_aperture=z1, aperture_radius=1e9,
        z_aperture_to_output=z2, wavelength=lam, n_paths=20000, rng=7)
    print(f"  vector, n_paths=20000 on a {N}x{N} grid: "
          f"{np.count_nonzero(Ex)} of {Ex.size} pixels non-zero "
          f"({100*np.count_nonzero(Ex)/Ex.size:.2f}%), warnings={len(w)}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    Es = H.propagate_hfpi(E0, z1, lam, dx, aperture_radius=1e9,
                          z_aperture_to_output=z2, n_paths=20000, rng=7)
    print(f"  scalar, same settings: warnings={len(w)} "
          f"-> {[str(x.category.__name__) for x in w]}")
