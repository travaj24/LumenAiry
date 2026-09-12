"""PROP-HF p3b: rng=None determinism, with a cone narrow enough that paths
actually land (the p3 run had zero landed paths so every seed tied at 0)."""
import sys
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hfpi import (  # noqa: E402
    init_paths_from_field, propagate_to_plane, apply_aperture_diffraction,
    accumulate_to_grid, propagate_hfpi_freespace_aperture)
from lumenairy.propagators.vectorial_hfpi import (  # noqa: E402
    init_vector_paths_from_field, propagate_vector_to_plane,
    apply_vector_aperture_diffraction, accumulate_vector_to_grid)

LAM = 633e-9
N, dx = 32, 2e-6
E = np.ones((N, N), dtype=np.complex128)


def run(rng, cone=0.05, n_paths=200000):
    """3-leg pipeline built by hand so cone_half_angle can be narrowed."""
    p = init_paths_from_field(E, dx, n_paths=n_paths, wavelength=LAM,
                              rng=rng, cone_half_angle=cone)
    p = propagate_to_plane(p, 1e-3, LAM)
    p = apply_aperture_diffraction(p, 60e-6, wavelength=LAM, rng=rng,
                                   cone_half_angle=cone)
    p = propagate_to_plane(p, 2e-3, LAM)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return accumulate_to_grid(p, Ny=N, Nx=N, dx=dx)


print("(1) hand-built pipeline, cone=0.05 rad so paths land")
a, b = run(None), run(None)
c = run(0)
d = run(7)
print(f"  landed energy  sum|E|^2 = {np.sum(np.abs(a)**2):.4e} (non-zero => paths landed)")
print(f"  rng=None twice identical?     {np.array_equal(a, b)}")
print(f"  rng=None == rng=0 ?           {np.array_equal(a, c)}")
print(f"  rng=7 differs from rng=None ? {not np.array_equal(a, d)}")

print("\n(2) the end-to-end entry point, same narrow-cone geometry")


def run_e2e(rng):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return propagate_hfpi_freespace_aperture(
            E, dx, z_to_aperture=1e-3, aperture_radius=60e-6,
            z_aperture_to_output=1e-3, wavelength=LAM, n_paths=200000,
            rng=rng, on_undersampled='silent')


x0, x1 = run_e2e(None), run_e2e(None)
y = run_e2e(0)
zz = run_e2e(7)
print(f"  landed energy = {np.sum(np.abs(x0)**2):.4e}")
print(f"  rng=None twice identical?     {np.array_equal(x0, x1)}")
print(f"  rng=None == rng=0 ?           {np.array_equal(x0, y)}")
print(f"  rng=7 differs?                {not np.array_equal(x0, zz)}")

print("\n(3) init / aperture correlation on the rng=None default")
p = init_paths_from_field(E, dx, n_paths=6, wavelength=LAM, rng=None,
                          cone_half_angle=0.05)
q = apply_aperture_diffraction(p, 1.0, wavelength=LAM, rng=None,
                               cone_half_angle=0.05)
print(f"  init  dir z: {np.round(p.directions[:, 2], 9)}")
print(f"  reemit dir z:{np.round(q.directions[:, 2], 9)}")
print(f"  init  dir x: {np.round(p.directions[:, 0], 9)}")
print(f"  reemit dir x:{np.round(q.directions[:, 0], 9)}")

print("\n(4) vector_projection=True: does it create Ez / conserve energy?")
Ex_in = np.ones((16, 16), dtype=np.complex128)
Ey_in = np.zeros((16, 16), dtype=np.complex128)
vp = init_vector_paths_from_field(Ex_in, Ey_in, dx, n_paths=50000,
                                  wavelength=LAM, rng=2, cone_half_angle=0.8)
vp = propagate_vector_to_plane(vp, 1e-4, LAM)
for flag in (False, True):
    w = apply_vector_aperture_diffraction(
        vp, 1.0, wavelength=LAM, rng=3, cone_half_angle=0.8,
        vector_projection=flag)
    e_before = float(np.sum(np.abs(vp.Ex) ** 2 + np.abs(vp.Ey) ** 2))
    e_after = float(np.sum(np.abs(w.Ex) ** 2 + np.abs(w.Ey) ** 2))
    # what the dropped longitudinal component would have carried
    L = w.directions[:, 0]
    M = w.directions[:, 1]
    Nz = w.directions[:, 2]
    proj = vp.Ex * L + vp.Ey * M
    e_lost = float(np.sum(np.abs(proj * Nz) ** 2))
    print(f"  vector_projection={flag!s:5s}  |E_t|^2/|E_in|^2 (per-path, pre-kirchhoff)"
          f" = {e_after/e_before:.6e}   dropped Ez' energy share = "
          f"{e_lost/float(np.sum(np.abs(vp.Ex)**2)):.4e}")
print("  VectorPathBundle fields:",
      [f for f in vp.__dataclass_fields__])
