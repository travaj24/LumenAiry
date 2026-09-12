"""PROP-HF p3: HFPI Monte-Carlo estimator checks.

(1) rng=None determinism -- the default path.
(2) MC normalisation: is the source-plane area factor N_pix*dx^2 or dx^2?
(3) _spawn_rng(Generator, i) reproducibility.
(4) vectorial HFPI == scalar HFPI per component (is there any vector physics?)
(5) vectorial accumulate has no undersampling guard.
"""
import sys
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")

from lumenairy.propagators.hfpi import (  # noqa: E402
    init_paths_from_field, propagate_to_plane, accumulate_to_grid,
    propagate_hfpi_freespace_aperture, _spawn_rng,
)
from lumenairy.propagators.vectorial_hfpi import (  # noqa: E402
    propagate_vector_hfpi_freespace_aperture,
)

LAM = 633e-9


def sec(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


sec("(1) rng=None is NOT random -- the documented 'system entropy' path")
N, dx = 32, 2e-6
E = np.ones((N, N), dtype=np.complex128)
kw = dict(z_to_aperture=1e-3, aperture_radius=30e-6,
          z_aperture_to_output=1e-3, wavelength=LAM, n_paths=20000,
          on_undersampled='silent')
a = propagate_hfpi_freespace_aperture(E, dx, rng=None, **kw)
b = propagate_hfpi_freespace_aperture(E, dx, rng=None, **kw)
print(f"  two calls with rng=None identical? {np.array_equal(a, b)}   "
      f"max|a-b| = {np.max(np.abs(a - b)):.3e}")
c = propagate_hfpi_freespace_aperture(E, dx, rng=0, **kw)
print(f"  rng=None equals rng=0 ?            {np.array_equal(a, c)}")
d = propagate_hfpi_freespace_aperture(E, dx, rng=7, **kw)
print(f"  rng=7   differs from rng=None ?    {not np.array_equal(a, d)}")

# and the correlation the 4.11.2 fix claims to have removed:
from lumenairy.backend import RandomState  # noqa: E402
rs1 = RandomState(rng=0); rs2 = RandomState(rng=0)
_ = rs1.integers((5,), low=0, high=10); _ = rs1.integers((5,), low=0, high=10)
u_init = np.asarray(rs1.uniform((5,)))
u_ap = np.asarray(rs2.uniform((5,)))
print(f"  init-stage uniforms  {np.round(u_init, 6)}")
print(f"  aperture uniforms    {np.round(u_ap, 6)}   (both from seed 0)")

sec("(2) MC source-plane normalisation: dx^2 or (N_pix dx^2)?")
# A single source pixel of unit amplitude.  Accumulate EVERYTHING that
# is emitted (huge output grid) and compare the total complex sum to the
# analytic total the HF weights should carry:
#   sum_over_paths w  ->  E * dx^2 * (1/(i lam)) * Omega     [if correct]
# The code's per-path solid angle is Omega/n_paths -- which is only the
# correct MC weight if EVERY path starts at that one pixel.
for Ns in (1, 4, 16, 64):
    Ein = np.zeros((Ns, Ns), dtype=np.complex128)
    Ein[Ns // 2, Ns // 2] = 1.0
    cone = 0.20
    p = init_paths_from_field(Ein, dx, n_paths=200000, wavelength=LAM,
                              rng=1, cone_half_angle=cone)
    tot = complex(np.sum(p.weights))
    Omega = 2 * np.pi * (1 - np.cos(cone))
    # exact integral of E*cos(th)*dx^2/(i lam) over the cone
    exact = (dx * dx) / (1j * LAM) * 2 * np.pi * (1 - np.cos(cone) ** 2) / 2
    print(f"  N_pix={Ns*Ns:5d}  sum(w) = {tot:.6e}   exact = {exact:.6e}   "
          f"ratio = {abs(tot)/abs(exact):.6f}   1/N_pix = {1.0/(Ns*Ns):.6f}")

sec("(3) _spawn_rng with a Generator is not reproducible for a given stream")
g = np.random.default_rng(12345)
s0 = _spawn_rng(g, 1)
g2 = np.random.default_rng(12345)
s1 = _spawn_rng(g2, 1)
print(f"  same parent seed, same stream index -> same draws? "
      f"{np.array_equal(s0.random(3), s1.random(3))}")
g3 = np.random.default_rng(12345)
x1 = _spawn_rng(g3, 0).random(3)
x2 = _spawn_rng(g3, 1).random(3)   # parent state was mutated by the first spawn
g4 = np.random.default_rng(12345)
y2 = _spawn_rng(g4, 1).random(3)
print(f"  stream 1 after stream 0 == stream 1 alone? {np.array_equal(x2, y2)}")

sec("(4) vectorial HFPI: is it anything but two scalar runs?")
Nv = 24
Ex_in = np.ones((Nv, Nv), dtype=np.complex128)
Ey_in = np.zeros((Nv, Nv), dtype=np.complex128)
kwv = dict(z_to_aperture=2e-4, aperture_radius=40e-6,
           z_aperture_to_output=2e-4, wavelength=LAM, n_paths=60000)
vx, vy = propagate_vector_hfpi_freespace_aperture(Ex_in, Ey_in, dx, rng=3, **kwv)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sx = propagate_hfpi_freespace_aperture(
        Ex_in, dx, rng=3, on_undersampled='silent', **kwv)
print(f"  vector Ex == scalar run on Ex_in ?  {np.allclose(vx, sx)}   "
      f"max|diff| = {np.max(np.abs(vx - sx)):.3e}")
print(f"  vector Ey all zero (y-pol never generated from x-pol)? "
      f"{np.all(vy == 0)}")
# now a 45-degree linear input: does any cross-coupling / depolarisation appear?
Ex45 = np.ones((Nv, Nv), dtype=np.complex128) / np.sqrt(2)
Ey45 = np.ones((Nv, Nv), dtype=np.complex128) / np.sqrt(2)
wx, wy = propagate_vector_hfpi_freespace_aperture(Ex45, Ey45, dx, rng=3, **kwv)
r = wy[np.abs(wx) > 0] / wx[np.abs(wx) > 0]
print(f"  45 deg input: Ey/Ex over the whole output grid -- "
      f"min={np.min(np.abs(r - 1)):.2e} max={np.max(np.abs(r - 1)):.2e} "
      f"(0 => polarisation is rigidly unchanged everywhere)")
print("  -> no Ez component exists in the API at all: "
      f"returns {len(propagate_vector_hfpi_freespace_aperture.__annotations__.get('return', ()).__args__ if hasattr(propagate_vector_hfpi_freespace_aperture.__annotations__.get('return', ()), '__args__') else [])} components")

sec("(5) undersampling guard coverage")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    propagate_hfpi_freespace_aperture(E, dx, rng=1, **{**kw, 'on_undersampled': 'warn'})
    print(f"  scalar end-to-end warned: {any('UNDER-SAMPLED' in str(x.message) for x in w)}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    propagate_vector_hfpi_freespace_aperture(E, np.zeros_like(E), dx, rng=1,
                                             z_to_aperture=1e-3,
                                             aperture_radius=30e-6,
                                             z_aperture_to_output=1e-3,
                                             wavelength=LAM, n_paths=20000)
    print(f"  vector end-to-end warned: {any('UNDER-SAMPLED' in str(x.message) for x in w)}")
import inspect  # noqa: E402
print("  vector entry point accepts on_undersampled? "
      f"{'on_undersampled' in inspect.signature(propagate_vector_hfpi_freespace_aperture).parameters}")
