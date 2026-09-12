"""PROP-HF p9:
(A) hf.py:167 uses np.isclose(dx, target_dx, rtol=1e-12) -- np.isclose's
    DEFAULT atol=1e-8 is absolute METRES, which swamps any realistic pitch.
    => a requested output_dx within 1e-8 m of the input pitch is silently
    ignored, and the field is returned LABELLED with the requested pitch.
(B) propagate_hfpi_freespace_aperture has no cone_half_angle parameter, so
    the remedy its own undersampling guard recommends is unreachable.
"""
import sys
import inspect
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hf import (  # noqa: E402
    propagate_huygens_fresnel_freespace as hf_free)
from lumenairy.propagators.hfpi import (  # noqa: E402
    propagate_hfpi_freespace_aperture)
from lumenairy.propagators.vectorial_hfpi import (  # noqa: E402
    propagate_vector_hfpi_freespace_aperture)

LAM = 633e-9
print("=" * 74)
print("(A) hf.py:167 np.isclose default atol=1e-8 m swallows the resample")
print("=" * 74)
print(f"  np.isclose(1e-6, 1.005e-6, rtol=1e-12) = "
      f"{np.isclose(1e-6, 1.005e-6, rtol=1e-12)}   (0.5 % pitch change)")
print(f"  np.isclose(1e-7, 1.1e-7,   rtol=1e-12) = "
      f"{np.isclose(1e-7, 1.1e-7, rtol=1e-12)}   (10 % pitch change at 100 nm)")
print(f"  np.isclose(1e-7, 2.0e-7,   rtol=1e-12) = "
      f"{np.isclose(1e-7, 2.0e-7, rtol=1e-12)}   (2x pitch change at 100 nm)")
print()
N = 32
for dx, target in ((1e-6, 1.005e-6), (1e-7, 1.1e-7), (1e-7, 2.0e-7),
                   (1e-6, 2.0e-6)):
    E = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out, dx_rep = hf_free(E, 1e-3, LAM, dx, output_dx=target)
    ref = hf_free(E, 1e-3, LAM, dx)          # the un-resampled native field
    identical = np.array_equal(np.asarray(out), np.asarray(ref))
    print(f"  dx={dx:.3e} -> output_dx={target:.3e} ({target/dx:.3f}x): "
          f"returned dx={dx_rep:.4e}, field identical to the UN-resampled "
          f"native field: {identical}"
          + ("   <-- SILENT NO-OP, wrong pitch label" if identical and
             abs(target - dx) > 1e-15 * dx else ""))

print()
print("=" * 74)
print("(B) cone_half_angle is not reachable through the entry points that")
print("    emit the undersampling guard recommending it")
print("=" * 74)
for fn in (propagate_hfpi_freespace_aperture,
           propagate_vector_hfpi_freespace_aperture):
    ps = list(inspect.signature(fn).parameters)
    print(f"  {fn.__name__}:")
    print(f"     cone_half_angle in signature: {'cone_half_angle' in ps}")
    print(f"     params: {ps}")
E = np.ones((16, 16), dtype=np.complex128)
try:
    propagate_hfpi_freespace_aperture(
        E, 2e-6, z_to_aperture=1e-3, aperture_radius=60e-6,
        z_aperture_to_output=1e-3, wavelength=LAM, n_paths=1000,
        cone_half_angle=0.05)
except TypeError as e:
    print(f"  passing it anyway -> TypeError: {e}")
# and through the dispatcher:
from lumenairy.propagators.dispatch import propagate  # noqa: E402
try:
    propagate(E, z=1e-3, wavelength=LAM, dx=2e-6, method='hfpi',
              z_to_aperture=1e-3, aperture_radius=60e-6,
              z_aperture_to_output=1e-3, n_paths=1000,
              cone_half_angle=0.05, return_result=False)
except TypeError as e:
    print(f"  via propagate(method='hfpi') -> TypeError: {e}")
