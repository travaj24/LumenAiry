"""PROP-HF p6: exports, stratified-sampling path blow-up, complex64 OPL
accumulator precision, wavelength=0 default."""
import sys
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la  # noqa: E402
from lumenairy import propagators as P  # noqa: E402
from lumenairy.propagators import hf as HF, hfpi as HFPI  # noqa: E402
from lumenairy.propagators import vectorial_hfpi as VHF  # noqa: E402
from lumenairy.propagators.hfpi import (  # noqa: E402
    init_paths_from_field, init_paths_stratified, apply_aperture_diffraction,
    propagate_to_plane)

LAM = 633e-9


def sec(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


sec("(1) canonical entry points missing from __all__")
for mod, name in ((HF, 'propagate_huygens_fresnel'),
                  (HFPI, 'propagate_hfpi')):
    print(f"  {mod.__name__}.__all__ contains {name!r}: "
          f"{name in mod.__all__}   "
          f"reachable as lumenairy.{name}: {hasattr(la, name)}   "
          f"as lumenairy.propagators.{name}: {hasattr(P, name)}")
print(f"  vectorial_hfpi reachable from top level: "
      f"{[n for n in VHF.__all__ if hasattr(la, n)]}")
print(f"  vectorial_hfpi names in lumenairy.propagators: "
      f"{[n for n in VHF.__all__ if hasattr(P, n)]}")

sec("(2) init_paths_stratified: n_paths is not a cap")
E = np.ones((16, 16), dtype=np.complex128)
for kwargs, label in (
        (dict(n_paths=20000), 'defaults, n_paths=20000'),
        (dict(n_paths=100, n_strata_xy=(16, 16), n_strata_dir=(16, 16)),
         'n_paths=100, strata 16^4'),
        (dict(n_paths=100, n_strata_xy=(32, 32), n_strata_dir=(32, 32)),
         'n_paths=100, strata 32^4')):
    p = init_paths_stratified(E, 2e-6, wavelength=LAM, rng=1, **kwargs)
    n = len(p)
    mem = n * (3 + 3) * 8 + n * 16 + n * 8 + n  # pos+dir+weights+opl+alive
    print(f"  {label:34s} requested {kwargs['n_paths']:8d} -> allocated "
          f"{n:10d} paths ({n/kwargs['n_paths']:8.1f}x, ~{mem/1e6:8.2f} MB)")

sec("(3) complex64 source -> float32 OPL accumulator")
E32 = np.ones((8, 8), dtype=np.complex64)
p = init_paths_from_field(E32, 2e-6, n_paths=5, wavelength=LAM, rng=1)
print(f"  complex64 E_in  -> opl dtype {p.opl.dtype}, weights {p.weights.dtype},"
      f" positions {p.positions.dtype}")
q = propagate_to_plane(p, 1e-3, LAM)
print(f"  after one free-space hop, opl dtype = {q.opl.dtype} "
      f"(numpy promotion rescues the free-space leg)")
# but the prescription leg reads paths.opl straight into RayBundle.opd:
for L in (1e-3, 1e-2, 1e-1, 1.0):
    opd32 = np.float32(L)
    eps = float(np.spacing(opd32))
    print(f"  OPD {L:6.3f} m in float32: spacing {eps:.3e} m -> "
          f"phase quantum {2*np.pi*eps/LAM:8.4f} rad at 633 nm")

sec("(4) apply_aperture_diffraction default wavelength=0 silently drops 1/(i lam)")
import inspect  # noqa: E402
print("  signature default:",
      inspect.signature(apply_aperture_diffraction).parameters['wavelength'].default)
p = init_paths_from_field(np.ones((8, 8), np.complex128), 2e-6, n_paths=4,
                          wavelength=LAM, rng=1, cone_half_angle=0.1)
a = apply_aperture_diffraction(p, 1.0, rng=1, cone_half_angle=0.1)          # no wavelength
b = apply_aperture_diffraction(p, 1.0, rng=1, wavelength=LAM, cone_half_angle=0.1)
print(f"  |w| ratio (default vs correct) = "
      f"{np.abs(a.weights[0])/np.abs(b.weights[0]):.4e}   "
      f"(should be lam = {LAM:.3e}); phase differs by "
      f"{np.angle(a.weights[0]/b.weights[0]):+.4f} rad")

sec("(5) _resolve_output_shape duplicated verbatim in hf.py and hfpi.py")
import difflib  # noqa: E402
a_src = inspect.getsource(HF._resolve_output_shape)
b_src = inspect.getsource(HFPI._resolve_output_shape)
same = a_src.replace("'hf'", "'X'") == b_src.replace("'hfpi'", "'X'")
print(f"  identical apart from the method name in the warning text: {same}")
print(f"  vectorial_hfpi imports hfpi's private helpers: "
      f"{[n for n in ('_resolve_output_shape', '_spawn_rng', '_complex_output_dtype') if hasattr(VHF, n)]}")
