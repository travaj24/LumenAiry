"""PROP-HF p10: is 'hf' the ONLY _OUTPUT_GRID_CAPABLE_METHODS member whose
free-space return TYPE changes when an output grid is requested?"""
import sys
import warnings
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.dispatch import propagate  # noqa: E402

LAM = 633e-9
N, dx = 32, 20e-6
E = np.ones((N, N), dtype=np.complex128)
hfpi_kw = dict(z_to_aperture=1e-3, aperture_radius=300e-6,
               z_aperture_to_output=1e-3, n_paths=5000,
               on_undersampled='silent')
gbd_kw = dict(sample_step=8, chunk_beamlets=256)


def describe(o):
    if isinstance(o, tuple):
        return (f"TUPLE len={len(o)} "
                + ", ".join(f"{type(e).__name__}{getattr(e,'shape','')}" for e in o))
    return f"{type(o).__name__} {getattr(o, 'shape', '')}"


print(f"{'method':6s} | {'no grid kwargs':36s} | with output_grid")
print("-" * 100)
for m, kw in (('asm', {}), ('gbd', gbd_kw), ('hf', {}), ('hfpi', hfpi_kw)):
    row = []
    for grid in ({}, {'output_grid': {'N': 16, 'dx': 2 * dx},
                      'output_dx': 2 * dx}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                o = propagate(E, z=1e-3, wavelength=LAM, dx=dx, method=m,
                              return_result=False, **kw, **grid)
            row.append(describe(o))
        except Exception as e:
            row.append(f"{type(e).__name__}: {str(e)[:40]}")
    print(f"{m:6s} | {row[0]:36s} | {row[1]}")
