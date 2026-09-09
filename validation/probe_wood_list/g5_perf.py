import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
assert os.path.abspath(lumenairy.__file__).replace("\\", "/").lower().startswith(_ROOT)
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

mo = np.arange(-7, 8)
mx, my = np.tile(mo, 15), np.repeat(mo, 15)
cell = np.where(np.random.default_rng(0).random((64, 64)) > 0.5, 4.0, 1.0)
diag = np.repeat(cell.ravel(), 3)          # the pre-fix tensor list shape
uniq = list(np.unique(np.real(diag)))
for tag, lst in (("raw diagonals (%d)" % len(diag), list(diag)),
                 ("deduped (%d)" % len(uniq), uniq)):
    t = time.perf_counter()
    for _ in range(3):
        wl = _grazing_safe_wavelength(0.633e-6, 0.0, 0.0, mx, my, 0.8e-6,
                                      0.8e-6, lst)
    print(f"  {tag}: {(time.perf_counter()-t)/3*1e3:.2f} ms/call  wl={wl!r}")
