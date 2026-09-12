"""Return-contract check: propagate(method='hf', output_dx=...) vs others."""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.dispatch import propagate

lam = 633e-9
N, dx = 32, 2e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
E = np.exp(-(np.hypot(X, Y)/12e-6)**2).astype(np.complex128)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for kw in ({}, {'output_dx': dx*0.5}, {'output_grid': {'N': 32, 'dx': dx*0.5}}):
        for meth in ('hf', 'gbd'):
            try:
                r = propagate(E, wavelength=lam, dx=dx, z=1e-3,
                              method=meth, **kw)
                d = getattr(r, 'field', None)
                print(f"  method={meth:<5s} kwargs={str(kw)[:38]:<40s} -> "
                      f"{type(r).__name__}"
                      + (f"  .field={type(d).__name__}" if d is not None else ""))
                if isinstance(r, tuple):
                    print(f"        tuple contents: "
                          f"{[type(t).__name__ for t in r]}")
                elif d is not None and isinstance(d, tuple):
                    print(f"        !! PropagationResult.field is a TUPLE: "
                          f"{[type(t).__name__ for t in d]}")
            except Exception as e:
                print(f"  method={meth:<5s} kwargs={str(kw)[:38]:<40s} -> "
                      f"{type(e).__name__}: {str(e)[:90]}")
