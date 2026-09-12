"""TR-INFRA 3b: end-to-end reachability of the _geometric_lens_phase AttributeError."""
import traceback, warnings
import numpy as np
from lumenairy.elements._lens_traced import apply_real_lens_traced, _geometric_lens_phase
from lumenairy.io.prescriptions_builders import make_singlet

lam=587.6e-9; N=128; dx=40e-6
P = make_singlet(0.100,-0.100,2e-3,'N-BK7',aperture=0.004)
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
E=np.exp(-(X**2+Y**2)/(1.0e-3)**2).astype(np.complex128)

print("--- direct _geometric_lens_phase on a REFRACTING prescription ---")
try:
    _geometric_lens_phase(P, lam, dx, N)
    print("  returned OK")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print("\n--- apply_real_lens_traced(fast_analytic_phase=True) ---")
for pip in (True, 'remap', False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens_traced(E, prescription=P, wavelength=lam, dx=dx,
                                         ray_subsample=8, fast_analytic_phase=True,
                                         preserve_input_phase=pip,
                                         on_undersample='silent', on_noncollimated='silent')
        print(f"  preserve_input_phase={pip!r}: returned OK, |E|max={np.abs(out).max():.4e}")
    except Exception as e:
        print(f"  preserve_input_phase={pip!r}: {type(e).__name__}: {e}")

print("\n--- control: fast_analytic_phase=False ---")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    out = apply_real_lens_traced(E, prescription=P, wavelength=lam, dx=dx,
                                 ray_subsample=8, on_undersample='silent',
                                 on_noncollimated='silent')
print(f"  returned OK, |E|max={np.abs(out).max():.4e}")
