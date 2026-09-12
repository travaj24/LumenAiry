"""Extra: REAL-dtype E_in + carrier -> _reference_input() casts exp(i k0 W)
to E_in.dtype, which for a real input DISCARDS the imaginary part."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import _compute_carrier
WL = common.WL; k0 = 2*np.pi/WL
N = 256; AP = 6e-3; dx = 1.35*AP/N
rx = common.plano_convex(R=100e-3, t=4e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R2 = X**2+Y**2
S = 200e-3
W = np.sqrt(R2+S*S)-S
env = np.exp(-R2/(1.0e-3)**2)
E_c = (env*np.exp(1j*k0*W)).astype(np.complex128)
E_r = env.astype(np.float64)      # a REAL field (allowed by the validator?)

# demonstrate the cast the code performs at _reference_input()
ref = np.exp(1j*k0*W)
print("np.exp(1j k0 W).astype(float64) keeps only the real part: "
      f"max|imag dropped| = {np.abs(ref.imag).max():.4f} (of unit modulus)")
with warnings.catch_warnings(record=True) as wl_:
    warnings.simplefilter('always')
    cast = ref.astype(np.float64)
    print("  cast warnings:", [str(m.message)[:70] for m in wl_])
print("  |cast| range:", float(np.abs(cast).min()), float(np.abs(cast).max()))

kw = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=8, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent', on_noncollimated='off')
for tag, E in (('complex128 input', E_c), ('float64 input', E_r)):
    for c in (None, S):
        try:
            with warnings.catch_warnings(record=True) as wl_:
                warnings.simplefilter('always')
                out = apply_real_lens_traced(E, carrier=c, **kw)
            msgs = sorted({type(m.message).__name__ for m in wl_})
            print(f"  {tag}, carrier={c}: out dtype={out.dtype} "
                  f"finite={np.isfinite(out).all()} |E|max={np.abs(out).max():.4e} "
                  f"warns={msgs}")
        except Exception as e:
            print(f"  {tag}, carrier={c}: {type(e).__name__}: {str(e)[:120]}")
