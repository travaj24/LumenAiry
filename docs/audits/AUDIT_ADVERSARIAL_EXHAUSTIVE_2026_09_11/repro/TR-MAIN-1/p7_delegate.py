"""Probe 7: on_noncollimated='delegate' -- what is forwarded, what is silently
dropped, and dtype/shape parity."""
import warnings, inspect, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens, apply_real_lens_traced
from lumenairy.elements import _lens_traced as LT
WL = common.WL; k0 = 2*np.pi/WL

sig = inspect.signature(apply_real_lens_traced).parameters
reported = ['carrier','amplitude_model','preserve_input_phase','remap_sampling',
            'caustic','output_plane_distance','tilt_aware_rays',
            'fit_radius_beam_factor','inversion_method','inverse_map',
            'newton_fit','newton_max_iters','newton_poly_order',
            'decentred_fit_poly_order','ray_subsample','return_screen']
forwarded = ['prescription','wavelength','dx','bandlimit','amp_use_gpu',
             'wave_propagator','sag_dtype','sag_chunk_rows','progress','carrier']
allk = [k for k in sig if not k.startswith('_') and k not in ('E_in',)]
silent = [k for k in allk if k not in reported and k not in forwarded]
print("kwargs neither forwarded to apply_real_lens nor reported as dropped:")
for k in silent:
    print(f"   {k:34s} default={sig[k].default!r}")

# --- behaviour: does the delegate return the same dtype/shape? ---
N = 256; AP = 4e-3; dx = 1.4*AP/N
rx = common.plano_convex(R=60e-3, t=3e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
w = 0.6e-3
# strongly diverging input -> trips the collimation guard
Rin = 8e-3
E_in = (np.exp(-(X**2+Y**2)/w**2)*np.exp(1j*k0*(X**2+Y**2)/(2*Rin))).astype(np.complex128)
for dt in (np.complex128, np.complex64):
    Ei = E_in.astype(dt)
    with warnings.catch_warnings(record=True) as wl_:
        warnings.simplefilter('always')
        Ed = apply_real_lens_traced(Ei, prescription=rx, wavelength=WL, dx=dx,
                                    ray_subsample=8, n_workers=1,
                                    on_noncollimated='delegate',
                                    on_undersample='silent',
                                    min_coarse_samples_per_aperture=0,
                                    on_pool_memory='silent')
        msgs = [str(m.message)[:100] for m in wl_]
    Ea = apply_real_lens(Ei, prescription=rx, wavelength=WL, dx=dx)
    print(f"dtype in={np.dtype(dt)}  delegate out={Ed.dtype} shape={Ed.shape} "
          f" == apply_real_lens? {np.array_equal(Ed, Ea)} (dtype {Ea.dtype})")

# --- return_screen=True + delegate: a FIELD is returned where a SCREEN was asked
with warnings.catch_warnings(record=True) as wl_:
    warnings.simplefilter('always')
    Es = apply_real_lens_traced(E_in, prescription=rx, wavelength=WL, dx=dx,
                                ray_subsample=8, n_workers=1,
                                on_noncollimated='delegate', return_screen=True,
                                on_undersample='silent',
                                min_coarse_samples_per_aperture=0,
                                on_pool_memory='silent')
Eref = apply_real_lens(E_in, prescription=rx, wavelength=WL, dx=dx)
print(f"\nreturn_screen=True + delegate -> returned array == apply_real_lens(E_in)?"
      f" {np.array_equal(Es, Eref)}  (i.e. an INPUT-DEPENDENT field, not a screen)")
print(" warnings:", [str(m.message)[:150] for m in wl_])

# --- newton_amp_mask_rel / beam_centre are dropped with no mention ---
with warnings.catch_warnings(record=True) as wl_:
    warnings.simplefilter('always')
    apply_real_lens_traced(E_in, prescription=rx, wavelength=WL, dx=dx,
                           ray_subsample=8, n_workers=1,
                           on_noncollimated='delegate',
                           newton_amp_mask_rel=0.0,
                           beam_centre=(1e-4, 0.0),
                           newton_mask_dilate_coarse_px=7,
                           fast_analytic_phase=True,
                           on_undersample='silent',
                           min_coarse_samples_per_aperture=0,
                           on_pool_memory='silent')
print("\nwith newton_amp_mask_rel=0, beam_centre, dilate=7, fast_analytic_phase=True:")
for m in wl_:
    print("   ", str(m.message)[:300])
