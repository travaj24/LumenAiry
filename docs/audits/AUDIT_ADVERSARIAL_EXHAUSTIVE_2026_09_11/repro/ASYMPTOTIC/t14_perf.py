"""T14: performance / memory profile of the asymptotic family."""
import sys, time, tracemalloc, cProfile, pstats, io
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic, aberration_tensor,
    _solve_envelope_stationary_batch, _compute_M_b_batch)

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
t0=time.perf_counter()
fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                pupil_box_half=0.02, n_field=8, n_pupil=8, poly_order=6)
print(f"fit: {time.perf_counter()-t0:.3f}s, n_basis={len(fit.multi_indices)}")

def grid(n, frac=0.4):
    L = fit.s2x_halfrange*frac
    a = np.linspace(-L, L, n)
    return np.meshgrid(a+fit.s2x_centre, a+fit.s2y_centre, indexing='xy')

vc = (fit.v2x_centre, fit.v2y_centre); w_s, w_p = 20e-6, 0.02
for n in (32, 64, 128, 256):
    X, Y = grid(n)
    t = time.perf_counter()
    E = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc,
                                   s2_grid_x=X, s2_grid_y=Y)
    dt = time.perf_counter()-t
    tracemalloc.start()
    E = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc,
                                   s2_grid_x=X, s2_grid_y=Y)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"  propagate {n:4d}x{n:<4d}  {dt*1e3:8.1f} ms  ({dt/n**2*1e6:6.2f} us/px)"
          f"  peak alloc {peak/1e6:8.2f} MB  = {peak/(n*n):.0f} B/px")

# multi-mode scaling
X, Y = grid(64)
for nm in (1, 3, 10):
    amps = {(p, l): 1.0+0.0j for p in range(4) for l in range(-3, 4)}
    amps = dict(list(amps.items())[:nm])
    t = time.perf_counter()
    propagate_modal_asymptotic(fit, source_amplitudes=amps, w_s=w_s, w_p=w_p,
                               v2_centre=vc, s2_grid_x=X, s2_grid_y=Y)
    print(f"  propagate 64x64, {nm:2d} source modes: {(time.perf_counter()-t)*1e3:8.1f} ms")

# profile the 128x128 case
X, Y = grid(128)
pr = cProfile.Profile(); pr.enable()
propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc, s2_grid_x=X, s2_grid_y=Y)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(16)
print("\n--- profile propagate 128x128 ---")
print("\n".join(s.getvalue().splitlines()[4:28]))

# aberration_tensor with defaults (sigma branch)
t = time.perf_counter()
res = aberration_tensor(fit, (0.0, 0.0), w_s=w_s, w_p=w_p, v2_centre=vc)
print(f"\naberration_tensor default (11 output modes): {time.perf_counter()-t:.2f} s, "
      f"sigma_grid_n={res.sigma_grid_n}, w_o={res.w_o:.4e}")
t = time.perf_counter()
res64 = aberration_tensor(fit, (0.0, 0.0), w_s=w_s, w_p=w_p, v2_centre=vc,
                          sigma_grid_n=64)
print(f"aberration_tensor sigma_grid_n=64:          {time.perf_counter()-t:.2f} s")
t = time.perf_counter()
res00 = aberration_tensor(fit, (0.0, 0.0), w_s=w_s, w_p=w_p, v2_centre=vc,
                          output_modes=[(0, 0)])
print(f"aberration_tensor [(0,0)] closed form:      {time.perf_counter()-t:.4f} s")
