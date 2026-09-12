"""T19: the NumPy `_lg00_sampling_waist` clamps w_o to [1e-9, 1.0];
the JAX twin does not.  The docstrings claim the two are identical
'character for character' / the documented cross-backend contract."""
import sys, math
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import jax; jax.config.update('jax_enable_x64', True)
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, aberration_tensor, aberration_tensor_lg00_jax,
    solve_envelope_stationary, _compute_M_b)
from lumenairy.propagators.asymptotic_aberration_tensor import _lg00_sampling_waist

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                pupil_box_half=0.02, n_field=8, n_pupil=8, poly_order=6)
vc = (fit.v2x_centre, fit.v2y_centre)
for (w_s, w_p) in ((20e-6, 0.02), (1.0, 1.0), (30.0, 30.0), (100.0, 100.0)):
    v, _, _ = solve_envelope_stationary(fit, (0.0,0.0), (0.0,0.0), w_s=w_s,
                                        w_p=w_p, v2_centre=vc)
    M, *_ = _compute_M_b(fit, 0.0, 0.0, v[0], v[1], 0.0, 0.0, w_s, w_p, vc[0], vc[1])
    ev = np.linalg.eigvalsh(np.real(M))
    wo_np = _lg00_sampling_waist(M)
    wo_jx = 1.0/math.sqrt(max(float(ev[-1]), 1e-30))
    Lnp = aberration_tensor(fit, (0.0,0.0), output_modes=[(0,0)], w_s=w_s, w_p=w_p,
                            v2_centre=vc).L[0,0]
    Ljx = complex(aberration_tensor_lg00_jax(fit, (0.0,0.0), v, w_s=w_s, w_p=w_p,
                                             v2_centre=vc))
    rel = abs(Lnp-Ljx)/max(abs(Lnp), 1e-300)
    flag = "  <-- DIVERGES" if rel > 1e-8 else ""
    print(f"w_s={w_s:<8g} w_p={w_p:<6g} lam_max(ReM)={ev[-1]:.4e}  "
          f"w_o numpy={wo_np:.6e} jax={wo_jx:.6e}   |L| np={abs(Lnp):.5e} "
          f"jax={abs(Ljx):.5e}  rel={rel:.3e}{flag}")
