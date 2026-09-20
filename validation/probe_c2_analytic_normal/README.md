# WP-C2 probes -- `sphere_normal='analytic'` and `renormalize='exit'` as defaults

Child-process probes for WP-C2 (5.49.0 default flip).  Every one of them
puts `LUMENAIRY_ROOT` on `sys.path` and asserts `lumenairy.__file__` is
inside it before doing anything, so an installed `lumenairy` in
site-packages cannot bind silently.  None of them runs under pytest.

Run each on BOTH builds:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LUMENAIRY_ROOT=<root> PYTHONPATH=<root> python <probe>.py <out>.json
```

Files ending `_win.json` are Windows py3.14 / numpy 2.4.4; `_wsl.json`
are WSL py3.12 / numpy 2.4.6.

Where a probe sweeps the ray tracer's defaults it rewrites
`trace.__defaults__` / `trace_world.__defaults__` on the function
objects -- which is what "flip the default" means and reaches every
module that already imported them -- exactly as VERIFY-WP-B9 section 4
did.  Any `lru_cache`d fixture that is built by TRACING (for instance
`test_niche_audit_w6_asymptotic._fit`) is cleared between combinations;
forgetting that is a real trap, and it silently reports the first
combination's number four times.

## Item 1 -- the two knife-edge pins

| probe | what it measures |
|---|---|
| `pins.py` | the raw readings of the three arms under all four `(renormalize, sphere_normal)` combinations, with the per-pixel population of the modal comparison |
| `modal_mechanism.py` | WHERE the modal disagreement comes from: batch vs scalar saddle LOCATION against field evaluation |
| `modal_conditioning.py` | the field's amplification `kappa` of a relative perturbation of its own inputs, with a linearity ladder -- the `eps * kappa` agreement floor |
| `w6a2_oracle.py` | a 60-digit `decimal` Newton on the same polynomial system: the exact root of the fit |
| `w6a2_asymmetry.py` | is the w6_a2 offset rounding or the fit's own asymmetry?  float64 vs 60-digit residual at the pupil centre |
| `w6a2_resolution.py` | the solver's reproducibility over a ladder of independent Newton starts |
| `pins_restated.py` | the RESTATED pins run under all four combinations, with the margins each derived |

Headline findings, both builds:

* the modal quantity is **not** bimodal and there is **no basin flip**.
  The batched and scalar solvers put the saddle in the same place to
  7.3e-18 (8.7e-17 of the pupil half-range) at every one of 1024 pixels;
  the disagreement is a dense field (996-1011 pixels above 1e-12
  relative, median 2.1e-09) at the cancellation floor of the moment
  contraction.  `kappa = 6.24e+06`, flat to 0.3 % over four decades, so
  `eps * kappa = 1.39e-09` and the two routines land at 7.3x to 9.0x it.
* the w6_a2 root is **not** at the pupil centre and never was.  The
  60-digit oracle reproduces what the library returns to <= 4.5e-22, and
  the offset is `H^-1 r(v_c)` where `r(v_c)` -- the fit's own asymmetry
  -- is resolved about 8 decades above its rounding floor.  `1e-15` was
  an S4 bar on how asymmetric a least-squares fit happened to come out.

## Items 2-6 -- the flip itself

| probe | what it measures |
|---|---|
| `sphere_oracle.py` | the 60-digit `decimal` normal ladder on WP-C2's own sphere set (radii, heights to the clamp, both signs, mirrors) |
| `timing.py` | whole-trace wall clock on several prescriptions, interleaved, with the load stated |
| `vignetting.py` | which rays' `alive` flags move between the two routes, and whether any shipped fixture's vignetting count changes |
| `byte_identity.py` | archive-to-archive byte identity against `git archive 49ddf4bd`, with the old keywords passed explicitly |
| `jax_parity.py` | CPU / JAX trace parity re-derived under all four CPU settings (the JAX tracer has NEITHER switch) |
| `renorm_ladder.py` | item 3: the `'surface'` vs `'exit'` difference against the derived `n_surfaces * eps * |t|` envelope, over a 3-to-13-surface ladder, plus the history-drift contract and the single-pass call count |
| `trace_touching_files.txt` | the grep-selected blast-radius file list (368 files: every ray-trace-touching test plus the lens family's traced / FGA / GBD / multibranch files) |

Headline findings, both builds:

* the closed form is within **1.75 ULP** of the 60-digit oracle out to
  `h = 0.95 |R|` against the generic route's 2.00 (Windows) and 2.25
  (WSL), and never worse there by more than 1 ULP at any of 672 points --
  but above `0.95 |R|` both routes are at the conditioning limit of
  `sqrt(1 - u)` and neither dominates point by point;
* `sphere_normal='analytic'` is worth **1.08x to 1.44x** on prescriptions
  that contain spheres, against controls (no sphere) at 0.93x to 1.03x --
  which is this method's own resolution on a loaded box;
  `renormalize='exit'` reads 0.95x to 1.13x, i.e. nothing this method can
  see;
* the rim band VERIFY-B9 3.3 predicted is real and reaches `alive`, but
  360 000 traced rays over twelve prescription and field-angle
  combinations move ZERO alive flags and ZERO error codes;
* CPU / JAX parity is **identical to the last digit** under all four CPU
  settings (3.469e-18 m position, 3.123e-17 m OPL).
