# VERIFY-WP-C2 probes

Independent adversarial verification of WP-C2 (`sphere_normal='analytic'` and
`renormalize='exit'` as the ray tracer's defaults).  Report:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C2.md`.

Every probe runs in a child process with `LUMENAIRY_ROOT` on `sys.path` and
`lumenairy.__file__` asserted inside it, on Windows py3.14.6 / numpy 2.4.4 and
WSL py3.12.3 / numpy 2.4.6, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line.  `*_win.json` / `*_wsl.json` are the
recorded outputs.

| probe | what it settles | headline |
|---|---|---|
| `vc2_sphere_oracle.py` | the normal against a 60-digit oracle fed the EXACT float64 inputs, on 2240 points of its own | closed form 1.00 against the generic route's 1.75 out to `h = 0.95 R`, 0 of 1280 worse by > 1; the shipped probe's `repr` input conversion contributes 1.00 at 0.95 R and 41.6 at the clamp |
| `vc2_trace_truth.py` | a 60-digit END-TO-END trace oracle, and the norm-drift sensitivity | all four default combinations within 5.2e-18 m / 7.0e-17 m of the truth; drift sensitivity 1.786e-3 m and 8.95e-4 m per unit, LINEAR |
| `vc2_opcount.py` | what each switch costs, COUNTED (ndarray subclass on both dispatch protocols) | hoist 0.9910x at 2 surfaces, 1.0237x at 13; normal 1.1612x-1.1973x, controls exactly 1.0000x |
| `vc2_renorm_cost.py`, `vc2_timing.py` | the timed instruments, and their failure modes | the Windows process tick read the block as ZERO; two wall-clock runs disagreed in SIGN |
| `vc2_vignetting.py` | both gates located by bisection; the straddle; twelve sweeps | the gates coincide EXACTLY on the axis; 580 000 rays move zero flags; a BALL LENS crosses the clamp and loses 4.9 % to RAY_NAN |
| `vc2_jax.py` | CPU/JAX parity, and whether JAX has the clamp | parity 3.469e-18 m under all five CPU settings; JAX has NO clamp and keeps 1991 of 1991 rays the CPU kills |
| `vc2_entrypoints.py` | the way-back census, from the package's AST | 21 exported directly-tracing entry points with neither keyword (16 CPU-affected), against the report's six |
| `vc2_wayback.py` | the way back, archive to archive in a second process | 594 / 594 byte-identical on both builds |
| `vc2_pins.py` | the two knife-edge pins, re-derived | the disagreement is DENSE (71.5-75.6 % of pixels above 10 % of max, not one); `w6_a2` confirmed to 9.9e-32; kappa is direction-dependent by 3.8x |
| `vc2_d3_bars.py` | the two restated d3 bars | readings reproduce; the arm-2 floor's own spread is 3.22x / 4.79x against a 3.0 multiplier |
| `vc2_ghost_and_reanchor.py` | ghost's normal, and the `EDITED_IN_PLACE` guard | ghost differs by 2.1e-14 mm; the override accepts a reverted default, a nonsense value and a stale copy |
| `vc2_mutplugin.py` | the mutation matrix (a pytest plugin; nothing under `lumenairy/` is edited) | 8 of 9 caught on both builds; `jax_gets_a_clamp` survives 294 tests; `whole_normal_sign_flipped` survives correctly |
