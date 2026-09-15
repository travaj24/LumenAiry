# VERIFY-WP-B7b -- independent adversarial re-verification of WP-B7b (the caustic routing to `phase_screen` for a single-valued field inside the aberration envelope, the `caustic='uniform'` fold envelope + warning, the FGA analytic-Jacobian predicate)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B7b is commit 9cf94fa5 (its parent
`ea374607` is the pre-change library).  You own `lumenairy/propagators/fga.py`, `lumenairy/elements/_lens_traced_uniform.py`, the test files WP-B7b
changed (`tests/unit/test_audit2609_a4_fga_s10.py`, `tests/unit/test_fga_h4_h5.py`), `tests/unit/test_audit2609_b7b_caustic_routing.py` and your two
report files; nothing else.  Neither owned module has a `docs/history/` document (`scripts/record_history_fingerprints.py <path> --check` says so), so
there is nothing to re-record; do not create one.

You did not write WP-B7b.  Two of its three items MOVE DEFAULTS, so the whole question is whether each move is supported by measurement -- and
whether what it declined to move is supported too.  Read `docs/TESTING_STANDARDS.md`, `CONVENTIONS.md` sec. 2 and 7, WP-B7b's brief
(`...\scratchpad\briefs\WP-B7b.md`), its report and changelog (`fixes/WP-B7b_REPORT.md`, `WP-B7b_CHANGELOG.md`), `fixes/WP-B7_REPORT.md`
sections 2, 6, 8.2-8.4 and `fixes/VERIFY_WP-B7.md`.

## What WP-B7b actually shipped (verify THIS, not the brief's original wording)

* **Item 1.** `_universal_route`'s caustic branch returns `"fga" if aberrated else "phase_screen"` -- the route to the screen is GATED on the H2
  sag-screen aberration envelope (`_sag_screen_aberration_rad <= aberration_threshold`), because the G1 matrix designs (M1 / M2 / M4 / M6, 20-2893 rad)
  all reach the caustic branch and `test_g1_gate_generality.py` / `test_fga.py`'s H2 rows pin them away from the screen on a dual-oracle measurement.
  Nine routing decisions move.  WP-B7b measured that on an over-budget fixture at the SAME NA (k = 30, 2.289 rad) the gate keeps the WORSE member
  (`fga` 0.1228 vs `phase_screen` 0.9990) and escalated: "score the three members at the H2 f/5 singlet's image plane; if `phase_screen` wins, drop
  the condition".
* **Item 2.** No field change.  WP-B7's diagnosis was found wrong twice (the 0.0030 was an under-resolved-grid FALLBACK; the CFU control-parameter
  fit is exact to 7 digits).  The defect found instead: `zeta = kappa (r_c - r)` is EXTRAPOLATED past the two-branch band it was fitted on (454x on the
  fixture), and the error is the dark tail's ENERGY (+22.8 % at ratio 454, +12.5 % at 9.8, +4.9 % at 5.4, within 5 % below that).  Shipped:
  `_trace_meridional_fold` returns `band`; `zeta_band` / `zeta_extrapolation` diagnostics; a `RuntimeWarning` above `_ZETA_EXTRAPOLATION_MAX = 8.0`
  (placed between the 5.4 and 9.8 rungs); a docstring envelope.  Field byte-identical on all three probe cases.
* **Item 3.** `_is_all_conic` -> `_analytic_jacobian_applies`: the predicate is the analytic primitive's own domain -- an even-aspheric departure
  now takes `ray_transfer_jacobian_analytic` (it was silently FD, `exact_jacobian=True` ignored), and a field-decentred / tilted / sag-callable conic
  now FALLS BACK to FD (it raised `NotImplementedError` at call time on the parent).  All-conic fields byte-identical.

## What to do, in order

1. **Re-derive item 1 on your OWN fixture** (a different singlet, glass, wavelength, NA in 0.10-0.20): your own brute-force Rayleigh-Sommerfeld from
   an exact conic raytrace (converged on the pupil sampling, floor stated; check the exit SLOPE-vs-direction-cosine trap WP-B7b fell into) at the
   focus and at two defocused planes; fidelity and intensity-rms width for `'fga'` (defaults and its best sampling) vs `'phase_screen'` vs `'traced'`
   vs the oracle; confirm or refute "phase_screen is closer at every NA" and "FGA converges to the wrong field".  Check the MULTI-VALUED branch still
   routes to `'fga'` (build a genuinely multi-valued input) and that the restated tests' single-valued rows are the ones that changed.
   **Then attack the GATE**: WP-B7b could not reach the ~20 rad regime on a tractable grid.  Try: the H2 f/5 singlet (M6 of the G1 matrix, 20.4 rad)
   at its image plane with an oracle you can converge (a Debye / vector-Debye or a well-converged RS over a decimated but ADEQUATE pupil -- state the
   floor); if that is intractable in your budget, build the largest-aberration fixture you CAN converge and report where between 2 and 20 rad the
   member ranking flips, if it does.  Decide with numbers whether `aberrated` should stay on the route; do NOT move it yourself -- report.
   Byte-identity of every routing decision OUTSIDE the caustic gate against `git archive 9cf94fa5^ lumenairy` (child process, cwd + PYTHONPATH = the
   archive, `lumenairy.__file__` asserted; never pytest, never the shared tree).
2. **Attack item 2**: reproduce a fold on a different f/# singlet and grid that the module's resolution gate ACCEPTS (`l_airy >= 1.2 dx` -- WP-B7b's
   report sec. 3.1 shows how WP-B7 tripped the fallback); score `caustic='uniform'` / `'multibranch'` / `'wave'` / `amplitude_model='ray_density'`
   against your oracle at two or more extrapolation ratios; is the `_ZETA_EXTRAPOLATION_MAX = 8.0` bar derived (two-sided on YOUR ladder: silent where
   the energy error is inside ~5 %, warning where it is past ~10 %) or is it one fixture's number?  Is the field byte-identical to the parent on
   your fixtures (archive-to-archive)?  Does `zeta_extrapolation` reproduce from `band` and `l_airy` independently?
3. **Attack item 3**: an aspheric prescription under `exact_jacobian=None` / `True` now takes the analytic Jacobian in FGA -- compare against FD with
   a step ladder and against `jax.jacfwd` through an independent trace on your own asphere; a field-decentred conic must now FALL BACK to FD instead
   of raising (fail-before on the parent, isolated tree); a biconic still falls back; `exact_jacobian=False` still forces FD; the all-conic field is
   byte-identical.
4. **Attack the Migration notes**: every call the notes say changes its answer must be shown changing by about the stated amount on the stated
   fixture, and the stated way back (`method='fga'`, `exact_jacobian=False`) must restore the parent's bytes.
5. **Attack the pins**: derived envelopes or per-build numbers?  Mutate (revert the route; drop the `aberrated` condition; drop the predicate
   widening; move the `zeta` bar) and confirm the right tests go red and ONLY those.
6. **Fix what you find, in the owned files.**  Anything outside goes under "Requested changes outside my ownership" with the exact edit.  Do not move
   any further default, and do not drop the `aberrated` condition -- report the measurement that decides it.
7. **Report** `fixes/VERIFY_WP-B7b.md` (verdict table with your oracle and numbers, defects with fail-before, Follow-up, every command + counts +
   durations); `VERIFY_WP-B7b_CHANGELOG.md` only if you changed library code.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time, in the FOREGROUND (a turn boundary
  kills background children); `-X faulthandler` on long selections.  Scratch files under `...\scratchpad\vb7b\`.
* NO git write commands of any kind (no add / commit / stash / checkout / restore / reset); do not kill processes.  The orchestrator commits with an
  explicit file list.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
