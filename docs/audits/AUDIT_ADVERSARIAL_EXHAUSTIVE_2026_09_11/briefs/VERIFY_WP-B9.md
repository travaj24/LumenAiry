# VERIFY-WP-B9 -- independent adversarial re-verification of WP-B9 (ray tracing: renormalise hoist, analytic sphere normal, one-bundle fans, trace_jax prescription cache, `pattern=` pupil sampling, aspheric analytic Jacobian)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B9 is commit 7592af4a (its parent
is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers may be editing other subsystems concurrently; never touch
those, and expect their fingerprint checks to be red until they re-record.  You own `lumenairy/raytrace/*` for this pass; `analysis/ghost.py` and
`elements/_lens_traced.py` consume the tracer and are NOT yours.

You did not write WP-B9.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B9's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B9_RAYTRACE_PERFORMANCE.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B9_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b9_raytrace_perf.py`, `fixes/WP-A1_REPORT.md` section 6 (the designs), `VERIFY_WP-A1.md` (60-digit closed-form oracles:
reuse the METHOD) and `WP-A26_REPORT.md` section 2.4 (the exact conic trace).

## What to do, in order

1. **Attack the renormalise hoist.**  On YOUR OWN prescriptions (a 12-surface double Gauss, a Cassegrain with two mirrors, a TIR prism path, a
   grazing-incidence marginal ray) compare 7592af4a with the parent: byte-identical, or within the derived ~1e-16-per-surface drift the report claims
   (measure the drift's actual growth with surface count in mpmath).  Degenerate rays: a ray that misses a surface and one that goes evanescent must
   still be `RAY_NAN` at the SAME surface index.  `analysis/ghost.py` and the differential FD path: identical to the parent by construction? Prove it.
2. **Attack the analytic sphere normal.**  R > 0 and R < 0, a ray at the vertex (r = 0), a marginal ray at r -> |R| (grazing), a plano surface
   (R = inf must NOT take the spherical path), a conic with k = 0 exactly versus a surface flagged `is_pure_spherical`.  The gate: the Maslov
   cross-backend asymptotic test, `tests/unit/test_audit_propagation.py -k ModalAsymptoticStillBitEqual`,
   `test_niche_audit_w6_asymptotic.py::test_w6_a2_v2_star...`, `test_niche_d7_decentred_fit.py` (2.371 / 1.683 urad byte-identical) and every
   `_lens_traced` fixture -- run them and confirm the numbers, not just green.
3. **Attack the one-bundle fans.**  RT-5 (`ey(0) == ex(0) == 0`) on a decentred and a tilted prescription; the concatenated bundle versus four
   separate traces byte-identical, INCLUDING when one sub-fan contains a `RAY_NAN` ray (does a `where` on the merged bundle couple fans?).
4. **Attack the `trace_jax` prescription cache.**  Mutate each field of a prescription one at a time (radius, thickness, index, conic, aspheric
   coefficients, semi-diameter, any tilt/decentre, wavelength / dispersion inputs) and confirm a cache miss and the right answer; mutate a numpy
   array IN PLACE after the first call (a stale hit?); check the cache bound and eviction; two prescriptions differing only in a float at 1 ulp.
5. **Attack `pattern=`.**  Mean r/R of the default (0.5806) and the sunflower (0.6667) on your own ring counts; the default byte-identical; the
   spot-rms shift on a third prescription; the pass-through in every `spot_rms` consumer (grep them).
6. **Attack the aspheric analytic Jacobian.**  Your OWN oracle: JAX autodiff (or complex-step) through an independent aspheric trace at three
   heights and two aspheric orders; FD with the engineer's step and with a 10x smaller step (the tolerance must be derived from the step);
   `aspheric_coeffs` all zero byte-identical to the conic path; an aspheric term large enough that the Newton intersection needs more iterations.
7. **Attack the byte-identity claims** against `git archive 7592af4a^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b9\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest), and **the pins** (derived envelopes or
   per-build numbers? fail-before real? mutate in memory -- renormalise per surface again, use the generic normal, drop a field from the cache key --
   and confirm the tests go red).
8. **Fix what you find, in `lumenairy/raytrace/*`**; re-record every history document you touch.  Anything outside goes under "Requested changes
   outside my ownership" with the exact edit.  Do not change any default.
9. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B9.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to a
   new `VERIFY_WP-B9_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/raytrace/*.py`, their `docs/history/` documents, `tests/unit/test_audit2609_b9_raytrace_perf.py` (add, do not weaken), your
  two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
