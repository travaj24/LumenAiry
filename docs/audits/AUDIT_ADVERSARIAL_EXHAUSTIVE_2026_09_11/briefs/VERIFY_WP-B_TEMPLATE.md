# VERIFY-{WP} -- independent adversarial re-verification of {WP} ({TITLE})

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`, HEAD {HEAD}
(the {WP} commit).  Other Wave-4 engineers may be editing other subsystems concurrently; the files you may touch are listed at the end.

You did not write {WP}.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then {WP}'s own brief
(`{BRIEF_PATH}`), its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/{WP}_REPORT.md` and changelog
`{WP}_CHANGELOG.md`, and the WP's new test file {TEST_FILE}.

## What to do, in order

1. **Re-derive every number in the report's summary table on a fixture the engineer did not use.**  Build your own oracle for the
   package's claim -- an analytic closed form, a brute-force quadrature, an independent exact trace -- sharing no code with the library
   path under test (the repository's `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/` scripts show the METHODS used in
   Wave 3: brute-force Rayleigh-Sommerfeld, Newton+Snell tracers, closed-form Fresnel, ABCD Gaussians).  State the oracle's own floor.
2. **Attack the byte-identity claims.**  Every path the report says did not move: prove it with `np.array_equal` against
   `git archive {HEAD}^ lumenairy` extracted READ-ONLY into your scratch directory
   (`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_{wp}\`;
   import the archived library from a child process whose cwd and PYTHONPATH are the archive -- never through pytest, which puts the
   repository root first -- and assert `lumenairy.__file__` per probe).
3. **Attack the new pins.**  For every new test: is the envelope DERIVED (oracle, floor, defect scale) or a per-build number?  Does the
   fail-before actually fail on the pre-change library?  Does the test pass for the wrong reason (mutate the fix in memory and confirm
   the test goes red)?
4. **Attack the edges.**  Decentred / tilted / converging inputs, the smallest and largest grids the guards admit, complex64,
   the JAX/GPU twin where one exists, `LensConfig` objects where the entry point takes them, the `_multi` / parallel path where one exists.
5. **Fix what you find, in the package's own files**, with the same standards ({WP}'s ownership list below); re-record the history
   documents you touch (`python scripts/record_history_fingerprints.py <module> --reason "..."`).  Anything outside those files goes under
   "Requested changes outside my ownership" with the exact edit.
6. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_{WP}.md`: a verdict table (each report claim ->
   VERIFIED / VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), the defects you found and fixed (with
   fail-before), a "Follow-up" section for anything ruled open, and the exact commands + counts + durations of every test run.  If you
   changed library code, add `### Fixed -- ...` release text to a new `VERIFY_{WP}_CHANGELOG.md` in the same style as the WP changelog.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind (no add/commit/stash/checkout/restore/reset); read-only `git show` / `git archive` / `git log` are
  fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: {OWNED_FILES}, their `docs/history/` documents, {TEST_FILE}, your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
