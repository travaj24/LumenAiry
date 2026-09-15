# VERIFY-WP-B1 -- independent adversarial re-verification of WP-B1 (Maslov S6 proper: the saddle follows the input field's local wavevector)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B1 is commit {HEAD} (its
parent is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers are editing `lumenairy/propagators/carrier.py`,
`lumenairy/elements/rcwa/*`, `eme/*`, `bor/*`, `pmm/*`, `lumenairy/analysis/*`, `lumenairy/sources/*`, `lumenairy/raytrace/*`, `_lens_traced.py`,
`hf.py`/`hfpi.py`/`rs.py`/`mft.py` concurrently; never touch those, and expect their fingerprint checks to be red until they re-record.  A follow-up
(WP-B7) will edit `lenses_maslov.py` and `asymptotic*.py` AFTER you finish; you own them until then.

You did not write WP-B1.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B1's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B1_MASLOV_S6_INPUT_WAVEVECTOR.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B1_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b1_maslov_input_wavevector.py`, and `fixes/WP-A4_REPORT.md` section 6 item 1 (the design; WP-B1 deviated from it in two
stated places -- judge whether those deviations are right).

## What to do, in order

1. **Re-derive the headline numbers on fixtures the engineer did not use.**  Build your OWN oracle: an exact trace of the input's own rays (launch
   direction = the input's local wavevector) plus a direct Rayleigh-Sommerfeld or Kirchhoff sum from the exit surface (the method of the d6 file's
   inline oracle and `repro/orch/p7_rs.py`; write your own code).  Use a different singlet (different f, aperture, glass), a different wavelength,
   and inputs the report did not: an off-axis converging input, a tilt at 2x the lens NA, an astigmatic input.  Report field fidelity, focal
   centroid and EE against the oracle for `stationary_phase` and `local_quadrature`, before (`_S6_INPUT_WAVEVECTOR_SADDLE = False`) and after.
2. **Attack the two deviations.**  (a) WP-B1 says adding the input phase to `opd_star` / `opd_v` as the WP-A4 design proposed would double-count
   because both integrators sample the complex `E_in` at the saddle.  Check that with the leading-order formula on your fixture: does the shipped
   form reproduce the oracle's PHASE as well as its intensity?  (b) the separate `_solve_fit` for `k1`: confirm the `'quadrature'` and `'levin'`
   paths are bit-identical to the parent on a non-collimated input (the report says a wider RHS would have moved them).
3. **Attack the thresholds.**  `_SADDLE_FLAT_INPUT_NA = 1e-3` (engagement, absolute) and `_K1_FIT_RESIDUAL_MAX = 0.5` (fallback).  Measure the
   field error the un-engaged saddle leaves just BELOW the engagement bar on an f = 100 mm and an f = 1 m system (the report admits the bar is
   absolute, not chart-relative); measure the fallback ladder on your own speckle / hard-edge inputs and check the 0.5 bar sits between "helps"
   and "hurts" there too.  Check the aliasing blind spot the report names (a tilt past the grid's Nyquist angle reads as smooth): what does the
   user get, and does anything say so?
4. **Attack the byte-identity claims** (24 rows): repeat a representative subset against `git archive {HEAD}^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b1\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest), including `collimated_input=True` on a
   non-collimated input and a numerically-real input with float dirt (`E * (1 + 1e-17j)`).
5. **Attack the new pins** (25 tests): derived envelopes or per-build numbers?  Fail-before real on the parent?  Mutate the fix in memory (zero the
   `k1` fit; drop the Hessian term but keep the gradient; flip the sign of `k1`) and confirm the corresponding tests go red.  Check the CPU/xp
   parity pin with `xp = np`.
6. **Fix what you find, in the package's own files** (`lumenairy/elements/lenses_maslov.py`, and `lumenairy/propagators/asymptotic*.py` only if
   the fix must reach them); re-record the history documents you touch.  Anything outside goes under "Requested changes outside my ownership"
   with the exact edit (the JAX sibling `_lens_jax.apply_real_lens_maslov_jax` is WP-B7's, not yours -- but measure how wrong it is on your
   tilted input so B7 has a number).
7. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B1.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), the defects you found and fixed (with fail-before), a
   "Follow-up" section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...`
   release text to a new `VERIFY_WP-B1_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/elements/lenses_maslov.py`, `lumenairy/propagators/asymptotic*.py`, their `docs/history/` documents,
  `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` (add, do not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
