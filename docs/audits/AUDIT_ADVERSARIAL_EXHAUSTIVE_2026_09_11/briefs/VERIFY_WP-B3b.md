# VERIFY-WP-B3b -- independent adversarial re-verification of WP-B3b (the K6 call sites: the chain's Fresnel leg onto the chain grid; the SAS and in-glass resample-back legs gated on window against period)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B3b is commit 908c02d6 (its
parent 908c02d6^ is the pre-change library for byte-identity and fail-before probes).  Other engineers and verifiers are concurrently editing
`analysis/*`, `sources/*`, `lenses_maslov.py`, `carrier.py`, `raytrace/*`, `rcwa/*`, `_lens_traced.py` and the REMAP functions of `_lens_real.py`
(VERIFY-B2); never touch those, and expect their fingerprint checks to be red until they re-record.  You own `lumenairy/propagators/system.py`, the
two `resample_field` calls in `lumenairy/elements/_lens_real.py::_propagate_through_glass` (nothing else in that file), and `resample_field`'s
docstring in `lumenairy/propagators/mft.py`.

You did not write WP-B3b.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B3b's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B3b_RESAMPLE_CALL_SITES.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B3b_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b3b_resample_call_sites.py`, and `fixes/WP-B3_REPORT.md` sec. 5 + `fixes/VERIFY_WP-B3.md` sec. 5 (the designs and the
gate correction).  This package MOVES A DEFAULT (the Fresnel chain leg's numbers), so the correctness of the new numbers is the whole question.

## What to do, in order

1. **Re-derive the Fresnel leg against your OWN oracle** -- the Fresnel integral as an explicit double sum written by you (no FFT, no library call),
   on fixtures the engineer did not use: odd N, a non-square input with the LONGER side in y, a decentred hard-edged input, complex64, a z at the
   K1 chirp-sampling bound and just below it, and a three-element chain where the Fresnel leg sits between two lenses.  Report relL2 before
   (908c02d6^) and after, and the window power against the oracle.  Check the leg's `dy_in` / `dy_out` handling on anamorphic pitches (the engineer
   says the old leg scaled y by the x ratio).
2. **Attack the gate.**  Its form is `N_out dx_out <= min(N_in) dx_in * (1 + 1e-9)` per axis.  Construct: a window EXACTLY one period (both sides
   of the 1e-9 slack), a non-square input whose shorter side sets the period, a contained Gaussian just past one period (the engineer measured
   1.000594 at 0.6182), a grid-filling field just inside one period; for each report chirp-Z vs spline vs a direct evaluation, and whether the
   gate's choice was the better one.  Then the in-glass legs: a glass whose index makes `dx_new/dx` cross 1 inside a thickness sweep; confirm
   both directions are exercised and byte-identical to 908c02d6^ where the spline is selected.
3. **Attack the byte-identity claims** (33 arrays) against `git archive 908c02d6^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b3b\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest), including `method='asm'` chains with
   tilted elements, `propagate_through_system_jax`, `apply_real_lens(wave_propagator='rs'|'asm')`, and every guard text.
4. **Attack the pins** (39 tests): derived envelopes or per-build numbers?  Fail-before real?  Mutate in memory (drop the per-axis `min`; use
   `dx_new >= dx`; remove the 1e-9 slack; resample the Fresnel leg again) and confirm the corresponding tests go red.  Check the F6 docstring's
   claims (`N_out dx_out == N_in dx_in`; which scale factors the default satisfies) against your own measurement.
5. **Fix what you find, in the owned files**; re-record `docs/history/lumenairy.propagators.system.md` / `lumenairy.elements._lens_real.md` if you
   change the modules.  Anything outside goes under "Requested changes outside my ownership" with the exact edit.
6. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B3b.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to
   a new `VERIFY_WP-B3b_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.  Do not change any default.
* Own only: the files named above, their `docs/history/` documents, `tests/unit/test_audit2609_b3b_resample_call_sites.py` (add, do not weaken),
  your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
