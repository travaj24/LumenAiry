# VERIFY-WP-B8 -- independent adversarial re-verification of WP-B8 (analysis and sources: PSF memory + MFT sampling, encircled-energy profile, Zernike recurrence + DM IF cache, Gori pseudo-modes, two dtype/out= items)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B8 is commit ed40e169 (its parent
is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers may be editing other subsystems concurrently; never touch
those, and expect their fingerprint checks to be red until they re-record.  You own `lumenairy/analysis/*`, `lumenairy/sources/*`,
`lumenairy/elements/polarization.py` for this pass (`propagators/mft.py` is consumed, never edited).

You did not write WP-B8.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B8's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B8_ANALYSIS_SOURCES_PERFORMANCE.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B8_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b8_analysis_sources.py`, and `fixes/WP-A7_REPORT.md` + `WP-A11_REPORT.md` section 6 (the designs).

## What to do, in order

1. **Attack the PSF items.**  (a) The chessboard identity: even N exact -- prove `np.array_equal` on random complex128 AND complex64 pupils, on
   non-square grids with one odd and one even side, and confirm what the code does for odd N (explicit shifts, or a derived tolerance you can
   reproduce).  Measure peak memory with `tracemalloc` at oversample 4 on the parent and at ed40e169 (a measurement, not an assertion).
   (b) `method='mft'`: at the FFT's OWN pitch and centring, the Soummer MFT is the DFT up to roundoff -- compare against the padded-FFT PSF at
   a derived tolerance; against YOUR analytic Airy (circular pupil, exact Bessel form) and Gaussian PSF at an arbitrary pitch; on an odd-N pupil,
   a decentred pupil and an annular pupil.  Confirm the FFT default's grid contract is byte-identical to the parent.
2. **Attack the encircled-energy profile.**  "The radius is the exact inverse of the curve": test fractions 0 and 1, plateaus (ties in the sorted
   radii, a PSF with an exact zero ring), `dy != dx`, a supplied `centroid`, and a `profile=` that does not belong to `E` (does anything validate
   it, and should it?).  Byte-identity of both functions without `profile=`.
3. **Attack the Zernike recurrence and the DM IF cache.**  The recurrence against YOUR factorial sum in extended precision (mpmath) at every (n, m)
   of the shipped tables and past them to the stated stability limit, at r = 0, r = 1, r = 1 - 1e-12 and r = 1e-12; confirm the derived tolerance
   and that the limit is where it is claimed.  `_banded_IF_apply` versus the dense build bit-identically; the `cache_basis` byte budget's derivation;
   the warning fires exactly once with the caller's frame.
4. **Attack the Gori pseudo-modes.**  Your own chi-square on the marginal at several M; the measured correlation function against the target for
   BOTH generators at s/L in {0.02, 0.1, 0.5}; the small-M exactness of the `zgemm` factorisation against a direct sum; seed reproducibility;
   `generator='fft'` byte-identical to the parent; the M heuristic's derivation.
5. **Attack the two bit-identity-sensitive items.**  `geometry_dtype=` float32: the documented tolerance on beams at the extremes of the grid; the
   default byte-identical.  `apply_jones_matrix` with `out=`: repeat the A11 bit-identity matrix on random complex64 and complex128 Jones fields,
   NON-contiguous and Fortran-ordered inputs, aliased `out`, and the JAX/xp path if one exists -- the complex-multiply operand order must be preserved
   exactly.  If the engineer left the item alone, check the reason with numbers.
6. **Attack the byte-identity claims** against `git archive ed40e169^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b8\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest), and **the pins** (derived envelopes or
   per-build numbers? fail-before real? mutate in memory -- drop the chessboard on one axis, shift the MFT centre by half a pixel, break the
   recurrence's seed term -- and confirm the tests go red).
7. **Fix what you find, in the owned files**; re-record every history document you touch.  Anything outside goes under "Requested changes outside
   my ownership" with the exact edit.  Do not change any default.
8. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B8.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to a
   new `VERIFY_WP-B8_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/analysis/*.py`, `lumenairy/sources/*.py`, `lumenairy/elements/polarization.py`, their `docs/history/` documents,
  `tests/unit/test_audit2609_b8_analysis_sources.py` (add, do not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
