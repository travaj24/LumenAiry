# VERIFY-WP-B2 -- independent adversarial re-verification of WP-B2 (the 2-D displaced remap: symmetric input window, structured inversion, lattice 181 -> 257, `displaced_n_side`)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B2 is commit 47bc7a79 (its
parent is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers are editing `lumenairy/elements/lenses_maslov.py`
+ `lumenairy/propagators/asymptotic*.py`, `lumenairy/propagators/carrier.py` + `carrier_field.py`, `lumenairy/elements/rcwa/*`, `eme/*`, `bor/*`,
`lumenairy/elements/pmm/*`, `lumenairy/analysis/*`, `lumenairy/sources/*`, `lumenairy/raytrace/*` concurrently; never touch those, and expect
their fingerprint checks to be red until they re-record.  A follow-up engineer (WP-B3b) will edit `_lens_real.py`'s `_propagate_gap` resample call
AFTER you finish; if you must change `_lens_real.py`, keep to the remap functions.

You did not write WP-B2.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B2's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B2_ANALYTIC_REMAP_STRUCTURED_INVERSION.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B2_REPORT.md`, its changelog `WP-B2_CHANGELOG.md`, its test file
`tests/unit/test_audit2609_b2_displaced_remap_inversion.py`, and `fixes/WP-A2_REPORT.md` section 2 (L9) + `VERIFY_WP-A2.md` (the reflection
instability WP-A2 attributed to Delaunay -- WP-B2 says that attribution was wrong; decide who is right with your own measurement).

## What to do, in order

1. **Re-derive the headline claim on fixtures the engineer did not use.**  WP-B2 says the mirror instability was a one-pixel input-window
   asymmetry (`map_coordinates(mode='constant')` on an axis that runs one sample further on -x), selected by whether a launch point falls in
   the band `(x[-1], x[-1] + dx]`.  Build your own decentred fixture (different N, dx, w0, decentre, a non-conic surface), sweep `n_side` over
   values the report did not use, and measure the +d / -d image-plane mirror residual before (parent commit, read-only archive) and after.
   Then check the 1-D symmetric remap the report says carries the same asymmetry (`_apply_displaced_remap`, deferred item 1) with a
   DECENTRED INPUT FIELD through a rotationally symmetric element: does it show the asymmetry?  Report the number even if you do not fix it.
2. **Re-derive the structured inversion's accuracy with your own oracle.**  WP-B2's ray-exact oracle re-traces individual rays with Newton
   on the true trace.  Build a DIFFERENT one -- e.g. the tilted-plate affine case at a different tilt/thickness/index, or a direct Kirchhoff/RS
   sum from the exit-vertex field on a small grid -- and score both backends (`interp_method` is private; call the internal functions) at
   two lattices.  Confirm the "no hull holes inside the illuminated pupil" and the transmitted-power claim (0.97720 -> 0.98658) on your fixture.
3. **Attack the derivation of 257.**  The report derives it as the largest lattice whose trace costs less than the interpolation on every grid
   measured.  Re-time that crossover on this box (interleaved medians) and check the observable-convergence table (centroid / RMS radius / EE80
   flat from 181 to 2049) on a fixture with real input structure (a sinusoidal ripple, an aperture with a hard edge), where the report says the
   lattice DOES matter -- does the restated warning name a `displaced_n_side` that actually clears the bar?
4. **Attack the byte-identity and validation claims.**  Every path the report says did not move (the 1-D remap's byte-identity pin, the
   non-displaced surface models, the traced family); `displaced_n_side` validation (bool, string, tuple, float with fraction, below the
   3-ray floor, a call that would discard it -- each must raise with the sec. 2 prefix); the `LensNumerics` round trip (`from_kwargs` /
   `to_kwargs`, `narrowed_to`); the covering array.
5. **Attack the new pins.**  For each of the 50 tests: derived envelope or per-build number?  Fail-before real on the parent library?  Mutate
   the fix in memory (restore the asymmetric window; stop the Newton at one sweep; drop the affine seed; put 181 back) and confirm the
   corresponding tests go red.  The inversion's residual bar `_DISP_REMAP_2D_INV_TOL_FRAC` is derived in the source -- check the derivation.
6. **Fix what you find, in the package's own files**, with the same standards; re-record `docs/history/lumenairy.elements._lens_real.md`
   in the same change (`python scripts/record_history_fingerprints.py lumenairy/elements/_lens_real.py --reason "..."`).  Anything outside
   goes under "Requested changes outside my ownership" with the exact edit.
7. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B2.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), the defects you found and fixed (with fail-before), a
   "Follow-up" section for anything ruled open, and the exact commands + counts + durations of every test run.  If you changed library code,
   add `### Fixed -- ...` release text to a new `VERIFY_WP-B2_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine (archive the parent into
  `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b2\`
  and import it from a child process whose cwd + PYTHONPATH are the archive, `lumenairy.__file__` asserted; never through pytest).  Do not kill
  processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/elements/_lens_real.py` (remap functions), `lumenairy/elements/lens_config.py`, their `docs/history/` documents,
  `tests/unit/test_audit2609_b2_displaced_remap_inversion.py` (add, do not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
