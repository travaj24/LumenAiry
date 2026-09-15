# VERIFY-WP-B3 -- independent adversarial re-verification of WP-B3 (propagator kernels: HFPI output plane + `normalisation='auto'`, Sobol sampler, Shen-Wang pixel-integrated RS kernel, chirp-Z resampler)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B3 is the
newest commit on the branch (`git log -1` shows it; call it {HEAD}).  Three other Wave-4 engineers are editing `lumenairy/elements/lenses_maslov.py`
+ `lumenairy/propagators/asymptotic*.py`, `lumenairy/elements/_lens_real.py` + `lens_config.py`, and `lumenairy/propagators/carrier.py` +
`carrier_field.py` concurrently; never touch those, and expect their history-fingerprint checks to be red until they re-record (not your concern).

You did not write WP-B3.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B3's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B3_PROPAGATOR_KERNELS.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B3_REPORT.md`, its changelog `WP-B3_CHANGELOG.md`, and its test
file `tests/unit/test_audit2609_b3_propagator_kernels.py`.  Also `fixes/WP-A5_REPORT.md` section 6 (the designs it implemented) and
`fixes/VERIFY_WP-A5.md` (the verifier's independent RS-I double quadrature and the HFPI transparency property, both reusable METHODS).

## What to do, in order

1. **Re-derive every number in the report's summary table on a fixture the engineer did not use.**  In particular:
   * K13: the report's photometric scale vs band-limited ASM (0.9906 at the native pitch) was measured on ONE flat-optics geometry.  Take a
     different aperture, wavelength, distance and cone; confirm the scale is flat in output pixel area and path count, and confirm the
     `'auto'` refusal through a powered element is right by measuring the factor yourself (the report says 4879x through a thin singlet).
   * K22: re-measure the convergence order of both samplers with your own reference and your own seeds; the claim is `p ~ 0.54-0.56` for
     both and a 1.00-1.13x constant.  Check the Owen-scramble seeding really makes the bundle a pure function of `rng`.
   * K9: build your OWN pixel-integrated reference (a different quadrature from both the module's Gauss-Legendre and the report's
     super-sampled midpoint rule -- e.g. an adaptive or Romberg rule over one pixel, or the exact analytic pixel integral in the far
     field) and score both kernels; confirm the "reverses by 4-5 decades on a smooth field" finding on a fixture of your choosing; check
     the `'RS_INT'` cache tag cannot be confused with `'spatial'`'s; check the six-node rule at the worst legal geometry.
   * K6: confirm the chirp-Z leg's unit MTF and its 2x2 tiling in the converging direction with your own carrier fixture; check the odd-N
     half-pixel origin handling against the Dirichlet interpolant on an ODD grid with a non-square extent.
2. **Attack the byte-identity claims** (52 comparisons in the report): repeat a representative subset against `git archive {HEAD}^ lumenairy`
   extracted READ-ONLY into `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b3\`,
   importing the archived library from a child process whose cwd and PYTHONPATH are the archive (never through pytest, which puts the
   repository root first; assert `lumenairy.__file__` per probe).  Include `propagate_hfpi_through_prescription` with the OLD default
   (`normalisation='legacy'` explicitly) against the new default (`'auto'` -> `'legacy'` when no `z_output`): identical bits and the identical
   warning?
3. **Attack the new pins.**  For each of the 33 tests: derived envelope or per-build number?  Does the fail-before really fail on the
   pre-change library?  Mutate the fix in memory (e.g. drop the `r` from the closing leg, unscramble the Sobol draw, sample the RS kernel at
   the pixel centre inside the "integrated" branch, skip the odd-N origin shift) and confirm the corresponding test goes red.
4. **Attack the edges.**  `z_output` behind a mirror (the report says the plane must sit on the reflected side -- try the wrong side and see
   what the user gets); an immersed image space; `sampler='sobol'` with a non-power-of-two `n_paths` (warning only?  should it round?);
   `kernel='spatial-integrated'` at `z` just under the alias threshold, on complex64, on an anamorphic grid; `resample_field(method='chirpz')`
   with `dx_out < dx_in` on an odd grid; `propagate(method='hfpi', ...)` reaching the new keywords through the dispatcher.
5. **Fix what you find, in the package's own files** (`lumenairy/propagators/hf.py`, `hfpi.py`, `rs.py`, `mft.py`), with the same
   standards; re-record the history documents you touch (`python scripts/record_history_fingerprints.py <module> --reason "..."`).  Anything
   outside those files goes under "Requested changes outside my ownership" with the exact edit.  WP-B3's own section 5 (the `system.py` and
   `_lens_real.py` call-site switches) is NOT yours to apply; the orchestrator is routing it after WP-B2 lands -- but do say whether its
   measurements and its gating rule (`'chirpz'` only when the pitch coarsens) hold on your fixtures.
6. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B3.md`: a verdict table (each report claim ->
   VERIFIED / VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), the defects you found and fixed (with
   fail-before), a "Follow-up" section for anything ruled open, and the exact commands + counts + durations of every test run.  If you
   changed library code, add `### Fixed -- ...` release text to a new `VERIFY_WP-B3_CHANGELOG.md` in the style of `fixes/VERIFY_WP-A5.md`'s
   changelog entries.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared with three
  engineers).
* NO git write commands of any kind (no add/commit/stash/checkout/restore/reset); read-only `git show` / `git archive` / `git log` are
  fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/propagators/hf.py`, `hfpi.py`, `rs.py`, `mft.py`, their `docs/history/` documents,
  `tests/unit/test_audit2609_b3_propagator_kernels.py` (you may add tests; do not weaken existing ones without a measured reason), your two
  report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
