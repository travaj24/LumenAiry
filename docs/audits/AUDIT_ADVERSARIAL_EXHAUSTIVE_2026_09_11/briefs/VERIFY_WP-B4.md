# VERIFY-WP-B4 -- independent adversarial re-verification of WP-B4 (carrier chain: Collins / ABCD-Fresnel transport with a Bluestein output pitch, `transport='collins'`)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B4 is commit 185d64cd (its parent
is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers may be editing other subsystems concurrently (rcwa/eme/bor,
analysis, sources, raytrace, `_lens_traced`, `lenses_maslov`, `system.py`, the propagator kernels); never touch those, and expect their fingerprint checks
to be red until they re-record.  You own `lumenairy/propagators/carrier.py` and `carrier_field.py` for this pass.

You did not write WP-B4.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B4's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B4_COLLINS_BLUESTEIN_TRANSPORT.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B4_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b4_collins_transport.py`, and `fixes/WP-A6_REPORT.md` section 6.1 (the design and the four-part acceptance gate) plus
`WP-A25_REPORT.md` (the replica regime the new transport is supposed to make unnecessary).

## What to do, in order

1. **Re-derive the four-part gate on fixtures the engineer did not use.**  Your OWN closed-form Gaussian-ABCD oracle (q-parameter transport through
   the same ABCD matrix; state its floor) at NAs and grid extents between and beyond the report's cells, on a CONVERGING leg (B < 0 in the report's
   stated convention -- check the convention is the one CONVENTIONS sec. 7 implies and that a diverging leg gives the mirror result), on a leg that
   lands EXACTLY at the paraxial focus (B -> 0: what does `'collins'` do at the imaging singularity, and does anything say so?), and on an
   output pitch coarser than the field's Nyquist pitch.  Rebuild the two-group chain oracle by the METHOD of `repro/CARRIER/p5_chain.py`
   (brute-force ASM + `apply_real_lens_traced`; write your own code) on a different pair of lenses and compare `'collins'` with `'sziklas'`.
2. **Attack the sampling guard.**  It is claimed derived from Kelly (2014).  Re-derive the bound yourself for the chirp-Z stage; construct an
   input just below and just above it and show the guard refuses (sec. 2 prefix) exactly where the answer actually degrades -- not at a geometric
   margin.  Check both signs of B, the `_multi` K = 2 path, and a decentred (WP-A24 style) carrier.
3. **Attack the byte-identity of the default.**  `transport='sziklas'` on the WP-A6 / A24 / A25 fixtures and the P2 design battery against
   `git archive 185d64cd^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b4\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest).  Include the near-focus bridge fixtures
   (`_near_focus_needs_bridge` true) and `replica_fill='zero'`.
4. **Attack the cost claim** by counting FFT calls (patch `numpy.fft` / the xp fft in memory) for N, N_out pairs, not by timing; confirm
   `next_fast_len(N + N_out - 1)` and 3 FFTs per axis-pass.  Report wall-clock medians of interleaved runs as a measurement.
5. **Attack the pins.**  Derived two-sided envelopes or per-build numbers?  Fail-before real on the parent?  Mutate in memory (drop the second
   separable screen; flip the sign of the D-term; skip the `i/(lambda B)` prefactor; use `m dx` instead of the chosen pitch) and confirm the
   corresponding tests go red.  Energy: Parseval on the chosen output lattice against the input power, with the floor stated.
6. **Fix what you find, in `carrier.py` / `carrier_field.py`**; re-record `docs/history/carrier.md` (and `carrier_field`'s document if it has one).
   Anything outside goes under "Requested changes outside my ownership" with the exact edit.  Do not change the default transport.
7. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B4.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section (including your view of what a later default flip to `'collins'` would retire and what measurements would justify it), and the exact
   commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to a new
   `VERIFY_WP-B4_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/propagators/carrier.py`, `lumenairy/propagators/carrier_field.py`, their `docs/history/` documents,
  `tests/unit/test_audit2609_b4_collins_transport.py` (add, do not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
