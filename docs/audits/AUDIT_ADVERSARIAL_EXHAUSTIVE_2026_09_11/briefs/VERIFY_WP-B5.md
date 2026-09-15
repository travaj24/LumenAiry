# VERIFY-WP-B5 -- independent adversarial re-verification of WP-B5 (RCWA / EME / BOR: Toeplitz solves D1, the two-interface closed form D2, off-plane `fff_nv` symmetrisation D3)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B5 is commit 4ea16066 (its parent
is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers may be editing other subsystems concurrently; never touch
those, and expect their fingerprint checks to be red until they re-record.  You own `lumenairy/elements/rcwa/*`, `eme/*`, `bor/*` for this pass;
`pmm/*` and `berreman.py` are NOT yours (use them as oracles, never edit them).

You did not write WP-B5.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B5's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B5_RCWA_EME_BOR_PERFORMANCE.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B5_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b5_rcwa_eme_bor.py`, and `fixes/WP-A14_REPORT.md` sections 2 and 6 (H2/H3/H4/M1 and the D1-D3 designs) + `VERIFY_WP-A14.md`.

## What to do, in order

1. **Attack D1 (Toeplitz solves).**  `scipy.linalg.solve_toeplitz` IS a Levinson-Durbin recursion -- the weakly stable algorithm WP-A14 refused
   for the explicit inverse.  On YOUR OWN metallic ladder (a different metal -- Al or Cu -- a different period / wavelength / fill factor, TM and TE,
   `n_orders` 50..400) compare each replaced site's product against a pivoted-LU `solve` in float64 AND in extended precision (mpmath or float128
   where available) on the SAME matrix, at the M1 census's `cond ~ 1e13` regime (construct it: a thin high-contrast metallic stripe).  Is the
   documented tolerance honest at that conditioning?  Does the ladder converge to the same efficiency to the census's residual floor?  If the
   engineer left a site alone "because the tolerance would exceed the census", check that claim with numbers too.
2. **Attack D2 (two-interface closed form).**  (a) A fixture that trips `_guarded_inverse` on the parent must trip it identically (same census
   entry, same refusal text) at 4ea16066.  (b) Thick layers with strongly evanescent orders: the Redheffer star is stable because it never forms
   growing exponentials; show whether the closed form keeps that property (a 1-D metallic grating 5-20 wavelengths thick, `n_orders` 200) or
   overflows / loses digits, both formulations, both polarisations.  (c) Thickness -> 0 must reduce to a single interface; thickness -> the star
   product's own fast path must agree to the derived tolerance.  (d) Confirm the inverse count (patch `_guarded_inverse` in memory and count).
3. **Attack D3 (off-plane `fff_nv` symmetrisation).**  Your OWN conical Berreman oracle (`lumenairy/elements/berreman.py`, called not edited) on a
   rotated-director uniaxial cell at angles and azimuths the report did not use, a biaxial cell, and a cell with `ezz` != `exx`; the x <-> y
   reflection-symmetry property (swap the cell's axes, swap the result); prove the `l3-` Schur fold is applied AFTER the mean (a hand-built 2x2-block
   example where mean-then-Schur and Schur-then-mean differ); in-plane path byte-identical to the parent.
4. **Attack the byte-identity claims** against `git archive 4ea16066^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b5\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest), including EME and BOR entry points the
   package says it did not touch, and `tests/unit/test_ci_kernel_consistency.py` (no decision may move).
5. **Attack the pins.**  Derived envelopes or per-build numbers?  Fail-before real on the parent?  Mutate in memory (serve `inv(T) @ X` again; drop
   the guard on the remaining inverse; apply the Schur fold before the mean) and confirm the corresponding tests go red.
6. **Fix what you find, in the three packages' own files**; re-record every history document you touch.  Anything outside goes under "Requested
   changes outside my ownership" with the exact edit.  Do not change any default.
7. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B5.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to a
   new `VERIFY_WP-B5_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/elements/rcwa/*.py`, `lumenairy/elements/eme/*.py`, `lumenairy/elements/bor/*.py`, their `docs/history/` documents,
  `tests/unit/test_audit2609_b5_rcwa_eme_bor.py` (add, do not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
