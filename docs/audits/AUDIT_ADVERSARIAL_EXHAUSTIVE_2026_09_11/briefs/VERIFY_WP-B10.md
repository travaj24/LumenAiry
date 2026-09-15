# VERIFY-WP-B10 -- independent adversarial re-verification of WP-B10 (traced lens: the disc-orthogonal Zernike fit basis, `fit_basis='zernike'`, opt-in)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B10 is commit f8d5466f (its parent
is the pre-change library for byte-identity and fail-before probes; the commit also carries the orchestrator's `lens_config.py` / census edits for the
new `LensNumerics` field).  Other Wave-4 engineers may be editing other subsystems concurrently (`_lens_real.py` has a verifier and a follow-up in it);
never touch those, and expect their fingerprint checks to be red until they re-record.  You own `lumenairy/elements/_lens_traced.py` (and
`_lens_imap.py` only if a fix must reach it) for this pass.

You did not write WP-B10.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B10's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B10_ZERNIKE_TRACED_FIT_BASIS.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B10_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b10_zernike_fit_basis.py`, `fixes/WP-A26_REPORT.md` sections 2.5-2.6, 3 and 8 (the skirt weights and the three refused
cures) and `WP-A3_REPORT.md` (the T-findings on the fits).

## What to do, in order

1. **Attack the "same polynomial space" claim.**  On YOUR OWN launch lattices (a square lattice clipped to the disc, a disc-uniform lattice, a
   decentred disc, D1's ghost geometry) at orders 6..20: rank of both design matrices, the projector difference `||P_cheb - P_zern||` (must be
   roundoff if the spaces coincide), and the Gram `rcond` of each WITH the skirt weights and outside-disc points included -- Zernike polynomials are
   orthogonal on the disc only, so measure how the conditioning degrades on the skirt versus Chebyshev.
2. **Re-derive the ladder on fixtures the engineer did not use.**  Rebuild the d7 analytic Fermat oracle by its METHOD on a different singlet
   (f, glass, aperture), decentres 0.25 / 0.75 / 1.25 beam radii, and an independent exact conic trace as a second oracle; exit-slope error, arbiter
   engagements and step-down firings for both bases, orders 6..20.  Is the Zernike basis EVER worse than Chebyshev on your fixtures?  Does the claim
   "marginal rays no longer couple into defocus" hold -- measure the defocus coefficient's sensitivity to the marginal-ray weight on both bases.
3. **Attack the byte-identity of the default.**  `fit_basis='chebyshev'` and the omitted keyword on the d7 / c11 / c12 / c13 / c6 / d1 / a26 / a3
   fixtures against `git archive f8d5466f^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b10\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest); the d7 numbers 2.371 / 1.683 urad
   character for character.
4. **Attack the configuration seam.**  `fit_basis='foo'` refuses with the sec. 2 prefix; `LensNumerics(fit_basis=...)` round-trips and the
   precedence-or-raise contract holds when both the object and the keyword are passed; the a16 census (`test_audit2609_a16_lens_config_round_trip.py`,
   `_NUMERICS_FOR`) counts the field; the `_lens_imap` inverse-characteristic evaluator (the d6 crossover) either honours the basis or says it does
   not; the JAX traced twin, if one exists, either honours it or refuses it explicitly.
5. **Attack the pins.**  Derived envelopes or per-build numbers?  Is the stated fail-before (the Chebyshev coupling of marginal rays into defocus)
   actually measured on the parent?  Mutate in memory (evaluate the Zernike basis on unscaled coordinates; drop the skirt weight; permute the basis
   order) and confirm the corresponding tests go red.  Cost: interleaved medians at the same order, reported.
6. **Fix what you find, in `_lens_traced.py`**; re-record `docs/history/lumenairy.elements._lens_traced.md`.  Anything outside goes under
   "Requested changes outside my ownership" with the exact edit.  Do not change the default basis.
7. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B10.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section (your recommendation on whether the disc basis could ever replace the arbiter / predictor apparatus, with the measurements that would
   justify it), and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text
   to a new `VERIFY_WP-B10_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/elements/_lens_traced.py` (+ `_lens_imap.py` if unavoidable), their `docs/history/` documents,
  `tests/unit/test_audit2609_b10_zernike_fit_basis.py` (add, do not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
