# VERIFY-WP-B6 -- independent adversarial re-verification of WP-B6 (PMM: the Gegenbauer-basis negative result, and the PMM-2D k0-free tensor operator cache)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B6 is commit {HEAD} (its parent
is the pre-change library for byte-identity and fail-before probes).  Other Wave-4 engineers are editing the carrier, rcwa/eme/bor, analysis, sources,
raytrace, `_lens_traced`, `lenses_maslov` and propagator-kernel files concurrently; never touch those, and expect their fingerprint checks to be red
until they re-record.  You own `lumenairy/elements/pmm/*` for this pass.

You did not write WP-B6.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B6's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B6_PMM_GEGENBAUER_TENSOR_CACHE.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B6_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py`, and `fixes/WP-A12_REPORT.md` section 6 item 3 + `WP-A13_REPORT.md` section 6.1 (the
designs), plus the audit's PMM-1D partition report ("Alternative algorithms" (c), which WP-B6 says cannot work as stated).

## What to do, in order

1. **Attack the negative result (item 1).**  WP-B6 claims a nodal-basis change with exact integration cannot move a Galerkin answer on the fixed
   C0 piecewise-P_N space (six lambda agree to 3e-13), that what lambda varies is the quadrature, and that only GLL satisfies summation-by-parts
   (`M D + (M D)^T = diag(-1, 0..0, +1)`, residual 1e-14 vs 0.4-1.5 for other lambda), so the LUMPED lambda = 0 "TM win" is error cancellation
   that breaks the lossless energy identity (closure 3e-14 -> 1e-8..1e-5) and regresses TE 40-110x.  Re-derive this on your OWN fixture (a
   different metal, period, depth, angle; a dielectric cell): build the ultraspherical Gauss-Lobatto family yourself (Golub-Welsch on the
   symmetric Jacobi recurrence), the exact-integration control, and the extrapolated RCWA oracle (`rcwa_efficiency_1d` + `rcwa_extrapolate`,
   called not edited; state the oracle floor).  Confirm or refute: (a) invariance under exact integration, (b) the SBP residuals, (c) the TE
   regression and the closure loss for lambda != 1/2, (d) that the local convergence rate never leaves the algebraic regime.  If the audit's
   claim ("recovers exponential convergence") holds anywhere in the family on your fixture, say so with the table -- that would reopen item 1.
2. **Attack the tensor cache (item 2).**  Repeat the bit-identity comparison against `git archive {HEAD}^ lumenairy` extracted READ-ONLY into
   `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\verify_b6\`
   (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest) on cells the engineer did not use:
   a crossed cell with an anisotropic (3x3) tile at oblique incidence and non-zero phi, `fff_nv`, circular truncation, an out-of-plane tile, a
   slanted layer, `symmetry='auto'` on a symmetric cell, and a wavelength AND angle sweep through `PMM2DStackHybrid`.  Then attack the cache
   itself: (a) mutate `formulation` after a solve -- is the stale entry ever served?  (b) a stack whose two layers differ only in a tile value
   that the key should catch; (c) `cache_max_bytes` just below one entry's size -- rebuild, never a wrong answer; (d) the frozen cached arrays --
   attempt a write through every returned reference; (e) `_symmetric_layer_specs`' `tops=None` path -- can a tensor layer reach it?
3. **Attack the pins** (44 tests): derived envelopes or per-build numbers?  Do the identity tests (SBP, exact-integration invariance) re-derive
   on the running build or pin constants?  Mutate the code in memory (serve a cached entry with the wrong `kind`; skip the `formulation` key;
   reorder the k0-free split) and confirm the corresponding tests go red.
4. **Check the census and the known-red test.**  `tests/unit/test_ci_kernel_consistency.py` (no PMM decision may move) and
   `tests/unit/test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band` (red at the audit base on this
   box; its screened tuple must be identical character for character before and after -- reproduce that by direct import, not pytest).
5. **Fix what you find, in `lumenairy/elements/pmm/*.py`**; re-record the history documents you touch (`python scripts/record_history_fingerprints.py
   <module> --reason "..."`).  Anything outside goes under "Requested changes outside my ownership" with the exact edit.
6. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B6.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to
   a new `VERIFY_WP-B6_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.  Do not change any PMM default.
* Own only: `lumenairy/elements/pmm/*.py`, their `docs/history/` documents, `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` (add, do
  not weaken), your two report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
