# WP-B5 (Wave 4) -- RCWA / EME / BOR: the H4 performance items WP-A14 deferred (Toeplitz solves, the two-interface closed form) and the off-plane `fff_nv` symmetrisation (H3)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD is the WP-B3 commit
284daccc or later).  Other Wave-4 engineers are concurrently editing `lumenairy/elements/lenses_maslov.py` + `lumenairy/propagators/asymptotic*.py`,
`lumenairy/elements/_lens_real.py` + `lens_config.py`, `lumenairy/propagators/carrier.py` + `carrier_field.py`, and (soon) `lumenairy/elements/pmm/*`;
never touch any of those.  You own the RCWA, EME and BOR packages only.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 (error prefix) and sec. 7, the comment rule in `CONTRIBUTING.md`
("Modules with a history document": `lumenairy/elements/rcwa/_core.py`, `rcwa/oned.py`, `rcwa/twod.py`, `rcwa/stack.py`, `eme/*.py`, `bor/*.py`
have `docs/history/<dotted.module>.md`; a code change MUST re-record with `python scripts/record_history_fingerprints.py <module path> --reason "..."`
in the same change).  Source comments describe what the code does NOW and why; no version narrative (the history-lint ratchet fails on it).

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A14_REPORT.md` (section 2 for H2/H3/H4/M1 -- what shipped and the
conditioning census `_guarded_inverse` carries; section 6 items D1, D2, D3 -- the designs you are implementing, reproduced below),
`WP-A14_CHANGELOG.md`, `VERIFY_WP-A14.md`, and the audit's RCWA-EME-BOR partition report.

## The items

**D1 -- H4, the two genuinely Toeplitz inverses** (`rcwa/oned.py`: `inv(Toeplitz(1/eps))` in `_binary_grating_convolutions`, and `inv(EPS)` for
the P block; ~8 % of a 1-D TM solve at `n_orders = 200`).  WP-A14 refused classical Levinson because it is only weakly stable and the matrix is a
general complex Toeplitz that the M1 census found at `cond ~ 1e13`.  Take the cheap 80 % it named: wherever the inverse is immediately MULTIPLIED,
replace `inv(T) @ X` by `scipy.linalg.solve_toeplitz` (or a `solve`) so no explicit inverse is formed, and MEASURE: bit-identical or a documented
tolerance against the metallic convergence ladder (Ag / Au TM, `n_orders` 50..400), plus the wall-clock share recovered.  If the tolerance needed
would exceed what the H2 / M1 census admits, say so with numbers and leave the site alone.  Do not implement Gohberg-Semencul unless you can also
supply the two-sided stability census the report asks for.

**D2 -- H4, the direct two-interface closed form for a single-layer 1-D stack.**  The `interface -> propagation -> interface` chain ends in a
`_redheffer_star` whose zero-block fast path cannot fire, so it pays two `_guarded_inverse` calls (~8 % at N = 401).  Design: the algebraic
closed form for `S_a * P * S_b` with the single inverse `(I - B11 A22)^-1` (Moharam 1995 enhanced transmittance is the alternative).  Hard
constraint: `_guarded_inverse`'s conditioning census, residual probe and refusal path must be preserved on the default path -- the one inverse
that remains goes through the same guard, and a fixture that trips the guard today must trip it identically after.  Gate: bit-tolerance sweep
over the metallic ladder, both formulations, both polarisations, with the tolerance derived from the census's own residual floor.

**D3 -- H3, symmetrise the OFF-PLANE (full 3x3) `fff_nv` operator.**  `_li_convolutions_2d_tensor_full` still uses the fixed `L2 L1` order; the
in-plane fix (H3, shipped) left it as it was.  Design: the same transpose argument, but the 3x3 transpose permutes `(x, y, z) -> (y, x, z)` and
swaps `exz <-> eyz`, `ezx <-> ezy`, leaving `ezz` alone; the `l3-` Schur fold is applied AFTER the mean, not before (the mean of two Schur
complements is not the Schur complement of the mean).  Oracle: a conical Berreman solve (`lumenairy/elements/berreman.py`, which you do not
change) on a ROTATED-director uniaxial cell, plus the reflection-symmetry property (swap x and y in the cell and the result must swap).  Report the
before/after residuals and the in-plane path's byte-identity.

Everything whose default did not move must be proved byte-identical against the pre-change modules (`git archive <HEAD>^ lumenairy` extracted
READ-ONLY into your scratch directory under `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b5\`,
imported from a child process whose cwd and PYTHONPATH are the archive -- never through pytest -- with `lumenairy.__file__` asserted).

## Deliverable

Implementation with the sec. 2 prefix on every new validation; oracles the library did not produce; derived two-sided envelopes in a new
`tests/unit/test_audit2609_b5_rcwa_eme_bor.py` (S5; no wall-clock assertions -- operation counts, inverse counts or convergence orders instead);
stated fail-befores; byte-identity proofs; wall-clock medians of interleaved runs reported, not asserted.  Re-record every history document you
touch.  Report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B5_REPORT.md` (summary table: item / status / files:lines / tests /
oracle / measured before -> after; per-item sections; files touched; tests run with counts and durations; requested changes outside your
ownership; deferred) and `WP-B5_CHANGELOG.md` (5.47.0 release text in the `### Performance -- ...` / `### Fixed -- ...` style of
`fixes/WP-A14_CHANGELOG.md`, findings H3/H4 named, `lumenairy/elements/rcwa/<file>.py:N` citations on non-trivial lines).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a14_rcwa_eme_bor.py` and the VERIFY-A14 file(s) (`a14` in `tests/unit/`), `pytest tests/unit -k "rcwa or eme or bor"`
(slow: budget ~15 min), `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py` (reads the RCWA fff_nv operators too), `python validation/run_all.py`
for the rcwa / eme / bor files (`ls validation/`), `ruff check`, `python scripts/record_history_fingerprints.py --check` (your documents; others'
may be red while their owners work), `tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/elements/rcwa/*.py`, `lumenairy/elements/eme/*.py`, `lumenairy/elements/bor/*.py`, their `docs/history/` documents, the new
  b5 test file, your two report files.  Anything else (including `pmm/`, `berreman.py`, `_branchcut` consolidation D4/D5 -- a later pass owns those):
  "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
