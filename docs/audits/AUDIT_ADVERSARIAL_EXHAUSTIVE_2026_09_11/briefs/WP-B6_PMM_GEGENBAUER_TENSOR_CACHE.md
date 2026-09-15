# WP-B6 (Wave 4) -- PMM: the Gegenbauer / ultraspherical basis for the TM wall corner (opt-in), and the PMM-2D k0-free tensor operator cache

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD is the WP-B3 commit
284daccc or later).  Other Wave-4 engineers are concurrently editing `lumenairy/elements/lenses_maslov.py` + `lumenairy/propagators/asymptotic*.py`,
`lumenairy/elements/_lens_real.py` + `lens_config.py`, `lumenairy/propagators/carrier.py` + `carrier_field.py`, and `lumenairy/elements/rcwa/*`,
`eme/*`, `bor/*` (WP-B5); never touch any of those.  You own `lumenairy/elements/pmm/*` only.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document": `lumenairy/elements/pmm/_core.py`, `oned.py`, `twod.py`, `stack.py`, `stack2d.py`, `stack2d_pure.py`, `conical.py` have
`docs/history/lumenairy.elements.pmm.<name>.md`; a code change MUST re-record with `python scripts/record_history_fingerprints.py <module path>
--reason "..."` in the same change).  Source comments describe what the code does NOW and why; no version narrative.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A12_REPORT.md` (section 2 G1-G4 and section 6 item 3 -- the Gegenbauer
design, reproduced below), `WP-A13_REPORT.md` (section 2 G10 and section 6.1 -- the tensor operator cache design, reproduced below; section 2.6
on why `cascade='fused'` was NOT adopted, a ruling you do not reopen), `VERIFY_WP-A12.md`, `VERIFY_WP-A13.md`, and the audit's PMM-1D and PMM-2D
partition reports.  The user's project memory records that the PMM-2D default formulation is kept with `'auto'` -- do not change defaults.

## Item 1 -- the Gegenbauer / ultraspherical basis (WP-A12 section 6 item 3), OPT-IN

The audit's alternative (c) and, in WP-A12's words, "the highest-value algorithmic move in this partition".  `_gll_nodes_weights(degree)` and
`_lagrange_derivative_matrix(nodes)` are the only two places the basis enters.  An ultraspherical Gauss-Lobatto rule with parameter lambda
(lambda = 1/2 recovers today's Legendre/GLL exactly) gives a one-parameter family.  The element mass is diagonal only for lambda = 1/2, so
`_build_sem*` must stop assuming a lumped mass: `_real_diagonal` already returns `None` for a non-diagonal `S0` and the G3 fast paths degrade
correctly; check that nothing else in the assembly hard-codes diagonality.  Ship it as `basis=` (or the vocabulary the module suggests) with the
default being today's basis, BYTE-IDENTICAL (prove it).  GATE: the auditor's Au/air TM fixture with the extrapolated RCWA oracle -- the option is
worth exposing only if the measured local convergence rate leaves the `O(N^-2.7)` regime; report the rate ladder for lambda in a small set
(e.g. 0.5, 0.75, 1.0, 1.5) and say which, if any, is worth recommending.  If none beats Legendre on the fixture, ship nothing for item 1 and
say so with the table -- a measured negative is a deliverable.

## Item 2 -- the k0-free projected TENSOR operators cache (WP-A13 section 6.1)

`PMM2DStackHybrid._build_layer_modes` passes `lops=None` for tensor layers, so `_geom_cache` stores `(ax, ay, None)` and `_tensor_layer_modes`
rebuilds the per-axis projections and the `_proj` sandwiches at every wavelength and angle, where the scalar branch caches them.  Design (move the
code, do not rewrite it): extract `_tensor_projected_ops(ax, ay, x_walls, y_walls, tile_i, ox, oy, formulation) -> dict(kind, Gx0F|None,
IpxF|None, Gy0F|None, IpyF|None, Cxx, Cxy, Cyx, Cyy, EZZ, oop)` by MOVING the three assembly branches verbatim, `kind` naming which axes are
k0-free (uniform: `diag(kxv)`, no k0-free part; separable: `kron(Iy, g1)/k0 + kx0*kron(Iy, ip1)` on the patterned axis; crossed:
`Gx0F/k0 + kx0*Ip`); `_tensor_layer_modes` keeps its signature, gains `ops=None`, and rebuilds `GxF`/`GyF` from `kind` when given one;
`stack2d`'s `_geom_key` caches `ops` for tensor layers as it caches `lops` for scalar ones (the key already covers walls, tile bytes, degree,
grade, periods, `n_orders`).  GATE: the moved code must be BIT-IDENTICAL -- `np.array_equal` on all seven operators against the current function
for uniform / separable / crossed / out-of-plane / `fff_nv` cells, before and after, and the cached path must equal the uncached path bit for
bit across a wavelength AND an angle sweep.  Report the wall-clock share recovered (the audit timed `_proj` at 0.47 s of a ~20 s solve; measure
on the shipped fixtures).

## Deliverable

Implementation with the sec. 2 prefix on every new validation; oracles the library did not produce (the extrapolated RCWA oracle for item 1 is the
one the audit used -- build it from `rcwa_efficiency_1d` + `rcwa_extrapolate`, which you may CALL but not edit); derived envelopes in a new
`tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` (S5; no wall-clock assertions -- count the projection builds instead); stated
fail-befores; byte-identity proofs against `git archive <HEAD>^ lumenairy` extracted READ-ONLY into your scratch directory under
`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b6\`
(child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest).  Re-record every history document you
touch.  Report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B6_REPORT.md` (summary table: item / status / files:lines / tests /
oracle / measured before -> after; per-item sections; files touched; tests run; requested changes outside your ownership; deferred) and
`WP-B6_CHANGELOG.md` (5.47.0 release text in the style of `fixes/WP-A12_CHANGELOG.md` / `WP-A13_CHANGELOG.md`, findings G3/G10 named,
`lumenairy/elements/pmm/<file>.py:N` citations on non-trivial lines, a Migration note only if a default moves -- none should).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a12_*.py`, `test_audit2609_a13_*.py` and the VERIFY-A12/A13 files, `pytest tests/unit -k "pmm"` (slow: budget
~20 min), `tests/unit/test_v5_14_0_pmm_jones_2d.py`, `test_v5_14_0_pmm2d_oop.py`, `test_pmm2d_oop_block_eig.py`,
`test_v5_20_13_pmm_jones_2d_fff_nv.py`, `tests/unit/test_ci_kernel_consistency.py` (the census; PMM decisions must not move),
`python validation/run_all.py` for the pmm files (`ls validation/`), `ruff check`, `python scripts/record_history_fingerprints.py --check`
(your documents), `tests/unit/test_audit2609_a17_history_lint.py`.  Note: `tests/unit/test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band`
is red on this workstation at the audit base already (BLAS-build classification); it is not yours to fix, but nothing you do may change its
numbers -- report them before and after.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/elements/pmm/*.py`, their `docs/history/` documents, the new b6 test file, your two report files.  Anything else:
  "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
