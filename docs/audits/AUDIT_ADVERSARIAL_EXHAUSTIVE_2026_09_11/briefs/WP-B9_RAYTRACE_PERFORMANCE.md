# WP-B9 (Wave 4) -- ray tracing: the performance and completeness items WP-A1 deferred

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD 284daccc or later).  Other
Wave-4 engineers are concurrently editing `lumenairy/elements/lenses_maslov.py` + `lumenairy/propagators/asymptotic*.py`, `lumenairy/elements/_lens_real.py`
+ `lens_config.py`, `lumenairy/propagators/carrier.py` + `carrier_field.py`, `lumenairy/propagators/hf.py`, `hfpi.py`, `rs.py`, `mft.py`,
`lumenairy/elements/rcwa/*`, `eme/*`, `bor/*`, `lumenairy/elements/pmm/*`, `lumenairy/analysis/*`, `lumenairy/sources/*`; never touch any of those.
You own `lumenairy/raytrace/*` only.  `lumenairy/elements/_lens_traced.py` consumes the tracer and was re-derived by WP-A26 last night: do not edit it,
and keep its fixtures byte-identical (they are in the verification set).

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document": every `lumenairy/raytrace/*.py` module has `docs/history/lumenairy.raytrace.<name>.md`; a code change MUST re-record with
`python scripts/record_history_fingerprints.py <module path> --reason "..."` in the same change).  Source comments describe what the code does
NOW and why; no version narrative.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_REPORT.md` (section 2 for R1-R7 and the shared exit-vertex helper; section 6
items 1-6, reproduced below), `WP-A1_CHANGELOG.md`, `VERIFY_WP-A1.md` (the 60-digit closed-form oracles -- reuse the METHOD), `WP-A26_REPORT.md`
(section 2.4: the exact conic trace written there is an oracle you can rebuild), and the audit's RAYTRACE partition report.

## The items (WP-A1 section 6)

1. **`_refract` / `_reflect` renormalise hoist** (perf #3).  Exact vector Snell with a unit normal returns a unit vector identically; the
   per-surface `sqrt` + 3 divides + 3 `np.where` remove ~1e-16 of drift.  Keep the degenerate-ray detection per surface (`mag < 1e-30 |
   ~isfinite` -> `RAY_NAN`), move the DIVISION to a single pass at the end of `trace` / `trace_world`.  `analysis/ghost.py` and the differential
   FD path also call `_refract`: either put the hoist behind a flag every caller passes explicitly, or keep those callers' behaviour identical by
   construction -- prove the choice with a bit-identity sweep (the tolerance, if any, derived from the ~1e-16 drift).
2. **Analytic sphere normal for `is_pure_spherical`** (perf #2, the 24 % block): `(x, y, z - R)/R` instead of the generic sqrt / where / divide /
   sqrt.  The v4.12.0 attempt failed because it was applied without the matching intersection change; the intersections are aligned now (R4),
   so retry with the Maslov cross-backend asymptotic test as the gate (the report names it).
3. **`ray_fan_data` / `opd_fan_data` issue four `trace()` calls** (perf #4): one concatenated bundle, preserving the RT-5 invariant
   (`ey(0) == ex(0) == 0`) across the different `ep_off` offsets and field-tilt axes.
4. **`trace_jax` per-call prep** (perf #7, ~750 us/call in `_build_jax_prescription`): cache the built `JaxPrescription` on a content hash of the
   prescription (audit sec. 15.5: the key must cover every field that changes the trace), or document the pre-built fast path -- measure which.
5. **Area-uniform pupil sampling** (alt-algorithm #6): a Vogel / sunflower generator behind `pattern=` on `make_rings` (and whatever `spot_rms`
   consumers pass it through), DEFAULT UNCHANGED -- the orchestrator's decision is that the current equal-radius default stays until the whole
   spot corpus is re-measured; measure the mean `r/R` (0.5806 today vs 0.6667 uniform) and the spot-rms shift on two prescriptions and put both
   in the changelog.
6. **Aspheric support in `ray_transfer_jacobian_analytic`** (alt #4): the polynomial terms in `_adrt_step`'s implicit `F` and its gradient, so
   the analytic path stops raising `NotImplementedError` for `aspheric_coeffs`; gate: finite-difference cross-check at three heights on an
   aspheric singlet, tolerance derived from the FD step.

Item 7 of the report (polarisation ray tracing) stays out of scope; say so.

## Deliverable

Implementation with the sec. 2 prefix on every new validation; oracles the library did not produce (closed-form sphere/conic normals, the exact
conic trace, FD Jacobians); derived envelopes in a new `tests/unit/test_audit2609_b9_raytrace_perf.py` (S5; no wall-clock assertions -- count
`trace()` calls, sqrt/divide operations or cache hits instead); stated fail-befores; byte-identity proofs against `git archive <HEAD>^ lumenairy`
extracted READ-ONLY into your scratch directory under `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b9\`
(child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest); wall-clock medians of interleaved runs
reported.  Re-record every history document you touch.  Report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B9_REPORT.md`
(summary table: item / status / files:lines / tests / oracle / measured before -> after; per-item sections; files touched; tests run; requested
changes outside your ownership; deferred) and `WP-B9_CHANGELOG.md` (5.47.0 release text in the `### Performance -- ...` / `### Added -- ...` style
of `fixes/WP-A1_CHANGELOG.md`, `lumenairy/raytrace/<file>.py:N` citations on non-trivial lines, Migration notes only for defaults that move --
none should).

## Verification set (all green when you finish)

The four A1 test files and the VERIFY-A1 file(s) (`a1` in `tests/unit/`), `tests/unit/test_audit2609_a26_decentred_exit_reference.py` (10),
`tests/unit/test_niche_d7_decentred_fit.py` (slow, ~4.5 min: must stay byte-identical at 2.371 / 1.683 urad), `pytest tests/unit -k "raytrace or
exit_vertex or seidel or opd_fan or ghost or jax_trace"`, `pytest tests/unit -k real_lens` (~4 min), the Maslov cross-backend asymptotic test WP-A1
names as the gate for item 2, `python validation/run_all.py test_raytrace test_lenses`, `ruff check`, `python scripts/record_history_fingerprints.py --check`
(your documents), `tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/raytrace/*.py`, their `docs/history/` documents, the new b9 test file, your two report files.  Anything else (including
  `analysis/ghost.py`, `_lens_traced.py`): "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
