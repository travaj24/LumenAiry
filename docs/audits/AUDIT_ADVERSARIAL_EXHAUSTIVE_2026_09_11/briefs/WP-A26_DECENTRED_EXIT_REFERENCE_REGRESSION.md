# WP-A26 -- the traced lens's DECENTRED exit wavefront regressed at WP-A1 (ray tracing): 2 urad -> 44 urad against a decentre-invariant analytic oracle, on axis unchanged

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy` (branch `audit-fixes-2026-09`,
HEAD 6345d99d).  Another engineer (WP-A25) is editing `lumenairy/propagators/carrier.py`, `docs/history/carrier.md`
and `tests/unit/test_audit2609_a25_carrier_focus_readout.py` right now -- do not touch those.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2, and the comment rule in `CONTRIBUTING.md`
("Modules with a history document": a code change to a module that has `docs/history/<dotted.module>.md` MUST be
re-recorded in the same change with `python scripts/record_history_fingerprints.py <module path> --reason "..."`;
`lumenairy/elements/_lens_traced.py` and most of `lumenairy/raytrace/*.py` have one).  Source comments describe
what the code does NOW and why; no "v5.xx (audit ...): pre-fix this did A" narrative in the source.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_REPORT.md`, `WP-A1_CHANGELOG.md` (the
sub-changes: the shared exit-vertex transfer F-O3; R1 reference sphere for `opd_fan_data`; R2 launch-plane tilt on
off-axis OPD fans; R3 seidel conic/aspheric terms; R4 conic RAY_MISSED_SURFACE; R5 diffraction-order medium index;
R6 `rays_from_field` aliasing, `_transfer` grazing rays, in-place hot path; R7 the P3 bundle) and `VERIFY_WP-A1.md`
(60-digit closed-form checks of the on-axis conic intersection and eikonal -- note what it did NOT cover: a
decentred Gaussian through `apply_real_lens_traced`).

## The measurement (orchestrator, 2026-09-13, read-only `git archive` bisect; the CURRENT test module imported by path
inside a child whose cwd + PYTHONPATH are the archived library, the imported library asserted per probe)

`tests/unit/test_niche_d7_decentred_fit.py::test_the_off_centre_fit_order_raise_flattens_the_exit_wavefront[0.5|1.0]`:
a Gaussian (waist `_W`) decentred by `frac * _W` through a conic singlet via `la.apply_real_lens_traced(...,
ray_subsample=1, fit_radius_beam_factor=_FRBF, beam_centre=(cx, 0), preserve_input_phase=False,
amplitude_model='screen', ...)`; the exit-slope RMS of the returned phase against the test's OWN analytic oracle
(`_exit_slope_rms`, `_bfd_by_inline_raytrace`, `_oracle_opl`: an inline exact conic raytrace to the exit VERTEX
plane, closed form, decentre-INVARIANT, sharing no code with the library).  `pre_d7=True` pins
`decentred_fit_poly_order` to the concentric order (the D7 fail-before switch).

| tree | on axis | frac 0.5: pre-D7 / D7 | frac 1.0: pre-D7 / D7 | test |
|---|---|---|---|---|
| a1ff1e6e (audit base) | 41.089 urad | 134.767 / **2.162** urad | 118.337 / **1.958** urad | PASS |
| 0067d63b (WP-A8, = the commit before A1 on this path) | 41.089 | 134.767 / 2.162 | 118.337 / 1.958 | PASS |
| **f602b72c (WP-A1)** | 41.089 | **233.859 / 44.457** | **354.413 / 31.556** | FAIL |
| every later commit incl. HEAD | 41.089 | 233.859 / 44.457 | 354.413 / 31.556 | FAIL |

The bars: `on_axis < 100 urad`, `before > 2 * on_axis` (fail-before), `after < 0.25 * on_axis`, `after < 0.1 *
before`.  The on-axis figure is unchanged to the last digit, so WP-A1 moved ONLY the decentred path -- both the
concentric-order fit (134 -> 234 / 118 -> 354 urad) and the raised-order fit (2.2 -> 44.5 / 2.0 -> 31.6 urad).  The
oracle is decentre-invariant by construction, so a correct traced lens must return to ~2 urad off axis; the new
44 urad happens to sit at the on-axis figure, which suggests the decentred beam is now referenced (exit vertex,
reference sphere, launch plane or chief ray) as if it were on axis, or its rays are launched/transferred with a
decentre-dependent error.  `_lens_traced.py` consumes the raytrace package (see the import list in your first read).
WP-A24's d6 study saw the same commit move an on-axis-ish figure by only 0.17 %, consistent with an off-axis-only
mechanism.

## Deliverable

1. **Diagnose with measurements.**  Reproduce with the test module's helpers (import it by path; call `_apply`,
   `_exit_slope_rms`, `_bfd_by_inline_raytrace`).  Extract the pre-A1 library read-only (`git archive 0067d63b
   lumenairy` into `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\a26\`; never check out or stash) and compare the decentred exit
   OPD MAPS (not just the RMS) before and after: is the difference a tilt, a defocus, a decentre-dependent
   offset, a launch-lattice change?  Then attribute it to ONE of A1's sub-changes by neutralising them one at a
   time in-process (monkeypatch / flag) or by reading which raytrace helpers `_lens_traced.py` calls on this path.
2. **Decide correctness against the oracle, then fix.**  The test's inline exact conic raytrace is the arbiter.  If
   A1's change is wrong off axis (the likely case: a helper that assumes the chief ray is the axis), fix it in the
   raytrace module(s) so that the decentred path returns to the oracle while VERIFY-A1's 60-digit on-axis pins and
   every WP-A1 test stay green.  If instead the test's oracle or its `_apply` convention is what is stale (say
   why with a closed-form argument, not a reading of the diff), restate the test with the derivation and a
   fail-before.  Do not pick silently.
3. **Pin it.**  New `tests/unit/test_audit2609_a26_decentred_exit_reference.py`: the decentred exit slope against
   the analytic oracle as a DERIVED two-sided envelope (S5: the oracle, its floor, the defect scale this catches --
   the 44 urad reading), and a pin on the mechanism you found, with a stated fail-before.
4. **Re-record** the history document of every module you change (same commit, one-line reason each).
5. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A26_REPORT.md` (summary table:
   status / files:lines / tests / oracle / measured before -> after; per-finding sections; files touched; tests run
   with counts and durations; requested changes outside your ownership; deferred) and `WP-A26_CHANGELOG.md`
   (release text in the `### Fixed -- ...` style; cite `path.py:N` only on non-trivial lines).

## Verification set (all must be green when you finish)

`tests/unit/test_niche_d7_decentred_fit.py` (slow, ~1 min), `tests/unit/test_audit2609_a1_raytrace.py` and the
VERIFY-A1 test file(s) (find them by `a1` in `tests/unit/`), `tests/unit/test_niche_d6_exact_tilted_leg.py` (slow,
~2.5 min; the d6 fixture reads `f602b72c` as a -0.17 % move, so re-measure and report whether your fix moves it),
`pytest tests/unit -k "raytrace or exit_vertex or seidel or opd_fan"`, `pytest tests/unit -k real_lens` (~4 min),
`python validation/run_all.py test_raytrace test_lenses`, `ruff check`, and `python
scripts/record_history_fingerprints.py --check`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
* NO git write commands of any kind (no add/commit/stash/checkout/restore/reset); read-only `git show` / `git
  archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/raytrace/*.py`, `lumenairy/elements/_lens_traced.py` (only if the fix must live there), their
  `docs/history/` documents, the new a26 test file, your two report files.  Do not modify
  `tests/unit/test_niche_d7_decentred_fit.py` unless its oracle is provably wrong (item 2).  Anything else goes under
  "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why.
* Finish with the report's full text as your final message.
