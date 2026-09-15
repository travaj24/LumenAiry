# WP-A25 -- the traced-carrier chain's paraxial focus readout regressed at WP-A6 (C1): an unclipped Gaussian doublet lost half its focal power

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy` (branch `audit-fixes-2026-09`,
HEAD 6c83ac91 = release 5.46.0, plus a few uncommitted test fixes in the working tree that are NOT yours: leave
`lumenairy/memory.py`, `lumenairy/elements/lens_config.py`, `tests/unit/test_audit_v5_24_2_g07_dedup.py`,
`tests/unit/test_niche_r3_gbd_mem_lstsq.py`, `tests/unit/test_perf_v4_12_0_asymptotic.py`,
`tests/unit/test_pmm_m2_window_contract.py` alone).

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 (error prefix) and the comment rule in
`CONTRIBUTING.md` ("Modules with a history document": a code change to `lumenairy/propagators/carrier.py` MUST
re-record `docs/history/carrier.md` in the same change:
`python scripts/record_history_fingerprints.py lumenairy/propagators/carrier.py --reason "..."`).  Source comments
describe what the code does NOW and why -- no "v5.xx (audit ...): pre-fix this did A" narrative in the source.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A6_REPORT.md` and `WP-A6_CHANGELOG.md` (C1:
"the focal peak no longer collapses when the reference carrier is a few percent off the beam"; C2-C5),
`VERIFY_WP-A6.md`, and `WP-A24_REPORT.md` (the d6 investigation: it proved `final_leg='exact'` never enters
`_default_focus_standoff`; THIS finding is on the paraxial leg, which does).

## The measurement (orchestrator, 2026-09-13, read-only `git archive` bisect, current test module against each tree)

`tests/unit/test_niche_p2_design_battery.py::test_battery_through_focus_unclipped_doublet_matches_gaussian` runs
`_through_focus(_d_doublet, 2.0e-3, 2.5)`: a 2 mm-waist Gaussian through a well-corrected doublet with a 2.5x
aperture (unclipped), `la.propagate_traced_carrier_chain(..., final_distance=-R, traced_kwargs=dict(parallel_amp=False,
on_undersample='silent'), focus_readout=dict(dx_out=0.5e-6, N_out=512, on_replica='ignore'))`, then a
9-step through-focus scan with `angular_spectrum_propagate_mft` over +/- z_R, reporting the best-focus FWHM against
the analytic Gaussian `1.177 * lambda |R| / (pi w_exit)` and the encircled energy inside 1 / 2 / 3 waists relative
to the LAUNCHED power.

| tree | FWHM | theory | ratio | EE 1w | EE 2w | dz_best |
|---|---|---|---|---|---|---|
| a1ff1e6e (audit base, 5.45.0) | 16.50 um | 17.41 um | 0.948 | 0.843 | 0.953 | +0.131 mm |
| 818251fd (= a18ab074^, VERIFY-A14) | 16.50 um | 17.41 um | 0.948 | 0.843 | 0.953 | +0.131 mm |
| **a18ab074 (WP-A6, C1-C5)** | **20.50 um** | 17.41 um | **1.177** | **0.353** | **0.495** | **+0.393 mm** |
| every later commit incl. HEAD | 20.50 um | 17.41 um | 1.177 | 0.353 | 0.495 | +0.393 mm |

The test's bar is `fwhm/fwhm_th < 1.10` and `EE 2w > 0.95` (docstring: measured 2026-07-25 as 18.5 um / 1.062x,
EE1w 86.0 %, EE2w 99.7 %).  A single step, at WP-A6, and the step is physically impossible for this fixture: an
unclipped Gaussian through a well-corrected doublet cannot put half of its power outside two waists at best focus.
Either the readout plane is no longer where the chain says it is (dz_best tripled, and +0.39 mm is near the edge of
the +/- z_R = +/-0.52 mm scan, so the true best focus may lie OUTSIDE the scan), the readout field is no longer
normalised to the launched power, or the readout window/period (`on_replica='ignore'`) now aliases.  WP-A6's C1
resized the paraxial focus readout from the BEAM (`_default_focus_standoff` -> `_beam_containment_standoff`,
`_check_focus_containment`, `_achievable_focus_margin`, `carrier_referenced_focus_readout` at
`lumenairy/propagators/carrier.py`); VERIFY-A6 confirmed C1 on its own fixtures (165 pins) but never ran this
battery, and neither did WP-A6.  The two sibling battery tests (`..._truncated_aperture_broadens_predictably`,
`..._relay_cliff_is_a_focal_catastrophe`) still pass; report their values before/after as well.

## Deliverable

1. **Diagnose.**  Reproduce with the battery helper (import the test module by path and call `_through_focus`).
   Instrument the chain: where is the readout plane relative to the geometric/Gaussian focus before and after
   C1 (`git archive 818251fd lumenairy` into the scratchpad for the pre-C1 library -- READ-ONLY, never check out or
   stash), what standoff does `_beam_containment_standoff` choose here and why, is the readout power equal to the
   launched power, does the requested 512 x 0.5 um window alias.  Name the mechanism with a measurement, not a
   reading of the diff.
2. **Fix it in `carrier.py`** so that this fixture returns to the analytic Gaussian focus (FWHM within the 10 %
   bar, EE2w >= 0.95) WITHOUT undoing what C1 fixed: the C1 fixture(s) in `tests/unit/test_audit2609_a6_carrier.py`
   / `test_audit2609_a6_verify_carrier.py` (165 ids) must stay green, and so must WP-A24's
   `tests/unit/test_audit2609_a24_decentre_calibration.py` (7) and `tests/unit/test_niche_d6_exact_tilted_leg.py`
   (38, slow, ~2.5 min).  If C1's beam-sized standoff and this fixture's needs genuinely conflict, say so with
   numbers and propose the rule that satisfies both; do not pick one silently.
3. **Pin it.**  A new `tests/unit/test_audit2609_a25_carrier_focus_readout.py`: the battery fixture's focus
   metrics as a DERIVED two-sided envelope (S5: state the oracle -- the analytic Gaussian -- its floor, and the
   defect scale this catches: the 0.495 reading), fail-before on the pre-fix tree (say how you showed it), and a
   pin on whatever mechanism you found (the standoff choice, the normalisation, the window), so the next resizing
   of the readout cannot silently reopen it.
4. **Re-record** `docs/history/carrier.md` in the same change with a one-line reason.
5. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A25_REPORT.md` (the usual shape:
   summary table with status / files:lines / tests / oracle / measured before -> after; per-finding sections;
   files touched; tests run with counts and durations; requested changes outside your ownership; deferred) and
   `WP-A25_CHANGELOG.md` (release text in the `### Fixed -- ...` style of the other WP changelog files; cite paths
   as `lumenairy/propagators/carrier.py:N` on non-trivial lines).

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time
  (a slow-lane timing run may still be finishing on this box).
* NO git write commands of any kind (no add/commit/stash/checkout/restore/reset); the orchestrator commits with an
  explicit file list.  Read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.
* Own only: `lumenairy/propagators/carrier.py`, `docs/history/carrier.md`, the new a25 test file, your two report
  files.  If the fix needs another file, list it under "requested changes outside my ownership" with the exact
  edit.  Do not modify `tests/unit/test_niche_p2_design_battery.py` unless its bar is provably wrong (it is not:
  the base tree passes it).
* Comments say what the code does now and why.  Measured derivations of live constants stay, dated, in the code
  and the tests.
* Finish with the report's full text as your final message.
