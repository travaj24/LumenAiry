# WP-B3 (Wave 4) -- propagator kernels: the HFPI walk's output plane, the Sobol sampler, the Shen-Wang pixel-integrated RS kernel, the band-limited (chirp-Z) resampler

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`, HEAD 81d5b586
(= release 5.46.0, Wave 3 closed).  Two other Wave-4 engineers are working concurrently on `lumenairy/elements/lenses_maslov.py` +
`lumenairy/propagators/asymptotic*.py` (WP-B1) and on `lumenairy/elements/_lens_real.py` + `lumenairy/elements/lens_config.py` (WP-B2);
never touch those files.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 (error prefix) and sec. 7 (signs), the comment rule in
`CONTRIBUTING.md` ("Modules with a history document": `lumenairy/propagators/rs.py`, `hf.py`, `hfpi.py`, `mft.py` and `system.py` have
`docs/history/lumenairy.propagators.<name>.md`; a code change MUST re-record with `python scripts/record_history_fingerprints.py <module path>
--reason "..."` in the same change).  Source comments describe what the code does NOW and why; no version narrative.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A5_REPORT.md` section 6 (the four deferred designs, reproduced below,
with their measured motivations), `WP-A5_CHANGELOG.md`, `VERIFY_WP-A5.md` (the verifier's independent RS-I double quadrature and its
transparency property for HFPI), and the audit findings K6, K9, K13, K22 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`.

## The four items (WP-A5 section 6; implement all four)

1. **K13 -- `propagate_hfpi_through_prescription` gets an explicit output plane.**  Add `z_output: float | None = None` (default: the
   last surface's `z`, today's behaviour) and, when it differs, a closing `propagate_to_plane(paths, z_target=z_output, wavelength=...)`
   so every path has a non-zero final leg; then `normalisation='physical'` is well-defined there and becomes the default for the walk,
   retiring the "amplitudes are not photometric" warning for that case.  Oracle: ASM through a thin-lens prescription (power, centroid,
   EE against `angular_spectrum_propagate`).
2. **K22 -- `sampler='jittered' | 'sobol'` for HFPI.**  `scipy.stats.qmc.Sobol(d=4).random(n)` over the same `(pixel_x, pixel_y,
   cos theta, phi)` cube `init_paths_stratified` strata-samples; `jittered` stays the default; `n_paths` an exact cap.  MEASURE the
   convergence rate of both against the same reference field before advertising the QMC gain (the integrand has a hard aperture edge, so
   `O(N^-1)` is not guaranteed); the changelog states what was measured, not the textbook rate.
3. **K9 second half -- Shen-Wang (2006) pixel-integrated RS kernel** for the `'spatial'` branch of `rs.py`: the analytic integral of
   the RS-I Green's function over each pixel (Appl. Opt. 45, 1102, already in `rs.py`'s references).  The audit measured the point-sampled
   kernel converging `O(dx)` on hard-aperture inputs (2.86e-2 -> 1.08e-3 over N = 256..2048); the integrated kernel should give `O(dx^2)`.
   Oracle: the brute-force RS quadrature `repro/orch/p7_rs.py`'s METHOD (write your own; do not import the repro), and the closed-form
   Fresnel/Lommel cases the A5 tests already use.  Ship it as `kernel='spatial-integrated'` (or the name the module's vocabulary suggests),
   keep today's kernel bit-identical under its existing name, and say whether it should become the default with the convergence table.
   The same kernel serves `hf.py`'s quadrature floor: do it there too if the change is contained, else specify the edit.
4. **K6 second half -- band-limited (chirp-Z) resampler for `resample_field`** in `mft.py`, routed through `_bluestein_centred_2d`
   (which already performs exactly that operation), as `method='chirpz'` beside today's method (bit-identical default).  Do NOT switch
   the `fresnel`/`sas` legs in `system.py` or `_lens_real.py` yourself: measure what the switch would do on those legs (the MTF, the
   crop warning) and put the exact call-site edits under "requested changes outside my ownership" -- `_lens_real.py` is WP-B2's file
   right now and the orchestrator will coordinate the switch after both land.

## Deliverable

For each item: implementation with the sec. 2 error prefix on every validation, an ORACLE the library did not produce, a derived
two-sided envelope in a new test file `tests/unit/test_audit2609_b3_propagator_kernels.py` (S5: oracle, floor, defect scale; no wall-clock
assertions -- operation counts or convergence orders instead), a stated fail-before, byte-identity proofs for every path whose default did
not move, and wall-clock medians of interleaved runs for the cost of each new option (report, do not assert).  Re-record every history
document you touch.  Report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B3_REPORT.md` (summary table: item / status /
files:lines / tests / oracle / measured before -> after; per-item sections; files touched; tests run with counts and durations; requested
changes outside your ownership; deferred) and `WP-B3_CHANGELOG.md` (5.47.0 release text in the `### Added -- ...` / `### Changed -- ...`
style of `fixes/WP-A5_CHANGELOG.md`, findings K6/K9/K13/K22 named, `lumenairy/propagators/<file>.py:N` citations on non-trivial lines,
Migration notes for the walk's new default normalisation and anything else whose default output moves).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a5_*.py` and the VERIFY-A5 file(s) (`a5` in `tests/unit/`), `pytest tests/unit -k "hfpi or hf_ or rs_ or resample or mft or bluestein"`,
`tests/unit/test_niche_d2_chain_multi.py` (uses the MFT), `tests/unit/test_audit2609_a25_carrier_focus_readout.py` (uses `_bluestein_centred_2d`
through the readouts), `python validation/run_all.py test_propagators test_hfpi` (or the files that exist for those subsystems -- `ls validation/`),
`ruff check`, `python scripts/record_history_fingerprints.py --check`, `tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator
  commits with an explicit file list.
* Own only: `lumenairy/propagators/hf.py`, `hfpi.py`, `rs.py`, `mft.py`, their `docs/history/` documents, the new b3 test file, your two
  report files.  `system.py`, `_lens_real.py`, `carrier.py` and every other file: "requested changes outside my ownership" with the exact
  edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
