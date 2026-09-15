# WP-B8 (Wave 4) -- analysis and sources: the performance designs WP-A7 and WP-A11 deferred, MFT-based PSF sampling (audit sec. 15.9), Gori pseudo-modes for Schell sources

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD 284daccc or later).  Other
Wave-4 engineers are concurrently editing `lumenairy/elements/lenses_maslov.py` + `lumenairy/propagators/asymptotic*.py`, `lumenairy/elements/_lens_real.py`
+ `lens_config.py`, `lumenairy/propagators/carrier.py` + `carrier_field.py`, `lumenairy/propagators/hf.py`, `hfpi.py`, `rs.py`, `mft.py`,
`lumenairy/elements/rcwa/*`, `eme/*`, `bor/*`, `lumenairy/elements/pmm/*`, and (WP-B9) `lumenairy/raytrace/*`; never touch any of those.  You CONSUME
`propagators/mft.py`'s `fraunhofer_propagate_mft` / `_bluestein_centred_2d` for item 1; if you need a change there, specify it.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document": `lumenairy/analysis/psf_mtf_otf.py`, `zernike.py`, `ao.py`, `coherence.py`, `lumenairy/sources/core.py`, `lumenairy/elements/polarization.py`
have `docs/history/<dotted.module>.md`; a code change MUST re-record with `python scripts/record_history_fingerprints.py <module path> --reason "..."`
in the same change).  Source comments describe what the code does NOW and why; no version narrative.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A7_REPORT.md` section 6 (items 1-3, reproduced below), `WP-A11_REPORT.md`
section 6 (items 1-3), their changelogs and verify reports, and the audit's ANALYSIS and POLAR-SOURCES-INFRA partition reports.

## The items

1. **`compute_psf` memory (A7 sec. 6.1) and MFT-based PSF sampling (audit sec. 15.9).**  Two steps, both gated on bit-identity or a derived
   tolerance: (a) replace `fftshift(fft2(ifftshift(a)))` with the separable chessboard identity `chess * fft2(chess * a)`, `chess = (-1)^(i+j)`
   applied in place -- exact for even N; for odd N keep the explicit shifts or derive the tolerance (the report says the identity differs by a
   cyclic shift there); measured target: two padded copies removed (~536 MB at oversample 4).  (b) `compute_psf(..., method='mft')` (name per the
   module's vocabulary): the Soummer (2007) matrix Fourier transform through `fraunhofer_propagate_mft` -- arbitrary focal-plane pitch, no
   padding, ~0 extra memory -- as an OPT-IN alongside the padded FFT default, with a compatibility statement of how its grid relates to the
   FFT grid (the PSF grid contract must not move for existing callers).  Oracle: the analytic Airy pattern of a circular pupil and the
   closed-form Gaussian PSF, both sampled exactly at the MFT's own pitch.
2. **`encircled_energy_curve` / `encircled_energy_radius` share one profile (A7 sec. 6.2):** expose `encircled_energy_profile(E, dx, dy=None,
   centroid=None) -> (r_sorted, p_cum, r_max)` and accept it as `profile=` on both; pin "the radius is the exact inverse of the curve".  No
   content-keyed cache (audit sec. 15.5).
3. **Zernike recurrence and the DM influence-function cache (A7 sec. 6.3):** Kintner or Prata-Rusch recurrence in `n` at fixed `m` for
   `_zernike_radial`, gated on bit-near-identity against the factorial sum at every (n, m) in the shipped tables (state the derived tolerance and
   the stability limit); `ao.fit_phase`'s banded influence-function build hoisted into a `_banded_IF_apply`, `cache_basis` defaulting to False above
   a byte budget with one warning (say how the budget is derived).
4. **Gori pseudo-modes for Schell sources (A11 sec. 6.1):** `generator='fft' | 'modes'` on the Schell-model source (find the entry point in
   `lumenairy/sources/` / `analysis/coherence.py`): a stationary field with Gaussian correlation `exp(-|d|^2/(2 s^2))` as
   `phi(r) = M^(-1/2) sum_j exp(i(k_j . r + psi_j))`, `k_j ~ N(0, s^-2)`, `psi_j` uniform -- one `zgemm` per realisation (`A = exp(i(y (x) k_y + psi))`,
   `B = exp(i k_x (x) x)`), exact by construction, no grid periodisation.  Default stays `'fft'` (byte-identical).  Deliver an `M` heuristic from
   `(L/s)^2`, a chi-square test that the marginal is circular-Gaussian, and the measured correlation function against the target for both
   generators; report the cost crossover (the report estimates ~100x the padded FFT at M = 256, N = 512).
5. **Two bit-identity-sensitive memory items (A11 sec. 6.2-6.3):** `create_gaussian_beam(geometry_dtype=)` opt-in computing the geometry in
   float32 for a complex64 output (documented tolerance; default unchanged), and `apply_jones_matrix` accumulating with `np.multiply(out=)` +
   `+=` -- ONLY if bit-identical (the report warns the complex-multiply operand order must be preserved exactly; prove it on the A11 bit-identity
   matrix or leave it and say why).

## Deliverable

Implementation with the sec. 2 prefix on every new validation; oracles the library did not produce; derived envelopes in a new
`tests/unit/test_audit2609_b8_analysis_sources.py` (S5; no wall-clock assertions -- peak-memory via tracemalloc is a MEASUREMENT, not an assertion,
unless you can derive a bar from the array counts); stated fail-befores; byte-identity proofs against `git archive <HEAD>^ lumenairy` extracted
READ-ONLY into your scratch directory under `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b8\`
(child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest); wall-clock and peak-memory medians of interleaved
runs reported.  Re-record every history document you touch.  Report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B8_REPORT.md`
(summary table: item / status / files:lines / tests / oracle / measured before -> after; per-item sections; files touched; tests run; requested
changes outside your ownership; deferred) and `WP-B8_CHANGELOG.md` (5.47.0 release text in the `### Performance -- ...` / `### Added -- ...` style
of `fixes/WP-A7_CHANGELOG.md`, findings A6 / Z3 named, sec. 15.9 named for the MFT PSF, `lumenairy/analysis/<file>.py:N` citations on non-trivial
lines, Migration notes only for defaults that move -- none should).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a7_*.py`, `test_audit2609_a11_*.py` and their VERIFY files, `pytest tests/unit -k "psf or mtf or zernike or encircled or
ao_ or coherence or schell or gaussian_beam or jones"`, `tests/unit/test_audit2609_a15b_reexports.py`, `python validation/run_all.py` for the analysis
and sources files (`ls validation/`), `ruff check`, `python scripts/record_history_fingerprints.py --check` (your documents), 
`tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only: `lumenairy/analysis/*.py`, `lumenairy/sources/*.py`, `lumenairy/elements/polarization.py`, their `docs/history/` documents, the new b8
  test file, your two report files.  Anything else: "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
