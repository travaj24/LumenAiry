# WP-B10 (Wave 4) -- a disc-orthogonal (Zernike) basis for the traced lens's OPL / exit-map fits (audit sec. 15.9), OPT-IN

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD is the newest commit).  Other
Wave-4 engineers are editing `lumenairy/elements/lenses_maslov.py` + `lumenairy/propagators/asymptotic*.py`, `lumenairy/elements/_lens_real.py` (a
verifier and a follow-up), `lumenairy/propagators/carrier.py` + `carrier_field.py`, `lumenairy/propagators/system.py`, `lumenairy/elements/rcwa/*`,
`eme/*`, `bor/*`, `lumenairy/elements/pmm/*`, `lumenairy/analysis/*`, `lumenairy/sources/*`, `lumenairy/raytrace/*`; never touch any of those.
You own `lumenairy/elements/_lens_traced.py` (and `_lens_imap.py` only if the inverse-characteristic evaluator's fit must follow).

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document": `lumenairy/elements/_lens_traced.py` has `docs/history/lumenairy.elements._lens_traced.md`; a code change MUST re-record it with
`python scripts/record_history_fingerprints.py lumenairy/elements/_lens_traced.py --reason "..."` in the same change).  Source comments describe what
the code does NOW and why; no version narrative.

Then: the audit's sec. 15.9 line -- "Zernike (disc-orthogonal) basis for the traced fits: the entire fit-radius / arbiter / predictor apparatus
exists because a square Chebyshev basis couples marginal rays into defocus on a disc" -- and its TR-MAIN-1 / TR-MAIN-2 partition reports;
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A3_REPORT.md` (T-findings on the fits), `VERIFY_WP-A3.md`, `WP-A26_REPORT.md`
(sections 2.5-2.6, 3 and 8: the decentred fit's skirt weights, the square-lattice-in-a-disc-domain observation A26-7, the residual-magnitude
blindness A26-6 -- the three cures it measured and refused are the ones you must not repeat), and `tests/unit/test_niche_d7_decentred_fit.py`,
`test_niche_c11_decentred_fit_arbiter.py`, `test_niche_c12_physics_fit_selection.py`, `test_niche_c13_lstsq_conditioning.py`, `test_niche_d1_tilted_carrier.py`,
`test_fix_d5_fit_domain_basis.py` (the fit apparatus's own pins).  `_Cheb2DEvaluator`, `_solve_fit`, `_fit_disc`, `_DECENTRED_FIT_POLY_ORDER` (= 16
since WP-A26), `_FIT_DISC_OUTSIDE_WEIGHT_REL` and the conditioning step-down are the code you are extending.

## The design

Add `fit_basis='chebyshev' | 'zernike'` (name per the module's vocabulary; default `'chebyshev'`, byte-identical) to the traced lens's fits.  On the
zernike path the fit's design matrix is the Zernike polynomials (orthogonal on the unit disc) evaluated on the launch coordinates scaled to the FIT
DISC (`_fit_disc`'s radius; for the decentred branch the disc is centred on the beam), with the total degree matching the shipped order so the
two bases span the same polynomial space on the disc (say so, and prove it with a rank/projection check); the weighted restriction (D1's skirt) is
kept, because it is what keeps the map single-valued outside the disc (WP-A26 section 3.1 measured that removing it folds the map).  The claim to
test is the audit's: with a disc-orthogonal basis the marginal rays no longer couple into defocus, so the fit-radius arbiter / predictor
(`test_niche_c11`, `c12`) should have nothing to arbitrate, the conditioning step-down should fire less often (Gram rcond improves), and the
decentred exit slope against the d7 analytic oracle should be at least as good as the Chebyshev fit at the same order.  Measure all three on the
d7 Fermat singlet (0.5 / 1.0 beam radii of decentre) and on D1's adversarial ghost geometry (folds, off-beam amplitude, sign changes of
`d(x_out)/dx`), for orders 6..20, both bases, and report the ladder.  If the Zernike fit is NOT better on those fixtures, ship it opt-in with the
table, and say what the disc basis does buy (conditioning, interpretability) and what it does not.  Do not change the default in this package.

Also measure the cost (interleaved medians, decentred call, both bases at the same order) and the conditioning census the C13 tests use (Gram
rcond and `||b - Ax||` against an independent QR solve).

## Deliverable

Implementation with the sec. 2 prefix on every new validation; the d7 analytic Fermat oracle and an independent exact conic trace as oracles (the
d7 file's inline oracle is decentre-invariant; do not import the test, rebuild the method); derived envelopes in a new
`tests/unit/test_audit2609_b10_zernike_fit_basis.py` (S5; no wall-clock assertions; count arbiter engagements / step-down firings instead); the
byte-identity of the default basis proved against `git archive <HEAD>^ lumenairy` extracted READ-ONLY into your scratch directory under
`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b10\`
(child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest); a stated fail-before (the coupling of marginal
rays into defocus on the Chebyshev basis, measured); re-record the history document; report
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B10_REPORT.md` (summary table: item / status / files:lines / tests / oracle / measured
before -> after; sections; files touched; tests run; requested changes outside your ownership; deferred) and `WP-B10_CHANGELOG.md` (5.47.0 release
text, `### Added -- ...` style of `fixes/WP-A3_CHANGELOG.md`, sec. 15.9 named, `lumenairy/elements/_lens_traced.py:N` citations on non-trivial lines,
no Migration note because no default moves).  `LensNumerics` would carry the new field (`lumenairy/elements/lens_config.py`, `_NUMERICS_FOR` for
the traced entry points, and the census count in `tests/unit/test_audit2609_a16_lens_config_round_trip.py` +1): put those exact edits under
"requested changes outside my ownership" -- the orchestrator applies them, since a verifier is working in `lens_config.py`.

## Verification set (all green when you finish)

`tests/unit/test_niche_d7_decentred_fit.py` (slow, ~4.5 min; the default must stay at 2.371 / 1.683 urad), `test_niche_c11_decentred_fit_arbiter.py`,
`test_niche_c12_physics_fit_selection.py`, `test_niche_c13_lstsq_conditioning.py`, `test_niche_c6_fit_guard.py`, `test_fix_d5_fit_domain_basis.py`,
`test_niche_d1_tilted_carrier.py`, `test_audit2609_a26_decentred_exit_reference.py`, `test_audit2609_a3_*.py` + VERIFY-A3 files,
`test_audit2609_a16_*.py`, `pytest tests/unit -k real_lens`, `tests/unit/test_niche_d6_exact_tilted_leg.py` (slow), `python validation/run_all.py
test_lenses`, `ruff check`, `python scripts/record_history_fingerprints.py --check` (your document), `tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only git is fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/elements/_lens_traced.py` (+ `_lens_imap.py` if unavoidable), their `docs/history/` documents, the new b10 test file, your
  two report files.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
