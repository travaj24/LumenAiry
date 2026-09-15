# WP-B7 (Wave 4) -- the asymptotic family: Y4 performance, the FGA-vs-phase_screen convergence question, the uniform asymptotics (Pearcey / fold Airy) behind `integration_method='uniform'`, GBD kernel clipping (S9)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD is the newest commit; WP-B1's
Maslov S6 work has landed in `lenses_maslov.py` and `asymptotic*.py` -- read its report `fixes/WP-B1_REPORT.md` first, and do not undo it).  Other
Wave-4 engineers and verifiers are editing `lumenairy/elements/_lens_real.py`, `lens_config.py`, `_lens_traced.py`, `lumenairy/propagators/carrier.py`,
`carrier_field.py`, `system.py`, `hf.py`, `hfpi.py`, `rs.py`, `mft.py`, `lumenairy/elements/rcwa/*`, `eme/*`, `bor/*`, `pmm/*`, `lumenairy/analysis/*`,
`lumenairy/sources/*`, `lumenairy/raytrace/*`; never touch any of those.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document": `lumenairy/elements/lenses_maslov.py`, `lenses_gbd.py`, `lumenairy/propagators/asymptotic*.py`, `fga.py`, `gbd.py` have documents;
re-record each you change with `python scripts/record_history_fingerprints.py <module path> --reason "..."` in the same change).  Source comments
describe what the code does NOW and why; no version narrative.

Then: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A4_REPORT.md` section 6 items 3-8 (the designs, reproduced below),
`WP-A4_CHANGELOG.md`, `VERIFY_WP-A4.md` (+ its changelog: the dimensionless LG merit, the sigma-grid branch), `WP-B1_REPORT.md`, and the audit's
MASLOV-GBD-FGA and ASYMPTOTIC partition reports (Probe 3, the fold-caustic oracle that was never run, is yours to run).

## The items (WP-A4 section 6)

3. **FGA vs `phase_screen` at NA 0.145.**  At the focus of an f = 1.2 mm singlet the two members disagree 3.9x in intensity-rms spot width
   (12.72 um for `fga` at default sampling vs 3.27 um for `phase_screen`, diffraction limit 3.44 um), and the dispatcher prefers `fga` there.
   Run the convergence sweep of FGA's `w0_factor` / `dq_step` / `p_max` against a converged Rayleigh-Sommerfeld (brute-force quadrature, your
   own) or `apply_real_lens_traced` reference at this NA.  Decide with the table: if FGA converges to `phase_screen`, its default sampling is
   the defect (fix it, with the derivation); if it does not, `na_threshold` is mis-set (re-derive it).  Either way the routing must end on the
   member that is closer to the oracle at every NA in the sweep.
4. **Y4 performance, three pure refactors gated on bit-identity:** (a) one fused `_basis_and_grad34` per evaluation contracted against
   `np.stack([coef_s1x, coef_s1y, coef_phi])` (the auditor measured 2.3-5.1x on the dominant 79 % of runtime); (b) hoist the loop-invariant `T1`,
   `T2`, `T12` out of `_solve_envelope_stationary_batch`'s Newton loop (1.4x); (c) a scale-relative Newton tolerance `tol * max(r0, 1)` so the loop
   stops instead of always running 12 iterations -- (c) moves bits: measure against the oracle and derive its bar, or ship it opt-in.
5. **Y4, `aberration_tensor` default cost:** build only the requested `(p, l)` pairs in `decompose_lg` (21 built / 11 used), cache
   `_measure_image_plane_waist` per `(fit, s2_image, w_s, w_p, v2_centre)` (audit sec. 15.5: the key covers everything that changes the answer),
   and revisit the `sigma_grid_n` cap once (4) makes 512 affordable -- the default currently pays 12x the n = 64 cost and still returns an aliased
   answer (its own warning asks for 494, the cap truncates to 256).  Measure before/after against the analytic LG orthogonality and the
   dimensionless LG merit pins.
6. **Y5 structural:** the triplicated `_compute_M_b` / Newton / polynomial-substitution kernels (the Y1 defect landed in one copy and not the
   others).  Collapse them ONLY with a bit-identity harness across the three call sites; if the harness cannot be made green, document the
   plan and stop -- this is the most invasive item and the least urgent.
7. **Sec. 15.9 -- the uniform asymptotics.**  `uniform_fold_airy` and `pearcey` are dead code.  Wire them behind `integration_method='uniform'`
   (default unchanged): locate the two coalescing saddles of the full exponent, form `zeta = (3/4 (S2 - S1))^(2/3)` (Chester-Friedman-Ursell),
   evaluate `Ai(-k^(2/3) zeta)` / `Ai'`, and gate the path against a brute-force Rayleigh-Sommerfeld integral through the marginal focus of an f/2
   singlet (a genuine fold) -- the oracle the auditor's Probe 3 specified and nobody ran.  If the uniform path does not beat `stationary_phase`
   at the fold against that oracle, leave it dead and publish the table.
9. **The JAX Maslov sibling is a thin-OPD phase screen, NOT a saddle solver** -- WP-B1's request 5 was wrong and is struck (VERIFY-B1 section 7,
   WP-B1_REPORT.md addendum).  `lumenairy/elements/_lens_jax.py::apply_real_lens_maslov_jax` contains no stationary-point solve, so S6 cannot
   apply and `_input_phase_terms` has nothing to feed there.  Its REAL defect, measured by VERIFY-B1 on an N-SF11 f = 13.3 mm singlet: the
   thin-screen approximation under-shoots the chief-ray displacement of a tilted input by 2.6 % at half the lens NA and 3.7 % at the lens NA
   (about 1.6 diffraction-spot radii).  Re-derive that on your own fixture against an exact trace of the input's own rays, then EITHER correct
   the screen (a chief-ray displacement term, byte-identical on a collimated input) OR document the error envelope on the function and warn
   above a derived tilt bar -- your measurement decides which, and the report says why.  You own `_lens_jax.py` for this item only.
10. **Re-found the S6 fallback statistic** (VERIFY-B1 F1, P1 owner decision): `_K1_FIT_RESIDUAL_MAX = 0.5` scores the fit's VALUE error while
    the saddle consumes its two DERIVATIVES -- on a tilted carrier the answer is gone at residual 0.05 (fidelity 0.0004) while the bar does not
    fire until 0.5, and the WP-B1 section 3 speckle ladder the bar was derived from does not reproduce (measured 20x lower; the measured column
    is now in the source comment).  Candidates VERIFY-B1 names: the fit's derivative error (refit `k1` at `poly_order - 1` and compare `dk1/du`
    at the ray points) or the Newton's in-box non-convergence fraction (WP-B1 section 6 item 3).  Derive the new bar two-sided on BOTH fixtures
    (WP-B1's and VERIFY-B1's fixture B), keep every case that works today engaging, and restate `test_verify_b1_the_k1_fit_residual_is_the_statistic_it_claims_to_be`
    only if the statistic itself changes.  `_S1_FIT_RESIDUAL_MAX = 2.5e-3` (VERIFY-B1 V1) stays unless your measurement moves it.
11. **The pupil chart is sized three times too wide on a tilted input** (VERIFY-B1 F2): `na_proxy = na_lens + na_input` uses the 3-sigma
    angular-spectrum moment ABOUT ZERO, so a uniform tilt theta contributes 3 theta; at tilt 2x the lens NA `na_proxy` reads 0.368 against a
    needed 0.158, the order-4 chart's OPD fit residual is 0.164 waves and fidelity 0.078, while `input_na=theta` gives 1.3e-3 waves and 0.898.
    Size the chart from the MEAN launch direction plus the SPREAD; gate on bit-identity for a collimated input and on the b1 file's fidelity
    envelopes for the tilted ones.  This also cuts runtime on every tilted call (measure it).

8. **S9 -- GBD `_reconstruct_fft` kernel clipping:** clip the kernel to `+-ceil(R_cut/dx)` (the bound `_reconstruct_windowed` already computes)
   and pad to `scipy.fft.next_fast_len`; the auditor measured the FFT peak at 36x the output-grid bytes, scaling as N^2 with no cap (~9.7 GB at
   N = 4096).  Bit-identical or a derived tolerance; measure peak memory before/after.

## State at launch (read before item 1)

* WP-B1 landed as 2871e92e and its follow-up 8dab7de5 made the S6 policy a per-call keyword: `apply_real_lens_maslov(input_wavevector_saddle=None
  | False | True)` overrides the module seam `_S6_INPUT_WAVEVECTOR_SADDLE` (the process default), is forwarded through the `fold_split` leg, and is
  classified in `lens_config.KWARG_ONLY`.  Item 9 (the JAX sibling `_lens_jax.apply_real_lens_maslov_jax`) must carry the SAME keyword with the same
  three-valued semantics and the same fallback rule (k1-fit residual > `_K1_FIT_RESIDUAL_MAX`), so a caller can switch backends without changing the
  saddle they asked for; measure the JAX sibling's error on WP-B1's tilted fixture before and after (VERIFY-B1 reports a number for it).
* VERIFY-B1 added a second gate to the S6 fallback: `_S1_FIT_RESIDUAL_MAX = 2.5e-3` on the entrance-coordinate chart fit's relative residual
  (a uniform tilt fits `k1` perfectly while an over-sized chart misplaces the coordinates the term is contracted against); the b1 test file is
  at 33 ids.  Read `fixes/VERIFY_WP-B1.md` sections 4 and 6 before items 10-11.
* `fixes/VERIFY_WP-B1.md` exists by the time you start: read its verdict table and "Follow-up" section first and do not undo anything it fixed in
  `lenses_maslov.py` / `asymptotic*.py`; its requested changes that fall in your files are yours to apply.
* WP-B9 (7592af4a) shipped `trace(sphere_normal=, renormalize=)` opt-in with byte-identical defaults; nothing in the asymptotic family moves because
  of it, but the knife-edge asymptotic pins (`test_audit_propagation.py -k ModalAsymptoticStillBitEqual`, `test_niche_audit_w6_asymptotic.py -k
  test_w6_a2_v2_star`) are in your verification set and must stay green at their current bars -- do not restate them.
* Byte-identity proofs are ARCHIVE-TO-ARCHIVE (`git archive <HEAD-at-launch> lumenairy` vs your tree's modules imported from a child process,
  `lumenairy.__file__` asserted), never against the shared working tree.

## Folded in from WP-B9 (request 4)

`propagators/gbd.py:3373` and `propagators/fga.py:817` build `_jac_candidates = [ray_transfer_jacobian_analytic, ray_transfer_jacobian]` for
`jacobian='auto'` and rely on `NotImplementedError` to fall back.  WP-B9 added aspheric support to the analytic path, so for an ASPHERIC prescription
`'auto'` now returns the exact analytic Jacobian (agreement with FD ~1e-8 relative, analytic side exact) instead of the FD one.  The dispatch needs no
code change; the comment's parenthetical list of uncovered surface kinds must drop "aspheric" (freeforms, biconics and field-frame decenter / tilt still
raise), and one derived test should pin that an aspheric prescription reaches the analytic path through `'auto'`.

## Deliverable

Implementation with the sec. 2 prefix on every new validation; oracles the library did not produce; derived envelopes in a new
`tests/unit/test_audit2609_b7_asymptotic_family.py` (S5; no wall-clock assertions -- operation counts, evaluation counts, peak-memory measurements
reported); stated fail-befores; byte-identity proofs against `git archive <HEAD>^ lumenairy` extracted READ-ONLY into your scratch directory under
`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b7\`
(child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest); wall-clock medians of interleaved runs reported.
Re-record every history document you touch.  Report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7_REPORT.md` (summary table:
item / status / files:lines / tests / oracle / measured before -> after; sections; files touched; tests run; requested changes outside your
ownership; deferred) and `WP-B7_CHANGELOG.md` (5.47.0 release text in the style of `fixes/WP-A4_CHANGELOG.md`, findings Y4 / Y5 / S9 named,
sec. 15.9 named for the uniform path, citations on non-trivial lines, Migration notes for any default that moves -- item 3 may move one; say so).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a4_maslov_gbd.py`, the VERIFY-A4 file(s), `tests/unit/test_audit2609_b1_maslov_input_wavevector.py`,
`tests/unit/test_perf_v4_12_0_asymptotic.py`, `tests/unit/test_niche_audit_w6_asymptotic.py`, `pytest tests/unit -k "maslov or asymptotic or fga or
gbd"` (slow: budget ~20 min), `tests/unit/test_audit2609_a15a_lens_covering_array.py`, `tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py`,
`python validation/run_all.py test_lenses`, `ruff check`, `python scripts/record_history_fingerprints.py --check` (your documents),
`tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only git is fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/elements/lenses_maslov.py`, `lenses_gbd.py`, `_lens_jax.py` (item 9 only), `lumenairy/propagators/asymptotic*.py`, `fga.py`, `gbd.py`, their
  `docs/history/` documents, the new b7 test file, your two report files.  `apply_real_lens_auto`'s routing lives in `fga.py` (`_universal_route`);
  the dispatcher elsewhere is not yours.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
