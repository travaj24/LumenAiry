# WP-B1 (Wave 4, first) -- Maslov S6 proper: the saddle must follow the INPUT field's local wavevector

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`, HEAD 81d5b586
(= release 5.46.0, Wave 3 closed).  Two other Wave-4 engineers are working concurrently on `lumenairy/elements/_lens_real.py`
(WP-B2) and on `lumenairy/propagators/hf.py` / `hfpi.py` (WP-B3); never touch those files.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 (error prefix `f"{fn_name}: ..."`), sec. 7 (signs), the
comment rule in `CONTRIBUTING.md` ("Modules with a history document": a code change to `lumenairy/elements/lenses_maslov.py` or any other
module with a `docs/history/<dotted.module>.md` MUST be re-recorded in the same change with
`python scripts/record_history_fingerprints.py <module path> --reason "..."`).  Source comments describe what the code does NOW and why;
no "v5.xx (audit ...): pre-fix" narrative in the source (the history-lint ratchet `tests/unit/test_audit2609_a17_history_lint.py` fails on it).

Then, in this order: the audit finding S6 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`;
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A4_REPORT.md` (section 2 S6, and section 6 item 1 -- the deferred DESIGN
you are implementing, reproduced below), `WP-A4_CHANGELOG.md` (the S6 warning that ships today), `VERIFY_WP-A4.md` (its S6 verdict and
the dimensionless LG merit follow-up), and `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s6_*` (the shipped pins: the warning, and
the symplectic identity `dOPD/dv2 = -n1 (v1 . ds1/dv2)` closing to 5.8e-7).

## The defect (from WP-A4, measured)

`_maslov_newton_saddle_cpu` and its GPU twin solve `grad_v2 OPD = 0`.  By the symplectic identity that is the `v1 = 0` launch ray at
EVERY pixel -- the on-axis collimated ray -- for every input.  The stationary point of the TOTAL integrand phase is
`grad_v2 [arg E_in(s1(v2)) + k . OPD] = 0`, i.e. `(v1_in - v1) . ds1/dv2 = 0`: the ray whose LAUNCH direction matches the input field's
local wavevector.  Measured: at the 2 % smallest `|grad_v2 OPD|` the traced rays have mean `|v1| = 6.93e-03` on a chart of NA 0.05
whose all-ray mean is 3.79e-02.  Today a non-collimated input gets a `RuntimeWarning` naming the measured input NA and the two correct
alternatives; the two asymptotic evaluators (`apply_real_lens_maslov` and the JAX twin) are still wrong for it.

## The design (WP-A4 section 6 item 1; implement it)

Fit the input's local wavevector `(k1x, k1y)(s1)` as two more columns of the SAME Chebyshev design matrix the chart already builds: the
trace carries `v1x, v1y` per ray and `_solve_fit` already takes a stacked RHS (it solves OPD, s1x, s1y together), so the marginal cost is
one wider RHS and no extra factorisation.  Then add `k1 . ds1/dv2` to the Newton gradient and `d(k1 . ds1/dv2)/dv2` to the Hessian in
`_maslov_newton_saddle_cpu` / `_maslov_newton_saddle_xp`, and the same term to `opd_star` in `_integrate_stationary_phase` and to `opd_v`
in `_integrate_local_quadrature` (the two sites that already carry `lin_v3` / `lin_v4`, so the threading pattern exists).  `E_in`'s own
amplitude stays where it is -- only its PHASE joins the exponent.  Where the input phase is not smooth enough to differentiate (speckle,
a hard-edged aperture), say so with a measurement and define the fallback (the shipped warning path is the obvious one; state the
criterion).

## Deliverable

1. **Implement S6 proper** on the CPU path and the `xp` (GPU/JAX) twin, keeping the two bit-identical where they were bit-identical
   before (`tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py` and the A4 parity pins say where).  The collimated case must be
   BYTE-IDENTICAL to today's output (`k1 = 0` everywhere): prove it with `np.array_equal` on the A4 fixtures and on
   `tests/unit/test_audit2609_a15a_lens_covering_array.py`'s Maslov cells.
2. **Oracle.**  A tilted or converging Gaussian input through the A4 conic singlet, scored against an oracle the library did not produce:
   the exact ray trace of the INPUT's own rays (launch direction = the input's local wavevector) plus a brute-force stationary-phase or
   direct Kirchhoff sum (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/` and `tests/unit/test_niche_d6_exact_tilted_leg.py`
   carry inline Kirchhoff oracles you may copy the METHOD of, not the code).  Report field fidelity, focal centroid and EE(2 um) before
   (today's evaluator) and after, for at least tilt = 0 / 0.5 / 1 x the input NA and one converging input.  Then retire or restate the
   S6 warning: it fires today for the case you are now computing correctly; keep a warning only for the fallback regime you defined.
3. **Pin it** in a new `tests/unit/test_audit2609_b1_maslov_input_wavevector.py`: derived two-sided envelopes (S5: the oracle, its floor,
   the defect scale -- the `|v1|` census above is one), the byte-identity of the collimated case, the CPU/xp parity, a fail-before
   (say how you showed it: the pre-change evaluator's centroid error on the tilted input), and the fallback criterion.
4. **Re-record** every history document you touch, in the same change, one-line reason each.
5. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B1_REPORT.md` (summary table: finding / status /
   files:lines / tests / oracle / measured before -> after; per-item sections; files touched; tests run with counts and durations;
   requested changes outside your ownership; deferred) and `WP-B1_CHANGELOG.md` (release text for 5.47.0 in the `### Fixed -- ...`
   style of `fixes/WP-A4_CHANGELOG.md`; cite `lumenairy/elements/lenses_maslov.py:N` only on non-trivial lines; name the finding S6).
   Include a Migration note if the default output for non-collimated inputs changes (it will: say what a caller who wants today's
   numbers passes).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a4_maslov_gbd.py`, the VERIFY-A4 test file(s) (`a4` in `tests/unit/`), `tests/unit/test_audit2609_a15a_lens_covering_array.py -k maslov`,
`pytest tests/unit -k "maslov or asymptotic"`, `tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py`, `python validation/run_all.py test_lenses`,
`ruff check`, `python scripts/record_history_fingerprints.py --check`, `tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
* NO git write commands of any kind (no add/commit/stash/checkout/restore/reset); read-only `git show` / `git archive` / `git log` are
  fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/elements/lenses_maslov.py` and the asymptotic modules it calls for the saddle (`lumenairy/propagators/asymptotic*.py`)
  if the fix must reach them, their `docs/history/` documents, the new b1 test file, your two report files.  Anything else goes under
  "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why; measured derivations of live constants stay, dated, beside them.
* Finish with the report's full text as your final message.
