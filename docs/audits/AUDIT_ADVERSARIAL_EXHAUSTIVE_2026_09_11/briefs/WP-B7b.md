# WP-B7b -- the two behaviour changes WP-B7 measured and escalated (launch after VERIFY-B7 lands; one reviewed commit each)

Read `WP-B7_ASYMPTOTIC_FGA_GBD.md` (same directory) for standards and rules, `fixes/WP-B7_REPORT.md` sections 2, 6 and 8.3-8.4, and `fixes/VERIFY_WP-B7.md`.
HEAD is the newest commit; the pre-change library is `git archive <HEAD-at-launch> lumenairy` (child process, cwd + PYTHONPATH = the archive,
`lumenairy.__file__` asserted; never through pytest).  You own `lumenairy/propagators/fga.py`, `lumenairy/elements/_lens_traced_uniform.py`, the
five test files item 1 names, `tests/unit/test_fga_h4_h5.py`, a new `tests/unit/test_audit2609_b7b_caustic_routing.py`, their `docs/history/` documents, and your two report files.

1. **`_universal_route`'s caustic branch routes a SINGLE-VALUED field at a caustic to `'phase_screen'`, not `'fga'`** (WP-B7 sec. 2 / 8.3).  Re-derive
   first, on your own fixture, WP-B7's measurement (brute-force Rayleigh-Sommerfeld from an exact conic raytrace, converged on the pupil sampling):
   fidelity 0.32-0.38 for `'fga'` at its best sampling against 0.9965 for `'phase_screen'` on the f = 1.2 mm NA 0.145 singlet at its focus, and
   `'phase_screen'` closer at every NA 0.039-0.192.  If it holds, ship the diff WP-B7 wrote (the multi-valued branch above it is unchanged -- that
   is what `'fga'` uniquely provides), restate the `== 'fga'` expectations for single-valued caustic rows in `test_audit2609_a4_fga_s10.py`,
   `test_fga.py`, `test_g1_gate_generality.py`, `test_niche_audit_w9_dispatch2.py`, `test_niche_p8_capstone.py` (and `test_niche_p7_seidel_gate.py`
   if it pins the route) with the measurement in each docstring, leave the multi-valued rows alone, and write the Migration note (which calls
   change their answer, by how much on the measured fixture, and how to get the old route back -- `method='fga'` explicitly).  If it does not
   hold, say so with the table and ship nothing.
2. **`caustic='uniform'` at a real fold** (WP-B7 sec. 6.2 / 8.4): at the marginal focus of an f/1.92 singlet `apply_real_lens_traced(caustic=
   'uniform', amplitude_model='ray_density')` scores 0.0030 against a brute-force RS oracle converged to 4.6e-8, `'multibranch'` 0.0018, `'wave'`
   0.1110, the Maslov evaluator 0.5921; the CFU kernel itself is validated to 1e-14 against exact cubic-phase integrals, so the failure is in the
   FITTING of the control parameters to the traced branches.  Reproduce on your own fold fixture (a different singlet), localise the failing step
   (branch pairing, the `zeta` / amplitude fits, the branch-cut of the fold), fix it if the fix is bounded and bit-identical away from folds, and
   otherwise document the measured envelope on the function and warn where the fit residual says the parameters are not carried.  Either way the
   report carries the ladder and the oracle floor.

3. **FGA's analytic-Jacobian whitelist** (WP-B7 sec. 8.2): `_pick_ray_transfer` gates on `_is_all_conic`, which still excludes `aspheric_coeffs`
   although WP-B9 gave `ray_transfer_jacobian_analytic` even-aspheric support -- so an aspheric prescription traces the 9-ray FD bundle whatever
   `exact_jacobian` says, `exact_jacobian=True` included (silently ignored); and the predicate does NOT check `field_decenter` / `field_tilt` /
   `field_sag_callable`, which the analytic primitive rejects, so a field-decentred conic reaches the analytic path and raises `NotImplementedError`
   at call time instead of falling back (latent bug).  Ship WP-B7's exact edit (`_analytic_jacobian_applies`, with `_is_all_conic` kept as an
   alias) and restate the three assertions in `tests/unit/test_fga_h4_h5.py` as written there; measure the swap on an A4 singlet (analytic vs FD
   exit heights agree to the FD truncation floor; trace count 9N -> N) and write the Migration note -- it is a default move for aspheric
   prescriptions in FGA (auto switches from the FD to the exact analytic Jacobian).  Gate the field-decentred conic fallback with a fail-before.

Deliverables: `fixes/WP-B7b_REPORT.md` and `WP-B7b_CHANGELOG.md` in the usual shape; derived S5 tests; every history document re-recorded in the same
change; the verification set = WP-B7's plus the five test files plus `-k "fga or caustic or uniform or traced"`, `validation/run_all.py test_lenses
test_propagation`, ruff, fingerprints, a17 lint.  Rules: thread env vars, one process at a time, no git write commands, do not kill processes,
comments say what the code does now and why.  Finish with the report's full text and the exact commit file list.
