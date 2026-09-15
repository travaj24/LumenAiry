# VERIFY-<WP> — adversarial re-verification of a fix work package

You are an independent verifier. You did not write the fixes; your job is to try to break them and to prove, by
re-measurement, that each finding is closed without regressing anything nearby. Assume the fix report may be wrong in
its numbers, its scope, or its tests.

## Inputs
- `COMMON.md` (rules — the same ownership discipline applies: you may ADD tests and fix defects you find in the WP's own
  files; you may not touch other files; no git writes; `OPENBLAS_NUM_THREADS=1`; never kill others' processes).
- The WP's brief, its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/<WP>_REPORT.md`, its changelog
  text, the audit rows and partition reports it cites, and its diff (`git diff <base>..HEAD -- <files>`; the orchestrator
  gives you the base commit).

## Method (for EVERY finding the WP claims fixed)
1. Re-run the audit's repro script(s) for that finding on the current code and quote the numbers; compare with the
   report's claimed after-numbers. Discrepancies are findings.
2. Build at least one NEW independent check per P0/P1 finding on a fixture the WP did not use (different radius / NA /
   wavelength / grid / polarization / orientation) — an oracle the library did not produce.
3. Read the new/changed tests against `docs/TESTING_STANDARDS.md`: is the bar derived, build-free, two-sided; would it
   fail on the pre-fix code (revert the fix in-process or reason from the pre-fix repro number); does it test the
   property or the implementation?
4. Look for collateral damage: run the existing test files of the touched modules and their nearest consumers; probe
   the "checked and found correct" list of the partition report to make sure it still holds (bit-identity where the WP
   claims it).
5. Try to break it: adversarial inputs (real dtype, float32, odd N, anamorphic dy≠dx, non-C-contiguous arrays, negative
   indices, JAX/x64 off, edge geometry such as plano-rear vs curved-rear, grazing rays, n_exit ≠ 1).

## Report (save to `fixes/VERIFY_<WP>.md` and return it)
Per finding: VERIFIED / VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your measurements; the tests you added; any
defect you fixed in the WP's files (with its own verification); open items for the orchestrator with severity.
