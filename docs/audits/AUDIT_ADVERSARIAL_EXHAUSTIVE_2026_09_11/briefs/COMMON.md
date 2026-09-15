# Common brief for every fix work-package (WP) agent — Lumenairy audit remediation

You are one of several Opus engineers implementing the findings of the 2026-09-11 adversarial audit of the
Lumenairy optics library. Work carefully and exhaustively; verify everything by MEASUREMENT.

## Where things are
- Repository: `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy` (git branch
  `audit-fixes-2026-09`, already checked out — do not switch branches).
- The audit report (consolidated): `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Finding rows are
  identified by letter+number (L12, K16, G5, …); each row gives Where / Evidence / Fix.
- The full auditor reports: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/<PARTITION>.md` — read the
  partition report(s) named in your WP file IN FULL before changing anything; they carry the derivations,
  the exact code sites and the measured numbers.
- Reproduction scripts: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/<PARTITION>/*.py` and
  `repro/orch/*.py`. They insert the repo root into `sys.path` by absolute path — they run as-is on this
  machine. Re-run the ones for your findings BEFORE (to confirm the defect on the current HEAD) and AFTER
  (to confirm the fix) and quote the numbers.
- Line numbers in the report refer to commit `a1ff1e6e`; only `elements/bor/*`, `elements/eme/*` changed
  since — everything else is unchanged, but always re-locate by content.
- Repo process docs you must follow: `CONVENTIONS.md` (§2 error prefix `f"{fn_name}: ..."`, §7 sign
  conventions, §9 sentinel, §10 optional deps), `docs/TESTING_STANDARDS.md` (S1–S5: no wall-clock or
  speedup assertions, no `pytest.skip` on resource preconditions, no per-build bars, every numeric bar carries
  its derivation and measured values in a comment, independent oracles with derived bounds), `CONTRIBUTING.md`
  (why-comments are welcome; no silent default changes).

## Ownership and coordination rules (hard rules — other agents are editing other files right now)
1. Edit ONLY the files listed under "Files you own" in your WP file, plus NEW test files you create. If a fix
   genuinely needs a change elsewhere, do not make it: describe the exact change in your report under
   "Requested changes outside my ownership" and, where possible, implement your side so that it degrades
   gracefully until the other change lands.
2. Do NOT edit `CHANGELOG.md`, `README.md`, `Migration-Guide.md`, `CONVENTIONS.md`, `pyproject.toml`,
   `lumenairy/__init__.py` (unless your WP file explicitly assigns one of them). Instead write your changelog
   text to `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/<WP>_CHANGELOG.md` in the repository's
   CHANGELOG voice (`### Fixed -- <area>: <headline>` / `### Changed` / `### Added` / `### Performance`,
   naming the finding IDs, the files, the tests added, the measured before/after numbers and any migration
   note). The orchestrator assembles the real CHANGELOG entry from these files.
3. No git WRITE commands of any kind (no add/commit/stash/checkout/reset/clean/branch). Read-only git
   (`git diff`, `git log`, `git blame`, `git status`) is fine. The orchestrator commits per WP.
4. Never kill processes you did not start (no blanket `taskkill`/`pkill` on python). Kill only your own PIDs.
5. Every python invocation: set `OPENBLAS_NUM_THREADS=1` (this workstation's OpenBLAS is ~400× slower
   unpinned). Do NOT run the full test suite (3.65 h): run the test FILES that cover the modules you touched
   plus your new tests (`python -m pytest tests/unit/<file>.py -q -x --no-header -p no:cacheprovider`), and the
   `validation/test_*.py` topic file(s) for your area if one exists (`python validation/run_all.py <name>`).
   The machine is shared with several other agents — be economical, avoid > 2 GB experiments, and prefer
   background runs with timeouts for anything over a minute.
6. Test files: strengthen EXISTING tests in place when one already covers the behaviour (the audit's V1/V2
   findings are exactly that pattern); put genuinely new coverage in new files named
   `tests/unit/test_audit2609_<wp>_<topic>.py`. Do not edit a test file that is not about your modules — if
   another agent's module owns it, ask in your report.
7. Do not add version-history narrative comments ("v5.xx (audit …): pre-fix this did …") to the source. Write
   comments that explain what the code does now and why (why-comments are valued here); the history goes into
   your changelog text.
8. Keep behaviour backward compatible unless the finding says the default is WRONG. When a default or a
   convention changes, say so explicitly (with the measured before/after) in the changelog text and provide a
   migration note. Removed or no-op'd behaviour must warn through the existing `lumenairy/_deprecation.py`
   patterns (read that module first).
9. If a NumPy path you fix has a JAX or CuPy twin, fix the twin too or make it refuse loudly; test JAX parity
   where jax is importable (enable x64). CuPy is not installed — desk-check and say so.
10. New public API needs a docstring in the surrounding style; new kwargs need validation with the §2 prefix.

## Verification bar (this is the core of the job)
- Re-measure, never read: a finding is fixed when the audit's repro script (or an equivalent independent
  oracle) shows the corrected number, AND a regression test pins the property build-free (TESTING_STANDARDS).
- Write the regression test so that it FAILS on the pre-fix code (state how you confirmed that — e.g. by
  stashing nothing, but by temporarily reverting your change in-process, or by asserting the pre-fix number
  from the repro output) and passes after.
- Every numeric bar in a test needs a derivation comment: what the oracle is, its error floor, the measured
  value, the decades of gap on both sides.
- Run the existing tests for the touched modules; if any existing test asserts the OLD (wrong) behaviour,
  fix the test and say so (this audit found tests that pin defects — see §14 of the report).
- If you cannot reproduce a finding on the current HEAD, say so with the measurement and leave the code
  alone for that item. If a fix is too large or too risky for this pass, implement the guard/warning/refusal
  path that stops silent-wrong output, and record the remainder as deferred with a concrete design.
- Performance items: measure with `time.perf_counter` medians of ≥ 5 interleaved runs AND tracemalloc peaks;
  require bit-identical or documented-tolerance output vs the previous implementation; never assert timings
  in tests.

## Report format (return this as your final message; also save it to
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/<WP>_REPORT.md`)
1. Summary table: finding ID | status (fixed / partially fixed / deferred / not-reproducible) | files:lines |
   tests (path::name) | oracle | measured before → after.
2. Per finding: what was wrong, what you changed and why, how you verified (numbers), residual risk.
3. Files touched (complete list) and new files.
4. Tests run: exact commands, pass/fail counts, durations; any pre-existing failures you found (with your
   judgement whether they are related).
5. Requested changes outside your ownership (exact file, exact change, why).
6. Deferred items with a concrete design and effort estimate.
7. Path to your `<WP>_CHANGELOG.md`.
