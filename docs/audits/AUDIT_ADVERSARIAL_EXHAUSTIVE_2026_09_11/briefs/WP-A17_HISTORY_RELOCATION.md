# WP-A17 — Relocate version-history narrative out of the source (runs AFTER every physics WP)

Read first: `COMMON.md`, then `TESTS-ARCH.md` [P2-4] and report §14 V6 / §15.7 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`.

## Files you own
`lumenairy/elements/_lens_traced.py`, `lumenairy/propagators/carrier.py`, `lumenairy/elements/_lens_real.py`,
`lumenairy/elements/_lens_imap.py`, `lumenairy/propagators/fft_infra.py`, `lumenairy/elements/lenses_maslov.py`
(comments and docstrings ONLY — zero code change), the new `docs/history/<module>.md` files, and the test files that
assert on those modules' docstrings / source text (find them: `grep -rln "getsource\|__doc__" tests/unit`).

## Deliverable
Move every "v5.xx (audit …): pre-fix this did A, which was wrong because B; now it does C" block — comments and the
history-heavy parts of docstrings — verbatim into `docs/history/<module>.md` with the source line it came from, leaving
in the code a one-line pointer where the rationale is load-bearing for the CURRENT behaviour (keep genuine why-comments
that explain what the code does now). Targets from the audit's token census: `_lens_traced.py` 37.6 % history (~4 970
lines), `carrier.py` 36.9 % (~3 900), `_lens_real.py` 23.3 % (~1 600), `_lens_imap.py` 30.3 %, `fft_infra.py` 34.4 %,
`lenses_maslov.py` 16.8 %.

## Verification (this must be provably behaviour-free)
- For each module: `ast.dump` of the module with all docstring nodes removed must be IDENTICAL before and after
  (write the checker; include it as a test that compares against the committed pre-change AST hash stored in the
  history doc). Byte-identical outputs on the module's existing test files.
- Run every test that reads these modules' docstrings or source (`inspect.getsource`, `__doc__`); where a test asserted
  on history prose, retire that assertion (it is a doc-prose test, TESTING_STANDARDS) and say which.
- Report the line counts before/after per module.
