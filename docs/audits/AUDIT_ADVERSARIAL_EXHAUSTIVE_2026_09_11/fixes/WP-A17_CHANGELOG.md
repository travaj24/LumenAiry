# WP-A17 (part 1) — changelog text

Voice: repository CHANGELOG.  Part 2 (the lens family) will add its own entry.

---

### Changed -- propagators: version-history narrative moved out of `carrier.py`, `carrier_field.py` and `fft_infra.py` into `docs/history/`

Audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` finding **P2-4**
(`TESTS-ARCH.md`) / consolidated report **§14 V6** and **§15.7** measured the
carrier and FFT-infrastructure modules at 32-37 % *git history* — blocks shaped
"v5.xx (audit X): pre-fix this did A, which was wrong because B; now it does C".
Those blocks are now in three new documents, reproduced **verbatim** under the
source line they came from, with a table of contents in original-line order:

* `docs/history/carrier.md` (35 blocks, 534 prose lines)
* `docs/history/fft_infra.md` (43 blocks, 372 prose lines)
* `docs/history/carrier_field.md` (5 blocks, 41 prose lines)

Line counts: `carrier.py` 11 602 → 11 275, `fft_infra.py` 2 678 → 2 501,
`carrier_field.py` 1 857 → 1 852.  Measured history share (the audit's own
`tokenize` classifier): `fft_infra.py` 34.8 % → **13.3 %**, `carrier.py`
35.8 % → **30.5 %**, `carrier_field.py` 20.2 % → **14.9 %**.  Measured on the
strict form the finding actually names — an explicit `vN.N (`, `pre-fix`,
`used to`, `formerly`, `previously`, `was wrong` — the reduction is larger:
**−80 %**, **−79 %** and **−38 %** of those lines respectively.

**No behaviour changed, and that is proved rather than asserted.** For every
module the SHA-256 of (a) its AST with all docstrings removed and source
positions ignored and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/STRING
with comments and docstrings dropped are **identical** before and after.

What stayed in the source: every derivation of a live numeric bar
(`docs/TESTING_STANDARDS.md` S5 requires one), every live deprecation and
migration statement, and a condensed why-comment plus a pointer to the history
document wherever the rationale is load-bearing for the current behaviour.  What
moved: superseded derivations of constants that have since been re-measured
(`_FINE_GRID_WORK_ARRAYS` alone carried four, 4 → 16 → 20 → 22 → 24), and
comments correcting earlier *comments*.

### Fixed -- propagators: four comments that stated the opposite of the code beneath them

Audit §15.7 ("a stale comment is worse than none").  Documentation-only; no
executable change:

* `fft_infra.warmup_fft_plans` — the `threads` parameter documented a default of
  `available_cpus()`; the audit-F-32 fix had already changed the call site to
  `FFTW_THREADS` and the doc had not followed.  Getting this wrong makes a
  warm-up build plans under a key the runtime never queries — a silent no-op.
* `carrier._sphere_parab_conversion` — "…while the untapered swap breaks a
  coarse chain", a claim the flag note 30 lines above already re-derives as a
  mis-citation of a measurement that says the opposite.
* `carrier._fourier_upsample_crop` — the dtype-parity promotion's stated premise
  ("numpy's FFT is double-only") lapsed at numpy 2.0; the comment now states
  only what the promotion does today.
* `fft_infra._PYFFTW_DOUBLE_BUFFER` — priced the single-buffer copy at "~1-3 %
  of a large transform"; the byte-cap note 40 lines below measures it at ~65 %
  of the transform at N = 8192.

### Added -- `tests/unit/test_audit2609_a17_history_relocation.py`: a reusable behaviour-free gate on documentation-only edits

19 tests.  Each `docs/history/*.md` header records the module it came from and
the two pre-relocation fingerprints; the test re-computes both from the live
file on every run, so an edit that changes behaviour while claiming to be
history-only fails here.  The registry is discovered from the documents
themselves, so WP-A17 part 2 extends the gate by adding its documents — no
change to the test file.

Two fingerprints rather than one, because they catch different things: the AST
fingerprint sees a statement being deleted, added, reordered or re-spelled but
folds `5` and `0x5` to the same `Constant`; the token fingerprint sees the
literal's *spelling*.  `test_the_fingerprints_are_actually_sensitive` proves
both are live by mutating an in-memory copy of each module three ways —
delete a statement (AST must move), rename an identifier (both must move),
re-spell an integer to the same value (token must move, AST must **not**) —
with every mutation anchored on a real AST node so it cannot land in a comment
or docstring and pass vacuously.

### Known issue (pre-existing, not introduced here)

`tests/unit/test_v5_2_walker_pep562_forwarding.py::test_v14_every_mutable_fft_infra_global_is_in_live_forward_names`
fails: `_PYFFTW_FIRST_FFT_THREAD` and `_PYFFTW_SHARED_BUFFERS_UNSAFE` (added by
WP-A5 / audit K4) are rebound at runtime but are not listed in
`propagation._LIVE_FORWARD_NAMES`, so `propagation.X` returns the stale
import-time snapshot after the multi-thread latch flips.  Confirmed pre-existing
by running the walker against the unmodified baseline file.  The fix — two names
added to `_LIVE_FORWARD_NAMES` in `lumenairy/propagators/propagation.py` — is
outside WP-A17's ownership and is recorded in `WP-A17_REPORT.md` §6.

### Migration

None.  No public API, default, signature or numeric result changed.  Readers
looking for the rationale behind a past change will find it in
`docs/history/<module>.md`, indexed by the source line it used to occupy.
