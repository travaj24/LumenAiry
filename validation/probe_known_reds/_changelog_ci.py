"""Add the CI-matrix entries to the existing ## [Unreleased] block."""
import io
import sys

P = 'CHANGELOG.md'
ANCHOR = "## [5.47.0] — 2026-09-14\n"

BLOCK = '''### Fixed -- the CI matrix (run 34914295323 on the 5.47.0 commit: 34 jobs, 30 red)

- **The whole Python 3.10 lane was aborting at collection and testing nothing.**
  `tests/unit/test_audit2609_a15a_packaging.py` imported `tomllib` unconditionally;
  it is 3.11+ stdlib, CI installs the `tomli` backport for exactly this reason, and
  pytest turns any collection error into `Interrupted` -- so each of the five 3.10
  shards ran **0 of its ~2 910 selected ids** and reported `18 skipped, 1 error`.
  The import takes the standard fallback, and the no-parser case is premise-gated
  inside the one helper that needs a parser rather than skipped at module scope:
  only 4 of the file's 11 gates read `pyproject.toml`, and a module-level skip would
  have silently dropped the other 7.  Proved on three arms by making `tomllib`
  unimportable in a child process -- with the backport present 11 passed, with
  neither 7 passed / 4 skipped and no collection error.

- **The history token-stream gate was interpreter-dependent (PEP 701).**  49
  distinct ids of `test_the_module_token_stream_is_unchanged_since_the_history_move`
  failed on every 3.11 shard and none on 3.12 / 3.13 / 3.14, while the sibling AST
  gate was 123/123 green on 3.11 -- so the module sources were identical on every
  arm and what moved with the interpreter was the digest's own definition.  CPython
  3.12 tokenises `f"a{b}c"` as seven records (`FSTRING_START` / `FSTRING_MIDDLE` /
  ... / `FSTRING_END`) where 3.11 emits one `STRING`; **110 of the 123 registered
  modules contain an f-string and 13 do not**, and the 7 ids that passed on 3.11 are
  exactly the 13.  The digest now collapses each f-string run to one record carrying
  the literal's exact SOURCE SLICE, which is version-stable and strictly FINER than
  the token run (which normalises `{{` to `{` and says nothing about spacing inside
  a replacement field), so the literal-spelling claim the gate exists for is kept
  and is newly proved by six falsifiability cases.  Established without a 3.11
  interpreter on the box: the new scheme reproduces, bit for bit on 3.12, 3.13 and
  3.14, **49 of 49** of the digests CPython 3.11 itself computed on the runners.
  110 of 123 history documents re-recorded by the recorder; `ast_sha256` moved on
  0 of 123, which is the arithmetic proof that no code changed.

- **The `a8` glass tests asserted a package-dependent fact unconditionally.**  CI
  deliberately omits the glass extra.  The suspected library contract violation is
  not one -- `lumenairy/glass.py` is unchanged: of 49 tuple-registered glasses
  exactly one (`SILICON`) has no bundled Sellmeier row, and its `ImportError` comes
  from the REAL index, not the extinction path, so "falling back to kappa = 0" there
  would mean fabricating a real index -- the silent-wrong shape the same audit
  finding exists to kill.  Each test now asserts the documented no-package fact as a
  two-sided partition (the exempt set must EQUAL the independently computed "tuple
  entry with no bundled row on this install" set), never `pytest.importorskip`,
  which `docs/TESTING_STANDARDS.md` rule 4 forbids on a resource check.  64 passed
  and 0 skipped with the package present AND with it blocked by a fixture.

- **Six platform / kernel bit pins, root-caused before any bar moved -- and in
  three of them the cause was not rounding.**  The off-plane `fff_nv` fixture's
  exact mirror precondition failed on Linux because `sin(pi/4)` and `cos(pi/4)`
  differ by one ULP under glibc and not under MSVC; it is repaired in the
  CONSTRUCTION (the fixture is averaged with its own mirror, which is exact on every
  IEEE-754 platform) so the `array_equal` pin is kept.  The BOR pencil's 1e-12
  residual bar was one arm's reading and is re-derived from LAPACK's backward-error
  result for a Cholesky-reduced symmetric-definite pencil, with an 11x kernel swing
  measured on one machine and a WSL control that isolates it to the kernel rather
  than the OS.  The carrier transfer-function "bit-identical" pin was comparing two
  different ASSOCIATIONS of the same three factors -- `sum(xm * (wgt*slope))` against
  `sum((wgt*xm) * slope)` -- so byte equality was never an invariant of the pair;
  the projection correction itself is 3e-21 to 2e-19 ULP and cannot move a bit.  Two
  peak-array budgets moved because NumPy's temporary-elision rewrite is a BUILD
  property (it needs `backtrace()`), which does not follow the operating system --
  measured on the running arm with a companion arm that puts elision out of reach,
  so "no elision here" can never come from an instrument that measured nothing.  A
  lens-memory pre-flight bar read 0.88 against a fail-safe 1.0 because the test
  warmed on a grid BELOW the deferred-import threshold, putting ~11.2 MB of one-time
  imports inside the measured region; warming above it makes all four cells
  byte-identical on both arms with both bars unchanged.  And an optimiser merit
  whose aberration-free reference is COLLAPSED swings 2.7x across BLAS kernels, so
  its scale is not a measurement: the physics pins keep their bars, the scale gets a
  derived order-of-magnitude band, and the collapse becomes a positive pin.

### Changed -- CI configuration

- `--maxfail` per unit shard 10 -> **50**.  At 10, every py3.11 shard of the run
  stopped after executing 6.4-7.9 % of its selection, so more than 92 % of the lane
  never ran and the census of what was broken on 3.11 was unknowable from the
  artefacts: 49 of the 50 failures were one 123-case gate whose true size, measured
  independently, is 110.  The early-abort property the lower cap was bought for
  survives, because a job's wall clock is bounded by its OWN budget and a
  catastrophically broken job reaches 50 in its first minutes exactly as it reaches
  10; the cap never fired on the healthy lanes (3 / 2 / 4 / 2 / 4 / 8 / 1 failures).
- The slow lane is **8 shards** (was 5), with the step cap 30 -> 45 minutes and the
  job cap 35 -> 50.  All five slow shards timed out on the run, and it was
  predictable before it started: the lane now totals 9 630.0 s against the 5 914.9 s
  last recorded (+62.8 %, from later waves moving files over the two-minute bar
  without re-running the sum), which `least_duration` balances to 1 926.0 s per
  shard against an 1 800 s cap.  `.test_durations` was NOT the problem and is
  untouched -- coverage of the slow selection measured 935 of 935 ids, zero gap.
  Both changes are needed: at 8 splits and the runners' worst measured scale factor
  the per-shard cost is ~1 782 s, which is a 1 % margin under the old cap and 34 %
  headroom under the new one.

### Fixed -- typing

- `lumenairy.propagators.asymptotic_modes.decompose_lg` normalised its `only`
  argument with `tuple(tuple(k) for k in only)`, and `tuple(iterable)` types as
  `tuple[_T_co, ...]` -- so the constructor widened `tuple[int, int]` to
  `tuple[int, ...]` at the call site and destroyed the length information the callee
  genuinely requires (it unpacks `for (p, ell) in only`).  Unpacking by name instead
  keeps the normalisation, keeps the loud failure one frame earlier, and clears the
  one error `mypy --strict` reported; neither the whitelist nor an `ignore` was
  touched.

'''


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\n', '\n').replace('\r\n', '\n')
    if '### Fixed -- the CI matrix' in src:
        print('already present')
        return 1
    if src.count(ANCHOR) != 1:
        print('anchor not unique:', src.count(ANCHOR))
        return 1
    src = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('CI entries added to [Unreleased]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
