# LumenAiry release process

This page documents the maintainer's tag-time workflow.  The
canonical CHANGELOG / per-release notes live in
[CHANGELOG.md](../CHANGELOG.md); this document covers ONLY the
release-process tooling and the order in which the steps run.

## Why a release-process page exists

v5.2.5's Part 7 audit ("recursive self-citation drift") observed
that each CHANGELOG block self-cites build-time empirical numbers
(test counts, file counts, line counts), but the CHANGELOG entry
itself is part of the diff that establishes those numbers -- so the
numbers an author types in the block are ALWAYS at-write-time, not
at-ship-time.  v5.3 ships the V17 walker
(`tests/unit/test_v5_3_walker_changelog_self_citation.py`) which
DETECTS the drift; v5.3.2 ships
[`scripts/stamp_changelog.py`](../scripts/stamp_changelog.py) which
FIXES it by stamping the block with empirical values immediately
before tag commit.

## Tools at a glance

| Tool                                           | When                | What                                                                       |
| ---------------------------------------------- | ------------------- | -------------------------------------------------------------------------- |
| `scripts/verify_changelog_closures.py`         | Pre-tag, post-stamp | Content-level CHANGELOG fabrication walker (V12.x companion).              |
| `scripts/stamp_changelog.py`                   | Pre-tag, pre-commit | CHANGELOG ship-time-stamp injection (V17 fix companion).                   |
| `scripts/check_dep_metadata.py`                | Weekly cron + ad-hoc | Optional-dep `requires-python` drift detector.                              |
| `tests/unit/test_v5_3_walker_changelog_self_citation.py` | Always (CI)         | V17 walker -- pins ship-time numbers against the topmost block.            |

## Stamping the CHANGELOG before tag

The canonical sequence:

```
1. Write the CHANGELOG entry for v<NEW> with at-write-time placeholder
   counts.  These are first-pass numbers (e.g. "3848 unit tests pass
   (collected = 3866 = pass + 17 skip + 1 xfail)") -- they will be
   refreshed in step 3.

2. ``git add CHANGELOG.md`` along with every other file in the
   release (lumenairy/*, tests/*, pyproject.toml, ROADMAP.md, etc.).
   The point of staging CHANGELOG.md NOW is so the file-count number
   from ``git diff PREV_TAG..HEAD --name-only`` includes the
   CHANGELOG.md edit itself.

3. ``python scripts/stamp_changelog.py --quick --apply``

   This rewrites the topmost ``## [X.Y.Z]`` block with the
   empirical numbers:
     - test counts (collected total from ``pytest --collect-only``)
     - file count (from ``git diff PREV_TAG..HEAD --name-only``)
     - CHANGELOG.md line count (from ``wc -l CHANGELOG.md``)
   Patterns NOT present in the block are quietly skipped.

4. ``git add CHANGELOG.md`` again to fold the stamp updates into
   the staged release.

5. ``git commit -m "Release v<NEW>"``

6. ``git tag v<NEW>``

7. ``git push origin main --tags``
```

## Invocation patterns

The script supports four common invocations.  Pick based on what
you need:

### Pattern 1 -- dry-run preview (default)

```
python scripts/stamp_changelog.py
```

Prints a unified diff of the proposed stamp without touching
CHANGELOG.md.  Exit codes:

* `0` -- no drift; the block already cites current empirical values.
* `1` -- drift detected; re-run with `--apply` to commit.
* `2` -- skip-clean; no stampable patterns in the block.
* `3` -- input error; CHANGELOG.md not parseable.

### Pattern 2 -- the canonical pre-tag stamp

```
python scripts/stamp_changelog.py --quick --apply
```

Updates the topmost block in place.  `--quick` uses
`pytest --collect-only` (fast, ~5 seconds) and refreshes only the
collected total; the at-write-time skip / xfail counts are held
constant.  Use this 95% of the time.

### Pattern 3 -- full-suite stamp (slow)

```
python scripts/stamp_changelog.py --full --apply
```

Runs the full unit suite (~5 minutes) and refreshes pass / skip /
xfail counts empirically.  Use this when you want every number in
the test-count line to reflect real ship-time behaviour (the
collect-only path can't distinguish skip / xfail).

### Pattern 4 -- back-stamp a past block

```
python scripts/stamp_changelog.py --version 5.2.3 --apply
```

Targets a specific past `## [X.Y.Z]` block instead of the topmost
one.  Useful for retroactive cleanup of historical entries when the
V17 walker flags drift on a past release.  Note: the `git diff
PREV_TAG..HEAD --name-only` step uses HEAD as the right-hand side
even when back-stamping; the file count reflects the diff against
the older block's PREV_TAG, NOT against the target's own tag.

## What the script does NOT do

* **It does not write the initial CHANGELOG entry.**  Step 1 is
  always manual -- the maintainer writes the bullet structure, the
  prose, and the audit-closure list.  The script only refreshes
  numbers.
* **It does not fabricate new pattern sites.**  If the topmost
  block contains no `X unit tests pass (collected = Y = pass + Z
  skip + W xfail)` clause, the script logs "skipped: no pattern to
  update" and moves on.  Adding a pattern is a manual editor step.
* **It does not touch ROADMAP.md.**  ROADMAP also cites empirical
  test counts; refresh those by hand for now (a future v5.x patch
  may extend the script to multi-file stamping).
* **It does not run any git commit / tag / push.**  Step 5 / 6 / 7
  above are pure-`git` steps; the script's role ends at writing
  the stamped CHANGELOG.

## Integration with the V17 walker

The V17 walker
(`tests/unit/test_v5_3_walker_changelog_self_citation.py`) enforces
three numeric self-citation invariants on the topmost block:

* V17.1 -- test-count arithmetic reconciles (`pass + skip + xfail
  == collected`).
* V17.2 -- file count claim within +/- 5 of `git diff
  PREV_TAG..HEAD --name-only | wc -l`.
* V17.3 -- CHANGELOG.md line count claim within +/- 300 lines of
  current `wc -l CHANGELOG.md`.

If `scripts/stamp_changelog.py` ran cleanly before tag, all three
walkers pass at ship time.  If a maintainer SKIPS the stamp step
(e.g. for a docs-only patch with no number changes), the walker
catches the drift at the next CI run.

## Troubleshooting

**`pytest (quick): unavailable`** in the dry-run output means the
script's `subprocess` call to `python -m pytest ... --collect-only`
returned non-zero or no collection line.  Most often this is an
import error in a `tests/unit/` test file; run the same command by
hand to see the traceback.

**`PREV_TAG: unresolvable`** means `git tag -l 'v*'` returned no
tags older than the current target.  This is expected on a brand-
new repo; the file-count stamp is skipped cleanly in that case.

**`drift detected but --apply not set (exit 1)`** is the dry-run
contract surfacing real drift.  Re-run with `--apply` once you've
inspected the proposed diff and accepted it.

**`no drift detected` (exit 0)** is the green path -- the topmost
block's numbers already match empirical state.  No action needed.
