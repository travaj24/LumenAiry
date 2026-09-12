# Legacy audit probe scratch (moved out of `scripts/`, 2026-09-12)

These eleven files are one-off reproduction probes from past audit rounds
(`_d5_*` — the `newton_fit='spline'` / decentred-fit round; `_g8_*` — the
PMM-2D G8 round).  They were committed into `scripts/` beside the four real
repository tools and, being unqualified `.py` files in a tools directory, read
as maintained utilities.  They are not: each reproduces one finding against one
build, several import each other by bare module name (`import _d5_byteid`), and
none is referenced by any test, workflow or document.

The audit of 2026-09-11 flagged this (`TESTS-ARCH.md` P3-4).  They are moved
here rather than deleted so the reproductions stay recoverable, and this
directory matches `validation/probe_*`, which `MANIFEST.in` prunes — so they no
longer ship in the sdist either.

`scripts/` now holds only the four maintained tools:

| tool | used by |
|---|---|
| `check_dep_metadata.py` | `dep-drift.yml` |
| `check_source_line_citations.py` | `publish.yml` (V18 walker) |
| `verify_changelog_closures.py` | `publish.yml` (V16 walker) |
| `stamp_changelog.py` | release mechanics |

To run one of these probes, run it from inside this directory (they resolve
their neighbours by `sys.path` insertion of their own folder).
