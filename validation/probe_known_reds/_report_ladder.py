"""Rewrite report section 7.3 with the arms actually measured."""
import io
import sys

P = ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/'
     'WP-B14_KNOWN_REDS_REPORT.md')

OLD = """### 7.3 The kernel ladder

`OPENBLAS_CORETYPE` x threads, confirmed per arm with `threadpoolctl`.  T3-1
before the fix (the red), and after:

| arm | T3-1 before | T3-1 after |
|---|---|---|
| HASWELL 1 | pass | pass |
| HASWELL 4 | pass | pass |
| NEHALEM 1 | pass | pass |
| NEHALEM 4 | pass | pass |
| KATMAI 1 | pass | pass |
| KATMAI 4 | pass | pass |
| SANDYBRIDGE 1 | pass | pass |
| **SANDYBRIDGE 4** | **FAIL** | **skip, carrying the two measured cells and the screened census** |
"""

NEW = """### 7.3 The kernel ladder

`OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x threads {1, 4},
the kernel confirmed per arm with `threadpoolctl`.  ZEN aliases Haswell on this
box and SKYLAKEX crashes, so neither is used.

**T3-1**, the one test whose OUTCOME moved with the build.  The "before" column
is the complete eight-arm ladder run against the unmodified test
(`validation/probe_known_reds/ladder_t31/`); the "after" column is the same
selection re-run against the fix:

| arm | T3-1 before | T3-1 after | after-run log |
|---|---|---|---|
| HASWELL 1 | pass | pass | `ladder_final/T31_HASWELL_t1.log` |
| HASWELL 4 | pass | pass | `ladder_final/T31_HASWELL_t4.log` |
| NEHALEM 1 | pass | pass | `ladder_final/T31_NEHALEM_t1.log` |
| NEHALEM 4 | pass | pass | `ladder_final/T31_NEHALEM_t4.log` |
| KATMAI 1 | pass | pass | `ladder_final/T31_KATMAI_t1.log` |
| KATMAI 4 | pass | **not re-run** -- see below | -- |
| SANDYBRIDGE 1 | pass | pass | `ladder_t31_after/` |
| **SANDYBRIDGE 4** | **FAIL** | **skip, carrying the two measured cells and the screened census** | `ladder_t31_after/` |

Seven of the eight arms were re-run after the fix, including **the arm that was
red**, which is the one the fix is about.  KATMAI at four threads was not
reached: the ladder shell wedged after its fifth arm (see section 8), and it is
the only arm of the eight whose post-fix reading is missing.  It passed before
the fix, and the change to that test can only turn a hard failure into a skip
or leave the outcome alone, so nothing it could report would contradict the
table -- but it is not measured and is not claimed.

**c7 / c8 / the GBD budget** (33 ids), re-run after the fix:

| arm | outcome |
|---|---|
| HASWELL 1 | 33 passed |
| HASWELL 4 | 33 passed |
| NEHALEM 1 | 33 passed |
| NEHALEM 4 | 33 passed |
| KATMAI 1 | 33 passed |
| KATMAI 4, SANDYBRIDGE 1, SANDYBRIDGE 4 | not reached (section 8) |

Five arms plus the WSL (Linux, py3.12, numpy 2.4.6) run of 7.1, which is a
different libm and a different BLAS and is the arm that matters most for RED 1,
since CI is Linux.  The restored order-10 stimulus manufactures the lobe there
too.

**What the ladder was NOT used for.**  `test_v4_16_0_agent_d_validity_ranges`,
`test_audit_w4_glass_registry_meshgrid` and `test_audit2609_b11_hygiene` are
registry-state, warn-once and AST checks with no floating-point kernel in them,
so a BLAS ladder measures nothing there; they were run on the default arm and
under WSL instead.  The GBD budget test reads `tracemalloc` allocation counts,
which are likewise kernel-independent by construction -- it is on the ladder
above only because it shares a selection with c7 / c8, and it reads the same on
every arm.
"""


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    if src.count(OLD) != 1:
        print('anchor not unique:', src.count(OLD))
        return 1
    src = src.replace(OLD, NEW)
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('7.3 rewritten')
    return 0


if __name__ == '__main__':
    sys.exit(main())
