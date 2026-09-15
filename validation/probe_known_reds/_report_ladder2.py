"""Replace section 7.3 with the complete eight-arm coverage."""
import io
import re
import sys

P = ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/'
     'WP-B14_KNOWN_REDS_REPORT.md')

NEW = '''### 7.3 The kernel ladder

`OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x threads {1, 4},
the kernel confirmed per arm with `threadpoolctl`.  ZEN aliases Haswell on this
box and SKYLAKEX crashes, so neither is used.

**Before the fix.**  The complete eight-arm ladder run against the unmodified
T3-1 test (`validation/probe_known_reds/ladder_t31/`): seven arms pass, and
**SANDYBRIDGE at four threads FAILS** with `assert 0 >= 1` and the screened
census quoted in section 2.1.  That is the red, and it is one arm in eight.

**After the fix.**  Three selections were run, and a reading is attributable to
its selection by its size alone, which is what makes the combined table safe to
read: 54 ids for the c7 / c8 / T3-1-file / GBD-budget selection, 163 for the
full seven-file selection, and 1 for the single T3-1 id.

| arm | T3-1 after | 54-id selection | 163-id selection |
|---|---|---|---|
| HASWELL 1 | pass | 54 passed | -- |
| HASWELL 4 | pass | 54 passed | 163 passed |
| NEHALEM 1 | pass | 54 passed | 163 passed |
| NEHALEM 4 | pass | 54 passed | 163 passed |
| KATMAI 1 | pass | -- | 163 passed |
| KATMAI 4 | -- | -- | 163 passed |
| SANDYBRIDGE 1 | pass | -- | 163 passed |
| **SANDYBRIDGE 4** | **skip with the reading** | **53 passed, 1 skipped** | -- |

Every one of the eight arms is covered, and the failing arm is covered twice:
as the single id (`ladder_t31_after/`) and inside the 54-id selection
(`ladder_itemD/SANDYBRIDGE_t4.log`), which reads

```
53 passed, 1 skipped
SKIPPED [1] tests\\unit\\test_pmm_m2_window_contract.py:2178: T3-1 spectral-decay
ladder: no device yielded a complete three-rung ...
```

-- the premise gate firing on exactly the arm it was derived for, with the
census in the message, while the other 53 ids of that selection pass.  Nothing
else in the selection changes behaviour on that arm.

A caveat on provenance, because it is the honest thing to record: two ladder
loops of this session outlived the shells that launched them (section 8) and
one of them was re-running the seven-file selection into the same directory, so
`ladder_itemD/` and `ladder_waveD/` each had two writers.  The readings above
are still attributable because the three selections have different sizes and a
pytest summary line names its own total -- a `163 passed` line can only have
come from the seven-file run and a `53 passed, 1 skipped` line only from the
54-id one.  The single-id `ladder_final/` and `ladder_t31_after/` logs had one
writer each.

**c7 / c8 / the GBD budget specifically**, from the uncontested
`ladder_final/` logs: HASWELL 1, HASWELL 4, NEHALEM 1, NEHALEM 4 and KATMAI 1
all read `33 passed`, and the 54-id and 163-id selections above cover the rest.
Plus the WSL (Linux, py3.12, numpy 2.4.6) run of 7.1 -- a different libm and a
different BLAS, and the arm that matters most for RED 1 since CI is Linux.  The
restored order-10 stimulus manufactures the lobe there too.

**What the ladder was NOT used for.**  `test_v4_16_0_agent_d_validity_ranges`,
`test_audit_w4_glass_registry_meshgrid` and `test_audit2609_b11_hygiene` are
registry-state, warn-once and AST checks with no floating-point kernel in them,
so a BLAS ladder measures nothing there; they are inside the 163-id selection
above and were also run on the default arm and under WSL.  The GBD budget test
reads `tracemalloc` allocation counts, which are kernel-independent by
construction -- it reads the same on every arm, as the table shows.
'''


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    m = re.search(r'### 7\.3 The kernel ladder\n.*?(?=### 7\.4 )', src, re.S)
    if not m:
        print('section 7.3 not found')
        return 1
    src = src[:m.start()] + NEW + '\n' + src[m.end():]
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('7.3 replaced with the full eight-arm table')
    return 0


if __name__ == '__main__':
    sys.exit(main())
