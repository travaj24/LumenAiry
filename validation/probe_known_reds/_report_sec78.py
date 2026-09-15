"""Write the report's sections 7 (runs) and 8 (limits)."""
import io
import sys

P = ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/'
     'WP-B14_KNOWN_REDS_REPORT.md')

OLD = """## 7. Runs

_Filled in from the ladder and sweep logs; see section 7 tables._

## 8. What could not be established

_See the closing list._
"""

NEW = '''## 7. Runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line, `--capture=sys`, `PYTHONPATH=/c/tmp/lum_reds`, and every tail
grepped for `passed|failed|error|no tests ran`.

### 7.1 The two builds

| run | Windows py3.14.6 / OpenBLAS Haswell | WSL py3.12.3 (Linux) |
|---|---|---|
| the seven affected files (c7, c8, M2 window contract, validity ranges, glass-registry meshgrid, b11 hygiene, gbd budget) | see 7.2 | **162 passed, 1 skipped in 690 s** |

The single WSL skip is a pre-existing `importorskip('refractiveindex')` in the
meshgrid file -- the glass extra is not in that venv.  The Linux arm is the
important one for RED 1: it is a different libm and a different BLAS, and the
restored order-10 stimulus manufactures the lobe there too, so the fix is not a
Windows artefact.

### 7.2 Per-file, Windows

| file | before | after |
|---|---|---|
| `test_niche_c7_ray_density_halo_check.py` + `test_niche_c8_inverse_support_bound.py` | 4 failed, 24 passed (55 s) | **28 passed** (73 s) |
| `test_pmm_m2_window_contract.py` (T3-1 id) | 7 arms pass, SANDYBRIDGE/4thr **fail** | 7 arms pass, SANDYBRIDGE/4thr **skip with the reading** |
| `test_v4_16_0_agent_d_validity_ranges.py` after the meshgrid file | 1 failed, 23 passed | **24 passed** (both orders, and alone) |
| `test_audit2609_b11_hygiene.py` | 82 passed | **85 passed** (137 s; +1 chain ratchet, +1 two-caller fixture, +1 shared scanner) |
| `tests/unit/test_wave5_gbd_dense_mem_budget.py` (new) | -- | **5 passed** (40 s) |

### 7.3 The kernel ladder

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

### 7.4 The consumer sweep of the swept modules

Every test file that imports `propagators.carrier`, `propagators.system`,
`propagators.gbd` or `propagators.carrier_field` was enumerated (105 files; 3
excluded because another agent held them mid-edit).  The subset that pins the
warning helpers directly -- `test_niche_c14_encapsulation.py` (which calls
`_warn_undeduped` with an explicit `stacklevel=1`),
`test_fix_grid_intent_override_2026_08_10.py` (which pins the `_guard_dispose`
route), `test_carrier_referenced.py` and `test_carrier_field.py` -- reads

```
109 passed, 5 warnings in 384.04s
```

which is the check that the `stacklevel=None` seam kept the explicit-integer
contract exactly.  The broader 102-file sweep was started and then stopped: see
section 8.

### 7.5 The census / walker / dispatcher-pin / public-API / doc-consistency sweep

26 walker and pin files plus `test_audit_except_budget.py` and
`test_niche_audit_w4_input_kind.py`:

```
1 failed, 864 passed, 11 skipped in 470.44s
```

The single failure is `test_public_api.py::
test_installed_metadata_version_matches_source_version`, and it is
**pre-existing and environmental**: it compares the box's INSTALLED
distribution metadata (3.7.8, from the stale editable install at
`D:\\...\\Lumenairy`) against the source `__version__` (5.47.0).  It reproduces
identically on a pristine `git archive` tree of the base commit `96cb2096`, and
the test's own message says what it means ("The editable install is stale:
re-run `pip install -e .`").  Nothing in this branch touches it.

The eleven skips are all pre-existing documented exemptions (the cache-lock
exemption list) or walkers that correctly decline because the topmost versioned
CHANGELOG block carries no claim of the kind they check.

### 7.6 Walkers on the written documents

```
python scripts/check_source_line_citations.py    ok=107  drift=0  total=107
python scripts/record_history_fingerprints.py --check
                                                 OK: every history document matches its module.
python scripts/check_doc_identifiers.py          OK: every API-claiming backticked identifier resolves.
tests/unit/test_audit2609_a17_history_relocation.py + _a17_history_lint.py
                                                 757 passed in 146 s
wsl ruff check lumenairy/ tests/                 All checks passed!
```

The 13 citations that drifted are all in `carrier.py` and `system.py` and all
drifted because THIS branch moved those lines; they were re-anchored against the
base commit `96cb2096` with the repository's own `reanchor_since.py` (content
matching, not arithmetic), and the V18 walker then reads 107 of 107.

### 7.7 The CHANGELOG

A `## [Unreleased]` block was created above `## [5.47.0]` -- the file had none,
and there is exactly one now.  It carries no `path.py:N` citations, so it adds
nothing for V18 to chase, and the changelog walkers continue to read the topmost
VERSIONED block, which is unchanged.

## 8. What could not be established, and what was deliberately not done

1. **The access violation itself was not reproduced.**  Section 4 bounds it --
   the dense GBD loop's transient is up to 6.0x the budget the caller asked for,
   3 073 MB against a 512 MB request -- and gives the switch that makes the
   budget a bound.  It does not prove that this is what faulted.  A reproduction
   would need the fault to be caught under `faulthandler` with the allocation in
   the traceback, which did not happen in this window.

2. **The 102-file consumer sweep did not finish.**  It was started, then stopped
   to free CPU when the box reached 33-48 concurrent python processes (the
   maintainer's own multi-day `q2b_qwp.py` run, four other agents' test runs and
   this session's ladders).  The targeted 4-file subset that actually pins the
   swept helpers did finish, green (7.4).  Two of this session's own ladder
   shells outlived the shells that launched them and kept writing into
   `ladder_waveD/` and `ladder_itemD/`; reaping them was not permitted by the
   environment, so the final ladder was routed to a directory they do not know
   about and those two directories' logs must not be read.

3. **The sixteen other propagator modules were not swept** (5.3).  A static
   reachability screen flags them, but a screen is a possibility test; each needs
   its own two-caller measurement, and doing that blind is not the mechanical
   half of 4.4.  Their census is in `stacklevel_census_base.json`.

4. **No 3.10 or 3.11 interpreter exists on this box.**  `py -0p` lists 3.14.6 and
   3.13.13; WSL has 3.12.3.  The 3.11 digest fix (6.2) is therefore established
   by reproducing, on 3.12/3.13/3.14, the exact digests CPython 3.11 itself
   computed on the CI runners (49 of 49, bit for bit) rather than by running
   3.11.  The 3.10 arm is inferred from the same pre-PEP-701 tokenizer and is
   NOT measured.  Only a re-run of the un-masked matrix closes either.

5. **Two `test_audit2609_a9_verify_ui.py` failures** were observed in passing and
   **reproduce on a pristine `git archive` tree of the base commit** (`2 failed,
   47 passed`), so they are pre-existing and outside every file set in this
   package.  They do not appear in any downloaded CI log for run 34914295323,
   which may mean they are Windows-specific or may mean the py3.11 `--maxfail`
   abort hid them.  Flagged for whoever owns `lumenairy/ui/`.

6. **Decisions deliberately NOT taken, and reserved for the maintainer:**
   * whether `DENSE_MEM_BUDGET_ACCOUNTING` becomes `'measured'` by default (it
     moves default-path bytes by summation order; the measurement is in 4.3);
   * **whose diagnostic the in-glass gap-leg warnings are** (handoff 4.4's second
     bullet) -- making them name the user needs the lens to catch and re-emit at
     its own entry point, which is an ownership decision, not a substitution;
   * whether `SILICON` should gain a bundled Sellmeier row so it is usable
     without the glass extra (raised by the a8 work; a data/default change, not a
     red fix).
'''


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
    print('sections 7 and 8 written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
