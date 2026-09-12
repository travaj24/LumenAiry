<!-- lumenairy-history-doc
module: lumenairy/memory.py
ast_sha256: 4f9c43586dd1d1419b2e5afcca135cfc82d44f845db577a7a9f086c1ecbfe521
token_sha256: 3dcd8672dcbd1ee9169073cabf0f2815affc462eb35b0b93b633aa82b5c1be33
pre_relocation_lines: 1332
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/memory.py`

This file holds the version-history narrative that used to live in
`lumenairy/memory.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: every measured calibration this module's numbers
rest on (`docs/TESTING_STANDARDS.md` S5) -- the v5.17.1 whole-grid and chunked
anchors, the 2026-09-12 `apply_real_lens` bytes-per-pixel solve
(`8*F + 16*C = 176.02`, `8*F + 8*C = 120.01`) and its ~7 % margin, the
`_ASM_FIRST_CALL_*` figures, and the plan-cache byte-cap arithmetic.  The live
`set_low_memory` / `set_max_ram` contracts stayed too.

`prefix : str` in `memory_report`'s parameter list is a false positive of the
finding's own `\bpre-?fix\b` pattern (the English word "prefix") and was left
alone; the history-lint ratchet baselines it rather than demanding zero.


Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment plus a pointer to this file; those are noted per block
below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L108-113 | `set_max_ram`, the negative-budget guard | "Pre-v4.14 a negative value was silently accepted" |
| L306-315 | `pick_batch_size` docstring, ``cost_per_item`` / ``available`` | "pre-v5.29.1 it was silently treated as free" and "audit P2-21: pre-v5.17.2 this bypassed the override" |
| L517-518 | the `estimate_lens_memory` CALIBRATION banner | the obsolete pre-v5.17.0 anchors (44.5/37.2/57.3 GB) and the two since-fixed defects they reflected |
| L530-538 | the `lens_model='real'` constants | "The pre-v5.46 branch reused the traced calibration above and then scaled the float64 core DOWN ..." |
| L932-945 | the plan-workspace dtype handed to `_plan_entry_n_bufs` | "the first cut built ``np.dtype(f'c{2 * cb}')``" -- the defect, its two platform-specific failures and the pins that missed it |

---

### L108-113 -- `set_max_ram`, the negative-budget guard -- "Pre-v4.14 a negative value was silently accepted"

*Left in the source:* the hazard in the present tense -- what a silently-accepted negative budget looks like downstream -- and why zero is rejected too

```text
    # v4.14 (audit P3 #18): reject negative budgets explicitly.  Pre-
    # v4.14 a negative value was silently accepted (treated as
    # negative bytes); ``pick_batch_size`` then clamped via
    # ``min_batch=1`` so the bug only surfaced as quiet single-batch
    # processing on huge workloads.  Zero is also nonsensical (no
    # work could ever fit) so reject it too.
```

### L306-315 -- `pick_batch_size` docstring, ``cost_per_item`` / ``available`` -- "pre-v5.29.1 it was silently treated as free" and "audit P2-21: pre-v5.17.2 this bypassed the override"

*Left in the source:* both live rules -- a negative cost is rejected, and the budget comes from `get_ram_budget` -- each with the failure it prevents

```text
        workload fits in one batch); a NEGATIVE cost is rejected (audit
        A-9..A-14 -- pre-v5.29.1 it was silently treated as free, so a
        sign-flipped or subtracted-in-the-wrong-order caller got the
        maximum batch size and OOMed instead of being told).
    available : int or None
        Available memory in bytes.  If ``None``, uses
        :func:`get_ram_budget` (the :func:`set_max_ram` override when
        set, else the auto-detected available memory) -- audit P2-21:
        pre-v5.17.2 this bypassed the override via
        :func:`available_memory_bytes`.
```

### L517-518 -- the `estimate_lens_memory` CALIBRATION banner -- the obsolete pre-v5.17.0 anchors (44.5/37.2/57.3 GB) and the two since-fixed defects they reflected

*Left in the source:* the LIVE calibration anchors above them, in full

```text
# Pre-v5.17.0 anchors (44.5/37.2/57.3 GB) reflected the since-fixed v4.10
# tilt-check leak + upsample double-build and are obsolete.
```

### L530-538 -- the `lens_model='real'` constants -- "The pre-v5.46 branch reused the traced calibration above and then scaled the float64 core DOWN ..."

*Left in the source:* that these constants are measured on the bare entry point, and the measured 2.8x / 1.6x under-prediction that reusing the traced calibration produces

```text
# v5.46 (audit Z3).  These are its OWN constants, measured on it.  The
# pre-v5.46 branch reused the traced calibration above and then scaled the
# float64 core DOWN by ``5 / _LENS_F64_ARRAYS`` on the reasoning that the bare
# entry point "omits the traced final-assembly float64 arrays".  Measured, it
# does not: ``estimate_lens_memory(..., lens_model='real')`` under-predicted
# ``apply_real_lens``'s tracemalloc peak by 2.8x (parallel_amp=False) / 1.6x
# (parallel_amp=True, the default) -- i.e. a pre-flight budget computed with
# the DOCUMENTED model for that entry point under-reserved by up to 2.8x,
# which is the exact failure ``check_sim_memory`` exists to prevent.
```

### L932-945 -- the plan-workspace dtype handed to `_plan_entry_n_bufs` -- "the first cut built ``np.dtype(f'c{2 * cb}')``" -- the defect, its two platform-specific failures and the pins that missed it

*Left in the source:* the rule (pass the caller's dtype through) plus the trap and both measured under-estimates, as a do-not-do-this

```text
    # v5.33.3 (VERIFY_PERF_BRANCH_2026_08_10 D1): the dtype handed to the
    # predicate is the CALLER's, not a re-spelled one.  The first cut built
    # ``np.dtype(f'c{2 * cb}')`` -- but ``cb`` is ALREADY the complex
    # itemsize, so that asked for a dtype of twice the element size and the
    # workspace was priced at 2x.  It failed differently on each platform,
    # which is why no pin saw it: ``'c32'`` is ``complex256`` on Linux (a
    # silent 43 % UNDER-estimate at N=8192/complex128, because a 2.147 GB
    # phantom workspace fails the 2 GB cap the true 1.074 GB one passes),
    # and a ``TypeError`` on MSVC that the except below swallowed into
    # ``n_bufs = 2`` -- a no-op for complex128 and, via the perfectly valid
    # ``'c16'``, a 9.66 GB UNDER-estimate for complex64 at N=12288 on
    # Windows too.  ``_plan_entry_n_bufs`` reads only ``.itemsize``, so
    # passing ``complex_dtype`` straight through is both correct and
    # incapable of raising for any dtype ``_as_complex_itemsize`` accepted.
```
