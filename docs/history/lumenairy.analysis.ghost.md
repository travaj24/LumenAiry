<!-- lumenairy-history-doc
module: lumenairy/analysis/ghost.py
ast_sha256: 21444e785f8d6c2c5012ee24406b46ae5ba8774f6edc3516f5ada67faa176818
token_sha256: 82b248194fc5b5445cbd58f7cc4869a2f358337512b19b84787d1d1cf0544db5
pre_relocation_lines: 1034
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/analysis/ghost.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/ghost.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

Three blocks, all release chronology around behaviour that is now simply the
behaviour: the `seed=None` default on the BSDF Monte Carlo (twice -- once in
the parameter doc and once at the call site) and the tombstone explaining why
`_ghost_intersect` is an alias.

What stayed is the part a caller acts on: `seed=None` gives a
sample-to-sample spread usable as a stand-in for the integration error, an
`int` gives a reproducible single sample with no uncertainty band, and
`direction` is accepted but ignored.  The `ghost_analysis` upper-bound caveat
(B2-1) and the "this is a first-pass estimate, use FRED/ASAP for sign-off"
note are live contract and were not touched.

Nothing the interpreter executes changed in the move.  The header above
records the SHA-256 of (a) the module's AST with every docstring removed and
source positions ignored, and (b) its `tokenize` stream reduced to
NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both taken from
the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from
the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L399-407 | `stray_light_report` -- `seed` | that the seed was hard-coded to 0 before v4.15 and what that hid |
| L447-450 | `stray_light_report` -- the TIS integration | the `Pre-4.15 default_rng(0) pinned the TIS number` framing |
| L571-588 | `_ghost_intersect` | what this helper inlined at v5.4.0 and how v5.4.1 promoted the fix into `_intersect_surface` |

---

### L399-407 -- `stray_light_report` -- `seed` -- that the seed was hard-coded to 0 before v4.15 and what that hid

*Left in the source:* what each value of the knob gives the caller, which is the whole of the live contract

```text
        Seed for the BSDF-integration Monte Carlo.  v4.15 (P1-GH-1):
        pre-4.15 this was hard-coded to ``0``, which made the TIS
        estimate fully deterministic with no uncertainty band -- you
        could not estimate the Monte-Carlo variance because every
        call returned identical numbers.  Pass an ``int`` (e.g.
        ``seed=0``) for the old reproducible behaviour; pass
        ``None`` (the new default) to draw fresh randomness from
        system entropy so repeated calls give a sample-to-sample
        spread you can use as a stand-in for the integration error.
```

### L447-450 -- `stray_light_report` -- the TIS integration -- the `Pre-4.15 default_rng(0) pinned the TIS number` framing

*Left in the source:* where the seed comes from and what the default buys

```text
        # v4.15 (P1-GH-1): RNG now seeded from the user-supplied
        # ``seed`` kwarg (None = system entropy = different samples
        # each call).  Pre-4.15 ``default_rng(0)`` pinned the TIS
        # number to a single sample, hiding the Monte-Carlo error.
```

### L571-588 -- `_ghost_intersect` -- what this helper inlined at v5.4.0 and how v5.4.1 promoted the fix into `_intersect_surface`

*Left in the source:* that the helper is redundant, that it is kept so external importers resolve, and that ``direction`` is accepted but ignored

```text
    v5.4.1 (audit P1): ``_ghost_intersect`` is now redundant; the
    library's ``_intersect_surface`` uses direction-aware root pick
    canonically.  Kept as an alias for backward compatibility so any
    external caller that imported this helper continues to resolve.

    Historically (v5.4.0) this helper inlined a direction-aware
    ray-sphere quadratic to work around a direction-blind root pick
    in ``_intersect_surface`` that landed backward-propagating rays
    on the wrong side of the sphere.  The v5.4.1 patch promoted that
    fix into the canonical library function (``intersection.py``
    fast path at the spherical branch + Newton initial guess), so
    ``_intersect_surface`` is now correct for both forward and
    backward legs.

    The ``direction`` keyword is retained for signature compatibility
    but ignored -- the library inspects each ray's ``N`` cosine to
    select the near root, so a bundle-wide direction hint is no
    longer required.
```
