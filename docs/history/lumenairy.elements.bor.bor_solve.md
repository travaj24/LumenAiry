<!-- lumenairy-history-doc
module: lumenairy/elements/bor/bor_solve.py
ast_sha256: f8fdde67bce6974d1bab6a8e9e637d78b031f361214ef86071e3cc042a99730b
token_sha256: 2664f44cafc4bc8de5558e4c2c6f986c382907104f19674999cb20fcdaa7491b
pre_relocation_lines: 826
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/elements/bor/bor_solve.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/bor_solve.py` -- the `ROUND 2` / `ROUND 3` passages that
tell the reader what a predicate USED TO say, what it NO LONGER gates, and which
round moved it.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

What did NOT move: the measured bars (`_BOR_NODAL_SUPERUNITY_BAR`,
`_BOR_NODAL_SUPERUNITY_WARN`, `_BOR_LOSSLESS_REL_IM`, `_BOR_PASSIVE_DEADBAND`)
and their censuses, the fail-before switch's description, and the ladders that
size the two detectors.

One correction, not a move: `_BOR_NODAL_SUPERUNITY_WARN` said "3.71 decades
above the healthy ceiling measured above" and a `ROUND 2 RESTATEMENT` ten lines
below said the binding number is **0.57** decades on the two-sided measure the
screen actually arms.  The source now states the binding margin first, with the
one-sided figure named for what it is.

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
| L90-108 | `_BOR_NODAL_SUPERUNITY_WARN` | a ROUND 2 RESTATEMENT correcting the margin claimed three lines above it |
| L125-128 | `_BOR_LOSSLESS_REL_IM` | a ROUND 2 NOTE recording what this constant used to gate |
| L263-281 | `_stack_is_provably_passive` docstring | ROUND 2 -- "WHAT THIS USED TO SAY, AND WHY IT WAS DEFECT D1", the superseded reasoning quoted, and the pre-round-2 rung census |
| L291-299 | `_stack_is_provably_passive` docstring, the round-3 split | ROUND 3 -- "THIS PREDICATE NO LONGER GATES THE INDEX CEILING", and what the shared early return let through |
| L326-334 | `_stack_media_are_passive` docstring | ROUND 3 -- "WHY THE TWO ARE NOW SEPARATE" and what the shared early return disarmed until then |
| L663-664 | `build_layer`, the ``Rbig/lambda`` hint | "RETIRED AS A DECISION and kept only as" -- the retirement framing |
| L786-788 | `solve` docstring | W6-B11 -- what a ``thickness=None`` middle layer used to die of |

---

### L90-108 -- `_BOR_NODAL_SUPERUNITY_WARN` -- a ROUND 2 RESTATEMENT correcting the margin claimed three lines above it

*Left in the source:* both margins, with the BINDING (two-sided) one stated first and the one-sided 3.71 named as what it is, plus the cost argument for keeping the edge where it is.

```text
#: The WARNING edge.  Between this and the refusal bar the answer is returned
#: with a ``UserWarning`` quoting the measured violation -- so a mildly damaged
#: nodal solve is never SILENT, which is what the retired ``Rbig/lambda > 4``
#: proxy allowed.  3.71 decades above the healthy ceiling measured above.
#:
#: ROUND 2 RESTATEMENT OF THAT MARGIN.  "3.71 decades" is against the ONE-SIDED
#: healthy ceiling (4.4336e-09).  On the TWO-SIDED measure this edge now reads
#: -- ``max(|R+T-1|)`` on the accurate family, which is what D2 armed -- the
#: ceiling is **2.714107e-07** (uniform nodal rows with the right channel set
#: and a silent index ceiling; ``r1_passivity_census`` -> ``r4_bars``), so the
#: edge carries **0.57 decades, not 3.71**.
#:
#: It is kept at 1e-6 anyway, and the reason is what this edge COSTS when it is
#: wrong: one ``UserWarning``, never a refusal.  The quantity is also
#: kernel-stable by construction -- the nodal blow-up is a deterministic
#: discretisation defect, not an arithmetic one -- so 3.7x is not the same kind
#: of thin margin a backward-error-driven bar would have at 3.7x.  The number is
#: recorded here rather than left at 3.71 so the next round re-derives from the
#: measurement.
```

### L125-128 -- `_BOR_LOSSLESS_REL_IM` -- a ROUND 2 NOTE recording what this constant used to gate

*Left in the source:* what the constant means now and the measurement that sizes it; what gates what is stated at the predicates themselves.

```text
#:
#: ROUND 2 NOTE.  Until round 2 this ONE constant gated the WHOLE screen, in
#: both directions, and that was defect D1: see
#: :func:`_stack_is_provably_passive`.
```

### L263-281 -- `_stack_is_provably_passive` docstring -- ROUND 2 -- "WHAT THIS USED TO SAY, AND WHY IT WAS DEFECT D1", the superseded reasoning quoted, and the pre-round-2 rung census

*Left in the source:* the ENERGY argument for passive-not-lossless, the discontinuity at ``Im(eps) = 0+`` it removes, and the measurement that the violation does not move with the loss at all.

```text
    ROUND 2 -- WHAT THIS USED TO SAY, AND WHY IT WAS DEFECT D1.  As first built
    it required every layer to be LOSSLESS to 1e-12 and disarmed otherwise, on
    the stated reasoning *"on a lossy stack there is no theorem to violate."*
    That reasoning covers the BELOW-unity direction only.  For a stack of
    passive media energy conservation reads ``R + T + A = 1`` with the absorbed
    fraction ``A >= 0``, so ``R + T <= 1`` holds on EVERY passive stack,
    absorbing or not -- and the guard had a discontinuity at ``Im(eps) = 0+`` of
    exactly the shape this same wave removed from the EME module.

    MEASURED (``validation/probe_fix_bor_round2/r2_loss_ladder.py``, ladder A):
    the shipped refusal fixture with a relative loss on the ring's high region.
    The returned violation is the SAME 2.9 % across the whole ladder -- the
    nodal excess reads 2.881869e-02 at ``Im/Re`` = 0, at 1e-12, at 3e-12 and out
    to 1e-04 -- while the pre-round-2 predicate refused the first four rungs and
    RETURNED every rung from 3e-12 up.  A "lossless" glass entered as
    ``n = 1.5 + 1e-8i``, or any dispersion fit with a residual imaginary part,
    took the whole guard out.

    WHAT STAYS, AND WHAT IT IS NO LONGER ALLOWED TO REACH.  The LOSSLESS
```

### L291-299 -- `_stack_is_provably_passive` docstring, the round-3 split -- ROUND 3 -- "THIS PREDICATE NO LONGER GATES THE INDEX CEILING", and what the shared early return let through

*Left in the source:* what this predicate gates NOW, which sibling gates the other detector, and the measured hole that separating them closes.

```text
    ROUND 3 (verification round 2, GAP 2) -- THIS PREDICATE NO LONGER GATES THE
    INDEX CEILING.  Until round 3 it was the single early return
    :func:`_check_nodal_passivity` took for BOTH detectors, so the conjunct
    above -- an energy argument -- disarmed the ceiling too, and D1's own
    discontinuity at ``Im(eps) = 0+`` survived on ``layers[0]``.  Measured: a
    3e-12 loss there walked a returned ``max(R + T) = 2.41297`` past both
    detectors against a twin closing to 1 exactly.  The media half is now
    :func:`_stack_media_are_passive`, which gates both; THIS predicate gates
    the energy detector alone.
```

### L326-334 -- `_stack_media_are_passive` docstring -- ROUND 3 -- "WHY THE TWO ARE NOW SEPARATE" and what the shared early return disarmed until then

*Left in the source:* the whole argument for the split, in present tense, including the 3e-12 rung.

```text
    ROUND 3 (verification round 2, GAP 2) -- WHY THE TWO ARE NOW SEPARATE.
    The incidence-lossless conjunct is about the ENERGY: ``R`` and ``T`` are
    formed from a unit-``|z-flux|`` basis, so in an absorbing incidence medium
    they are not power fractions and no energy bar can mean anything.  It is
    NOT about the INDEX CEILING, whose Rayleigh argument concerns one
    half-space's own ``eps`` and is untouched by whether the incidence medium's
    flux is conserved.  Until round 3 the two detectors shared one early
    return, so a loss of ANY size on ``layers[0]`` -- 3e-12 included, the exact
    rung defect D1 was named for -- disarmed BOTH.
```

### L663-664 -- `build_layer`, the ``Rbig/lambda`` hint -- "RETIRED AS A DECISION and kept only as" -- the retirement framing

*Left in the source:* what the test IS now (a hint, never a decision), with the scoping measurement and the pointer to where the decision lives.

```text
        # 5.45.1: this ``Rbig/lambda > 4`` test is RETIRED AS A DECISION and
        # kept only as an early, cheap hint.  It is a PROXY, and the scoping
```

### L786-788 -- `solve` docstring -- W6-B11 -- what a ``thickness=None`` middle layer used to die of

*Left in the source:* the same failure as the reason the entry check exists, with the bare TypeError it replaces named so the symptom is searchable.

```text
    Audit W6-B11: a MIDDLE layer left at the ``build_layer`` default
    ``thickness=None`` used to die inside ``propagation_smatrix`` with a bare
    ``TypeError: unsupported operand type(s) for *: 'complex' and 'NoneType'``.
```
