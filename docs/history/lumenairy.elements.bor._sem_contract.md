<!-- lumenairy-history-doc
module: lumenairy/elements/bor/_sem_contract.py
ast_sha256: 6dea7b84f8d4ef0c84a06b7b30c986d9fd9f9616ec123240c8d8db35da2ef054
token_sha256: dddb7fb977382766b8591d59b0e71313d67d15d345d1928f5ba56ccec23f157a
pre_relocation_lines: 569
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bor/_sem_contract.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/_sem_contract.py`: the two `ROUND 2 RESTATEMENT` blocks
that retract a margin claimed a few lines above them, the `ROUND 2 (D4)` /
`ROUND 3 (GAP 3)` passages recording what a branch USED TO read, and the
"the scoping's own note said ..." meta-notes.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the measured censuses behind `_BOR_Q_EXCESS`,
`_BOR_SLIVER_BAND_FRAC`, `_BOR_MIN_ELEM_FRAC` and `_BOR_FRAC_DEADBAND`, the hp
refinement ladder, and the argument for NOT moving the warn edge.
`docs/TESTING_STANDARDS.md` S5 wants those on the bar, and
`tests/unit/test_fix_bor_multilayer_guards.py` re-measures the ordinary census
on the running build.

Two corrections, not moves: both restatement blocks were the source of truth
for the binding margin while the sentence they corrected still stood above
them.  The source now states the binding margin ONCE -- `1.20 decades (15.9x)`
one-sided for `_BOR_Q_EXCESS`, and `1.003x` (graded hp refinement) for
`_BOR_SLIVER_BAND_FRAC`.

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
| L107-140 | `_BOR_Q_EXCESS` | a ROUND 2 RESTATEMENT retracting the two-sided margin the block above it claimed, and the "the scoping quoted ..." notes |
| L142-145 | `_BOR_Q_EXCESS`, the non-finite reading | "This bar is NOW reached only when ... :func:`verdict` NOW reads it as HOT" |
| L185-189 | `_BOR_SLIVER_BAND_FRAC`, the refuted candidate edge | "The scoping's own note said its 3.9x margin ... this is that measurement, and it moved the edge a decade" |
| L198-203 | `_BOR_SLIVER_BAND_FRAC`, the binding margin | a ROUND 2 RESTATEMENT retracting the 9.77x margin claimed seven lines above it |
| L233-236 | `_BOR_SLIVER_BAND_FRAC`, why the edge is not moved | "Round 2 therefore records the margin honestly ... the next round has the measurement to decide with" |
| L347-357 | `measure_layer` docstring, the ``w_min_own`` field | D4 -- that the ``warn_own`` branch used to read ``w_min``, and the two wrongly-blamed layers it named |
| L444-450 | `verdict`, the non-finite ``q_excess`` reading | GAP 3 -- "A NON-FINITE ``q_excess`` used to read as NOT hot" |
| L460-461 | `verdict`, the never-formed ratio | "and the pre-round-3 behaviour for that case" |
| L465-466 | `verdict`, the OWN arm | the ``ROUND 2 (D4)`` chronology tag |
| L473 | `verdict`, the width comparisons | the ``ROUND 3 (GAP 4)`` chronology tag |

---

### L107-140 -- `_BOR_Q_EXCESS` -- a ROUND 2 RESTATEMENT retracting the two-sided margin the block above it claimed, and the "the scoping quoted ..." notes

*Left in the source:* what the bar does (it suppresses refusals, so its failure mode is a MISS), why an 'ordinary' margin is vacuous here, and the one-sided margin that is real -- stated once.

```text
#: ROUND 2 RESTATEMENT -- THIS BAR'S "ORDINARY" MARGIN PROTECTS NOTHING, AND ITS
#: ONLY OPERATIVE ROLE IS IN THE OTHER DIRECTION.
#:
#: The build stated the margins two-sidedly: "0.95 decades (8.82x) above the
#: worst ordinary geometry measured" and "1.20 decades (15.9x) below the
#: mildest rung it must refuse".  The SECOND is real.  The FIRST is vacuous,
#: and the verification (section 5.3) is right about why: the refusal is a
#: CONJUNCTION, ``w_min_union_frac < _BOR_MIN_ELEM_FRAC`` **and**
#: ``q_excess > _BOR_Q_EXCESS``, so a geometry with no cross-layer cell at all
#: -- ``w_min_union_frac = inf``, which is what every ordinary family measured
#: reads -- can never be refused **whatever its q_excess**.  Quoting a distance
#: from ordinary geometry to this bar therefore describes a comparison the
#: conjunction cannot reach.
#:
#: WHAT THE BAR ACTUALLY DOES: it SUPPRESSES refusals.  A manufactured cell
#: narrow enough to pass the geometric conjunct is refused only if its spectrum
#: also shows the damage, and this is the threshold that decides.  Its failure
#: mode is therefore a MISS, not a false positive, and the honest statement of
#: its margin is the one-sided one: 1.20 decades (15.9x) below the mildest rung
#: the ladder measured as damaging (the union ladder at ``delta/Rbig`` = 1e-6,
#: degree 6, ``|q|max``/ceiling = 1.5934e+05).
#:
#: The ordinary readings below are kept because the number should be RIGHT even
#: where it is not load-bearing, and round 2's census moved it: the build quoted
#: a non-taper maximum of 164.4, and hp refinement with GRADING on -- a family
#: the build's census did not sweep -- reads higher (see
#: ``_BOR_SLIVER_BAND_FRAC`` and
#: ``validation/probe_fix_bor_round2/r7_sem_hp_census.py``).
#:
#: The scoping quoted 1.63 decades of headroom on the ordinary side; that was a
#: SAMPLE property of a census that stopped at a 64-slice taper (235.8).  The
#: widened census reaches 1134.2.  A LOWER k0 RAISES this ratio, so the low-k0
#: taper arm is the demanding one and is why the census sweeps k0 as well as the
#: slice count.  Kernel spread of the quantity itself: 1.1895x.
```

### L142-145 -- `_BOR_Q_EXCESS`, the non-finite reading -- "This bar is NOW reached only when ... :func:`verdict` NOW reads it as HOT"

*Left in the source:* what a non-finite ratio means and how :func:`verdict` treats it, without the round framing.

```text
#: ROUND 3 (verification round 2, GAP 3) -- WHAT A NON-FINITE RATIO MEANS.
#: This bar is now reached only when the ratio is FINITE.  A ratio FORMED
#: from a real spectrum that comes back ``inf`` or ``nan`` is PAST every bar
#: there is, and :func:`verdict` now reads it as HOT rather than as benign; a
```

### L185-189 -- `_BOR_SLIVER_BAND_FRAC`, the refuted candidate edge -- "The scoping's own note said its 3.9x margin ... this is that measurement, and it moved the edge a decade"

*Left in the source:* the refutation itself, which is why the edge is 1e-4 and not 1e-3.

```text
#: **THE SCOPING'S CANDIDATE EDGE OF 1e-3 IS REFUTED BY THIS CENSUS**: a
#: 256-slice taper lands at 9.766e-04, i.e. 0.977x -- INSIDE the band it would
#: have warned on.  The scoping's own note said its 3.9x margin at 64 slices
#: was sample-scoped and had to be re-measured before the edge was fixed; this
#: is that measurement, and it moved the edge a decade.
```

### L198-203 -- `_BOR_SLIVER_BAND_FRAC`, the binding margin -- a ROUND 2 RESTATEMENT retracting the 9.77x margin claimed seven lines above it

*Left in the source:* the binding margin and the family that sets it, stated as a fact rather than as a correction.

```text
#: ROUND 2 RESTATEMENT -- **9.77x IS NOT THE BINDING ORDINARY MARGIN.  1.003x
#: IS.**  Both statements above are properties of the census's own two families
#: (taper staircases and single-layer hp).  The verification found a third the
#: census did not sweep, and round 2 measured it
#: (``validation/probe_fix_bor_round2/r7_sem_hp_census.py``): hp refinement with
#: GRADING ON, on an ORDINARY two-layer ring pair.
```

### L233-236 -- `_BOR_SLIVER_BAND_FRAC`, why the edge is not moved -- "Round 2 therefore records the margin honestly ... the next round has the measurement to decide with"

*Left in the source:* the decision and its cost, which is what a later editor weighing the same trade needs.

```text
#: Round 2 therefore records the margin honestly rather than trading a true
#: warning for a false one; the cost is a ``UserWarning``, never a refusal
#: (``warn_manufactured`` carries no spectral conjunct and cannot escalate), and
#: the next round has the measurement to decide with.
```

### L347-357 -- `measure_layer` docstring, the ``w_min_own`` field -- D4 -- that the ``warn_own`` branch used to read ``w_min``, and the two wrongly-blamed layers it named

*Left in the source:* what ``w_min_own`` is and why it is NOT ``w_min``, which is the distinction the message's wording depends on.

```text
                         does).  ROUND 2, verification D4: the ``warn_own``
                         branch used to read ``w_min``, the narrowest cell of
                         the POST-WINDOW mesh, and then tell the caller that
                         "the LAYER'S OWN segment list asked for" it -- which is
                         false for the NEIGHBOUR of a liner, whose own segment
                         list is a single full-radius entry and which has the
                         cell only because the +-1 enrichment window put it
                         there.  Measured: the liner ladder emitted TWO
                         warn_own messages, blaming layers [0, 1], where layer 1
                         is ``add_layer(0.5, eps=1.21)``.  ``w_min_own`` is the
                         quantity the message's own words describe.
```

### L444-450 -- `verdict`, the non-finite ``q_excess`` reading -- GAP 3 -- "A NON-FINITE ``q_excess`` used to read as NOT hot"

*Left in the source:* the whole argument in present tense, including the two-kernel measurement, because it is the reason the test is shaped this way and TESTING_STANDARDS forbids the alternative.

```text
    # ROUND 3 (verification round 2, GAP 3).  A NON-FINITE ``q_excess`` used to
    # read as NOT hot -- ``np.isfinite(excess) and excess > bar`` -- so the
    # contract fell SILENT exactly where the damage is worst, and because
    # ``inf`` is a backward-error outcome the VERDICT moved with the BLAS
    # kernel.  Measured on one geometry (a caller-prescribed liner 1e-8 of
    # ``Rbig`` wide AT THE AXIS; ``BORStack(Rbig=24, m=1, N=120, basis='sem',
    # degree=8)``), both builds, 2026-09-12:
```

### L460-461 -- `verdict`, the never-formed ratio -- "and the pre-round-3 behaviour for that case"

*Left in the source:* the conservative reading, which is the rule.

```text
    # never formed is no evidence in either direction and stays cold, which is
    # the conservative reading and the pre-round-3 behaviour for that case.
```

### L465-466 -- `verdict`, the OWN arm -- the ``ROUND 2 (D4)`` chronology tag

*Left in the source:* which quantity the arm reads and which it must not, plus the explanation below it.

```text
    # ROUND 2 (D4): the OWN arm reads ``w_min_own_frac`` -- the narrowest cell
    # BOTH of whose walls this layer's own segment list asked for -- and not
```

### L473 -- `verdict`, the width comparisons -- the ``ROUND 3 (GAP 4)`` chronology tag

*Left in the source:* what ``_below`` does and why, unchanged.

```text
    # ROUND 3 (GAP 4): the three width comparisons go through ``_below``, which
```
