<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/stack.py
ast_sha256: 9c0cea3a2b7cd1d95113832f54bf3904330b751af37cc354bafcae64fb66b37a
token_sha256: d3d0dea91c4b129671c180e076dbc5845c76b7790f9442bd8f9dbbca9a8bb762
pre_relocation_lines: 5360
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- per-layer stabilize='slices' refusal message reworded to present tense; the retracted wording is recorded in this document at L3209-3215 (WP-A22 follow-up)
-->

# Version history -- `lumenairy/elements/pmm/stack.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/stack.py` -- the "round N did A, which was wrong
because B, so round N+1 does C" passages of the O-11 near-coincident-wall
(sliver) campaign, and the smaller "pre-fix this returned X" notes elsewhere in
the module.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file, so `git log -S` on any phrase here still lands
on the commit that wrote it.

What did NOT move: the measured derivations of the live bars
(`_STACK_SUPERUNITY_BAR`, `_SLIVER_TRIGGER_BAR`, `_SLIVER_OWN_SCALE_RATIO`,
`_SLIVER_MOVE_FACTOR`, `_SLIVER_ATTRIB_CLOSURE`, `_SLIVER_CLOSURE_FRACTION`,
`_SLIVER_WALL_RATIO`, `_MIN_FEATURE_DEFAULT_FRAC`, `_SLIVER_Q_EXCESS`).
`docs/TESTING_STANDARDS.md` S5 requires a numeric bar to carry its oracle, its
error floor and its measured populations, and
`tests/unit/test_fix_pmmstack_sliver_round4.py` reads two of those numbers
(`833.78`, `906.56`) straight out of the source.  The live migration note on
`min_feature`, the per-layer / window contracts, and the fail-before switches'
descriptions are current behaviour and also stayed.

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
| L93-181 | `<module>` -- the O-11 section header | rounds 1-4 of the sliver campaign: the super-unity conjunction, its 110/648 false positives, the arbiter, the relative closure |
| L183-186 | `PMM_SLIVER_GUARD` | the round-1 "BOTH conjuncts" description of when the guard fires -- the opposite of what the code does since round 4 |
| L1358-1361 | `_warn_stack_energy` docstring, the NON-FINITE arm | "used to reach the far field and return ``tot = [nan, nan]``" |
| L1377-1384 | `_warn_stack_energy` docstring | ROUND 4 -- what rounds 1-3 gated the arbiter on, and the 5.45.0 CI matrix that refuted it |
| L1396-1398 | `_warn_stack_energy` docstring, the ``'truncation'`` arm | round 1's 110-of-648 false positives on this arm |
| L1399-1402 | `_warn_stack_energy` docstring, the ``'unknown'`` arm | that this arm is round 1's behaviour, unchanged |
| L1411-1414 | `_warn_stack_energy` docstring, the within-layer arm | ROUND 4 -- that this arm, too, used to be gated on super-unity |
| L1438-1444 | `_warn_stack_energy`, before the arbiter call | ROUND 4 -- the early return rounds 1-3 took below the trigger |
| L1811-1846 | `PMMStack.__init__`, the ``min_feature`` block | the G2 two-fixture degree-scatter ladder, duplicated verbatim from :data:`_MIN_FEATURE_DEFAULT_FRAC`, and the superseded ``period*1e-5`` default |
| L1890-1917 | `PMMStack.__init__`, "WHAT REPORTS A SLIVER, EXACTLY" | a comment correcting an earlier COMMENT -- what this block used to claim `_pmm_union_grid` reported |
| L2716 | `PMMStack._solve_conical`, the tensor gate | "the old path was silently wrong for it" |
| L2977-2979 | `PMMStack.solve`, the ``stabilize`` validation | P3-31 -- what the covariant dispatch used to accept |
| L3025-3034 | `PMMStack.solve`, the incidence guard | audit M3 -- the negative efficiencies the JAX twin returned before the mirror guard existed |
| L3134-3141 | `PMMStack.solve`, the out-of-plane routing | that the pre-fix covariant-for-OOP routing was validated against engines sharing the same defect |
| L3209-3215 | `PMMStack.solve`, the per-layer `stabilize='slices'` refusal MESSAGE | a message retracting its own earlier wording -- "this message used to say per-layer grids have 'no cross-layer walls to perturb', which is WRONG" |
| L3445-3447 | `_solve_vertical_perlayer` docstring | a comment correcting an earlier COMMENT -- this docstring's own retracted "``min_feature`` never enters" claim |
| L3832-3837 | `_slices_consensus_check`, the no-recipe fallback | "previously the ENTIRE geometry-built path ... was silently unprotected" |
| L4044-4045 | `PMMStack.internal_field` docstring, Returns | which release changed the H scale, and what the day-one method returned |
| L4377-4380 | `PMMStack.layer_absorption`, the per-material split | P2-13 -- that the ezz channel used to be omitted from the split |
| L5093-5114 | `_solve_conical`, the half-space kz gauge bridge | W7 F-C -- the measured pre-fix T = [0, 0] table on three lossy substrates |

---

### L93-181 -- `<module>` -- the O-11 section header -- rounds 1-4 of the sliver campaign: the super-unity conjunction, its 110/648 false positives, the arbiter, the relative closure

*Left in the source:* the mechanism of the pathology (unchanged, above), the statement that nothing per-layer and free separates the two populations, and a description of the guard as it stands now -- geometric screen, arbiter, then refusal or warning.

```text
# WHAT SEPARATES, MEASURED, AND WHAT DOES NOT.  Nothing per-layer and free
# separates the correct solves from the wrong ones on this family: the T3-4
# residual ``n_grow_post`` reads 0 on every row (the 2026-08-06
# ``_forward_growth_flip`` repair redirects them all), the T3-4 margin reads
# 1.0-1.3 on BOTH populations, and ``q_excess`` -- which is the right IDEA, a
# mode called propagating that no propagating mode can be -- crosses 1 while
# the answer is still right and then SATURATES across the onset (3.66 on the
# last correct degree-14 row and 3.66 on the first wrong one).  What DOES
# separate, by 5.4 decades over 138 dense rows on three degrees, is the
# assembled answer's PASSIVITY:
#
#     max |R+T-1| among CORRECT rows   4.13e-06     (5.4 decades)
#     min  (R+T-1) among WRONG rows    1.159e+00
#
# so this guard is a CONJUNCTION of that theorem violation with the GEOMETRIC
# cause -- M1's ``_guarded_lstsq`` lesson (rank AND residual) again:
#
#   (a) the union grid MANUFACTURED a cell -- one whose two walls share no
#       owning layer -- at least ``_SLIVER_OWN_SCALE_RATIO`` times finer than
#       the finest wall spacing any single layer asked for; and
#   (b) the solve reads super-unity above ``_STACK_SUPERUNITY_BAR`` on a
#       PROVABLY PASSIVE stack with a LOSSLESS propagating incidence medium,
#       where ``R + T <= 1`` is a theorem and not a tolerance.
#
# (b) alone would promote to a refusal every solve that today only warns,
# including the documented many-slice quasi-resonance the warning was
# deliberately left a warning for; (a) alone fires on correct solves (it is
# true from ``delta`` = 3e-3 down, where the answer still tracks the physical
# shift to 1e-4).  The conjunction confines the behaviour change to stacks that
# BOTH carry the sliver and violate the theorem.
#
# ROUND 2 (2026-09-11) -- THE ARBITER, and why the conjunction alone is not
# enough.  ``docs/audits/VERIFY_PMMSTACK_SLIVER_WALLS_2026_09_11.md`` measured
# the conjunction BOTH ways and refuted its margins in both directions:
#
#   * FALSE POSITIVES.  On a passive stack, super-unity is just as often
#     ordinary under-convergence as a theorem violation.  110 of 648 realistic
#     staircase configurations (lossy substrate, theta 1.2-1.45, degree 6-10,
#     wall steps 0.36-3.6 nm) were REFUSED although their answer tracks the
#     exact ``delta -> 0`` limit to 0.35-8.8x the physical wall shift -- and
#     the first-named remedy silenced the refusal without moving the number
#     (1.03559 vs 1.03557), because it removes the ATTRIBUTION, not the error.
#   * FALSE NEGATIVES.  On a 120-delta x 3-degree grid the CORRECT population
#     reaches ``|R+T-1|`` = 9.87e-05 and the WRONG one reaches DOWN to
#     +7.14e-03 -- BELOW the 1e-2 bar -- so 8 of 660 wrong solves returned
#     unwarned, with errors to 2.8e-03.
#
# Both are one defect: super-unity is the DETECTOR but not the ATTRIBUTION.
# What attributes, measured, is ONE extra solve at the point where the library
# is about to raise anyway --
#
#     re-solve on the grid the prescribed ``min_feature`` would produce.
#     If the super-unity VANISHES and the answer MOVES far past the geometric
#     perturbation that snap describes, the sliver caused it -> REFUSE.
#     If it SURVIVES, the sliver did not -> fall through to the WARNING and
#     name degree / n_slices.
#
# so round 2 lowers the trigger to ``_SLIVER_TRIGGER_BAR`` (one decade above
# the correct population's measured envelope) and gates the refusal on that
# arbiter.  Measured on the shipped bars, 2026-09-11, both builds: false
# positives 110/648 -> 0/648, false negatives 8/660 -> 4/660, and the arbiter
# fires on 0 of 600 CONVERGED correct rows (it can only fire on a stack that
# already reads super-unity above the trigger).
#
# ROUND 3 (2026-09-11) -- "VANISHES" HAD TO BE READ RELATIVELY.  Round 2's
# verification (``docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md``,
# defect D-5) measured the one case the round-2 criterion cannot express: a
# stack whose SLIVER-FREE truncation super-unity already sits above the
# ABSOLUTE closure bar.  There the snapped super-unity can never reach that
# bar, so the arbiter says ``truncation`` however completely the snap restores
# the answer -- measured, it removed a 621x-5,181x super-unity and put the
# answer back on the sliver-free reference to err/delta = 0.0019, and round 2
# returned the wrong number (off by 1,811x-3,501x the physical wall shift, at
# R+T = 1.19) under a warning saying the prescribed remedy would silence
# nothing.  Round 1 refused all three rows, so it was a behaviour change.
#
# So the closure asks the snap to REMOVE most of the violation rather than to
# reach a fixed floor:
#
#     su_snapped <= max(_SLIVER_ATTRIB_CLOSURE,
#                       (max(R+T) - 1) * _SLIVER_CLOSURE_FRACTION)
#
# with the round-2 value kept as the lower arm, so the criterion is a widening
# and never a tightening.  Measured on both builds: the two censuses are
# unchanged row for row (false positives 0/648 with all 648 returned answers
# bit-identical to the unguarded ones, false negatives 4/660), and over 615
# arbitrated rows of eight devices the only verdicts that move are 85 rows of
# the D-5 class, every one of them WRONG by both continuity rules and the
# mildest of them off by 517.7x the physical wall shift.
```

### L183-186 -- `PMM_SLIVER_GUARD` -- the round-1 "BOTH conjuncts" description of when the guard fires -- the opposite of what the code does since round 4

*Left in the source:* what the switch restores, and that it is a switch and not a policy, with the trigger corrected to the geometric screen.

```text
#: FAIL-BEFORE SWITCH for the refusal (2026-09-11).  ``False`` restores the
#: pre-fix behaviour bit for bit: the super-unity WARNING below, and the wrong
#: answer returned.  A switch, not a policy -- the guard changes nothing on any
#: solve that does not trip BOTH conjuncts.
```

### L1358-1361 -- `_warn_stack_energy` docstring, the NON-FINITE arm -- "used to reach the far field and return ``tot = [nan, nan]``"

*Left in the source:* the same failure as the reason for the raise, plus the NaN-blind comparison that lets it through.

```text
    * NON-FINITE total -> **raise**.  A NaN/inf half-space index or
      permittivity used to reach the far field and return ``tot = [nan, nan]``
      completely silently (audit M3 2026-07-25): a one-sided ``>`` comparison
      is NaN-blind, so the tripwire never fired.
```

### L1377-1384 -- `_warn_stack_energy` docstring -- ROUND 4 -- what rounds 1-3 gated the arbiter on, and the 5.45.0 CI matrix that refuted it

*Left in the source:* the live rule: no super-unity precondition, the geometric screen decides, and why (a super-unity total is a property of the running BLAS kernel).

```text
      ROUND 4 (2026-09-11) removed the super-unity PRECONDITION on this arm.
      Rounds 1-3 asked the question only when the solve read above
      ``_SLIVER_TRIGGER_BAR``, so a sliver-corrupted answer that happened to
      read 1+1.15e-04 on the running BLAS kernel was returned silently while
      the SAME row on another kernel read 1+2.17 and was refused -- the
      release CI matrix for 5.45.0 measured exactly that.  What decides
      whether the arbiter runs is now the GEOMETRIC screen, which is a
      deterministic fact about the wall coordinates and the ``min_feature``.
```

### L1396-1398 -- `_warn_stack_energy` docstring, the ``'truncation'`` arm -- round 1's 110-of-648 false positives on this arm

*Left in the source:* what the arm does now -- warn above the bar with one sentence naming the sliver, silent below it.

```text
        and is NOT the cause), silent below it.  Round 1 refused 110 of 648
        realistic staircases here whose answers were within 0.35-8.8x the
        physical wall shift.
```

### L1399-1402 -- `_warn_stack_energy` docstring, the ``'unknown'`` arm -- that this arm is round 1's behaviour, unchanged

*Left in the source:* the rule itself, which is what a caller needs.

```text
      - ``'unknown'`` (the three extra solves could not be run: keyed /
        dispersive materials, no resolved source) -> ROUND 1's behaviour,
        unchanged: raise above ``_STACK_SUPERUNITY_BAR`` on a provably
        passive stack, warn below it.
```

### L1411-1414 -- `_warn_stack_energy` docstring, the within-layer arm -- ROUND 4 -- that this arm, too, used to be gated on super-unity

*Left in the source:* the measurement that makes the gate wrong (a liner 1.06e-03 wrong reads SUB-unity), stated as the reason there is no gate.

```text
      geometry the caller ASKED for.  ROUND 4 removed the super-unity
      precondition here too (defect R2-B: the measured liner that is
      1.06e-03 wrong reads ``R+T`` = 0.999221, i.e. SUB-unity, and the arm
      was silent on it).
```

### L1438-1444 -- `_warn_stack_energy`, before the arbiter call -- ROUND 4 -- the early return rounds 1-3 took below the trigger

*Left in the source:* the live rule and its reason, in three lines.

```text
    # ROUND 4 (2026-09-11): the arbiter is NOT gated on the super-unity
    # reading any more.  Rounds 1-3 returned here whenever the solve read
    # below the trigger, so a sliver-corrupted answer that happened to read
    # 1+1.15e-04 on the running BLAS kernel was returned silently while the
    # SAME row on another kernel read 1+2.17 and was refused.  The screen
    # inside the arbiter is a deterministic fact about the wall coordinates,
    # so it is what decides whether the three extra solves are paid.
```

### L1811-1846 -- `PMMStack.__init__`, the ``min_feature`` block -- the G2 two-fixture degree-scatter ladder, duplicated verbatim from :data:`_MIN_FEATURE_DEFAULT_FRAC`, and the superseded ``period*1e-5`` default

*Left in the source:* the sizing RULE (the pathology's width is absolute, so the threshold must sit above the collision scale) and a pointer to the constant that carries the measured ladder.  The derivation is not lost: TESTING_STANDARDS S5 wants it on the bar, and that is where the live copy is.

```text
        # WHY 1e-3 AND NOT 1e-5 (audit finding G2, 2026-09-12).  The snap is
        # the only thing standing between a staircased stack and the
        # MANUFACTURED-sliver pathology below, and the pathology has a MEASURED
        # width that is ABSOLUTE rather than a multiple of this knob: an
        # unsnapped cross-layer wall collision of size ``s`` corrupts the solve
        # for ``s`` at roughly ``1e-5 .. 1e-4`` of a PERIOD and is harmless
        # outside that -- which is what makes raising the threshold a cure at
        # all (a band that scaled WITH the knob could never be cleared by
        # raising it; the ladder below shows it does not, because at 1e-3 the
        # rungs at 1x .. 8x of the threshold are clean where at 1e-5 they were
        # the whole hazard).  The old default of ``period*1e-5`` snapped away only
        # the collisions that were already harmless and left the whole
        # dangerous decade exposed.  Measured on two independent fixtures (a
        # Si/SiO2 1.0/1.55 um pair at 12 deg and a TiO2-like 0.55/0.70 um pair
        # at 31 deg), sweeping ``s`` over a 0.3x..100x ladder of ``min_feature``
        # at degrees 10/14/18/22/26 with the refusal disarmed, and scoring
        # DEGREE-SCATTER at fixed ``s`` (a smooth drift with ``s`` is a
        # genuinely different geometry and is correct physics; an answer that
        # jumps between branches as ``degree`` changes is the pathology):
        #
        #     min_feature      rungs showing degree-scatter
        #     period*1e-5      6 of 11   (every rung from 1x to 8x; T0 reads
        #                                 0.2645 / 0.3082 / 0.1939 against a
        #                                 correct 0.199230 -- up to 55% wrong,
        #                                 scattering +-5% between adjacent
        #                                 degrees; on the first fixture the
        #                                 same band reaches T0 = 22.4 and 147.7)
        #     period*1e-4      1 of 11   (only the 1.0x rung, one degree of 5)
        #     period*1e-3      0 of 11   (every rung degree-independent to 7
        #                                 digits)
        #
        # and, decisively, where two settings both leave a collision unsnapped
        # they agree EXACTLY: s = 1.5e-4 reads 0.19839028 under 1e-5 and 1e-4,
        # s = 3e-4 reads 0.19755266 under both, s = 1e-3 reads 0.19366045 under
        # 1e-5 and 1e-3.  Raising the knob does not perturb the cases it does
        # not touch -- it only removes cells the union manufactured.
```

### L1890-1917 -- `PMMStack.__init__`, "WHAT REPORTS A SLIVER, EXACTLY" -- a comment correcting an earlier COMMENT -- what this block used to claim `_pmm_union_grid` reported

*Left in the source:* the corrected rule, stated once: which two reports exist, what each fires on, and the deliberate false-negative trade.

```text
        # WHAT REPORTS A SLIVER, EXACTLY (open item C, 2026-09-11; corrected
        # here 2026-09-11 by the verification of that fix).  This block used to
        # say "a cross-layer sliver left in the grid is now reported by
        # `_pmm_union_grid`".  It is NOT: `_pmm_union_grid` warns only about
        # the pairs it SNAPS, and a sliver left in the grid is by definition
        # one it did not snap.  There are two reports, neither of them that:
        #   * `_pmm_union_grid`'s warning -- fires when the snap MERGES pairs,
        #     and names the pairs and the max wall displacement;
        #   * `_sliver_refusal` (O-11, below) -- RAISES on a sliver LEFT in
        #     the grid, but only when super-unity above `_SLIVER_TRIGGER_BAR`
        #     on a provably passive stack is ATTRIBUTED to it by
        #     `_sliver_arbiter`: one re-solve on the `min_feature` grid the
        #     refusal prescribes, which must both clear the super-unity and
        #     move the answer far past the snap's own displacement (round 2,
        #     2026-09-11).  Where the arbiter says the sliver is NOT the cause
        #     the solve returns under the plain super-unity warning, which
        #     then says so; where the arbiter cannot run the round-1 decision
        #     stands.  A sliver whose answer still closes is not reported at
        #     all: that is the deliberate trade of S4.2 of
        #     docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md (a plain
        #     report would fire on correct solves), and its measured cost --
        #     8 rows in 660 under round 1, 4 under round 2 -- is the
        #     false-negative census in
        #     docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md S4;
        #   * the WITHIN-LAYER warning (`_within_layer_hazard`) -- a
        #     sliver-thin feature ONE layer owns is the geometry the caller
        #     asked for and no `min_feature` removes it, so it is warned
        #     about, never refused.
```

### L2716 -- `PMMStack._solve_conical`, the tensor gate -- "the old path was silently wrong for it"

*Left in the source:* the refusal and the silent-wrong alternative, as a property of the alternative.

```text
        # tensor is rejected loudly (the old path was silently wrong for it);
```

### L2977-2979 -- `PMMStack.solve`, the ``stabilize`` validation -- P3-31 -- what the covariant dispatch used to accept

*Left in the source:* the hazard in present tense: the covariant dispatch returns before the late vertical-path check, so the check must be eager.

```text
        # Validate ``stabilize`` EAGERLY, before ANY dispatch/early return
        # (audit P3-31): the covariant (uniform-slant) dispatch used to return
        # before the late vertical-path check, silently accepting garbage.
```

### L3025-3034 -- `PMMStack.solve`, the incidence guard -- audit M3 -- the negative efficiencies the JAX twin returned before the mirror guard existed

*Left in the source:* why the mirror is here rather than below the dispatch, what it skips on a traced input, and that the NumPy call below is then a no-op repeat.

```text
        # The NumPy branch's own ``_require_propagating_incidence`` sits below
        # the JAX dispatch, so the differentiable twin used to return BEFORE
        # it: a fully CONCRETE gain superstrate (n_sup = 1 - 1e-3j) reached the
        # far field and returned R+T = [-0.848, -0.863] -- negative
        # efficiencies, silently -- which is the audit-M3 2026-07-25 defect the
        # NumPy path was fixed for, still alive on the twin.  The concrete-only
        # mirror runs here so BOTH branches refuse it; a TRACED n_sup / angle
        # skips it exactly as the single-layer twins do (concretizing would
        # sever the trace), and the NumPy call below is then a no-op repeat of
        # two float comparisons.
```

### L3134-3141 -- `PMMStack.solve`, the out-of-plane routing -- that the pre-fix covariant-for-OOP routing was validated against engines sharing the same defect

*Left in the source:* the routing rule, the measured defect size, and the documented limitation that explicit 'covariant' still solves OOP.

```text
            # OUT-OF-PLANE slanted stacks route to CONVECTION (2026-07-14):
            # the covariant layout's discontinuous off-plane TM channel has a
            # ~0.1 per-order factorization defect under the corrected
            # factor-i physics (the pre-fix covariant-for-OOP routing was
            # validated against engines sharing the same defect; convection
            # and the RCWA tensor staircase now agree at ~4e-3 while
            # covariant is the outlier).  Explicit 'covariant' still solves
            # OOP (documented limitation; AUDIT_OOP_GENERATOR_FACTOR_I).
```

### L3209-3215 -- `PMMStack.solve`, the per-layer `stabilize='slices'` refusal MESSAGE -- a message retracting its own earlier wording

This one lived inside a STRING the interpreter executes -- the text of the
`NotImplementedError` a user sees -- so it was out of scope for a
documentation-only sweep (both fingerprints move when it changes) and was
rewritten separately, by WP-A22, with the fingerprints re-recorded in the same
commit.  It is the same retraction the `_solve_vertical_perlayer` docstring
below carries, reaching the user instead of the reader.

*Left in the source:* the corrected claim, stated positively and without the
retraction -- "min_feature IS live on this path even so: a window is itself a
union and contains the adjacent-slice collisions.  Vary min_feature directly
and compare, or run the shared path for the consensus."  Everything the caller
needs in order to act is still there; what went was the account of what an
earlier release's message said.

```text
                raise NotImplementedError(
                    "PMMStack.solve(stabilize='slices'): not applicable with "
                    "layer_grids='per-layer' -- the consensus probes perturb "
                    "the ONE SHARED union grid (n_slices re-slice / "
                    "min_feature) and read the spread across those probes, "
                    "and there is no shared union grid here: every layer "
                    "carries its own window grid.  (M4 / N-6, 2026-08-04: "
                    "this message used to say per-layer grids have 'no "
                    "cross-layer walls to perturb', which is WRONG -- a "
                    "window IS a union and contains the adjacent-slice "
                    "collisions, so min_feature is live here too.  Vary "
                    "min_feature directly and compare, or run the shared "
                    "path for the consensus.)")
```

### L3445-3447 -- `_solve_vertical_perlayer` docstring -- a comment correcting an earlier COMMENT -- this docstring's own retracted "``min_feature`` never enters" claim

*Left in the source:* the corrected statement: ``min_feature`` is the lever here exactly as on the shared path, with the pointer to the measured contract.

```text
        here exactly as on the shared path (M2 / N-6 -- this docstring used to
        say "``min_feature`` never enters", which was wrong; see the measured
        contract on :func:`_perlayer_window_grids`).  Interfaces between
```

### L3832-3837 -- `_slices_consensus_check`, the no-recipe fallback -- "previously the ENTIRE geometry-built path ... was silently unprotected"

*Left in the source:* what the fallback is and why it exists -- it needs no builder recipe, so it reaches the documented device route, which nothing else protects.

```text
            # No builder recipe -> the n_slices probe is impossible.  Fall back
            # to the UNION-GRID consensus (audit 2026-07-28, R-1): it needs no
            # recipe, so the guard is finally reachable on hand-added and
            # SegmentStackGeometry-built stacks -- previously the ENTIRE
            # geometry-built path (the documented device route) was silently
            # unprotected against the very pathology this check exists for.
```

### L4044-4045 -- `PMMStack.internal_field` docstring, Returns -- which release changed the H scale, and what the day-one method returned

*Left in the source:* a condensed ``versionchanged`` -- the scale is a public contract, so the fact that it changed stays, without the day-one description.

```text
            ``-i eta0`` scale (CHANGED in v5.14.3 from the modal-convention
            envelope the day-one v5.14.2 method returned).  ``E_x`` is the
```

### L4377-4380 -- `PMMStack.layer_absorption`, the per-material split -- P2-13 -- that the ezz channel used to be omitted from the split

*Left in the source:* the live rule: all three diagonal channels are summed, so an ezz-only-lossy segment is attributed.

```text
        # Im(exx)|Ex|^2 + Im(eyy)|Ey|^2 + Im(ezz)|Ez|^2 (audit P2-13: the
        # ezz channel used to be omitted, so a segment whose ONLY loss is
        # Im(ezz) -- a uniaxial absorber with the lossy axis along z -- was
        # dropped from the dict entirely), GLL-quadratured in x and Gauss-
```

### L5093-5114 -- `_solve_conical`, the half-space kz gauge bridge -- W7 F-C -- the measured pre-fix T = [0, 0] table on three lossy substrates

*Left in the source:* the gauge rule and the failure mode it prevents, stated in present tense, plus the two sibling bridges and the note that un-conjugating is the identity for a real eps.

```text
        # W7 F-C (2026-07-26), the Berreman-F-1 twin.  ``_kz_forward`` is a
        # PUBLIC-gauge helper (``Im(kz) >= 0`` for ``exp(-iwt)``), but
        # ``eps_sup``/``eps_sub`` are INTERNAL exp(+iwt) here (conjugated at the
        # top of this method) -- so a LOSSY half-space arrived double-
        # conjugated, ``sqrt`` landed in the 4th quadrant, the ``Im < 0`` flip
        # sent ``Re(kz) < 0``, and the ``Re(kz) > 0`` propagating mask inside
        # _assemble_jones_farfield SILENTLY ZEROED T.  Measured pre-fix on a
        # HOMOGENEOUS eps=2.25 slab (where the slant is a physical no-op, so
        # the vertical cascade is the exact oracle), P=0.30 um, 0.22 um deep,
        # wl 0.55 um, slant 0.35 rad, theta=0.3:
        #     n_sub 1.5+0.01j  ->  T = [0, 0]   (oracle [0.96586, 0.95613])
        #     n_sub 1.5+0.30j  ->  T = [0, 0]   (oracle [0.98578, 0.98047])
        #     n_sub 0.2+3.5j   ->  T = [0, 0]   (oracle [0.08641, 0.08135])
        # with ZERO warnings -- ``_warn_stack_energy`` only sees super-unity /
        # negative totals, and 0.014 is "passive".  An absorbing SUPERSTRATE
        # was worse: ``kz_inc = -1.14651`` tripped the "non-propagating
        # incidence" raise on a perfectly propagating medium.  Un-conjugating
        # restores the public gauge (identity for a real eps -> every lossless
        # solve is BYTE-UNCHANGED); this is exactly the ``kz_ord`` bridge in
        # ``_core._pmm_jones_oblique_core`` and rcwa's ``_forward_flux_kz``.
        # (The MODAL kz inside the cascade keeps the internal convention -- that
        # path is already correct.)
```
