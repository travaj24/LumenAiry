# Testing standards — build-free assertions

Distilled from the v5.35.x release campaign (2026-08-13..16), in which five
consecutive release tags were refused by tests asserting per-build facts while
the library itself regressed zero times.  Full evidence chain:
`docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md` and the
`FIX_EME_CENSUS_2026_08_12.md` S1-S12 arc.

## The invariant

**Anything a build is entitled to move must be measured in the test, never
assumed from elsewhere.**

LAPACK/BLAS builds legitimately differ in the last bits.  Anything downstream
of a near-degenerate eigenproblem, an at-threshold comparison, or a large
reduction amplifies those bits into discrete differences: a mode found or not
found, a count of 0 vs 2, a bar crossed.  A test is per-build if and only if
its pass/fail boundary sits inside the cross-build spread of the quantity it
reads.  No amount of local greenness proves such a test sound — both release
shards that refused v5.35.1 and v5.35.2 were green on every local mount.

## The five fragile shapes (S1-S5)

| shape | example that burned a tag |
|---|---|
| S1 magnitude-ratio defect pin | "the open theta=0 defect manifests at >100x" (runner: 16.6x) |
| S2 pre-fix-referencing arm | "the pre-fix census flips under a 1-ULP nudge" (runner: stable) |
| S3 env-dependent precondition | "the spawn pool engaged" (runner: pricing rightly refused workers) |
| S4 floor bar | closure `< 1e-8` on a quantity whose cross-build spread reaches 2.6e-8 |
| S5 exact count/set of nondeterministic machinery | `abs(len(a)-len(b)) <= 2` on knife-edge mode censuses |

Two sub-shapes discovered late, both recurrences one level deeper:
parametrized injections whose *effect* is per-build (a fixed +0.012 stray that
lands harmlessly on some builds), and contracts conditioned on a fixed census
reading (`[0,0,0,0,2]` is a legal reading on some builds).

## The five build-free restatements

1. **Assert decisions, not readings.**  "A mode exists within the isolation
   radius of the 40-digit oracle's root" (radius >> spread), never "the census
   reads 201.8862661906".  Match by basin (half the minimum spacing the data
   itself exhibits), never by a converged tolerance.
2. **Independent oracles with derived bounds.**  mpmath / analytic /
   finite-difference truths, bounded by the oracle's own error floor
   (derived, e.g. `eps*|R|/h` for an FD) times documented decades — never by
   one build's residual.
3. **Engineer the state; don't hope the build produces it.**  Fail-before
   demonstrations construct their near-tie / stray / degeneracy through the
   current API, deriving the injected quantities from the running build's own
   measured geometry.  Scan a ladder if needed; hard-fail only when the ladder
   is exhausted AND the fixed path misbehaves — the fixed-path claim stays
   unconditional on every arm.
4. **Force preconditions; assert the gate separately.**  The pool engages
   because the test sets worker count and budget hooks, not because the box is
   big.  The pricing decision is its own two-sided claim (engaged when priced
   in, refused when priced out).  Never `pytest.skip` on a resource check —
   two skips silently removed five tests from the gate on exactly the runners
   that mattered.
5. **Bars need a gap on both sides.**  Measured cross-build envelope below
   (with the measurements and date in the comment), smallest real signal
   above, decades to each.  If no such gap exists the assertion is testing
   noise — restate the property one level up.  "It passed on the run I looked
   at" is not an envelope; it is one build's sampling (this mistake was made
   *inside* this campaign, twice).

## Durability rules

- Derived-at-runtime quantities track legitimate library evolution; nothing
  may pin a prior version's numbers.
- An envelope bar firing after an intentional algorithm change is the gate
  working: re-derive it once, with the new measurement dated in the comment.
- Every changed bar carries its derivation and its measured values.  A
  numeric constant in a test without a stated origin is a defect (two
  fabricated-adjacent constants were caught this campaign only by
  re-measuring — never by reading).
- Verification of a claimed fix means *re-measuring* its numbers, not reading
  its comments.  Right-conclusion-wrong-numbers is the most dangerous shape:
  it reads as authoritative and it passes.

## Process rules (release mechanics)

- The release gate is the FULL un-masked main-CI matrix on the merge, green
  end to end, BEFORE the tag.  Release-verify fail-fast cancels shards and
  masks independent failures (a cancelled shard hid a real library bug for
  one tag).
- One failing runner python proves nothing about the axis: the discriminator
  is the wheel's LAPACK, not the interpreter version (py3.10/3.11 were green
  locally on files their CI counterparts failed).  Deterministic injectors
  beat environment coverage.
