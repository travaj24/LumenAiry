<!-- lumenairy-history-doc
module: lumenairy/elements/bor/_orient.py
ast_sha256: 9245fb6b3bf3ef3e1cccf19a46bfcce12362f2939331278db29d9c1cba9f6e3f
token_sha256: 4571967b21f3fabfcf36b0a4e6f22995d3c787ea5ba0c52613c03b174563cae2
pre_relocation_lines: 296
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- the on-cut band comparison and the forward selector moved to the shared lumenairy/_branchcut.py leaf; each engine keeps its own derived scale (bit-identical, WP-B11a item 1)
-->

# Version history -- `lumenairy/elements/bor/_orient.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/_orient.py` -- the two `ROUND 2 RESTATEMENT` blocks, the
second of which retracts the thin-end margin printed in the table above it.
Each block is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the 39-rung near-cutoff ladder, the two-sided population
table, the unit-safety argument for the `k0` floor, and the harmlessness
measurement -- `docs/TESTING_STANDARDS.md` S5 wants those on the bar, and
`tests/unit/test_fix_bor_multilayer_guards.py::test_band_two_sided_population`
re-measures the table on every running build.

One correction, not a move: the table prints `0.98 dec` for the thin SIGNAL end
and the restatement below it measured the union minimum at **0.38** decades.
The source now says so where the table is read, instead of ten lines later.

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
| L132-137 | `orient_band_scale` docstring, the floor | the ROUND 2 RESTATEMENT framing on the unit-safety floor |
| L169-186 | `orient_band_scale` docstring, the SIGNAL margins | a ROUND 2 RESTATEMENT retracting the 0.98-decade thin end printed in the table above it, and the "what changed" meta-paragraph |

---

### L132-137 -- `orient_band_scale` docstring, the floor -- the ROUND 2 RESTATEMENT framing on the unit-safety floor

*Left in the source:* the whole argument -- the floor never binds, why it is kept anyway, and the EME sibling that shipped the dimensioned literal.

```text
    ROUND 2 RESTATEMENT -- THE FLOOR IS A UNIT-SAFETY FLOOR, NOT A MEASURED
    BAR, AND IT NEVER BINDS.  ``max|q|`` over a layer's spectrum is dominated
    by the largest transverse eigenvalue, ``~ N / Rbig``, which exceeds ``k0``
    on any grid that resolves the wavelength.  Over the 122 layers the
    verification measured, and over the 135 measured again in round 2
    (``validation/probe_fix_bor_round2/r8_band_sides.py``, which includes an
```

### L169-186 -- `orient_band_scale` docstring, the SIGNAL margins -- a ROUND 2 RESTATEMENT retracting the 0.98-decade thin end printed in the table above it, and the "what changed" meta-paragraph

*Left in the source:* the corrected figure and the reason it is the honest one -- the SIGNAL rows are a SAMPLE property of the swept population, and the minimum over a union of populations is the smaller of the two.

```text
    ROUND 2 RESTATEMENT -- THE SIGNAL MARGINS ARE SAMPLE-SCOPED, AND THE THIN
    END IS 0.38 DECADES, NOT 0.98.  Re-measured
    (``validation/probe_fix_bor_round2/r8_band_sides.py``), the two SIGNAL rows
    ARE the minima OF THE POPULATION THE GATE SWEEPS -- ``m`` = 0/1/2 x ``k0``
    = 2.0/3.5, 359 and 362 physically propagating modes -- reproduced here to
    all seven digits.  What they are not is a property of the BAND: widening
    the population by ONE ``k0`` rung (adding 0.8, giving 407 and 410 modes)
    lowers the minimum to **3.7752e-05 and 3.7752e-08**, i.e. 3.58 and **0.58
    decades**, and the independent verification's own lossy population reaches
    **2.3820e-08**, i.e. **0.38 decades**.  The minimum over a union of
    populations is the smaller of the two, so the honest figure for the thin
    end is 0.38 decades.

    That is the 2-D peer's round-4 correction in this module: a margin measured
    on one fixture family is a SAMPLE property, not a library one.  The
    decision is unchanged and still right, for the reason below and re-measured
    on every arm; what changed is that the gate now sweeps the wider ``k0``
    population and its floor is derived from the measured envelope over it.
```
