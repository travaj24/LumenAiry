<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic.py
ast_sha256: 24449fce106838b55c226094d5d8dc1e624f49c77bc87bbd17045cd6d56561d7
token_sha256: 2a5d7180a0e478970cd2884db2a2eb1e2f7599f6fb0ddd1d43f5c6ee4fb4e6f4
pre_relocation_lines: 893
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/asymptotic.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asymptotic.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

Six blocks moved, and they are all of one kind: a paragraph that explains a
present-day design decision by narrating the release that made it.  The
decisions stayed; the releases are here.

Three things deliberately did NOT move:

* **The `maslov_tracking` derivation.**  The proof that `arg det M` is
  confined to `(-pi, +pi)` -- hence that `'principal'` is the unique globally
  analytic continuation -- and the measurement that shows what the legacy
  raster unwrap really detects (742 of 4209 pixels sign-flipped, 60 of 289
  shared points opposite on two grids, `'principal'` reproducing them to
  3.4e-11) is the justification for a live default and for two live legacy
  options.  It stays.
* **The measured accuracy envelope** (W6-A3/A9/A6) and the **pupil-mode
  power caveat**: both bound what the live evaluator can be trusted to
  return.
* **The shell-retention note** at Section 6.  It reads like history ("Pre-v5.1.0
  ... monkey-patches ...") but it is a *do not move this function* instruction
  whose reason is a live Python semantic: intra-module name lookups resolve
  against `__globals__`.

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
| L120-127 | `<module>` -- module layout | that the v5.1.0 split was 'purely mechanical' and bit-for-bit identical |
| L433-436 | `propagate_modal_asymptotic` -- the v2-linear note | a comment correcting an earlier COMMENT: the v5.30 W6-A4 note claimed the v2-linear part WAS removed, on two on-axis fixtures |
| L492-495 | `propagate_modal_asymptotic` -- radiometric normalisation | what the weight was before v5.46 and what the old text called it |
| L510-524 | `propagate_modal_asymptotic` -- vectorisation closure | the pre-v4.15 per-pixel warm-started Newton chain, the bit-equal pins it broke and the release notes that recorded the relaxation |
| L547-551 | `propagate_modal_asymptotic` -- rank guard | the `Pre-fix` framing of the bare unpack error |
| L595-598 | `propagate_modal_asymptotic` -- batching banner | the release/roadmap tag and the 'from the scalar path' framing |

---

### L120-127 -- `<module>` -- module layout -- that the v5.1.0 split was 'purely mechanical' and bit-for-bit identical

*Left in the source:* the re-export guarantee and the whole reason ``propagate_modal_asymptotic`` is defined in the shell

```text
The split is purely mechanical -- every previously-public name remains
importable from ``lumenairy.propagators.asymptotic`` and behaves
bit-for-bit identically.  ``propagate_modal_asymptotic`` is defined
*in this shell* (not in a submodule) so its body resolves
``_solve_envelope_stationary_batch`` and the batched helpers against
this module's globals.  This preserves the pre-v5.1.0 monkey-patch
contract relied upon by
``tests/unit/test_audit_fixes_v4_14_1_agent_a.py``.
```

### L433-436 -- `propagate_modal_asymptotic` -- the v2-linear note -- a comment correcting an earlier COMMENT: the v5.30 W6-A4 note claimed the v2-linear part WAS removed, on two on-axis fixtures

*Left in the source:* the live statement and the whole measured argument for it -- ``a3 u3 + a4 u4`` is linear in the INTEGRATION variable, and dropping it moved the PSF 700 um and shrank the peak 88x

```text
           **The v2-linear part is NOT removed (v5.46, audit Y1).**  The
           v5.30 W6-A4 note used to say it was, on the strength of two
           ON-AXIS measurements where ``|a3| + |a4| <= 1.7e-09`` waves.
           That is a property of those fixtures, not of the convention:
```

### L492-495 -- `propagate_modal_asymptotic` -- radiometric normalisation -- what the weight was before v5.46 and what the old text called it

*Left in the source:* the live weight, the exact free-space-chart verification and the ratio it produced

```text
    4.2e-11.  Before v5.46 the weight was ``|det ds1/dv2|`` with no
    ``1/lambda``, i.e. the output was ``i lambda sqrt(|det J|)`` times the
    true field -- wavelength- AND field-point-dependent, not the "arbitrary
    constant" the old text described.
```

### L510-524 -- `propagate_modal_asymptotic` -- vectorisation closure -- the pre-v4.15 per-pixel warm-started Newton chain, the bit-equal pins it broke and the release notes that recorded the relaxation

*Left in the source:* the live algorithm (one batched cold-start solve from ``v2_centre``) and the hazard that makes the cold start mandatory -- a warm-started chain lands in wrong-saddle basins at the grid edges, where the ``|b_quad| > 700`` overflow guard zeroes the pixel silently

```text
    **v4.15 vectorisation closure.**  Prior to v4.15 this function
    used a per-pixel Python loop with a warm-started Newton chain.
    The chain landed in *wrong-saddle basins* near the grid edges,
    where the overflow guard ``|b_quad| > 700`` silently zeroed those
    pixels.  v4.15 switches to a single batched call into
    :func:`_solve_envelope_stationary_batch` followed by
    :func:`_compute_M_b_batch` and the batched polynomial-substitution
    helpers (cold-start Newton for every pixel from ``v2_centre``).
    The new body finds the physical saddle uniformly and produces
    strictly more non-zero pixels at grid edges -- physically MORE
    correct, but breaks the pre-v4.15 bit-equal pin against the
    warm-start reference (the v4.12.0 and v4.14.0 bit-equal pins
    were relaxed to property pins in the same v4.15 patch; see
    ``docs/audits/AUDIT_V4_14_2_2026_05_17.md`` Part 3.5 closure and
    ``docs/release_notes/.release_notes_v4_15_agent_a.md``).
```

### L547-551 -- `propagate_modal_asymptotic` -- rank guard -- the `Pre-fix` framing of the bare unpack error

*Left in the source:* what the guard prevents, which is the reason it exists

```text
    # v5.30 (audit W6-A5): explicit rank guard.  Pre-fix the row-major
    # unpack below (``Ny, Nx = s2x_arr.shape``) raised a bare
    # ``ValueError: too many values to unpack (expected 2, got 3)`` from
    # the middle of the function for any ndim >= 3 input -- an internal
    # error message with no mention of the offending argument.
```

### L595-598 -- `propagate_modal_asymptotic` -- batching banner -- the release/roadmap tag and the 'from the scalar path' framing

*Left in the source:* what the section does

```text
    # v4.15 (ROADMAP #1) -- batched cold-start Newton + batched M/b/poly
    # ------------------------------------------------------------------
    # Per-pixel ``continue`` from the scalar path becomes element-wise
    # ``valid`` masking here.
```
