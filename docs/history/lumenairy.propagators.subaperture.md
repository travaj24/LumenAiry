<!-- lumenairy-history-doc
module: lumenairy/propagators/subaperture.py
ast_sha256: d1030e6626743f3fb86f6da6a9a7414ef56eef1ba336c2c4c104d1a1ff0d22b8
token_sha256: bd4166da9c7f79307fe37fac94257ab3a35a8d527d154c3b78848d1eaa1cb922
pre_relocation_lines: 614
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/subaperture.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/subaperture.py`.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

Every block here is the same shape, and it is the shape the WP-A17 SWEEP-1
follow-up pass was asked to close: a **live guard whose comment explained
itself by naming the release that added it** ("Pre-fix X happened", "Pre-4.12
the dispatcher only passed ...").  The hazard X is still reachable -- the guard
is the only thing preventing it -- so the source now states X in the present
tense, as what goes wrong WITHOUT the guard, together with every measurement
that sizes it.  What moved is the release attribution and the
"bit-identical to pre-fix" reassurance that travelled with it.

The first block is also the V6 duplication pattern: two `.. versionchanged::`
directives, v5.2 and v5.2.3, stacked on the same two kwargs, the second
partly superseding the first.  A caller reading them in order learns the
contract twice and has to work out which half still applies.  The source now
states it once, including the fallback branch that still emits the v5.2
`UserWarning` when the automatic image-plane mapping is unavailable.

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
| L79 | `make_patch_grid` -- `centred=False` | the `pre-v5.30` name for a LIVE option |
| L88 | `make_patch_grid` -- layout offset | the past tense on a live option (`the legacy layout pushed`) |
| L131-132 | `make_patch_grid` -- the un-centred branch | the `Pre-v5.30 layout` / `Bit-for-bit preserved` framing |
| L192-215 | `combine_patch_fields` -- two stacked `versionchanged` directives | the v5.2-then-v5.2.3 sequence: opt-in kwargs added, then computed automatically, with the warning first introduced and then 'silenced for the typical-call case' |
| L238-241 | `combine_patch_fields` -- centre selection | the `Legacy callers ... the bit-for-bit pre-v5.2 path` framing |
| L366-376 | `propagate_subaperture_asymptotic` -- image-plane mapping | the `v5.2.0 surfaced the bug as a UserWarning -- v5.2.3 now FIXES it` sequence |
| L440-446 | `propagate_subaperture_asymptotic` -- the source grid | the `Pre-4.13.2 E_in was silently replaced` framing |
| L526-535 | `propagate_subaperture_asymptotic` -- the kernel call | two stacked call-site fixes: the pre-4.10 TypeError that left this path dead on import, and the 4.10 3-D `np.stack` the 4.11.1 patch undid |
| L537-542 | `propagate_subaperture_asymptotic` -- the LG projection | the `Pre-4.13.2 the source_amplitudes were hard-coded` framing |

---

### L79 -- `make_patch_grid` -- `centred=False` -- the `pre-v5.30` name for a LIVE option

*Left in the source:* the option and its formula

```text
        * ``False``: the pre-v5.30 layout, ``c_i = -W/2 + w/2 + i*step``
```

### L88 -- `make_patch_grid` -- layout offset -- the past tense on a live option (`the legacy layout pushed`)

*Left in the source:* what each layout does with the surplus coverage

```text
        lower) -- i.e. the legacy layout pushed the entire surplus past
```

### L131-132 -- `make_patch_grid` -- the un-centred branch -- the `Pre-v5.30 layout` / `Bit-for-bit preserved` framing

*Left in the source:* what the branch computes

```text
        # Pre-v5.30 layout: first patch flush with the box's low edge, all
        # the surplus coverage past the high edge.  Bit-for-bit preserved.
```

### L192-215 -- `combine_patch_fields` -- two stacked `versionchanged` directives -- the v5.2-then-v5.2.3 sequence: opt-in kwargs added, then computed automatically, with the warning first introduced and then 'silenced for the typical-call case'

*Left in the source:* the live contract in one directive -- what the windows centre on, who supplies it, when the source-plane fallback and its ``UserWarning`` still fire, and what ``None`` means

```text
    .. versionchanged:: 5.2
        v5.2 (AUDIT_V4_13_1 Part 2 P1-F closure): two new optional
        kwargs ``image_centres`` and ``image_half_widths`` let the
        caller pass image-plane (post-magnification + tilt) patch
        coordinates that the partition-of-unity windows centre on.
        Pre-v5.2 the windows were always centred on
        ``patch_grid.centres``, which are SOURCE-plane positions --
        correct only for unit-magnification, no-tilt geometries.  When
        ``image_centres`` is ``None`` (default) the legacy
        source-plane behaviour is preserved bit-for-bit; the caller
        :func:`propagate_subaperture_asymptotic` emits a
        ``UserWarning`` advising the user to supply mapped centres for
        non-unit-magnification systems.

    .. versionchanged:: 5.2.3
        v5.2.3 (AUDIT_V4_13_1 P1-F substantive closure):
        :func:`propagate_subaperture_asymptotic` now computes the
        image-plane centres / half-widths internally from the system
        ABCD and passes them through these kwargs automatically, so
        the v5.2.0 ``UserWarning`` is silenced for the typical-call
        case.  The opt-in kwargs still work and take precedence over
        the auto-computed values for callers who want to override the
        paraxial mapping (e.g. when the prescription's nominal image
        plane is not the desired output plane).
```

### L238-241 -- `combine_patch_fields` -- centre selection -- the `Legacy callers ... the bit-for-bit pre-v5.2 path` framing

*Left in the source:* which coordinates each branch picks and what the fallback means physically

```text
    # v5.2 (AUDIT_V4_13_1 Part 2 P1-F closure): pick the centres /
    # half-widths used for the partition-of-unity windows.  Legacy
    # callers see ``None`` and inherit ``patch_grid.centres`` /
    # ``patch_grid.half_widths`` -- the bit-for-bit pre-v5.2 path.
```

### L366-376 -- `propagate_subaperture_asymptotic` -- image-plane mapping -- the `v5.2.0 surfaced the bug as a UserWarning -- v5.2.3 now FIXES it` sequence

*Left in the source:* where the mapped centres go, the unit-magnification degeneracy, and the one corner case in which the ``UserWarning`` still fires

```text
    # half-width onto its image-plane footprint.  v5.2.0 surfaced the
    # source-plane-centred-window bug as a ``UserWarning`` -- v5.2.3
    # now FIXES it by routing the mapped centres through
    # :func:`combine_patch_fields`'s ``image_centres`` /
    # ``image_half_widths`` kwargs.  For a unit-magnification system
    # (``A == 1, B == 0``) the mapped centres equal the source-plane
    # centres bit-for-bit, so v5.1 / pre-v5.2 numerics are preserved
    # exactly.  The warning is retained only as a fallback for the
    # corner case where the ABCD computation itself fails (degenerate
    # / coord-break-heavy prescriptions without a clean paraxial
    # imaging chain).
```

### L440-446 -- `propagate_subaperture_asymptotic` -- the source grid -- the `Pre-4.13.2 E_in was silently replaced` framing

*Left in the source:* what the grid is for and exactly what is discarded without it

```text
    # 4.13.2 (P1-NEW-B): build the source-plane coordinate grid for
    # E_in so we can project the actual input field onto the LG basis
    # per patch.  Pre-4.13.2 ``E_in`` was silently replaced by a unit
    # fundamental Gaussian at every patch -- structured input (off-
    # axis Gaussian, vortex, Airy) was completely discarded.  Mirrors
    # the v4.11.2 fix in
    # :func:`hf.propagate_huygens_fresnel_through_prescription`.
```

### L526-535 -- `propagate_subaperture_asymptotic` -- the kernel call -- two stacked call-site fixes: the pre-4.10 TypeError that left this path dead on import, and the 4.10 3-D `np.stack` the 4.11.1 patch undid

*Left in the source:* the signature the kernel actually wants and the unpacking failure that follows from getting it wrong

```text
        # Propagate from this patch's source point.  4.10: the actual
        # `propagate_modal_asymptotic` signature uses
        # `source_amplitudes` / `pupil_amplitudes` (not
        # `source_lg_amps` / `pupil_lg_amps`) and `s2_grid_x` /
        # `s2_grid_y` (not `output_grid`).  Pre-4.10 calls raised
        # TypeError on first invocation -- the subaperture path was
        # dead on import.  4.11.1: feed the (Ny, Nx) meshgrids
        # directly; the 4.10 patch built a 3-D ``np.stack(...,axis=-1)``
        # array and then tried to unpack it 2-ways, which always raised
        # ``ValueError: too many values to unpack`` for any Ny != 2.
```

### L537-542 -- `propagate_subaperture_asymptotic` -- the LG projection -- the `Pre-4.13.2 the source_amplitudes were hard-coded` framing

*Left in the source:* what the projection does and what hard-coding it costs

```text
        # 4.13.2 (P1-NEW-B): project the *actual* input field onto the
        # LG basis centred at this patch's source point.  Pre-4.13.2
        # the source_amplitudes were hard-coded to a unit LG_{0,0},
        # so any structured E_in was silently replaced by a fundamental
        # Gaussian and the function returned a Gaussian output
        # regardless of the input.  Mirrors v4.11.2 hf.py fix.
```
