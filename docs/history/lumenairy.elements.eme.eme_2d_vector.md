<!-- lumenairy-history-doc
module: lumenairy/elements/eme/eme_2d_vector.py
ast_sha256: 8377da84ed9c9bf14e2cc667a550d1339d7b91617ab0bc930c5cdfe9ee2ee5e5
token_sha256: 88df6d7edbb97001e572b47abd49999afa4fc28f89a1917f7fd023c6e7c5c67e
pre_relocation_lines: 1376
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/elements/eme/eme_2d_vector.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/eme/eme_2d_vector.py`: the `ROUND 2 (D13)` passage recording
which floor the forward/backward band used to carry and retracting an earlier
claim in its own docstring, and the "used to / pre-fix / previously" clauses in
front of the W6 guards and the guarded-polish block.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the `_CENSUS_BAND` / `_STRUCTURAL_SAT` derivations and their
measured populations, the guarded-improvement argument, and the band definition
itself.

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
| L257-271 | `_forward_indices_vector` docstring | ROUND 2 (D13) -- the literal 1.0 floor this site used to carry, and a retraction of an earlier claim in this same docstring |
| L528 | `_CENSUS_BAND` | "keeps the pre-fix code path exactly" |
| L639 | `_sigma_min_invpow` | "iters <= 0 used to raise UnboundLocalError" |
| L745 | `_sigma_min_dispatch` docstring | "it used to fall through to dense silently" |
| L831-832 | `_mode_census` docstring, the census band | "used to be decided in the last bits of the LAPACK reduction" |
| L922-932 | `_mode_census`, the guarded polish | "Before this guard ... was held by the pre-fix path" -- the build-shard recall loss the guard removes |
| L1253-1254 | `ref_2d_modes_vector`, the ``sigma`` guard | "sigma was silently INERT" |
| L1324-1325 | `strips_to_eps_xy` docstring | "un-covered y rows used to be left silently at eps = 0" |
| L1366-1367 | `eps_xy_to_strips`, the tensor-grid refusal | "a tensor (Nx, Ny, 3, 3) grid used to die on the shape unpack" |

---

### L257-271 -- `_forward_indices_vector` docstring -- ROUND 2 (D13) -- the literal 1.0 floor this site used to carry, and a retraction of an earlier claim in this same docstring

*Left in the source:* what the band IS and what its floor decides, which is the only part that describes the running code.

```text
    This site has ALWAYS carried the correct relative band; 5.45.1 made that
    band THE one definition (:func:`lumenairy.elements.eme._branch.cut_band`)
    and moved the module's two scalar siblings onto it, which until then
    carried an exact-zero pin instead.

    ROUND 2 (D13): the band's FLOOR moved from a literal 1.0 to ``|k0|``, so
    the split no longer depends on the caller's unit system, and ``k0`` is
    passed in from :func:`strip_vector_modes`, which has it.  On a spectrum
    whose top already exceeds ``|k0|`` -- every ordinary strip -- the two
    floors give the identical number and this site's answer does not move; the
    floor is what decides on a sub-``k0`` spectrum, which is where the literal
    was wrong.  (The earlier claim here that ``cut_band`` "computes the
    identical quantity this line always did" was also false for an EMPTY ``ky``
    array, where the pre-5.45.1 inline expression raised ``ValueError``:
    verification D16.)"""
```

### L528 -- `_CENSUS_BAND` -- "keeps the pre-fix code path exactly"

*Left in the source:* the scope statement itself: outside the band the code path is untouched.

```text
#   pre-fix code path exactly.
```

### L639 -- `_sigma_min_invpow` -- "iters <= 0 used to raise UnboundLocalError"

*Left in the source:* why the accumulator is pre-initialised.

```text
    cur = 0.0                # AUDIT W6: iters <= 0 used to raise UnboundLocalError
```

### L745 -- `_sigma_min_dispatch` docstring -- "it used to fall through to dense silently"

*Left in the source:* the refusal and the silent fallback it replaces.

```text
    other value raises (it used to fall through to dense silently)."""
```

### L831-832 -- `_mode_census` docstring, the census band -- "used to be decided in the last bits of the LAPACK reduction"

*Left in the source:* the instability itself, as the reason the band is adjudicated.

```text
    reading of a near-threshold candidate -- used to be decided in the last bits
    of the LAPACK reduction.  A candidate whose rank-drop lands inside
```

### L922-932 -- `_mode_census`, the guarded polish -- "Before this guard ... was held by the pre-fix path" -- the build-shard recall loss the guard removes

*Left in the source:* what the guard does and the measured shard case as the reason for it, in present tense.

```text
            # adopted iff it is a DEEPER zero than the minimiser's stop.  Before
            # this guard the step was one-way -- a polish that strayed onto a
            # neighbouring wiggle of the min-of-branches, or landed somewhere
            # ``_mode_reading`` could not evaluate, discarded a candidate whose
            # pre-polish reading was a clean accept, and did so SILENTLY.  That
            # is a build-dependent RECALL loss of exactly the kind this block
            # exists to remove: measured on the 2026-08-13 ubuntu py3.10 shard,
            # the genuine Nx=16 mode at 201.8868828456 (FD distance 0.074,
            # sigma_min 2.7e-15, structural ratio 8.0e-15 -- a mode by every
            # oracle) was held by the pre-fix path at its stop 201.8862661906
            # and DROPPED by this branch, while our mounts keep it.
```

### L1253-1254 -- `ref_2d_modes_vector`, the ``sigma`` guard -- "sigma was silently INERT"

*Left in the source:* the inertness as a present fact and the measurement that shows it.

```text
        # AUDIT W6 (sibling of the scalar oracle): sigma was silently INERT
        # without k -- measured bit-identical results for sigma=1e9.
```

### L1324-1325 -- `strips_to_eps_xy` docstring -- "un-covered y rows used to be left silently at eps = 0"

*Left in the source:* the same failure as the reason for the contract.

```text
    un-covered y rows used to be left silently at ``eps = 0``, which the vector
    oracle's ``1/(k0 eps)`` turns into ``inf``/``NaN``."""
```

### L1366-1367 -- `eps_xy_to_strips`, the tensor-grid refusal -- "a tensor (Nx, Ny, 3, 3) grid used to die on the shape unpack"

*Left in the source:* the bare error the named refusal replaces.

```text
        # AUDIT W6: a tensor (Nx, Ny, 3, 3) grid used to die on the shape unpack
        # with a bare "too many values to unpack (expected 2, got 4)".
```
