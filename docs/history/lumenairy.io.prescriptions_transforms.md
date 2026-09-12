<!-- lumenairy-history-doc
module: lumenairy/io/prescriptions_transforms.py
ast_sha256: c2e8ccfe5cc46cfccd83523d32daea6c348298c95e52c3230ac1e285cf18457c
token_sha256: 8d4e112cc86638e7e64c2544695a190237ebd3e3b011aafdf37144967bdae383
pre_relocation_lines: 742
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/io/prescriptions_transforms.py`

This file holds the version-history narrative that used to live in
`lumenairy/io/prescriptions_transforms.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Five blocks.  `scale_prescription` promises geometric self-similarity, and
every block here records a key that was NOT scaled and therefore broke that
promise silently.  The MEASURED consequences (r_max 7.5 mm instead of
1.875 mm, period 2 um instead of 0.5 um, gap_before 10 mm instead of 2.5 mm)
stayed with their rules: they are what tells a future editor that adding a
new length-valued key without scaling it is a silent wrong answer.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L224-227 | `scale_prescription / Forbes-Q` | the "Pre-fix neither was touched" framing |
| L265-268 | `scale_prescription / diffractives` | the release tag and the "was untouched, so a scaled system kept" framing |
| L391-393 | `normalize_prescription` | the "pre-v4.11.2 this checked" framing |
| L404-407 | `normalize_prescription` | the audit id and the "Pre-fix they were plain surface dicts" framing |
| L530-533 | `split_prescription_legs` | the "Pre-fix this early return was silent" framing |
| L584-586 | `split_prescription_legs` | "(previously dropped)" |

---

### L224-227 -- `scale_prescription / Forbes-Q` -- the "Pre-fix neither was touched" framing

*Left in the source:* the dimensional argument and the measured failure.

```text
        # are functions of the dimensionless u = r / r_max).  Pre-fix neither
        # was touched, so a scaled Q-type surface kept its original freeform
        # sag on a rescaled base conic (measured at s = 0.25: r_max stayed
        # 7.5 mm instead of 1.875 mm).
```

### L265-268 -- `scale_prescription / diffractives` -- the release tag and the "was untouched, so a scaled system kept" framing

*Left in the source:* the rule and both measured failures.

```text
    # I7: the v5.32 diffractive payload is entirely lengths and was untouched,
    # so a scaled system kept the original DOE pitch and axial gaps (measured
    # at s = 0.25: period 2 um instead of 0.5 um, gap_before 10 mm instead of
    # 2.5 mm) -- i.e. the "self-similar" result was not self-similar at all.
```

### L391-393 -- `normalize_prescription` -- the "pre-v4.11.2 this checked" framing

*Left in the source:* the canonical flag and the leak a wrong key produces.

```text
        # is ``element_type='mirror'`` -- pre-v4.11.2 this checked
        # ``e.get('mirror')`` which is never set, making the filter a
        # no-op (mirrors leaked through to apply_real_lens).
```

### L404-407 -- `normalize_prescription` -- the audit id and the "Pre-fix they were plain surface dicts" framing

*Left in the source:* the stamping rule and the KeyError it prevents.

```text
        # ``element_type='surface'`` on the mirrored entries.  Pre-fix they
        # were plain surface dicts with no ``element_type``, so
        # ``generate_simulation_script`` -- which subscripts
        # ``elem['element_type']`` -- raised ``KeyError: 'element_type'`` on
```

### L530-533 -- `split_prescription_legs` -- the "Pre-fix this early return was silent" framing

*Left in the source:* the ambiguity the diagnostic removes.

```text
        # I7: say so.  Pre-fix this early return was silent, so a
        # prescription whose loader simply does not emit ``elements``
        # reported "one refractive leg, no folds" -- indistinguishable from a
        # genuinely unfolded design, and the docstring above promises the
```

### L584-586 -- `split_prescription_legs` -- "(previously dropped)"

*Left in the source:* the rule and the workflow it serves.

```text
            # and OUT OF the mirror (previously dropped), so the folded-
            # design walking workflow can reconstruct the inter-leg
            # geometry.  all_th[i] is the gap from element i to element i+1.
```

