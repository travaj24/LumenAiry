<!-- lumenairy-history-doc
module: lumenairy/elements/coatings.py
ast_sha256: 8d3b29ce1e4a354976fbeae0b7c41a5e2171d1f96176d408e01cbd3ef3008413
token_sha256: 8444a18c1708aeea66a582cdbcdb221cc3ad41de4a50618dcfce3239a177ac53
pre_relocation_lines: 904
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/coatings.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/coatings.py`: the pre-v5.30 hard-coded `n_H` / `n_L` pair on
`v_coat_ar` and the measured reflectance table showing what it cost, plus the
smaller "used to fall through" / "no longer" clauses.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the `.. versionchanged:: 5.30` statement itself (it names a
DEFAULT that changed and a formula a caller may be relying on), the complex-angle
TMM factorization notes, and the measured residuals of the shipped design.

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
| L55-57 | `_normalize_polarization` docstring, ``Raises`` | "An unrecognised string used to fall through to the p branch" |
| L142 | `_tmm` docstring, the Snell factorization | "``n.imag`` is no longer dropped" |
| L551-561 | `v_coat_ar` docstring, ``versionchanged:: 5.30`` | the measured worse-than-bare-glass reflectance table the hard-coded pair produced |
| L769-770 | `_coating_sellmeier_array`, the shared wavelength guard | "a guard fix in one no longer silently skips the other" |

---

### L55-57 -- `_normalize_polarization` docstring, ``Raises`` -- "An unrecognised string used to fall through to the p branch"

*Left in the source:* the same failure as the reason for the refusal, in present tense -- including that the perfectly legal ``'te'`` is one of them.

```text
        ``'tm'``, ``'avg'`` (any case).  An unrecognised string used to fall
        through to the p branch silently, so a typo -- or the perfectly legal
        ``'te'`` -- returned the TM coefficient for a TE wave.
```

### L142 -- `_tmm` docstring, the Snell factorization -- "``n.imag`` is no longer dropped"

*Left in the source:* what the complex angle carries, stated as a property of the code.

```text
      (``n.imag`` is no longer dropped), so absorbing-layer phase
```

### L551-561 -- `v_coat_ar` docstring, ``versionchanged:: 5.30`` -- the measured worse-than-bare-glass reflectance table the hard-coded pair produced

*Left in the source:* what changed and why the old pair could only match one substrate -- the part a caller relying on the pre-5.30 numbers needs -- and the measured residuals of the shipped design below.

```text
        ``n_substrate`` is now READ (audit E-H6,
        ``AUDIT_ADVERSARIAL_CODEBASE_2026_07_25``).  Pre-v5.30 this
        function returned a hard-coded ``n_H = 2.3`` / ``n_L = 1.38``
        pair for every substrate, so the ``n_substrate`` argument was
        inert -- and by (2) that fixed pair is matched only to a
        substrate of ``(2.3/1.38)**2 = 2.778``.  Measured with this
        module's own TMM at 550 nm, the old stack was WORSE THAN BARE
        GLASS over the whole common range: R = 0.0986 vs 0.0337 bare at
        n_s=1.45, R = 0.0856 vs 0.0426 bare at n_s=1.52 (N-BK7, i.e.
        double the uncoated reflectance), R = 0.0515 vs 0.0744 at 1.75,
        and only R = 0.0 at n_s=2.778.  With (2) the design is exact for
```

### L769-770 -- `_coating_sellmeier_array`, the shared wavelength guard -- "a guard fix in one no longer silently skips the other"

*Left in the source:* the reason for sharing the guard, as a property rather than a change.

```text
    # wavelength identically (a guard fix in one no longer silently skips
    # the other).  Sellmeier is sign-symmetric, so the guard warns on a
```
