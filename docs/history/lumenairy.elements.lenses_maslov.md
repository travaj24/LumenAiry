<!-- lumenairy-history-doc
module: lumenairy/elements/lenses_maslov.py
ast_sha256: 5f32b2e80cf42ae4aaafa706e50b6ddb03f6c27367e3e1539c0f3380f5d0d27f
token_sha256: e0c70c5fa612e2d4f315a0a88f3860472b795cf52cc4629f4ea30d11d5d81659
pre_relocation_lines: 4484
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
-->

# Version history -- `lumenairy/elements/lenses_maslov.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/lenses_maslov.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: the measured derivations of the live constants
(`docs/TESTING_STANDARDS.md` S5) -- the `local_quadrature` window geometry's
three-part argument with its 9.1 / 40 % / 8.09e-02 measurements, the
`_GRAM_COND_MAX` conditioning numbers (`rank(A) = 65` of 70, `cond(G) =
6.18e+18`, the 0.869-wave null-space spread), the E-H1 lattice measurement
behind `affine_transform`, and the auto-`poly_order` / auto-`n_v2` ladders.
The public docstring's live migration statements (`fold_split`,
`output_plane_distance`, `normalize_output`, the anamorphic support note) also
stayed.


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
| L1138-1144 | `_tukey_window` docstring | that this window was "formerly written out THREE times" and which two call sites carried the copies |
| L1163-1177 | `_solve_fit` docstring, the Conditioning gate | a docstring retracting its own earlier v5.21 justification ("``A`` is well-conditioned ... so squaring the condition number in ``G`` is safe") |
| L1691-1698 | the mirror-in-surfaces guard | "Pre-fix a hand-built prescription ... would slip past" -- the hazard written as a release note |
| L1920-1926 | the ``stop_index`` validation | "where a negative index used to read as 'non-entrance stop' and a float would have raised a bare TypeError" -- the WP-A2 fix as a note about what the code used to do |
| L2740-2756 | the coarse->fine PHASE upsample | "pre-3.5.6 used line-by-line np.unwrap then cubic zoom" and "3.5.6 fix: ..." -- the change written as a two-release narrative |
| L4183-4188 | the out-of-box sample handling in the v2 integrand | "the pre-fix ``np.clip`` folded every out-of-box sample onto the box edge" |

---

### L1138-1144 -- `_tukey_window` docstring -- that this window was "formerly written out THREE times" and which two call sites carried the copies

*Left in the source:* that this is the one definition, the two formula constants it reproduces, and the bit-identity claim

```text
    This is the ONE definition of the Maslov quadrature window, which was
    formerly written out THREE times -- the ``tukey(n)`` helper in
    :func:`_integrate_quadrature` (which built its own ``linspace(-1, 1, n)``)
    and the ``_tuk(u)`` closure in :func:`_integrate_levin`.  It reproduces both
    former formulas operation-for-operation (``1.0 - alpha`` taper start, the
    ``0.5*(1 + cos(pi*(|u| - (1 - alpha))/alpha))`` roll-off), so every routed
    call is bit-identical."""
```

### L1163-1177 -- `_solve_fit` docstring, the Conditioning gate -- a docstring retracting its own earlier v5.21 justification ("``A`` is well-conditioned ... so squaring the condition number in ``G`` is safe")

*Left in the source:* the settled statement -- when squaring the condition number is safe, when it is not, and why the fallback ladder cannot see the failure

```text
    v5.21 (M-P5 follow-up): normal-equations Cholesky (``G = A^T A``; solve
    ``G coef = A^T RHS``) instead of the ``gelsd`` full-SVD ``lstsq``, which is
    O(M^3) with tiny ``M`` (70 at poly_order=4) rather than O(n_rays M^2).  A
    caller sweeping the SAME optic can precompute ``gram_factor`` and pass it
    in (only the cheap ``A^T RHS`` GEMM + back-substitution then re-run per
    field).  Returns ``coef`` (M, k).

    Conditioning gate
    -----------------
    The v5.21 justification -- "``A`` is a normalized tensor-Chebyshev
    Vandermonde, well-conditioned and ~1.5x oversampled, so squaring the
    condition number in ``G`` is safe" -- does NOT hold on every chart, and the
    ``LinAlgError`` fallback ladder below cannot see the failure: a numerically
    positive-semidefinite but RANK-DEFICIENT ``G`` factors happily and returns
    an arbitrary member of the solution set.
```

### L1691-1698 -- the mirror-in-surfaces guard -- "Pre-fix a hand-built prescription ... would slip past" -- the hazard written as a release note

*Left in the source:* the hazard itself, present tense: which prescriptions slip past the shared fold check and what the Maslov leg would do with them

```text
    # v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard
    # from ``apply_real_lens_traced``.  Pre-fix a hand-built prescription
    # with ``surfaces[i]['is_mirror']=True`` (or ``glass_after='MIRROR'``)
    # would slip past the shared ``_check_no_silent_fold_drop`` (which
    # only inspects ``prescription['elements']``), and the Maslov leg
    # would silently treat the mirror as a refractor with the wrong
    # sign.  Fail loudly with the same mirror-specific message as
    # ``apply_real_lens_traced``.
```

### L1920-1926 -- the ``stop_index`` validation -- "where a negative index used to read as 'non-entrance stop' and a float would have raised a bare TypeError" -- the WP-A2 fix as a note about what the code used to do

*Left in the source:* the live rule and what normalising buys

```text
    # WP-A2: validate the key the way ``apply_real_lens`` now does -- an
    # out-of-range or non-integer ``stop_index`` RAISES with a Section 2
    # prefix rather than being int()-ed into a silent warning path (where a
    # negative index used to read as "non-entrance stop" and a float would
    # have raised a bare TypeError).  Normalising also makes
    # ``stop_index=-1`` on a 2-surface lens mean surface 1, as Python
    # indexing does, instead of tripping the non-entrance warning.
```

### L2740-2756 -- the coarse->fine PHASE upsample -- "pre-3.5.6 used line-by-line np.unwrap then cubic zoom" and "3.5.6 fix: ..." -- the change written as a two-release narrative

*Left in the source:* what the upsample does, the seam failure it avoids, and the < pi validity condition it inherits

```text
        # Phase upsampling: pre-3.5.6 used line-by-line np.unwrap then
        # cubic zoom of the unwrapped phase.  Line-by-line unwrap is
        # fragile near caustics / focal saddles where the phase wraps
        # along both axes; the resulting cubic-interpolated phase had
        # ~4% RMS errors from line-mismatched seams.
        #
        # 3.5.6 fix: interpolate the COMPLEX exp(i*phase) directly via
        # cubic zoom of its real and imaginary parts, then take
        # ``angle()``.  This avoids any 2-D phase-unwrap step
        # (and therefore any unwrap-induced seams) at the cost of
        # only being well-behaved when the local phase variation
        # between adjacent coarse pixels is < pi -- which is the same
        # condition the original line-unwrap silently relied on.
        # For Maslov outputs that satisfy that bound (typical
        # refractive systems with output_subsample <= 8), the new
        # path agrees with the OLD output to ~0.3% RMS while
        # eliminating the caustic-seam artifact.
```

### L4183-4188 -- the out-of-box sample handling in the v2 integrand -- "the pre-fix ``np.clip`` folded every out-of-box sample onto the box edge"

*Left in the source:* the hazard in the present tense and the rule that follows from it

```text
    # S2/P2 (audit): the pre-fix ``np.clip`` folded every out-of-box sample
    # onto the box edge and still counted it at the full unclipped cell area,
    # over-counting by up to 3 decades on a weakly-curved chart.  Samples
    # outside the fitted chart carry no information (the Chebyshev recurrences
    # are not even accurate there), so DROP them: clip to keep the polynomial
    # evaluation in its accurate range, and zero the contribution.
```
