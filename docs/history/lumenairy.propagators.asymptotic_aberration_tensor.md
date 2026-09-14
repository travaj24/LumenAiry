<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic_aberration_tensor.py
ast_sha256: 47444f941cb16b7e1f48f1443f7a0caef386c3477e645ae481bc8f2771df64dd
token_sha256: add271903ca2908e491a983471736750a46c8e688208906bcb6b1c7b95ff1e97
pre_relocation_lines: 1376
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-B7 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, WP-A4 sec. 6 items 3-8 + VERIFY-B1 F1/F2): the Y4 fused basis evaluation and hoisted Newton factor, the aberration_tensor mode/waist caches, the S6 gate's k1-slope statistic and mean-plus-spread chart sizing, the JAX screen's chief-ray displacement term, and the S9 FFT kernel clip
-->

# Version history -- `lumenairy/propagators/asymptotic_aberration_tensor.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asymptotic_aberration_tensor.py`.  Each block is
reproduced **verbatim** under the source line it came from in the
pre-relocation file.

This module is mostly NOT history, and very little moved.  Its long comment
blocks are the **measured derivations** that `docs/TESTING_STANDARDS.md` S5
requires a numeric bar to carry, and they stayed in the source in full:

* the W4-T1 chirp-Nyquist argument `n >= 4*extent*v_max/lambda` and the
  fringe-rate convergence measurements behind it (`_required_sigma_grid_n`);
* the 7-rung accuracy/cost table that sets `_SIGMA_GRID_N_MAX_DEFAULT = 256`;
* the 32-point probe-grid calibration for the default `w_o`;
* the W4-T2 sign-flip / ptp table behind `curvature_matched_basis`;
* the Gram-matrix clamp measurements on `sigma_grid_extent`.

Four blocks did move.  Two are release chronology attached to behaviour that
is now simply the behaviour (the Chebyshev-helper move, and the two "removed
limitation" paragraphs in `aberration_tensor`'s Notes); one is a comment
correcting an earlier COMMENT (`van_vleck_weight`'s "VERIFY-A4: the
pre-v5.46-final wording gave the SQUARED factor"); and one is the dimensional
post-mortem of a superseded default, kept here because the live argument that
replaces it -- no function of `M` alone can supply an image-plane width --
carries its own measurement and stayed in the source.

The `aberration_tensor` Notes rewrite is the one place where prose changed
shape rather than shrinking: the two paragraphs recorded a defect that the
CURRENT branch rule exists to prevent, so the source now states the property
("the closed-form branch is a point-sampling functional; every `(p, 0)` mode
gives the same number and every `l != 0` mode gives zero") as a present-tense
do-not-widen warning instead of as release history.

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
| L41-44 | `<module>` -- Chebyshev alias import | which release moved the helpers and which roadmap item it was |
| L113-114 | `AberrationTensorResult.van_vleck_weight` | a comment correcting an earlier COMMENT -- that the pre-v5.46-final wording gave the SQUARED factor |
| L840-865 | `aberration_tensor` -- Notes | the pre-4.9 all-zero ell != 0 limitation and the 4.9-5.28 ell = 0 degeneracy, both as release chronology |
| L941-955 | `aberration_tensor` -- the measured `w_o` default | the superseded ``1/sqrt(lambda_max(Re M))`` default: its dimensional analysis and the two measurements (15x too narrow, 255x too wide) that condemned it |

---

### L41-44 -- `<module>` -- Chebyshev alias import -- which release moved the helpers and which roadmap item it was

*Left in the source:* what the aliases are for

```text
# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# Chebyshev helpers moved to lumenairy._math.chebyshev; binding the
# new public names to the legacy underscore-prefixed locals keeps the
# existing call sites in this module unchanged.
```

### L113-114 -- `AberrationTensorResult.van_vleck_weight` -- a comment correcting an earlier COMMENT -- that the pre-v5.46-final wording gave the SQUARED factor

*Left in the source:* the live migration algebra (``L_legacy = L * |det J| / van_vleck_weight`` and both derived forms), which is what the attribute exists for

```text
        (VERIFY-A4: the pre-v5.46-final wording gave the SQUARED factor for
        ``L`` itself.)  ``None`` on the sigma-grid branch, which evaluates
```

### L840-865 -- `aberration_tensor` -- Notes -- the pre-4.9 all-zero ell != 0 limitation and the 4.9-5.28 ell = 0 degeneracy, both as release chronology

*Left in the source:* the degeneracy itself, restated as a present-tense property of the closed-form branch plus an explicit do-not-widen warning; the branch rule; and the ~1e-14 independent-quadrature check

```text
    **Pre-4.9 limitation removed.**  Pre-4.9 the projection at the
    chief image collapsed to the constant term of the LG output
    polynomial, which is identically zero for any ``ℓ ≠ 0`` mode
    ((σ_x + j·σ_y)^|ℓ| · Laguerre has no constant term).  That made
    coma ``(1, ±1)``, astigmatism ``(0, ±2)``, tilt ``(0, ±1)``, and
    every other ℓ ≠ 0 entry of the returned tensor silently zero,
    even when the underlying aberration was present.  4.9 fixes this
    by doing the actual σ-integration via a small output-plane grid
    and a numerical LG projection (``propagate_modal_asymptotic`` +
    ``decompose_lg``).

    **v5.28.x (audit W3-T3) -- ℓ = 0 degeneracy removed.**  4.9-5.28
    kept EVERY ℓ = 0 output mode on the closed-form chief-ray path,
    which is a point-sampling functional whose whole output-mode
    dependence is ``conj(LG_k)`` evaluated at one point:  that equals
    ``N_{p,0} = sqrt(2/(π w_o²))`` for every ``(p, 0)`` mode, so
    ``L`` came back BIT-IDENTICAL for piston / defocus / spherical /
    every higher ``(p, 0)`` channel (and across separate single-mode
    calls, with no warning).  Only the pure ``[(0, 0)]`` request --
    whose LG polynomial genuinely IS that constant, and which is the
    documented cross-backend contract of
    ``aberration_tensor_lg00_jax`` -- still uses the closed form; every
    other request routes to the σ-integration, whose overlaps an
    independent from-scratch LG quadrature reproduces to ~1e-14
    relative.  The two paths carry different overall scales (sampling
    vs. overlap integral); see the note at the branch.
```

### L941-955 -- `aberration_tensor` -- the measured `w_o` default -- the superseded ``1/sqrt(lambda_max(Re M))`` default: its dimensional analysis and the two measurements (15x too narrow, 255x too wide) that condemned it

*Left in the source:* the live argument -- that NO function of ``M`` alone can supply an image-plane width -- together with its own measurement, and the instruction to measure it

```text
            #
            # v5.29 (audit W3-T3b).  The pre-fix default was
            # ``1/sqrt(lambda_max(Re M))``, which is DIMENSIONALLY a pupil
            # quantity: ``M``'s entries are ``J^T J / w_s^2 + I / w_p^2 -
            # i·pi·H_phi`` with ``J = ds1/dv2`` [m/direction-cosine], so
            # ``M`` is in 1/direction-cosine^2 and its inverse square root
            # is an ANGLE -- the effective pupil acceptance -- used as if
            # it were metres.  Being dimensionally wrong, its error had no
            # fixed sign: measured 1.01e-4 "m" against a true field waist
            # of 1.559e-3 m (15x too NARROW, so ``4·w_o`` sampled only the
            # flat central 10 % of the field) on the validation singlet at
            # w_p = 0.02, but 255x too WIDE (grid entirely outside the
            # validity box, every entry of L exactly 0) at w_p = 0.05.
            #
            # Nor can any function of ``M`` alone be right: the image-plane
```
