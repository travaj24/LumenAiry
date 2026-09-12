<!-- lumenairy-history-doc
module: lumenairy/analysis/through_focus.py
ast_sha256: 9e8a789cd2c3ec59c4589c351df78b8119590e8df3713e4663f6f5f9c358289b
token_sha256: 00ad9b40602f84e97ff74cd482b8b71c7c8da7a0371c08f440e082c62c0995aa
pre_relocation_lines: 1981
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/through_focus.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/through_focus.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

Four blocks moved, each a live design statement written as the release that
made it: the single-pass moment path, the fixed Strehl denominator, the
factored JAX kernel builder, and the JAX backend's metric reconciliation.  In
every case the reason stayed and the chronology moved -- "recomputing
`ideal_peak` inside the loop yields a non-physical mean Strehl ~1.09" is a
live hazard, whereas "Previously `ideal_peak` was recomputed inside the loop"
is a changelog entry.

The metric-reconciliation block is the interesting one.  It documented two
band-edge conventions the JAX backend adopted from `single_plane_metrics` --
an INCLUSIVE `R2 <= r*r` bucket mask, and `rms_radius` derived from
`beam_d4sigma` so a zero plane reports 0 rather than NaN.  Both conventions
are live, observable and pinned by named tests, so both stayed; what moved is
that this path used to be a hand-inlined twin.

The v4.12.2 z-invariant hoist note in `through_focus_scan`'s Notes stayed as
written: "bit-near-exact (abs err 0.0) vs the per-z
`angular_spectrum_propagate` reference" is the verification of the live
implementation, not a changelog entry about it.

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
| L195-200 | `single_plane_metrics` -- the single-pass moment path | the `Pre-fix` framing, the 21-plane scan figure and the bit-identity reassurance |
| L1084-1086 | `monte_carlo_tolerancing` -- the Strehl denominator | the `Previously ... which is what produced` framing |
| L1181-1187 | `_build_through_focus_kernel` | the `previously the kernel was a Python closure` framing |
| L1361-1385 | `through_focus_scan_jax` -- metric reconciliation | that this path was formerly a hand-inlined twin of `single_plane_metrics`, and the `previously dropped` / `was NaN on JAX` framing |

---

### L195-200 -- `single_plane_metrics` -- the single-pass moment path -- the `Pre-fix` framing, the 21-plane scan figure and the bit-identity reassurance

*Left in the source:* what the shared helper does and the measured cost of not sharing it

```text
    # ``_centroid_and_d4sigma``.  Pre-fix this built |E|**2 three times
    # (here, inside beam_centroid, inside beam_d4sigma) and the centroid
    # twice -- 81.4 ms/plane at N = 1024 against 52.4 single-pass, with
    # the whole 21-plane scan going 202.3 -> 128.8 ms/plane once the
    # transfer-function recurrence lands too.  Bit-identical: same
    # helper, same order of operations.
```

### L1084-1086 -- `monte_carlo_tolerancing` -- the Strehl denominator -- the `Previously ... which is what produced` framing

*Left in the source:* the rule and the non-physical Strehl it prevents

```text
    # Previously ``ideal_peak`` was recomputed inside the loop from the
    # per-trial perturbed ``E_exit``, which is what produced the
    # non-physical mean Strehl ~1.09 in example 09.
```

### L1181-1187 -- `_build_through_focus_kernel` -- the `previously the kernel was a Python closure` framing

*Left in the source:* why the builder is factored out and what a closure would cost -- stated as the reason not to re-inline it

```text
    v4.12.2 D4 closure -- previously the kernel was a Python closure
    inside `through_focus_scan_jax` that captured `kz_safe`,
    `propagating`, `E_fft_shifted` from the enclosing scope.  Each
    Python call rebuilt the closure and therefore re-traced.  The
    factored builder lets a module-scope cache hold the compiled
    kernel across calls keyed on `(Ny, Nx, dx, wavelength, bandlimit,
    dtype)` -- the structural pieces that participate in tracing.
```

### L1361-1385 -- `through_focus_scan_jax` -- metric reconciliation -- that this path was formerly a hand-inlined twin of `single_plane_metrics`, and the `previously dropped` / `was NaN on JAX` framing

*Left in the source:* both band-edge conventions as live statements, their observable consequences, and the named tests that pin them

```text
    # RECONCILED (AUDIT_V5_24_2 S3-19): this path was formerly a
    # hand-inlined twin of ``single_plane_metrics`` that diverged in two
    # documented, band-edge ways.  Both are now resolved by adopting the
    # shared function's canonical conventions:
    #   * power_in_bucket -- the inline twin masked with a STRICT
    #     ``r < bucket_radius``; ``single_plane_metrics`` ->
    #     ``radial_power_bands`` uses the canonical ``R2 <= r*r`` (``<=``).
    #     A pixel sitting EXACTLY on the bucket radius is now INCLUDED on
    #     the JAX backend too (previously dropped) -- so a bucket that
    #     lands on the on-radius ring gains that ring's energy.  Both
    #     forms compute the SAME centroid-relative distance
    #     ``(pix - centroid)*dx``, so this is the only bucket change.
    #   * rms_radius -- the inline twin computed an independent second
    #     moment guarded by ``I_sum > 0`` (leaving NaN on an all-zero
    #     plane); the shared path derives it from ``beam_d4sigma``
    #     (rms = sqrt(var_x + var_y)), which is the SAME second moment
    #     analytically (agree ~1e-15 in summation order) but returns 0 on
    #     a zero field.  A zero plane now reports rms 0 on BOTH backends
    #     (was NaN on JAX), which also removes the zero-field
    #     best_focus_spot divergence the S3-8 guard papered over.
    # Parity is pinned by tests/unit/test_through_focus_metric_parity.py
    # (smooth field) plus tests/unit/test_through_focus_bucket_boundary.py
    # (on-radius pixel + zero-field convention).  This entry point still
    # exposes no background/aperture pedestal controls (always whole-grid);
    # pass those via ``through_focus_scan(backend='numpy')``.
```
