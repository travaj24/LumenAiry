# PLAN -- Wave 5: the open work left by the 2026-09-11 adversarial audit (HANDOFF_2026_09_14.md section 4)

Binding plan for the wave that closes the handoff's open items. Base: 4bf26c5e (the
5.47.0 release commit). Integration branch: `wave5/audit-leftovers`. Every item is
built by one agent, independently re-measured by another, and merged only after the
merged tree's own gates are green. Rules of the house apply: decisions not readings;
two-sided derived bars with stated margins; a decision that moves with the BLAS kernel,
the thread count or the build is a defect; tests assert invariants unconditionally and
premise-gate any claim that a pathology reproduces on the running arm (the CI runners
are a per-job mix of AMD EPYC 9V74 / Zen 4 and EPYC 7763 with older wheels); one shared
implementation per numerical kernel; explicit-path `git add` only; no push from an
agent worktree.

## Wave 5.1 (parallel builds)

| item | handoff | owner branch | what ships |
|---|---|---|---|
| A. WP-B12 FGA reference plane | 4.1 (P1) | `fix/wp-b12-fga-reference-plane` | project the last-surface ray state to the exit-vertex plane at the four `fga.py` sites; own oracle ladder; re-pin the eight test files; re-score the caustic route and measure (not decide) the `aberrated` condition on the H2 f/5 fixture |
| B. Newton pool hang | 4.1b (P1) | `fix/newton-pool-broken-fallback` | the broken-pool fallback reaches the serial path (`shutdown(wait=False, cancel_futures=True)` + reap); the pool-breaking race between the two chain arms found and closed; worker-side diagnostics |
| C. Multibranch blow-up + zeta envelope | 4.2, 4.3 (P2) | `fix/multibranch-zeta-envelope` | the `1/sqrt|J|` blow-up bounded and surfaced in the uniform diagnostics with a derived threshold; `_ZETA_EXTRAPOLATION_MAX` re-derived two-sided on >= 2 optics; the member-selection pass measured; `_AIRY_TAIL_CELLS` clipping where the extrapolation exceeds a derived bound |
| D. Known reds + warning attribution | 4.4 (mechanical part), 4.6 (known reds) | `fix/known-reds-and-stacklevels` | the T3-1 BLAS-build classification test, the order-dependent glass-validity one-shot pin, and the c7/c8 halo pins root-caused (never masked); `carrier.py` literal stacklevels -> `caller_stacklevel()` |

## Wave 5.2 (parallel verifications of 5.1), then merge into the integration branch.

## Wave 5.3 (hygiene, after 5.2): the WP-B11 items not reached (4.5) -- the rest of the
`rcwa/_core.py` split, the two import cycles, the `lenses <-> lenses_maslov` back-edge,
the direct-matrix MFT branch, `_collins_transport` on JAX, the near-focus exact-kernel
table -- each with its own verifier.

## Decisions reserved for the maintainer (handoff 4.7): presented with measured
recommendations at the end of the wave, not taken here. Items that turn on those
decisions (the caustic route once WP-B12 lands, `edge='gray'`, `transport='collins'`,
`sphere_normal='analytic'`, the `_sphere_normal` clamp, `gap_kernel='auto'` near a
focus, the odd-N centring, `replica_fill='zero'`, whose diagnostic the in-glass gap-leg
warnings are) are MEASURED in this wave and left switchable.

## Release: 5.48.0 (WP-B12 moves FGA answers on every curved-last-surface prescription,
a behaviour change with a Migration note); `NEXT_REMOVAL_VERSION` slips 5.48 -> 5.50 in
the release commit.
