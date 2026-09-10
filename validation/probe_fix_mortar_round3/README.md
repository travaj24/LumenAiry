# ROUND-3 probes -- PMM2D staggered per-layer mortar

Measurement suite for `docs/audits/FIX_PMM2D_MORTAR_ROUND3_2026_09_11.md`,
which fixes **DEFECT V1** (P1, ship-blocking) and answers the **S5.4** ask of
`docs/audits/VERIFY_PMM2D_MORTAR_ROUND2_2026_09_11.md`.

Every script pins the worktree's own `lumenairy` (it inserts the repo root at
the front of `sys.path` and then ASSERTS `lumenairy.__file__` is under it --
without that, a probe launched from `validation/...` silently imports whatever
else is on `PYTHONPATH`, which happened once here) and writes its JSON beside
itself, tagged `win` / `wsl`.

Run from the worktree root:

```
python validation/probe_fix_mortar_round3/<script>.py win
wsl -e bash -lc "cd /mnt/c/tmp/lum_mortar3 && \
  ~/lumvenv/bin/python validation/probe_fix_mortar_round3/<script>.py wsl"
```

| script | what it measures |
|---|---|
| `_fixtures.py` | the shared stacks: the VERIFY audit's v8 device, an out-of-plane layer next to a plain UNIFORM SPACER, and the healthy both-out-of-plane / both-slanted controls |
| `r1_mechanism.py` | **the mechanism of DEFECT V1, to closure**: SVD, numerical rank, null-space dimension, WHICH block column the near-null right vector lives in, the residual of the returned answer and the right-hand side's component along the LEFT null space -- at `M` = 4..7, on the mixed and both-out-of-plane fixtures |
| `r2_populations.py` | the three populations at the generalized site: 28 ORDINARY stacks, the SLIVER ladder with `PMM2D_STAG_MIN_SEG_GUARD` lifted, and synthetic singular / inconsistent operands -- read on `rcond` AND on the residual, which is what shows `rcond` has no two-sided gap there |
| `r3_residual_screen.py` | the residual as an instrument: exact vs one-probe-vector forms, a three-layer stack whose middle interface has BOTH sides promoted, an `M` = 8 point, and a first (load-polluted) cost reading |
| `r4_cost.py` | what the screen costs, min-of-N interleaved, against the FLOP argument |
| `r5_degradation_band.py` | the S5.4 degradation curve re-measured on an INDEPENDENT fixture at `M` = 6/7/8 against the exact 1-D `PMMStack`, plus the ordinary-geometry census that binds the band's upper edge |
| `r6_v2_conforming.py` | DEFECT V2: where the width contract is raised from (three constructors, independently) and how `delta`-insensitive a CONFORMING per-layer stack really is |
| `splice_durations.py` | adds the new gates node ids to `.test_durations` from a `--durations=0` log, dict-union and sorted, preserving the file CRLF |

## Bit-identity

The round-2 verifier's own 30-fixture / 165-hash harness is reused rather than
rewritten, against a pristine copy of the three changed files:

```
mkdir -p /c/tmp/lum_r3pre && cp -r lumenairy /c/tmp/lum_r3pre/
for f in lumenairy/elements/pmm/{_core,twod_staggered,stack2d_pure}.py; do
  git show HEAD:$f > /c/tmp/lum_r3pre/$f; done
V1_TAG=r3pre  PYTHONPATH=/c/tmp/lum_r3pre  V1_EXPECT_ROOT=/c/tmp/lum_r3pre \
  python validation/probe_verify_mortar_round2/v1_bitid.py
V1_TAG=r3with PYTHONPATH=/c/tmp/lum_mortar3 V1_EXPECT_ROOT=/c/tmp/lum_mortar3 \
  python validation/probe_verify_mortar_round2/v1_bitid.py
python validation/probe_verify_mortar_round2/v1_compare.py r3with r3pre
```

Result 2026-09-11: **165 / 165 identical, 0 warning-set differences.**

## A trap recorded because it produces a FALSE PASS

A probe that monkeypatches `_pc._guarded_mortar_solve` must accept the
round-3 `screen=` keyword and FORWARD it, or it silently turns the generalized
site's residual decision back into the round-2 `rcond` one (and, if it does not
accept the keyword at all, raises `TypeError` mid-solve -- which is how this
was caught).
