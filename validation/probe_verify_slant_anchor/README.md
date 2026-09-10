# probe_verify_slant_anchor

The INDEPENDENT verification of `docs/audits/FIX_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md`
(items V1, V2 and O2).  Verdicts, defects and every number:
`docs/audits/VERIFY_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md`.

Nothing here reads a number from the fix's document or reuses one of its
fixtures: the periods, wavelengths, permittivities, duties, shears,
thicknesses, mounts, degrees, order counts and cells are all chosen in
`fixtures.py`, `t2_v2_sign.py`, `t3_v1_jax.py` and `t4_o2_census.py`.

## The three trees

`_lib.arm()` decides WHICH TREE answered from `lumenairy.__file__` alone,
refuses any tree that is not one of these three, and stamps tree, interpreter,
numpy, scipy, jax and thread caps into every JSON:

| arm | tree | commit |
|---|---|---|
| `post` | `C:/tmp/lum_vslant` | `2ec4359` -- the wave2 tip, holds the fix |
| `pre` | `C:/tmp/lum_vslant_pre` | `4a987e3` -- the fix's branch point |
| `rev` | `C:/tmp/lum_vslant_rev` | `2ec4359` with ONLY `c8cb563`, `006c031`, `b678747` reverted |

`pre` alone cannot attribute a moved byte to this fix (wave2 moved for other
reasons in between); `rev` is the surgical isolation.  Recreate them with:

```sh
git -C C:/tmp/lum_vslant worktree add --detach C:/tmp/lum_vslant_pre 4a987e3
git -C C:/tmp/lum_vslant worktree add --detach C:/tmp/lum_vslant_rev 2ec4359
cd C:/tmp/lum_vslant_rev
git checkout 4a987e3 -- lumenairy/elements/rcwa/_core.py \
    lumenairy/elements/pmm/twod_jones.py \
    lumenairy/elements/pmm/_jax_twod_jones.py
git revert --no-commit --no-edit 006c031
```

## The scripts

| script | task |
|---|---|
| `fixtures.py` | 46 `PMMStack` mounts + the surface hasher (13 surfaces each) |
| `t1_identity.py` | TASK 1 -- the bit-identity census, 1-D + 9 two-dimensional entries |
| `t2_v2_sign.py` | TASK 2 -- V2's sign: a HAND-BUILT z-staircase ladder, the analytic uniform-film oracle, the `PMM2DStackPure` cross-engine arm, opposite/net-zero walks, and the break attempts |
| `t2b_rcwa_cross.py` | TASK 2 -- the same decision against a THIRD engine (`RCWAStack`) |
| `t3_v1_jax.py` | TASK 3 -- the seven traced routes, the constant-tile no-op at five mounts, and the 13-candidate eighth-route hunt |
| `t4_o2_census.py` | TASK 4 -- 134 solves / 229 interfaces over every consumer of `_interface_smatrix_general`, census-armed and census-off |
| `t5_durability.py`, `t5b_durability_rest.py` | TASK 5 -- every numeric bar in the fix's gate, re-measured through the gate module's own helpers |
| `summarize.py` | re-reads the JSONs and prints the report's tables (`identity`, `o2`, `v1`, `v2`, `dur`, `dur2`, `census`) |
| `splice_durations.py` | merges the new node ids into `.test_durations` (pytest-split's own `--store-durations` would REPLACE the file) |
| `run_win.sh`, `run_wsl.sh`, `run_batch*.sh` | the arm drivers, thread caps pinned |

## Running one

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LUM_ARM_TREE=C:/tmp/lum_vslant python t1_identity.py
```

`LUM_ARM_TREE` is put ahead of the script's own directory on `sys.path`, so
`import lumenairy` resolves to the arm being measured.  Results land in
`results/<probe>.<arm>.<build>.json`.
