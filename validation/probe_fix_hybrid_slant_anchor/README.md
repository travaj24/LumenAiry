# `probe_fix_hybrid_slant_anchor` -- the hybrid 2-D PMM's frame-anchor fix

Evidence for `docs/audits/FIX_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md`
(defect **D1** of `docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10.md` S3.4):
`PMM2DStackHybrid.jones_transmission()` and
`per_order_amplitudes('transmission')` were **FRAME-referenced** on a stack
holding a slanted PATTERNED layer -- the per-order phase that maps the sheared
frame's exit plane back to the lab was never applied.  `R`, `T` and the
REFLECTION Jones were exact, so no energy check and no existing test could see
it.

## The discipline

* every script imports `_lib`, whose **`arm()` decides `fix` / `base` from
  `lumenairy.__file__` itself**, never from a flag, and stamps the resolved
  path, interpreter, numpy, scipy and thread caps into its JSON.  A mis-set
  `PYTHONPATH` cannot mislabel a run; an unexpected tree is a hard exit;
* `FIXSTAGE=pre|post` names which side of the library edit a run is on and goes
  into the JSON filename, so `p1_census_pre_base_win.json` and
  `p1_census_post_fix_win.json` sit side by side;
* `FIXTAG=win|wsl` separates the two builds.

## The arms

| arm | tree | what it is |
|---|---|---|
| `fix` | `C:/tmp/lum_hyb` (`fix/hybrid-slant-transmission-anchor`) | the worktree the fix was made in |
| `base` | `D:/Metacept/.../Lumenairy` | the **READ-ONLY** main clone.  Its `stack2d.py`, `stack2d_pure.py`, `twod.py`, `twod_jones.py` and `rcwa/_core.py` are **byte-identical** to the pre-fix worktree (verified by `diff`), so it is a valid independent "pre-fix" arm AND the reference for every bit-identity hash |

## The probes

| script | what it measures |
|---|---|
| `p1_census.py` | reproduces D1 on the verification's own fixture (a HALF-period walk, `n_orders = 9`) and censuses every public output at normal / oblique 25 / conical 25-40: `solve`'s R / T / reflection Jones, `jones_transmission`, `per_order_amplitudes` on both ports, plus a QUARTER-walk arm at normal incidence (where a half walk makes `P_m` real and the conjugate arm degenerate), the two NULL rows, and the reachability of `retain_internal` / `internal_field` / `layer_absorption` / `solve_vs_wavelength` / `pmm_jones_2d` |
| `p2_composition.py` | the COMPOSITION rule: a film below, a film above, two slanted halves (the SUM), a slanted UNIFORM layer and a CONSTANT-tile cell below (which must contribute NOTHING), and the SCOPE case -- a PATTERNED layer below a slanted one, placed three ways |
| `p3_bit_identity.py` / `p3_compare.py` | 21 fixtures x 7 arrays, sha256, against the read-only main clone.  `p3_compare.py` diffs two JSONs and marks any hash outside `slanted_patterned_*.{jones,per_order}_transmission` as UNEXPECTED |
| `p4_test_bars.py` | every quantity `tests/unit/test_fix_hybrid_slant_transmission_anchor.py` asserts, at the TEST fixture's own size, on both builds.  Assumes the fix is present; reconstructs "no anchor" / "conjugate" / "half the sum" by re-referencing the shipped anchor |
| `p5_restated_bars.py` | re-measures the FOUR thin bars the verification flagged in `tests/unit/test_pmm2d_staggered_slant.py` (the cost ratio idle AND under load, the m4c direction ratios, the m4 ladder, the b3 conj/none range).  Imports the fixtures FROM the test file, so it re-measures the file's own bars |

## Running them

```
cd validation/probe_fix_hybrid_slant_anchor
PYTHONPATH=C:/tmp/lum_hyb FIXSTAGE=post FIXTAG=win \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -W ignore -u p1_census.py          # and p2 .. p5
```

the WSL arm:

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_hyb/validation/probe_fix_hybrid_slant_anchor \
  && PYTHONPATH=/mnt/c/tmp/lum_hyb FIXSTAGE=post FIXTAG=wsl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/lumvenv/bin/python -W ignore -u p1_census.py"
```

and the PRE-fix arm, which is the read-only main clone (never the worktree,
which now carries the fix):

```
PYTHONPATH="D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy" \
  FIXSTAGE=pre FIXTAG=win python -W ignore -u p1_census.py
python p3_compare.py results/p3_bit_identity_post_fix_win.json \
                     results/p3_bit_identity_pre_base_win.json
```

## Traps these probes hit first, recorded so the next run does not

1. **A WHOLE-period walk degenerates the anchor.**  With `t d = px`,
   `exp(2 pi i m t d / px) = 1` for every order: the per-order factor becomes a
   single global phase and no measurement can tell it from any other.  The
   verification's `v8` hit this; every probe here walks HALF a period.  At
   NORMAL incidence a HALF walk degenerates one level further -- `P_m =
   exp(i pi m)` is REAL, so "x P_m" and "x conj(P_m)" coincide -- which is why
   `p1` adds a QUARTER-walk normal-incidence arm (`P_m = i^m`).

2. **A CONSTANT-tile patterned layer must NOT be anchored.**  It stores a slant
   but `_build_layer_modes` short-circuits it to `_homogeneous_modes` before
   the slant is read, so it never enters a frame.  Summing "every layer with a
   slant keyword" breaks it by `1.371e+00` on an answer that is exact at
   `0.000e+00`.  Measured in `p1` (NULL 2) and `p2` (case F).

3. **The first fixture was numerically unstable, and only in some
   `(n_orders, mount)` cells.**  `px = 1.2 um` with a cell containing
   `eps = 1.0` (= the superstrate) and a `+1` order at `|alpha| = 0.989`
   against the superstrate's cut-off of 1.0 drove the slanted-layer-over-a-film
   cascade to `sum R + T = 2.6e+27` -- while the SINGLE-layer rows on the same
   fixture were fine.  `p2` / `p4` / the test file therefore run `px = 1.0 um`,
   where every order is clear of a cut-off and no cell value equals a
   half-space `eps`.  A one-mount smoke test would have missed it.

4. **`p1`'s `pmm_jones_2d` reachability call is expected to raise
   `_EnergyError`** on that (deliberately unstable, `px = 1.2 um`) fixture; the
   probe catches it and records the reachability answer, which is what it is
   there for: `pmm_jones_2d` returns `(orders, R, T, jones_REFLECTION)` and
   exposes no transmission surface at all, so the single-layer entry never had
   the defect.
