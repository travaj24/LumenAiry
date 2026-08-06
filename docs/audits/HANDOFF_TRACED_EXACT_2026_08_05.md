# Handoff — exact traced chain, speed/memory, and the open 121 grid question
**2026-08-05.** Nothing in this session is committed to git. `git status` shows
the full working set; `lumenairy/elements/pmm/stack.py` is **NOT part of this
work** (pre-existing modification, PMM BLAS-pool threading — leave it alone or
review it separately).

---

## 1. Library defaults that CHANGED (all verified, all regression-backed)

| what | from | to | why |
|---|---|---|---|
| `newton_fit` | `'polynomial'` | `'auto'` → spline (CPU) / polynomial (GPU) | accuracy identical, spline parallelises 1.29–1.31× vs 1.10–1.13× |
| `gap_kernel` | `'fresnel'` | `'auto'` → **exact on every backend** | exact matches an independent ASM oracle to 1e-13 vs Fresnel 3e-6…1.28 |
| `_FOCUS_STANDOFF_ZR` | 6.0 z_R (was `_BRIDGE_ZR_FACTOR`) | **0.8 z_R** | readout waist error 8.84% → 0.60% vs analytic complex-q truth |
| Newton pool threshold | single 200 000 | two-tier: 200 000 cold / 8 000 warm | measured; see §3 |

`final_leg` was **not** changed and should not be: `'auto'` already selects the
exact leg for 121 (measured `na_exit` = 0.405 against the 0.15 threshold).
The problem was that `validation/repro_traced_carrier_121/_d121_common.py`
*overrode* it with `final_leg='paraxial'`. Now fixed — and `final_leg` is part
of the `_chainA_*.npz` **cache key**, because it previously was not: switching
the leg silently reloaded the old paraxial field and looked like a no-op.

### The exact kernel is now backend-generic
`_exact_tf_2d_xp` mirrors `_fresnel_tf_2d_xp`'s scaffolding (`_freq_1d_bld`,
`_tf_phase_to_H`), so CuPy/JAX get the exact kernel too — validated on a real
JAX backend at 3–5e-16 vs the NumPy path. It was NumPy-only purely because the
physics was validated there first; a NumPy-only exact kernel would have meant
other backends silently running the *paraxial* transfer function.

---

## 2. What is still PARAXIAL (and cannot easily stop being)

The **Sziklas–Siegman co-moving frame** — the `m = R_out/R` scaling,
`z_eff = z·R/R_out`, the `1/m` amplitude, the piston. It derives from the
dilation symmetry of the paraxial wave equation, which `sqrt(k² + ∇⊥²)` breaks,
so there is no drop-in exact replacement. Everything else on the 121 path is
exact: ray trace, in-glass ASM, exact-sphere carrier reference, exact gap
kernel, exact final leg.

Measured consequence at high NA (paraxial readout, analytic Gaussian truth):
0.06% at NA 0.20, 0.36% at 0.278, 2.8% at 0.40, 5.8% at 0.50.

---

## 3. Newton pool — the two-tier threshold, and why

Cold and warm behave oppositely, which is why a single constant was wrong:

```
points     COLD (fresh process)      WARM (pool already alive)
 16 384    1.707 → 2.768  0.62x       0.60 → 0.50   1.20x
 65 536    4.129 → 4.797  0.86x       2.88 → 2.35   1.22x
262 144   11.427 → 9.360  1.22x      10.98 → 7.17   1.53x
  1 024        —                       0.04 → 0.03   1.14x
```

The cold crossover really is ~200k, so the shipped value was right and lowering
it outright would make one-shot runs **1.6× slower** at 16k. But a multi-group
chain calls `apply_real_lens_traced` once per group and only the FIRST is cold —
so at 121's rs=4 (65 536 points/group) every group used to run serial. Hence
`_POOL_MIN_PIXELS` (cold, 200 000) + `_POOL_MIN_PIXELS_WARM` (8 000).

**Speedup tracks the ratio of Newton work to whole-grid work, not the absolute
point count**: the same 262k points gives 1.53× at rs=2/N=1024 but 1.24× at
rs=4/N=2048 (4× more non-Newton work → Amdahl). Finer `ray_subsample` on a
given grid parallelises better than a bigger grid at fixed subsample.

---

## 4. Speed / memory guidance for a larger box

* **Working set is ~8·N² complex128**, not the 22× figure quoted earlier in this
  session's notes. Measured peak RSS: N=2048 → 0.50 GB, N=4096 → 1.98 GB. So
  N=8192 ≈ 8 GB and N=16384 ≈ 32 GB for the CHAIN.
* **The exact final leg, not the chain, is the memory-dominant stage.** With the
  default `window_factor=7.0` it built an **8712² complex128** internal grid
  (~1.1 GB per array, several live at once) and OOM'd at 17 GB free. It scales
  **quadratically in `window_factor`**; `n_fine_cap` does NOT bound that
  dimension despite the name. `window_factor=4.0` is documented as the no-op
  regime (≥ 4) and is what d6 uses.
* **`n_fine_cap`: do NOT lower the default to 8192.** The library's own table on
  real 121 shows 8192 is 3.7× faster (84 s vs 312 s) at identical FWHM/EE — but
  D6's paraxial pre-check *refuses* 8192, so it is a valid opt-in for speed
  (with `on_tilt_exact_grid` downgraded), not a safe default.
* **complex64** would halve the working set; phase precision ~1e-7 rad is far
  under any tolerance here. **Unmeasured** through a 6-group chain — worth
  testing on the big box, since accumulated error is exactly where it could bite.

---

## 5. NOT DONE — pick these up

1. **Re-profile with spline as the default.** The only profile on record
   (`_pip_sample_residual` 42.7%, `_poly` 39.6%, `_invert_newton` 18.4%, FFT
   9.2%) was taken with **polynomial**. `_poly` is gone from the CPU path now,
   so any optimisation derived from that profile must be re-derived.
2. **The 121 chain-grid convergence study — the open correctness question.**
   Every convergence measurement in the repo varies `n_fine_cap` at a FIXED
   N=1024, dx0=2.0 µm. Nothing establishes the CHAIN grid is sufficient.
3. **Full unit-suite regression** for the default flips. Partial run reached 204
   passed; the single failure (`test_audit_io … complex64`) is a **pre-existing
   missing `filelock` dependency**, confirmed by forcing polynomial and seeing
   it fail identically.

### State of the convergence harness (scratchpad, not in repo)
`d121_grid_convergence.py` runs pre-DOE → order (−4,−2) → post-DOE 6 groups →
image, at fixed 2.048 mm window. **It does not work yet** — the readout still
returns a non-finite/empty field (`in_window: false`). Fixed so far:

* `centre_out` must come from `_chain_chief_ray_at_target` — the chief ray walks
  **3.19 mm** over the 69.2 mm post-DOE path, far outside a ±12.8 µm window.
* `fit_radius_beam_factor` (1.5) is **mandatory**, not optional: the post-DOE
  groups carry 20–32 mm apertures against a sub-mm beam (~75×), far past the
  1.5× aperture:beam cliff that corrupts the traced OPL fit.
* `window_factor=4.0` to fit memory.

Still failing after all three. **Beware the failure signature**: the dead runs
reported FWHM identical to five decimals and `gap_env_theta` identical to twelve
significant figures across two different grids — which reads exactly like a
converged result. It agreed because both measured an empty window. An
`in_window` flag is now emitted; trust it before any metric.

---

## 6. Verification status

Green with the new defaults: `test_niche_d2_chain_multi` (38),
`test_niche_d6_exact_tilted_leg` (38), `test_niche_exact_gap_kernel` (23),
`test_niche_newton_pool_both_fits` (6), `test_niche_tight_focus_readout` (10),
plus 469 across nine focus-readout-dependent suites.

`test_niche_d6`'s two contrast margins were re-baselined (0.60→0.75, 5.0→4.0)
because the standoff change **improved** the paraxial leg (1.857× → 1.476× the
oracle FWHM), collapsing margins sized against the worse baseline. The exact leg
is unchanged and still lands exactly on the ray oracle:

| | exact | paraxial | oracle |
|---|---|---|---|
| FWHM | **3.1500 µm** | 4.6500 µm | **3.1500 µm** |
| EE2 | 0.7033 | 0.1557 | 0.7163 |
| peak offset | 0 | −6.3 µm | 0 |

That contrast — for a *tilted* congruence, i.e. the DOE-order case — is the
reason `final_leg='exact'` matters for 121.
