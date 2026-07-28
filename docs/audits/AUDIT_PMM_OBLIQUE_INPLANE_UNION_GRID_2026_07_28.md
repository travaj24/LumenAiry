# AUDIT — PMM tapered stacks: the shared union grid caps accuracy on in-plane oblique solves

**Date:** 2026-07-28 · **Library:** lumenairy 5.31.0 · **Branch:** `fix/pmm-union-grid-conditioning`
**Reporter:** metasurface LC-QWP out-coupler campaign (exp18–exp22), λ = 1310 nm
**Severity:** **P1 — silent-wrong, user-facing.** Energy-clean results wrong by up to 20×, no warning on
the affected construction path.

---

## 0. How to read this report

Findings are labelled by evidence class, and the distinction is load-bearing:

| tag | meaning |
|---|---|
| **[M]** | **Measured** — a number produced by a run, reproducible from §11 |
| **[A]** | **Analysis** — arithmetic/derivation from measured quantities or from the source |
| **[H]** | **Hypothesis** — consistent with evidence, *not* established. Explicitly flagged. |

One hypothesis in an earlier draft of this audit was **refuted by measurement and withdrawn**; it is kept
in §4.2 as a record, because the refutation constrains what the mechanism can be.

---

## 1. Executive summary

A 1-D `PMMStack` built from a **2° tapered** `SegmentStackGeometry` returns **in-plane (φ=0) oblique
extinction that does not converge in polynomial `degree`** — deg 6/8/10 give mutually inconsistent answers
(spread **91 %** at θ=8°) while total power stays clean and nothing warns. The **out-of-plane (φ=90) cut of
the identical stack converges to 0.03 %** [M], so the defect is not the conical path, the materials, or the
geometry description.

**Cause [M]:** the solution depends on **how the shared union grid was built** — specifically on
`min_feature`, the threshold that merges near-coincident cross-layer walls. Its default,
`period × 1e-5` = **7 pm** on a 700 nm pitch, is ~200× below the ~1.5 nm the library's own docstring
recommends, so on this geometry it merged **nothing**.

**Immediate fix [M]:** `min_feature = 1.5e-9` collapses the degree spread **91 % → 0.1 %** and runs
**2.1× faster**. Every physical film survives (≤1.2 % width drift).

**Geometric origin of the collisions [A]:** the per-slice taper wall offset is `slice_thickness × tan(2°)`
= **5.41 nm** against a **5.00 nm** Al₂O₃ coat — a **0.41 nm** mismatch, and three of the six observed
collision pairs are *exactly* 0.41 nm [M]. This explains **where the collisions come from**. It does **not**
explain the loss of convergence: choosing `n_slice = 3` to avoid the resonance was predicted to restore
conditioning and **was measured and REFUTED** (§4.4). Collision *geometry* and conditioning *failure* are
not the same thing, and no cheap geometric workaround has survived testing — which strengthens the case for
the structural fix (R-6) over further tuning.

**Structural limit [M]:** fixing `min_feature` cures `degree` and then exposes the next wall. Refining the
staircase (`n_slice` 2 → 4) improves the geometry but **re-breaks conditioning** (degree spread 15.7 % at
one angle). Converging a taper needs more slices; more slices grow the *shared* grid and multiply
collisions. **Accuracy on tapered devices is capped by the shared-grid architecture, not by the staircase
law.** The tractable fix is per-layer element grids with interface projection (**R-6**), estimated
**1–3 weeks**, with a large *performance* upside as well.

---

## 2. Symptom and scope

**Device.** Coated Cu pillar out-coupler; period 700 nm; λ = 1310 nm; `sidewall_deg = 2.0`;
`n_slice = n_slice_A = n_refl_slice = 2`; 38 layers; absorbing Cu substrate (`n = √ε_Cu`); LC superstrate
`n = 1.50`; anisotropic **planar** LC (no out-of-plane tensor components).
**Metric.** Zeroth-order reflection Jones → `reflective_outcoupling`; ER = peak/null over the LC switch,
LC angles frozen at their normal-incidence values (φ_pk = 0°, φ_nl = 90°).

**In-plane ER at θ = 8°, default `min_feature` [M]:**

| degree | ER |
|---|---|
| 6 | 41.363 |
| 8 | 3.822 |
| 10 | 3.674 |

A second harness differing only in the LC-search grid gave **60.1 / 2.2 / 41.1** at the same degrees [M] —
the result is not stable even across equivalent call sites.

**Out-of-plane ER at θ = 8°, same stack, same degrees [M]:** 14.418 / 14.380 / 14.376 (**0.03 %**).

Peak coupling stays physical throughout (0.82–0.91) and energy closes, so no existing tripwire fires. This
is the class the library elsewhere names **"passive-but-wrong."**

### 2.1 Scope: what is and is not affected

| affected | not affected |
|---|---|
| In-plane oblique extinction on **tapered** stacks with **deep nulls** | Normal incidence (0.2 % across degrees) [M] |
| Any deep-null ratio at oblique incidence | All reflectivity / peak-coupling values [M] |
| | The entire out-of-plane dataset (0.03 %) [M] |
| | Vertical (untapered) stacks — no staircase, no collisions [A] |
| | Shallow nulls (ER ≲ 20) — see §5.4 |

---

## 3. What was ruled out

Recorded because eliminating these is what localised the defect.

| candidate | verdict | evidence |
|---|---|---|
| Library version change (5.24 → 5.31) | **Not the cause** | Fully deterministic: cross-machine **0.00 %**, run-to-run **0.000e+00** [M] |
| Machine / BLAS nondeterminism | **Not the cause** | Same [M] |
| Far-field truncation (`ffo`) | **Not the cause** | ffo 11/21/31/41 **identical at every degree** [M] |
| API/behaviour break from the audit campaign | **Not the cause** | 14 methods + 8 imports verified; the one apparent hit was `RCWAStack.set_source(polarization=)`, a different class [M] |
| The v5.31 `shear` addition | **Not the cause** | `shear` omitted is bit-identical to the pre-change builder (library-measured) |
| Conical/OOP solver path | **Not the cause** | OOP converges to 0.03 % on the same stack [M] |
| Staircase truncation order | **Not the cause** | Library-measured `O(1/n_slices²)`, 3.9×/doubling vs an n_slices = 768 twin |

**`ffo` deserves emphasis.** The campaign had been validating convergence in `ffo` and treating `degree` as
settled. `ffo` was converged at 11 the entire time; **`degree` was the live variable.** Checking only the
order count gave false confidence for months.

---

## 4. Root cause

### 4.1 The shared union grid

`_pmm_union_grid` (`elements/pmm/_core.py:3300`) builds **one nodal grid for the whole stack** as the union
of every layer's wall positions. Every layer is then assembled on that shared grid, because the S-matrix
cascade matches modes at interfaces and adjacent layers must share a basis.

A tapered staircase offsets each slice's walls, so the union accumulates closely-spaced cross-layer pairs.
`min_feature` merges pairs closer than its threshold **to their midpoint**, and only when the two walls
have **no common owning layer** — a close pair owned by one layer is that layer's intentional thin feature
(a 1 nm liner) and is never merged.

`PMMStack.__init__` (`stack.py:194`):

```python
self.min_feature = (float(period) * 1e-5 if min_feature is None else float(min_feature))
```

**Default values [A]:**

| design | period | default `min_feature` |
|---|---|---|
| W_hiER2 / W_tolHi | 700 nm | **7.00 pm** |
| cBal_db240 | 690 nm | 6.90 pm |
| W_knee | 680 nm | 6.80 pm |
| W_bal | 650 nm | 6.50 pm |

7 pm is ~3 % of a copper atom's diameter. Nothing physical is that small, and on these stacks it merged
**zero** pairs [M].

The mismatch is dimensional: the default scales with the **period**, but the collision scale of a taper is
the **per-slice wall offset** `≈ (thickness / n_slices) · tan(sidewall)` — nanometres, and **independent of
the period**.

### 4.2 Mechanism: what is established, and what was refuted

**Refuted and withdrawn [M].** The initial hypothesis was a single degenerate element
(`J = ½(x_r − x_l) → 0`) whose `1/J` operators blow up. The measured grid statistics do not support it:

| `min_feature` | elements | median width | thinnest | thinnest / median |
|---|---|---|---|---|
| 0.007 nm (default) | 61 | 1.095 nm | 0.214 nm | 0.196 |
| 0.5 nm | 45 | 5.206 nm | 0.946 nm | 0.182 |
| 1.5 nm | 43 | 5.206 nm | 0.706 nm | 0.136 |

The thinnest element is only ~5× below the median — not a near-zero-width sliver — and the device's own
1 nm Ta liner legitimately produces ~1 nm elements. Note also that the thin/median *ratio* barely changes
across the sweep (0.196 → 0.136) even as behaviour goes from broken to converged, so that ratio is not the
discriminator either.

**Established [M].** The answer depends causally on `min_feature`, and with it the degree-convergence:

| `min_feature` | ER θ8 (deg 6 / 8 / 10) | degree spread θ8 | deg-6 runtime |
|---|---|---|---|
| **0.007 nm** (default) | 41.363 / 3.822 / 3.674 | **91.1 % SCATTERED** | 147 s |
| 0.5 nm | 57.694 / 36.139 / 36.023 | 37.6 % SCATTERED | 78 s |
| **1.5 nm** | **54.889 / 54.834 / 54.833** | **0.1 % CONVERGED** | **71 s** |
| 3.0 nm | 18.368 / 11.328 / 18.340 | 38.3 % SCATTERED | 57 s |

The usable window is bounded on **both** sides: too small and collisions survive; too large and the snap
over-perturbs the geometry (§6.2) and conditioning degrades again.

**Corroboration [M].** Removing the thin conformal coats (`ta = al = sin = 0`), which eliminates most
cross-layer collisions, makes the **default** converge: deg 6/8/10 → 2.811 / 2.800 / 2.795 (0.5 %). The
solver core is sound; the defect is in the grid the geometry induces.

**Open [H].** The precise route from "grid A vs grid B" to "deep null corrupted" is **not pinned down**.
Element-width extremes are ruled out as the sole explanation. Candidates not discriminated here: total
element count and its effect on the eig spectrum; conditioning of the interface mode-match; sensitivity of
a near-cancelling null to sub-nm wall placement (which would make it partly *physical* — see §6.2). The
recommendations below deliberately do not depend on resolving this.

### 4.3 Why in-plane fails and out-of-plane does not

Both cuts share the grid and are perturbed identically. The in-plane null is far deeper (ER ≈ 50–60 ⇒ null
≈ 1.7 %) than the out-of-plane null (ER ≈ 8–26). ER is a ratio with the null in the denominator, so the
same absolute error is amplified several times more in-plane [A]. Measured directly [M]: moving
`min_feature` from the default to 1.5 nm moves the Jones by **0.00709 in the peak state** and **0.03923 in
the null state** — a 5.5× difference, on the same stack at the same angle.

**This generalises:** deep-null figures of merit are the sensitive observable. Any PMM result quoting
extinction ≳ 30 at oblique incidence on a tapered stack should be treated as suspect until convergence in
`degree` **and** `min_feature` is demonstrated.

### 4.4 The geometric origin: a taper/coat resonance [A]

The collisions are not generic — they are an accidental resonance between two independent lengths.

Region 1 thickness = `H − t − al` = 385 − 70 − 5 = **310 nm**; at `n_slice = 2`, slices are **155 nm**.

```
per-slice wall offset = slice_thickness × tan(2°) = 155 × 0.034921 = 5.41 nm
Al2O3 conformal coat                                             = 5.00 nm
mismatch                                                         = 0.41 nm
```

So a slice's **ridge** wall lands 0.41 nm from the neighbouring slice's **coat** wall. Measured collision
pair separations [M]: **0.22, 0.41, 0.41, 0.44, 0.81, 1.09 nm** — three are exactly the predicted 0.41 nm.

Scanning `n_slice` (offset = 310/n × tan 2°), and comparing against the coat thicknesses (5 nm, and
al+sin = 20 nm) at integer multiples `k`:

| n_slice | offset | nearest collision | severity |
|---|---|---|---|
| 1 | 10.82 nm | k=2 → 21.6 vs 20 | 1.6 nm — mild |
| **2** | **5.41 nm** | **k=1 → 5.41 vs 5** | **0.41 nm — severe** |
| 3 | 3.61 nm | k=1 → 3.61 vs 5 | 1.4 nm — mild |
| **4** | **2.70 nm** | **k=2 → 5.41 vs 5** | **0.41 nm — severe** |

**`n_slice` 2 and 4 both hit the resonance; 1 and 3 do not** [A].

### 4.5 The resonance does NOT drive the convergence failure — prediction tested and REFUTED [M]

§4.4 predicted that `n_slice = 3`, which avoids the 0.41 nm resonance, would be well-conditioned **without**
any snapping. Tested at the **library-default** `min_feature` (nothing snapped, confirmed: 0 merged pairs):

| n_slice | θ0 (deg6 / deg8) | θ8 (deg6 / deg8) | θ10 (deg6 / deg8) | worst degree spread |
|---|---|---|---|---|
| 2 (resonant) | 58.889 / 58.768 | 60.130 / **2.222** | 56.359 / 51.637 | **96.3 %** (θ8) |
| 3 (non-resonant) | 65.756 / 58.399 | 49.977 / 47.074 | 53.525 / **3.887** | **92.7 %** (θ10) |

**`n_slice = 3` is just as badly non-convergent as `n_slice = 2`** — the failure simply moves to a
different angle. The prediction is **refuted**.

What survives and what does not:

* **Survives [M/A]:** the arithmetic explaining *where the collisions come from*. `155 × tan 2° − 5.00 =
  0.41 nm` matches three of six measured pair separations exactly. That is a correct account of the
  **geometry**.
* **Refuted [M]:** the inference that those collisions **cause** the loss of degree-convergence, and hence
  that avoiding them fixes it. Collision geometry and conditioning failure are **not** the same thing.

This is the **second** hypothesis falsified in this audit (after the `J → 0` sliver, §4.2). Both were
plausible, quantitatively consistent with some of the evidence, and wrong. The pattern in the failures is
notable: the collapse lands on an apparently arbitrary `(n_slice, degree, θ)` cell (θ8 at ns=2/deg8;
θ10 at ns=3/deg8; θ10 at ns=4/deg8) rather than tracking any geometric parameter — which is what one would
expect from a conditioning threshold being crossed unpredictably, not from a systematic geometric
resonance.

**Consequence for the recommendations:** no cheap geometric workaround has survived testing. `min_feature`
remains the only lever measured to work (and only at `n_slice = 2`, §5.3). That is an argument for the
structural fix — **R-6** — rather than for further parameter tuning, and **R-8 is withdrawn**.

---

## 5. The four-knob convergence picture

A tapered PMM result is trustworthy only when stationary in all four discretisation knobs.

### 5.1 `ffo` (far-field orders) — converged, never the issue [M]
Identical values at ffo 11 / 21 / 31 / 41, at every degree, for both designs tested. `ffo = 11` suffices.

### 5.2 `degree` — the live variable [M]
Cured by `min_feature` (§4.2): 91.1 % → 0.1 %.

### 5.3 `n_slice` — the staircase, and the next wall [M]
At `min_feature = 1.5 nm` (W_hiER2, in-plane ER):

| n_slice | θ0 (deg6 / deg8) | θ8 (deg6 / deg8) | θ10 (deg6 / deg8) | degree spread |
|---|---|---|---|---|
| 1 | 17.442 / 17.470 | 12.506 / 12.525 | 10.398 / 10.413 | ≤ 0.16 % |
| 2 | 49.640 / 49.555 | 54.889 / 54.834 | 54.702 / 54.679 | ≤ 0.17 % |
| 4 | 49.964 / 49.848 | 55.959 / 55.877 | 56.233 / **47.400** | θ10 **15.7 %** |

1. **The staircase is converging.** `n_slice = 1` is degenerate (one slab, no taper resolution) and must be
   excluded. On the meaningful **2 → 4** step, five of six entries agree to **0.6–2.8 %**, consistent with
   the documented `O(1/n_slices²)` law. Residual staircase error at `n_slice = 2` is therefore **~1–3 %**.
2. **But refining re-breaks conditioning.** At ns=4/θ10 the degree spread is **15.7 %** where every other
   cell is ≤ 0.23 %. `min_feature = 1.5 nm` is sufficient at ns=2 and **not** at ns=4 — consistent with
   §4.4, since ns=4 re-enters the 0.41 nm resonance.

**This is the structural vise:** geometry fidelity and grid conditioning move in opposite directions, with
cost `O(n_slices^3.4)` on top.

### 5.4 `min_feature` — an accuracy knob, not just a cost knob [M]
The converged value itself depends on it (θ0 ER: 58.8 → 52.6 → 49.6 → 15.5 at 0.007 / 0.5 / 1.5 / 3.0 nm).
It is documented only as a cost knob ("**ALSO THE COST KNOB**"), and nothing reports how far the snap has
moved the geometry. Addressed by R-4 (now implemented).

---

## 6. Does the fix distort the device?

### 6.1 Films survive [M]
Total x-width per material across all layers, versus the exact reference:

| material | reference | at 0.5 nm | at 1.5 nm | at 3.0 nm |
|---|---|---|---|---|
| Cu | 13954.12 nm | −0.05 % | **−0.07 %** | −0.44 % |
| SiO₂ | 3432.48 | −0.12 % | **−0.03 %** | −0.03 % |
| Si₃N₄ (15 nm) | 3369.91 | −0.20 % | **−0.20 %** | −0.20 % |
| LC | 2338.99 | +0.00 % | **−0.00 %** | −0.00 % |
| SiCN | 1404.32 | +0.11 % | **−0.06 %** | −0.24 % |
| Al₂O₃ (5 nm) | 1134.48 | +1.21 % | **+1.20 %** | **+5.77 %** |
| Ta (1 nm liner) | 965.71 | +0.23 % | **+0.53 %** | +0.80 % |

At 1.5 nm every material is preserved to ≤ 1.2 %: the snap is coarsening **staircase wall positions** (a
discretisation artifact), not the physical films. At 3.0 nm real distortion begins (Al₂O₃ +5.8 %), which is
also where the degree scatter returns — the two failures coincide.

### 6.2 A caveat worth stating
`min_feature = 1.5 nm` moves walls by ≤ 0.75 nm, and the ER changes by ~16 % (58.8 → 49.6 at θ0). Some of
that may be a **real** sensitivity of a deep resonant null to sub-nm wall placement rather than pure
numerics [H]. If so it is a **fabrication-tolerance statement** about the design, and a useful one: an
extinction that moves 16 % for 0.75 nm of wall placement is fragile against etch variation. Distinguishing
"numerical" from "physical" here requires a staircase-free reference the codebase cannot currently provide
for a trapezoid (§8.3).

---

## 7. Why no independent cross-check was available

RCWA was the obvious oracle: `SegmentStackGeometry.to_rcwa_stack` builds the **same geometry object** with
a different solver. It does not converge on this device [M]:

| n_orders | 21 | 31 | 41 | 81 | 151 | 251 |
|---|---|---|---|---|---|---|
| ER at θ0 | 1.105 | 0.198 | 0.727 | 3.653 | 6.366 | 22.285 |
| peak coupling | 0.329 | 0.129 | 0.333 | — | — | — |

against a PMM value of ~50–59 and peak **0.824**. Expected [A]: the 1 nm Ta liner and 5 nm conformal coats
are far below the Fourier floor `≈ P/2N` (8.5 nm at N = 41; 1.4 nm even at N = 251), and the coats are thin
in **x** as well as **z** because they are conformal. This is a legitimate, documented strength of PMM
(laterally exact, no Gibbs floor) — but the consequence is that **no independent in-repo oracle exists for
this device class**, which is why the defect survived. A `min_feature`-perturbation self-consistency check
is the practical substitute, and is what R-1 implements.

---

## 8. Options for a real fix

### 8.1 Lattice quantisation of wall positions
**Proposal:** round every wall to a multiple of Δ at build time, instead of the current cascading pairwise
midpoint merge.

**Genuine advantages over the status quo [A]:** deterministic and reproducible; bounded displacement
(≤ Δ/2) versus a cascade whose displacement had to be *instrumented to be reported at all* (R-4); walls
that *should* coincide coincide **exactly**, so those pairs merge cleanly rather than leaving a residue.

**Sizing — the counter-intuitive part [A].** A lattice merges two walls only if they round to the same
point, which requires roughly `d < Δ`. **Finer Δ therefore merges fewer pairs, not more.** A Δ of 0.01 nm
turns a 0.22 nm collision into exactly 22Δ — well-defined, reproducible, and *still a 0.22 nm element*.
Quantisation regularises positions; it does not by itself remove separations.

Measured separations are 0.22–1.09 nm [M], so removing them needs **Δ ≈ 1.1 nm**, at which the real films
remain exactly representable (Ta 1 nm ≈ 1Δ, Al₂O₃ 5 nm ≈ 5Δ, Si₃N₄ 15 nm ≈ 14Δ). The sizing rule should
therefore be driven by **the separations to eliminate**, not by the thinnest feature to preserve.

**Verdict:** worth adopting as a **complement** to `min_feature` (determinism, bounded error), not a
replacement (conditioning). Recorded as **R-7**.

### 8.2 Per-layer element grids with interface projection — **the recommended fix (R-6)**
Give each layer its own grid; insert an L2 projection `P_{i,i+1}` between adjacent bases at each interface
(mass-weighted inner products by quadrature over the intersection of the two element sets).

**Effort [A]: ~1–3 weeks** for a developer fluent in the codebase, including validation. It is bounded, not
a rewrite: the per-layer **eigenproblems are already independent** — only the nodal basis is shared. Work
items: (a) per-layer grid construction; (b) the projection operator; (c) re-validating energy closure,
reciprocity and S-matrix stability. Main risk: getting the projection right for both tangential **E** and
**H** so interface conditions hold exactly.

**Performance upside, which may exceed the accuracy motivation [A].** Today every layer carries every other
layer's walls: this stack has a **61-element union grid** [M] while a typical layer has only ~6–10 walls of
its own. The eig is `O(N³)` in node count, so per-layer grids shrink a typical layer's `N` by ~6× — roughly
**two orders of magnitude less work per layer** — before subtracting projection overhead. For a 38-layer
stack that plausibly dominates. It also removes the `O(n_slices^3.4)` cost law that currently makes taper
convergence unaffordable, and retires `min_feature` as an accuracy knob.

### 8.3 Covariant taper-metric layer (the existing roadmap item)
Coordinate-transform the tapered walls into coordinate surfaces and solve **one** layer exactly, no
staircase. Already shipped for the **pure-shear** case: `add_sheared_grating` emits one exact slanted layer
because `u = x − z·tanφ` has a **z-invariant** metric, so the modal coefficients collapse to a single
eigenproblem.

**Why the symmetric trapezoid is genuinely harder [A]:** its natural map `u = x/w(z)` has a
**z-dependent** metric — the eigenproblem varies with depth and does not reduce to one solve. This is a
real mathematical obstruction, not an oversight, and it is why shear shipped and the general taper did not.
A shear can absorb a **translation** of the walls with depth; it cannot absorb a **dilation**.

### 8.4 Choose `n_slice` to avoid the resonance — free, available today
Per §4.4, pick `n_slice` so the per-slice offset `(thickness/n_slices)·tan(sidewall)` is not close to a
coat thickness or a small multiple thereof. For this device, **`n_slice = 3` instead of 2 or 4**. The
builder could compute this and warn automatically (**R-8**).

---

## 9. Fixes implemented on this branch

Commits `886eeb0`, `832ed56`.

| id | change | verification |
|---|---|---|
| **R-1** | `solve(stabilize='slices')` previously **skipped entirely** with no taper recipe, leaving every `SegmentStackGeometry`-built stack — the documented device route — unprotected. It now falls back to a **union-grid consensus** (`_min_feature_clone` + perturbed `min_feature`), needing no recipe. | **Fires** on the pathological stack (Jones moved 0.0392 > 0.02, naming the 7e-12 → 2e-9 m perturbation); **silent** on a clean vertical stack and on a sheared grating [M] |
| **R-1a** | The probe is anchored to a **physical ~nm scale**, not a multiple of the current value — scaling a pathologically small `min_feature` by a small factor stays in the same broken regime (a ±4× probe moved the Jones by **0.0**) [M] | as above |
| **R-1b** | A failed probe is **skipped**, not scored as maximal disagreement — this guard runs on every stabilize call and a false pathology claim is worse than silence | negative controls pass [M] |
| **R-4** | The snap warning reports **max wall displacement**; `PMMStack.__init__` documents `min_feature` as an **accuracy** knob and why the default is wrong for tapered stacks | live run reports 0.55 nm displacement at 1.5 nm [M] |

Two tests pinned the old skip-and-announce behaviour and were updated to pin the new contract (consensus
runs; stays silent when there is nothing to snap). **PMM suite: 997 passed**, the only two failures being
those, now green [M].

**Deliberately not changed:** the `min_feature` **default**. Changing it would silently alter results for
every existing user, and this repo has strict deprecation discipline. The defect is addressed by
documentation + detection; see R-3 for the principled default.

---

## 10. Recommendations

| # | Priority | Recommendation | Status |
|---|---|---|---|
| **R-1** | P1 | Make the staircase guard reachable on `SegmentStackGeometry`-built stacks | **done** |
| **R-2** | P1 | Report a direct observable of grid quality regardless of construction path. *An element-width-ratio detector was implemented, then removed: §4.2 refutes its premise. A correct detector needs the mechanism resolved first — do not ship one on an unconfirmed story.* | **open** |
| **R-3** | P2 | Make the `min_feature` default **taper-aware**: derive from `(thickness/n_slices)·tan(sidewall)` when a taper is recorded, instead of `period × 1e-5` | open |
| **R-4** | P2 | Document `min_feature` as an accuracy knob; report snapped displacement | **done** |
| **R-5** | P3 | Covariant taper-metric layer for the general trapezoid (§8.3) — research-adjacent | open |
| **R-6** | **P2** | **Per-layer element grids with interface projection (§8.2)** — removes the collision class outright, retires `min_feature` as an accuracy knob, lets `n_slice` scale, and is plausibly a large net **speed-up**. ~1–3 weeks. **The recommended structural fix.** | open |
| **R-7** | P3 | Replace the cascading pairwise snap with **lattice quantisation** (§8.1), sized by the separations to remove (~1 nm here), for determinism and bounded displacement | open |
| ~~R-8~~ | — | ~~Warn when the per-slice taper offset resonates with a coat thickness (§4.4)~~ — **WITHDRAWN**: the resonance explains where collisions come from but **not** the convergence failure; `n_slice = 3` avoids it and is equally non-convergent (§4.5) [M] | **withdrawn** |

### For users of tapered `PMMStack` geometry, today
1. **Set `min_feature` explicitly.** Do not accept the default on a tapered stack. Validate by sweeping it:
   the usable value is where the answer is stationary in **both** `degree` and `min_feature`.
2. **Converge in `degree`, not just `ffo`.** Here `ffo` was converged at 11 throughout while `degree` was
   the live variable — checking only order count gives false confidence.
3. **Check `n_slice` against your coat thicknesses** (§4.4) before trusting a tapered result.
4. **Treat oblique extinction ≳ 30 as ill-conditioned** until convergence is demonstrated; quote it with a
   band or as a bound.

---

## 11. Impact on the reporting campaign

- **Unaffected [M]:** all normal-incidence extinction (0.2 % across degrees); all reflectivity /
  peak-coupling; the entire out-of-plane angular dataset (0.03 %). The campaign's central physical
  conclusion — the pillar is out-of-plane-fragile, i.e. the axis flip — rests on out-of-plane data and
  **stands**.
- **Corrected [M]:** in-plane oblique extinction, recomputed at `min_feature = 1.5e-9`, deg 8. W_hiER2
  in-plane (dB): **17.0 / 17.2 / 17.4 / 17.4 / 16.7 / 14.9 / 12.1 / 10.5 / 7.4** at θ = 0/5/8/10/12/15/18/20/25
  — a smooth monotone roll-off, replacing a curve that oscillated 17.7 → 3.5 → 17.1 → 3.5 → 14.4. The
  in-plane tolerance claim is restored, now on a degree-converged footing.
- **Residual uncertainty, stated:** at `n_slice = 2` the recomputed values carry a **~1–3 % staircase
  band** (§5.3) and should be quoted as such. Driving it lower is blocked by R-6/R-5, not by any user
  setting. For conclusions that turn on 10 dB-scale differences, 3 % is immaterial.

---

## 12. Reproduction

Scripts in `Metasurface_QWP/experiments/exp21/`:

| script | purpose |
|---|---|
| `min_feature_matrix.py` | the decisive `min_feature` × `degree` matrix (§4.2) |
| `nslice_convergence.py` | `n_slice` convergence at fixed `min_feature` (§5.3) |
| `washout_check.py` | per-material width preservation (§6.1) |
| `grid_stats.py` | union-grid element statistics (§4.2) |
| `oblique_convergence_v531.py` | degree/ffo ladders, in-plane vs out-of-plane |
| `rcwa_crosscheck.py` | the RCWA oracle attempt (§7) |
| `pmm_pathology_probe.py` | `stabilize='slices'` inertness + simplified-structure control |
| `verify_fixes.py`, `verify_null_state.py` | verification of the R-1/R-4 fixes (§9) |

**Geometry:** `exp11_common.coated_pillar_geometry(w_c=340, w_f=170, g=95, H=385, t=70, pillar_w=90,
sio2_gap=80, d_refl=120, d_below=120)`, `n_slice = n_slice_A = n_refl_slice = 2`, `sidewall_deg = 2.0`,
λ = 1310 nm, `n_superstrate = 1.50`, `n_substrate = √ε_Cu`, planar anisotropic LC.

**Note on imports:** these experiments load lumenairy from the working tree via `sys.path.insert`, not from
an installed wheel — library changes reach them with no version gate.
