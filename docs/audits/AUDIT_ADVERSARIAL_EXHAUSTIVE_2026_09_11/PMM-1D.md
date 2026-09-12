# PMM-1D audit — the 1-D Polynomial (spectral-element) Modal Method: `_core.py`, `stack.py`, `oned.py`, `conical.py`, `_jax_stack.py`

Tree `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`, branch `main`, HEAD `a1ff1e6e`.
Repro scripts + raw outputs: `…/scratchpad/PMM-1D/` (`oracle.py`, `p1*…p15*.py`, `out_*.txt`).

**Caveat on every wall-clock number below**: this machine was running ~19 sibling auditors throughout
(15 concurrent `python.exe` at the time of measurement). Absolute timings are worthless; I therefore
report only (a) *deterministic call counts*, (b) *structural* facts (operator sparsity, flop counts),
and (c) **interleaved min-of-N A/B ratios**, which are the only timing form that survives contention.

---

## 0. What "PMM" is here, and the equations as coded

Despite the name it is **not** the Botten/Li *true-mode* method and not a Fourier method. It is a
**C⁰ nodal spectral-element (GLL Lagrange) Galerkin discretisation** of the transverse modal
eigenproblem on the periodic cell, with element boundaries **pinned to the material walls** — the
subsectional polynomial modal method of Morf 1995 / Edee 2011 in its GLL-nodal (mass-lumped)
realisation. This matters for the audit: a Galerkin eigenproblem **structurally cannot miss a mode**,
so the classic PMM/true-mode "missed root" failure does not exist here (confirmed numerically, §CFC-2).

Operators (`_build_sem`, `_core.py:409`), per element with `w = ref_w·J`, `D = Dref/J`:

```
S0 = Σ diag(w)          Peps = Σ ε diag(w)          Pinv = Σ (1/ε) diag(w)
L  = Σ Dᵀ diag(w) D     Linv = Σ (1/ε) Dᵀ diag(w) D
C  = Σ diag(w) D        Cinv = Σ (1/ε) diag(w) D
```

Layer modes (`_sem_modes`, `_core.py:1678`), Bloch shift `kx0 = Re(n_sup)·sinθ·k0` (DIMENSIONAL, rad/m):

```
TE (E_y):  (Peps − Lop/k0²) x = q² S0 x,    Lop = L    − i·kx0·(C    − Cᵀ)    + kx0²·S0
TM (H_y):  (S0   − Lop/k0²) x = q² Pinv x,  Lop = Linv − i·kx0·(Cinv − Cinvᵀ) + kx0²·Pinv
q = γ/k0 (DIMENSIONLESS),  lam = −i·q,  propagator X = exp(−lam·k0·L) = exp(+i·γ·L)
V_TE = W·diag(q),   V_TM = (S0⁻¹ Pinv)·W·diag(q)
```

I re-derived both weak forms from Maxwell. With `D = d/dx + i·kx0`, `D̄ = d/dx − i·kx0`:
`∫φ(−D²u) = L − 2i·kx0·C + kx0²·S0`, and `C` is **exactly antisymmetric** under periodicity, so
`−i·kx0·(C − Cᵀ) ≡ −2i·kx0·C`. For TM, `∫φ(−D(ε⁻¹Du)) = ∫(D̄φ)(ε⁻¹)(Du) = Linv − i·kx0·(Cinv − Cinvᵀ)
+ kx0²·Pinv`, and `Cinv` is **not** antisymmetric (the 1/ε jump at the wall gives
`Cinv + Cinvᵀ = −∫(1/ε)′φφ ≠ 0`). The in-code claim that the `(Cinv − Cinvᵀ)` spelling is required
rather than `2·Cinv` is **correct**, and it is the crux of the TM oblique path.

Interface S-matrix (`_core.py:1846`), with `a = Wb⁻¹Wa`, `b = Vb⁻¹Va`:
`S11 = −(a+b)⁻¹(a−b)`, `S12 = 2(a+b)⁻¹`, `S21 = ½[(a+b) − (a−b)(a+b)⁻¹(a−b)]`, `S22 = (a−b)(a+b)⁻¹`
— derived and confirmed (`S11` = reflection, `S21` = transmission, no T-matrix anywhere).

Anisotropic 2-component layer (`_sem_modes_tensor`, `_core.py:2354`): Li-1996 in-plane factorisation
with the wall-normal inverse rule realised **exactly** (`Cxx = inv(S0⁻¹∫φ(1/εxx)φ)`, and the jump is on
an element boundary so the quadrature is exact), plus the full dimension-agnostic `P/Q` blocks with a
scalar `Ky = ky0/k0` for conical.

1-D mortar (per-layer grids, `_core.py:5497`): tangential E tested on grid *b*, tangential H on grid *a*;
`A = (M_b W_b)⁻¹ C_abᵀ W_a`, `B = (M_a V_a)⁻¹ C_ab V_b`,
`S11 = (I+BA)⁻¹(I−BA)`, `S12 = 2(I+BA)⁻¹B`, `S21 = A(I+S11)`, `S22 = A·S12 − I` — derived, confirmed;
the classic energy-consistent mode-matching pairing.

---

## Scope read

* `pmm/_core.py` (7514) — read line-by-line: 1–880 (basis, resolvers, guards, `_forward_branch_flip`,
  flux cut), 880–1500 (T3-4 instruments, `_forward_growth_flip`, passivity predicates), 1678–2120
  (`_sem_modes`, geo-eig, Fourier projection, interface/propagation S-matrices, `_assemble_jones_farfield`,
  `_scalar_farfield_RT`), 2138–2500 (`_pmm_solve_core`, `_build_sem_tensor`, `_sem_modes_tensor`),
  2494–2630 (geo-eig cache, `_sem_modes_uniform`), 2960–3200 (`_stabilize_*`, `_lossy_incidence`),
  3775–3830 (JAX incidence guard), 4460–4760 (`_pmm_union_grid`, `_perlayer_window_grids`), 4816–5030
  (exact/cross mass, per-layer geometry cache, `_kron2_apply`), 5290–5640 (`_guarded_solve`, both
  mortars), 5807–5900 (`_redheffer_star_rect`, slant build).
* `pmm/stack.py` (5050) — 185–430 (guard constants), 429–1265 (the whole sliver guard: screen /
  arbiter / refusal / within-layer hazard), 1265–1440 (`_warn_stack_energy`), 1436–1790 (`__init__`,
  `add_layer`), 2599–3290 (`solve`, `_solve_vertical_perlayer`), 3605–3660
  (`_internal_amplitudes`, `_flux_at`), 3663–3985 (`internal_field`, both grid branches).
* `pmm/oned.py` (1811) — all public signatures and dispatch; `pmm/conical.py` (690) — in full;
  `pmm/_jax_stack.py` (722) — in full for `_pmm_stack_solve_jax`; `pmm/__init__.py`.
* Context: `CONVENTIONS.md` §6, §7, §7.1, §11; `docs/audits/FIX_BRANCH_CUT_ROUND3_2026_09_11.md`;
  `FIX_PMMSTACK_SLIVER_WALLS_ROUND*`; `rcwa/_core.py:1290–1450` (`_sqrt_decay`) and `:2696–2960`
  (the shared Redheffer / interface algebra); `git log --oneline -60`.
* **Not reached** (out of budget): the slant/covariant metric generators' line-by-line algebra
  (`_build_generator_metric` 226 lines, `_cov_generator_4n`, `_pmm_jones_slant_*` — read for contracts
  and routing only; the slant solver was exercised numerically, CFC-21), `_PreparedPMMStack`,
  `PMMStack.solve_vs_wavelength` and its threaded per-layer twin, and the tapered-geometry builders
  (`add_tapered_grating` / `add_sheared_grating` / `add_tapered_ridges`).

---

## Findings

### **[P1] The differentiable (JAX) `PMMStack.solve` branch runs NONE of the NumPy branch's guards, and returns silently-wrong answers where NumPy raises** — `stack.py:2720–2724` vs `stack.py:2739`; `_jax_stack.py` (whole file)

`PMMStack.solve` dispatches to the JAX twin and **returns** at `stack.py:2722/2724`, i.e. *before*
`_require_propagating_incidence` at `stack.py:2739`. `_jax_stack.py` contains **zero** occurrences of
`_warn_stack_energy`, `_require_propagating_incidence`, `_guarded_lstsq`, `_sliver` or `energy`
(counted). Its only guard is `_grazing_guard_concrete`, which tests `|kz_inc| < 1e-9` and nothing else.
It also replaces `_guarded_lstsq` with an explicit normal-equation pseudo-inverse
`Hᴴ(HHᴴ)⁻¹` (`_jax_stack.py:413–419`), which squares the condition number, with no residual screen.

Measured (`p11_jax_guards.py` → `out_jaxg.txt`), routing to JAX by making **one layer's eps** a `jnp`
array while every other input stays a plain Python/NumPy value:

| case | NumPy branch | JAX branch |
|---|---|---|
| gain superstrate `n_sup = 1 − 1e-3j` (**concrete**) | `ValueError: gain incidence medium …` | returns **ΣR+ΣT = [−0.848, −0.863]** — negative efficiencies, silently |
| 2-layer stack, manufactured sliver `s = 1.5e-5` of the period, degree 14 | `ValueError` (sliver refusal) | **T₀(E_y) = 1.3417, max ΣR+ΣT = 8.35** — a 735 % energy violation, no warning |
| same, degree 16 / 18 | `ValueError` | 1.34167 / 1.34162, ΣR+ΣT = 8.349 / 8.349 |
| same, degree 12 / 20 (NumPy returns) | 0.7658701 / — | 0.7658918 / 0.7659349 (agrees) |

The gain case is **exactly the audit-M3 2026-07-25 defect** the NumPy path was fixed for ("a gain
superstrate … SILENTLY negated every efficiency (measured tot = [-0.95, -0.82] with no warning)"),
still alive on the JAX twin — and `n_superstrate` is fully **concrete** there, so the documented
"a TRACED value skips the guard" carve-out does not cover it. The asymmetry is `PMMStack`-specific:
the **single-layer** JAX entry points do call `_jpmm_concrete_incidence_guard` (`_core.py:3775`).

The `PMMStack.solve` docstring asserts the opposite of what I measured — *"same physics, same order set,
**forward-identical to NumPy at ~1e-15**"* — and lists as raising only slant / out-of-plane / `stabilize`
/ `retain_internal` / sweep. The absent incidence guard is not mentioned anywhere; the absent sliver
guard is mentioned only inside `_sliver_screen`'s body ("A TRACED stack is outside the guard entirely"),
never in the user-facing docstring, and its consequence (a silently 735 %-wrong answer) nowhere at all.

*Impact*: a differentiable design loop over a `PMMStack` — the LC-QWP inverse-design use case this
library exists for — optimises against unguarded numbers. The sliver case is worse than the gain case
because it is degree-dependent, so it passes any spot check the user makes at a neighbouring degree.

*Fix*: (1) hoist `_jpmm_concrete_incidence_guard(label, self.n_sup, angle)` to **before** the
`_holds_traced()` dispatch — it already degrades correctly on a traced index; (2) run `_sliver_screen`
before the dispatch — the screen is **pure geometry** (wall coordinates + `min_feature`) and needs no
traced value at all, so at minimum the warning can fire; (3) after the twin returns, when `R_eff`/
`T_eff` are concrete (an eager call, not under `jit`), run `_warn_stack_energy`. None of this touches a
traced code path.

---

### **[P2] The default `min_feature` sits at the BOTTOM of the sliver hazard band, so the default exposes the whole band** — `stack.py:1597` (`self.min_feature = float(period) * 1e-5`)

Fixture (`p3b_sliver_scan.py`, `p3d_guard_census.py`): two 0.25 µm layers, Λ = 1 µm, λ = 1.55 µm, 12°,
Si/SiO₂; layer 1 ridge `[0, 0.5]`, layer 2 ridge `[0, 0.5+s]`, so the union manufactures **one** cell of
width `s`. The physics is continuous and **linear** in `s` (measured `dT₀/ds = 0.7825`, reproduced to
three digits from `s` = 1e-6 to 1e-3).

Measured hazard band, guard **disarmed**, degree ladder 10…24:

| `s` (period fraction) | behaviour |
|---|---|
| ≤ 1e-6 | snapped away by `min_feature`; correct at every degree |
| **1e-5 … 8e-5** | **degree-dependent blow-up**: T₀ jumps from the correct 0.7577 to **1.3668** at many degrees, and to **22.38** (s=3e-5, deg 18) and **147.67** (s=8e-5, deg 20) |
| ≥ 1e-4 | correct at every degree (max spread 5e-7) |

So the band is `s ∈ [min_feature, ~8·min_feature]`: **the default snaps away only the slivers that were
already harmless and leaves the entire dangerous decade.** With the guard armed (the shipped default)
this surfaces as: at `s` = 1.5e-5 and 2e-5, **every degree from 10 to 24 is REFUSED** — the stack cannot
be solved at all at the library default.

Reproduced on a **second, independent** fixture (`p3e_band2.py` → `out_band2.txt`: Λ = 0.55 µm,
λ = 0.7 µm, 31°, TiO₂-like 2.35 / 1.46, two layers, guard disarmed), and here the diagnostic is
**degree-SCATTER at fixed `s`**, not distance from the `s = 0` value: a large `s` is a genuinely
different geometry, so a smooth degree-independent drift with `s` is *correct physics*, while an answer
that jumps between branches as `degree` changes is the pathology. Reference `T₀ = 0.199229666`.

At the **library default** `min_feature = 1e-5·P`:

| `s` (× mf) | deg 10 | 14 | 18 | 22 | 26 | verdict |
|---|---|---|---|---|---|---|
| 0.3× / 0.6× | 0.199220 | 0.199220 | 0.199220 | 0.199220 | 0.199220 | clean (snapped away) |
| **1.0×** | 0.264548 | 0.264443 | 0.264642 | 0.198917 | 0.308070 | **scatter** |
| **1.5×** | 0.199142 | 0.264504 | 0.264380 | 0.264460 | 0.199198 | **scatter** |
| **2.0×** | 0.302335 | 0.199130 | 0.199094 | 0.264403 | 0.264468 | **scatter** |
| **3.0×** | 0.302284 | 0.193933 | 0.282317 | 0.199052 | 0.264453 | **scatter** |
| **5.0×** | 0.198950 | 0.198947 | 0.193886 | 0.198952 | 0.282130 | **scatter** |
| **8.0×** | 0.198782 | 0.198781 | 0.198783 | 0.193820 | 0.281859 | **scatter** |
| 15× / 30× / 100× | 0.198390 | 0.198390 | 0.198390 | 0.198391 | 0.198391 | clean |

— i.e. the band is `s ∈ [1×, 8×]·min_feature` on this fixture too, the same decade as the first one, up
to 55 % wrong and scattering by ±5 % between adjacent degrees.

**The cure, measured.** Re-running the identical `s`-ladder with `min_feature` raised:

| `min_feature` | rows showing degree-scatter |
|---|---|
| `1e-5·P` (**library default**) | 6 of 11 (every rung from 1× to 8×) |
| `1e-4·P` | **1 of 11** (only `s = 1.0×`, and only at one degree of five) |
| `1e-3·P` | **0 of 11** — every rung degree-independent to 7 digits |

and where the two settings both leave a sliver unsnapped they agree **exactly** (`s` = 1.5e-4: 0.198390
under both; `s` = 3e-4: 0.197553 under both; `s` = 1e-3: 0.193660 under both), so raising the knob does
not perturb the cases it does not touch. That is the direct evidence the recommendation below needs.

*Impact*: a staircased/tapered stack at the library default is systematically driven into the refusal
(or, on the JAX path, into finding P1).
*Fix*: documentation + default, not algorithm — and now with a measured target. Raise the default to
~`period·1e-3` (which eliminated the pathology outright on this fixture, and is the scale the refusal
itself already prescribes via `mf_fix = 2·w_wide·P`), or at minimum state in `PMMStack.__init__` that
the default is a *float-noise* snap and that any stack with cross-layer wall collisions needs
`min_feature` ≳ 10× the collision scale. The existing docstring says the default is "~200x too small"
for a 2-deg taper; this generalises it — the default sits one decade *below* the band it must clear,
and two decades above it the band is gone.

---

### **[P2] The round-4 sliver arbiter pays 3 extra full stack solves on EVERY solve of a stack that carries a manufactured sliver** — `stack.py:1350–1357`, `_sliver_arbiter` at `stack.py:977`

Round 4 deliberately removed the super-unity precondition ("what decides whether the arbiter runs is now
the GEOMETRIC screen"). The arbiter then runs `_sliver_probe_solve` + two `_sliver_collapse_solve`
calls. Measured deterministically by wrapping `PMMStack.solve` and counting invocations
(`p10e_arb_count.py` → `out_arbc.txt`), six stack configurations:

| n_slices | sliver `s` | solves, guard ON | guard OFF | ratio | |
|---|---|---|---|---|---|
| 4 | 3e-4 | 4 | 1 | **4.00** | |
| 8 | 3e-4 | 4 | 1 | **4.00** | |
| 8 | 1e-4 | 4 | 1 | **4.00** | |
| 16 | 5e-5 | 4 | 1 | **4.00** | |
| 4 | 0 | 1 | 1 | 1.00 | *control — no sliver* |
| 1 | 0 | 1 | 1 | 1.00 | *control — single layer* |

The two controls are what make the **scope** of the finding measured rather than inferred: the cost is
paid on exactly the stacks that carry a manufactured cross-layer sliver, and is zero otherwise. The 4×
is flat in `n_slices` (4 → 16), because the arbiter runs three *whole-stack* re-solves regardless of
depth — so it scales with the stack, and again with every wavelength of a sweep.

*Provenance note.* The first version of this probe was **wrong** and I caught it: it monkeypatched
`_core._sem_modes_tensor`, but `stack.py` binds that name at import, so the patch never took and it
counted 0 for both arms. The corrected version wraps `PMMStack.solve` itself. The stale job then
finished late and briefly interleaved its zeros into `out_arbc.txt`; the table above is the completed
corrected run, read after both jobs exited. It is reproduced **row for row** by an independent second
process (`out_arbc_v2.txt`), so the table is not an artefact of that interleaving.

Every tapered/staircased stack carries such slivers (that is the population the feature targets), and the
cost multiplies through `solve_vs_wavelength`, where it is paid per wavelength.

*Why this is a solve COUNT and not a wall-clock figure.* I tried the timing version first
(`p10b_arbiter_cost.py` → `out_arb.txt`) and it is **unusable** under the sibling-auditor load: three of
its five rows read a ratio **below 1.0** (0.90, 0.92, 0.86), i.e. the guard apparently making the solve
*faster*, which is impossible since it strictly adds three solves. Magnitudes were equally broken
(818 s for an 8-layer degree-20 solve). Its one non-contradictory row is the no-sliver control at
0.012 s vs 0.014 s — same order, consistent with the counted 1×, but far too noisy to lean on. The
deterministic count is the claim; do not re-attempt the timing under load.

*Fix*: memoise the verdict on the key it actually depends on — the wall coordinates, `min_feature`,
`degree`, `wl`, `angle` — so a sweep pays it once; and make the two `_sliver_collapse_solve` calls lazy
(only the branch that needs `d12`, i.e. when `d0` already cleared the geometric floor, uses them).

---

### **[P2] Every SEM nodal MASS operator is EXACTLY diagonal, and every one of them is inverted / multiplied densely; `Q @ W2` is additionally computed twice** — `_core.py:2396–2407, 2467, 2489, 2539–2572`, `_core.py:1678`

Measured (`p10_perf.py` → `out_perf.txt`): at every degree and segmentation tested,
`mass["one"/"eyy"/"exy_xx"/"eyx_xx"/"schur"/"ezz"/"inv_xx"/"inv_ezz"]` and the scalar `S0`/`Peps`/`Pinv`
read `max|offdiag| = 0.000e+00`, `nnz_offdiag = 0`. GLL mass lumping makes this **structural**
(`Mloc = np.diag(wel)` and the only overlaps are shared boundary nodes), not accidental. Yet:

* `_sem_modes_tensor` runs `iS0 = _safe_inv(S0)` and `Cxx = _safe_inv(Cinv_xx)` — dense LU inverses of
  diagonal matrices. Counted over a full stack solve: **5 of 5** `_safe_inv` calls are on exactly
  diagonal matrices (n = 64 and n = 144), and `_safe_inv(S0) == np.diag(1/np.diag(S0))` to **0.00e+00**.
* `Cinv_xx, EXY_XX, EYX_XX, SCHUR, Cxy, Cyx, Cyy` are diagonal×diagonal products formed as dense `n³`
  gemms; `iS0 @ op` (diag×dense) is a dense gemm where a row scale would do; `G @ Cxx`, `G @ Cxy` are
  dense×diag gemms.
* `_sem_modes` runs `invop = _safe_solve(S0, Pinv)` — a dense solve of a diagonal system.
* **`Q @ W2` is built twice** — `_core.py:2467` (`V0`) and `:2489` (`V2`) — and each
  `… @ np.diag(v)` is a further full `(2n)³` gemm, so that is `4·(2n)³ = 32 n³` of gemm. `V2` differs
  from `V0` **only** by the per-column sign of the branch flip, so
  `V2 = V0 · where(flip & (|lam0| ≥ 1e-12), −1, +1)` is exact and `O(n²)`; `(2n)³ + O(n²)` suffices.

A/B with a diagonal-aware `_safe_inv`/`_safe_solve` only (`p10c_diag.py`, interleaved, min-of-7):
output **bit-identical** (`max|Δ| = 0.00e+00` on R, T and the Jones) at **1.02–1.14×** end-to-end.
Here bit-identity is *expected and meaningful*: replacing a dense LU inverse of an exactly diagonal
matrix by `diag(1/d)` is arithmetically exact, so there is no last-bit to move. That the patch was
genuinely live is established separately, by the call counter (5 of 5 diagonal calls observed), which
can only fire on a live patch.

The `Q@W2` reuse is the larger of the two and I could **not** measure it. My attempt
(`p10d_modes_opt.py` → `out_opt.txt`) is **invalid by construction** and is reported here only so nobody
repeats it: it rebinds `_core._sem_modes_tensor`, but `stack.py` does `from ._core import
_sem_modes_tensor` at import, so the patch is invisible to the caller — the same import-binding mistake
I made in the arbiter counter. Its own output proves the patch was dead: `max|Δ| = 0.00e+00` on all
three rows, which the real optimisation *cannot* produce (it swaps two gemms for a column scale and a
row scale, both of which perturb the last bits). So that figure is shipped-vs-shipped. By flop count the
duplication is ~16 n³ of waste against ~344 n³ per layer (below), i.e. **~5 %, unverified**.

*Fix*: carry the masses as 1-D diagonals in `mats` (`mats["massd"][k]`); replace `A @ np.diag(v)` with
`A * v[None, :]` throughout; compute `QW = Q @ W2` once.

---

### **[P3] The module's own headline performance claim does not hold on the stack path** — `_core.py:284`

> "The dominant cost of a PMM solve is the dense `np.linalg.eig` on the layer generator (~85% of runtime)"

Deterministic call counts from a cProfile of an 8-layer degree-24 `PMMStack.solve`: `np.linalg.inv`
**42 calls**, `np.linalg.solve` **18**, `np.linalg.eig` **8**. Flop accounting for an `L`-layer shared-grid
stack at `n = n_glob` (complex `zgeev` with eigenvectors ≈ 25N³; `2N = 2n`):

| term | per layer | share |
|---|---|---|
| `eig(Mbig)` 2n×2n | 25·(2n)³ = **200 n³** | **58 %** |
| `_sem_modes_tensor` assembly (`Q@W2` ×2, `@diag` ×2) | 32 n³ | 9 % |
| interface (2 solves + 1 inverse + 3 gemms on 2n) | ≈ 48 n³ | 14 % |
| Redheffer star (2 inverses + 6 gemms on 2n) | ≈ 64 n³ | 19 % |

i.e. the eig is ~**58 %**, not 85 %, and a third of the work is in the interface/star inverses. The
practical consequence is that the note steers optimisation away from where the calls actually are.

**Honest limits on this one.** The flop model is arithmetic, and the call counts are deterministic, but I
could **not** validate the split empirically. Two independent attempts both came back unusable under the
sibling-auditor load: the cProfile (inv 56 % / solve 27 % / eig 15 %), and a dedicated interleaved
min-of-9 kernel timer (`p10f_split.py` → `out_split.txt`) which returned 490 ms for a 144×144 complex
eig (should be ~5 ms), 15 s for a 144-wide interface (~2 ms), and `inv(384)` = 3.78 s against
`inv(320)` = 15.2 s — **non-monotonic in matrix size**, i.e. pure contention, not signal. So: the "~85 %"
claim is contradicted by the call counts and by the flop model, and I am confident the eig is not 85 %
on a multi-layer stack — but the exact share should be re-measured on a quiet machine before anyone
acts on the 58 % figure. Do not repeat the wall-clock attempt under load; it yields nothing.

---

### **[P3] `elements_per_region > 1` + `grade=True` is documented as "the speed lever for TM (hp-refinement)"; measured, it is a pessimisation at matched DOF** — `oned.py:365–376`; same wording in `PMMStack.__init__`

Fixture: Au `n = 0.18 + 3.43j` / air, Λ = 0.6 µm, λ = 0.633 µm, d = 0.1 µm, f = 0.5, 10°, **TM**, order-0
reflectance. Reference `R_∞ = 0.415514656` from an `R = R_∞ + c/n` fit to the library's own RCWA at
`n_orders` = 401/601/801/1201 — an extrapolated oracle this problem *needs*: the raw `n_orders = 401`
value (0.415478859) is itself **3.6e-5** from the limit, which is the whole size of the effect, and
grading the PMM against it produces a spurious "PMM stalls at 4e-5" reading.

| DOF = 2·n_el·degree | `eper=1` | `eper=3, grade=True` | `eper=3, grade=False` | `eper=6, grade=True` |
|---|---|---|---|---|
| 48 | **4.36e-5** | 1.07e-4 | 1.51e-4 | — |
| 96 | **≈5.6e-6** (interp. deg 44→60) | 2.00e-5 | 2.94e-5 | 2.05e-5 |
| 120 | **1.89e-6** | — | — | — |
| 144 | — | 5.61e-6 | 9.40e-6 | 5.98e-6 |
| 192 | — | 9.90e-7 | 2.97e-6 | 1.23e-6 |

Grading *does* beat non-grading at fixed `eper` (1.5–3×), but **single-element p-refinement beats both at
matched DOF by 2.5–4×**, and the eig cost goes as DOF³. So the knob buys accuracy per *element*, never
per *flop*, on the metal-corner case it is advertised for.

*Fix*: correct the docstring, or replace the uniform-degree Chebyshev–Lobatto grading with a genuine
hp mesh (geometric grading σ ≈ 0.15 **with a linearly decreasing degree toward the corner**,
Babuška–Guo), which is the construction that actually gives `exp(−c√DOF)` for a corner singularity.

---

### **[P3] "Converges SPECTRALLY … with no accuracy floor" is true for TE and false for TM, and the Jones entry point carries the claim unqualified** — `oned.py:71–79` (`pmm_jones_1d`)

| cell | 8 | 12 | 16 | 20 | 24 | 32 | 44 | 60 | local rate |
|---|---|---|---|---|---|---|---|---|---|
| **TE**, Au/air | 9.4e-6 | 2.6e-7 | 1.8e-8 | 4.9e-9 | — | — | — | — | ≈ 9, rising → **spectral** |
| **TM**, Au/air | 5.4e-4 | 2.2e-4 | 1.1e-4 | — | 4.4e-5 | 2.1e-5 | 8.0e-6 | 1.9e-6 | 1.9 → 4.7 |
| **TM**, lossless n = 3.48/1 | 8.3e-4 | 2.7e-4 | 1.2e-4 | — | 3.9e-5 | 1.8e-5 | 7.8e-6 | — | flat **2.7** → purely algebraic |

At degree 28 the TM order-0 error is ~2.4e-6 where TE is ~2.8e-11 — five orders apart on the same cell,
and the lossless high-contrast TM case shows **no acceleration at all** (O(N^−2.7) out to degree 44), so
this is the wall-corner singularity, not the metal. `pmm_efficiency_1d`'s docstring does carry a
"TM … only spectral-*ish*" caveat; **`pmm_jones_1d`'s says flatly "Converges SPECTRALLY in the polynomial
`degree` with no accuracy floor"** — and `pmm_jones_1d` is the physics `PMMStack` (the default multilayer
path) runs. Add the caveat there and to `PMMStack`'s class docstring.

---

### **[P3] `CONVENTIONS.md` §7.1 mislabels the 1-D Jones basis, and the mislabel hides a sign** — `CONVENTIONS.md:196–204`

§7.1 says *"The 1-D solvers return `te`/`tm` (`s`/`p`)"* and that at `phi = 0` the bases coincide "up to
the `tm` ↔ `x`, `te` ↔ `y` identification". Measured on a uniform slab (n = 2.1, d = 0.32 µm, λ = 0.55 µm,
n_sup = 1, n_sub = 1.5) against an independent analytic TMM (`p12_jones_basis.py`):

| angle | `pmm J[0,0]` / analytic `r_p` | `pmm J[1,1]` / analytic `r_s` | `rcwa J` / `pmm J` (both entries) |
|---|---|---|---|
| 0° | **−1.000000 + 0.000000j** | +1.000000 | +1.000000 |
| 30° | **−1.000000 − 0.000000j** | +1.000000 | +1.000000 |
| 60° | **−1.000000 − 0.000000j** | +1.000000 | +1.000000 |

Both 1-D Jones solvers return the **lab Cartesian (E_x, E_y)** basis (their own docstrings say so), whose
`xx` entry is `−r_p` in the standard Fresnel p convention; at normal incidence they correctly give
`J_xx = J_yy` (lab-frame isotropy), which a true `te/tm` matrix would not. The library is internally
consistent (PMM and RCWA agree to 1e-15), but §7.1 is the declared source of truth and says something
different: a consumer taking it literally picks up a sign on the p row/column at `phi = 0`.

*Fix*: §7.1 → "the 1-D solvers return the lab Cartesian `(E_x, E_y)` Jones; at `phi = 0` the `x` column is
the p channel **up to the sign of the p unit vector** (`J_xx = −r_p`)."

---

### **[P3] `PMMStack.solve` is 479 lines, and the far-field order-budget block is copy-pasted ~15 times** — `stack.py:2599`; counted across `_core.py` ×6, `stack.py` ×7, `conical.py` ×1–2

AST-measured function sizes: `PMMStack.solve` 479 lines (454 code), `internal_field` 322,
`solve_vs_wavelength` 310, `_solve_conical` 247, `_solve_vs_wavelength_perlayer` 224,
`_solve_vertical_perlayer` 206; `_core.py::_build_generator_metric` 226.

Counted occurrences of the order-budget idiom (`m_prop = _n_propagating_orders` → `n_proj = max(…)` →
`cap = …` → `if n_proj % 2 == 0` → `"too low to resolve the"`): **6 in `_core.py`, 7–8 in `stack.py`,
1–2 in `conical.py`**. This is the exact multi-copy shape this codebase's own audits blame for its three
worst recent defects — the six-copy factor-i defect (S1-8), the six-copy `_sqrt_decay` branch-cut defect
(round 2: "this function had SIX independent bodies … each still carrying the EXACT `r.real == 0` pin
that round 1 removed"), and the T3-3 conical order-cap defect, which was *one copy of this very block*
computing the cap from the union `n_glob` instead of the window half-spaces'. The consolidation template
already exists in the same file (`_forward_branch_flip`, `_mass_flux_threshold`,
`_perlayer_window_grids`): extract
`_farfield_order_set(period, wl, n_max, ffo, n_glob_cap, label) -> (orders, kx, half)`.

---

### **[P3] `PMMStack.internal_field(pol=…)` takes an integer 0/1 where the whole family takes `'te'`/`'tm'`/`'s'`/`'p'`** — `stack.py:3723–3728`

`internal_field(..., pol="x")` raises `ValueError: pol must be 0 or 1`. CONVENTIONS §7 pins that the
`s`/`te` and `p`/`tm` aliases are "accepted everywhere (case-insensitive)"; this is the one place in the
PMM surface that is an index into the Jones columns instead. Accept the strings (mapping
`te/s → 1`, `tm/p → 0`, matching the rows of the returned `R_eff`/`T_eff`).

---

## Performance opportunities

1. **Diagonal mass operators + the duplicated `Q @ W2`** (finding above): bit-identical, 1.02–1.14×
   measured for the `_safe_inv`/`_safe_solve` half alone; ~5 % more from the `Q@W2` reuse by flop count.
2. **`_sem_fourier_projection`** (`_core.py:1818`) scatters with a Python loop over `n_el·(degree+1)`
   global nodes (`for a in range(degree+1): T[:, idx[a]] += contrib[:, a]`);
   `np.add.at(T, (slice(None), idx), contrib)` is one call.
3. **Sliver-arbiter memoisation** (finding above): measured **4×** on every solve of a staircased stack,
   multiplied again by `solve_vs_wavelength`.
4. **`_GEO_EIG_CACHE`** (`_core.py:2504`) is a 64-entry LRU whose **key** is the full `B.tobytes()`
   (`n_glob²` complex128) — at a production `n_glob = 300` that is ~1.4 MB of key *plus* ~1.4 MB of value
   per entry, up to ~180 MB retained, and unlike `_PERLAYER_GEO_CACHE` it is **not** enrolled with
   `ByteBudgetedLRU`/`LUMENAIRY_CACHE_BUDGET_MB`. A `blake2b` digest drops the key to 32 B; enrolling it
   bounds the values. (Building `B.tobytes()` is also an `O(n²)` copy on every call.)
5. **Cross-wavelength eig reuse at OBLIQUE incidence.** The geometric eig is cached on the operator bytes,
   which bake in the *absolute* `kx0 = n·sinθ·k0`, so a fixed-angle wavelength sweep re-eigs at every
   point (`_core.py:2563` says so). Reformulating the pencil in the *dimensionless* `kx0/k0` and carrying
   the `k0` scaling analytically — the same trick already used for the `1/k0²` factor — would make the
   half-space eig wavelength-independent at every angle, not only at normal incidence. That is the
   51–64 % of 1-D eig time the S5-P1 note already banked, recovered for the oblique sweep too.

## Alternative algorithms / methods

* **(a) FMM with adaptive spatial resolution (Granet 1999; Granet & Guizal 1996).** Not worth switching
  to: the SEM basis puts the ε jump exactly on an element boundary, so it has **no Gibbs floor at all in
  TE** (measured spectral, 4.9e-9 at degree 20 on a *metal* grating) where ASR-FMM still has one. ASR's
  only advantage is in TM, where both are algebraic.
* **(b) The C-method (Chandezon 1982; Granet 2001).** Wrong tool here — it needs a *smooth* profile and
  degrades on a lamellar one. It is, however, the right tool for the sinusoidal/blazed profiles that
  `PMMStack` currently has to staircase, and adopting it for those would remove the staircase and with it
  the entire sliver-guard mechanism for that class.
* **(c) Gegenbauer / ultraspherical subsectional modal method (Edee, Plumey & Guizal 2013; Edee 2011).**
  This *is* the implemented family, in its GLL-nodal realisation. The concrete upgrade the literature
  offers is exactly the gap measured above: an ultraspherical (Gegenbauer) basis recovers exponential
  convergence for the **TM wall-corner singularity** where the Legendre/GLL basis is `O(N^−2.7)`. It is a
  basis swap in `_gll_nodes_weights` + `_lagrange_derivative_matrix`, not an architectural change, and it
  is the highest-value algorithmic move available to this partition.
* **(c′) Classical hp-FEM corner mesh (Babuška–Guo).** Geometric grading σ ≈ 0.15 toward the wall **with a
  linearly decreasing polynomial degree** gives `exp(−c√DOF)` for a corner singularity; the library's
  `grade=True` grades at *uniform* degree, which is the construction measured to be a matched-DOF
  pessimisation. `_graded_boundaries` already produces the mesh — only the per-element degree is missing.
* **(d) Aperiodic FMM with PML (Lalanne & Silberstein 2000; Hugonin & Lalanne 2005).** Out of scope for a
  strictly periodic cell, but the SEM basis takes a complex-stretched coordinate for free (a PML is just a
  complex element Jacobian in `_build_sem`), so this is an unusually cheap way to give `PMMStack` an
  isolated-structure mode.
* **(e) Eigen-recycling across a sweep** (Jacobi–Davidson, or a Newton correction seeded from the previous
  sweep point). Already explored for 2-D in `docs/audits/EXPERIMENT_PMM2D_EIG_RECYCLE_2026_08_16.md`;
  applies unchanged in 1-D and is the natural partner to opportunity 5.

## Code organization observations

* `_core.py` 7514 lines / 191 functions and `stack.py` 5050 / 79 in one namespace each. The comment mass
  is extreme (`_forward_growth_flip` is 131 lines of which **20** are code). The comments are genuine
  audit history, but the *sequence of superseded claims inside a single docstring*
  ("UPDATE 2026-08-05 … UPDATE 2026-08-08 … The first bullet above was WRONG") is itself a hazard: a
  reader who stops early reads a retracted statement as current. Move the history to `docs/audits/` and
  leave the current contract plus a pointer.
* Eight near-duplicate cascade implementations (`PMMStack.solve` shared / per-layer-vertical /
  per-layer-general / conical / sweep / sweep-per-layer, plus `_pmm_solve_core`, `_pmm_jones_solve_core`
  and the two JAX twins) each re-implement the grid build, the eig memo, the Redheffer loop, the order
  budget and the far-field tail. Only the far-field tail is genuinely shared
  (`_assemble_jones_farfield`).
* `_resolve_order_count` / `_resolve_incidence` implement an "alias wins, **no** equality check" rule that
  is deliberate and test-pinned — but it means `set_source(angle=A, theta=T)` with `A ≠ T` silently solves
  `T`. "Pass only one spelling in practice" is not enforceable; raising on a mismatch and updating the
  test would be strictly safer.
* `pmm_jones_1d` mixes conventions in one signature: `eps_ridge`/`eps_groove` are **permittivities** while
  `n_substrate`/`n_superstrate` are **indices**. The docstring flags it ("a wrong-convention value is
  silently accepted"); an `eps_substrate`/`eps_superstrate` alias would at least let callers be
  unambiguous.

## Unverified suspicions

* `_pmm_union_grid`'s hard pre-merge at `tol = 1e-9` (`_core.py:4524–4531`) is unconditional and, unlike the
  `min_feature` snap immediately below it, does **not** apply the single-layer-ownership rule — a genuine
  within-layer feature narrower than 1e-9 of a period would be merged silently and without a warning.
  At 1 fm on a 1 µm pitch this is unreachable in practice; not tested.
* `_energy_clean_pick` (`_core.py:2961`) tie-breaks on `|total − 1|`, which on a **lossless** structure is
  ~1e-14 at every degree — so the pick is decided by round-off among the cluster. I looked for a case
  where that returns a materially worse degree than requested and did not find one (CFC-19), but the
  tie-break is structurally unprincipled on exactly the class of structure it targets.
* `_stabilize_*` calls `solve_at_degree` for up to 16 degrees, each rebuilding the whole SEM grid and
  re-eigging; only the degree-keyed reference machinery is memoised. A sparse-degree probe is explicitly
  flagged as risky at `_core.py:296`; not tested.
* The covariant/metric slant generators were read for contracts only. `_COV_MIN_SLANT_RAD = 1e-3` is a
  hard routing threshold whose continuity across the switch I did not test (I did verify the
  *convection* slant path's `slant → 0` limit: see CFC-21).
* `_guarded_mortar_solve` (`_core.py:5328`, 156 lines — the "own conditioning decision for the
  generalized mortar site" of the round-3/4 commits) is reached only from the **2-D staggered** mortar
  sites, not from the 1-D `_interface_smatrix_mortar` (which uses `_guarded_solve`). I read it but it
  belongs to the PMM-2D partition and I did not exercise it.

## Checked and found correct

**CFC-1 Formulation.** Both weak forms re-derived from Maxwell independently; the `(Cinv − Cinvᵀ)`
antisymmetrisation, the `Peps`/`Pinv` roles, the `V = (1/ε)·q·W` TM partner and all four interface
S-matrix blocks are right. `_propagation_smatrix` uses `exp(−lam·k0·L) = exp(+iγL)` with `Im γ ≥ 0`, so
`|X| ≤ 1` — **no T-matrix growth path exists anywhere in the 1-D stack.**

**CFC-2 Mode spectrum vs the exact Botten-1981 transcendental relation** (`p1_modes.py`,
`p1b_lossy_modes.py`; my own dense real-line bisection and complex Newton sweep, no library code).
SEM eigenvalues match the exact roots in `u = (γ/k0)²` to **≤ 3.8e-13** (lowest 10 modes) at degree 32,
for Si/air Λ=λ, Si/air Λ=3λ (49 roots above u = −60), Si/SiO₂ f=0.3, at 0° and 25°, TE and TM. Residual
`|F(u_SEM)|/|F′|` ≤ 4.5e-13 lossless and ≤ 4.1e-13 for Au/air (complex roots). **No mode missed, none
duplicated.** (At degree 28 the deepest evanescent roots, `Re u < −10`, are converged only to 1e-3–1e-5;
they carry no far-field power.)

**CFC-3 Forward branch / branch cut.** Over 5 fixtures × 2 polarisations × 56 modes: **0 modes with
`Im(q) < 0`** and **0 near-real modes with `Re(q) < 0`** after `_forward_branch_flip`. The rule is the
correct `exp(+iγz)`-decay rule. `tol = 1e-8·max(|q|, 1)` is scale-free because `q = γ/k0` is
dimensionless — I checked this specifically; it is *not* another instance of the `_mass_flux_cut` unit
bug.

**CFC-4 "The on-cut flip is `−r`, not `conj(r)`"** (`rcwa/_core.py:1429`) — independently confirmed
**correct**. `±r` are the two *exact* roots of the computed `lam²`; `conj(r)` is not a root of `lam²`
unless `lam²` is real, so it departs from the exact forward/backward involution by `2·Re(r)`, which the
band admits out to `1e-8·scale`. The S-matrix assembly *is* exactly invariant under `lam → −lam` with
`V → −V`. Non-holomorphy of `conj` additionally breaks the reverse-mode cotangent. **The 1-D PMM was
never affected**: `_forward_branch_flip` has always returned `where(flip, −q, q)`. I also checked the
gauge consistency: RCWA is internally `exp(+iωt)` (outgoing = `+i|kz|`) while the 1-D PMM is public
`exp(−iωt)` end-to-end (outgoing = `lam = −i·q`, `Im q ≥ 0`) — two different but individually correct
conventions, confirmed by the TMM/energy/reciprocity checks below.

**CFC-5 Orthogonality / biorthogonality** (`p2_ortho.py`), overlap matrices built exactly as the code
builds its operators, unit-normalised columns, reported as `max|offdiag| / max|diag|`:

| case | `Wᴴ B W` | `Wᵀ B W` | `W(−kx0)ᵀ B W` |
|---|---|---|---|
| lossless, normal | 2.5e-13 | 2.5e-13 | 2.5e-13 |
| lossless, 25° | **6.3e-14** | 9.3e-01 | 6.3e-14 |
| lossy (Au), normal | 3.7e-01 | **6.6e-14** | 6.6e-14 |
| lossy (Au), 25° | 3.8e-01 | 4.9e-01 | **1.0e-12** |

Exactly the textbook structure: Hermitian-orthogonal when the pencil is Hermitian (lossless),
unconjugated-orthogonal when complex-symmetric (lossy at normal), and **bi**orthogonal against the
adjoint (`−kx0`) problem otherwise. TM is weighted by `Pinv` (1/ε) throughout, as required. The code does
not *rely* on orthogonality (the interface uses explicit inverses), so this is a consistency check — and
it passes.

**CFC-6 Unpatterned limit vs analytic Fresnel/TMM** (`p6_unpatterned.py`, my own TMM, not `coatings.py`).
n_layer = 2.1, d = 0.32 µm, λ = 0.55 µm, Λ = 0.4 µm, n_sup = 1, n_sub = 1.5: `|ΔR|, |ΔT| ≤ 3.8e-14` at
0°/30°/60° for both `te` and `tm`; through `pmm_jones_1d`, `Δ|r| ≤ 2.3e-14` and **Δphase ≤ 1.7e-13 rad**
— **no reference-plane error**. Cross-pol `|J_xy| ≤ 9e-45`. Lossy layers (n = 2+0.1j and 0.2+3.5j,
d = 0.05 µm, 0° and 40°): `≤ 9.5e-14`. No power leaks into `m ≠ 0` (`< 1e-14`).

**CFC-7 TE/TM labelling vs CONVENTIONS §7.** At 30°, `polarization='te'` reproduces the **s** TMM
(0.2692774850 vs 0.2692774850) and `'tm'` the **p** TMM (0.1688348080 vs 0.1688348080). A swap would have
read as a 0.10 discrepancy at 30° and 0.35 at 60°; it does not.

**CFC-8 Energy, lossless Si/SiO₂ lamellar** (Λ = 1 µm, λ = 1.55 µm, d = 0.45 µm, f = 0.5), 0°/17°/45° ×
te/tm × degree 16/24/32, 18 cells: `|ΣR+ΣT−1| ≤ 3.1e-12`, median ~2e-14.

**CFC-9 Reciprocity.** `T(0 → m)` at 17° vs the reverse illumination at the order's exit angle with the
half-spaces swapped: TE `m = −1` 0.123683315125 vs 0.123683315125 (Δ = 1.5e-14), `m = 0` Δ = 4.7e-14;
TM `m = −1` Δ = **1.8e-11**, `m = 0` Δ = 2.6e-14.

**CFC-10 Against the library's own RCWA at `n_orders = 201`**: max per-order `|Δ|` 4.6e-8 (TE 0°),
7.5e-8 (TE 17°), 1.2e-6 (TE 45°), 3.4e-6…1.2e-5 (TM at degree 28, where PMM is not yet converged).
Absorbed fraction on the Au cell: PMM `A = 0.052611` vs RCWA `A = 0.052711` at n=401 (5e-5 apart, which
is the oracle's own residual — see the P3 convergence finding).

**CFC-11 S-matrix stability.** A 200 µm uniform layer with κ = 0.3 (≈ 3.8e5 nepers) reproduces the
analytic TMM to full double precision (R = 0.1350310987 vs 0.1350310987, T = 0 vs 0) for both s and p at
20°; likewise 50 µm and 10 µm at κ = 0, 0.01, 0.3 — 24 cells, all exact to the printed 10 digits. A
50 µm **absorbing grating** layer returns finite, sub-unity R and T. Splitting one 0.45 µm grating layer
into 1 / 4 / 12 / 40 identical sub-layers returns a **bit-identical** Jones
(`0.0898671568 − 0.0032436835j` in all four).

**CFC-12 JAX twin parity and gradients.** `PMMStack` NumPy vs JAX: `max|ΔR| = 5.9e-14`, `|ΔT| = 8.7e-14`,
`|ΔJ| = 1.3e-13` (x64 enforced). `jax.grad` w.r.t. `eps_ridge` is finite and matches central FD to
**5.8e-8 relative** (AD 0.21751584296, FD 0.21751585555). (Parity and gradients are fine; it is the
*guards* that are missing — finding P1.)

**CFC-13 Conical** (`p7_conical.py`). Cross-pol vanishes exactly linearly as φ→0: `|J_xy|` =
6.474e-3 / 6.474e-6 / 6.474e-9 / 6.398e-12 at φ = 1° / 1e-3° / 1e-6° / 1e-9°, and 1.5e-32 at φ = 0.
φ = 0 conical reproduces the classical `pmm_jones_1d` to **2.3e-13** and the scalar `te`/`tm` efficiencies
to 1e-13. φ = 90°: cross-pol 4.6e-12. `pmm_jones_1d_conical` and `PMMStack(..., phi=π/2).solve()` are
**bit-identical** (`max|ΔJ| = 0.0`). Energy over φ ∈ {0,30,60,90}° × degree ∈ {10,14,18,22}:
`|tot−1| ≤ 6.2e-11`. I also verified the conical far-field normalisation
`einc² = 1 + ((kx0·ex0 + ky0·ey0)/kz)²` analytically from `Sz = ½/(ωµ0)·kz·(|Ex|²+|Ey|²+|Ez|²)` — correct.

**CFC-14 Mortar** (`p3c_mortar.py`). On identical grids (`|M_a − M_b| = 0`, `|C_ab − M_a| = 1.2e-22`) the
mortar reduces to `_interface_smatrix` at relative **≤ 4.3e-14** in all four blocks — the algebraic
identity holds numerically. On a 3-layer non-conforming stack, per-layer vs shared: `max|ΔJ|` =
1.85e-5 / 6.51e-6 / 2.95e-6 at degree 10 / 14 / 18 (≈ O(deg⁻³)); the per-layer path's own closure defect
is 2.4e-7.

**CFC-15 The round-4 sliver guard: 0 false negatives in 88 cells** (`p3d_guard_census.py`), on a fixture
that is **not** one of the audit's own. `s` ∈ {1.0, 1.2, 1.5, 2, 3, 4, 5, 6.5, 8}e-5 ∪ {1, 1.5}e-4 ×
degree ∈ {10…24}: every cell whose unguarded answer is wrong (1.3668, 22.38 or 147.67 against a correct
0.7577) is **REFUSED**, and every cell that returns a value returns the correct one to ≤ 1.6e-4. The
returned answer is continuous in the sliver width across the trigger (linear, `dT/ds = 0.7825`, held from
`s` = 1e-6 to 1e-3), so the guard introduces **no discontinuity in what it returns**. On this evidence it
is a correct and complete guard for the shared-grid NumPy path; its two gaps are *scope* (JAX — P1) and
*cost* (4× — P2).

**CFC-16 Caches and thread-safety** (`p9_cache_threads.py`). The geometric-eig caches are keyed on the
**operator bytes**, not on a knob tuple, so a wavelength / angle / `n_ridge` / glass-name change cannot
produce a stale hit: 8 `(wl, angle, n_ridge)` cases replayed after `_clear_pmm_caches()` and again in
shuffled order returned **bit-identical** answers (0 mismatches, twice). 16 concurrent threaded
`pmm_jones_1d` solves matched their serial counterparts to **exactly 0.0**. The cached GLL nodes/weights,
the barycentric derivative matrix and the geometric eigenvectors are read-only.

**CFC-17 Unit (scale) invariance.** The same physical problem expressed in km / m / µm / nm / Å returns
`R₀ = 0.08536814608764 ± 6e-14` and the same `J₀₀` to 1e-12 (13 decades of length scale). The
`_mass_flux_cut` unit-safety repair holds and no other absolute-length constant remains on the 1-D path.

**CFC-18 `PMMStack.layer_absorption()`** closes to `1 − ΣR − ΣT` at **≤ 7.5e-13** on a 3-layer stack with
three different lossy media, at degree 12 / 16 / 20 / 26, for both incident polarisations; the
`by_material` split reproduces the per-layer array exactly; and on a **lossless** stack the total
absorption reads `−4.6e-14 / −1.6e-12` (i.e. zero).

**CFC-19 `stabilize=True` (the default) is never materially worse than `stabilize=False`** at the same
requested degree: over 6 structures (3 geometries × te/tm) × 8 degrees, `err(True)/err(False)` is
0.40–1.00 in 47 of 48 cells and 1.80 in one (both errors ~1e-6). It typically **improves** the answer by
20–60 %.

**CFC-20 The Rayleigh-projection capacity clamp works.** `n_proj = min(n_proj, cap)` with
`cap = n_glob | n_glob−1` keeps the projector inside the grid's nodal capacity on every 1-D path, so
`_guarded_lstsq` never has to refuse on the shared single-layer path: degree 4–8 with
`far_field_orders` 41/61 all return with `|ΣR+ΣT−1| ≤ 4e-4` and NumPy/JAX agreeing to 3.4e-15.

**CFC-21 The slanted (inclined-coordinate) solver's `slant → 0` limit** reproduces the vertical solver:
`max|ΔR|, |ΔT|` = 2.1e-10 / 2.4e-9 (TE/TM) at slant 1e-9 rad, 7.0e-9 / 4.5e-9 at 1e-6, and 6.0e-7 /
1.8e-6 at 1e-3 — a smooth `O(slant)` approach with no discontinuity, and `|tot−1| ≤ 1.1e-8` throughout.
At a 20° slant the single-layer inclined-coordinate solve is already converged at degree 16
(R₋₁ = 0.04073747, R₀ = 0.07794570, R₊₁ = 0.01065607, T₀ = 0.07982690 — **identical to 8 digits** at
degree 16, 22 and 28, `|tot−1| ≤ 1.5e-8`), i.e. one layer of 32 DOF replaces the z-staircase entirely.
**The independent z-staircase cross-check now closes, for TE** (`p15b_slant.py` → `out_slant2.txt`):
vertical layers through the *symmetric* cascade — a completely different code path, with no slant
generator, no inclined coordinate and no generalized S-matrix. Staircase convergence in `n_slices`
against the slanted single-layer answer:

| `n_slices` | R₋₁ | R₀ | T₀ | ΔT₀ vs slanted |
|---|---|---|---|---|
| 8 | 0.04090110 | 0.07730068 | 0.07614784 | 3.68e-3 |
| 24 | 0.04075542 | 0.07786931 | 0.07940367 | 4.23e-4 |
| 64 | 0.04073998 | 0.07793480 | 0.07976677 | 6.01e-5 |
| **slanted, 1 layer** | **0.04073747** | **0.07794570** | **0.07982690** | — |

The staircase converges as **O(n_slices⁻²)** (error ratios 8.7 and 7.0 against slice ratios 3 and 2.67),
and Richardson-extrapolating the last two rungs lands on the slanted answer to
**2e-8 (R₋₁), 1.8e-7 (R₀), 7.1e-7 (T₀)**. That is an independent confirmation of the inclined-coordinate
solver, and it quantifies the efficiency claim: 32 DOF in one layer versus a 64-slice staircase that had
not yet matched it (and `n_slices = 128` died with a `MemoryError` on a 3084×3084 union grid — which is
precisely the cross-stack wall accumulation the slant path exists to avoid).

**TM is inconclusive, and honestly so.** The staircase was run at `degree = 12`, where the TM channel's
own truncation error is ~1e-4 (the P3 convergence finding above) — and the slanted-vs-staircase TM gap
after extrapolation is ~9e-5, i.e. the same size. The comparison is therefore consistent but carries no
discriminating power for TM; it neither confirms nor refutes at finer resolution.

**New scope note (not a defect, but a limitation worth stating).** On this *lossless* cell the slanted
TM path reads `ΣR+ΣT` = 1.0000908 / 1.0000604 / 1.0000432 at degree 16 / 22 / 28 — super-unity by
4–9e-5, **decreasing with degree** (so it converges, and is not an instability), while the vertical
cascade on the same solid reads 1.0000000000. The slanted generator therefore does **not** close energy
to machine precision the way the vertical path does; its closure is a convergence diagnostic in its own
right. TE is unaffected (`|tot−1| ≤ 1.5e-8` at every degree). Anyone using `|ΣR+ΣT−1|` as a correctness
tripwire on a slanted TM stack should calibrate against this floor rather than against 1e-10.

**CFC-22 `internal_field` tangential-field continuity across a z-interface — no defect** (I flagged this
mid-audit and then refuted it; recorded because the refutation is the useful part). A naive test reads a
"jump" in Ex (4.4e-3 relative at degree 22) and Hy (3.2e-4) while Ey, Hx and Hz read exactly 0 and Ez
jumps legitimately. It is entirely the **finite sampling gap**: the two probes sit `2Δz` apart, and the
field varies by `k0·n·2Δz` over it. Control (`p14d_fictitious.py`) across a **fictitious** interface —
two adjacent layers with identical segments, where any true jump would be a reconstruction bug:

| region | deg 10 | 14 | 18 | 24 | analytic `k0·n·2Δz` |
|---|---|---|---|---|---|
| uniform (no walls at all) | 2.716e-4 | 2.716e-4 | 2.716e-4 | 2.716e-4 | **2.1e-4** |
| patterned | 1.588e-4 | 1.620e-4 | 1.635e-4 | 1.646e-4 | — |

degree-**independent** and matching the analytic gap; and the residual is spread uniformly over x
(median 1.59e-4 against a max 2.39e-4, top-10 locations scattered, not at the wall). The decisive test
is the **Δz scaling** (`p14e_dz.py`) — a discontinuity is Δz-independent, a sampling artefact is ∝ Δz:

| Δz | Ex | Ey | Hx | Hy | Hz | **Ez** |
|---|---|---|---|---|---|---|
| 1e-10 m | 4.112e-2 | 0 | 0 | 3.159e-3 | 0 | **5.649e-1** |
| 1e-11 m | 4.371e-3 | 0 | 0 | 3.208e-4 | 0 | **5.619e-1** |
| 1e-12 m | 4.398e-4 | 0 | 0 | 3.213e-5 | 0 | **5.615e-1** |
| 1e-13 m | 4.401e-5 | 0 | 0 | 3.214e-6 | 0 | **5.615e-1** |
| 1e-14 m | 4.401e-6 | 0 | 0 | 3.214e-7 | 0 | **5.615e-1** |

Every tangential component scales **exactly linearly** with Δz over four decades (4.401e-5 → 4.401e-6
for a 10× change), while the **normal**
`Ez` — the one component that is *physically* discontinuous — stays put. The extra degree-growth seen on
the 3-layer stack scales as `N²`, which is the highest-|q| evanescent nodal content's own z-derivative
(`|q|max·k0·2Δz ≈ 8e-3` predicted at degree 22 vs 4.4e-3 observed). The nodal `Ex`/`Hy` are read straight
off `W`/`V` (`stack.py:3954`), which the interface S-matrix matches exactly on a shared grid, so
continuity holds by construction — and does.
