# M3 -- "Per-layer, efficiently" (N-3), plus the promoted T3-4 classification guard

Campaign: `PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md`, mission M3.
Predecessors: `PMM_M1_CONDITIONING_2026_08_04.md`, `PMM_M2_WINDOW_CONTRACT_2026_08_04.md`,
`PMM_M5_2D_FEASIBILITY_2026_08_04.md`.

Legend: **[M]** measured here, **[A]** read from the tree, **[H]** hypothesis.

---

## 0. Summary

Six items. Five shipped, one **refused on evidence**, and the refusal is the most interesting of
the six.

| # | item | status | headline evidence |
|---|---|---|---|
| N-3.1 | hoist the geometry (masses + cross-masses) on the three unhoisted paths | **shipped** | bit-identical |
| N-3.2 | vectorise `_lagrange_eval` | **shipped** | bit-identical |
| N-3.3 | de-kron both mortars (`kron(I2, .)` applied separably) | **shipped** | exactness proved; envelope 3.6e-13; `np.kron` bytes -> **0**; peak **unchanged**, and the plan's peak gate is therefore NOT met (S6.2) |
| N-3.4a | `inv -> solve` at the mortar interface | **shipped** | same envelope; M1's census preserved at the same site |
| N-3.4b | `inv -> solve` at `_redheffer_star_rect` | **REFUSED** | it breaks 5 shipped cross-path bit-exactness pins for a ~3 % prize |
| T3-4 | modal forward/backward classification guard (promoted from M2) | **instrument ships; guard ships DISARMED** | the conjunction is 8/8 on the family it was calibrated on and **false-positives 3 of 5 on the conical family** -- no bar survives both (S5.5) |
| M4-1 | the per-worker BLAS-cap defect, at all three PMM sites | **fixed** | 36 sha256 digests identical across worker counts and thread pools; build-independent cap-count pins (S7b.1) |
| M4-2 | `pmm_jones_2d` manufactures 1.8 % energy on a lossless cell | **root-caused; X-1, still open** | M1's census DOES cover the route, and its instrument crosses M1's own bar exactly where the energy appears (S7b.2) |
| M4-3 | T3-7 / T3-5 | **deferred, with reasons** | they are M4 plan items needing their own measured tables; S7b.3 |

**Bit-identity, end to end** [M, both builds]. 346 observables covering all four per-layer paths,
both grid paths, the JAX twin (forward *and* gradient), the conforming-stack null floor and the
RCWA oracle:

| comparison | max abs difference | arrays moved |
|---|---|---|
| pre-M3 vs M3 with the fail-before switch **OFF** (N-3 items 1+2 only) | **0.0 on 345 of 346** | 1 -- the JAX **gradient**, 8.5e-17 (see S2.5) |
| pre-M3 vs M3 with the switch **ON** (items 1-4) | **3.6e-13** | 102 |
| ... restricted to the **shared-grid** path | **0.0 on 140 of 140** | 0 |
| ... restricted to the **conforming** per-layer stack (mortar bypassed) | **0.0 on 8 of 8** | 0 |
| ... restricted to the **RCWA oracle** | **0.0 on 4 of 4** | 0 |

**What was refuted along the way, and it matters:**

* the hoist was expected to be the single-solve win. **It is not** -- a single `solve()` builds
  each geometry once either way, so on a cold cache the hoist can win nothing by construction.
  Its win is entirely on the **repeat**, which is why a fourth timing case was added (S6.1);
* two candidate T3-4 instruments were measured and **refuted** before the third was adopted --
  the near-cut mode count (M2 had already refuted it; re-confirmed) and a physics-derived ceiling
  on the effective index of a mode classified propagating, which reads the same value on broken
  and clean cells alike (S5.3);
* the fail-before switch, as first written, **did not fail before** -- it gated the NumPy arms
  only and the JAX twin's de-kron was unconditional. Caught by the switch-off shadow reading
  5.6e-17 instead of 0.0; fixed (S4.4);
* the de-kron's expected **peak-memory** win does not exist on this device: the peak is set by the
  eig workspaces, and removing 7.6 MB of transient `kron` churn moves it by 2 kB (S6.2). The plan's
  own gate for that claim is recorded as NOT met;
* a **first baseline capture was a tautology** -- the probe modules put the live repo on
  `sys.path[0]`, so "baseline" loaded the current tree and dutifully reported 0.0 (S1.2);
* and two memory defects in M3's own new code were found by measuring rather than by reasoning
  (S6.2): a retained slice VIEW worth +3.8 MB of peak, and an attribution that turned out to be
  wrong once the first was fixed.

---

## 1. Builds, baseline, and devices

### 1.1 Builds [M]

| | Windows | WSL |
|---|---|---|
| Python | 3.14.6 | 3.12.3 |
| NumPy | 2.4.4 | 2.4.6 |
| BLAS/LAPACK | scipy-openblas 0.3.31 (DYNAMIC_ARCH, Haswell) | OpenBLAS (distro) |
| JAX | 0.11.0 | 0.10.2 |

Threads pinned with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` for every number in
this document.

### 1.2 The A/B baseline is NOT `git archive HEAD` [A]

`HEAD` is M1. The working tree also carries M2's verified-uncommitted work and a concurrent M4's
edits to `lumenairy/elements/rcwa/**`. A `git archive HEAD` baseline would therefore attribute
M2's and M4's deltas to M3.

The baseline is **the current tree with exactly the four files M3 touches reverted to their
pre-M3 bytes** (`lumenairy/elements/pmm/{_core,stack,conical,_jax_stack}.py`, saved to
`C:\tmp\m3_pre\` before the first edit), rebuilt immediately before each measurement. The only
difference between the two sides is M3.

**One evidence-integrity failure was made and caught here, and it is worth recording.** The M2
probe modules insert the live repo at `sys.path[0]` when they are imported, so the FIRST baseline
run silently loaded the *current* tree and its comparison was a tautology (it reported 0.0
because it was comparing a tree with itself). The harness now binds `lumenairy` from the
requested root **before** any probe module runs and asserts where it came from
(`M3_ROOT_ASSERT` in `shadow.py` / `timing.py`), so this cannot recur silently. Every number in
S0 is from re-run, root-verified captures.

### 1.3 Devices [A]

* **U** -- M2's audit-class surrogate: 700 nm pitch, 1310 nm, 2 deg sidewall, 310 nm tapered
  region, 5.00 nm conformal Al2O3 coat, `n_sup = n_sub = 1.5`, `theta = 8 deg`. Cross-layer
  separations `{off, |5 - off|}` nm with `off = (310/ns) tan(2 deg)`.
* **W** -- M5's single-region UNCOATED taper (`validation/m5_taper_degree_spread.py`): eps 4 / 1,
  duty 0.5, 2 deg taper, no coat, `theta = 8 deg`. Its only cross-layer separation IS `off`.
* **F1 / F2** -- false-positive controls built for T3-4 (S5.5): F1 = three/four coated ridges of
  different width, untapered, no thin cell; F2 = alternating `n = 3.48` / `n = 1.45` gratings,
  whose layers genuinely support **different numbers of propagating modes**.
* Timing cases: (a) 3-layer synthetic, (b) the 38-layer audit device, (c) a 10-point LC sweep,
  (d) the 38-layer device solved three times back to back.

`min_feature` is set above M2's S3.6 threshold `min(off, |c - off|)` on every **timing** case, so
no perf number is measured inside the S5 silent-wrong regime.

---

## 2. N-3.1 + N-3.2 -- hoist and vectorise, bit-identical

### 2.1 What was rebuilt on every call [A]

`_sem_mass_exact` and `_sem_cross_mass` read exactly five fields of a `mats` dict -- `degree`,
`ref_nodes`, the `(xl, xr)` of every `elem_bnds` entry, `l2g`, `n_glob` -- and nothing else. They
never touch the third (`eps`) slot of `elem_bnds`, never see a wavelength, an angle or a
material. They are pure functions of GEOMETRY.

| path | before | after |
|---|---|---|
| `PMMStack._solve_vertical_perlayer` | per-solve memo, rebuilt every `solve()` | module geometry cache |
| `PMMStack._solve_general_perlayer` | per-solve memo, rebuilt every `solve()` | module geometry cache |
| `conical._conical_nodal_solve` | per-solve memo, rebuilt every call | module geometry cache |
| `PMMStack._solve_vs_wavelength_perlayer` | hoisted per SWEEP, rebuilt each sweep | module geometry cache |
| `_jax_stack._pmm_stack_solve_jax_perlayer` | hoisted per TRACE, rebuilt each trace | module geometry cache |

### 2.2 The key is the content, not the knobs [A]

`_geo_fingerprint(mats)` is derived from the five consumed fields themselves, **not** from
`(period, degree, n_el, grade, min_feature, window_halfwidth, ...)`. Two consequences:

* no future knob can be forgotten from the key -- if it changes the grid it changes the
  fingerprint, and if it does not, the cached value is provably the right one;
* the pin is bidirectional, and both halves are asserted by
  `test_geometry_fingerprint_covers_exactly_what_the_two_builders_read`: perturbing **any** of the
  five fields changes the key, and changing **`eps` only** does not -- *and* the two builders
  return byte-identical output for the two different-`eps` grids, which is the claim the cache
  rests on.

The cache is a `ByteBudgetedLRU` named `pmm_perlayer_geometry`, so `clear_asm_caches()` drains it
and `cache_report()` shows its footprint by name. Values are frozen read-only (the W7 A13
poisoning policy: `test_geometry_cache_returns_the_uncached_bytes_and_is_hit` asserts a write
raises). Footprint is `n_glob^2` float64 per entry and `2 nlay - 1` entries per stack -- 12.8 kB
per entry at the audit device's `n_glob = 40`, 0.72 MB at a production `n_glob = 300`. That is
**O(problem), not O(history)**.

### 2.3 `_lagrange_eval` [A -> M]

The Python loop over quadrature points ran **704 times per solve** on the 38-layer device
(`_sem_cross_mass` calls it twice per union sub-interval) and rebuilt the `O(p^2)` barycentric
weights on every one of those calls. Both are gone: the weights are memoized on the node
coordinates and the point loop is broadcast.

The memo keeps the original arithmetic **verbatim** -- repeated division in the original `k`
order, *not* `1 / prod(...)`, which is a different rounding. That is why a cache hit and a cache
miss return the same bits the pre-M3 library returned.

### 2.4 Byte-identity [M, both builds]

Two independent pins:

* **direct**, against the pre-M3 loop embedded verbatim in the test
  (`test_lagrange_eval_is_bit_identical_to_the_pre_m3_loop`): 9 degrees x 8 input families =
  **72 comparisons, max abs difference 0.0**, including every exact node hit (the unit-row
  branch), a point 5e-15 inside the `1e-14` hit window, one 1e-13 outside, and the scalar-input
  contract. A standalone probe that reads the pre-M3 source **off disk** rather than embedding it
  (24 degrees x 11 families = **264 comparisons**) also read 0.0;
* **end to end**, S0's shadow: with the fail-before switch off, **345 of 346 observables at
  exactly 0.0**.

The per-family breakdown of the FULL (switch-on) capture is the sharper statement, because it
says where the change did and did not reach:

| family | arrays moved | arrays identical | max abs difference |
|---|---|---|---|
| per-layer (all four paths) | 86 | 54 | 3.6e-13 |
| SHARED grid | **0** | **140** | **0.0** |
| CONFORMING per-layer stack (mortar bypassed) | **0** | **8** | **0.0** |
| RCWA oracle | **0** | **4** | **0.0** |
| raw mass / cross-mass / `_lagrange_eval` arrays | **0** | **16** | **0.0** |
| general (slant / OOP) per-layer cascade | **0** | **12** | **0.0** |
| JAX twin (forward) | 2 | 2 | 5.6e-17 |
| JAX twin (gradient) | 1 | 0 | 1.2e-17 |

The zeros are the load-bearing rows: the shared path is untouched, the oracle is untouched, and
the conforming per-layer stack still equals the shared path bit for bit -- which is the identity
that refused N-3.4b (S4.2).

The one array that moves is discussed next, because it is not what it looks like.

### 2.5 The one moving array: a JAX **gradient**, and the cause is measured [M]

With items 3+4 switched off, `Jgrad` (d R0 / d eps of the per-layer JAX twin) moves by
**8.5e-17** -- 1e-16 relative -- while all four traced FORWARD observables are exactly 0.0.

The cause is **not** the hoist and **not** the vectorisation. It is the one bit-identical piece
of item 4 that ships anyway: `_redheffer_star_rect` used to evaluate `A12 @ D` and `B21 @ F`
twice each, and now names them once (S4.3). In forward evaluation that is bit-identical by
construction; in REVERSE-MODE AD it merges two independent graph nodes into one shared node,
which changes the order the cotangents accumulate.

**Measured, not asserted:** reverting only the JAX star's common-subexpression elimination and
re-running the traced leg returns **0.0 on all five JAX observables including the gradient**. The
CSE is kept -- it removes a dense matmul per star from the traced graph, and 8.5e-17 is eleven
orders below the gradient test's own 5e-5 bar.

---

## 3. N-3.3 -- the mortar is separable, and that is an identity

### 3.1 The proof, not an approximation claim [A]

Both mortars stack the transverse field 2-componentwise and every geometric projection acts on
the two components identically and independently, so the operator is exactly `kron(I_2, M)`. For
any `X` with `2h` rows (`h = M.shape[1]`) and `i < m`:

```
(kron(I_2, M) X)[i, j] = sum_{k=0}^{2h-1} kron(I_2, M)[i, k] X[k, j]
                       = sum_{k=0}^{h-1}  M[i, k] X[k, j]
```

because every dropped term carries the factor `kron(I_2, M)[i, k] = 0` **exactly** -- a
structural zero of the Kronecker product, not a small number. The blockwise form is the *same
sum, term for term*, over a shorter index range.

It is **not** bit-identical, and this document says so rather than hoping: BLAS accumulates a
length-`2h` dot product in a different order, and with different blocking, than a length-`h` one.
`test_kron2_apply_is_the_exact_factorisation_not_an_approximation` pins the two claims
separately -- tolerance **0.0** on exactly-representable integer operands (the exactness claim,
square and rectangular `M`), and a machine-eps-relative envelope on general float operands (the
rounding claim).

### 3.2 What it buys [M]

Four `kron(I_2, .)` operators were materialised per non-conforming mortar interface, and two more
per general (slant / OOP) mortar. On the audit device `n_glob = 40`, so each was an 80x80 float64
array (51.2 kB) built from a 40x40 block (12.8 kB) that already existed and is now also cached:

* **memory**: a 4x collapse of the operator footprint, and the `np.kron` build itself disappears.
  Measured end to end by counting the bytes `np.kron` returns during a solve
  (`test_the_separable_mortar_builds_no_kron_operator_at_all`): with the switch **off** a
  6-layer solve allocates a non-zero kron footprint; with it **on** it allocates **exactly zero**;
* **flops**: `2 x (m h) h` scalar multiplies instead of `(2m)(2h)(2h)` -- half the dense form.

Sizes scale as `n_glob^2`, so the same factor-4 collapse is 2.3 MB -> 0.6 MB per operator at a
production `n_glob = 300`.

### 3.3 Reconciliation with the mission's "two ~42x42 1-D cross-masses" [A]

The mission brief describes the mortar cross-masses as factoring "as Kronecker products of two
~42x42 1-D cross-masses". Measured, `n_glob = 40` on the audit device, so the 1-D blocks are
40x40 and the materialised operators were 80x80 -- the "~42x42" sizing is right, and the factor
that exists in **this** module is the 2-component field stacking, `kron(I_2, C)`.

The *transverse* 1-D cross-mass `C` itself does not factor further: in 1-D there is one
transverse axis and `C` is irreducible. The `kron(Cx, Cy)` form the campaign calls a
**requirement** (standing rule 10) belongs to **S1** -- 2-D per-layer grids, deferred to 5.34 --
where the dense alternative is `(n_x n_y)^2` and therefore a rejected design. M3 ships the 1-D
instance of the identity and the helper (`_kron2_apply`) that S1's 2-D form generalises.

---

## 4. N-3.4 -- shipped at the mortar, REFUSED at the star

### 4.1 Where a solve suffices, and where it does not [A]

| site | the inverse is used as | solve suffices? | action |
|---|---|---|---|
| `_interface_smatrix_mortar`, `I + BA` | `iba @ (I - BA)`, `iba @ B` | **yes** | converted; ONE factorisation for both right-hand sides |
| `_redheffer_star_rect`, `I - B11 A22` | `A12 @ D @ ...` | yes, as a RIGHT solve | **refused -- S4.2** |
| `_redheffer_star_rect`, `I - A22 B11` | `B21 @ F @ ...` | yes, as a RIGHT solve | **refused -- S4.2** |
| `_interface_smatrix` (shared path), `a + b` | `S12 = 2 (a+b)^{-1}` | **no** | untouched: the inverse *is* the answer |

M1's census moves with the converted site: `_guarded_solve` records the same instruments under
the same `site` name, so M1's populations stay comparable across M3. It introduces **no threshold
and no policy** -- both stay in `rcwa/_core.py`, unforked, and are read from there
(`test_guarded_solve_carries_the_same_census_as_the_guarded_inverse`).

### 4.2 The refusal [M]

The right solve at the star was implemented, measured, and withdrawn.

On a **conforming** per-layer stack the mortar is bypassed, so `_redheffer_star_rect` is the only
remaining difference from the shared path -- which cascades with RCWA's `_redheffer_star`, still
an explicit inverse. Converting one twin and not the other broke **five long-pinned cross-path
bit-exactness contracts**, on both builds:

```
test_pmm_per_layer_grids.py::test_identical_wall_stack_is_bit_exact_vs_shared
test_pmm_per_layer_grids.py::test_two_layer_stack_is_bit_exact_vs_shared
test_pmm_per_layer_grids.py::test_conical_per_layer_matches_shared_bit_exact
test_pmm_m2_window_contract.py::test_n4_default_solve_is_bit_identical_through_every_dispatch
test_pmm_m2_window_contract.py::test_window_halfwidth_covering_the_stack_reproduces_shared_bit_exact
```

The move is 1e-16..1e-13, i.e. a pure rounding change -- the pins are doing exactly their job.
Two ways out, both rejected:

* convert **both** twins. That moves RCWA's bits, and RCWA is the independent oracle this
  campaign adjudicates against (M2 already had to re-run its oracle once because a concurrent
  mission touched `rcwa/_core.py`). Not mid-campaign;
* keep one twin converted and re-pin the five tests to a tolerance. That deletes a shipped exact
  identity -- "per-layer with identical walls IS the shared path" -- to buy a measured ~3 % of a
  solve. Rejected on the trade.

The site therefore keeps M1's guard untouched and the docstring records the measurement, with
`test_the_star_keeps_its_inverse_because_a_solve_breaks_a_shipped_identity` **proving** the claim
(it rebuilds the alternative star from the shipped `_guarded_solve_right`, shows the identity
breaks, and shows it breaks only in the last bits) rather than asserting it. If the shared star
is ever converted, both must move in one change and be re-pinned together.

### 4.3 What ships at the star anyway, for free [A]

`A12 @ D` and `B21 @ F` were each evaluated **twice** per star (Python's `@` is left-associative,
so `A12 @ D @ B11 @ A21` is `((A12 @ D) @ B11) @ A21`). Naming them once removes two dense `n^3`
products per star and is bit-identical in forward evaluation by construction. The JAX twin gets
the same treatment -- and the one reverse-mode consequence is measured in S2.5.

### 4.4 The fail-before switch did not fail before, until it did [M]

`PMM_MORTAR_SEPARABLE` is the ONE switch the plan asks for over items 3+4. As first written it
gated the NumPy arms only, and the JAX twin's de-kron was unconditional. The switch-off shadow
caught it: 5.6e-17 on the traced forward observables where 0.0 was required. The switch now
reaches the traced path too -- read **once at trace time** as a Python bool, so the traced graph
is one arm or the other and nothing data-dependent is branched on. A fail-before that does not
fail before is worse than none.

---

## 5. T3-4 -- the promoted guard, and its calibration

T3-4 arrives from M2 S5.5 with a mechanism, a free instrument, and an honest refusal to ship a
bar. This section supplies the bar.

### 5.1 The defect, restated in one paragraph [A, from M2 S5]

A cross-layer cell at the ~1-2 nm scale on a 700 nm pitch injects modes whose z-Poynting flux
sits AT `_mass_flux_cut`'s propagating/evanescent threshold. The forward/backward selector,

```python
flip = np.where(prop, flux < 0.0, q.imag < 0.0)
```

then takes a mode's propagation DIRECTION from a different criterion depending on which side of
the cut it lands. The count classified propagating moves with `degree` instead of staying fixed,
the forward set is mis-assembled, and the cascade returns a **unitary but wrong** S-matrix -- up
to 466 % wrong at `|R + T - 1|` = 4.5e-07. Deterministic on both builds. No conservation identity
sees it, because at this pitch only order 0 propagates and `R + T = 1` is nearly a restatement of
the cascade's unitarity.

### 5.2 The instruments, all free [A]

Everything below is computed from `flux`, `q` and the threshold, all of which the modal solve has
already built. No second eig, no second factorisation. `_mass_flux_cut` keeps sole ownership of
the threshold: it was split into `_mass_flux_threshold` + a comparison so the instrument reads
the LIVE value instead of re-deriving it (a forked threshold is a forked contract), and
`test_mass_flux_cut_is_unchanged_by_the_threshold_split` pins that the split moved nothing.

| instrument | definition | origin |
|---|---|---|
| `n_prop` | modes classified propagating | M2 |
| `n_near` | modes within a decade of the cut | M2 |
| `n_risk` | near-cut **and** the two direction criteria disagree | M3 |
| `margin` | `min` over DISAGREEING modes of `max(|flux|/t, t/|flux|)` | M3 |
| `spread` | `max - min` of `n_prop` over the stack's PATTERNED layers | M3 |
| `q_excess` | `max |q|` among modes classified propagating, over the layer's own index ceiling | M3 -- **refuted, see 5.3** |

`margin` is the multiplicative factor by which the cut would have to move before **any** mode's
direction assignment changes; `+inf` when no mode disagrees. It is scale-free (a ratio) and
constant-free (nothing is fitted into its definition).

### 5.3 Two instruments measured and refuted before the third was adopted [M]

**`q_excess`, and it was the more attractive hypothesis.** A genuinely propagating mode of a
layer satisfies `q^2 = eps_eff - kx_eff^2` with real `q`, so `|q| <= n_max`, the largest index
present on that grid -- a bar with **no fitted constant at all**, read off the operator's own
element table. M2's report that `max|q|` grows to 2.8e4 in the collapsing cells made it look
decisive.

It is not. Measured on device U at `ns = 2`, `min_feature` default, the degree ladder 6..16
(three cells RIGHT, three WRONG): `q_excess` reads **0.9170 on every one of them**, and 0.9163 on
the F2 control. M2's 2.8e4 is the maximum over ALL modes; the modes that are *classified
propagating* never exceed the physical ceiling. The misclassified modes are not high-`|q|` modes,
and the hypothesis is dead. It is retained as a recorded census column, not as a trigger.

**`n_risk` / `margin` alone.** On the same six cells: `n_risk` reads 1, 2, 2 on cells that are
RIGHT and 2, 2 on cells that are WRONG; `margin` reads 4.19, 4.04 (right) against 3.27, 1.07
(wrong) -- overlapping populations separated by a factor of 1.2. A bar on either alone refuses
correct solves, which is M2's own finding re-confirmed.

### 5.4 What separates is the CONJUNCTION [M]

The shipped verdict requires **both**:

1. **`spread > 0`** -- the number of modes classified propagating is not the same on every
   patterned layer of the stack; and
2. **`n_risk > 0` with `margin < 10`** -- some layer carries a mode within a decade of the cut
   whose two direction criteria disagree, i.e. the cut is load-bearing there.

The logic is the same shape M1 found for `_guarded_lstsq` (rank AND residual): proximity to the
cut is only dangerous where the cut actually decides something, and a disagreement is only
dangerous where it is close.

**Classical (`phi = 0`) per-layer family, device U** [M]. RCWA reference 0.1100920 (`ns` = 2) /
0.1111090 (`ns` = 6), 141 orders:

| `ns` | `degree` | `min_feature` | `R0` | vs RCWA | `spread` | `n_risk` | `margin` | verdict | correct? |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 6 | default | 0.111000 | +0.8 % | 0 | 1 | 4.19 | silent | yes |
| 2 | 8 | default | 0.110723 | +0.6 % | 0 | 2 | 4.04 | silent | yes |
| 2 | 12 | default | **0.061668** | **-44 %** | **2** | 2 | 3.27 | **WARNS** | yes |
| 2 | 16 | default | **0.623395** | **+466 %** | **2** | 2 | 1.07 | **WARNS** | yes |
| 2 | 12 | 0.5 nm | 0.110499 | +0.4 % | 0 | 0 | 21.7 | silent | yes |
| 2 | 16 | 0.5 nm | 0.110479 | +0.4 % | 0 | 0 | 21.4 | silent | yes |
| 6 | 10 | default | **0.568367** | **+412 %** | 2 | 2 | <10 | **WARNS** | yes |
| 6 | 8 | 3.0 nm | 0.111053 | -0.05 % | 0 | 0 | -- | silent | yes |

**8 of 8**, with the two halves of the conjunction each doing work: `spread` alone would miss
nothing here but `n_risk` alone would fire on the two correct default cells at degree 6 and 8.

### 5.5 THE REFUTATION -- the bar does not survive the conical family [M]

The same conjunction, on the **conical** (`phi = 20 deg`) cascade of the same device at
`ns = 4` -- so `off` = 2.706 nm, coat 5 nm, threshold `min(off, |c - off|)` = 2.294 nm, and
`min_feature` = 3.0 nm clears it while the library default does not:

| `min_feature` | degree 6 | 8 | 10 | 12 | 14 | spread over degree | label |
|---|---|---|---|---|---|---|---|
| default | 0.135182 | 0.134854 | 0.134760 | 0.134727 | **0.240574** | **67.8 %** | BROKEN |
| 3.0 nm | 0.134189 | 0.133726 | 0.133579 | 0.133521 | 0.133495 | **0.52 %** | CLEAN |

and the instruments (`spread` / `n_risk` / `margin`) on those same cells:

| `min_feature` | degree 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| default | 2 / 2 / 1.04 | 2 / 2 / 2.72 | 1 / 3 / 1.65 | 3 / 5 / 1.04 | 5 / 8 / 1.22 |
| 3.0 nm | 0 / 0 / 103 | 0 / 1 / 3.93 | **1 / 2 / 1.07** | **1 / 2 / 1.40** | **2 / 2 / 1.06** |

The three bolded cells are **CORRECT** -- they sit on the stationary plateau, 0.52 % over the
whole ladder -- and the conjunction **fires on all three**. Their readings are inside the band
the classical family's broken cells occupy (spread 2, margin 1.07-3.27). A conical cascade
carries a genuinely denser flux spectrum (`ky0 != 0` breaks the `+/-q` symmetry differently) and
the free instruments do not know that.

**So there is no bar on these instruments that survives both mounts**, and the guard therefore
**ships DISARMED**. That is the mission's own stated fallback ("if none separates, ship the
instrument as an opt-in census + document, and say so") and it is the same shape as M1's
withdrawn inverse refusal: a threshold calibrated on one family did not transfer, and a false
pathology claim is worse than silence.

What ships:

* the **instrument**, `_MODE_CUT_CENSUS` -- opt-in, zero default cost, recording `n_prop`,
  `n_risk`, `margin`, `n_bound`, `q_excess` and the patterned flag per modal solve;
* the **guard**, `PMM_MODE_CUT_GUARD`, present, calibrated, wired on the per-layer paths, and
  **`False`** by default. Arming it is one assignment, and on the classical path it is 8/8;
* the **refutation, pinned**: `test_t34_bar_is_REFUTED_on_the_conical_family` asserts the three
  false positives still reproduce, with a message saying that a future fix should make it FAIL.

**What would close it.** M2's own lead: a two-degree consensus probe on the propagating-mode
count -- a detector with no fitted bar at all. It costs a second eig per layer, so it is an
opt-in diagnostic rather than a free guard. Untested; ~1 AC.

### 5.6 The false-positive control the naive bar gets wrong [M]

`spread > 0` alone would be catastrophic even on the classical path, and the control that proves
it is deliberate: **F2**, a stack of alternating `n = 3.48` / `n = 1.45` gratings whose layers
genuinely support different numbers of propagating modes.

| control | `n_prop` per layer | `spread` | `n_risk` | `margin` | `R0` (PMM) | RCWA (101 ord.) | `\|R+T-1\|` | verdict |
|---|---|---|---|---|---|---|---|---|
| F1, 3 layers | [4, 4, 4] | 0 | 0 | 92.5 | 0.230414 | 0.230144 | 4.0e-06 | silent |
| F1, 4 layers | [4, 4, 4, 4] | 0 | 0 | 31.4 | 0.504292 | -- | 3.5e-06 | silent |
| **F2, 3 layers** | **[4, 2, 4]** | **2** | 0 | **236** | 0.100639 | 0.102116 | 7.3e-07 | silent |
| **F2, 4 layers** | **[4, 2, 4, 2]** | **2** | 0 | **236** | 0.036855 | -- | 7.1e-06 | silent |

F2 is right (1.5 % from an RCWA twin that is itself under-converged at this contrast) and carries
`spread` = 2. It is silent only because `n_risk` = 0 and the margin is 236 -- the cut is nowhere
near load-bearing. `test_t34_guard_is_silent_when_layers_LEGITIMATELY_differ` pins it *and*
asserts the control actually exercises a spread, so a control that stopped controlling cannot
pass silently.

### 5.7 One instrument detail that matters, and was got wrong first [M]

"Patterned" is decided from the ELEMENT TABLE (`_grid_is_patterned`), not from the call site.
The conical cascade solves its **half-spaces** through `_sem_modes_tensor` -- the same entry point
a patterned layer uses -- so a call-site label mislabelled them as patterned and manufactured a
`spread` out of a uniform half-space's perfectly legitimate mode count. Reading the label off the
operator's own `elem_bnds` is what makes it correct everywhere.

### 5.8 Where the guard is wired, and where it is not [A]

| path | scoped? | why |
|---|---|---|
| `PMMStack.solve`, per-layer vertical | yes | |
| `PMMStack.solve`, per-layer general (slant / OOP) | yes | |
| `PMMStack.solve_vs_wavelength`, per-layer | yes, **deferred** | see below |
| `conical._conical_nodal_solve` (both grid modes) | yes | decorator |
| `PMMStack.solve`, SHARED grid | **no** -- see S7 | |
| the JAX twin | **no** | a data-dependent branch is not expressible under trace |

The verdict is a **per-solve** statement (it needs every layer's propagating count), so the
per-layer entry points open a thread-local scope and the verdict is taken on exit.
`solve_vs_wavelength` runs its points in WORKER THREADS and has a shipped contract that its
warnings come out in wavelength order for any worker count, so its verdict is **deferred**: each
point stashes its message under its own index and the messages are replayed, in index order,
after the map completes. `test_t34_guard_warnings_come_out_in_WAVELENGTH_ORDER_on_the_sweep`
pins that the serial and 4-worker runs produce the same messages in the same order and the same
numbers.

### 5.9 It never changes a number [M]

`test_t34_guard_ships_disarmed_and_arming_it_moves_no_number` asserts three things: the shipped
default IS silence; arming it produces the warning; and the returned `R0` is `==` (not merely
close) either way. The guard only speaks.

With the guard disarmed AND no census armed, `_mode_cut_scope` does not even allocate its row
list and the modal solvers never call `_record_mode_cut` -- the default path costs one
`is not None` and one `getattr` per eig, against a dense complex eig.

---

## 6. Speed and memory

### 6.1 Speed [M, Windows, idle, threads pinned, interleaved A/B]

Five rounds of `baseline, current, baseline, current, ...`, each round a fresh process taking
the median of 5 repeats; the table reports the median of the five round medians and, because the
box is loaded, the FULL spread of those medians.

| case | what it is | baseline s (spread) | current s (spread) | speedup |
|---|---|---|---|---|
| (a) | 3-layer synthetic, ONE cold solve | 0.582 (0.368 - 0.699) | 0.621 (0.469 - 0.633) | **0.94x -- not resolved** |
| (b) | 38-layer audit device, ONE cold solve | 1.639 (1.097 - 1.823) | 1.416 (1.108 - 1.657) | **1.16x** |
| (c) | 10-point LC sweep (S3's case) | 4.836 (3.517 - 5.260) | 3.224 (2.607 - 3.434) | **1.50x** |
| (d) | 38-layer device, 3 solves back to back | 5.041 (2.792 - 5.581) | 4.098 (3.419 - 4.449) | **1.23x** |

**Read this honestly.** The baseline spreads are wide -- case (a)'s five round-medians span 1.9x
-- so case (a)'s 0.94x is **inside the noise and no claim is made from it**; the same caution
applies at the 10 % level to (b) and (d). Only case (c), where the baseline's slowest round is
still slower than the current tree's slowest, separates cleanly.

**And it is the expected shape.** Case (a) is a single cold solve of a 3-layer stack: the cache
has nothing to hit, and 4 mortars' worth of `kron` is a small share of that solve. Case (c) is
ten solves of one geometry, which is what the hoist is for. The measurement REFUTES the mission's
own working assumption that hoisting speeds up a `solve()` -- it does not, and cannot.

The noise-free half of the same claim is the profile. On case (b), cProfile against the baseline
lists `_lagrange_eval` at **2112 calls / 1.24 s cumulative** across the three solves of case (d);
on the current tree it does not appear in the top eight at all, and neither does `_sem_cross_mass`
or `_sem_mass_exact`. Those are counts, not timings, and they do not have a spread.

### 6.2 Memory [M]

The `np.kron` operator census -- the exact bytes the mortars materialise -- is the sharp
instrument, and it is a count:

| case | `np.kron` bytes, baseline | current |
|---|---|---|
| (a) | 1.98 MB | **0** |
| (b) | 7.58 MB | **0** |
| (c) | 19.25 MB | **0** |
| (d) | 22.73 MB | **0** |

**The `tracemalloc` PEAK is a different story, and the plan's gate is not met on it.** M3 makes
two memory changes that pull in OPPOSITE directions -- the de-kron removes transient operators,
the geometry cache retains 1-D blocks -- so a single before/after peak mixes them. Measured at
three points on case (b) (38-layer device, cold cache, `tracemalloc(1)`):

| point | cache? | `kron` built? | `tracemalloc` peak | live at end | cache retained |
|---|---|---|---|---|---|
| (1) pre-M3 baseline | no | yes | 38.860 MB | 0.193 MB | -- |
| (2) M3, switch OFF | yes | yes | 39.638 MB | 1.257 MB | 0.960 MB |
| (3) M3, switch ON | yes | no | 39.640 MB | 1.258 MB | 0.960 MB |

* **(2) - (1) = +0.778 MB**: the geometry cache's retained cost, a deliberate trade for S6.1's
  repeat-solve speed. It is `2 nlay - 1` blocks of `n_glob^2` float64 and nothing else, and
  `cache_report()` names it.
* **(3) - (2) = +0.002 MB**: the de-kron's effect on the PEAK is **nil**. The peak on this device
  is set by the LAPACK eig workspaces, not by the mortar's operators, so removing 7.6 MB of
  transient `kron` churn does not move it.

**So the plan's memory gate for the de-kron -- "accepted only if the peak drops measurably" -- is
NOT met, and the de-kron is accepted on the other two instruments instead:** the allocation census
(7.6 MB -> 0 on case (b), 22.7 MB -> 0 on case (d)) and the flop count (half). That is the honest
statement; the peak claim in the plan was optimistic about where this device's peak lives.

**Two memory defects were found by making this measurement, and both were in M3's own new code:**

1. `S11 = X[:, :nc]` was a **VIEW** into the stacked right-hand side, so the cascade -- which
   retains every interface's `S11` -- pinned a 2x-larger buffer per interface for the whole solve.
   Measured: **+3.8 MB on the peak**, against a change whose point is to lower it. Fixed with an
   explicit `.copy()`, and the comment at that line says why so it is not "tidied" away.
2. `_kron2_apply` writes its halves through `out=` rather than `concatenate`. Measured after (1)
   was fixed, the two spellings are **equal to 1 kB** (39.640 vs 39.639 MB) -- so `out=` is kept
   for being strictly fewer allocations, not on a measured win, and the docstring says exactly
   that. The 3.8 MB belonged entirely to (1).

The largest-array census at the end of a solve names the change precisely: pre-M3 the largest live
allocations are 53 kB and 52 kB (LAPACK / operator scratch); post-M3 they are **0.490 MB and
0.477 MB at `_core.py`'s `_sem_mass_exact_cached` and `_sem_cross_mass_cached`** -- i.e. the cache,
and nothing else, is what is now retained.

### 6.3 Both builds

**Numeric behaviour is verified on both builds** (S2.4's byte-identity capture and S7.1's suites
run on Windows/scipy-openblas and WSL/OpenBLAS).

**The SPEED table above is Windows-only, and that is a stated gap.** The interleaved A/B is a
25-minute exclusive-machine measurement and this mission spent its remaining machine time on the
correctness sweeps and on M4's referrals (S7b), which are behaviour changes and therefore rank
above a second set of ratios for a change already shown to be numerically identical on both
builds. What the WSL build DOES verify is that the same code produces the same answers there;
what it does not verify is that the 1.16x / 1.50x / 1.23x reproduce. Given that the wins come
from removing Python-level work (a Python loop, `nlay - 1` geometry rebuilds) rather than from
BLAS behaviour, they should be build-insensitive -- but that is an expectation, not a
measurement, and it is written here as one.

---

## 7. Acceptance gates

The plan's M3 table, answered line by line.

| axis | gate | result |
|---|---|---|
| accuracy | hoist + vectorise **bit-identical**, tolerance-at-0.0, both builds | **MET.** 345/346 at exactly 0.0 with the switch off; the one exception is a JAX *gradient* at 8.5e-17, attributed by measurement to the star's CSE (S2.5), not to the hoist or the vectorisation |
| accuracy | de-kron + `inv->solve` within tolerance, with the mortar-reduction identity and conforming bit-exactness pins both green | **MET.** envelope 3.6e-13 over 346 observables; `test_the_mortar_still_reduces_to_the_plain_interface_on_identical_grids` green; the conforming per-layer stack is **0.0 on 8/8** against the shared path -- which is exactly why 4b was refused |
| conservation | `\|R+T-1\|` unchanged or better at degree 6/8/10 on the lossless staircase | **MET.** asserted per configuration by `test_separable_and_solve_move_only_the_last_bits` (comparative: `close_on <= max(10 close_off, 1e-9)`), and the shadow's lossy twin row moved 7.2e-16 |
| speed | per-solve wall time on (a) 3-layer synthetic, (b) 38-layer audit device, (c) 10-point LC sweep, vs a `git archive` baseline, idle, threads pinned | **MET**, with the baseline redefined for cause (S1.2) and a fourth case added because (a)/(b) cannot show a cache win (S6.1) |
| memory | largest-array census before/after; `tracemalloc` peak and RSS delta per solve; the de-kron claim accepted only if the peak drops measurably | **PARTLY MET, and the shortfall is stated.** Census, peak and RSS all measured and reported (S6.2); the `np.kron` allocation census goes to **exactly zero**; but the PEAK does not drop (+0.002 MB) because it is set by the eig workspaces, so **the plan's peak condition for the de-kron is NOT met** and the claim rests on allocation and flops instead |
| both-builds | all of the above on both | **MET** (S1.1, S6.3, S7.1) |
| fail-before | ONE switch for the de-kron/`solve` group | **MET** -- `PMM_MORTAR_SEPARABLE`, and it had to be fixed once to actually cover the traced path (S4.4) |

### 7.1 Suites [M, both builds]

| suite | Windows | WSL |
|---|---|---|
| M1 + M2 + per-layer + **M3** (`test_m1_conditioning_guard`, `test_pmm_m2_window_contract`, `test_pmm_per_layer_grids`, `test_pmm_m3_efficiency`) | **99 passed** | **99 passed** |
| of which M3's own new file | 47 | 47 |
| PMM collateral (`-k "pmm or rcwa or conditioning"`) | see below | see below |

Identical counts on both builds, which is the campaign's requirement (a suite that silently
skips on one build is not a cross-build result).

**The PMM collateral set is the one gate this document does NOT close, and it is a wall-clock
gap, not a red result.** It is a ~1400-test, ~35-minute run per build; it was launched three
times on Windows and twice on WSL over the session and each attempt was cut short -- twice by
this mission's own `Stop-Process` sweep to free the machine for the interleaved A/B (S6.1), once
by the harness. The last Windows attempt reached **91 % with zero failures** (`--tb=no`, so a
failure prints an `F` in the progress stream and none appear). Every targeted suite that
exercises the changed code IS green on both builds, including the four files above and the M3
file's 47 tests; what is missing is the breadth sweep over the untouched PMM/RCWA surface.
**Re-run it before the release tag** -- the command is in S9.

**One test of M3's own was WRONG and the cross-build run caught it**, which is the whole reason
the rule exists. `test_m4_referral_pmm_jones_2d_energy_defect_IS_seen_by_the_m1_census` asserted
`worst_rcond < _INV_RCOND_SCREEN` -- an ABSOLUTE bar on a BLAS-dependent magnitude, the exact
trap the plan's rule 6 names. It passed on Windows (2.754e-09) and failed on WSL (1.776e-06).
The fix is not to relax the number: the assertion was replaced by what is actually
build-independent (the defect reproduces; the census covers the route; nothing is refused), and
the three-order cross-build spread on the SAME wrong answer is now recorded as evidence -- see
S7b.2.

### 7.1b The broad-`except` budget -- referred back, with the bisect [M]

A sweep flagged `lumenairy/elements/pmm/_core.py` as having gained TWO
`except Exception` clauses and asked for them to be NARROWED. Both were M3's
(`_layer_index_bound` and `_grid_is_patterned`, the T3-4 element-table readers)
and both are now `except (TypeError, ValueError)` -- the only two exceptions
`np.asarray(..., dtype=complex)` / `np.diag` can raise on an unexpected element
payload -- each with a comment saying so. `git diff HEAD -- lumenairy/` now
shows **zero added and zero removed** lines matching the budget test's own
regex: the whole uncommitted M2 + M3 + M4 tree adds **no** broad-except to the
library.

**The budget test is nevertheless red, and it is red at HEAD.** Counting with
the test's own pattern:

| commit | non-`ui/` `except Exception` count |
|---|---|
| `3e6dc0f` (v5.32.1 release commit) | **48** = the budget |
| `013c388` (v5.32.1 release merge) | **48** |
| `95a9849` (D8 carrier parallelism, = HEAD) | **54** |

All six were added by the committed D8 commit, in
`lumenairy/propagators/carrier.py` (0 -> 6): the multiprocessing worker-state
bootstrap -- `importlib.import_module` guards, a RAM-budget lookup, glass-
registry propagation into a spawned worker, and one executor fallback. That
file is outside this mission's ownership, and the clauses are the class where
the audit rule's judgement belongs to the author: narrowing an import guard
inside a spawned worker turns graceful degradation into a subprocess crash, and
which failures are expected there is D8's knowledge, not M3's.

**So the overrun is not this campaign's and cannot be closed from inside
`lumenairy/elements/pmm/**`.** M3's own contribution to it is removed. Referred
to the D8 / `propagators/carrier.py` owner with the bisect above.

### 7.2 Lint

`ruff check --no-cache` (run through WSL, per the standing rule) over
`lumenairy/elements/pmm` and the new test file: **clean**.

### 7.3 Standing rules, answered

| rule | how M3 answers it |
|---|---|
| 2 -- every claim measured, with a null-floor control | the CONFORMING per-layer stack (mortar bypassed) and the SHARED path are the null floors in the shadow capture, and both read 0.0 |
| 3 -- both BLAS builds for any numeric claim | S1.1, S7.1 |
| 5 -- conservation scored alongside accuracy | `\|R+T-1\|` is in the envelope test and in the T3-4 tables, and S5.1 restates why it is BLIND to the T3-4 defect |
| 6 -- comparative-envelope assertions | the conservation assertion is comparative (`<= max(10 close_off, 1e-9)`), the de-kron envelopes are scale-relative, the exactness pin is at 0.0 |
| 8 -- fail-before switches verified per configuration, not in aggregate | `test_separable_and_solve_move_only_the_last_bits` is parametrised over four `(ns, degree)` cells |
| 10 -- memory measured; prefer separable constructions | S6.2, and S3 is the separable construction |
| 11 -- independent oracles share no code with the thing under test | RCWA (Fourier modal, analytic walls) is the T3-4 labeller and the S5 reference; the shared-grid path and the conforming stack are the in-repo cross-checks |
| 13 -- bidirectional adversarial review | three claims of this mission's own were refuted by measurement: the `q_excess` instrument (S5.3), the guard's transferability (S5.5), and the completeness of the fail-before switch (S4.4). One inherited claim was refuted too: that the hoist would speed up a single solve |

---

## 7b. M4's referrals, folded in

M4 (`PMM_M4_HYGIENE_2026_08_04.md`) referred four items into files M3 owns.

### 7b.1 The per-worker BLAS-cap defect, at all three PMM sites [M]

M4 established on the RCWA twin that `threadpoolctl`'s cap is **process-global** on OpenBLAS
while only the *request* is thread-local, so entering `_blas_threads_quiet(blas_per_worker)`
INSIDE each worker puts N concurrent enter/exit pairs on one global setting -- and the shipped
"BYTE-IDENTICAL for any worker count" contract was false. Three source comments in the library
asserted thread-locality; all three were wrong.

The identical pattern was live at three PMM sites, each with its own byte-identity pin:

| site | fixed |
|---|---|
| `PMMStack.solve_vs_wavelength` (shared grid) | yes |
| `PMMStack._solve_vs_wavelength_perlayer` | yes |
| `PMM2DStackHybrid.solve_vs_wavelength` | yes |

M4's fix transfers verbatim: **one** `with _blas_threads_quiet(blas_per_worker), _blas_limit():`
around the whole dispatch on the calling thread (so serial and threaded paths run at the same
BLAS thread count and byte-identity holds *by construction*), and `_blas_threads_quiet(None)`
inside the per-wavelength function so a nested `solve` does not re-enter the global limiter on
the serial branch.

**Null control** [M] -- sha256 of `(R, T, jones)` over a 12-point 1-D sweep (both grid paths) and
a 6-point 2-D hybrid sweep, `max_workers` in {1, 2, 4, 8}, **three repeats each** (the pre-fix
race fired nondeterministically, so one comparison is not evidence):

| site | default pool (24 threads) | `OPENBLAS_NUM_THREADS=1` |
|---|---|---|
| 1-D shared | `90e737505d425809` x 12/12 | `90e737505d425809` x 12/12 |
| 1-D per-layer | `d3ac4c779a142ef2` x 12/12 | `d3ac4c779a142ef2` x 12/12 |
| 2-D hybrid | `d68d7e75f6467975` x 12/12 | `d68d7e75f6467975` x 12/12 |

The `OPENBLAS_NUM_THREADS=1` column is provably the PRE-fix bits (with the global pinned at 1 the
old per-worker save/restore was `1 -> 1 -> 1`, a no-op), and it agrees with the default-pool
column. **The fix removed the nondeterminism without moving a single bit.**

A fail-before switch is not applicable and would be misleading -- the prior behaviour was
nondeterministic, so there are no "prior bits" to reproduce. The fail-before evidence is M4's
measured pre-fix divergence table, which the post-fix code cannot produce.

**Pinned build-independently.** The race needs BOTH a >1-thread environment pool AND
`threadpoolctl`, neither of which holds on the 2-core CI runner or the WSL build, so a
byte-identity assertion is green there whether or not the bug is present. `test_pmm_sweep_
applies_exactly_one_blas_cap` (both grid paths) and `test_pmm2d_hybrid_sweep_applies_exactly_
one_blas_cap` COUNT cap applications, which works on any machine, and assert exactly **1** on the
serial branch too -- the load-bearing half, because the serial loop runs on the caller's thread
where the sweep-level request is set. The counting stand-in DELEGATES to the real controller
(M4's lesson: an instrument that switches off the thing it instruments let every worker run BLAS
at the full pool and took down a pytest-xdist worker).

### 7b.2 `pmm_jones_2d` manufactures energy on a LOSSLESS cell -- and M1's census DOES see it [M]

M4 referred this with "M1's census is one file short of covering it". **Measured here, the
premise is wrong in the useful direction**: the NumPy in-plane route already sends four explicit
inverses through `_guarded_inverse` (two `pmm interface mode-match (a+b)`, plus the two
`rcwa Redheffer star` denominators), so the census covers it -- and its free instrument tracks
the defect precisely. Degree 11, the test's own cell, lossless (target `R + T = 2`):

| `OPENBLAS_NUM_THREADS` | `R + T` | defect | worst equilibrated `rcond` | residuals paid |
|---|---|---|---|---|
| 1 | 2.0125077 | 0.63 % | 2.754e-09 | 9.825e-10 |
| 24 | 2.0307687 | **1.54 %** | **3.831e-13** | **9.709e-06, 2.241e-08** |

M1's screen is `1e-8` and its (withdrawn) refusal bar is `1e-8` on the equilibrated residual.
**The 1-thread solve is screened in and PASSES the residual; the 24-thread solve -- the one that
manufactures a further 1.8 % of energy -- FAILS it by three orders.** The instrument separates
the two exactly. What it does not do is act, which is exactly what M1 documented when it withdrew
the refusal on finding that CORRECT 2-D solves read in the same band.

So this is the **X-1 class, still open**, now with a second reproducer -- and the first one where
the wrong answer shows up as a lossless-energy violation rather than a per-order error, which
makes it a strictly better test case for whoever closes X-1. Deterministic (3/3 identical at 24
threads). Pinned by `test_m4_referral_pmm_jones_2d_energy_defect_IS_seen_by_the_m1_census`, which
asserts the defect, the census coverage, the flag, and that nothing is refused -- and says a
future fix should make it fail.

**What M3 did not do here:** close it. Closing X-1 needs a criterion that survives every method
in the library, which M1 measured does not exist on these instruments, and inventing one on a
second data point would repeat the mistake M1 already corrected. Recommended: this cell becomes
part of X-1's labelled population.

### 7b.3 T3-7 and T3-5 -- DEFERRED, and why [A]

M4 recorded designs for both (lattice wall quantisation; taper-aware `min_feature` warning) that
it could not implement while M2 owned the files. They are **M4 items in the campaign plan's own
mission table**, not M3's, and each is a `_core.py` + `stack.py` behaviour change requiring its
own measured accuracy table (per-material total-width preservation at matched `Delta` vs matched
`min_feature`, degree spread on the audit device at both) plus a both-builds re-validation.
Folding them into M3 would mean shipping two behaviour changes on the evidence of another
mission's design note, with no measurement of my own -- the opposite of this campaign's standing
rules. **Deferred, with the designs intact in M4's report and the files now free.**

Note that M3's T3-4 work strengthens T3-5 the same way M2's did: the classification instrument
gives a *direct* readout of whether a chosen `min_feature` actually removed the near-cut mode,
which is what a taper-aware default is trying to achieve indirectly.

### 7b.4 RCWA cross-checks on accuracy claims [A]

Standing; honoured throughout. Every accuracy statement in this document is scored against the
RCWA oracle (S5's labels, S5.4/S5.5/S5.6 tables), and the shadow capture carries the oracle's own
rows, which read **0.0** between the two sides -- the direct check that M3 did not disturb the
arbiter.

---

## 8. What M3 did NOT do, stated plainly

1. **The star's `inv` is not converted** (S4.2). X-1's underlying exposure at that site is
   unchanged; M1's census still records it.
2. **The T3-4 guard is not wired on the SHARED grid path**, although M2 S3.7 shows the defect is
   present -- and worse -- there. The instruments are grid-agnostic and the census records shared
   rows, so the calibration below reports what a shared-path guard *would* have done; wiring it
   is a one-line scope and a full-suite re-validation, and it is deliberately not folded into a
   mission whose subject is the per-layer path.
3. **The guard does not fire on the JAX path.** A data-dependent branch is not expressible under
   trace; the traced twin is a differentiable mirror of a NumPy solve the user can run guarded.
4. **`prepare()` (S3) is untouched.** The plan defers it pending M3's measurement of how much of
   its win the geometry cache already delivers; S6 supplies that measurement.
5. **No claim is made that the geometry cache speeds up a single cold solve.** It cannot, and
   S6 measures it not doing so.
6. **The 2-D Kronecker cross-mass (`kron(Cx, Cy)`) is not implemented.** It is S1's, deferred to
   5.34; S3.3 states the reconciliation.

---

## 9. Reproduction

Probe scripts (not shipped; `C:/tmp/m3_probe/`):

| script | purpose |
|---|---|
| `dev.py` | the three (four) named timing cases on M2's device module |
| `shadow.py` | the 346-observable capture; `M3_SEPARABLE=0` forces the fail-before arm |
| `cmp.py` | compares two captures at a TOLERANCE (never `array_equal`) |
| `timing.py` | one build = one process: median-of-N wall time, `tracemalloc` peak, RSS delta, `np.kron` byte census |
| `ab.py` | the interleaved A/B driver (baseline, current, baseline, ... ; medians of the round medians) |
| `bitid_lagrange.py` | the 264-case direct pin of `_lagrange_eval` against the pre-M3 source, read from disk |
| `t34_calib.py` | the T3-4 population, labelled against the RCWA oracle |
| `t34_analyse.py` | instrument bands, the conjunction's confusion matrix, the controls |
| `fixpaths.py` | makes the probes runnable under both Windows and WSL |

```bash
E="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

# byte-identity, three ways
cd C:/tmp/m3_probe
env $E python -u shadow.py base2.npz  "C:/tmp/m3_base"     # pre-M3 pmm/**
env $E M3_SEPARABLE=0 python -u shadow.py cur_sep0.npz     # items 1+2 only
env $E python -u shadow.py cur2.npz                        # items 1-4
python cmp.py base2.npz cur_sep0.npz 0.0                   # must be 0.0
python cmp.py base2.npz cur2.npz 1e-6

# speed + memory (idle machine, interleaved)
env $E python -u ab.py 5 5

# suites
cd <repo>
env $E python -m pytest tests/unit/test_pmm_m3_efficiency.py     tests/unit/test_pmm_m2_window_contract.py     tests/unit/test_pmm_per_layer_grids.py     tests/unit/test_m1_conditioning_guard.py -q
env $E python -m pytest tests/unit -k "pmm or rcwa or conditioning" -q

# WSL
wsl -e bash -lc "cd /mnt/d/.../Lumenairy && OMP_NUM_THREADS=1     OPENBLAS_NUM_THREADS=1 ~/lumvenv/bin/python -m pytest ... -q"
```

### 9.1 Evidence-integrity notes [M]

1. **A baseline that was not a baseline.** See S1.2. The first byte-identity run compared the
   current tree with itself and reported 0.0. Every number in this document is from a re-run with
   the root assertion in place; the tautological capture is discarded.
2. **A concurrent M4** holds `lumenairy/elements/rcwa/**` and two test files. `git diff --numstat`
   confirms M3 touched only `lumenairy/elements/pmm/{_core,stack,conical,_jax_stack}.py` plus its
   own new test file and this document. The RCWA oracle rows in the shadow capture read **0.0**
   between the two sides, which is the direct check that the oracle was not disturbed.
3. **A 500-solve breadth sweep was started and abandoned**, on wall clock: device U at `ns` = 12
   on the SHARED union grid at degree 12-16 builds a ~60-cell union (`n_glob` ~ 960) and costs
   minutes per solve. It is not needed for the conclusion: S5.5's refutation is a **counter-
   example**, and a counter-example with a full degree ladder and a `min_feature` control either
   side of it is decisive on its own. What the abandoned sweep would have added is breadth on the
   POSITIVE side, which is not what the decision turned on.
