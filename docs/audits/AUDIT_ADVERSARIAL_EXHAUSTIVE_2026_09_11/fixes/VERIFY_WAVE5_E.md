# VERIFY-WAVE5-E -- independent adversarial re-verification of Wave-5 item E

Branch `verify/wave5-item-e`, worktree `C:/tmp/lum_ve` at
`fix/wave5-item-e-leftovers` (`e4a3c304`); base `ede07f30`.  PRE trees are my
own `git archive` extractions -- `C:/tmp/lum_ve_pre` at `ede07f30` (E5) and
`C:/tmp/lum_ve_b14pre` at `96cb2096` (E4/D2).  The builder's PRE worktrees were
not used.

Every number below was **re-measured here**, on both builds, on fixtures I
wrote; nothing is quoted out of `WAVE5_E_LEFTOVERS_REPORT.md` except to say
whether it reproduced.  Probes and per-arm JSON:
`validation/probe_verify_wave5_e/`.

## 0. The two builds, and how every run was pinned

| | Windows | WSL (the CI condition) | third interpreter |
|---|---|---|---|
| interpreter | CPython **3.14.6** (MSC v.1944) | CPython **3.12.3** | CPython 3.14.6 (MSC v.1944) |
| numpy | 2.4.4 | 2.4.6 | **2.4.6**, clean venv `C:/tmp/ve_np246` |
| jax | 0.11.0 | 0.11.0 | -- |

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line of every invocation, `PYTHONPATH` pinned to the tree under test, every
probe prints `lumenairy.__file__`, pytest with `--capture=sys -p no:randomly`,
every tail grepped for `passed|failed|error|no tests ran`.

---

## 1. Verdict table

| claim | verdict | my numbers |
|---|---|---|
| **E1(a)** privatising the dispatchers costs +18.6..22.1 % of ASM at 512^2..2048^2 (Win) / +13 % (WSL); one `buf.copy()` is 0.30-0.42 of a forward transform | **DECISION CONFIRMED, NUMBERS DO NOT REPRODUCE** | Win **+28.3 / +32.7 / +23.7 %**; WSL **+2.4 / +4.5 / +16.2 %**; copy/fft **0.366-0.555** (Win), **0.011-0.475** (WSL).  The cost is larger than published, so the decision to refuse (a) is *a fortiori* right; the specific percentages are contention-sensitive readings |
| **E1(b)** every in-library product site names its other operand; 4 entry points x 3 shapes x 2 builds byte-identical (12/12) | **CONFIRMED, and the enumeration is incomplete** | AST walk: **10 sites, 0 elidable** (the note names 6); switch probe **12/12 identical on Windows AND 12/12 on WSL** |
| **E1(c)** the caller-visible move is rel 3.4e-16..4.0e-16 (WSL) and exactly 0 (Windows) | **CONFIRMED** | WSL **3.28e-16 / 3.80e-16 / 3.92e-16** at n = 256/512/1024, 79-82 % of the doubles; Windows **0.0, 0 doubles** |
| **E1(d)** the reproducer is lumenairy-free and reproduces on WSL; the version axis is ruled out; `np.multiply(a,b,out=b)` matches the named form while the elided spelling does not | **CONFIRMED, all three** | WSL `2.4.6 False 1.776e-16`; clean Windows venv **`2.4.6 True 0.0`**; explicit `out=` matches named on every build and every n, elided differs at n >= 128 on WSL only.  NEW: the **both-elidable** spelling also matches named, so it is right-*only* elision |
| **E1(e)** D4's sentence is verbatim in the knob doc, the setter docstring and the module note | **CONFIRMED 3/3** | verbatim (whitespace-normalised) in all three; `set_fft_plan_max_bytes_per_buffer` cross-references, as stated |
| **E2** N = 256 is the smallest sampling >= 10x that survives a doubling (12.9x / 490.9x; suppression 0.596 -> 1.79e-7, 0.216 -> 7.52e-8); premise gate re-runs the order-10 control (51.5x) and hard-fails; the annulus axis is monotone | **CONFIRMED except two statements** | ratio@3w **12.9243 (Win) / 12.9242 (WSL)** at 256 and **490.910 / 490.911** at 512; both seven-rung suppression rows reproduce **to every printed digit**; order-10 control **51.548** from **4.594672922387141e-02 -> 8.913411901995892e-04**.  **"seven of seven rungs clear 10x at 512" is 6 of 7** (the 2.0 w rung reads 4.63x -- the report's own table says 0.216).  **The annulus axis is monotone only inside the halo's support**: on a 25-rung sweep to 8 w it rises at 5.25 w (n = 256) and 5.50 w (n = 512) |
| **E3** the new bar is 27.3-34.0 ULP against readings 0-3 and a signal 5.16e+08, over 16 arms; the two-sided arm is reconstructed; the corrected readings are 1.7777776632250892 and -6.4436e-08 | **CONFIRMED, every figure** | 16 arms: readings **0..3 ULP**, products **5.333**, reduction **1.493..3.173**, bar **27.3..34.0**, independent-per-leg **5.158e+08..5.168e+08**; `r_none` on Windows/HASWELL/t1 is **1.7777776632250892** exactly and `dev` **-6.4436e-08** (the old -6.4376e-08 is outside the 16-arm envelope -6.4552e-08..-6.4420e-08).  A SECOND fixture of my own reads 0..2 ULP against a 27.1..30.7 bar and a 2.5e+09 signal |
| **E4 / D2** the glass red reproduces ALONE (1 failed / 7 passed), red in both orders (1/23), the single id alone green | **CONFIRMED, all five arms** | on my own `96cb2096` archive: **1f/7p**, **1f/23p**, **1f/23p**, **1f/1p**, **1 passed** |
| **E4 / D3** 2.39x / 9.58x (Win), 2.23x / 8.92x (WSL), one-column floor 11.53 MB at N = 256 | **CONFIRMED to the digit, both builds** | Win 0.84 / 0.60 / **2.39** / **9.58**, peak 432.42 / 9.58 MB; WSL 0.76 / 0.56 / **2.23** / **8.92**, peak 389.56 / 8.92 MB; floor **11.53 MB**.  Extended to N = 192 (floor 6.49 MB): the bound holds **iff** the budget clears the floor |
| **E5 / O-1** reached rows byte-identical 30/30; missed rows unfrozen 17/23 -> 0/23; opd drift 1.4e-5..1.9e-4 -> 0; Jacobian drift up to 1.5e300 -> 0 | **CONFIRMED and stronger on my fixtures** | 44 scored cells: reached rows **byte-identical 44/44** PRE vs POST, both builds; unfrozen **36/44 -> 0/44**; missed-row opd drift **up to 1.231e-03 m -> 0.0**; Jacobian drift **up to 1.3498e+300 -> 0.0**.  Windows and WSL summaries identical |
| **E5 / O-1** the frozen rows now equal `at_exit_vertex()`'s frozen state | **HALF CONFIRMED** | **24 of 44 cells** bit-exact (PRE: 4) -- **every FD cell**.  The 20 ANALYTIC cells still differ by **8.6e-04..1.1e-03 m**, for a reason upstream of the projection (it is also 8.8e-04 on `flat`, where the projection short-circuits).  Pre-existing, not introduced |
| **E5 / O-1** the round-1 literal `np.where(alive, ...)` breaks 7 of 18 WP-B12 pins | **REPRODUCED EXACTLY** | mutation `e5_round1_alive`: **7 failed, 11 passed** of the 18 |
| **E5 / O-3** a named `NotImplementedError` at all four fga sites; silent in air; byte-identical archive-to-archive with the budget pinned | **CONFIRMED** | 4/4 sites fire on my immersed prescription with `IMMERSED` in the message and 0/4 on the air twin, both builds; **12/12 digests byte-identical PRE vs POST** with `LUMENAIRY_MEM_BUDGET_MB=2000` (3 fixtures x 4 sites), both builds |
| **E5 / O-3** the tolerance is two-sided: 3.6x above STP air, ~500x below water | **REFUTED as stated** | the guard's boundary is `waves*lam/max(abs(z),lam)` exactly (bisected through the guard itself, 8 image distances, both builds).  The "3.6x above STP air" holds **only at a zero-length leg**.  At `z_image >= 1.0e-5 m` a real STP air index (1.000277) **IS refused**, and at the fixture's own `z_image = 0.35 mm` it is refused by **63x** |
| **E5-new** the JAX path of `ray_transfer_jacobian_analytic` reports every ray alive whatever the aperture | **CONFIRMED** | six-rung aperture ladder, 201 rays: JAX dead **0 at every rung** against NumPy-analytic **0 / 68 / 104 / 134 / 156 / 174** and the bundle tracer's identical set.  `_adrt_jax` ends `alive=jnp.ones((n,), dtype=bool)` -- there is no vignetting logic on that path at all.  The VALUES agree to **1.08e-19 m** on the rays NumPy kills, so it is the mask alone |
| **E6** the report records VERIFY-WP-B14 7a as closed | **CONFIRMED** | section 7 of the item report; nothing was built and nothing was needed |

Defects below: **D1** (P2, behaviour vs documentation, O-3's air claim), **D2**
(P2, the JAX analytic backend's missing vignetting -- the builder's own
out-of-scope finding, confirmed and characterised), **D3** (P2, durability --
the O-3 tolerance test asserts a copy of the derivation, not the derivation),
**D4** (P3, E2's trip counter discards a totally-suppressed rung), **D5** (P3,
"seven of seven at 512" is six of seven), **D6** (P3, the elision note's
enumeration of in-library sites is incomplete), **D9** (P3, durability -- the
`_fga_coarse` guard can be deleted invisibly), **D7** (P3, "where the bound is
inert the two fields are BIT-identical" is not what an inert cell reads),
**D8** (P3, E3's products term under-counts its own roundings).

---

## 2. E1 -- the `fft_infra` remedy decision

### 2.1 (a) The cost, re-measured

`probe_v_e1_cost.py` wraps `_fft2`/`_ifft2` in the consumer modules with a
`.copy()`-returning shim -- the cheapest faithful stand-in for remedy (a) --
and times the three arms **round-robin** (best of 15) so box contention hits
them equally.  A `nocopy_shim` arm prices the wrapper itself, so the copy's own
cost can be read shim-to-shim as well as against the shipped code.

| n | ASM Win (vs base / vs shim) | ASM WSL (vs base / vs shim) | ASM-no-bandlimit Win | Fresnel Win | copy / `_fft2` Win | WSL |
|---|---|---|---|---|---|---|
| 512 | **+28.3 %** / +39.0 % | +2.4 % / +31.5 % | +73.3 % | +9.8 % | 0.521 | 0.011 |
| 1024 | **+32.7 %** / +27.8 % | +4.5 % / +26.7 % | +31.3 % | -1.1 % | 0.555 | 0.058 |
| 2048 | **+23.7 %** / +32.0 % | **+16.2 %** / +13.2 % | +24.1 % | -0.1 % | 0.366 | 0.475 |

The published table reads +22.1 / +19.0 / +18.6 % (Windows) and -12.5 / +0.9 /
+13.0 % (WSL).  Mine are **higher on every Windows cell and on the WSL 2048
cell**, and the WSL 512 cell is +2.4 % rather than the published -12.5 %.  I do
not treat either set as reproducible: these are wall clocks on a box that was
running the rest of this verification.  What survives the difference is the
decision -- the copy is tens of percent of the ASM hot path at the shapes that
matter on both builds -- and that is the only thing the remedy turns on.

### 2.2 (b) The exposure, as an AST walk

`probe_v_e1_ast2.py` walks all 236 `lumenairy/*.py` files for every binary op
(`*`, `+`, `-`, `/`) one of whose operands is an FFT-dispatcher result, in
**both** spellings: the dispatcher called inline, and its result held under a
name (`spec = _fft2(E); spec * H`).  The other operand is classified for
ELIDABILITY, not for syntax.

**10 sites, 0 with an elidable operand**, on this tree:

| file:line | scope | spelling | other operand |
|---|---|---|---|
| `asm.py:919` | `angular_spectrum_propagate` | `_fft2(E_in) * H` | NAME |
| `asm.py:922` | `angular_spectrum_propagate` | `_fft2(ifftshift(E_in)) * H` | NAME |
| `asm.py:1147` | `angular_spectrum_propagate_batch` | `_fft2_nd(...) * H[None, :, :]` | **basic-slice VIEW** |
| `asm.py:1391` | `angular_spectrum_propagate_tilted` | `_fft2(ifftshift(E_demod)) * H` | NAME |
| `carrier.py:1401` | `_exact_envelope_tf_step` | `_fft2(E) * H` | NAME |
| `carrier.py:7126` | `_shift_envelope` | `_fft2(_e) * ramp` | NAME |
| `fresnel.py:216` | `fresnel_tf_propagate` | `_fft2(...) * H` | NAME |
| `rs.py:936/939/942` | `rayleigh_sommerfeld_propagate` | `E_fft * H`, `E_fft = _fft2(...)` | NAME |

The claim holds.  The **enumeration in the module note does not** -- it names
six sites and says "every"; four more exist (D6).  Three of them (`rs.py`) are
the more fragile shape, because the dispatcher's non-owning view is held under
a name and the next edit to that line does not have `_fft2(` staring at it.

`asm.py:1147`'s `H[None, :, :]` is a *fresh object* but a **view**
(`owndata` False), so `temp_elide` cannot claim it either.  An advanced index
(`H[mask]`) at the same place WOULD be elidable; that is why the walk
classifies by elidability rather than by "is it a Name".

`probe_v_e1_switch.py` then measures the four entry points
(`angular_spectrum_propagate`, the same with `bandlimit=False`,
`fresnel_propagate`, `carrier._exact_envelope_tf_step`) x n = 256/512/1024
across `set_fft_double_buffer`: **12 of 12 byte-identical on Windows and 12 of
12 on WSL**.  (The published phrasing, "four entry points x three shapes x both
builds: byte-identical ... 12 of 12 cells", counts one build's twelve and then
says "x both builds"; the cell count over both builds is 24.  Both readings are
12/12 here.)

### 2.3 (c) The caller spelling that IS exposed

Same probe, `_ifft2(_fft2(E) * np.exp(1j*P))` against its named twin:

| n | Windows numpy 2.4.4 | WSL numpy 2.4.6 |
|---|---|---|
| 256 | identical, 0 / 131 072 doubles | rel **3.277e-16**, 104 212 / 131 072 |
| 512 | identical | rel **3.798e-16**, 425 215 / 524 288 |
| 1024 | identical | rel **3.916e-16**, 1 727 262 / 2 097 152 |

and in the same process `np.multiply(a, h, out=h)` is bit-identical to the
named form at every n on both builds.  `_fft2`'s return reads `owndata False`
in both modes' measurement above because the probe measures it with the
ping-pong on; the mode-dependence of `owndata` is what
`test_wave5_e_fft_elision.py`'s gated arm asserts and I did not re-derive it.

### 2.4 (d) The upstream half

`numpy_elision_reproducer.py` imports nothing but NumPy (verified: one
occurrence of the string "lumenairy", in the docstring).  Readings:

| interpreter | reading |
|---|---|
| Windows py3.14.6, numpy **2.4.4** | `2.4.4 True 0.0` |
| Windows py3.14.6, numpy **2.4.6**, clean venv built here | `2.4.6 True 0.0` |
| WSL py3.12.3, numpy **2.4.6** manylinux | `2.4.6 False 1.776e-16` |

So **the version axis is ruled out** -- independently of the builder's venv,
which I did not use.  My spelling matrix (`probe_v_e1_spellings.py`) adds one
row the published matrix does not have: on the affected build the
**both-operands-elidable** spelling also equals the named form.  So it is not
"elision changes the answer"; it is specifically the RIGHT-only elision.

| spelling | WSL 2.4.6 | Win 2.4.4 | Win 2.4.6 |
|---|---|---|---|
| `(A*1.0) * h` (left elidable) | = named | = named | = named |
| `a * np.exp(1j*P)` (right elidable) | **differs**, n >= 128 | = named | = named |
| `(A*1.0) * np.exp(1j*P)` (both) | = named | = named | = named |
| `np.multiply(a, h2, out=h2)` | = named | = named | = named |
| `np.multiply(a2, h, out=a2)` | = named | = named | = named |
| `np.multiply(a, h, out=o)` | = named | = named | = named |

### 2.5 (e) The sentence

`probe_v_e1_doc.py` normalises whitespace and emphasis and looks for D4's
sentence in the **running** library: present in the registered knob doc
(through `lumenairy._knobs.knob_doc`), in `set_fft_double_buffer.__doc__` and
in the `_PYFFTW_DOUBLE_BUFFER` module note -- **3 of 3**.

---

## 3. E2 -- the C8 default-order ladder

`probe_v_e2_c8.py` re-implements the element call, the halo field and the
annulus ladder from scratch and adds a **25-rung** radial sweep (2.0 to 8.0
beam widths in 0.25 steps) the pin does not have.

### 3.1 What reproduced

Both builds, shipped `_DECENTRED_FIT_POLY_ORDER = 16`, no order passed:

| r > | 2.0 w | 2.5 w | 3.0 w | 3.5 w | 4.0 w | 4.5 w | 5.0 w |
|---|---|---|---|---|---|---|---|
| n = 256 suppression | 0.5964 | 0.5964 | 7.737e-2 | 8.951e-3 | 8.951e-3 | 2.369e-5 | 1.790e-7 |
| n = 512 suppression | 0.2158 | 2.280e-2 | 2.037e-3 | 1.663e-4 | 1.299e-5 | 9.950e-7 | 7.517e-8 |

-- every printed digit of the published table, on both builds.  Ratio beyond
three beam widths: **12.9243 (Win) / 12.9242 (WSL)** at 256 and **490.910 /
490.911** at 512.  The order-10 `_GHOST` control reads **51.548** from
`4.594672922387141e-02 -> 8.913411901995892e-04` -- the published pair, bit for
bit, on both builds (at the r > 3.0 w rung; the report quotes the pair without
saying which rung it is read at).

### 3.2 What did not

**"seven of seven rungs clear 10x at n = 512" is six of seven** (D5).  The
2.0 w rung's suppression is 0.2158, i.e. **4.63x**, which is below the file's
own `_TRIP = 10.0`.  The sentence contradicts the table printed three lines
above it in the same documents.  Nothing fails because of it -- the assertions
are `>= 3` rungs -- but it is a stated measurement that is wrong.

**The annulus-radius axis is not monotone as a mechanism.**  Over 25 rungs the
suppression is strictly non-increasing out to **5.25 w** (n = 256) and
**5.50 w** (n = 512) and then rises, because the OFF field's own halo ends:

| n = 256 | 5.25 w | 5.50 w | ... | 6.50 w | 7.00 w |
|---|---|---|---|---|---|
| off (normalised) | 1.495e-01 | **4.953e-03** | | 1.041e-08 | 0.0 |
| on | 2.065e-08 | 2.065e-08 | | 0.0 | 0.0 |
| suppression | 1.381e-07 | **4.169e-06** | | 0.0 | 1.0 (convention) |

So the pin's seven rungs sit inside the region where the claim is true, and
the claim as stated in the test docstring ("a property of the mechanism, not
of the cell") is a property of the mechanism **within the halo's support**.
This is not a defect in the pin -- it is a scope its docstring does not carry,
and a build whose halo shrank below 5 w would fire the monotone arm for a
reason that is not the bound.

**"where the bound is inert the two fields are BIT-IDENTICAL"** (D7).  On an
inert cell of my own (`alpha = 3.0, cx = 1.40 mm, z = 12 mm, frbf = 2.0,
n = 256`) all four annuli read ratio exactly 1.0 -- but the two fields are
**not** bit-identical: total power moves by 3e-16 relative on Windows and
3.3e-14 on WSL.  The conclusion (the inert reading is exactly 1.0, so a 10x bar
has a decade of gap) survives; the stated mechanism does not.  The ratio is
exactly 1.0 because the annulus *maximum* is an untouched pixel.

### 3.3 The trip counter discards its own strongest rung (D4)

```python
trips = [r for r in rows if r[3] > 0.0 and (1.0 / r[3]) >= _TRIP]
```

A rung the bound empties **completely** has suppression exactly 0.0 and is
therefore counted as **not tripping** -- the strongest possible signal is
discarded.  `_fmt` and `_require` both treat the same reading as `inf`, so the
file is internally inconsistent about it.  Measured: at n = 256 the 6.50 w and
6.75 w rungs read `on == 0.0` exactly, so they are silently dropped.  It makes
the pin harder to pass, never easier, but it can turn a live stimulus into a
skip.

---

## 4. E3 -- the Maslov joint-scale ULP bar

`probe_v_e3_ulp.py` re-derives the bar from scratch and runs **two** fixtures:
the pin's own (f/3.3 N-BK7, 96^2, input ratio 1.7777777777777779) and one of
mine (N-SF11, 128^2, 12 um pitch, 1.03 um, input ratio 6.3178...) so the bar is
exercised where the pin has never run.  16 arms: both builds x
`OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x 1 and 4
threads, every arm's architecture confirmed by `threadpoolctl`.

| | pinned fixture, 16 arms | my fixture, 16 arms |
|---|---|---|
| `'power'` / `'peak'` reading | **0 .. 3 ULP** | 0 .. 2 ULP |
| products term (derived) | 5.333 ULP, every arm | 4.738 ULP |
| reduction term (measured) | **1.493 .. 3.173 ULP** | 2.034 .. 2.932 |
| **the bar** | **27.3 .. 34.0 ULP** | 27.1 .. 30.7 |
| independent per-leg scale | **5.158e+08 .. 5.168e+08 ULP** | 2.520e+09 .. 2.565e+09 |
| bar / widest reading, worst arm | **10.6x** | 13.6x |
| bar > widest reading, every arm | **yes, 16/16** | yes, 16/16 |
| independent > 1e3 x bar, every arm | **yes, 16/16** | yes, 16/16 |

Every published figure reproduces: the envelope 0-3 ULP, the products 5.333,
the reduction 1.493-3.173, the bar 27.3-34.0, the signal 5.16e+08.  The
report's "SANDYBRIDGE/4 reads 3 ULP against its own 2.623 ULP reduction term"
reproduces exactly on my Windows SANDYBRIDGE t4 arm.

The two corrected readings reproduce: Windows / HASWELL / t1 gives
`r_none = 1.7777776632250892` exactly, and `dev = -6.443589e-08`.  The 16-arm
envelope of `dev` is **-6.4552e-08 .. -6.4420e-08**, which contains the new
figure (-6.4436e-08) and **excludes** the old one (-6.4376e-08).  So the
correction is right and the old value was not a per-build reading.

**Is the bar derived from the right condition?**  Yes in kind: each power is a
sum of N^2 non-negative terms, the reduction's rounding is a genuine second
source, and it is measured at run time over five summation trees against
`math.fsum` rather than assumed.  **Is 4x a free constant?**  Yes -- it is a
safety factor with an argument (the derived quantity spans 2.1x across the
ladder) rather than a derivation, and the report says so.  It is not
load-bearing either way: with the safety factor the bar clears the widest
reading by 10.6x at the worst arm and sits 1.5e+07x below the signal, and
WITHOUT it (safety 1) the bar would be 6.8 to 8.5 ULP against a 3 ULP reading
-- still two-sided, but at 2.3x rather than 10.6x, which is the S4 shape this
re-derivation exists to get away from.  So 4 is doing work and is not
decoration.

One inaccuracy (D8): the stated "at most three roundings per leg"
under-counts.  Per complex element the chain is `fl(s*re)`, `fl(s*im)`,
`fl((s*re)^2)`, `fl((s*im)^2)`, `fl(sum)` -- five, not three -- so the products
term should be `10u`, not `6u`.  Re-derived over the same 16 arms, that moves
the bar to **41.53-48.25 ULP**, still 15.4x above the widest reading and
1.07e+07x below the signal, so not one arm changes verdict; it is the
docstring's arithmetic that is wrong, not the gate.

---

## 5. E4 -- the two corrected claims

### 5.1 D2, on my own `96cb2096` archive

| selection | measured here | published |
|---|---|---|
| `test_v4_16_0_agent_d_validity_ranges.py` **alone** | **1 failed, 7 passed** | 1 failed, 7 passed |
| meshgrid file FIRST, then it | **1 failed, 23 passed** | same |
| it FIRST, then the meshgrid file | **1 failed, 23 passed** | same |
| its own first test + the failing id | **1 failed, 1 passed** | same |
| the failing id in isolation | **1 passed** | same |

The failing id is
`test_validity_warning_is_one_shot_per_pair`; the correction is right and the
published reproduction reproduces exactly.

### 5.2 D3, on an independently written probe

`probe_v_e4_gbd.py` imports nothing from the builder's probes: the bundle, the
chunk formula and the `tracemalloc` harness are written from the public API and
the two module constants, and a **second grid** (N = 192) is swept so the
statement is checked as a function of the floor.

| N | floor `Ny*Nx*(48 + 128)` | budget | chunk | Win peak / ratio | WSL peak / ratio |
|---|---|---|---|---|---|
| 256 | **11.53 MB** | 512 MB | 61 | 432.42 MB / **0.84x** | 389.56 MB / **0.76x** |
| 256 | | 16 MB | 1 | 9.58 MB / **0.60x** | 8.92 MB / **0.56x** |
| 256 | | 4 MB | 1 | 9.58 MB / **2.39x** | 8.92 MB / **2.23x** |
| 256 | | 1 MB | 1 | 9.58 MB / **9.58x** | 8.92 MB / **8.92x** |
| 192 | **6.49 MB** | 512 MB | 108 | 384.11 MB / 0.75x | 352.13 MB / 0.69x |
| 192 | | 16 MB | 3 | 12.52 MB / 0.78x | 11.51 MB / 0.72x |
| 192 | | 4 MB | 1 | 5.45 MB / 1.36x | 5.02 MB / 1.25x |
| 192 | | 1 MB | 1 | 5.45 MB / 5.45x | 5.02 MB / 5.02x |

All eight of the published figures reproduce **to the digit on the build they
were recorded on**, on a bundle with a different seed, a different position
scale, a different `Q` and a different waist -- which also shows the peak is
set by the grid, not by the beamlets.  The N = 192 rows are new here and give
the scope sentence its build-free form: **the budget is a bound exactly when it
clears the one-column floor**, at both grids, on both builds.  That is what my
new id `test_the_measured_budget_is_a_bound_exactly_above_the_one_column_floor`
pins.

---

## 6. E5 -- the exit-vertex freeze and the FGA image leg

### 6.1 O-1, on my own optics

`probe_v_e5_freeze.py` builds six surface classes with radii, conics,
aspheric coefficients and tilts different from the item-E fixture (conic,
asphere, biconic, mirror, oblique at 8 degrees, flat) and runs each at four
aperture over-fills (1.6x, 2.2x, 3.0x, 4.0x) on both backends -- 48 cells, 44
scored (the analytic backend refuses biconic).  It runs unchanged in the PRE
archive and the POST tree.

| | PRE (`ede07f30`) | POST | both builds |
|---|---|---|---|
| rows that REACHED the surface, PRE vs POST digest | -- | **byte-identical 44/44** | identical |
| cells whose MISSED rows are frozen | 8 of 44 | **44 of 44** | identical |
| missed-row `opd` drift | up to **1.231e-03 m** | **0.0** | identical |
| missed-row Jacobian drift | up to **1.3498e+300** | **0.0** | identical |
| frozen rows == `at_exit_vertex`, bit-exact | 4 of 44 | **24 of 44** | identical |
| companion-dead-but-base-alive rays | 17 over 12 cells | 17 over 12 cells | identical |

Windows and WSL produce **identical summaries**, cell for cell.

The `at_exit_vertex` row is the one the brief asks about directly.  The answer
is: **for the finite-difference backend, yes, all 24 cells**; for the analytic
backend, no -- its frozen `opd` on rays that missed the surface sits
8.6e-04..1.1e-03 m from `at_exit_vertex`'s, including on `flat`, where the
projection short-circuits and does nothing at all.  So that gap is the analytic
tracer's own dead-ray state, not this change: it was there before (PRE reads
the same 20 cells apart, and the drift is the same 8.8e-04 on `flat`).  O-1's
goal -- the module's two vertex-plane operators agreeing on dead rays -- is
therefore met on the FD backend and still open on the analytic one, which is
the primitive `fga._pick_ray_transfer` PREFERS (`exact=None` auto-selects
analytic wherever it applies).  It stays unobservable for the same reason O-1
was: all four `fga.py` consumers zero the dead beamlets before the
reconstruction.  Recorded, not scored against item E -- it is upstream of the
projection and outside O-1's wording.

### 6.2 The round-1 fail-before, reproduced

Mutation `e5_round1_alive` restores VERIFY-WP-B12 O-1's literal one-liner by
passing `reached_surface=None` at both `ray_transfer_jacobian` call sites, so
the freeze keys on the FD backend's `base & companion` alive again:

    tests/unit/test_audit2609_b12_fga_reference_plane.py
    tests/unit/test_verify_b12_fga_reference_plane.py   ->  7 failed, 11 passed

**7 of 18**, exactly as the report states.  The same mutation reddens 5 of the
29 ids of the new E5 file, so the repair is pinned on its own terms as well.

### 6.3 O-3, the guard

`probe_v_e5_o3_guard.py`, on an N-LAK22 singlet at 1.55 um (not the item-E
fixture):

* the guard fires at **4 of 4** sites (`apply_real_lens_fga`, the same with
  `coarse_stride`, `apply_real_lens_fga_vector`, `_caustic_zone`) with
  `IMMERSED` in the message, and at **0 of 4** on the air-terminated twin, on
  both builds;
* `get_glass_index('air', lambda)` returns **exactly 1.0** at 633, 780, 1030,
  1060, 1310 and 1550 nm, so the air control really does sit at the tolerance's
  origin;
* `probe_v_e5_fga_bytes.py` runs three air-terminated FGA fixtures x four sites
  in the PRE archive and the POST tree with `LUMENAIRY_MEM_BUDGET_MB=2000`
  exported (VERIFY-WP-B12 D-4): **12 of 12 digests byte-identical**, on both
  builds.  So neither the guard nor the freeze moves a byte of an
  air-terminated FGA field.

The tolerance, bisected **through the guard itself** rather than through a copy
of its formula:

| `z_image` | refusal boundary `abs(n-1)` | predicted | STP air (1.000277) | n = 1.0001 | water |
|---|---|---|---|---|---|
| 0 | 1.0000e-03 | 1.0000e-03 | not refused | not refused | refused |
| 1.55e-6 (= lambda) | 1.0000e-03 | 1.0000e-03 | not refused | not refused | refused |
| 1e-5 | 1.5500e-04 | 1.5500e-04 | **REFUSED** | not refused | refused |
| 1e-4 | 1.5500e-05 | 1.5500e-05 | **REFUSED** | **REFUSED** | refused |
| 3.5e-4 | 4.4286e-06 | 4.4286e-06 | **REFUSED** | **REFUSED** | refused |
| 1e-3 | 1.5500e-06 | 1.5500e-06 | **REFUSED** | **REFUSED** | refused |
| 1e-2 | 1.5500e-07 | 1.5500e-07 | **REFUSED** | **REFUSED** | refused |

Identical on both builds.  The boundary IS the derivation, to every digit.
What is not true is the two-sidedness as the docstring, the report and the pin
all state it: see D1.

### 6.4 E5-new -- the JAX analytic backend

`probe_v_e5_jax_alive.py`, 201 rays across a fixed fan, clear aperture shrunk
rung by rung:

| semi-diameter | fan / semi | bundle dead | FD dead | NumPy-analytic dead | **JAX-analytic dead** |
|---|---|---|---|---|---|
| 0.45 mm | 1.0 | 0 | 2 | 0 | **0** |
| 0.30 mm | 1.5 | 68 | 68 | 68 | **0** |
| 0.22 mm | 2.05 | 104 | 104 | 104 | **0** |
| 0.15 mm | 3.0 | 134 | 134 | 134 | **0** |
| 0.10 mm | 4.5 | 156 | 157 | 156 | **0** |
| 0.06 mm | 7.5 | 174 | 174 | 174 | **0** |

The 0.15 mm rung is the report's cell (0 of 201 against 134) and it
reproduces.  Characterisation is in D2 below.

---

## 7. Durability -- the mutation matrix

Every mutation is applied to a **copy** of the tree (`C:/tmp/lum_ve_mut`); the
verification worktree's `lumenairy/` was never edited
(`git status --porcelain lumenairy/` empty).  `validation/probe_verify_wave5_e/
mutate.py` restores every mutable file from the pristine tree before each arm
-- a first pass that restored only the file a given mutation named produced
three arms that silently read the arm before them, which is recorded here
because it is exactly the failure mode a mutation matrix exists to avoid.

Fourteen mutations plus the identity arm, across all three item-E files, the
two library modules they defend and the 18 WP-B12 reference-plane pins.

| # | mutation | what it breaks | selection | Windows | WSL | caught? |
|---|---|---|---|---|---|---|
| 0 | *identity* | -- | the three item-E files | 44 passed, 1 skipped | 45 passed | -- |
| 1 | `e5_unfreeze` | the Jacobian freeze is disabled (`reached \| True`) | E5 file | **9 failed**, 20 passed | **9 failed**, 20 passed | **yes** |
| 2 | `e5_unfreeze` | same | the 18 WP-B12 pins | 18 passed | 18 passed | no -- they mask dead rays, as VERIFY-WP-B12 said |
| 3 | `e5_freeze_everything` | `reached &= False`, i.e. the projection becomes the identity | E5 file | **14 failed**, 15 passed | **14 failed**, 15 passed | **yes** -- the live-motion arm |
| 4 | `e5_round1_alive` | the ROUND-1 literal: freeze on `transfer.alive` at both call sites | E5 file | **5 failed**, 24 passed | **5 failed**, 24 passed | **yes** |
| 5 | `e5_round1_alive` | same | the 18 WP-B12 pins | **7 failed**, 11 passed | **7 failed**, 11 passed | **yes -- the published fail-before, reproduced exactly on both builds** |
| 6 | `e5_tol_drop_z` | the guard's tolerance loses `max(abs(z_image), lam)` | E5 file | **29 passed** | **29 passed** | **NO -- defect D3** |
| 7 | `e5_guard_off_at_coarse` | `_require_non_immersed_exit` deleted from `_fga_coarse` | E5 file | **29 passed** | **29 passed** | **NO -- defect D9** |
| 8 | `e5_guard_off_at_through_lens` | deleted from `_fga_through_lens` instead | E5 file | 1 failed, 28 passed | 1 failed, 28 passed | partly -- only the `through_lens` arm |
| 9 | both 7 and 8 | deleted from BOTH | E5 file | **2 failed**, 27 passed | **2 failed**, 27 passed | yes -- which is how D9 is proved |
| 10 | `e1_unnamed_right_operand` | `asm.py:919` becomes `_fft2(E_in) * np.exp(np.log(H))` | E1 file | 11 passed, 1 skipped | **2 failed**, 10 passed | **build-dependent** -- closed build-free by my new AST id |
| 11 | `e1_privatise_dispatchers` | the pyFFTW dispatchers always return `buf.copy()` | E1 file | 11 passed, 1 skipped | **1 failed**, 11 passed | build-dependent (the gated arm skips on Windows) |
| 12 | `e1_drop_scope_sentence` | the knob doc reverts to "values are byte-identical either way" | E1 file | **1 failed**, 10 passed, 1 skipped | **1 failed**, 11 passed | **yes** |
| 13 | `e2_bound_disabled` | `REMAP_INVERSE_SUPPORT_BOUND` forced False everywhere | E2 file | **4 failed** | **4 failed** | **yes, and it HARD-FAILS** -- the premise gate's control is dead too |
| 14 | `e2_annuli_reversed` | the ladder's rungs run outward-in | E2 file | **2 failed**, 2 passed | **2 failed**, 2 passed | **yes** -- the monotone arms |
| 15 | `e2_stimulus_dead` | no candidate reaches the bar | E2 file | 4 skipped | 4 skipped | **correct**: SKIPS, with the order-10 control alive |
| 16 | `e2_stimulus_dead` + `e2_control_weak` | and the order-10 control cannot see the bound either | E2 file | **4 failed** | **4 failed** | **yes -- HARD FAIL, not a skip**, which is what the gate promises |

Arms 15 and 16 together are the premise gate's two-sided claim, and they behave
as the file documents: the stimulus going away is a skip that prints every
candidate's reading, and the stimulus going away *while the bound is dead* is a
hard failure.

Two process notes, recorded because they are the failure modes a mutation
matrix exists to catch and I hit both:

* a first version of `mutate.py` restored only the file the current mutation
  named, so three arms silently ran with the previous arm's edit still in
  place and reported its verdict.  The harness now restores every mutable file
  before every arm.
* editing `run_mutations.sh` while it was executing corrupted the running
  shell's byte offsets mid-file (bash reads scripts incrementally), which made
  it try to execute a Python test file.  The matrix was re-run from a frozen
  copy.

---

## 8. Defects

### D1 (P2, documentation vs behaviour) -- O-3's tolerance is two-sided only at a zero-length leg, and the sentence that says otherwise is the one a caller reads

`_require_non_immersed_exit`'s docstring, `WAVE5_E_LEFTOVERS_REPORT.md` sec. 6.3
and `test_wave5_e_exit_vertex_dead_rays.py::
test_the_guard_tolerance_is_the_wavefront_it_protects` all state the tolerance's
lower gap as "3.6x the air-vs-vacuum index difference at STP (2.77e-4, so a
caller who registers a real air index is NOT refused)".

**Measured** (`probe_v_e5_o3_guard.py`, bisecting the guard itself, both
builds, identical): the tolerance is `waves*lambda/max(|z_image|, lambda)` and
the 3.6x margin exists only while `|z_image| <= lambda`.

| `z_image` | tolerance | STP air (n-1 = 2.77e-4) |
|---|---|---|
| 0 | 1.0e-03 | not refused (3.6x margin) |
| 1e-5 m | 1.55e-04 | **refused** |
| 3.5e-4 m (the fixture's own leg) | 4.43e-06 | **refused, by 63x** |
| 1e-2 m | 1.55e-07 | **refused, by 1800x** |

The BEHAVIOUR is self-consistent -- real air over a 0.35 mm leg costs 63
milliwaves against a 1-milliwave budget, so refusing it is what the budget
says.  What is wrong is the claim, and it is wrong in the direction that
matters: it tells a reader the guard will not refuse a real air index, when at
every image distance the FGA actually runs it will.  `get_glass_index('air',
lambda)` returns exactly 1.0 on this registry, so nothing served today is
affected -- but the guard is not the "immersion" guard its message and its
docstring say it is; it is a near-unity-exit-index guard, and a caller who
registers air, a purge gas or an index-matching fluid at n = 1.0001 meets a
`NotImplementedError` that says "the optic is IMMERSED".

**Requested edit**, in `lumenairy/propagators/fga.py`'s
`_require_non_immersed_exit` docstring, replacing the parenthetical:

```
    The ``max(..., wavelength)`` clamp is what makes a zero-length leg
    (``_caustic_zone``, or ``output_plane_distance=0``) still refuse a real
    immersion medium: the tolerance then floors at the budget itself, 1e-3,
    which is ~500x below any immersion medium (water 1.33, oil 1.52).

    THE FLOOR IS NOT THE TOLERANCE.  At a real image leg the tolerance is
    ``waves_budget * wavelength / |z_image|``, and it is TIGHT: at
    ``z_image = 0.35 mm``, ``lambda = 1.55 um`` it is 4.4e-06, so an exit
    medium registered as REAL AIR (n - 1 = 2.77e-4 at STP) is refused by 63x
    (measured 2026-09-19, VERIFY-WAVE5-E sec. 6.3).  That is the wave budget
    working as derived -- real air over that leg costs 63 milliwaves, not one
    -- but it means this guard refuses any near-unity exit medium and not only
    immersion.  ``get_glass_index('air', lambda)`` returns EXACTLY 1.0 on this
    registry at every wavelength measured, so no prescription served today is
    affected.
```

and, in the raised message, after "i.e. the optic is IMMERSED", add
"(or, at a long image leg, merely has an exit index far enough from 1 to spend
the wavefront budget over that leg)" -- the `IMMERSED` token stays, so the four
existing `pytest.raises(..., match='IMMERSED')` arms keep passing.  The same
two sentences belong in `WAVE5_E_LEFTOVERS_REPORT.md` sec. 6.3 and in the
test's docstring.

---

### D2 (P2, correctness, pre-existing, confirmed from the builder's own out-of-scope note) -- the JAX path of `ray_transfer_jacobian_analytic` reports every ray alive

`lumenairy/raytrace/differential.py::_adrt_jax` ends

```python
    return DifferentialTransfer(
        jacobian=jac, x=st[:, 0], y=st[:, 1], ux=st[:, 2], uy=st[:, 3],
        opd=opd, alive=jnp.ones((n,), dtype=bool))
```

and calls `_adrt_step(..., compute_dead=False)`.  There is no vignetting,
no TIR and no missed-surface logic on that path at all.

**Measured** (`probe_v_e5_jax_alive.py`, 201 rays, a fixed fan, the clear
aperture shrunk rung by rung; both builds identical):

| semi-diameter | fan / semi | bundle tracer dead | NumPy analytic dead | **JAX analytic dead** |
|---|---|---|---|---|
| 0.45 mm | 1.0 | 0 | 0 | 0 |
| 0.30 mm | 1.5 | 68 | 68 | **0** |
| 0.22 mm | 2.05 | 104 | 104 | **0** |
| 0.15 mm | 3.0 | 134 | 134 | **0** |
| 0.10 mm | 4.5 | 156 | 156 | **0** |
| 0.06 mm | 7.5 | 174 | 174 | **0** |

The two analytic paths agree on the VALUES they compute for the rays NumPy
kills, to **1.08e-19 m** of OPL, so the defect is the mask alone and the fix is
local.

**Who reads it.**  `ray_transfer_jacobian_analytic` is public
(`lumenairy.ray_transfer_jacobian_analytic`) and its own docstring advertises
the JAX backend as the `jax.grad`/`jax.jit`-differentiable one, so a
differentiable-optics caller is the exposed consumer.  The two in-library
consumers cannot reach the JAX path at all: `fga._pick_ray_transfer`
(`fga.py:828-859`, used at `fga.py:1544`) hands it NumPy arrays, and
`gbd.py:3568` calls it with `per_surface=True`, which `_adrt_jax` refuses with
its own `NotImplementedError` before any tracing happens.  So nothing the
library ships reads the JAX `alive` today and the exposure is the public entry
point alone.  It also makes 7 of the item-E
probe's 30 cells vacuous, which is how the builder found it.

**Requested edit** (NOT made here).  `_adrt_step` already computes the same
`dead` predicate for both namespaces; `_adrt_jax` switches it off.  In
`_adrt_jax._full`:

```python
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        opd = jnp.zeros(())
        dead = jnp.zeros((), dtype=bool)                       # NEW
        for si, s in enumerate(surfaces):
            xx, yy, uxx, uyy, dopd, d = _adrt_step(
                xx, yy, uxx, uyy, s, wavelength, si < nsurf - 1, jnp_ops,
                compute_dead=True)                             # was False
            opd = opd + dopd
            dead = dead | d                                    # NEW
        state = jnp.stack([xx, yy, uxx, uyy])
        return state, (state, opd, dead)                       # NEW
```

and at the call and the return:

```python
    jac, (st, opd, dead) = jax.vmap(jax.jacfwd(_full, has_aux=True),
                                    in_axes=1, out_axes=(0, (0, 0, 0)))(s4)
    return DifferentialTransfer(
        jacobian=jac, x=st[:, 0], y=st[:, 1], ux=st[:, 2], uy=st[:, 3],
        opd=opd, alive=jnp.logical_not(dead))
```

The in-line comment at `differential.py:897` ("the JAX path returns alive=True
and would trip a tracer->ndarray conversion here") is the reason the switch
exists; every `np.isfinite` / `float()` in that block reads SURFACE attributes,
not ray state, so under `vmap(jacfwd(...))` `dead` should come back as a scalar
bool tracer and vmap to `(n,)`.  That has to be MEASURED, not assumed -- if a
conversion does trip, the fallback is to compute the mask in a separate
un-differentiated `jax.vmap` pass over the same `_adrt_step` with
`compute_dead=True`, which costs one extra primal walk and no gradient.

Until it is fixed, `tests/unit/test_verify_wave5_e.py::
test_the_jax_analytic_backend_reports_no_vignetting_KNOWN_DEFECT` pins it in
the shape `test_verify_b14_known_reds.py` uses, and says in its own failure
message what to replace it with.

---

### D3 (P2, durability) -- the O-3 tolerance pin asserts a COPY of the derivation, so the derivation can change under it

`tests/unit/test_wave5_e_exit_vertex_dead_rays.py::
test_the_guard_tolerance_is_the_wavefront_it_protects` defines

```python
    def tol(z):
        return max(_fga._FGA_EXIT_INDEX_NOISE_FLOOR,
                   waves * lam / max(abs(z), lam))
```

inside the test and then asserts four properties of that local function.  It
reads two module constants but never the formula, so it cannot see the
library's formula move.

**Measured** (mutation `e5_tol_drop_z`): rewriting the guard's tolerance from
`waves*lam/max(|z_image|, lam)` to `waves*lam/lam` -- deleting the image leg's
length from the derivation entirely -- leaves all 29 ids of that file GREEN.
The four site arms survive too, because a glass exit (n = 1.617) is refused
under either formula and an exactly-1.0 air exit is refused under neither.

**Requested edit**: replace the local `tol` with a bisection through the guard,
as `tests/unit/test_verify_wave5_e.py::
test_the_immersed_exit_guards_boundary_is_the_guards_own_derivation` does (that
id is the closure of this gap and takes 0.01 s), or, minimally, add two
`_require_non_immersed_exit` calls per `z` -- one just inside the predicted
boundary and one just outside -- so the assertion is about the guard.

---

### D4 (P3, test logic) -- E2's trip counter discards the strongest rung it can measure

`tests/unit/test_wave5_e_c8_default_order.py::_evaluate`:

```python
    trips = [r for r in rows if r[3] > 0.0 and (1.0 / r[3]) >= _TRIP]
```

A rung the bound empties COMPLETELY has `suppression == 0.0` exactly, and
`r[3] > 0.0` drops it -- total removal is counted as *not tripping*.  `_fmt`
and `_require` both read the same 0.0 as `inf`, so the file contradicts itself.

**Measured**: on the selected cell at n = 256 the 6.50 w and 6.75 w rungs read
`on == 0.0` exactly, so they are silently dropped from the trip count.  It can
only make the pin harder to pass, never easier -- but it can turn a live
stimulus into a skip.

**Requested edit**:

```python
    trips = [r for r in rows if r[3] <= 1.0 / _TRIP]
```

which reads exactly-zero suppression as a trip and is the same predicate
`_fmt`/`_require` already use.

---

### D5 (P3, documentation) -- "seven of seven rungs clear 10x at n = 512" is six of seven

`WAVE5_E_LEFTOVERS_REPORT.md` sec. 3.3 and
`tests/unit/test_wave5_e_c8_default_order.py`'s header both say "five of seven
rungs clear 10x at 256 and seven of seven at 512".  The suppression table
printed immediately above the sentence in both documents gives 0.216 at the
2.0 w rung, i.e. **4.63x**, which is below the file's own `_TRIP = 10.0`.
Re-measured here on both builds: **6 of 7**.  Nothing fails on it (the
assertions are `>= 3`), but it is a stated measurement that contradicts its own
table.

**Requested edit**: "…and six of seven do at 512 (the 2.0 w rung reads 4.63x)".

---

### D6 (P3, documentation) -- the elision note's list of in-library product sites is incomplete, and it is maintained by hand

`lumenairy/propagators/fft_infra.py`'s `_PYFFTW_DOUBLE_BUFFER` note says "every
in-library site that multiplies a dispatcher result names the other operand
(``asm.py:919/922/1391``, ``carrier.py:1401/7126``, ``fresnel.py:216`` all
spell it ``_fft2(...) * H``)"; `test_wave5_e_fft_elision.py` repeats the six.

**Measured** (AST walk over all 236 modules, both the inline and the
held-under-a-name spellings): **10 sites**, not six.  The four not listed are
`asm.py:1147` (`_fft2_nd(...) * H[None, :, :]` -- a basic-slice VIEW, so also
not elidable) and `rs.py:936/939/942` (`E_fft * H` where
`E_fft = _fft2(E_padded)`).  All ten are safe, so the CLAIM holds; the
enumeration does not, and the three `rs.py` sites are the more fragile shape
because the non-owning view is held under a name and the next edit to that line
has no `_fft2(` in front of it.

**Requested edit**: add the four sites, and point the note at
`tests/unit/test_verify_wave5_e.py::
test_no_in_library_fft_product_spells_an_elidable_operand`, which checks the
property structurally on every build instead of restating a list.  (That id
also closes a real hole: the byte-identity arm can only fire on a build whose
NumPy shows the asymmetry, and the mutation `e1_unnamed_right_operand` --
rewriting `asm.py:919` to an unnamed right operand -- is GREEN on Windows.)

---

### D7 (P3, documentation) -- "where the bound is inert the two fields are BIT-IDENTICAL" is not what an inert cell reads

`WAVE5_E_LEFTOVERS_REPORT.md` sec. 3.4 and the `_TRIP` docstring in
`test_wave5_e_c8_default_order.py` justify the gap below the bar with "where
the bound is inert the two fields are BIT-IDENTICAL, so the inert reading is
exactly 1.0 and not 1.0000001".

**Measured** on an inert cell (`alpha = 3.0, cx = 1.40 mm, z = 12 mm,
frbf = 2.0, n = 256`): all four annuli read ratio **exactly 1.0**, and the two
fields are **not** bit-identical -- total power moves by 3e-16 relative on
Windows and 3.3e-14 on WSL.  The conclusion survives (the inert reading really
is exactly 1.0, so the 10x bar has a decade of gap); the stated mechanism does
not.  The reading is exactly 1.0 because the annulus MAXIMUM is an untouched
pixel, which is a weaker and truer statement.

**Requested edit**: "where the bound is inert the annulus maximum is an
untouched pixel, so the inert reading is exactly 1.0 and not 1.0000001
(measured 2026-09-19: an inert cell moved total power by 3e-16 relative while
every annulus still read exactly 1.0, so the fields are not necessarily
bit-identical)".

---

### D8 (P3, derivation) -- E3's products term counts three roundings per leg where there are five

`tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py` derives

> the PRODUCTS: ``fl(s*a)`` then ``fl(re^2)``, ``fl(im^2)``, ``fl(+)`` is at
> most three roundings of half an ULP each, so <= ``3u`` relative per leg and
> ``6u`` for the ratio

and codes `product_rel = 6.0 * u`.  Per complex element the chain is
`fl(s*re)`, `fl(s*im)`, `fl((s*re)^2)`, `fl((s*im)^2)`, `fl(+)` -- **five**
roundings, so `5u` per leg and `10u` for the ratio.

**Measured effect: none.**  With `10u` the bar moves from 27.3-34.0 ULP to
**41.53-48.25 ULP** over the same 16 arms, still above every reading (0-3 ULP)
by 15.4x at the worst arm and still 1.07e+07x below the independent-per-leg
signal.
Every arm keeps its verdict.  It is the arithmetic in the docstring that is
wrong, and `docs/TESTING_STANDARDS.md`'s "every changed bar carries its
derivation" is the rule it sits under.

**Requested edit**: `product_rel = 10.0 * u` and the chain spelled out as five
roundings, with the re-measured bar (41.53-48.25 ULP) and its margins in the
docstring.

---

### D9 (P3, durability) -- the "four sites" guard pin exercises three, and the `_fga_coarse` guard can be deleted invisibly

`test_wave5_e_exit_vertex_dead_rays.py::test_every_fga_site_refuses_an_immersed_exit`
is parametrized over `through_lens`, `coarse`, `vector` and `caustic_zone`,
which the item report reads as "the guard fires at each of the four sites".
Those are four ENTRY PATHS, not four guard call sites: `_fga_coarse` is reached
only from inside `_fga_through_lens` (`fga.py:1564`), whose own guard runs
first.

**Measured** (three mutation arms, identical on both builds):

| mutation | E5 file |
|---|---|
| `_require_non_immersed_exit` deleted from `_fga_coarse` | **29 passed** |
| deleted from `_fga_through_lens` instead | 1 failed, 28 passed |
| deleted from BOTH | 2 failed, 27 passed |

So the `coarse` arm is satisfied by EITHER guard and the `_fga_coarse` site is
not independently pinned at all: it can be deleted and the whole suite stays
green.  The guard itself is still worth having -- `_fga_coarse` is a
module-level function a future caller could enter directly -- but the pin does
not say so.

**Requested edit**: add one arm that calls `_fga._fga_coarse` directly (the
file already calls `_fga._caustic_zone` directly, so the precedent is there),
or state in the test's docstring that the `coarse` parametrization exercises
`_fga_through_lens`'s guard and that `_fga_coarse`'s is defence-in-depth for a
direct caller.

---

## 9. Runs

Every invocation carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on the command line,
`PYTHONPATH` pinned to the tree under test and pytest
`--capture=sys -p no:randomly`; every tail was grepped for
`passed|failed|error|skipped|no tests ran`.  Wall clocks carry contention --
this box ran the two mutation matrices, the gbd suite and the sweep
concurrently -- and are reported, never asserted.

| selection | Windows py3.14.6 / numpy 2.4.4 | WSL py3.12.3 / numpy 2.4.6 |
|---|---|---|
| the three item-E files together | **44 passed, 1 skipped** (26.4 s) | -- |
| the three + my `test_verify_wave5_e.py` | -- | **50 passed** (114.8 s) |
| `test_verify_wave5_e.py` alone (NEW, 5 ids) | **5 passed** (61.4 s); slowest id 25.6 s | (in the 50 above) |
| `test_audit2609_a4_verify_maslov_asymptotic.py`, `test_v4_16_0_agent_d_validity_ranges.py`, `test_verify_b14_known_reds.py`, `test_wave5_gbd_dense_mem_budget.py`, `test_audit2609_a5_verify_fft_buffer_threads.py` | **75 passed** (273.0 s) | **75 passed** (359.4 s) |
| `test_audit2609_b12_fga_reference_plane.py` + `test_verify_b12_fga_reference_plane.py` | **18 passed** (58.6 s) | -- |
| `tests/unit/test_gbd*.py` (the brief's glob -- one file) + `test_verify_b14_known_reds.py` + `test_wave5_gbd_dense_mem_budget.py` | **36 passed** (501.8 s) | -- |
| all twelve `*gbd*.py` files + `test_verify_b14_known_reds.py` (the item report's wider selection) | **165 passed, 0 failed** (3820.9 s -- under the contention above) | -- |
| census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget.py` (30 files) | **1394 passed, 11 skipped, 0 failed** (841.8 s) | -- |
| the mutation matrix, 14 mutations + identity | see sec. 7 | see sec. 7 |
| E3's 16-arm `OPENBLAS_CORETYPE` x threads ladder | 8 arms | 8 arms |
| `scripts/record_history_fingerprints.py --check` | **OK, every history document matches its module** | -- |
| `ruff check lumenairy/ tests/ validation/probe_verify_wave5_e/ validation/probe_wave5_e/ validation/probe_fft_elision/ scripts/` (WSL) | -- | **All checks passed!** |
| `git status --porcelain lumenairy/` | **empty** | **empty** |
| `.test_durations` | valid JSON, **16 282 entries** (+5, spliced by `--store-durations`), largest new id **25.64 s** | -- |

PRE-tree runs, same pinning: `96cb2096` (my own archive) for E4/D2's five
selections, `ede07f30` (my own archive) for E5's PRE arm and for the FGA
byte-identity pair, both builds.

---

## 10. Ship recommendation

**SHIP, with two defects to fix before the merge and one to file.**

Item E's six leftovers are, on re-measurement, done and done honestly.  The
three claims that carry the most weight -- E5's freeze (44/44 reached rows
byte-identical PRE vs POST, 36/44 unfrozen cells -> 0, drift to 1.3e+300 -> 0,
on my own optics, both builds), E3's re-derived bar (every one of the published
figures reproduced across 16 arms, plus a second fixture of my own), and E4's
two corrections (all five D2 arms and all eight D3 figures reproduced to the
digit) -- are exactly what the report says they are.  E2's stimulus reproduced
to every printed digit of its suppression table on both builds, and its premise
gate does what it claims: stimulus dead alone SKIPS with the control alive
(4 skipped), stimulus dead AND control weakened HARD-FAILS (4 failed), the
bound globally disabled HARD-FAILS (4 failed).  E1's decision reproduced even
though its timings did not, and its conclusion is the same under my larger
numbers.  The round-1 fail-before -- 7 of the 18 WP-B12 pins -- reproduced
exactly.

The library edits are safe: `git status --porcelain lumenairy/` is empty in my
tree, the reached rows are byte-identical PRE to POST, and three air-terminated
FGA fixtures x four sites hash identically archive-to-archive with
`LUMENAIRY_MEM_BUDGET_MB` pinned, on both builds.

**Before the merge:**

1. **D3** -- one test asserts a copy of a derivation, and a mutation proves it:
   the O-3 tolerance can be rewritten to drop the image leg entirely and all 29
   ids stay green.  The fix is the four-line bisection my
   `test_the_immersed_exit_guards_boundary_is_the_guards_own_derivation`
   already carries; it runs in 0.01 s.
2. **D1** -- the guard's documented two-sidedness ("a caller who registers a
   real air index is NOT refused") holds only at a zero-length leg and is false
   by 63x at the fixture's own image distance.  The behaviour is right; the
   sentence is not, and it is the sentence a caller reads.  It appears in three
   places.

Both are edits to prose and to one test, not to the repair.

**File, do not fix in this item: D2** -- the JAX analytic backend reports no
vignetting at any aperture.  It is pre-existing, it is out of item E's scope,
the builder recorded it and I confirmed and characterised it (0 dead of 201 at
every rung against NumPy's 174, values agreeing to 1.08e-19 m, so it is the
mask alone), and it is now pinned as a known defect with the exact edit in
hand.

D4-D9 are documentation, test-logic and coverage corrections; none changes a
verdict and none blocks.  D9 is worth doing at the same time as D3, since both
are one arm each in the same file.

---

## 11. What I could not measure

1. **Whether E1(a)'s published percentages are reproducible at all.**  Mine are
   larger than the report's on every Windows cell (+23.7..+32.7 % against
   +18.6..+22.1 %) and larger on the WSL 2048 cell (+16.2 % against +13.0 %).
   Both sets are wall clocks taken on a box running the rest of a verification;
   neither is a per-build fact.  Separating the shim's own frame cost from the
   copy's (my `nocopy_shim` arm) narrows it but does not settle it -- a clean
   box, or `perf`-level instrumentation, would.  The DECISION is insensitive to
   the difference in either direction.

2. **Whether the D2 fix for `_adrt_jax` actually traces.**  The requested edit
   in D2 is written from the code and from the fact that `_adrt_step` already
   computes the identical predicate for both namespaces, but the in-line
   comment at `differential.py:897` claims the dead computation "would trip a
   tracer->ndarray conversion".  I did not apply the edit (the brief says not
   to fix it), so I did not measure whether that claim is still true under
   `vmap(jacfwd(...))`.  The fallback path is named in the defect.

3. **Whether the analytic backend's dead-ray OPD gap is reachable.**  POST, the
   20 analytic cells' frozen rows sit 8.6e-04..1.1e-03 m from
   `at_exit_vertex`'s, including on a FLAT last surface where the projection
   short-circuits.  I established that it is pre-existing and upstream of the
   projection, not that any consumer reads it -- all four `fga.py` consumers
   zero dead beamlets, and the analytic backend is only selected through
   `_pick_ray_transfer`.

4. **CI's runner mix.**  Everything here is two builds on one box.  E2's
   stimulus in particular is a chaotic function of the ray grid, and the
   EPYC/py3.10/numpy 2.2.6 shards have not run it; the premise gate is what
   protects that arm, and the mutation matrix shows the gate discriminates
   correctly (skip when only the stimulus is gone, hard-fail when the control
   is dead too).

5. **The 768^2 F1 cells at the shipped default.**  Like the builder, I
   re-measured 256^2 and 512^2 and the 768^2 order-10 control (51.548,
   bit-reproducing), and did not re-run the published 301x / 3164x 768^2 cells.

6. **A full CI matrix on the merge.**  The gates run here are the
   library-touching selections, the two touched files, the gbd suite and the
   census/walker/dispatcher-pin/public-API/doc-consistency sweep.  A green
   un-masked main-CI matrix on the merge remains the authority
   (`docs/TESTING_STANDARDS.md`, process rules).
