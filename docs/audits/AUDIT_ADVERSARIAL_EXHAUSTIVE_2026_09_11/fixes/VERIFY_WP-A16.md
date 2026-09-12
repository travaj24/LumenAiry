# VERIFY-A16 — independent re-verification of WP-A16

Configuration objects for the `apply_real_lens` family, plus the lens-side
architecture and import-time items.  Commits under test: `c7c9ebbb`
(`feat(lens): WP-A16 …`) and `65bea863` (`feat(lens): WP-A16 follow-up …`),
re-measured on the working tree at HEAD `65bea863` (branch `audit-fixes-2026-09`,
no git writes).

Everything below is a measurement made by this agent, on this box (Windows 11,
CPython 3.14, NumPy 2.x, numba present, **CuPy absent**, jax present,
`OPENBLAS_NUM_THREADS=1` on every invocation, box shared with other agents).
The WP's own fixture (WP-A15a's covering array) was deliberately **not** reused
for the identity work: a second fixture was built here.

---

## 1. Verdicts

| # | deliverable | verdict | evidence |
|---|---|---|---|
| 1 | **V6 / §14-13 — the config objects** (bit-identity of `config=` / triple / keyword, all 7 entry points) | **VERIFIED-WITH-NOTES** | 30 arms on a NEW fixture, `np.array_equal`, **0 failures**; 1 defect found and fixed (§4.1) |
| 2 | The precedence rule and its two refusals | **VERIFIED** | §3.2 — 14 probes, message text checked clause by clause |
| 3 | The `dataclasses.fields()` / signature gate | **VERIFIED** | §3.3 — independently re-derived, and the shipped gate driven with a corrupted signature view from outside |
| 4 | The three documented naming defects (`ray_subsample`, `min_area_ratio`, `input_carrier`) | **VERIFIED** | §3.4 — the wrong spelling cannot get through, in either direction |
| 5 | `sag_dtype` strictness — "coherent contract?" | **NOT COHERENT as shipped → fixed** | §4.2 — the per-call keyword accepted `float16`/`complex128`/`int32` and returned the float64 field byte for byte, with no warning |
| 6 | The re-dispatch design (frames, recursion, cost, `stacklevel`) | **VERIFIED-WITH-NOTES** | §3.5 — terminates, +1 frame exactly; warning **attribution** moves into the library (open item O1) |
| 7 | Addendum 8 — the five optional-dependency sites | **VERIFIED** | §3.6 — including a *dynamic* test of the `cp`-alias coupling, which the shipped test only inspects structurally |
| 8 | Addendum 9 — the three lens knobs | **VERIFIED** | §3.7 — registry 18, apply/restore/raise, the 2²⁰ asymmetry re-measured |
| 9 | Addendum 10 — the module cycles (deferred) | **VERIFIED-WITH-NOTES** | §3.8 — **3 cycles now, not 4**; the docs page was stale and is fixed |
| 10 | Addendum 11 — the four exports | **VERIFIED** | §3.9 |
| 11 | Addendum 12 — narrowed except + the two F541 | **VERIFIED** | §3.10 — F541: 2 on the pre-change blob, 0 now, by ruff itself |
| 12 | Addendum 14 — lazy `backend.scipy` + the `_airy()` accessor | **VERIFIED (claim now an understatement)** | §3.11 — the deferral is worth **+360 ms**, not the +71 ms the report could measure at its own commit |
| 13 | §8.1 — `mypy --strict` on the four package roots | **VERIFIED** | §3.12 — 0 errors, all four, together and separately |
| 14 | §8.2 — the `lens_config` vocabulary cache clearer | **VERIFIED** | §3.13 |
| 15 | The A16 test files read against `TESTING_STANDARDS` | **VERIFIED-WITH-NOTES** | §5 — one test did not test its stated property (§4.3), fixed |

**Defects found and fixed by this pass: 3** (§4).  **Open items: 4** (§7).
No regression of any kind was found in the shipped behaviour.

---

## 2. What I ran

| command | result | duration |
|---|---|---|
| `pytest` the three A16 files (as shipped, before my changes) | **138 passed** | 12.2 s |
| `pytest tests/unit -k real_lens` (before my changes) | **158 passed, 3 skipped**, 14936 deselected | 239.9 s |
| `pytest` 13 collateral files (covering array, `test_public_api`, the three `__all__`/PEP-562 walkers, the cache-registry pin, the 2-D-scalar-field dispatcher pin, both a15b files, `niche_audit_e`, `audit_except_budget`, `niche_k1_kmah_caustic`, `lens_chunked_sag`) | **215 passed** | 104.4 s |
| `pytest` 19 files **after** my changes (the four A16 files + the 13 above + `test_v4_14_0_dispatcher_pin_apply_lens` + `test_v5_21_lens_accuracy_extensions`) | **441 passed**, 2 deselected | 249.2 s |
| `pytest tests/unit -k real_lens` (after my changes) | **160 passed, 3 skipped**, 14962 deselected | 257.3 s |
| `pytest tests/unit/test_audit2609_a16_verify_config_and_arch.py` (new) | **27 passed** | 9.1 s |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py` | **697 passed** | 20.7 s |
| `python validation/run_all.py test_lenses` | **PASS** (1/1) | 25.2 s |
| `mypy --strict --follow-imports silent --ignore-missing-imports` × 4 package roots | **Success: no issues found in 4 source files** | |
| `ruff check lumenairy --no-cache`; `ruff check tests/unit/test_audit2609_a16_*.py` | **All checks passed** (both) | |
| `scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** | |
| 4 standalone probe scripts (bit-identity, precedence/protocol, architecture, docs/cycles) | 30 + 71 + 51 + 1 checks, all green after the fixes | |

Plus `pytest` of the four A16 files + the a15a covering array re-run after
another agent's concurrent `carrier.py` change landed: **211 passed**, 14.1 s.

Pre-existing failures found: **one**, not WP-A16's, and closed by its own owner
during this pass — see §6.

---

## 3. Per deliverable

### 3.1 Bit-identity on an independent fixture (main deliverable)

The WP's evidence stands on ONE geometry: WP-A15a's curved-rear AC254-ish
**cemented doublet**, N = 64, square grid, complex128, on-axis, 632.8 nm.  I
built a different one and re-asked the question on five axes it holds fixed:

* a curved-rear **singlet** (2 surfaces, N-SF11), not a doublet;
* **λ = 1064 nm**;
* **N = 56** (and N = 40 in the committed test) — not a power of two;
* an **anamorphic grid**, `dy = 1.15 dx`;
* a **complex64** input, and an **off-centre** diverging source at −80 mm;
* a **decentred + tilted rear surface** (`decenter=(25, −18) µm`,
  `tilt=(8, 3)·10⁻⁴ rad`) — asymmetric, so a coordinate change cannot undo it.

For every entry point, three spellings (`**settings`, `config=`,
`geometry=/numerics=/resources=`) plus `LensConfig()` against no config, plus a
non-vacuity check that the settings **move** the field:

```
=== c128  complex128  dx=1.857143e-04 dy=2.135714e-04 (dy/dx=1.1500) ===
  ok  apply_real_lens/displaced        |E|sum=3.788184672e+02  moved=True
  ok  apply_real_lens/remap+f32sag     |E|sum=3.789654661e+02  moved=True
  ok  apply_real_lens_traced/newton    REFUSED identically (dx != dy)
  ok  apply_real_lens_traced/caustic   |E|sum=2.286231901e+02  moved=True
  ok  apply_real_lens_traced/origin    REFUSED identically (dx != dy)
  ok  apply_real_lens_maslov           |E|sum=3.086069056e+02  moved=True
  ok  apply_real_lens_gbd              |E|sum=1.542336151e+02  moved=True
  ok  apply_real_lens_fga              |E|sum=4.652136524e+04  moved=True
  ok  multibranch                      |E|sum=2.286231901e+02  moved=True
  ok  prepare_real_lens_traced screen  |S|sum=1.457000000e+03  moved=True  empty_ok=True
=== c64   complex64   (same grid) ===            9 arms, all ok
=== dec   complex128, decentred+tilted rear ===  9 arms, all ok
FAILS: 0
```

`max |diff| = 0` **exactly** on every successful arm (`np.array_equal`, not a
tolerance — a re-dispatch has no numerical content, so any difference is a
defect).  The output **dtype** was compared too, not just the values: the
complex64 arms return complex64 from `apply_real_lens` / `maslov` and
complex128 from `gbd` / `fga` / `multibranch`, identically on all three
spellings.

**The half the WP did not test: identical REFUSAL.**  The existing bit-identity
file compares successful calls only.  A resolver that silently dropped a setting
would make the configured call *succeed* where the keyword call refuses — which
is the audit's failure class, and is invisible to a comparison of two returned
fields.  Measured on four genuine library refusals (`dx != dy` on the traced
engine; `origin` under `amplitude_model='screen'`; `surface_model='displaced'`
with a non-ASM propagator; an analytic Jacobian on a decentred surface): the
keyword call, `config=` and the triple raise the **same exception type with the
same message**, byte for byte, in every case.  Now pinned:
`test_an_illegal_combination_refuses_identically_through_every_spelling`.

### 3.2 The precedence rule

All measured at a real entry point, message text checked clause by clause.

| probe | result |
|---|---|
| field at its default next to a differing keyword | keyword wins, byte-identical to the keyword-only call ✓ (documented) |
| both set and disagreeing | `ValueError`, prefix `apply_real_lens:`, names the setting, **both** values, and "must agree" ✓ |
| set field the entry point lacks | `ValueError` naming `apply_real_lens_traced, prepare_real_lens_traced`, "would silently discard it", and `narrowed_to` ✓ |
| agreeing keyword + field (both `config=` and component) | accepted, byte-identical ✓ |
| **array-valued** conflict (`carrier=` two different wavefronts) | conflict detected, "must agree" ✓ |
| **equal but not identical** arrays (`w` vs `w.copy()`) | *no* spurious conflict, byte-identical ✓ |
| `narrowed_to` | drops exactly the inapplicable requests; idempotent; refuses an unknown entry point ✓ |
| `dataclasses.replace` | keeps the other fields and **re-runs `__post_init__`** (a bad value raises) ✓ |
| pickle / unpickle of all four classes | round trips, and an **unpickled** config resolves byte-identically at the call ✓ |
| `from_kwargs` with a misspelling / a documented exclusion (`levin_tol`) / a contract name (`E_in`) / a config parameter (`geometry`) | all four refused, naming the offender ✓ |
| `to_kwargs(include_defaults=True)` | emits exactly **38** fields; per entry point every emitted keyword exists on that entry point **and equals its signature default** (7/7) ✓ |
| `from_kwargs(**to_kwargs()) == cfg` with all 38 fields set | ✓ |
| frozen (all four) | `FrozenInstanceError` ✓ |
| hashable / `__eq__` | ✓ — **except with an ndarray field**, which was a defect (§4.1) |

### 3.3 The `dataclasses.fields()` gate

Re-derived independently (my own walk over `inspect.signature` × `_GROUPS`):
**every** (entry point, field) pair's dataclass default equals its signature
default — 0 drifts across all 7 entry points.

Falsifiability, driven **from outside** the shipped file so a gate that had gone
vacuous would be caught: `_kwonly` is patched in-process so `apply_real_lens`
appears to default `remap_order` to 5, and the shipped test function is called
directly.

```
  PASS  shipped gate catches a DRIFTED signature default
          LensNumerics.remap_order defaults to 3 but apply_real_lens(remap_order=...) defaults to 5
  PASS  shipped gate catches an UNCLASSIFIED new keyword   (brand_new_knob on gbd)
```

and the un-patched call still passes, so the patch — not an unrelated change —
is what turned it red.  Both are now committed as
`test_the_shipped_default_agreement_gate_would_catch_a_drifted_default` /
`…_classification_gate_would_catch_a_new_keyword`.

One assumption the shipped gate does **not** cover and I added: the re-dispatch
forwards only `KEYWORD_ONLY` parameters, so a parameter added as
`POSITIONAL_OR_KEYWORD` would be silently dropped by the configured path.
Measured: today `E_in` is the only non-keyword-only parameter on all seven, and
the forwarded set is exactly the keyword-only set minus the four config names.
Pinned (`test_the_forwarded_parameter_set_is_every_keyword_only_one`).

**A second unpinned assumption, now pinned.**  `resolve_entry_point_kwargs`
reads the entry point's `locals()` and trusts that **no parameter has been
rebound yet**.  A normalisation above the block — `dy = dx if dy is None else
dy` is the obvious one — would make the resolver read the normalised value,
conclude the caller had requested it, and then refuse a config that set the same
field.  Nothing asserted the ordering.  An AST gate now walks all seven bodies
and fails on any assignment to a signature parameter above the `_wants_config`
block (`test_the_config_block_runs_before_any_parameter_is_rebound`); measured
clean today.

Counts re-derived: **38** config fields, **119** distinct keyword-only names
across the seven entry points, **72** documented exclusions, **9** contract
parameters, **0** unaccounted.

### 3.4 The three naming defects

The question asked was whether a user can silently pass the wrong one through a
config.  Measured — they cannot, in either direction:

```
traced ray_subsample=4        -> requests {'ray_subsample': 4}
  spelled for multibranch     -> {}                      (correctly nothing)
  passed to multibranch       -> ValueError: numerics.ray_subsample=4 is not a setting
                                 apply_real_lens_traced_multibranch accepts (it applies to:
                                 apply_real_lens_traced, prepare_real_lens_traced)
multibranch ray_subsample=4   -> requests {'caustic_ray_subsample': 4}
  traced OPL field stays      -> 8   (the default; NOT overwritten)
  spelled for traced          -> {'caustic_ray_subsample': 4}
  spelled for multibranch     -> {'ray_subsample': 4}
traced caustic_min_area_ratio -> multibranch spelling {'min_area_ratio': 0.0001}
multibranch input_carrier     -> ValueError from from_kwargs ('is not a lens config field of …')
geometry.carrier=0.25         -> ValueError at the multibranch call site
```

A config carrying **both** `ray_subsample=4` and `caustic_ray_subsample=3`
narrows to `{'ray_subsample': 3}` for multibranch and
`{'ray_subsample': 4, 'caustic_ray_subsample': 3}` for the traced engine — the
rename is applied per entry point, not globally.  Passing that config
un-narrowed to multibranch refuses.  **Verified.**  (One residual sharp edge:
`to_kwargs(entry_point=…)` drops the inapplicable half *silently* by design, so
`**cfg.to_kwargs(entry_point='…multibranch')` hands multibranch `ray_subsample=3`
where the config also said `ray_subsample=4`.  That is the method's documented
contract and the call-site refusal is the safety net — recorded as open item O4,
severity P3.)

### 3.5 The re-dispatch design

* **Recursion terminates** for an all-default `config=`, for a one-field
  component, and for all three components at once (the three shapes that take
  the branch).  The merged mapping has the four config names removed, so the
  second entry cannot take the branch again.
* **Exactly one extra frame**, measured by counting `apply_real_lens` frames
  from inside the call through the `progress` sink: plain = 1, configured = 2.
* **Cost when nothing is configured**: four `is not None` tests.  `lens_config`
  itself costs **3.08 ms** cumulative at `import lumenairy` (`-X importtime`).
* **`warnings.warn(stacklevel=…)` attribution — measured, and it is worse than
  "one frame too shallow" reads.**  Same warning, same text, both paths:

  | path | attributed file:line |
  |---|---|
  | keyword | `v_a16_precedence.py:483` (the caller's own line) |
  | configured | `_lens_real.py:5515` (the re-dispatch statement) |

  So a configured caller is pointed at a **lumenairy source line**, not at their
  own code — `python -W error::UserWarning` and any filter keyed on the module
  behave differently for the two spellings.  The WP recorded this as a residual
  and declined to fix it (~40 `warn` sites); I agree with the decision but the
  report understates the symptom.  Open item **O1**.  The safe half — that the
  same warnings, with the same categories and the same messages, fire on both
  paths — is now pinned
  (`test_the_configured_path_emits_the_same_warnings`); the attribution itself
  is deliberately *not* pinned, so a future fix does not turn a test red.

### 3.6 Addendum 8 — the five optional-dependency sites

* AST census over the five modules: **0** own `find_spec('cupy')` /
  `find_spec('numba')` probes, **0** own `import cupy` / `import numba` inside
  `_ensure_cupy_loaded` / `_is_cupy_array` / `_load_numba`.
* **The `cp`-alias coupling, exercised rather than inspected.**  The shipped
  test checks the coupling by looking for the string `_ensure_cupy_loaded` in
  the source.  CuPy is absent here, so the True branch is never taken — I faked
  it (`CUPY_AVAILABLE=True`, a stub `ensure_cupy`, `is_cupy_array → True`) and
  measured that `_is_cupy_array(x) is True` leaves the module-level `cp` bound
  to the stub, on all three consumers, with `cp` and the probes restored
  afterwards.  That is what a CUDA box would check.  Committed as
  `test_a_true_cupy_answer_really_binds_the_module_cp`.
* `_NUMBA_AVAILABLE` is a module attribute that monkeypatch reaches on all
  three consumers (`_load_numba()` returns False), and equals the shared
  constant after restore.
* `fga` keeps its deliberate `ImportError` naming `apply_real_lens_fga` and
  carrying `pip install lumenairy[numba]`.
* Collateral: `_lens_thin` no longer routes CuPy through `lenses._lenses_module`
  (that attribute is gone) — the WP's §5 item 3 request has landed.

### 3.7 Addendum 9 — the three lens knobs

Registry size at plain `import lumenairy`: **18**, all three present with docs
> 40 chars.  For each of `lens_sag_dtype`, `lens_parallel_amp`,
`pointwise_cos_grid_cache_budget`: `override()` applies, restores on normal
exit, **and restores when the block raises**; `setter(getter())` is an exact
no-op.

The asymmetry that forced the private pair, re-measured:

```
set_pointwise_cos_grid_cache_budget(64)  ->  get_...() = 67108864   (x 1048576 = 2**20)
setter(getter()) on the PUBLIC pair      ->  67108864 -> 70368744177664
private byte pair restored exactly       ->  0
```

i.e. registering the public pair would have multiplied the budget by 2²⁰ **on
every restore** — the WP's justification is real and is now re-measured by its
own test each run.  `set_low_memory(True)` flips `lens_parallel_amp` to False
and `_knobs.restore()` puts all four macro knobs back.

### 3.8 Addendum 10 — the cycles

Re-measured with my own module-level-only AST walk over the 11 lens modules:

```
_lens_real   <-> lenses
_lens_traced <-> lenses
lenses       <-> lenses_maslov
(3 cycles)
lens_config module-scope imports from lumenairy: none (leaf)
```

**Three, not four.**  The audit's `_lens_thin <-> lenses` is gone — WP-A22's
follow-up (`d96c4cd4`) actioned WP-A16's §5 item 3.  `docs/lens_configuration.md`
still said "four", tabled the dead `_lens_thin:39` back-edge, and closed with
"**One of the four is already a two-line fix and should be taken first**" for a
fix that had already landed.  Fixed (§4.4).  `lens_config` being a leaf is now
pinned (`test_lens_config_stays_a_leaf`).

### 3.9 Addendum 11 — exports

`LensGeometry` / `LensNumerics` / `LensResources` / `LensConfig` are the **same
objects** through `lumenairy`, `lumenairy.elements` and
`lumenairy.elements.lens_config`, and all four are in both `__all__`s.
`test_public_api.py`, `test_v4_16_0_walker_all_symmetry.py`,
`test_v5_2_walker_shell_vs_canonical.py` and
`test_v5_2_walker_pep562_forwarding.py` are green (the last of those was red in
the WP's report because of an in-flight `fft_infra` diff; WP-A21 closed it).

### 3.10 Addendum 12 — the narrowed except and the two F541

* `ruff check --isolated --select F541` on the **pre-change committed blob**
  (`git show c7c9ebbb~1:lumenairy/elements/_lens_real.py`): **2 errors**,
  at the two `f"…split_prescription_at_mirrors…"` continuation lines.  On the
  working tree: **0**.  `ruff check lumenairy` overall: clean.
* AST census: `_lens_imap.py` carries **0** broad `except` clauses;
  `lens_config.py` **0**.  `_lens_real.py` still has 2 (lines 882/910 —
  `_glass_key_value` / `_sag_callable_fingerprint`), which WP-A15a justifies and
  which are outside this item.
* Both of the WP's §5 requests have landed: the
  `"lumenairy/elements/_lens_real.py" = ["F541"]` per-file ignore is gone from
  `pyproject.toml`, and `test_audit_except_budget.py`'s `_lens_imap` entry is
  deleted (census 51 → 50).

### 3.11 Addendum 14 — lazy `backend.scipy` and `_airy()`

**Structural facts, fresh interpreter, after `import lumenairy`:**

```
lumenairy.backend.scipy  False
scipy.linalg             False
scipy.special            False
scipy.fft                False
numba                    False
cupy                     False
```

and the gate is **not vacuous** — touching `lumenairy.backend.scipy` in the same
interpreter flips `scipy.linalg` to `True`.  Every access path still works and
is the same object: `la.backend.scipy`, `from lumenairy.backend import scipy`,
`importlib.import_module('lumenairy.backend.scipy')`, `'scipy' in dir(la.backend)`,
`'scipy' in la.backend.__all__`, the `globals()` cache, a real call
(`la.backend.scipy.jv(0, [0, 1]) = [1, 0.76519769]`), `AttributeError` (never
`ImportError`) for an unknown name, and `getattr(..., default)` / `hasattr`.

**Import time — interleaved same-build A/B, fresh interpreters, arms
alternating, medians of 7 pairs (two independent runs, hours apart):**

| arm | run A median (ms) | run B median (ms) | Δ vs bare |
|---|---|---|---|
| `import lumenairy` (bare, this tree) | **278.1** | **294.2** | — |
| + `lumenairy.backend.scipy` | 667.9 | 657.6 | **+363 … +390** |
| + `from scipy.special import airy` | 583.4 | 591.7 | **+298 … +305** |
| + **both** (a same-build stand-in for pre-WP-A16) | 647.3 | 654.4 | **+360 … +369** |

A 9-pair bare-vs-both run gave 309.7 → 717.2 ms, **Δ +407.5 ms**.  The two
deferrals overlap (both pull the `scipy._lib._array_api` → `array_api_compat`
→ `numpy.f2py` → `charset_normalizer` prefix), which is why "both" ≈ either one
alone.

**The WP's claim is now a large understatement, for a reason its own §2.6
predicted.**  It measured +70.8 ms and wrote: "the `airy` move cannot save that
prefix while `fft_infra` is eager".  WP-A22 (`7d03d799`) then made `scipy.fft`
lazy, and the prefix is now genuinely unpaid: `import lumenairy` is **≈ 294 ms**
instead of the **≈ 654 ms** it would be with these two imports eager.  Named
cumulative rows on this tree (`-X importtime`): `lumenairy` 278.8 ms,
`lumenairy.elements` 115.6 ms, `lumenairy.propagators.fft_infra` 52.7 ms,
`lumenairy.backend` **2.0 ms** (A15b measured 567.3 ms before the forward),
`lumenairy.elements.lens_config` 3.1 ms; `scipy.fft` / `scipy.special` /
`scipy.linalg` do not appear at all.  **Nothing is asserted as a time** — the
committed gate is the `sys.modules` membership, per TESTING_STANDARDS S1.
Request **O2**: the CHANGELOG text's `745.0 → 673.8 ms / −71.2 ms` is correct
for WP-A16's own commit but wrong for the release tree, and should be restated.

**`_airy()`**: exactly equal to `scipy.special.airy` on 23 points spanning
`[-6, 3]` (all four returned arrays, `np.array_equal`); caches the handle;
**is not `airye`** (checked, because that is the plausible mis-binding).
`ludwig_fold` verified against a closed-form oracle — see §4.3.

### 3.12 §8.1 — mypy

```
mypy --strict --follow-imports silent --ignore-missing-imports \
     lumenairy/__init__.py lumenairy/backend/__init__.py \
     lumenairy/elements/__init__.py lumenairy/elements/lens_config.py
Success: no issues found in 4 source files
```

0 errors together and 0 separately, before **and after** my changes to
`lens_config.py`.  All four are in `[tool.mypy] files` in `pyproject.toml`.

### 3.13 §8.2 — the vocabulary cache clearer

`_VOCAB_CACHE` holds 4 entries after one validation; `la.clear_asm_caches()`
drains it to 0; the next validation refills it to 4.  Registry enrollment is
green (`test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`).

---

## 4. Defects found, and what I did about them

### 4.1 `__eq__` raised `ValueError` on an array-valued field  (P2 — fixed)

`LensGeometry.carrier` is documented to accept "a wavefront ndarray", and
`lens_config._same` carries an array arm *because* "several of these settings
legitimately carry one (`carrier` may be a wavefront, `beam_centre` a
2-vector)".  But the **dataclass-generated `__eq__`** compares the two field
TUPLES, and a tuple comparison calls `bool()` on each element's `==`:

```
FAIL-BEFORE (the generated __eq__, restored in-process):
  ValueError: The truth value of an array with more than one element is
  ambiguous. Use a.any() or a.all()
```

So, for exactly the values the module was written to support, these all raised
rather than answering:

* `LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg` — **the round trip the
  class docstring advertises**, and the one the docs page shows;
* `cfg.narrowed_to(ep) == other` — the idempotence the shipped round-trip test
  asserts (it just never uses an array);
* `pickle.loads(pickle.dumps(cfg)) == cfg`;
* any `==` between two configs in a study table.

The calls themselves were never affected (`resolve_entry_point_kwargs` compares
field by field through `_same`), so this is a value-semantics defect, not a
numerical one — but the docs page claims the objects are "comparable" and the
module claims the round trip closes.

**Fix** (`lumenairy/elements/lens_config.py`): the three dataclasses declare
`eq=False` and bind `__eq__ = _dataclass_eq` / `__hash__ = _dataclass_hash`.
`_dataclass_eq` compares field by field through the same `_same` the resolver
uses (so an ndarray compares with `np.array_equal`) and returns `NotImplemented`
for a foreign type, exactly like the generated one.  `__hash__` is spelled out
rather than left to the `_set_new_attribute` subtlety, and is the same
field-tuple hash as before — so `a == b ⇒ hash(a) == hash(b)` still holds: the
two definitions differ only on arrays, and an array-valued field makes the
instance unhashable either way.

**Verified**: `test_value_equality_survives_an_array_valued_field` (fails with
the generated `__eq__` restored in-process, transcript above) and its
counter-pin `test_hash_is_still_by_value_and_agrees_with_equality` (two equal
instances hash equal, `len({a, b}) == 1`, and an ndarray field is `TypeError`
to hash — stated so the limit is documented rather than discovered).
`mypy --strict` and `ruff` still clean; all 138 shipped A16 ids still pass.

### 4.2 `sag_dtype`: the per-call keyword silently discarded the setting  (P2 — fixed)

The brief asked whether "refusing `np.float16` in the config but accepting it
(as float64) as a keyword" is a coherent contract.  **Decision: it is not, and
the incoherent half is the keyword path, not the config.**  Measured before the
fix:

| spelling | `np.float16` | `np.complex128` | `np.int32` |
|---|---|---|---|
| `LensResources(sag_dtype=…)` | `ValueError` | `ValueError` | `ValueError` |
| `set_lens_sag_dtype(…)` (the process knob) | `ValueError` | `ValueError` | `ValueError` |
| `apply_real_lens(…, sag_dtype=…)` | **accepted** | **accepted** | **accepted** |

and what "accepted" meant: `apply_real_lens(..., sag_dtype=np.float16)` returned
the float64 field **byte for byte** (`max |diff| = 0.000e+00`) with **no
warning**.  A caller who asked for half-precision geometry got the default and
no way to find out — the discarded-setting class this entire work package exists
to close, sitting inside it.  Two spellings of one setting disagreeing about
what is legal is also exactly how a config object becomes a trap: a study that
works through `LensResources` refuses, the same study written with keywords does
not.

**Fix** (`lumenairy/elements/_lens_real.py:_resolve_sag_real`): the shared
resolver now refuses anything but `None` / float32 / float64, with the
CONVENTIONS §2 prefix and the caller's own name (`apply_real_lens:` /
`prepare_real_lens:`), and says what to pass instead.  One implementation, so
`apply_real_lens_traced` and `prepare_real_lens_traced` inherit it through the
analytic leg (measured).  The documented domain in the `sag_dtype` docstring —
`{None, np.float32, np.float64}` — is now the enforced one.

**Verified**: `test_all_three_spellings_of_sag_dtype_refuse_the_same_set`
(parametrised over `float16`, `complex128`, `int32`, `'int32'`; each asserts the
field, the knob, the per-call keyword and `prepare_real_lens`).  FAIL-BEFORE
confirmed by restoring the pre-fix resolver in-process — all four report
`Failed: DID NOT RAISE ValueError`.  Counter-pin
`test_the_legal_sag_dtypes_are_untouched_by_that_refusal`: `None`, `np.float64`
and the string `'float64'` stay **byte-identical** to the call that omits the
keyword, `'f4'` still resolves to float32, and `np.float32` still moves the
field (so the test cannot pass by refusing everything).
`test_lens_chunked_sag.py` (which owns this knob) and the `-k real_lens` slice
are green.

**Behaviour change, for the changelog** (I do not own `WP-A16_CHANGELOG.md`;
text for the orchestrator):

> ### Fixed — `sag_dtype`: one rule for all three spellings
> `apply_real_lens` / `prepare_real_lens` (and the traced entry points through
> them) resolved an unrecognised `sag_dtype` to float64 in silence, while
> `set_lens_sag_dtype` and the new `LensResources.sag_dtype` refused it.
> `sag_dtype=np.float16` returned the float64 field byte for byte with no
> warning — a setting discarded without a word.  The shared resolver now
> refuses anything but `None` / `np.float64` / `np.float32`, with the
> `CONVENTIONS.md` §2 prefix.  `None`, `np.float64`, `'float64'` and `'f4'` are
> unchanged and byte-identical; only previously-ignored values now raise.
> Test: `tests/unit/test_audit2609_a16_verify_config_and_arch.py::test_all_three_spellings_of_sag_dtype_refuse_the_same_set`.

### 4.3 `test_ludwig_fold_…` called the function with its arguments scrambled  (P2 — fixed)

`tests/unit/test_audit2609_a16_lens_arch.py:401` (a file I own) read:

```python
out = _mb.ludwig_fold(a, a, s, s, k)      # a = 1+0j, s = 1e-3, k = 9.93e6
```

but the signature is `ludwig_fold(k, S_plus, S_minus, A_plus, A_minus)` — every
other caller in the repo (the library at `_lens_traced_multibranch.py:1013`,
`test_niche_k1_kmah_caustic.py`, `test_v5_21_lens_accuracy_extensions.py`, the
audit's own `repro/TR-SIBLINGS/ludwig_check.py`) uses that order.  The call as
written therefore evaluated `k=1+0j, S_plus=1+0j, S_minus=1e-3, A_plus=1e-3,
A_minus=9929180.3`, and the test's stated property — "exactly ON the fold
(`S_plus == S_minus`) the plain branch sum diverges and the Ludwig form must
stay finite" — was not what it measured (`S_plus != S_minus`, and the "plain
branch sum" was never computed).  Its `isfinite` assertion passed on a
`8.96e6 - 7.93e5j` value.

Worse, the property as stated is not testable the way the docstring implies:
calling it correctly with `S_plus == S_minus` and `A_plus == A_minus` gives
`≈ 6.2e298` — finite **only because of the `1e-300` guard in `g1`'s
denominator**, because equal amplitudes at a fold are unphysical (the branch
that has touched the caustic carries the KMAH `−π/2`, so `A₊ = −i A₋` and
`A₊ + iA₋ = 0`).

**Fix**: replaced with two tests carrying derived bars and a closed-form oracle.

*`test_ludwig_fold_stays_bounded_where_the_branch_sum_diverges`* — approach the
fold with the physical pair (`A₋ = A₀ ρ^(−1/4)`, `A₊ = −i A₋`), for which the
limit is analytic:

```
u(0) = sqrt(2π) k^(1/6) e^{iπ/4} e^{ikS₀} (−i√2 A₀) Ai(0),   Ai(0) = 0.3550280538878172 (DLMF 9.2.3)
     -> |u(0)| = 18.450996939883858   for k = 2π/632.8 nm, S₀ = 1 mm, A₀ = 1
```

| half-separation δ | `|ludwig_fold|` | `|plain branch sum|` | rel. to the closed form |
|---|---|---|---|
| 1e-12 m | 18.4591 | 132.18 | **4.41e-4** |
| 1e-10 m | 18.6264 | 61.41 | 9.51e-3 |
| 1e-09 m | 19.2645 | 42.21 | 4.41e-2 |

Bars: `rel < 1e-2` (23× the measured 4.41e-4, and the residual is the genuine
`O(k^{2/3}ρ)` curvature of `Ai`, not noise) and `plain/|u| > 4` (measured 7.16,
and the ratio grows without bound as δ→0, so the gap to a Ludwig form that
tracked the divergent sum is unbounded above).

*`test_ludwig_fold_reduces_to_the_plain_branch_sum_far_from_the_fold`* — the
other asymptote, where the plain sum is an oracle the test computes itself:

| separation | rel. distance to the plain sum | bar |
|---|---|---|
| 20 λ | **1.11e-3** | 5e-3 |
| 500 λ | **4.42e-5** | 2e-4 |

falling as the expected `O(1/(k ΔS))`.  Together the two bracket the Airy call
from both sides: `airye` differs by `exp(2/3 z^{3/2})` and `Bi` by a growing
exponential, so either mis-binding lands at `O(1)` or larger.

**FAIL-BEFORE**: feeding the *old* argument rotation back into the new test
gives `AssertionError: ludwig_fold is 4.350e+00 away from the closed-form
on-fold limit` — i.e. the new test detects precisely what the old call could not.

### 4.4 `docs/lens_configuration.md` was stale in four places  (P3 — fixed)

1. "**four** genuine module↔module import cycles", the `_lens_thin:39` back-edge
   row, and "One of the four is already a two-line fix and **should be taken
   first**" — all describe a state that WP-A22's follow-up (`d96c4cd4`) already
   changed.  Re-measured to 3 and rewritten, with the `_lens_thin` quarter
   recorded as landed.
2. "hashable-by-value" with no mention that an ndarray field is unhashable, and
   no mention that `==` now compares through `_same`.  Corrected.
3. "nor one of the **four** contract names (`prescription`, `wavelength`, `dx`,
   `N`)" — `CONTRACT_PARAMETERS` has **nine**.  Corrected.
4. "One place the config is *stricter* than the keyword: `sag_dtype`" — no
   longer true after §4.2.  Rewritten as the single rule, with the reason the
   fix went into the shared resolver rather than the config.

Also added: a pointer to this report and to the new verify test file, and the
statement that `lens_config`'s leaf status is now pinned.

---

## 5. The A16 tests read against `docs/TESTING_STANDARDS.md`

| standard | finding |
|---|---|
| **S1** no wall-clock / speedup assertions | ✓ — the import-time item is asserted as `sys.modules` membership, never as a time; I added none either |
| **S2** no `pytest.skip` on resource preconditions | ✓ — `test_load_numba_returns_true_and_binds_the_handles_when_available` asserts the *other half* of the contract on a numba-free box instead of skipping, which is the right pattern |
| **S3** no per-build bars | ✓ for the config files (every comparison is exact: `np.array_equal`, set equality, dataclass equality, exception text). The one exception was §4.3, which had **no bar at all** where its docstring implied one; now two derived bars |
| **S4** every numeric bar carries its derivation and measured value | the three A16 files had essentially no numeric bars, so this was vacuously satisfied; my additions carry derivations (§4.3, and the `sag_dtype` counter-pin is exact) |
| **S5** independent oracles with derived bounds | ✓ after §4.3 — the Ludwig tests now use `Ai(0)` from DLMF and the plain branch sum, neither produced by the code under test. The bit-identity files' "oracle" is the keyword call, which is correct here: the claim *is* "these two calls are the same call" |
| non-vacuity | strong throughout — the bit-identity file asserts every row **moves** the field, `_NO_MOVE_EXPECTED` is asserted in both directions, the classification walker has an explicit counter-pin, and `test_the_megabyte_byte_asymmetry_…` re-measures its own justification each run. This is above the house average |
| testing the property vs the implementation | mostly the property. Two exceptions, both defensible and both documented in place: `test_cupy_probe_is_the_shared_one_…` greps the source (CuPy is absent, so the True branch is unreachable — I added the dynamic version beside it), and `test_the_mirror_guard_warning_…` greps for two literals (it also fires the warning and checks the interpolation, which is the property) |

---

## 6. Pre-existing failures

**None remain.**  One was live when this pass started and closed during it, by
another agent:
`tests/unit/test_niche_d6_exact_tilted_leg.py::test_decentred_carrier_decentre_penalty_envelope`
— **1 failed** when I first ran it (the 0.969787 the WP's §8.3 bisected to
WP-A6), **1 passed** on the re-run an hour later, with
`lumenairy/propagators/carrier.py` and that test file both `M` in the working
tree and the shipped `decentre_fit_frac` warning now carrying a re-measured
table ("MEASURED 2026-09-12 … 0.00 w -> 0.970 … against an oracle floor of
6.4e-04") and an explicit note on the ordering.  That is WP-A16's §5 items 7 and
8 being closed by their owner (A24) while I worked; I did not touch either file
and did not re-run the WP's nine-commit bisect.  My own A16 set was re-run after
that change landed: **211 passed** (the four A16 files plus the a15a covering
array).

The `-k real_lens` slice (160 passed, 3 skipped) and the 441-id collateral
battery were measured *before* A24's change landed; nothing in them depends on
`carrier.py`'s warning text, and the A16 set is green on both sides of it.

**Closed since the WP report was written**, verified here:

* its §4(a) — the 12 `test_v4_14_0_dispatcher_pin_apply_lens.py` jax-x64
  isolation failures: **35 passed, 0 failed** running that file ALONE
  (WP-A22's "JAX x64 isolation", `7d03d799`);
* its §4(b) — `test_v5_2_walker_pep562_forwarding.py`: green (WP-A21,
  `949edb3b`);
* its §5 item 1 (the `pyproject.toml` F541 per-file ignore): deleted;
* its §5 item 2 (the `_lens_imap` except-budget entry): deleted, census 51 → 50;
* its §5 item 3 (`_lens_thin`'s cycle): landed;
* its §5 items 7 and 8 (the d6 bar and the stale `carrier.py` calibration
  table): closed by A24 during this pass — see above.

---

## 7. Open items for the orchestrator

| id | item | severity |
|---|---|---|
| **O1** | **`warnings.warn` attribution on the configured path.** Measured: the keyword call attributes a warning to the caller's own file:line; the configured call attributes it to `lumenairy/elements/_lens_real.py:5515` (the re-dispatch statement). A user filtering or `-W error`-ing by module sees different behaviour for the two spellings. The WP recorded it as "one frame too shallow", which understates it. Fixing it properly means threading a `stacklevel` offset through ~40 `warn` sites in `_lens_real`/`_lens_traced` (the WP's estimate, ~2 h); a cheaper 80 % is to fix only the handful of warnings a *configuration* can trigger (aperture-vs-grid, undersample, non-collimated). Not fixed here: it is a cosmetic attribution and the change surface is large. | **P3** |
| **O2** | **The WP's CHANGELOG import-time numbers are stale for the release tree.** `745.0 → 673.8 ms (−71.2 ms / −9.6 %)` was true at `c7c9ebbb`; on HEAD, with WP-A22's `scipy.fft` deferral in place, the same two lazy imports are worth **+360 to +407 ms** and `import lumenairy` is **≈ 294 ms** against **≈ 654 ms** with them eager. Table in §3.11. Restate before assembling `CHANGELOG.md`, or the released note undersells the change by 5×. | **P2 (docs)** |
| **O3** | **`_lens_traced`'s `sag_dtype` refusal is prefixed with the inner function's name.** After §4.2, `apply_real_lens_traced(..., sag_dtype=np.float16)` raises `apply_real_lens: sag_dtype=…` (the analytic leg refuses). Accurate and actionable, but a caller who never wrote `apply_real_lens` has to infer the chain. A three-line early validation in `apply_real_lens_traced` / `prepare_real_lens_traced` would name the entry point the user called. Left alone: it duplicates the rule, and duplicated validation is what `lens_config` deliberately avoids. | **P3** |
| **O4** | **`to_kwargs(entry_point=…)` drops silently by design.** A config carrying both `ray_subsample=4` (traced OPL) and `caustic_ray_subsample=3` narrows to `{'ray_subsample': 3}` for multibranch — the caustic value under multibranch's spelling, with the traced-only request dropped. That is the documented contract of the method (it is the explicit narrowing step, and passing the same config to the call refuses), but it is the one place the three-way naming trap can still bite someone who splats `**cfg.to_kwargs(entry_point=…)`. Consider a `strict=True` option on `to_kwargs` that raises instead, for symmetry with the call site. | **P3** |

---

## 8. Files I changed

All five are inside VERIFY-A16's ownership list.  No file with a
`docs/history/<dotted>.md` document was touched — `scripts/record_history_fingerprints.py
--check` reports "OK: every history document matches its module", and
`test_audit2609_a17_history_relocation.py` is 697 passed, so no re-record was
needed.

| file | change |
|---|---|
| `lumenairy/elements/lens_config.py` | `_dataclass_eq` / `_dataclass_hash`; the three dataclasses declare `eq=False` and bind them (§4.1); the stale "stricter than the keyword path" comment on `sag_dtype` corrected (§4.2) |
| `lumenairy/elements/_lens_real.py` | `_resolve_sag_real` refuses an unrecognised dtype, with the §2 prefix and the caller's name (§4.2); the `prepare_real_lens` call site names itself |
| `tests/unit/test_audit2609_a16_lens_arch.py` | the scrambled `ludwig_fold` call replaced by two tests with derived bars and a closed-form oracle (§4.3) |
| `tests/unit/test_audit2609_a16_verify_config_and_arch.py` | **new**, 27 ids, ~9 s — the independent fixture, identical refusal, the `locals()` contract, value equality, the `sag_dtype` contract, gate falsifiability, the dynamic `cp`-alias test, the leaf and module-scope-scipy gates |
| `docs/lens_configuration.md` | four stale statements corrected (§4.4) plus pointers to this report and the new test file |
