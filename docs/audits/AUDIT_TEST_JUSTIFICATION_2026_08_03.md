# Audit — does every test in `tests/unit/` still earn its place? (2026-08-03)

**Scope:** all 446 `test_*.py` files in `tests/unit/` (~10,991 collected tests), audited file-by-file
against `docs/audits/` (116 files), `CHANGELOG.md`, `ROADMAP.md`, and live `lumenairy/` symbols. This
is a **different, deeper** pass than `AUDIT_CI_TEST_TIME_2026_08_03.md` — that audit asked "is this
test too expensive for what it checks?"; this one asks **"does this test trace to a real, still-valid
reason to exist?"** Method: 10 parallel Opus agents, one per ~45-file chunk, read-only static analysis
(no pytest execution) — matching each test's claimed purpose against the evidence, checking for
staleness/supersession, and verifying every referenced symbol still resolves in the current codebase.
**No repository files were modified.**

---

## 0. Bottom line

**The suite is, on the whole, honestly and traceably justified.** Across 445 files and ~10,991 tests,
only **2 individual tests** were found to be genuinely stale (guarding a discredited/reverted
convention with no remaining value), and **zero files or symbols were found testing dead code** —
every one of the many thousands of `lumenairy.*` imports checked across all 10 chunks resolved to a
real, live definition. The codebase's convention of naming test files after and citing a specific
audit finding (by code — F1, R7, C11, P1-05, S2-14, a version number, etc.) works: the large majority
of citations were independently verified against `docs/audits/`, `CHANGELOG.md`, or `ROADMAP.md` and
found to accurately describe what the test checks.

What the audit **did** find, in descending order of how much attention each deserves:

1. **2 genuinely stale tests** (§1) — should be deleted or rewritten.
2. **1 concrete, actionable bug** in test infrastructure itself (§2) — not a staleness issue, a real
   defect that will break the suite for anyone without an optional dependency installed.
3. **~30 confirmed duplicate tests** across ~19 files (§3) — assert the identical claim as another
   test with no distinguishing angle. Safe to delete/merge; several were traced to a specific
   mechanical cause (parallel "agent" backfills during a historical multi-agent audit campaign).
4. **~14 cases of docstring/module-comment drift** (§4) — the test itself is current and correct,
   but its prose describes an earlier, superseded state of the code. Purely a documentation fix.
5. **A large number of untraceable-but-harmless citation labels** (§5) — informal codes ("G08",
   "C2", "Phase 7", "ZX-5", "P1-V5_1_1-1"...) that don't resolve to any document, where the
   *underlying finding* was independently confirmed real via `CHANGELOG.md` or live code. Cosmetic.
6. **A new hygiene category**: 9 "always-pass diagnostic" tests across 6 files that cannot fail by
   construction (§6) — informational only, not defects.
7. **Two direct connections back to this session's earlier Part-E work** (§7), including one file
   that raises the *exact same* keep-vs-delete question this session already had to answer once.
8. **A consolidated list of ~15 items needing an owner's judgment call** (§8) — things the audit
   could not resolve from static evidence alone.

---

## 1. Genuinely stale tests (recommend delete or rewrite)

### 1.1 `test_audit_misc.py` — `TestAuditFixesV4_11_1_EvenAsphParmRoundTrip::test_evenasph_export_then_load_preserves_coeffs` (chunk 1)

Pins the **v4.11.2** EVENASPH parameter mapping (`power = 2 + 2*parm_num`), which was **reversed in
v5.16.1** to `power = 2*parm_num` (live at `lumenairy/io/prescriptions_zemax.py:932`, with an explicit
reversal comment). Three compounding problems: (a) the class docstring still prescribes the
discredited convention; (b) the test's own `pytest.skip(...)` references
`prescriptions.py:580-585`, a file/line that no longer exists (the loader moved to
`io/prescriptions_zemax.py`) — this is the exact "green-mask" pattern flagged elsewhere in this
repo's own audit history as test-quality debt (S5-7); (c) the surviving convention-agnostic claim is
already covered, correctly, by `test_audit_io.py:196 test_evenasph_full_round_trip_preserves_alpha4`.
**Recommend deletion**, or a rewrite to the v5.16.1 convention with the skip removed.

### 1.2 `test_niche_audit_w3_elements.py:186` — `TestDoeLegacyUnitsDeprecationReachability::test_auto_mode_post_rescale_bound_pin_is_SUPERSEDED` (chunk 3)

Self-admitted tombstone: its own docstring states *"With `'auto'` removed there is no post-rescale
bound to test: the mode is rejected before any value is inspected."* Its only assertion
(`pytest.raises(ValueError, match='REMOVED in v5.30')` on `_legacy_units='auto'`) duplicates a
sibling in the same file (line 153) and `test_niche_audit_p2b_infra_contracts.py:533`. **Safe to
delete** (or reduce to a one-line comment pointing at the superseding test).

---

## 2. A concrete, actionable bug (not staleness)

**`test_niche_r4_fga_dual_vectorize.py:95`** (chunk 5) — `_assert_dual_numba_match` does
`assert nb is not None, "numba kernel unexpectedly unavailable"` with **no** `importorskip`/`skipif`
guard anywhere in the file. numba is a genuinely optional extra (`pyproject.toml:121`), and the
file's own module docstring says the kernel path is *"opt-out-safe: numba unavailable ... falls back
to the dual path"* — directly contradicted by the hard assertion. All 10 tests in the file that route
through this helper will **fail outright** (not skip) on a base install without numba. The sibling
`test_niche_k2_carrier_backends.py:63,79` does this correctly with `pytest.importorskip` — same fix
pattern is available in-repo. Cheap, high-value fix.

---

## 3. Confirmed duplicate tests (safe to merge/delete — zero new information)

The single largest cluster, and the most mechanically interesting: **8 confirmed byte-for-byte
backfilled duplicates** in `test_v4_15_1_agent_f.py` (chunk 6), verified via `git log -S` to have
landed in commits `672051c`/`7808107` **alongside** the canonical versions of the same tests in
`test_v4_15_2_agent_d.py` and `test_v4_15_3_agent_d.py` — an artifact of a historical multi-agent
audit campaign (the "agent_a...agent_g" v4.15.x file family) where parallel agents' backfills
overlapped and were never pruned. Full list, by file:

| File | Duplicate(s) | Duplicate of | Chunk |
|---|---|---|---|
| `test_v4_15_1_agent_f.py` | 8 tests (uniform placement ×2, unwrap-gradient ×2, complex-gradient, anamorphic direction cosines, CDF reproducibility, threshold-boundary) | `test_v4_15_2_agent_d.py`, `test_v4_15_3_agent_d.py` | 6 |
| `test_v4_15_3_agent_a.py` | 10 "migration sanity" tests | `test_v4_15_2_agent_c.py` (20 tests, same 10 entry points) | 6 |
| `test_v4_15_3_agent_c.py` | 2 tests (self-admitted, cites its own canonical duplicate) | `test_v4_15_2_agent_e.py` | 6 |
| `test_v4_15_1_agent_g_application.py` | 1 test (`test_fourier_transform_field_matches_3_stage_chain`) | `test_v4_15_2_agent_b.py` | 6 |
| `test_through_focus_bucket_boundary.py` | 1 test, strict subset (2 of 6 metrics) | `test_through_focus_metric_parity.py` | 6 |
| `test_v4_16_2_agent_d.py` | 4 tests (README/requirements/ROADMAP-count checks) | `test_v4_16_2_dispatcher_pin_doc_consistency.py` (more general) | 7 |
| `test_v4_16_1_agent_c.py` | 1 test (cache-clear registry walk) | `test_v4_16_0_agent_d_cache_registry.py` (strictly stronger) | 7 |
| `test_v4_15_4_agent_d.py` | 1 test (OPD-fan backcompat units) | `test_v4_15_5_agent_b.py` | 7 |
| `test_v5_2_physics_fixes.py` | 3 tests (Maslov subdomain-grid-loss claims) — **recommend merge, not delete**: the class also documents a real option-a→option-b historical flip | `test_v5_2_3_mhs_maslov_resampling.py` | 9 |
| `test_v5_4_5_coating_edge_cases.py` | 1 test (`test_no_warning_for_valid_wavelength`), claim-for-claim identical | `test_v5_4_coating_materials.py` | 10 |
| `test_audit_except_budget.py` | 1 test, byte-identical assertion to its own file's predecessor | (same file) | 1 |
| `test_audit_g06_perf.py` | 1 test — same finding as a more robust white-box pair, and the weaker/more fragile of the two (its own name once made it a false-fail source under `pytest-timeout`) | `test_audit_s5_8_perf_noloss.py` | 1 |
| `test_niche_audit_p2b_infra_contracts.py` | 1 test | `test_niche_audit_w3_elements.py` (superset) | 3 |
| `test_niche_audit_w4_p5_return_contract.py` | 2 assertions | `test_niche_audit_w3_propagators.py` (fuller) | 3 |

**Milder overlaps, flagged but not recommended for action** (each has *some* distinguishing angle —
recorded for completeness, not urgency): `test_v5_11_0_rcwa_device_helpers.py` vs
`test_v5_11_0_rcwa_segments.py` (chunk 8); `test_v5_11_1_rcwa_lowerpri.py`'s stabilize tests vs
`test_v5_11_0_rcwa_stack_stabilize.py` (chunk 8); `test_niche_p3_pointwise_obliquity.py` vs
`test_niche_p10_transverse_walk_remap.py` (chunk 5); `test_niche_r9_dx_scaling_fix.py` vs
`test_niche_s10_sibling_patterns.py` (chunk 5); `test_niche_r0_byte_budgeted_cache.py` vs
`test_niche_r1_cosgrid_cache.py` (chunk 5).

---

## 4. Docstring / module-comment drift (test correct, prose stale)

The test itself remains accurate and current; only its explanatory prose describes an earlier state.
Zero action needed on the assertions; a documentation pass would close these.

- `test_audit_v5_24_2_b1_source_conventions.py` — docstring describes the v5.25 *deprecation-warning*
  phase; the tests correctly assert the v5.30 *hard-removal* that superseded it (chunk 2).
- `test_audit_v5_24_2_g05_seams.py` — header bullet describes an abandoned "lockstep" classifier
  approach; the tests correctly guard the *accepted* per-basis-specific resolution (chunk 2).
- `test_fga.py` — in-code CI-membership comment cites `xdist --dist loadfile`, which the workflow no
  longer uses; already independently flagged dead in `AUDIT_CI_TEST_TIME_2026_08_03.md` (chunk 3).
- `test_niche_audit_w3_oracles.py` — module docstring describes "four oracles" but the file (2,915
  lines) also contains an entire unmentioned W3-3b section and W4 wave (~30% of the file) (chunk 3).
- `test_v4_15_3_agent_b.py` — docstring says a helper/sentinel is "preserved for back-compat"; the
  tests correctly assert its v5.30 removal (chunk 6).
- `test_v4_16_3_agent_c.py` — docstring's own test-count taxonomy claims 8 tests; 5 were deliberately
  deleted in commit `155141b7` and the docstring never updated (chunk 7) — the most substantial of
  this category, bordering on §1's severity, but the deletion itself was correct and intentional.
- `test_v4_16_3_agent_b.py` — 4 tests became tautological after a warning mechanism was retired in
  v5.1.0 (asserting "no warning fires" when the emission branch no longer exists at all); correctly
  inverted in spirit, just now vacuous (chunk 7).
- `test_v5_11_0_rcwa_fff_nv_2d.py` — docstring claims `fff_nv` "lands closer than `li`"; the actual
  test's own docstring and a backlog doc both say "neither robustly beats the other" (chunk 8).
- `test_v5_14_5_emt_and_berreman_jax.py`, `test_v5_14_5_viewer_polarization.py` — both filenames/
  docstrings say v5.14.5; `CHANGELOG.md` places the shipped features in v5.14.4 (chunk 8).
- `test_niche_s8_sphere_carrier_reference.py` — two spots still describe a "cos² band-limit taper"
  that niche C9 removed; the shipped-exact-conversion assertions are themselves correct (chunk 5).
- `test_v5_4_1_coating_sellmeier_nan_fix.py` — docstring describes a validity range since narrowed
  by a later audit fix; the NaN-regression pin itself remains valid (chunk 10).
- `test_v5_2_glass_formula3.py` — docstring describes the v5.2 "0 of 24 stubs ingested" ship state,
  since completed; 4 tests now pass vacuously over an empty set but are explicitly intended as
  forward guards that regain teeth if a stub regresses (chunk 9) — lowest severity in this list.
- `test_v5_2_chebyshev_extraction.py` — cites a `ROADMAP.md` line range that now holds unrelated
  content (the referenced item moved) (chunk 9).

---

## 5. Untraceable-but-harmless citation labels

A large, low-severity bucket: every chunk found informal codes that don't resolve to any document,
where the **underlying finding was independently confirmed real** (via `CHANGELOG.md`, live code, or
a differently-named doc). None of these indicate an unjustified test — only a broken/missing
citation trail. Representative sample (not exhaustive — see individual chunk transcripts for full
detail): W6/W9-series group labels with no dedicated `docs/audits/` file (chunks 1, 4); "G08"/"G10"
group labels (chunk 1); FGA "C2" campaign label (chunk 3); PMM "Phase 0/1/3/4/6/7" program labels
(chunk 8); coatings "C10" label (chunk 10); pre-CHANGELOG "v4.7" / "v4.8.1" citations that predate
the earliest CHANGELOG entry (chunks 2, 10); a locally-invented "ZX-5" ID for a real but unnumbered
audit "Nits" bullet (chunk 9); a non-existent "P1-V5_1_1-1" ID for a real, correctly-described gap
(chunk 9); inconsistent "P1-1" vs "P1-I" spelling in `CHANGELOG.md` for the same item (chunk 9); an
invented "Part 3.5" section number, repeated across 3 files (chunk 7); "P2-VAL-1"/"P2-FAC-1"/
"P2-DEP-1" dispatch-brief labels not present in any audit doc (chunk 7); roadmap "item E/F/G"
lettering with no canonical source document (chunk 10).

---

## 6. New hygiene category: always-pass "diagnostic" tests (chunk 7)

9 tests across 6 files (`test_v4_15_3_dispatcher_pin_2d_scalar_field.py`,
`test_v4_15_dispatcher_pin_validate_grid_params.py`, `test_v4_16_0_walker_all_symmetry.py`,
`test_v4_16_0_walker_dy_threading.py`, `test_v4_16_0_walker_sentinel_propagation.py`,
`test_v4_16_0_walker_xp_of_dispatch.py`, `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`,
`test_v4_16_1_agent_d.py`) are explicitly documented as *"not really a test — always passes"*,
existing only to give `pytest -v` a triage inspection point. They cannot fail by construction, so
under a justification lens they guard nothing. Not a defect — a design choice worth being aware of;
arguably better as a `--collect`-time report than a collected test.

---

## 7. Direct connections to this session's earlier Part-E work

1. **`test_niche_c12_physics_fit_selection.py` (chunk 4) independently raises the identical
   keep-vs-delete question this session already resolved once for Part E.** C12 pins a closed-form
   fit-branch predictor that was flipped on in 5.32.1 and then **reverted on measurement** — 9 tests
   across niches D6/D7 regressed, with one case dropping a fidelity metric from 0.98 to 0.67 against
   an exact oracle; `lumenairy/elements/_lens_traced.py:2238` confirms the flag now ships `False`.
   The tests were correctly updated to assert the predictor *stays off*, so nothing is broken — but
   the file's justification has shifted from "pin new physics" to "pin a path measurement rejected,"
   structurally identical to the `wavefront_aware` launch this session deleted outright after the
   same kind of measurement-driven reversal. Unlike Part E, this repo's own C6 fit-guard precedent
   argues for *keeping* a measurement-rejected path opt-in when it acts on a genuinely different
   object than its replacement (documented in `C8_INVERSE_SUPPORT_BOUND_2026_08_01.md §8`) — but no
   equivalent argument is recorded for C12. **This is a live decision the repo owner should make**,
   informed by exactly the precedent this session set with Part E.
2. **Confirmed clean, independently, by two separate chunks:** zero tests anywhere in the 445-file
   sweep guard the reverted `wavefront_aware` / `tilt_aware_rays` / `_carrier_relative_launch`
   mechanism (chunks 4 and 9 both searched explicitly and found nothing) — the Part-E revert left no
   orphaned test debt behind.
3. **A live inconsistency worth flagging alongside C12:** the working tree currently has
   `DECENTRED_FIT_ARBITER = True` (uncommitted, in-flight work on `feat/d121-final-closure`), which
   contradicts `C11_PHYSICAL_DECENTRE_GATE_2026_08_03.md`'s stated ship value of `False` (chunk 5). If
   the arbiter is about to ship default-on, `test_niche_d6_exact_tilted_leg.py` and
   `test_niche_d7_decentred_fit.py`'s branch-selection pins should be re-confirmed against whichever
   value actually ships.

---

## 8. Consolidated list of items needing an owner's judgment call

1. **C12 predictor** (§7.1) — retire like Part E, or keep opt-in like the C6 fit guard? No recorded
   "acts on a different object" argument exists for C12 the way it does for C6.
2. **`test_niche_d7_decentred_fit.py`'s two double-era-pinned fail-before witnesses** (chunk 5) —
   honestly documented (they monkeypatch the library back to a superseded flag state to keep
   witnessing a fixed bug), but now that a pass-after test also exists, are both still worth the
   runtime? Uses raw `monkeypatch.setattr` rather than the repo's own sanctioned era-pin registry
   (`lumenairy/elements/_traced_flags.py`).
3. **`DECENTRED_FIT_ARBITER` ship value** (§7.3) — confirm before it lands whether D6/D7's pins still
   describe the default path.
4. **`test_v4_16_3_agent_c.py` docstring-vs-CHANGELOG disagreement** (chunk 7) — a `DeprecationWarning`
   the docstring calls "live with a v5.0 removal target" was retired as stale per a commit message,
   but `CHANGELOG.md:1598` still lists it as active with no horizon. Update docs, or was retirement
   premature?
5. **Whether the 30 confirmed duplicates in §3 should simply be deleted** — none carry risk; this is
   purely a "does the owner want this done" question, not an open investigation.
6. Several **citation-numbering-only** questions from §5 (does the repo want "G08"/"C2"/"Phase 7"/
   etc. formally mapped to a doc, or left as informal internal shorthand) — lowest priority.
7. **`test_v5_2_physics_fixes.py`'s duplicate cluster** (chunk 9) — merge into
   `test_v5_2_3_mhs_maslov_resampling.py`, or keep as a historical marker of the v5.2.0→v5.2.3
   option-a→option-b flip? The agent recommends the latter be preserved in some form.
8. A handful of **weak/near-vacuous gates** independently flagged by both this audit and the repo's
   own prior audit history (e.g. the 50-wave RMS gate in `test_audit_glass.py:147`, tagged S5-12 as
   test-quality debt, unremediated) — whether to tighten now or continue deferring.

---

## 9. What came back clean (worth stating explicitly)

- **Zero orphaned imports across all 445 files.** Every chunk performed an explicit AST-level (or
  equivalent) resolution of every `lumenairy.*` symbol referenced — many hundreds per chunk — and
  found no test importing or exercising a removed, renamed, or otherwise dead code path.
- **Supersession is handled correctly almost everywhere.** Of the small number of genuine
  implement→measure→revert events in this repo's history (the 408b8c3 coordinate-break revert later
  itself retracted; the N6 eigen-rotated `local_quadrature` window; the HFPI bincount swap; the
  Part-E `wavefront_aware` launch; C12's predictor), only C12 (§7.1) currently lacks a clear
  keep-or-delete resolution — every other reverted approach has either no surviving test, or a test
  correctly rewritten to guard the *accepted* resolution rather than the abandoned one.
- **The audit-citation convention works.** The overwhelming majority of the ~10,991 tests checked
  trace cleanly to a real, specific, still-valid finding — this is a genuinely well-disciplined test
  suite for its size, and the issues found are consistent with organic drift at scale (docstrings not
  updated post-fix, parallel-agent backfills not pruned) rather than systemic neglect.
