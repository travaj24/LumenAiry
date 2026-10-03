# BUILD -- the 5.50.0 deprecation removals (2026-10-03)

Branch `feat/5.50-removals`, cut from the integration commit `5ea82b44`.
Builder: claude-opus-5-5.  Scope: execute the removals that 5.46.0 deprecated
with `version_removed='5.48'`, which 5.48.0 slipped once to 5.50 through
`_deprecation.REMOVAL_SCHEDULE = {'5.48': '5.50'}`.  Without this change a
`__version__ = "5.50.0"` fold cannot pass `check_removal_schedule()`.

## 1. What was removed

| item | deprecated | horizon (original -> slipped) | removed from |
|---|---|---|---|
| `gbd_asm_gouy_phase` | 5.46 (audit S5) | 5.48 -> 5.50 | `lumenairy/propagators/gbd.py`, `lumenairy/propagators/__init__.py` (import + `__all__`), `lumenairy/__init__.py` (import + `__all__`) |
| `gbd_field_to_asm` | 5.46 (audit S5) | 5.48 -> 5.50 | same three files |
| `asm_field_to_gbd` | 5.46 (audit S5) | 5.48 -> 5.50 | same three files |
| `CarrierField` attribute assignment (warn-and-allow shim) | 5.46 (audit C5) | 5.48 -> 5.50 | `lumenairy/propagators/carrier_field.py`: the class is `@dataclass(frozen=True)`; the `__setattr__` override, the `_built` gate, `_CARRIER_FIELD_FROZEN_SINCE`, `_CARRIER_FIELD_FROZEN_IN` and the two imports only the shim used (`warnings`, `caller_stacklevel`) are deleted; `__hash__ = None` is explicit so the class stays unhashable as it was while mutable |
| `REMOVAL_SCHEDULE['5.48']` | -- | -- | `lumenairy/_deprecation.py`: registry is `{}` with a tombstone; `NEXT_REMOVAL_VERSION` `'5.50'` -> `'5.52'` |

`gbd.py` also lost its now-unused `warn_deprecated_alias` import.

### 1.1 The misnamed third alias -- `match_global_phase` is NOT removed

The brief (and the 5.48.0 CHANGELOG, Migration Guide and `_deprecation.py`
comment) listed the three GBD aliases as `gbd_field_to_asm`,
`asm_field_to_gbd`, `match_global_phase`.  The source says otherwise:

* the 5.46.0 CHANGELOG heading is "Deprecated -- `gbd_asm_gouy_phase`,
  `gbd_field_to_asm`, `asm_field_to_gbd` (audit S5)", and its migration line
  says "use `match_global_phase`, which is unchanged";
* the three `version_removed='5.48'` call sites in `gbd.py` were exactly those
  three functions; `match_global_phase` has no warning and no horizon, and its
  own `.. deprecated::` text is absent;
* the test pins (`test_v5_21_gbd_asm_interop.py`,
  `test_audit2609_a4_maslov_gbd.py`, `test_audit2609_a4_verify_maslov_asymptotic.py`)
  all name `gbd_asm_gouy_phase` as the third deprecated function.

So `match_global_phase` is the documented REPLACEMENT and stays public and
unchanged.  The misnaming is corrected in the `_deprecation.py` tombstone, the
Migration Guide's 5.48.0 paragraph (inline correction note), the new CHANGELOG
entry, and `docs/history/lumenairy._deprecation.md`.

## 2. The internal `match_global_phase` call (gbd.py:754, `converge_gbd_sampling`)

Decision: **kept, unchanged** (the function is public, never deprecated, and
the call is load-bearing).  Measured with a scratch probe that replaces it by
the identity (Windows py3.14, 64x64 Gaussian, w = 40 um, dx = 4 um,
lambda = 1 um, z = 200 um, overlaps 1.0 / 1.5 / 2.0):

| reference | with `match_global_phase` | identity in its place |
|---|---|---|
| ASM oracle (default) | 2.7462e-02 / 5.9629e-02 / 1.0108e-01 | 2.7462e-02 / 5.9629e-02 / 1.0108e-01 |
| supplied, ASM x exp(1j*1.0) | 2.7462e-02 / 5.9629e-02 / 1.0108e-01 | 9.4980e-01 / 9.3987e-01 / 9.2817e-01 |

Residual global phase GBD vs ASM with no fit: 3.02e-07 rad.  Against the ASM
oracle the call is near-identity (post-S5), but a supplied `reference=` from
another solver carries an arbitrary absolute phase and without the call a
perfect decomposition scores ~0.95.  The comment and docstring at that site
still described the pre-S5 width-dependent Gouy "convention"; both were
reworded to state the measured reason above.

### 2.1 Byte-identity of public outputs, before vs after

Probe (`converge_gbd_sampling` on two fixtures -- a centred Gaussian and an
offset tilted Gaussian -- every overlap's error as a float hex, plus a SHA-256
of `match_global_phase` on a seeded random pair) and a `CarrierField` probe
(`re_reference` A->B and B->A envelopes and provenance, `aggregate`,
`full_field`, `with_provenance` identity, `dataclasses.replace`, pickle round
trip, field list, repr), each run on the HEAD tree before any edit and on the
final tree:

| probe | Windows py3.14 | WSL py3.12 |
|---|---|---|
| GBD (3 keys, 6 error floats + 1 hash) | identical | identical |
| CarrierField (13 keys) | identical | identical |

(The two builds differ from EACH OTHER in the last ulps of the GBD errors,
e.g. `0x1.c1f0f9e6cc3afp-6` vs `0x1.c1f0f9e6cc399p-6` -- different FFT/BLAS;
the comparison that matters is before vs after within one build.)

## 3. `_deprecation.py` and the simulated fold

`REMOVAL_SCHEDULE = {}` (entry deleted, as invariant 2 requires for an executed
removal) and `NEXT_REMOVAL_VERSION = '5.52'`: invariant 1 requires a horizon
after the running version, and 5.52 is one two-minor cycle ahead (the cadence
of 5.46 -> 5.48 -> 5.50), covering a 5.51 release and every 5.50.x / 5.51.x
patch without another bump.  `API_TRANSITION_VERSION` stays bound to it (two
pins assert the binding; both pass).  No live deprecation states 5.52; the
package's remaining `version_removed=` call sites state `'6.0'`.

Simulated fold (scratch, not committed), `check_removal_schedule()` with
`lumenairy.__version__` patched:

| `__version__` | HEAD registry (`'5.50'`, `{'5.48': '5.50'}`) | this branch |
|---|---|---|
| 5.49.0 (as shipped) | [] | [] |
| 5.50.0 | 4 violations (NEXT_REMOVAL_VERSION, REMOVAL_SCHEDULE['5.48'], resolve('5.48'), resolve('5.50')) | [] |
| 5.50.1 / 5.51.0 / 5.51.9 | -- | [] |
| 5.52.0 | -- | 1 violation (NEXT_REMOVAL_VERSION) -- the intended refusal |

`resolve_removal_version('5.48')` and `('5.27')` both return `'5.52'` (the
backstop), so no deleted entry can resurrect a past-horizon banner.

The deprecation test files under the simulated fold: see section 5.

## 4. Test changes (decisions, not deletions)

| file | was | now |
|---|---|---|
| `test_v5_21_gbd_asm_interop.py` | `test_converters_are_deprecated_no_ops[3]`: warn + no-op | `test_converters_are_removed[3 names x 3 modules]`: absent as attribute, absent from `__all__`, `from ... import` raises `ImportError`; premise: `match_global_phase` still public |
| `test_audit2609_a4_maslov_gbd.py` | `..._is_a_deprecated_no_op` | `test_s5_gouy_compensator_api_is_removed[3]`: `ImportError` |
| `test_audit2609_a4_verify_maslov_asymptotic.py` | warned no-op that still validates | `..._is_removed_and_its_horizon_is_retired`: names gone AND no `version_removed='5.48'` / `warn_deprecated_alias(` left in `gbd.py` |
| `test_niche_audit_w4_input_kind.py` | 70 guard sites, 65 'field' | 68 sites, 63 'field' (two rows deleted with the two guarded functions; docstring rollout history extended) |
| `test_carrier_field.py` | `..._is_deprecated`: warns per attribute, still applies | `test_mutating_a_built_carrier_field_raises`: `FrozenInstanceError` for all five fields, field unchanged, unhashable; two-sided with the migration routes (`with_provenance`, `dataclasses.replace` warn nothing, `replace` re-runs the shape and wavelength checks, `np.add(..., out=)` bit-identical); constants gone |
| `test_audit2609_a6_carrier.py` | `..._is_announced` | `..._raises`; siblings-frozen check kept |
| `test_audit2609_a6_verify_carrier.py` | one warning per assignment, horizon resolves forward | same name (`..._deprecation_cycle_is_complete`), RESTATED: deepcopy / pickle / replace / with_provenance / `np.add` silent; deepcopy and pickle copies are frozen values; every assignment raises, nothing changes, no warning; constants and `_built` gone |
| `test_niche_audit_w3_ui_deprecation.py` | registry-entry invariant loop (vacuous on an empty registry) | RESTATED: loop kept, plus `'5.48'` gone from the registry and from every call site, plus the invariant run on two SIMULATED entries -- the retired `{'5.48': NEXT}` must fail the call-site clause, a not-yet-shipped key the first clause |
| `test_public_api.py` | `'_FROZEN_IN'` forward-version context; docstring example | context dropped with the constant; docstring example updated |
| `test_audit2609_a21_doc_identifiers.py` + `scripts/check_doc_identifiers.py` | curated ceiling 51 (list at 50) | three "removed API, documented as removed" entries added; ceiling 53, reason at the constant |

Unchanged and still passing: `test_niche_audit_w5_shim_removals.py` (its
registry pin holds on `{}`), `test_niche_audit_w4_p5_return_contract.py` (the
`API_TRANSITION_VERSION == NEXT_REMOVAL_VERSION` binding),
`test_v5_21_lens_accuracy_extensions.py` (uses `match_global_phase`), the
"rescheduled from v5.27" banner test (reads `NEXT_REMOVAL_VERSION`, now 5.52).

`.test_durations`: the renamed ids were re-keyed in place with their old
measured values (17147 -> 17149 entries; six input-kind ids for the two
deleted guard rows dropped).

## 5. Documentation and bookkeeping

* CHANGELOG `## [Unreleased]`: `### Removed -- the 5.46 deprecations ...` and
  `### Changed -- the deprecation horizon moves from 5.50 to 5.52 ...` at the
  top.
* Migration Guide: a `## 5.50.0` section (both removals, the way forward with
  code, the horizon); the 5.48.0 paragraph corrected inline; "Versions
  covered" extended to 5.50.
* History: removal records appended to `docs/history/lumenairy.propagators.gbd.md`,
  `docs/history/carrier_field.md`, `docs/history/lumenairy._deprecation.md`;
  fingerprints re-recorded for those three modules
  (`record_history_fingerprints.py --check` exits 0).  The tombstone
  narrative was first written into `gbd.py` / `carrier_field.py` and the
  a17 history lint rejected it (gbd.py 0 -> 1 narrative lines); it now lives in
  the history documents and the source says what is true now.  The same
  forward-version gate in `test_public_api.py` would have rejected a `v5.50`
  token in those two files on the 5.49.0 tree, so they name no release.
* Not touched: `validation/probe_verify_b14/probe_v5_two_caller.py` (an
  archived WP-B14 probe whose `carrier_field.setattr_deprecation` case now
  raises instead of warning; no test runs that case) and the audit records
  under `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/` (historical).

## 6. Gates and measurements

Targeted files (both builds): the eleven pinning files above,
`test_audit2609_a17_history_relocation.py`, `test_audit2609_a17_history_lint.py`,
the three `test_audit2609_a21_*`, `test_pipeline.py`,
`test_verify_b14_known_reds.py`; WSL also `test_audit2609_a15a_durations_staleness.py`.

| gate | Windows py3.14 | WSL py3.12 |
|---|---|---|
| import path asserted under this tree | yes (`c:/tmp/lum_removals/...`, 5.49.0) | yes (`/mnt/c/tmp/lum_removals/...`, 5.49.0) |
| targeted pytest, `-n 2` | **1537 passed, 44 skipped, 0 failed** (19 files, 1199 s) | **1540 passed, 44 skipped, 1 failed** (20 files incl. a15a, 721 s) |
| `test_audit2609_a15a_durations_staleness.py` | 4 passed (run separately, 84 s) | in the batch above, passed |
| ruff `lumenairy tests scripts` | All checks passed | All checks passed |
| mypy (project config) | Success: no issues found in 33 source files | -- |
| `record_history_fingerprints.py --check` | exit 0 | -- |
| `scripts/check_doc_identifiers.py` | 684 / 684 resolve, 0 unresolved | -- |

All 44 skips on both builds are the same three sites in
`test_niche_audit_w5_shim_removals.py` (lines 402 / 408 x42 / 416): frozen
SHA-256 digests that are host-specific by construction and run only with
`LUMENAIRY_W5_DIGEST_HOST=1` on the capturing host.  No xfail.

The one WSL failure is `test_public_api.py::test_installed_metadata_version_matches_source_version`:
`importlib.metadata.version('lumenairy')` reads **5.11.0** because the WSL
venv carries `/home/travaj/lumvenv/lib/python3.12/site-packages/lumenairy-5.11.0.dist-info`
while the tree under test is imported through `PYTHONPATH`.  It is an
environment fact independent of this tree (the same test passes on Windows,
whose editable install reports 5.49.0); it was not rerun, and the shared venv
was not modified.

Simulated fold, pytest: a scratch plugin (`-p fold550`, not committed) sets
`lumenairy.__version__ = "5.50.0"` in every process; a one-line probe test
confirmed the patched value and `check_removal_schedule() == []` inside the
xdist workers.  Under it, `test_niche_audit_w3_ui_deprecation.py`,
`test_niche_audit_w5_shim_removals.py`, `test_niche_audit_w4_p5_return_contract.py`,
`test_v5_21_gbd_asm_interop.py`, `test_audit2609_a4_maslov_gbd.py`,
`test_carrier_field.py`, `test_audit2609_a6_carrier.py`,
`test_audit2609_a6_verify_carrier.py` and `test_public_api.py` gave **394
passed, 44 skipped, 1 failed** on Windows.  The failure is the same metadata
test, now reading installed 5.49.0 against the patched 5.50.0 -- the mismatch
the fold itself removes when the maintainer bumps `__version__` and
reinstalls; every registry and forward-version pin passed at 5.50.0.
