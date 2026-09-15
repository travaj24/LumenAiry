# WP-B4 (Wave 4) -- carrier chain: Collins / ABCD-Fresnel transport with a freely chosen (Bluestein) output pitch, behind `transport='collins'`

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`, HEAD 81d5b586
(= release 5.46.0, Wave 3 closed).  Three other Wave-4 engineers are working concurrently: WP-B1 on `lumenairy/elements/lenses_maslov.py`
+ `lumenairy/propagators/asymptotic*.py`, WP-B2 on `lumenairy/elements/_lens_real.py` + `lumenairy/elements/lens_config.py`, WP-B3 on
`lumenairy/propagators/hf.py`, `hfpi.py`, `rs.py`, `mft.py`.  Never touch those files.  You CONSUME `mft.py`'s `_bluestein_centred_2d` and
`angular_spectrum_propagate_mft`; if you need a change there, specify it under "requested changes outside my ownership".

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 (error prefix) and sec. 7 (signs), the comment rule in
`CONTRIBUTING.md` ("Modules with a history document": `lumenairy/propagators/carrier.py` has `docs/history/carrier.md`; a code change MUST
re-record it in the same change with `python scripts/record_history_fingerprints.py lumenairy/propagators/carrier.py --reason "..."`).
Source comments describe what the code does NOW and why; no version narrative (the history-lint ratchet fails on it).

Then, in this order: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A6_REPORT.md` section 6.1 (the DESIGN and the
acceptance GATE you are implementing, reproduced below) and its C1-C5 sections; `VERIFY_WP-A6.md`; `WP-A24_REPORT.md` (the d6 decentre
calibration and the exact-vs-paraxial leg routing); `WP-A25_REPORT.md` (the replica regime of the readout, `replica_fill`, and why the
readout's period is coupled to the standoff today -- the coupling this transport removes); the audit's CARRIER partition
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/CARRIER.md`, "Alternative algorithms" 1-2) and sec. 15.9 of the main report.
Kelly, *Appl. Opt.* 53, 2861 (2014) gives the sampling conditions the guard must be written against.

## The design (WP-A6 section 6.1; implement it)

Collins (1970, JOSA 60, 1168): for an ABCD system
`E_out(x) = (i/(lambda B)) integral E_in(u) exp(-i k (A u^2 - 2 u x + D x^2)/(2B)) du`.  Factor it as chirp x chirp-Z x chirp:

1. `g(u) = E_in(u) exp(-i k A u^2/(2B))` -- one separable screen (`_radial_carrier_phase`'s outer-product build already provides it);
2. a chirp-Z (Bluestein) of `g` onto the CHOSEN output lattice -- exactly what `propagators/mft.py::angular_spectrum_propagate_mft` /
   `_bluestein_centred_2d` perform, with the separable variant carrier.py already ships (`_EXACT_READOUT_SEPARABLE_BLUESTEIN`);
3. `E_out(x) = (i/(lambda B)) exp(-i k D x^2/(2B)) * (2)` -- a second separable screen.

For a quadratic carrier this is EXACTLY the Sziklas-Siegman result with the output pitch chosen freely instead of forced to `m dx`, so
`m -> 0` stops being a singularity: the near-focus apparatus (`_near_focus_needs_bridge`, `_propagate_carrier_focus_crossing`,
`_axis_bridge`, `_default_focus_standoff`, `_small_extent_focus_standoff_f`, the replica guard's standoff coupling, and C1's contracted
co-moving grid) is not needed on this transport.  Cost: 3 FFTs of `next_fast_len(N + N_out - 1)` instead of 2 of `N`, before the
separable Bluestein's own 2.4-6.7x.

## Deliverable

1. **Ship it behind `transport='sziklas' | 'collins'`** on `propagate_traced_carrier_chain` (and `_multi`), default `'sziklas'` --
   nothing existing moves, and prove that with `np.array_equal` on the WP-A6 / A24 / A25 fixtures.  Validate the vocabulary with the
   sec. 2 prefix.  The sign conventions follow CONVENTIONS sec. 7; state the convention of `B`'s sign for a converging leg explicitly.
2. **The acceptance gate, in order** (all measured, all in the report):
   (a) an analytic Gaussian-ABCD oracle (closed-form Gaussian beam through the same ABCD matrix; the repo has ABCD Gaussian oracles in
   `tests/unit/test_audit2609_a6_verify_carrier.py` -- write your own) at NA 0.03-0.45 and grid extents 1.5-10 w: `'collins'` must be no
   worse than `'sziklas'` in every cell and materially better in the cells the small-extent branch exists for;
   (b) the C1 mismatch matrix of `WP-A6_REPORT.md`: `'collins'` should read peak ratio 1.0000 at every R/R0, because the output pitch no
   longer depends on the carrier;
   (c) a two-group chain against the brute-force ASM + `apply_real_lens_traced` arm the audit built
   (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/CARRIER/p5_chain.py`: power 1.000067, r2m 0.44 % -- read its method, do not
   import it), requiring agreement at least as close;
   (d) `propagate_traced_carrier_chain_multi` K = 1 versus K = 2 exactness on the new transport.
   Also re-run the P2 design battery (`tests/unit/test_niche_p2_design_battery.py`) with `transport='collins'` injected and report its
   through-focus metrics against the analytic Gaussian: the fixture that WP-A25 found scoring a replica is the natural demonstration
   that a freely chosen output pitch needs no replica handling.
3. **The guard**: a sampling guard derived from Kelly (2014) for the chirp-Z stage, with a derived tolerance and a stated fail-before,
   not a geometric margin.
4. **Pin it** in `tests/unit/test_audit2609_b4_collins_transport.py`: the gate's four parts as derived two-sided envelopes (S5: oracle,
   floor, defect scale), the byte-identity of the default, the guard, and the K-congruence.  No wall-clock assertions; report timings.
5. **Do NOT remove** the near-focus apparatus in this package even though `'collins'` does not need it: the default still does.  Say in
   the report what a later default flip would retire, with the measurements that would justify it.
6. **Re-record** `docs/history/carrier.md`; report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B4_REPORT.md` and
   `WP-B4_CHANGELOG.md` (5.47.0 release text, `### Added -- ...` style of `fixes/WP-A6_CHANGELOG.md`, sec. 15.9 named,
   `lumenairy/propagators/carrier.py:N` citations on non-trivial lines, a Migration note saying the default did not move and how to opt in).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a6_carrier.py`, `test_audit2609_a6_verify_carrier.py`, `test_audit2609_a24_decentre_calibration.py`,
`test_audit2609_a25_carrier_focus_readout.py`, `test_niche_d6_exact_tilted_leg.py` (slow, ~3 min), `test_niche_d1_tilted_carrier.py`,
`test_niche_d2_chain_multi.py`, `test_niche_p2_design_battery.py`, `pytest tests/unit -k carrier`, `python validation/run_all.py`
(the carrier-related files; `ls validation/`), `ruff check`, `python scripts/record_history_fingerprints.py --check`,
`tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (three other engineers share
  this box).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator
  commits with an explicit file list.
* Own only: `lumenairy/propagators/carrier.py`, `lumenairy/propagators/carrier_field.py` (if the separable screens need it), their
  `docs/history/` documents, the new b4 test file, your two report files.  Anything else: "requested changes outside my ownership" with the
  exact edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
