<!-- lumenairy-history-doc
module: lumenairy/elements/_lens_traced.py
ast_sha256: 211541f016b0a7525f744b00a0a967523c8826dabea08ec8ad9aeb2c97d51b08
token_sha256: b3ff4bbacfa6bafbef189ad3fa6afede396c8f8516efec54a5352cc1402b9471
pre_relocation_lines: 14898
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-A26: _DECENTRED_FIT_POLY_ORDER re-derived 10 -> 16 against the ray set WP-A1's exact conic intersection produces -- the lowest order that returns the decentred exit slope to the pre-truncation scale on both decentres
re_recorded: 2026-09-13 -- WP-B10: an opt-in disc-orthogonal design basis for the traced ray fits -- fit_basis='chebyshev' (the default, byte-identical) or 'zernike', orthonormal on the ray-fit disc at the same total degree; the samples, D1's weights, the D7 order and the C11 arbiter are unchanged and only the conditioning of the least-squares solve moves
re_recorded: 2026-09-13 -- VERIFY-B10 landing: the inverse-map cache key names the fit basis (parity_tag += str(_fit_basis)); the verifier's fit_basis reachability paragraph
re_recorded: 2026-09-13 -- the grid-versus-aperture bookkeeping moved to the new elements/_lens_kernels.py leaf and is re-exported; _lens_traced reads the leaf, closing its module-level 2-cycle with the lenses facade (WP-B11a item 4)
re_recorded: 2026-09-14 -- WP-B11b item 8: every warning in the two lens bodies asks _lens_kernels.caller_stacklevel for its level, so it names the first frame outside the package whatever wrapper / configuration re-entry reached it; doe.py's zone-plate fill takes T's own dtype.
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-14 -- P1 4.1b: the Newton pool's teardown becomes non-blocking (background reaper + bounded close) and the rebuild rule becomes a ceiling with in-flight accounting, so a broken pool reaches the serial fallback instead of wedging
re_recorded: 2026-09-15 -- P1 4.1b follow-up: the dispatcher's post-claim re-read of the cached pool is removed -- it overrode whatever _get_persistent_worker_pool returned, which broke the H2 executor-spy pins; the microsecond window it closed costs only a bit-identical serial fallback
re_recorded: 2026-09-15 -- merge of verify/wp-b13 into wave5/audit-leftovers: the WP-B13 pool repair and the wave-5 comment edits land in one tree (both sides' re_recorded lines kept)
re_recorded: 2026-09-19 -- WP-B13 follow-ups D2/D3/D4: _shutdown_pool_bounded drops its executor from _ABANDONED_POOLS when the teardown returns (census, not ledger, with the expiry boundary ordered); _get_persistent_worker_pool's idle-worker footprint re-measured in the state the ceiling rule leaves behind; _POOL_INFLIGHT documented as a DISPATCH count
re_recorded: 2026-09-19 -- VERIFY-WP-B13-FOLLOWUPS round 2: VD2 -- _shutdown_pool_bounded's helper removes its _ABANDONED_POOLS entry only when its own caller added it (a caller-set 'added' flag under _ABANDONED_POOLS_LOCK), so a bounded teardown that completes inside its bound can no longer drop an entry _abandon_pool's reaper is still outstanding on (6/6 -> 0/6 by construction, both builds).  The publish-before-lock ordering is unchanged.  VD5 (docstring only) -- _get_persistent_worker_pool no longer calls the never-served worker 'the SURPLUS of a pool wider than the clamp': ProcessPoolExecutor spawns lazily (0 processes after construction, 4 after a 4-chunk dispatch, both builds), so that column prices a worker the labelling step created; the served figure is restated as the measured range over two independent measurements.
re_recorded: 2026-09-20 -- WP-C2 round 2 (VERIFY-WP-C2 D4): this module's exported internally-tracing entry point(s) take the tracer's own sphere_normal= / renormalize= keywords (default None, which stamps nothing) and forward them verbatim to the trace call, so the pre-WP-C2 arithmetic is one keyword away; 742/742 arrays byte-identical archive to archive on both builds
-->

# Version history -- `lumenairy/elements/_lens_traced.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/_lens_traced.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: the measured derivations of the live constants.
`docs/TESTING_STANDARDS.md` S5 requires a numeric bar to carry its oracle, and
this module is mostly that -- the `_RD_HALO_*` 180-call calibration with its
123x separation table, the `_FIT_DISC_OUTSIDE_WEIGHT_REL` sweep and its C2
envelope, the `_DECENTRE_GATE_*` 0-1.0 w table, the niche-C11 arbiter's 42-point
validation, the `_NEWTON_WORKER_*_BYTES` 267.2 B/point commit sweep, the
`REMAP_STATIONARY_PHASE_FIT_GUARD` chain-level reach map, and the degree-6
support-bound table all stayed in full.  The live migration statements, the
fail-before switches and the do-not-do-this hazards stayed too; where one was
written as history ("pre-fix this did X"), the hazard was restated in the
present tense at the source site and the original wording recorded here.


Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment plus a pointer to this file; those are noted per block
below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L86-92 | `<module>`, between the CuPy probe and the numba import | the v5.30 E-L4 tombstone for the deleted numexpr scaffold |
| L207-214 | `_NEWTON_MAX_ITERS` | the 3.5.5 drop to 8 and the 3.5.6 revert -- the cap's release chronology |
| L241-251 | `_RAY_DENSITY_CAUSTIC_MAXMIN`, the det J scan | D1's retraction of an earlier D1 revision that masked the scan |
| L588-594 | `<module>`, the sibling-import block | the v5.30 E-L4 tombstone for the deleted `surface_sag_general` import and alias |
| L802-835 | `_newton_invert_chunk` docstring | three stacked release blocks (v5.29.1 E-H2, v5.32.3 FIX_CI_POOL, v5.33.0 FIX_POOL_REBUILD) each recording what the worker did before that release's key joined the payload |
| L865-870 | `_newton_invert_chunk`, the paraxial-magnification keys | the `pre-3.1.3 callers` / `historical 1.10` framing on a live payload fallback |
| L873-875 | `_newton_invert_chunk`, the Newton cap key | the `Pre-5.29.1 payloads` framing on a live payload fallback |
| L988-994 | `<module>`, the persistent-ProcessPool banner | the pre-3.5.5 per-call pool teardown cost and the 3.5.5 change that replaced it |
| L1190-1198 | `_PERSISTENT_POOL_LOCK` | the v5.30 E-L2 record that the lock used to be lazily created |
| L1200-1205 | `_PERSISTENT_POOL_ATEXIT_REGISTERED` | the v5.30 E-L1 record that the handler used to be registered per pool creation |
| L1233-1243 | `_get_persistent_worker_pool`, the start-method choice | the v4.16.1 M-2 tag and the clause about which CHANGELOG claim the change made true |
| L1453-1458 | the Newton-pool RESOURCE clamp banner | `_invert_newton_parallel` "used to submit n_workers chunks with NO memory accounting" |
| L1773-1787 | `_script_has_main_guard` docstring | "The predicate used to be ..." -- the superseded EXISTS-a-guard proxy and what it classified as guarded |
| L2707-2717 | `_det_normal_equations`, the empty-fit guard | "This branch used to sit below ``B.reshape(...)``" -- where the guard was before the P4 close-out |
| L3117-3127 | `_solve_lstsq_thread_safe` docstring, CONDITIONING | a docstring correcting its OWN earlier assertion that ``A`` is well-conditioned so squaring its condition number is safe |
| L3430-3440 | `_Cheb2DEvaluator.from_state` banner | the v5.33.0 FIX_POOL_REBUILD record that a Newton pool worker used to call `__init__` |
| L3676-3680 | `_analytic_lens_phase` docstring, the thickness budget | a docstring correcting its OWN earlier "under 10 nm on F/10+" rule |
| L4170-4180 | `DECENTRED_FIT_ARBITER`, the default-flip paragraph | which release shipped the arbiter ON, that C11 shipped it OFF, and that "that decision has now been taken" |
| L5255-5259 | the rationalised exact-sphere expression | "the k0*eps*|s| error it used to carry" -- the pre-rationalisation cancellation error as history |
| L5349-5429 | `_REMAP_RESID_EIKONAL_DEGREE` | the whole degree-4 era: the WHY-4 end-to-end table, the 0.103 %-of-input-power on-axis ghost it shipped with, the MECHANISM note that refuted the note above it, the RESOLVED note that corrected a guess in the same block, and the 2026-08-02 "RAISED TO 6" announcement |
| L5627-5631 | `REMAP_STATIONARY_PHASE_FIT_GUARD`, THE DEFECT | the retraction of a hypothesis that was recorded in `_REMAP_RESID_EIKONAL_DEGREE` and has itself now moved |
| L5666-5673 | `REMAP_STATIONARY_PHASE_FIT_GUARD`, THE STRUCTURAL FIX | the dated "2026-08-01: THE FIRST OF THOSE SHIPPED" follow-up appended under a sentence that said neither was attempted |
| L5675-5679 | `REMAP_STATIONARY_PHASE_FIT_GUARD`, the chain-level block | "The two claims above that were element-level inferences are now chain measurements, and ONE OF THEM WAS WRONG" |
| L5934-5956 | the niche-C14 UNIT C banner | "WHAT WAS WRONG.  There were THREE notions ..." -- the three rules, the duplicated convex-hull algebra and the two copies of the half-plane evaluator that the extraction replaced |
| L7139-7145 | `_reverse_prescription` docstring | a docstring correcting its OWN earlier assertion that even-power aspheric coefficients are invariant under the reversal |
| L7257-7276 | `_opl_by_backward_trace` docstring, Validation | a docstring retracting its OWN "~35-40 nm on singlets at N=512" figure and recording that no fixture reproduces it |
| L7403-7410 | the coarse->fine index mapping | the ``ii * N_c / N`` predecessor written as what the code previously did |
| L7845-7873 | `apply_real_lens_traced` docstring, ``tilt_aware_rays`` | "Why the default flipped from True to False in 3.1.3" and "The 3.1.4 default ... restores the ... launch that pre-3.1.2 releases used" |
| L7939-7947 | `apply_real_lens_traced` docstring, ``on_noncollimated`` | "It used to be documented as removing its one-FFT-free cost" and the v5.29.1 E-M3 note that an unrecognised value used to select 'warn' |
| L7992-7997 | `apply_real_lens_traced` docstring, ``newton_max_iters`` | "Honoured ... since v5.29.1 (audit E-H2 -- the pool worker previously hard-coded 12 ...)" |
| L8145-8148 | `apply_real_lens_traced` docstring, ``on_fit_domain_basis`` | "Validated at entry since v5.32.2 (finding V4) ... Before that gate every unrecognised value ... silently selected 'warn'" |
| L8365-8377 | `apply_real_lens_traced` docstring, ``remap_sampling`` | the dated "**The POINT GAIN is gone once niche C6 lands**" follow-up and its "Nothing above is retracted" clause |
| L9002-9010 | the ``newton_amp_mask_rel`` override in the `caustic='wave'` block | "this override used to be SILENT while every other requirement in this block raises" |
| L9467-9477 | the ``sag_chunk_rows`` resolution | two stacked release notes (v5.17.0 then v5.17.1 P2-05) on the same kwarg, the second correcting what the first shipped |
| L10609-10617 | the `min_coarse_samples_per_aperture` apertureless arm | "the floor was documented as enforced ... but the guard was silently skipped for apertureless prescriptions" |
| L11023-11035 | the entrance-grid reshape | a comment correcting an earlier COMMENT at the same site -- "the comment that used to sit at this site ... was wrong twice over" |
| L11137-11149 | the on-axis OPL reference, FIX_TILT_QUADRATIC_OPL | "What was WRONG is that the constant was then dropped" -- the defect as a fix narrative |
| L11704-11713 | the paraxial-magnification stencil | "4.11.2: the indices match ... Pre-4.11.2 the indices were swapped" -- the index-order defect as a release note |
| L11775-11781 | the Newton cap in the pickled worker payload | "``_newton_invert_chunk`` used to hard-code ``_NEWTON_MAX_ITERS``, so ``newton_max_iters`` was inert whenever the process pool engaged" |
| L11792-11795 | `_warn_newton_unconverged` docstring | "Pre-3.5.6 unconverged pixels were silently kept ...; the POOL path stayed silent until v5.29.1" |
| L12256-12259 | the pool path's unconverged report | "the pool used to be silent, so the one regime whose convergence the message's own advice addresses never reported" |
| L12838-12843 | the (2, N, N) coordinate stack | "Pre-fix the stack was constructed twice ... ~4 extra full-grid float64 (~34 GB at N=32768)" |
| L14158-14163 | `apply_real_lens_traced_multi`, the `_FORCED` contract | "Pre-v5.29 they were popped SILENTLY, so a caller asking for e.g. ``preserve_input_phase='remap'`` got ``True`` with no diagnostic" |
| L14787-14800 | `prepare_real_lens_traced` docstring, WHAT A PREPARED LENS FREEZES | "Pre-v5.29.1 the unresolved ``None`` sentinels were stored, so the frozen screen and the per-call analytic amplitude leg silently desynchronised" with its two measured numbers |

---

### L86-92 -- `<module>`, between the CuPy probe and the numba import -- the v5.30 E-L4 tombstone for the deleted numexpr scaffold

*Left in the source:* nothing -- the names it describes do not exist in this module

```text
# v5.30 (audit E-L4): the numexpr scaffold that used to sit here
# (``NUMEXPR_AVAILABLE`` / ``_ne`` / ``_ensure_numexpr_loaded`` /
# ``_NUMEXPR_MIN_SIZE``) was DEAD -- 0 readers in this module and no importer
# anywhere (the public ``NUMEXPR_AVAILABLE`` re-export comes from ``lenses.py``,
# and the live numexpr phase-screen gate lives in ``_lens_real.py``).  Deleted:
# it advertised a fused-expression fast path this module never had.

```

### L207-214 -- `_NEWTON_MAX_ITERS` -- the 3.5.5 drop to 8 and the 3.5.6 revert -- the cap's release chronology

*Left in the source:* the live reason the cap is 12 and not 8, and the per-call override

```text
# Newton iter cap default.  Set to 12 (the historical value).
# 3.5.5 dropped this to 8 based on an audit recommendation, but the
# active-mask early-exit already short-circuits converged pixels -- the
# cap only matters for outlier pixels that genuinely need 9-12 iters.
# Truncating those at 8 silently lost accuracy on cemented multi-element
# / strongly-aberrated systems.  3.5.6 reverts to the safe 12.  Override
# via apply_real_lens_traced(newton_max_iters=N) when profiling shows
# Newton dominates.
```

### L241-251 -- `_RAY_DENSITY_CAUSTIC_MAXMIN`, the det J scan -- D1's retraction of an earlier D1 revision that masked the scan

*Left in the source:* the present-tense reason the scan is unmasked, and the measured lobe that makes masking it a silent wrong answer

```text
# D1 (2026-07-28): the whole-grid det J scan is NOT masked to the beam's own
# support when the ray-fit disc sits off centre.  An earlier D1 revision did
# mask it, on the reading that an off-centre disc leaves the rest of the launch
# domain to polynomial EXTRAPOLATION whose det J is not a property of the
# optics.  That reading was wrong in the only way that matters: the fold the
# scan reported was REAL -- the hard-masked off-centre fit genuinely folded, and
# the same calls returned a spurious lobe at 0.75 of the on-beam peak (see
# ``_FIT_DISC_OUTSIDE_WEIGHT_REL``).  Masking the scan would have converted a
# loud wrong answer into a silent one.  With the fit regularised the fold is
# gone at the source and the unmasked scan is silent on the same cases, so the
# scan stays exactly as it was on every path.
```

### L588-594 -- `<module>`, the sibling-import block -- the v5.30 E-L4 tombstone for the deleted `surface_sag_general` import and alias

*Left in the source:* nothing -- this module gets its sag from the raytrace core via the trace

```text
# v5.30 (audit E-L4): ``surface_sag_general`` and its private
# ``_surface_sag_general`` alias were imported/defined here with ZERO readers in
# this module (grep-verified: the only two occurrences were the import and the
# alias itself) and nothing imports either name FROM this module -- the live
# users are ``_lens_real.py`` and ``elements.py``, which import from
# ``lenses.py`` directly.  Both deleted; this module gets its sag from the
# raytrace core via the trace, not from the analytic helper.
```

### L802-835 -- `_newton_invert_chunk` docstring -- three stacked release blocks (v5.29.1 E-H2, v5.32.3 FIX_CI_POOL, v5.33.0 FIX_POOL_REBUILD) each recording what the worker did before that release's key joined the payload

*Left in the source:* the live payload contract for all three keys, their shared reason and the measured bit-identity numbers, in one paragraph

```text
    v5.29.1 (audit E-H2): the cap used to be the module constant
    ``_NEWTON_MAX_ITERS``, which made ``newton_max_iters`` INERT on the
    pool path (then: >=200k points with ``newton_fit='spline'``; since
    v5.30.1 / v5.32.2 the pool serves EITHER fit, above the two-tier
    200k-cold / 8k-warm size gate) -- the caller's
    cap was honoured by the serial closure only, so the OPL (pool) and the
    ray-density amplitude (always serial) could come from DIFFERENT Newton
    solutions.  The resolved cap now travels in the pickled payload;
    payloads written by older callers (no key) keep the historical 12.
    The unconverged count travels back so the pool path can emit the same
    "did not converge ... increase newton_max_iters" warning the serial
    path emits (pre-fix the pool was silent, and the advice in that very
    message did nothing on this path).

    v5.32.3 (FIX_CI_POOL): so does the parent's Chebyshev EVALUATOR BACKEND
    (``cheb_backend``), for the same reason and with the same fallback for
    payloads that predate the key.  A worker that resolved a different branch
    of ``_Cheb2DEvaluator.ev_value_and_grad`` than the parent ran the same
    mathematics in a different floating-point order, which cost the pool its
    bit-identity to serial (MEASURED 5.167e-14 locally, 1.358e-11 on CI).  A
    worker that cannot honour a pinned ``'numba'`` raises
    :class:`NewtonWorkerBackendUnavailable` rather than substituting the other
    order.

    v5.33.0 (FIX_POOL_REBUILD): and so does the parent's BUILT Chebyshev FIT
    (``cheb_fit``), which retires the polynomial re-fit above entirely.
    Rebuilding it here re-ran ``_solve_lstsq_thread_safe`` -- a BLAS reduction
    over a ~78 000-row design matrix -- in a fresh interpreter, and OpenBLAS
    reduces in a thread-count-dependent order, so a worker whose BLAS width
    differed from its parent's recovered DIFFERENT coefficients on identical
    data (MEASURED max|dc| 4.6e-15, which the Newton convergence threshold
    amplifies to 1.370e-11 of the field -- CI's 1.341e-11 / 1.358e-11).  The
    worker now EVALUATES the parent's coefficients and fits nothing; payloads
    with no ``cheb_fit`` key keep the historical rebuild.
```

### L865-870 -- `_newton_invert_chunk`, the paraxial-magnification keys -- the `pre-3.1.3 callers` / `historical 1.10` framing on a live payload fallback

*Left in the source:* the fallback itself, in the present tense

```text
    # Paraxial-magnification initial-guess factors.  See the docstring
    # in ``apply_real_lens_traced`` where these are computed from the
    # central finite-difference slope of the forward map.  Older knot
    # data written by pre-3.1.3 callers won't have these keys -- fall
    # back to the historical 1.10 multiplier so the worker stays
    # backwards compatible.
```

### L873-875 -- `_newton_invert_chunk`, the Newton cap key -- the `Pre-5.29.1 payloads` framing on a live payload fallback

*Left in the source:* the fallback itself, in the present tense

```text
    # Newton iteration cap, resolved by the caller (caller override >
    # module default).  Pre-5.29.1 payloads have no key -- fall back to
    # the module default so an old pickled payload still runs (audit E-H2).
```

### L988-994 -- `<module>`, the persistent-ProcessPool banner -- the pre-3.5.5 per-call pool teardown cost and the 3.5.5 change that replaced it

*Left in the source:* what the pool does now, and how to close it early

```text
#
# Pre-3.5.5: every apply_real_lens_traced call created+torn-down its own
# pool, paying the Windows-spawn startup cost (~5 s for n_workers=8) once
# per call.  For optimisation runs and tolerancing studies that call
# apply_real_lens_traced 100+ times the cumulative cost was minutes.
#
# 3.5.5+: a module-level pool is lazily created on first parallel-Newton
```

### L1190-1198 -- `_PERSISTENT_POOL_LOCK` -- the v5.30 E-L2 record that the lock used to be lazily created

*Left in the source:* the hazard itself, present tense -- a 'simplification' back to a lazy lock re-opens it

```text
# v5.30 (audit E-L2): the lock is built AT MODULE SCOPE.  It used to be a lazy
# ``None`` that ``_get_persistent_worker_pool`` created on first use -- the
# classic broken double-checked-locking shape, since the ``if
# _PERSISTENT_POOL_LOCK is None: ... = threading.Lock()`` guard is itself
# unsynchronised, so two threads racing the first parallel-Newton call could
# each build a lock, each acquire their own, and both construct a pool (the
# second overwriting the first, leaking its workers).  Building it here is
# free (a ``threading.Lock`` costs nothing at import) and makes the guard
# unnecessary.
```

### L1200-1205 -- `_PERSISTENT_POOL_ATEXIT_REGISTERED` -- the v5.30 E-L1 record that the handler used to be registered per pool creation

*Left in the source:* the hazard and its measurement, present tense

```text
# v5.30 (audit E-L1): one-shot flag for the atexit registration.  The handler
# used to be registered INSIDE the pool-construction block, i.e. once per pool
# creation: measured ``atexit._ncallbacks()`` growing 2 -> 8 across five
# creations (one extra ``close_worker_pool`` callback each time after the
# executor's own).  Every duplicate re-runs a full ``shutdown(wait=True)`` at
# interpreter exit.
```

### L1233-1243 -- `_get_persistent_worker_pool`, the start-method choice -- the v4.16.1 M-2 tag and the clause about which CHANGELOG claim the change made true

*Left in the source:* why ``spawn`` is forced, which is the whole live content

```text
        # v4.16.1 (audit M-2): force the ``spawn`` start method.  The
        # default on Linux is ``fork``, which inherits the parent's
        # FFT plan caches and threading state -- both of which are
        # unsafe to share between forked processes (pyFFTW's plan
        # cache holds module-private locks that the forked child
        # cannot release; numpy/MKL spin up a duplicate thread pool
        # that races with the parent).  ``spawn`` is portable across
        # Linux + macOS + Windows and matches the v4.16.0 CHANGELOG
        # claim that the library uses spawn (which was previously
        # only true of the multi-process storage tests, not the
        # library worker pool itself).
```

### L1453-1458 -- the Newton-pool RESOURCE clamp banner -- `_invert_newton_parallel` "used to submit n_workers chunks with NO memory accounting"

*Left in the source:* why the clamp exists and where its twin lives; the whole measured 267.2 B/point derivation below it is untouched

```text
# ``_invert_newton_parallel`` used to submit ``n_workers`` chunks with NO memory
# accounting of any kind, while the fine grid those chunks come from is sized by
# ``carrier._memory_bounded_n_fine`` with a SINGLE-PROCESS cost model.  The other
# process pool in this library already has the clamp -- see
# ``carrier._multi_resolve_workers``, whose comment records the identical failure
# being fixed there -- and this is the same treatment for the Newton pool.
```

### L1773-1787 -- `_script_has_main_guard` docstring -- "The predicate used to be ..." -- the superseded EXISTS-a-guard proxy and what it classified as guarded

*Left in the source:* the same trap as a do-not-do-this, with the driver-script shape and the 22.1 GB/worker failure it causes

```text
    The predicate used to be "does a top-level ``__name__`` guard EXIST
    anywhere", which is a proxy that cannot see the module BODY -- the very
    thing the warning it feeds is about.  The ordinary shape of a real driver
    script defeats it::

        import numpy as np
        if __name__ == '__main__':
            pass                      # decorative
        BIG = np.zeros((4096, 4096))  # UNGUARDED: 134 MB re-run in EVERY worker
        main()

    which the old form classified as guarded, so the pool ran and every worker
    paid the cost -- precisely the 22.1 GB/worker failure the warning exists
    for.  ``ast.Match`` is accepted as a guard shape alongside ``ast.If``
    (``match __name__: case '__main__':`` used to read as unguarded).
```

### L2707-2717 -- `_det_normal_equations`, the empty-fit guard -- "This branch used to sit below ``B.reshape(...)``" -- where the guard was before the P4 close-out

*Left in the source:* why the guard sits ABOVE the reshape and what the contract is

```text
    # EMPTY FIT, guarded ABOVE the reshape (P4 close-out, 2026-08-24).  This
    # branch used to sit below ``B.reshape(B.shape[0], -1)``, which raises
    # ``cannot reshape array of size 0 into shape (0,newaxis)`` on a zero-row
    # ``b`` -- so it was dead by construction and the deterministic route
    # RAISED where the ``(A.T @ A, A.T @ b)`` it replaces returns zeros.  This
    # function's whole contract is "the same two arrays as the BLAS
    # expression, in a fixed summation order", and a shape the BLAS expression
    # handles is part of that contract.  No live caller reaches it (every fit
    # site enforces a samples-per-term floor), so nothing on any shipped path
    # moves: ``_solve_lstsq_thread_safe`` still raises on an empty fit, from
    # ``_solve_lstsq_qr``'s own reshape, on BOTH routes exactly as before.
```

### L3117-3127 -- `_solve_lstsq_thread_safe` docstring, CONDITIONING -- a docstring correcting its OWN earlier assertion that ``A`` is well-conditioned so squaring its condition number is safe

*Left in the source:* the settled fact -- which fits are well conditioned, which are not, the measured cond(A) and what the screen does

```text
    CONDITIONING (niche C13, 2026-08-03).  This function used to assert that
    ``A`` "is a well-conditioned normalised tensor-Chebyshev / monomial
    Vandermonde (~1.5x oversampled), so squaring the condition number in ``G``
    is safe".  That holds for the concentric unweighted fits and is FALSE for
    the weighted decentred ones, where ``cond(A)`` = 1.4e10 was measured and
    ``G`` is therefore numerically singular.  Rather than assume either way,
    the Gram is now SCREENED and, where it is singular, both answers are scored
    on the data.  A solve that passes the screen -- or one whose two candidates
    tie -- returns the identical bits it returned before, which is what keeps
    the byte-identity contracts of niches C1/C6/C8/C9 intact.
    See ``LSTSQ_CONDITIONING_STEPDOWN``.
```

### L3430-3440 -- `_Cheb2DEvaluator.from_state` banner -- the v5.33.0 FIX_POOL_REBUILD record that a Newton pool worker used to call `__init__`

*Left in the source:* why this entry point exists, as a statement about what `__init__` costs a worker

```text
    # ----------------------------------------------------------------
    # v5.33.0 (FIX_POOL_REBUILD): construct from an ALREADY-BUILT fit.
    #
    # ``__init__`` RUNS the least-squares fit.  A Newton pool worker used to
    # call it, which meant every worker re-solved the same normal equations in
    # its own interpreter -- and that solve is a BLAS reduction whose ORDER
    # depends on the BLAS thread regime, so a worker whose regime differed from
    # its parent's recovered coefficients that differ in the last bits.  See
    # ``_cheb_fit_state`` for the measurement.  This entry point takes the
    # parent's coefficients and does no arithmetic at all.
    # ----------------------------------------------------------------
```

### L3676-3680 -- `_analytic_lens_phase` docstring, the thickness budget -- a docstring correcting its OWN earlier "under 10 nm on F/10+" rule

*Left in the source:* the measured budget and the thickness ceiling the rule of thumb actually has

```text
    0.7 nm at 100 um and **14.1 nm rms / 41.6 nm PV at 2 mm** -- so the
    "under 10 nm on F/10+" rule this docstring used to state holds only
    for elements thinner than ~1.5 mm, whatever their speed.  Budget
    ``~7 nm rms per mm of glass`` and validate before trusting a thick
    or fast element.
```

### L4170-4180 -- `DECENTRED_FIT_ARBITER`, the default-flip paragraph -- which release shipped the arbiter ON, that C11 shipped it OFF, and that "that decision has now been taken"

*Left in the source:* the trade the shipped default takes, measured, and the audit that re-measured it on both BLAS builds

```text
#: SHIPPED ON since 5.32.1 (2026-08-03), by an EXPLICIT DECISION, and the
#: trade it takes is stated rather than buried.  On design 121 the arbiter
#: improves four of the five tilted orders (by 0.017 / 0.052 / 0.110 / 0.082
#: points), takes the worst-case residual from 0.152 to 0.069 and removes the
#: residual's growth with field angle -- and makes ONE order, (-1,0), worse by
#: 0.026 points against a 0.003-0.015 differential floor.  That per-order
#: "improve or hold" failure is why C11 shipped it OFF: it is a judgement
#: about a design rather than a library fact, and it was left to an explicit
#: decision instead of taken silently in a patch release.  **That decision has
#: now been taken** -- see ``docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md``
#: S10 for the re-measurement of the trade on BOTH BLAS builds.
```

### L5255-5259 -- the rationalised exact-sphere expression -- "the k0*eps*|s| error it used to carry" -- the pre-rationalisation cancellation error as history

*Left in the source:* the same hazard in the present tense: why this expression, and why a cancellation error here would be coherent across three legs

```text
    # RATIONALIZED: sqrt(r^2+s^2) - |s| == r^2 / (sqrt(r^2+s^2) + |s|).  Same
    # fix and same reason as a185cfc in propagators/carrier.py.  This one
    # feeds the ray launch, the H6 entrance eikonal AND the exp(i k0 W)
    # reference leg, so the k0*eps*|s| error it used to carry was COHERENT
    # across all three.
```

### L5349-5429 -- `_REMAP_RESID_EIKONAL_DEGREE` -- the whole degree-4 era: the WHY-4 end-to-end table, the 0.103 %-of-input-power on-axis ghost it shipped with, the MECHANISM note that refuted the note above it, the RESOLVED note that corrected a guess in the same block, and the 2026-08-02 "RAISED TO 6" announcement

*Left in the source:* the present-tense WHY 6 (the r^4/r^6 form argument), the fail-before, and the live degree-4-vs-6 C8 table that follows it

```text
#: WHY 4.  Measured on design 121's last group at order (-4,-2), the worst
#: case -- the element pass scored pointwise against the exact-ray oracle, and
#: the same setting scored END TO END through the whole post-DOE chain:
#:
#:   degree  rms grad(a-a_fit)  element WFE (waves)  ghost power  chain EE3 %
#:     off        --                 0.0659            0.00000       73.66
#:      2       8.83e-4              0.0388            0.00000       81.95
#:      3       2.99e-4              0.0074            0.00000       88.66
#:      4       1.03e-4              0.0140            0.00000       88.49
#:      5       1.01e-4              0.0144            0.00132       88.49*
#:      6       8.5e-5               0.0136            0.01255       88.72
#:
#: (* interpolated; degree 5 was not run end to end.  Oracle ceiling 89.78.)
#:
#: Read it as: everything from degree 3 on lands within 0.25 EE3 points of
#: everything else END TO END, so the choice is made on GENERALITY.  Degree 4
#: is where the residual's own content is -- a carrier-referenced relay carries
#: an r^4-dominant correction (see ``remap_sampling``), and degree 4 spans that
#: EXACTLY: on a synthetic fixture whose residual IS r^4 the corrected launch
#: reads 2e-5 waves against 0.021 uncorrected, a factor of 1000, where degree 3
#: reads 0.014 and removes only a third.  Degree 3 happens to fit design 121's
#: particular group-5 residual better (90 % of its slope) but cannot represent
#: the generic case at all.  Degrees 5-6 buy nothing and start self-caustiking
#: in the 2-3 w skirt.
#:
#: KNOWN COST OF THIS CHOICE, measured and NOT fixed: at degree 4 the ON-AXIS
#: design-121 call gains **0.103 % of the input power** as a far ghost lobe
#: (exit power 0.9959 -> 0.9970), where degree 3 gains none.  It appears only
#: on the CONCENTRIC fit branch (hard NaN mask, ``newton_poly_order=6``); the
#: off-centre branch (weighted, ``_DECENTRED_FIT_POLY_ORDER=10``) is clean at
#: every order.  It does not move the spot (on-axis EE3 89.21 at degree 4
#: against 88.87 at degree 3, and the field's second moment inside r < 1 mm
#: moves by 0.0003 mm) but it IS spurious energy -- 1.03e-03 of Pin as an
#: annulus at 6.298-7.216 mm exit radius carrying 33 % of the peak amplitude --
#: so any halo / second-moment metric taken through this path on axis should be
#: checked against ``REMAP_STATIONARY_PHASE_LAUNCH = False``.
#:
#: MECHANISM (2026-07-31), and it is NOT the one first recorded here.  The
#: original note read this as "the order-6 forward-map fit being unable to
#: carry a degree-4 launch augmentation".  That is REFUTED: raising the order
#: on the hard-mask branch makes it 86x WORSE (``newton_poly_order=10`` gains
#: 8.5 % of the input power).  It is the D1 fold -- see
#: ``REMAP_STATIONARY_PHASE_FIT_GUARD`` for the mechanism, the sweeps and the
#: opt-in remedy.
#:
#: RESOLVED (2026-07-31): the ELEMENT-vs-oracle column's NON-MONOTONICITY in
#: the degree (3 reading 0.0074 against degree 4's 0.0140, while the model's
#: own slope residual keeps improving) is an artefact of the PROBE's 2 %-of-
#: peak amplitude threshold, not of the fit and not -- as was guessed here --
#: of the oracle's band-limited representation of ``a``.  Measured
#: (validation/repro_traced_carrier_121/probe_c6_degree_oracle.py): the
#: oracle's own upsample factor is converged (4 / 8 / 16 move degree 4 by
#: 1.8 % and never reorder anything), the scored patch size is irrelevant, and
#: raising the amplitude threshold to 10 % of peak REVERSES the ordering to
#: 4 < 6 < 3 -- the order ``grad(a - a_fit)`` predicts -- while dropping
#: degree 4's reading 6x, 0.0140 -> 0.0024 waves.  The whole penalty lives in
#: the 2-10 %-of-peak skirt, where degree 4's extra terms are constrained by
#: the core (``_REMAP_RESID_BRIGHT_FRAC`` = 0.05) and then evaluated outside
#: it.  On a synthetic fixture with an ANALYTIC oracle the response is
#: perfectly ordered -- 2.065e-02 (off) -> 1.406e-02 (degrees 2 and 3) ->
#: 2.344e-05 (degrees 4, 5, 6) -- so nothing is wrong with the fit.  Degree 4
#: is the best model over the bright core.
#:
#: ---------------------------------------------------------------------------
#: 2026-08-02 -- RAISED TO 6.  THE ONLY THING KEEPING IT AT 4 WAS A GHOST THAT
#: NICHE C8 NOW BOUNDS.  docs/audits/D121_RESIDUAL_CLOSURE_2026_08_02.md.
#:
#: The fail-before is this constant: setting it back to ``4`` restores the
#: v5.32.0 / niche-C9 behaviour exactly, since it is the ONLY thing that
#: changes (it is read once, at :func:`_fit_residual_eikonal`, and clamped by
#: ``_REMAP_RESID_DEGREE_CAP``).
#:
#: WHAT THE "degrees 5-6 buy nothing and start self-caustiking" line above was
#: measuring, and why it no longer holds.  Both records of that ghost -- the
#: ``ghost power`` column here (1.255e-02 at degree 6, order (-4,-2)) and
#: ``REMAP_STATIONARY_PHASE_FIT_GUARD``'s "degree 6 still reads 9.78e-03" --
#: predate ``REMAP_INVERSE_SUPPORT_BOUND`` (niche C8, 2026-08-01), whose whole
#: job is to stop the library CLAIMING amplitude outside the traced ray
#: support.  The degree-6 ghost is exactly such a claim.  Re-measured on the
#: post-C9 tree through ``energy_stage_audit_121.py`` (unedited), design 121
#: order (-4,-2), ``RN=1024``, ``rs=4``, six post-DOE groups:
```

### L5627-5631 -- `REMAP_STATIONARY_PHASE_FIT_GUARD`, THE DEFECT -- the retraction of a hypothesis that was recorded in `_REMAP_RESID_EIKONAL_DEGREE` and has itself now moved

*Left in the source:* the measured fact the retraction rested on (86x worse with more terms) and the mechanism

```text
#: This RETRACTS the hypothesis recorded in ``_REMAP_RESID_EIKONAL_DEGREE`` --
#: "the order-6 forward-map fit [is] unable to carry a degree-4 launch
#: augmentation".  More terms on the hard-mask branch makes it 86x WORSE (row
#: 3), which is what an unconstrained extrapolation does and not what an
#: under-resolved fit does.  The mechanism is the fitted entrance->exit map
```

### L5666-5673 -- `REMAP_STATIONARY_PHASE_FIT_GUARD`, THE STRUCTURAL FIX -- the dated "2026-08-01: THE FIRST OF THOSE SHIPPED" follow-up appended under a sentence that said neither was attempted

*Left in the source:* one present-tense sentence: which structural fix ships, and that it makes this flag redundant

```text
#: THE STRUCTURAL FIX is not on this axis at all: bound the Newton inverse to
#: the traced samples' own support, or use a caustic-faithful amplitude model
#: (``apply_real_lens_gbd`` / ``apply_real_lens_fga``).  Neither is attempted
#: here.  See docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S3.
#:
#: 2026-08-01: THE FIRST OF THOSE SHIPPED, as ``REMAP_INVERSE_SUPPORT_BOUND``
#: (niche C8), and it makes this flag REDUNDANT on every case measured -- see
#: the closing note at the bottom of this docstring.
```

### L5675-5679 -- `REMAP_STATIONARY_PHASE_FIT_GUARD`, the chain-level block -- "The two claims above that were element-level inferences are now chain measurements, and ONE OF THEM WAS WRONG"

*Left in the source:* the heading and the audit reference; the three numbered chain measurements under it are untouched

```text
#: ---------------------------------------------------------------------------
#: 2026-07-31: MEASURED AT CHAIN LEVEL, AND THE DEFAULT IS CONFIRMED ``False``.
#: docs/audits/C6_FIT_GUARD_DECISION_2026_07_31.md.  The two claims above that
#: were element-level inferences are now chain measurements, and ONE OF THEM
#: WAS WRONG.
```

### L5934-5956 -- the niche-C14 UNIT C banner -- "WHAT WAS WRONG.  There were THREE notions ..." -- the three rules, the duplicated convex-hull algebra and the two copies of the half-plane evaluator that the extraction replaced

*Left in the source:* what the object IS (one construction, three named views) and the measured consequence that follows

```text
# ===========================================================================
# NICHE C14 (2026-08-03) -- UNIT C: THE TRACED EXIT SUPPORT, AS ONE OBJECT
# ===========================================================================
# WHAT WAS WRONG.  There were THREE notions of "the region the traced rays
# reached", computed from the same arrays, at nearly the same point in
# ``apply_real_lens_traced``, by three different rules and three separate
# copies of the same convex-hull algebra:
#
#   1. the C7 halo radius -- amplitude-weighted centroid + max radius over the
#      samples above the ``e^-_RD_HALO_AMP_CONTOUR`` amplitude contour, times
#      ``_RD_HALO_RADIUS_FACTOR`` at report time;
#   2. the C8 support hull -- convex hull of the alive STOP-PASSING landings,
#      plus a ``sqrt(2) sub dx`` plateau and one exit-lattice cell of feather;
#   3. the direct-fit hull -- the ``inversion_method='fit'`` path's own
#      long-standing exit hull mask over the post-restriction samples.
#
# (2) and (3) are the same idea implemented twice, and the C8 audit says so:
# "This bound gives the Newton path the containment the direct-fit path has had
# all along."  They had two copies of the ConvexHull call, two copies of the
# ``equations -> (A, b)`` unpacking and two different half-plane evaluators
# (one chunked over a BLAS product, one a full-width ``np.all``).
#
# THE MEASURED CONSEQUENCE, and the reason this is not cosmetics.
```

### L7139-7145 -- `_reverse_prescription` docstring -- a docstring correcting its OWN earlier assertion that even-power aspheric coefficients are invariant under the reversal

*Left in the source:* the rule (every sag term flips; the radius flip carries the conic) and the measured 2.13-wave error that getting it wrong costs

```text
    ``sag_callable`` have no radius to flip, so each has to be negated
    explicitly -- which is what this does now.  The docstring used to assert
    the opposite ("even-power aspheric coefficients are invariant"); measured
    on a 100 mm/plano N-BK7 singlet with ``aspheric_coeffs={4: 1.0e3}`` at
    h = 5 mm, the reversed sag came out ``-2.500031447e-04 m`` against the
    correct ``-2.512531447e-04 m`` -- an error of 1.25e-06 m = 2.13 waves at
    588 nm, i.e. the whole aspheric departure with the wrong sign.
```

### L7257-7276 -- `_opl_by_backward_trace` docstring, Validation -- a docstring retracting its OWN "~35-40 nm on singlets at N=512" figure and recording that no fixture reproduces it

*Left in the source:* the measured 1.93 rad rms disagreement, the undersampled regime it was taken in, and the EXPERIMENTAL verdict that follows

```text
    *   End-to-end exit-phase agreement with the Newton path: the
        "**~35-40 nm** on singlets at N=512" figure this docstring used to
        quote is **NOT REPRODUCIBLE ON ANY FIXTURE IN THE TREE, and no test
        pins the one it was measured on**.  Re-measured on an N-BK7 100/-100
        singlet (2 mm thick, 6 mm aperture, N = 256, dx = 30 um, 587.6 nm,
        ray_subsample = 8, a 1.5 mm Gaussian): backward-vs-Newton exit phase
        **1.93 rad rms (180 nm), max 372 nm** inside r < w -- 1.93 rad is the
        1.81 rad of a UNIFORMLY-DISTRIBUTED wrapped difference and the maximum
        sits on the wrap boundary, i.e. the two inversions differ by more than
        a wave, not by tens of nanometres.
        That fixture is exit-UNDERSAMPLED (grid Nyquist direction cosine
        ``lambda/(2 dx)`` = 0.0098 against an exit NA ~0.031), which is the
        regime in which ``_sample_local_tilts`` -- the source of this route's
        launch directions -- aliases, so the measurement is consistent with
        the attribution below; but the old claim excluded no such regime, and
        the two ``_sample_local_tilts`` defects the 2026-09 audit fixed (the
        ``np.roll`` wrap and the half-pixel storage offset) fed straight into
        that budget and were not acknowledged in it.  Treat this route as
        EXPERIMENTAL with an unquantified exit-phase error; use Newton (the
        default) for anything with a tolerance.
```

### L7403-7410 -- the coarse->fine index mapping -- the ``ii * N_c / N`` predecessor written as what the code previously did

*Left in the source:* the same trap as a do-not-do-this, with the measured diagonal focus walk it produces

```text
    # Coarse sample u sits at FINE index u*sub (idx_c = arange(0, N, sub)),
    # so fine pixel ii maps to coarse coordinate ii/sub -- EXACT for any sub.
    # The previous ``ii * N_c / N`` equals ii/sub only when sub divides N;
    # otherwise it is a corner-anchored scale error that displaces the whole
    # map diagonally by (N/2)*(N_c*sub - N)/N pixels (audit
    # AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24: the traced chain's diagonal
    # focus walk -- measured -6.100 um at N=8192/sub=50 vs -6.11 predicted).
    # Bit-identical to the old expression whenever sub | N.
```

### L7845-7873 -- `apply_real_lens_traced` docstring, ``tilt_aware_rays`` -- "Why the default flipped from True to False in 3.1.3" and "The 3.1.4 default ... restores the ... launch that pre-3.1.2 releases used"

*Left in the source:* the same reference-consistency argument as the reason the default IS False, and when to pass True

```text
        **Why the default flipped from True to False in 3.1.3:**  When
        ``preserve_input_phase=True`` (also the default), the exit
        field is assembled as

            E_out = E_analytic * exp(i * delta_phase)
            delta_phase = k0 * opl_traced - phase_analytic_lens

        where ``phase_analytic_lens`` is the phase produced by running
        :func:`apply_real_lens` on a unit PLANE WAVE -- i.e. a
        plane-wave reference.  For ``delta_phase`` to be a
        mathematically clean "ray-traced minus analytic" correction,
        ``opl_traced`` must use the same reference: a plane-wave
        entrance launch.  With ``tilt_aware_rays=True``, ``opl_traced``
        instead mixes the lens-model correction with per-pixel
        tilt-induced phase shifts that the plane-wave ``phase_analytic_lens``
        does not contain.  The resulting ``delta_phase`` is only
        approximately right for small/uniform input tilts, and breaks
        materially on multi-mode inputs (post-DOE fields, strongly
        off-axis compound beams) where the per-pixel tilts vary
        significantly across the pupil.

        The 3.1.4 default ``tilt_aware_rays=False`` restores the
        reference-consistent plane-wave launch that pre-3.1.2 releases
        used, so ``delta_phase`` remains well-defined for any input the
        wave model can represent.  If you have a specifically small,
        uniform input tilt and want the per-ray OPL variation (e.g.
        rigorous off-axis lens characterisation with a single tilted
        input), pass ``tilt_aware_rays=True`` explicitly and validate
        against the default on your specific case.
```

### L7939-7947 -- `apply_real_lens_traced` docstring, ``on_noncollimated`` -- "It used to be documented as removing its one-FFT-free cost" and the v5.29.1 E-M3 note that an unrecognised value used to select 'warn'

*Left in the source:* the measured no-saving statement (which is why the knob is documented as suppression-only) and the live validation rule

```text
        ``'off'`` is a suppression knob, not a cost knob.  It used to be
        documented as removing "its one-FFT-free cost"; it does not, because
        the ``else`` branch recomputes the SAME ``_input_tilt_stats`` for the
        tilt warning.  Measured at N = 1024, ray_subsample = 4, median of 4:
        ``'warn'`` 4.013 s against ``'off'`` 4.209 s -- no saving.  (cProfile
        puts ``_input_tilt_stats`` at 0.310 s of a 5.760 s call and
        ``_input_beam_amp_radius`` at 0.092 s, both for warnings only.)  v5.29.1 (audit E-M3): any OTHER value now raises --
        it used to select ``'warn'`` silently, and ``'silent'`` in particular
        therefore warned instead of suppressing.
```

### L7992-7997 -- `apply_real_lens_traced` docstring, ``newton_max_iters`` -- "Honoured ... since v5.29.1 (audit E-H2 -- the pool worker previously hard-coded 12 ...)"

*Left in the source:* that both paths honour the cap, which is the contract a caller needs

```text
        (12).  Honoured by BOTH the serial and the process-pool inversion
        paths since v5.29.1 (audit E-H2 -- the pool worker previously
        hard-coded 12, making this knob inert whenever the pool engaged; at
        that time that meant >=200k Newton points with ``newton_fit='spline'``
        on the CPU path, whereas the pool now serves EITHER fit above a
        two-tier size gate -- see ``n_workers``).
```

### L8145-8148 -- `apply_real_lens_traced` docstring, ``on_fit_domain_basis`` -- "Validated at entry since v5.32.2 (finding V4) ... Before that gate every unrecognised value ... silently selected 'warn'"

*Left in the source:* the live validation rule

```text
        Validated at entry since v5.32.2 (finding V4): any other value raises
        ``ValueError``.  Before that gate every unrecognised value -- including
        ``'Error'`` and ``'ignore'`` -- silently selected ``'warn'``, so a
        caller asking for a fatal got a warning and a returned field.
```

### L8365-8377 -- `apply_real_lens_traced` docstring, ``remap_sampling`` -- the dated "**The POINT GAIN is gone once niche C6 lands**" follow-up and its "Nothing above is retracted" clause

*Left in the source:* the live expectation -- ``'full'`` does not buy EE points on a post-C6 chain -- and the measurement behind it

```text
        **The POINT GAIN is gone once niche C6 lands; the CONVERGENCE argument
        is not (2026-07-31).**  On the worst tilted order of design 121,
        `(-4,-2)`, end to end against the landed C6 launch, ``'lattice'``
        measures **+0.0988 EE3 points** -- i.e. marginally BETTER than
        ``'full'``, against **-17.73 points** for the same substitution on
        pinned HEAD with the C6 defect open.  That is expected: at HEAD the
        launch went along ``grad(W)``, so the residual was being SAMPLED at the
        wrong foot and the sampling resolution mattered enormously; with the
        stationary-phase launch it is sampled at the right one.  Nothing above
        is retracted -- the dx-independence measurement is what this default
        rests on, and it is untouched -- but do NOT expect ``'full'`` to buy
        EE points on a post-C6 chain.  See
        docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S1.
```

### L9002-9010 -- the ``newton_amp_mask_rel`` override in the `caustic='wave'` block -- "this override used to be SILENT while every other requirement in this block raises"

*Left in the source:* the live rule, including which value is read as 'not requested' and why

```text
        #
        # v5.29.1 (audit E-M5): this override used to be SILENT while every
        # other requirement in this block raises.  Match the block (and the
        # ``_FORCED`` contract in apply_real_lens_traced_multi): a value equal
        # to the forced 0.0 is accepted, anything else raises with the reason.
        # The shipped default ``_NEWTON_AMP_MASK_REL_DEFAULT`` is read as "not
        # requested" (there is no separate not-passed sentinel), so a caller
        # who explicitly passes exactly the default still gets the silent
        # override -- pass 0.0 to state the intent.
```

### L9467-9477 -- the ``sag_chunk_rows`` resolution -- two stacked release notes (v5.17.0 then v5.17.1 P2-05) on the same kwarg, the second correcting what the first shipped

*Left in the source:* the live rule: what the raw value means and why the RAW value is forwarded

```text
    # v5.17.0: sag_chunk_rows=None resolves to AUTO (banded when N >= 4096);
    # pass 0 to force the whole-grid path.  The caller's RAW kwarg also flows
    # to the apply_real_lens amp legs so both stages resolve -- and band --
    # consistently.
    # v5.17.1 (audit P2-05): forward the RAW kwarg, not the resolved value.
    # The resolver maps 0 -> None, and apply_real_lens re-resolves None ->
    # AUTO, so forwarding the resolved value silently re-enabled row-banding
    # in the amp legs when the caller passed the documented force-whole-grid
    # sentinel 0.  Both stages resolve the raw value against the same N, so
    # None / positive ints band identically in both stages and 0 now forces
    # whole-grid in BOTH.
```

### L10609-10617 -- the `min_coarse_samples_per_aperture` apertureless arm -- "the floor was documented as enforced ... but the guard was silently skipped for apertureless prescriptions"

*Left in the source:* what the effective pupil is on an apertureless prescription, and why

```text
            # v5.17.1 (audit P3-08): the floor was documented as enforced
            # against the launch radius when no ``aperture_diameter`` is
            # set, but the guard was silently skipped for apertureless
            # prescriptions.  Derive the effective pupil from the largest
            # per-surface ``clear_aperture`` when present (the actual
            # pupil-limiting hardware, capped at the launch diameter the
            # coarse grid actually spans), else the launch diameter itself
            # (= the grid extent), so apertureless prescriptions get the
            # same aliasing protection.
```

### L11023-11035 -- the entrance-grid reshape -- a comment correcting an earlier COMMENT at the same site -- "the comment that used to sit at this site ... was wrong twice over"

*Left in the source:* both settled facts: the launch square's corners really are outside the clear aperture, and FITPACK does not ignore NaN

```text
    # Vignetting is NOT rare here, and the comment that used to sit at this
    # site ("vignetting is rare for normal lenses but we guard against it by
    # filling dead entries with NaN and extrapolating with the spline's
    # natural extrapolation") was wrong twice over.  The launch lattice is a
    # SQUARE of half-width ``launch_radius = 0.75*aperture``, so its corners
    # sit at ``sqrt(2)*0.75 = 1.06`` aperture radii -- past any per-surface
    # ``semi_diameter`` of ``aperture/2``, i.e. past 2.12 clear-aperture radii.
    # And ``RectBivariateSpline`` is an interpolating (``s = 0``) FITPACK fit
    # that does not ignore NaN: one NaN sample makes ~90 % of the spline
    # coefficients NaN, ``So.ev`` NaN everywhere, ``valid = isfinite(opl_map)``
    # all-False and the returned field IDENTICALLY ZERO -- reported to the
    # caller only as a 100 %-unconverged Newton warning, which misdiagnoses
    # both the cause and the outcome.
```

### L11137-11149 -- the on-axis OPL reference, FIX_TILT_QUADRATIC_OPL -- "What was WRONG is that the constant was then dropped" -- the defect as a fix narrative

*Left in the source:* why the subtraction happens and why the constant is re-applied at assembly, with the full tilt-quadratic derivation and its measurement

```text
    # constant.  What was WRONG is that the constant was then dropped: every
    # branch below builds the exit phase from ``k0 * opl_map`` alone, so the
    # returned field's absolute phase was referenced to
    #
    #     Lam(0) = W(0, 0) + a_fit(0, 0) + P(0, 0)
    #
    # -- the entrance eikonal of the launched congruence at the LAUNCH-LATTICE
    # AXIS (the H6 / niche-C6 terms added to ``final.opd`` above) plus the
    # geometric path of the ray launched there.  On an UNTILTED, UNDECENTRED
    # congruence the axis IS the chief ray, so this only cost an unobservable
    # global phase -- which is why it survived.  Under a
    # :class:`TiltedCarrier` the axis is NOT the chief ray, and BOTH pieces
    # become functions of the tilt:
```

### L11704-11713 -- the paraxial-magnification stencil -- "4.11.2: the indices match ... Pre-4.11.2 the indices were swapped" -- the index-order defect as a release note

*Left in the source:* the index convention itself and why it is the one the meshgrid demands

```text
    # because n_launch is odd).  4.11.2: the indices match the meshgrid
    # at the launch step
    #     ``Xs_in, Ys_in = np.meshgrid(xs_in, xs_in, indexing='ij')``
    # which puts x along axis 0 and y along axis 1, so ∂x_out/∂x_in
    # varies the FIRST index, not the second.  Pre-4.11.2 the indices
    # were swapped, computing ∂x_out/∂y_in (~zero by rotational
    # symmetry) instead of ∂x_out/∂x_in.  Newton still converged
    # because the polynomial Jacobian is right, but every pixel started
    # at the clipped-to-boundary initial guess (0.91-fallback) instead
    # of the actual paraxial slope.
```

### L11775-11781 -- the Newton cap in the pickled worker payload -- "``_newton_invert_chunk`` used to hard-code ``_NEWTON_MAX_ITERS``, so ``newton_max_iters`` was inert whenever the process pool engaged"

*Left in the source:* what the payload key is for, in the present tense

```text
    # v5.29.1 (audit E-H2): carry the RESOLVED cap into the pickled worker
    # payload.  ``_newton_invert_chunk`` used to hard-code
    # ``_NEWTON_MAX_ITERS``, so ``newton_max_iters`` was inert whenever the
    # process pool engaged (then >=200k points, newton_fit='spline', CPU;
    # now either fit, above the two-tier cold/warm gate) -- and
    # the ray-density amplitude leg, which always runs the SERIAL closure,
    # could then be built from a different Newton solution than the OPL.
```

### L11792-11795 -- `_warn_newton_unconverged` docstring -- "Pre-3.5.6 unconverged pixels were silently kept ...; the POOL path stayed silent until v5.29.1"

*Left in the source:* that both paths report identically, which the opening sentence already states

```text
        rest of the function uses ('silent' suppresses).  Pre-3.5.6
        unconverged pixels were silently kept at their last Newton value; the
        POOL path stayed silent until v5.29.1 (audit E-H2) even though the
        message's own advice is "increase newton_max_iters".
```

### L12256-12259 -- the pool path's unconverged report -- "the pool used to be silent, so the one regime whose convergence the message's own advice addresses never reported"

*Left in the source:* what the worker returns and that the pool emits the same warning as the serial path

```text
        # v5.29.1 (audit E-H2): the worker returns (opl, n_unconverged); sum
        # the counts and emit the SAME warning the serial path emits (the
        # pool used to be silent, so the one regime whose convergence the
        # message's own advice addresses never reported).
```

### L12838-12843 -- the (2, N, N) coordinate stack -- "Pre-fix the stack was constructed twice ... ~4 extra full-grid float64 (~34 GB at N=32768)"

*Left in the source:* the rule and its cost, present tense, ahead of the FIX_PERF_ROUND2 measurement that follows

```text
            # v5.16.2 (memory root-cause): build the (2, N, N) coordinate
            # stack ONCE and free ii/jj before interpolating.  Pre-fix the
            # stack was constructed twice (once per map_coordinates call)
            # with ii/jj held throughout -- ~4 extra full-grid float64
            # (~34 GB at N=32768) at the upsample peak.  Same coords,
            # same map_coordinates inputs -> byte-identical outputs.
```

### L14158-14163 -- `apply_real_lens_traced_multi`, the `_FORCED` contract -- "Pre-v5.29 they were popped SILENTLY, so a caller asking for e.g. ``preserve_input_phase='remap'`` got ``True`` with no diagnostic"

*Left in the source:* the live contract: a value equal to the forced one is accepted, anything else raises

```text
    # the phase is preserved.  These values are FIXED by the per-emitter
    # contract.  Pre-v5.29 they were popped SILENTLY, so a caller asking for
    # e.g. ``preserve_input_phase='remap'`` got ``True`` with no diagnostic
    # (measured byte-identical to True while the direct element call differs by
    # 7.5e-3).  Now: a value EQUAL to the forced one is accepted (no behaviour
    # change), anything else raises with the reason.
```

### L14787-14800 -- `prepare_real_lens_traced` docstring, WHAT A PREPARED LENS FREEZES -- "Pre-v5.29.1 the unresolved ``None`` sentinels were stored, so the frozen screen and the per-call analytic amplitude leg silently desynchronised" with its two measured numbers

*Left in the source:* what a prepared lens freezes and what a caller must do to pick up a changed default

```text
    WHAT A PREPARED LENS FREEZES (v5.29.1; audit E-H4).  **A prepared object
    freezes the settings that were live when it was prepared.**  Concretely:
    ``wave_propagator`` and ``sag_dtype`` are RESOLVED here against the
    process-wide defaults (:func:`set_default_wave_propagator` /
    :func:`set_lens_sag_dtype`) and the resolved values are stored on the
    returned object, and the ``prescription`` is DEEP-COPIED.  So flipping a
    global default -- or mutating the prescription dict in place -- after
    preparing leaves the prepared lens unchanged; rebuild it to pick up the
    new settings.  Pre-v5.29.1 the unresolved ``None`` sentinels were stored,
    so the frozen screen (prepare-time defaults) and the per-call analytic
    amplitude leg (call-time defaults) silently desynchronised on a global
    flip (measured 49.6 on a singlet), and an in-place prescription edit --
    the optimizer / tolerancing pattern this class advertises -- produced a
    stale-OPL x new-amplitude hybrid (measured 0.71 from a correct rebuild).
```
