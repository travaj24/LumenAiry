<!-- lumenairy-history-doc
module: lumenairy/propagators/fft_infra.py
ast_sha256: bd5002a4bb2ed943962b9bac214eb699c952d4d55eb715f4d96c4e9401ee6eab
token_sha256: 5a24405d51265962a660e4d2ce29457a6327702551f184e910387661e7011c40
pre_relocation_lines: 2678
recorded_by: WP-A17 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- scipy.fft deferred behind find_spec + a first-use accessor; no behaviour change (WP-A22 item 9)
-->

# Version history -- `lumenairy/propagators/fft_infra.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/fft_infra.py` -- the "vX.Y (audit Z): pre-fix this did A,
which was wrong because B, now it does C" blocks.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

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
| L5-7 | `<module> docstring` | the v5.1.0 propagation.py split |
| L21-24 | `<module> docstring` | "no public behaviour changed in the split" |
| L43-45 | `<module>` | why the CuPy probe was consolidated (P2-9) |
| L71-76 | `_is_cupy_array` | the NumPy 2.x ``.device`` duck-type break |
| L123-127 | `<module>` | why the knobs are registered (P2-5) |
| L138-148 | `FFTW_THREADS` | S5-8c -- the 8-thread oversubscription cap |
| L196-206 | `_PYFFTW_BAD_SHAPES` | K7 -- the blacklist key is a triple, not a bare shape |
| L303-323 | `DEFAULT_* knobs` | the v4.16.2 -> v5.1.0 per-knob consumer-rollout status table |
| L364-374 | `DEFAULT_WAVE_PROPAGATOR_SHIPPED` | W9-8 -- why the shipped default is a frozen CONSTANT |
| L379-388 | `_DEFAULT_*_NO_CONSUMER_WARNED` | why the two one-shot latches are pinned True |
| L463-466 | `set_default_wave_propagator` | the retired pre-v5.1.0 "API-only" UserWarning |
| L511-514 | `set_default_dy` | the retired pre-v5.1.0 "API-only" UserWarning |
| L550-556 | `_resolve_jax_complex_dtype` | L2 -- why a central JAX dtype resolver exists |
| L617-621 | `_resolve_jax_real_dtype` | L2 companion -- the real-dtype twin |
| L655-665 | `_PYFFTW_PLAN_CACHE` | why the single-slot plan cache became a multi-slot LRU |
| L694-701 | `_PYFFTW_SHARED_BUFFERS_UNSAFE` | K4 -- the measured 100 %-wrong fields that the latch prevents |
| L743-761 | `_PYFFTW_AUTO_PROMOTE` | W9 -- why auto-promote stopped being the default |
| L854-857 | `reset_fft_backend` | P3-14 -- clear the blacklist in place, do not rebind it |
| L870-874 | `reset_fft_backend` | S5-8 -- why pyfftw.interfaces.cache is not toggled here |
| L908-914 | `_PYFFTW_DOUBLE_BUFFER` | v5.16.2 -- the ping-pong opt-out |
| L917-921 | `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` | v5.33.2 -- the per-key byte cap: the 24.77 GB that prompted it |
| L961-966 | `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` | v5.33.3 -- the two corrected figures (11586 -> 11181, 4.5 s -> 1.6 s) |
| L1077-1079 | `warmup_fft_plans` | the ``threads`` default: the docstring contradicted the code |
| L1100-1104 | `warmup_fft_plans` | F-32 -- warm up on FFTW_THREADS, not _available_cpus() |
| L1115-1119 | `_FFT_STATE_KEYS` | P3-16 -- why snapshot/restore exists |
| L1124-1129 | `_FFT_STATE_KEYS` | P3-54 -- the keys added after the first snapshot contract |
| L1163-1168 | `snapshot_fft_state` | what each release added to the snapshot |
| L1244-1257 | `_build_plan_entry` | v5.16.2 / v5.33.2 -- the two routes to a single-buffer entry |
| L1286-1298 | `_build_plan_entry` | K4 -- one lock per SLOT, not one per entry |
| L1316-1322 | `_promote_entry_to_measure` | P13 -- the dropped dead ``entry`` parameter |
| L1437-1448 | `_get_or_make_plan` | P3-13 -- the known planner-under-lock stall |
| L1554-1566 | `_clear_local_asm_caches` | P3-55 -- which lock covers the pyFFTW structures, and why |
| L1577-1599 | `<module>` | the late-binding registry hook: v4.16.0 / v4.16.1 / v5.1.0 |
| L1633-1644 | `clear_asm_caches` | v4.16.0 -- the lazy-import fan-out replaced by the registry |
| L1682-1691 | `clear_asm_caches` | the "Historical notes" block -- which release chained which cache |
| L1880-1887 | `_get_or_make_bandlimit` | K8 -- multiply-by-reciprocal, not division, in the mask bins |
| L1946-1956 | `_h_cache_store` | K8 -- the cached H is made read-only |
| L2083-2097 | `set_fft_auto_promote` | v5.30.1 -- the versionchanged block for the opt-in flip |
| L2201-2206 | `_handle_pyfftw_failure` | P3-14 -- the read-test-then-add must be atomic |
| L2212-2220 | `_handle_pyfftw_failure` | S5-8 -- the removed pyfftw.interfaces.cache toggle |
| L2281-2288 | `_fft2` | W5 P2-26 -- the complex-only gate on the four dispatchers |
| L2588-2594 | `_validate_propagator_inputs` | 4.9 -- the > 1 mm pixel-pitch guard loosened to > 100 mm |
| L2630-2638 | `__all__` | v5.1.0 -- which backend flags are deliberately out of __all__ |

---

### L5-7 -- `<module> docstring` -- the v5.1.0 propagation.py split

*Left in the source:* the same "what this module owns" lead-in, without the split narrative.

```text
v5.1.0 split (Agent C): the formerly-monolithic
``lumenairy/propagators/propagation.py`` is reorganised into six
submodules sharing this one infrastructure layer.  This module owns:
```

### L21-24 -- `<module> docstring` -- "no public behaviour changed in the split"

*Left in the source:* the re-export contract (still true), plus the pointer to this file.

```text
Public API contract: every name previously importable from
``lumenairy.propagators.propagation`` is re-exported there unchanged
(see ``propagation.py``).  No public behaviour changed in the v5.1.0
split -- it is a pure file-level refactor.
```

### L43-45 -- `<module>` -- why the CuPy probe was consolidated (P2-9)

*Left in the source:* the one-place rule and the audit id.

```text
# The probe, the first-use import and the isinstance test live in ONE place
# (audit 2026-09-11 TESTS-ARCH P2-9: five hand-copied pairs, one of which is
# the only place the accelerator-absent path could be tested).
```

### L71-76 -- `_is_cupy_array` -- the NumPy 2.x ``.device`` duck-type break

*Left in the source:* the rule itself -- isinstance, never hasattr -- because a "simplification" back to the duck-type test silently routes every NumPy array to CuPy.

```text
    Historically this module used ``hasattr(x, 'device')`` as a duck-type
    test for a CuPy device array.  That broke in NumPy 2.x: ``ndarray``
    now exposes ``.device`` as part of the Python Array API standard, so
    every NumPy array falsely tests as a CuPy array and gets routed
    through the (unusable without CUDA) CuPy FFT path.  Use ``isinstance``
    against the real CuPy type instead.
```

### L123-127 -- `<module>` -- why the knobs are registered (P2-5)

*Left in the source:* what registration buys, without the census that prompted it.

```text
# Every process-global knob below is registered with the central registry
# (audit 2026-09-11 TESTS-ARCH P2-5: 12 of the library's ~20 process globals
# live in this module, and none of them had a context-manager form or a
# reset).  Registration is what makes ``lumenairy.override(...)`` and the
# suite's autouse snapshot/restore fixture reach them.
```

### L138-148 -- `FFTW_THREADS` -- S5-8c -- the 8-thread oversubscription cap

*Left in the source:* the measured knee and the "default only, not a pin" rule, which is why set_fft_threads(n) still takes any count.

```text
# S5-8c (perf, audit AUDIT_V5_24_2): on a many-core box (> 8 physical cores)
# libfftw3 OVERSUBSCRIBES a single 2-D transform -- a 1024^2..2048^2 complex128
# FFT is measured 11-18% FASTER at 8 threads than at all 24, because the
# transform is memory-bandwidth bound past ~8 threads and the extra threads
# only add butterfly / barrier contention.  So the DEFAULT thread count is
# capped at ``_FFTW_DEFAULT_THREAD_CAP``.  This is NOT bit-identical to the
# all-core count (a different thread count changes the FFT reduction order at
# the LSB, ~1e-15 relative -- the same order-of-magnitude perturbation the
# opt-in ESTIMATE->MEASURE auto-promote introduces), so it is applied only to
# the DEFAULT; pass an explicit ``set_fft_threads(n)`` to pin ANY count,
# including all cores via ``set_fft_threads(<available_cpus()>)``.
```

### L196-206 -- `_PYFFTW_BAD_SHAPES` -- K7 -- the blacklist key is a triple, not a bare shape

*Left in the source:* what the key IS (a reader who sees the name "BAD_SHAPES" would otherwise assume a shape) and why the name no longer matches its contents.

```text
# K7 (audit 2026-09-11): entries are ``(shape, dtype.str, direction)``
# triples, not bare shapes.  Keyed on the bare shape, ONE complex128
# MemoryError at (512, 512) also blacklisted complex64 at the same shape
# -- half the memory, likely to succeed -- and the inverse direction,
# which has its own separate plan and buffer.  Measured: after a single
# simulated complex128 failure at (512, 512) the blacklist was
# ``{(512, 512)}`` and a subsequent complex64 transform skipped pyFFTW.
# The key now matches the plan cache's own
# ``(direction, shape, dtype, threads)``, minus ``threads`` (an
# allocation failure is not thread-count-specific).  The name is kept
# for the reset / snapshot machinery and the tests that reference it.
```

### L303-323 -- `DEFAULT_* knobs` -- the v4.16.2 -> v5.1.0 per-knob consumer-rollout status table

*Left in the source:* nothing: which entry point reads which knob is documented on the setters themselves, and the rollout dates are the history.  The Multiprocess / fork notes below are NOT history -- they are a current-behaviour contract and stay in full.

```text
# v4.16.3 (audit P2-NEW-F1-3 + P2-NEW-F1-4) status of consumer rollout:
#
#   * ``DEFAULT_REAL_DTYPE``       -- consumed at one site
#     (``propagate_ensemble``'s no-input-dtype real-accumulator
#     fallback, ``ensemble.py:~347``).  v4.16.2 shipped this consumer
#     behind an unreachable ``except`` branch; v4.16.3 re-shapes the
#     consumer so the ``in_dtype is None`` path is the canonical
#     fallback.
#   * ``DEFAULT_WAVE_PROPAGATOR``  -- v5.1.0: consumer-wired across
#     ``apply_real_lens`` (``_lens_real.py``),
#     ``apply_real_lens_traced`` (``_lens_traced.py``), and the
#     ``method=`` argument of ``propagate_through_system``
#     (``system.py``).  v4.16.2 / v4.16.3 / v5.0.x shipped the SET /
#     GET API only and emitted a one-shot UserWarning that no
#     library consumer existed; v5.1.0 retires that warning and
#     wires the resolver at the entry points listed above.
#   * ``DEFAULT_DY``               -- v5.1.0: consumer-wired across
#     ``apply_real_lens`` and ``apply_real_lens_traced``.  Same
#     status history as ``DEFAULT_WAVE_PROPAGATOR``: API-only at
#     v4.16.2-v5.0.x, library-wide rollout at v5.1.0.
#
```

### L364-374 -- `DEFAULT_WAVE_PROPAGATOR_SHIPPED` -- W9-8 -- why the shipped default is a frozen CONSTANT

*Left in the source:* the whole rationale: it is load-bearing for how ``propagate()`` reads the knob.

```text
# v5.31 (audit W9-8): the SHIPPED value of the knob above, frozen at import and
# never reassigned.  ``propagate()`` compares against it to tell "the caller
# moved the library default" from "nobody touched it, so the value is still the
# factory one" -- it honours the knob only in the first case, because resolving
# it unconditionally would silently retire ``propagate``'s far-field
# auto-selection for every caller who never asked for that.  A CONSTANT rather
# than a "setter was called" latch on purpose: comparison is stateless, so
# restoring the knob with ``set_default_wave_propagator('asm')`` restores the
# behaviour with no cross-call or cross-process residue.
# ``propagate_through_system`` is unaffected -- it resolves the knob
# unconditionally, as it has since v5.1.0.
```

### L379-388 -- `_DEFAULT_*_NO_CONSUMER_WARNED` -- why the two one-shot latches are pinned True

*Left in the source:* the fact that they are inert and that flipping them back does nothing -- without it the two module globals look like live switches.

```text
# v4.16.3 (audit P2-NEW-F1-4): module-level latches that gated the
# "API-only in v4.16.2/v4.16.3; consumer wiring follows in v5.0"
# one-shot UserWarning emitted by the wave_propagator / dy setters.
# v5.1.0 retired both warnings (the resolver is now wired across
# ``apply_real_lens`` / ``apply_real_lens_traced`` /
# ``propagate_through_system``), so the setters no longer reach the
# emission branch.  The latch globals are preserved (pinned permanently
# to ``True``) for back-compat with code that introspects the module's
# attribute surface; flipping them back to ``False`` no longer revives
# the warning.
```

### L463-466 -- `set_default_wave_propagator` -- the retired pre-v5.1.0 "API-only" UserWarning

```text

    Pre-v5.1.0 the setter stored the value but no library consumer
    read it (API-only) and a one-shot ``UserWarning`` advertised the
    gap; v5.1.0 retired that warning.
```

### L511-514 -- `set_default_dy` -- the retired pre-v5.1.0 "API-only" UserWarning

```text

    Pre-v5.1.0 the setter stored the value but no library consumer
    read it (API-only) and a one-shot ``UserWarning`` advertised the
    gap; v5.1.0 retired that warning.
```

### L550-556 -- `_resolve_jax_complex_dtype` -- L2 -- why a central JAX dtype resolver exists

*Left in the source:* the failure mode it prevents, since a new JAX entry point that hard-casts is exactly how the bug comes back.

```text
    v4.13.0 (audit L2): JAX-side code historically hard-cast to
    ``jnp.complex64`` (or read ``jax.config.jax_enable_x64`` directly),
    which silently overrides the user's
    :func:`set_default_complex_dtype` setting and gives float32-precision
    answers with no warning.  This helper centralises the
    NumPy-default-dtype -> JAX-dtype mapping so every JAX entry point
    obeys the same configuration knob.
```

### L617-621 -- `_resolve_jax_real_dtype` -- L2 companion -- the real-dtype twin

*Left in the source:* the mapping itself, which is the contract.

```text
    v4.13.0 (audit L2 companion): real-valued JAX kernels (phase
    arrays, masks, prefactors) also need a precision twin to match the
    complex dtype.  Returns ``jnp.float64`` when the default complex
    dtype is ``np.complex128`` and ``jnp.float32`` for
    ``np.complex64``.
```

### L655-665 -- `_PYFFTW_PLAN_CACHE` -- why the single-slot plan cache became a multi-slot LRU

*Left in the source:* the callers that need several resident shapes -- the sizing argument for _PYFFTW_PLAN_CACHE_SIZE.

```text
# Multi-slot pyFFTW plan cache (3.2.14)
# ----------------------------------------------------------------------------
# Earlier the cache held *one* plan per direction (forward / inverse).
# That worked when a single call site dominated, but optimization
# loops, JonesField (Ex/Ey at one shape, then a 3D batch shape),
# Maslov (mixes the input grid and per-axis 1-D FFTs), and any code
# that propagates at multiple sizes thrashes the single slot --
# every call between two shapes has to reallocate the bound buffer
# and re-plan.  An LRU dict keyed by ``(direction, shape, dtype,
# threads)`` lets several recently-used plans stay resident, with
# bounded memory because old entries fall out the back of the LRU.
```

### L694-701 -- `_PYFFTW_SHARED_BUFFERS_UNSAFE` -- K4 -- the measured 100 %-wrong fields that the latch prevents

*Left in the source:* the hazard and the latch rule; the 8-thread x 40-call measurement is history.

```text
# The per-slot locks (audit K4) removed the entry-wide serialisation that
# had been masking this: measured on this box, 8 threads x 40 concurrent
# ``rayleigh_sommerfeld_propagate`` calls at (128, 128) / complex128 /
# z = 5 mm returned 7 fields with max|out - ref|/max|ref| = 1.14 (i.e.
# 100 % wrong) with per-slot locks and 0 with one shared lock per entry.
# ``angular_spectrum_propagate`` measured 0 / 320 either way.
#
# Fix: latch the first thread that reaches the plan cache; the instant a
```

### L743-761 -- `_PYFFTW_AUTO_PROMOTE` -- W9 -- why auto-promote stopped being the default

*Left in the source:* the reproducibility argument in two sentences, because it is the reason the shipped default is False and the reason to prefer set_pyfftw_planner.

```text
# v5.30.1 (audit W9): the DEFAULT is now False -- auto-promote is OPT-IN.
# It was on by default from 4.12 through v5.30, which made lumenairy
# silently non-reproducible, in two separate ways:
#
#   1. IN-PROCESS.  The switch happens mid-session, at whichever call
#      crosses the threshold at that key.  A caller doing N transforms per
#      user-level call sees its output change bits after ceil(5/N) calls
#      on one FIXED input -- measured on apply_real_lens_traced (4
#      transforms per call at one 256^2 key): calls 0-1 give one value,
#      calls 2+ a different one, max|d| ~ 2.8e-15.  Because 'calls' is
#      global state keyed on (direction, shape, dtype, threads), an
#      UNRELATED earlier caller at the same shape moves the boundary --
#      which is how this reached CI as a collection-order-dependent
#      failure of a byte-identity pin.
#   2. ACROSS PROCESSES.  FFTW_MEASURE picks its algorithm by TIMING
#      candidate plans at plan time, so the winner depends on machine
#      noise.  Measured: 4 fresh processes, same input, 4 DIFFERENT
#      post-promotion bit patterns -- while the ESTIMATE result was
#      identical in all 4.  Only ESTIMATE is a deterministic planner.
```

### L854-857 -- `reset_fft_backend` -- P3-14 -- clear the blacklist in place, do not rebind it

*Left in the source:* the invariant (in place, under the plan lock) and the race it closes.

```text
    # v5.4.6 (audit P3-14): clear the bad-shapes set IN PLACE under the
    # plan lock instead of rebinding the global, so a concurrent
    # _handle_pyfftw_failure cannot add to an orphaned set that is then
    # GC'd (silently dropping the blacklist entry).
```

### L870-874 -- `reset_fft_backend` -- S5-8 -- why pyfftw.interfaces.cache is not toggled here

*Left in the source:* the "deliberately absent" note, which is what stops the toggle being helpfully re-added.

```text
    # S5-8 (perf, no-loss): the real plan buffers live in
    # ``_PYFFTW_PLAN_CACHE`` (cleared above under the lock); the
    # ``pyfftw.interfaces.cache`` we used to disable/enable here is never
    # populated by lumenairy (raw ``pyfftw.FFTW`` plans only), so toggling it
    # freed nothing and only reset the idle keep-alive daemon.  Dropped.
```

### L908-914 -- `_PYFFTW_DOUBLE_BUFFER` -- v5.16.2 -- the ping-pong opt-out

*Left in the source:* what the switch trades, without the "pre-v4.12" framing.  The "~1-3 %" figure is deliberately dropped: the byte-cap block below measured it at ~65 % of the transform at N = 8192.

```text
# v5.16.2: opt-out for the v4.12 two-buffer ping-pong.  With the ping-pong,
# ``_fft2``/``_ifft2`` return one of two live workspace buffers with no copy
# (speed), at the cost of a SECOND resident full-grid aligned buffer per plan
# key (16 GiB/key at N=32768 complex128).  ``set_fft_double_buffer(False)``
# restores the pre-v4.12 single-buffer footprint; the dispatchers then return
# ``buf.copy()`` so results stay private -- byte-identical values, ~one extra
# array copy per FFT (~1-3% of a large transform).
```

### L917-921 -- `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` -- v5.33.2 -- the per-key byte cap: the 24.77 GB that prompted it

*Left in the source:* the MEASURED table and the derivation of the 2e9 constant stay (they are the bar's derivation, TESTING_STANDARDS S5); only the "the cache had no byte bound of any kind" framing moves.

```text
# v5.33.2 PER-KEY BYTE CAP on the ping-pong (audit
# AUDIT_TRACED_MEMORY_2026_08_09 rows 2 and 5.4).  ``_PYFFTW_DOUBLE_BUFFER``
# above is an all-or-nothing process switch, and the plan cache had no byte
# bound of any kind -- 8 KEYS, each holding TWO full-grid aligned workspaces,
# priced only in KEYS.  MEASURED retained after ONE design-121 order:
```

### L961-966 -- `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` -- v5.33.3 -- the two corrected figures (11586 -> 11181, 4.5 s -> 1.6 s)

*Left in the source:* the reason the threshold is DECIMAL 2e9, which is the part that constrains any future edit of the constant.

```text
# a common shape on the direction of a ``<=``.  (2 GiB would instead bind at
# N >= 11586, which is where this comment's own earlier binding figure came
# from; the constant is decimal, so 11181 is the one that is true.  The cost
# line likewise read "~4.5 s ... (0.5 %)" against the 1.6 s / 0.2 % its own
# measured table gives -- both corrected v5.33.3,
# VERIFY_PERF_BRANCH_2026_08_10 D6.)
```

### L1077-1079 -- `warmup_fft_plans` -- the ``threads`` default: the docstring contradicted the code

*Left in the source:* the CORRECT default.  The pre-relocation text said ``available_cpus``, which the F-32 fix at the call site had already replaced with ``FFTW_THREADS`` -- a stale doc line, the class the audit flags in sec. 15.7.

```text
    threads : int, optional
        Threads per plan.  Defaults to
        :func:`lumenairy._backends.available_cpus`.
```

### L1100-1104 -- `warmup_fft_plans` -- F-32 -- warm up on FFTW_THREADS, not _available_cpus()

*Left in the source:* the whole note: the plan key includes the thread count, so getting this wrong makes warmup a silent no-op.

```text
        # v5.4.6 (audit F-32): default to the FFTW_THREADS global that
        # _fft2/_ifft2 actually dispatch on, NOT _available_cpus().  The
        # plan cache key includes the thread count, so after
        # set_fft_threads(k) a warmup built at _available_cpus() threads
        # lands under a key the runtime never queries -- a silent no-op.
```

### L1115-1119 -- `_FFT_STATE_KEYS` -- P3-16 -- why snapshot/restore exists

*Left in the source:* the reason, in the present tense.

```text
# v5.4.6 (audit P3-16): the FFT/precision dispatch globals below are plain
# module globals, so a spawn-based worker re-imports this module at library
# defaults and silently loses any parent ``set_default_*`` / ``set_fft_*``
# overrides.  ``snapshot_fft_state`` / ``restore_fft_state`` let a caller
# carry the parent's configuration across the spawn boundary.
```

### L1124-1129 -- `_FFT_STATE_KEYS` -- P3-54 -- the keys added after the first snapshot contract

*Left in the source:* the consequence of a missing key, and the mixed-version tolerance.

```text
    # v5.17.1 (audit P3-54): setter-backed globals added after v5.4.6 that
    # spawned workers must inherit too -- without these a worker silently
    # reverts to double-buffered plans / default cache budgets, i.e. ~2x
    # the FFT workspace the parent's knobs were set to prevent.
    # restore_fft_state tolerates snapshots from older library versions
    # that lack these keys (mixed-version worker pools).
```

### L1163-1168 -- `snapshot_fft_state` -- what each release added to the snapshot

*Left in the source:* the return contract; the per-release key list is history.

```text
    worker.  See :func:`restore_fft_state`.  v5.4.6 (audit P3-16).

    v5.17.1 (audit P3-54): also captures the later-added knobs --
    ``USE_SCIPY_FFT``, :func:`set_fft_fallback`,
    :func:`set_fft_double_buffer`, :func:`set_fft_plan_cache_size` and
    the :func:`set_asm_cache_size` bounds.
```

### L1244-1257 -- `_build_plan_entry` -- v5.16.2 / v5.33.2 -- the two routes to a single-buffer entry

*Left in the source:* the two conditions and the copy contract, which is what the dispatchers branch on.

```text
    v5.16.2: when :func:`set_fft_double_buffer` disabled the ping-pong
    (``_PYFFTW_DOUBLE_BUFFER = False``), a SINGLE buffer + plan is built
    instead -- halving the resident aligned-workspace memory (one
    full-grid array per plan key; 16 GiB/key at N=32768 complex128,
    matching the pre-v4.12 single-buffer behaviour).  ``_fft2`` /
    ``_ifft2`` then return ``buf.copy()`` instead of the live buffer, so
    results stay private (byte-identical values; ~one extra copy per
    FFT).

    v5.33.2: the same single-buffer entry is built, per key, whenever the
    ping-pong would exceed :data:`_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` -- see
    :func:`_plan_entry_n_bufs`.  The dispatchers read the entry's buffer
    COUNT (not the global switch) to decide whether to copy, so a
    byte-capped key and a globally-disabled one behave identically.
```

### L1286-1298 -- `_build_plan_entry` -- K4 -- one lock per SLOT, not one per entry

*Left in the source:* the invariant and the hazard it does guard; the measured serialisation is history.

```text
        # K4 (audit 2026-09-11): ONE LOCK PER SLOT, not one per entry.
        # Each pyFFTW plan is bound to its own buffer and the ping-pong
        # slot index is advanced under ``_PYFFTW_PLAN_LOCK``, so two
        # threads at the same key always receive DIFFERENT plans and
        # DIFFERENT buffers -- there is nothing for them to race on.  A
        # single entry-wide lock nevertheless serialised them: measured
        # max simultaneous threads inside the pyFFTW critical section =
        # 1 with 4 threads x 6 calls on one (1024, 1024) complex128
        # shape, i.e. the double buffer could never deliver any
        # concurrency at all.  The per-slot lock still guards the one
        # real hazard -- ``pyfftw.FFTW.__call__`` on the SAME buffer --
        # which only arises when two callers wrap around to the same
        # slot.
```

### L1316-1322 -- `_promote_entry_to_measure` -- P13 -- the dropped dead ``entry`` parameter

*Left in the source:* a one-line versionchanged, which is the part a caller needs.

```text
    .. versionchanged:: 5.30
        Dropped the leading ``entry`` parameter (audit P13): despite the
        "in-place" wording it was never read -- the function builds a
        FRESH entry from ``(direction, shape_t, dt, threads)`` and the
        single call site copies ``entry['calls']`` across itself.
        Module-private (underscore, one caller inside this module), so no
        public signature changed.
```

### L1437-1448 -- `_get_or_make_plan` -- P3-13 -- the known planner-under-lock stall

*Left in the source:* the limitation itself (it is still present) and the shape of the fix, without the release it was deferred to.

```text
                    #
                    # v5.4.6 (audit P3-13): KNOWN LIMITATION (perf, not
                    # correctness).  The FFTW_MEASURE planner here can take
                    # 100-1000 ms on 4k+ grids and runs while holding
                    # _PYFFTW_PLAN_LOCK, so every concurrent _fft2/_ifft2
                    # blocks for that window.  The proper fix is a
                    # double-checked lock: stash a "promote requested"
                    # marker, drop the lock, build the new entry, then
                    # reacquire and swap only if the slot is unchanged.
                    # That concurrency-sensitive refactor is deferred to
                    # v5.5; single-thread / single-process callers (the
                    # common case) are unaffected.
```

### L1554-1566 -- `_clear_local_asm_caches` -- P3-55 -- which lock covers the pyFFTW structures, and why

*Left in the source:* the lock rule and the no-nesting rule, which any future edit here must obey.

```text
    v5.17.1 (audit P3-55): the two pyFFTW structures are cleared under
    ``_PYFFTW_PLAN_LOCK`` -- the lock that serialises every other
    mutation of them (``_get_or_make_plan``, ``_handle_pyfftw_failure``,
    ``reset_fft_backend`` per the v5.4.6 P3-14 fix).  Clearing them
    under ``_ASM_CACHE_LOCK`` let this clearer empty the plan cache
    between ``_get_or_make_plan``'s membership check and its indexing
    (both performed while HOLDING the plan lock), raising an uncaught
    ``KeyError`` out of ``_fft2`` in a concurrent clear.  The two locks
    are acquired SEQUENTIALLY (never nested) so no lock order is
    established with any other holder (``restore_fft_state`` ->
    ``set_fft_plan_cache_size`` takes the plan lock alone;
    ``reset_fft_backend`` releases it before calling
    ``clear_asm_caches``).
```

### L1577-1599 -- `<module>` -- the late-binding registry hook: v4.16.0 / v4.16.1 / v5.1.0

*Left in the source:* why the entry is a lambda and why it resolves through ``propagation`` -- both are load-bearing for the monkey-patch points people rely on.

```text
# v4.16.0 (ROADMAP #15): register the local-ASM clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
#
# v4.16.1 (audit P1-NEW-F1-2 / C.5): late-binding lambda matching
# the canonical pattern used by the other 8 cache-owning modules.
# The registered entry re-resolves ``_clear_local_asm_caches`` from
# the module's current namespace at call time -- this preserves the
# pre-v4.16 ``mock.patch.object`` semantic where tests that monkey-
# patch the clear-function still observe their counter increment
# when ``clear_asm_caches`` walks the registry.  Pre-v4.16.1 the
# registration captured the function object directly (early-binding),
# which silently bypassed ``mock.patch.object`` -- a tester-visible
# inconsistency vs the other 8 caches.  The cost is one attribute
# lookup per cache per drain (negligible vs the cache-clear work).
#
# v5.1.0 (Agent C, propagation.py split): the registry entry must
# remain registered under the legacy module path so external code that
# monkey-patches ``lumenairy.propagators.propagation._clear_local_asm_caches``
# still observes the patch.  ``propagation`` re-exports
# ``_clear_local_asm_caches`` from this module, and the registry hook
# below resolves the attribute on the ``propagation`` module
# namespace so the legacy patch path stays intact.
```

### L1633-1644 -- `clear_asm_caches` -- v4.16.0 -- the lazy-import fan-out replaced by the registry

*Left in the source:* how it works now (walks the registry) and the swallowed-exception contract.

```text
    v4.16.0 retires the lazy-import fan-out that this function used
    pre-v4.15 in favour of the central cache-clearer registry (see
    :mod:`lumenairy._cache_registry`).  Each cache-owning module
    registers its own clear function at import time via
    :func:`lumenairy.register_cache_clearer`, and this function simply
    walks the registry.

    The external contract is preserved bit-for-bit: every cache that
    was drained pre-v4.16 is still drained, and the same narrowed-
    except classes (``ImportError``, ``RuntimeError``,
    ``AttributeError``) are swallowed so a partial install does not
    strand the rest of the chain.
```

### L1682-1691 -- `clear_asm_caches` -- the "Historical notes" block -- which release chained which cache

*Left in the source:* nothing: the live list of drained caches is above it and is the contract.

```text

    Historical notes:
        The original 3.2.14 perf-pass only cleared the first three
        local caches.  v4.12.2 extended this to drop the pyFFTW plan
        cache + bad-shape memo.  v4.14.1 chained the LG/HG mode-stack
        and wrapper-merit caches.  v4.14.2 (AUDIT_V4_14_1_2026_05_17
        P1-NEW-3 / Agent C) chained the five additional sibling caches.
        v4.14.3 chained the LG polynomial coefficient cache (8th
        sibling).  v4.16.0 retires the lazy-import fan-out entirely
        in favour of the central registry.
```

### L1880-1887 -- `_get_or_make_bandlimit` -- K8 -- multiply-by-reciprocal, not division, in the mask bins

*Left in the source:* the rule and the reason (1-ULP label mismatch against the H grids); the 400-trial latency measurement is history.

```text
    # audit P1: integer DC anchor, matching _get_or_make_freq_grids.
    # K8 (audit 2026-09-11): and the SAME multiply-by-reciprocal
    # expression, not a division.  These masks label the bins of an H
    # built from ``_get_or_make_freq_grids``, and the two forms differ by
    # up to 1 ULP whenever ``1/(N*d)`` is not exactly representable -- a
    # mask/kernel label mismatch.  Latent only (0 flipped mask bins in
    # 400 randomised (N, dx, lambda, z) trials), but there is no reason
    # for two expressions where one will do.
```

### L1946-1956 -- `_h_cache_store` -- K8 -- the cached H is made read-only

*Left in the source:* the invariant and the one public exception (get_asm_transfer_function copies), because a caller who needs to write must know to copy.

```text
    # K8 (audit 2026-09-11): ``_h_cache_lookup`` hands the STORED array
    # back by reference (no copy -- that is the point of the cache), and
    # the internal consumers (``angular_spectrum_propagate``,
    # ``angular_spectrum_propagate_batch``, ``shack_hartmann``,
    # ``rayleigh_sommerfeld_propagate``) hold the live object.  The
    # convention "callers must not mutate it in place" was a comment;
    # make it an enforced invariant at zero cost.  Measured pre-fix: two
    # successive ``_get_asm_H_natural`` calls at the same key returned
    # arrays for which ``np.shares_memory(...) is True`` and both were
    # writeable.  The public ``get_asm_transfer_function`` already
    # copies, so its return stays writeable.
```

### L2083-2097 -- `set_fft_auto_promote` -- v5.30.1 -- the versionchanged block for the opt-in flip

*Left in the source:* a condensed versionchanged: users need to know the default moved and why.

```text
    .. versionchanged:: 5.30.1
        **The default is now** ``False`` **(opt-in)**; it was ``True``
        from 4.12 through v5.30.  Auto-promote is not reproducible, and
        it was on by default: (1) it swaps the plan MID-SESSION, so one
        FIXED input returns one bit pattern before the threshold call
        and a different one after (measured ~2.8e-15 on a 256^2 traced
        lens: calls 0-1 vs calls 2+), and the counter is GLOBAL per
        ``(direction, shape, dtype, threads)`` key, so an unrelated
        earlier caller at the same shape moves the boundary; (2)
        ``FFTW_MEASURE`` selects its algorithm by timing candidates at
        plan time, so the winner -- and therefore the output bits --
        varies with machine noise (measured: 4 fresh processes, 4
        distinct post-promotion results, where ESTIMATE gave one).
        Neither result is more accurate; the default now favours
        reproducibility.
```

### L2201-2206 -- `_handle_pyfftw_failure` -- P3-14 -- the read-test-then-add must be atomic

*Left in the source:* the atomicity requirement and the no-deadlock argument for taking the non-reentrant lock here.

```text
    # v5.4.6 (audit P3-14): the read-test-then-add must be atomic under the
    # plan lock, else two threads failing on the same key both observe
    # was_new=True (duplicate warnings) and a concurrent reset can swap the
    # binding underfoot.  This handler runs OUTSIDE the plan-lookup lock
    # (it is called from the _fft2/_ifft2 execution except-blocks), so
    # acquiring the (non-reentrant) lock here does not deadlock.
```

### L2212-2220 -- `_handle_pyfftw_failure` -- S5-8 -- the removed pyfftw.interfaces.cache toggle

*Left in the source:* what the user should do instead, which is the actionable half.

```text
        # S5-8 (perf, no-loss): we no longer toggle
        # ``pyfftw.interfaces.cache`` here.  That cache belongs to the
        # ``pyfftw.interfaces.*`` wrapper API, which lumenairy never uses (raw
        # ``pyfftw.FFTW`` plans only), so it was always empty for us -- the
        # disable/enable freed no failed buffers despite the old comment's
        # claim.  This shape is recorded in ``_PYFFTW_BAD_SHAPES`` above, so
        # subsequent calls at this shape route straight to scipy/numpy; call
        # ``reset_fft_backend()`` to drop the resident aligned plan buffers
        # in ``_PYFFTW_PLAN_CACHE`` once the memory pressure has passed.
```

### L2281-2288 -- `_fft2` -- W5 P2-26 -- the complex-only gate on the four dispatchers

*Left in the source:* what the gate is for, in one line, at each of the four sites.

```text
    # v5.17.x (audit W5 P2-26 hardening): the pyFFTW path is complex-to-
    # complex only -- a real-dtype input used to reach _get_or_make_plan,
    # fail (ValueError), and permanently poison the bare-shape
    # _PYFFTW_BAD_SHAPES blacklist for ALL dtypes at that shape, with a
    # misleading "memory pressure" warning.  Gate on iscomplexobj here
    # (and in the three sibling dispatchers) so real input routes
    # directly to the scipy/numpy fallback -- correct result, no
    # blacklist poisoning -- regardless of how future callers cast.
```

### L2588-2594 -- `_validate_propagator_inputs` -- 4.9 -- the > 1 mm pixel-pitch guard loosened to > 100 mm

*Left in the source:* which bound raises and which only warns, and the telescope case that is the reason for the split.

```text
    # 4.9 fix (audit #4.3): the > 1 mm guard was over-strict.  Large
    # telescope pupils (Hale-class ground apertures, JWST-scale segment
    # arrays) are legitimately sampled at the mm scale.  Loosened to
    # > 100 mm (above which a unit-error is far more likely than a
    # genuine telescope-class problem); the legacy < 1 mm sanity floor
    # downgrades to a one-time RuntimeWarning so users who really do
    # have mm-scale pixel pitch see the warning once and proceed.
```

### L2630-2638 -- `__all__` -- v5.1.0 -- which backend flags are deliberately out of __all__

*Left in the source:* the rule (module-attribute-accessible, not public API) and the two exceptions, because it governs every future addition to this list.

```text
    # v5.1.0 (Wave-4 integration / V9 walker symmetry): module handles
    # ``cp`` / ``pyfftw`` and internal config flags (FFTW_MIN_SIZE /
    # FFTW_THREADS / PYFFTW_FALLBACK_ON_ERROR / SCIPY_FFT_AVAILABLE /
    # SCIPY_FFT_WORKERS / USE_PYFFTW / USE_SCIPY_FFT) are intentionally
    # NOT in __all__ -- they're module-attribute-accessible for power
    # users (``lumenairy.propagators.propagation.USE_PYFFTW = True``)
    # but not part of the public top-level API.  CUPY_AVAILABLE /
    # PYFFTW_AVAILABLE stay public because they're documented as
    # capability-probe flags in the README.
```

