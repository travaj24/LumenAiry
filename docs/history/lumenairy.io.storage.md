<!-- lumenairy-history-doc
module: lumenairy/io/storage.py
ast_sha256: dfd2c16e28614204ea52d48e1ff32a83280518beb16b25ebbd29d8d16094317e
token_sha256: f70dca9dbcacb2908c0dd68e238ebda914bf00207ab05c891b90ef6ad2f77378
pre_relocation_lines: 2159
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/io/storage.py`

This file holds the version-history narrative that used to live in
`lumenairy/io/storage.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file, so `git log -S`
on any phrase here still lands on the commit that wrote it.

`storage.py` carries three recurring shapes of history, and this document is
mostly those three:

* **the writer version stamp** -- nine sites all annotated `v4.15.0
  (P2-VERSTAMP)`.  The stamp is live behaviour; only the release tag moved, and
  the per-site reason for the stamp (group-level vs per-plane, first-touch vs
  re-open) stayed in the source at every one of them.
* **the metadata codec** -- five sites carrying the measured before-state of
  the raw h5py/zarr attribute loop the type-tagged codec replaced.  The
  MEASURED numbers over the module's own 19-type probe set stayed, because they
  are what tells a future editor why the codec cannot be simplified away; the
  "this SUPERSEDES the S4-9 `None`-skip" supersession narrative and the
  commented-out pre-fix code moved here.
* **the append atomicity contract** -- bump-then-create, roll back on failure,
  never overwrite.  The reason the reverse order is wrong is live (it strands
  an orphan that collides with the next append), so it stayed, re-stated in the
  present tense; the release tags moved.

What deliberately did NOT move: the four `'auto'` compression / chunk-size
docstrings' **migration statements** (`(v5.46 default)`, "pass
`compression='gzip'` explicitly to restore the pre-v5.46 behaviour",
"(the historical value)").  Those tell a user upgrading what to do now, which
is current-behaviour documentation, not narrative -- the same line part 1 drew
around `set_pyfftw_planner`'s migration note.  The measured compression
numbers (57.9x write, 7.1x read, 5.6 % of space) stayed for the same reason
`docs/TESTING_STANDARDS.md` S5 keeps a derivation next to its bar.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L138-148 | `_ZARR_MKDIR_PATCH_LOCK` | the "L8 fix (v4.13.0)" framing on the mkdir-patch race |
| L155-160 | `_get_lumenairy_version` | the release and the audit item that introduced the writer stamp |
| L206-206 | `<module> metadata contract header` | the release tag on the section header |
| L209-220 | `<module> metadata contract` | the two fidelity bugs of the pre-contract scheme -- list/ndarray coercion and un-reversed dict flattening |
| L230-233 | `<module> metadata contract` | the word "historical" describing the flat fallback |
| L561-562 | `save_field_h5` | the release tag on the writer stamp |
| L568-578 | `save_field_h5` | the commit reference, the "SUPERSEDES the storage-nit None-skip" supersession narrative, and the two commented-out lines of pre-fix code |
| L655-656 | `save_planes_h5` | the release tag on the group-level stamp |
| L660-663 | `save_planes_h5` | the audit id and the repeated measured before-state |
| L704-704 | `save_planes_h5` | the release tag |
| L798-799 | `save_jones_field_h5` | the release tag |
| L805-811 | `save_jones_field_h5` | the S4-9 supersession narrative and the measured before-state of the raw loop |
| L878-887 | `append_plane` | the release and audit that made nested metadata faithful, and what h5py attrs did with it before |
| L904-907 | `append_plane` | the old fixed 1024-edge chunk and the 16 MiB chunk it produced |
| L912-912 | `append_plane` | "v4.13.0 onward" on the preserve_dtype flag |
| L1020-1021 | `append_plane_h5` | the release tag |
| L1044-1051 | `append_plane_h5` | the release and audit id on the atomicity fix |
| L1058-1058 | `append_plane_h5` | the release tag |
| L1064-1070 | `append_plane_h5` | the audit id and the S4-9 supersession narrative |
| L1072-1072 | `append_plane_h5` | the release tag on the SWMR block |
| L1099-1099 | `append_plane_h5` | the release tag on the unconditional flush |
| L1112-1115 | `append_plane_h5` | the release/audit tag and the "(overwrite was removed in v4.14.3)" aside |
| L1312-1312 | `TempFieldStore.save` | the release tag |
| L1430-1438 | `_open_zarr_group_safe` | the "L8 fix, v4.13.0" framing |
| L1499-1507 | `append_plane_zarr` | that the Zarr path once passed `overwrite=True`, and which release dropped it |
| L1573-1573 | `append_plane_zarr` | the release tag |
| L1582-1586 | `append_plane_zarr` | the release and audit id |
| L1593-1593 | `append_plane_zarr` | the release tag |
| L1599-1609 | `append_plane_zarr` | the audit id and the "pre-A-4 raw loop" framing |
| L1751-1754 | `set_storage_backend` | that the zarr-availability check used to be lazy |
| L1800-1813 | `_detect_backend` | the release and audit id, and the "Pre-v4.16.1 any directory routed to Zarr" framing |
| L2030-2038 | `replay_run` | the release/audit tag and the pre-fix behaviour (always reporting 0.0) |
| L2097-2102 | `replay_run` | the release/audit tag and the "pre-fix replay_run ignored it entirely" sentence |

---

### L138-148 -- `_ZARR_MKDIR_PATCH_LOCK` -- the "L8 fix (v4.13.0)" framing on the mkdir-patch race

*Left in the source:* the entire race description, re-stated as what happens WITHOUT the lock -- both interleavings, and the permanent `Path.mkdir` corruption they produce.  That is the only argument for the lock's existence, so it cannot leave the source.

```text
# Module-level lock guarding the ``Path.mkdir`` monkey-patch inside
# :func:`_open_zarr_group_safe`.  Two threads racing through
# ``append_plane_h5`` -> ``_open_zarr_group_safe`` previously could
# leave the patch installed indefinitely if one thread restored its
# saved ``_orig_mkdir`` while the other was still running, or both
# could call ``_orig_mkdir = _PL.mkdir`` simultaneously and end up
# saving the patched version as the "original" -- making the patch
# permanent and corrupting every later mkdir call.  L8 fix (v4.13.0):
# serialise the patch install/restore window.  The lock is process-
# scoped (one per import) so it imposes no overhead on the common
# single-threaded code path.
```

### L155-160 -- `_get_lumenairy_version` -- the release and the audit item that introduced the writer stamp

*Left in the source:* what the stamp is for (a future reader distinguishing versions, migration paths, flagging a known-buggy release), the lazy-import rationale, and the 'unknown' fallback.

```text
    v4.15.0 (P2-VERSTAMP from v4.14.2 audit): every HDF5
    ``create_dataset`` and Zarr ``create_array`` site stamps a
    ``lumenairy_version`` attribute so a future reader can
    distinguish data written by different library versions
    (e.g. for migration paths around schema changes or to flag
    data written by a known-buggy release).  Lazy import + cached
```

### L206-206 -- `<module> metadata contract header` -- the release tag on the section header

*Left in the source:* the section header and the whole contract under it.

```text
# Canonical nested-metadata serialization contract (S4-19, v5.24.x)
```

### L209-220 -- `<module> metadata contract` -- the two fidelity bugs of the pre-contract scheme -- list/ndarray coercion and un-reversed dict flattening

*Left in the source:* the contract itself: one type-tagged JSON blob under one reserved key, byte-identical in both backends, with a best-effort flattened copy alongside it for external inspectors.

```text
# ``write_sim_metadata`` / ``read_sim_metadata`` used to lower each
# metadata value directly into a native h5py / zarr attribute.  Two
# fidelity bugs followed from that (audit S4-19):
#
#   * **list -> ndarray coercion** -- a native attribute cannot tell a
#     Python ``list`` from a NumPy ``ndarray``; both round-tripped to the
#     same type, so the ``list`` vs ``ndarray`` distinction was lost.
#   * **un-reversed dict flattening** -- nested dicts were flattened to
#     ``"parent.child"`` keys on write but never re-nested on read, so a
#     round-trip returned a *flat* dict, not the caller's structure.
#
# The fix is one canonical, backend-agnostic serialization: the whole
```

### L230-233 -- `<module> metadata contract` -- the word "historical" describing the flat fallback

*Left in the source:* the whole back-compat rule (a blob-less file still loads, reconstructed from the native attributes) plus the pointer to this document.

```text
# BACK-COMPAT: a file with no blob (written by the pre-contract scheme)
# still loads -- the reader detects the missing blob and falls back to
# reconstructing the mapping from the individual native attributes,
# reproducing the historical (flat) behavior exactly.
```

### L561-562 -- `save_field_h5` -- the release tag on the writer stamp

*Left in the source:* what the stamp buys and the pointer to _get_lumenairy_version.

```text
        # v4.15.0 (P2-VERSTAMP): writer version stamp for migration /
        # known-buggy-release traceability.  See _get_lumenairy_version.
```

### L568-578 -- `save_field_h5` -- the commit reference, the "SUPERSEDES the storage-nit None-skip" supersession narrative, and the two commented-out lines of pre-fix code

*Left in the source:* why the codec is used instead of raw attrs, and the measured 19/19 vs 14/4/1/7 fidelity of the two -- the number that stops the codec being 'simplified' back to an attribute loop.

```text
        # A-4 follow-up (recorded in d045980, closed 2026-07-26): route
        # ``metadata`` through the canonical type-tagged codec instead of
        # handing raw values to h5py attrs.  Measured over the module's
        # 19-type probe set, the raw loop wrote 14 without raising / 4
        # raised (heterogeneous list, flat + nested dict, arbitrary
        # object) / 1 was silently DROPPED (``None``) / 7 came back a
        # different type; the codec round-trips 19/19.  This SUPERSEDES
        # the storage-nit ``None``-skip below it (the blob stores ``None``
        # faithfully, which is what the zarr backend already did):
        #     if value is None: continue        # <- dropped the value
        #     dset.attrs[str(key)] = value      # <- raised on containers
```

### L655-656 -- `save_planes_h5` -- the release tag on the group-level stamp

*Left in the source:* the scope of the stamp and its relation to the per-dataset stamps.

```text
        # v4.15.0 (P2-VERSTAMP): group-level stamp covers the whole
        # batch, complementing per-dataset stamps below.
```

### L660-663 -- `save_planes_h5` -- the audit id and the repeated measured before-state

*Left in the source:* the rule (run-level metadata goes through the codec) and a pointer to the site that carries the measurement.

```text
        # A-4 follow-up (see save_field_h5): the run-level ``metadata``
        # mapping goes through the canonical codec, not raw h5py attrs.
        # Same measured before-state as its siblings (14 wrote / 4 raised /
        # ``None`` dropped / 7 coerced over the 19-type probe set).
```

### L704-704 -- `save_planes_h5` -- the release tag

*Left in the source:* the stamp's scope.

```text
            # v4.15.0 (P2-VERSTAMP): per-plane writer-version stamp.
```

### L798-799 -- `save_jones_field_h5` -- the release tag

*Left in the source:* why a single group-level stamp is right for a Jones pair.

```text
        # v4.15.0 (P2-VERSTAMP): single group-level stamp for the
        # Jones-field pair so the (Ex, Ey) provenance is consistent.
```

### L805-811 -- `save_jones_field_h5` -- the S4-9 supersession narrative and the measured before-state of the raw loop

*Left in the source:* the rule and the one live consequence: the codec stores `None` rather than dropping the key, matching zarr.

```text
        # A-4 follow-up (see save_field_h5).  This SUPERSEDES the S4-9
        # ``None``-skip that used to live here: S4-9's stated goal was to
        # stop the h5py C-layer crash and "match zarr, which stores
        # ``None`` fine", which it reached by DROPPING the key; the codec
        # reaches it by actually storing ``None``.  Measured before-state
        # over the 19-type probe set: 14 wrote without raising / 4 raised /
        # 1 silently dropped (``None``) / 7 type-coerced -> now 19/19.
```

### L878-887 -- `append_plane` -- the release and audit that made nested metadata faithful, and what h5py attrs did with it before

*Left in the source:* the whole current contract -- which Python types survive a round-trip, through which loaders, and that a flattened copy is still written for external tools.

```text
        nested mapping: v5.29.1+ (audit A-4) serialises it through the
        same canonical type-tagged JSON codec ``write_sim_metadata``
        uses, so ``None``, ``bytes``, ``complex``, ``tuple``, empty and
        heterogeneous lists, and nested dicts all survive the round-trip
        through :func:`load_planes` / :func:`load_planes_h5` as the exact
        Python types written.  Pre-v5.29.1 the values were handed
        straight to h5py attrs, which raised on nested / heterogeneous
        containers and silently dropped ``None``.  A flattened
        native-attr copy of the scalars is still written for external
        inspection tools.
```

### L904-907 -- `append_plane` -- the old fixed 1024-edge chunk and the 16 MiB chunk it produced

*Left in the source:* the live sizing rule and its reason: 1 MiB is HDF5's own default chunk-cache size, so a chunk above it makes a partial read touch more than the cache holds.

```text
        <= 1 MiB -- 256 for complex128, 362->256 for complex64.  The old
        fixed ``1024`` made a **16 MiB** chunk for complex128, sixteen
        times HDF5's 1 MiB default chunk cache, so every partial read
        touched a whole 16 MiB chunk.
```

### L912-912 -- `append_plane` -- "v4.13.0 onward" on the preserve_dtype flag

*Left in the source:* the flag's meaning and the cross-reference to save_field_h5.

```text
        :func:`save_field_h5` for the same flag.  v4.13.0 onward.
```

### L1020-1021 -- `append_plane_h5` -- the release tag

*Left in the source:* the whole first-touch rule, including why the stamp is NOT overwritten on re-open and how a multi-version append is still resolvable from the per-dataset attr.

```text
                # v4.15.0 (P2-VERSTAMP): group-level stamp on
                # first-touch.  Note we don't overwrite the stamp on
```

### L1044-1051 -- `append_plane_h5` -- the release and audit id on the atomicity fix

*Left in the source:* the full rule and, crucially, why the reverse order is wrong -- an orphan dataset that collides with the next append's computed name.  Re-stated in the present tense.

```text
            # v4.14.3 (P0-NEW-1): reserve the slot atomically by
            # bumping ``n_planes`` BEFORE the dataset is created.  If
            # ``create_dataset`` crashes (disk full, dtype mismatch,
            # etc.) the increment is rolled back so a subsequent
            # append re-uses this same slot rather than skipping it.
            # The reverse order (create, then bump) left an orphan
            # dataset on crash that collided with the next append's
            # computed name.
```

### L1058-1058 -- `append_plane_h5` -- the release tag

*Left in the source:* the stamp's scope.

```text
                # v4.15.0 (P2-VERSTAMP): per-plane writer stamp.
```

### L1064-1070 -- `append_plane_h5` -- the audit id and the S4-9 supersession narrative

*Left in the source:* the rule and both live consequences: `None` is stored faithfully, and nested / heterogeneous / empty containers no longer raise.

```text
                # A-4 (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25): route
                # ``metadata`` through the canonical type-tagged codec
                # instead of handing raw values to h5py attrs.  This
                # SUPERSEDES the S4-9 ``None``-skip: the blob stores
                # ``None`` faithfully (matching zarr, which was S4-9's
                # stated goal) instead of dropping it, and nested /
                # heterogeneous / empty containers no longer raise.
```

### L1072-1072 -- `append_plane_h5` -- the release tag on the SWMR block

*Left in the source:* the entire SWMR ordering contract -- schema work must finish before `swmr_mode = True`, the flag is per-open while `libver` is persisted, and the try/except rationale.

```text
                # v4.16.0 SWMR: enable single-writer-multiple-reader
```

### L1099-1099 -- `append_plane_h5` -- the release tag on the unconditional flush

*Left in the source:* why the flush is unconditional.

```text
                # v4.16.0: flush in all modes so even the non-SWMR
```

### L1112-1115 -- `append_plane_h5` -- the release/audit tag and the "(overwrite was removed in v4.14.3)" aside

*Left in the source:* the whole orphan-cleanup rule, with the aside re-stated as the live invariant it actually is: this writer never overwrites.

```text
                # v5.4.6 (audit P3-15): if create_dataset SUCCEEDED but a
                # later attrs/swmr/flush step raised, the orphan plane_NN
                # dataset would block the next append at the same name
                # (overwrite was removed in v4.14.3).  Delete it so the
```

### L1312-1312 -- `TempFieldStore.save` -- the release tag

*Left in the source:* why the temp store stamps at all -- a crash-recovered process detecting cross-version temp files in a reused temp dir.

```text
            # v4.15.0 (P2-VERSTAMP): temp-store stamps too, so a
```

### L1430-1438 -- `_open_zarr_group_safe` -- the "L8 fix, v4.13.0" framing

*Left in the source:* the entire race and its consequence, re-stated as what happens without the lock, plus the contention note.

```text
    Thread-safety (L8 fix, v4.13.0): the install / restore pair is
    serialised through a module-level :class:`threading.Lock`
    (``_ZARR_MKDIR_PATCH_LOCK``).  Two threads racing through
    ``append_plane_h5`` -> ``_open_zarr_group_safe`` previously could
    save the patched ``mkdir`` as the "original" and never restore
    the real implementation, permanently corrupting ``Path.mkdir`` for
    the whole process.  The lock is contended only on Windows where
    the patch is needed at all, and is uncontended on the (vastly more
    common) single-threaded code path.
```

### L1499-1507 -- `append_plane_zarr` -- that the Zarr path once passed `overwrite=True`, and which release dropped it

*Left in the source:* the live contract: no `overwrite=True`, attribute-before-create, and a stale `plane_NN` at the computed slot raises instead of being clobbered.

```text
    fix: bump ``n_planes`` first, roll back on create failure.  In
    addition the Zarr code path historically passed
    ``overwrite=True`` to ``create_array``, which combined with the
    pre-fix attribute-after-create order meant a crashed-then-restarted
    appender would *silently clobber* the orphan dataset.  v4.14.3
    drops ``overwrite=True``: the attribute-before-create order leaves
    the orphan slot reserved (next append computes ``n + 1``), and if
    a stale ``plane_NN`` already exists at the computed slot the
    create now raises rather than silently overwriting.
```

### L1573-1573 -- `append_plane_zarr` -- the release tag

*Left in the source:* the initial-create-only rule and its relation to the per-plane stamps.

```text
            # v4.15.0 (P2-VERSTAMP): mirror the HDF5 group-level
```

### L1582-1586 -- `append_plane_zarr` -- the release and audit id

*Left in the source:* the discipline itself and what `overwrite=False` buys.

```text
        # v4.14.3 (P0-NEW-1): same attribute-before-create discipline
        # as the HDF5 path.  Roll back on any create failure so the
        # slot is not stranded.  ``overwrite=False`` (the default) is
        # now required to detect-and-raise on a stale orphan rather
        # than silently clobber it.
```

### L1593-1593 -- `append_plane_zarr` -- the release tag

*Left in the source:* the stamp.

```text
            # v4.15.0 (P2-VERSTAMP): per-plane stamp.
```

### L1599-1609 -- `append_plane_zarr` -- the audit id and the "pre-A-4 raw loop" framing

*Left in the source:* the rule and the measurement that justifies it -- 13/19 vs 18/19, and the ndarray -> str() truncation that loses the values outright.  Also kept: the real hazard, that the same call had different fidelity depending on a global backend switch.

```text
            # I6 (AUDIT_ADVERSARIAL_EXHAUSTIVE 2026-09-11): route per-plane
            # metadata through the SAME type-tagged codec every HDF5 write
            # site uses (the A-4 / S4-19 contract) instead of the pre-A-4 raw
            # loop.  Measured over the module's own 19-type probe set, the raw
            # loop kept 13/19 faithful against HDF5's 18/19: complex -> str,
            # bytes -> str, tuple -> list, np scalar -> str, and -- the
            # irrecoverable one -- ndarray -> str(), which inserts ``...``
            # beyond numpy's 1000-element print threshold, so the values were
            # GONE.  ``_zarr_write_sim_metadata`` already used the codec, so
            # the same call with the same arguments had different fidelity
            # depending on a global backend switch.
```

### L1751-1754 -- `set_storage_backend` -- that the zarr-availability check used to be lazy

*Left in the source:* the live reason for checking eagerly: deferring the failure to the first `append_plane` call is much harder to debug in a long-running simulation.

```text
    raises ``ImportError`` immediately.  Previously the check was
    lazy: ``set_storage_backend`` succeeded and the missing library
    only surfaced on the first ``append_plane`` call, which is harder
    to debug in a long-running simulation.
```

### L1800-1813 -- `_detect_backend` -- the release and audit id, and the "Pre-v4.16.1 any directory routed to Zarr" framing

*Left in the source:* both details as current-behaviour notes, including the stale `output.h5/` hazard that is the reason the store-marker check exists.  `tests/unit/test_v4_16_1_agent_a.py` pins the literals `'zarr.json'`, `'.zarray'` and `str(path)`, all of which are code and unaffected.

```text
    v4.16.1 (AUDIT_V4_16_0_DEEP P1-DEEP-3-1) fix:

    1. ``str(path)`` cast added so ``pathlib.Path`` inputs no longer
       raise ``AttributeError`` on ``.endswith``.  The dispatcher
       docstrings advertise Path support; this enforces it.

    2. Directory-routing restricted to actual Zarr stores via the
       canonical store-marker file (``zarr.json`` per the Zarr v3
       spec at the group root, or ``.zarray`` per Zarr v2 at the
       array root).  Pre-v4.16.1 any directory at ``path`` silently
       routed to the Zarr backend, so a stale ``output.h5/``
       directory next to ``output.h5`` would re-route subsequent
       reads to Zarr and silently produce a wrong-backend error
       trace.
```

### L2030-2038 -- `replay_run` -- the release/audit tag and the pre-fix behaviour (always reporting 0.0)

*Left in the source:* the full resolution order for the reported wavelength, which is the contract a caller needs.

```text
        Forwarded onto the result's ``wavelength`` field.  v5.24.x
        (audit S4-19): when omitted (``None``), the STORED wavelength is
        used -- the per-plane ``wavelength`` attribute of the last plane
        (preferred), then any per-plane value, then the file-level
        ``wavelength`` metadata -- falling back to ``0.0`` only when the
        store carries none.  ``append_plane`` has persisted per-plane
        wavelength since v4.0; pre-fix ``replay_run`` ignored it and the
        replayed result always reported ``0.0`` unless the caller
        re-supplied the wavelength by hand.
```

### L2097-2102 -- `replay_run` -- the release/audit tag and the "pre-fix replay_run ignored it entirely" sentence

*Left in the source:* the resolution order, restated at the code that implements it.

```text
    # v5.24.x (audit S4-19): resolve the reported wavelength from the
    # STORED per-plane attribute when the caller omitted it.  Prefer the
    # last plane (the exit-plane by convention), then any per-plane
    # value, then the file-level metadata; 0.0 only if the store carries
    # none.  ``append_plane`` has persisted per-plane wavelength since
    # v4.0, but pre-fix replay_run ignored it entirely.
```

