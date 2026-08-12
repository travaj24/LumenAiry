"""validation.pipeline -- a staged, resumable, config-driven driver for
Lumenairy field-aggregation workflows.

    decompose -> chains -> aggregate -> leg -> readout

WHAT IT IS FOR
--------------
``propagate_traced_carrier_chain`` runs ONE ray congruence: its entrance->exit
map is single-congruence by construction, so a fan of DOE orders is a fan of
chains.  Doing anything COHERENT across those chains -- summing them,
measuring their crosstalk, handing them to a downstream leg once instead of K
times -- needs the per-beam fields brought onto ONE grid against ONE analytic
carrier first.  ``lumenairy.propagators.carrier_field`` provides the physics
primitives for that (``CarrierField``, ``re_reference``, ``aggregate``, the
Nyquist guard, Zarr IO).  THIS package is the run-level driver that composes
them: it decides what the beams are, carries each one to the summable plane,
checkpoints every stage, and resumes from any of them.

WHY IT IS RUNNER-SIDE (``validation/pipeline``) AND NOT ``lumenairy/workflows``
------------------------------------------------------------------------------
Adjudicated from this repo's own conventions, and the first reason is
decisive:

1.  **The chain-A cache key.**  ``_d121_common._chain_a_key`` hashes the
    CONTENT of every ``lumenairy/**/*.py`` -- deliberately, because defect D6
    (REVIEW_TRACED_EXACT_2026_08_05) was a cache keyed on less than the field
    depended on, and one commit flipped two library DEFAULTS that changed the
    cached field.  The consequence is stated in that file: *any edit anywhere
    in the library invalidates chain A*.  So putting this driver under
    ``lumenairy/**`` would orphan every design-121 chain-A cache in every tree
    on every edit of the DRIVER -- under running measurements.
    ``BUILD_CARRIER_FIELD_2026_08_11`` S0.1 records having to build in an
    isolated worktree for exactly this reason, and
    ``PROBE_SUM_AT_APERTURE`` S1.2 records the damage when it is not
    respected (a library edit mid-run moved a chain's absolute output phase by
    2.88 rad while every intensity metric held).

2.  **It cannot live in the library.**  ``pyproject.toml`` packages
    ``lumenairy*`` only, and the first decomposer needs the design-121
    ``.zmx``, ``tx_design_study_sim`` and the design-study runner's
    ``_NEW_GLASSES`` Sellmeier table -- LOCAL-ONLY absolute paths on a dev
    box.  A library module that imported those would be unimportable in CI and
    unshippable in a wheel.

3.  **The repo already draws this line.**  ``lumenairy/`` holds physics and
    per-call orchestration (including
    ``propagate_traced_carrier_chain_multi``, which recombines a fan in ONE
    call).  Run-level machinery -- campaign configuration, checkpoints,
    resume, grid-intent pre-flights, acceptance banners -- lives in
    ``validation/``: ``_grid_intent.py``, ``_d121_common.py``,
    ``fan_multi_121.py``, ``sumap_probe_121.py`` are all that, and this
    package generalises exactly those.  There is no ``lumenairy/workflows/``
    today and nothing in the library imports upward.

4.  **Blast radius.**  A staged driver is edited constantly during a
    campaign.  Runner-side, an edit costs nothing but this package's own
    artifact key (``artifacts.pipeline_source_sha``); library-side it would
    cost a release, a CHANGELOG entry and every field cache in the mesh.

The physics stays in the library: this package computes NOTHING itself.
Every apply-side operation is a lumenairy call.

WHAT THE STAGES ARE
-------------------
``decompose``  a system -> a beam list: initial ``CarrierSpec``-equivalents
               and complex amplitudes.  Pluggable; ``design121_doe`` is the
               first implementation (the Dammann cell's own DFT, which is the
               DOE's EXACT action), with ``sumap_cache`` and ``explicit``
               alongside.

``chains``     each beam, independently, to a NAMED plane.  Today the only
               legal plane is ``'fine_retrace_exit'`` -- the last group's back
               aperture ON THE EXACT LEG'S FINE RETRACE GRID.  That is not a
               preference: the coarse co-moving exit vertex is 7.8x
               under-sampled on its own exit sphere (design 121: dx 33.2 um
               against 4.26 um needed) and anything before the last group
               cannot be traced at all, because ``apply_real_lens_traced``
               maps entrance to exit along ONE congruence.
               PROBE_SUM_AT_APERTURE S2.  Per-beam checkpoints are
               ``CarrierField`` Zarr stores (8.9x compression, measured) plus
               a JSON energy ledger.

``aggregate``  ``re_reference`` + weighted sum onto one common carrier, with
               the Nyquist guard LIVE (default ``on_nyquist='error'``) and a
               per-beam ledger of power landed, power lost, support radius,
               containment margin and which term bound the pitch.

``leg``        resolve ONE exact final leg against the summed field.

``readout``    one Bluestein zoom per frame, plus per-frame metrics and a
               like-for-like comparison against the shipped per-order tile.

RESUME
------
A checkpoint is a stage boundary.  Every artifact carries a KEY covering the
pipeline schema, ``lumenairy.__version__``, a content hash of
``lumenairy/**``, a content hash of this package, the stage's slice of the
spec, the upstream stage's digest, and per-artifact extras -- the same
discipline ``_d121_common._chain_a_key`` applies to chain A, applied per
artifact.  A stage is resumed only when its stored key matches EXACTLY.

COHERENT SUM
------------
Read ``driver.admissibility_banner``.  On a library without
``FIX_TILT_QUADRATIC_OPL_2026_08_11`` the inter-chain piston is reproducible
and wrong by 0.050-0.416 waves, so intensity columns are scored and piston
columns are recorded pre-fix and flagged.

Author: Andrew Traverso
"""
from .artifacts import (Manifest, lumenairy_source_sha, peak_rss_gb,
                        pipeline_source_sha, selection_tag, stage_key)
from .driver import (PipelineError, RunState, admissibility_banner,
                     piston_fix_status, run)
from .metrics import compare, spot
from .sources import (Beam, ChainOutput, DecomposeResult, get_chain_runner,
                      get_decomposer, order_tag, register_chain_runner,
                      register_decomposer)
from .spec import (STAGES, AggregateSpec, ChainSpec, DecomposeSpec, LegSpec,
                   PipelineSpec, ReadoutSpec, SpecError, resolve_selection)

__all__ = [
    'STAGES', 'AggregateSpec', 'Beam', 'ChainOutput', 'ChainSpec',
    'DecomposeResult', 'DecomposeSpec', 'LegSpec', 'Manifest',
    'PipelineError', 'PipelineSpec', 'ReadoutSpec', 'RunState', 'SpecError',
    'admissibility_banner', 'compare', 'get_chain_runner', 'get_decomposer',
    'lumenairy_source_sha', 'order_tag', 'peak_rss_gb', 'pipeline_source_sha',
    'piston_fix_status', 'register_chain_runner', 'register_decomposer',
    'resolve_selection', 'run', 'selection_tag', 'spot', 'stage_key',
]
