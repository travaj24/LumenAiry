# validation/pipeline/artifacts.py -- checkpoints, and the KEY DISCIPLINE that
# makes resuming from one safe.
#
# THE RULE, TAKEN FROM ``_d121_common._chain_a_key`` AND APPLIED PER ARTIFACT
# ---------------------------------------------------------------------------
# A cache that is keyed on less than the artifact depends on is worse than no
# cache: it serves a field computed under different physics, silently, wearing
# the label of the run that asked for it.  That is not hypothetical here --
# defect D6 (REVIEW_TRACED_EXACT_2026_08_05) was exactly this: the chain-A
# cache was keyed on a hand-spelled filename, and one commit flipped two
# LIBRARY DEFAULTS (``gap_kernel``, ``newton_fit``) that change the cached
# field and could never appear in that name.
#
# So every artifact this pipeline writes carries a KEY DICT covering
# everything it can depend on:
#
#   1. the pipeline schema salt;
#   2. ``lumenairy.__version__``;
#   3. A CONTENT HASH OF EVERY ``lumenairy/**/*.py``.  This is the field that
#      catches a default flip WITHIN one version, and it is also what protects
#      this pipeline from the hazard PROBE_SUM_AT_APERTURE S1.2 measured: the
#      working tree's ``lumenairy/**`` was edited by another process WHILE a
#      probe was running, and the returned field's absolute phase moved by up
#      to 2.88 rad while every intensity metric held to 0.001 EE points.  The
#      cache key is the only thing that noticed.
#   4. A CONTENT HASH OF THIS PIPELINE PACKAGE.  The driver is as much an
#      input to its artifacts as the library is.
#   5. the STAGE's own slice of the spec, serialised;
#   6. the UPSTREAM stage's key digest, so a change anywhere upstream orphans
#      everything downstream of it without each stage having to know why;
#   7. per-artifact extras (a beam key, an include-set tag).
#
# The digest names the file AND is stored inside it, and the stored copy is
# CHECKED on load -- so a file renamed onto a matching name is refused rather
# than used.
#
# cp1252-safe ASCII only.
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .spec import PIPELINE_SCHEMA, STAGES, PipelineSpec

#: Bumped by hand to orphan every artifact this package has ever written.
ARTIFACT_SCHEMA = 1

_HERE = os.path.dirname(os.path.abspath(__file__))

# The two source hashes are process-invariant and cost ~40 ms and ~2 ms
# respectively, so they are computed once.  A run that edits the library or
# this package MID-RUN gets the pre-edit hash for the rest of the process --
# which is the honest reading: the artifacts of that process were produced by
# the code the process imported, not by what is on disk now.
_LIB_SHA: Optional[str] = None
_PKG_SHA: Optional[str] = None


#: Package modules EXCLUDED from :func:`pipeline_source_sha`.
#:
#: THE RULE, AND WHY IT IS A SHORT EXPLICIT LIST RATHER THAN A PATTERN.  The
#: key must cover everything an artifact DEPENDS ON.  A module that the driver
#: never imports and that never writes an artifact is provably not one of
#: those, and including it means a docstring edit in a read-only reporter
#: throws away hours of chains -- which is how a caching layer trains its
#: users to delete it (the same argument S2 of the note makes for keeping the
#: chains stage off the readout's spec slice).
#:
#: A NAME GOES IN HERE ONLY IF BOTH HOLD: (1) nothing under
#: ``validation/pipeline`` imports it, and (2) it opens artifacts read-only.
#: ``report.py`` is the only such module; ``run_pipeline.py`` is deliberately
#: NOT excluded even though it merely builds a spec, because it is an entry
#: point and the conservative side of that judgement is cheap.
_PKG_HASH_EXCLUDE = ('report.py',)


def _sha_tree(root: str, suffix: str = '.py', exclude=()) -> str:
    """SHA-256 over every ``suffix`` file under ``root``.

    Path-relative name + bytes, walked in sorted order with ``__pycache__``
    pruned, so it is stable across machines and independent of mtime (this
    repo is Syncthing-mirrored across a mesh, where mtime is not)."""
    ex = set(exclude)
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__')
        for name in sorted(filenames):
            if not name.endswith(suffix) or name in ex:
                continue
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).replace('\\', '/').encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()


def lumenairy_source_sha() -> str:
    global _LIB_SHA
    if _LIB_SHA is None:
        import lumenairy
        _LIB_SHA = _sha_tree(os.path.dirname(os.path.abspath(
            lumenairy.__file__)))
    return _LIB_SHA


def pipeline_source_sha() -> str:
    global _PKG_SHA
    if _PKG_SHA is None:
        _PKG_SHA = _sha_tree(_HERE)
    return _PKG_SHA


def _stage_slice(spec: PipelineSpec, stage: str) -> Dict[str, Any]:
    """The part of the spec a given stage's output can depend on.

    Deliberately NOT the whole spec: keying the ``chains`` stage on the
    readout's ``dx_out`` would orphan 32 aperture fields (each a ~2 minute
    chain) every time a frame window changed, which is how a caching layer
    trains its users to delete it."""
    d = spec.to_dict()
    common = {'wavelength': d['wavelength']}
    if stage == 'decompose':
        return {**common, 'decompose': d['decompose']}
    if stage == 'chains':
        return {**common, 'chain': d['chain']}
    if stage == 'aggregate':
        return {**common, 'aggregate': d['aggregate']}
    if stage == 'leg':
        return {**common, 'leg': d['leg'], 'readout_dx_out':
                d['readout']['dx_out'], 'readout_n_out': d['readout']['n_out']}
    if stage == 'readout':
        return {**common, 'leg': d['leg'], 'readout': d['readout']}
    raise KeyError(stage)


def stage_key(spec: PipelineSpec, stage: str, upstream: Optional[str] = None,
              extra: Optional[Dict[str, Any]] = None):
    """``(key_dict, hexdigest)`` for one artifact.  See the module header."""
    if stage not in STAGES:
        raise KeyError(f"stage_key: {stage!r} is not one of {STAGES}.")
    import lumenairy
    key = {
        'artifact_schema': int(ARTIFACT_SCHEMA),
        'pipeline_schema': int(PIPELINE_SCHEMA),
        'stage': str(stage),
        'lumenairy_version': str(lumenairy.__version__),
        'lumenairy_source_sha256': lumenairy_source_sha(),
        'pipeline_source_sha256': pipeline_source_sha(),
        'spec': _stage_slice(spec, stage),
        'upstream': str(upstream) if upstream else None,
        'extra': json_safe(dict(extra or {})),
    }
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)
    return key, hashlib.sha256(blob.encode('ascii')).hexdigest()


def key_blob(key: Dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


#: Decomposer parameters that SELECT beams rather than DEFINE them.  Named
#: here because the distinction is load-bearing for resume.
_SELECTION_PARAM_KEYS = ('orders', 'beams')


def decompose_lineage_digest(spec: PipelineSpec) -> str:
    """The upstream digest a PER-BEAM chain artifact hangs off.

    WHY THIS IS NOT ``stage_key(spec, 'decompose')``.  A per-beam aperture
    field depends on how that beam is DEFINED (the decomposer, the launch
    chain, the cell resolution, the wavelength) and on the beam's own payload
    -- which the chain key already carries in full.  It does NOT depend on
    which SIBLINGS were asked for.  Keying it on the whole decompose stage
    would mean that dropping a 32-order fan to 8 orders, or adding one order
    to a 3-order study, orphans every aperture field: 32 chains at ~3 minutes
    each, thrown away because the list they were listed in changed.

    So the selection keys are excluded here and only here.  Everything that
    can change a beam's FIELD is still in the key: the decomposer name, every
    other parameter (``chain_a``, ``cell_pixels``, ``cache_dir``, ...), the
    wavelength, the beam's own weight, frame and payload, the chain spec, the
    library hash and this package's hash.  The ``decompose`` artifact itself
    is still keyed on the full slice, so a 3-beam ``beams.json`` can never be
    served to a 32-beam run."""
    d = spec.to_dict()
    dec = d['decompose']
    lineage = {'kind': dec['kind'],
               'params': {k: v for k, v in (dec.get('params') or {}).items()
                          if k not in _SELECTION_PARAM_KEYS},
               'wavelength': d['wavelength'],
               'artifact_schema': int(ARTIFACT_SCHEMA)}
    return hashlib.sha256(key_blob(lineage).encode('ascii')).hexdigest()


# ---------------------------------------------------------------------------
# JSON artifacts
# ---------------------------------------------------------------------------
def json_safe(v):
    """Encode a payload so ``allow_nan=False`` can serialise it.

    Non-finite floats become their ``repr`` STRING (``'nan'``, ``'inf'``),
    which is the same convention ``carrier_field`` uses for a collimated
    ``R = +/-inf``.  They are NOT dropped and NOT replaced by ``null``: a
    metric that could not be measured -- an FWHM whose radial profile never
    crosses half maximum inside the readout tile, say -- is a real, reportable
    outcome, and a report that silently omitted it would read as if the
    measurement had not been attempted.  ``allow_nan=True`` is the other
    option and is refused: bare ``NaN`` is not JSON and a non-Python reader
    of these artifacts is entitled to reject it."""
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, (np.floating, np.integer)):
        return json_safe(v.item())
    if isinstance(v, np.ndarray):
        return json_safe(v.tolist())
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    return str(v)


def write_json(path: str, payload: Dict[str, Any], key: Dict[str, Any]):
    """Write a JSON artifact with its key embedded.

    Written to a temporary file and renamed, so an interrupted run leaves
    either the previous artifact or none -- never a half-written one that the
    next run's key check would happily accept because the KEY was flushed
    first."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc = {'_key': key, '_written': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'payload': json_safe(payload)}
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='cp1252', errors='strict') as fh:
        json.dump(doc, fh, sort_keys=True, indent=1, allow_nan=False)
    os.replace(tmp, path)


def read_json(path: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the payload iff the stored key matches EXACTLY, else ``None``.

    ``None`` means recompute.  There is deliberately no 'close enough': the
    whole purpose of the key is that a mismatch is a different artifact."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='cp1252', errors='strict') as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if key_blob(doc.get('_key', {})) != key_blob(key):
        return None
    return doc.get('payload')


# ---------------------------------------------------------------------------
# CarrierField artifacts (Zarr)
# ---------------------------------------------------------------------------
def have_zarr() -> bool:
    try:
        import zarr                                          # noqa: F401
        return True
    except ImportError:
        return False


def _npz_path(path: str) -> str:
    return path + '.npz'


def field_exists(path: str) -> bool:
    """Is there a field checkpoint at ``path`` in EITHER container?"""
    return os.path.exists(path) or os.path.exists(_npz_path(path))


def _save_field_npz(path: str, field) -> str:
    """FALLBACK checkpoint format for a box without the ``zarr`` extra.

    ``zarr`` is an OPTIONAL dependency of this library (``pyproject``
    ``[zarr]``), and a pipeline whose checkpoints are unavailable on half the
    mesh is a pipeline that cannot be resumed where it is running.  The npz
    carries the SAME information -- envelope, grid (origin included), carrier
    (piston included), wavelength, provenance -- because the thing that must
    never be lost is the ABSOLUTE geometry: a mis-read origin relocates the
    field without changing a single sample of it.  It is bigger (npz's
    DEFLATE against zstd+shuffle) and it is not the format of record."""
    from lumenairy.propagators.carrier_field import CARRIER_FIELD_SCHEMA
    np.savez_compressed(
        _npz_path(path), envelope=np.ascontiguousarray(field.envelope,
                                                       dtype=np.complex128),
        meta=np.array(json.dumps(
            {'schema': int(CARRIER_FIELD_SCHEMA),
             'wavelength': float(field.wavelength),
             'grid': field.grid.to_dict(), 'carrier': field.carrier.to_dict(),
             'provenance': json_safe(field.provenance)},
            sort_keys=True, allow_nan=False)))
    return _npz_path(path)


def _load_field_npz(path: str):
    from lumenairy.propagators.carrier_field import (CARRIER_FIELD_SCHEMA,
                                                     CarrierField, CarrierSpec,
                                                     FieldGrid)
    d = np.load(_npz_path(path), allow_pickle=False)
    m = json.loads(str(d['meta']))
    if int(m.get('schema', -1)) != int(CARRIER_FIELD_SCHEMA):
        raise ValueError(
            f"{_npz_path(path)}: CarrierField schema {m.get('schema')} against "
            f"this build's {CARRIER_FIELD_SCHEMA}.  Refusing to guess.")
    return CarrierField(envelope=np.asarray(d['envelope']),
                        grid=FieldGrid.from_dict(m['grid']),
                        carrier=CarrierSpec.from_dict(m['carrier']),
                        wavelength=float(m['wavelength']),
                        provenance=m['provenance'])


def save_field(path: str, field, key: Dict[str, Any], *, name='field',
               extra_provenance: Optional[Dict[str, Any]] = None) -> str:
    """Checkpoint a :class:`CarrierField`, key stamped into its provenance.

    ZARR IS THE FORMAT OF RECORD, for two measured reasons
    (BUILD_CARRIER_FIELD S6.3): a design-121 aperture ENVELOPE compresses
    8.923x (1.0737 GB -> 0.1203 GB) under the shipped zstd+shuffle default,
    and the store's attributes carry the grid ORIGIN and the carrier -- which
    a bare array cannot, and getting either wrong relocates the field in
    absolute coordinates without changing a single sample of it.  When the
    optional dependency is absent the npz fallback above is used instead, so a
    run is never blocked by a missing extra."""
    prov = dict(field.provenance)
    prov.update(extra_provenance or {})
    prov['pipeline_key'] = key
    stamped = field.__class__(envelope=field.envelope, grid=field.grid,
                              carrier=field.carrier,
                              wavelength=field.wavelength, provenance=prov)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    if not have_zarr():
        return _save_field_npz(path, stamped)
    from lumenairy.propagators.carrier_field import save_carrier_field_zarr
    return save_carrier_field_zarr(path, stamped, name=name, overwrite=True)


def load_field(path: str, key: Dict[str, Any], *, name='field'):
    """Load a checkpointed field iff its stored key matches, else ``None``.

    ``None`` means recompute.  Both formats are read whichever is present, so
    a workdir written on a box with ``zarr`` resumes on one without it and
    vice versa -- the KEY is what decides validity, not the container."""
    f = None
    try:
        if os.path.exists(path) and have_zarr():
            from lumenairy.propagators.carrier_field import (
                load_carrier_field_zarr)
            f = load_carrier_field_zarr(path, name=name)
        elif os.path.exists(_npz_path(path)):
            f = _load_field_npz(path)
    except Exception:
        return None
    if f is None:
        return None
    if key_blob(f.provenance.get('pipeline_key', {})) != key_blob(key):
        return None
    return f


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
class Manifest:
    """The run's stage ledger: what ran, under which key, how long it took.

    Kept as a plain JSON file rather than a database because the interesting
    failure mode is a human reading it after a 3-hour run died, and because
    two processes must never write the same workdir (which is stated, and
    which a lock would only appear to solve)."""

    def __init__(self, path: str):
        self.path = path
        self.doc: Dict[str, Any] = {'schema': ARTIFACT_SCHEMA, 'stages': {}}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='cp1252', errors='strict') as fh:
                    self.doc = json.load(fh)
            except (OSError, ValueError):
                pass
        self.doc.setdefault('stages', {})

    def get(self, stage: str) -> Dict[str, Any]:
        return self.doc['stages'].get(stage, {})

    def record(self, stage: str, digest: str, **info):
        ent = {'digest': digest,
               'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        ent.update(info)
        self.doc['stages'][stage] = ent
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.',
                    exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='cp1252', errors='strict') as fh:
            json.dump(json_safe(self.doc), fh, sort_keys=True, indent=1,
                      allow_nan=False)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# process accounting
# ---------------------------------------------------------------------------
def peak_rss_gb():
    """``(peak working set, peak commit)`` of THIS process in GB.

    Windows keeps both counters itself, so this is the true peak over the
    whole run rather than a sample -- which matters, because the quantity a
    32-order run has to report is the PEAK, and a sampler misses it by
    construction."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return (float(getattr(mi, 'peak_wset', mi.rss)) / 1e9,
                float(getattr(mi, 'peak_pagefile', 0.0)) / 1e9)
    except Exception:
        return (float('nan'), float('nan'))


def selection_tag(keys: Sequence[str], prefix: str = 'sum') -> str:
    """A short, deterministic, FILESYSTEM-safe tag for a set of beam keys.

    Spelled out while it is short enough to read (which is the 3-order case
    the acceptance is scored on) and hashed once it is not (the 32-order
    case), because a directory a human cannot read is a directory a human
    deletes."""
    body = '_'.join(keys)
    if len(body) <= 48:
        return f"{prefix}_{body}" if body else f"{prefix}_none"
    h = hashlib.sha256('|'.join(keys).encode('ascii')).hexdigest()[:12]
    return f"{prefix}_{len(keys)}beams_{h}"
