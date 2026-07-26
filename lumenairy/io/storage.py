"""
Unified Storage Backend for Optical Fields
===========================================

Provides a single API for reading and writing optical field data to
either HDF5 (``.h5``) or Zarr (``.zarr``) storage, selectable at
runtime via :func:`set_storage_backend`.

The user sets the backend ONCE (or accepts the default ``'hdf5'``),
and all subsequent read/write calls auto-dispatch to the correct
implementation.  On reads, the format is auto-detected from the file
extension or path structure, so HDF5 files written by older code are
always readable regardless of the current backend setting.

This module also contains all HDF5-specific I/O functions (previously
in ``hdf5_io.py``) for single-field, multi-plane, and Jones-field
storage.  The ``hdf5_io`` module is retained as a backwards-compatible
re-export shim.

Quick start::

    import lumenairy as la

    # Choose backend for NEW files (default = 'hdf5')
    la.set_storage_backend('zarr')

    # Unified API — works with both HDF5 and Zarr:
    la.append_plane('output.zarr', E, dx=dx, label='After L1')
    planes = la.load_planes('output.zarr', indices=[0, 5])

    # HDF5-specific (explicit):
    la.save_field_h5('field.h5', E, dx=2e-6, wavelength=1.31e-6)
    E, meta = la.load_field_h5('field.h5')

Concurrency model (v4.16.0+)
----------------------------

Both :func:`append_plane_h5` and :func:`_zarr_append_plane` are now
safe to call from multiple PROCESSES against the same file.  The
implementation combines two cooperating mechanisms:

1. **Single-process atomicity** (v4.14.3, preserved): the
   ``n_planes`` attribute is bumped BEFORE the dataset is created
   and rolled back on failure.  A mid-call crash never strands the
   slot and never silently clobbers an orphan.

2. **Multi-process serialisation** (v4.16.0): every append acquires
   a :class:`filelock.FileLock` on a sibling ``<path>.lock`` file
   for the duration of the open/create/close window.  Concurrent
   appenders are SERIALISED (race-free) rather than parallelised --
   throughput is one writer at a time, but no data is ever lost.

3. **Concurrent reads** (v4.16.0, HDF5 only): when ``swmr=True``
   (the default) the writer opens the file with ``libver='latest'``
   and sets ``f.swmr_mode = True`` after creating the ``planes``
   group, so reader processes can open the file with
   ``swmr=True`` and see consistent state while the writer is
   active.  Multiple readers + a single writer can coexist.  Set
   ``swmr=False`` only for legacy callers that need to interoperate
   with pre-HDF5-1.10 readers; SWMR-disabled mode loses the
   concurrent-reader guarantee.

The lock file is automatically created in the same directory as the
data file (it is essentially empty -- the lock state lives in the
file-system's advisory-lock layer).  v5.24.x (audit S4-19): whether
the ``.lock`` file remains on disk after the lock is released is
filelock-version- and platform-dependent (some ``filelock`` releases
unlink it on POSIX release; on Windows it is typically left behind),
so callers must not rely on either outcome.  Either way it is
harmless: a stray ``.lock`` sibling carries no state and can be
removed manually, and library cleanup never depends on its presence.

Requirements: ``filelock>=3.0`` (declared in the ``hdf5`` and
``zarr`` optional-dependency groups in ``pyproject.toml``).  HDF5
SWMR mode requires HDF5 1.10+, which ships with all h5py>=3.0
wheels.

Author: Andrew Traverso
"""

from __future__ import annotations

import base64
import json
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# v4.16.0: distributed (multi-process) advisory lock.  Imported lazily
# inside :func:`_require_filelock` so simply importing
# ``lumenairy.io.storage`` does not require ``filelock`` to be
# installed -- it is only fetched when the multi-writer append path
# is actually invoked.  ``filelock`` is declared in the ``hdf5`` and
# ``zarr`` optional-dependency groups in ``pyproject.toml``.
_FILELOCK_MODULE = None  # populated by :func:`_require_filelock`


def _require_filelock():
    """Lazy-import :mod:`filelock` for the multi-process append path.

    Raises :class:`ImportError` with an actionable message if the
    optional dependency is missing.  Called from the multi-writer
    append code path on every invocation; cached at module scope on
    first success so the import cost is paid once per process.
    """
    global _FILELOCK_MODULE
    if _FILELOCK_MODULE is None:
        try:
            import filelock as _fl
            _FILELOCK_MODULE = _fl
        except ImportError as e:
            raise ImportError(
                "filelock is required for multi-process atomic append "
                "(v4.16.0+).  Install with: pip install filelock  "
                "(or pip install 'lumenairy[hdf5]' / 'lumenairy[zarr]' "
                "for the full optional-dependency group)."
            ) from e
    return _FILELOCK_MODULE


def _append_lock_path(filepath) -> str:
    """Return the sibling ``<path>.lock`` file path used by the
    multi-process append lock.

    The lock file lives next to the data file so it inherits the
    same permissions / file-system as the data, and is trivially
    discoverable by an administrator inspecting a stuck job.  It is
    essentially empty -- the OS advisory-lock state lives in the
    file-system layer, not in the file contents.
    """
    return str(filepath) + '.lock'

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
_ZARR_MKDIR_PATCH_LOCK = threading.Lock()


def _get_lumenairy_version() -> str:
    """Return the installed lumenairy version string.

    v4.15.0 (P2-VERSTAMP from v4.14.2 audit): every HDF5
    ``create_dataset`` and Zarr ``create_array`` site stamps a
    ``lumenairy_version`` attribute so a future reader can
    distinguish data written by different library versions
    (e.g. for migration paths around schema changes or to flag
    data written by a known-buggy release).  Lazy import + cached
    via the parent module's ``__version__`` attribute so the
    storage layer doesn't pay a top-level-import cost on every
    write call.  Falls back to the literal ``'unknown'`` if the
    parent package can't be imported (which would only happen if
    storage.py were copied out of the package and used standalone).
    """
    try:
        from .. import __version__
        return str(__version__)
    except (ImportError, AttributeError):
        return 'unknown'

# =========================================================================
# HDF5 backend
# =========================================================================

import importlib.util as _importlib_util

_H5PY_AVAILABLE = _importlib_util.find_spec('h5py') is not None
h5py = None  # populated lazily on first use


def _require_h5py():
    global h5py
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O. "
                          "Install with: pip install h5py")
    if h5py is None:
        import h5py as _h
        globals()['h5py'] = _h


def _decode_attr(val):
    """Convert HDF5 attribute values to plain Python types where sensible."""
    if isinstance(val, bytes):
        return val.decode('utf-8')
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ('S', 'O'):  # bytes / object
            return [v.decode('utf-8') if isinstance(v, bytes) else v
                    for v in val]
        return val
    return val


# =========================================================================
# Canonical nested-metadata serialization contract (S4-19, v5.24.x)
# =========================================================================
#
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
# metadata mapping is encoded to a single JSON string with explicit
# type tags (``_meta_dumps``) and stored under ONE reserved attribute
# (``_META_BLOB_KEY``), byte-for-byte identical in both backends.  On
# read the blob is decoded back to the exact Python structure
# (``_meta_loads``).  A best-effort *flattened* copy of the scalars is
# still written alongside the blob so external tools (HDFView, ``zarr``
# inspectors) and older library versions can still see readable keys --
# but the blob is authoritative for the faithful round-trip.
#
# BACK-COMPAT: a file with no blob (written by the pre-contract scheme)
# still loads -- the reader detects the missing blob and falls back to
# reconstructing the mapping from the individual native attributes,
# reproducing the historical (flat) behavior exactly.

_META_BLOB_KEY = '__lumenairy_meta_json__'
_META_TAG = '__lumen_t__'


def _meta_encode(obj):
    """Lower a metadata value to a JSON-serializable, type-tagged form.

    Native JSON scalars (``None``/``bool``/``int``/``float``/``str``)
    and ``list`` pass through as themselves (a ``list`` therefore stays
    a JSON array and decodes back to a ``list``).  Every other type
    carries an explicit ``_META_TAG`` so the decoder can restore it
    exactly -- crucially distinguishing ``ndarray`` (tagged) from
    ``list`` (untagged array).
    """
    # bool is a subclass of int; both are native JSON and need no tag.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, complex):
        return {_META_TAG: 'complex', 're': obj.real, 'im': obj.imag}
    if isinstance(obj, bytes):
        return {_META_TAG: 'bytes',
                'b64': base64.b64encode(obj).decode('ascii')}
    if isinstance(obj, np.generic):
        # NumPy scalar -> Python scalar, then re-encode (a complex
        # numpy scalar becomes a Python complex and picks up the tag).
        return _meta_encode(obj.item())
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            data = {'re': np.real(obj).tolist(), 'im': np.imag(obj).tolist()}
        else:
            data = obj.tolist()
        return {_META_TAG: 'ndarray', 'dtype': str(obj.dtype),
                'shape': list(obj.shape), 'data': data}
    if isinstance(obj, tuple):
        return {_META_TAG: 'tuple', 'items': [_meta_encode(v) for v in obj]}
    if isinstance(obj, list):
        return [_meta_encode(v) for v in obj]
    if isinstance(obj, dict):
        enc = {k: _meta_encode(v) for k, v in obj.items()}
        # A user dict that literally contains the tag key would be
        # mis-read as a tagged wrapper; wrap it explicitly so the
        # decoder restores it as a plain dict.
        if _META_TAG in enc:
            return {_META_TAG: 'dict',
                    'items': [[k, _meta_encode(v)] for k, v in obj.items()]}
        return enc
    # Unknown / non-serializable type: preserve its ``str()`` form,
    # mirroring the historical ``except TypeError: str(v)`` fallback.
    return {_META_TAG: 'repr', 'value': str(obj)}


def _meta_decode(obj):
    """Inverse of :func:`_meta_encode`."""
    if isinstance(obj, list):
        return [_meta_decode(v) for v in obj]
    if isinstance(obj, dict):
        kind = obj.get(_META_TAG)
        if kind is not None:
            if kind == 'complex':
                return complex(obj['re'], obj['im'])
            if kind == 'bytes':
                return base64.b64decode(obj['b64'])
            if kind == 'tuple':
                return tuple(_meta_decode(v) for v in obj['items'])
            if kind == 'dict':
                return {k: _meta_decode(v) for k, v in obj['items']}
            if kind == 'ndarray':
                dtype = np.dtype(obj['dtype'])
                shape = tuple(obj['shape'])
                data = obj['data']
                if isinstance(data, dict) and 're' in data:  # complex
                    arr = (np.asarray(data['re'], dtype=np.float64)
                           + 1j * np.asarray(data['im'], dtype=np.float64))
                    arr = arr.astype(dtype)
                else:
                    arr = np.asarray(data, dtype=dtype)
                return arr.reshape(shape)
            if kind == 'repr':
                return obj['value']
            # Unknown tag (written by a newer library): fall through and
            # return it as a plain dict so nothing is silently dropped.
        return {k: _meta_decode(v) for k, v in obj.items()}
    return obj


def _meta_dumps(metadata: Dict[str, Any]) -> str:
    """Serialize a nested metadata mapping to the canonical JSON string.

    ``sort_keys`` is deliberately OFF so insertion order (dict ordering)
    survives the round-trip.
    """
    return json.dumps(_meta_encode(dict(metadata)), sort_keys=False)


def _meta_loads(blob) -> Dict[str, Any]:
    """Parse a canonical metadata blob back to its Python structure."""
    if isinstance(blob, bytes):
        blob = blob.decode('utf-8')
    return _meta_decode(json.loads(blob))


def _flatten_metadata(d, prefix=''):
    """Flatten a nested mapping to dotted keys (best-effort native attrs).

    Shared by both backends' sim-metadata writers.  This is the *lossy*
    external-inspection view (it cannot re-nest and coerces list vs
    ndarray); the authoritative round-trip is the JSON blob.
    """
    items = {}
    for k, v in d.items():
        key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
        if isinstance(v, dict):
            items.update(_flatten_metadata(v, key))
        else:
            items[key] = v
    return items


def _h5_write_meta_attrs(holder, metadata) -> None:
    """Write a user ``metadata`` mapping onto an h5py attrs holder faithfully.

    A-4 (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25).  Handing the raw values
    to ``holder.attrs[k] = v`` measured 4 hard raises (nested dict,
    heterogeneous list, any non-serializable object) plus a silent DROP of
    ``None`` and four silent type coercions (``list``/``tuple`` ->
    ``ndarray``, ``bytes`` -> ``str``) over the module's own 19-type probe
    set -- while :func:`_meta_dumps` (already the contract for
    ``write_sim_metadata``, S4-19) round-trips all 19.  This helper routes
    ``metadata`` through the SAME canonical codec:

    * the authoritative, type-tagged JSON blob under ``_META_BLOB_KEY``;
    * plus the best-effort *flattened* native-attr copy so external tools
      (HDFView, ``h5dump``) and older library readers still see readable
      keys.  The flat copy is lossy by construction and is discarded on
      read in favour of the blob (see :func:`_h5_read_attrs`).

    BACK-COMPAT: files written by the pre-A-4 raw-attr scheme carry no
    blob and read back byte-for-byte as before -- :func:`_h5_read_attrs`
    only consults the blob when it is present.
    """
    if not metadata:
        return
    holder.attrs[_META_BLOB_KEY] = _meta_dumps(metadata)
    for k, v in _flatten_metadata(metadata).items():
        if v is None:   # h5py cannot store a None attr; the blob carries it
            continue
        # Best-effort external-inspection view only -- never aborts the
        # write (the blob is authoritative).  Mirrors
        # :func:`_h5_write_sim_metadata`.
        try:
            holder.attrs[str(k)] = v
        except (TypeError, ValueError):
            try:
                holder.attrs[str(k)] = str(v)
            except (TypeError, ValueError):
                pass


# The plane-dict keys ``save_planes_h5`` handles STRUCTURALLY -- exactly
# the set ``append_plane_h5`` writes as native attributes itself.  They
# stay raw h5py attributes so their read-back types are unchanged for
# every file ever written (measured: ``dx``/``dy``/``z``/``wavelength``
# -> ``np.float64``, ``label``/``dtype``/``lumenairy_version`` -> ``str``).
# Every OTHER key in a plane dict is caller metadata -- the per-plane
# equivalent of ``append_plane_h5(metadata=)`` -- and is routed through
# the canonical codec (A-4 follow-up).
_PLANE_NATIVE_KEYS = frozenset({
    'dx', 'dy', 'z', 'label', 'wavelength', 'dtype', 'lumenairy_version',
})


def _h5_read_attrs(holder) -> Dict[str, Any]:
    """Decode every attribute of an h5py dataset / group to Python types.

    Equivalent to ``{k: _decode_attr(v) for k, v in holder.attrs.items()}``
    (the historical behaviour, preserved exactly for files with no
    metadata blob) plus the A-4 blob overlay: when ``_META_BLOB_KEY`` is
    present its decoded mapping is authoritative, and the lossy flattened
    shadow copies of those same keys are dropped so the caller sees the
    structure it wrote rather than a mix of both views.  The reserved blob
    key itself is never surfaced.
    """
    out = {k: _decode_attr(holder.attrs[k]) for k in holder.attrs.keys()}
    blob = out.pop(_META_BLOB_KEY, None)
    if blob is None:
        return out
    try:
        decoded = _meta_loads(blob)
    except (ValueError, TypeError):
        # Truncated / foreign blob: fall back to the native attrs rather
        # than losing them (the flat copy is still there).
        return out
    if not isinstance(decoded, dict):
        return out
    # Drop the flattened shadow copies the writer left for external tools.
    for flat_key in _flatten_metadata(decoded):
        out.pop(flat_key, None)
    out.update(decoded)
    return out


# ── Single field I/O (HDF5-specific) ────────────────────────────────────

def save_field_h5(filepath: str, E: np.ndarray, dx: float,
                  dy: Optional[float] = None,
                  wavelength: Optional[float] = None,
                  label: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  compression: Optional[str] = 'gzip',
                  compression_opts: Optional[int] = 4,
                  preserve_dtype: bool = False) -> None:
    """
    Save a single complex optical field to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output .h5 file path.
    E : ndarray (complex, Ny x Nx)
        Complex electric field.  CuPy / JAX arrays are accepted and
        converted to NumPy for storage.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m]. Defaults to ``dx``.
    wavelength : float, optional
        Optical wavelength [m].
    label : str, optional
        Human-readable label for the field.
    metadata : dict, optional
        Additional attributes to store.
    compression : str or None, default 'gzip'
        HDF5 compression filter ('gzip', 'lzf', or None).
    compression_opts : int, default 4
        Compression level (for gzip, 1-9).
    preserve_dtype : bool, default False
        If True, store ``E`` at its native complex precision
        (``complex64`` or ``complex128``).  If False (the historical
        default), coerce to ``complex128`` for storage.  ``complex64``
        is ~2× smaller on disk and is sufficient for most propagation
        simulations; use ``preserve_dtype=True`` to keep the original
        precision through a round-trip (``save_field_h5`` ->
        ``load_field_h5``).
    """
    _require_h5py()
    if dy is None:
        dy = dx
    # Coerce CuPy / JAX -> NumPy without disturbing dtype if requested.
    from ..backend import to_numpy
    E_np = to_numpy(E) if not isinstance(E, np.ndarray) else E
    if not preserve_dtype:
        E_np = E_np.astype(np.complex128, copy=False)
    with h5py.File(filepath, 'w') as f:
        dset = f.create_dataset(
            'field', data=E_np,
            compression=compression, compression_opts=compression_opts
        )
        dset.attrs['dx'] = float(dx)
        dset.attrs['dy'] = float(dy)
        dset.attrs['dtype'] = str(E_np.dtype)
        # v4.15.0 (P2-VERSTAMP): writer version stamp for migration /
        # known-buggy-release traceability.  See _get_lumenairy_version.
        dset.attrs['lumenairy_version'] = _get_lumenairy_version()
        if wavelength is not None:
            dset.attrs['wavelength'] = float(wavelength)
        if label is not None:
            dset.attrs['label'] = str(label)
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
        _h5_write_meta_attrs(dset, metadata)


def load_field_h5(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a single complex field from an HDF5 file.

    Returns
    -------
    E : ndarray (complex)
    metadata : dict
    """
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
        if 'field' not in f:
            raise KeyError(f"File {filepath} does not contain a '/field' "
                           f"dataset. For multi-plane files, use "
                           f"load_planes instead.")
        dset = f['field']
        E = np.array(dset[()])
        metadata = _h5_read_attrs(dset)
    return E, metadata


# ── Multi-plane I/O (HDF5-specific) ─────────────────────────────────────

def save_planes_h5(filepath: str, planes: Sequence[Dict[str, Any]],
                   wavelength: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   compression: Optional[str] = 'gzip',
                   compression_opts: Optional[int] = 4,
                   preserve_dtype: bool = False) -> None:
    """
    Save a sequence of complex fields to a single HDF5 file.

    Parameters
    ----------
    filepath : str
    planes : list of dict
        Each dict has ``'field'`` (ndarray), ``'dx'`` (float), and
        optional ``'dy'``, ``'z'``, ``'wavelength'``, ``'label'``,
        plus any extra keys.  Per-plane ``'wavelength'`` (4.0+)
        survives the round-trip -- useful for chromatic-design runs
        where different planes were propagated at different
        wavelengths.
    wavelength : float, optional
        Run-level (file-level) wavelength.  Per-plane wavelengths
        in ``planes[i]['wavelength']`` are also stored and override
        the run-level value on load.
    metadata : dict, optional
    compression : str, default 'gzip'
    compression_opts : int, default 4
    preserve_dtype : bool, default False
        Preserve the per-plane complex precision (``complex64`` /
        ``complex128``).  See :func:`save_field_h5` for the same
        flag.
    """
    _require_h5py()
    from ..backend import to_numpy
    n = len(planes)
    if n == 0:
        raise ValueError("At least one plane is required")
    with h5py.File(filepath, 'w') as f:
        grp = f.create_group('planes')
        grp.attrs['n_planes'] = n
        # v4.15.0 (P2-VERSTAMP): group-level stamp covers the whole
        # batch, complementing per-dataset stamps below.
        grp.attrs['lumenairy_version'] = _get_lumenairy_version()
        if wavelength is not None:
            grp.attrs['wavelength'] = float(wavelength)
        # A-4 follow-up (see save_field_h5): the run-level ``metadata``
        # mapping goes through the canonical codec, not raw h5py attrs.
        # Same measured before-state as its siblings (14 wrote / 4 raised /
        # ``None`` dropped / 7 coerced over the 19-type probe set).
        _h5_write_meta_attrs(grp, metadata)
        for i, plane in enumerate(planes):
            if 'field' not in plane or 'dx' not in plane:
                raise ValueError(
                    f"Plane {i} missing required 'field' or 'dx'")
            name = f'plane_{i:02d}'
            E_in = plane['field']
            E = (to_numpy(E_in) if not isinstance(E_in, np.ndarray)
                 else E_in)
            if not preserve_dtype:
                E = E.astype(np.complex128, copy=False)
            dset = grp.create_dataset(
                name, data=E,
                compression=compression,
                compression_opts=compression_opts
            )
            # A-4 follow-up: structural keys stay native (see
            # _PLANE_NATIVE_KEYS); the caller's extra keys are per-plane
            # metadata and go through the codec.  Raw, those extras
            # measured the same as every other A-4 site over the 19-type
            # probe set: 14 wrote without raising / 4 raised / 1 silently
            # dropped (``None``) / 7 type-coerced.
            extras = {}
            for key, value in plane.items():
                if key == 'field':
                    continue
                if key in _PLANE_NATIVE_KEYS:
                    if value is None:   # h5py cannot store a None attr
                        continue
                    dset.attrs[str(key)] = value
                else:
                    extras[str(key)] = value
            _h5_write_meta_attrs(dset, extras)
            if 'dy' not in plane:
                dset.attrs['dy'] = float(plane['dx'])
            dset.attrs['dtype'] = str(E.dtype)
            # v4.15.0 (P2-VERSTAMP): per-plane writer-version stamp.
            dset.attrs['lumenairy_version'] = _get_lumenairy_version()


def load_planes_h5(filepath: str,
                   indices: Optional[Sequence[int]] = None
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load multi-plane fields from an HDF5 file.

    Returns
    -------
    planes : list of dict
    file_metadata : dict
    """
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
        if 'planes' not in f:
            raise KeyError(f"File {filepath} does not contain a '/planes' "
                           f"group. For single-field files, use "
                           f"load_field_h5 instead.")
        grp = f['planes']
        n = int(grp.attrs['n_planes'])
        if indices is None:
            indices = list(range(n))
        file_metadata = _h5_read_attrs(grp)
        planes = []
        for i in indices:
            name = f'plane_{i:02d}'
            if name not in grp:
                raise KeyError(f"Plane {name} not found in file")
            dset = grp[name]
            plane = {'field': np.array(dset[()])}
            plane.update(_h5_read_attrs(dset))
            planes.append(plane)
    return planes, file_metadata


# ── Jones field I/O (HDF5-specific) ─────────────────────────────────────

def save_jones_field_h5(filepath: str, jones_field: Any,
                        wavelength: Optional[float] = None,
                        label: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        compression: Optional[str] = 'gzip',
                        compression_opts: Optional[int] = 4,
                        preserve_dtype: bool = False) -> None:
    """Save a JonesField (polarized field) to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output ``.h5`` file path.
    jones_field : JonesField
        Polarized field carrying ``Ex``, ``Ey``, ``dx``, ``dy``.
    wavelength : float, optional
        Optical wavelength [m].
    label : str, optional
        Human-readable label for the field.
    metadata : dict, optional
        Additional attributes to store.
    compression : str or None, default ``'gzip'``
        HDF5 compression filter.
    compression_opts : int, default 4
        Compression level (for gzip, 1-9).
    preserve_dtype : bool, default False
        If True, store ``Ex`` / ``Ey`` at their native complex precision
        (``complex64`` or ``complex128``).  If False (the historical
        default), coerce to ``complex128`` for storage.  See
        :func:`save_field_h5` for the same flag.  v4.13.0 onward.
    """
    _require_h5py()
    Ex_in = jones_field.Ex
    Ey_in = jones_field.Ey
    if preserve_dtype:
        Ex = np.asarray(Ex_in)
        Ey = np.asarray(Ey_in)
    else:
        Ex = np.asarray(Ex_in, dtype=np.complex128)
        Ey = np.asarray(Ey_in, dtype=np.complex128)
    with h5py.File(filepath, 'w') as f:
        grp = f.create_group('jones')
        grp.attrs['dx'] = float(jones_field.dx)
        grp.attrs['dy'] = float(jones_field.dy)
        # v4.15.0 (P2-VERSTAMP): single group-level stamp for the
        # Jones-field pair so the (Ex, Ey) provenance is consistent.
        grp.attrs['lumenairy_version'] = _get_lumenairy_version()
        if wavelength is not None:
            grp.attrs['wavelength'] = float(wavelength)
        if label is not None:
            grp.attrs['label'] = str(label)
        # A-4 follow-up (see save_field_h5).  This SUPERSEDES the S4-9
        # ``None``-skip that used to live here: S4-9's stated goal was to
        # stop the h5py C-layer crash and "match zarr, which stores
        # ``None`` fine", which it reached by DROPPING the key; the codec
        # reaches it by actually storing ``None``.  Measured before-state
        # over the 19-type probe set: 14 wrote without raising / 4 raised /
        # 1 silently dropped (``None``) / 7 type-coerced -> now 19/19.
        _h5_write_meta_attrs(grp, metadata)
        dset_ex = grp.create_dataset(
            'Ex', data=Ex,
            compression=compression, compression_opts=compression_opts)
        dset_ex.attrs['lumenairy_version'] = _get_lumenairy_version()
        dset_ey = grp.create_dataset(
            'Ey', data=Ey,
            compression=compression, compression_opts=compression_opts)
        dset_ey.attrs['lumenairy_version'] = _get_lumenairy_version()


def load_jones_field_h5(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a JonesField from an HDF5 file.

    Returns
    -------
    jones_field : JonesField
    metadata : dict
    """
    _require_h5py()
    from ..elements.polarization import JonesField
    with h5py.File(filepath, 'r') as f:
        if 'jones' not in f:
            raise KeyError(
                f"File {filepath} does not contain a '/jones' group.")
        grp = f['jones']
        Ex = np.array(grp['Ex'][()])
        Ey = np.array(grp['Ey'][()])
        dx = float(grp.attrs['dx'])
        dy = float(grp.attrs['dy'])
        metadata = _h5_read_attrs(grp)
    return JonesField(Ex, Ey, dx, dy), metadata


# ── Append / inspect (HDF5-specific) ────────────────────────────────────

def append_plane_h5(filepath: str, field: np.ndarray, dx: float,
                    dy: Optional[float] = None,
                    z: Optional[float] = None,
                    label: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    compression: Optional[str] = 'gzip',
                    compression_opts: Optional[int] = 4,
                    chunk_size: int = 1024,
                    preserve_dtype: bool = False,
                    swmr: bool = True,
                    lock_timeout: float = 30.0) -> None:
    """Append a single plane to a multi-plane HDF5 file (or create one).

    Parameters
    ----------
    filepath : str
        Path to the multi-plane HDF5 file (created on first append).
    field : ndarray (complex, Ny x Nx)
        Complex electric field.  CuPy / JAX arrays are accepted and
        converted to NumPy for storage.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    z, label, metadata
        Per-plane stored attributes.  ``metadata`` may be an arbitrarily
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
    compression, compression_opts
        HDF5 compression filter and level.
    chunk_size : int, default 1024
        HDF5 chunk-size cap (clipped to the field extent).
    preserve_dtype : bool, default False
        If True, store ``field`` at its native complex precision
        (``complex64`` or ``complex128``).  If False (the historical
        default), coerce to ``complex128`` for storage.  See
        :func:`save_field_h5` for the same flag.  v4.13.0 onward.
    swmr : bool, default True
        v4.16.0+.  Open the underlying HDF5 file with
        ``libver='latest'`` and enable Single-Writer-Multiple-Reader
        mode after creating the ``planes`` group.  This lets reader
        processes open the file (with their own ``swmr=True``) and
        follow appends in real-time without crashing or seeing
        partially-written data.  Set ``swmr=False`` only for legacy
        callers that need to interoperate with pre-HDF5-1.10 readers
        -- in that mode no concurrent-reader guarantee is provided.
        Requires HDF5 1.10+ (ships with h5py>=3.0 wheels).
    lock_timeout : float, default 30.0
        v4.16.0+.  Seconds to wait for the multi-process append lock
        before raising :class:`TimeoutError`.  The lock file is a
        sibling ``<filepath>.lock`` created automatically.  Set
        higher for very slow disks or contended remote filesystems.
        Negative or zero raises immediately on contention.

    Notes
    -----
    **Atomicity (v4.14.3, P0-NEW-1).**  The ``n_planes`` attribute is
    bumped to ``n + 1`` *before* the dataset is created and rolled back
    to ``n`` if the create fails.  Pre-4.14.3 the order was reversed
    (create first, then bump), which had two failure modes: (1) a
    crash between create and bump left an orphan dataset and a stale
    ``n_planes`` so the next append would compute the same dataset
    name and fail; (2) two concurrent appenders read the same
    ``n_planes`` and one silently overwrote the other (silent data
    loss).  The new order keeps the slot reserved even if create
    crashes: a subsequent append computes ``n + 1`` and skips the
    orphan.  The load path tolerates a "reserved but absent" slot
    (see :func:`_h5_list_planes`'s ``if name not in grp: continue``).

    **Multi-writer support (v4.16.0, P0-from-v4.14.3-deferred).**
    The single-process atomicity above is now augmented with a
    cross-process advisory lock so multiple WRITER processes calling
    ``append_plane_h5`` against the same file are SERIALISED
    race-free.  Implementation: every call acquires a
    :class:`filelock.FileLock` on the sibling ``<filepath>.lock``
    file for the duration of the open/create/close window.
    Throughput is one writer at a time (the lock is exclusive); the
    contract is "no data loss", not "parallel writes".  Tested via
    ``tests/unit/test_v4_16_0_agent_b_multiprocess_storage.py``.

    **Concurrent readers (v4.16.0, SWMR mode).**  When ``swmr=True``
    (the default for new files) reader processes can open the file
    with ``h5py.File(filepath, 'r', swmr=True, libver='latest')`` and
    safely follow appends in real-time.  SWMR requires the file to
    be opened with ``libver='latest'`` AND have ``swmr_mode = True``
    set on the writer's handle AFTER the schema (``planes`` group)
    has been created.  Both happen automatically here.  Set
    ``swmr=False`` to fall back to the pre-4.16 behaviour for
    interoperability with HDF5<1.10 readers.

    **Legacy multi-process restriction (v4.14.3, NOW LIFTED).**
    Prior to v4.16.0 this docstring warned against multi-process
    use entirely.  That restriction is now relaxed via the
    filelock + SWMR mechanisms above.  Single-process callers see
    NO behaviour change: the v4.14.3 atomicity guarantees are
    preserved bit-for-bit; the lock is uncontended in the
    single-process path and adds <1 ms of overhead per append.
    """
    _require_h5py()
    if dy is None:
        dy = dx
    from ..backend import to_numpy
    field_np = (to_numpy(field) if not isinstance(field, np.ndarray)
                else field)
    if preserve_dtype:
        E = np.asarray(field_np)
    else:
        E = np.asarray(field_np, dtype=np.complex128)

    # v4.16.0: acquire the cross-process append lock BEFORE opening
    # the HDF5 file.  Two concurrent writer processes will be
    # serialised here -- second one blocks until first releases.
    # On timeout, raise TimeoutError with a clear pointer to the
    # lock file so the user can inspect / manually remove it if
    # a previous job died holding it (filelock cleans up on normal
    # exit but a SIGKILL leaves the FS-layer lock orphaned).
    _fl = _require_filelock()
    lock_path = _append_lock_path(filepath)
    file_lock = _fl.FileLock(lock_path, timeout=lock_timeout)
    try:
        file_lock.acquire()
    except _fl.Timeout as e:
        raise TimeoutError(
            f"append_plane_h5: failed to acquire multi-process lock "
            f"on {lock_path} within {lock_timeout}s.  Another writer "
            f"process is holding the lock, or a previous job crashed "
            f"while holding it.  Inspect or remove {lock_path} to "
            f"recover."
        ) from e

    try:
        # v4.16.0 SWMR: ``libver='latest'`` is required to enable
        # ``swmr_mode``.  We keep ``libver='earliest'`` (the h5py
        # default) when SWMR is disabled so legacy readers stay
        # happy.  The ``planes`` group MUST exist before we set
        # ``f.swmr_mode = True`` -- SWMR freezes the schema, so a
        # later ``create_group`` would raise.
        h5py_kwargs = dict()
        if swmr:
            h5py_kwargs['libver'] = 'latest'
        with h5py.File(filepath, 'a', **h5py_kwargs) as f:
            if 'planes' not in f:
                grp = f.create_group('planes')
                grp.attrs['n_planes'] = 0
                # v4.15.0 (P2-VERSTAMP): group-level stamp on
                # first-touch.  Note we don't overwrite the stamp on
                # a re-open (the ``else`` branch below) so the value
                # remains the version that *created* the file.
                # Per-plane stamps below carry the version that wrote
                # each plane, so multi-version appends are still
                # resolvable from the per-dataset attr.
                grp.attrs['lumenairy_version'] = _get_lumenairy_version()
            else:
                grp = f['planes']

            n = int(grp.attrs.get('n_planes', 0))
            name = f'plane_{n:02d}'
            Ny, Nx = E.shape
            chunks = (min(chunk_size, Ny), min(chunk_size, Nx))
            ds_kwargs = dict(chunks=chunks)
            if compression is not None:
                ds_kwargs['compression'] = compression
                if compression_opts is not None:
                    ds_kwargs['compression_opts'] = compression_opts
            # v4.14.3 (P0-NEW-1): reserve the slot atomically by
            # bumping ``n_planes`` BEFORE the dataset is created.  If
            # ``create_dataset`` crashes (disk full, dtype mismatch,
            # etc.) the increment is rolled back so a subsequent
            # append re-uses this same slot rather than skipping it.
            # The reverse order (create, then bump) left an orphan
            # dataset on crash that collided with the next append's
            # computed name.
            grp.attrs['n_planes'] = n + 1
            try:
                dset = grp.create_dataset(name, data=E, **ds_kwargs)
                dset.attrs['dx'] = float(dx)
                dset.attrs['dy'] = float(dy)
                dset.attrs['dtype'] = str(E.dtype)
                # v4.15.0 (P2-VERSTAMP): per-plane writer stamp.
                dset.attrs['lumenairy_version'] = _get_lumenairy_version()
                if z is not None:
                    dset.attrs['z'] = float(z)
                if label is not None:
                    dset.attrs['label'] = str(label)
                # A-4 (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25): route
                # ``metadata`` through the canonical type-tagged codec
                # instead of handing raw values to h5py attrs.  This
                # SUPERSEDES the S4-9 ``None``-skip: the blob stores
                # ``None`` faithfully (matching zarr, which was S4-9's
                # stated goal) instead of dropping it, and nested /
                # heterogeneous / empty containers no longer raise.
                _h5_write_meta_attrs(dset, metadata)
                # v4.16.0 SWMR: enable single-writer-multiple-reader
                # mode AFTER the new dataset + attributes are in
                # place.  HDF5 SWMR requires that no schema changes
                # happen after ``swmr_mode = True``, so this MUST
                # come at the end of the writer's schema work.  The
                # next writer open re-creates its handle so this
                # flag is per-open, not persisted (the ``libver``
                # marker IS persisted and makes the file SWMR-
                # compatible across opens).  After this point,
                # reader processes can open the file with
                # ``swmr=True`` and follow the new plane.
                #
                # Wrap in try/except because some h5py versions /
                # HDF5 builds raise if the file was opened with a
                # libver lower than 'latest' (which we just
                # guaranteed above when ``swmr=True``), or on
                # rare race conditions where the SWMR transition
                # fails.  We follow with a final flush either way
                # so readers see consistent state.
                if swmr:
                    try:
                        f.swmr_mode = True
                    except (ValueError, RuntimeError):
                        # Fall back: filelock still serialises
                        # writers; only the concurrent-reader
                        # guarantee is forfeited.
                        pass
                # v4.16.0: flush in all modes so even the non-SWMR
                # legacy path commits the new plane to disk before
                # ``close()`` triggers final flush.  Helps readers
                # that opportunistically retry-open.
                f.flush()
            except Exception:
                # Roll back the slot-reservation so a re-try lands on
                # the same name (don't strand the slot).  Narrow the
                # exception tuple to (RuntimeError, OSError,
                # ValueError, TypeError) would be ideal but the h5py
                # create-dataset path can raise anything from the
                # HDF5 native layer; preserve the original Exception
                # class for the user via ``raise``.
                # v5.4.6 (audit P3-15): if create_dataset SUCCEEDED but a
                # later attrs/swmr/flush step raised, the orphan plane_NN
                # dataset would block the next append at the same name
                # (overwrite was removed in v4.14.3).  Delete it so the
                # file sits at exactly n planes -- the documented
                # atomicity contract -- before decrementing n_planes.
                try:
                    if name in grp:
                        del grp[name]
                except Exception:
                    pass
                grp.attrs['n_planes'] = n
                raise
    finally:
        # Always release the multi-process lock, even on exception.
        # ``filelock.FileLock.release()`` is idempotent so a
        # second-time release (e.g. if filelock acquired-but-failed)
        # is harmless.
        file_lock.release()


def _h5_list_planes(filepath):
    """List planes in an HDF5 file without loading field data."""
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
        if 'planes' not in f:
            raise KeyError(
                f"File {filepath} does not contain a '/planes' group.")
        grp = f['planes']
        n = int(grp.attrs['n_planes'])
        file_metadata = _h5_read_attrs(grp)
        planes = []
        for i in range(n):
            name = f'plane_{i:02d}'
            if name not in grp:
                continue
            dset = grp[name]
            info = {'index': i, 'shape': tuple(dset.shape)}
            info.update(_h5_read_attrs(dset))
            planes.append(info)
    return planes, file_metadata


def _h5_load_plane_by_label(filepath, label_substring, *,
                            case_sensitive=False):
    """Load the first plane matching a label substring from HDF5."""
    _require_h5py()
    target = label_substring if case_sensitive else label_substring.lower()
    with h5py.File(filepath, 'r') as f:
        if 'planes' not in f:
            raise KeyError(
                f"File {filepath} does not contain a '/planes' group.")
        grp = f['planes']
        n = int(grp.attrs['n_planes'])
        for i in range(n):
            name = f'plane_{i:02d}'
            if name not in grp:
                continue
            dset = grp[name]
            label = _decode_attr(dset.attrs.get('label', ''))
            haystack = label if case_sensitive else label.lower()
            if target in haystack:
                plane = {'index': i, 'field': np.array(dset[()])}
                plane.update(_h5_read_attrs(dset))
                return plane
    raise KeyError(f"No plane in {filepath} with label containing "
                   f"{label_substring!r}")


def _h5_load_plane_slice(filepath, plane_index, y_slice, x_slice):
    """Load a rectangular slice of a plane from HDF5."""
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
        if 'planes' not in f:
            raise KeyError(f"File {filepath} has no '/planes' group")
        grp = f['planes']
        name = f'plane_{plane_index:02d}'
        if name not in grp:
            raise KeyError(f"Plane {name} not found in {filepath}")
        dset = grp[name]
        E_sub = dset[y_slice, x_slice]
        attrs = _h5_read_attrs(dset)
    return E_sub, attrs


def _h5_write_sim_metadata(filepath, metadata):
    """Write simulation metadata to HDF5 root attributes.

    Stores the canonical, type-tagged JSON blob under ``_META_BLOB_KEY``
    (authoritative for the faithful round-trip) plus a best-effort
    flattened copy of the scalars as native attributes (for external
    tools / older readers).  See the codec block above.
    """
    _require_h5py()
    blob = _meta_dumps(metadata)
    flat = _flatten_metadata(metadata)
    with h5py.File(filepath, 'a') as f:
        f.attrs[_META_BLOB_KEY] = blob
        for k, v in flat.items():
            if v is None:   # h5py cannot store a None attr; skip it
                continue
            # Best-effort external-inspection view only -- the blob is
            # authoritative, so a value h5py cannot store (e.g. bytes
            # with embedded NULLs raise ValueError, not TypeError) is
            # stringified or, failing that, skipped.  It never aborts.
            try:
                f.attrs[str(k)] = v
            except (TypeError, ValueError):
                try:
                    f.attrs[str(k)] = str(v)
                except (TypeError, ValueError):
                    pass


def _h5_read_sim_metadata(filepath):
    """Read simulation metadata from HDF5 root attributes.

    Prefers the canonical JSON blob (faithful round-trip).  Falls back
    to reconstructing a flat mapping from the individual native
    attributes when the blob is absent -- i.e. for files written by the
    pre-contract scheme (back-compat).
    """
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
        if _META_BLOB_KEY in f.attrs:
            return _meta_loads(f.attrs[_META_BLOB_KEY])
        meta = {}
        for k in f.attrs:
            v = f.attrs[k]
            if isinstance(v, bytes):
                v = v.decode()
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            meta[k] = v
    return meta


def list_h5_contents(filepath: str) -> Dict[str, Any]:
    """Print and return a summary of an HDF5 file's contents."""
    _require_h5py()
    info = {}
    with h5py.File(filepath, 'r') as f:
        print(f"HDF5 file: {filepath}")
        print("=" * 60)

        def _walk(name, obj):
            attrs = _h5_read_attrs(obj)
            if isinstance(obj, h5py.Dataset):
                entry = {
                    'type': 'dataset', 'shape': obj.shape,
                    'dtype': str(obj.dtype), 'attrs': attrs,
                }
                print(f"  {name} : dataset, shape={obj.shape}, "
                      f"dtype={obj.dtype}")
            else:
                entry = {'type': 'group', 'attrs': attrs}
                print(f"  {name}/ : group")
            for k, v in attrs.items():
                print(f"      @{k} = {v!r}")
            info[name] = entry

        f.visititems(_walk)
        root_attrs = _h5_read_attrs(f)
        if root_attrs:
            info['/'] = {'type': 'root', 'attrs': root_attrs}
            print("  / (root attrs):")
            for k, v in root_attrs.items():
                print(f"      @{k} = {v!r}")
    return info


class TempFieldStore:
    """Context manager for HDF5-backed temporary field storage.

    When a simulation step needs to free a large field from RAM
    temporarily, ``store()`` it to a temp HDF5 file and ``load()``
    it back later.  The temp file is cleaned up on context exit.
    """

    def __init__(self, prefix: str = 'op_temp_') -> None:
        _require_h5py()
        import tempfile as _tf
        self._tmpfile = _tf.NamedTemporaryFile(
            suffix='.h5', prefix=prefix, delete=False)
        self._path = self._tmpfile.name
        self._tmpfile.close()
        self._counter = 0

    def store(self, field: np.ndarray,
              dx: Optional[float] = None) -> str:
        """Write a field to the temp file and return a handle string."""
        name = f'tmp_{self._counter:04d}'
        self._counter += 1
        E = np.asarray(field)
        chunks = tuple(min(1024, s) for s in E.shape)
        with h5py.File(self._path, 'a') as f:
            dset = f.create_dataset(name, data=E, chunks=chunks,
                                    compression=None)
            if dx is not None:
                dset.attrs['dx'] = float(dx)
            # v4.15.0 (P2-VERSTAMP): temp-store stamps too, so a
            # crash-recovered process can detect cross-version temp
            # files if a job-runner reused the temp dir.
            dset.attrs['lumenairy_version'] = _get_lumenairy_version()
        return name

    def load(self, handle: str) -> np.ndarray:
        """Reload a field from the temp file by handle."""
        with h5py.File(self._path, 'r') as f:
            return np.array(f[handle])

    def load_slice(self, handle: str, *slices: Any) -> np.ndarray:
        """Load a sub-region of a stored field."""
        with h5py.File(self._path, 'r') as f:
            return np.array(f[handle][slices])

    def cleanup(self) -> None:
        """Delete the temp file."""
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def __enter__(self) -> "TempFieldStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()


# =========================================================================
# Zarr backend
# =========================================================================

def _require_zarr():
    try:
        import zarr
        return zarr
    except ImportError:
        raise ImportError(
            "zarr is required for Zarr storage. "
            "Install with: pip install zarr")


def _open_zarr_group_safe(zarr_mod, filepath, writable=False):
    """Open a zarr group safely across zarr v2/v3 on Windows.

    Background: zarr v3's ``LocalStore._open`` unconditionally calls
    ``Path.mkdir(parents=True, exist_ok=True)`` on the store root
    -- regardless of open mode (`'r+'`, `'a'`, `'w'`).  On Python 3.14
    + Windows this raises ``FileExistsError`` (WinError 183) when the
    directory already exists, despite the ``exist_ok=True`` flag --
    a platform / version-specific regression in CPython's path handling
    (documented CPython issue on Windows where mkdir with exist_ok
    still raises if the path is a directory that already exists).
    The bug is triggered on EVERY reopen because every mode's
    internal ``_open`` calls the same buggy mkdir.

    Workaround: wrap all zarr open_group calls in a context that
    monkey-patches ``pathlib.Path.mkdir`` to swallow FileExistsError
    when exist_ok=True (restoring the documented semantics).  The
    patch is scoped to the duration of the zarr call and restored in
    the finally block, so it never leaks to other code paths.

    Thread-safety (L8 fix, v4.13.0): the install / restore pair is
    serialised through a module-level :class:`threading.Lock`
    (``_ZARR_MKDIR_PATCH_LOCK``).  Two threads racing through
    ``append_plane_h5`` -> ``_open_zarr_group_safe`` previously could
    save the patched ``mkdir`` as the "original" and never restore
    the real implementation, permanently corrupting ``Path.mkdir`` for
    the whole process.  The lock is contended only on Windows where
    the patch is needed at all, and is uncontended on the (vastly more
    common) single-threaded code path.

    Parameters
    ----------
    zarr_mod : module
        The ``zarr`` module handle (passed through for cheap).
    filepath : str or Path
        Path to the zarr store (directory) to open.
    writable : bool, optional
        If False, open read-only (``'r'``) and ignore the Windows bug
        entirely (read path never mkdir's).
    """
    if not writable:
        return zarr_mod.open_group(str(filepath), mode='r')

    # Local monkey-patch of Path.mkdir so zarr's internal call
    # ``self.root.mkdir(parents=True, exist_ok=True)`` correctly
    # no-ops when the directory already exists, as the documented
    # Python semantics promise but Windows + Python 3.14 currently
    # break.
    from pathlib import Path as _PL

    # Acquire the module-level lock before the install/restore window
    # to keep the original mkdir reference single-source.  If a second
    # thread arrives here it will block until we restore.
    with _ZARR_MKDIR_PATCH_LOCK:
        _orig_mkdir = _PL.mkdir

        def _patched_mkdir(self, mode=0o777, parents=False,
                           exist_ok=False):
            try:
                return _orig_mkdir(self, mode=mode, parents=parents,
                                   exist_ok=exist_ok)
            except FileExistsError:
                if exist_ok and self.is_dir():
                    return None    # documented exist_ok semantics
                raise

        _PL.mkdir = _patched_mkdir
        try:
            # Writable path: prefer r+ when the store already exists so
            # the patched mkdir is only needed as a safety net.
            if os.path.isdir(str(filepath)):
                try:
                    return zarr_mod.open_group(str(filepath), mode='r+')
                except (FileNotFoundError, KeyError):
                    # Directory exists but no zarr metadata yet -> fall
                    # through to creating mode.
                    pass
            return zarr_mod.open_group(str(filepath), mode='a')
        finally:
            _PL.mkdir = _orig_mkdir


def _zarr_append_plane(filepath, field, dx, dy=None, z=None, label=None,
                       metadata=None, chunk_size=1024,
                       preserve_dtype=False, lock_timeout=30.0,
                       **_kwargs):
    """Append one plane to a Zarr store, ``planes/plane_NN`` dataset.

    See :func:`append_plane_h5` for the atomicity contract.  Same
    fix: bump ``n_planes`` first, roll back on create failure.  In
    addition the Zarr code path historically passed
    ``overwrite=True`` to ``create_array``, which combined with the
    pre-fix attribute-after-create order meant a crashed-then-restarted
    appender would *silently clobber* the orphan dataset.  v4.14.3
    drops ``overwrite=True``: the attribute-before-create order leaves
    the orphan slot reserved (next append computes ``n + 1``), and if
    a stale ``plane_NN`` already exists at the computed slot the
    create now raises rather than silently overwriting.

    Parameters
    ----------
    lock_timeout : float, default 30.0
        v4.16.0+.  Seconds to wait for the multi-process append lock
        (sibling ``<filepath>.lock`` file) before raising
        :class:`TimeoutError`.  See :func:`append_plane_h5` for the
        full contract.

    Notes
    -----
    **Multi-writer support (v4.16.0).**  Zarr v3 has atomic
    ``create_array`` semantics within a single process but does not
    ship a built-in distributed lock for the read-modify-write
    sequence ``attrs['n_planes'] += 1`` + ``create_array``.  v4.16.0
    wraps the whole window in a :class:`filelock.FileLock` on the
    sibling ``<filepath>.lock`` file so cross-process appenders are
    serialised race-free.  Single-process callers see no behaviour
    change (the lock is uncontended and adds <1 ms overhead per
    append).
    """
    zarr = _require_zarr()
    if dy is None:
        dy = dx
    from ..backend import to_numpy
    field_np = (to_numpy(field) if not isinstance(field, np.ndarray)
                else field)
    if preserve_dtype:
        E = np.asarray(field_np)
    else:
        E = np.asarray(field_np, dtype=np.complex128)
    Ny, Nx = E.shape
    chunks = (min(chunk_size, Ny), min(chunk_size, Nx))

    # v4.16.0: acquire the cross-process append lock BEFORE opening
    # the Zarr store.  Mirrors the HDF5 path; see ``append_plane_h5``
    # for the rationale.
    _fl = _require_filelock()
    lock_path = _append_lock_path(filepath)
    file_lock = _fl.FileLock(lock_path, timeout=lock_timeout)
    try:
        file_lock.acquire()
    except _fl.Timeout as e:
        raise TimeoutError(
            f"_zarr_append_plane: failed to acquire multi-process "
            f"lock on {lock_path} within {lock_timeout}s.  Another "
            f"writer process is holding the lock, or a previous job "
            f"crashed while holding it.  Inspect or remove "
            f"{lock_path} to recover."
        ) from e

    try:
        # Windows + Python 3.14 + zarr v3 workaround:
        # ``open_group(mode='a')`` internally does
        # ``Path.mkdir(parents=True, exist_ok=True)`` which raises
        # ``FileExistsError`` on this platform when the zarr
        # directory already exists.  Use ``r+`` for re-opens (no
        # mkdir) and only ``a`` when we genuinely need to create the
        # store.
        store = _open_zarr_group_safe(zarr, filepath, writable=True)
        if 'planes' not in store:
            planes_grp = store.create_group('planes')
            planes_grp.attrs['n_planes'] = 0
            # v4.15.0 (P2-VERSTAMP): mirror the HDF5 group-level
            # stamp.  Only set on initial create; per-plane stamps
            # below cover appended-by-different-version planes.
            planes_grp.attrs['lumenairy_version'] = (
                _get_lumenairy_version())
        else:
            planes_grp = store['planes']
        n = int(planes_grp.attrs.get('n_planes', 0))
        name = f'plane_{n:02d}'
        # v4.14.3 (P0-NEW-1): same attribute-before-create discipline
        # as the HDF5 path.  Roll back on any create failure so the
        # slot is not stranded.  ``overwrite=False`` (the default) is
        # now required to detect-and-raise on a stale orphan rather
        # than silently clobber it.
        planes_grp.attrs['n_planes'] = n + 1
        try:
            ds = planes_grp.create_array(
                name=name, data=E, chunks=chunks)
            ds.attrs['dx'] = float(dx)
            ds.attrs['dy'] = float(dy)
            # v4.15.0 (P2-VERSTAMP): per-plane stamp.
            ds.attrs['lumenairy_version'] = _get_lumenairy_version()
            if z is not None:
                ds.attrs['z'] = float(z)
            if label is not None:
                ds.attrs['label'] = str(label)
            if metadata:
                for k, v in metadata.items():
                    try:
                        ds.attrs[str(k)] = v
                    except TypeError:
                        ds.attrs[str(k)] = str(v)
        except Exception:
            # Roll back the slot reservation so a retry re-uses the
            # same name.  Re-raise the original exception class
            # unchanged.
            planes_grp.attrs['n_planes'] = n
            raise
    finally:
        # Always release the multi-process lock, even on exception.
        file_lock.release()


def _zarr_load_planes(filepath, indices=None):
    zarr = _require_zarr()
    store = zarr.open_group(filepath, mode='r')
    if 'planes' not in store:
        raise KeyError(f"Zarr store {filepath} has no 'planes' group")
    grp = store['planes']
    n = int(grp.attrs['n_planes'])
    if indices is None:
        indices = list(range(n))
    file_metadata = dict(grp.attrs)
    planes = []
    for i in indices:
        name = f'plane_{i:02d}'
        if name not in grp:
            raise KeyError(f"Plane {name} not found")
        ds = grp[name]
        plane = {'field': np.array(ds[:])}
        for k in ds.attrs:
            v = ds.attrs[k]
            if isinstance(v, bytes):
                v = v.decode()
            plane[k] = v
        planes.append(plane)
    return planes, file_metadata


def _zarr_list_planes(filepath):
    zarr = _require_zarr()
    store = zarr.open_group(filepath, mode='r')
    if 'planes' not in store:
        raise KeyError(f"Zarr store {filepath} has no 'planes' group")
    grp = store['planes']
    n = int(grp.attrs['n_planes'])
    file_metadata = dict(grp.attrs)
    planes = []
    for i in range(n):
        name = f'plane_{i:02d}'
        if name not in grp:
            continue
        ds = grp[name]
        info = {'index': i, 'shape': tuple(ds.shape)}
        for k in ds.attrs:
            v = ds.attrs[k]
            if isinstance(v, bytes):
                v = v.decode()
            info[k] = v
        planes.append(info)
    return planes, file_metadata


def _zarr_load_plane_by_label(filepath, label_substring,
                              case_sensitive=False):
    zarr = _require_zarr()
    target = label_substring if case_sensitive else label_substring.lower()
    store = zarr.open_group(filepath, mode='r')
    if 'planes' not in store:
        raise KeyError(f"Zarr store {filepath} has no 'planes' group")
    grp = store['planes']
    n = int(grp.attrs['n_planes'])
    for i in range(n):
        name = f'plane_{i:02d}'
        if name not in grp:
            continue
        ds = grp[name]
        label = ds.attrs.get('label', '')
        if isinstance(label, bytes):
            label = label.decode()
        haystack = label if case_sensitive else label.lower()
        if target in haystack:
            plane = {'index': i, 'field': np.array(ds[:])}
            for k in ds.attrs:
                v = ds.attrs[k]
                if isinstance(v, bytes):
                    v = v.decode()
                plane[k] = v
            return plane
    raise KeyError(f"No plane with label containing {label_substring!r}")


def _zarr_load_plane_slice(filepath, plane_index, y_slice, x_slice):
    zarr = _require_zarr()
    store = zarr.open_group(filepath, mode='r')
    if 'planes' not in store:
        raise KeyError(f"Zarr store {filepath} has no 'planes' group")
    grp = store['planes']
    name = f'plane_{plane_index:02d}'
    if name not in grp:
        raise KeyError(f"Plane {name} not found")
    ds = grp[name]
    E_sub = np.array(ds[y_slice, x_slice])
    attrs = dict(ds.attrs)
    return E_sub, attrs


def _zarr_write_sim_metadata(filepath, metadata):
    """Write simulation metadata to Zarr group attributes.

    Uses the same canonical contract as the HDF5 path: the type-tagged
    JSON blob under ``_META_BLOB_KEY`` (authoritative) plus a
    best-effort flattened copy for external inspection.  The blob is
    byte-for-byte identical to the HDF5 blob for the same input.
    """
    zarr = _require_zarr()
    store = _open_zarr_group_safe(zarr, filepath, writable=True)
    store.attrs[_META_BLOB_KEY] = _meta_dumps(metadata)
    flat = _flatten_metadata(metadata)
    for k, v in flat.items():
        # Best-effort external-inspection view (blob is authoritative).
        try:
            store.attrs[str(k)] = v
        except (TypeError, ValueError):
            try:
                store.attrs[str(k)] = str(v)
            except (TypeError, ValueError):
                pass


def _zarr_read_sim_metadata(filepath):
    """Read simulation metadata from Zarr group attributes.

    Prefers the canonical JSON blob; falls back to the flat native
    attributes for pre-contract stores (back-compat).
    """
    zarr = _require_zarr()
    store = zarr.open_group(filepath, mode='r')
    if _META_BLOB_KEY in store.attrs:
        return _meta_loads(store.attrs[_META_BLOB_KEY])
    meta = {}
    for k in store.attrs:
        v = store.attrs[k]
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        meta[k] = v
    return meta


# =========================================================================
# Backend registry
# =========================================================================
_BACKEND = 'hdf5'


def set_storage_backend(backend: str) -> None:
    """Set the storage backend for NEW file creation ('hdf5' or 'zarr').

    If ``backend='zarr'`` is requested but zarr is not installed this
    raises ``ImportError`` immediately.  Previously the check was
    lazy: ``set_storage_backend`` succeeded and the missing library
    only surfaced on the first ``append_plane`` call, which is harder
    to debug in a long-running simulation.
    """
    global _BACKEND
    if backend not in ('hdf5', 'zarr'):
        raise ValueError(f"backend must be 'hdf5' or 'zarr', got {backend!r}")
    if backend == 'zarr':
        try:
            import zarr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "zarr backend requested but zarr is not installed. "
                "Install with: pip install zarr"
            ) from e
    _BACKEND = backend


def get_storage_backend() -> str:
    """Return the current default storage backend name."""
    return _BACKEND


def default_extension() -> str:
    """Return the file extension for the current backend."""
    return '.zarr' if _BACKEND == 'zarr' else '.h5'


def _detect_backend(path):
    """Auto-detect backend from file extension or path structure.

    Supports both ``str`` and ``pathlib.Path`` inputs.  Returns
    ``'zarr'`` when the path either (a) has a ``.zarr`` extension, or
    (b) is an existing directory containing a Zarr store marker
    (``zarr.json`` for Zarr v3 at the group root, or ``.zarray`` for
    Zarr v2 at the array root).  Returns ``'hdf5'`` otherwise.

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
    """
    s = str(path)
    if s.endswith('.zarr'):
        return 'zarr'
    if os.path.isdir(s):
        # Restrict directory-routing to a real Zarr store via the
        # canonical store-marker files.  Zarr v3 group root:
        # ``zarr.json``.  Zarr v2 array root: ``.zarray``.  Without a
        # marker file the directory is treated as a non-Zarr
        # directory and dispatched to HDF5 (which will then raise a
        # clear "not an HDF5 file" error if the user pointed at a
        # generic directory).
        if (os.path.exists(os.path.join(s, 'zarr.json'))
                or os.path.exists(os.path.join(s, '.zarray'))):
            return 'zarr'
    return 'hdf5'


# =========================================================================
# Unified dispatch API
# =========================================================================

def append_plane(filepath: str, field: np.ndarray, dx: float,
                 dy: Optional[float] = None,
                 z: Optional[float] = None,
                 label: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 chunk_size: int = 1024,
                 preserve_dtype: bool = False,
                 **kwargs: Any) -> None:
    """Append a plane to a multi-plane file (HDF5 or Zarr, auto-dispatch).

    Backend detection precedence: existing file inspected first; then
    path extension (``.zarr`` -> zarr); else global ``_BACKEND``.

    ``preserve_dtype`` (v4.13.0 onward) forwards to the chosen backend's
    append impl so ``complex64`` inputs round-trip without being
    silently promoted to ``complex128``.  Default ``False`` matches the
    historical single-shot behaviour.

    ``**kwargs`` are forwarded to the chosen backend.  In particular,
    v4.16.0 adds:

    * ``swmr=True`` (HDF5 only) -- enable Single-Writer-Multiple-Reader
      mode so concurrent reader processes can follow appends safely.
    * ``lock_timeout=30.0`` (both backends) -- seconds to wait for the
      multi-process append lock before raising ``TimeoutError``.

    See :func:`append_plane_h5` for the full multi-writer / SWMR
    contract documentation.
    """
    if os.path.exists(filepath):
        backend = _detect_backend(filepath)
    elif str(filepath).endswith('.zarr'):
        backend = 'zarr'
    else:
        backend = _BACKEND
    if backend == 'zarr':
        # v4.16.0: forward lock_timeout to the zarr path; drop
        # ``swmr`` (HDF5-only) silently to keep the unified API
        # ergonomic for backend-agnostic call sites.
        zarr_kwargs = {k: v for k, v in kwargs.items()
                       if k in ('lock_timeout',)}
        _zarr_append_plane(filepath, field, dx, dy=dy, z=z, label=label,
                           metadata=metadata, chunk_size=chunk_size,
                           preserve_dtype=preserve_dtype, **zarr_kwargs)
    else:
        append_plane_h5(filepath, field, dx, dy=dy, z=z, label=label,
                        metadata=metadata, chunk_size=chunk_size,
                        preserve_dtype=preserve_dtype, **kwargs)


def load_planes(filepath: str,
                indices: Optional[Sequence[int]] = None
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load planes from a multi-plane file (auto-detected format)."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_load_planes(filepath, indices=indices)
    else:
        return load_planes_h5(filepath, indices=indices)


def list_planes(filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """List planes in a multi-plane file without loading data."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_list_planes(filepath)
    else:
        return _h5_list_planes(filepath)


def load_plane_by_label(filepath: str, label_substring: str, *,
                        case_sensitive: bool = False) -> Dict[str, Any]:
    """Load the first plane matching a label substring (auto-detected)."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_load_plane_by_label(filepath, label_substring,
                                         case_sensitive=case_sensitive)
    else:
        return _h5_load_plane_by_label(filepath, label_substring,
                                       case_sensitive=case_sensitive)


def load_plane_slice(filepath: str, plane_index: int,
                     y_slice: slice, x_slice: slice
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a rectangular slice of a plane (auto-detected format).

    Returns
    -------
    arr : ndarray
        The requested rectangular slice of the plane dataset.
    attrs : dict
        The full attribute dictionary of the plane dataset (label,
        coordinates, units, etc.), decoded to native Python types.

    Notes
    -----
    The ``(arr, attrs)`` tuple shape is shared between the HDF5 and
    Zarr backends; v4.13.2 pinned the docstring to match the
    long-standing dispatcher behaviour.
    """
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_load_plane_slice(filepath, plane_index,
                                      y_slice, x_slice)
    else:
        return _h5_load_plane_slice(filepath, plane_index,
                                    y_slice, x_slice)


def write_sim_metadata(filepath: str, metadata: Dict[str, Any]) -> None:
    """Write simulation metadata to a file's root attributes.

    Backend detection precedence, in order:
      1. If the file/dir already exists, inspect it to pick h5 or zarr.
      2. If the path's EXTENSION is ``.zarr``, treat as zarr (this
         avoids a subtle bug where write_sim_metadata is called
         BEFORE any plane has been saved, so no store exists yet --
         falling back to the global ``_BACKEND`` default in that
         case would write an HDF5 file at a ``.zarr`` path, later
         corrupting the zarr dispatch.)
      3. Fall back to the global default ``_BACKEND``.
    """
    if os.path.exists(filepath):
        backend = _detect_backend(filepath)
    elif str(filepath).endswith('.zarr'):
        backend = 'zarr'
    else:
        backend = _BACKEND
    if backend == 'zarr':
        _zarr_write_sim_metadata(filepath, metadata)
    else:
        _h5_write_sim_metadata(filepath, metadata)


def read_sim_metadata(filepath: str) -> Dict[str, Any]:
    """Read simulation metadata from a file's root attributes."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_read_sim_metadata(filepath)
    else:
        return _h5_read_sim_metadata(filepath)


# Backwards-compatible aliases (old storage.py names)
list_planes_store = list_planes
load_plane_by_label_store = load_plane_by_label
load_plane_slice_store = load_plane_slice
write_metadata = write_sim_metadata
read_metadata = read_sim_metadata


def replay_run(filepath: str, *,
               label_prefix: Optional[str] = None,
               wavelength: Optional[float] = None,
               method: Optional[str] = None) -> Any:
    """Read every plane from a stored run and return a
    :class:`lumenairy.propagators.PropagationResult`.

    The store is the write-side end of the pipeline established by
    :meth:`MhsPipeline.run(store=...)`,
    :func:`propagate_through_system(store=...)`, and
    :class:`design_optimize(plane_logger=...)` (when the user wires
    a logger that calls :func:`append_plane`).  This function closes
    the loop: a single call returns the field at every plane in the
    order they were written, paired with their dx and label, so
    callers can replay, plot, or diff without re-running the
    simulation.

    Parameters
    ----------
    filepath : str or pathlib.Path
        HDF5 or Zarr store written via :func:`append_plane`.
    label_prefix : str, optional
        If given, only planes whose ``label`` starts with this prefix
        are included (e.g. ``label_prefix='mhs'`` to filter MHS-only
        planes from a mixed run).
    wavelength : float, optional
        Forwarded onto the result's ``wavelength`` field.  v5.24.x
        (audit S4-19): when omitted (``None``), the STORED wavelength is
        used -- the per-plane ``wavelength`` attribute of the last plane
        (preferred), then any per-plane value, then the file-level
        ``wavelength`` metadata -- falling back to ``0.0`` only when the
        store carries none.  ``append_plane`` has persisted per-plane
        wavelength since v4.0; pre-fix ``replay_run`` ignored it and the
        replayed result always reported ``0.0`` unless the caller
        re-supplied the wavelength by hand.
    method : str, optional
        Free-form method tag for the result (e.g. ``'mhs'``,
        ``'system'``).  Defaults to ``'replay'``.

    Returns
    -------
    PropagationResult
        ``.field`` is the LAST plane (the exit-plane field by
        convention).  ``.history`` carries every plane as
        ``(label, field, dx)`` tuples.  ``.dx`` is taken from the
        last plane.
    """
    from ..propagators.result import PropagationResult

    planes_info, _file_meta = list_planes(filepath)

    def _file_meta_wavelength():
        """File-level (run) wavelength, if the store carries one."""
        if isinstance(_file_meta, dict):
            wl = _file_meta.get('wavelength')
            if wl is not None:
                return float(wl)
        return None

    # Filter + sort by stored index.
    selected = []
    for info in planes_info:
        label = info.get('label', '') or ''
        if label_prefix is not None and not label.startswith(label_prefix):
            continue
        selected.append(info)
    selected.sort(key=lambda d: d.get('index', 0))

    if not selected:
        # v5.24.x (audit S4-19): even the empty-selection path honours a
        # stored file-level wavelength when the caller did not supply one.
        resolved_wl = wavelength
        if resolved_wl is None:
            resolved_wl = _file_meta_wavelength()
        return PropagationResult(
            field=np.zeros((0, 0), dtype=np.complex128),
            dx=0.0, wavelength=float(resolved_wl or 0.0),
            method=method or 'replay',
            history=[],
        )

    # Bulk-load the actual field arrays.
    indices = [info['index'] for info in selected]
    full_planes, _ = load_planes(filepath, indices=indices)

    history = []
    for info, plane in zip(selected, full_planes):
        history.append((
            plane.get('label', info.get('label', '')) or f"plane_{info.get('index', 0)}",
            plane['field'],
            float(plane.get('dx', info.get('dx', 0.0)) or 0.0),
        ))

    # v5.24.x (audit S4-19): resolve the reported wavelength from the
    # STORED per-plane attribute when the caller omitted it.  Prefer the
    # last plane (the exit-plane by convention), then any per-plane
    # value, then the file-level metadata; 0.0 only if the store carries
    # none.  ``append_plane`` has persisted per-plane wavelength since
    # v4.0, but pre-fix replay_run ignored it entirely.
    resolved_wl = wavelength
    if resolved_wl is None:
        for plane, info in zip(reversed(full_planes), reversed(selected)):
            wl = plane.get('wavelength', info.get('wavelength'))
            if wl is not None:
                resolved_wl = float(wl)
                break
    if resolved_wl is None:
        resolved_wl = _file_meta_wavelength()

    last_label, last_field, last_dx = history[-1]
    return PropagationResult(
        field=last_field, dx=last_dx,
        wavelength=float(resolved_wl or 0.0),
        method=method or 'replay',
        history=history,
        metadata={'source': str(filepath)},
    )


# A-13ish (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25): every module under
# ``analysis/`` declares its public surface with ``__all__``; the ``io/``
# modules did not, so ``from ... import *`` leaked private helpers and the
# ``__all__``-symmetry walkers had nothing to check against here.  This
# list is the module's public surface exactly as re-exported by
# ``lumenairy.io.__init__`` (plus the backwards-compatible ``*_store`` /
# ``*_metadata`` aliases defined above).
__all__ = [
    # HDF5-specific
    'save_field_h5',
    'load_field_h5',
    'save_planes_h5',
    'load_planes_h5',
    'save_jones_field_h5',
    'load_jones_field_h5',
    'append_plane_h5',
    'list_h5_contents',
    # Unified backend dispatch
    'set_storage_backend',
    'get_storage_backend',
    'default_extension',
    'append_plane',
    'load_planes',
    'list_planes',
    'load_plane_by_label',
    'load_plane_slice',
    'write_sim_metadata',
    'read_sim_metadata',
    'replay_run',
    'TempFieldStore',
    # Backwards-compatible aliases
    'list_planes_store',
    'load_plane_by_label_store',
    'load_plane_slice_store',
    'write_metadata',
    'read_metadata',
]
