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

Author: Andrew Traverso
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

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
        if metadata:
            for key, value in metadata.items():
                dset.attrs[str(key)] = value


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
        metadata = {k: _decode_attr(dset.attrs[k])
                    for k in dset.attrs.keys()}
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
        if metadata:
            for k, v in metadata.items():
                grp.attrs[str(k)] = v
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
            for key, value in plane.items():
                if key == 'field':
                    continue
                dset.attrs[str(key)] = value
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
        file_metadata = {k: _decode_attr(grp.attrs[k])
                         for k in grp.attrs.keys()}
        planes = []
        for i in indices:
            name = f'plane_{i:02d}'
            if name not in grp:
                raise KeyError(f"Plane {name} not found in file")
            dset = grp[name]
            plane = {'field': np.array(dset[()])}
            for k in dset.attrs.keys():
                plane[k] = _decode_attr(dset.attrs[k])
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
        if metadata:
            for k, v in metadata.items():
                grp.attrs[str(k)] = v
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
        metadata = {k: _decode_attr(grp.attrs[k])
                    for k in grp.attrs.keys()}
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
                    preserve_dtype: bool = False) -> None:
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
        Per-plane stored attributes.
    compression, compression_opts
        HDF5 compression filter and level.
    chunk_size : int, default 1024
        HDF5 chunk-size cap (clipped to the field extent).
    preserve_dtype : bool, default False
        If True, store ``field`` at its native complex precision
        (``complex64`` or ``complex128``).  If False (the historical
        default), coerce to ``complex128`` for storage.  See
        :func:`save_field_h5` for the same flag.  v4.13.0 onward.

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

    **Multi-process restriction (v4.14.3).**  The single-process
    atomicity above is sufficient for the canonical sequential
    streaming-write path (HFPI, Monte-Carlo, ``MhsPipeline.run``).
    True multi-process concurrent appenders still require platform
    locking (HDF5 SWMR mode or a Zarr distributed lock).  Do not
    call ``append_plane_h5`` against the same file from multiple
    processes simultaneously.  HDF5 enforces single-writer semantics
    at the file-handle level so a concurrent ``File(..., 'a')`` will
    raise ``BlockingIOError`` on most platforms -- but this is a
    side-effect of the underlying library, not a guarantee.
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
    with h5py.File(filepath, 'a') as f:
        if 'planes' not in f:
            grp = f.create_group('planes')
            grp.attrs['n_planes'] = 0
            # v4.15.0 (P2-VERSTAMP): group-level stamp on first-touch.
            # Note we don't overwrite the stamp on a re-open (the
            # ``else`` branch below) so the value remains the version
            # that *created* the file.  Per-plane stamps below carry
            # the version that wrote each plane, so multi-version
            # appends are still resolvable from the per-dataset attr.
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
        # v4.14.3 (P0-NEW-1): reserve the slot atomically by bumping
        # ``n_planes`` BEFORE the dataset is created.  If
        # ``create_dataset`` crashes (disk full, dtype mismatch, etc.)
        # the increment is rolled back so a subsequent append re-uses
        # this same slot rather than skipping it.  The reverse order
        # (create, then bump) left an orphan dataset on crash that
        # collided with the next append's computed name.
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
            if metadata:
                for k, v in metadata.items():
                    dset.attrs[str(k)] = v
        except Exception:
            # Roll back the slot-reservation so a re-try lands on the
            # same name (don't strand the slot).  Narrow the exception
            # tuple to (RuntimeError, OSError, ValueError, TypeError)
            # would be ideal but the h5py create-dataset path can raise
            # anything from the HDF5 native layer; preserve the
            # original Exception class for the user via ``raise``.
            grp.attrs['n_planes'] = n
            raise


def _h5_list_planes(filepath):
    """List planes in an HDF5 file without loading field data."""
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
        if 'planes' not in f:
            raise KeyError(
                f"File {filepath} does not contain a '/planes' group.")
        grp = f['planes']
        n = int(grp.attrs['n_planes'])
        file_metadata = {k: _decode_attr(grp.attrs[k])
                         for k in grp.attrs.keys()}
        planes = []
        for i in range(n):
            name = f'plane_{i:02d}'
            if name not in grp:
                continue
            dset = grp[name]
            info = {'index': i, 'shape': tuple(dset.shape)}
            for k in dset.attrs.keys():
                info[k] = _decode_attr(dset.attrs[k])
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
                for k in dset.attrs.keys():
                    plane[k] = _decode_attr(dset.attrs[k])
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
        attrs = {k: _decode_attr(dset.attrs[k])
                 for k in dset.attrs.keys()}
    return E_sub, attrs


def _h5_write_sim_metadata(filepath, metadata):
    """Write simulation metadata to HDF5 root attributes."""
    _require_h5py()

    def _flatten(d, prefix=''):
        items = {}
        for k, v in d.items():
            key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    flat = _flatten(metadata)
    with h5py.File(filepath, 'a') as f:
        for k, v in flat.items():
            try:
                f.attrs[str(k)] = v
            except TypeError:
                f.attrs[str(k)] = str(v)


def _h5_read_sim_metadata(filepath):
    """Read simulation metadata from HDF5 root attributes."""
    _require_h5py()
    with h5py.File(filepath, 'r') as f:
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
            attrs = {k: _decode_attr(obj.attrs[k])
                     for k in obj.attrs.keys()}
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
        root_attrs = {k: _decode_attr(f.attrs[k])
                      for k in f.attrs.keys()}
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
                       preserve_dtype=False, **_kwargs):
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
    # Windows + Python 3.14 + zarr v3 workaround: ``open_group(mode='a')``
    # internally does ``Path.mkdir(parents=True, exist_ok=True)`` which
    # raises ``FileExistsError`` on this platform when the zarr
    # directory already exists.  Use ``r+`` for re-opens (no mkdir)
    # and only ``a`` when we genuinely need to create the store.
    store = _open_zarr_group_safe(zarr, filepath, writable=True)
    if 'planes' not in store:
        planes_grp = store.create_group('planes')
        planes_grp.attrs['n_planes'] = 0
        # v4.15.0 (P2-VERSTAMP): mirror the HDF5 group-level stamp.
        # Only set on initial create; per-plane stamps below cover
        # appended-by-different-version planes.
        planes_grp.attrs['lumenairy_version'] = _get_lumenairy_version()
    else:
        planes_grp = store['planes']
    n = int(planes_grp.attrs.get('n_planes', 0))
    name = f'plane_{n:02d}'
    # v4.14.3 (P0-NEW-1): same attribute-before-create discipline as
    # the HDF5 path.  Roll back on any create failure so the slot is
    # not stranded.  ``overwrite=False`` (the default) is now required
    # to detect-and-raise on a stale orphan rather than silently
    # clobber it.
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
        # Roll back the slot reservation so a retry re-uses the same
        # name.  Re-raise the original exception class unchanged.
        planes_grp.attrs['n_planes'] = n
        raise


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
    zarr = _require_zarr()
    store = _open_zarr_group_safe(zarr, filepath, writable=True)

    def _flatten(d, prefix=''):
        items = {}
        for k, v in d.items():
            key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    flat = _flatten(metadata)
    for k, v in flat.items():
        try:
            store.attrs[str(k)] = v
        except TypeError:
            store.attrs[str(k)] = str(v)


def _zarr_read_sim_metadata(filepath):
    zarr = _require_zarr()
    store = zarr.open_group(filepath, mode='r')
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
    """Auto-detect backend from file extension or path structure."""
    if path.endswith('.zarr') or os.path.isdir(path):
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
    """
    if os.path.exists(filepath):
        backend = _detect_backend(filepath)
    elif str(filepath).endswith('.zarr'):
        backend = 'zarr'
    else:
        backend = _BACKEND
    if backend == 'zarr':
        _zarr_append_plane(filepath, field, dx, dy=dy, z=z, label=label,
                           metadata=metadata, chunk_size=chunk_size,
                           preserve_dtype=preserve_dtype)
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
        Forwarded onto the result's ``wavelength`` field.  Default 0
        (storage doesn't carry wavelength as a per-plane attribute).
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

    # Filter + sort by stored index.
    selected = []
    for info in planes_info:
        label = info.get('label', '') or ''
        if label_prefix is not None and not label.startswith(label_prefix):
            continue
        selected.append(info)
    selected.sort(key=lambda d: d.get('index', 0))

    if not selected:
        return PropagationResult(
            field=np.zeros((0, 0), dtype=np.complex128),
            dx=0.0, wavelength=float(wavelength or 0.0),
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

    last_label, last_field, last_dx = history[-1]
    return PropagationResult(
        field=last_field, dx=last_dx,
        wavelength=float(wavelength or 0.0),
        method=method or 'replay',
        history=history,
        metadata={'source': str(filepath)},
    )
