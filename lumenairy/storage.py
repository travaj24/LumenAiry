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

    import lumenairy as op

    # Choose backend for NEW files (default = 'hdf5')
    op.set_storage_backend('zarr')

    # Unified API — works with both HDF5 and Zarr:
    op.append_plane('output.zarr', E, dx=dx, label='After L1')
    planes = op.load_planes('output.zarr', indices=[0, 5])

    # HDF5-specific (explicit):
    op.save_field_h5('field.h5', E, dx=2e-6, wavelength=1.31e-6)
    E, meta = op.load_field_h5('field.h5')

Author: Andrew Traverso
"""

import os
import numpy as np

# =========================================================================
# HDF5 backend
# =========================================================================

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def _require_h5py():
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required for HDF5 I/O. "
                          "Install with: pip install h5py")


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

def save_field_h5(filepath, E, dx, dy=None, wavelength=None, label=None,
                  metadata=None, compression='gzip', compression_opts=4):
    """
    Save a single complex optical field to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output .h5 file path.
    E : ndarray (complex, Ny x Nx)
        Complex electric field.
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
    """
    _require_h5py()
    if dy is None:
        dy = dx
    E = np.asarray(E, dtype=np.complex128)
    with h5py.File(filepath, 'w') as f:
        dset = f.create_dataset(
            'field', data=E,
            compression=compression, compression_opts=compression_opts
        )
        dset.attrs['dx'] = float(dx)
        dset.attrs['dy'] = float(dy)
        if wavelength is not None:
            dset.attrs['wavelength'] = float(wavelength)
        if label is not None:
            dset.attrs['label'] = str(label)
        if metadata:
            for key, value in metadata.items():
                dset.attrs[str(key)] = value


def load_field_h5(filepath):
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

def save_planes_h5(filepath, planes, wavelength=None, metadata=None,
                   compression='gzip', compression_opts=4):
    """
    Save a sequence of complex fields to a single HDF5 file.

    Parameters
    ----------
    filepath : str
    planes : list of dict
        Each dict has ``'field'`` (ndarray), ``'dx'`` (float), and
        optional ``'dy'``, ``'z'``, ``'label'``, plus any extra keys.
    wavelength : float, optional
    metadata : dict, optional
    compression : str, default 'gzip'
    compression_opts : int, default 4
    """
    _require_h5py()
    n = len(planes)
    if n == 0:
        raise ValueError("At least one plane is required")
    with h5py.File(filepath, 'w') as f:
        grp = f.create_group('planes')
        grp.attrs['n_planes'] = n
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
            E = np.asarray(plane['field'], dtype=np.complex128)
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


def load_planes_h5(filepath, indices=None):
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

def save_jones_field_h5(filepath, jones_field, wavelength=None, label=None,
                        metadata=None, compression='gzip',
                        compression_opts=4):
    """Save a JonesField (polarized field) to an HDF5 file."""
    _require_h5py()
    Ex = np.asarray(jones_field.Ex, dtype=np.complex128)
    Ey = np.asarray(jones_field.Ey, dtype=np.complex128)
    with h5py.File(filepath, 'w') as f:
        grp = f.create_group('jones')
        grp.attrs['dx'] = float(jones_field.dx)
        grp.attrs['dy'] = float(jones_field.dy)
        if wavelength is not None:
            grp.attrs['wavelength'] = float(wavelength)
        if label is not None:
            grp.attrs['label'] = str(label)
        if metadata:
            for k, v in metadata.items():
                grp.attrs[str(k)] = v
        grp.create_dataset(
            'Ex', data=Ex,
            compression=compression, compression_opts=compression_opts)
        grp.create_dataset(
            'Ey', data=Ey,
            compression=compression, compression_opts=compression_opts)


def load_jones_field_h5(filepath):
    """
    Load a JonesField from an HDF5 file.

    Returns
    -------
    jones_field : JonesField
    metadata : dict
    """
    _require_h5py()
    from .polarization import JonesField
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

def append_plane_h5(filepath, field, dx, dy=None, z=None, label=None,
                    metadata=None, compression='gzip', compression_opts=4,
                    chunk_size=1024):
    """Append a single plane to a multi-plane HDF5 file (or create one)."""
    _require_h5py()
    if dy is None:
        dy = dx
    E = np.asarray(field, dtype=np.complex128)
    with h5py.File(filepath, 'a') as f:
        if 'planes' not in f:
            grp = f.create_group('planes')
            grp.attrs['n_planes'] = 0
        else:
            grp = f['planes']
        n = int(grp.attrs['n_planes'])
        name = f'plane_{n:02d}'
        Ny, Nx = E.shape
        chunks = (min(chunk_size, Ny), min(chunk_size, Nx))
        ds_kwargs = dict(chunks=chunks)
        if compression is not None:
            ds_kwargs['compression'] = compression
            if compression_opts is not None:
                ds_kwargs['compression_opts'] = compression_opts
        dset = grp.create_dataset(name, data=E, **ds_kwargs)
        dset.attrs['dx'] = float(dx)
        dset.attrs['dy'] = float(dy)
        if z is not None:
            dset.attrs['z'] = float(z)
        if label is not None:
            dset.attrs['label'] = str(label)
        if metadata:
            for k, v in metadata.items():
                dset.attrs[str(k)] = v
        grp.attrs['n_planes'] = n + 1


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


def list_h5_contents(filepath):
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

    def __init__(self, prefix='op_temp_'):
        _require_h5py()
        import tempfile as _tf
        self._tmpfile = _tf.NamedTemporaryFile(
            suffix='.h5', prefix=prefix, delete=False)
        self._path = self._tmpfile.name
        self._tmpfile.close()
        self._counter = 0

    def store(self, field, dx=None):
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
        return name

    def load(self, handle):
        """Reload a field from the temp file by handle."""
        with h5py.File(self._path, 'r') as f:
            return np.array(f[handle])

    def load_slice(self, handle, *slices):
        """Load a sub-region of a stored field."""
        with h5py.File(self._path, 'r') as f:
            return np.array(f[handle][slices])

    def cleanup(self):
        """Delete the temp file."""
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
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
    _orig_mkdir = _PL.mkdir

    def _patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
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
                       metadata=None, chunk_size=1024, **_kwargs):
    zarr = _require_zarr()
    if dy is None:
        dy = dx
    E = np.asarray(field, dtype=np.complex128)
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
    else:
        planes_grp = store['planes']
    n = int(planes_grp.attrs['n_planes'])
    name = f'plane_{n:02d}'
    ds = planes_grp.create_array(
        name=name, data=E, chunks=chunks, overwrite=True)
    ds.attrs['dx'] = float(dx)
    ds.attrs['dy'] = float(dy)
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
    planes_grp.attrs['n_planes'] = n + 1


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


def set_storage_backend(backend):
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


def get_storage_backend():
    """Return the current default storage backend name."""
    return _BACKEND


def default_extension():
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

def append_plane(filepath, field, dx, dy=None, z=None, label=None,
                 metadata=None, chunk_size=1024, **kwargs):
    """Append a plane to a multi-plane file (HDF5 or Zarr, auto-dispatch).

    Backend detection precedence: existing file inspected first; then
    path extension (``.zarr`` -> zarr); else global ``_BACKEND``."""
    if os.path.exists(filepath):
        backend = _detect_backend(filepath)
    elif str(filepath).endswith('.zarr'):
        backend = 'zarr'
    else:
        backend = _BACKEND
    if backend == 'zarr':
        _zarr_append_plane(filepath, field, dx, dy=dy, z=z, label=label,
                           metadata=metadata, chunk_size=chunk_size)
    else:
        append_plane_h5(filepath, field, dx, dy=dy, z=z, label=label,
                        metadata=metadata, chunk_size=chunk_size, **kwargs)


def load_planes(filepath, indices=None):
    """Load planes from a multi-plane file (auto-detected format)."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_load_planes(filepath, indices=indices)
    else:
        return load_planes_h5(filepath, indices=indices)


def list_planes(filepath):
    """List planes in a multi-plane file without loading data."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_list_planes(filepath)
    else:
        return _h5_list_planes(filepath)


def load_plane_by_label(filepath, label_substring, *,
                        case_sensitive=False):
    """Load the first plane matching a label substring (auto-detected)."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_load_plane_by_label(filepath, label_substring,
                                         case_sensitive=case_sensitive)
    else:
        return _h5_load_plane_by_label(filepath, label_substring,
                                       case_sensitive=case_sensitive)


def load_plane_slice(filepath, plane_index, y_slice, x_slice):
    """Load a rectangular slice of a plane (auto-detected format)."""
    backend = _detect_backend(filepath)
    if backend == 'zarr':
        return _zarr_load_plane_slice(filepath, plane_index,
                                      y_slice, x_slice)
    else:
        return _h5_load_plane_slice(filepath, plane_index,
                                    y_slice, x_slice)


def write_sim_metadata(filepath, metadata):
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


def read_sim_metadata(filepath):
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
