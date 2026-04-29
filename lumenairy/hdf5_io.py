"""
HDF5 I/O — backwards-compatible re-export shim.

All HDF5 I/O functions now live in :mod:`lumenairy.storage`,
which also provides unified HDF5/Zarr dispatch.  This module re-exports
the HDF5-specific functions so that existing ``from lumenairy.hdf5_io
import ...`` statements continue to work.

For new code, prefer importing from ``lumenairy.storage`` or
directly from the top-level ``lumenairy`` namespace.
"""

from .storage import (  # noqa: F401
    save_field_h5,
    load_field_h5,
    save_planes_h5,
    load_planes_h5,
    save_jones_field_h5,
    load_jones_field_h5,
    append_plane_h5,
    list_h5_contents,
    TempFieldStore,
    list_planes,
    load_plane_by_label,
    load_plane_slice,
    write_sim_metadata,
    read_sim_metadata,
)
