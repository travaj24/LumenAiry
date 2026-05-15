"""
lumenairy.io -- prescription I/O (Zemax, CODE V, Quadoa), HDF5 /
Zarr field storage, and code generation.

Submodules:

* :mod:`lumenairy.io.prescriptions` -- read/write Zemax ``.zmx``,
  Zemax ``.txt``, CODE V ``.seq``, Quadoa Optikos ``.qos``;
  prescription scaling, Thorlabs catalog, ``make_singlet`` /
  ``make_doublet`` / ``make_cylindrical`` / ``make_biconic``
  factories.
* :mod:`lumenairy.io.hdf5` -- single-field and multi-plane HDF5
  I/O.  (Most public HDF5 names are also re-exported through
  :mod:`lumenairy.io.storage`.)
* :mod:`lumenairy.io.storage` -- unified HDF5 / Zarr storage
  backend with auto-dispatch.
* :mod:`lumenairy.io.codegen` -- generate standalone simulation
  scripts from a prescription.
"""

from .prescriptions import (
    make_singlet,
    make_doublet,
    make_cylindrical,
    make_biconic,
    thorlabs_lens,
    load_zemax_zmx,
    load_zemax_prescription_data_txt,
    export_zemax_lens_data,
    export_zemax_zmx,
    load_codev_seq,
    export_codev_seq,
    export_quadoa_qos,
    load_quadoa_qos,
    QUADOA_SCHEMA_VERSION,
    scale_prescription,
    normalize_prescription,
    split_prescription_at_mirrors,
    has_mirrors,
    THORLABS_CATALOG,
)
from .storage import (
    set_storage_backend,
    get_storage_backend,
    default_extension,
    append_plane,
    load_planes,
    list_planes,
    load_plane_by_label,
    load_plane_slice,
    write_sim_metadata,
    read_sim_metadata,
    replay_run,
    list_planes_store,
    load_plane_by_label_store,
    load_plane_slice_store,
    write_metadata,
    read_metadata,
    save_field_h5,
    load_field_h5,
    save_planes_h5,
    load_planes_h5,
    save_jones_field_h5,
    load_jones_field_h5,
    append_plane_h5,
    list_h5_contents,
    TempFieldStore,
)
from .codegen import (
    generate_simulation_script,
    generate_script_from_zmx,
    generate_script_from_txt,
)


__all__ = [
    # prescriptions
    'make_singlet',
    'make_doublet',
    'make_cylindrical',
    'make_biconic',
    'thorlabs_lens',
    'load_zemax_zmx',
    'load_zemax_prescription_data_txt',
    'export_zemax_lens_data',
    'export_zemax_zmx',
    'load_codev_seq',
    'export_codev_seq',
    'export_quadoa_qos',
    'load_quadoa_qos',
    'QUADOA_SCHEMA_VERSION',
    'scale_prescription',
    'normalize_prescription',
    'split_prescription_at_mirrors',
    'has_mirrors',
    'THORLABS_CATALOG',
    # storage
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
    'list_planes_store',
    'load_plane_by_label_store',
    'load_plane_slice_store',
    'write_metadata',
    'read_metadata',
    'save_field_h5',
    'load_field_h5',
    'save_planes_h5',
    'load_planes_h5',
    'save_jones_field_h5',
    'load_jones_field_h5',
    'append_plane_h5',
    'list_h5_contents',
    'TempFieldStore',
    # codegen
    'generate_simulation_script',
    'generate_script_from_zmx',
    'generate_script_from_txt',
]
