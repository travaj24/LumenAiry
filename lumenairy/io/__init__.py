"""
lumenairy.io -- prescription I/O (Zemax, CODE V, Quadoa), HDF5 /
Zarr field storage, code generation.

Implementation in :mod:`lumenairy.io.prescriptions`,
:mod:`lumenairy.io.hdf5`, :mod:`lumenairy.io.storage`, and
:mod:`lumenairy.io.codegen`.  This package's ``__init__`` mirrors
all four submodule namespaces so existing user imports of the form
``from lumenairy.prescriptions import X`` (via the top-level shim)
or ``from lumenairy.io import X`` continue to work.
"""
from . import prescriptions as _p
from . import hdf5 as _h
from . import storage as _s
from . import codegen as _c
globals().update({k: v for k, v in _p.__dict__.items() if not k.startswith('__')})
globals().update({k: v for k, v in _h.__dict__.items() if not k.startswith('__')})
globals().update({k: v for k, v in _s.__dict__.items() if not k.startswith('__')})
globals().update({k: v for k, v in _c.__dict__.items() if not k.startswith('__')})
