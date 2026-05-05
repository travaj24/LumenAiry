"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.io.hdf5`."""
from .io import hdf5 as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
