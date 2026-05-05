"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.io.prescriptions`."""
from .io import prescriptions as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
