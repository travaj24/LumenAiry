"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.elements.doe`."""
from .elements import doe as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
