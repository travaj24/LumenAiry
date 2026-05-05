"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.elements.freeform`."""
from .elements import freeform as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
