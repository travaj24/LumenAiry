"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.elements.polarization`."""
from .elements import polarization as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
