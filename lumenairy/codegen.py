"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.io.codegen`."""
from .io import codegen as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
