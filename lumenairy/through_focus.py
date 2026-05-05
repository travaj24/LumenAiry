"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.analysis.through_focus`."""
from .analysis import through_focus as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
