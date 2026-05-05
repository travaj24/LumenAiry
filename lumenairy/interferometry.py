"""Backwards-compat shim.  Implementation in
:mod:`lumenairy.analysis.interferometry`."""
from .analysis import interferometry as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
