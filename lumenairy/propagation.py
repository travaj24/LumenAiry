"""
lumenairy.propagation -- backwards-compat shim.

The implementation lives in :mod:`lumenairy.propagators.propagation`
as of v3.4.0; this shim mirrors the entire submodule namespace
(including private FFT-cache state and underscore-prefixed helpers)
so existing user code that does ``from lumenairy.propagation import
X`` continues to work unchanged.

New code should prefer ``from lumenairy.propagators.propagation
import X`` or ``from lumenairy import X`` (top-level re-export).
"""
from .propagators import propagation as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
