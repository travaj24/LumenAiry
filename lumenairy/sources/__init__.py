"""
lumenairy.sources -- source-field generators (Gaussian, plane wave,
fiber mode, point source, top hat, etc.).

The implementation lives in :mod:`lumenairy.sources.core`.  This
package's ``__init__`` mirrors the entire submodule namespace so
existing user imports of the form ``from lumenairy.sources import
X`` continue to work unchanged.
"""
from . import core as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
