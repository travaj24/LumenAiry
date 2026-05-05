"""
lumenairy.asymptotic -- backwards-compat shim.

The implementation lives in :mod:`lumenairy.propagators.asymptotic`
as of v3.4.0; this shim re-exports the entire module namespace
(including private helpers) so existing user code that does
``from lumenairy.asymptotic import X`` continues to work.

New code should prefer ``from lumenairy.propagators.asymptotic
import X``.
"""
from .propagators import asymptotic as _impl
# Mirror the entire submodule namespace into this shim (includes
# private names like _multiply_polys_2d that some tests reach in for).
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
