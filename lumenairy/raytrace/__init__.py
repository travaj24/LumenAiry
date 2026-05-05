"""
lumenairy.raytrace -- geometric ray tracing through sequential
optical prescriptions.

The implementation lives in :mod:`lumenairy.raytrace.core`.  This
package's ``__init__`` mirrors the entire submodule namespace
(including private helpers) so existing user imports of the form
``from lumenairy.raytrace import X`` continue to work unchanged.

New code may use either ``from lumenairy.raytrace import X`` (works
unchanged) or ``from lumenairy.raytrace.core import X`` (explicit).
"""
from . import core as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
