"""Back-compat shim for the renamed :mod:`lumenairy.analysis.core`.

In 4.7.0 the long-confusing :mod:`lumenairy.analysis.analysis` submodule
(the nested-name was an accident of history -- the package and its
"main" submodule had the same name) was renamed to
:mod:`lumenairy.analysis.core`.  This shim re-exports the new module's
public surface under the old dotted path so existing user code that
imported from ``lumenairy.analysis.analysis`` keeps working.

New code should import from :mod:`lumenairy.analysis` directly
(``from lumenairy.analysis import beam_centroid``) or from the
explicitly-named ``core`` submodule.
"""
from __future__ import annotations

from .core import *  # noqa: F401,F403
from .core import __all__  # noqa: F401
