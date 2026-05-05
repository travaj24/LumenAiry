"""
lumenairy.optimize -- prescription optimization, multi-config
parameterization, design merits.

Implementation in :mod:`lumenairy.optimize.core` (main optimizer
+ merits) and :mod:`lumenairy.optimize.multiconfig` (multi-config
parameterization).  This package's ``__init__`` mirrors both
submodules' namespaces so existing user imports of the form
``from lumenairy.optimize import X`` continue to work unchanged.
"""
from . import core as _core
from . import multiconfig as _multiconfig
globals().update({k: v for k, v in _core.__dict__.items() if not k.startswith('__')})
globals().update({k: v for k, v in _multiconfig.__dict__.items() if not k.startswith('__')})
