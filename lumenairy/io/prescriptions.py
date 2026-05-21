"""
Lens prescription construction and Zemax / CODE V / Quadoa import utilities.

Provides functions to build lens prescription dicts (singlets, doublets),
a catalog of Thorlabs stock lenses, and parsers / writers for the
sequential lens-design file formats LumenAiry supports.  All
prescriptions use glass name strings rather than numeric refractive
indices so they remain wavelength-independent; indices are resolved
at runtime by the propagation engine.

v5.1.0 split (Agent F):  what used to be a single 3224-LOC monolith
is now organised across five sibling submodules.  This module is a
thin re-export shell -- public API is **unchanged**: every name
previously importable from :mod:`lumenairy.io.prescriptions` (and
re-exported through :mod:`lumenairy`) is still importable from the
same locations.  See the per-submodule docstrings for the source
implementation:

* :mod:`lumenairy.io.prescriptions_builders` -- ``make_singlet``,
  ``make_cylindrical``, ``make_biconic``, ``make_doublet``,
  ``make_off_axis_parabola``, ``THORLABS_CATALOG``, ``thorlabs_lens``.
* :mod:`lumenairy.io.prescriptions_zemax` -- ``load_zemax_zmx``,
  ``load_zemax_prescription_data_txt``, ``export_zemax_lens_data``,
  ``export_zemax_zmx``.
* :mod:`lumenairy.io.prescriptions_code_v` -- ``export_codev_seq``,
  ``load_codev_seq``.
* :mod:`lumenairy.io.prescriptions_quadoa` -- ``QUADOA_SCHEMA_VERSION``,
  ``export_quadoa_qos``, ``load_quadoa_qos``.
* :mod:`lumenairy.io.prescriptions_transforms` -- ``scale_prescription``,
  ``normalize_prescription``, ``split_prescription_at_mirrors``,
  ``has_mirrors``.

Author: Andrew Traverso
"""

from __future__ import annotations

# Re-exports from the v5.1.0 split submodules.  Public API is preserved
# bit-for-bit: every name previously defined in this module continues
# to live here as a re-exported alias so existing
# ``from lumenairy.io.prescriptions import X`` and
# ``from lumenairy import X`` continue to work unchanged.

from .prescriptions_builders import (
    make_singlet,
    make_cylindrical,
    make_biconic,
    make_doublet,
    make_off_axis_parabola,
    THORLABS_CATALOG,
    thorlabs_lens,
)
from .prescriptions_zemax import (
    load_zemax_zmx,
    load_zemax_prescription_data_txt,
    export_zemax_lens_data,
    export_zemax_zmx,
    _export_zemax_zmx_full,
)
from .prescriptions_code_v import (
    export_codev_seq,
    load_codev_seq,
)
from .prescriptions_quadoa import (
    QUADOA_SCHEMA_VERSION,
    _quadoa_serialize_radius,
    _quadoa_serialize_aspheric,
    _quadoa_deserialize_aspheric,
    export_quadoa_qos,
    load_quadoa_qos,
)
from .prescriptions_transforms import (
    scale_prescription,
    normalize_prescription,
    split_prescription_at_mirrors,
    has_mirrors,
)


__all__ = [
    # builders
    'make_singlet',
    'make_cylindrical',
    'make_biconic',
    'make_doublet',
    'make_off_axis_parabola',
    'THORLABS_CATALOG',
    'thorlabs_lens',
    # Zemax I/O
    'load_zemax_zmx',
    'load_zemax_prescription_data_txt',
    'export_zemax_lens_data',
    'export_zemax_zmx',
    # CODE V I/O
    'export_codev_seq',
    'load_codev_seq',
    # Quadoa I/O
    'QUADOA_SCHEMA_VERSION',
    'export_quadoa_qos',
    'load_quadoa_qos',
    # transforms
    'scale_prescription',
    'normalize_prescription',
    'split_prescription_at_mirrors',
    'has_mirrors',
]
