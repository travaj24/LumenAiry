"""v5.1.0 split: this file is the back-compat re-export shell.

The actual implementation moved to::

    lumenairy/raytrace/surface.py        -- data classes + sag / normal
    lumenairy/raytrace/intersection.py   -- intersection / refract /
                                            reflect / transfer / coord-break
    lumenairy/raytrace/trace.py          -- trace + prescription conv +
                                            ray generators + DOE + element bridge
    lumenairy/raytrace/world_trace.py    -- world-frame trace + helpers
    lumenairy/raytrace/seidel.py         -- paraxial + ABCD + Lens/Pupil /
                                            first-order + Seidel coefficients
    lumenairy/raytrace/ray_fan.py        -- spot / ray-fan / OPD /
                                            refocus / through-focus
    lumenairy/raytrace/layout.py         -- trace_summary / prescription_summary

Every public (and underscore-private "needed externally") name imported
from ``lumenairy.raytrace.core`` (or from ``lumenairy.raytrace``)
continues to resolve here, bit-for-bit identical to v5.0.x.

No physics change: the v5.1.0 split is purely a mechanical
reorganisation -- see ``docs/release_notes/.release_notes_v5_1_0_agent_b.md``.
"""

from __future__ import annotations

from .intersection import *  # noqa: F401,F403
from .layout import *  # noqa: F401,F403
from .ray_fan import *  # noqa: F401,F403

# Two world-frame analytics live in ``ray_fan`` but are kept off its
# ``__all__`` to preserve the pre-v5.1.0 "module-attribute visible but
# not in the advertised __all__" status: the pre-split
# ``lumenairy.raytrace.__init__`` did ``from .core import
# opd_fan_data_world, ray_fan_data_world`` (so they were attributes of
# ``lumenairy.raytrace.core`` and ``lumenairy.raytrace``) but did NOT
# add them to ``lumenairy.raytrace.__all__``.  Re-exporting them here
# by explicit name keeps ``from lumenairy.raytrace.core import
# ray_fan_data_world`` working bit-for-bit with pre-v5.1.0.
from .ray_fan import (  # noqa: F401
    opd_fan_data_world,
    ray_fan_data_world,
)
from .seidel import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Star-import every submodule so the legacy ``from lumenairy.raytrace.core
# import X`` (or the public ``from lumenairy.raytrace import X``) keeps
# resolving for every symbol that used to live here.  Each submodule
# defines its own ``__all__`` (or omits underscore-prefixed entries on
# purpose); the union of the seven captures the legacy public surface.
# ---------------------------------------------------------------------------
from .surface import *  # noqa: F401,F403
from .trace import *  # noqa: F401,F403
from .world_trace import *  # noqa: F401,F403

# NB: this re-export shell deliberately does NOT define ``__all__``.
# Pre-v5.1.0 ``core.py`` had no ``__all__`` either -- ``from
# lumenairy.raytrace.core import *`` therefore fell back to "every
# module-level non-underscore name" semantics, and the v4.16.0 walker
# symmetry test (tests/unit/test_v4_16_0_walker_all_symmetry.py)
# only examined submodules that DO define ``__all__``.  Adding an
# aggregated ``__all__`` here would advertise ``ray_fan_data_world`` /
# ``opd_fan_data_world`` as core's public surface even though
# ``lumenairy.__all__`` deliberately omits them -- a sibling-gap
# regression the walker would catch.  Keep ``__all__`` off this
# shell so the public surface stays exactly what pre-v5.1.0 promised.
