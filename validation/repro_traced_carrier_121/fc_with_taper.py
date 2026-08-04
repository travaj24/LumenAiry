# Run any runner in this directory with ``_sphere_parab_conversion``'s cos^2
# taper forced to a chosen state, WITHOUT editing that runner -- the same
# device as ``c8_with_bound.py``, for the other knob this study prices.
#
#   TAPER=on   python fc_with_taper.py focus_scan_121.py      # shipped
#   TAPER=off  python fc_with_taper.py focus_scan_121.py      # T == 1
#   TAPER=2.0  python fc_with_taper.py focus_scan_121.py      # r_safe x 2
#
# The patch is installed on ``lumenairy.propagators.carrier`` itself (module
# attribute), which is where every call site resolves it, so a runner that
# imports the chain entry point still gets it.  ``LUMEN_C8`` mirrors
# ``c8_with_bound``'s knob so the two can be combined in one process.
import hashlib
import os
import runpy
import sys

# see fc_instrument_121.py: ``approx_common`` DEFAULTS LUMEN_PIN to a stale
# v5.31 export that still exists on this box.
os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as _C                                      # noqa: E402,F401
import approx_ablate_121 as AB                                 # noqa: E402
import lumenairy                                               # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402

# AFTER niche C9 landed in the working tree the LIBRARY DEFAULT is the exact
# (untapered) conversion, so ``TAPER=on`` no longer means "the taper".  Use
# ``C9=0`` to get the historical tapered library through its own fail-before
# switch -- that is the supported way to reproduce every pre-C9 number.  The
# TAPER monkeypatch is kept because every measurement taken BEFORE the flag
# existed used it, and the two must agree (they do: same function, and the
# ``C9=0`` / ``TAPER=off`` pair is checked in the audit).
if 'C9' in os.environ:
    CM.SPHERE_PARAB_CONVERSION_EXACT = os.environ['C9'] not in ('0', '')
_spec = os.environ.get('TAPER', 'on')
if _spec not in ('on', '1', ''):
    if _spec in ('off', '0'):
        _items = AB.p_sphere_taper_off()
    else:
        _items = AB.p_sphere_taper_scale(float(_spec))
    for _mod, _attr, _val in _items:
        setattr(_mod, _attr, _val)
if 'LUMEN_C8' in os.environ:
    LT.REMAP_INVERSE_SUPPORT_BOUND = os.environ['LUMEN_C8'] not in ('0', '')
_h = [hashlib.sha256(open(m.__file__, 'rb').read()).hexdigest()[:16]
      for m in (LT, CM)]
print(f"[fc_with_taper] TAPER={_spec}  C9="
      f"{getattr(CM, 'SPHERE_PARAB_CONVERSION_EXACT', 'ABSENT')}  "
      f"_sphere_parab_conversion={CM._sphere_parab_conversion.__qualname__}  "
      f"C8={LT.REMAP_INVERSE_SUPPORT_BOUND}\n"
      f"[fc_with_taper] lumenairy {lumenairy.__version__} @ "
      f"{lumenairy.__file__}\n"
      f"[fc_with_taper] _lens_traced {_h[0]}  carrier {_h[1]}", flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
