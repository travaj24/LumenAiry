# Run any runner in this directory with the niche-C11 DECENTRED FIT ARBITER
# pinned, WITHOUT editing that runner -- ``rc_with_gate.py``'s device for the
# knob this study prices.
#
#   ARBITER=1 python c11_with_arbiter.py focus_scan_121.py     # shipped
#   ARBITER=0 python c11_with_arbiter.py focus_scan_121.py     # the fail-before
#
# Both arms are pinned EXPLICITLY so neither depends on what the module default
# happens to be -- the trap D121_FINAL_CLOSURE S10 item 7 records.
import hashlib
import os
import runpy
import sys

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as _C                                      # noqa: E402,F401
import lumenairy                                               # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402

_A = os.environ.get('ARBITER')
if _A is None:
    raise SystemExit("set ARBITER=0 or ARBITER=1 -- both arms are pinned "
                     "explicitly, neither inherits the default")
LT.DECENTRED_FIT_ARBITER = _A not in ('0', '')
if 'RESID_DEG' in os.environ:
    LT._REMAP_RESID_EIKONAL_DEGREE = int(os.environ['RESID_DEG'])
_h = [hashlib.sha256(open(m.__file__, 'rb').read()).hexdigest()[:16]
      for m in (LT, CM)]
print(f"[c11_with_arbiter] DECENTRED_FIT_ARBITER={LT.DECENTRED_FIT_ARBITER}  "
      f"RESID_DEG={LT._REMAP_RESID_EIKONAL_DEGREE}  "
      f"_DECENTRE_GATE_W_FRAC={LT._DECENTRE_GATE_W_FRAC}\n"
      f"[c11_with_arbiter] lumenairy {lumenairy.__version__} @ "
      f"{lumenairy.__file__}\n"
      f"[c11_with_arbiter] _lens_traced {_h[0]}  carrier {_h[1]}", flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
