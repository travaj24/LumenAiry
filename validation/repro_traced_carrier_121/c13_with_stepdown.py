# Run any runner in this directory with the niche-C13 CONDITIONING STEP-DOWN
# pinned, WITHOUT editing that runner -- ``c11_with_arbiter.py``'s device for
# the knob this study prices.
#
#   STEPDOWN=1 python c13_with_stepdown.py focus_scan_121.py     # shipped
#   STEPDOWN=0 python c13_with_stepdown.py focus_scan_121.py     # fail-before
#
# Both arms are pinned EXPLICITLY so neither depends on what the module default
# happens to be -- the trap D121_FINAL_CLOSURE S10 item 7 records.
#
# ``RESID_DEG`` is honoured too, because C13's whole subject is the interaction
# between the residual-eikonal DEGREE (niche C10) and the least-squares solve:
# the cliff this fixes appears only at degree 6, and the degree sweep has to be
# runnable with the step-down pinned either way.
import hashlib
import os
import runpy
import sys

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as _C                                      # noqa: E402,F401
import lumenairy                                               # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402

_S = os.environ.get('STEPDOWN')
if _S is None:
    raise SystemExit("set STEPDOWN=0 or STEPDOWN=1 -- both arms are pinned "
                     "explicitly, neither inherits the default")
LT.LSTSQ_CONDITIONING_STEPDOWN = _S not in ('0', '')
if 'RESID_DEG' in os.environ:
    LT._REMAP_RESID_EIKONAL_DEGREE = int(os.environ['RESID_DEG'])
# SELECTOR pins the 5.32.1 ray-fit branch ladder EXPLICITLY, because the
# default moved and an arm that inherited it would stop meaning what it says:
#   SELECTOR=predictor  -> the SHIPPED pair (C12 decides, C11 checks)
#   SELECTOR=arbiter    -> the C11 era (predictor off)
#   SELECTOR=gate       -> the v5.32 era, the fail-before for the flip
_SEL = os.environ.get('SELECTOR')
if _SEL is not None:
    if _SEL == 'predictor':
        LT.DECENTRED_FIT_PREDICTOR, LT.DECENTRED_FIT_ARBITER = True, True
    elif _SEL == 'arbiter':
        LT.DECENTRED_FIT_PREDICTOR, LT.DECENTRED_FIT_ARBITER = False, True
    elif _SEL == 'gate':
        LT.DECENTRED_FIT_PREDICTOR, LT.DECENTRED_FIT_ARBITER = False, False
    else:
        raise SystemExit("SELECTOR must be predictor|arbiter|gate, got "
                         f"{_SEL!r}")
_h = [hashlib.sha256(open(m.__file__, 'rb').read()).hexdigest()[:16]
      for m in (LT, CM)]
print(f"[c13_with_stepdown] LSTSQ_CONDITIONING_STEPDOWN="
      f"{LT.LSTSQ_CONDITIONING_STEPDOWN}  "
      f"rcond_min={LT._LSTSQ_GRAM_RCOND_MIN:.1e}  "
      f"resid_margin={LT._LSTSQ_RESID_MARGIN:.1e}  "
      f"RESID_DEG={LT._REMAP_RESID_EIKONAL_DEGREE}  "
      f"PREDICTOR={LT.DECENTRED_FIT_PREDICTOR}  "
      f"ARBITER={LT.DECENTRED_FIT_ARBITER}\n"
      f"[c13_with_stepdown] lumenairy {lumenairy.__version__} @ "
      f"{lumenairy.__file__}\n"
      f"[c13_with_stepdown] _lens_traced {_h[0]}  carrier {_h[1]}", flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
