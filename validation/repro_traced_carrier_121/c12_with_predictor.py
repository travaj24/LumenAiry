# Run any runner in this directory with the niche-C12 PHYSICS PREDICTOR (and
# niche C11's arbiter) pinned, WITHOUT editing that runner -- the same device
# as ``c11_with_arbiter.py`` / ``rc_with_gate.py``.
#
#   PRED=1 ARBITER=0 python c12_with_predictor.py focus_scan_121.py   # C12
#   PRED=0 ARBITER=1 python c12_with_predictor.py focus_scan_121.py   # C11
#   PRED=0 ARBITER=0 python c12_with_predictor.py focus_scan_121.py   # v5.32
#
# BOTH flags are pinned EXPLICITLY so no arm depends on what the module default
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

_P, _A = os.environ.get('PRED'), os.environ.get('ARBITER')
if _P is None or _A is None:
    raise SystemExit("set PRED=0|1 and ARBITER=0|1 -- both arms are pinned "
                     "explicitly, neither inherits the default")
LT.DECENTRED_FIT_PREDICTOR = _P not in ('0', '')
LT.DECENTRED_FIT_ARBITER = _A not in ('0', '')
if 'RESID_DEG' in os.environ:
    LT._REMAP_RESID_EIKONAL_DEGREE = int(os.environ['RESID_DEG'])
if 'SCORE_FLOOR' in os.environ:
    LT._DECENTRED_FIT_SCORE_FLOOR = float(os.environ['SCORE_FLOOR'])
_h = [hashlib.sha256(open(m.__file__, 'rb').read()).hexdigest()[:16]
      for m in (LT, CM)]
print(f"[c12_with_predictor] DECENTRED_FIT_PREDICTOR="
      f"{LT.DECENTRED_FIT_PREDICTOR}  "
      f"DECENTRED_FIT_ARBITER={LT.DECENTRED_FIT_ARBITER}  "
      f"score_floor={LT._DECENTRED_FIT_SCORE_FLOOR}  "
      f"spectrum_order={LT._DECENTRED_FIT_SPECTRUM_ORDER}\n"
      f"[c12_with_predictor] RESID_DEG={LT._REMAP_RESID_EIKONAL_DEGREE}  "
      f"_DECENTRE_GATE_W_FRAC={LT._DECENTRE_GATE_W_FRAC}\n"
      f"[c12_with_predictor] lumenairy {lumenairy.__version__} @ "
      f"{lumenairy.__file__}\n"
      f"[c12_with_predictor] _lens_traced {_h[0]}  carrier {_h[1]}", flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
