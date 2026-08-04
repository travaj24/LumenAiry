# Run any runner in this directory with the niche-C1 DECENTRE GATE forced,
# WITHOUT editing that runner -- ``fc_with_taper.py``'s device for the knob
# this study prices.
#
#   GATE=shipped     python rc_with_gate.py energy_stage_audit_121.py
#   GATE=concentric  python rc_with_gate.py energy_stage_audit_121.py   # inf
#   GATE=offcentre   python rc_with_gate.py energy_stage_audit_121.py   # 0
#
# ``GATE=concentric`` forces EVERY traced element call onto the historical
# concentric ray-fit path -- hard NaN sample mask, ORIGIN-referenced beam
# radius (so the fit disc grows as the beam decentres), and the concentric fit
# order.  ``GATE=offcentre`` forces the D1/D7 weighted path on every call.
# Both are the library's own documented fail-before selectors, set through the
# module attributes every call site resolves.
#
# WHY THIS EXISTS.  ``rc_gate_121.py`` shows the branch is worth 0.88 EE3
# points at design 121's first order.  EE is a blind currency (the campaign has
# said so since the energy audit), so the same intervention has to be scored on
# CONSERVATION and HALO through ``energy_stage_audit_121.py`` UNEDITED -- and
# the concentric branch is exactly the one D1 replaced because its fit can go
# non-monotone outside the disc and let Newton find a second root.  If forcing
# it manufactures a fold or a lobe, that shows up there and not in EE3.
import hashlib
import os
import runpy
import sys

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as _C                                      # noqa: E402,F401
import lumenairy                                               # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402

_G = os.environ.get('GATE', 'shipped')
if _G == 'concentric':
    LT._DECENTRE_GATE_W_FRAC = float('inf')
elif _G == 'offcentre':
    LT._DECENTRE_GATE_W_FRAC = 0.0
    LT._DECENTRE_GATE_PIXELS = 0.0
elif _G != 'shipped':
    raise SystemExit(f"GATE must be shipped|concentric|offcentre, got {_G!r}")
if 'C9' in os.environ:
    CM.SPHERE_PARAB_CONVERSION_EXACT = os.environ['C9'] not in ('0', '')
if 'C6' in os.environ:
    LT.REMAP_STATIONARY_PHASE_LAUNCH = os.environ['C6'] not in ('0', '')
if 'RESID_DEG' in os.environ:
    LT._REMAP_RESID_EIKONAL_DEGREE = int(os.environ['RESID_DEG'])
if 'DEC_ORDER' in os.environ:
    LT._DECENTRED_FIT_POLY_ORDER = int(os.environ['DEC_ORDER'])
if 'LUMEN_C8' in os.environ:
    LT.REMAP_INVERSE_SUPPORT_BOUND = os.environ['LUMEN_C8'] not in ('0', '')
_h = [hashlib.sha256(open(m.__file__, 'rb').read()).hexdigest()[:16]
      for m in (LT, CM)]
print(f"[rc_with_gate] GATE={_G}  RESID_DEG={LT._REMAP_RESID_EIKONAL_DEGREE}  DEC_ORDER={LT._DECENTRED_FIT_POLY_ORDER}  _DECENTRE_GATE_W_FRAC="
      f"{LT._DECENTRE_GATE_W_FRAC}  _DECENTRE_GATE_PIXELS="
      f"{LT._DECENTRE_GATE_PIXELS}  C6={LT.REMAP_STATIONARY_PHASE_LAUNCH}  "
      f"C9={CM.SPHERE_PARAB_CONVERSION_EXACT}\n"
      f"[rc_with_gate] lumenairy {lumenairy.__version__} @ "
      f"{lumenairy.__file__}\n"
      f"[rc_with_gate] _lens_traced {_h[0]}  carrier {_h[1]}", flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
