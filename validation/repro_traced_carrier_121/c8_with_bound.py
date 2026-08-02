# Run any runner in this directory with the niche-C8 support bound forced to a
# chosen state, WITHOUT editing that runner.  The bound is a module attribute,
# so setting it before the runner's ``main()`` is enough -- none of the runners
# touch it, and none of them fork.
#
#   C8=0        python c8_with_bound.py energy_stage_audit_121.py     # = HEAD
#   C8=1 C8F=1  python c8_with_bound.py energy_stage_audit_121.py     # bounded
#
# Everything after the runner's path is left in ``sys.argv`` for it.
import os
import runpy
import sys

import lumenairy.elements._lens_traced as LT

LT.REMAP_INVERSE_SUPPORT_BOUND = os.environ.get('C8', '1') not in ('0', '')
LT._SUPPORT_BOUND_FEATHER_CELLS = float(os.environ.get('C8F', '1'))
print("[c8_with_bound] REMAP_INVERSE_SUPPORT_BOUND=%s  feather=%g cells"
      % (LT.REMAP_INVERSE_SUPPORT_BOUND, LT._SUPPORT_BOUND_FEATHER_CELLS),
      flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
