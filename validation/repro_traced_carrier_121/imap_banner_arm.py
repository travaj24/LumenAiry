# Banner arm driver: run focus_scan_121.py unmodified under a stated
# TRACED_INVERSE_MAP / newton_max_iters combination.  The point of the
# newton_max_iters arm is that it is an INDEPENDENT way to make the SHIPPED
# path more faithful: if the shipped banner walks toward the inverse map's
# answer when the incumbent's Newton is allowed to converge, the difference
# between the two banners is the incumbent's convergence and not the map.
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lumenairy.elements._lens_imap as IM          # noqa: E402
import lumenairy.elements._lens_traced as LT        # noqa: E402

IM.TRACED_INVERSE_MAP = os.environ.get('ARM_IMAP', '0') == '1'
if os.environ.get('ARM_NEWTON_ITERS'):
    LT._NEWTON_MAX_ITERS = int(os.environ['ARM_NEWTON_ITERS'])
print('ARM: TRACED_INVERSE_MAP=%s  _NEWTON_MAX_ITERS=%d'
      % (IM.TRACED_INVERSE_MAP, LT._NEWTON_MAX_ITERS), flush=True)
sys.argv = ['focus_scan_121.py']
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'focus_scan_121.py'), run_name='__main__')
