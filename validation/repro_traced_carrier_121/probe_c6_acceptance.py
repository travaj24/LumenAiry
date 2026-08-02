# Niche C6: what the SHIPPED single-beam design-121 acceptance reads with the
# stationary-phase launch on and off.
#
# The acceptance runner is focus_scan_121.py at its shipping defaults
# (N=2048 / NFC=8192 / WF=4.0), whose recorded line is
# FWHM 3.450 um / EE3 88.8 / EE6 99.6 / EE12 99.8 at best focus, ON AXIS.
# C6 is NOT inert on that path -- the on-axis input residual is real
# (grad a rms 0.66 mrad) -- so this is the number the user has to decide about.
#
# usage:  CASES=off,on python probe_c6_acceptance.py
import os
import runpy
import sys

import lumenairy.elements._lens_traced as LT

HERE = os.path.dirname(os.path.abspath(__file__))

for case in os.environ.get('CASES', 'off,on').split(','):
    case = case.strip()
    LT.REMAP_STATIONARY_PHASE_LAUNCH = (case != 'off')
    print(f"\n########## SINGLE-BEAM ACCEPTANCE, C6 = {case} "
          f"(flag={LT.REMAP_STATIONARY_PHASE_LAUNCH}) ##########", flush=True)
    sys.argv = ['focus_scan_121.py']
    runpy.run_path(os.path.join(HERE, 'focus_scan_121.py'),
                   run_name='__main__')
