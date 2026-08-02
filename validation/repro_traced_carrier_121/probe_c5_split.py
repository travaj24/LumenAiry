# Niche C5: split the fix between the CHAIN's carrier bookkeeping and the
# ELEMENT's eikonal, so each half is scored on its own.
#
# ELEM=0 forces ``_tilted_carrier_parts`` back to the pre-C5 sphere-plus-ramp
# form for the duration of the run while the chain keeps the exact reference
# (and vice versa with CHAIN=0).  Physically inconsistent on purpose -- it is
# a differential diagnostic, not a proposal.
#
# Env: ORD, CHAIN=0|1, ELEM=0|1, plus every hybrid_localize_121 knob.
import io
import os
import re
import sys
import warnings
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')
import _d121_common as C  # noqa: E402,F401

import lumenairy.elements._lens_traced as _LT  # noqa: E402
import lumenairy.propagators.carrier as _CA  # noqa: E402


def main():
    chain = os.environ.get('CHAIN', '1') == '1'
    elem = os.environ.get('ELEM', '1') == '1'
    os.environ.setdefault('NMIN', '6')
    os.environ.setdefault('NMAX', '6')
    os.environ.setdefault('NOUT', '61')
    os.environ.setdefault('DXO', '0.4')
    os.environ.setdefault('NL', '121')
    _real_flag = _LT.TILTED_CARRIER_EXACT_EIKONAL
    _real_xf = _CA._tilt_exactness_phase
    _real_parts = _LT._tilted_carrier_parts

    def _no_xf(*a, **k):
        return None

    def _old_parts(spec, X, Y):
        _f = _LT.TILTED_CARRIER_EXACT_EIKONAL
        _LT.TILTED_CARRIER_EXACT_EIKONAL = False
        try:
            return _real_parts(spec, X, Y)
        finally:
            _LT.TILTED_CARRIER_EXACT_EIKONAL = _f

    if not chain:
        _CA._tilt_exactness_phase = _no_xf
    if not elem:
        # patch the ELEMENT's eikonal only -- the chain's exactness screen
        # reads the same flag, so flipping the flag would move both.
        _LT._tilted_carrier_parts = _old_parts
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            import hybrid_localize_121 as H
            H.main()
    finally:
        _LT.TILTED_CARRIER_EXACT_EIKONAL = _real_flag
        _CA._tilt_exactness_phase = _real_xf
        _LT._tilted_carrier_parts = _real_parts
    txt = buf.getvalue()
    ee = re.findall(r"EE3\s+([0-9.]+)\s+EE6\s+([0-9.]+)\s+EE12\s+([0-9.]+)",
                    txt)
    fw = re.findall(r"FWHM\s+([0-9.]+)\s+um", txt)
    print(f"RESULT  ORD={os.environ.get('ORD', '-4,-2')}  "
          f"CHAIN={'exact' if chain else 'sphere+ramp'}  "
          f"ELEM={'exact' if elem else 'sphere+ramp'}  "
          + "  ".join(f"EE3={a} EE6={b} EE12={c} FWHM={f}"
                      for (a, b, c), f in zip(ee, fw)))


if __name__ == '__main__':
    sys.exit(main())
