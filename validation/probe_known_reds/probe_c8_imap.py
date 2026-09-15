"""Probe 2: is the missing lobe caused by the G8 inverse-map refusal?

Cells: guard on/off x inverse_map on(default, may refuse)/off(fail-before
switch).  Reports the halo beyond 3 w, the G8 reading and whether the
inverse-map model was actually BUILT.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from probe_c8_guard_reach import _GHOST, call, halo  # noqa: E402
import lumenairy as la  # noqa: E402
from lumenairy.elements import _lens_imap as IM  # noqa: E402


def main():
    assert 'lum_reds' in la.__file__, la.__file__
    out = {'lumenairy_file': la.__file__,
           'TRACED_INVERSE_MAP_default': bool(IM.TRACED_INVERSE_MAP)
           if hasattr(IM, 'TRACED_INVERSE_MAP') else None,
           'INVERSE_MAP_GUARD_default': getattr(IM, 'INVERSE_MAP_GUARD', None),
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS')},
           'cells': {}}
    for guard in (False, True):
        for imap in (True, False):
            F, msgs = call(_GHOST, bound=False, guard=guard,
                           inverse_map=imap)
            refused = any('guard G8' in m for m in msgs)
            out['cells'][f'guard={guard},inverse_map={imap}'] = {
                'halo_3w': halo(F, _GHOST),
                'peak': float(np.abs(F).max()),
                'power': float((np.abs(F) ** 2).sum()),
                'g8_refused': refused,
                'g8_msg': next((m[:400] for m in msgs if 'guard G8' in m),
                               None)}
    print(json.dumps(out, indent=1))
    tag = os.environ.get('PROBE_TAG', 'default')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'c8_imap_{tag}.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
