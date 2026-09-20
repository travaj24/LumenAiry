"""High-budget byte identity: does the V-D5 threshold change move any bytes
at a budget where the WARNING differs between the two trees?"""
import hashlib
import json
import sys
import warnings

import numpy as np


def main(out):
    import lumenairy
    from lumenairy.propagators import _bluestein as BL
    rng = np.random.default_rng(20260920)
    keys = {}
    for (nin, nout) in ((24, 12), (32, 16)):
        E = (rng.standard_normal((nin, nin))
             + 1j * rng.standard_normal((nin, nin))).astype(np.complex128)
        for budget in (1e8, 1e10, 1e12, 1e15):
            alpha = budget / float(nin) ** 2
            for meth in ('auto', 'bluestein', 'separable', 'direct'):
                for prim in ('plain', 'centred'):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter('always')
                        fn = (BL._bluestein_2d if prim == 'plain'
                              else BL._bluestein_centred_2d)
                        F = fn(E, alpha, alpha, nout, nout, sign=-1, xp=np,
                               fft2=np.fft.fft2, ifft2=np.fft.ifft2,
                               method=meth)
                        warned = [str(x.message) for x in w
                                  if issubclass(x.category, RuntimeWarning)]
                    k = f"{prim}.{nin}x{nout}.b{budget:.0e}.{meth}"
                    keys[k] = hashlib.sha256(
                        np.ascontiguousarray(F).tobytes()).hexdigest()
                    keys[k + '.warned'] = hashlib.sha256(
                        ('|'.join(warned)).encode()).hexdigest()
                    keys[k + '.nwarn'] = str(len(warned))
    json.dump({'lumenairy_file': lumenairy.__file__,
               'platform': sys.platform, 'n': len(keys), 'keys': keys},
              open(out, 'w', encoding='cp1252'), indent=1, sort_keys=True)
    print(lumenairy.__file__, len(keys), '->', out)


main(sys.argv[1])
