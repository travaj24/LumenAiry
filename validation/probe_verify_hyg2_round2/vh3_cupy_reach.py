"""Where does an eager CuPy envelope now fail in the public Collins leg?"""
import json
import sys

import numpy as np


def main(out):
    import lumenairy
    from lumenairy.propagators import carrier as CA
    res = {'lumenairy_file': lumenairy.__file__}
    try:
        import cupy as cp
        res['cupy'] = cp.__version__
    except Exception as e:                      # noqa: BLE001
        res['cupy'] = f'UNAVAILABLE {type(e).__name__}'
        json.dump(res, open(out, 'w', encoding='cp1252'), indent=1)
        print(res)
        return
    n, dx = 64, 3e-6
    x = (np.arange(n) - n / 2.0) * dx
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / (24e-6) ** 2).astype(np.complex128)
    try:
        d = cp.asarray(env)
        res['device_array'] = type(d).__module__ + '.' + type(d).__name__
    except Exception as e:                      # noqa: BLE001
        res['device_array'] = f'{type(e).__name__}: {e}'
        json.dump(res, open(out, 'w', encoding='cp1252'), indent=1)
        print(res)
        return
    try:
        r = CA.propagate_carrier_referenced(
            d, 0.04, 5e-3, 1.55e-6, dx, dx_out=dx,
            on_collins_sampling='ignore', transport='collins')
        res['public_collins'] = ('OK ' + type(r.env).__module__ + '.'
                                 + type(r.env).__name__)
    except BaseException as e:                  # noqa: BLE001
        res['public_collins'] = f'{type(e).__name__}: {str(e)[:220]}'
    # where the failure sits: a bare cupy fft, for comparison
    try:
        cp.fft.fft2(d)
        res['cupy_fft2'] = 'OK'
    except BaseException as e:                  # noqa: BLE001
        res['cupy_fft2'] = f'{type(e).__name__}: {str(e)[:120]}'
    json.dump(res, open(out, 'w', encoding='cp1252'), indent=1)
    for k, v in res.items():
        print(f'  {k}: {v}')


main(sys.argv[1])
