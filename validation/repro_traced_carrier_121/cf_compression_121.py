# COMPRESSION MEASUREMENT for lumenairy.propagators.carrier_field's Zarr IO,
# on a REAL design-121 back-aperture field (8192^2 complex128 = 1.074 GB).
#
# WHY THE MEASUREMENT HAS TO BE ON A REAL FIELD.  A synthetic Gaussian
# compresses far better than a traced envelope: its mantissas are smooth
# everywhere, while a real aperture field carries the group's residual
# aberration, the ray-density amplitude and a numerical skirt.  Quoting a
# ratio from a fixture would overstate what a consumer will see by a lot.
#
# WHAT IS COMPARED, and why each row is there:
#   full field   -- the carrier NOT divided out.  The fringes of a converging
#                   NA-0.4 sphere are near-incompressible; this is the
#                   baseline the split has to beat.
#   envelope     -- the carrier divided out.  This is what a CarrierField
#                   stores, and the smoothness is the whole point of the
#                   split.
# and, on the envelope, four codec chains, so "zstd + shuffle" is a
# measurement rather than a habit:
#   zstd alone / shuffle+zstd / blosc-zstd-shuffle / blosc-zstd-bitshuffle.
#
#   python cf_compression_121.py [--order 0,0]
import argparse
import os
import shutil
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sumap_newapi_null_121 import LAM, _ap_path, load_meta  # noqa: E402
from zarr.codecs import (  # noqa: E402
    BloscCodec,
    BloscShuffle,
    BytesCodec,
    ZstdCodec,
)

from lumenairy.propagators.carrier_field import (  # noqa: E402
    CarrierField,
    CarrierSpec,
    FieldGrid,
    save_carrier_field_zarr,
)

SCRATCH = os.environ.get('CF_SCRATCH', r'C:\tmp\_cf_compress')


def _du(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _d, fs in os.walk(path) for f in fs)


def _shuffle_codec(itemsize):
    try:                                    # zarr >= 3.1.3
        from zarr.codecs.numcodecs import Shuffle
    except ImportError:                     # older zarr / numcodecs
        from numcodecs.zarr3 import Shuffle
    return Shuffle(elementsize=int(itemsize))


def measure(field, label, chains):
    raw = field.envelope.nbytes
    print(f"\n{label}   raw {raw / 1e9:.4f} GB  ({field.envelope.shape[0]}^2 "
          f"complex128)")
    print(f"  {'codec chain':<34s} {'stored':>12s} {'ratio':>8s} "
          f"{'GB/s w':>8s}")
    out = {}
    for name, (ser, comp) in chains.items():
        p = os.path.join(SCRATCH, 'x.zarr')
        shutil.rmtree(p, ignore_errors=True)
        t0 = time.perf_counter()
        save_carrier_field_zarr(p, field, name='f', serializer=ser,
                                compressors=comp, overwrite=True)
        dt = time.perf_counter() - t0
        n = _du(p)
        out[name] = (n, raw / n)
        print(f"  {name:<34s} {n / 1e9:>10.4f} GB {raw / n:>8.3f}x "
              f"{raw / 1e9 / dt:>8.3f}")
        shutil.rmtree(p, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', default='0,0')
    a = ap.parse_args()
    m, n = (int(v) for v in a.order.split(','))
    os.makedirs(SCRATCH, exist_ok=True)
    M_, _tile = load_meta(m, n)
    dx_k = float(M_['dx_ap'])
    grid = FieldGrid((int(M_['n_ap']), int(M_['n_ap'])), dx_k,
                     origin=tuple(M_['grid_origin']))
    car = CarrierSpec(R=float(M_['R_out']), centre=tuple(M_['chief_exit']),
                      tilt=tuple(M_['tilt_exit']), piston=0.0)
    E = np.load(_ap_path(m, n))

    chains = {
        'zstd(5) only, no shuffle':
            (BytesCodec(), [ZstdCodec(level=5)]),
        'numcodecs Shuffle(16) + zstd(5)':
            (BytesCodec(), [_shuffle_codec(16), ZstdCodec(level=5)]),
        'blosc zstd(5) + byte shuffle [DEFAULT]':
            (BytesCodec(), [BloscCodec(cname='zstd', clevel=5,
                                       shuffle=BloscShuffle.shuffle)]),
        'blosc zstd(5) + bitshuffle':
            (BytesCodec(), [BloscCodec(cname='zstd', clevel=5,
                                       shuffle=BloscShuffle.bitshuffle)]),
    }

    full = CarrierField(E, grid, CarrierSpec(R=float('inf')), LAM,
                        {'what': 'FULL field, carrier NOT divided out'})
    measure(full, f"FULL FIELD, order ({m:+d},{n:+d})", chains)
    del full

    f = CarrierField.from_full_field(E, grid, car, LAM,
                                     {'order': [m, n], 'what': 'envelope'})
    del E
    measure(f, f"ENVELOPE (carrier divided out), order ({m:+d},{n:+d})",
            chains)
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == '__main__':
    main()
