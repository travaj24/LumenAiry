"""Shared plumbing for the WP-B12b ROUND 2 probes (VERIFY-WP-B12b D-2..D-8).

Round 2 closes the two LIBRARY defects VERIFY-WP-B12b filed against
``lumenairy.propagators.gbd.apply_prescription_persurface_to_beamlets``:

* **D-5** -- an IMMERSED exit medium was reachable through the public
  ``apply_real_lens_gbd`` and served silently (1846.26 waves omitted at a
  2 mm leg in n = 1.72).  The same guard the four ``propagators.fga`` sites
  carry now runs at the GBD local branch, through the SAME helper, so the
  tolerance still has exactly one definition.
* **D-4** -- a MIRROR-terminated prescription through the LOCAL branch
  returned a wrong field silently, and the branch WP-B12b told the caller to
  use refuses a curved terminating mirror itself.  A named refusal now says
  so.

NOTHING is re-implemented here.  The fixtures and the 3-D oracle are the
VERIFIER's own (``validation/probe_verify_b12b/vb12b_common.py``), imported
by path from THIS file's tree, so a PRE arm running against an archived
``lumenairy`` still scores against one oracle and one fixture table.  The
library under test is whatever ``PYTHONPATH`` resolves, and its ARM is
DETECTED from the library itself -- never passed on the command line.

Author: Andrew Traverso
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_VB12B = _HERE.parent / 'probe_verify_b12b'
if str(_VB12B) not in sys.path:
    sys.path.insert(0, str(_VB12B))

import vb12b_common as VB  # noqa: E402

OUTDIR = _HERE


# ---------------------------------------------------------------------------
# which tree, which arm
# ---------------------------------------------------------------------------
def assert_tree(root_env='R2_TREE'):
    """``lumenairy`` must resolve under the tree this arm names.

    The same discipline VERIFY-WP-B12b used: the arm is a TREE, not a flag,
    and the assertion is made from inside the child process.
    """
    import lumenairy
    root = os.environ.get(root_env)
    got = Path(lumenairy.__file__).resolve()
    print(f'lumenairy.__file__ = {got}')
    if root:
        want = Path(root).resolve()
        assert str(got).lower().startswith(str(want).lower()), (
            f'{root_env}={want} but lumenairy resolved to {got}')
    return str(got)


def detect_arm():
    """PRE or POST, read off the LIBRARY's own source, never passed in.

    POST is the tree that carries the round-2 guards; the token searched for
    is the guard helper's NAME, which exists only after this package.
    """
    from lumenairy.propagators import gbd as G
    src = inspect.getsource(G)
    has_mirror_guard = '_require_forward_going_local_exit' in src
    has_immersed_guard = ('_require_non_immersed_exit' in src)
    return ('post' if (has_mirror_guard and has_immersed_guard) else 'pre',
            dict(mirror_guard=has_mirror_guard,
                 immersed_guard=has_immersed_guard))


def build_tag():
    return f'{sys.platform}_{sys.version_info.major}{sys.version_info.minor}'


def env_block():
    import numpy
    import scipy

    import lumenairy
    arm, tokens = detect_arm()
    return dict(
        arm=arm, arm_tokens=tokens,
        build=build_tag(),
        python=sys.version,
        numpy=numpy.__version__, scipy=scipy.__version__,
        platform=platform.platform(),
        lumenairy_file=str(Path(lumenairy.__file__).resolve()),
        lumenairy_version=getattr(lumenairy, '__version__', '?'),
        env={k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'LUMENAIRY_MEM_BUDGET_MB', 'R2_TREE', 'PYTHONPATH')},
    )


# ---------------------------------------------------------------------------
# digests / scores (the verifier's, re-exported so there is one definition)
# ---------------------------------------------------------------------------
sha = VB.sha
fidelity = VB.fidelity
rel_l2 = VB.rel_l2


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o


def dump(obj, name, tag=None):
    tag = tag or build_tag()
    path = OUTDIR / f'{name}_{tag}.json'
    with open(path, 'w', encoding='cp1252', newline='\n') as fh:
        json.dump(_jsonable(obj), fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(f'wrote {path}')
    return str(path)


def digest_bytes(*blobs):
    h = hashlib.sha256()
    for b in blobs:
        h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()[:24]


# ---------------------------------------------------------------------------
# my own fixtures -- a DIFFERENT optic from the verifier's, so "fires on the
# verifier's fixture and on mine" is two optics and not one twice
# ---------------------------------------------------------------------------
#: A model immersion medium registered probe-locally (water-like), and a
#: near-unity one, so the guard's boundary can be approached from a real
#: prescription as well as through the helper.
R2_MEDIA = {'R2-WATER133': 1.333, 'R2-OIL152': 1.5180}


def register_r2_media():
    from lumenairy import glass as _g
    for name, n in R2_MEDIA.items():
        _g.GLASS_REGISTRY[name] = (lambda v: (lambda wl: v))(n)
    VB.register_model_glasses()


#: My own optic: a faster, shorter-wavelength biconvex singlet in a different
#: model glass, with a CONIC last surface (so nothing here depends on the sag
#: repair) -- 633 nm, R1 = +4.0 mm, R2 = -4.0 mm, t = 0.8 mm, semi = 0.20 mm.
R2_LAM = 633e-9
R2_SEMI = 0.20e-3
R2_W0 = 0.12e-3
R2_N = 128
R2_DX = 3.0e-6
R2_GLASS = 'VB12B-M158'


def r2_prescription(exit_glass='air', mirror=False, R2=-4.0e-3, conic=0.0):
    """MY fixture, built the way the library documents, not the verifier's."""
    register_r2_media()
    if mirror:
        s = {'radius': R2, 'conic': conic, 'thickness': 0.0,
             'glass_before': 'air', 'glass_after': 'MIRROR',
             'semi_diameter': 0.30e-3}
        return {'name': 'r2_mirror', 'aperture_diameter': 0.60e-3,
                'surfaces': [s], 'thicknesses': [0.0], 'stop_index': 0}
    s0 = {'radius': 4.0e-3, 'conic': 0.0, 'thickness': 0.8e-3,
          'glass_before': 'air', 'glass_after': R2_GLASS,
          'semi_diameter': R2_SEMI}
    s1 = {'radius': R2, 'conic': conic, 'thickness': 0.0,
          'glass_before': R2_GLASS, 'glass_after': exit_glass,
          'semi_diameter': R2_SEMI}
    return {'name': 'r2_singlet', 'aperture_diameter': 2 * R2_SEMI,
            'surfaces': [s0, s1], 'thicknesses': [0.8e-3], 'stop_index': 0}


def r2_input_field(n=R2_N, dx=R2_DX, w0=R2_W0):
    xs = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


#: The beamlet frame every arm names, so the three public entry points are
#: comparable byte for byte (they have different sampling DEFAULTS).  Same
#: frame VERIFY-WP-B12b justified in its section 8.3.
FRAME = dict(sample_step=4, waist_factor=4.0)


def gbd_field(prescription, E, dx, lam, z, **kw):
    """The PUBLIC entry, with the frame named and the budget already pinned
    on the command line (VERIFY-WP-B12b D-6: the digest depends on it)."""
    import warnings

    import lumenairy as la
    k = dict(FRAME)
    k.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_gbd(
            E, prescription=prescription, wavelength=lam, dx=dx,
            output_plane_distance=float(z), **k))
