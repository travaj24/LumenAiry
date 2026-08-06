# Shared design-121 setup for the 2026-07-30 tilted-coarse-leg scoping study.
#
# LOCAL-ONLY: needs the 121 .zmx and the design-study runner (Schott Sellmeier
# coefficients).  Factored out of fan_multi_121.py VERBATIM so every script in
# this study starts from a bit-identical prescription split, DOE period, order
# table and chain-A (source -> DOE) field.  Nothing here is new physics; it is
# the same construction fan_multi_121.py performs inline.
#
# Chain A is CACHED because it costs ~1-2 min and is order-INDEPENDENT (the
# DOE decomposition happens after it).  See ``chain_a`` / ``_chain_a_key`` for
# the cache key and why it is hashed rather than spelled into the filename.
import ast
import dataclasses
import hashlib
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

_HERE = os.path.dirname(os.path.abspath(__file__))
# LOCAL-ONLY paths.  ``D121_ROOT`` overrides the Windows dev-box location, so
# the same runners drive from the WSL CI proxy as well -- niche C13 needed the
# design's OWN per-order table on the other BLAS build, and this one literal
# was the only thing that stopped it (2026-08-03).
_ROOT = os.environ.get(
    'D121_ROOT', r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics")
sys.path.insert(0, os.path.join(_ROOT, "Lumenairy"))
sys.path.insert(0, os.path.join(_ROOT, "Reverse_Symmetric_ASM"))

import lumenairy as la  # noqa: E402
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS  # noqa: E402
from lumenairy.propagators.carrier import carrier_referenced_envelope  # noqa: E402
from lumenairy.raytrace import Surface  # noqa: E402
from lumenairy.raytrace.seidel import system_abcd_prescription  # noqa: E402
from lumenairy.raytrace.trace import surfaces_from_prescription  # noqa: E402

RUNNER = os.path.join(_ROOT, "Reverse_Symmetric_ASM",
                      "run_poc_119_120_v518.py")
ZMX = os.path.join(_ROOT, "Reverse_Symmetric_ASM", "tx4designstudy121",
                   "20260707 dll Tx02-MSOP16.zmx")
LAM = 1.31e-6
W0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3
FRAME_PITCH = 480e-6
DOE_ELEM = 6
ORDERS = (4, 8)
Z1 = 2e-3


def _register_glasses():
    src = open(RUNNER, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'_NEW_GLASSES\s*=\s*\{', src)
    i0 = m.end() - 1
    d = 0
    for i in range(i0, len(src)):
        if src[i] == '{':
            d += 1
        elif src[i] == '}':
            d -= 1
            if d == 0:
                break
    for g, (c, r, _nd) in ast.literal_eval(src[i0:i + 1]).items():
        if g not in SELLMEIER_COEFFICIENTS:
            SELLMEIER_COEFFICIENTS[g] = c
            GLASS_REGISTRY[g] = '__sellmeier__'
            GLASS_VALIDITY[g] = r


_register_glasses()
import copy  # noqa: E402

import tx_design_study_sim as sim  # noqa: E402


def geometry():
    """Return ``(groups_pre, groups_post, gap_to_doe, doe_period)``."""
    rx = la.load_zemax_zmx(ZMX)
    events = sim.group_elements_into_lenses(rx)
    rx_doe = copy.deepcopy(rx)
    for k in ('elements', 'all_thicknesses', 'surfaces', 'thicknesses'):
        rx_doe[k] = rx[k][DOE_ELEM:]
    b = float(system_abcd_prescription(rx_doe, LAM)[0][0, 1])
    period = LAM * abs(b) / FRAME_PITCH
    gapsum = 0.0
    first = True
    pre, post = [], []
    gap_to_doe = None
    for ev in events:
        if ev['type'] == 'doe' and gap_to_doe is None:
            gap_to_doe = ev['z_before'] + gapsum
            gapsum = 0.0
            continue
        if ev['type'] != 'lens':
            gapsum += ev['z_before']
            continue
        g = (OBJ_GAP - Z1) if first else (ev['z_before'] + gapsum)
        gapsum = 0.0
        (pre if gap_to_doe is None else post).append(
            {'prescription': ev['prescription'], 'gap_before': g})
        first = False
    return pre, post, gap_to_doe, period


def order_table(period, n_per=128):
    """Dammann order indices ``(mx, my)`` and complex amplitudes."""
    cache = os.path.join(_HERE, f'_dammann_121_{ORDERS[0]}x{ORDERS[1]}_'
                                f'{n_per}.npy')
    if os.path.exists(cache):
        nf = np.load(cache)
    else:
        nf, _f, _c = la.makedammann2d(
            periodx=period, periody=period, waveln=LAM,
            diforders=np.ones(ORDERS), phaselevels=8, phasesteps=4, itr=3000,
            seed=42, plot=False, cell_pixels=n_per)
        np.save(cache, nf)
    A = np.fft.fftshift(np.fft.fft2(nf)) / nf.size
    nx, ny = ORDERS
    cx = cy = n_per // 2
    P = np.abs(A) ** 2
    flat = np.argsort(P.ravel())[::-1][:nx * ny]
    oy, ox = np.unravel_index(flat, P.shape)
    mx, my = ox - cx, oy - cy
    o = np.lexsort((mx, my))
    mx, my = mx[o], my[o]
    return mx, my, A[my + cy, mx + cx]


# ===========================================================================
# chain-A cache key (defect D6, REVIEW_TRACED_EXACT_2026_08_05; fixed
# 2026-08-06)
# ===========================================================================
# HISTORY, because it is the whole justification for the machinery below.
# The key started as ``_chainA_{n}_{dx0}nm_rs{rs}.npz``.  ``final_leg`` was
# hard-coded 'paraxial' and absent, so switching the leg silently reloaded the
# paraxial field and looked like a no-op; 6dfc79d added ``final_leg`` to the
# filename, which was correct as far as it went.  It did not go far enough:
# THE SAME COMMIT FLIPPED TWO LIBRARY DEFAULTS THAT CHANGE THE CACHED FIELD --
# ``gap_kernel`` 'fresnel' -> 'auto' (= exact) and ``newton_fit`` 'polynomial'
# -> 'auto' (= spline on CPU) -- and neither is a ``chain_a`` argument, so
# neither could ever appear in a hand-spelled filename.  A cache written
# between those two changes holds a FRESNEL-kernel, POLYNOMIAL-fit field and
# would have been served, silently, as if it were exact/spline.
#
# WHAT IS KEYED NOW.  Everything the cached field can depend on:
#
#   1. schema salt (_CHAIN_A_SCHEMA) -- bump to orphan every cache by hand.
#   2. lumenairy.__version__.
#   3. A CONTENT HASH OF EVERY lumenairy/**/*.py.  This is the part that
#      catches a default flip WITHIN one version -- the exact D6 failure.  A
#      version string cannot: 6dfc79d flipped two defaults and shipped under
#      the same 5.32.1.  It also catches semantic flips that leave the default
#      STRING alone (e.g. 'auto' re-resolving from exact to fresnel) and
#      module-constant flips (_FOCUS_STANDOFF_ZR 6.0 -> 0.8), neither of which
#      any signature inspection can see.  Cost: ~40 ms over 218 files / 9.2 MB,
#      against the 1-2 min the cache saves.  CONSEQUENCE, stated plainly: any
#      edit anywhere in the library invalidates chain A.  That is the intended
#      trade -- on this branch the alternative is a plausible-looking wrong
#      answer, which is the exact failure class this campaign exists to remove.
#   4. The explicit chain_a arguments: n, dx0 (FULL float repr -- the old
#      '%.0f nm' rounding collided two pitches differing by < 0.5 nm), rs, nw.
#      ``nw`` (n_workers) is expected to be field-inert -- the Newton pool is
#      deterministic -- but it is a chain argument, it is not PROVEN inert, and
#      a cache is the wrong place to bet on that.
#   5. final_leg (the 6dfc79d fix, kept).
#   6. EVERY OTHER argument of ``propagate_traced_carrier_chain``, taken as its
#      signature default, since chain_a passes none of them -- so gap_kernel,
#      carrier_reference, na_exact_threshold, the guard actions and the
#      tolerances are all keyed by VALUE, and a default flip changes the key
#      even before the source hash notices.  Same for every default of
#      ``apply_real_lens_traced`` (newton_fit, amplitude_model,
#      preserve_input_phase, remap_sampling, newton_poly_order, ...), which the
#      chain drives with ``traced_kwargs=None``.
#   7. The d121 inputs this module owns: LAM, W0, OBJ_GAP, Z1, FRAME_PITCH,
#      DOE_ELEM, ORDERS -- they build env0, R1 and the group split.
#   8. A content hash of the .zmx, of the design-study RUNNER (its
#      ``_NEW_GLASSES`` Sellmeier coefficients are parsed and registered at
#      import, so they set the dispersion the trace uses) and of THIS module
#      (``geometry`` owns the group split; ``chain_a`` owns the source beam).
#      All three are inputs to the cached field.  A file that cannot be read
#      takes a distinct sentinel, so a cache written WITH it can never be read
#      back without it.
#
# The key dict is hashed into the filename (short, deterministic) AND stored
# verbatim inside the .npz, and the stored copy is CHECKED on load.  So even a
# file renamed onto a matching hash is refused rather than silently used.
_CHAIN_A_SCHEMA = 2


def _numba_available():
    # Read the LIBRARY's own gate, not a re-derived `import numba` probe:
    # the two disagree in the numba/numpy ABI-mismatch case (find_spec sees
    # the package, import fails), and it is the gate that selects the code
    # path whose output this cache stores (verifier round 2, V7 caveat).
    from lumenairy.elements import _lens_traced
    return bool(_lens_traced._NUMBA_AVAILABLE)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _lumenairy_source_sha():
    """SHA-256 over every ``.py`` in the lumenairy package actually imported
    (path-relative name + bytes, walked in sorted order, so it is stable
    across machines and independent of mtime / Syncthing copies)."""
    root = os.path.dirname(os.path.abspath(la.__file__))
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__')
        for name in sorted(filenames):
            if not name.endswith('.py'):
                continue
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).replace('\\', '/').encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()


def _signature_defaults(fn):
    """``{name: repr(default)}`` for every defaulted parameter of ``fn``."""
    import inspect
    return {k: repr(v.default)
            for k, v in inspect.signature(fn).parameters.items()
            if v.default is not inspect.Parameter.empty}


def _chain_a_key(n, dx0, rs, nw, final_leg):
    """The full parameter dict the chain-A cache is keyed on, and its hash.

    Returns ``(key_dict, hexdigest)``.  Deterministic: the dict is JSON-dumped
    with sorted keys, so the digest depends on values only, not on insertion
    order."""
    def _sha_or_sentinel(path):
        try:
            return _sha256_file(path)
        except OSError:
            return '<unreadable:%s>' % os.path.basename(path)

    key = {
        'schema': _CHAIN_A_SCHEMA,
        'lumenairy_version': str(la.__version__),
        'lumenairy_source_sha256': _lumenairy_source_sha(),
        # the design file (the prescription), the glass table's source (the
        # runner's _NEW_GLASSES Sellmeier coefficients, registered at import)
        # and THIS module (geometry() owns the group split, and chain_a owns
        # the source-beam construction) -- all three are inputs to the field.
        'zmx_sha256': _sha_or_sentinel(ZMX),
        'runner_sha256': _sha_or_sentinel(RUNNER),
        'd121_common_sha256': _sha_or_sentinel(os.path.abspath(__file__)),
        # explicit chain_a arguments
        'n': int(n),
        'dx0': repr(float(dx0)),
        'ray_subsample': int(rs),
        'n_workers': repr(nw),
        'final_leg': repr(final_leg),
        # every OTHER chain argument, at the default actually in force
        'chain_defaults': _signature_defaults(la.propagate_traced_carrier_chain),
        'traced_defaults': _signature_defaults(la.apply_real_lens_traced),
        # the d121 constants this module owns
        'LAM': repr(LAM), 'W0': repr(W0), 'OBJ_GAP': repr(OBJ_GAP),
        'Z1': repr(Z1), 'FRAME_PITCH': repr(FRAME_PITCH),
        'DOE_ELEM': int(DOE_ELEM), 'ORDERS': list(ORDERS),
        # verifier V7: numba availability perturbs the chain at ~5e-12
        # (the Chebyshev evaluator falls back to NumPy without it), so a
        # cache written on one runtime must not serve the other.
        'numba_available': _numba_available(),
    }
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'))
    return key, hashlib.sha256(blob.encode('ascii')).hexdigest()


def chain_a(n=1024, dx0=None, rs=4, nw=8, cache=True, final_leg='exact'):
    """Source envelope -> DOE plane.  Returns ``(env, R, dx, P_in)``.

    CACHING.  The cache file is named from a hash of :func:`_chain_a_key` --
    every parameter the field depends on, including the LIBRARY DEFAULTS in
    force and a content hash of the library source.  Read the block above
    ``_CHAIN_A_SCHEMA`` before changing any of it: the whole point is that a
    default flip (or any library edit) orphans the cache automatically instead
    of silently serving a field computed under the old physics.
    """
    dx0 = float(dx0 if dx0 is not None else 1.0e-6 * 2048 / n)
    key, digest = _chain_a_key(n, dx0, rs, nw, final_leg)
    key_blob = json.dumps(key, sort_keys=True, separators=(',', ':'))
    fn = os.path.join(_HERE, f'_chainA_v{_CHAIN_A_SCHEMA}_n{n}_rs{rs}_'
                             f'{digest[:12]}.npz')
    if cache and os.path.exists(fn):
        d = np.load(fn)
        # Belt AND braces: the hash names the file, the stored key proves it.
        # A file renamed / copied onto a matching name is refused, not used.
        stored = str(d['key_json']) if 'key_json' in d.files else None
        if stored != key_blob:
            raise RuntimeError(
                f"chain_a: {os.path.basename(fn)} does not carry the cache key "
                f"its name claims (stored key "
                f"{'absent' if stored is None else 'differs'}).  This file was "
                f"not written by this configuration -- delete it rather than "
                f"trust it.")
        return d['env'], float(d['R']), float(d['dx']), float(d['P_in'])
    pre, _post, gap_to_doe, _per = geometry()
    zR = np.pi * W0 * W0 / LAM
    w_z1 = W0 * np.sqrt(1 + (Z1 / zR) ** 2)
    R1 = Z1 * (1 + (zR / Z1) ** 2)
    x = (np.arange(n) - n // 2) * dx0
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                  / w_z1 ** 2).astype(np.complex128)
    P_in = float(np.sum(np.abs(env0) ** 2)) * dx0 * dx0
    res = la.propagate_traced_carrier_chain(
        env0, pre, LAM, dx0, r_in=R1, ray_subsample=rs, n_workers=nw,
        final_distance=gap_to_doe, final_leg=final_leg)
    env = carrier_referenced_envelope(res.field, res.R, LAM, res.dx)
    if cache:
        np.savez_compressed(fn, env=env, R=float(res.R), dx=float(res.dx),
                            P_in=P_in, key_json=np.array(key_blob))
    return env, float(res.R), float(res.dx), P_in


def post_surfaces(groups_post, trailing=TRAILING, stop_after=None,
                  back_off=0.0):
    """Sequential surface list for the post-DOE relay.

    ``stop_after`` (0-based group index) truncates after that group, and the
    trailing thickness becomes the following group's ``gap_before``.
    ``back_off`` pulls the final plane back by that many metres (used to place
    an exit REFERENCE plane short of the image plane)."""
    surfs = [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                     glass_before='air', glass_after='air', is_mirror=False,
                     thickness=float(groups_post[0]['gap_before']),
                     label='doe_plane')]
    last = len(groups_post) - 1 if stop_after is None else int(stop_after)
    for j, g in enumerate(groups_post[:last + 1]):
        sf = surfaces_from_prescription(g['prescription'])
        if j < last:
            nxt = float(groups_post[j + 1]['gap_before'])
        else:
            nxt = (float(trailing) if stop_after is None
                   else float(groups_post[j + 1]['gap_before']))
            nxt -= float(back_off)      # ONLY the final leg is pulled back
        sf[-1] = dataclasses.replace(sf[-1], thickness=nxt)
        surfs.extend(sf)
    surfs.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                         glass_before='air', glass_after='air',
                         is_mirror=False, thickness=0.0, label='img'))
    return surfs
