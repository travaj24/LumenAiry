"""VERIFY ROUND 3 -- the verifier's OWN fixtures and helpers.

Written from scratch for this verification.  Nothing here is imported from
``validation/probe_fix_sliver_round3/`` or from
``validation/probe_verify_sliver_round2/``, and no number in the report is
read out of either one's JSON: every statistic is re-measured on the running
build with the devices below.

The GEOMETRIES are deliberately NOT the fix's.  The fix sizes its constant on
a 576-configuration box built from one wall pair (0.27865 / 0.62505), three
periods 0.9 / 1.2 / 1.6 um, wavelengths 0.85 / 1.05 um and superstrates
2.4 / 3.2.  The box below uses a different wall pair, three different periods,
two different wavelengths (a visible HeNe line and a 1.064 um line), different
superstrates, different lossy substrates, different ridge permittivities and
different grazing angles -- so "the correct population's finite drop envelope"
is measured on an independent sample rather than re-read.

Families
--------
``vstair``  the ordinary two-to-four-slice staircase whose walls open by
            ``delta`` of a period across the stack.  This is the family that
            manufactures a cross-layer sliver of exactly that width on the
            shared union grid, and it is the CORRECT population when ``delta``
            is small enough that the answer still tracks the physical shift.
``vgmr``    a guided-mode-resonance grating: a shallow high-index corrugation
            over a high-index slab, in a dense-superstrate grazing mount.
            Used both as a D-5 mount and as a resonant counter-fixture.
``vfp``     a Fabry-Perot cavity: two corrugated mirrors around a thick
            uniform spacer, so the reflectance is set by a cavity resonance.
``vwood``   a near-Wood mount: the incidence angle is solved so one diffracted
            order sits just inside its Rayleigh cutoff.
``vtensor`` / ``voop`` rotated uniaxial (in-plane and out-of-plane) directors.
``vliner``  a stack carrying a sliver-thin feature ONE layer owns.
``vplain``  an ordinary stack with NO manufactured cell at all.

Conventions
-----------
* ``err`` is the polarization-1 per-order distance to the exact ``delta -> 0``
  solve of the same device, the convention every probe in this campaign
  classifies with; ``classify`` is the fitted-constant-free rule
  ``err <= 10 delta`` RIGHT / ``err > 100 delta`` WRONG.
* ``move`` is what the library's arbiter compares: the largest per-order
  efficiency difference over the shared (centred) orders and over BOTH
  incident polarizations.
* every population statistic is taken with ``PMM_SLIVER_GUARD`` DISARMED, so
  nothing the library decides can feed back into the measurement, and the
  verdicts are then scored two ways -- analytically from the recorded
  ``(worst, su_snapped, move, w_wide)`` and by asking the library.
"""
import itertools
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

#: The tree this probe belongs to, pinned from ``__file__`` and put FIRST on
#: ``sys.path``.  Not decoration: this box carries an EDITABLE install of
#: ``lumenairy`` pointing at a different checkout, and an editable install
#: answers through a ``sys.meta_path`` finder.  ``python <probe>.py`` puts the
#: PROBE directory on ``sys.path[0]`` and does NOT put the working directory
#: there, so without this line a probe launched from a worktree root silently
#: imports the OTHER checkout.  Every probe here therefore reports the
#: resolved ``lumenairy`` path in its JSON, and :func:`assert_tree` refuses to
#: run against a tree other than its own.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if sys.path[:1] != [ROOT]:
    sys.path.insert(0, ROOT)

import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps


def assert_tree():
    """Fail loudly if ``lumenairy`` did not resolve inside :data:`ROOT`."""
    got = os.path.abspath(ps.__file__)
    if not got.lower().startswith(ROOT.lower() + os.sep):
        raise RuntimeError(
            f"probe tree mismatch: lumenairy resolved to {got}, expected a "
            f"module under {ROOT}.  Refusing to measure the wrong checkout.")
    return got

NO_SNAP = 1e-12          # a min_feature so fine the union snap never fires


# ------------------------------------------------------------ solving ------
def unguarded(st):
    """The pre-guard code path, bit for bit.

    Returns ``dict(o, R, T, worst)`` with the orders sorted ascending and
    ``R`` / ``T`` the full REAL ``(2, n_orders)`` arrays, so two solves on
    different grids can be compared on the orders they share."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    R = np.real(np.asarray(R))[:, i]
    T = np.real(np.asarray(T))[:, i]
    tot = R.sum(axis=-1) + T.sum(axis=-1)
    return dict(o=o[i], R=R, T=T, J=np.asarray(J),
                worst=float(np.max(tot)), least=float(np.min(tot)))


def guarded(st):
    """What the LIBRARY actually does, end to end:
    ``(refused, message, payload_or_None, warning_texts)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            o, R, T, J = st.solve()
        except ValueError as exc:
            return (True, str(exc), None, [str(w.message) for w in rec])
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return (False, "", dict(o=o[i], R=np.real(np.asarray(R))[:, i],
                            T=np.real(np.asarray(T))[:, i],
                            J=np.asarray(J)),
            [str(w.message) for w in rec])


def move_shared(a, b, *, pol=None):
    """Largest per-order efficiency difference over the orders two solves
    SHARE.  ``pol=None`` scores both polarizations (the library's arbiter);
    ``pol=1`` is the campaign's ``err`` convention."""
    c = np.intersect1d(a["o"], b["o"])
    if c.size == 0:
        return None
    ia = np.searchsorted(a["o"], c)
    ib = np.searchsorted(b["o"], c)
    sl = slice(None) if pol is None else slice(pol, pol + 1)
    return float(max(np.abs(a["R"][sl][:, ia] - b["R"][sl][:, ib]).max(),
                     np.abs(a["T"][sl][:, ia] - b["T"][sl][:, ib]).max()))


def classify(err, delta):
    """``err <= 10 delta`` RIGHT, ``err > 100 delta`` WRONG, else GREY."""
    if delta <= 0.0:
        return "ref"
    return ("wrong" if err > 100.0 * delta
            else "right" if err <= 10.0 * delta else "grey")


def screen_hit(st):
    """The library's OWN geometric screen on this stack, or ``None``."""
    return ps._cross_layer_sliver([L[1] for L in st._layers],
                                  float(st.min_feature) / float(st.period))


def prescribed(st):
    """``dict(mf, w_wide, w_narrow, own, n_hit)`` -- the ``min_feature`` the
    refusal's first remedy would prescribe, read off the library's own screen
    so this probe cannot drift from it."""
    hit = screen_hit(st)
    if hit is None:
        return None
    return dict(mf=2.0 * hit[3] * float(st.period), w_wide=hit[3],
                w_narrow=hit[0], own=hit[4], n_hit=int(hit[5]))


def snapped(st, mf):
    """The UNGUARDED solve of the same stack on the ``min_feature = mf``
    grid -- the arbiter's single measurement, taken here independently."""
    clone = st._min_feature_clone(float(mf))
    clone._src = dict(st._src)
    return unguarded(clone)


# ------------------------------------------------- the two criteria --------
def drop(worst, su_snap):
    """``(worst - 1) / su_snapped`` -- how much of the violation the
    prescribed snap REMOVES.  ``inf`` when the snapped solve reads at or
    below unity."""
    v = max(worst - 1.0, 0.0)
    return (v / su_snap) if su_snap > 0.0 else float("inf")


def v_round2(worst, su_snap, move, w_wide):
    """Round 2's criterion as an expression of the recorded quantities."""
    return ("sliver" if (su_snap <= 1.0e-5 and move > 100.0 * w_wide)
            else "truncation")


def v_round3(worst, su_snap, move, w_wide, frac):
    """The round-3 criterion at an arbitrary closure fraction."""
    bar = max(1.0e-5, max(worst - 1.0, 0.0) * frac)
    return ("sliver" if (su_snap <= bar and move > 100.0 * w_wide)
            else "truncation")


def closure_admits(worst, su_snap, frac):
    """The CLOSURE arm alone (the question the fix's sizing rule asks)."""
    return bool(su_snap <= max(1.0e-5, max(worst - 1.0, 0.0) * frac))


# =========================================================== devices =======
def vstair(delta, *, period, wl, theta, a0, b0, e_lo, e_hi, dz, nl=2,
           degree=8, nsub=1.0, nsup=1.0, mf=None, ffo=31):
    """``nl`` z-slices whose ridge walls open by ``delta`` (a fraction of a
    period) in total across the stack: layer ``k`` uses
    ``a0 - delta k/(nl-1)`` and ``b0 + delta k/(nl-1)``.  The union of those
    wall sets manufactures cells no single layer asked for -- the sliver."""
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(dz, segments=[(a0 - dd, e_lo),
                                   (b0 + dd - (a0 - dd), e_hi),
                                   (1.0 - (b0 + dd), e_lo)])
    st.set_source(wl, theta=theta)
    return st


def vgmr(delta, *, period=0.82e-6, wl=0.78e-6, theta=1.28, duty=0.46,
         degree=8, nsup=2.15, nsub=complex(1.52, 0.04), e_lo=4.41, e_hi=5.29,
         t_gr=0.24e-6, t_slab=0.13e-6, e_slab=5.29, nl=2, mf=None, ffo=15):
    """A GUIDED-MODE-RESONANCE grating: a shallow high-index corrugation over
    a high-index slab, in a DENSE-superstrate grazing mount.  The slab guides,
    the corrugation couples, and the far field near the resonance moves fast
    with any change to the duty cycle -- which is what makes it the natural
    attack on a bar expressed in units of a wall displacement."""
    a = 0.5 - duty / 2.0
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(t_gr / nl, segments=[(a - dd, e_lo),
                                          (duty + 2.0 * dd, e_hi),
                                          (1.0 - a - duty - dd, e_lo)])
    st.add_layer(t_slab, eps=e_slab)
    st.set_source(wl, theta=theta)
    return st


def vfp(delta, *, period=1.45e-6, wl=1.31e-6, theta=0.94, degree=8,
        nsup=1.78, nsub=complex(3.48, 0.6), a0=0.2140, b0=0.7360,
        e_lo=2.10, e_hi=10.24, dz=0.11e-6, t_cav=1.02e-6, e_cav=2.10,
        mf=None, ffo=21):
    """A FABRY-PEROT cavity: two identical corrugated mirrors around a thick
    uniform spacer, so the reflectance is set by a cavity resonance whose
    phase is sensitive to anything that shifts either mirror's effective
    index.  The second mirror's walls are the ones that open by ``delta``."""
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    st.add_layer(dz, segments=[(a0, e_lo), (b0 - a0, e_hi), (1.0 - b0, e_lo)])
    st.add_layer(t_cav, eps=e_cav)
    st.add_layer(dz, segments=[(a0 - delta, e_lo),
                               (b0 + delta - (a0 - delta), e_hi),
                               (1.0 - (b0 + delta), e_lo)])
    st.set_source(wl, theta=theta)
    return st


def vhcg(delta, *, wl=1.4508e-6, duty=0.6505, period=0.70e-6, t_gr=0.35e-6,
         e_hi=12.25, e_lo=1.0, nsup=1.0, nsub=1.0, theta=0.0, degree=12,
         nl=2, mf=None, ffo=9):
    """A HIGH-CONTRAST GRATING at a Fano resonance -- the steep counter-fixture
    the MOVE bar is attacked with.

    A single high-index bar layer in air supports leaky guided resonances whose
    far field swings between 0 and 1 over a fraction of a percent of duty
    cycle.  ``dR/d(duty)`` there is the DEVICE's own slope, and it is the
    quantity the move bar is really competing against: the prescribed snap
    changes the duty by about one widest manufactured cell, so a device whose
    slope exceeds the bar puts a CORRECT answer past it with no numerical
    pathology at all.  The resonance is LOCATED by the probe, not pinned here;
    the defaults are only a seed."""
    a = 0.5 - duty / 2.0
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(t_gr / nl, segments=[(a - dd, e_lo),
                                          (duty + 2.0 * dd, e_hi),
                                          (1.0 - a - duty - dd, e_lo)])
    st.set_source(wl, theta=theta)
    return st


def wood_theta(period, wl, n_sup, order, *, inside=1e-3):
    """The incidence angle at which diffracted ``order`` sits ``inside``
    (relative) of its RAYLEIGH cutoff in the superstrate.

    The cutoff is ``n_sup sin(theta) + order wl / period = -n_sup``; solving
    for ``sin(theta)`` and backing off by ``inside`` puts the order just
    propagating, where its ``kz`` is smallest and the far field is most
    sensitive to the discretisation."""
    s = (-1.0 * n_sup - order * wl / period) / n_sup
    s = s * (1.0 - inside)
    s = float(np.clip(s, -0.999999, 0.999999))
    return float(np.arcsin(s))


def vwood(delta, *, period=1.62e-6, wl=1.064e-6, n_sup=1.90, order=-2,
          inside=2e-3, degree=8, nsub=complex(2.60, 0.35), a0=0.2410,
          b0=0.7040, e_lo=2.31, e_hi=9.61, dz=0.17e-6, nl=2, mf=None,
          ffo=25):
    """A near-WOOD mount: ``theta`` is solved so one diffracted order sits
    just inside its Rayleigh cutoff."""
    th = wood_theta(period, wl, n_sup, order, inside=inside)
    return vstair(delta, period=period, wl=wl, theta=th, a0=a0, b0=b0,
                  e_lo=e_lo, e_hi=e_hi, dz=dz, nl=nl, degree=degree,
                  nsub=nsub, nsup=n_sup, mf=mf, ffo=ffo)


def _uniaxial(no, ne, tilt, azim):
    """``R diag(no^2, no^2, ne^2) R^T`` for a director tilted ``tilt`` from z
    and rotated ``azim`` about z."""
    ct, stt = np.cos(tilt), np.sin(tilt)
    ca, sa = np.cos(azim), np.sin(azim)
    d = np.array([stt * ca, stt * sa, ct], dtype=float)
    return (no ** 2) * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)


def vtensor(delta, *, period=1.15e-6, wl=0.905e-6, theta=0.36, degree=10,
            no=1.51, ne=1.74, tilt=np.pi / 2, azim=0.7, a0=0.2615, b0=0.6840,
            e_lo=2.10, dz=0.13e-6, nl=2, nsup=1.0, nsub=1.46, mf=None,
            ffo=21):
    """An IN-PLANE rotated uniaxial director as the ridge material
    (``tilt = pi/2`` puts the director in the xy plane)."""
    eps = _uniaxial(no, ne, tilt, azim)
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(dz, segments=[(a0 - dd, e_lo * np.eye(3)),
                                   (b0 + dd - (a0 - dd), eps),
                                   (1.0 - (b0 + dd), e_lo * np.eye(3))])
    st.set_source(wl, theta=theta)
    return st


def voop(delta, **kw):
    """The OUT-OF-PLANE director: the same device with the director tilted
    60 degrees from z, which is the class the round-2 pol-0 evidence turns
    on."""
    kw.setdefault("tilt", np.pi / 3)
    kw.setdefault("azim", 0.25)
    return vtensor(delta, **kw)


def vliner(delta, *, liner=1e-6, period=1.35e-6, wl=1.064e-6, theta=0.28,
           degree=10, a0=0.3120, b0=0.6790, e_lo=2.56, e_hi=12.25,
           e_liner=4.0, dz=0.12e-6, nsup=1.0, nsub=1.52, mf=None, ffo=21):
    """A stack whose FIRST layer OWNS a sliver-thin liner (both walls in the
    same layer), while the two layers' ridge walls differ by ``delta``.

    This is the D-1 / R3-B geometry: the owned liner lowers the GLOBAL
    own-scale ``_cross_layer_sliver`` measures against."""
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    st.add_layer(dz, segments=[(a0, e_lo), (liner, e_liner),
                               (b0 - a0 - liner, e_hi), (1.0 - b0, e_lo)])
    st.add_layer(dz, segments=[(a0 - delta, e_lo),
                               (b0 + delta - (a0 - delta), e_hi),
                               (1.0 - (b0 + delta), e_lo)])
    st.set_source(wl, theta=theta)
    return st


def vliner_free(delta, **kw):
    """:func:`vliner` with the OWNED liner removed and nothing else changed
    -- the control arm of the R3-B pair."""
    kw.pop("liner", None)
    p = dict(period=1.35e-6, wl=1.064e-6, theta=0.28, degree=10, a0=0.3120,
             b0=0.6790, e_lo=2.56, e_hi=12.25, dz=0.12e-6, nsup=1.0,
             nsub=1.52, mf=None, ffo=21)
    p.update(kw)
    st = PMMStack(p["period"], n_superstrate=p["nsup"], n_substrate=p["nsub"],
                  degree=p["degree"], far_field_orders=p["ffo"],
                  min_feature=(p["period"] * NO_SNAP if p["mf"] is None
                               else float(p["mf"])))
    for k in (0, 1):
        dd = delta * k
        st.add_layer(p["dz"], segments=[(p["a0"] - dd, p["e_lo"]),
                                        (p["b0"] + dd - (p["a0"] - dd),
                                         p["e_hi"]),
                                        (1.0 - (p["b0"] + dd), p["e_lo"])])
    st.set_source(p["wl"], theta=p["theta"])
    return st


def vplain(*, period=1.35e-6, wl=1.064e-6, theta=0.28, degree=10, nl=2,
           a0=0.3120, b0=0.6790, e_lo=2.56, e_hi=12.25, dz=0.12e-6,
           nsup=1.0, nsub=1.52, mf=None, ffo=21, spacer=None):
    """An ORDINARY stack with NO sliver: every layer shares the same wall
    set, optionally with a uniform spacer.  The screen must be silent here."""
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=ffo,
                  min_feature=(period * NO_SNAP if mf is None else float(mf)))
    for _k in range(nl):
        st.add_layer(dz, segments=[(a0, e_lo), (b0 - a0, e_hi),
                                   (1.0 - b0, e_lo)])
    if spacer is not None:
        st.add_layer(spacer, eps=e_lo)
    st.set_source(wl, theta=theta)
    return st


# ------------------------------------------------------------- box --------
#: The verifier's OWN configuration box for the closure sizing (task 2).
#: 3 periods x 2 wavelengths x 2 superstrates x 3 lossy substrates x 2 angles
#: x 2 degrees x 2 ridge permittivities x 2 slice counts = 576 mounts, each
#: at 4 wall steps = 2,304 rows.  Every axis value differs from the fix's box.
#: The substrate indices and the widest period are capped so that a degree-6
#: reference solve (3 elements, 18 global nodes) still resolves every
#: propagating order -- ``PMMStack.solve`` refuses below that, and a skipped
#: configuration is a hole in the population rather than a measurement.
BOX = dict(
    period=(0.68e-6, 1.02e-6, 1.35e-6),
    wl=(0.633e-6, 1.064e-6),
    nsup=(2.05, 3.10),
    nsub=(complex(1.52, 0.03), complex(2.90, 1.10), complex(3.45, 0.90)),
    theta=(1.18, 1.35),
    degree=(6, 8),
    e_hi=(6.76, 12.25),
    nl=(2, 3),
)
BOX_DELTAS = (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
BOX_WALLS = dict(a0=0.3120, b0=0.6790, e_lo=2.56, dz=0.12e-6)


def box_stack(delta, *, period, wl, nsup, nsub, theta, degree, e_hi, nl,
              mf=None):
    return vstair(delta, period=period, wl=wl, theta=theta, degree=degree,
                  nsub=nsub, nsup=nsup, e_hi=e_hi, nl=nl, mf=mf, ffo=31,
                  **BOX_WALLS)


def box_configs():
    """Every mount of :data:`BOX`, as a list of kwargs dicts."""
    keys = list(BOX)
    out = []
    for vals in itertools.product(*(BOX[k] for k in keys)):
        out.append(dict(zip(keys, vals)))
    return out


def jsonable(x):
    """NumPy / complex -> JSON."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if np.isfinite(f) else ("inf" if f > 0 else
                                         ("-inf" if f < 0 else "nan"))
    if isinstance(x, (np.complexfloating, complex)):
        return [float(np.real(x)), float(np.imag(x))]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def build_info():
    import platform
    return dict(python=sys.version.split()[0], numpy=np.__version__,
                platform=platform.platform(),
                lumenairy=assert_tree(),
                guard=bool(ps.PMM_SLIVER_GUARD),
                closure_fraction=float(
                    getattr(ps, "_SLIVER_CLOSURE_FRACTION", float("nan"))),
                attrib_closure=float(ps._SLIVER_ATTRIB_CLOSURE),
                move_factor=float(ps._SLIVER_MOVE_FACTOR),
                trigger=float(ps._SLIVER_TRIGGER_BAR))
