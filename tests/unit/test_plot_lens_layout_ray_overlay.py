"""``plot_lens_layout`` ray-overlay contract.

Closes the zero-coverage hole recorded as S5.4 of
``docs/audits/AUDIT_PLOT_LENS_LAYOUT_RAY_OVERLAY_2026_08_19.md``: through
5.39.1 no test anywhere in ``tests/`` referenced ``plot_lens_layout``,
which is how a silent-wrong renderer survived to a release.

THE DEFECT these tests pin.  ``make_fan`` builds its fan entirely in the
y-z plane -- every launched ray has ``x = 0`` and ``L = 0``, with both
the aperture spread and the field angle in ``y`` / ``M``.  The overlay
in ``plot_lens_layout`` read ``rb.x`` / ``ir.L``, so every ray was drawn
flat at ``h = 0``: not omitted, not raised, but rendered as a clean
horizontal line -- a coherent and entirely false picture of a system
whose rays all travel on the axis and arrive on axis.

Assertions are on ``Line2D`` artist data, never on pixels: an image
comparison would be both fragile across matplotlib versions and unable
to say *which* number was wrong.

Every bar below carries its derivation and its measured values, per
``docs/TESTING_STANDARDS.md``.
"""

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from lumenairy.analysis.plotting import (  # noqa: E402
    _layout_finite_conjugate_image_distance,
    plot_lens_layout,
)
from lumenairy.io.prescriptions_builders import (  # noqa: E402
    make_singlet,
    thorlabs_lens,
)
from lumenairy.raytrace import (  # noqa: E402
    find_paraxial_focus,
    make_fan,
    surfaces_from_prescription,
    system_abcd,
    trace,
)
from lumenairy.raytrace.seidel import _object_space_index  # noqa: E402
from lumenairy.raytrace.trace import _make_bundle  # noqa: E402

WL = 1.31e-6

# Two independent prescriptions, matching the audit's out-of-tree
# verification (which used designs 121 and 122 -- both multi-surface
# 1.31 um relays, neither in-tree).  A cemented catalog achromat and a
# bare biconvex singlet differ in surface count, glass count and focal
# length, so a fix that happened to work on one shape is not enough.
_FIXTURES = {
    'AC254-100-C achromat': lambda: thorlabs_lens('AC254-100-C'),
    'N-BK7 biconvex singlet': lambda: make_singlet(
        50e-3, -50e-3, 4e-3, 'N-BK7', aperture=25.4e-3,
        name='Biconvex singlet'),
}


def _ray_lines(ax, n_surfaces, has_object_leg=False):
    """The ray polylines among an axes' ``Line2D`` artists.

    ``plot_lens_layout`` draws exactly four kinds of line: surface sag
    curves (81 points, hard-coded), the optical axis and the image plane
    (2 points each), and one polyline per drawn ray.  A ray polyline has
    one point per surface plus the image plane, plus the object-space
    leg when a finite ``object_distance`` is honoured.  The counts are
    read off the running build rather than assumed, and the identifying
    length is asserted unambiguous.
    """
    n_pts = int(n_surfaces) + 1 + (1 if has_object_leg else 0)
    assert n_pts not in (2, 81), (
        f"artist identification is ambiguous on this fixture: a ray "
        f"polyline has {n_pts} points, colliding with the axis/image "
        f"lines (2) or a surface curve (81).  Use a fixture with a "
        f"different surface count.")
    return [ln for ln in ax.get_lines() if len(ln.get_xdata()) == n_pts]


def _by_field_angle(ray_lines):
    """Group ray polylines by field angle.

    ``plot_lens_layout`` colours one fan per field angle from viridis,
    so the RGBA tuple *is* the field-angle label.  Returned in draw
    order, which is ascending field angle.
    """
    groups = []
    for ln in ray_lines:
        col = ln.get_color()
        key = col if isinstance(col, str) else tuple(np.round(col, 9))
        for k, members in groups:
            if k == key:
                members.append(ln)
                break
        else:
            groups.append((key, [ln]))
    return [members for _, members in groups]


def _chief(members):
    """The chief ray of a fan: the one launched at zero pupil height."""
    return min(members, key=lambda ln: abs(float(ln.get_ydata()[0])))


# =========================================================================
# 1 -- rays are not degenerate  (audit section 6.1)
# =========================================================================

@pytest.mark.parametrize('label', sorted(_FIXTURES))
def test_layout_ray_overlay_is_not_degenerate(label):
    """Every field angle must contribute at least one ray with real
    transverse extent.

    FAIL-BEFORE, demonstrated rather than asserted from memory: the arm
    below re-traces the same fans through the same surfaces and shows
    that ``x`` -- the coordinate the pre-fix renderer plotted -- is
    *identically* zero at every surface and at the image, so the pre-fix
    polyline had ``ptp(ydata) == 0.0`` exactly and could not clear any
    positive bar.  That zero is algebraic, not a small residual: a
    meridional ray through rotationally symmetric surfaces never
    acquires an x component, so it is the same zero on every build.

    Measured pre-fix (2026-08-22, this build), rendering both fixtures
    at n_field_angles=3, max_field_deg=2.64, rays_per_fan=5:
    max ``ptp(ydata)`` over ALL 13 ray polylines was 0.000000e+00 on
    both.  Post-fix the same render gives 1.668368e-02 m (achromat) and
    1.694031e-02 m (singlet).

    BAR: ``0.1 * semi_diameter`` = 1.270000e-03 m for these fixtures.
    Gap below: the pre-fix value is exactly 0 (a decade is meaningless
    against zero -- the bar is unreachable, not merely far).  Gap above:
    the marginal ray spans 1.67e-2 m, 13x the bar.  Nothing here sits
    near a build-sensitive boundary; the quantity is a full-aperture ray
    height, not a reduction over a near-degenerate eigenproblem.
    """
    rx = _FIXTURES[label]()
    surfaces = surfaces_from_prescription(rx)
    semi = float(rx['aperture_diameter']) / 2.0

    # --- FAIL-BEFORE arm: the pre-fix coordinate carries no signal ----
    for deg in (0.0, 1.32, 2.64):
        fan = make_fan(semi_aperture=semi, n_rays=5,
                       field_angle=float(np.radians(deg)), wavelength=WL)
        assert np.ptp(fan.x) == 0.0 and np.ptp(fan.L) == 0.0
        r = trace(fan, surfaces, WL)
        max_x = max(float(np.max(np.abs(rb.x))) for rb in r.ray_history)
        max_L = float(np.max(np.abs(r.image_rays.L)))
        assert max_x == 0.0, (
            f"{label} @ {deg} deg: the fail-before premise no longer "
            f"holds -- traced rays have max|x| = {max_x:.6e}, so "
            f"plotting x would no longer be provably flat.  make_fan's "
            f"plane or the trace's symmetry has changed; re-derive this "
            f"test before trusting it.")
        assert max_L == 0.0, (
            f"{label} @ {deg} deg: image-plane max|L| = {max_L:.6e}, "
            f"expected exactly 0 for a meridional y-z fan.")

    # --- FIXED arm: the rendered artists carry the fan ----------------
    fig, ax = plt.subplots()
    try:
        plot_lens_layout(rx, ax=ax, wavelength=WL, show_rays=True,
                         n_field_angles=3, max_field_deg=2.64,
                         rays_per_fan=5)
        lines = _ray_lines(ax, len(surfaces))
        assert lines, f"{label}: no ray polylines were drawn at all."
        groups = _by_field_angle(lines)
        assert len(groups) == 3, (
            f"{label}: expected one colour group per field angle; got "
            f"{len(groups)}.")
        bar = 0.1 * semi
        for i, members in enumerate(groups):
            spans = [float(np.ptp(ln.get_ydata())) for ln in members]
            assert max(spans) > bar, (
                f"{label}, field-angle group {i}: no ray spans more "
                f"than 0.1*semi_diameter ({bar:.6e} m); the largest "
                f"ptp(ydata) over {len(members)} rays is "
                f"{max(spans):.6e} m.  The overlay is plotting a "
                f"coordinate the fan does not live in (the pre-5.39.2 "
                f"defect: rb.x instead of rb.y), or the fan collapsed.")
    finally:
        plt.close(fig)


# =========================================================================
# 2 -- field angles separate at the image plane  (audit section 6.2)
# =========================================================================

@pytest.mark.parametrize('label', sorted(_FIXTURES))
def test_layout_chief_ray_heights_follow_f_tan_theta(label):
    """Chief-ray image heights must be monotonic in field angle and
    equal the paraxial prediction ``f_obj * tan(theta)``.

    Catches what test 1 lets through: a sign flip, or a single
    non-degenerate fan drawn three times.  ``f_obj = n_obj * efl``
    because ``efl`` is the REDUCED focal length (see the EFL CONVENTION
    note in ``system_abcd``); both fixtures are air-to-air, so
    ``n_obj == 1``, but the index is threaded so an immersed fixture
    would not silently drift.

    BAR: relative agreement ``< 2e-2``.  Measured 2026-08-22 on this
    build: 1.9e-5 and 7.7e-5 (achromat, 1.32 / 2.64 deg), 2.8e-5 and
    1.1e-4 (singlet).  The residual is real distortion, not noise, and
    it grows with field -- so the bar is set two decades above the
    largest measured value rather than at a converged tolerance.  Gap
    above: the failures this is aimed at are gross -- a sign flip gives
    relative error 2.0, a collapse to the axis gives 1.0, both two
    decades the other side of the bar.
    """
    rx = _FIXTURES[label]()
    surfaces = surfaces_from_prescription(rx)
    _, efl, _, _ = system_abcd(surfaces, WL)
    f_obj = float(_object_space_index(surfaces, WL)) * float(efl)
    angles = (0.0, 1.32, 2.64)

    fig, ax = plt.subplots()
    try:
        plot_lens_layout(rx, ax=ax, wavelength=WL, show_rays=True,
                         n_field_angles=len(angles),
                         max_field_deg=angles[-1], rays_per_fan=5)
        groups = _by_field_angle(_ray_lines(ax, len(surfaces)))
        assert len(groups) == len(angles)
        heights = [float(_chief(g).get_ydata()[-1]) for g in groups]
    finally:
        plt.close(fig)

    diffs = np.diff(heights)
    assert np.all(diffs > 0), (
        f"{label}: chief-ray image heights {heights} are not strictly "
        f"increasing with field angle.  A flat sequence means the fans "
        f"are collapsed onto the axis; a decreasing one means the "
        f"transverse coordinate or the image-plane extrapolation "
        f"carries the wrong sign.")

    for deg, got in zip(angles[1:], heights[1:]):
        want = f_obj * np.tan(np.radians(deg))
        rel = abs(got - want) / abs(want)
        assert rel < 2e-2, (
            f"{label} @ {deg} deg: chief-ray image height {got:.9e} m "
            f"disagrees with f_obj*tan(theta) = {want:.9e} m by "
            f"{rel:.3e} relative (bar 2e-2).")


# =========================================================================
# 3 -- the fan-plane coupling is pinned  (audit section 6.3)
# =========================================================================

def test_make_fan_plane_is_y_z_as_plot_lens_layout_assumes():
    """``make_fan`` builds its fan in y-z; the layout overlay reads y.

    THIS TEST NAMES A COUPLING, not a property of ``make_fan`` alone.
    ``plot_lens_layout`` builds its own fan through ``make_fan`` and
    plots ``rb.y`` / ``ir.M`` on the height axis (see the FAN-PLANE
    COUPLING comment in ``analysis/plotting.py``).  If ``make_fan`` is
    ever re-planed -- to x-z, or to a caller-selectable plane -- this
    test fails FIRST and points at the renderer that must be updated
    alongside it, instead of letting the layout silently break in the
    other direction.

    ``make_fan``'s own per-axis field convention is pinned separately by
    ``test_niche_audit_w3_raytrace_sources.py``; this is the rendering
    consumer's stake in it.
    """
    fan = make_fan(semi_aperture=9.0e-3, n_rays=5,
                   field_angle=float(np.radians(2.64)), wavelength=WL)

    assert np.ptp(fan.x) == 0.0, (
        f"make_fan spread its fan in x (ptp = {np.ptp(fan.x):.6e}); "
        f"plot_lens_layout plots y and would now render a flat line.")
    assert np.ptp(fan.y) > 0.0, (
        f"make_fan produced no y extent (ptp = {np.ptp(fan.y):.6e}); "
        f"plot_lens_layout's height axis would carry no information.")
    assert float(np.max(np.abs(fan.L))) == 0.0, (
        "make_fan put field angle in L; plot_lens_layout extrapolates "
        "to the image plane with M.")
    assert float(np.max(np.abs(fan.M))) > 0.0, (
        "make_fan put no field angle in M.")


# =========================================================================
# 4 -- S5.3: object_distance is honoured
# =========================================================================

def _real_ray_image_distance(surfaces, wavelength, obj_d, y_pupil=1e-5):
    """Independent oracle for the finite-conjugate image distance.

    Real-ray, not ABCD: launch a near-axial ray from the on-axis object
    point, trace it, and read the axis crossing past the last surface.
    Shares no algebra with
    ``_layout_finite_conjugate_image_distance``'s Gauss-at-the-
    principal-planes derivation.
    """
    norm = float(np.hypot(y_pupil, obj_d))
    b = _make_bundle(x=[0.0], y=[0.0], L=[0.0], M=[y_pupil / norm],
                     wavelength=wavelength)
    b.z = np.array([-float(obj_d)])
    img = trace(b, surfaces, wavelength).image_rays
    assert bool(img.alive[0]), 'oracle ray did not survive the trace'
    return float(-img.y[0] / (img.M[0] / img.N[0]))


@pytest.mark.parametrize('obj_d', [0.3, 0.5, 2.0])
def test_layout_honours_finite_object_distance(obj_d):
    """A finite ``object_distance`` must move BOTH the ray launch and
    the image plane (audit S5.3).

    Pre-5.39.2 the function traced from infinity and placed the image
    plane at the BFL regardless, which is simply the wrong model for a
    finite conjugate.

    The image distance is checked against an independent REAL-RAY
    oracle, bounded by that oracle's own paraxial-limit error rather
    than by one build's residual: the oracle ray is launched at
    ``y_pupil = 1e-5 m``, so its departure from the paraxial limit is
    O(y^2/R^2) ~ 1e-8 relative for these ~50 mm radii.  BAR: 1e-6
    relative, two decades above that floor.  Measured 2026-08-22:
    2.3e-09 / 4.3e-09 / 7.6e-09 at obj_d = 2.0 / 0.5 / 0.3 m.

    Separation from the infinite-conjugate answer is asserted too, so
    the test cannot pass by ignoring ``object_distance``: the finite
    conjugate sits 3.7 mm (2.0 m) to 32.6 mm (0.3 m) beyond the
    80.03 mm BFL, against an 8.5 mm total track.
    """
    rx = dict(thorlabs_lens('AC254-100-C'), object_distance=obj_d)
    surfaces = surfaces_from_prescription(rx)

    image_z = _layout_finite_conjugate_image_distance(surfaces, WL, obj_d)
    oracle = _real_ray_image_distance(surfaces, WL, obj_d)
    rel = abs(image_z - oracle) / abs(oracle)
    assert rel < 1e-6, (
        f"finite-conjugate image distance {image_z:.9e} m disagrees "
        f"with the real-ray oracle {oracle:.9e} m by {rel:.3e} "
        f"relative (bar 1e-6).")

    bfl = float(find_paraxial_focus(surfaces, WL))
    track = float(sum(float(t) for t in rx['thicknesses']))
    assert abs(image_z - bfl) > 0.1 * track, (
        f"the finite conjugate ({image_z:.6e} m) is not separated from "
        f"the BFL ({bfl:.6e} m) by enough of the {track:.6e} m track "
        f"for this fixture to prove anything.")

    fig, ax = plt.subplots()
    try:
        plot_lens_layout(rx, ax=ax, wavelength=WL, show_rays=True,
                         n_field_angles=2, max_field_deg=1.0,
                         rays_per_fan=5)
        lines = _ray_lines(ax, len(surfaces), has_object_leg=True)
        assert lines, (
            "no ray polyline carries an object-space leg; the finite "
            "object_distance was ignored for the ray launch.")
        z_start = min(float(np.min(ln.get_xdata())) for ln in lines)
        assert z_start == pytest.approx(-obj_d, rel=1e-12), (
            f"rays launch at z = {z_start:.9e} m, not at the object "
            f"plane z = {-obj_d:.9e} m.")

        # The image plane must be the finite conjugate, not the BFL.
        verticals = [ln for ln in ax.get_lines()
                     if len(ln.get_xdata()) == 2
                     and float(ln.get_xdata()[0])
                     == float(ln.get_xdata()[1])]
        assert len(verticals) == 1, (
            f"expected exactly one image-plane vline; got "
            f"{len(verticals)}.")
        z_img = float(verticals[0].get_xdata()[0])
        assert z_img == pytest.approx(track + image_z, rel=1e-12), (
            f"image plane drawn at z = {z_img:.9e} m; the finite "
            f"conjugate is at {track + image_z:.9e} m (the BFL would "
            f"put it at {track + bfl:.9e} m).")
    finally:
        plt.close(fig)


def test_layout_infinite_conjugate_is_unchanged_without_object_distance():
    """The historical behaviour must survive: no ``object_distance``
    (or a zero / non-finite one) still means collimated fans launched at
    the first vertex and an image plane at the BFL.

    The second half of rule 4 in ``docs/TESTING_STANDARDS.md``: the S5.3
    behaviour change is a two-sided claim, and this is the arm that says
    it did not leak into every other caller.
    """
    base = thorlabs_lens('AC254-100-C')
    surfaces = surfaces_from_prescription(base)
    bfl = float(find_paraxial_focus(surfaces, WL))
    track = float(sum(float(t) for t in base['thicknesses']))

    for label, rx in (('absent', base),
                      ('zero', dict(base, object_distance=0.0)),
                      ('None', dict(base, object_distance=None)),
                      ('inf', dict(base, object_distance=float('inf'))),
                      ('negative', dict(base, object_distance=-0.5))):
        fig, ax = plt.subplots()
        try:
            plot_lens_layout(rx, ax=ax, wavelength=WL, show_rays=True,
                             n_field_angles=2, max_field_deg=1.0,
                             rays_per_fan=5)
            lines = _ray_lines(ax, len(surfaces))
            assert lines, f"object_distance={label}: no rays drawn."
            z_start = min(float(np.min(ln.get_xdata())) for ln in lines)
            assert z_start == pytest.approx(0.0, abs=1e-15), (
                f"object_distance={label}: rays launch at "
                f"z = {z_start:.6e}, not at the first vertex.")
            verticals = [ln for ln in ax.get_lines()
                         if len(ln.get_xdata()) == 2
                         and float(ln.get_xdata()[0])
                         == float(ln.get_xdata()[1])]
            assert len(verticals) == 1
            assert float(verticals[0].get_xdata()[0]) == pytest.approx(
                track + bfl, rel=1e-12), (
                f"object_distance={label}: image plane moved off the "
                f"BFL.")
        finally:
            plt.close(fig)


# =========================================================================
# 5 -- S5.3: the runaway-focus warning
# =========================================================================

def _galilean_expander(f1, f2, detune):
    """A Galilean beam expander, detuned off its afocal separation.

    Two plano singlets of thin-lens focal lengths ``f1`` (negative) and
    ``f2`` (positive) separated by ``(f1 + f2) * (1 + detune)``.  At
    ``detune = 0`` the pair is afocal; nearby, the system power is small
    but finite, so ``efl`` and the BFL both run away together while
    their RATIO stays pinned at the expander's magnification
    ``|f2 / f1|`` -- which is exactly the S5.3 pathology and exactly
    what the guard measures.
    """
    n = 1.5010                      # N-BK7 near 1.31 um; only sets the
    R1 = f1 * (n - 1.0)             # nominal powers, the ladder below
    R2 = -f2 * (n - 1.0)            # absorbs the inaccuracy.
    return {
        'name': 'Galilean expander', 'aperture_diameter': 25.4e-3,
        'surfaces': (make_singlet(R1, float('inf'), 2e-3,
                                  'N-BK7')['surfaces']
                     + make_singlet(float('inf'), R2, 2e-3,
                                    'N-BK7')['surfaces']),
        'thicknesses': [2e-3, (f1 + f2) * (1.0 + detune), 2e-3],
    }


def _runaway_ratio(rx):
    """The quantity the guard tests, measured from the running build."""
    surfaces = surfaces_from_prescription(rx)
    image_z = float(find_paraxial_focus(surfaces, WL))
    _, efl, _, _ = system_abcd(surfaces, WL)
    track = float(sum(float(t) for t in rx['thicknesses']))
    ap = float(rx['aperture_diameter'])
    scale = max(abs(track), ap, abs(efl) if np.isfinite(efl) else 0.0)
    return abs(image_z) / scale if scale > 0 else float('inf')


def test_layout_warns_when_the_paraxial_focus_runs_away():
    """A near-afocal system fed collimated must WARN, not silently
    autoscale the panel to a runaway image plane (audit S5.3).

    The state is ENGINEERED through the public builders and its ratio is
    MEASURED on the running build, not hoped for: a ladder of detunes is
    scanned and the first fixture that actually clears the bar is used.
    Hard-fail only if the whole ladder is exhausted.

    BAR (``_LAYOUT_FOCUS_RUNAWAY_RATIO`` = 10x): see the measured
    envelope recorded above ``plot_lens_layout`` in
    ``analysis/plotting.py`` -- every focusing prescription in the tree
    sits at 0.868..0.9998 on this ratio, so the bar is a full decade
    above the sane envelope.  Measured for this fixture on 2026-08-22:
    20.94 / 20.56 / 20.42 / 20.38 at detunes 0.03 / 0.01 / 3e-3 / 1e-3,
    asymptoting to the 20x expander magnification -- a design constant,
    not a knife-edge, so it is a full decade above the bar on both
    sides.
    """
    from lumenairy.analysis.plotting import _LAYOUT_FOCUS_RUNAWAY_RATIO

    ladder = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    measured = []
    chosen = None
    for detune in ladder:
        rx = _galilean_expander(-0.010, 0.200, detune)
        try:
            ratio = _runaway_ratio(rx)
        except (ValueError, RuntimeError, ZeroDivisionError):
            continue
        measured.append((detune, ratio))
        if ratio > _LAYOUT_FOCUS_RUNAWAY_RATIO:
            chosen = rx
            break
    assert chosen is not None, (
        f"could not engineer a runaway-focus system on this build; "
        f"ratios measured across the detune ladder: "
        f"{[(d, round(r, 4)) for d, r in measured]} against bar "
        f"{_LAYOUT_FOCUS_RUNAWAY_RATIO}.")

    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plot_lens_layout(chosen, ax=ax, wavelength=WL,
                             show_rays=True, n_field_angles=1)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
    finally:
        plt.close(fig)

    assert any('near-afocal' in m for m in msgs), (
        f"no runaway-focus UserWarning was raised for a system whose "
        f"image plane sits {measured[-1][1]:.3g}x its own scale away.  "
        f"Warnings seen: {msgs}")


def test_layout_does_not_warn_on_ordinary_prescriptions():
    """The other side of the gate: normal lenses must stay SILENT.

    Rule 4 of ``docs/TESTING_STANDARDS.md``.  This arm is what stops the
    S5.3 guard being tightened into a warning that fires on every
    layout -- and it is the arm that rejected the audit's own suggested
    formulation.  S5.3 proposed warning at ``10x the TRACK LENGTH``;
    remeasured on this build, ``|image_z| / track`` is 74.79 for a stock
    LA1301-C and 24.28 for an LA1050-C, so that bar would fire on
    ordinary catalog singlets purely because they are thin.  Every
    prescription below is a real, perfectly drawable lens.
    """
    cases = {
        'LA1050-C': thorlabs_lens('LA1050-C'),
        'LA1509-C': thorlabs_lens('LA1509-C'),
        'LA1301-C': thorlabs_lens('LA1301-C'),
        'AC254-050-C': thorlabs_lens('AC254-050-C'),
        'AC254-100-C': thorlabs_lens('AC254-100-C'),
        'AC254-200-C': thorlabs_lens('AC254-200-C'),
        'f/1 biconvex': make_singlet(25e-3, -25e-3, 8e-3, 'N-BK7',
                                     aperture=25.4e-3),
        'f/100 plano-convex': make_singlet(5.0, float('inf'), 3e-3,
                                           'N-BK7', aperture=25.4e-3),
    }
    for label, rx in cases.items():
        fig, ax = plt.subplots()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                plot_lens_layout(rx, ax=ax, wavelength=WL,
                                 show_rays=True, n_field_angles=3,
                                 max_field_deg=2.0)
            runaway = [str(w.message) for w in caught
                       if 'near-afocal' in str(w.message)]
        finally:
            plt.close(fig)
        assert not runaway, (
            f"{label} is an ordinary focusing lens (ratio "
            f"{_runaway_ratio(rx):.4f}) but tripped the runaway-focus "
            f"guard: {runaway}")
