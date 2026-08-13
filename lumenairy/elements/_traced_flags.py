"""Registry, context manager and era presets for the traced-lens layer flags.

NICHE C14 (2026-08-03).  This is step 2 of
``docs/audits/ARCH_TRACED_ENCAPSULATION_2026_08_03.md`` S8, and it is
deliberately ADDITIVE: it introduces no new behaviour, changes no default,
renames nothing, and requires no existing test, probe or validation runner to
move.  Every flag below keeps its name, its module and its semantics; this
module only gives them ONE TABLE, ONE audited save/restore, and a name for the
configurations the campaign actually compares.

WHY NOT ONE ORDINAL ``TRACED_MODEL_ERA`` SWITCH.  The assessment (S5.2)
measured the alternative and rejected it, and the reason is worth restating
here because it is what shapes this API: **the flags are a LATTICE, not a
timeline.**  The single most-cited comparison in the whole campaign --
``REMAP_STATIONARY_PHASE_LAUNCH = True`` with ``REMAP_INVERSE_SUPPORT_BOUND =
False``, which is how niche C8's entire case is made -- exists at NO point in
history: before C8 landed it was HEAD, and after it landed it is not any era.
An ordinal switch cannot express it.  So :func:`traced_era` applies a preset
AND accepts per-flag overrides, and every one of the 2^N corners stays
reachable::

    with traced_era('v5.32', REMAP_INVERSE_SUPPORT_BOUND=False):
        ...                       # the C8 fail-before arm, named

THE DEFECT THIS CLOSES, measured three times by the campaign's own
instruments (S8 step 2): **an intervention expressed relative to a default,
evaluated after the default moved.**

  1. ``fc_production_taper.py`` ran a nine-minute "BASELINE (taper as
     shipped)" row that was in fact the EXACT conversion, because the library
     default had flipped underneath it -- its baseline read 89.235 where
     v5.32.0 reads 87.834.  *"An intervention expressed as 'leave the library
     alone' is not an intervention once the library moves."*
  2. ``approx_common.py`` silently defaulted ``LUMEN_PIN`` to a frozen v5.31
     export that still existed on the machine; caught only by an
     ``AttributeError`` on a constant that does not exist in v5.31.
  3. ``wfe_probe_orders.py`` cached results keyed on the CONFIGURATION and not
     on the LIBRARY, and was hit by it twice in one session.

:func:`traced_flag_state` is the direct answer to all three: it is what a
runner should print in its provenance banner beside the file hash, and what a
result cache should include in its key.  A run that records its flag state
cannot silently become a run of something else.

RELATIONSHIP TO THE TEST-SIDE LEAK GUARD.  ``tests/conftest.py``'s niche-C11
guard DISCOVERS flags (every upper-case module-level scalar in three modules,
~91 of them) so that it cannot go stale.  This registry is the opposite and
complementary thing: a CURATED table of the flags that carry a documented
fail-before contract, with the values that contract names.  Discovery answers
"what could leak"; the registry answers "what does this flag mean, and what
does turning it off restore".  Neither replaces the other, and the liveness
test in ``tests/unit/test_niche_c14_encapsulation.py`` checks the registry
against the live modules so a renamed or deleted flag fails CI here.
"""
from __future__ import annotations

import contextlib
import importlib
import threading

_LT = 'lumenairy.elements._lens_traced'
_CM = 'lumenairy.propagators.carrier'
_IM = 'lumenairy.elements._lens_imap'

#: Era names, oldest first.  ``v5.31`` is the library BEFORE the D1-D7 / C1-C8
#: campaign, ``v5.32`` is v5.32.0 as released, ``v5.32.1`` is the current tree
#: (C9, C10, C11 and C13 on top).  A flag missing from an era carries its
#: nearest OLDER value; a flag whose era is not expressible as a value (see
#: ``fail_before`` being :data:`NOT_A_VALUE`) is simply absent from the table.
#:
#: ``v5.32.1`` IS A SOURCE-ONLY ERA at the time of writing: ``pyproject.toml``
#: and ``lumenairy/__init__.py`` both still say ``5.32.0``, ``CHANGELOG.md``
#: has no ``5.32.1`` header (C9-C12 sit under ``[Unreleased]``) and C13 is not
#: in the CHANGELOG at all.  The name is used here because the source
#: docstrings already use it; it is not a released version.
#:
#: NOT EVERY LAYER'S ERA IS IN RANGE.  F3, R7/F2 and P2 shipped at 5.28.0 and
#: 5.29.0 -- BEFORE ``v5.31`` -- so at every era in this table they are already
#: at their shipped values.  Their ``fail_before`` is still recorded (it is
#: what the regression tests force) but it names a pre-5.28 library, not an
#: era below.
#:
#: ``v5.34`` is the tree AFTER the inverse-characteristic per-pixel
#: evaluator landed.  It is the first era to carry a flag from a module
#: other than the two above, and the reason it exists at all is that
#: ``TRACED_INVERSE_MAP`` is NOT inert at the older eras -- a flag that
#: changes a returned bit has to be given a value in every era, or an
#: era arm quietly runs the NEW code and reports it as the old library.
#: (Contrast ``_REMAP_RESID_EIKONAL_DEGREE``, which IS inert at ``v5.31``
#: because C6 is off there, and is therefore correctly absent.)
ERAS = ('v5.31', 'v5.32', 'v5.32.1', 'v5.34')

#: Sentinel: this flag's fail-before is NOT a value of this flag (it is a
#: kwarg, or the absence of a carrier), so the era table cannot express it and
#: the layer map's prose must be read instead.
NOT_A_VALUE = object()


class _Flag:
    """One registered switch: where it lives, what layer it belongs to, what
    its documented fail-before is, and which eras give it which value."""

    __slots__ = ('module', 'name', 'layer', 'fail_before', 'eras', 'inert',
                 'note')

    def __init__(self, module, name, layer, fail_before, eras=(), inert=False,
                 note=''):
        self.module = module
        self.name = name
        self.layer = layer
        self.fail_before = fail_before
        self.eras = dict(eras)
        self.inert = bool(inert)
        self.note = note

    @property
    def key(self):
        return (self.module, self.name)

    def __repr__(self):                                   # pragma: no cover
        return f'<_Flag {self.layer} {self.module}.{self.name}>'


def _f(*a, **k):
    return _Flag(*a, **k)


#: THE TABLE.  Ordered by the dependency graph of
#: ``docs/audits/TRACED_LAYER_MAP.md`` S2, not alphabetically, so reading it
#: top to bottom walks the pipeline.  ``fail_before`` is quoted from each
#: constant's own note; where a note says the fail-before is a kwarg rather
#: than a value, ``fail_before`` is :data:`NOT_A_VALUE` and ``note`` says what
#: to do instead.
_TRACED_ERA_FLAGS = (
    # ---- carrier resolution ------------------------------------------------
    _f(_LT, '_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER', 'F3', False,
       {'v5.31': True, 'v5.32': True},
       note='tilt-aware guard on an explicitly-passed carrier.  Shipped at '
            '5.28.0, i.e. BEFORE the oldest era in this table -- "Exposed as '
            'a module flag so the regression test can force the pre-fix '
            'per-pixel-tilt launch (fail-before) via monkeypatch"'),
    _f(_LT, '_CARRIER_FIT_RADIUS_FRAC', 'R7/F2', NOT_A_VALUE, (),
       note="the gate is ``carrier is not None``, not a value of this "
            "constant; with no carrier the restriction is byte-identically "
            "absent"),
    _f(_LT, '_CARRIER_FIT_MIN_SAMPLES', 'R7/F2', NOT_A_VALUE, (),
       note='sample floor below which the R7/P2 restriction is abandoned'),
    # ---- the entrance eikonal (UNIT A) -------------------------------------
    _f(_LT, 'TILTED_CARRIER_EXACT_EIKONAL', 'C5', False,
       {'v5.31': False, 'v5.32': True},
       note='exact tilted reference wavefront; read at CALL time so the '
            'element and the chain can never be configured apart'),
    _f(_LT, 'REMAP_STATIONARY_PHASE_LAUNCH', 'C6', False,
       {'v5.31': False, 'v5.32': True},
       note='stationary-phase launch: augments the launch congruence by '
            'grad(a_fit) of the fitted residual eikonal'),
    _f(_LT, '_REMAP_RESID_EIKONAL_DEGREE', 'C10', 4,
       {'v5.32': 4, 'v5.32.1': 6},
       note='"setting it back to 4 restores the v5.32.0 / niche-C9 behaviour '
            'exactly, since it is the only thing that changes"'),
    _f(_LT, '_REMAP_RESID_DEGREE_CAP', 'C10', NOT_A_VALUE, (),
       note='VESTIGIAL since C10 raised the default to 6: the cap is now '
            'EQUAL to the default it exists to bound, and its justification '
            'text was written when the default was 4'),
    _f(_LT, '_REMAP_RESID_FREEZE_MARGIN', 'C6', NOT_A_VALUE, (),
       note="the residual model's radial freeze circle must sit strictly "
            "OUTSIDE the ray-fit disc; coincident, the polynomial and spline "
            "backends stop describing the same map (5.608 um vs 0.006 um)"),
    # ---- the ray-fit domain policy (UNIT B) --------------------------------
    _f(_LT, '_FIT_RADIUS_BEAM_FACTOR_DEFAULT', 'P2', NOT_A_VALUE, (),
       note='fail-before is the kwarg ``fit_radius_beam_factor=None``'),
    _f(_LT, '_APERTURE_BEAM_WARN_RATIO', 'P2', NOT_A_VALUE, (),
       note='aperture:beam ratio above which the cliff warning fires'),
    _f(_LT, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 'D1', 0.0,
       {'v5.31': 0.0, 'v5.32': 1e-8},
       note='"Set to 0.0 to restore the historical hard NaN mask exactly -- '
            'that is the fail-before switch the D1 tests use, not a supported '
            'configuration"'),
    _f(_LT, '_DECENTRED_FIT_POLY_ORDER', 'D7', NOT_A_VALUE, (),
       note='fail-before is the kwarg ``decentred_fit_poly_order = '
            'newton_poly_order``.  OPEN ANOMALY: orders 6, 8 and 12 all close '
            'the (-1,0) chain residual and the shipped 10 does not'),
    _f(_LT, '_DECENTRE_GATE_PIXELS', 'C1', 0.0,
       {'v5.31': 0.0, 'v5.32': 0.5},
       note='null-decentre gate, stage 1 (absolute pixels)'),
    _f(_LT, '_DECENTRE_GATE_W_FRAC', 'C1', 0.0,
       {'v5.31': 0.0, 'v5.32': 0.05},
       note='null-decentre gate, stage 2 (fraction of beam radius).  OPEN: '
            'sits 10-14x BELOW its own measured 0.48-0.72 w crossover'),
    _f(_LT, 'DECENTRED_FIT_ARBITER', 'C11', False,
       {'v5.32': False, 'v5.32.1': True},
       note='"With BOTH False the whole C11/C12 layer is BYTE-IDENTICAL to '
            'the pure _DECENTRE_GATE_W_FRAC selector ... That remains the '
            'era-pinned fail-before."  THE SHIPPED SELECTOR since 5.32.1'),
    _f(_LT, 'DECENTRED_FIT_PREDICTOR', 'C12', False,
       {'v5.32': False, 'v5.32.1': False},
       note='opt-in; C12 shipped it OFF and 5.32.1 keeps it OFF -- "inert '
            'where it was wanted and harmful where it is live"'),
    _f(_LT, '_DECENTRED_FIT_SCORE_FLOOR', 'C12', 0.0, (), inert=True,
       note='"SHIPPED 0.0 -- INERT, and the reason is a measurement, not '
            'caution": swept, the 26 picks are the shipped 26.  Pinned by '
            'test_niche_c12::test_the_score_weight_floor_ships_inert'),
    _f(_LT, '_DECENTRED_FIT_SPECTRUM_ORDER', 'C12', 0, (),
       note='order of the predictor spectral tail; 0 disables the spectral '
            'half.  Only reachable when DECENTRED_FIT_PREDICTOR is True, '
            'which it is not in any era of this table'),
    _f(_LT, 'REMAP_STATIONARY_PHASE_FIT_GUARD', 'C6-guard', False,
       {'v5.31': False, 'v5.32': False, 'v5.32.1': False},
       note='OPT-IN, never shipped True.  Superseded by C8 on every case '
            'measured, but KEPT: "the two act on different objects" -- a '
            'defect depositing energy INSIDE the traced support is invisible '
            'to a support bound by construction'),
    # ---- the least-squares solver ------------------------------------------
    _f(_LT, 'LSTSQ_CONDITIONING_STEPDOWN', 'C13', False,
       {'v5.32': False, 'v5.32.1': True},
       note='"False restores the pre-C13 solver EXACTLY -- '
            '_solve_lstsq_thread_safe is the fail-before for everything '
            'below"'),
    _f(_LT, '_LSTSQ_GRAM_RCOND_MIN', 'C13', NOT_A_VALUE, (),
       note='Gram rcond below which the C13 step-down engages'),
    _f(_LT, '_LSTSQ_RESID_MARGIN', 'C13', NOT_A_VALUE, (),
       note='relative margin a stepped-down route must BEAT to be taken; '
            'ties go to the shipped path'),
    # ---- the traced exit support (UNIT C) ----------------------------------
    _f(_LT, 'REMAP_INVERSE_SUPPORT_BOUND', 'C8', False,
       {'v5.31': False, 'v5.32': True},
       note='"REMAP_INVERSE_SUPPORT_BOUND = False restores the pre-C8 '
            'library bit for bit"'),
    _f(_LT, '_SUPPORT_BOUND_FEATHER_CELLS', 'C8', NOT_A_VALUE, (),
       note='feather width in exit-lattice cells; 0.0 gives a hard cut, not '
            'the pre-C8 library'),
    _f(_LT, 'RAY_DENSITY_HALO_CHECK', 'C7', 'silent',
       {'v5.31': 'silent', 'v5.32': 'warn'},
       note="'silent' suppresses the halo report entirely; the check is "
            'REPORTING-ONLY and never changes a returned bit'),
    _f(_LT, '_RD_HALO_AMP_CONTOUR', 'C7', NOT_A_VALUE, (),
       note='e^-N amplitude contour defining the reporting hull.  OPEN: '
            'understates the true support (1.6161 mm vs 1.8115 mm over all '
            'alive rays)'),
    _f(_LT, '_RD_HALO_RADIUS_FACTOR', 'C7', NOT_A_VALUE, (),
       note='x r_hull, the reporting radius.  Calibrated over 180 element '
            'calls with a measured 123x separation between the clean and '
            'defective populations at 1.25'),
    _f(_LT, '_RD_HALO_AMAX_TOL', 'C7', NOT_A_VALUE, (),
       note='amplitude tolerance beyond the reporting radius'),
    _f(_LT, 'SUPPORT_BAND_CHECK', 'C14', 'silent',
       {'v5.31': 'silent', 'v5.32': 'silent', 'v5.32.1': 'warn'},
       note="'silent' restores the pre-C14 reporting exactly.  Watches the "
            'band C8 RETAINS outside the traced hull -- the plateau plus the '
            'feather -- which C7 cannot see because its reporting radius lies '
            'in territory the bound has already zeroed.  Like C7 it is '
            'REPORTING-ONLY and cannot change a returned bit'),
    _f(_LT, '_SUPPORT_BAND_PEAK_RATIO_TOL', 'C14', NOT_A_VALUE, (),
       note='ratio of the in-support peak above which the retained band is '
            'reported.  1.0 = "the global maximum may not sit outside the '
            'traced support", which needs no calibration because no correct '
            'field can answer yes'),
    # ---- the inverse characteristic (UNIT D) -------------------------------
    _f(_IM, 'TRACED_INVERSE_MAP', 'IMAP', False,
       {'v5.31': False, 'v5.32': False, 'v5.32.1': False, 'v5.34': True},
       note='SHIPS ON at its own era, and scoped by carrier.py to the '
            'terminal fine retrace.  "Setting it False restores the shipped '
            'coarse-Newton + map_coordinates upsample chain BIT FOR '
            'BIT."  Valued at EVERY era, not just its own: it changes a '
            'returned bit, so leaving it alone in an older preset would '
            'run the new evaluator and label the result v5.31.  Its own '
            'note carries the exact-trace oracle that decided the accuracy '
            'case and the off-lattice G8 probe that unblocked the default'),
    _f(_IM, 'INVERSE_MAP_GUARD', 'IMAP', 'silent',
       {'v5.34': 'warn'}, note='REPORTING-ONLY: it selects what a REFUSED '
            'build says, and a refused build keeps the shipped path '
            'whatever this holds, so it cannot change a returned bit.  '
            'Absent from the older eras because TRACED_INVERSE_MAP is '
            'False there, which makes it inert'),
    # ---- chain side --------------------------------------------------------
    _f(_CM, 'SPHERE_PARAB_CONVERSION_EXACT', 'C9', False,
       {'v5.32': False, 'v5.32.1': True},
       note='removal of a cos^2 band-limit taper on an otherwise-exact '
            'conversion; needs the git-archive byte-identity device, not the '
            'shadow-module one, because the chain resolves carrier.py under '
            'one name from several call sites'),
)

#: ``(module, name) -> _Flag``.
FLAGS = {f.key: f for f in _TRACED_ERA_FLAGS}

_BY_NAME: dict = {}
for _flag in _TRACED_ERA_FLAGS:
    _BY_NAME.setdefault(_flag.name, []).append(_flag)
del _flag

#: Re-entrancy lock.  The flags are process-global module attributes, so
#: concurrent :func:`traced_flags` blocks in different THREADS cannot both be
#: honoured; this serialises them rather than interleaving two save/restore
#: pairs into a lost update.  It does NOT make the flags thread-safe for
#: readers -- nothing can, short of moving them off the module -- and CI runs
#: serially (pytest-split shards across runners, not in-process xdist).
_LOCK = threading.RLock()


def _resolve(name):
    """One flag from a bare identifier, raising on unknown or ambiguous.

    Raising on an UNKNOWN name is the point: a typo'd override that silently
    did nothing would be exactly the class of failure this module exists to
    close (an intervention that is not an intervention)."""
    hits = _BY_NAME.get(name)
    if not hits:
        raise KeyError(
            f'{name!r} is not a registered traced-lens flag.  Registered: '
            + ', '.join(sorted(_BY_NAME)))
    if len(hits) > 1:                                     # pragma: no cover
        raise KeyError(f'{name!r} is ambiguous across modules: '
                       + ', '.join(h.module for h in hits))
    return hits[0]


def traced_flag_state(module=None):
    """Current value of every registered flag, as ``{'MODULE.NAME': value}``.

    Print this in a runner's provenance banner beside the library file hash,
    and include it in any result cache key.  Both are answers to measured
    failures -- see this module's header."""
    out = {}
    for f in _TRACED_ERA_FLAGS:
        if module is not None and f.module != module:
            continue
        mod = importlib.import_module(f.module)
        out[f'{f.module}.{f.name}'] = getattr(mod, f.name)
    return out


def resolve_era(era):
    """``{(module, name): value}`` for one era name.

    Resolution, exactly:

    * a flag with an entry at ``era`` takes it;
    * a flag whose newest entry is OLDER than ``era`` carries that entry
      forward (values persist until something changes them);
    * a flag whose OLDEST entry is NEWER than ``era`` is **absent from the
      result** -- the constant did not exist yet, or existed but could not be
      reached, and the preset therefore leaves it alone rather than inventing a
      value for it.  ``_REMAP_RESID_EIKONAL_DEGREE`` at ``v5.31`` is the
      example: C6 is off in that era, so the residual fit never runs and the
      degree is inert whatever it holds;
    * a flag with no era table at all (``fail_before is NOT_A_VALUE``) is
      always absent, because its era is not expressible as a value of itself.

    So a preset is a set of ASSIGNMENTS, not a complete state.  Combine it with
    ``overrides`` (see :func:`traced_era`) whenever an arm depends on a flag
    the table does not place in that era."""
    if era not in ERAS:
        raise ValueError(f'unknown era {era!r}; known: {ERAS}')
    upto = ERAS[:ERAS.index(era) + 1]
    out = {}
    for f in _TRACED_ERA_FLAGS:
        for name in upto:                    # oldest -> newest, last wins
            if name in f.eras:
                out[f.key] = f.eras[name]
    return out


@contextlib.contextmanager
def traced_flags(**overrides):
    """Set registered flags by bare name for the duration of the block, then
    restore EVERY one of them -- including on an exception raised while the
    values are being applied.

    ``with traced_flags(REMAP_INVERSE_SUPPORT_BOUND=False): ...``

    The restore is atomic in the sense that matters: the saved values are
    captured before any is written and are written back in a ``finally``, so a
    failure part-way through leaves no flag dirty.  This is the discipline all
    37 in-tree test sites already follow by hand; the point of having it here
    is that it becomes the path of least resistance instead of a rule each
    author must remember."""
    targets = [(_resolve(n), v) for n, v in overrides.items()]
    with _LOCK:
        saved = []
        try:
            for f, val in targets:
                mod = importlib.import_module(f.module)
                saved.append((mod, f.name, getattr(mod, f.name)))
                setattr(mod, f.name, val)
            yield
        finally:
            for mod, name, old in reversed(saved):
                setattr(mod, name, old)


@contextlib.contextmanager
def traced_era(era, **overrides):
    """Apply an era preset, then ``overrides`` on top, then restore.

    The overrides are what keep the LATTICE reachable -- an era alone can only
    name the ~3 points on the historical sequence, and the campaign's evidence
    lives on the product space::

        with traced_era('v5.32.1', REMAP_INVERSE_SUPPORT_BOUND=False):
            ...        # C6 on, C8 off: the row C8's whole case rests on
    """
    preset = resolve_era(era)
    with _LOCK:
        saved = []
        try:
            for (module, name), val in preset.items():
                mod = importlib.import_module(module)
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, val)
            with traced_flags(**overrides):
                yield
        finally:
            for mod, name, old in reversed(saved):
                setattr(mod, name, old)
