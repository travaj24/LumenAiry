# THE RECORD-RUN GRID CHOICE, MADE LOUD (docs/audits/FIX_PERF_PARALLEL_2026_08_10.md)
#
# WHY THIS EXISTS.  ADJUDICATION_NFC_8192_2026_08_10.md sec 2.1 found that on
# a 137 GB box the shipped ``n_fine_cap = 16384`` -- the setting of record for
# design 121 -- ALREADY RAN AT 8192.  ``carrier._memory_bounded_n_fine``
# degraded it, said so in a ``RuntimeWarning`` labelled "RESOLUTION-LIMITED
# (non-converged)", and the run then printed a full acceptance banner that
# passed.  Nothing in the runner compared the grid it asked for against the
# grid it got, so a 4.7-hour reference run and a 1.3-hour exploration run were
# the same run with different labels.  A naive dual-arm comparison would have
# reported "8192 and 16384 agree exactly".
#
# The remedy is not a louder warning.  It is that the runner must state its
# INTENT (which grid, on which budget), PROVE before the chain starts that the
# clamp cannot bind, and FAIL if it would -- because a warning is a thing a
# production log swallows and an exit code is not.
#
# THREE THINGS ARE CHECKED, in this order, all before any chain runs:
#
#   1. CLAMP.  ``carrier._fine_grid_ceiling(budget)`` is the largest fine grid
#      the RAM clamp will approve, as a pure function of the budget.  The leg
#      asks for ``min(n_fine_req, n_fine_cap)``, so ``ceiling >= n_fine_cap``
#      PROVES the clamp cannot bind, for any request, without running it.
#   2. BOX.  With ``ram_budget=inf`` the clamp is switched off, so something
#      else has to refuse an impossible run.  ``k`` congruence workers each
#      hold ``carrier._fine_grid_peak_bytes(...)``; the parent additionally
#      holds the common-grid accumulator.  That total is checked against the
#      free memory the box actually reports, with a stated reserve.
#   3. AFTER THE RUN.  Every warning raised during the chain is recorded (and
#      still shown), then scanned for the clamp's own vocabulary.  This is the
#      only check that survives ``congruence_workers > 1``: the clamp then
#      runs INSIDE the workers, where no parent-side wrapper can see it, and
#      the multi entry point re-emits the workers' warnings in the parent.
#
# THE TWO CHECKS FAIL DIFFERENTLY, AND THAT IS THE WHOLE DESIGN
# (docs/audits/FIX_CLAMP_RECAL_OVERRIDE_2026_08_10.md sec 3).
#
#   * Check 1 is about a WRONG ANSWER.  If the clamp binds, the leg silently
#     returns a smaller grid under the label the runner configured -- the
#     silent-8192 failure this module is named for.  There is no disposition
#     for that and there never will be: it stays a hard ``SystemExit(2)``,
#     whatever ``on_box_budget`` says.  The run gets the grid it named or it
#     refuses; it never quietly shrinks.
#   * Check 2 is about CAPACITY, and capacity is the operator's call to make.
#     The model is a deliberate UPPER-BOUND ENVELOPE over measured peaks
#     (``carrier._FINE_GRID_WORK_ARRAYS`` / ``_FINE_GRID_BASE_BYTES``; worst
#     ratio ~1.3x), so "modelled need > free" does NOT mean "will not fit" --
#     it means "the envelope says it might not".  Refusing an explicit intent
#     on an envelope is the model over-ruling a human who has already read
#     what the run costs.  So on the EXPLICIT-intent path (which is what
#     calling :func:`preflight` IS -- the runner states its grid and its
#     budget) a binding box check emits a prominent ``RuntimeWarning`` naming
#     modelled need, free memory and the MEASURED reference peak, and
#     PROCEEDS at the requested grid.  ``on_box_budget='error'`` (``ONBOX=
#     error``) restores the refusal for an unattended batch.
#
#     The refusal SURVIVES for the one case that is not a judgement call: a
#     run the box physically cannot allocate, i.e. modelled need >= TOTAL
#     physical memory rather than merely more than what is free right now.
#     Freeing memory cannot rescue that one, so no warning is going to help.
#
# THE WORKER BUDGET IS DIVIDED, NOT COPIED.  ``_multi_capture_worker_state``
# hands each congruence worker ``get_ram_budget() // k`` (carrier.py, and it is
# right to: K workers each sizing themselves for the whole box is K-fold
# over-subscription).  So the effective budget on the leg is k times SMALLER
# than the box's, and the same ``n_fine_cap`` that is honoured at k=1 can be
# silently degraded at k=4.  That is why ``workers`` is an argument here and
# why the record runs pass an explicit ``ram_budget``.
#
# Import-safe by construction: no module-level work beyond the imports, so a
# spawn child that re-imports the runner pays nothing for it.
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lumenairy.memory import get_ram_budget  # noqa: E402
from lumenairy.propagators import carrier as _C  # noqa: E402

#: Free memory left unclaimed by the BOX check.  Defaults to the same 8 GB
#: reserve ``_multi_resolve_workers`` uses for congruence workers, so the
#: runner-side refusal and the library-side clamp agree about what "fits"
#: means.  ``RAMRES=<GB>`` lowers it for a run whose operator has decided to
#: spend the margin -- a DELIBERATE, logged choice, which is the whole point
#: of this module; it is not a way to make the check go away, because the
#: modelled peak still has to fit what is left.
RESERVE_GB = float(os.environ.get('RAMRES', '8') or 8.0)

#: Disposition of a BINDING BOX check on an explicit intent -- the ``on_*``
#: action pattern the library uses everywhere else (``on_ram_cap``,
#: ``on_mem_budget``, ``on_tilt_exact_grid``), spelled the same way and
#: dispatched through ``carrier._guard_dispose``.  ``ONBOX=<action>``
#: overrides it for a whole run; ``preflight(on_box_budget=...)`` for one call.
#:
#: ``'warn'`` is the default BECAUSE the intent is explicit: see the header.
#: ``'ignore'`` is deliberately NOT accepted -- the entire purpose of this
#: module is that the grid choice is loud, and a silent over-commit is the
#: failure mode it exists to prevent.  The vocabulary is therefore two words,
#: not the library's three, and that is checked rather than assumed.
_BOX_ACTIONS = ('warn', 'error')
ON_BOX_BUDGET = (os.environ.get('ONBOX', '') or 'warn').strip().lower()

#: Substrings of the clamp's own guard messages.  A run of record must not see
#: any of them.  Kept as data rather than one regex so the failure report can
#: name which one fired.
#:
#: NOTE FOR ANYONE EDITING THE OVERRIDE WARNING BELOW: its text must contain
#: NONE of these.  ``record_warnings`` tees every warning the run raises into
#: :func:`assert_no_grid_degradation`, so an over-commit warning that spelled
#: one of these marks would make the pre-flight fail its own post-run check
#: with exit 3 ("the model and the clamp disagree") on a run where nothing
#: degraded.  Pinned by the override module's test.
_DEGRADE_MARKS = (
    'MEMORY-LIMITED',            # _memory_bounded_n_fine
    'COUNT-LIMITED',             # readout n_fine_cap bind
    'RESOLUTION-LIMITED',        # both, the shared verdict phrase
)

#: MEASURED whole-process / worker-child peak RSS, in GB, on this branch --
#: the reference the over-commit warning quotes so the operator sees the model
#: and the measurement side by side rather than the envelope alone.
#:
#: RE-MEASURED 2026-08-10 on the FINAL round-2 tree
#: (docs/audits/FIX_CLAMP_RECAL_OVERRIDE_2026_08_10.md sec 2); the authority
#: for the full set of rows, with their labels and their model ratios, is
#: ``tests/unit/test_niche_d8_congruence_workers.py::_MEASURED_PEAK_BYTES``.
#: What is kept here is one number per grid -- the LARGEST measured peak at
#: that grid, i.e. the honest thing to quote to someone deciding whether to
#: spend the box on it.
MEASURED_PEAK_GB = {
    4096: 7.09,
    8192: 26.74,
    16384: 87.93,
}


def resolve_ram_budget(env_value, workers=1):
    """Turn the ``RAMB`` env string into (override, effective, description).

    ``override`` is what to put in ``output_grid['ram_budget']`` /
    ``focus_readout['ram_budget']`` -- ``None`` means "pass nothing, use the
    box".  ``effective`` is the budget the clamp will actually see ON THE LEG,
    which is where the division by ``workers`` is applied.
    """
    workers = max(1, int(workers))
    v = str(env_value).strip().lower()
    if v in ('', 'auto', 'box', 'none'):
        box = float(get_ram_budget())
        return None, box / workers, (
            'auto (the box: get_ram_budget() = %.2f GB%s)'
            % (box / 1e9, '' if workers == 1
               else ', DIVIDED by %d workers -> %.2f GB each'
                    % (workers, box / workers / 1e9)))
    if v == 'inf':
        return float('inf'), float('inf'), 'inf (RAM clamp DISABLED, explicitly)'
    try:
        gb = float(v)
    except ValueError:
        raise SystemExit(
            "RAMB must be 'auto', 'inf' or a number of GB, got %r" % env_value)
    if not (gb > 0.0):
        raise SystemExit("RAMB must be > 0 GB, got %r" % env_value)
    return gb * 1e9, gb * 1e9, '%.2f GB, per worker, explicitly' % gb


def _box_memory():
    """``(available, total)`` physical memory in bytes.

    One seam, for one reason: the BOX check now has two thresholds -- free
    memory (a judgement call, disposed by ``on_box_budget``) and TOTAL
    physical memory (not a judgement call) -- and a test that cannot set both
    cannot pin the difference between them.  Imported lazily, exactly as the
    inline ``import psutil`` it replaces was.
    """
    import psutil
    vm = psutil.virtual_memory()
    return float(vm.available), float(vm.total)


def _measured_reference(n_fine_cap):
    """The MEASURED peak to quote alongside the model, as ``(n, gb)``.

    Exact grid if one was measured, else the largest measured grid below the
    request (whose peak is a LOWER bound on this one's, which is the honest
    direction), else ``None`` -- and the caller words the sentence for each
    case rather than inventing a number by extrapolation.
    """
    n = int(n_fine_cap)
    if n in MEASURED_PEAK_GB:
        return n, MEASURED_PEAK_GB[n]
    below = [k for k in MEASURED_PEAK_GB if k < n]
    if below:
        k = max(below)
        return k, MEASURED_PEAK_GB[k]
    return None


def preflight(n_fine_cap, ramb_env='auto', workers=1, n_px=0, n_out=0,
              label='run', paraxial=False, stream=None, on_box_budget=None):
    """PROVE the run gets ``n_fine_cap``, or say so loudly.

    Returns the ``ram_budget`` value to pass through (possibly ``None``).

    Raises ``SystemExit(2)`` -- not a warning -- when

      * the RAM CLAMP would bind, i.e. the leg would run at a smaller grid
        than the label says.  No disposition: a silently-degraded answer is
        the failure this module exists for; and
      * the modelled need is at or above the box's TOTAL physical memory,
        which no amount of freeing can rescue; and
      * ``on_box_budget='error'`` and the modelled need exceeds free memory.

    ``on_box_budget`` (``'warn'`` default, from ``ONBOX``) disposes of the
    remaining case -- modelled need above free memory but below the box's
    total.  ``'warn'`` emits a ``RuntimeWarning`` naming modelled need, free
    memory and the measured reference peak, and PROCEEDS at the requested
    grid; ``'error'`` refuses as before.  See the header for why an explicit
    intent is not over-ruled by an upper-bound envelope.

    ``paraxial=True`` (``final_leg='paraxial'``) builds no fine grid, so both
    checks reduce to the chain's own footprint and ``n_fine_cap`` is not
    asserted.
    """
    out = stream or sys.stdout
    workers = max(1, int(workers))
    act = (ON_BOX_BUDGET if on_box_budget is None
           else str(on_box_budget)).strip().lower()
    if act not in _BOX_ACTIONS:
        raise SystemExit(
            "on_box_budget (ONBOX) must be one of %s, got %r.  'ignore' is "
            "deliberately not accepted: an over-commit that nobody is told "
            "about is the failure this pre-flight exists to prevent."
            % (', '.join(repr(a) for a in _BOX_ACTIONS), on_box_budget
               if on_box_budget is not None else ON_BOX_BUDGET))
    override, effective, how = resolve_ram_budget(ramb_env, workers)

    def _say(s):
        print(s, file=out, flush=True)

    _say("")
    _say("GRID INTENT -- %s" % label)
    _say("  n_fine_cap requested : %s" % ('n/a (paraxial leg)' if paraxial
                                          else str(int(n_fine_cap))))
    _say("  ram_budget           : %s" % how)
    _say("  congruence workers   : %d" % workers)

    if paraxial:
        _say("  paraxial leg: no fine grid is built, so neither the clamp nor "
             "the box check applies.")
        return override

    n_fine_cap = int(n_fine_cap)
    # ---- 1. CLAMP -------------------------------------------------------
    ceiling = _C._fine_grid_ceiling(effective)
    need_b = (_C._FINE_GRID_WORK_ARRAYS * 16.0 * float(n_fine_cap) ** 2
              / _C._FINE_GRID_RAM_FRAC)
    _say("  clamp ceiling        : %s  (the largest grid %s approves; %d "
         "needs %.1f GB of budget)"
         % (('unbounded' if effective == float('inf') else '%d' % ceiling),
            ('an unbounded budget' if effective == float('inf')
             else 'a %.2f GB budget' % (effective / 1e9)),
            n_fine_cap, need_b / 1e9))
    if ceiling < n_fine_cap:
        _refuse(out, label,
                "the RAM clamp would DEGRADE the fine grid from %d to %d."
                % (n_fine_cap, ceiling),
                [("this run would have produced a %dx%d answer wearing a %d "
                  "label -- the silent-8192 failure of "
                  "ADJUDICATION_NFC_8192_2026_08_10.md sec 2.1."
                  % (ceiling, ceiling, n_fine_cap)),
                 ("the clamp needs %.1f GB of budget for %d "
                  "(%d work arrays x 16 B x %d^2 over a %.0f%% reserve); the "
                  "leg will see %.2f GB."
                  % (need_b / 1e9, n_fine_cap, _C._FINE_GRID_WORK_ARRAYS,
                     n_fine_cap, 100.0 * _C._FINE_GRID_RAM_FRAC,
                     effective / 1e9))],
                ["%-20s-- disable the clamp, having read that one order is "
                 "modelled at %.1f GB"
                 % ('RAMB=inf',
                    _C._fine_grid_peak_bytes(n_fine_cap, n_px) / 1e9),
                 "%-20s-- pin the budget (>= %.1f)" % ('RAMB=<GB>',
                                                       need_b / 1e9),
                 "%-20s-- ask for the grid the box can hold"
                 % ('NFC=%d' % ceiling),
                 ("%-20s-- stop dividing the budget by %d"
                  % ('CW=1', workers)) if workers > 1 else
                 "%-20s-- close what is holding it" % 'free memory'])

    # ---- 2. BOX ---------------------------------------------------------
    avail, physical = _box_memory()
    per_worker = _C._fine_grid_peak_bytes(n_fine_cap, n_px)
    acc = 16.0 * float(n_out) ** 2 if n_out else 0.0
    # THE ACCUMULATOR IS CHARGED ONLY AT k > 1, and that is measured, not a
    # convenience.  At k=1 the congruence runs IN the parent, so the fine
    # leg's peak and the common-grid accumulator are SEQUENTIAL: the leg's
    # arrays are gone before the accumulator is filled.  ADJUDICATION_NFC_8192
    # sec 5.3's arm B is the direct evidence -- 32 orders at NFC=16384 on the
    # NOUT=32768 grid (a 17.2 GB accumulator) peaked at 71.9-80.5 GiB = 77-86
    # GB, against a leg peak of ~83.5 GB on its own; i.e. +3 GB, not +17.2.
    # AUDIT_TRACED_SPEED sec 3.3 recorded the same thing from the other side
    # ("the N_out=32768 accumulator is NOT yet allocated at that instant").
    # At k > 1 they DO co-occur: the parent holds the accumulator while the
    # workers hold their legs, which is the whole shape of the k-way peak.
    parent = (acc + _C._FINE_GRID_BASE_BYTES) if workers > 1 else 0.0
    total = workers * per_worker + parent
    _say("  modelled peak        : %d x %.1f GB per worker + %.1f GB parent "
         "= %.1f GB   (free %.1f GB, reserve %.0f GB)"
         % (workers, per_worker / 1e9, parent / 1e9, total / 1e9,
            avail / 1e9, RESERVE_GB))
    if total > avail - RESERVE_GB * 1e9:
        # THREE OUTCOMES, and which one applies is not a matter of degree.
        #
        # (a) IMPOSSIBLE.  The modelled need is at or above the box's TOTAL
        #     physical memory.  Freeing every byte on the machine would not
        #     make this run fit, so there is nothing for an operator to decide
        #     and no disposition applies -- it refuses whatever ONBOX says.
        #     This is the check that keeps `k=2 at NFC=16384` refused on this
        #     box (2 x 105.2 GB modelled, and 2 x 87.9 GB MEASURED, against
        #     137.4 GB physical) now that a binding FREE check no longer
        #     refuses on its own.
        # (b) 'error'.  The caller asked for the pre-change behaviour.
        # (c) 'warn' (the default).  Loud, and it proceeds -- see the header.
        why = [("%d worker(s) x %.1f GB (fine grid %d^2 at %d work arrays + "
                "a %.1f GB process floor) + %.1f GB parent."
                % (workers, per_worker / 1e9, n_fine_cap,
                   _C._FINE_GRID_WORK_ARRAYS,
                   _C._FINE_GRID_BASE_BYTES / 1e9, parent / 1e9)),
               ("the model is a MEASURED upper-bound envelope, not a "
                "prediction: see FIX_CLAMP_RECAL_OVERRIDE_2026_08_10.md "
                "sec 2 for the peaks it bounds and by how much.")]
        remedies = [
            "%-20s-- the k this box can actually hold"
            % ('CW=%d' % max(1, int((avail - RESERVE_GB * 1e9 - parent)
                                    // max(per_worker, 1.0)))),
            "%-20s-- a lighter grid" % ('NFC=%d' % (n_fine_cap // 2)),
            "%-20s-- close what is holding it" % 'free memory',
            "%-20s-- spend the reserve, deliberately (currently %.0f GB)"
            % ('RAMRES=<GB>', RESERVE_GB)]
        if total >= physical:
            _refuse(out, label,
                    "the box PHYSICALLY cannot allocate this run: %.1f GB "
                    "modelled against %.1f GB of TOTAL physical memory "
                    "(%.1f GB of it free)."
                    % (total / 1e9, physical / 1e9, avail / 1e9),
                    why + ["freeing memory cannot rescue this one, which is "
                           "why on_box_budget does not dispose of it."],
                    remedies[:2] + [remedies[-1]])
        if act == 'error':
            _refuse(out, label,
                    "the box cannot hold this run: %.1f GB modelled against "
                    "%.1f GB free (%.0f GB reserved), and "
                    "on_box_budget='error'."
                    % (total / 1e9, avail / 1e9, RESERVE_GB),
                    why, remedies + [
                        "%-20s-- proceed anyway, loudly (the default)"
                        % 'ONBOX=warn'])
        # ---- (c) allow, with warnings ----------------------------------
        ref = _measured_reference(n_fine_cap)
        if ref is None:
            ref_txt = ("no peak has been MEASURED at any grid at or below "
                       "%d, so the envelope is all there is here" % n_fine_cap)
        elif ref[0] == n_fine_cap:
            ref_txt = ("the MEASURED peak at n_fine=%d on this branch is "
                       "%.1f GB per process (model/measured %.2fx)"
                       % (ref[0], ref[1], (per_worker / 1e9) / max(ref[1],
                                                                  1e-9)))
        else:
            ref_txt = ("no peak has been MEASURED at n_fine=%d; the largest "
                       "grid that has is %d at %.1f GB per process, which is "
                       "a LOWER bound on this one"
                       % (n_fine_cap, ref[0], ref[1]))
        msg = (
            "GRID INTENT (%s): PROCEEDING OVER A BINDING MEMORY MODEL.  The "
            "modelled peak is %.1f GB (%d x %.1f GB per worker + %.1f GB "
            "parent) against %.1f GB free with a %.0f GB reserve, on a box "
            "with %.1f GB of physical memory -- so the model says this may "
            "not fit.  It is allowed to proceed because the intent is "
            "EXPLICIT and the model is an UPPER-BOUND envelope over measured "
            "peaks, not a prediction: %s.  The grid is NOT degraded to make "
            "it fit -- this run gets the %d it asked for or it dies trying, "
            "which is the point.  Watch it, or pass ONBOX=error "
            "(on_box_budget='error') to refuse instead."
            % (label, total / 1e9, workers, per_worker / 1e9, parent / 1e9,
               avail / 1e9, RESERVE_GB, physical / 1e9, ref_txt, n_fine_cap))
        _C._guard_dispose('warn', msg, stacklevel=2)
        bar = '~' * 74
        _say("")
        _say(bar)
        _say("PROCEEDING UNDER WARNING -- %s" % label)
        _say(bar)
        _say("  %.1f GB modelled  >  %.1f GB free - %.0f GB reserve   "
             "(%.1f GB physical)" % (total / 1e9, avail / 1e9, RESERVE_GB,
                                     physical / 1e9))
        _say("  %s" % ref_txt)
        _say("  the requested grid is UNCHANGED at %d -- nothing here "
             "degrades it." % n_fine_cap)
        _say("  ONBOX=error refuses instead; ONBOX=warn (default) is what "
             "you have.")
        _say(bar)
        return override
    _say("  VERDICT: the clamp cannot bind and the box can hold it.")
    return override


def _refuse(out, label, headline, why, remedies, code=2):
    bar = '!' * 74
    print("", file=out)
    print(bar, file=out)
    print("REFUSED -- %s" % label, file=out)
    print(bar, file=out)
    print("  %s" % headline, file=out)
    for w in why:
        print("  * %s" % w, file=out)
    print("  Remedies:", file=out)
    for r in remedies:
        print("    %s" % r, file=out)
    print(bar, file=out, flush=True)
    raise SystemExit(code)


class record_warnings:
    """Record every warning raised inside the block AND still show it.

    ``warnings.catch_warnings(record=True)`` would swallow them, and these
    runners' warnings are diagnostics the campaign reads (AUDIT_TRACED_SPEED
    sec 8.4).  Teeing ``showwarning`` keeps both.  Congruence workers'
    warnings arrive here too: the multi entry point re-raises them in the
    parent, tagged with the congruence name.
    """

    def __init__(self):
        self.records = []

    def __enter__(self):
        self._orig = warnings.showwarning

        def _tee(message, category, filename, lineno, file=None, line=None):
            self.records.append({
                'msg': str(message), 'cat': getattr(category, '__name__', '?'),
                'where': '%s:%d' % (os.path.basename(str(filename)), lineno)})
            return self._orig(message, category, filename, lineno, file, line)

        warnings.showwarning = _tee
        return self

    def __exit__(self, *exc):
        warnings.showwarning = self._orig
        return False

    def degradations(self):
        return [r for r in self.records
                if any(m in r['msg'] for m in _DEGRADE_MARKS)]


def assert_no_grid_degradation(rec, n_fine_cap, label='run', stream=None):
    """Post-run half of the intent check: the clamp must not have fired.

    Returns silently, or raises ``SystemExit(3)``.  Separate exit code from
    the pre-flight's 2 on purpose: a 2 means "this box cannot do what you
    asked", a 3 means "the pre-flight said it could and it did not", which is
    a defect in the model, not in the request.
    """
    out = stream or sys.stdout
    bad = rec.degradations()
    if not bad:
        print("  grid check: no MEMORY/COUNT/RESOLUTION-LIMITED warning was "
              "raised -- the leg ran at the %d it was asked for."
              % int(n_fine_cap), file=out, flush=True)
        return
    _refuse(out, label,
            "the fine grid WAS degraded during the run, after the pre-flight "
            "proved it could not be.",
            ['%s  [%s]' % (b['msg'][:220], b['where']) for b in bad[:4]],
            ["re-run with RAMB=inf or a pinned RAMB=<GB>",
             "report this: the pre-flight model and the clamp disagree"],
            code=3)
