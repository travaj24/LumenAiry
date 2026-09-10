"""CROSS-KERNEL CONSISTENCY of the library's guard DECISIONS.

Fix doc: ``docs/audits/CI_KERNEL_SWEEP_2026_09_11.md``.

WHY THIS FILE EXISTS.  The 5.45.0 release matrix went red on CI while the
same commit was green on both local builds, and every failure was a
DECISION -- a guard that refuses on one machine and returns on another, or a
test whose bar was pinned on a build-dependent reading.  Nothing in the gate
could have caught that locally, because the gate only ever ran on one BLAS
micro-kernel.  This file is the missing instrument: it reads a committed
census of every guard decision on eight (build, kernel) arms and asserts they
AGREE, then re-takes the cheap half of that census on whatever arm is running
now and asserts it joins the consensus.

WHAT AN ARM IS, AND WHY THE KERNEL IS THE AXIS.  The bundled scipy-openblas
is DYNAMIC_ARCH, so ``OPENBLAS_CORETYPE`` re-dispatches the whole BLAS/LAPACK
kernel set at import time -- a different reduction order and blocking for the
same arithmetic, which is exactly the axis that moves a rounding-level guard
reading.  The census was taken at ONE thread on every arm, so the kernel is
the only thing that varies.  Two measured caveats are recorded in
``validation/probe_ci_kernel_sweep/probe_decisions.py``: ``ZEN`` is not a
distinct kernel in these wheels (it resolves to ``Haswell``, which is also
what auto-detection gives on a Zen CPU, so the default arm ALREADY IS the CI
Zen arm), and ``SKYLAKEX`` is unreachable on a non-AVX-512 host (SIGILL).

THE TWO CLAIMS, and they point in opposite directions on purpose:

  * ``decisions``     -- what the library actually DOES.  Every arm must
                         agree.  A disagreement here is a P1: some user gets
                         a refusal, and another user gets an answer, from the
                         same code on the same input.
  * ``hypothetical``  -- what a bar the library does NOT ship would decide at
                         a site the library deliberately leaves UNGUARDED.
                         These must STAY non-unanimous.  They are the
                         standing evidence that the site is undecidable, and
                         if they ever became unanimous the omission would
                         need re-arguing rather than silently keeping.

BUDGET.  Reading and comparing the table is free; the current-arm re-take is
restricted to the four CHEAP sections (~3 s, measured).  The mortar and T22
sections are the expensive half and are covered by the committed table plus
their own fix/verify files, so this gate stays well inside its 30 s budget.

REGENERATING THE TABLE (after a deliberate, argued decision change)::

    for k in HASWELL SANDYBRIDGE NEHALEM PRESCOTT; do
      OPENBLAS_CORETYPE=$k OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        MKL_NUM_THREADS=1 python \\
        validation/probe_ci_kernel_sweep/probe_decisions.py \\
        --out validation/probe_ci_kernel_sweep/arms/<build>_$k.json
    done
    python validation/probe_ci_kernel_sweep/merge_arms.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE_DIR = _ROOT / "validation" / "probe_ci_kernel_sweep"
_TABLE = _PROBE_DIR / "decisions.json"

#: the sections the current arm RE-TAKES.  Cheap by measurement (1.4 / 0.1 /
#: 1.8 / 0.0 s); the mortar and T22 sections cost 4.7 and 5.3 s and are left
#: to the committed table so this gate keeps a wide margin under 30 s.
_CHEAP = ("interface", "sliver", "band", "branch_cut")

#: the decision PREFIXES each cheap section owns, so the re-take can be
#: compared against exactly the rows it is responsible for.
_PREFIX = {
    "interface": "pmm1d_interface/",
    "sliver": "sliver/",
    "band": "band/",
    "branch_cut": "branch_cut/",
}


def _load_table():
    if not _TABLE.is_file():
        pytest.fail(
            "the CI kernel census is missing: %s.  It is a COMMITTED file; "
            "regenerate it with validation/probe_ci_kernel_sweep/"
            "probe_decisions.py on each arm followed by merge_arms.py (see "
            "this module's docstring)." % _TABLE)
    with _TABLE.open(encoding="cp1252") as fh:
        return json.load(fh)


def _probe_module():
    """Load the probe BY PATH.

    ``validation/`` is not a package on ``sys.path``, and making it one to
    satisfy an import would put probe scripts into the library's import
    namespace.  Loading by path also guarantees this gate and the census are
    reading the SAME fixture definitions -- duplicating them here is the one
    thing a consistency check must not do.
    """
    path = _PROBE_DIR / "probe_decisions.py"
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("_ci_kernel_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# 1 -- the table itself has to be worth reading
# ======================================================================
def test_the_census_spans_more_than_one_kernel_and_more_than_one_build():
    """A census taken on one machine proves nothing about another.

    The table must carry at least two distinct BLAS kernels and at least two
    distinct builds, and every arm must have been taken at ONE thread -- an
    arm with threads unpinned is measuring the reduction order of a thread
    split, not of a kernel, and cannot be compared with the others.
    """
    t = _load_table()
    arms = t["arms"]
    assert len(arms) >= 4, sorted(arms)
    kernels = {m["kernel"] for m in arms.values()}
    builds = {m["build"] for m in arms.values()}
    assert len(kernels) >= 2, kernels
    assert len(builds) >= 2, builds
    for arm, m in arms.items():
        for var, val in m["threads"].items():
            assert val == "1", (arm, var, val)
    # the arm KEY is the MEASURED kernel, not the requested one: several
    # requested coretypes resolve to the same kernel (``ZEN`` and an
    # unrecognised name both land on ``Haswell``), and two arms that ran
    # identical code must not look like independent evidence.
    for arm, m in arms.items():
        assert arm == "%s-%s" % (m["build"], m["kernel"]), (arm, m)
    assert len(set(arms)) == len(arms)


# ======================================================================
# 2 -- THE claim: every arm takes the same decision
# ======================================================================
def test_every_arm_takes_the_same_guard_decision():
    """The headline.  Each guard OUTCOME must be identical on every arm.

    A row that disagrees is a P1 by construction: the same code, on the same
    input, refuses for one user and answers for another.  The failure message
    names the row and prints each arm's verdict, because the useful next step
    is always "which side is right", not "which arm is odd".
    """
    t = _load_table()
    dec = t["decisions"]
    arms = sorted(dec)
    keys = sorted({k for d in dec.values() for k in d})
    assert keys, "the census carries no decisions at all"
    split = {}
    for k in keys:
        seen = {a: dec[a].get(k, "<absent>") for a in arms}
        if len(set(seen.values())) > 1:
            split[k] = seen
    assert not split, (
        "%d guard DECISION(s) are not the same on every arm -- this is a P1, "
        "not a flake:\n%s" % (len(split), json.dumps(split, indent=1,
                                                     sort_keys=True)))
    # and every arm answered every row: a row missing from one arm is a
    # census hole, which reads as agreement above and must not.
    holes = {a: sorted(set(keys) - set(dec[a])) for a in arms
             if set(keys) - set(dec[a])}
    assert not holes, holes


# ======================================================================
# 3 -- the counterexample must STAY a counterexample
# ======================================================================
def test_the_unguarded_sites_stay_undecidable_across_the_kernels():
    """The other direction, and it is not symmetry for its own sake.

    ``hypothetical`` holds the verdict a bar the library does NOT ship would
    return at a site it deliberately leaves unguarded -- today, the 1e-12
    rcond bar at the plain 1-D ``_interface_smatrix`` site (see
    ``tests/unit/test_fix_pmm2d_mortar_round2.py::
    test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why``).
    The reason that site ships unguarded is that the bar's verdict FLIPS with
    the BLAS kernel.  If this ever became unanimous, the omission would rest
    on a premise that no longer holds and would have to be re-argued -- so
    the non-unanimity is asserted, not assumed.
    """
    t = _load_table()
    hyp = t.get("hypothetical", {})
    arms = sorted(a for a in hyp if hyp[a])
    assert arms, "the census carries no hypothetical-bar verdicts"
    keys = sorted({k for a in arms for k in hyp[a]})
    undecidable = {
        k: {a: hyp[a].get(k, "<absent>") for a in arms}
        for k in keys
        if len({hyp[a].get(k) for a in arms}) > 1
    }
    assert "pmm1d_interface/bar_1e-12_would@1e-05" in undecidable, (
        "the 1e-12 bar at the plain 1-D interface site now reads the SAME on "
        "every arm in the census.  That is the premise the site's unguarded "
        "status rests on, so it has changed: re-argue the omission (or arm "
        "the guard) rather than deleting this assertion.  Verdicts: %s"
        % json.dumps({a: hyp[a].get("pmm1d_interface/bar_1e-12_would@1e-05")
                      for a in arms}, sort_keys=True))
    assert set(undecidable["pmm1d_interface/bar_1e-12_would@1e-05"].values()) \
        == {"refuse", "accept"}, undecidable


# ======================================================================
# 4 -- the arm running RIGHT NOW has to join the consensus
# ======================================================================
def test_this_arm_agrees_with_the_committed_census():
    """Re-take the cheap half of the census here and compare.

    This is what makes the gate local rather than archival: a kernel nobody
    has censused yet (a future CI runner, a colleague's laptop) either agrees
    with the table or fails here with its own arm named.  Only the four cheap
    sections are re-taken -- see the module docstring's BUDGET note.
    """
    t = _load_table()
    dec = t["decisions"]
    consensus = {}
    for k in sorted({k for d in dec.values() for k in d}):
        vals = {d.get(k) for d in dec.values()}
        if len(vals) == 1:
            consensus[k] = vals.pop()

    m = _probe_module()
    here, hyp, rea = {}, {}, {}
    t0 = time.perf_counter()
    m._interface_site_population(here, hyp, rea)
    m._sliver_decisions(here, rea)
    m._band_decisions(here, rea)
    m._branch_cut_decisions(here, rea)
    elapsed = time.perf_counter() - t0

    arm, _build, _kernel = m._arm_id()
    prefixes = tuple(_PREFIX[s] for s in _CHEAP)
    mismatch = {k: (v, consensus[k]) for k, v in here.items()
                if k in consensus and consensus[k] != v}
    assert not mismatch, (
        "arm %r disagrees with the committed census on %d guard decision(s) "
        "(measured here, census value): %s.  Either this arm has found a "
        "kernel-dependent decision the census does not cover -- add it with "
        "probe_decisions.py + merge_arms.py and FIX the guard -- or a guard "
        "changed and the census is stale."
        % (arm, len(mismatch), json.dumps(mismatch, indent=1, sort_keys=True)))
    # every cheap row this arm produced must EXIST in the census: a new
    # decision that nobody has censused is a hole, and it would pass the
    # comparison above by being absent from ``consensus``.
    uncensused = sorted(k for k in here
                        if k.startswith(prefixes) and k not in consensus
                        and not any(k in d for d in dec.values()))
    assert not uncensused, (
        "these decisions are produced by the probe but are in NO arm of the "
        "census: %s.  Re-run the arms and merge." % uncensused)
    # the budget this file promises, measured on the running build
    assert elapsed < 20.0, elapsed


def test_the_cheap_sections_cover_every_area_the_sweep_found_a_flip_in():
    """A guard against the gate quietly narrowing.

    The re-take above is only as good as the sections it runs.  This pins
    which prefixes are re-taken locally, so dropping one (to save a second,
    say) is a visible edit rather than a silent loss of coverage.
    """
    t = _load_table()
    keys = {k for d in t["decisions"].values() for k in d}
    for section, prefix in _PREFIX.items():
        assert section in _CHEAP, section
        assert any(k.startswith(prefix) for k in keys), (section, prefix)
    m = _probe_module()
    for section in _CHEAP:
        assert hasattr(m, {"interface": "_interface_site_population",
                           "sliver": "_sliver_decisions",
                           "band": "_band_decisions",
                           "branch_cut": "_branch_cut_decisions"}[section])
