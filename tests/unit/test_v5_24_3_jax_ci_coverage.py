"""v5.24.4 (AUDIT_V5_24_2 S4-4): JAX must be exercised in a CI leg.

Finding S4-4 [P2][test-coverage]: no workflow installed ``jax``, so all
jax-guarded unit files (module- or function-level
``pytest.importorskip('jax')``) skipped in every CI leg -- including the
numpy<->jax parity gates and the P1 regression files.  Every JAX-side
finding in the audit was therefore invisible to CI.

The fix adds one dedicated, NON-matrix ``jax-unit`` job to
``.github/workflows/unit-tests.yml`` that installs the ``jax`` extra and
runs exactly the jax-guarded files (selected by the guard, not by a
``-k jax`` name filter, so module-level-jax files and the
``through_focus_stream_matches_fused`` parity gate -- both of which a
``-k jax`` selection silently drops -- are covered).

This walker pins that job in place.  It is an INDEPENDENT oracle, not a
tautology: it recomputes the jax-guarded file set from the test tree with
its own scan and asserts (a) the set is non-empty and includes the two
files a ``-k jax`` filter would miss, and (b) the workflow has a single
non-matrix job that installs the ``jax`` extra AND runs pytest over the
jax-guarded selection.  It parses the workflow as text (no PyYAML, which
is not in the CI ``dev`` install) and imports neither jax nor lumenairy,
so it runs in every CI leg -- exactly the coverage the finding is about.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "unit-tests.yml"
_TESTS_UNIT = _REPO_ROOT / "tests" / "unit"

# A jax-guard line: ``pytest.importorskip('jax')`` /
# ``importorskip("jax")`` / ``jax = pytest.importorskip('jax')``.  The
# small ``.{0,4}`` gap tolerates the quote/paren between token and name
# and mirrors the selection grep used by the CI job.
_JAX_GUARD_RE = re.compile(r"importorskip.{0,4}jax")

# Two files that a ``-k jax`` node-id filter would DROP but the
# guard-based selection covers, so the fix must keep selecting by guard:
#  * module-level ``importorskip('jax')`` whose file/test names lack "jax"
#  * a numpy<->jax parity gate whose function name lacks "jax"
_K_JAX_BLIND_SPOTS = (
    "test_v5_10_3_rcwa_2d_autodiff.py",
    "test_v5_18_1_residuals.py",
)


def _read_workflow() -> str:
    assert _WORKFLOW.is_file(), (
        f"S4-4 pin: CI workflow missing at {_WORKFLOW}; the jax CI leg "
        "cannot be verified."
    )
    return _WORKFLOW.read_text(encoding="utf-8")


def _job_blocks(text: str) -> dict[str, str]:
    """Split ``.github/workflows/unit-tests.yml`` into {job_name: body}.

    Jobs are the 2-space-indented keys under the top-level ``jobs:``
    mapping; their bodies are indented 4+ spaces.  This is the same
    text-parsing idiom the other walker tests use (no PyYAML dependency,
    which is absent from the CI ``dev`` install).
    """
    jobs_idx = text.find("\njobs:")
    assert jobs_idx != -1, "workflow has no top-level ``jobs:`` mapping"
    region = text[jobs_idx:]
    header_re = re.compile(r"^  ([A-Za-z0-9_-]+):[ \t]*$", re.MULTILINE)
    matches = list(header_re.finditer(region))
    blocks: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(region)
        blocks[m.group(1)] = region[m.start():end]
    return blocks


def _installs_jax_extra(block: str) -> bool:
    """True if the job block pip-installs the package with the ``jax`` extra."""
    return bool(
        re.search(r"pip install[^\n]*\[[^\]\n]*\bjax\b[^\]\n]*\]", block)
    )


def _runs_jax_guarded_selection(block: str) -> bool:
    """True if the job runs pytest over the jax-guarded file selection."""
    return (
        "importorskip" in block
        and "tests/unit" in block
        and re.search(r"python -m pytest", block) is not None
    )


def _has_matrix(block: str) -> bool:
    return bool(re.search(r"^\s*matrix:\s*$", block, re.MULTILINE))


def _scan_jax_guarded_files() -> list[str]:
    """Independently enumerate jax-guarded unit files from the tree.

    This is the oracle the CI job's grep must agree with; recomputing it
    here (rather than asserting a hard-coded list) keeps the pin honest
    as jax coverage grows.
    """
    found = []
    for path in sorted(_TESTS_UNIT.glob("*.py")):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover -- defensive
            continue
        if _JAX_GUARD_RE.search(src):
            found.append(path.name)
    return found


# ---------------------------------------------------------------------------
# S4-4.1 -- a dedicated non-matrix CI job installs jax AND runs the jax tests
# ---------------------------------------------------------------------------


def test_s4_4_dedicated_jax_ci_job_installs_and_runs_jax() -> None:
    """A single non-matrix job in unit-tests.yml installs the ``jax``
    extra and runs pytest over the jax-guarded selection.

    Closes S4-4: before the fix no workflow installed jax, so the
    ``importorskip('jax')`` guards skipped everywhere.
    """
    text = _read_workflow()
    blocks = _job_blocks(text)
    assert blocks, "no jobs parsed from unit-tests.yml"

    qualifying = [
        name
        for name, body in blocks.items()
        if _installs_jax_extra(body) and _runs_jax_guarded_selection(body)
    ]
    assert qualifying, (
        "AUDIT_V5_24_2 S4-4 regression: no CI job in "
        ".github/workflows/unit-tests.yml both installs the ``jax`` extra "
        "(``pip install -e \".[...,jax]\"``) AND runs pytest over the "
        "jax-guarded unit files.  Without such a job every "
        "``pytest.importorskip('jax')`` test skips in CI (the exact "
        f"S4-4 condition).  Jobs present: {sorted(blocks)}."
    )

    # Honor the finding's constraint: keep it ONE dedicated job so the
    # 4x3 fast matrix is not re-bloated (the documented v5.0.1 reason jax
    # was dropped there).  The jax job itself must not carry a matrix.
    non_matrix = [name for name in qualifying if not _has_matrix(blocks[name])]
    assert non_matrix, (
        "S4-4: the jax CI job(s) "
        f"{qualifying} all carry a ``matrix:`` -- the fix must be a single "
        "dedicated non-matrix job so the fast matrix is not re-bloated."
    )


def test_s4_4_jax_job_pins_blas_threads_to_avoid_openblas_deadlock() -> None:
    """The dedicated jax job MUST pin the numpy / OpenBLAS thread count
    (``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``) so the numpy-side
    linear algebra runs single-threaded.

    v5.24.4: JAX and OpenBLAS coexisting in ONE process deadlock on the
    first large *multi-threaded* ``numpy.linalg`` BLAS call -- concretely
    the traced-lens Chebyshev-fit ``lstsq`` in ``apply_real_lens_traced``
    (``_lens_traced.py:558``), reached by the non-jax cookbook tests that
    share a file with a jax-guarded test.  JAX's runtime and OpenBLAS both
    spin up OpenMP pools; the nested OpenMP hangs the worker, which is
    SIGTERM-killed (exit 143) mid-``lstsq`` -- silently failing the whole
    job with no ``FAILED`` line.  Pinning BLAS to one thread makes the
    numpy side deadlock-free (JAX/XLA parallelism is governed by XLA, not
    ``OMP_NUM_THREADS``, so the paths under test are not slowed).  A
    regression that drops the env re-introduces the hang, so pin it here.
    """
    text = _read_workflow()
    blocks = _job_blocks(text)
    qualifying = [
        body
        for name, body in blocks.items()
        if _installs_jax_extra(body) and _runs_jax_guarded_selection(body)
    ]
    assert qualifying, (
        "no qualifying jax job found (see the S4-4 install/run pin above)")
    for body in qualifying:
        assert 'OMP_NUM_THREADS' in body and 'OPENBLAS_NUM_THREADS' in body, (
            "v5.24.4 regression: the dedicated jax CI job no longer pins "
            "OMP_NUM_THREADS / OPENBLAS_NUM_THREADS.  JAX + multi-threaded "
            "OpenBLAS deadlock on the first large lstsq (the traced-lens "
            "Chebyshev fit in apply_real_lens_traced), SIGTERM-killing the "
            "job.  Restore the thread-limit ``env:`` on the run step.")


# ---------------------------------------------------------------------------
# S4-4.2 -- the selection actually covers the guarded files (independent oracle)
# ---------------------------------------------------------------------------


def test_s4_4_jax_guarded_files_exist_and_cover_k_jax_blind_spots() -> None:
    """There ARE jax-guarded unit files (so the job is not a no-op), and
    the guarded set includes the files a ``-k jax`` filter would drop --
    proving the fix must select by the guard, not by name.
    """
    guarded = _scan_jax_guarded_files()
    # There is a substantial jax surface to protect; a tiny count would
    # mean the scan (or the test layout) broke.
    assert len(guarded) >= 20, (
        "S4-4 oracle: expected many jax-guarded unit files "
        f"(>=20), found {len(guarded)}: {guarded}.  The scan pattern or "
        "the test tree changed -- re-verify before trusting the CI leg."
    )

    missing = [f for f in _K_JAX_BLIND_SPOTS if f not in guarded]
    assert not missing, (
        "S4-4 oracle: expected the ``-k jax`` blind-spot files "
        f"{list(_K_JAX_BLIND_SPOTS)} to be jax-guarded, but these were not "
        f"detected as guarded: {missing}.  If they were renamed/removed, "
        "update _K_JAX_BLIND_SPOTS; the point is that the CI job must keep "
        "selecting by the ``importorskip('jax')`` guard (name-independent) "
        "rather than by a ``-k jax`` node-id filter, which silently drops "
        "these module-level-jax / non-'jax'-named parity tests."
    )

    # And the CI job must select by that guard so those files run.
    blocks = _job_blocks(_read_workflow())
    selects_by_guard = any(
        _installs_jax_extra(body)
        and "importorskip" in body
        and re.search(r"grep\b", body) is not None
        for body in blocks.values()
    )
    assert selects_by_guard, (
        "S4-4: the jax CI job must select the jax-guarded files by the "
        "``importorskip('jax')`` guard (a ``grep`` over tests/unit), so the "
        f"name-independent blind-spot files {list(_K_JAX_BLIND_SPOTS)} are "
        "included.  A ``-k jax`` name filter alone would drop them."
    )
