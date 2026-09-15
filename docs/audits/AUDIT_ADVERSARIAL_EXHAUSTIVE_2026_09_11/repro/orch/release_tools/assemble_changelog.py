#!/usr/bin/env python
"""Assemble the release block of CHANGELOG.md from the per-WP changelog files.

    python assemble_changelog.py --version 5.46.0 --date 2026-09-12            # dry run: block -> stdout, pre-flight -> stderr
    python assemble_changelog.py --version 5.46.0 --date 2026-09-12 --write    # insert/replace the block in CHANGELOG.md
    python assemble_changelog.py --version 5.46.0 --bump                       # also bump pyproject.toml + lumenairy/__init__.py

The pre-flight applies the repository's own CHANGELOG walkers' regexes (V12 file-path citations, V12.2 audit-ID
tokens, V12.3 test-count claim arithmetic, V17 file/line-count claims) to the assembled block so a walker failure is
seen here first.  Every WP file is included in ORDER; a missing file is reported, not fatal.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
AUD_REL = "docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11"
AUD = REPO / AUD_REL
FIX = AUD / "fixes"
HERE = pathlib.Path(__file__).resolve().parent

# Release-block order: the apply_real_lens family first (the audit's focus), then the rest of the physics,
# then infrastructure, packaging and documentation.
ORDER_547 = [
    ("B1", "Maslov: the asymptotic saddle follows the input field's local wavevector (S6 proper)"),
    ("VERIFY_WP-B1", "Maslov S6: verifier follow-ups"),
    ("B2", "Analytic lens: the 2-D displaced remap -- symmetric window, structured inversion, displaced_n_side (L9)"),
    ("VERIFY_WP-B2", "Analytic remap: verifier follow-ups"),
    ("B10", "Traced lens: a disc-orthogonal (Zernike) fit basis, opt-in"),
    ("VERIFY_WP-B10", "Traced lens fit basis: verifier follow-ups"),
    ("B3", "Propagator kernels: HFPI output plane and Sobol sampler, the pixel-integrated RS kernel, the chirp-Z resampler"),
    ("VERIFY_WP-B3", "Propagator kernels: verifier follow-ups"),
    ("B3b", "Propagator call sites: the Fresnel and SAS legs of system.py and the analytic lens gaps"),
    ("VERIFY_WP-B3b", "Propagator call sites: verifier follow-ups"),
    ("B4", "Traced-carrier chain: Collins / ABCD-Fresnel transport with a Bluestein output grid"),
    ("VERIFY_WP-B4", "Carrier transport: verifier follow-ups"),
    ("B7", "Asymptotic family: Y4 performance, FGA routing, the uniform asymptotics, GBD kernel clipping"),
    ("VERIFY_WP-B7", "Asymptotic family: verifier follow-ups"),
    ("B7b", "FGA / uniform asymptotics: the caustic route, the fold envelope, the analytic-Jacobian predicate"),
    ("VERIFY_WP-B7b", "FGA / uniform asymptotics: verifier follow-ups"),
    ("B9", "Ray tracing: the performance items and the aspheric analytic Jacobian"),
    ("VERIFY_WP-B9", "Ray tracing: verifier follow-ups"),
    ("B5", "RCWA / EME / BOR: Toeplitz solves, the two-interface closed form, off-plane fff_nv symmetrisation"),
    ("VERIFY_WP-B5", "RCWA / EME / BOR: verifier follow-ups"),
    ("B6", "PMM: the k0-free tensor operator cache; the Gegenbauer basis measured and not shipped"),
    ("VERIFY_WP-B6", "PMM: verifier follow-ups"),
    ("B8", "Analysis and sources: PSF memory and MFT sampling, encircled-energy profile, Zernike recurrence, Gori pseudo-modes"),
    ("VERIFY_WP-B8", "Analysis and sources: verifier follow-ups"),
    ("B11", "Hygiene pass: consolidations and the small deferred items"),
]

ORDER = [
    ("A2", "Analytic `apply_real_lens` and its displaced-model siblings"),
    ("A3", "Traced lens family (`apply_real_lens_traced`, caustic siblings)"),
    ("A4", "Maslov / GBD / FGA / asymptotic / JAX lens models"),
    ("VERIFY_WP-A4", "Maslov / asymptotic verification follow-ups (dimensionless LG Strehl merit)"),
    ("A6", "Traced-carrier chain"),
    ("A24", "Traced-carrier chain: decentre calibration re-measured, d6 bar restated"),
    ("A25", "Traced-carrier chain: paraxial focus readout regression (full-run follow-up)"),
    ("A1", "Ray tracing and the exit-vertex helper"),
    ("A26", "Ray tracing: decentred exit reference regression (full-run follow-up)"),
    ("A5", "Propagator kernels"),
    ("A7", "Analysis"),
    ("A8", "Thin elements, apertures, BSDF, gratings, glass catalogue"),
    ("A12", "PMM 1-D"),
    ("A13", "PMM 2-D"),
    ("A14", "RCWA / EME / BOR"),
    ("A11", "Polarization, coatings, sources, algebra, infrastructure"),
    ("A10", "I/O, prescriptions, optimisation"),
    ("A9", "Designer UI"),
    ("A16", "Lens configuration objects (new feature)"),
    ("VERIFY_WP-A16", "Lens configuration objects: verifier fixes (sag_dtype rule, value equality, import-time restated)"),
    ("A15a", "Tests, CI, packaging"),
    ("A15b", "Architecture: optional dependencies, knob overrides, lazy loading"),
    ("A20", "Loose ends: legacy folded library entries, rs_alias_free_distance, two restated pins"),
    ("A21", "Final hygiene pass"),
    ("A22", "Final code pass (lazy scipy.fft, fingerprint recorder, isolation fixes)"),
    ("A23", "CI kernel census re-recorded"),
    ("A17", "History relocation (part 1: carrier chain, FFT layer)"),
    ("A17_SWEEP1", "History relocation sweep 1 (propagators, analysis, sources)"),
    ("A17_SWEEP2", "History relocation sweep 2 (elements)"),
    ("A17_SWEEP3", "History relocation sweep 3 (root, io, optimize, raytrace, ui, backend)"),
    ("A17_SWEEP4", "History relocation sweep 4 (lens family, hygiene-pass files)"),
    ("A18", "Documentation"),
]

# The repository's V12 walker regexes (tests/unit/test_v5_2_walker_changelog_changeset.py), copied verbatim.
FILE_PATH_PATTERN = re.compile(
    r'`(?P<path>'
    r'(?:lumenairy|tests|examples|docs|scripts|benchmarks|validation)'
    r'/[\w\-./]+\.\w{1,5}'
    r'|\.github/[\w\-./]+\.\w{1,5}'
    r'|(?:README|CHANGELOG|ROADMAP|Migration-Guide|CONTRIBUTING|CONVENTIONS)\.md'
    r'|pyproject\.toml'
    r'|requirements\.txt'
    r')'
    r'(?::\d+(?:-\d+)?)?'
    r'`')
AUDIT_ID_PATTERN = re.compile(r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z0-9][A-Z0-9_-]*)')
TEST_COUNT_PATTERN = re.compile(
    r'(?P<pass>\d{3,5})\s+unit tests pass[^(]*\(\s*collected\s*=?\s*(?P<coll>\d{3,5})\s*=\s*'
    r'pass\s*\+\s*(?P<skip>\d+)\s+skip\s*\+\s*(?P<xfail>\d+)\s+xfail', re.IGNORECASE)
V17_PATTERNS = [re.compile(p) for p in (
    r'\d+\s+files?\s+(?:touched|changed|modified)',
    r'CHANGELOG\.md:\s*\d+\s*->\s*\d+\s*lines',
    r'Net LOC:\s*\+\d+\s*/\s*-\d+',
)]


def load_wp(path: pathlib.Path) -> str:
    """Return the WP file's body from its first '### ' heading on, with stray H1/H2 headings demoted so the
    release block is never split (the walkers end a block at the next '## [')."""
    lines = path.read_text(encoding="utf-8").splitlines()
    # WP-A9's file has two blocks: BLOCK 1 for CHANGELOG.md, BLOCK 2 for GUI_CHANGELOG.md (see gui_block()).
    cut = next((i for i, l in enumerate(lines) if l.startswith("## BLOCK 2")), None)
    if cut is not None:
        lines = lines[:cut]
    idx = next((i for i, l in enumerate(lines) if l.startswith("### ")), None)
    if idx is None:
        raise SystemExit(f"{path.name}: no '### ' heading found")
    out = []
    for l in lines[idx:]:
        if l.startswith("## BLOCK 1"):
            continue
        if l.startswith("# "):
            l = "##" + l
        elif l.startswith("## ") and not l.startswith("## ["):
            l = "#" + l
        out.append(l.rstrip())
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def rewrite_paths(text: str) -> str:
    """Qualify audit-relative citations so they resolve from the repository root (and are therefore checked by V12)."""
    text = text.replace("docs/audits/.../", AUD_REL + "/")
    text = re.sub(r"`(repro|fixes)/", "`" + AUD_REL + r"/\1/", text)
    return text


def rewrite_audit_id_lookalikes(text: str) -> str:
    """Tokens such as ``P3-43`` (a priority + row number in a WP file) match the walkers' audit-ID regex and would
    have to resolve to an audit document.  Spell them ``P3 #43`` instead; genuine long-form IDs are left alone."""
    return re.sub(r"(?<![A-Z0-9])(P[0-3])-(\d+)(?![\w-])", r"\1 #\2", text)


def assemble(version: str, date: str, intro: str | None) -> tuple[str, list[str]]:
    notes: list[str] = []
    parts = [f"## [{version}] — {date}", ""]
    if intro:
        parts += [intro.rstrip(), ""]
    order = ORDER_547 if str(version).startswith('5.47') else ORDER
    for wp, area in order:
        f = FIX / (f"{wp}_CHANGELOG.md" if wp.startswith("VERIFY_") else f"WP-{wp}_CHANGELOG.md")
        if not f.exists():
            notes.append(f"missing: {f.name} ({area})")
            continue
        body = rewrite_paths(load_wp(f))
        parts += [f"<!-- WP-{wp}: {area} -->", body]
    return "\n".join(parts).rstrip() + "\n\n", notes


def preflight(block: str, notes: list[str]) -> None:
    err = sys.stderr
    print("=== pre-flight ===", file=err)
    for n in notes:
        print("NOTE", n, file=err)
    cited = {m.group("path") for m in FILE_PATH_PATTERN.finditer(block)}
    missing = sorted(p for p in cited if not (REPO / p).exists())
    print(f"V12.1 cited paths: {len(cited)}; unresolved: {len(missing)}", file=err)
    for p in missing:
        print("   MISSING", p, file=err)
    ids = sorted({m.group(1) for m in AUDIT_ID_PATTERN.finditer(block)})
    aud_text = chr(10).join(f.read_text(encoding="utf-8", errors="replace") for f in (REPO / "docs/audits").rglob("*.md"))
    unresolved_ids = [i for i in ids if i not in aud_text]
    print(f"V12.2 audit-ID tokens: {ids}; unresolved under docs/audits: {unresolved_ids}", file=err)
    m = TEST_COUNT_PATTERN.search(block)
    if m:
        ok = int(m["pass"]) + int(m["skip"]) + int(m["xfail"]) == int(m["coll"])
        print(f"V12.3 test-count claim: {m.group(0)!r} arithmetic {'OK' if ok else 'BROKEN'}", file=err)
    else:
        print("V12.3 test-count claim: none", file=err)
    for pat in V17_PATTERNS:
        for mm in pat.finditer(block):
            print(f"V17 claim present (stamp_changelog will re-stamp it): {mm.group(0)!r}", file=err)
    h2 = [l for l in block.splitlines()[1:] if l.startswith("## ")]
    if h2:
        print("H2 headings inside the block (would split it):", h2, file=err)
    print(f"block size: {len(block)} chars, {block.count(chr(10))} lines", file=err)


def gui_block(version: str, date: str) -> str:
    """BLOCK 2 of WP-A9's changelog file (from '## BLOCK 2' up to the next '## ' heading) under a GUI_CHANGELOG heading."""
    lines = (FIX / "WP-A9_CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## BLOCK 2")) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = [l.rstrip() for l in lines[start:end]]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return f"## [{version}] — {date}\n\n" + rewrite_paths("\n".join(body)) + "\n\n"


def write_block(block: str, version: str, target: str = "CHANGELOG.md") -> None:
    cl = REPO / target
    text = cl.read_text(encoding="utf-8")
    existing = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n.*?(?=^## \[|\Z)", re.DOTALL | re.MULTILINE)
    if existing.search(text):
        text = existing.sub(lambda _m: block, text, count=1)
        action = "replaced"
    else:
        first = re.search(r"^## \[", text, re.MULTILINE)
        if not first:
            raise SystemExit("CHANGELOG.md has no '## [' block to insert before")
        text = text[: first.start()] + block + text[first.start():]
        action = "inserted"
    cl.write_text(text, encoding="utf-8", newline="\n")
    print(f"{target}: {action} block [{version}]", file=sys.stderr)


def bump(version: str) -> None:
    for rel, pat in (("pyproject.toml", r'^(version\s*=\s*")(\d+\.\d+\.\d+)(")'),
                     ("lumenairy/__init__.py", r'^(__version__\s*=\s*")(\d+\.\d+\.\d+)(")')):
        p = REPO / rel
        t = p.read_text(encoding="utf-8")
        new, n = re.subn(pat, lambda m: m.group(1) + version + m.group(3), t, count=1, flags=re.MULTILINE)
        if n != 1:
            raise SystemExit(f"{rel}: version line not found")
        p.write_text(new, encoding="utf-8", newline="\n")
        print(f"{rel}: version -> {version}", file=sys.stderr)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--intro", default=None, help="markdown file inserted under the heading")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--write-gui", action="store_true", help="also insert WP-A9 BLOCK 2 into GUI_CHANGELOG.md")
    ap.add_argument("--print-gui", action="store_true", help="print the GUI block to stdout and exit")
    ap.add_argument("--bump", action="store_true")
    a = ap.parse_args()
    if a.print_gui:
        import datetime as _dt
        sys.stdout.write(gui_block(a.version, a.date or _dt.date.today().isoformat()))
        return
    import datetime
    date = a.date or datetime.date.today().isoformat()
    intro = pathlib.Path(a.intro).read_text(encoding="utf-8") if a.intro else None
    if intro and "{{" in intro and a.write:
        raise SystemExit("intro still contains {{placeholders}} -- fill them before --write")
    block, notes = assemble(a.version, date, intro)
    preflight(block, notes)
    if a.write:
        write_block(block, a.version)
        if a.write_gui:
            write_block(gui_block(a.version, date), a.version, target="GUI_CHANGELOG.md")
    else:
        sys.stdout.write(block)
    if a.bump:
        bump(a.version)


if __name__ == "__main__":
    main()
