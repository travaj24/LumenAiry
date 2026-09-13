#!/usr/bin/env python
"""Build docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/RESOLUTION_STATUS.md.

For every finding row of the consolidated audit report (ID, severity, section) this joins the work-package
report that claims it (status cell of the WP summary table), the independent verifier's verdict (verdict cell of
the VERIFY report), and the commits on the branch whose subject names the package.  Findings no package claims
are listed as GAPS at the end so the orchestrator can rule on them; WP rows naming an ID the audit does not have
are listed as EXTRA.

    python build_resolution_status.py            # dry run: writes RESOLUTION_STATUS.md to the scratchpad only
    python build_resolution_status.py --write    # writes it into the repo's audit directory
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict

REPO = pathlib.Path(r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
AUD_REL = "docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11"
AUD = REPO / AUD_REL
REPORT = REPO / (AUD_REL + ".md")
FIX = AUD / "fixes"
HERE = pathlib.Path(__file__).resolve().parent

ID_RE = re.compile(r"\b([A-Z]{1,2}\d{1,2})[a-z]?\b")
WP_LABEL_RE = re.compile(r"A\d{1,2}[ab]?")
SEV_RE = re.compile(r"\bP[0-3]\b")
STATUS_WORDS = [
    ("regression", "REGRESSION"), ("not fixed", "NOT FIXED"), ("not started", "not started"),
    ("not reproducible", "not reproducible"), ("not done", "deferred"), ("deferred", "deferred"),
    ("partial", "partially fixed"), ("mostly", "mostly fixed"), ("withdrawn", "withdrawn"),
    ("verified", "verified"), ("correct", "verified correct"), ("docs", "fixed (docs)"),
    ("fixed", "fixed"), ("added", "added"), ("done", "fixed"), ("closed", "fixed"), ("resolved", "fixed"),
    ("implemented", "fixed"), ("landed", "fixed"), ("✅", "fixed"), ("✔", "fixed"), ("yes", "fixed"),
]
VERDICT_ORDER = ["REGRESSION", "NOT FIXED", "VERIFIED-WITH-NOTES", "VERIFIED"]


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def ids_in(cell: str) -> list[str]:
    """Finding IDs in a table cell.  Work-package labels (WP-A2, VERIFY-A13, A15b, A10..A19) are stripped first;
    the analysis partition's own findings are A1..A7, so single-digit A-tokens are kept."""
    cell = re.sub(r"(?:WP|VERIFY)-A\d{1,2}[ab]?", " ", cell)
    cell = re.sub(r"A\d{2}[ab]?", " ", cell)
    return [t for t in ID_RE.findall(cell) if not SEV_RE.fullmatch(t)]


def parse_audit() -> "OrderedDict[str, dict]":
    findings: "OrderedDict[str, dict]" = OrderedDict()
    section = ""
    for line in REPORT.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^## (\d+)\. (.*)$", line)
        if m:
            section = f"§{m.group(1)} {m.group(2).strip()}"
            continue
        if not line.startswith("|"):
            continue
        c = cells(line)
        if len(c) < 6:
            continue
        idm = re.fullmatch(r"\*{0,2}([A-Z]{1,2}\d{1,2})\*{0,2}", c[0])
        sev = SEV_RE.search(c[1])
        if not idm or not sev:
            continue
        fid = idm.group(1)
        if fid in findings:
            continue
        findings[fid] = {
            "sev": sev.group(0), "auditor_verified": "✔" in c[1], "section": section,
            "title": re.sub(r"\s+", " ", c[2])[:110],
        }
    return findings


def norm_status(cell: str) -> str:
    """The status word that appears EARLIEST in the cell wins ("fixed (was deferred)" is fixed; "deferred; fixed
    in follow-up" is deferred) -- ties broken by STATUS_WORDS order (so "not fixed" beats "fixed")."""
    low = cell.lower()
    hits = [(low.find(word), i, label) for i, (word, label) in enumerate(STATUS_WORDS) if word in low]
    if not hits:
        return cell.strip("* ")[:30] or "?"
    return min(hits)[2]


# (finding, WP) claims that are parser artefacts -- a WP table row that mentions another package's ID in prose.
DROP_CLAIMS = {("A6", "A5"), ("C1", "A24")}  # (finding, WP): WP-A24 discusses C1 only to refute an attribution; C1 is A6's


def parse_wp_reports() -> dict[str, list[tuple[str, str, str]]]:
    """ID -> [(WP, status, first-cell text)] from every WP-*_REPORT.md summary table."""
    claims: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for f in sorted(FIX.glob("WP-*_REPORT.md")):
        wp = re.match(r"WP-(A\d+[ab]?(?:_SWEEP\d)?)_REPORT", f.name).group(1)
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("|"):
                continue
            c = cells(line)
            if len(c) < 3 or "coordinator" in c[0].lower():
                continue
            fids = ids_in(c[0])
            if not fids or c[0].lower().startswith(("id", "#", "---")):
                continue
            labels = {lab for _, lab in STATUS_WORDS}
            # The status cell is column 2 in most packages, column 3 where column 2 carries the severity or
            # a one-line description (WP-A5, WP-A11).
            status = norm_status(c[1])
            if status not in labels and len(c) > 3 and not SEV_RE.search(c[2]):
                status = norm_status(c[2])
            if status not in labels:
                continue
            for fid in fids:
                claims[fid].append((wp, status, c[0].strip("* ")[:60]))
    return claims


def parse_verify_reports() -> dict[str, list[tuple[str, str]]]:
    verdicts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in sorted(FIX.glob("VERIFY_WP-*.md")):
        wp = re.match(r"VERIFY_WP-(A\d+[ab]?)", f.name).group(1)
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("|"):
                continue
            c = cells(line)
            if len(c) < 2:
                continue
            up = c[1].upper()
            verdict = next((v for v in VERDICT_ORDER if v in up), None)
            if verdict is None:
                continue
            # "NOT FIXED as shipped -> FIXED here" / "REGRESSION found and fixed": the verifier closed it in the
            # same pass, so the finding's final state is verified, with the verifier's fix noted.
            if verdict in ("NOT FIXED", "REGRESSION") and ("FIXED HERE" in up or "AND FIXED" in up
                                                            or "FIXED BELOW" in up or "-> FIXED" in up
                                                            or "→ FIXED" in up):
                verdict = "VERIFIED-WITH-NOTES"
            for fid in ids_in(c[0]):
                verdicts[fid].append((wp, verdict))
    return verdicts


def commits_by_wp() -> dict[str, list[str]]:
    out = subprocess.run(["git", "log", "--format=%h|%s", "audit-fixes-2026-09"], cwd=REPO,
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    by: dict[str, list[str]] = defaultdict(list)
    for line in out.splitlines():
        h, _, subj = line.partition("|")
        for m in re.finditer(r"\b(?:WP-|VERIFY-)?(A\d+[ab]?)\b", subj):
            if re.search(r"\b(WP|VERIFY)-" + re.escape(m.group(1)) + r"\b", subj):
                by[m.group(1)].append(h)
    return by


def best_verdict(vs: list[tuple[str, str]]) -> str:
    if not vs:
        return ""
    return sorted({v for _, v in vs}, key=VERDICT_ORDER.index)[0]


def build() -> tuple[str, dict]:
    findings = parse_audit()
    prefixes = {re.match(r"[A-Z]+", f).group(0) for f in findings}
    raw_claims = parse_wp_reports()
    claims = {k: [c for c in v if (k, c[0]) not in DROP_CLAIMS]
              for k, v in raw_claims.items() if re.match(r"[A-Z]+", k).group(0) in prefixes}
    claims = {k: v for k, v in claims.items() if v}
    verdicts = parse_verify_reports()
    verified_wps = {re.match(r"VERIFY_WP-(A\d+[ab]?)", f.name).group(1) for f in FIX.glob("VERIFY_WP-*.md")}
    commits = commits_by_wp()
    rows, gaps, counts = [], [], defaultdict(int)
    for fid, f in findings.items():
        cl = claims.get(fid, [])
        wps = sorted({wp for wp, _, _ in cl})
        statuses = "; ".join(sorted({f"{wp}: {st}" for wp, st, _ in cl})) if cl else "—"
        verdict = best_verdict(verdicts.get(fid, []))
        if not verdict and any(wp in verified_wps for wp in wps):
            verdict = "see VERIFY_WP-" + "/".join(wp for wp in wps if wp in verified_wps) + ".md"
        shas = sorted({h for wp in wps for h in commits.get(wp, [])})
        if not cl:
            gaps.append(fid)
            counts["unclaimed"] += 1
        else:
            key = "fixed" if all(st.startswith(("fixed", "added", "mostly", "verified")) for _, st, _ in cl) else \
                  "partial/deferred" if any(st.startswith(("partial", "deferred", "not started", "not done")) for _, st, _ in cl) else "other"
            counts[key] += 1
        counts["verifier:" + (verdict or "none")] += 1
        rows.append(f"| {fid} | {f['sev']}{' ✔' if f['auditor_verified'] else ''} | {f['section'].split(' ', 1)[0]} | "
                    f"{', '.join(wps) or '—'} | {statuses} | {verdict or '—'} | {' '.join(shas[:4]) or '—'} |")
    extra = sorted(fid for fid in raw_claims if fid not in findings)
    lines = [
        "# Resolution status of the 2026-09-11 adversarial audit",
        "",
        f"Generated by the orchestrator's `build_resolution_status.py` from the {len(findings)} finding rows of "
        f"`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`, the work-package reports (`fixes/WP-*_REPORT.md`), the "
        f"independent verification reports (`fixes/VERIFY_WP-*.md`) and the commit log of branch `audit-fixes-2026-09`.",
        "",
        "Status is the work package's own summary-table cell; verdict is the strictest cell the verifier gave that ID "
        "(a verifier's NOT FIXED / REGRESSION rows were followed by a ruled follow-up — see the report's Follow-up section).",
        "",
        "## Counts", "",
    ] + [f"- {k}: {v}" for k, v in sorted(counts.items())] + [
        "", "## Findings", "",
        "| ID | Sev | § | WP | WP status | Verifier | Commits |", "|---|---|---|---|---|---|---|",
    ] + rows
    if gaps:
        lines += ["", "## GAPS — audit findings no work-package report claims", ""] + \
                 [f"- {g} ({findings[g]['sev']}, {findings[g]['section']}): {findings[g]['title']}" for g in gaps]
    if extra:
        lines += ["", "## EXTRA — IDs in work-package tables that are not audit finding IDs (check the parser or the report)",
                  "", ", ".join(extra)]
    return "\n".join(lines) + "\n", {"findings": len(findings), "gaps": gaps, "extra": extra, "counts": dict(counts)}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    text, info = build()
    target = (AUD / "RESOLUTION_STATUS.md") if a.write else (HERE / "RESOLUTION_STATUS_dryrun.md")
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}", file=sys.stderr)
    print(f"findings={info['findings']} gaps={len(info['gaps'])} extra={len(info['extra'])}", file=sys.stderr)
    print("counts:", info["counts"], file=sys.stderr)
    if info["gaps"]:
        print("GAPS:", " ".join(info["gaps"]), file=sys.stderr)
    if info["extra"]:
        print("EXTRA:", " ".join(info["extra"]), file=sys.stderr)


if __name__ == "__main__":
    main()
