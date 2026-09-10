#!/bin/sh
# Assemble results/RUNS.md -- the pytest and lint transcripts, which cannot be
# committed as *.log (.gitignore:43).
D=/c/tmp/lum_vslant/validation/probe_verify_slant_anchor/results
{
echo "# Run transcripts"
echo
echo "Assembled by \`make_runs_md.sh\` from the raw logs (\`*.log\` is gitignored)."
echo
echo '## The 22 named files, Windows'
echo
echo '`tests/unit/test_pmm*.py tests/unit/test_fix_pmm*.py tests/unit/test_verify_pmm*.py tests/unit/test_rcwa*.py`,'
echo '`OMP` = `OPENBLAS` = `MKL` = 1, `-p no:randomly`:'
echo
echo '```'
tail -20 "$D/suite_win.log"
echo '```'
echo
echo '## The fixs gate, both builds'
echo
echo '```'
echo '--- WIN'
tail -10 "$D/gate_win.log"
echo '--- WSL'
tail -10 "$D/gate_wsl.log"
echo '```'
echo
echo '## This verifications gate, both builds'
echo
echo '```'
cat "$D/verify_gate_win.txt" 2>/dev/null
cat "$D/verify_gate_wsl.txt" 2>/dev/null
echo '```'
echo
echo '## ruff (WSL)'
echo
echo '```'
cat "$D/ruff_wsl.txt" 2>/dev/null
echo '```'
} > "$D/RUNS.md"
wc -l "$D/RUNS.md"
