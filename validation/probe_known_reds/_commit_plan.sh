#!/bin/bash
# The four commits of Wave 5 item D, explicit-path `git add` only.
# Run from /c/tmp/lum_reds.  Each commit is checked for a non-empty staged set
# before it is made; `.test_durations` is json-validated before it is staged.
set -e
cd /c/tmp/lum_reds

CO="Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"

stage() {
  for p in "$@"; do
    if [ -e "$p" ]; then git add -- "$p"; else echo "MISSING: $p" >&2; fi
  done
}

check_staged() {
  if git diff --cached --quiet; then
    echo "NOTHING STAGED for: $1" >&2
    exit 1
  fi
  echo "--- staged for $1:"
  git diff --cached --name-only
}
