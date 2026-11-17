#!/usr/bin/env bash
# Resource watch for a long CI pytest step (the sharded slow lanes in
# unit-tests.yml and publish.yml).  Started in the BACKGROUND by the step that
# runs pytest, killed by that step when pytest returns.
#
#   scripts/ci_resource_watch.sh <pytest_log> <watch_log> [interval_s]
#
# WHY.  A slow shard that exhausts the runner's RAM does not fail: the VM
# thrashes, the runner agent stops answering, the step's own timeout-minutes
# never fires, the job is killed at its limit with every step result missing,
# and the job log is never assembled (BlobNotFound).  Nothing is left to say
# which test was running or how much memory it had.  This script makes that
# case diagnosable and, where it can, turns it into an ordinary failure:
#
# * every <interval_s> (default 60) it prints, to the step's streamed log and
#   to <watch_log>: free -m, the five largest processes by RSS, and the test
#   pytest is currently running (pytest -v writes the node id before it runs
#   the test, so the tail of <pytest_log> names it);
# * every 2 s it reads MemAvailable.  If it stays below
#   LUMEN_CI_MIN_AVAIL_MB (default 600) for three polls in a row, it prints
#   the same report flagged ::error::, then sends SIGABRT to the pytest
#   process.  pytest's faulthandler turns SIGABRT into a stack dump of every
#   thread on stderr (which the step tees into <pytest_log>) before the
#   process dies, so the step FAILS while the runner can still record it and
#   the log-upload step still runs.  A process that close to the end of RAM
#   was about to wedge the runner anyway; this only chooses the failure mode.
set -u
PYLOG=${1:?pytest log path}
OUT=${2:?watch log path}
INTERVAL=${3:-60}
MIN_AVAIL_MB=${LUMEN_CI_MIN_AVAIL_MB:-600}

avail_mb() {
  # free's "available" column is MemAvailable where the kernel reports it
  # and procps' own estimate where it does not
  free -m | awk '/^Mem:/ {print $7; exit}'
}

pytest_pid() {
  # the oldest process whose argv starts with "python -m pytest" (not the
  # ``timeout`` wrapper, whose argv[0] is ``timeout``)
  pgrep -o -f '^python[0-9.]* -m pytest' 2>/dev/null
}

report() {
  {
    echo "---- resource watch $(date -u +%H:%M:%S) ${1:-} ----"
    free -m
    ps -eo pid,ppid,rss,etime,comm --sort=-rss | head -n 6
    if [ -f "$PYLOG" ]; then
      printf 'current test: '
      tail -c 400 "$PYLOG" | tr '\r' '\n' | tail -n 1
      echo
    fi
  } 2>&1 | tee -a "$OUT"
}

low=0
next=0
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$next" ]; then
    report
    next=$((now + INTERVAL))
  fi
  a=$(avail_mb)
  if [ -n "$a" ] && [ "$a" -lt "$MIN_AVAIL_MB" ]; then
    low=$((low + 1))
  else
    low=0
  fi
  if [ "$low" -ge 3 ]; then
    echo "::error::MemAvailable ${a} MB < ${MIN_AVAIL_MB} MB for 6 s: aborting pytest with a stack dump before the runner wedges" | tee -a "$OUT"
    report "LOW MEMORY"
    pid=$(pytest_pid)
    if [ -n "$pid" ]; then
      kill -ABRT "$pid" 2>/dev/null
    fi
    exit 0
  fi
  sleep 2
done
