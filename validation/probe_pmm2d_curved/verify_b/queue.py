"""Detached job queue for the verification probes: runs every line of a jobs
file as ``python <script> <args>`` with at most P concurrent processes, BLAS
pinned to one thread, PYTHONPATH pinned to this worktree; a line whose
output JSON already exists is skipped when ``--skip`` names a pattern.

  python queue.py <script.py> <jobs.txt> <P> <logfile>
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
script, jobs, P, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
lines = [ln.split() for ln in open(os.path.join(HERE, jobs)) if ln.strip()]
running = []
with open(os.path.join(HERE, log), "a") as lg:
    lg.write(f"START {time.ctime()} {script} {jobs} P={P} n={len(lines)}\n")
    lg.flush()
    while lines or running:
        while lines and len(running) < P:
            a = lines.pop(0)
            p = subprocess.Popen([sys.executable, script] + a, cwd=HERE,
                                 env=env, stdout=lg, stderr=subprocess.STDOUT)
            running.append((p, a, time.time()))
        time.sleep(2)
        for item in list(running):
            p, a, t0 = item
            if p.poll() is not None:
                lg.write(f"END rc={p.returncode} {time.time() - t0:.0f}s "
                         f"{' '.join(a)}\n")
                lg.flush()
                running.remove(item)
    lg.write(f"ALL_DONE {time.ctime()} {jobs}\n")
