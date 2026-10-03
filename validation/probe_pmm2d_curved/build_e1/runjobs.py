"""Run the probe commands listed in a jobs file (one per line, arguments to
``python``) with at most N at a time, BLAS pinned to one thread, from this
directory; each job's stdout / stderr goes to logs/<n>.log.

usage: python runjobs.py <jobs.txt> <N>
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [ln.strip() for ln in open(os.path.join(HERE, sys.argv[1]))
        if ln.strip() and not ln.startswith("#")]
N = int(sys.argv[2])
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1",
           PYTHONPATH=os.path.normpath(os.path.join(HERE, "..", "..", "..")))
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
tag = os.path.splitext(os.path.basename(sys.argv[1]))[0]
running = []
k = 0
t0 = time.time()
while jobs or running:
    while jobs and len(running) < N:
        cmd = jobs.pop(0)
        k += 1
        log = open(os.path.join(HERE, "logs", f"{tag}_{k:03d}.log"), "w")
        log.write(cmd + "\n")
        log.flush()
        p = subprocess.Popen([sys.executable] + cmd.split(), cwd=HERE,
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((p, cmd, log))
    time.sleep(2)
    for item in list(running):
        p, cmd, log = item
        if p.poll() is not None:
            log.close()
            running.remove(item)
            print(f"[{time.time() - t0:7.0f}s] rc={p.returncode} {cmd}",
                  flush=True)
print("ALL DONE", flush=True)
