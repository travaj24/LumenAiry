#!/bin/bash
# one worker: pops the first line of queue.txt (mkdir lock) and runs it, until the queue is empty
cd /c/tmp/lum_vcurved_e2/validation/probe_pmm2d_curved/verify_e2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e2
while true; do
  until mkdir q.lock 2>/dev/null; do sleep 1; done
  job=$(head -n 1 queue.txt)
  tail -n +2 queue.txt > queue.tmp && mv queue.tmp queue.txt
  rmdir q.lock
  [ -z "$job" ] && break
  n=$(echo "$job" | tr ' ' '_')
  t0=$(date +%s)
  timeout 1780 python $job > logs/$n.log 2>&1
  echo "done rc=$? $(( $(date +%s) - t0 ))s $job" >> logs/runq.out
done
