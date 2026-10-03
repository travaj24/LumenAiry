#!/bin/bash
# VERIFY-B: FEM oracle at r = 360 (provenance re-run of one saved mesh) and at two NEW radii.
cd /c/tmp/lum_vcurved_b/validation/probe_pmm2d_curved/verify_b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python fem_circle_r.py --rad 360 --h 1.0 --p 4 --tag h1.0 --threads 6
for R in 480 240; do
  python fem_circle_r.py --rad $R --h 1.0 --hedge 20 --p 4 --tag h1.0_e20 --threads 6
  python fem_circle_r.py --rad $R --h 0.8 --hedge 30 --p 4 --tag h0.8_e30 --threads 6
  python fem_circle_r.py --rad $R --h 1.0 --p 6 --tag h1.0 --threads 6
done
echo FEM_ALL_DONE
