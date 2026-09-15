#!/bin/sh
# Task A / C / D -- the oracle-scored ladders, one process per (fixture, build).
# $1 = tree to import lumenairy from, $2 = tag for the JSON names
T="$1"; TAG="$2"
D=/c/tmp/lum_vmb/validation/probe_verify_b7c
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$T"
P_Z="840 856 872 888 904 920 960 1000 1015.00 1015.18 1015.20 1015.24 1015.28 1015.30 1015.34 1015.38 1016"
M_Z="2201.74 2215.22 2228.70 2242.17 2255.65 2269.13 2282.61 2330 2342.24 2342.28 2342.32 2342.36 2343"
S_Z="3188.70 3195.22 3201.74 3208.26 3214.78 3221.30 3227.83 3253.91 3273.95 3273.97 3274.02"
Q_Z="5400 5408 5412 5416 5420 5422 5426 5430 5432 5436 5444 5460 5466 5660 5680 5700"
python "$D/probe1_band.py" P "$D/band_P_$TAG.json" $P_Z > "$D/log_band_P_$TAG.txt" 2>&1 &
python "$D/probe1_band.py" M "$D/band_M_$TAG.json" $M_Z > "$D/log_band_M_$TAG.txt" 2>&1 &
python "$D/probe1_band.py" S "$D/band_S_$TAG.json" $S_Z > "$D/log_band_S_$TAG.txt" 2>&1 &
python "$D/probe1_band.py" Q "$D/band_Q_$TAG.json" $Q_Z > "$D/log_band_Q_$TAG.txt" 2>&1 &
wait
echo "BAND_DONE $TAG"
