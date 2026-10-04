#!/bin/bash
cd /mnt/c/tmp/lum_symgrad_verify2/validation/probe_jax_symgrad/verify_r2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/lumvenv/bin/python
$PY -c "import sys;sys.path.insert(0,'/mnt/c/tmp/lum_symgrad_verify2');import lumenairy;assert lumenairy.__file__.startswith('/mnt/c/tmp/lum_symgrad_verify2'),lumenairy.__file__;print('import ok',lumenairy.__file__)"
$PY p1_rayleigh_anchor.py 1e-2 1e-3 3e-4 1e-4 2>&1 | grep "^{" | sed 's/.scales.*//'
$PY p1b_anchor_attrib.py 1e-3 1e-4 2>&1 | grep -v Warn | grep "^eff1d\|^jones"
$PY p3_mechanism.py 2>&1 | grep "^[a-j] " | cut -c1-400
$PY p5_switch.py 2>&1 | grep "^leak\|^env\|^jit\|^vmap\|^count" | cut -c1-500
LUMROOT=/mnt/c/tmp/lum_symgrad_verify2_base $PY p5_switch.py 2>&1 | grep "^leak"
$PY p9_uniform.py post 2>&1 | grep "^{" | cut -c1-200
LUMROOT=/mnt/c/tmp/lum_symgrad_verify2_base $PY p9_uniform.py pre 2>&1 | grep "^{" | cut -c1-200
$PY p8_controls.py post 2>&1 | grep -v Warn | tail -4
LUMROOT=/mnt/c/tmp/lum_symgrad_verify2_base $PY p8_controls.py pre 2>&1 | grep -v Warn | tail -4
