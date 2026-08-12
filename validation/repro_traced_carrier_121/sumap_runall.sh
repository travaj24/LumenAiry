#!/bin/sh
# SUM-AT-APERTURE probe -- the arm-B run set, serially (each run peaks at
# ~57 GB on the 'full' leg, so they must NOT overlap).
#   1. the 3-order sum, both leg variants          -- the headline
#   2. the single-order null controls              -- crosstalk + linearity
#   3. a pitch ablation on the null control        -- is dx_c converged?
set -x
cd "$(dirname "$0")"
P="python -W ignore::UserWarning sumap_probe_121.py armb"
$P --orders "0,0;-2,0;-4,-2" --leg both
$P --orders "0,0"            --leg crop
$P --orders "-2,0"           --leg both
$P --orders "-4,-2"          --leg both
DXC=1.2292e-6 NC=8192 $P --orders "0,0" --leg full --suffix _dxc1229
