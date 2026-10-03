#!/bin/bash
# runmut.sh <mutant-id> [extra test file] : the Phase C tests in a mutant tree
d=/c/tmp/vcc_mut/$1
cd $d || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/vcc_mut/$1
python -c "import lumenairy,os; f=os.path.abspath(lumenairy.__file__).lower().replace(os.sep,'/'); assert 'vcc_mut/$1/' in f, f; print('lumenairy from', f)" > pytest.log 2>&1 || exit 2
timeout 5400 python -m pytest tests/unit/test_pmm2d_staggered_curved_c.py $2 --capture=sys -p no:randomly -p no:cacheprovider -n 3 -q -rfE >> pytest.log 2>&1
echo "$1: $(tail -n 1 pytest.log)"
