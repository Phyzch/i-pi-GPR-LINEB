#!/bin/bash -f
source /Users/chenghao.zhang/venv/i-pi/bin/activate   # location of i-pi environment.
source /Users/chenghao.zhang/software/i-pi/env.sh     # source i-pi environment script.

python3 -u /Users/chenghao.zhang/software/i-pi/gpr_script/neb_gpr.py input.xml > log.ipi &   # run i-pi

echo "Waiting 30 seconds..."
sleep 30

export NPROCS=4
cd ./nw/
mpirun -np $NPROCS nwchem input.nw > log.nw &

cd ..
