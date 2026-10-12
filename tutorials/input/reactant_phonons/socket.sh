#!/bin/bash -f

# source /home/niri/sources/nwchem/build_noether.sh

export NWCHEM_BASIS_LIBRARY="/Users/zhan602/software/nwchem/src/basis/libraries/"
export NWCHEM_NWPW_LIBRARY="/Users/zhan602/software/nwchem/src/nwpw/libraryps/"
export NWCHEM_BIN="/Users/zhan602/software/nwchem/bin/MACX64"
export NPROCS=4

source /Users/zhan602/venv/i-pi/bin/activate   # location of i-pi environment.
source /Users/zhan602/software/i-pi/env.sh     # source i-pi environment script.


python3 -u /Users/zhan602/software/i-pi/bin/i-pi input.xml > log.ipi &   # run i-pi

echo "Waiting 10 seconds..."
sleep 10

#srun --mpi=pmi2 -n $SLURM_NPROCS -K1 $NWCHEM_BIN/nwchem input.nw
mpirun -np $NPROCS $NWCHEM_BIN/nwchem input.nw > log.nw
