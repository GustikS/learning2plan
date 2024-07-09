#!/bin/bash
#SBATCH --mem=8gb # Job memory request
#SBATCH --time=48:00:00 # Time limit hrs:min:sec

# Show commands
set -x

# set to english
export LC_ALL=C

# log some hardware stats
pwd; hostname; date; echo ""; lscpu; echo ""

# go into root directory with run.py
cd ..

$CMD

date
