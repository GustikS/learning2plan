#!/bin/bash
#SBATCH --mem=64gb # Job memory request
#SBATCH --cpus-per-task=16 # Number of CPU cores
#SBATCH --time=96:00:00 # Time limit hrs:min:sec

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
