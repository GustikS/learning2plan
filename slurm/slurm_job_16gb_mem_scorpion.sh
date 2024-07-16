#!/bin/bash
#SBATCH --mem=16gb # Job memory request
#SBATCH --time=96:00:00 # Time limit hrs:min:sec

# Show commands
set -x

# set to english
export LC_ALL=C

# log some hardware stats
pwd; hostname; date; echo ""; lscpu; echo ""

# go into desired directory
cd ../datasets/ipc23lt/$DOMAIN

# make tmp dir for trash files
mkdir -p $PROBLEM
cd $PROBLEM
cp ../domain.pddl .
cp ../testing/p$PROBLEM.pddl .

ls

# apptainer command
$CMD

# remove trash dir
cd ..
rm -rf $PROBLEM

date
