#!/usr/bin/env python3
from itertools import product
import os
import subprocess
import argparse

## paths
# assume we are in slurm/ directory
CUR_DIR = os.getcwd()  
# assume you have built the container and put it in root directory
CONTAINER = f"{CUR_DIR}/../lrnn_planning.sif"  
assert os.path.exists(CONTAINER), CONTAINER
EXPERIMENTS_DIR = f"{CUR_DIR}/__experiments"
LOG_DIR = f"{EXPERIMENTS_DIR}/train_logs"
SAVE_DIR = f"{EXPERIMENTS_DIR}/models"
LOCK_DIR = f"{CUR_DIR}/.lock_train"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
JOB_SCRIPT = f"{CUR_DIR}/pbs_train.sh"
assert os.path.exists(JOB_SCRIPT), JOB_SCRIPT

PBS_TRAIN_NCPU = 2
PBS_TRAIN_TIMEOUT = "01:00:00"
PBS_TRAIN_MEMOUT = "8GB"

DIMENSIONS = [1, 2, 4, 8, 16, 32, 64]
LAYERS = [1, 2, 3, 4]
REPEATS = [0, 1, 2]
DOMAINS = [
    # "blocksworld", 
    "ferry", 
    # "miconic", 
    # "satellite", 
    "transport",
]

CONFIGS = product(DIMENSIONS, LAYERS, REPEATS, DOMAINS)


""" Main loop """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    submissions = args.submissions
    ## gadi has queue limits
    assert args.submissions <= 1000

    submitted = 0
    skipped = 0
    to_go = 0

    for dim, layer, repeat, domain in CONFIGS:
        description = f"{domain}_{layer}_{dim}_{repeat}"

        log_file = f"{LOG_DIR}/{description}.log"
        save_file = f"{SAVE_DIR}/{description}.model"
        lock_file = f"{LOCK_DIR}/{description}.lock"

        if (os.path.exists(log_file) or os.path.exists(lock_file)) and not args.force:
            skipped += 1
            # print(log_file)
            continue

        if submitted >= submissions:
            to_go += 1
            continue

        ## empty lock file
        with open(lock_file, "w") as f:
            pass

        cmd = f"python3 run.py -d {domain} --embedding {dim} --layers {layer} -s {repeat} --save_file {save_file}"

        job_cmd = [
            "qsub",
            "-o",
            log_file,
            "-j",
            "oe",
            "-N",
            "train_" + description,
            "-l",
            f"ncpus={PBS_TRAIN_NCPU}",
            "-l",
            f"walltime={PBS_TRAIN_TIMEOUT}",
            "-l",
            f"mem={PBS_TRAIN_MEMOUT}",
            "-v",
            f"CMD={cmd}",
            JOB_SCRIPT,
        ]

        p = subprocess.Popen(job_cmd)
        p.wait()
        print(log_file)
        submitted += 1

    print("Submitted:", submitted)
    print("Skipped:", skipped)
    print("To go:", to_go)


if __name__ == "__main__":
    main()
