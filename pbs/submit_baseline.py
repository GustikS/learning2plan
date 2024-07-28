#!/usr/bin/env python3
from itertools import product
import os
import subprocess
import argparse
import json

## paths
# make everything relative to where this script is located
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# assume you have built the container and put it in root directory
CONTAINER = f"{CUR_DIR}/../lrnn_planning.sif"  
assert os.path.exists(CONTAINER), CONTAINER
EXPERIMENTS_DIR = f"{CUR_DIR}/__experiments"
LOG_DIR = f"{EXPERIMENTS_DIR}/baseline_logs"
SAVE_DIR = f"{EXPERIMENTS_DIR}/models"
LOCK_DIR = f"{CUR_DIR}/.lock_test"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
JOB_SCRIPT = f"{CUR_DIR}/pbs_job.sh"
assert os.path.exists(JOB_SCRIPT), JOB_SCRIPT

PARAMETER_FILE = f"{CUR_DIR}/../parameters.json"
assert os.path.exists(PARAMETER_FILE), PARAMETER_FILE
with open(PARAMETER_FILE, "r") as f:
    parameters = json.load(f)
DIMENSIONS = parameters["dimensions"]
LAYERS = parameters["layers"]
REPEATS = parameters["repeats"]
POLICY_SAMPLE = parameters["policy_sample"]

PBS_TEST_NCPU = 2
PBS_TEST_TIMEOUT = "1:00:00"
PBS_TEST_MEMOUT = "8GB"

DOMAINS = [
    # "blocksworld", 
    # "ferry", 
    "satellite", 
    # "transport"
]
PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1]]

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

    for domain, problem, repeat in product(DOMAINS, PROBLEMS, REPEATS):
        description = f"{domain}_{problem}_r{repeat}"

        log_file = f"{LOG_DIR}/{description}.log"
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

        cmd = f"python3 run.py -d {domain} -p {problem} -s {repeat} -c sample -b 10000"

        job_cmd = [
            "qsub",
            "-o",
            log_file,
            "-j",
            "oe",
            "-N",
            "BA" + description,
            "-l",
            f"ncpus={PBS_TEST_NCPU}",
            "-l",
            f"walltime={PBS_TEST_TIMEOUT}",
            "-l",
            f"mem={PBS_TEST_MEMOUT}",
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
