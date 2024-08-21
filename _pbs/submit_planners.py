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
CONTAINER = f"{CUR_DIR}/../scorpion.sif"  
assert os.path.exists(CONTAINER), CONTAINER
EXPERIMENTS_DIR = f"{CUR_DIR}/__experiments"
LOG_DIR = f"{EXPERIMENTS_DIR}/planner_logs"
LOCK_DIR = f"{CUR_DIR}/.lock_test"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
PLANNERS = [
    "lama",
    "scorpion",
]
JOB_SCRIPT = {planner: f"{CUR_DIR}/{planner}_job.sh" for planner in PLANNERS}
for planner, job_script in JOB_SCRIPT.items():
    assert os.path.exists(job_script), job_script

PBS_TEST_NCPU = 1
PBS_TEST_TIMEOUT = "1:00:00"
PBS_TEST_MEMOUT = "8GB"

DOMAINS = [
    # "blocksworld", 
    # "ferry", 
    "rover",
    # "satellite", 
    # "transport"
]
# PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1, 2]]
PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1]]

CONFIGS = product(PLANNERS, DOMAINS, PROBLEMS)


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

    for planner, domain, problem in CONFIGS:
        description = f"{domain}_{problem}_{planner}"

        log_file = f"{LOG_DIR}/{description}.log"
        lock_file = f"{LOCK_DIR}/{description}.lock"

        if os.path.exists(lock_file) and not args.force:
            skipped += 1
            continue

        if os.path.exists(log_file) and not args.force:
            with open(log_file, "r") as f:
                content = f.read()
            if "Solution found!" in content:
                skipped += 1
                continue
            else:
                # print(log_file)
                pass
                # continue
            # skipped += 1
            # print(log_file)
            # continue

        if submitted >= submissions:
            to_go += 1
            continue

        domain_file = f"datasets/pddl/{domain}/domain.pddl"
        problem_file = f"datasets/pddl/{domain}/testing/p{problem}.pddl"

        ## empty lock file
        with open(lock_file, "w") as f:
            pass

        job_cmd = [
            "qsub",
            "-o",
            log_file,
            "-j",
            "oe",
            "-N",
            "PL_" + description,
            "-l",
            f"ncpus={PBS_TEST_NCPU}",
            "-l",
            f"walltime={PBS_TEST_TIMEOUT}",
            "-l",
            f"mem={PBS_TEST_MEMOUT}",
            "-v",
            f"DOMAIN_FILE={domain_file},PROBLEM_FILE={problem_file},DESC={description}",
            JOB_SCRIPT[planner],
        ]

        p = subprocess.Popen(job_cmd)
        p.wait()
        print(log_file)
        submitted += 1

    print("Submitted:", submitted)
    print("Skipped from log or lock:", skipped)
    print("To go:", to_go)


if __name__ == "__main__":
    main()
