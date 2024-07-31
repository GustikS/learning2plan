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
LOG_DIR = f"{EXPERIMENTS_DIR}/test_logs"
SAVE_DIR = f"{EXPERIMENTS_DIR}/models"
LOCK_DIR = f"{CUR_DIR}/.lock_test"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
JOB_SCRIPT = f"{CUR_DIR}/pbs_job.sh"
assert os.path.exists(JOB_SCRIPT), JOB_SCRIPT

PBS_TEST_NCPU = 4
PBS_TEST_TIMEOUT = "1:00:00"
PBS_TEST_MEMOUT = "16GB"
# 4SU per job for 8GB

PARAMETER_FILE = f"{CUR_DIR}/../parameters.json"
assert os.path.exists(PARAMETER_FILE), PARAMETER_FILE
with open(PARAMETER_FILE, "r") as f:
    parameters = json.load(f)
DIMENSIONS = parameters["dimensions"]
LAYERS = parameters["layers"]
REPEATS = parameters["repeats"]
POLICY_SAMPLE = parameters["policy_sample"]

DOMAINS = [
    # "blocksworld", 
    # "ferry", 
    "rover",
    "satellite", 
]
# PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1, 2]]
PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1]]

CONFIGS = product(DIMENSIONS, LAYERS, REPEATS, POLICY_SAMPLE, DOMAINS, PROBLEMS)


""" Main loop """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    submissions = args.submissions
    ## gadi has queue limits
    assert args.submissions <= 1000

    missing_models = set()
    submitted = 0
    skipped_from_log = 0
    skipped_from_lock = 0
    model_missing = 0
    to_go = 0

    for dim, layer, repeat, sample, domain, problem in CONFIGS:
        choice = "sample" if sample else "best"
        description = f"{domain}_{layer}_{dim}_{choice}_{problem}_{repeat}"

        log_file = f"{LOG_DIR}/{description}.log"
        save_file = f"{SAVE_DIR}/{domain}_{layer}_{dim}_{repeat}.model"
        lock_file = f"{LOCK_DIR}/{description}.lock"

        if not os.path.exists(save_file):
            model_missing += 1
            missing_models.add(save_file)
            continue

        if os.path.exists(log_file) and not args.force:
            with open(log_file, "r") as f:
                content = f.read()
            if "Plan generated!" in content:
                skipped_from_log += 1
                continue
            # else:
            #     print(description)

        if os.path.exists(lock_file) and not args.force:
            skipped_from_lock += 1
            continue

        if submitted >= submissions:
            to_go += 1
            continue

        ## empty lock file
        with open(lock_file, "w") as f:
            pass

        cmd = f"python3 run.py -d {domain} -dnc --embedding {dim} --layers {layer} -s {repeat} -b 3000 -c {choice} -p {problem} --load_file {save_file}"

        job_cmd = [
            "qsub",
            "-o",
            log_file,
            "-j",
            "oe",
            "-N",
            "test_" + description,
            "-l",
            f"ncpus={PBS_TEST_NCPU}",
            "-l",
            f"walltime={PBS_TEST_TIMEOUT}",
            "-l",
            f"mem={PBS_TEST_MEMOUT}",
            "-v",
            f"CMD={cmd},LOCK_FILE={lock_file}",
            JOB_SCRIPT,
        ]

        p = subprocess.Popen(job_cmd)
        p.wait()
        print(log_file)
        submitted += 1

    print("Models missing:")
    for m in sorted(missing_models):
        print(m)
    print("Submitted:", submitted)
    print("Skipped from log:", skipped_from_log)
    print("Skipped from lock:", skipped_from_lock)
    print("Skipped from missing model:", model_missing)
    print("To go:", to_go)


if __name__ == "__main__":
    main()
