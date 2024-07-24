import logging
import subprocess

from tqdm import tqdm


def _log_subprocess_output(pipe):
    plan_length = 0
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        line = line.decode("utf-8").strip()
        logging.info(line)
        if "plan_length=" in line:
            plan_length = int(line.split("=")[1])
    return plan_length

# test all "easy" problems
PROBLEMS = []
for i in range (1, 31):
    PROBLEMS.append(f"0_{i:02}")

def test_domain(domain, debug=False, problems=1000, seed=2024):
    lengths = []
    iterator = list(enumerate(PROBLEMS))
    if debug:
        iterator = tqdm(iterator)
    for i, problem in iterator:
        if i >= problems:
            break
        cmd = ["python3", "run.py", "-s", str(seed), "-d", domain, "-p", problem, "-c", "sample", "-b", "10000"]
        cmd_str = " ".join(cmd)
        if not debug:
            logging.critical(cmd_str)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            plan_length = _log_subprocess_output(process.stdout)
        lengths.append(plan_length)
        rc = process.wait()  # 0 means success
        assert rc == 0, cmd_str
    return lengths
