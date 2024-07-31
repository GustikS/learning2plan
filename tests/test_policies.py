import logging
import subprocess

from tqdm import tqdm


def _log_subprocess_output(pipe):
    plan_length = -1
    cycle_detected = -1
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        line = line.decode("utf-8").strip()
        logging.info(line)
        if "plan_length=" in line:
            plan_length = int(line.split("=")[1])
        if "cycles_detected=" in line and "," not in line:
            cycle_detected = int(line.split("=")[1])
    return plan_length, cycle_detected

def test_domain(domain, debug=False, problems=1000, seed=2024):
    # test "easy" problems
    PROBLEMS = []
    iterator = [10, 20, 30]
    # iterator = range(1, 31)
    for i in iterator:
        PROBLEMS.append(f"0_{i:02}")
        
    lengths = []
    if not isinstance(problems, int):
        PROBLEMS = problems
        problems = 1000
    iterator = list(enumerate(PROBLEMS))
    if debug:
        iterator = tqdm(iterator)
    for i, problem in iterator:
        if i >= problems:
            break
        cmd = ["python3", "run.py", "-s", str(seed), "-d", domain, "-p", problem, "-b", "1000"]
        cmd_str = " ".join(cmd)
        if not debug:
            logging.critical(cmd_str)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            plan_length, cycle_detected = _log_subprocess_output(process.stdout)
        lengths.append(plan_length)
        assert cycle_detected == 0, cmd_str
        rc = process.wait()  # 0 means success
        assert rc == 0, cmd_str
    if debug:
        return lengths
