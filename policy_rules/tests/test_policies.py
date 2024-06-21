import logging
import subprocess


def _log_subprocess_output(pipe):
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        line = line.decode("utf-8").strip()
        logging.info(line)

# test all "easy" problems
PROBLEMS = []
for i in range (1, 31):
    PROBLEMS.append(f"0_{i:02}")

def test_domain(domain):
    for problem in PROBLEMS:
        cmd = ["python3", "run.py", "-d", domain, "-p", problem, "-b", "1000"]
        cmd_str = " ".join(cmd)
        logging.critical(cmd_str)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            _log_subprocess_output(process.stdout)
        rc = process.wait()  # 0 means success
        assert rc == 0, cmd_str
