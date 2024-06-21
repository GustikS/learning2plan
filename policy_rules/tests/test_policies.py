import logging
import subprocess


def _log_subprocess_output(pipe):
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        line = line.decode("utf-8").strip()
        logging.info(line)


def test_domain(domain):
    for problem in ["0_01", "0_30", "1_01"]:
        cmd = ["python3", "run.py", "-d", domain, "-p", problem]
        cmd_str = " ".join(cmd)
        logging.critical(cmd_str)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            _log_subprocess_output(process.stdout)
        rc = process.wait()  # 0 means success
        assert rc == 0, cmd_str
