import logging
import subprocess

from tqdm import tqdm


def _log_subprocess_output(pipe, get_plan_length=True):
    plan_length = 0
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        line = line.decode("utf-8").strip()
        logging.info(line)
        if get_plan_length and"plan_length=" in line:
            plan_length = int(line.split("=")[1])
    if get_plan_length:
        return plan_length


# test all "easy" problems
PROBLEMS = []
for i in range(1, 31):
    PROBLEMS.append(f"0_{i:02}")


def test_train_eval_domain(domain, debug=False, problems=5, seed=2024):
    """ Test mainly that nothing breaks, does not really care about performance. """
    lengths = []
    iterator = list(enumerate(PROBLEMS))

    model_file = f"{domain}.model"

    cmd = [
        "python3",
        "run.py",
        "--train",
        "-d",
        domain,
        "--save_file",
        model_file,
        "-l",
        "1",
        "-e",
        "1",
        "-s",
        str(seed),
        "--epochs",
        "20",
    ]
    cmd_str = " ".join(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout:
        _log_subprocess_output(process.stdout)
    rc = process.wait()  # 0 means success
    assert rc == 0, cmd_str

    if debug:
        iterator = tqdm(iterator)
    for i, problem in iterator:
        if i >= problems:
            break
        cmd = [
            "python3",
            "run.py",
            "-s",
            str(seed),
            "-d",
            domain,
            "-p",
            problem,
            "--load_file",
            model_file,
            "-b",
            "1000",
        ]
        cmd_str = " ".join(cmd)
        if not debug:
            logging.critical(cmd_str)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            plan_length = _log_subprocess_output(process.stdout)
        lengths.append(plan_length)
        rc = process.wait()  # 0 means success
        assert rc == 0, cmd_str
    if debug:
        return lengths
