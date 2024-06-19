import subprocess

DOMAINS = [
    "blocksworld",
    "childsnack",
    "ferry",
    "floortile",
    "miconic",
    "rovers",
    "satellite",
    "sokoban",
    "spanner",
    "transport",
]


def test_domain(domain):
    for problem in ["0_01", "0_30", "1_01"]:
        cmd = ["python3", "run.py", "-d", domain, "-p", problem]
        subprocess.run(cmd, check=True)
