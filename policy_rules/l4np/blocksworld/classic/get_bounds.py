import os

testing_dir = "testing"
plan_lengths = []

for f in sorted(os.listdir(testing_dir)):
    # seems to time out after this
    if f == "p2_01.pddl":
        break

    problem_f = os.path.join(testing_dir, f)

    cmd = f"./blocks-world-generator-and-planner/solve_optimally.py {problem_f}"
    plan_length = int(os.popen(cmd).read().split("=")[1])
    print(f"{f=} {plan_length=}")
    plan_lengths.append(plan_length)

with open("blocksworld_opt_plan_lengths.txt", "w") as f:
    f.write(" ".join(map(str, plan_lengths)))
