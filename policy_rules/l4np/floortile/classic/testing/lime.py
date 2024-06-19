import os

for diff, pre in [("easy", 0), ("medium", 1), ("hard", 2)]:
    for f in sorted(os.listdir(diff)):
        file = f"{diff}/{f}"
        new_f = f"p{pre}_{f[1:3]}.pddl"
        os.system(f"mv {file} {new_f}")