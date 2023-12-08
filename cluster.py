from os import listdir


def create_script(dataset):
    script = "#!/bin/bash \n#SBATCH --partition=cpufast\n#SBATCH --mem=24g \n\n"
    script += "ml PyTorch-Geometric/2.4.0-foss-2022a-CUDA-11.7.0 \n"
    script += "cd /home/souregus/planning/ \n"
    # script += "pip install neuralogic \n"
    script += "python ./code/experiments.py " + dataset
    return script


def create_scripts(source_folder):
    datasets = sorted(listdir(source_folder))
    script_names = []
    for dataset in datasets:
        script = create_script(source_folder + "/" + dataset)
        script_name = dataset.split("/")[-1] + ".sh"
        with open("./scripts/" + script_name, 'w', newline='\n') as f:
            f.write(script)
        script_names.append(script_name)
    with open("./scripts/_batch.sh", 'w', newline='\n') as f:
        for script_name in script_names:
            f.write("sbatch " + script_name + "\n")


def load_results(path, result_name="_merged_results.csv"):
    datasets = sorted(listdir(path))
    merged_lines = []
    for dataset in datasets:
        with open(path + "/" + dataset) as f:
            lines = f.readlines()
            if merged_lines:
                merged_lines.extend(lines[1:])
            else:
                merged_lines.extend(lines)

    with open(path + "/" + result_name, 'w') as f:
        f.writelines(merged_lines)


# create_scripts("./datasets/rosta")
load_results("Y:/planning/results")
