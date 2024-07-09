# Apptainer instructions

I use apptainer as a "virtual environment" for managing packages on a cluster, rather than as a binary of the actual source code. Build the container by

    apptainer build Apptainer_cpu_environment.sif Apptainer_cpu_environment.def

Run the code using the container e.g. by

    cd policy_rules/
    ../Apptainer_cpu_environment.sif python3 run.py
