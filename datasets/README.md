Go into this directory, and run the script to get state space data as jsons. They will NOT be added to github because they may be too large, hence the .gitignore file.

    cd datasets/
    ./to_jsons.py

The json files containing the optimal actions for all states in the state spaces of small problems should then be located in 

    datasets/jsons/<DOMAIN>/classic/state_space_data.json
