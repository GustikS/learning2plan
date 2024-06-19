# learning-4-numeric-planning-dataset

Numeric extensions of [ipc2023-learning-dataset](https://github.com/DillonZChen/goose-dataset).

## Repository structure
The repo consists of PDDL files for 8 planning domains. 
```
blocksworld childsnack ferry miconic rovers satellite spanner transport
```

Each domain has a `numeric` and `classic` version. The `classic` versions do not have any functions. Each `<domain>/<version>` directory contains the following files and subdirectories:

- `domain.pddl` contains the domain information in PDDL format
- `training/` contains small PDDL problems from which training data is generated
- `training_plans/` contains optimal plans for a subset of problems in `training/`
- `testing/` contains large evaluation PDDL problems
- `data.json` consists of nicely formatted information about domains, and the training data without labels. Since these are pretty big, they must be extracted with 
```
unzip jsons.zip
```

### json objects
See [json_readme.md](json_readme.md)
