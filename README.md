Use python 3.10 (exactly) for the pymimir (`pymdzcf`) dependency.

After installing the `./requirements.txt` dependencies, you should be able to `python3 run.py --help` from the root dir of the repo

All the core functionality of the workflow is exposed to the arguments of the run script.

## Benchmark notes

No hyphens allowed in predicates or action schemata in PDDL and plan files! I have either replaced them with underscores or removed them entirely for some domains but have not checked this thoroughly. The reason for this is to ensure consistency in the code and to minimise bugs.

## Workflow instructions

### Dataset
Firstly, you need to create the training datasets from pymimir by running `datasets/to_jsons.py` 
which exports them to respective domain JSON files `datasets/jsons/DOMAIN` . 
This preprocessing action is separate and not part of the main `run.py` workflow, as it takes quite some time.

    cd datasets; python3 to_jsons.py; cd ..

The dataset folder also contains `datasets/lrnn/*` subdirs that get automatically extracted out of 
these json files as part of the main workflow, depending on the requested format settings, particularly:
 - `--state_regression` for ADDING the current state distance target as a regression label
 - `--action_regression` for switching between classification/regression of the action targets

The resulting lrnn training samples will get exported into
 - `--limit` state samples from the JSON file

split into `*/examples.txt` and `*/queries.txt`, linked together via example ids (rows), in a human-readable format 
so that you can check them even manually.

### Domains with handcrafted policies
The domains with background knowledge (BK) in forms of handcrafted policies so far are 
- `blocksworld`
- `ferry`
- `miconic`
- `satellite`
- `transport`

### Training
A similar workflow follows for the **templates**, which represent the core logic of the whole policy. They are either created w.r.t. 
the default logic provided programmatically for each domain in `policy_rules/policy/handcraft`, or they can be loaded from drive with
 - `--load_file`

If not found, it will be created following the programmatic logic (w.r.t. the current setup), and possibly exported to
- `--save_file` location in both serialized model with weights (*.model) and readable (`*_template.txt`) manner

There are two main part to each policy template - the handcrafted domain knowledge and the generic learning/modelling construct(s).
The handcrafted logic (from `policy_rules/policy/handcraft`) can optionally be skipped with
- `--no-knowledge`

The learning part is conceptualized as a very generic modelling construct surrounding the policy logic with learnable parameters, 
embeddings, and basic message-passing between the incorporated objects. On this most generic level, its whole logic 
is compressed to
- `--embedding` dimensionality (global) parameter of the model
- `--layers` count for the number of (GNN-like) message passing steps (1=embedding only)

The training occurs if either 
- `--train` is specified, or a `save_file` from above is specified

The training workflow then consists of
 1. building the given training samples with the given template, yielding dynamically structured neural networks
    - one per each target label (action), but sharing neurons across the same state
 2. training the (shared) parameters of these networks against the data labels
    - for the given `--epochs` number of steps

### Evaluating policies
There are two types of policies that can be evaluated
1. handcrafted policies
2. learned LRNN policies (either with or without BK from handcrafted policies)

The arguments are
 - `--problem` - the problem from the given domain to test the current policy on
 - `--bound` - the number of actions/steps taken greedily through the state space
 - `--verbose` - levels of verbosity from [0-6] to inspect the different parts of the workflow right from the console

### Examples
#### Run just Ferry handcrafted policy

    python3 run.py -d ferry -p 0_30

#### Train Ferry but do not save

    python3 run.py -d ferry --embedding 8 --layers 2 --train

#### Train and save Ferry policy

    python3 run.py -d ferry --embedding 8 --layers 2 --save_file ferry.model

#### Load and run Ferry policy

    python3 run.py -d ferry --embedding 8 --layers 2 --load_file ferry.model -p 0_30

#### Save visualisation of Ferry template to file
We use a low embedding to prevent a lot of numbers being seen

    python3 run.py -d ferry --embedding 3 --layers 2 --visualise ferry_template.png

## Apptainer instructions

I use apptainer as a "virtual environment" for managing packages on a cluster, rather than as a binary of the actual source code. Build the container by

    apptainer build Apptainer_cpu_environment.sif Apptainer_cpu_environment.def

Run the code using the container e.g. by

    cd policy_rules/
    ../Apptainer_cpu_environment.sif python3 run.py
