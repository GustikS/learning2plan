# Lifted Relational Neural Networks for Planning

- [Lifted Relational Neural Networks for Planning](#lifted-relational-neural-networks-for-planning)
  - [Setup](#setup)
    - [Virtual environment](#virtual-environment)
    - [Apptainer environment](#apptainer-environment)
  - [Code Usage](#code-usage)
    - [Run BK Policy](#run-bk-policy)
    - [Train and Save an LRNN Policy](#train-and-save-an-lrnn-policy)
    - [Load and Run an LRNN Policy](#load-and-run-an-lrnn-policy)
  - [Benchmark notes](#benchmark-notes)
  - [Profiling Instructions](#profiling-instructions)
  - [Code Details](#code-details)
    - [Dataset](#dataset)
    - [Domains with handcrafted policies](#domains-with-handcrafted-policies)
    - [Training](#training)
    - [Evaluating policies](#evaluating-policies)

## Setup
### Virtual environment
Use Python 3.10 (exactly) for the pymimir (`pymdzcf`) dependency.

After installing the `./requirements.txt` dependencies, you should be able to `python3 run.py --help` from the root dir of the repo

All the core functionality of the workflow is exposed to the arguments of the run script.

Setup example using conda:

    conda create --name lrnnplan python=3.10
    conda activate lrnnplan
    pip install -r requirements.txt

### Apptainer environment
Alternatively, we can use an apptainer environment. Build the environment with 

    apptainer build lrnnplan.sif Apptainer_env.def

and run the driver script by 

    apptainer run lrnnplan.sif python3 run.py <arguments>

## Code Usage
Call `python3 run.py -h` for up to date usage. Some examples of usage are as follows.

### Run BK Policy

    python3 run.py -d satellite -p 0_30

### Train and Save an LRNN Policy

    python3 run.py -d satellite --embedding 8 --layers 2 --save_file satellite.model

### Load and Run an LRNN Policy

    python3 run.py -d satellite --embedding 8 --layers 2 --load_file satellite.model -p 0_30

## Benchmark notes

Hyphens are NOT allowed in predicates or action schemata in PDDL files! e.g. `turn_to` is ok but not `turn-to`. I have either replaced them with underscores or removed them entirely for some domains but have not checked this thoroughly. The reason for this is to ensure consistency in the code and to minimise bugs when parsing.

## Profiling Instructions
Run the following

    python3 -m cProfile -o out.profile run.py <arguments>
    snakeviz out.profile

## Code Details

### Dataset
The dataset is automatically generated in the when training the model. The below is the old instructions and information.

---

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

---

### Domains with handcrafted policies
The domains with background knowledge (BK) in forms of handcrafted policies so far are 
- `blocksworld`
- `ferry`
- `satellite`
- `rover`

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
 - `--verbosity` - levels of verbosity to inspect the different parts of the workflow right from the console
