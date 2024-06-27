import argparse
import pickle

import neuralogic
import torch

if not neuralogic.is_initialized():
    neuralogic.initialize(jar_path="../jar/NeuraLogic.jar", debug_mode=False)  # custom momentary backend upgrades

from neuralogic.core import R
from neuralogic.dataset import Dataset

from modelling.planning import State
from modelling.samples import flatten_states, parse_domain
from modelling.templates import build_template
from modelling.training import prepare_model, store_template, train


def load_model(save_file, model=None):
    """This serialization will be done more correctly as a function in the next release of neuralogic..."""

    with open(save_file + "_weights", 'rb') as f:
        weights = torch.load(f)

    if model is None:
        with open(save_file + "_template", 'rb') as f:
            template = pickle.load(f)
        model = build_template(template)

    model.load_state_dict(weights)
    return model


def get_init_state(domain_name):
    """get some adhoc init state from the jsons, without the labeled queries"""
    problems, predicates, actions = parse_domain(domain_name, problem_limit=1)
    for states, goal_state in problems.values():
        for state in states:
            break
        break
    init_state = State(state)
    init_state.setup_ILG(goal_state)
    return init_state, actions

def prepare_action_queries(actions):
    """turn action headers into lifted queries"""
    action_queries = [R.get(action.name)(action.parameters) for action in actions]
    actions = {action.name: action for action in actions}
    return action_queries, actions

def test_model(domain_name, model_file, model=None):
    """Test a stored/trained model/template on a given domain"""
    init_state, actions = get_init_state(domain_name)
    print(f'init_state: {init_state.atoms}')
    action_queries, actions = prepare_action_queries(actions)

    if model is None:
        if model_file:
            model = load_model(model_file)
        else:
            model, template = train(domain_name, numeric=False, save_file="./target/tmp", plotting=True)
    model.test()

    sorted_actions, indexed_state = score_applicable_actions(action_queries, init_state, model)
    print(f'applicable: {sorted_actions}')

    best_action = sorted_actions[0]
    split = best_action[0].split("(")
    action_name = split[0]
    action_terms = split[1][:-1]
    action = actions[action_name]
    print(f'selecting: {best_action}')
    ground_action = action.ground(action_terms)
    next_state = ground_action.successor(init_state.backend())
    print(f'next_state: {next_state}')

def score_applicable_actions(action_queries, init_state, model):
    """The core step where the model get evaluated, and we check the values of the queries corresponding to actions"""
    # TODO(DZC): action_queries: list[str] of lifted actions, init_state: list[str] with prefixes
    dataset = Dataset()
    dataset.add_example(init_state.make_relations())
    dataset.add_queries(action_queries)
    ground_samples = model.ground(dataset)
    indexed_state = State.from_backend(ground_samples[0])
    bd = model.build_dataset(ground_samples)  # todo skip postprocessing here for speedup

    scored_actions = [(str(sample.java_sample.query.neuron.name), model(sample)) for sample in bd]
    return sorted(scored_actions, key=lambda item: item[1], reverse=True), indexed_state


def _inner_atom_values(sample, action_queries):
    """this can be used to query values of the actions even if they are not the outputs, e.g. in the regression setting"""
    for q in action_queries:
        atoms = sample.get_atom(q)
        if atoms:
            for a in atoms:
                print(a.substitutions)
                print(a.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, default="blocksworld", choices=["satellite", "blocksworld"])
    parser.add_argument("--save_file", type=str, default='./target/stored_model')
    args = parser.parse_args()
    domain_name = args.domain
    saved_file = args.save_file + f'_{domain_name}'
    print(f"{domain_name=}")
    print(f"{saved_file=}")

    test_model(domain_name, saved_file)
