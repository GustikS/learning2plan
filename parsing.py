import os
import warnings

from planning import *
from encoding import *


def get_dataset(file_path: str):
    with open(file_path) as f:

        all_lines = f.readlines()
        lines = [l.strip() for l in all_lines]

        try:
            types_start = lines.index("BEGIN_TYPES")
            types_end = lines.index("END_TYPES")
            # Process types
            types: [str] = []
            for i in range(types_start + 1, types_end):
                types.append(lines[i].split(" "))
        except:
            warnings.warn("No object types parsed")

        try:
            obj_start = lines.index("BEGIN_OBJECTS")
            obj_end = lines.index("END_OBJECTS")
            # Process objects
            objects: [str] = []
            for i in range(obj_start + 1, obj_end):
                objects.append(lines[i].split(" "))
        except:
            warnings.warn("No objects parsed")

        # some predicates have to be present
        pred_start = lines.index("BEGIN_PREDICATES")
        pred_end = lines.index("END_PREDICATES")
        # Process predicates
        predicates: [str] = []
        for i in range(pred_start + 1, pred_end):
            predicates.append(lines[i].split(" "))

        domain = DomainLanguage(objects, predicates, types)

        try:
            fact_start = lines.index("BEGIN_STATIC_FACTS")
            fact_end = lines.index("END_STATIC_FACTS")
            # Process facts
            facts: [Atom] = []
            for i in range(fact_start + 1, fact_end):
                facts.append(domain.parse_atom(lines[i]))
        except:
            warnings.warn("No static facts parsed")

        # goal has to be present
        goal_start = lines.index("BEGIN_GOAL")
        goal_end = lines.index("END_GOAL")
        # Process goal
        goal: [Atom] = []
        for i in range(goal_start + 1, goal_end):
            goal.append(domain.parse_atom(lines[i]))

        try:
            # actions_start = lines.index("BEGIN_ACTIONS")
            # actions_end = lines.index("END_ACTIONS")
            # Process actions
            actions_starts = [i for i, x in enumerate(lines) if x == "BEGIN_ACTION"]
            actions_ends = [i for i, x in enumerate(lines) if x == "END_ACTION"]
            actions: [Action] = []
            for action_start, action_end in zip(actions_starts, actions_ends):
                action_name = lines[action_start + 1]
                action_parameters = lines[lines.index("BEGIN_PARAMETERS", action_start) + 1: lines.index("END_PARAMETERS",
                                                                                                         action_start)]
                action_preconditions = lines[lines.index("BEGIN_PRECONDITION", action_start) + 1: lines.index(
                    "END_PRECONDITION", action_start)]
                action_add_effects = lines[lines.index("BEGIN_ADD_EFFECT", action_start) + 1: lines.index(
                    "END_ADD_EFFECTN", action_start)]
                action_del_effects = lines[lines.index("BEGIN_DEL_EFFECT", action_start) + 1: lines.index(
                    "END_DEL_EFFECT", action_start)]
                actions.append(
                    Action(action_name, domain, action_parameters, action_preconditions, action_add_effects,
                           action_del_effects))
        except:
            warnings.warn("No actions parsed")

        dataset = PlanningDataset(f.name, domain, facts, actions, goal)

        # states_start = lines.index("BEGIN_STATE_LIST")
        # states_end = lines.index("END_STATE_LIST")
        # Process states
        states_starts = [i for i, x in enumerate(lines) if x == "BEGIN_LABELED_STATE"]
        states_ends = [i for i, x in enumerate(lines) if x == "END_LABELED_STATE"]
        for state_start, state_end in zip(states_starts, states_ends):
            label_line = lines[state_start + 1]
            fact_lines = lines[state_start + 3: state_end - 1]
            dataset.add_state(PlanningState.parse(domain, label_line, fact_lines))

        # Create the dataset
        return dataset


# %%

# Getting multiple datasets from a folder
def get_datasets(folder: str, limit=float("inf"), descending=False):
    all_files = sorted(os.listdir(folder), reverse=descending)
    all_datasets = []

    for i, file in enumerate(all_files):
        dataset = get_dataset(folder + "/" + file)
        all_datasets.append(dataset)
        if i == limit - 1:
            break

    return all_datasets
