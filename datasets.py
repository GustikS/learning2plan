import sys
import os

from collections import namedtuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# %%

Object = namedtuple("Object", "name, type")
Predicate = namedtuple("Predicate", "name, arity, types")
Atom = namedtuple("Atom", "predicate, terms")


class LogicLanguage:
    objects: [Object]
    predicates: [Predicate]

    types: [str]
    supertypes: {str: str}  # type -> supertype

    def __init__(self, objects: [str], predicates: [str], types: [str] = []):
        self.types = [obj_type[1] for obj_type in types]
        self.supertypes = {obj_type[1]: self.types[int(obj_type[0])] for i, obj_type in enumerate(types)}

        self.objects = [Object(obj_name, self.types[int(obj_type)]) for obj_type, obj_name in objects]

        self.predicates = []
        for pred in predicates:
            pred_types = [self.types[int(arg_type)] for arg_type in pred[:-1]]
            pred_name = pred[-1]
            predicate = Predicate(pred_name, len(pred_types), tuple(pred_types))
            self.predicates.append(predicate)

    def parse_atom(self, int_line: str) -> Atom:
        ints = [int(i) for i in int_line.split(" ")]
        predicate = self.predicates[ints[0]]
        constants = [self.objects[i] for i in ints[1:]]
        atom = Atom(predicate, constants)
        return atom


class DomainLanguage(LogicLanguage):
    arities: {int: [Predicate]}

    proposition_types: [Predicate]  # zero arity predicates
    object_types: [Predicate]  # unary relations and actual types
    relation_types: [Predicate]  # all other relations with arity >=2 will be treated as relation types

    types_for_object: {Object: [Predicate]}  # concrete object types and supertypes

    def __init__(self, objects: [str], predicates: [str], types: [str] = []):
        super().__init__(objects, predicates, types)

        self.arities = {}
        for predicate in self.predicates:
            self.arities.setdefault(predicate.arity, []).append(predicate)

        self.propositions = self.arities[0]
        self.object_types = self.arities[1]
        type_predicates = [Predicate(obj_type, 1, -1) for obj_type in self.types]
        self.object_types.extend(type_predicates)

        self.relation_types = []
        for arity, predicate in self.arities.items():
            if arity <= 1: continue
            self.relation_types.append(predicate)

        for obj in self.objects:
            obj_types = []
            self.recursive_types(obj.type, obj_types)
            self.types_for_object[obj] = obj_types

    def recursive_types(self, obj_type, types):
        types.append(obj_type)
        if obj_type == "object":
            return
        else:
            self.recursive_types(self.supertypes[obj_type], types)


# %%

class PlanningState:
    domain: DomainLanguage

    label: int

    atoms: [Atom]  # all atoms
    propositions: [Atom]  # zero arity atoms
    relations: [Atom]  # >=2 arity atoms

    object_properties: {Object: [Predicate]}  # unary atoms

    def __init__(self, domain: DomainLanguage, atoms: [Atom], label: int = -1):
        self.domain = domain
        self.label = label
        self.atoms = atoms

        self.propositions = []
        self.relations = []
        self.object_properties = {}

        self.update(atoms)

    def update(self, atoms: [Atom]):
        for atom in atoms:
            if atom.predicate.arity == 0:
                self.propositions.append(atom)
            elif atom.predicate.arity == 1:
                self.object_properties.setdefault(atom.terms[0], []).append(atom.predicate)
            elif atom.predicate.arity >= 2:
                self.relations.append(atom)

    @staticmethod
    def parse(domain: DomainLanguage, label_line: str, facts_lines: [str]):
        label = int(label_line)
        facts: [Atom] = []
        for fact_line in facts_lines:
            fact = domain.parse_atom(fact_line)
            facts.append(fact)
        state = PlanningState(domain, facts, label)
        return state


# %%

class Action:
    name: str

    domain: DomainLanguage

    parameter_types: [str]  # term types

    preconditions: [Atom]
    add_effects: [Atom]
    delete_effects: [Atom]

    def __init__(self, name: str, domain: DomainLanguage, parameters: [str], preconditions: [str], add_effects: [str],
                 delete_effects: [str]):
        self.name = name
        self.domain = domain
        self.parameter_types = [self.domain.types[int(par.split(" ")[1])] for par in parameters]
        self.preconditions = [self.parse_atom(precondition) for precondition in preconditions]
        self.add_effects = [self.parse_atom(add_effect) for add_effect in add_effects]
        self.delete_effects = [self.parse_atom(delete_effect) for delete_effect in delete_effects]

    def parse_atom(self, int_line: str) -> Atom:
        ints = [int(i) for i in int_line.split(" ")]
        predicate = self.domain.predicates[ints[0]]
        arguments = ["X" + str(arg) for arg in ints[1:]]  # arguments are just variable indices
        atom = Atom(predicate, arguments)
        return atom


class PlanningDataset:
    name: str

    domain: DomainLanguage

    static_facts: [Atom]
    actions: [Action]
    goal: [Atom]

    states: [PlanningState]

    def __init__(self, name, domain: DomainLanguage, static_facts: [Atom], actions: [Action], goal: [Atom],
                 states: [PlanningState]):
        self.name = name
        self.domain = domain

        self.static_facts = static_facts
        self.actions = actions
        self.goal = goal

        self.states = states

    def enrich_states(self):
        for state in self.states:
            state.update(self.static_facts)
            # todo add also actions and goal somehow?


# %%

def get_dataset(file_path: str):
    with open(file_path) as f:

        all_lines = f.readlines()
        lines = [l.strip() for l in all_lines]

        types_start = lines.index("BEGIN_TYPES")
        types_end = lines.index("END_TYPES")

        actions_start = lines.index("BEGIN_ACTIONS")
        actions_end = lines.index("END_ACTIONS")

        obj_start = lines.index("BEGIN_OBJECTS")
        obj_end = lines.index("END_OBJECTS")

        pred_start = lines.index("BEGIN_PREDICATES")
        pred_end = lines.index("END_PREDICATES")

        fact_start = lines.index("BEGIN_STATIC_FACTS")
        fact_end = lines.index("END_STATIC_FACTS")

        goal_start = lines.index("BEGIN_GOAL")
        goal_end = lines.index("END_GOAL")

        states_start = lines.index("BEGIN_STATE_LIST")
        states_end = lines.index("END_STATE_LIST")

        # Process types
        types: [str] = []
        for i in range(types_start + 1, types_end):
            types.append(lines[i].split(" ")[1:])

        # Process objects
        objects: [str] = []
        for i in range(obj_start + 1, obj_end):
            objects.append(lines[i].split(" ")[1:])

        # Process predicates
        predicates: [str] = []
        for i in range(pred_start + 1, pred_end):
            predicates.append(lines[i].split(" ")[1:])

        domain = DomainLanguage(objects, predicates, types)

        # Process facts
        facts: [Atom] = []
        for i in range(fact_start + 1, fact_end):
            facts.append(domain.parse_atom(lines[i]))

        # Process goal
        goal: [Atom] = []
        for i in range(goal_start + 1, goal_end):
            goal.append(domain.parse_atom(lines[i]))

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

        # Process states
        states_starts = [i for i, x in enumerate(lines) if x == "BEGIN_LABELED_STATE"]
        states_ends = [i for i, x in enumerate(lines) if x == "END_LABELED_STATE"]
        states: [PlanningState] = []
        for state_start, state_end in zip(states_starts, states_ends):
            label_line = lines[state_start + 1]
            fact_lines = lines[state_start + 3: state_end - 1]
            states.append(PlanningState.parse(domain, label_line, fact_lines))

        # Create the dataset
        dataset = PlanningDataset(f.name, domain, facts, actions, goal, states)
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


# %%

# folder = "C:/Users/gusta/Downloads/planning/geffner/data/data/supervised/optimal/train/blocks-clear/blocks-clear"
folder = "C:/Users/gusta/Downloads/planning/rosta/blocks"

datasets = get_datasets(folder, limit=1)  # let's just get the first/smallest dataset for now
dataset = datasets[0]

# %%

from neuralogic.core.settings import Settings
from neuralogic.dataset import Dataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import CrossEntropy, MSE
from neuralogic.core import Template, Relation, R, Var, V, C, Rule
from neuralogic.core import Transformation, Aggregation, Combination, Metadata
from neuralogic.optim import Adam
from neuralogic.inference.inference_engine import InferenceEngine

# lets also turn on the backend logging for more detailed info
from neuralogic.logging import add_handler, clear_handlers, Formatter, Level

add_handler(sys.stdout, Level.FINE, Formatter.COLOR)

# from IPython.display import display

# %%

label_name = 'distance'
generic_name = 'relation'  # some generic name for n-ary relations


class LogicDataset:
    """Just keep the data as they are"""
    original: PlanningDataset

    object_types: [Relation]
    static_facts: [Relation]
    goal: [Relation]
    actions: [(str, [Rule])]

    states: [(int, [Rule])]

    supertypes: [Rule]

    def __init__(self, planning_dataset: PlanningDataset):
        """Creating a dataset this way (relation by relation) might be slow, but it's more instructive.
            For big datasets use FileDataset in neuralogic"""
        self.original = planning_dataset

        self.object_types = [R.get(obj.type)(obj.name) for obj in planning_dataset.domain.objects]
        self.static_facts = [R.get(atom.predicate.name)([term.name for term in atom.terms]) for atom in
                             planning_dataset.static_facts]
        self.goal = [R.get(atom.predicate.name)([term.name for term in atom.terms]) for atom in planning_dataset.goal]
        self.actions = []
        for action in planning_dataset.actions:
            rules = []
            body = [R.get(atom.predicate.name)(atom.terms) for atom in action.preconditions]
            for i, typename in enumerate(action.parameter_types):
                body.append(R.get(typename)("X" + str(i)))
            for add in action.add_effects:
                rules.append(R.get("add_" + add.predicate.name)(add.terms) <= body)
            for delete in action.delete_effects:
                rules.append(R.get("del_" + delete.predicate.name)(delete.terms) <= body)
            self.actions.append((action.name, rules))

        self.states = []
        for state in planning_dataset.states:
            relations = [R.get(atom.predicate.name)([term.name for term in atom.terms]) for atom in state.atoms]
            self.states.append((state.label, relations))

        self.supertypes = []
        for sub, super in enumerate(planning_dataset.domain.supertypes):
            self.supertypes.append(
                R.get(planning_dataset.domain.types[super])(V.X) <= R.get(planning_dataset.domain.types[sub])(V.X))

    def state_samples(self) -> [Dataset]:
        """Just the states ignoring everything else"""
        logic_dataset = Dataset()
        for sample in self.states:
            logic_dataset.add_example(sample[1])
            logic_dataset.add_query(R.get(label_name)[sample[0]])
        return logic_dataset

    def predicates_to_features(self, predicates: [Predicate], predicate_list: [Predicate]) -> [int]:
        feature_vector = [0] * len(predicate_list)
        for predicate in predicates:
            predicate_index = predicate_list.index(predicate)
            feature_vector[predicate_index] = 1
        return feature_vector


# %%

import torch
from torch_geometric.data import Data

#%%
class ObjectGraph(LogicDataset):
    logic_dataset: Dataset

    tensor_dataset: Data

    def __init__(self, planning_dataset: PlanningDataset):
        self.logic_dataset = Dataset()
        for state in planning_dataset.states:
            self.logic_dataset.add_example(self.get_object_nodes(state) + self.get_attributed_edges(state))
            self.logic_dataset.add_query(R.get(label_name)[state.label])

    def get_object_nodes(self, state: PlanningState) -> [Relation]:
        relations = []
        for obj, properties in state.object_properties.items():
            all_properties = self.original.domain.types_for_object[obj] + properties
            feature_vector = self.predicates_to_features(all_properties, self.original.domain.object_types)
            relations.append(R.get("features")(obj.name)[feature_vector])
        return relations

    def get_attributed_edges(self, state: PlanningState) -> [Relation]:
        relations = []
        types: {[str]: {Predicate}} = {}  # remember constants and all the predicates they satisfy

        for atom in state.relations:
            if atom.predicate.arity >= 2:  # split n-ary relations into multiple binary relations
                for const1 in atom.terms:  # connect every constant with every other
                    for const2 in atom.terms:
                        if const1 == const2:
                            continue
                        types.setdefault(tuple([const1, const2]), set()).add(atom.predicate)

        for constants, predicates in types.items():
            feature_vector = self.predicates_to_features(predicates, self.original.domain.relation_types)
            relations.append(R.get("edge")(constants)[feature_vector])


class AtomGraph(ObjectGraph):

    def __init__(self, planning_dataset: PlanningDataset):
        self.logic_dataset = Dataset()
        for state in planning_dataset.states:
            self.logic_dataset.add_example(self.get_object_nodes(state) + self.get_atom_edges(state))
            self.logic_dataset.add_query(R.get(label_name)[state.label])

    def get_atom_edges(self, state):
        relations = []
        for atom in state.relations:
            feature_vector = self.predicates_to_features(atom.predicate, self.original.domain.relation_types)
            joint_object = "-".join(atom.terms)
            relations.append(R.get("atom")(joint_object)[feature_vector])  # atom nodes
            for i, obj in enumerate(atom.terms):
                feature_vector = [0] * len(atom.terms)
                feature_vector[i] = 1  # remember also the term position
                relations.append(R.get("edge")(joint_object, obj.name)[feature_vector])  # atom-object edges


class HyperGraph(LogicDataset):
    pass


# %%
logic_dataset = LogicDataset(dataset)
logic_dataset

# %%

label_name = 'distance'
generic_name = 'relation'  # some generic name for n-ary relations


class LRNN:
    template_name: str

    dataset: PlanningDataset

    rules: []

    def __init__(self, dataset: PlanningDataset, rules=[]):
        self.dataset = dataset
        self.rules = rules

    @abstractmethod
    def get_rules(self) -> [Rule]:
        print("you need to define some rules that make up the model")
        return []

    def get_relations(self, atoms: [Atom], predicates_to_features=True, generic_name=generic_name) -> [Relation]:
        relations = []
        for atom in atoms:
            relations.append(R.get(atom.predicate.name)(atom.terms))  # just add the fact as it is

        # optionally, also encode the predicate types as a multi-hot feature vector over some generic relation within each arity
        if predicates_to_features:
            types: {[str]: [Predicate]} = {}  # remember constants and all the predicates they satisfy
            for atom in atoms:
                types.setdefault(tuple(atom.terms), []).append(atom.predicate)

            for constants, predicates in types.items():
                feature_vector = self.predicates_to_features(predicates, self.dataset.domain.arities[len(constants)])
                relations.append(R.get(generic_name + str(len(constants)))(constants)[feature_vector])

        return relations

    def predicates_to_features(self, predicates: [Predicate], predicate_list: [Predicate]) -> [int]:
        feature_vector = [0] * len(predicate_list)
        for predicate in predicates:
            predicate_index = predicate_list.index(predicate)
            feature_vector[predicate_index] = 1
        return feature_vector

    def get_dataset(self, predicates_to_features=True, generic_name=generic_name) -> Dataset:
        """Creating a dataset this way (relation by relation) might be slow, but it's more instructive.
            For big datasets use FileDataset in neuralogic"""

        logic_dataset = Dataset()
        for sample in self.dataset.states:
            atoms = self.get_relations(sample.atoms, predicates_to_features, generic_name)
            logic_dataset.add_example(atoms)
            logic_dataset.add_query(R.get(label_name)[sample.label])
        return logic_dataset

    def get_template(self) -> Template:
        template = Template()
        template += self.get_relations(self.dataset.static_facts)  # add static facts straight into the template/model
        template += self.get_rules()  # finally add the logic of the rules/model itself, which is up to the user

        template += R.get(label_name) / 0 | [
            Transformation.IDENTITY]  # we will want to end up with Identity (or Relu) as this is a distance regression task
        return template


# %%

class GraphModel(LRNN):
    object_types: [Relation]  # unary relations = "object types"
    relation_types: [Relation]  # all other relations with arity >=2 will be treated as relation types

    dim: int

    add_random: int = 0  # to be explained later in the notebook
    constants_random: {str: [float]} = {}

    def __init__(self, dataset, dim=3):
        super().__init__(dataset)
        self.dim = dim
        self.object_types = self.dataset.domain.arities[1]
        self.relation_types = [p for p in self.dataset.domain.predicates if p not in self.object_types]

    # overrides
    def get_relations(self, atoms: [Atom], predicates_to_features=True, generic_name=generic_name):
        relations = []
        types: {[str]: {Predicate}} = {}  # remember constants and all the predicates they satisfy

        for atom in atoms:
            if atom.predicate.arity > 2:  # split n-ary relations into multiple binary relations
                for const1 in atom.terms:  # connect every constant with every other
                    for const2 in atom.terms:
                        if const1 == const2:
                            continue
                        types.setdefault(tuple([const1, const2]), set()).add(atom.predicate)
            else:
                types.setdefault(tuple([term.name for term in atom.terms]), set()).add(atom.predicate)

        # todo add explicit types

        for constants, predicates in types.items():
            if len(constants) >= 2:
                predicate_list = self.relation_types
            elif len(constants) == 1:
                predicate_list = self.object_types
            elif len(constants) == 0:
                predicate_list = self.dataset.domain.arities[0]
            feature_vector = self.predicates_to_features(predicates, predicate_list)
            if self.constants_random and len(constants) == 1:
                feature_vector.extend(self.constants_random[constants[0]])
            relations.append(R.get(generic_name + str(len(constants)))(constants)[feature_vector])

        return relations

    # overrides
    def get_rules(self, generic_name=generic_name):
        rules = []

        # A classic message passing over the edges (preprocessed binary relations)
        rules.append(
            R.embedding(V.X)[self.dim, self.dim] <= (
                R.get(generic_name + str(2))(V.Y, V.X)[self.dim, len(self.relation_types)],
                R.get(generic_name + str(1))(V.Y)[self.dim, len(self.object_types) + self.add_random]))
        # Global pooling/readout
        rules.append(R.get(label_name)[1, self.dim] <= R.embedding(V.X))

        # Aggregate also the zero-order predicate(s)
        rules.append(
            R.get(label_name)[1, len(self.dataset.domain.arities[0])] <= R.get(generic_name + str(0)))

        # ...and the unary predicate(s) on their own
        rules.append(
            R.get(label_name)[1,] <= R.get(generic_name + str(1))(V.X)[1, len(self.object_types) + self.add_random])

        return rules

    # to be explained later
    def generate_random_constants(self, add_random=1) -> None:
        self.add_random = add_random
        for constant in self.dataset.domain.constants:
            self.constants_random[constant] = np.random.normal(size=add_random)


# %%

gnn = GraphModel(dataset)
template = gnn.get_template()
print(template)
template.draw()

settings = Settings(optimizer=Adam(lr=0.01), epochs=500, error_function=MSE(), chain_pruning=True,
                    iso_value_compression=False,  # turn off compression to get more interpretable drawings
                    rule_transformation=Transformation.IDENTITY, relation_transformation=Transformation.IDENTITY,
                    # these are just default activation fcn choices, IDENTITY = no hidden activation
                    )
evaluator = get_evaluator(template, settings)

model = template.build(settings)
model.reset_parameters()
model.draw()
model.draw()

graph_data = gnn.get_dataset()
built_dataset = evaluator.build_dataset(graph_data)

# %%

built_dataset[0].draw()

# todo add exhaustive unary X binary combinations
# add ternary and up with context
