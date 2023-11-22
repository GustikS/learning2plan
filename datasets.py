import sys

import numpy as np


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


# %%
class ObjectGraph(LogicDataset):
    logic_dataset: Dataset

    node_features: torch.tensor
    edge_matrix: torch.tensor

    def __init__(self, planning_dataset: PlanningDataset):
        self.logic_dataset = Dataset()
        for state in planning_dataset.states:
            self.logic_dataset.add_example(self.get_object_nodes(state) + self.get_attributed_edges(state))
            self.logic_dataset.add_query(R.get(label_name)[state.label])

    def get_object_nodes(self, state: PlanningState) -> [Relation]:
        relations = []
        feature_vectors = []
        for obj, properties in state.object_properties.items():
            all_properties = self.original.domain.object_types[obj] + properties
            feature_vector = self.predicates_to_features(all_properties, self.original.domain.unary_predicates)
            feature_vector.append(feature_vector)
            relations.append(R.get("features")(obj.name)[feature_vector])

        node_features = torch.tensor(feature_vectors, dtype=torch.float)
        return relations

    def get_attributed_edges(self, state: PlanningState) -> [Relation]:
        relations = []
        types: {[str]: {Predicate}} = {}  # remember constants and all the predicates they satisfy
        edge_index = []

        for atom in state.relations:
            if atom.predicate.arity >= 2:  # split n-ary relations into multiple binary relations
                for const1 in atom.terms:  # connect every constant with every other
                    for const2 in atom.terms:
                        if const1 == const2:
                            continue
                        types.setdefault(tuple([const1, const2]), set()).add(atom.predicate)

        for constants, predicates in types.items():
            feature_vector = self.predicates_to_features(predicates, self.original.domain.nary_predicates)
            relations.append(R.get("edge")(constants)[feature_vector])

            edge_index.append([state.domain.objects.index(constants[0]), state.domain.objects.index(constants[1])])

        return relations

    def get_tensor_data(self, state: PlanningState):

        x = torch.tensor(np.zeros(shape=(len(state.domain.objects), len(self.original.domain.unary_predicates))),
                         dtype=torch.float)

        for obj, properties in state.object_properties.items():
            all_properties = self.original.domain.object_types[obj] + properties
            feature_vector = self.predicates_to_features(all_properties, self.original.domain.unary_predicates)


class AtomGraph(ObjectGraph):

    def __init__(self, planning_dataset: PlanningDataset):
        self.logic_dataset = Dataset()
        for state in planning_dataset.states:
            self.logic_dataset.add_example(self.get_object_nodes(state) + self.get_atom_edges(state))
            self.logic_dataset.add_query(R.get(label_name)[state.label])

    def get_atom_edges(self, state):
        relations = []
        for atom in state.relations:
            feature_vector = self.predicates_to_features(atom.predicate, self.original.domain.nary_predicates)
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
