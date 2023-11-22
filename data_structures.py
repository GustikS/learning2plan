from abc import abstractmethod, ABC

import torch
from torch_geometric.data import Data
from neuralogic.core import Relation, R

from logic import Atom, Predicate, Object
from planning import PlanningState


def predicates_to_features(predicates: [Predicate], predicate_list: [Predicate]) -> [int]:
    feature_vector = [0] * len(predicate_list)
    for predicate in predicates:
        predicate_index = predicate_list.index(predicate)
        feature_vector[predicate_index] = 1
    return feature_vector


class Sample(ABC):
    state: PlanningState

    object2index: {object: int}

    def __init__(self, state: PlanningState):
        self.state = state
        self.object2index = {}

    @abstractmethod
    def to_relations(self) -> [Relation]:
        pass

    @abstractmethod
    def to_tensors(self) -> Data:
        pass


class Graph(Sample, ABC):
    node_features: {object: [float]}

    edges: [(object, object)]
    edge_features: [[float]]

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.node_features = {}
        self.edges = []
        self.edge_features = []

        self.load_nodes(state)
        self.load_edges(state)

    @abstractmethod
    def load_nodes(self, state: PlanningState, include_types=True):
        pass

    @abstractmethod
    def load_edges(self, state: PlanningState, symmetric_edges=True):
        pass

    def to_relations(self) -> [Relation]:
        relations = []
        for node, features in self.node_features.items():
            relations.append(R.get("node")(node.name)[features])
        for i, (node1, node2) in enumerate(self.edges):
            relations.append(R.get("edge")(node1.name, node2.name)[self.edge_features[i]])
        return relations

    def to_tensors(self) -> Data:
        x = torch.tensor(list(self.node_features.values()), dtype=torch.float)
        edge_index = [(self.object2index[i], self.object2index[j]) for i, j in self.edges]
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).reshape(2, len(edge_index))  # COO format
        edge_attr_tensor = torch.tensor(self.edge_features)

        data_tensor = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=float(self.state.label))

        # data_tensor.validate(raise_on_error=True)
        # data_tensor.has_isolated_nodes()
        # data_tensor.has_self_loops()
        # data_tensor.is_directed()

        return data_tensor

    def node_feature_names(self):
        return self.state.domain.unary_predicates

    def edge_feature_names(self):
        return self.state.domain.nary_predicates


class Multigraph(Graph, ABC):
    edges: [[(int, int)]]


class Bipartite(Graph, ABC):
    left: []
    right: []


class Hypergraph(Graph, ABC):
    # hyperedges are the atoms
    incidence: []


class NestedGraph(Graph, ABC):
    graphs: []


class RawRelational:
    pass


class Object2ObjectGraph(Graph):

    def __init__(self, state: PlanningState):
        super().__init__(state)

    def load_nodes(self, state: PlanningState, include_types=True):
        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i  # storing also indices for the tensor version
            feature_vector = predicates_to_features(properties, self.node_feature_names())
            self.node_features[obj] = feature_vector

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types: {[Object]: {Predicate}} = {}  # remember constants and all the predicates they satisfy

        for atom in state.atoms:
            if atom.predicate.arity >= 2:  # split n-ary relations into multiple binary relations
                for const1 in atom.terms:  # connect every constant with every other
                    for const2 in atom.terms:
                        if const1 == const2:
                            continue
                        edge_types.setdefault(tuple([const1, const2]), set()).add(atom.predicate)
                        if symmetric_edges:
                            edge_types.setdefault(tuple([const2, const1]), set()).add(atom.predicate)

        for constants, predicates in edge_types.items():
            self.edges.append((constants[0], constants[1]))
            feature_vector = predicates_to_features(predicates, self.edge_feature_names())
            self.edge_features.append(feature_vector)

# todo not objects but loading methods of the classes above
# class Object2Object:
#     pass
#
#
# class Objet2Atom:
#     pass
#
#
# class Atom2Atoms:
#     pass
