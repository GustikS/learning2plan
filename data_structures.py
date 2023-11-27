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


def one_hot(index, length) -> [int]:
    vector = [0] * length
    vector[index] = 1
    return vector


class BipartiteData(Data):
    def __init__(self, x_left, x_right, edge_index, edge_attr, y):
        super(BipartiteData, self).__init__(None, edge_index, edge_attr, y)
        self.x_left = x_left
        self.x_right = x_right


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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        for atom in self.state.atoms:
            strings.append(atom.predicate.name + "(" + ",".join([term.name for term in atom.terms]) + ")")
        return "\n" + ", ".join(strings)


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
            if len(self.edge_features[i]) == 1:
                feats = self.edge_features[i][0]
            else:
                feats = self.edge_features[i]
            relations.append(R.get("edge")(node1.name, node2.name)[feats])
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

    def object_feature_names(self):
        return self.state.domain.unary_predicates

    def relation_feature_names(self, include_nullary=False, include_unary=False):
        if include_nullary:
            if include_unary:
                return self.state.domain.predicates
            else:
                return self.state.domain.nary_predicates + self.state.domain.nullary_predicates
        else:
            return self.state.domain.nary_predicates


class Multigraph(Graph, ABC):
    edges: [[(int, int)]]


class Bipartite(Graph, ABC):
    node_features_left: {}
    node_features_right: {}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.node_features_left = {}
        self.node_features_right = {}

    def to_tensors(self) -> Data:
        x_left = torch.tensor(list(self.node_features_left.values()), dtype=torch.float)
        x_right = torch.tensor(list(self.node_features_right.values()), dtype=torch.float)
        edge_index = [(self.object2index[i], self.object2index[j]) for i, j in self.edges]
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).reshape(2, len(edge_index))  # COO format
        edge_attr_tensor = torch.tensor(self.edge_features)

        data_tensor = BipartiteData(x_left=x_left, x_right=x_right, edge_index=edge_index_tensor,
                                    edge_attr=edge_attr_tensor, y=float(self.state.label))
        return data_tensor


class Hypergraph(Graph, ABC):
    # hyperedges are the atoms
    incidence: []


class NestedGraph(Graph, ABC):
    graphs: []


class RawRelational:
    pass


class Object2ObjectGraph(Graph):

    def load_nodes(self, state: PlanningState, include_types=True):
        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i  # storing also indices for the tensor version
            feature_vector = predicates_to_features(properties, self.object_feature_names())
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
            feature_vector = predicates_to_features(predicates, self.relation_feature_names())
            self.edge_features.append(feature_vector)


class Object2AtomGraph(Graph):
    """Object-atom simple graph pretending nodes are of the same type (no object/atom features)"""

    scalar_edge_types = True

    def load_nodes(self, state: PlanningState, include_types=True):

        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i

        offset = i
        for i, atom in enumerate(state.atoms):
            self.object2index[atom] = offset + i

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for i, atom in enumerate(state.atoms):
            for j, term in enumerate(atom.terms):
                self.edges.append((term, atom))
                self.edge_features.append(self.encode_edge_type(j, state.domain.max_arity))
                if symmetric_edges:
                    self.edges.append((atom, term))
                    self.edge_features.append(self.encode_edge_type(j, state.domain.max_arity))

    def encode_edge_type(self, index, max_index):
        """Encoding the position of object in an atom - either scalar or one-hot"""
        if self.scalar_edge_types:
            self.edge_features.append(index)
        else:
            self.edge_features.append(one_hot(index, max_index))


class Object2AtomBipartite(Object2AtomGraph, Bipartite):
    """Object-atom bipartite graph, i.e. with 2 explicitly different types of nodes with different features"""

    def load_nodes(self, state: PlanningState, include_types=True):

        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i  # storing also indices for the tensor version

            feature_vector = predicates_to_features(properties, self.object_feature_names())
            self.node_features_left[obj] = feature_vector

        i = 0
        for atom in state.atoms:
            if atom.predicate.arity == 1:
                continue  # we skip the unary atoms here as they are the object features
            self.object2index[atom] = i
            i += 1

            feature_vector = predicates_to_features(atom.predicate, self.relation_feature_names(include_nullary=True))
            self.node_features_right[atom] = feature_vector


class Atom2AtomGraph(Graph):
    """Atom-atom graph with edges being their shared objects, and edge features the object ids"""
    term2atoms: {Object: [Atom]}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.term2atoms = {}

    def load_nodes(self, state: PlanningState, include_types=True):
        for i, atom in enumerate(state.atoms):
            self.object2index[atom] = i
            feature_vector = predicates_to_features(atom.predicate, self.relation_feature_names(include_nullary=True,
                                                                                                include_unary=True))
            self.node_features[atom] = feature_vector

            for term in atom.terms:
                self.term2atoms.setdefault(term, []).append(atom)

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for i, atom1 in enumerate(state.atoms):
            for term in atom1.terms:
                for atom2 in self.term2atoms[term]:
                    self.edges.append((atom1, atom2))
                    # edge feature will be the object id as one-hot
                    self.edge_features.append(one_hot(state.domain.objects.index(term), len(state.domain.objects)))
                    if symmetric_edges:
                        self.edges.append((atom2, atom2))
                        self.edge_features.append(one_hot(state.domain.objects.index(term), len(state.domain.objects)))
