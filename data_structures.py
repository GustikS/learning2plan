from abc import abstractmethod, ABC

import torch
from torch_geometric.data import Data, HeteroData
from neuralogic.core import Relation, R

from logic import Atom, Predicate, Object
from planning import PlanningState


def one_hot(index, length) -> [int]:
    vector = [0] * length
    vector[index] = 1
    return vector


def multi_hot(predicates: [Predicate], predicate_list: [Predicate]) -> [int]:
    feature_vector = [0] * len(predicate_list)
    for predicate in predicates:
        predicate_index = predicate_list.index(predicate)
        feature_vector[predicate_index] = 1
    return feature_vector


# class BipartiteData(Data):
#     def __init__(self, x_left, x_right, edge_index, edge_attr, y):
#         super(BipartiteData, self).__init__(None, edge_index, edge_attr, y)
#         self.x_left = x_left
#         self.x_right = x_right


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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        for atom in self.state.atoms:
            strings.append(atom.predicate.name + "(" + ",".join([term.name for term in atom.terms]) + ")")
        return "\n" + ", ".join(strings)


class Graph(Sample, ABC):
    node_features: {object: [float]}

    edges: [(object, object)]  # note that duplicate edges are allowed here!
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


class Bipartite(Graph, ABC):
    node_features_left: {object: [float]}
    node_features_right: {object: [float]}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.node_features_left = {}
        self.node_features_right = {}

    def to_tensors(self) -> Data:
        data: Data = super().to_tensors()
        x_left = torch.tensor(list(self.node_features_left.values()), dtype=torch.float)
        x_right = torch.tensor(list(self.node_features_right.values()), dtype=torch.float)
        data.x = (x_left, x_right)
        return data


class Hetero(Graph, ABC):
    relation_edges: {Predicate: (Object, Object)}
    relation_edge_features: {Predicate: [float]}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.relation_edges = {}
        self.relation_edge_features = {}


class Multi(Graph, ABC):
    """This is natively implemented by allowing duplicate/parallel edges in the Graph class"""
    pass
    # edges: [[(int, int)]]


class Hypergraph(Graph, ABC):
    # hyperedges are the atoms
    incidence: []
#
#
# class NestedGraph(Graph, ABC):
#     graphs: []
#
#
# class RawRelational:
#     pass


class Object2ObjectGraph(Graph):
    """"Object-object graph with edges corresponding to relations"""

    def load_nodes(self, state: PlanningState, include_types=True):
        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i  # storing also indices for the tensor version
            feature_vector = multi_hot(properties, self.object_feature_names())
            self.node_features[obj] = feature_vector

    # todo nullary predicates are missing here, how to include them?
    def load_edges(self, state: PlanningState, symmetric_edges=True):
        """Collecting all relation types into one multi-hot edge feature vector"""
        edge_types = self.get_edge_types(state, symmetric_edges)

        for constants, predicates in edge_types.items():
            self.edges.append((constants[0], constants[1]))
            feature_vector = multi_hot(predicates, self.relation_feature_names())
            self.edge_features.append(feature_vector)

    def get_edge_types(self, state, symmetric_edges):
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
        return edge_types


class Object2ObjectMultiGraph(Object2ObjectGraph, Multi):
    """Same as Object2ObjectGraph but each relation is a separate one-hot edge instead of one multi-hot"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for constants, predicates in edge_types.items():
            for predicate in predicates:
                self.edges.append((constants[0], constants[1]))
                feature_vector = one_hot(predicate, self.relation_feature_names())
                self.edge_features.append(feature_vector)


class Object2ObjectHeteroGraph(Object2ObjectGraph, Hetero):
    """Same as Object2ObjectGraph but each relation is a separate edge type with separate learning parameters"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for constants, predicates in edge_types.items():
            for predicate in predicates:
                self.relation_edges[predicate] = (constants[0], constants[1])
                self.edge_features.append(1.0)  # no edge features here - each relation is a separate dimension already


class Object2AtomGraph(Graph):
    """Object-atom simple graph pretending nodes are of the same type (no object/atom features)"""

    scalar_edge_types = True

    def load_nodes(self, state: PlanningState, include_types=True):

        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i
            self.node_features[obj] = 1  # no node features here

        offset = i
        for i, atom in enumerate(state.atoms):
            self.object2index[atom] = offset + i
            self.node_features[atom] = 1  # no node features here

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


class Object2AtomBipartiteGraph(Object2AtomGraph, Bipartite):
    """Object-atom bipartite graph, i.e. with 2 explicitly different types of nodes with different features"""

    def load_nodes(self, state: PlanningState, include_types=True):

        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.object2index[obj] = i  # storing also indices for the tensor version

            feature_vector = multi_hot(properties, self.object_feature_names())
            self.node_features_left[obj] = feature_vector

        i = 0
        for atom in state.atoms:
            if atom.predicate.arity == 1:
                continue  # we skip the unary atoms here as they are the object features
            self.object2index[atom] = i
            i += 1

            feature_vector = multi_hot(atom.predicate, self.relation_feature_names(include_nullary=True))
            self.node_features_right[atom] = feature_vector


class Object2AtomMultiGraph(Object2AtomBipartiteGraph, Multi):
    # todo done in Object2AtomGraph already, the multi-hot version is missing
    pass


class Object2AtomHeteroGraph(Object2AtomBipartiteGraph, Hetero):
    """Same as Object2ObjectGraph but each relation is a separate edge type with separate learning parameters"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for i, atom in enumerate(state.atoms):
            for j, term in enumerate(atom.terms):
                index_predicate = Predicate("X" + str(j), 0, [], -1)
                self.relation_edges[index_predicate]((term, atom))
                self.relation_edge_features[
                    index_predicate] = 1.0  # no edge features here - each relation is a separate dimension already
                if symmetric_edges:
                    index_predicate = Predicate("Y" + str(j), 0, [], -2)
                    self.relation_edges[index_predicate]((atom, term))  # reverse argument
                    self.relation_edge_features[index_predicate] = 1.0

    def to_tensors(self) -> Data:
        # todo
        pass


class Atom2AtomGraph(Graph):
    """Atom-atom graph with edges being their shared objects, and edge features the object ids"""
    term2atoms: {Object: [Atom]}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.term2atoms = {}

    def load_nodes(self, state: PlanningState, include_types=True):
        for i, atom in enumerate(state.atoms):
            self.object2index[atom] = i
            feature_vector = multi_hot(atom.predicate, self.relation_feature_names(include_nullary=True,
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


class Atom2AtomMultiGraph(Atom2AtomGraph, Multi):
    # todo done in Atom2AtomGraph already, the multi-hot version is missing
    pass


class Atom2AtomHeteroGraph(Atom2AtomGraph, Hetero):
    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for atom1 in state.atoms:
            for term in atom1.terms:
                for atom2 in self.term2atoms[term]:
                    self.relation_edges[term](atom1, atom2)
                    self.relation_edge_features[term] = 1
                    if symmetric_edges:
                        self.relation_edges[term](atom2, atom1)
                        self.relation_edge_features[term] = 1
