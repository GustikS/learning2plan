from abc import abstractmethod, ABC
from typing import Union

import torch
from torch_geometric.data import Data, HeteroData
from neuralogic.core import Relation, R

from logic import Atom, Predicate, Object
from planning import PlanningState


def one_hot(index, length) -> [float]:
    vector = [0.0] * length
    vector[index] = 1.0
    return vector


def multi_hot_index(ints, length) -> [float]:
    vector = [0.0] * length
    for i in ints:
        vector[i] = 1.0
    return vector


def multi_hot_object(predicates: [Predicate], predicate_list: [Predicate]) -> [float]:
    feature_vector = [0.0] * len(predicate_list)
    for predicate in predicates:
        predicate_index = predicate_list.index(predicate)
        feature_vector[predicate_index] = 1.0
    return feature_vector


class Sample(ABC):
    state: PlanningState

    node2index: {Object | Atom: int}

    def __init__(self, state: PlanningState):
        self.state = state
        self.node2index = {}

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
        return ", ".join(strings)


class Graph(Sample, ABC):
    node_features: {Object | Atom: [float]}

    edges: [(Object | Atom, Object | Atom)]  # note that duplicate edges are allowed here!
    edge_features: [[float]]

    edge_type_format: str

    def __init__(self, state: PlanningState, edge_type_format="index"):
        super().__init__(state)
        self.node_features = {}
        self.edges = []
        self.edge_features = []
        self.edge_type_format = edge_type_format

    def load_state(self, state: PlanningState):
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
        edge_index = [(self.node2index[i], self.node2index[j]) for i, j in self.edges]
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        edge_attr_tensor = torch.tensor(self.edge_features)

        data_tensor = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=float(self.state.label))

        # data_tensor.validate(raise_on_error=True)
        # data_tensor.has_isolated_nodes()
        # data_tensor.has_self_loops()
        # data_tensor.is_directed()

        return data_tensor


class Bipartite(Graph, ABC):
    graph_source: Graph  # source -> target
    graph_target: Graph  # target -> source

    def __init__(self, state: PlanningState):
        super().__init__(state)

    def to_tensors(self) -> Data:
        x_source = torch.tensor(list(self.graph_source.node_features.values()), dtype=torch.float)
        x_target = torch.tensor(list(self.graph_target.node_features.values()), dtype=torch.float)

        edges_s2t = [(self.graph_source.node2index[i], self.graph_target.node2index[j]) for i, j in
                     self.graph_source.edges]
        edges_s2t_tensor = torch.tensor(edges_s2t, dtype=torch.long).transpose(0, 1)
        edge_s2t_attr_tensor = torch.tensor(self.graph_source.edge_features)

        edges_t2s = [(self.graph_target.node2index[i], self.graph_source.node2index[j]) for i, j in
                     self.graph_target.edges]
        edges_t2s_tensor = torch.tensor(edges_t2s, dtype=torch.long).transpose(0, 1)
        edges_t2s_attr_tensor = torch.tensor(self.graph_target.edge_features)

        data_tensor = Data(x=(x_source, x_target), edge_index=(edges_s2t_tensor, edges_t2s_tensor),
                           edge_attr=(edge_s2t_attr_tensor, edges_t2s_attr_tensor), y=float(self.state.label))
        return data_tensor


class Hetero(Graph, ABC):
    node_types: {str: Graph}  # Graph is a carrier of node features and indices for each node type (Object|Atom) here

    relation_edges: {Predicate: [(Union[Object, Atom], Union[Object, Atom])]}
    relation_edge_features: {Predicate: [[float]]}

    def __init__(self, state: PlanningState):
        self.node_types = {}
        self.relation_edges = {}
        self.relation_edge_features = {}
        super().__init__(state)

    def to_tensors(self) -> HeteroData:
        data: HeteroData = HeteroData()

        for node_type, graph in self.node_types.items():
            data[node_type].x = torch.tensor(list(graph.node_features.values()), dtype=torch.float)

        for relation, edges in self.relation_edges.items():
            type1 = edges[0][0].__class__.__name__
            type2 = edges[0][1].__class__.__name__
            edge_index = [(self.node_types[type1].node2index[i], self.node_types[type2].node2index[j])
                          for i, j in edges]
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
            data[type1, relation.name, type2].edge_index = edge_index_tensor

            data[type1, relation.name, type2].edge_attr = torch.tensor(self.relation_edge_features[relation])
        return data


class Multi(Graph, ABC):
    """This is natively implemented by allowing duplicate/parallel edges in the Graph class"""

    def encode_edge_type(self, index, max_index):
        """Encoding a position/type of something - either scalar, index, or one-hot"""
        if self.edge_type_format == "index":
            return index
        elif self.edge_type_format == "weight":
            return [float(index + 1)]  # if scalar we start indexing from 1 (not to multiply by 0)
        else:  # one-hot
            return one_hot(index, max_index)


class Hypergraph(Graph, ABC):
    # hyperedges are the atoms
    incidence: []


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
            self.node2index[obj] = i  # storing also indices for the tensor version
            feature_vector = multi_hot_object(properties, self.object_feature_names())
            self.node_features[obj] = feature_vector

    # todo nullary predicates are missing here, how to include them?
    def load_edges(self, state: PlanningState, symmetric_edges=True):
        """Collecting all relation types into one multi-hot edge feature vector"""
        edge_types = self.get_edge_types(state, symmetric_edges)

        for constants, predicates in edge_types.items():
            self.edges.append((constants[0], constants[1]))
            feature_vector = multi_hot_object(predicates, self.relation_feature_names())
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
                feature_vector = self.encode_edge_type(self.relation_feature_names().index(predicate),
                                                       len(self.relation_feature_names()))
                self.edge_features.append(feature_vector)


class Object2ObjectHeteroGraph(Object2ObjectGraph, Hetero):
    """Same as Object2ObjectGraph but each relation is a separate edge type with separate learning parameters"""

    def load_nodes(self, state: PlanningState, include_types=True):
        super().load_nodes(state)
        self.node_types["Object"] = self

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for constants, predicates in edge_types.items():
            for predicate in predicates:
                self.relation_edges.setdefault(predicate, []).append((constants[0], constants[1]))
                # no edge features here - each relation is a separate dimension already
                self.relation_edge_features.setdefault(predicate, []).append([1.0])


class Object2AtomGraph(Graph):
    """Object-atom simple graph pretending nodes are of the same type (no object/atom features),
        and edge features are mult-hot positions of the terms in the atoms"""

    def load_nodes(self, state: PlanningState, include_types=True):

        for i, (obj, properties) in enumerate(state.object_properties.items()):
            self.node2index[obj] = i
            self.node_features[obj] = [1.0]  # no node features here

        offset = i + 1
        for i, atom in enumerate(state.atoms):
            self.node2index[atom] = offset + i
            self.node_features[atom] = [1.0]  # no node features here

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for term2atom, positions in edge_types.items():
            self.edges.append((term2atom[0], term2atom[1]))
            self.edge_features.append(multi_hot_index(positions, state.domain.max_arity))

    def get_edge_types(self, state, symmetric_edges):
        edge_types: {(Object, Atom): {int}} = {}
        for i, atom in enumerate(state.atoms):
            for j, term in enumerate(atom.terms):
                edge_types.setdefault(tuple([atom, term]), set()).add(j)
                if symmetric_edges:
                    edge_types.setdefault(tuple([term, atom]), set()).add(j)
        return edge_types


class Object2AtomMultiGraph(Object2AtomGraph, Multi):
    """Same as Object2AtomGraph but with parallel edges for the separate positions"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for i, atom in enumerate(state.atoms):
            for j, term in enumerate(atom.terms):
                self.edges.append((term, atom))
                self.edge_features.append(self.encode_edge_type(j, state.domain.max_arity))
                if symmetric_edges:
                    self.edges.append((atom, term))
                    self.edge_features.append(self.encode_edge_type(j, state.domain.max_arity))


class Object2AtomBipartiteGraph(Object2AtomGraph, Bipartite):
    """Object-atom bipartite graph, i.e. with 2 explicitly different types of nodes with different features"""

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.graph_source = Object2ObjectGraph(state)
        self.graph_target = Atom2AtomGraph(state)

    def load_nodes(self, state: PlanningState, include_types=True):
        self.graph_source.load_nodes(state)
        self.graph_target.load_nodes(state)

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges=False)

        for term2atom, positions in edge_types.items():
            edge_feature = multi_hot_index(positions, state.domain.max_arity)
            # source -> target
            self.graph_source.edges.append((term2atom[0], term2atom[1]))
            self.graph_source.edge_features.append(edge_feature)
            if symmetric_edges:
                # target -> source
                self.graph_target.edges.append((term2atom[1], term2atom[0]))
                self.graph_target.edge_features.append(edge_feature)


class Object2AtomBipartiteMultiGraph(Object2AtomGraph, Bipartite, Multi):
    """Same as Object2AtomBipartiteGraph but with parallel edges"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for i, atom in enumerate(state.atoms):
            for j, term in enumerate(atom.terms):
                edge_feature = self.encode_edge_type(j, state.domain.max_arity)
                # source -> target
                self.graph_source.edges.append((term, atom))
                self.graph_source.edge_features.append(edge_feature)
                if symmetric_edges:
                    # target -> source
                    self.graph_target.edges.append((atom, term))
                    self.graph_target.edge_features.append(edge_feature)


class Object2AtomHeteroGraph(Object2AtomBipartiteGraph, Hetero):
    """Same as Object2AtomGraph but each relation is a separate edge type with separate learning parameters"""

    def load_nodes(self, state: PlanningState, include_types=True):
        super().load_nodes(state)
        self.node_types["Object"] = self.graph_source
        self.node_types["Atom"] = self.graph_target

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges=False)

        for term2atom, positions in edge_types.items():
            for position in positions:
                index_predicate = Predicate("X" + str(position), 0, [], -1)
                self.relation_edges.setdefault(index_predicate, []).append((term2atom[0], term2atom[1]))
                # no edge features here - each relation is a separate dimension already
                self.relation_edge_features.setdefault(index_predicate, []).append([1.0])
                if symmetric_edges:
                    index_predicate = Predicate("Y" + str(position), 0, [], -2)
                    self.relation_edges.setdefault(index_predicate, []).append((term2atom[1], term2atom[0]))
                    self.relation_edge_features.setdefault(index_predicate, []).append([1.0])


class Atom2AtomGraph(Graph):
    """Atom-atom graph with edges being their shared objects, and edge features the multi-hot object ids"""
    term2atoms: {Object: [Atom]}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.term2atoms = {}

    def load_nodes(self, state: PlanningState, include_types=True):
        relations_scope = self.relation_feature_names(include_nullary=True, include_unary=True)
        for i, atom in enumerate(state.atoms):
            self.node2index[atom] = i
            feature_vector = one_hot(relations_scope.index(atom.predicate), len(relations_scope))
            self.node_features[atom] = feature_vector

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for (atom1, atom2), objects in edge_types.items():
            self.edges.append((atom1, atom2))
            # edge feature will be the object ids as multi-hot index vector
            object_ids = multi_hot_object(objects, state.domain.objects)
            self.edge_features.append(object_ids)

    def get_edge_types(self, state, symmetric_edges):
        for atom in state.atoms:
            for term in atom.terms:
                self.term2atoms.setdefault(term, []).append(atom)

        edge_types: {(Atom, Atom): {Object}} = {}
        for i, atom1 in enumerate(state.atoms):
            for term in atom1.terms:
                for atom2 in self.term2atoms[term]:
                    edge_types.setdefault(tuple([atom1, atom2]), set()).add(term)
                    if symmetric_edges:
                        edge_types.setdefault(tuple([atom2, atom1]), set()).add(term)
        return edge_types


class Atom2AtomMultiGraph(Atom2AtomGraph, Multi):
    """Same as Atom2AtomGraph but with parallel edges"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        for atom in state.atoms:
            for term in atom.terms:
                self.term2atoms.setdefault(term, []).append(atom)

        for i, atom1 in enumerate(state.atoms):
            for term in atom1.terms:
                for atom2 in self.term2atoms[term]:
                    self.edges.append((atom1, atom2))
                    # edge feature will be the object id index
                    edge_feature = self.encode_edge_type(state.domain.objects.index(term), len(state.domain.objects))
                    self.edge_features.append(edge_feature)
                    if symmetric_edges:
                        self.edges.append((atom2, atom1))
                        self.edge_features.append(edge_feature)


class Atom2AtomHeteroGraph(Atom2AtomGraph, Hetero):

    def load_nodes(self, state: PlanningState, include_types=True):
        super().load_nodes(state)
        self.node_types["Atom"] = self

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for (atom1, atom2), objects in edge_types.items():
            for object in objects:
                self.relation_edges.setdefault(object, []).append((atom1, atom2))
                # no edge features here - each relation is a separate dimension already
                self.relation_edge_features.setdefault(object, []).append([1.0])
