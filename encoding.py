import copy
from abc import abstractmethod, ABC
from typing import Union

import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
from neuralogic.core import Relation, R

from logic import Atom, Predicate, Object, atom2string
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


def multi_hot_aggregate(int_pairs: [(int, int)], max_arity):
    vector = [0.0] * max_arity
    for int_pair in int_pairs:
        vector[int_pair[1]] += 1
    return vector


class Sample(ABC):
    state: PlanningState

    node2index: {Union[Object, Atom]: int}

    cache: {}  # caching the original edge features that some models need to modify post-hoc

    def __init__(self, state: PlanningState):
        self.state = state
        self.node2index = {}
        self.cache = {}

    @abstractmethod
    def to_relations(self) -> [Relation]:
        pass

    @abstractmethod
    def to_tensors(self) -> Data:
        pass

    def object_feature_names(self, include_nullary=False):
        if include_nullary:
            return self.state.domain.unary_predicates + self.state.domain.nullary_predicates
        else:
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
            strings.append(atom2string(atom))
        return ", ".join(strings)


class Graph(Sample, ABC):
    node_features: {Union[Object, Atom]: [float]}
    node_features_symbolic: {Union[Object, Atom]: [str]}

    edges: [(Union[Object, Atom], Union[Object, Atom])]  # note that duplicate edges are allowed here!
    edge_features: [[float]]
    edge_features_symbolic: {(Union[Object, Atom], Union[Object, Atom]): [str]}

    def __init__(self, state: PlanningState):
        super().__init__(state)
        self.node_features = {}
        self.edges = []
        self.edge_features = []

        self.node_features_symbolic = {}
        self.edge_features_symbolic = {}

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

    def draw(self, symbolic=True, pos=None):
        data = self.to_tensors()

        g = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"], graph_attrs=["y"])
        if not pos:
            pos = nx.spring_layout(g)

        if symbolic:
            node_names = {}
            for node, index in self.node2index.items():
                try:
                    nodename = node.name
                except:
                    nodename = str(atom2string(node))
                node_names[index] = nodename

            node_attr = {self.node2index[node]: features for node, features in self.node_features_symbolic.items()}
            edge_attr = {(self.node2index[node1], self.node2index[node2]): features for (node1, node2), features in
                         self.edge_features_symbolic.items()}

            nx.draw_networkx(g, pos, with_labels=True, labels=node_names, font_size=8)
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_attr, font_size=6)

            pos_attrs = {}
            for node, coords in pos.items():
                pos_attrs[node] = (coords[0], coords[1] + 0.08)
            nx.draw_networkx_labels(g, pos_attrs, labels=node_attr, font_size=6)
        else:
            node_attr = {self.node2index[node]: features for node, features in self.node_features.items()}

            nx.draw_networkx(g, pos, with_labels=True, font_size=8)
            nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, 'edge_attr'), font_size=6)

            pos_attrs = {}
            for node, coords in pos.items():
                pos_attrs[node] = (coords[0], coords[1] + 0.08)
            nx.draw_networkx_labels(g, pos_attrs, labels=node_attr, font_size=6)
        plt.show()
        return pos


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

    def draw(self, symbolic=True, pos=None):
        data = self.to_tensors()

        num_nodes_source = data.x[0].size()[0]
        num_nodes_target = data.x[1].size()[0]

        g = nx.DiGraph(node_label_offset=0.2, node_size=0.5)
        g.add_nodes_from(range(num_nodes_source + num_nodes_target))

        for u, v in data.edge_index[0].t().tolist():
            g.add_edge(u, v + num_nodes_source)
        for u, v in data.edge_index[1].t().tolist():
            g.add_edge(u + num_nodes_source, v)

        if not pos:
            pos = nx.bipartite_layout(g, list(range(num_nodes_source)), scale=1)

        if symbolic:
            node_names = {index: node.name for node, index in self.graph_source.node2index.items()}
            node_names.update(
                {index + num_nodes_source: atom2string(node) for node, index in self.graph_target.node2index.items()})

            node_attr_source = {self.graph_source.node2index[node]: features for node, features in
                                self.graph_source.node_features_symbolic.items()}
            node_attr_target = {self.graph_target.node2index[node] + num_nodes_source: features for node, features in
                                self.graph_target.node_features_symbolic.items()}
            node_attr = {**node_attr_source, **node_attr_target}
            edge_attr = {
                (self.graph_source.node2index[node1], self.graph_target.node2index[node2] + num_nodes_source): features
                for (node1, node2), features in self.edge_features_symbolic.items()}
            edge_attr.update(
                {(self.graph_target.node2index[node2] + num_nodes_source, self.graph_source.node2index[node1]): features
                 for (node1, node2), features in self.edge_features_symbolic.items()})

        else:
            node_names = None
            node_attr_source = {self.graph_source.node2index[node]: features for node, features in
                                self.graph_source.node_features.items()}
            node_attr_target = {self.graph_target.node2index[node] + num_nodes_source: features for node, features in
                                self.graph_target.node_features.items()}
            node_attr = {**node_attr_source, **node_attr_target}
            edge_attr = {(node1, node2 + num_nodes_source): self.graph_source.edge_features[i]
                         for i, (node1, node2) in enumerate(data.edge_index[0].t().tolist())}
            edge_attr.update({(node1 + num_nodes_source, node2): self.graph_target.edge_features[i]
                              for i, (node1, node2) in enumerate(data.edge_index[1].t().tolist())})

        nx.draw_networkx(g, pos, with_labels=True, labels=node_names, font_size=8)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_attr, font_size=6)

        pos_attrs = {}
        for node, coords in pos.items():
            if node < + num_nodes_source:
                pos_attrs[node] = (coords[0] - 0.1, coords[1] - 0.05)
            else:
                pos_attrs[node] = (coords[0] + 0.1, coords[1] - 0.05)
        nx.draw_networkx_labels(g, pos_attrs, labels=node_attr, font_size=6)

        ax1 = plt.subplot(111)
        ax1.margins(0.3, 0.05)
        plt.show()
        return pos


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

    def draw(self, symbolic=True, pos=None):
        raise Exception("Drawing not supported for HeteroData")


class Multi(Graph, ABC):
    """This is natively implemented by allowing duplicate/parallel edges in the Graph class"""

    def __init__(self, state: PlanningState, edge_type_format="one_hot"):
        super().__init__(state)

    @staticmethod
    def encode_edge_type(index, max_index, edge_type_format="one_hot"):
        """Encoding a position/type of something - either scalar, index, or one-hot"""
        if edge_type_format == "index":
            return index
        elif edge_type_format == "weight":
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

    def load_nodes(self, state: PlanningState, include_types=True, add_nullary=True):
        object_feature_names = self.object_feature_names(include_nullary=add_nullary)
        object_features = state.object_properties

        if add_nullary:
            for null_pred in state.domain.nullary_predicates:
                for props in object_features.values():
                    props.append(null_pred)

        for i, (obj, properties) in enumerate(object_features.items()):
            self.node2index[obj] = i  # storing also indices for the tensor version
            feature_vector = multi_hot_object(properties, object_feature_names)
            self.node_features[obj] = feature_vector

            self.node_features_symbolic[obj] = [prop.name for prop in properties]

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

        for constants, predicates in edge_types.items():
            self.edge_features_symbolic[(constants[0], constants[1])] = [pred.name for pred in predicates]

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

            self.node_features_symbolic[obj] = [prop.name for prop in properties]

        offset = i + 1
        for i, atom in enumerate(state.atoms):
            self.node2index[atom] = offset + i
            self.node_features[atom] = [1.0]  # no node features here

            self.node_features_symbolic[atom] = [atom.predicate.name]

    def load_edges(self, state: PlanningState, symmetric_edges=True):
        edge_types = self.get_edge_types(state, symmetric_edges)

        for term2atom, positions in edge_types.items():
            self.edges.append((term2atom[0], term2atom[1]))
            self.edge_features.append(multi_hot_index(positions, state.domain.max_arity))

    def get_edge_types(self, state, symmetric_edges):
        edge_types: {(Object, Atom): {int}} = {}
        for i, atom in enumerate(state.atoms):
            for j, term in enumerate(atom.terms):
                edge_types.setdefault(tuple([term, atom]), set()).add(j)
                if symmetric_edges:
                    edge_types.setdefault(tuple([atom, term]), set()).add(j)

        for term2atom, positions in edge_types.items():
            self.edge_features_symbolic[(term2atom[0], term2atom[1])] = [str(pos) for pos in positions]

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
            self.add_edge(term2atom, edge_feature, symmetric_edges)

    def add_edge(self, term2atom, edge_feature, symmetric_edges):
        # source -> target
        self.graph_source.edges.append((term2atom[0], term2atom[1]))
        self.graph_source.edge_features.append(edge_feature)
        if symmetric_edges:
            # target -> source
            self.graph_target.edges.append((term2atom[1], term2atom[0]))
            self.graph_target.edge_features.append(edge_feature)


class Object2AtomBipartiteMultiGraph(Object2AtomBipartiteGraph, Multi):
    """Same as Object2AtomBipartiteGraph but with parallel edges"""

    def load_edges(self, state: PlanningState, symmetric_edges=True):

        edge_types = self.get_edge_types(state, symmetric_edges=False)

        for term2atom, positions in edge_types.items():
            for position in positions:
                edge_feature = self.encode_edge_type(position, state.domain.max_arity)
                self.add_edge(term2atom, edge_feature, symmetric_edges)


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
                index_predicate = Predicate("X" + str(position), 0, tuple([]), -1)
                self.relation_edges.setdefault(index_predicate, []).append((term2atom[0], term2atom[1]))
                # no edge features here - each relation is a separate dimension already
                self.relation_edge_features.setdefault(index_predicate, []).append([1.0])
                if symmetric_edges:
                    index_predicate = Predicate("Y" + str(position), 0, tuple([]), -2)
                    self.relation_edges.setdefault(index_predicate, []).append((term2atom[1], term2atom[0]))
                    self.relation_edge_features.setdefault(index_predicate, []).append([1.0])

    def to_tensors(self) -> Data:
        return Hetero.to_tensors(self)


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

            self.node_features_symbolic[atom] = [atom.predicate.name]

    def load_edges(self, state: PlanningState, symmetric_edges=True, object_ids=False, total_count=False):
        edge_types = self.get_edge_types(state, symmetric_edges=False, object_ids=False)

        for (atom1, atom2), objects in edge_types.items():
            self.edges.append((atom1, atom2))
            if object_ids:  # edge feature will be the object ids as multi-hot index vector
                feature = multi_hot_object(objects, state.domain.objects)
            else:  # edge feature will be the COUNT of the shared objects
                if total_count: # either simple total count of the shared objects between the two atoms
                    feature = one_hot(len(objects) - 1, self.state.domain.max_arity)
                else:   # or a count of shared object per each target position (in the atom2)
                    feature = multi_hot_aggregate(objects, self.state.domain.max_arity)
            self.edge_features.append(feature)

    def get_edge_types(self, state, symmetric_edges=False, object_ids=True):
        for atom in state.atoms:
            for term in atom.terms:
                self.term2atoms.setdefault(term, []).append(atom)

        edge_types: {(Atom, Atom): {Object}} = {}
        for atom1 in state.atoms:
            for i, term in enumerate(atom1.terms):
                for atom2 in self.term2atoms[term]:
                    if object_ids:
                        edge_types.setdefault(tuple([atom1, atom2]), []).append(term)
                    else:
                        edge_types.setdefault(tuple([atom1, atom2]), []).append((i, atom2.terms.index(term)))
                    if symmetric_edges:
                        if object_ids:
                            edge_types.setdefault(tuple([atom2, atom1]), []).append(term)
                        else:
                            edge_types.setdefault(tuple([atom2, atom1]), []).append((atom2.terms.index(term), i))

        for (atom1, atom2), objects in edge_types.items():
            if object_ids:
                self.edge_features_symbolic[(atom1, atom2)] = [obj.name for obj in objects]
            else:
                self.edge_features_symbolic[(atom1, atom2)] = [obj for obj in objects]

        return edge_types


class Atom2AtomMultiGraph(Atom2AtomGraph, Multi):
    """Same as Atom2AtomGraph but with parallel edges"""

    def load_edges(self, state: PlanningState, symmetric_edges=True, object_ids=False):
        edge_types = self.get_edge_types(state, symmetric_edges=False, object_ids=object_ids)

        for (atom1, atom2), objects in edge_types.items():
            for obj in objects:
                self.edges.append((atom1, atom2))
                if object_ids:  # edge feature will be the object id index
                    edge_feature = self.encode_edge_type(state.domain.objects.index(obj), len(state.domain.objects))
                else:  # edge feature will be the object POSITION pair source_atom_position -> target_atom_position
                    edge_feature = self.encode_edge_type(obj[0], self.state.domain.max_arity,
                                                         edge_type_format="one-hot") + \
                                   self.encode_edge_type(obj[1], self.state.domain.max_arity,
                                                         edge_type_format="one-hot")
                self.edge_features.append(edge_feature)


class Atom2AtomHeteroGraph(Atom2AtomGraph, Hetero):

    def load_nodes(self, state: PlanningState, include_types=True):
        super().load_nodes(state)
        self.node_types["Atom"] = self

    def load_edges(self, state: PlanningState, symmetric_edges=True, object_ids=False):
        edge_types = self.get_edge_types(state, symmetric_edges, object_ids)

        for (atom1, atom2), objects in edge_types.items():
            for obj in objects:
                self.relation_edges.setdefault(obj, []).append((atom1, atom2))
                # no edge features here - each relation is a separate dimension already
                self.relation_edge_features.setdefault(obj, []).append([1.0])
