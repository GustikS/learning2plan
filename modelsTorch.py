import copy
import warnings
from typing import List, Tuple

import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, RGCNConv, GATv2Conv, global_add_pool, \
    MessagePassing, to_hetero, HGTConv, HANConv, FiLMConv, RGATConv, GINEConv, NNConv, PDNConv
from torch_geometric.nn import Linear as Linear_pyg

from encoding import Bipartite, Hetero, Graph

torch.manual_seed(1)

multirelational_gnn_list = [RGCNConv, FiLMConv, RGATConv]
hetero_gnn_list = [HGTConv, HANConv]


class MyException(Exception):
    pass


def get_tensor_dataset(samples):
    data_tensors = []
    for sample in samples:
        data_tensors.append(sample.to_tensors())
    return data_tensors


def get_predictions_torch(model, tensor_samples):
    reset_model_weights(model)
    model.eval()
    predictions = []
    for tensor_sample in tensor_samples:
        prediction = model(tensor_sample)
        predictions.append(prediction.detach().item())
    return predictions


def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)


def model_call(conv, x, edge_index, edge_attr):
    if isinstance(conv, SAGEConv) or isinstance(conv, GINConvWrap):
        warnings.warn("Skipping edge features due to no support in the selected GNN model")
        x = conv(x=x, edge_index=edge_index)  # no edge feature support
    elif isinstance(conv, GCNConv):
        if edge_attr[0].dim() == 0 or edge_attr[0][0].dim() == 0:  # only scalar edge weights are supported in GCN
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
        else:
            warnings.warn("Skipping edge features with GCN")
            x = conv(x=x, edge_index=edge_index)
    elif conv.__class__ in multirelational_gnn_list:
        if edge_attr[0].dim() != 0:  # these need to have the edge types as index not one-hot
            raise MyException(
                "Calling multi-relational model (e.g. RGCN) with wrong edge feature (type-index) encoding")
        x = conv(x=x, edge_index=edge_index, edge_type=edge_attr)
    else:  # general support for edge features
        x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return x


def get_compatible_model(samples, model_class=SAGEConv, num_layers=3, hidden_channels=8, aggr="add",
                         previous_model=None):
    first_sample = samples[0]
    if model_class in hetero_gnn_list and not isinstance(first_sample, Hetero):
        raise MyException("Calling a hetero GNN model on a non-hetero encoding!")

    if isinstance(first_sample, Hetero):
        if model_class != previous_model:
            for sample in samples:
                for relation, edge_features in sample.relation_edge_features.items():
                    check_cache(sample)
                    update_edge_features(edge_features, model_class)
        model = HeteroGNN(samples, model_class, hidden_channels, num_layers, aggr=aggr)
    elif isinstance(first_sample, Bipartite):
        if model_class != previous_model:
            for sample in samples:
                check_cache(sample.graph_source)
                update_edge_features(sample.graph_source.edge_features, model_class)
                check_cache(sample.graph_target)
                update_edge_features(sample.graph_target.edge_features, model_class)
        model = BipartiteGNN(samples, model_class, hidden_channels, num_layers, aggr=aggr)
    else:  # plain graph
        if model_class != previous_model:
            for sample in samples:
                check_cache(sample)
                update_edge_features(sample.edge_features, model_class)
        model = PlainGNN(samples, model_class, hidden_channels, num_layers, aggr=aggr)

    return model


def check_cache(sample):
    # todo - timing - copying edge features is slow! Replace by keeping both versions from the start and switching
    if sample.cache:
        sample.edge_features = copy.deepcopy(sample.cache)
    else:
        sample.cache = copy.deepcopy(sample.edge_features)


def update_edge_features(edge_features_list: [], model_class):
    if model_class == GCNConv or model_class in multirelational_gnn_list:  # repairing edge features for compatibility
        if len(edge_features_list) == 0:
            return
        if isinstance(edge_features_list[0], List):  # only scalars supported here
            for i, edge_features in enumerate(edge_features_list):
                non_zero_idx = [i for i, e in enumerate(edge_features) if e > 0.]
                if len(non_zero_idx) == 0:
                    raise MyException("Calling (R)GCN on an EMPTY edge feature vector, can't extract edge type!)")
                if len(non_zero_idx) > 1:
                    raise MyException("Calling (R)GCN on a multi-hot edge feature vector (only scalars supported)")
                elif edge_features[non_zero_idx[0]] != 1.0:
                    raise MyException("Calling (R)GCN on a one-hot real value edge feature vector - ambiguous")
                else:  # standard one-hot back to index
                    # edge_features.clear()
                    if model_class == GCNConv:
                        # edge_features.append(float(non_zero_idx[0] + 1))
                        edge_features_list[i] = float(non_zero_idx[0] + 1)
                    else:
                        edge_features_list[i] = non_zero_idx[0]


class PlainGNN(torch.nn.Module):
    def __init__(self, samples=None, model_class=GCNConv, hidden_channels=16, num_layers=3, aggr="add"):
        super().__init__()
        self.aggr = aggr
        sample = samples[0]

        if sample:
            first_node_features = next(iter(sample.node_features.items()))[1]
            num_node_features = len(first_node_features)
            try:
                num_edge_features = len(sample.edge_features[0])
            except:
                num_edge_features = -1
                for sam in samples:
                    curr = max(sam.edge_features) + 1
                    if curr > num_edge_features:
                        num_edge_features = curr
        else:
            num_node_features = -1
            num_edge_features = -1

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            model_class(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                        add_self_loops=False, num_relations=num_edge_features, aggr=aggr,
                        hidden_channels=hidden_channels, in_edge_channels=num_edge_features))
        for i in range(num_layers - 1):
            self.convs.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features, add_self_loops=False,
                            num_relations=num_edge_features, aggr=aggr, hidden_channels=hidden_channels,
                            in_edge_channels=num_edge_features))
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data_sample: Data):
        x = data_sample.x
        edge_index = data_sample.edge_index
        edge_attr = data_sample.edge_attr

        for conv in self.convs:
            x = model_call(conv, x, edge_index, edge_attr)

        if self.aggr == "add":
            x = global_add_pool(x, None)
        else:
            x = global_mean_pool(x, None)

        x = self.lin(x)
        return x


class BipartiteGNN(torch.nn.Module):
    def __init__(self, samples, model_class=SAGEConv, hidden_channels=16, num_layers=3, aggr="add"):
        super().__init__()
        self.aggr = aggr
        sample = samples[0]

        if model_class in [GCNConv, RGATConv, PDNConv]:
            raise MyException("The selected GNN does not support Bipartite(Hetero) graphs!")

        node_features_source = len(next(iter(sample.graph_source.node_features.items()))[1])
        node_features_target = len(next(iter(sample.graph_target.node_features.items()))[1])
        try:
            num_edge_features_s2t = len(sample.graph_source.edge_features[0])
        except:
            num_edge_features_s2t = -1
            for sam in samples:
                curr = max(sam.graph_source.edge_features) + 1
                if curr > num_edge_features_s2t:
                    num_edge_features_s2t = curr
        try:
            num_edge_features_t2s = len(sample.graph_target.edge_features[0])
        except:
            num_edge_features_t2s = -1
            for sam in samples:
                curr = max(sam.graph_target.edge_features) + 1
                if curr > num_edge_features_t2s:
                    num_edge_features_t2s = curr

        self.convs_s2t = torch.nn.ModuleList()
        self.convs_s2t.append(
            model_class((node_features_source, node_features_target), hidden_channels,
                        edge_dim=num_edge_features_s2t, num_relations=num_edge_features_s2t, add_self_loops=False,
                        aggr=aggr, hidden_channels=hidden_channels, in_edge_channels=num_edge_features_s2t))
        for i in range(num_layers - 1):
            self.convs_s2t.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features_s2t,
                            num_relations=num_edge_features_s2t, add_self_loops=False, aggr=aggr,
                            hidden_channels=hidden_channels, in_edge_channels=num_edge_features_s2t))

        self.convs_t2s = torch.nn.ModuleList()
        self.convs_t2s.append(
            model_class((node_features_target, node_features_source), hidden_channels,
                        edge_dim=num_edge_features_t2s, num_relations=num_edge_features_t2s, add_self_loops=False,
                        aggr=aggr, hidden_channels=hidden_channels, in_edge_channels=num_edge_features_t2s))
        for i in range(num_layers - 1):
            self.convs_t2s.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features_t2s,
                            num_relations=num_edge_features_t2s, add_self_loops=False, aggr=aggr,
                            hidden_channels=hidden_channels, in_edge_channels=num_edge_features_t2s))

        self.lin = Linear(hidden_channels, 1)

    def forward(self, data_sample: Data, agg="sum"):
        x = data_sample.x
        edge_index = data_sample.edge_index
        edge_attr = data_sample.edge_attr

        x_s2t = x
        x_t2s = (x[1], x[0])
        edges_s2t = edge_index[0]
        edges_t2s = edge_index[1]
        edge_attr_s2t = edge_attr[0]
        edge_attr_t2s = edge_attr[1]

        # interleaving source->target and target->source message passing
        for conv_s2t, conv_t2s in zip(self.convs_s2t, self.convs_t2s):
            out_target = model_call(conv_s2t, x_s2t, edges_s2t, edge_attr_s2t)
            out_source = model_call(conv_t2s, x_t2s, edges_t2s, edge_attr_t2s)
            x_s2t = (out_source, out_target)
            x_t2s = (out_target, out_source)
        x = torch.concat(x_s2t, dim=0)

        if self.aggr == "add":
            x = global_add_pool(x, None)
        else:
            x = global_mean_pool(x, None)

        x = self.lin(x)
        return x


class HeteroGNN(torch.nn.Module):
    base_model: PlainGNN
    conv_class: object

    def __init__(self, samples, model_class=HGTConv, hidden_channels=16, num_layers=3, aggr="add"):
        super().__init__()
        self.aggr = aggr

        if not isinstance(samples[0], Hetero):
            raise MyException("HeteroData representation expected for HeteroGNN")

        self.conv_class = model_class
        self.base_model = None

        supported = hetero_gnn_list
        if model_class not in supported:
            raise MyException(f'Only {[sup.__name__ for sup in supported]} models are supported for HeteroGraphs')

            simpleGNN = PlainGNN(None, model_class=model_class, hidden_channels=16, num_layers=3, aggr=aggr)
            self.base_model = to_hetero(simpleGNN, sample.to_tensors().metadata(), aggr='sum')
        else:
            self.convs = torch.nn.ModuleList()
            # we need to collect the relations and object types from all the samples first
            obj_types = set()
            rel_types = set()
            for sample in samples:
                for obj_type in sample.node_types.keys():
                    obj_types.add(obj_type)
                for relation, edges in sample.relation_edges.items():
                    type1 = edges[0][0].__class__.__name__
                    type2 = edges[0][1].__class__.__name__
                    rel_types.add(tuple([type1, relation.name, type2]))
            meta = tuple([list(obj_types), list(rel_types)])
            self.convs.append(model_class(-1, hidden_channels, meta))
            for _ in range(num_layers - 1):
                conv = model_class(hidden_channels, hidden_channels, meta)
                self.convs.append(conv)

        self.lin = Linear(hidden_channels, 1)

    def forward(self, data_sample: HeteroData):
        if self.base_model:
            return self.base_model.forward(data_sample.x_dict, data_sample.edge_index_dict)
        else:
            x_dict = data_sample.x_dict
            for conv in self.convs:
                x_dict = conv(x_dict, data_sample.edge_index_dict)

            x = torch.concat(list(x_dict.values()), dim=0)
            if self.aggr == "add":
                x = global_add_pool(x, None)
            else:
                x = global_mean_pool(x, None)
            x = self.lin(x)
            return x


class GINConvWrap(GINConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        if isinstance(in_channels, Tuple):
            raise MyException("GIN does not (really) support bipartite graphs!")
        else:
            gin_nn = torch.nn.Sequential(
                Linear_pyg(in_channels, out_channels), torch.nn.Tanh(),
                Linear_pyg(out_channels, out_channels))
        super().__init__(gin_nn, **kwargs)


class GINEConvWrap(GINEConv):
    def __init__(self, in_channels, out_channels, edge_dim, **kwargs):
        if isinstance(in_channels, Tuple):
            raise MyException("GINE does not (really) support bipartite graphs!")
        else:
            gin_nn = torch.nn.Sequential(
                Linear_pyg(in_channels, out_channels), torch.nn.Tanh(),
                Linear_pyg(out_channels, out_channels))
        super().__init__(gin_nn, edge_dim=edge_dim, **kwargs)


class NNConvWrap(NNConv):
    def __init__(self, in_channels, out_channels, edge_dim, **kwargs):
        if isinstance(in_channels, Tuple):
            raise MyException("NNConv does not (really) support bipartite graphs!")
        else:
            gin_nn = torch.nn.Sequential(
                Linear_pyg(edge_dim, out_channels), torch.nn.Tanh(),
                Linear_pyg(out_channels, in_channels*out_channels))
        super().__init__(in_channels, out_channels, nn=gin_nn, **kwargs)
