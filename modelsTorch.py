from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data, HeteroData
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, RGCNConv, GATv2Conv, global_add_pool, \
    MessagePassing, to_hetero, HGTConv
from torch_geometric.nn import Linear as Linear_pyg

from data_structures import Bipartite, Hetero

torch.manual_seed(1)


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
        x = conv(x=x, edge_index=edge_index)
    elif isinstance(conv, GCNConv):
        if len(edge_attr[0]) == 1:
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
        else:  # only scalar edge weights are supported in GCN
            x = conv(x=x, edge_index=edge_index)
    elif isinstance(conv, RGCNConv):
        edge_type_index = torch.argmax(edge_attr, dim=1)  # RGCN need to have the edge types as index not one-hot
        x = conv(x=x, edge_index=edge_index, edge_type=edge_type_index)
    else:
        x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return x


class SimpleGNN(torch.nn.Module):
    def __init__(self, sample=None, model_class=GCNConv, hidden_channels=16, num_layers=3):
        super().__init__()

        if sample:
            first_node_features = next(iter(sample.node_features.items()))[1]
            num_node_features = len(first_node_features)
            num_edge_features = len(sample.edge_features[0])
        else:
            num_node_features = -1
            num_edge_features = -1

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            model_class(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                        add_self_loops=False, num_relations=num_edge_features))
        for i in range(num_layers - 1):
            self.convs.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features, add_self_loops=False,
                            num_relations=num_edge_features))
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data_sample: Data, agg="mean"):
        x = data_sample.x
        edge_index = data_sample.edge_index
        edge_attr = data_sample.edge_attr

        for conv in self.convs:
            x = model_call(conv, x, edge_index, edge_attr)

        if agg == "mean":
            x = global_mean_pool(x, None)
        elif agg == "sum":
            x = global_add_pool(x, None)

        x = self.lin(x)
        return x


class BipartiteGNN(torch.nn.Module):
    def __init__(self, sample, model_class=SAGEConv, hidden_channels=16, num_layers=3):
        super().__init__()

        node_features_source = len(next(iter(sample.graph_source.node_features.items()))[1])
        node_features_target = len(next(iter(sample.graph_target.node_features.items()))[1])
        num_edge_features_s2t = len(sample.graph_source.edge_features[0])
        num_edge_features_t2s = len(sample.graph_target.edge_features[0])

        self.convs_s2t = torch.nn.ModuleList()
        self.convs_s2t.append(
            model_class((node_features_source, node_features_target), hidden_channels,
                        edge_dim=num_edge_features_s2t, add_self_loops=False))
        for i in range(num_layers - 1):
            self.convs_s2t.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features_s2t, add_self_loops=False))

        self.convs_t2s = torch.nn.ModuleList()
        self.convs_t2s.append(
            model_class((node_features_target, node_features_source), hidden_channels,
                        edge_dim=num_edge_features_t2s, add_self_loops=False))
        for i in range(num_layers - 1):
            self.convs_t2s.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features_t2s, add_self_loops=False))

        self.lin = Linear(hidden_channels, 1)

    def forward(self, data_sample: Data, agg="mean"):
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

        if agg == "mean":
            x = global_mean_pool(x, None)
        elif agg == "sum":
            x = global_add_pool(x, None)

        x = self.lin(x)
        return x


class HeteroGNN(torch.nn.Module):
    base_model: SimpleGNN
    conv_class: object

    def __init__(self, sample, model_class=RGCNConv, hidden_channels=16, num_layers=3):
        super().__init__()
        if not isinstance(sample, Hetero):
            raise Exception("HeteroData representation expected for HeteroGNN")

        self.conv_class = model_class
        self.base_model = None
        if model_class not in [HGTConv]:
            simpleGNN = SimpleGNN(None, model_class=model_class, hidden_channels=16, num_layers=3)
            self.base_model = to_hetero(simpleGNN, sample.to_tensors().metadata(), aggr='sum')
        else:
            self.convs = torch.nn.ModuleList()
            self.convs.append(model_class(-1, hidden_channels, sample.to_tensors().metadata()))
            for _ in range(num_layers-1):
                conv = model_class(hidden_channels, hidden_channels, sample.to_tensors().metadata())
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
            x = global_mean_pool(x, None)
            x = self.lin(x)
            return x


class GINConvWrap(GINConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        if isinstance(in_channels, Tuple):
            raise Exception("GIN does not (really) support bipartite graphs!")
            # gin_nn = torch.nn.Sequential(
            #     Linear_pyg(in_channels[0], in_channels[1]), torch.nn.Tanh(),
            #     Linear_pyg(in_channels[1], out_channels))
        else:
            gin_nn = torch.nn.Sequential(
                Linear_pyg(in_channels, out_channels), torch.nn.Tanh(),
                Linear_pyg(out_channels, out_channels))
        super().__init__(gin_nn)
