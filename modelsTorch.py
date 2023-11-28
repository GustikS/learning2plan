from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, RGCNConv, GATv2Conv, global_add_pool, \
    MessagePassing
from torch_geometric.nn import Linear as Linear_pyg

from data_structures import Bipartite

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
    for sample in tensor_samples:
        prediction = model(sample.x, sample.edge_index, sample.edge_attr)
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
    else:
        x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return x


class SimpleGNN(torch.nn.Module):
    def __init__(self, sample, model_class=GCNConv, hidden_channels=16, num_layers=3):
        super().__init__()

        first_node_features = next(iter(sample.node_features.items()))[1]
        num_node_features = len(first_node_features)
        num_edge_features = len(sample.edge_features[0])

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            model_class(in_channels=num_node_features, out_channels=hidden_channels, edge_dim=num_edge_features,
                        add_self_loops=False))
        for i in range(num_layers - 1):
            self.convs.append(
                model_class(hidden_channels, hidden_channels, edge_dim=num_edge_features, add_self_loops=False))
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None, agg="mean"):
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

    def forward(self, x, edge_index, edge_attr=None, agg="mean"):
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


class RGCN(torch.nn.Module):
    def __init__(self, samples, hidden_channels=16, num_layers=3):
        super().__init__()
        num_node_features = len(samples[0].object_feature_names())
        num_relations = len(samples.edge_types)

        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(num_node_features, hidden_channels, num_relations))
        for i in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, None)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x
