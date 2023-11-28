from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.graphgym import GINConv
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, RGCNConv, GATv2Conv, global_add_pool

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

class BipartiteGNN(torch.nn.Module):
    def __init__(self, sample, hidden_channels=16, num_layers=3):
        super().__init__()


class GCN(torch.nn.Module):
    def __init__(self, samples, hidden_channels=16, num_layers=3):
        super().__init__()

        if isinstance(samples[0], Bipartite):
            raise Exception("GCN does not support bipartite graphs")

        first_node_features = next(iter(samples[0].node_features.items()))[1]
        num_node_features = len(first_node_features)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr)

        x = global_mean_pool(x, None)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, samples, hidden_channels=16, num_layers=3):
        super().__init__()

        if isinstance(samples[0], Bipartite):
            node_features_left = len(next(iter(samples[0].node_features_source.items()))[1])
            node_features_right = len(next(iter(samples[0].node_features_target.items()))[1])
            in_channels = (node_features_left, node_features_right)
        else:
            first_node_features = next(iter(samples[0].node_features.items()))[1]
            in_channels = len(first_node_features)

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, None)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, samples, hidden_channels=16, num_layers=3):
        super().__init__()

        if isinstance(samples[0], Bipartite):
            raise Exception("GIN does not support bipartite graphs")  # todo but it could

        first_node_features = next(iter(samples[0].node_features.items()))[1]
        num_node_features = len(first_node_features)

        config = LayerConfig()
        config.dim_in = num_node_features
        config.dim_out = hidden_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(config))

        config.dim_in = hidden_channels
        for i in range(num_layers):
            conv = GINConv(config)
            self.convs.append(conv)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = F.relu(conv.model(x, edge_index))
        x = global_add_pool(x, None)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


class GATv2(torch.nn.Module):
    # todo debug - returns too many collisions + can only use 1 head (+ no self-loops if bipartite)
    def __init__(self, samples, hidden_channels=16, num_layers=3, num_heads=1):
        super().__init__()

        if isinstance(samples[0], Bipartite):
            node_features_source = len(next(iter(samples[0].graph_source.node_features.items()))[1])
            node_features_target = len(next(iter(samples[0].graph_target.node_features.items()))[1])

            self.convs_s2t = torch.nn.ModuleList()
            self.convs_s2t.append(
                GATv2Conv((node_features_source, node_features_target), hidden_channels,
                          edge_dim=len(samples[0].graph_source.edge_features[0]),
                          heads=num_heads, add_self_loops=False))
            for i in range(num_layers - 1):
                self.convs_s2t.append(
                    GATv2Conv(hidden_channels, hidden_channels, edge_dim=len(samples[0].graph_source.edge_features[0]),
                              heads=num_heads, add_self_loops=False))

            self.convs_t2s = torch.nn.ModuleList()
            self.convs_t2s.append(
                GATv2Conv((node_features_target, node_features_source), hidden_channels,
                          edge_dim=len(samples[0].graph_target.edge_features[0]),
                          heads=num_heads, add_self_loops=False))
            for i in range(num_layers - 1):
                self.convs_t2s.append(
                    GATv2Conv(hidden_channels, hidden_channels, edge_dim=len(samples[0].graph_target.edge_features[0]),
                              heads=num_heads, add_self_loops=False))

        else:
            first_node_features = next(iter(samples[0].node_features.items()))[1]
            in_channels = len(first_node_features)
            num_edge_features = len(samples[0].edge_features[0])

            self.convs = torch.nn.ModuleList()
            self.convs.append(
                GATv2Conv(in_channels, hidden_channels, edge_dim=num_edge_features, heads=num_heads,
                          add_self_loops=False))
            for i in range(num_layers - 1):
                self.convs.append(
                    GATv2Conv(hidden_channels, hidden_channels, edge_dim=num_edge_features, heads=num_heads,
                              add_self_loops=False))

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        if isinstance(edge_index, Tuple):  # bipartite graph propagation
            edges_s2t = edge_index[0]
            edges_t2s = edge_index[1]
            x_s2t = x
            x_t2s = (x[1], x[0])
            edge_attr_s2t = edge_attr[0]
            edge_attr_t2s = edge_attr[1]

            for conv_s2t, conv_t2s in zip(self.convs_s2t, self.convs_t2s):
                out_target = conv_s2t(x_s2t, edges_s2t, edge_attr=edge_attr_s2t)
                out_source = conv_t2s(x_t2s, edges_t2s, edge_attr=edge_attr_t2s)
                x_s2t = (out_source, out_target)
                x_t2s = (out_target, out_source)
            x = torch.concat(x_s2t, dim=0)

        else:
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, None)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x


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
