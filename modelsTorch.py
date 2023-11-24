import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

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
        prediction = model(sample.x, sample.edge_index, None)
        predictions.append(prediction.detach().item())
    return predictions


def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)


class GCN(torch.nn.Module):
    def __init__(self, tensor_dataset, hidden_channels=16):
        super(GCN, self).__init__()
        num_node_features = len(tensor_dataset[0].node_feature_names())

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x
