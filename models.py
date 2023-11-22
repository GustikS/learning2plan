import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


def get_predictions(model, tensor_dataset):
    # model.eval()
    predictions = []
    for sample in tensor_dataset:
        prediction = model(sample.x, sample.edge_index, None)
        predictions.append(prediction.detach().item())
    return predictions

class GCN(torch.nn.Module):
    def __init__(self, tensor_dataset, hidden_channels=16):
        super(GCN, self).__init__()
        torch.manual_seed(1)
        num_node_features = tensor_dataset[0].x.size()[1]

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

