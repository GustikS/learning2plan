from encodings_engine.expressiveness.encoding import Object2ObjectGraph
from encodings_engine.learning.modelsLRNN import get_trained_model_lrnn
from encodings_engine.learning.modelsTorch import get_trained_model_torch
from encodings_engine.parsing import get_datasets

from neuralogic.nn.module import SAGEConv as SAGErel

from torch_geometric.nn import SAGEConv

def train(dataset, encoding=Object2ObjectGraph, framework="lrnn", epochs=100):
    if framework == "lrnn":
        model = get_trained_model_lrnn(dataset, encoding=encoding, model_type=SAGErel, optimizer="ADAM", epochs=epochs)
    elif framework == "torch":
        model = get_trained_model_torch(dataset, encoding=encoding, model_type=SAGEConv, optimizer="ADAM", epochs=epochs)
    return model


if __name__ == "__main__":
    folder = "../datasets/textfiles/blocks"
    datasets = get_datasets(folder, limit=1, descending=False)  # smallest first
    instance = datasets[0]  # choose one
    model = train(instance, framework="lrnn")