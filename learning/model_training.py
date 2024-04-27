from learning2plan.expressiveness.encoding import Object2ObjectGraph, Object2AtomGraph, Atom2AtomGraph
from learning2plan.learning.modelsLRNN import get_trained_model_lrnn
from learning2plan.learning.modelsTorch import get_trained_model_torch
from learning2plan.parsing import get_datasets


from neuralogic.nn.module import GCNConv as GCNrel
from neuralogic.nn.module import SAGEConv as SAGErel
from neuralogic.nn.module import GATv2Conv as GATrel
from neuralogic.nn.module import GINConv as GINrel

from torch_geometric.nn import GCNConv, SAGEConv

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