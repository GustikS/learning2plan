import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, RGCNConv, GATv2Conv

from modelsLRNN import GNN, get_predictions_LRNN, get_relational_dataset
from data_structures import Object2ObjectGraph, Object2ObjectMultiGraph, Object2ObjectHeteroGraph, \
    Object2AtomGraph, Object2AtomBipartiteGraph
from modelsTorch import get_predictions_torch, get_tensor_dataset, SimpleGNN, BipartiteGNN, GINConvWrap
from parsing import get_datasets
from planning import PlanningDataset, PlanningState


class DistanceHashing:
    precision: int
    repetitions: int

    true_distances: {int: [PlanningState]}
    predicted_distances: {tuple: [PlanningState]}

    def __init__(self, model, samples, precision=10, repetitions=3):
        self.precision = precision
        self.repetitions = repetitions

        self.true_distances = {}
        for sample in samples:
            self.true_distances.setdefault(sample.state.label, []).append(sample)

        self.repeated_predictions(model, samples)

    def repeated_predictions(self, model, samples):
        rep_pred = []
        if isinstance(model, torch.nn.Module):
            tensor_dataset = get_tensor_dataset(samples)
            for rep in range(self.repetitions):
                rep_pred.append(get_predictions_torch(model, tensor_dataset))
        else:
            logic_dataset = get_relational_dataset(samples)
            built_dataset = model.model.build_dataset(logic_dataset)
            for rep in range(self.repetitions):
                rep_pred.append(get_predictions_LRNN(model, built_dataset))

        rounded_predictions = []
        for predictions in rep_pred:
            rounded_predictions.append([round(distance, self.precision) for distance in predictions])
        rounded_predictions = list(map(list, zip(*rounded_predictions)))  # transpose the list of lists

        self.predicted_distances = {}
        for sample, distances in zip(samples, rounded_predictions):
            self.predicted_distances.setdefault(tuple(distances), []).append(sample)

    def get_all_collisions(self):
        """Remember that collisions are not always bad due to the desired symmetry invariance(s)"""
        return {distance: collisions for distance, collisions in self.predicted_distances.items() if
                len(collisions) > 1}

    def get_bad_collisions(self):
        """Collisions that are of a different true distance - that is always bad"""
        confusions = {}
        for distance, collisions in self.predicted_distances.items():
            if len(collisions) > 1:
                for state1 in collisions:
                    for state2 in collisions:
                        if state1.label != state2.label:
                            confusions.setdefault(state1, []).append(state2)
        return confusions


# %%

# folder = "C:/Users/gusta/Downloads/planning/geffner/data/data/supervised/optimal/train/blocks-clear/blocks-clear"
folder = "C:/Users/gusta/Downloads/planning/rosta/blocks"

datasets = get_datasets(folder, limit=1)  # let's just get the first/smallest dataset for now
dataset = datasets[0]

# %%

dataset.enrich_states()  # add info about types, static facts, goal...

# 1) choose an encoding
# samples = dataset.get_samples(Object2ObjectGraph)
# samples = dataset.get_samples(Object2ObjectMultiGraph)
# samples = dataset.get_samples(Object2AtomGraph)
samples = dataset.get_samples(Object2AtomBipartiteGraph)
# samples = dataset.get_samples(Object2ObjectHeteroGraph)

# 2) choose a model
# model = SimpleGNN(samples[0], model_class=GCNConv)
# model = SimpleGNN(samples[0], model_class=SAGEConv)
# model = SimpleGNN(samples[0], model_class=GINConvWrap)
# model = SimpleGNN(samples[0], model_class=GATv2Conv)

# model = BipartiteGNN(samples[0], model_class=SAGEConv)
model = BipartiteGNN(samples[0], model_class=GATv2Conv)

# model = GNN(samples)    # LRNN

distance_hashing = DistanceHashing(model, samples)

collisions = distance_hashing.get_all_collisions()
print(collisions)
