import torch
import torch.nn.functional as F

from data_structures import Object2ObjectGraph, Sample
from models import GCN, get_predictions
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

        self.repeated_predictions()
    def repeated_predictions(self):
        rep_pred = []
        for rep in range(self.repetitions):
            predictions = get_predictions(model, tensor_dataset)
            rep_pred.append([round(distance, self.precision) for distance in predictions])

        rep_pred = list(map(list, zip(*rep_pred)))  # transpose the list of lists

        self.predicted_distances = {}
        for sample, distances in zip(samples, rep_pred):
            self.predicted_distances.setdefault(tuple(distances), []).append(sample)

    def get_all_collisions(self):
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

dataset.enrich_states()  # add info about types, static facts, ...

samples = dataset.get_samples(Object2ObjectGraph)

tensor_dataset = dataset.get_tensor_dataset(samples)

model = GCN(tensor_dataset)

distance_hashing = DistanceHashing(model, samples)

collisions = distance_hashing.get_all_collisions()
print(collisions)
