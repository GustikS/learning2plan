import sys
import warnings
from decimal import Decimal, Context, ROUND_HALF_DOWN
from math import comb

import numpy as np
import torch

from modelsLRNN import GNN, get_predictions_LRNN, get_relational_dataset
from modelsTorch import get_tensor_dataset, get_predictions_torch
from planning import PlanningDataset, PlanningState


class DistanceHashing:
    precision: int
    repetitions: int

    true_distances: {int: [PlanningState]}
    predicted_distances: {tuple: [PlanningState]}

    def __init__(self, model, samples, precision=5, repetitions=3, epsilon_check=True, epsilon=1e-6):
        self.precision = precision
        self.repetitions = repetitions

        self.samples = samples
        self.true_distances = {}
        for sample in samples:
            self.true_distances.setdefault(sample.state.label, []).append(sample)

        self.repeated_predictions(model, samples)

        if epsilon_check:
            self.epsilon_sanity_check(epsilon=epsilon)

    def repeated_predictions(self, model, samples):
        rep_pred = []
        if isinstance(model, torch.nn.Module):
            tensor_dataset = get_tensor_dataset(samples)
            for rep in range(self.repetitions):
                rep_pred.append(get_predictions_torch(model, tensor_dataset))
        else:
            logic_dataset = get_relational_dataset(samples)
            built_dataset = model.base_model.build_dataset(logic_dataset)
            for rep in range(self.repetitions):
                rep_pred.append(get_predictions_LRNN(model, built_dataset))

        aligned_predictions = list(map(list, zip(*rep_pred)))  # transpose the list of lists
        rounded_predictions = self.round_predictions(aligned_predictions)

        self.predicted_distances = {}
        for sample, distances in zip(samples, rounded_predictions):
            distance_key = tuple(distances)
            # distance_key = np.array(distances)
            self.predicted_distances.setdefault(distance_key, []).append(sample)

    def round_predictions(self, aligned_predictions):
        context = Context(prec=self.precision, rounding=ROUND_HALF_DOWN)
        rounded_predictions = []
        for predictions in aligned_predictions:
            rounded_predictions.append([context.create_decimal(str(distance)) for distance in predictions])
        return rounded_predictions

    def epsilon_sanity_check(self, epsilon=1e-6):
        """Possibly expensive call for pairwise comparisons"""
        near_collisions = 0
        for i, (distances1, collisions1) in enumerate(self.predicted_distances.items()):
            for j, (distances2, collisions2) in enumerate(self.predicted_distances.items()):
                if i >= j: continue
                difference = sum(abs(d1 - d2) for d1, d2 in zip(distances1, distances2))
                if difference < epsilon:
                    near_collisions += 1
                    warnings.warn(("A different but very close prediction detected:"))
                    print('========Similar predicted values========')
                    print(distances1)
                    print(distances2)
                    print("For the following 2 sets of states:")
                    print(collisions1)
                    print(collisions2)
        return near_collisions

    def get_all_collisions(self):
        """Remember that collisions are not always bad due to the desired symmetry invariance(s)"""
        confusions = {}
        pairwise_count = 0
        for distance, collisions in self.predicted_distances.items():
            if len(collisions) > 1:
                confusions[distance] = collisions
                pairwise_count += comb(len(collisions), 2)  # all 2-size subsets
        return pairwise_count, confusions

    def get_bad_collisions(self):
        """Collisions that are of a different true distance - that is always bad"""
        confusions = {}
        pairwise_count = 0
        for distance, collisions in self.predicted_distances.items():
            if len(collisions) > 1:
                num_collisions = len(collisions)
                for i in range(num_collisions):
                    for j in range(i + 1, num_collisions):
                        sample1 = collisions[i]
                        sample2 = collisions[j]
                        if sample1.state.label != sample2.state.label:
                            pairwise_count += 1
                            confusions.setdefault(sample1, []).append(sample2)
        return pairwise_count, confusions

    def get_compression_rates(self):
        class_compression = len(self.true_distances) / len(self.predicted_distances)
        sample_compression = len(self.predicted_distances) / len(self.samples)
        return sample_compression, class_compression
