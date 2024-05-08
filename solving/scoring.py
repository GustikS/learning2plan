from abc import ABC, abstractmethod
from typing import List

from learning.modelsLRNN import LRNN, get_relational_dataset, get_predictions_LRNN
from learning.modelsTorch import get_compatible_model, get_tensor_dataset, get_predictions_torch
from planning import PlanningState, Action

from neuralogic.core import R, V


class Scorer(ABC):

    def __init__(self, model, encoding_type, backend, instance):
        self.model = model
        self.encoding_type = encoding_type
        self.backend = backend
        self.instance = instance

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))

        scores = self.get_scores(samples)

        scored_states = {state: score for (state, score) in zip(states, scores)}
        sorted_states = dict(sorted(scored_states.items(), key=lambda item: item[1]))
        return sorted_states, scores

    def score_actions(self, backend_state, frontend_state, backend_ground_actions,
                      frontend_ground_actions: [Action]) -> {Action: float}:
        successor_states = []
        for ground_action in backend_ground_actions:
            next_state = self.backend.planner.nextState(backend_state, ground_action)
            successor_states.append(PlanningState.from_backend(next_state, self.instance.domain))
        _, scores = self.score_states(successor_states)

        scored_actions = {ground_action: score for (ground_action, score) in zip(backend_ground_actions, scores)}
        sorted_actions = dict(sorted(scored_actions.items(), key=lambda item: item[1]))
        return sorted_actions, scores

    @abstractmethod
    def get_scores(self, samples):
        pass


class TorchScorer(Scorer):

    def __init__(self, model, encoding_type, backend, instance):
        super().__init__(model, encoding_type, backend, instance)

    def get_scores(self, samples):
        tensor_dataset = get_tensor_dataset(samples)
        return get_predictions_torch(self.model, tensor_dataset, reset_weights=False)


class LRNNScorer(Scorer):

    def __init__(self, model, encoding_type, backend, instance):
        super().__init__(model, encoding_type, backend, instance)

    def get_scores(self, samples):
        relational_dataset = get_relational_dataset(samples)
        built_dataset = self.model.model.build_dataset(relational_dataset)
        return get_predictions_LRNN(self.model, built_dataset, reset_weights=False)

    def direct_action_scoring(self, frontend_state, actions: [Action]) -> {Action: float}:
        relational_sample = get_relational_dataset([frontend_state.get_sample(self.encoding_type)])
        built_sample = self.model.model.build_dataset(relational_sample)
        for action in actions:
            groundings = built_sample[0].get_atom(R.get(action.name)(action.parameter_names))
            for grounding in groundings:
                print(grounding.substitutions)
                print(grounding.value)
        # todo next - transform these substitutions into the backend GroundAction representation and return...