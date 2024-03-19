from abc import ABC, abstractmethod

import jpype
import neuralogic
from collections import deque

from torch_geometric.nn import GENConv

from learning2plan.expressiveness.encoding import Object2ObjectGraph, Sample
from learning2plan.parsing import get_datasets
from learning2plan.planning import PlanningDataset, PlanningInstance, PlanningState
from learning2plan.solving.lrnn import Backend
from learning2plan.solving.scoring import Scorer, TorchScorer


class Search(ABC):

    def __init__(self, scorer, backend=None):
        self.scorer = scorer
        if backend is None:
            self.backend = Backend()
        else:
            self.backend = backend
        self.get_substitutions = self.backend.planner.getSubstitutions
        self.ground_actions = self.backend.planner.groundActions
        self.next_state = self.backend.planner.nextState

    @abstractmethod
    def solve(self, instance: PlanningInstance):
        pass


class Greedy(Search):
    def __init__(self, scorer: Scorer, backend=None):
        super().__init__(scorer, backend)

    def solve(self, instance: PlanningInstance):
        plan = []
        closed = set()

        backend_instance = instance.to_backend(self.backend)
        state = PlanningState(instance.domain, instance.init)
        backend_state = state.to_backend(self.backend)

        plan.append((None, state))

        while not backend_instance.isGoal(backend_state, self.backend.matching):
            possible_ground_actions = set()
            for action in backend_instance.actions:
                substitutions = self.get_substitutions(backend_state, action)
                ground_actions = self.ground_actions(action, substitutions)
                possible_ground_actions.update(ground_actions)

            # todo work here with the frontend actions instead?
            sorted_action_scores = self.scorer.score_actions(state, possible_ground_actions)
            for ground_action, action_score in sorted_action_scores.items():
                next_state = self.next_state(backend_state, ground_action)
                if next_state not in closed:
                    backend_state = next_state
                    state = PlanningState.from_backend(backend_state, instance.domain)
                    closed.add(next_state)  # just check for cycles...
                    plan.append((ground_action, state))
                    print(state)
                    break
        return plan


# %%

if __name__ == "__main__":
    folder = "../../datasets/rosta/blocks"
    datasets = get_datasets(folder, limit=1, descending=False)  # smallest dataset
    instance = datasets[0]
    instance.init = instance.states[0].atoms

    backend = Backend()
    scorer = TorchScorer(GENConv, Object2ObjectGraph, backend)
    search = Greedy(scorer, backend)
    search.solve(instance)
