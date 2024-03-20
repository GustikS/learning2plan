from abc import ABC, abstractmethod

import jpype
import neuralogic
from collections import deque

from neuralogic.nn.module import GCNConv as GCNrel
from neuralogic.nn.module import SAGEConv as SAGErel
from neuralogic.nn.module import GATv2Conv as GATrel
from neuralogic.nn.module import GINConv as GINrel
from torch_geometric.nn import GCNConv

from learning2plan.expressiveness.encoding import Object2ObjectGraph, Sample
from learning2plan.learning.modelsLRNN import LRNN, get_trained_model_lrnn
from learning2plan.learning.modelsTorch import get_trained_model_torch
from learning2plan.parsing import get_datasets
from learning2plan.planning import PlanningDataset, PlanningInstance, PlanningState, GroundAction
from learning2plan.solving.lrnn import Backend
from learning2plan.solving.scoring import Scorer, TorchScorer, LRNNScorer


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
        closed = jpype.java.util.HashSet()

        backend_instance = instance.to_backend(self.backend)
        state = PlanningState(instance.domain, instance.init)
        backend_state = state.to_backend(self.backend)

        plan.append((None, state))

        i = 0
        last_state = None
        while not backend_instance.isGoal(backend_state, self.backend.matching):
            if last_state == backend_state:
                print("greedy search got stuck: no unvisited states left to expand")
                break

            backend_ground_actions = set()
            for action in backend_instance.actions:
                substitutions = self.get_substitutions(backend_state, action)
                ground_actions = self.ground_actions(action, substitutions)
                backend_ground_actions.update(ground_actions)

            frontend_ground_actions = [GroundAction(action, instance.domain) for action in backend_ground_actions]
            sorted_ground_actions = self.scorer.score_actions(backend_state, state, backend_ground_actions,
                                                              frontend_ground_actions)

            for ground_action, action_score in sorted_ground_actions.items():
                # todo merge this so that the grounding is done only once
                next_state = self.next_state(backend_state, ground_action)
                if next_state.clause not in closed:
                    backend_state = next_state
                    state = PlanningState.from_backend(backend_state, instance.domain)
                    closed.add(next_state.clause)  # just check for cycles...
                    plan.append((ground_action, state))
                    print(i, " : ", state)
                    i += 1
                    break
                else:
                    print("already visited")
                    last_state = backend_state
                    continue
        return plan


# %%

if __name__ == "__main__":
    folder = "../../datasets/rosta/blocks"
    datasets = get_datasets(folder, limit=1, descending=False)  # smallest first
    instance = datasets[0]  # choose one
    instance.load_init(instance.states[0])  # setup an artificial init state

    encoding = Object2ObjectGraph
    backend = Backend()

    model = get_trained_model_lrnn(instance, encoding=encoding, model_type=GCNrel, epochs=10)
    scorer = LRNNScorer(model, encoding, backend, instance)

    # model = get_trained_model_torch(instance, encoding=encoding, model_type=GCNConv, epochs=10)
    # scorer = TorchScorer(model, encoding, backend, instance)

    search = Greedy(scorer, backend)
    search.solve(instance)
