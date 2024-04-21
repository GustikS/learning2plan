import heapq
from abc import ABC, abstractmethod

import jpype
import neuralogic
from collections import deque
from queue import PriorityQueue

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
from learning2plan.solving.backend import Backend
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


class GBFSStates(Search):
    def __init__(self, scorer: Scorer, backend=None):
        super().__init__(scorer, backend)

    def solve(self, instance: PlanningInstance):
        plan = []
        closed = set()

        backend_instance = instance.to_backend(self.backend)
        state = PlanningState(instance.domain, instance.init)
        backend_state = state.to_backend(self.backend)

        plan.append(state)

        i = 0
        last_state = None
        while not backend_instance.isGoal(backend_state, self.backend.matching):
            if last_state == backend_state:
                print("greedy search got stuck: no unvisited states left to expand")
                break

            sorted_next_states = self.sorted_succesors(backend_state, backend_instance.actions, instance.domain)

            for next_state in sorted_next_states:
                atoms = frozenset([str(atom) for atom in next_state.atoms])
                if atoms not in closed:
                    closed.add(atoms)  # just check for cycles...
                    backend_state = next_state.to_backend(self.backend)
                    plan.append(next_state)
                    print(i, " : ", next_state)
                    i += 1
                    break
                else:
                    print("already visited")
                    last_state = backend_state
                    continue
        return plan

    def sorted_succesors(self, backend_state, actions, domain):
        next_states = []
        for action in actions:
            substitutions = self.get_substitutions(backend_state, action)
            ground_actions = self.ground_actions(action, substitutions)
            for ground_action in ground_actions:
                next_state = self.next_state(backend_state, ground_action)
                next_states.append(PlanningState.from_backend(next_state, domain))
        sorted_next_states, _ = scorer.score_states(next_states)
        return sorted_next_states


class AStarStates(GBFSStates):
    def __init__(self, scorer: Scorer, backend=None):
        super().__init__(scorer, backend)

    def solve(self, instance: PlanningInstance):
        plan = []

        # open = PriorityQueue()
        open = []
        closed = set()
        parents = {}
        min_scores = {}

        backend_instance = instance.to_backend(self.backend)
        state = PlanningState(instance.domain, instance.init)

        heapq.heappush(open, (float("inf"), state))
        min_scores[state] = 0

        while open:
            # _, state = open.get()
            _, state = heapq.heappop(open)

            backend_state = state.to_backend(self.backend)
            if backend_instance.isGoal(backend_state, self.backend.matching):
                break

            scored_next_states = self.sorted_succesors(backend_state, backend_instance.actions, instance.domain)

            for next_state, next_score in scored_next_states.items():
                atoms = frozenset([str(atom) for atom in next_state.atoms])
                if atoms in closed:
                    print("already visited")
                    continue
                else:
                    curr_min = min_scores.get(state, (float("inf")))
                    next_f = curr_min + 1 + next_score
                    if next_state not in min_scores:
                        min_scores[next_state] = next_f
                        heapq.heappush(open, (next_f, next_state))
                        parents[next_state] = state
                    else:
                        next_min = min_scores[next_state]
                        if next_f < next_min:
                            min_scores[next_state] = next_f
                            open.remove((next_min, next_state))
                            heapq.heappush(open, (next_f, next_state))
                            parents[next_state] = state

            closed.add(frozenset([str(atom) for atom in state.atoms]))
            print(state)

        plan.append(state)
        while parents[state]:
            plan.append(parents[state])
            state = parents[state]

        return plan.reverse()


class GBFSActions(Search):
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

            sorted_ground_actions, _ = self.sorted_actions(state, backend_state, instance, backend_instance)

            for ground_action, action_score in sorted_ground_actions.items():
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

    def sorted_actions(self, state, backend_state, instance, backend_instance):
        """this does an unnecessarily repeated grounding computation in case of the LRNN scorer..."""
        backend_ground_actions = set()
        for action in backend_instance.actions:
            substitutions = self.get_substitutions(backend_state, action)
            ground_actions = self.ground_actions(action, substitutions)
            backend_ground_actions.update(ground_actions)
        frontend_ground_actions = [GroundAction(action, instance.domain) for action in backend_ground_actions]
        return self.scorer.score_actions(backend_state, state, backend_ground_actions, frontend_ground_actions)


class GBFSActionsLRNN(GBFSActions):
    def __init__(self, scorer: LRNNScorer, backend=None):
        if isinstance(scorer, LRNNScorer):
            if scorer.model.contains_actions:
                super().__init__(scorer, backend)
                return
        raise Exception("This search mode is only compatible with LRNNScorer")

    def sorted_actions(self, state, backend_state, instance, backend_instance):
        """this (LRNN-specific) version will ground and score the actions directly as part of the model inference..."""
        return self.scorer.direct_action_scoring(state, instance.actions)


# %%

if __name__ == "__main__":
    folder = "../datasets/blocks"
    datasets = get_datasets(folder, limit=1, descending=False)  # smallest first
    instance = datasets[0]  # choose one
    instance.load_init(instance.states[0])  # setup an artificial init state

    encoding = Object2ObjectGraph
    backend = Backend()

    model = get_trained_model_lrnn(instance, encoding=encoding, model_type=GCNrel, epochs=10)
    scorer = LRNNScorer(model, encoding, backend, instance)

    # model = get_trained_model_torch(instance, encoding=encoding, model_type=GCNConv, epochs=10)
    # scorer = TorchScorer(model, encoding, backend, instance)

    # search = GBFSStates(scorer, backend)
    # search = AStarStates(scorer, backend)
    search = GBFSActions(scorer, backend)
    # search = GBFSActionsLRNN(scorer, backend)

    search.solve(instance)
