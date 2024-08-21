import time
from argparse import Namespace
from typing import Union

import numpy as np
import pymimir
from pymimir import Action, Literal, State
from termcolor import colored

from policy_rules.policy.policy import Policy
from policy_rules.util.printing import print_mat
from policy_rules.util.timer import TimerContextManager

_K = lambda key: colored(key, "light_magenta")

class PolicyExecutor:
    def __init__(
        self,
        policy: Policy,
        initial_state: State,
        goal: list[Literal],
        run_baseline: bool,
        opts: Namespace,
    ) -> None:
        self.policy = policy
        self.initial_state = initial_state
        self.goal = goal
        self.run_baseline = run_baseline
        self._verbosity = opts.verbosity
        self._bound = opts.bound

        if run_baseline:
            self.sampling_method = "uniform"
        elif opts.choice == "sample":
            self.sampling_method = "sample"
        elif opts.action_regression:
            self.sampling_method = "lowest"
        else:
            self.sampling_method = "highest"

    def _goal_count(self, state: State) -> int:
        state_atoms = set([a.get_name() for a in state.get_atoms()])
        ret = 0
        for g in self.goal:
            g_name = g.atom.get_name()
            if g_name not in state_atoms and not g.negated:
                ret += 1
            elif g_name in state_atoms and g.negated:
                ret += 1
        return ret

    def _print_remaining_goals(self, state: State) -> None:
        rem_goals = {}
        ilg_state = self.policy.get_ilg_facts(state.get_atoms())
        for f in ilg_state:
            pred = f.predicate
            if pred.startswith("ug_"):
                pred = pred[3:]
                if pred not in rem_goals:
                    rem_goals[pred] = 0
                rem_goals[pred] += 1
        desc = ""
        for k, v in rem_goals.items():
            desc += f"{k}={v}, "
        self._matrix_log.append([_K("Remaining goals:"), desc])

    def _print_derived_actions(self, policy_actions: list[tuple[float, Action]]) -> None:
        schemata = {k.name: 0 for k in self.policy._domain.action_schemas}
        for a in policy_actions:
            schema = a[1].schema
            schemata[schema.name] += 1
        desc = ""
        for k, v in schemata.items():
            desc += f"{k}={v}, "
        self._matrix_log.append([_K("Derived:"), desc])

    def execute(self) -> None:
        """Main function for executing the policy"""

        _VERBOSITY_LEVEL = self._verbosity

        plan = []
        cycles_detected = 0
        state = self.initial_state
        seen_states = set()
        start_t = time.time()

        # print initial state
        if _VERBOSITY_LEVEL > 1:
            # may or may not be implemented depending on domain
            self.policy.print_state(state.get_atoms())

        print("=" * 80)
        if _VERBOSITY_LEVEL > 0:
            print("Initial state:")
            print(state_repr(state, is_goal=False))
            print("=" * 80)
            print("Goal:")
            print(state_repr(self.goal, is_goal=True))
            print("=" * 80)

        # Main policy execution loop
        while True:
            goals_left = self._goal_count(state)
            if goals_left == 0:
                break

            # log progress
            Step = len(plan)
            t = time.time() - start_t
            print(colored(f"[{Step=}, {goals_left=}, {cycles_detected=}, {t}s]", "blue"))
            self._matrix_log = []

            # actions from Datalog policy
            policy_actions: list[tuple[float, pymimir.Action]] = self.policy.solve(state.get_atoms())

            # sort for reproducibility
            policy_actions = sorted(policy_actions, key=lambda x: x[1].get_name())

            # print remaining goals
            if _VERBOSITY_LEVEL > 0:
                self._print_remaining_goals(state)

            # print number of derived actions per schema
            if _VERBOSITY_LEVEL > 0:
                self._print_derived_actions(policy_actions)

            if len(policy_actions) == 0:
                print(f"Error: At step {len(plan)} but no actions computed and not at goal state!")
                print("Terminating...")
                exit(-1)

            if _VERBOSITY_LEVEL > 1:
                action_names = [f"{v}:{a.get_name()}" for v, a in policy_actions]
                self._matrix_log.append([_K("Available actions:"), ", ".join(action_names)])

            # log seen states for counting cycles
            state_str = state_repr(state)
            seen_states.add(state_str)

            # sample action based on selected criterion
            action_idx = sample_action(policy_actions, self.sampling_method)
            action = policy_actions[action_idx][1]
            succ_state = action.apply(state)
            if state_repr(succ_state) in seen_states:
                cycles_detected += 1

            # add action to plan
            plan.append(action.get_name())
            state = succ_state
            if _VERBOSITY_LEVEL > 0:
                self._matrix_log.append([_K("Applying:"), colored(action.get_name(), "cyan")])

            if _VERBOSITY_LEVEL > 4:
                ilg_state = self.policy.get_ilg_facts(state.get_atoms())
                ilg_state = ", ".join([str(f) for f in ilg_state])
                self._matrix_log.append(["State after action: ", ilg_state])

            if len(self._matrix_log) > 0:
                print_mat(self._matrix_log, rjust=False)

            if _VERBOSITY_LEVEL > 1:
                # may or may not be implemented depending on domain
                self.policy.print_state(state.get_atoms())

            if len(plan) == self._bound:
                # fmt: off
                print(f"Terminating with failure after {self._bound} steps. Increase bound with -b <bound>", flush=True)
                # fmt: on
                exit(-1)

        total_time = time.time() - start_t
        plan_length = len(plan)

        print("=" * 80)
        print("Plan generated!")
        for action in plan:
            print(action)
        print(f"{plan_length=}")
        print(f"{cycles_detected=}")
        print(f"{total_time=}")
        print("=" * 80)
        return plan


def state_repr(state: Union[pymimir.State, list[pymimir.Literal]], is_goal=False):
    if not is_goal:
        atom_names = sorted([a.get_name() for a in state.get_atoms()])
        return " ".join(atom_names)
    else:
        goals = []
        for g in state:
            if g.negated:
                goals.append(f"~{g.atom.get_name()}")
            else:
                goals.append(g.atom.get_name())
        return " ".join(goals)


def sample_action(policy_actions, sampling_method):
    actions = [a[1] for a in policy_actions]
    p = [a[0] for a in policy_actions]
    p = np.array(p)
    indices = list(range(len(policy_actions)))
    match sampling_method:
        case "uniform":
            # uniform sampling
            action_idx = np.random.choice(indices)
        case "sample":
            # sample from distribution computed by scores
            p = p / sum(p)
            action_idx = np.random.choice(indices, p=p)
        # case "sample":
        #     # sample after converting distrubtion with softmax
        #     # p = 1/(1 + np.exp(-p))  # sigmoid so softmax does not overflow
        #     p = p / sum(p)  # normalise so softmax does not overflow
        #     div = sum(np.exp(p))
        #     p = np.exp(p) / div  # softmax
        #     print(p)
        #     action_idx = np.random.choice(indices, p=p)
        case "highest":
            # if action classification we take the highest,
            action_idx = np.argmax(p)
        case "lowest":
            # if action regression we take the lowest
            action_idx = np.argmin(p)
    return action_idx
