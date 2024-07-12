from neuralogic.core import R
from pymimir import Atom
from typing_extensions import override

from policy_rules.util.printing import print_mat

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy

# policy_rules/l4np/blocksworld/classic/testing/p0_01.pddl
# initial state
# 3
# 5 2
# 4 1
# =====
# goal state
# 4   1
# 3 2 5
# =====


class BlocksworldPolicy(FasterLearningPolicy):
    def _print_bw_state(self, state: list[Atom]):
        on_table = []
        block_dict = {}
        holding = None
        for atom in state:
            if atom.predicate.name == "on-table":
                on_table.append(atom.terms[0].name)
            elif atom.predicate.name == "on":
                above = atom.terms[0].name
                below = atom.terms[1].name
                block_dict[below] = above
            elif atom.predicate.name == "holding":
                holding = atom.terms[0].name
        towers = []
        for block in on_table:
            tower = [block]
            while block in block_dict:
                block = block_dict[block]
                tower.append(block)
            tower = list(reversed(tower))
            towers.append(tower)
        tower_max = max(len(tower) for tower in towers)
        for i in range(len(towers)):
            towers[i] = ([""] * (tower_max - len(towers[i]))) + towers[i]
        towers_transpose = list(zip(*towers))

        if holding:
            print(f"--[ {holding}")
        print()
        print_mat(towers_transpose)
        print("----" * len(towers))

    def print_state(self, state: list[Atom]):
        goal = []
        for g in self._goal:
            assert not g.negated
            goal.append(g.atom)

        print()
        print("Goal state:")
        self._print_bw_state(goal)

        if self._prev_state:
            print()
            print("Previous state:")
            self._print_bw_state(self._prev_state)

            print()
            print("Current state:")
            self._print_bw_state(state)
        else:
            print()
            print("Initial state:")
            self._print_bw_state(state)

        self._prev_state = state

    def _debug_inference(self):
        print("Inference for current state:")
        if self._debug > 2:  # this can be used for precise/complete debugging of the (neural) inference for each state
            super()._debug_inference()
        self._debug_inference_helper(R.well_placed_block("Ob"))
        print("-" * 80)
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        """well_placed_block"""
        # stacked on another well-placed block
        head = R.well_placed_block("A")
        body = [
            R.get("ag_on")("A", "B"),
            R.get("well_placed_block")("B"),
        ]
        self.add_rule(head, body)

        # correctly on the table
        head = R.well_placed_block("A")
        body = [
            R.get("ag_on-table")("A")
        ]
        self.add_rule(head, body)

    @override
    def _add_policy_rules(self):
        """ unstack(?ob, ?underob) """
        # [unstack block that is not well-placed]
        body = [
            ~R.get("well_placed_block")("Ob"),
        ]
        self.add_output_action("unstack", body)

        """stack(?ob, ?underob)"""
        # [stack on top of a well-placed goal block]
        body = [
            R.get("ug_on")("Ob", "Underob"),
            R.get("well_placed_block")("Underob"),
        ]
        self.add_output_action("stack", body)

        """pickup(?ob)"""
        # [pick up from table if goal underneath block is well-placed]
        body = [
            R.get("ug_on")("Ob", "Underob"),
            R.get("well_placed_block")("Underob"),
            R.get("clear")("Underob"),
        ]
        self.add_output_action("pickup", body)

        """ putdown(?ob) """
        # [put on table if goal block to put on is not well-placed]
        body = [
            R.get("ug_on")("Ob", "Underob"),
            ~R.get("well_placed_block")("Underob"),
        ]
        self.add_output_action("putdown", body)

        # [put on table if goal block to put on is blocked]
        body = [
            R.get("ug_on")("Ob", "Underob"),
            R.get("ap_on")("Otherblock", "Underob")
        ]
        self.add_output_action("putdown", body)

        # [put on table if goal is to put on table]
        body = [
            R.get("ug_on-table")("Ob"),
        ]
        self.add_output_action("putdown", body)
