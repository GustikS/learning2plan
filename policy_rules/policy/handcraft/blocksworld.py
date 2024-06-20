from neuralogic.core import R
from pymimir import Atom
from typing_extensions import override
from util.printing import print_mat

from ..policy import Policy

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


class BlocksworldPolicy(Policy):
    def print_state(self, state: list[Atom]):
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

        print()
        if holding:
            print(f"--[ {holding}")
        print()
        print_mat(towers_transpose)
        print("----" * len(towers))

    def _debug_inference(self):
        print("Inference for current state:")
        self._debug_inference_helper(R.well_placed_block("Ob"))
        # self._debug_inference_helper(R.not_well_placed("Ob"))
        self._debug_inference_helper(R.priority_1_pickup("Ob"))
        self._debug_inference_helper(R.priority_2_pickup("Ob"))
        self._debug_inference_helper(R.putdown_1("Ob"))
        self._debug_inference_helper(R.putdown_2("Ob"))
        print("-" * 80)
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        # well_placed_block 
        # stacked on another block
        head = R.well_placed_block("A")
        body = [
            ~R.get("ug_on")("A", "B"),
            ~R.get("ug_on-table")("A"),
            R.get("ag_on")("A", "C"),
            R.get("well_placed_block")("C"),
        ]
        self._template += head <= body

        # on the table
        head = R.well_placed_block("A")
        body = [
            R.get("ag_on-table")("A"),
        ]
        self._template += head <= body

    @override
    def _add_policy_rules(self):
        # pickup(?ob) - priority 1
        # [pick up from a stacked block if not well placed]
        priority_1_pickup = R.priority_1_pickup("Ob")
        body = [
            ~R.get("well_placed_block")("Ob"),
            ~R.get("ap_on-table")("Ob"),
            ~R.get("ag_on-table")("Ob"),
        ]
        self.add_hardcode_rule(priority_1_pickup, body)
        self.add_hardcode_rule("pickup", [priority_1_pickup])

        # pickup(?ob) - priority 2
        # [pick up from table if not well placed]
        priority_2_pickup = R.priority_2_pickup("Ob")
        body = [
            ~R.priority_1_pickup("Ob"),
            R.ug_on("Ob", "Underob"),
            R.well_placed_block("Underob"),
        ]
        self.add_hardcode_rule(priority_2_pickup, body)
        self.add_hardcode_rule("pickup", [priority_2_pickup])

        # putdown(?ob) - option 1 (options are just for debugging)
        # [put on table if goal block to put on is not well placed]
        # self._template += R.not_well_placed("Ob") <= ~R.well_placed_block("Ob")
        putdown_1 = R.putdown_1("Ob")
        body = [
            R.get("ug_on")("Ob", "Underob"),
            ~R.well_placed_block("Underob"),
        ]
        self.add_hardcode_rule(putdown_1, body)
        self.add_hardcode_rule("putdown", [putdown_1])

        # putdown(?ob) - option 2
        # [put on table if goal is to put on table]
        putdown_2 = R.putdown_2("Ob")
        body = [
            R.get("ug_on-table")("Ob"),
        ]
        self.add_hardcode_rule(putdown_2, body)
        self.add_hardcode_rule("putdown", [putdown_2])

        # stack(?ob, ?underob)
        # [stack on top of a well placed goal block]
        body = [
            R.get("ug_on")("Ob", "Underob"),
            R.get("well_placed_block")("Underob"),
        ]
        self.add_hardcode_rule("stack", body)

        # unstack(?ob, ?underob)
        body = [
            ~R.get("well_placed_block")("Ob"),
        ]
        self.add_hardcode_rule("unstack", body)
