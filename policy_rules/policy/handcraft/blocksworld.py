from neuralogic.core import R
from typing_extensions import override

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
    def _debug_inference(self):
        print("Inference for current state:")
        self._debug_inference_helper(R.well_placed_block("A"))
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
            R.get("ap_on")("A", "C"),
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
        body = [
            ~R.get("well_placed_block")("Ob"),
            ~R.get("ap_on-table")("Ob"),
        ]
        self.add_hardcode_rule("pickup", body)

        # pickup(?ob) - priority 2
        # TODO

        # putdown(?ob)
        body = [
        ]
        self.add_hardcode_rule("putdown", body)

        # putdown(?ob)
        body = [
            R.get("ug_on-table")("Ob"),
        ]
        self.add_hardcode_rule("putdown", body)

        # stack(?ob, ?underob)
        body = [
            ~R.get("ug_on-table")("Ob"),
            R.get("well_placed_block")("Underob"),
        ]
        self.add_hardcode_rule("stack", body)

        # unstack(?ob, ?underob)
        body = [
            ~R.get("well_placed_block")("A"),
        ]
        self.add_hardcode_rule("unstack", body)
