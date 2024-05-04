from neuralogic.core import R
from typing_extensions import override

from .policy import Policy


class FerryPolicy(Policy):
    @override
    def _add_derived_predicates(self):
        pass

    @override
    def _add_policy_rules(self):
        ## sail(?from - location ?to - location)
        ## ferry not empty
        sail_head = self.relation_from_schema("sail")
        sail_rule = self.get_schema_preconditions("sail")
        sail_rule += [
            R.get("ug_at")("Car", "To"),
            R.get("ap_at")("Car", "From"),
            R.get("ap_on")("Car", "Ferry"),
            self._get_negative_literal("empty-ferry", []),
        ]
        self._template += sail_head <= sail_rule

        ## sail(?from - location ?to - location)
        ## ferry is empty
        sail_head = self.relation_from_schema("sail")
        sail_rule = self.get_schema_preconditions("sail")
        sail_rule += [
            R.get("ug_at")("Car", "Goal_loc"),
            R.get("ap_at")("Car", "To"),
            R.get("empty-ferry")(),
            ## not exists Car2 R.get("ug_at")("Car2", "Goal_loc"), R.get("ap_at")("Car2", "From")
        ]
        self._template += sail_head <= sail_rule

        ## board(?car - car ?loc - location)
        board_head = self.relation_from_schema("board")
        board_rule = self.get_schema_preconditions("board")
        board_rule += [
            R.get("ug_at")("Car", "Goal_loc"),
        ]
        self._template += board_head <= board_rule

        ## debark(?car - car  ?loc - location)
        debark_head = self.relation_from_schema("debark")
        debark_rule = self.get_schema_preconditions("debark")
        debark_rule += [
            R.get("ug_at")("Car", "Loc"),
        ]
        self._template += debark_head <= debark_rule
