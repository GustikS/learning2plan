from neuralogic.core import R
from typing_extensions import override

from .policy import Policy


class FerryPolicy(Policy):
    @override
    def _add_derived_predicates(self):
        raise NotImplementedError

    @override
    def _add_policy_rules(self):
        ## sail
        sail_head = self.relation_from_schema("sail")
        sail_rule = self.get_schema_preconditions("sail")
        sail_rule += [
            R.ug_at("Car", "Goal_loc"),
            R.ap_at("Car", "Loc"),
        ]
        self._template += sail_head <= sail_rule


        ## board
        board_head = self.relation_from_schema("board")
        board_rule = self.get_schema_preconditions("board")
        board_rule += [
            R.ug_at("Car", "Goal_loc"),
        ]
        self._template += board_head <= board_rule


        ## debark
        debark_head = self.relation_from_schema("debark")
        debark_rule = self.get_schema_preconditions("debark")
        debark_rule += [
            R.ug_at("Car", "Loc"),
        ]
        self._template += debark_head <= debark_rule
