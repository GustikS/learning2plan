from neuralogic.core import R
from typing_extensions import override

from .policy import Policy


class SatellitePolicy(Policy):
    @override
    def add_policy_rules(self):
        self._template += (
            R.instrument_config("I", "M", "S", "D")
            <= R.supports("I", "M")
            & R.on_board("I", "S")
            & R.calibration_target("I", "D"),
        )

        turn_to_head = self.relation_from_schema("turn_to")

        turn_to_rule = self.get_schema_preconditions("turn_to")
        turn_to_rule += [
            R.ug_have_image("D", "M"),
            R.instrument_config("I", "M", "S", "D"),
        ]

        # turn_to_body =

        # ## schemata
        # for schema in self._schemata:
        #     head = relation_from_schema(schema)
        #     body, aux_rules = schema_preconditions(schema)
        #     for rule in aux_rules:
        #         self._template.add_rule(rule)
        #     self._template += (head <= body)
