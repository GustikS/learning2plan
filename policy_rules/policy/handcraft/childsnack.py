from neuralogic.core import R
from typing_extensions import override

from ..policy import Policy


class ChildsnackPolicy(Policy):
    @override
    def _add_derived_predicates(self):
        pass

    @override
    def _add_policy_rules(self):
        # make_sandwich_no_gluten(?s - sandwich ?b - bread-portion ?c - content-portion)
        self.add_hardcode_rule("make_sandwich_no_gluten", [])
        head, rule = self.head_rule_from_schema("make_sandwich_no_gluten")
        rule += [

        ]
        self.add_head_rule(head, rule)

        # make_sandwich(?s - sandwich ?b - bread-portion ?c - content-portion)


        # put_on_tray(?s - sandwich ?t - tray)


        # serve_sandwich_no_gluten(?s - sandwich ?c - child ?t - tray ?p - place)


        # serve_sandwich(?s - sandwich ?c - child ?t - tray ?p - place)


        # move_tray(?t - tray ?p1 ?p2 - place)e
