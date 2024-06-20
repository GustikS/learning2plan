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
        can_make_no_gluten_sandwich = R.can_make_no_gluten_sandwich("S", "B", "C")
        body = self.get_schema_preconditions("make_sandwich_no_gluten")
        self.add_rule(can_make_no_gluten_sandwich, body)
        self.add_rule("make_sandwich_no_gluten", [can_make_no_gluten_sandwich])

        # make_sandwich(?s - sandwich ?b - bread-portion ?c - content-portion)
        can_make_sandwich = R.can_make_sandwich("S", "B", "C")
        body = self.get_schema_preconditions("make_sandwich")
        body += [
            ~R.can_make_no_gluten_sandwich("S", "B", "C"),
        ]
        self.add_rule(can_make_sandwich, body)
        self.add_rule("make_sandwich", [can_make_sandwich])

        # put_on_tray(?s - sandwich ?t - tray)
        can_put_on_tray = R.can_put_on_tray("S", "T")
        body = self.get_schema_preconditions("put_on_tray")
        body += [
            ~R.can_make_no_gluten_sandwich("S", "B", "C"),
            ~R.can_make_sandwich("S", "B", "C"),
        ]
        self.add_rule(can_put_on_tray, body)
        self.add_rule("put_on_tray", [can_put_on_tray])

        # serve_sandwich_no_gluten(?s - sandwich ?c - child ?t - tray ?p - place)


        # serve_sandwich(?s - sandwich ?c - child ?t - tray ?p - place)


        # move_tray(?t - tray ?p1 ?p2 - place)e
