from neuralogic.core import R
from typing_extensions import override

from ..policy import Policy


class FerryPolicy(Policy):
    @override
    def _add_derived_predicates(self):
        pass

    @override
    def _add_policy_rules(self):
        # sail(?from - location ?to - location)
        # ferry not empty
        body = [
            R.get("ug_at")("Car", "To"),
            R.get("ap_on")("Car"),
        ]
        self.add_hardcode_rule("sail", body)

        # helper
        head = R.exists_goal_car_at("Loc")
        body = [
            R.get("ug_at")("Car2", "Goal_loc"),
            R.get("ap_at")("Car2", "Loc"),
        ]
        self._template += head <= body

        # sail(?from - location ?to - location)
        # ferry is empty
        body = [
            R.get("ug_at")("Car", "Goal_loc"),
            R.get("ap_at")("Car", "To"),
            R.get("empty-ferry")(),
            ~R.get("exists_goal_car_at")("From"),
        ]
        self.add_hardcode_rule("sail", body)

        # board(?car - car ?loc - location)
        body = [
            R.get("ug_at")("Car", "Goal_loc"),
        ]
        self.add_hardcode_rule("board", body)

        # debark(?car - car  ?loc - location)
        body = [
            R.get("ug_at")("Car", "Loc"),
        ]
        self.add_hardcode_rule("debark", body)
