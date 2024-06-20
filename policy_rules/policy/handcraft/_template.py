from neuralogic.core import R
from typing_extensions import override

from ..policy import Policy


class _TemplatePolicy(Policy):
    @override
    def _add_policy_rules(self):
        # sail(?from - location ?to - location)
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
