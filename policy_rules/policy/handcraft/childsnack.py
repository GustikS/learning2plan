from neuralogic.core import R
from typing_extensions import override

from ..policy import Policy


class ChildsnackPolicy(Policy):
    @override
    def _add_derived_predicates(self):
        pass

    @override
    def _add_policy_rules(self):
        # skip since optimally solvable in linear time
        raise NotImplementedError
