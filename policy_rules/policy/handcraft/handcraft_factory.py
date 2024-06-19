from ..policy import Policy
from .ferry import FerryPolicy
from .satellite import SatellitePolicy


def get_handcraft_policy(domain: str) -> Policy:
    return {
        "ferry": FerryPolicy,
        "satellite": SatellitePolicy,
    }[domain]
