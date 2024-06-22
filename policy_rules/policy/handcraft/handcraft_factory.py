from ..policy import Policy
from .blocksworld import BlocksworldPolicy
from .ferry import FerryPolicy
from .miconic import MiconicPolicy
from .satellite import SatellitePolicy


def get_handcraft_policy(domain: str) -> Policy:
    domains = {
        "blocksworld": BlocksworldPolicy,
        "ferry": FerryPolicy,
        "miconic": MiconicPolicy,
        "satellite": SatellitePolicy,
    }
    if domain not in domains:
        raise NotImplementedError
    return domains[domain]
