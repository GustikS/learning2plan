from ..policy import Policy
from .blocksworld import BlocksworldPolicy
from .ferry import FerryPolicy
from .miconic import MiconicPolicy
from .satellite_nullary import SatellitePolicyNullary
from .satellite_original import SatellitePolicy


def get_handcraft_policy(domain: str) -> Policy:
    domains = {
        "blocksworld": BlocksworldPolicy,
        "ferry": FerryPolicy,
        "miconic": MiconicPolicy,
        # "satellite": SatellitePolicy,
        "satellite": SatellitePolicyNullary,
    }
    if domain not in domains:
        raise NotImplementedError
    return domains[domain]
