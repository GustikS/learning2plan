from ..policy import Policy
from .blocksworld import BlocksworldPolicy
from .ferry import FerryPolicy
from .miconic import MiconicPolicy
from .satellite_nullary import SatellitePolicyNullary
from .satellite_original import SatellitePolicy
from .transport import TransportPolicy


def get_handcraft_policy(domain: str) -> Policy:
    domains = {
        "blocksworld": BlocksworldPolicy,
        "ferry": FerryPolicy,
        "miconic": MiconicPolicy,
        # "satellite": SatellitePolicy,
        "satellite": SatellitePolicyNullary,
        "transport": TransportPolicy,
    }
    if domain not in domains:
        raise NotImplementedError
    return domains[domain]
