from ..policy import Policy
from .blocksworld import BlocksworldPolicy
from .ferry import FerryPolicy
from .miconic import MiconicPolicy
from .satellite import SatellitePolicy
from .satellite_100 import SatellitePolicy100
from .transport import TransportPolicy


def get_handcraft_policy(domain: str) -> Policy:
    domains = {
        "blocksworld": BlocksworldPolicy,
        "ferry": FerryPolicy,
        "miconic": MiconicPolicy,
        "satellite": SatellitePolicy,
        "satellite100": SatellitePolicy100,
        "transport": TransportPolicy,
    }
    if domain not in domains:
        raise NotImplementedError
    return domains[domain]
