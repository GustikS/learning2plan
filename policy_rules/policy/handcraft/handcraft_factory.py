from ..policy import Policy
from .blocksworld import BlocksworldPolicy
from .ferry import FerryPolicy
from .miconic import MiconicPolicy
from .satellite_nullary_passes_statespace_checks import SatellitePolicyNullaryX as SatellitePolicy
# from .satellite_nullary import SatellitePolicyNullary as SatellitePolicy
# from .satellite_original import SatellitePolicy
from .transport import TransportPolicy


def get_handcraft_policy(domain: str) -> Policy:
    domains = {
        "blocksworld": BlocksworldPolicy,
        "ferry": FerryPolicy,
        "miconic": MiconicPolicy,
        "satellite": SatellitePolicy,
        "transport": TransportPolicy,
    }
    if domain not in domains:
        raise NotImplementedError
    return domains[domain]
