from ..policy import Policy
from .blocksworld import BlocksworldPolicy
from .childsnack import ChildsnackPolicy
from .ferry import FerryPolicy
from .satellite import SatellitePolicy


def get_handcraft_policy(domain: str) -> Policy:
    domains = {
        "blocksworld": BlocksworldPolicy,
        "childsnack": ChildsnackPolicy,
        "ferry": FerryPolicy,
        "satellite": SatellitePolicy,
    }
    if domain not in domains:
        raise NotImplementedError
    return domains[domain]
