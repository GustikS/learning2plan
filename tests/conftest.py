DOMAINS = [
    "blocksworld",
    "ferry",
    "rover",
    "satellite",
]


def pytest_addoption(parser):
    parser.addoption("--domain", action="store", default=None)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturedomains".
    option_value = metafunc.config.option.domain
    if "domain" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("domain", [option_value])
    elif "domain" in metafunc.fixturenames and option_value is None:
        metafunc.parametrize("domain", DOMAINS)
