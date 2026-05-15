def pytest_addoption(parser):
    parser.addoption(
        "--dirpath",
        action="store",
        default="tests/data_wrangling/data",
        help="Directory containing Intan test binaries (e.g. digitalin.dat), relative to repo root",
    )
