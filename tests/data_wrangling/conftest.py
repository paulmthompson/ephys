

# conftest.py
def pytest_addoption(parser):
    parser.addoption("--dirpath", action="store", default="data", help="Directory path for test files")