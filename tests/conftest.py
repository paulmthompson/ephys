"""Shared pytest options and temp-only filesystem write guard for ephys."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="pytest_ephys_mpl_"))

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

import pytest  # noqa: E402

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_SINGLE_PATH_WRITE_EVENTS = frozenset(
    {
        "os.mkdir",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.truncate",
        "shutil.rmtree",
    }
)
_DUAL_PATH_MOVE_EVENTS = frozenset({"os.rename", "os.replace", "shutil.move"})
_DEST_PATH_COPY_EVENTS = frozenset({"shutil.copyfile", "shutil.copytree"})
_NULL_DEVICE_NAMES = frozenset({"nul", "con", "prn", "aux"})

_ENFORCE_TEMP_ONLY_WRITES = False
_HOOK_INSTALLED = False
_REENTRANT = threading.local()


def _norm_path(path: Path) -> Path:
    """Resolve ``path`` without requiring existence and normalize case."""
    resolved = path.resolve(strict=False)
    return Path(os.path.normcase(str(resolved)))


_ALLOWED_WRITE_ROOTS: list[Path] = [
    _norm_path(Path(tempfile.gettempdir())),
]
_DEVNULL_NORM = os.path.normcase(os.path.abspath(os.devnull))


def _register_allowed_root(path: Path) -> None:
    """Allow test writes beneath ``path`` if not already registered."""
    norm = _norm_path(path)
    if norm not in _ALLOWED_WRITE_ROOTS:
        _ALLOWED_WRITE_ROOTS.append(norm)


def _is_allowed_write_target(raw_path: str) -> bool:
    """Return ``True`` when ``raw_path`` is inside an allowed temp/cache dir."""
    lower_raw = raw_path.lower()
    if lower_raw.startswith("\\\\.\\") and lower_raw[4:] in _NULL_DEVICE_NAMES:
        return True
    if os.path.normcase(os.path.abspath(raw_path)) == _DEVNULL_NORM:
        return True

    norm = _norm_path(Path(raw_path))
    if norm.name in _NULL_DEVICE_NAMES:
        return True
    if "__pycache__" in norm.parts:
        return True
    return any(norm.is_relative_to(root) for root in _ALLOWED_WRITE_ROOTS)


def _assert_writable_test_path(target: object, operation: str) -> None:
    """Raise ``PermissionError`` if ``target`` resolves outside temp roots."""
    if not _ENFORCE_TEMP_ONLY_WRITES:
        return
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    if getattr(_REENTRANT, "active", False):
        return
    _REENTRANT.active = True
    try:
        raw = os.fsdecode(target)
        if not _is_allowed_write_target(raw):
            resolved = Path(raw).resolve(strict=False)
            msg = (
                f"Blocked test filesystem write ({operation}) outside "
                f"temporary directories: {resolved}"
            )
            raise PermissionError(msg)
    finally:
        _REENTRANT.active = False


def _audit_filesystem_writes(event: str, args: tuple[object, ...]) -> None:
    """CPython runtime audit hook blocking non-temp writes before syscalls."""
    if not _ENFORCE_TEMP_ONLY_WRITES:
        return

    if event == "open":
        path_arg = args[0] if len(args) > 0 else None
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        is_write = (
            isinstance(mode, str)
            and any(ch in mode for ch in ("w", "a", "x", "+"))
        ) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if is_write:
            _assert_writable_test_path(path_arg, event)
    elif event in _SINGLE_PATH_WRITE_EVENTS:
        if len(args) > 0:
            _assert_writable_test_path(args[0], event)
    elif event in _DUAL_PATH_MOVE_EVENTS:
        if len(args) > 0:
            _assert_writable_test_path(args[0], f"{event}[src]")
        if len(args) > 1:
            _assert_writable_test_path(args[1], f"{event}[dst]")
    elif event in _DEST_PATH_COPY_EVENTS and len(args) > 1:
        _assert_writable_test_path(args[1], f"{event}[dst]")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for optional hardware/Intan integration fixtures."""
    parser.addoption(
        "--dirpath",
        action="store",
        default="tests/data_wrangling/data",
        help="Directory containing Intan test binaries (e.g. digitalin.dat), relative to repo root",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register allowed cache/temp roots and install the CPython audit hook."""
    global _HOOK_INSTALLED  # noqa: PLW0603
    _register_allowed_root(Path(config.rootpath) / ".pytest_cache")
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp is not None:
        _register_allowed_root(Path(basetemp))
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_filesystem_writes)
        _HOOK_INSTALLED = True


def pytest_sessionstart(session: pytest.Session) -> None:
    """Enable temp-only filesystem write enforcement for the test session."""
    global _ENFORCE_TEMP_ONLY_WRITES  # noqa: PLW0603
    _ = session
    _ENFORCE_TEMP_ONLY_WRITES = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Disable write enforcement after test teardown completes."""
    global _ENFORCE_TEMP_ONLY_WRITES  # noqa: PLW0603
    _ = (session, exitstatus)
    _ENFORCE_TEMP_ONLY_WRITES = False


@pytest.fixture(scope="session", autouse=True)
def _allow_pytest_basetemp(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Ensure custom ``--basetemp`` directories remain writable."""
    _register_allowed_root(tmp_path_factory.getbasetemp())


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close any open Matplotlib figures after each test."""
    yield
    plt_mod = sys.modules.get("matplotlib.pyplot")
    if plt_mod is not None:
        plt_mod.close("all")
