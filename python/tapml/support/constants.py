"""Environment variables used by the TapML."""
import os
import sys
from pathlib import Path


def _check():
    if TAPML_JIT_POLICY not in ["ON", "OFF", "REDO", "READONLY"]:
        raise ValueError(
            'Invalid TAPML_JIT_POLICY. It has to be one of "ON", "OFF", "REDO", "READONLY"'
            f"but got {TAPML_JIT_POLICY}."
        )


def _get_cache_dir() -> Path:
    if "TAPML_CACHE_DIR" in os.environ:
        result = Path(os.environ["TAPML_CACHE_DIR"])
    elif sys.platform == "win32":
        result = Path(os.environ["LOCALAPPDATA"])
        result = result / "tapml"
    elif os.getenv("XDG_CACHE_HOME", None) is not None:
        result = Path(os.getenv("XDG_CACHE_HOME"))
        result = result / "tapml"
    else:
        result = Path(os.path.expanduser("~/.cache"))
        result = result / "tapml"
    result.mkdir(parents=True, exist_ok=True)
    if not result.is_dir():
        raise ValueError(
            f"The default cache directory is not a directory: {result}. "
            "Use environment variable TAPML_CACHE_DIR to specify a valid cache directory."
        )
    (result / "model_weights").mkdir(parents=True, exist_ok=True)
    (result / "model_lib").mkdir(parents=True, exist_ok=True)
    return result


def _get_dso_suffix() -> str:
    if "TAPML_DSO_SUFFIX" in os.environ:
        return os.environ["TAPML_DSO_SUFFIX"]
    if sys.platform == "win32":
        return "dll"
    if sys.platform == "darwin":
        return "dylib"
    return "so"


TAPML_TEMP_DIR = os.getenv("TAPML_TEMP_DIR", None)
TAPML_MULTI_ARCH = os.environ.get("TAPML_MULTI_ARCH", None)
TAPML_CACHE_DIR: Path = _get_cache_dir()
TAPML_JIT_POLICY = os.environ.get("TAPML_JIT_POLICY", "ON")
TAPML_DSO_SUFFIX = _get_dso_suffix()


_check()
