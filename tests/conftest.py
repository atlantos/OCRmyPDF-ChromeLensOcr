import importlib
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_ocrmypdf_stub(version: str = "17.2.0") -> None:
    ocrmypdf_mod = types.ModuleType("ocrmypdf")

    class OcrEngine:
        pass

    def hookimpl(func=None, *args, **kwargs):
        if func is None:
            def decorator(inner):
                return inner
            return decorator
        return func

    ocrmypdf_mod.OcrEngine = OcrEngine
    ocrmypdf_mod.hookimpl = hookimpl
    ocrmypdf_mod.__version__ = version

    exec_mod = types.ModuleType("ocrmypdf._exec")
    exec_mod.tesseract = types.SimpleNamespace(
        get_orientation=lambda *args, **kwargs: 0.0
    )

    sys.modules["ocrmypdf"] = ocrmypdf_mod
    sys.modules["ocrmypdf._exec"] = exec_mod


@pytest.fixture
def plugin():
    prev_ocr = sys.modules.get("ocrmypdf")
    prev_exec = sys.modules.get("ocrmypdf._exec")
    prev_plugin = sys.modules.get("ocrmypdf_chromelens_ocr.plugin")
    prev_pkg = sys.modules.get("ocrmypdf_chromelens_ocr")

    _install_ocrmypdf_stub("17.2.0")
    sys.modules.pop("ocrmypdf_chromelens_ocr.plugin", None)
    sys.modules.pop("ocrmypdf_chromelens_ocr", None)
    mod = importlib.import_module("ocrmypdf_chromelens_ocr.plugin")
    try:
        yield mod
    finally:
        for name in ("ocrmypdf", "ocrmypdf._exec", "ocrmypdf_chromelens_ocr.plugin", "ocrmypdf_chromelens_ocr"):
            sys.modules.pop(name, None)
        if prev_ocr is not None:
            sys.modules["ocrmypdf"] = prev_ocr
        if prev_exec is not None:
            sys.modules["ocrmypdf._exec"] = prev_exec
        if prev_pkg is not None:
            sys.modules["ocrmypdf_chromelens_ocr"] = prev_pkg
        if prev_plugin is not None:
            sys.modules["ocrmypdf_chromelens_ocr.plugin"] = prev_plugin


@pytest.fixture
def plugin_v16():
    prev_ocr = sys.modules.get("ocrmypdf")
    prev_exec = sys.modules.get("ocrmypdf._exec")
    prev_plugin = sys.modules.get("ocrmypdf_chromelens_ocr.plugin")
    prev_pkg = sys.modules.get("ocrmypdf_chromelens_ocr")

    _install_ocrmypdf_stub("16.13.0")
    sys.modules.pop("ocrmypdf_chromelens_ocr.plugin", None)
    sys.modules.pop("ocrmypdf_chromelens_ocr", None)
    mod = importlib.import_module("ocrmypdf_chromelens_ocr.plugin")
    try:
        yield mod
    finally:
        for name in ("ocrmypdf", "ocrmypdf._exec", "ocrmypdf_chromelens_ocr.plugin", "ocrmypdf_chromelens_ocr"):
            sys.modules.pop(name, None)
        if prev_ocr is not None:
            sys.modules["ocrmypdf"] = prev_ocr
        if prev_exec is not None:
            sys.modules["ocrmypdf._exec"] = prev_exec
        if prev_pkg is not None:
            sys.modules["ocrmypdf_chromelens_ocr"] = prev_pkg
        if prev_plugin is not None:
            sys.modules["ocrmypdf_chromelens_ocr.plugin"] = prev_plugin
