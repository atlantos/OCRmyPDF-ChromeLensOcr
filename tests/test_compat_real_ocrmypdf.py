import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ocrmypdf = pytest.importorskip("ocrmypdf")


def _reload_plugin_module():
    sys.modules.pop("ocrmypdf_chromelens_ocr.plugin", None)
    sys.modules.pop("ocrmypdf_chromelens_ocr", None)
    return importlib.import_module("ocrmypdf_chromelens_ocr.plugin")


def _write_input_image(path: Path):
    Image.new("RGB", (120, 90), (255, 255, 255)).save(path, dpi=(300, 300))


def test_hook_signature_matches_ocrmypdf_major():
    plugin = _reload_plugin_module()
    major = int(str(getattr(ocrmypdf, "__version__", "0")).split(".")[0])

    sig = inspect.signature(plugin.get_ocr_engine)
    params = list(sig.parameters.keys())
    if major >= 17:
        assert params == ["options"]
        assert plugin.get_ocr_engine(SimpleNamespace(ocr_engine="tesseract")) is None
        assert isinstance(
            plugin.get_ocr_engine(SimpleNamespace(ocr_engine="auto")),
            plugin.ChromeLensEngine,
        )
    else:
        assert params == []
        assert isinstance(plugin.get_ocr_engine(), plugin.ChromeLensEngine)


def test_generate_hocr_smoke_with_real_ocrmypdf(monkeypatch, tmp_path):
    plugin = _reload_plugin_module()
    engine = plugin.ChromeLensEngine()

    input_img = tmp_path / "input.png"
    output_hocr = tmp_path / "out.hocr"
    output_text = tmp_path / "out.txt"
    _write_input_image(input_img)

    monkeypatch.setattr(engine, "_create_lens_proto_request", lambda *_: b"req")
    monkeypatch.setattr(engine, "_send_proto_request", lambda *_: b"resp")
    monkeypatch.setattr(
        engine,
        "_strict_parse_hierarchical",
        lambda *_args, **_kwargs: [
            {
                "bbox": [0, 0, 100, 50],
                "rotation": 0.0,
                "lines": [
                    {
                        "bbox": [0, 0, 50, 10],
                        "rotation": 0.0,
                        "words": [{"text": "hello", "bbox": [0, 0, 10, 10], "sep": " "}],
                    }
                ],
            }
        ],
    )

    opts = SimpleNamespace(
        chromelens_dump_debug=False,
        keep_temporary_files=False,
        chromelens_no_dehyphenation=True,
        chromelens_max_dehyphen_len=10,
    )
    engine.generate_hocr(input_img, output_hocr, output_text=output_text, options=opts)

    assert output_hocr.exists()
    assert output_text.exists()
    assert "hello" in output_text.read_text(encoding="utf-8")
