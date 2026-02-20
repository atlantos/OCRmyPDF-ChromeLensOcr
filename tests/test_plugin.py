import argparse
import json
import math
import struct
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _field_float(field_id: int, value: float) -> bytes:
    return _varint((field_id << 3) | 5) + struct.pack("<f", value)


def _field_varint(field_id: int, value: int) -> bytes:
    return _varint((field_id << 3) | 0) + _varint(value)


def _make_box(cx, cy, w, h, rotation, coordinate_type):
    return b"".join(
        [
            _field_float(1, cx),
            _field_float(2, cy),
            _field_float(3, w),
            _field_float(4, h),
            _field_float(5, rotation),
            _field_varint(6, coordinate_type),
        ]
    )


def _write_rgba_image(path: Path, size=(220, 120), dpi=(72, 73)):
    img = Image.new("RGBA", size, (255, 0, 0, 128))
    img.save(path, dpi=dpi)


def test_nfkc_patch_preserves_superscripts(plugin):
    original = unicodedata.normalize
    try:
        assert plugin._patched_normalize("NFKC", "x¹") == "x¹"
        assert plugin._patched_normalize("NFC", "x¹") == plugin._original_normalize("NFC", "x¹")
        plugin._apply_nfkc_patch()
        assert unicodedata.normalize("NFKC", "x¹") == "x¹"

        # _remove_nfkc_patch restores original behaviour
        plugin._remove_nfkc_patch()
        assert unicodedata.normalize("NFKC", "x¹") == plugin._original_normalize("NFKC", "x¹")
        assert plugin._nfkc_patch_applied is False

        # Calling remove again when already removed is a no-op
        plugin._remove_nfkc_patch()
        assert plugin._nfkc_patch_applied is False
    finally:
        unicodedata.normalize = original
        plugin._nfkc_patch_applied = False


def test_next_request_uuid_is_uint64(plugin):
    values = [plugin._next_request_uuid() for _ in range(200)]
    assert all(0 <= value < (1 << 64) for value in values)
    assert len(set(values)) == len(values)


def test_is_ocrmypdf_v17_invalid_version_fallback(plugin, monkeypatch):
    monkeypatch.setattr(plugin.ocrmypdf, "__version__", "v17-custom")
    assert plugin._is_ocrmypdf_v17_or_newer() is True
    monkeypatch.setattr(plugin.ocrmypdf, "__version__", "unknown")
    assert plugin._is_ocrmypdf_v17_or_newer() is False


@pytest.mark.parametrize(
    ("sys_platform", "profile"),
    [
        ("win32", "windows"),
        ("linux", "linux"),
        ("darwin", "macos"),
    ],
)
def test_runtime_lens_platform_and_identity(plugin, monkeypatch, sys_platform, profile):
    monkeypatch.setattr(plugin.sys, "platform", sys_platform)
    assert plugin._runtime_lens_platform() == profile
    api_key, user_agent, browser_year, platform_name = plugin._lens_request_identity()
    assert platform_name == profile
    assert api_key == plugin.LENS_PLATFORM_PROFILES[profile]["api_key"]
    assert user_agent == plugin.LENS_PLATFORM_PROFILES[profile]["user_agent"]
    assert browser_year == plugin.LENS_PLATFORM_PROFILES[profile]["browser_year"]
    assert (
        plugin.LENS_PLATFORM_PROFILES[profile]["sec_ch_ua_platform"]
        in plugin._sec_ch_ua_headers(profile, user_agent)["sec-ch-ua-platform"]
    )


def test_runtime_lens_platform_unknown_fallback(plugin, monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(plugin.sys, "platform", "plan9")
    assert plugin._runtime_lens_platform() == "macos"
    assert any("defaulting to macOS Lens profile" in rec.message for rec in caplog.records)


def test_sec_ch_ua_headers_fallback_major_on_unparseable_ua(plugin, caplog):
    caplog.set_level("WARNING")
    headers = plugin._sec_ch_ua_headers("linux", "NotAChromeUA/1.0")
    assert '"Google Chrome";v="144"' in headers["sec-ch-ua"]
    assert any("Unable to parse Chrome major version" in rec.message for rec in caplog.records)


def test_generate_x_browser_validation_matches_reference_value(plugin):
    api_key = plugin.LENS_PLATFORM_PROFILES["macos"]["api_key"]
    user_agent = plugin.LENS_PLATFORM_PROFILES["macos"]["user_agent"]
    assert (
        plugin._generate_x_browser_validation(api_key, user_agent)
        == "YX3LzjiV26KLi9dp+0FecwLxpEU="
    )


def test_xml_bbox_union_helpers(plugin):
    assert plugin.xml_sanitize("A\x00B\x1fC") == "ABC"
    assert plugin.xml_sanitize("") == ""
    assert plugin.bbox_str([1, 2, 3, 4]) == "bbox 1 2 3 4"
    assert plugin.union_bboxes([]) == [0, 0, 0, 0]
    assert plugin.union_bboxes([[1, 2, 5, 6], [0, 3, 4, 10]]) == [0, 2, 5, 10]


def test_proto_writer_and_mini_proto_roundtrip(plugin):
    nested = plugin.ProtoWriter()
    nested.add_string(1, "nested")

    msg = plugin.ProtoWriter()
    msg.add_varint(1, 150)
    msg.add_string(2, "hello")
    msg.add_message(3, nested)
    payload = msg.get_bytes()

    # Prepend a wire-type-1 field to exercise that branch in MiniProto.parse().
    with_wire1 = _varint((9 << 3) | 1) + (b"\x00" * 8) + payload
    fields = plugin.MiniProto(with_wire1).parse()

    assert fields[1][0] == 150
    assert fields[2][0] == b"hello"
    nested_fields = plugin.MiniProto(fields[3][0]).parse()
    assert nested_fields[1][0] == b"nested"


def test_miniproto_eof_paths(plugin):
    reader = plugin.MiniProto(b"")
    with pytest.raises(EOFError):
        reader.read_varint()
    with pytest.raises(EOFError):
        reader.read_fixed32()
    with pytest.raises(EOFError):
        reader.read_bytes(1)


def test_add_options_and_engine_metadata(plugin, monkeypatch):
    parser = argparse.ArgumentParser()
    plugin.add_options(parser)
    args = parser.parse_args(
        ["--chromelens-no-dehyphenation", "--chromelens-max-dehyphen-len", "7", "--chromelens-dump-debug"]
    )
    assert args.chromelens_no_dehyphenation is True
    assert args.chromelens_max_dehyphen_len == 7
    assert args.chromelens_dump_debug is True

    engine = plugin.ChromeLensEngine()
    assert plugin.ChromeLensEngine.version() == "1.0.5"
    assert plugin.ChromeLensEngine.creator_tag().startswith("OCRmyPDF-ChromeLens-Ocr ")
    assert str(engine) == "ChromeLensOcr"
    assert engine.engine_name() == "ChromeLensOcr"

    assert engine.languages(None) == {"eng", "auto"}
    assert engine.languages(SimpleNamespace(languages={"rus"})) == {"rus"}
    assert engine.get_deskew(Path("x"), SimpleNamespace()) == 0.0

    called = {}

    def fake_get_orientation(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return 42.0

    monkeypatch.setattr(plugin.tesseract, "get_orientation", fake_get_orientation)
    opts = SimpleNamespace(tesseract_oem=1, tesseract_non_ocr_timeout=5)
    assert engine.get_orientation(Path("image.png"), opts) == 42.0
    assert called["kwargs"]["engine_mode"] == 1
    assert called["kwargs"]["timeout"] == 5


def test_generate_hocr_success_with_debug_and_downscale(plugin, monkeypatch, tmp_path):
    input_img = tmp_path / "input.png"
    _write_rgba_image(input_img, size=(220, 120))
    output_hocr = tmp_path / "out.hocr"
    output_text = tmp_path / "out.txt"

    engine = plugin.ChromeLensEngine()
    monkeypatch.setattr(plugin, "MAX_DIMENSION_V17", 100)

    captured = {"dump_called": False}

    def fake_create(img_bytes, width, height):
        captured["create"] = (len(img_bytes), width, height)
        return b"request"

    def fake_send(payload):
        assert payload == b"request"
        return b"response"

    def fake_parse(binary_data, orig_w, orig_h, upload_w, upload_h):
        assert binary_data == b"response"
        captured["sizes"] = (orig_w, orig_h, upload_w, upload_h)
        return [
            {
                "bbox": [0, 0, 100, 100],
                "rotation": 0.0,
                "lines": [
                    {"bbox": [0, 0, 50, 10], "rotation": 0.0, "words": [{"text": "foo-", "bbox": [0, 0, 5, 5], "sep": None}]},
                    {"bbox": [0, 20, 50, 30], "rotation": 0.0, "words": [{"text": "bar", "bbox": [0, 20, 5, 25], "sep": " "}]},
                ],
            }
        ]

    def fake_dump(**kwargs):
        captured["dump_called"] = True
        assert kwargs["orig_size"] == (220, 120)

    def fake_write(layout, img_w, img_h, dpi, input_file, out_hocr, out_text):
        captured["write"] = (img_w, img_h, tuple(dpi), input_file, out_hocr, out_text)
        # De-hyphenation should merge foo- + bar.
        assert layout[0]["lines"][0]["words"][-1]["text"] == ""
        assert layout[0]["lines"][1]["words"][0]["text"] == "foobar"

    monkeypatch.setattr(engine, "_create_lens_proto_request", fake_create)
    monkeypatch.setattr(engine, "_send_proto_request", fake_send)
    monkeypatch.setattr(engine, "_strict_parse_hierarchical", fake_parse)
    monkeypatch.setattr(engine, "_dump_debug_artifacts", fake_dump)
    monkeypatch.setattr(engine, "_write_output_hierarchical", fake_write)

    opts = SimpleNamespace(
        chromelens_dump_debug=True,
        keep_temporary_files=True,
        chromelens_no_dehyphenation=False,
        chromelens_max_dehyphen_len=10,
    )
    engine.generate_hocr(input_img, output_hocr, output_text=output_text, options=opts)

    assert captured["sizes"] == (220, 120, 100, 55)
    assert captured["dump_called"] is True
    assert captured["write"][0:2] == (220, 120)


def test_generate_hocr_debug_warning_only_once(plugin, monkeypatch, caplog, tmp_path):
    input_img = tmp_path / "input.png"
    _write_rgba_image(input_img, size=(80, 60))
    engine = plugin.ChromeLensEngine()

    monkeypatch.setattr(engine, "_create_lens_proto_request", lambda *_: b"request")
    monkeypatch.setattr(engine, "_send_proto_request", lambda *_: b"response")
    monkeypatch.setattr(engine, "_strict_parse_hierarchical", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(engine, "_write_output_hierarchical", lambda *_args, **_kwargs: None)

    opts = SimpleNamespace(
        chromelens_dump_debug=True,
        keep_temporary_files=False,
        chromelens_no_dehyphenation=True,
        chromelens_max_dehyphen_len=10,
    )
    caplog.set_level("WARNING")
    engine.generate_hocr(input_img, tmp_path / "a.hocr", output_text=tmp_path / "a.txt", options=opts)
    engine.generate_hocr(input_img, tmp_path / "b.hocr", output_text=tmp_path / "b.txt", options=opts)
    warning_lines = [rec.message for rec in caplog.records if "Ignoring --chromelens-dump-debug" in rec.message]
    assert len(warning_lines) == 1


def test_generate_hocr_fails_on_invalid_image(plugin, tmp_path):
    engine = plugin.ChromeLensEngine()
    with pytest.raises(plugin.OcrEngineError, match="Failed to process image"):
        engine.generate_hocr(tmp_path / "does_not_exist.png", tmp_path / "out.hocr")


def test_generate_hocr_wraps_google_logic_failure(plugin, monkeypatch, tmp_path):
    input_img = tmp_path / "input.png"
    _write_rgba_image(input_img, size=(80, 60))
    engine = plugin.ChromeLensEngine()

    monkeypatch.setattr(engine, "_create_lens_proto_request", lambda *_: b"request")
    monkeypatch.setattr(engine, "_send_proto_request", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(plugin.OcrEngineError, match="Google Lens logic failed"):
        engine.generate_hocr(input_img, tmp_path / "out.hocr")


def test_dump_debug_artifacts_writes_expected_files(plugin, tmp_path):
    engine = plugin.ChromeLensEngine()
    out_hocr = tmp_path / "000001_ocr_hocr.hocr"

    engine._dump_debug_artifacts(
        output_hocr=out_hocr,
        request_proto=b"req",
        response_proto=b"resp",
        layout_structure=[{"k": "v"}],
        orig_size=(100, 200),
        upload_size=(80, 160),
        dpi=(300, 300),
    )

    base = out_hocr.with_suffix("")
    request_path = base.with_name(base.name + "_chromelens_request.pb")
    response_path = base.with_name(base.name + "_chromelens_response.pb")
    layout_path = base.with_name(base.name + "_chromelens_layout.json")
    meta_path = base.with_name(base.name + "_chromelens_meta.json")

    assert request_path.read_bytes() == b"req"
    assert response_path.read_bytes() == b"resp"
    assert json.loads(layout_path.read_text(encoding="utf-8"))["paragraphs"] == [{"k": "v"}]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["orig_size"]["width"] == 100
    assert meta["upload_size"]["height"] == 160


def test_dump_debug_artifacts_handles_write_error(plugin, caplog, tmp_path):
    engine = plugin.ChromeLensEngine()
    caplog.set_level("WARNING")
    engine._dump_debug_artifacts(
        output_hocr=tmp_path / "x.hocr",
        request_proto="not-bytes",
        response_proto=b"resp",
        layout_structure=[],
        orig_size=(1, 1),
        upload_size=(1, 1),
        dpi=(300, 300),
    )
    assert any("Failed to write ChromeLens debug dump files" in rec.message for rec in caplog.records)


def test_sort_lines_by_rotation(plugin):
    engine = plugin.ChromeLensEngine()
    paragraphs = [
        {
            "lines": [
                {"bbox": [0, 10, 1, 11], "rotation": -1.0},
                {"bbox": [0, 30, 1, 31], "rotation": -1.0},
            ]
        },
        {
            "lines": [
                {"bbox": [0, 20, 1, 21], "rotation": 1.0},
                {"bbox": [0, 5, 1, 6], "rotation": 1.0},
            ]
        },
        {"lines": []},
    ]
    updated = engine._sort_lines_by_rotation(paragraphs)
    assert [l["bbox"][1] for l in updated[0]["lines"]] == [30, 10]
    assert [l["bbox"][1] for l in updated[1]["lines"]] == [5, 20]


def test_dehyphenate_merges_short_split_words(plugin):
    engine = plugin.ChromeLensEngine()
    paragraphs = [
        {
            "lines": [
                {
                    "words": [
                        {"text": "более", "bbox": [0, 0, 0, 0], "sep": " "},
                        {"text": "летопи-", "bbox": [0, 0, 0, 0], "sep": None},
                    ],
                    "bbox": [0, 0, 0, 0],
                },
                {
                    "words": [
                        {"text": "сями", "bbox": [0, 0, 0, 0], "sep": " "},
                    ],
                    "bbox": [0, 0, 0, 0],
                },
            ]
        }
    ]

    updated = engine._dehyphenate(paragraphs, max_len_threshold=10)
    first_line_words = updated[0]["lines"][0]["words"]
    second_line_words = updated[0]["lines"][1]["words"]

    assert first_line_words[-1]["text"] == "летописями"
    assert first_line_words[-1]["sep"] == " "
    assert second_line_words[0]["text"] == ""
    assert second_line_words[0]["sep"] is None


def test_dehyphenate_skips_dash_and_uppercase_cases(plugin):
    engine = plugin.ChromeLensEngine()
    paragraphs = [
        {
            "lines": [
                {"words": [{"text": "text -", "bbox": [0, 0, 0, 0], "sep": None}]},
                {"words": [{"text": "Suffix", "bbox": [0, 0, 0, 0], "sep": " "}]},
            ]
        }
    ]
    updated = engine._dehyphenate(paragraphs, max_len_threshold=10)
    assert updated[0]["lines"][0]["words"][0]["text"] == "text -"
    assert updated[0]["lines"][1]["words"][0]["text"] == "Suffix"


def test_dehyphenate_does_not_merge_two_long_parts(plugin):
    engine = plugin.ChromeLensEngine()
    paragraphs = [
        {
            "lines": [
                {"words": [{"text": "extraordinary-", "bbox": [0, 0, 0, 0], "sep": None}]},
                {"words": [{"text": "compatibility", "bbox": [0, 0, 0, 0], "sep": " "}]},
            ]
        }
    ]

    updated = engine._dehyphenate(paragraphs, max_len_threshold=5)
    assert updated[0]["lines"][0]["words"][0]["text"] == "extraordinary-"
    assert updated[0]["lines"][1]["words"][0]["text"] == "compatibility"


def test_create_lens_proto_request_contains_expected_fields(plugin):
    engine = plugin.ChromeLensEngine()
    payload = engine._create_lens_proto_request(b"img", 321, 654)

    server = plugin.MiniProto(payload).parse()
    objects = plugin.MiniProto(server[1][0]).parse()

    request_context = plugin.MiniProto(objects[1][0]).parse()
    request_id = plugin.MiniProto(request_context[3][0]).parse()
    client_context = plugin.MiniProto(request_context[4][0]).parse()
    locale_context = plugin.MiniProto(client_context[4][0]).parse()
    image_data = plugin.MiniProto(objects[3][0]).parse()
    image_payload = plugin.MiniProto(image_data[1][0]).parse()
    image_metadata = plugin.MiniProto(image_data[3][0]).parse()
    request_payload = plugin.MiniProto(objects[4][0]).parse()

    assert request_id[2][0] == 1
    assert client_context[1][0] == plugin.LENS_CLIENT_PLATFORM
    assert client_context[2][0] == plugin.LENS_CLIENT_SURFACE
    assert locale_context[1][0].decode("utf-8") == plugin.LENS_LOCALE_LANGUAGE
    assert locale_context[2][0].decode("utf-8") == plugin.LENS_LOCALE_REGION
    assert locale_context[3][0].decode("utf-8") == plugin.LENS_LOCALE_TIMEZONE
    assert image_payload[1][0] == b"img"
    assert image_metadata[1][0] == 321
    assert image_metadata[2][0] == 654
    assert request_payload[6][0] == plugin.LENS_PAYLOAD_REQUEST_TYPE
    assert request_payload[4][0].decode("utf-8") == plugin.LENS_PAYLOAD_CONTENT_TYPE
    assert request_payload[5][0].decode("utf-8") == plugin.LENS_PAYLOAD_PAGE_URL


def test_send_proto_request_adds_chrome_validation_headers(plugin, monkeypatch):
    engine = plugin.ChromeLensEngine()
    captured = {}

    class _Resp:
        status_code = 200
        content = b"ok"
        text = "ok"

    def fake_post(url, data, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(plugin.requests, "post", fake_post)
    monkeypatch.setattr(plugin.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plugin.random, "uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(plugin.sys, "platform", "linux")

    response = engine._send_proto_request(b"payload")
    expected_api_key = plugin.LENS_PLATFORM_PROFILES["linux"]["api_key"]
    expected_user_agent = plugin.LENS_PLATFORM_PROFILES["linux"]["user_agent"]
    expected_browser_year = plugin.LENS_PLATFORM_PROFILES["linux"]["browser_year"]
    expected_platform = plugin.LENS_PLATFORM_PROFILES["linux"]["sec_ch_ua_platform"]

    assert response == b"ok"
    assert captured["url"] == plugin.LENS_PROTO_ENDPOINT
    assert captured["data"] == b"payload"
    assert captured["timeout"] == 120
    assert captured["headers"]["X-Goog-Api-Key"] == expected_api_key
    assert captured["headers"]["User-Agent"] == expected_user_agent
    assert captured["headers"]["x-browser-channel"] == "stable"
    assert captured["headers"]["x-browser-year"] == str(expected_browser_year)
    assert captured["headers"]["sec-ch-ua-mobile"] == "?0"
    assert captured["headers"]["sec-ch-ua-platform"] == f'"{expected_platform}"'
    assert '"Google Chrome";v="144"' in captured["headers"]["sec-ch-ua"]
    assert (
        captured["headers"]["x-browser-validation"]
        == plugin._generate_x_browser_validation(expected_api_key, expected_user_agent)
    )


def test_send_proto_request_retries_then_succeeds(plugin, monkeypatch):
    engine = plugin.ChromeLensEngine()
    calls = {"count": 0}

    class _Resp:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "error"
            self.content = b"ok"

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        return _Resp(500 if calls["count"] == 1 else 200)

    monkeypatch.setattr(plugin.requests, "post", fake_post)
    monkeypatch.setattr(plugin.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plugin.random, "uniform", lambda *_args, **_kwargs: 0.0)

    assert engine._send_proto_request(b"payload") == b"ok"
    assert calls["count"] == 2


def test_send_proto_request_raises_after_network_failures(plugin, monkeypatch):
    engine = plugin.ChromeLensEngine()

    def fake_post(*_args, **_kwargs):
        raise plugin.requests.RequestException("offline")

    monkeypatch.setattr(plugin.requests, "post", fake_post)
    monkeypatch.setattr(plugin.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plugin.random, "uniform", lambda *_args, **_kwargs: 0.0)

    with pytest.raises(plugin.OcrEngineError, match="Network error"):
        engine._send_proto_request(b"payload")


def test_send_proto_request_raises_after_server_failures(plugin, monkeypatch):
    engine = plugin.ChromeLensEngine()

    class _Resp:
        status_code = 500
        text = "bad"
        content = b""

    monkeypatch.setattr(plugin.requests, "post", lambda *_args, **_kwargs: _Resp())
    monkeypatch.setattr(plugin.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plugin.random, "uniform", lambda *_args, **_kwargs: 0.0)

    with pytest.raises(plugin.OcrEngineError, match="Server returned 500"):
        engine._send_proto_request(b"payload")


def test_parse_geometry_normalized(plugin):
    engine = plugin.ChromeLensEngine()
    box = _make_box(cx=0.5, cy=0.5, w=0.2, h=0.4, rotation=0.0, coordinate_type=1)

    bbox, rotation = engine._parse_geometry(box, img_w=100, img_h=200)

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    assert width in (20, 21)
    assert height in (80, 81)
    assert abs(center_x - 50) <= 1
    assert abs(center_y - 100) <= 1
    assert rotation == 0.0


def test_parse_geometry_image_coordinates_scaled_to_original(plugin):
    engine = plugin.ChromeLensEngine()
    box = _make_box(cx=50, cy=20, w=40, h=20, rotation=0.0, coordinate_type=2)

    bbox, rotation = engine._parse_geometry(
        box,
        img_w=200,
        img_h=200,
        upload_w=100,
        upload_h=100,
    )

    assert bbox == [60, 20, 140, 60]
    assert rotation == 0.0


def test_parse_geometry_unknown_type_and_rotation(plugin):
    engine = plugin.ChromeLensEngine()

    normalized_like = _make_box(cx=0.5, cy=0.5, w=0.6, h=0.2, rotation=math.pi / 2, coordinate_type=0)
    bbox, _ = engine._parse_geometry(normalized_like, img_w=100, img_h=100)
    assert (bbox[2] - bbox[0]) in (20, 21)
    assert (bbox[3] - bbox[1]) in (60, 61)

    absolute_like = _make_box(cx=50, cy=50, w=20, h=40, rotation=0.0, coordinate_type=0)
    bbox2, _ = engine._parse_geometry(absolute_like, img_w=200, img_h=200, upload_w=100, upload_h=100)
    assert bbox2 == [80, 60, 120, 140]

    clipping_box = _make_box(cx=0.0, cy=0.0, w=1.0, h=1.0, rotation=0.0, coordinate_type=1)
    bbox3, _ = engine._parse_geometry(clipping_box, img_w=100, img_h=100)
    assert bbox3 == [0, 0, 50, 50]


def test_strict_parse_hierarchical_returns_empty_for_missing_root_field(plugin):
    engine = plugin.ChromeLensEngine()
    assert engine._strict_parse_hierarchical(b"", orig_w=100, orig_h=100) == []


def test_strict_parse_hierarchical_parses_words_and_superscripts(plugin, monkeypatch):
    engine = plugin.ChromeLensEngine()
    real_miniproto = plugin.MiniProto

    para_box = _make_box(0.5, 0.5, 0.9, 0.9, 0.0, 1)
    line_box = _make_box(0.5, 0.2, 0.8, 0.1, 0.0, 1)
    word_box_a = _make_box(0.2, 0.2, 0.1, 0.05, 0.0, 1)
    word_box_b = _make_box(0.4, 0.2, 0.1, 0.05, 0.0, 1)

    mapping = {
        b"root": {2: [b"obj"]},
        b"obj": {3: [b"text"]},
        b"text": {1: [b"layout"]},
        b"layout": {1: [b"para"]},
        # No para geometry: exercises fallback bbox union at paragraph level.
        b"para": {2: [b"line1", b"line2"]},
        b"line1": {2: [b"line1_geo"], 1: [b"bad", b"blank", b"sup", b"plain", b"nobbox", b"allsup"]},
        b"line2": {1: [b"tail_only"]},
        b"line1_geo": {1: [line_box]},
        b"sup": {2: ["abc⁴".encode("utf-8")], 3: [b"."], 4: [b"word_geo_a"]},
        b"plain": {2: [b"plain"], 4: [b"word_geo_b"]},
        b"nobbox": {2: [b"nobbox"], 3: [b" "]},
        b"allsup": {2: ["⁵".encode("utf-8")], 4: [b"word_geo_b"]},
        b"tail_only": {2: [b"tail"], 3: [b" "]},
        b"bad": {2: [b"\xff"]},
        b"blank": {2: [b"   "]},
        b"word_geo_a": {1: [word_box_a]},
        b"word_geo_b": {1: [word_box_b]},
        b"para_geo_unused": {1: [para_box]},
    }

    class FakeMiniProto:
        def __init__(self, data):
            self.data = data

        def parse(self):
            if self.data in mapping:
                return mapping[self.data]
            return real_miniproto(self.data).parse()

    monkeypatch.setattr(plugin, "MiniProto", FakeMiniProto)

    paragraphs = engine._strict_parse_hierarchical(
        b"root",
        orig_w=100,
        orig_h=100,
        upload_w=100,
        upload_h=100,
    )

    assert len(paragraphs) == 1
    line1_words = paragraphs[0]["lines"][0]["words"]
    line2_words = paragraphs[0]["lines"][1]["words"]

    assert [w["text"] for w in line1_words] == ["abc", "⁴", "plain", "nobbox", "⁵"]
    assert line1_words[1]["sep"] == "."
    assert line1_words[3]["bbox"] == paragraphs[0]["lines"][0]["bbox"]
    # No line bbox and no word bbox for line2 path -> default tiny bbox.
    assert line2_words[0]["bbox"] == [0, 0, 1, 1]
    assert paragraphs[0]["bbox"] is not None


def test_line_title_and_plain_text(plugin):
    engine = plugin.ChromeLensEngine()
    assert engine._line_title({"bbox": [0, 1, 2, 3]}) == "bbox 0 1 2 3"
    assert engine._line_title({"bbox": [0, 1, 2, 3], "rotation": 0.0}) == "bbox 0 1 2 3"
    with_angle = engine._line_title({"bbox": [0, 1, 2, 3], "rotation": 1.0})
    assert "textangle" in with_angle

    assert engine._line_plain_text([]) == ""
    words = [{"text": "foo", "sep": " "}, {"text": "bar", "sep": "."}]
    assert engine._line_plain_text(words) == "foo bar."


def test_write_output_hierarchical_writes_hocr_and_text(plugin, tmp_path):
    engine = plugin.ChromeLensEngine()
    paragraphs = [
        {
            "bbox": [0, 0, 100, 30],
            "rotation": 0.0,
            "lines": [
                {
                    "bbox": [0, 0, 100, 10],
                    "rotation": 1.0,
                    "words": [
                        {"text": "foo", "bbox": [0, 0, 10, 10], "sep": " "},
                        {"text": "bar", "bbox": [10, 0, 20, 10], "sep": "."},
                    ],
                }
            ],
        }
    ]
    out_hocr = tmp_path / "out.hocr"
    out_txt = tmp_path / "out.txt"
    engine._write_output_hierarchical(
        paragraphs=paragraphs,
        img_w=100,
        img_h=100,
        dpi=(0, float("nan")),
        input_file=tmp_path / "input.png",
        output_hocr=out_hocr,
        output_text=out_txt,
    )

    hocr = out_hocr.read_text(encoding="utf-8")
    text = out_txt.read_text(encoding="utf-8")

    assert "scan_res 300 300" in hocr
    assert "textangle" in hocr
    assert "foo" in hocr and "bar." in hocr
    assert text == "foo bar."


def test_generate_pdf_raises_not_implemented(plugin):
    engine = plugin.ChromeLensEngine()
    with pytest.raises(NotImplementedError):
        engine.generate_pdf(Path("in.png"), Path("out.pdf"), Path("out.hocr"))


def test_get_ocr_engine_v17_selection(plugin):
    original = unicodedata.normalize
    try:
        plugin._nfkc_patch_applied = False
        unicodedata.normalize = plugin._original_normalize

        assert plugin.get_ocr_engine(SimpleNamespace(ocr_engine="tesseract")) is None
        assert isinstance(plugin.get_ocr_engine(SimpleNamespace(ocr_engine="auto")), plugin.ChromeLensEngine)
        assert isinstance(plugin.get_ocr_engine(SimpleNamespace(ocr_engine="chromelens")), plugin.ChromeLensEngine)
        # Patch should be active after selecting chromelens
        assert unicodedata.normalize("NFKC", "x¹") == "x¹"

        # Selecting another engine must reverse the patch
        assert plugin.get_ocr_engine(SimpleNamespace(ocr_engine="tesseract")) is None
        assert plugin._nfkc_patch_applied is False
        assert unicodedata.normalize("NFKC", "x¹") == plugin._original_normalize("NFKC", "x¹")
    finally:
        unicodedata.normalize = original
        plugin._nfkc_patch_applied = False


def test_get_ocr_engine_v16_selection(plugin_v16):
    original = unicodedata.normalize
    try:
        plugin_v16._nfkc_patch_applied = False
        unicodedata.normalize = plugin_v16._original_normalize
        assert isinstance(plugin_v16.get_ocr_engine(), plugin_v16.ChromeLensEngine)
        assert unicodedata.normalize("NFKC", "x¹") == "x¹"
    finally:
        unicodedata.normalize = original
        plugin_v16._nfkc_patch_applied = False
