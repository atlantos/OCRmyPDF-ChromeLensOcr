import io
import json
import logging
import math
import random
import re
import struct
import sys
import time
import unicodedata
import uuid
import base64
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from PIL import Image
from packaging.version import InvalidVersion, Version

__version__ = "1.0.5"

# ocrmypdf imports
import ocrmypdf
from ocrmypdf import OcrEngine, hookimpl
from ocrmypdf._exec import tesseract

class OcrEngineError(Exception):
    pass

logger = logging.getLogger(__name__)

# --- MONKEY PATCH: Disable NFKC Normalization ---
# ocrmypdf uses NFKC normalization which converts ¹ -> 1, ² -> 2, etc.
# We patch it to use NFC (Canonical) which preserves these glyphs.
_original_normalize = unicodedata.normalize
_nfkc_patch_applied = False

def _patched_normalize(form, unistr):
    if form == 'NFKC':
        # Fallback to NFC to preserve visual fidelity (superscripts, fractions)
        return _original_normalize('NFC', unistr)
    return _original_normalize(form, unistr)

def _apply_nfkc_patch():
    global _nfkc_patch_applied
    if _nfkc_patch_applied:
        return
    unicodedata.normalize = _patched_normalize
    _nfkc_patch_applied = True

def _remove_nfkc_patch():
    global _nfkc_patch_applied
    if not _nfkc_patch_applied:
        return
    unicodedata.normalize = _original_normalize
    _nfkc_patch_applied = False

# --- Constants ---
LENS_PROTO_ENDPOINT = 'https://lensfrontend-pa.googleapis.com/v1/crupload'
LENS_ACCEPT_LANGUAGE = 'en-US,en;q=0.9'
LENS_LOCALE_LANGUAGE = 'en'
LENS_LOCALE_REGION = 'US'
LENS_LOCALE_TIMEZONE = 'America/New_York'
LENS_CLIENT_PLATFORM = 6  # PLATFORM_LENS_OVERLAY
LENS_CLIENT_SURFACE = 4   # SURFACE_CHROMIUM
LENS_PAYLOAD_REQUEST_TYPE = 1  # REQUEST_TYPE_PDF
LENS_PAYLOAD_CONTENT_TYPE = "application/pdf"
LENS_PAYLOAD_PAGE_URL = "file:///document.pdf"
LENS_BROWSER_CHANNEL = "stable"

# Empirically best defaults from side-by-side benchmark on test corpus.
UPLOAD_FORMAT = "JPEG"
UPLOAD_JPEG_QUALITY = 95
MAX_DIMENSION_V16 = 1600
MAX_DIMENSION_V17 = 1200

LENS_PLATFORM_PROFILES = {
    "windows": {
        "api_key": "AIzaSyA2KlwBX3mkFo30om9LUFYQhpqLoa_BNhE",
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua_platform": "Windows",
        "browser_year": 2025,
    },
    "linux": {
        "api_key": "AIzaSyBqJZh-7pA44blAaAkH6490hUFOwX0KCYM",
        "user_agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua_platform": "Linux",
        "browser_year": 2025,
    },
    "macos": {
        "api_key": "AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua_platform": "macOS",
        "browser_year": 2025,
    },
}


def _next_request_uuid() -> int:
    # Lens request_id.uuid is uint64, so truncate UUIDv4 to 64 bits.
    return uuid.uuid4().int & ((1 << 64) - 1)


def _is_ocrmypdf_v17_or_newer() -> bool:
    raw_version = str(getattr(ocrmypdf, "__version__", "0"))
    try:
        return Version(raw_version) >= Version("17.0.0")
    except InvalidVersion:
        # Fallback for unconventional version strings.
        parts = re.findall(r"\d+", raw_version)
        major = int(parts[0]) if parts else 0
        return major >= 17


def _runtime_lens_platform() -> str:
    platform = sys.platform
    if platform.startswith("win"):
        return "windows"
    if platform.startswith("linux"):
        return "linux"
    if platform == "darwin":
        return "macos"

    logger.warning("Unknown platform '%s'; defaulting to macOS Lens profile", platform)
    return "macos"


def _lens_request_identity() -> tuple[str, str, int, str]:
    platform_name = _runtime_lens_platform()
    profile = LENS_PLATFORM_PROFILES[platform_name]
    return (
        profile["api_key"],
        profile["user_agent"],
        profile["browser_year"],
        platform_name,
    )


def _chrome_major_from_user_agent(user_agent: str) -> int:
    match = re.search(r"Chrome/(\d+)", user_agent)
    if match:
        return int(match.group(1))
    logger.warning("Unable to parse Chrome major version from user agent; defaulting to 144")
    return 144


def _sec_ch_ua_headers(platform_name: str, user_agent: str) -> dict[str, str]:
    chrome_major = _chrome_major_from_user_agent(user_agent)
    platform_value = LENS_PLATFORM_PROFILES[platform_name]["sec_ch_ua_platform"]
    return {
        "sec-ch-ua": (
            f'"Not:A-Brand";v="99", "Google Chrome";v="{chrome_major}", "Chromium";v="{chrome_major}"'
        ),
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": f'"{platform_value}"',
    }


def _generate_x_browser_validation(api_key: str, user_agent: str) -> str:
    # Mirrors Chrome's header generation: base64(sha1(api_key + user_agent))
    digest = hashlib.sha1((api_key + user_agent).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


# --- Utilities ---
def xml_sanitize(text):
    if not text: return ""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def bbox_str(bbox):
    return f"bbox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

def union_bboxes(bboxes):
    if not bboxes: return [0, 0, 0, 0]
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]

# --- Minimal Protobuf Tools ---
class ProtoWriter:
    def __init__(self):
        self.buf = bytearray()

    def _write_varint(self, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                self.buf.append(byte | 0x80)
            else:
                self.buf.append(byte)
                break

    def add_varint(self, field_id, value):
        self._write_varint((field_id << 3) | 0)
        self._write_varint(value)

    def add_bytes(self, field_id, data):
        self._write_varint((field_id << 3) | 2)
        self._write_varint(len(data))
        self.buf.extend(data)

    def add_string(self, field_id, text):
        self.add_bytes(field_id, text.encode('utf-8'))

    def add_message(self, field_id, writer):
        self.add_bytes(field_id, writer.buf)

    def get_bytes(self):
        return bytes(self.buf)

class MiniProto:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read_varint(self):
        result = 0
        shift = 0
        while True:
            if self.pos >= len(self.data): raise EOFError()
            byte = self.data[self.pos]
            self.pos += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                return result
            shift += 7

    def read_fixed32(self):
        if self.pos + 4 > len(self.data): raise EOFError()
        val = struct.unpack('<f', self.data[self.pos:self.pos+4])[0]
        self.pos += 4
        return val

    def read_bytes(self, length):
        if self.pos + length > len(self.data): raise EOFError()
        val = self.data[self.pos:self.pos+length]
        self.pos += length
        return val

    def parse(self):
        fields = {}
        while self.pos < len(self.data):
            try:
                tag = self.read_varint()
            except EOFError: break
            
            field_num = tag >> 3
            wire_type = tag & 0x07

            value = None
            if wire_type == 0: value = self.read_varint()
            elif wire_type == 1: self.read_bytes(8)
            elif wire_type == 2: value = self.read_bytes(self.read_varint())
            elif wire_type == 5: value = self.read_fixed32()
            else: break

            if value is not None:
                if field_num not in fields: fields[field_num] = []
                fields[field_num].append(value)
        return fields

# --- Plugin Definition ---

@hookimpl
def add_options(parser):
    group = parser.add_argument_group(
        "ChromeLens OCR", 
        "Options for the Google Lens OCR engine"
    )
    group.add_argument(
        "--chromelens-no-dehyphenation", 
        action="store_true", 
        help="Disable smart de-hyphenation (merging broken words across lines)"
    )
    group.add_argument(
        "--chromelens-max-dehyphen-len", 
        type=int, 
        default=10, 
        help="Maximum length of a word part to allow de-hyphenation (default: 10). "
             "If both parts are longer than this, they are assumed to be separate words/names."
    )
    group.add_argument(
        "--chromelens-dump-debug",
        action="store_true",
        help="Dump Chrome Lens request/response and parsed layout JSON next to *_ocr_hocr files "
             "(for debugging and cross-version comparison). Requires --keep-temporary-files.",
    )

class ChromeLensEngine(OcrEngine):
    @staticmethod
    def version():
        return __version__

    @classmethod
    def creator_tag(cls, options=None):
        return f"OCRmyPDF-ChromeLens-Ocr {cls.version()}"

    def __str__(self):
        return "ChromeLensOcr"

    def languages(self, options):
        if options and hasattr(options, 'languages') and options.languages:
            return options.languages
        return {"eng", "auto"}

    def get_orientation(self, input_file: Path, options):
        return tesseract.get_orientation(
            input_file,
            engine_mode=options.tesseract_oem,
            timeout=options.tesseract_non_ocr_timeout,
        )

    def get_deskew(self, input_file: Path, options) -> float:
        return 0.0

    def generate_hocr(self, input_file: Path, output_hocr: Path, output_text: Path = None, options=None):
        img_bytes = None
        width, height = 0, 0
        dpi = (300, 300)
        final_w, final_h = 0, 0
        dump_debug_requested = bool(getattr(options, "chromelens_dump_debug", False)) if options is not None else False
        keep_temporary_files = bool(getattr(options, "keep_temporary_files", False)) if options is not None else False
        if dump_debug_requested and not keep_temporary_files and not getattr(self, "_dump_debug_warning_logged", False):
            logger.warning("Ignoring --chromelens-dump-debug because --keep-temporary-files is not enabled.")
            self._dump_debug_warning_logged = True
        dump_debug = dump_debug_requested and keep_temporary_files

        try:
            with Image.open(input_file) as img:
                width, height = img.size
                dpi = img.info.get('dpi', (300, 300))
                max_dimension = MAX_DIMENSION_V17 if _is_ocrmypdf_v17_or_newer() else MAX_DIMENSION_V16
                process_img = img

                # Pillow's .convert('RGB') turns transparent pixels BLACK.
                # We must composite the image over a white background.
                if process_img.mode in ('RGBA', 'LA') or (process_img.mode == 'P' and 'transparency' in process_img.info):
                    process_img = process_img.convert('RGBA')
                    background = Image.new('RGB', process_img.size, (255, 255, 255))
                    background.paste(process_img, mask=process_img.split()[-1])
                    process_img = background
                elif process_img.mode != 'RGB':
                    process_img = process_img.convert('RGB')

                long_edge = max(width, height)
                scale = 1.0
                resize_reason = None
                if long_edge > max_dimension:
                    scale = max_dimension / long_edge
                    resize_reason = "downscaling"

                if scale != 1.0:
                    new_w = max(1, int(round(width * scale)))
                    new_h = max(1, int(round(height * scale)))
                    logger.debug(
                        "%s from %sx%s to %sx%s",
                        resize_reason or "resizing",
                        width,
                        height,
                        new_w,
                        new_h,
                    )
                    process_img = process_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                if UPLOAD_FORMAT == "JPEG":
                    process_img.save(
                        buffer,
                        format="JPEG",
                        quality=UPLOAD_JPEG_QUALITY,
                        optimize=False,
                        subsampling=0,
                    )
                else:
                    process_img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                final_w, final_h = process_img.size
        except Exception as e:
            raise OcrEngineError(f"Failed to process image: {e}") from e

        try:
            proto_payload = self._create_lens_proto_request(img_bytes, final_w, final_h)
            response_data = self._send_proto_request(proto_payload)
            layout_structure = self._strict_parse_hierarchical(
                response_data,
                orig_w=width,
                orig_h=height,
                upload_w=final_w,
                upload_h=final_h,
            )
            if dump_debug:
                self._dump_debug_artifacts(
                    output_hocr=output_hocr,
                    request_proto=proto_payload,
                    response_proto=response_data,
                    layout_structure=layout_structure,
                    orig_size=(width, height),
                    upload_size=(final_w, final_h),
                    dpi=dpi,
                )
            
            # --- De-hyphenation Configuration ---
            no_dehyphen = getattr(options, 'chromelens_no_dehyphenation', False)
            max_len = getattr(options, 'chromelens_max_dehyphen_len', 10)

            if not no_dehyphen:
                layout_structure = self._dehyphenate(layout_structure, max_len)
            
            # Sort by rotation/geometry to fix reading order
            layout_structure = self._sort_lines_by_rotation(layout_structure)
            
        except Exception as e:
            raise OcrEngineError(f"Google Lens logic failed: {e}") from e

        self._write_output_hierarchical(layout_structure, width, height, dpi, input_file, output_hocr, output_text)

    def _dump_debug_artifacts(
        self,
        output_hocr: Path,
        request_proto: bytes,
        response_proto: bytes,
        layout_structure,
        orig_size,
        upload_size,
        dpi,
    ):
        base = output_hocr.with_suffix('')
        request_path = base.with_name(base.name + "_chromelens_request.pb")
        response_path = base.with_name(base.name + "_chromelens_response.pb")
        layout_path = base.with_name(base.name + "_chromelens_layout.json")
        meta_path = base.with_name(base.name + "_chromelens_meta.json")

        try:
            request_path.write_bytes(request_proto)
            response_path.write_bytes(response_proto)
            layout_payload = {
                "paragraphs": layout_structure,
            }
            layout_path.write_text(
                json.dumps(layout_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            meta_payload = {
                "orig_size": {"width": int(orig_size[0]), "height": int(orig_size[1])},
                "upload_size": {"width": int(upload_size[0]), "height": int(upload_size[1])},
                "dpi": list(dpi) if isinstance(dpi, (tuple, list)) else dpi,
                "request_bytes": len(request_proto),
                "response_bytes": len(response_proto),
            }
            meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "ChromeLens debug dump written: %s, %s, %s, %s",
                request_path,
                response_path,
                layout_path,
                meta_path,
            )
        except Exception as e:
            logger.warning("Failed to write ChromeLens debug dump files: %s", e)

    def _sort_lines_by_rotation(self, paragraphs):
        for para in paragraphs:
            lines = para.get('lines', [])
            if not lines: continue
            
            total_rot = 0
            count = 0
            for line in lines:
                if 'rotation' in line:
                    total_rot += line['rotation']
                    count += 1
            
            avg_rot = total_rot / count if count > 0 else 0
            
            # Bottom-to-Top (-90 deg)
            if avg_rot < -0.8:
                lines.sort(key=lambda l: l['bbox'][1], reverse=True)
            # Top-to-Bottom Vertical (90 deg)
            elif avg_rot > 0.8:
                lines.sort(key=lambda l: l['bbox'][1])
                
        return paragraphs

    def _dehyphenate(self, paragraphs, max_len_threshold):
        for para in paragraphs:
            lines = para.get('lines', [])
            if len(lines) < 2: continue
            for i in range(len(lines) - 1):
                curr_line = lines[i]
                next_line = lines[i+1]
                if not curr_line['words'] or not next_line['words']: continue
                
                last_word = curr_line['words'][-1]
                first_next_word = next_line['words'][0]
                
                text = last_word['text']
                next_text = first_next_word['text']

                if not text or not text.endswith('-'):
                    continue

                if text.endswith(' -') or text.endswith(' –') or text.endswith(' —'):
                    continue

                if next_text and next_text[0].isupper():
                    continue

                prefix = text[:-1]
                suffix = next_text
                
                if len(prefix) > max_len_threshold and len(suffix) > max_len_threshold:
                    continue

                if not suffix: continue
                full_word = prefix + suffix
                
                if len(prefix) > len(suffix):
                    last_word['text'] = full_word
                    # Preserve separator that belongs to the suffix token.
                    last_word['sep'] = first_next_word.get('sep')
                    first_next_word['text'] = ""
                    first_next_word['sep'] = None
                else:
                    first_next_word['text'] = full_word
                    last_word['text'] = ""
                    last_word['sep'] = None
                    
        return paragraphs

    def _create_lens_proto_request(self, image_bytes, width, height):
        # 1. Request ID
        request_id = ProtoWriter()
        request_id.add_varint(1, _next_request_uuid())
        request_id.add_varint(2, 1) 
        
        # 2. Client Context
        locale_context = ProtoWriter()
        locale_context.add_string(1, LENS_LOCALE_LANGUAGE)
        locale_context.add_string(2, LENS_LOCALE_REGION)
        locale_context.add_string(3, LENS_LOCALE_TIMEZONE)
        
        client_context = ProtoWriter()
        client_context.add_varint(1, LENS_CLIENT_PLATFORM)
        client_context.add_varint(2, LENS_CLIENT_SURFACE)
        client_context.add_message(4, locale_context)
        
        # 3. Request Context
        request_context = ProtoWriter()
        request_context.add_message(3, request_id)
        request_context.add_message(4, client_context)
        
        # 4. Image Data
        image_payload = ProtoWriter()
        image_payload.add_bytes(1, image_bytes)
        
        image_metadata = ProtoWriter()
        image_metadata.add_varint(1, width)
        image_metadata.add_varint(2, height)
        
        image_data = ProtoWriter()
        image_data.add_message(1, image_payload)
        image_data.add_message(3, image_metadata) 
        
        # 6. Final Objects Request
        objects_request = ProtoWriter()
        objects_request.add_message(1, request_context)
        objects_request.add_message(3, image_data)
        payload = ProtoWriter()
        payload.add_varint(6, LENS_PAYLOAD_REQUEST_TYPE)
        payload.add_string(4, LENS_PAYLOAD_CONTENT_TYPE)
        payload.add_string(5, LENS_PAYLOAD_PAGE_URL)
        objects_request.add_message(4, payload)
        
        server_request = ProtoWriter()
        server_request.add_message(1, objects_request)
        
        return server_request.get_bytes()

    def _send_proto_request(self, proto_bytes):
        api_key, user_agent, browser_year, platform_name = _lens_request_identity()
        browser_validation = _generate_x_browser_validation(api_key, user_agent)
        headers = {
            'Content-Type': 'application/x-protobuf',
            'X-Goog-Api-Key': api_key,
            'User-Agent': user_agent,
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': LENS_ACCEPT_LANGUAGE,
            'x-browser-channel': LENS_BROWSER_CHANNEL,
            'x-browser-year': str(browser_year),
            'x-browser-copyright': f'Copyright {browser_year} Google LLC. All rights reserved.',
            'x-browser-validation': browser_validation,
        }
        headers.update(_sec_ch_ua_headers(platform_name, user_agent))
        
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                # Random sleep to mitigate rate limiting
                time.sleep(random.uniform(0.5, 1.5))

                response = requests.post(
                    LENS_PROTO_ENDPOINT, 
                    data=proto_bytes, 
                    headers=headers, 
                    timeout=120
                )
                
                if response.status_code == 200:
                    return response.content
                
                # If non-200, log and allow retry
                error_msg = f"Server returned {response.status_code}. Response: {response.text[:200]}"
                logger.warning(
                    "ChromeLens API attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    error_msg,
                )
                last_exception = OcrEngineError(error_msg)

            except (requests.RequestException, OcrEngineError) as e:
                logger.warning(
                    "ChromeLens API connection failed (Attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                last_exception = OcrEngineError(f"Network error: {e}")

            # Backoff logic if we haven't succeeded yet
            if attempt < max_retries - 1:
                sleep_time = random.uniform(4.0, 10.0)
                logger.info("Retrying in %.1f seconds...", sleep_time)
                time.sleep(sleep_time)

        # If loop finishes without success
        raise last_exception or OcrEngineError("Unknown failure after retries")

    def _parse_geometry(self, box_bytes, img_w, img_h, upload_w=None, upload_h=None):
        box = MiniProto(box_bytes).parse()
        cx = box.get(1, [0.5])[0]
        cy = box.get(2, [0.5])[0]
        w  = box.get(3, [0.0])[0]
        h  = box.get(4, [0.0])[0]
        rotation = box.get(5, [0.0])[0]
        coordinate_type = box.get(6, [1])[0]
        upload_w = upload_w or img_w
        upload_h = upload_h or img_h
        scale_x = (img_w / upload_w) if upload_w else 1.0
        scale_y = (img_h / upload_h) if upload_h else 1.0

        # Geometry may be normalized (0..1) or absolute image coordinates.
        # Absolute IMAGE coords are relative to the uploaded image, which may
        # have been downscaled before sending to Lens.
        if coordinate_type == 2:  # IMAGE
            px_cx = cx * scale_x
            px_cy = cy * scale_y
            px_w = w * scale_x
            px_h = h * scale_y
        elif coordinate_type == 1:  # NORMALIZED
            px_cx = cx * img_w
            px_cy = cy * img_h
            px_w = w * img_w
            px_h = h * img_h
        else:
            # Unknown/unspecified type: infer from value ranges.
            looks_normalized = (
                0.0 <= cx <= 1.0 and
                0.0 <= cy <= 1.0 and
                0.0 <= w <= 1.5 and
                0.0 <= h <= 1.5
            )
            if looks_normalized:
                px_cx = cx * img_w
                px_cy = cy * img_h
                px_w = w * img_w
                px_h = h * img_h
            else:
                px_cx = cx * scale_x
                px_cy = cy * scale_y
                px_w = w * scale_x
                px_h = h * scale_y
        
        if abs(rotation) > 0.1:
            cos_r = abs(math.cos(rotation))
            sin_r = abs(math.sin(rotation))
            new_w = (px_w * cos_r) + (px_h * sin_r)
            new_h = (px_w * sin_r) + (px_h * cos_r)
            px_w = new_w
            px_h = new_h

        x0 = int(px_cx - (px_w / 2))
        y0 = int(px_cy - (px_h / 2))
        x1 = int(px_cx + (px_w / 2))
        y1 = int(px_cy + (px_h / 2))
        
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img_w, x1)
        y1 = min(img_h, y1)
        return ([x0, y0, x1, y1], rotation)

    def _strict_parse_hierarchical(self, binary_data, orig_w, orig_h, upload_w=None, upload_h=None):
        paragraphs = []
        root = MiniProto(binary_data).parse()
        if 2 not in root: return []
        obj_resp = MiniProto(root[2][0]).parse()
        if 3 not in obj_resp: return []
        text_proto = MiniProto(obj_resp[3][0]).parse()
        if 1 not in text_proto: return []
        layout = MiniProto(text_proto[1][0]).parse()
        if 1 not in layout: return []
        
        # Superscript set for splitting logic
        superscripts = set("⁰¹²³⁴⁵⁶⁷⁸⁹")

        for para_bytes in layout[1]:
            para = MiniProto(para_bytes).parse()
            para_struct = {'lines': [], 'bbox': None, 'rotation': 0.0}
            if 3 in para:
                geo = MiniProto(para[3][0]).parse()
                if 1 in geo:
                    bbox, rot = self._parse_geometry(geo[1][0], orig_w, orig_h, upload_w, upload_h)
                    para_struct['bbox'] = bbox
                    para_struct['rotation'] = rot
            if 2 not in para: continue
            
            for line_bytes in para[2]:
                line = MiniProto(line_bytes).parse()
                line_struct = {'words': [], 'bbox': None, 'rotation': 0.0}
                if 2 in line:
                    geo = MiniProto(line[2][0]).parse()
                    if 1 in geo:
                        bbox, rot = self._parse_geometry(geo[1][0], orig_w, orig_h, upload_w, upload_h)
                        line_struct['bbox'] = bbox
                        line_struct['rotation'] = rot
                if 1 not in line: continue
                
                for word_bytes in line[1]:
                    word = MiniProto(word_bytes).parse()
                    if 2 not in word: continue
                    try:
                        text_val = word[2][0].decode('utf-8')
                    except Exception: continue
                    text_val = xml_sanitize(text_val)
                    if not text_val.strip(): continue
                    sep_val = None
                    if 3 in word:
                        try:
                            sep_val = word[3][0].decode('utf-8')
                            sep_val = xml_sanitize(sep_val)
                        except Exception:
                            sep_val = None
                    word_bbox = None
                    if 4 in word:
                        geo = MiniProto(word[4][0]).parse()
                        if 1 in geo:
                            word_bbox, _ = self._parse_geometry(geo[1][0], orig_w, orig_h, upload_w, upload_h)
                    
                    # --- Relaxed Parsing: Inherit bbox if missing ---
                    if word_bbox is None:
                        if line_struct['bbox']:
                            word_bbox, _ = line_struct['bbox'], 0
                        else:
                            word_bbox = [0, 0, 1, 1]

                    # 1. Find the split index (start of the superscript suffix)
                    split_idx = len(text_val)
                    while split_idx > 0 and text_val[split_idx-1] in superscripts:
                        split_idx -= 1
                    
                    # 2. Only split if we found superscripts AND there is a base word preceding them
                    if split_idx < len(text_val) and split_idx > 0:
                        base_text = text_val[:split_idx]
                        suffix_text = text_val[split_idx:]
                        
                        x0, y0, x1, y1 = word_bbox
                        total_w = x1 - x0
                        full_len = len(text_val)
                        suffix_len = len(suffix_text)
                        
                        # Calculate width of the suffix based on character count ratio
                        # We cap the suffix width at 40% of the word to prevent layout issues
                        ratio = suffix_len / full_len
                        split_w = int(total_w * ratio)
                        
                        # Safety cap
                        if split_w > (total_w * 0.4):
                            split_w = int(total_w * 0.4)
                            
                        # Ensure at least 1 pixel width
                        split_w = max(1, split_w)
                        
                        split_x = x1 - split_w
                        
                        # Add Base
                        bbox_base = [x0, y0, split_x, y1]
                        # Empty separator prevents fallback space insertion between
                        # base and superscript suffix.
                        line_struct['words'].append({'text': base_text, 'bbox': bbox_base, 'sep': ''})
                        
                        # Add Suffix (The superscript part)
                        bbox_suff = [split_x, y0, x1, y1]
                        line_struct['words'].append({'text': suffix_text, 'bbox': bbox_suff, 'sep': sep_val})
                    else:
                        # No superscript suffix found, or the whole word is superscripts (leave as is)
                        line_struct['words'].append({'text': text_val, 'bbox': word_bbox, 'sep': sep_val})
                
                if not line_struct['bbox'] and line_struct['words']:
                    line_struct['bbox'] = union_bboxes([w['bbox'] for w in line_struct['words']])
                if line_struct['words']:
                    para_struct['lines'].append(line_struct)

            if not para_struct['bbox'] and para_struct['lines']:
                para_struct['bbox'] = union_bboxes([l['bbox'] for l in para_struct['lines']])
            if para_struct['lines']:
                paragraphs.append(para_struct)
        return paragraphs

    def _line_title(self, line):
        bbox = line.get('bbox') or [0, 0, 0, 0]
        title = bbox_str(bbox)
        rotation = line.get('rotation')
        if rotation is None:
            return title

        # hOCR textangle is in degrees, counter-clockwise from horizontal.
        degrees = ((math.degrees(rotation) + 180.0) % 360.0) - 180.0
        if abs(degrees) < 0.1:
            return title
        return f"{title}; textangle {degrees:.2f}"

    @staticmethod
    def _line_plain_text(words):
        visible_words = [w for w in words if w.get('text')]
        if not visible_words:
            return ""

        parts = []
        for idx, word in enumerate(visible_words):
            parts.append(word.get('text', ''))
            sep = word.get('sep')
            if sep is not None:
                parts.append(sep)
            elif idx < len(visible_words) - 1:
                parts.append(' ')
        return "".join(parts).rstrip()

    def _write_output_hierarchical(self, paragraphs, img_w, img_h, dpi, input_file, output_hocr, output_text):
        html = ET.Element("html", {"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": "und"})
        head = ET.SubElement(html, "head")
        safe_title = xml_sanitize(str(input_file))
        ET.SubElement(head, "title").text = safe_title
        ET.SubElement(head, "meta", {"name": "ocr-system", "content": "chrome-lens-pure-py"})
        body = ET.SubElement(html, "body")

        # Format DPI for scan_res with safety fallback
        if not dpi:
            dpi = (300, 300)

        def _safe_scan_res(value, fallback):
            try:
                val = float(value)
            except (TypeError, ValueError):
                val = fallback
            if not math.isfinite(val) or val <= 0:
                val = fallback
            return max(1, int(round(val)))

        dpi_x_raw = dpi[0] if isinstance(dpi, (tuple, list)) and len(dpi) > 0 else 300
        dpi_x = _safe_scan_res(dpi_x_raw, 300.0)
        dpi_y_raw = dpi[1] if isinstance(dpi, (tuple, list)) and len(dpi) > 1 else dpi_x
        dpi_y = _safe_scan_res(dpi_y_raw, float(dpi_x))

        page_div = ET.SubElement(body, "div", {
            "class": "ocr_page", 
            "id": "page_1", 
            "title": f"bbox 0 0 {img_w} {img_h}; ppageno 0; scan_res {dpi_x} {dpi_y}"
        })

        full_text_lines = []

        for i, para in enumerate(paragraphs):
            carea_div = ET.SubElement(page_div, "div", {"class": "ocr_carea", "id": f"block_{i+1}", "title": bbox_str(para['bbox'])})
            par_p = ET.SubElement(carea_div, "p", {"class": "ocr_par", "id": f"par_{i+1}", "title": bbox_str(para['bbox'])})

            for j, line in enumerate(para['lines']):
                line_span = ET.SubElement(par_p, "span", {"class": "ocr_line", "id": f"line_{i+1}_{j+1}", "title": self._line_title(line)})
                for k, word in enumerate(line['words']):
                    if not word['text']: continue
                    word_span = ET.SubElement(line_span, "span", {"class": "ocrx_word", "id": f"word_{i+1}_{j+1}_{k+1}", "title": bbox_str(word['bbox'])})
                    sep = word.get('sep') or ""
                    # Preserve punctuation separators in hOCR word text.
                    word_sep_text = sep.strip()
                    word_span.text = f"{word['text']}{word_sep_text}" if word_sep_text else word['text']
                
                line_text = self._line_plain_text(line['words'])
                if line_text:
                    full_text_lines.append(line_text)
            
            full_text_lines.append("")

        tree = ET.ElementTree(html)
        with open(output_hocr, "wb") as f:
            f.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write(b"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n")
            tree.write(f, encoding="utf-8", xml_declaration=False)

        if output_text:
            with open(output_text, "w", encoding="utf-8") as f:
                f.write("\n".join(full_text_lines).strip())

    def generate_pdf(self, input_file: Path, output_pdf: Path, output_text: Path, options=None):
        raise NotImplementedError()

if _is_ocrmypdf_v17_or_newer():
    @hookimpl
    def get_ocr_engine(options):
        if options is not None:
            ocr_engine = getattr(options, "ocr_engine", "auto")
            # If the user explicitly requested another engine, do not return this one
            if ocr_engine not in ("auto", "chromelens"):
                _remove_nfkc_patch()
                return None
        _apply_nfkc_patch()
        return ChromeLensEngine()
else:
    @hookimpl
    def get_ocr_engine():
        _apply_nfkc_patch()
        return ChromeLensEngine()
