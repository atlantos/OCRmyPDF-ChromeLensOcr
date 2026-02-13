import io
import logging
import math
import random
import re
import struct
import time
import unicodedata
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from PIL import Image

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

def _patched_normalize(form, unistr):
    if form == 'NFKC':
        # Fallback to NFC to preserve visual fidelity (superscripts, fractions)
        return _original_normalize('NFC', unistr)
    return _original_normalize(form, unistr)

# Apply patch globally
unicodedata.normalize = _patched_normalize

# --- Constants ---
LENS_PROTO_ENDPOINT = 'https://lensfrontend-pa.googleapis.com/v1/crupload'
LENS_API_KEY = 'AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY'
LENS_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'


def _next_request_uuid() -> int:
    # Lens request_id.uuid is uint64, so truncate UUIDv4 to 64 bits.
    return uuid.uuid4().int & ((1 << 64) - 1)

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

class ChromeLensEngine(OcrEngine):
    @staticmethod
    def version():
        return "1.0.3"

    @classmethod
    def creator_tag(cls, options=None):
        return f"OCRmyPDF-ChromeLens-Ocr {cls.version()}"

    def __str__(self):
        return "ChromeLensOcr"

    def engine_name(self):
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

        try:
            with Image.open(input_file) as img:
                width, height = img.size
                dpi = img.info.get('dpi', (300, 300))
                MAX_DIMENSION = 4800
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

                if max(width, height) > MAX_DIMENSION:
                    scale = MAX_DIMENSION / max(width, height)
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    logger.debug(f"Downscaling from {width}x{height} to {new_w}x{new_h}")
                    process_img = process_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                process_img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                final_w, final_h = process_img.size
        except Exception as e:
            raise OcrEngineError(f"Failed to process image: {e}")

        try:
            proto_payload = self._create_lens_proto_request(img_bytes, final_w, final_h)
            response_data = self._send_proto_request(proto_payload)
            layout_structure = self._strict_parse_hierarchical(response_data, width, height)
            
            # --- De-hyphenation Configuration ---
            no_dehyphen = getattr(options, 'chromelens_no_dehyphenation', False)
            max_len = getattr(options, 'chromelens_max_dehyphen_len', 10)

            if not no_dehyphen:
                layout_structure = self._dehyphenate(layout_structure, max_len)
            
            # Sort by rotation/geometry to fix reading order
            layout_structure = self._sort_lines_by_rotation(layout_structure)
            
        except Exception as e:
            raise OcrEngineError(f"Google Lens logic failed: {e}")

        self._write_output_hierarchical(layout_structure, width, height, dpi, input_file, output_hocr, output_text)

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
                    first_next_word['text'] = ""
                else:
                    first_next_word['text'] = full_word
                    last_word['text'] = ""
                    
        return paragraphs

    def _create_lens_proto_request(self, image_bytes, width, height):
        # 1. Request ID
        request_id = ProtoWriter()
        request_id.add_varint(1, _next_request_uuid())
        request_id.add_varint(2, 1) 
        
        # 2. Client Context
        locale_context = ProtoWriter()
        locale_context.add_string(1, 'en') 
        locale_context.add_string(2, 'US') 
        locale_context.add_string(3, 'America/New_York') 
        
        client_context = ProtoWriter()
        client_context.add_varint(1, 6) 
        client_context.add_varint(2, 4) 
        client_context.add_message(4, locale_context)
        # Note: We aren't adding filters here, PDF mode usually implies full text
        
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
        
        # 5. Payload
        # We tell the server this is a PDF request (RequestType = 1)
        payload = ProtoWriter()
        payload.add_varint(6, 1) # field 6 is request_type, 1 is REQUEST_TYPE_PDF
        payload.add_string(4, "application/pdf") # field 4 is content_type
        payload.add_string(5, "file:///document.pdf")
        
        # 6. Final Objects Request
        objects_request = ProtoWriter()
        objects_request.add_message(1, request_context)
        objects_request.add_message(3, image_data)
        objects_request.add_message(4, payload) # Add the payload here
        
        server_request = ProtoWriter()
        server_request.add_message(1, objects_request)
        
        return server_request.get_bytes()

    def _send_proto_request(self, proto_bytes):
        headers = {
            'Content-Type': 'application/x-protobuf',
            'X-Goog-Api-Key': LENS_API_KEY,
            'User-Agent': LENS_USER_AGENT,
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
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
                logger.warning(f"ChromeLens API attempt {attempt+1}/{max_retries} failed: {error_msg}")
                last_exception = OcrEngineError(error_msg)

            except (requests.RequestException, OcrEngineError) as e:
                logger.warning(f"ChromeLens API connection failed (Attempt {attempt+1}/{max_retries}): {e}")
                last_exception = OcrEngineError(f"Network error: {e}")

            # Backoff logic if we haven't succeeded yet
            if attempt < max_retries - 1:
                sleep_time = random.uniform(4.0, 10.0)
                logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

        # If loop finishes without success
        raise last_exception or OcrEngineError("Unknown failure after retries")

    def _parse_geometry(self, box_bytes, img_w, img_h):
        box = MiniProto(box_bytes).parse()
        cx = box.get(1, [0.5])[0]
        cy = box.get(2, [0.5])[0]
        w  = box.get(3, [0.0])[0]
        h  = box.get(4, [0.0])[0]
        rotation = box.get(5, [0.0])[0]
        
        px_cx = cx * img_w
        px_cy = cy * img_h
        px_w = w * img_w
        px_h = h * img_h

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

    def _strict_parse_hierarchical(self, binary_data, img_w, img_h):
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
                    bbox, rot = self._parse_geometry(geo[1][0], img_w, img_h)
                    para_struct['bbox'] = bbox
                    para_struct['rotation'] = rot
            if 2 not in para: continue
            
            for line_bytes in para[2]:
                line = MiniProto(line_bytes).parse()
                line_struct = {'words': [], 'bbox': None, 'rotation': 0.0}
                if 2 in line:
                    geo = MiniProto(line[2][0]).parse()
                    if 1 in geo:
                        bbox, rot = self._parse_geometry(geo[1][0], img_w, img_h)
                        line_struct['bbox'] = bbox
                        line_struct['rotation'] = rot
                if 1 not in line: continue
                
                for word_bytes in line[1]:
                    word = MiniProto(word_bytes).parse()
                    if 2 not in word: continue
                    try:
                        text_val = word[2][0].decode('utf-8')
                    except: continue
                    text_val = xml_sanitize(text_val)
                    if not text_val.strip(): continue
                    word_bbox = None
                    if 4 in word:
                        geo = MiniProto(word[4][0]).parse()
                        if 1 in geo:
                            word_bbox, _ = self._parse_geometry(geo[1][0], img_w, img_h)
                    
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
                        line_struct['words'].append({'text': base_text, 'bbox': bbox_base})
                        
                        # Add Suffix (The superscript part)
                        bbox_suff = [split_x, y0, x1, y1]
                        line_struct['words'].append({'text': suffix_text, 'bbox': bbox_suff})
                    else:
                        # No superscript suffix found, or the whole word is superscripts (leave as is)
                        line_struct['words'].append({'text': text_val, 'bbox': word_bbox})
                
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
        
        dpi_x = int(dpi[0]) if isinstance(dpi, (tuple, list)) and dpi[0] else 300
        dpi_y = int(dpi[1]) if isinstance(dpi, (tuple, list)) and len(dpi) > 1 and dpi[1] else dpi_x

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
                line_text = []
                for k, word in enumerate(line['words']):
                    if not word['text']: continue
                    word_span = ET.SubElement(line_span, "span", {"class": "ocrx_word", "id": f"word_{i+1}_{j+1}_{k+1}", "title": bbox_str(word['bbox'])})
                    word_span.text = word['text']
                    line_text.append(word['text'])
                
                if line_text:
                    full_text_lines.append(" ".join(line_text))
            
            full_text_lines.append("")

        tree = ET.ElementTree(html)
        with open(output_hocr, "wb") as f:
            f.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write(b"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n")
            tree.write(f, encoding="utf-8", xml_declaration=False)

        if output_text:
            with open(output_text, "w", encoding="utf-8") as f:
                f.write("\n".join(full_text_lines).strip())

    def generate_pdf(self, input_file: Path, output_pdf: Path, hocr_file: Path, recalculate_coords: bool = False, options=None):
        raise NotImplementedError()

if ocrmypdf.__version__ >= "17.0.0":
    @hookimpl
    def get_ocr_engine(options):
        if options is not None:
            ocr_engine = getattr(options, "ocr_engine", "auto")
            # If the user explicitly requested another engine, do not return this one
            if ocr_engine not in ("auto", "chromelens"):
                return None
        return ChromeLensEngine()
else:
    @hookimpl
    def get_ocr_engine():
        return ChromeLensEngine()
