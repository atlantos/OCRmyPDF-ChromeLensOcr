# OCRmyPDF-ChromeLens-Ocr

OCRmyPDF plugin that uses Google Lens (`v1/crupload`) as OCR backend.

## What It Does

- Sends rasterized page images to Google Lens and parses protobuf response into hOCR + text.
- Preserves hierarchical layout (paragraphs/lines/words), including rotation metadata (`textangle` in hOCR lines).
- Handles word separators from Lens response for better spacing fidelity.
- Includes optional de-hyphenation for line-broken words.
- Tries to preserve superscript glyphs (for example `¹²³`) by overriding OCRmyPDF's NFKC normalization path.

## Installation

Prerequisite: install `ocrmypdf`.

Install from Git:

```bash
pip install git+https://github.com/atlantos/OCRmyPDF-ChromeLens-Ocr.git
```

Install from PyPI:

```bash
pip install ocrmypdf-chromelens-ocr
```

## Usage

Basic usage:

```bash
ocrmypdf --plugin ocrmypdf_chromelens_ocr input.pdf output.pdf
```

Debug dump example:

```bash
ocrmypdf \
  --plugin ocrmypdf_chromelens_ocr \
  --keep-temporary-files \
  --chromelens-dump-debug \
  input.pdf output.pdf
```

## Plugin CLI Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--chromelens-no-dehyphenation` | Disable de-hyphenation across adjacent lines. | `false` |
| `--chromelens-max-dehyphen-len` | Max prefix/suffix length threshold for de-hyphenation merge. | `10` |
| `--chromelens-dump-debug` | Write raw request/response + parsed layout artifacts next to `*_ocr_hocr.*` temp files. Works only with `--keep-temporary-files`. | `false` |

## Current Implementation Defaults

These are hardcoded in `src/ocrmypdf_chromelens_ocr/plugin.py`:

- Upload format: `JPEG`
- JPEG quality: `95`
- Max upload long edge:
  - OCRmyPDF v16: `1600`
  - OCRmyPDF v17+: `1200`
- Fixed request locale/context:
  - language `en`, region `US`, timezone `America/New_York`
- Chrome-style request headers:
  - `x-browser-channel`, `x-browser-year`, `x-browser-copyright`, `x-browser-validation`

Note: OCRmyPDF language flags (`-l/--language`) are not propagated to Lens request context; Lens auto-detection is relied on.

## Compatibility

- Python `>=3.9`
- OCRmyPDF `>=16.0.0`
- Tested with OCRmyPDF 16 and 17 code paths

## Limitations

- Uses undocumented/private Google API and may break without notice.
- Requires network access and uploads page images to Google servers.
- OCR quality depends on Lens behavior and can vary by document type.
- `generate_pdf()` in the plugin is not implemented; OCR output is produced through hOCR/text path.

## Credits

- [chrome-lens-ocr](https://github.com/dimdenGD/chrome-lens-ocr) for protobuf/API reverse-engineering ideas.
- [OCRmyPDF-AppleOCR](https://github.com/mkyt/OCRmyPDF-AppleOCR) for plugin architecture inspiration.

## License

MIT
