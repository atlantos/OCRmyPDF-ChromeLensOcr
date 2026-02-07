# OCRmyPDF-ChromeLens-Ocr

A plugin for [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) that uses the **Google Lens API** to perform Optical Character Recognition (OCR). 

## Features

-   **High Accuracy**: Leverages Google's advanced Lens models.
-   **Structure Preservation**: Correctly handles multi-column layouts and vertical text flows via strict API parsing and rotation-aware sorting.
-   **Smart De-hyphenation**: Merges words broken across lines while respecting punctuation dashes.

***

## ⚠️ Compatibility Warning

**This plugin is not currently compatible with OCRmyPDF v17 or higher.**

OCRmyPDF v17 introduced breaking changes to the plugin interface (migrating from `argparse.Namespace` to `OcrOptions` objects). Please use an older version of OCRmyPDF (v16.x or lower) until this plugin is updated.

To install a compatible version:
```bash
pip install "ocrmypdf<17"
```

## Installation

### Prerequisites
You must have `ocrmypdf` installed.

### Install from Git
```bash
pip install git+https://github.com/atlantos/OCRmyPDF-ChromeLens-Ocr.git
```

### Install from Pip
```bash
pip install ocrmypdf-chromelens-ocr
```

## Usage

To use this engine, pass the plugin name to OCRmyPDF. You generally do not need to specify a language, as Google Lens auto-detects it.

```bash
ocrmypdf --plugin ocrmypdf_chromelens_ocr input.pdf output.pdf
```

### Configuration Options

You can configure the behavior of the plugin using the following command-line arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--chromelens-no-dehyphenation` | Disables the logic that merges hyphenated words across lines. Useful if you prefer raw output. | Disabled |
| `--chromelens-max-dehyphen-len` | The **maximum** length of word parts allowed for de-hyphenation. If both the prefix (before hyphen) and suffix (after hyphen) are *longer* than this value, the plugin assumes it is a compound word or dash separator and will *not* merge them. | 10 |

**Example: Disable de-hyphenation**
```bash
ocrmypdf --plugin ocrmypdf_chromelens_ocr --chromelens-no-dehyphenation input.pdf output.pdf
```

**Example: Stricter de-hyphenation (only merge very short breaks)**
```bash
ocrmypdf --plugin ocrmypdf_chromelens_ocr --chromelens-max-dehyphen-len 4 input.pdf output.pdf
```

## Credits & Acknowledgements

This project is a Python port and adaptation based on ideas and logic from:

1.  **[chrome-lens-ocr](https://github.com/dimdenGD/chrome-lens-ocr)**:
    -   Provided the critical reverse-engineering of the Google Lens Protobuf API (`v1/crupload`).
    -   Logic for strict layout parsing and request structure.

2.  **[OCRmyPDF-AppleOCR](https://github.com/mkyt/OCRmyPDF-AppleOCR)**:
    -   Provided the architectural inspiration for creating an OCRmyPDF plugin that offloads recognition to an external engine.

## Disclaimer

This software is for educational purposes. It uses an undocumented private API from Google. 
-   **Privacy**: Your images are uploaded to Google servers. Do not process sensitive/confidential data.
-   **Stability**: The API may change or break at any time without notice.
-   **Rate Limits**: Excessive use may result in your IP being temporarily blocked by Google.

## License

MIT License
