#!/usr/bin/env python3
"""Compare OCR output against a reference text with robust similarity metrics."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(SUPERSCRIPT_DIGITS)
    text = text.replace("\u00ad", "")  # soft hyphen
    # Keep footnote markers comparable after superscript normalization
    # (e.g., "летописями⁴" -> "летописями 4").
    text = re.sub(r"(?<=\w)(?=\d)|(?<=\d)(?=\w)", " ", text, flags=re.UNICODE)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            ins = previous[j] + 1
            delete = current[j - 1] + 1
            subst = previous[j - 1] + (ca != cb)
            current.append(min(ins, delete, subst))
        previous = current
    return previous[-1]


def token_f1(reference_tokens: list[str], candidate_tokens: list[str]) -> float:
    ref_counts = Counter(reference_tokens)
    cand_counts = Counter(candidate_tokens)
    overlap = sum((ref_counts & cand_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / max(1, sum(cand_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compare(reference_text: str, candidate_text: str) -> dict[str, float | int]:
    ref_norm = normalize_text(reference_text)
    cand_norm = normalize_text(candidate_text)

    seq_ratio = difflib.SequenceMatcher(None, ref_norm, cand_norm).ratio()
    # Exact Levenshtein is O(N*M); for long OCR texts this can be very slow.
    # Use exact distance on moderate input sizes and a fast ratio-based
    # approximation on large inputs.
    exact_levenshtein = (len(ref_norm) * len(cand_norm)) <= 5_000_000
    if exact_levenshtein:
        edit_dist = levenshtein_distance(ref_norm, cand_norm)
        cer = edit_dist / max(1, len(ref_norm))
    else:
        edit_dist = int(round((1.0 - seq_ratio) * len(ref_norm)))
        cer = 1.0 - seq_ratio

    ref_tokens = ref_norm.split()
    cand_tokens = cand_norm.split()
    f1 = token_f1(ref_tokens, cand_tokens)
    jaccard = (
        len(set(ref_tokens) & set(cand_tokens)) / max(1, len(set(ref_tokens) | set(cand_tokens)))
    )

    return {
        "char_sequence_ratio": seq_ratio,
        "char_error_rate": cer,
        "token_f1": f1,
        "token_jaccard": jaccard,
        "reference_chars_normalized": len(ref_norm),
        "candidate_chars_normalized": len(cand_norm),
        "reference_tokens": len(ref_tokens),
        "candidate_tokens": len(cand_tokens),
        "char_error_rate_exact": exact_levenshtein,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Reference OCR text file")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate OCR text file")
    parser.add_argument("--json-out", type=Path, help="Write metrics as JSON to this path")
    parser.add_argument("--min-char-ratio", type=float, default=None, help="Fail if char ratio is below this value")
    parser.add_argument("--min-token-f1", type=float, default=None, help="Fail if token F1 is below this value")
    args = parser.parse_args()

    reference_text = args.reference.read_text(encoding="utf-8", errors="ignore")
    candidate_text = args.candidate.read_text(encoding="utf-8", errors="ignore")
    metrics = compare(reference_text, candidate_text)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.min_char_ratio is not None and metrics["char_sequence_ratio"] < args.min_char_ratio:
        print(
            f"char_sequence_ratio {metrics['char_sequence_ratio']:.4f} is below {args.min_char_ratio:.4f}",
            file=sys.stderr,
        )
        return 2

    if args.min_token_f1 is not None and metrics["token_f1"] < args.min_token_f1:
        print(f"token_f1 {metrics['token_f1']:.4f} is below {args.min_token_f1:.4f}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
