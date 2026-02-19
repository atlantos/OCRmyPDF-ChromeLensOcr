from tests.scripts.compare_ocr_texts import compare, levenshtein_distance, normalize_text


def test_normalize_text_maps_superscripts_and_strips_punct():
    assert normalize_text("Abc⁴, test.") == "abc 4 test"


def test_levenshtein_distance_basic():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "abc") == 0


def test_compare_metrics_high_for_similar_text():
    ref = "Пользовался летописями⁴."
    cand = "пользовался летописями 4"
    metrics = compare(ref, cand)

    assert metrics["char_sequence_ratio"] > 0.9
    assert metrics["token_f1"] > 0.9
    assert metrics["char_error_rate"] < 0.2
