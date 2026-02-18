from news_sentinel.data.preprocess import clean_text, whitespace_tokenize


def test_clean_text_normalizes_symbols_and_spaces() -> None:
    assert clean_text(" Hello,   World!! 123 ") == "hello world 123"


def test_whitespace_tokenize_handles_empty() -> None:
    assert whitespace_tokenize("") == []


def test_whitespace_tokenize_splits_normalized_text() -> None:
    assert whitespace_tokenize("Business & Markets") == ["business", "markets"]
