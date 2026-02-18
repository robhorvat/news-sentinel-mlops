import re
from typing import List

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def whitespace_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return clean_text(text).split(" ")
