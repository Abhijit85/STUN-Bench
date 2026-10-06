"""Leak-filter helpers."""
from .normalization import normalize_query_exact

def exact_normalized_match(a: str, b: str) -> bool:
    return normalize_query_exact(a) == normalize_query_exact(b)
