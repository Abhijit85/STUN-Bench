"""Normalization rules for STUN-Bench release artifacts."""
import hashlib, re

def normalize_tool_id(value: str) -> str:
    return re.sub(r"[_\-]+", " ", (value or "").strip().lower())

def normalize_query_exact(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())

def sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode()).hexdigest()
