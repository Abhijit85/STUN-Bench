"""STUN-Bench split helpers."""
import json
from pathlib import Path

def load_heldout_tools(path="release/heldout_tools.json"):
    return set(json.loads(Path(path).read_text())["heldout_tools"])

def subset_for_gold(gold_tools, heldout_tools):
    gold=list(gold_tools or [])
    in_h=[g for g in gold if g in heldout_tools]
    if in_h and len(in_h)==len(gold): return "heldout"
    if in_h: return "mixed"
    return "labeled"
