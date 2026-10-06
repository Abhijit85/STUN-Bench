#!/usr/bin/env python3
import gzip, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
need=["release/heldout_tools.json","release/query_subsets.json","release/leak_filter_removed.jsonl.gz","release/stun_removed.jsonl.gz","release/SHA256SUMS"]
missing=[p for p in need if not (root/p).exists()]
if missing: raise SystemExit(f"missing release files: {missing}")
held=json.loads((root/"release/heldout_tools.json").read_text())
assert held["heldout_tool_count"] == 172
qs=json.loads((root/"release/query_subsets.json").read_text())
assert qs["counts"] == {"heldout":309,"labeled":340,"mixed":80}, qs["counts"]
for gz in ["release/leak_filter_removed.jsonl.gz","release/stun_removed.jsonl.gz"]:
    with gzip.open(root/gz,'rt') as f:
        for i,line in enumerate(f):
            rec=json.loads(line)
            assert 'query' not in rec and 'query_text' not in rec
            if i>10: break
print("release verification passed")
