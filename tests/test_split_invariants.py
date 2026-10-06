import gzip, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_heldout_count_and_hash():
    data=json.loads((ROOT/'release/heldout_tools.json').read_text())
    assert data['heldout_tool_count'] == 172
    assert data['heldout_tools_sha256'] == '8e84717a6e4fbe985efe03e41c258cd04d4f61917cc1efb15df9f5471e200260'

def test_query_subsets():
    data=json.loads((ROOT/'release/query_subsets.json').read_text())
    assert data['counts'] == {'heldout':309,'labeled':340,'mixed':80}
    assert len(data['rows']) == 729

def test_filter_counts():
    leak=sum(1 for _ in gzip.open(ROOT/'release/leak_filter_removed.jsonl.gz','rt'))
    stun=sum(1 for _ in gzip.open(ROOT/'release/stun_removed.jsonl.gz','rt'))
    assert leak == 1916
    assert stun == 103403
