"""Candidate collection policy used by corrected STUN-Bench tables."""

def walk_down_distinct_tools(ranked_entries, k=5, cap=200):
    seen=set(); out=[]; depth=0
    for entry in ranked_entries[:cap]:
        depth += 1
        tool = entry["tool_id"] if isinstance(entry, dict) else entry[0]
        if tool in seen: continue
        seen.add(tool); out.append(entry)
        if len(out) == k: break
    return out, {"distinct_count": len(out), "depth_reached": depth, "shortfall": len(out) < k}
