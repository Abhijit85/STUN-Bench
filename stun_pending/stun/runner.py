"""Arm evaluation shared by P0-P6."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .common import (
    Entry, LLMIndexReranker, Query, Top1Reranker, clustered_bootstrap_diff, concat_experience,
    correct, default_render, description_entries, hit, load_catalog, load_client_experience,
    load_tests, make_retriever, mcnemar_exact, seed_everything, stratum_of, summarize, write_jsonl,
)


def base_args(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--catalog", required=True, help="catalog.jsonl: tool_id, name, description, category")
    p.add_argument("--tests", required=True, help="tests.jsonl: qid, query, gold[], group")
    p.add_argument("--heldout", required=True, help="heldout.json: list of tool ids (H)")
    p.add_argument("--clients-root", required=True, help="dir with seed42/, seed123/, seed456/ each holding client*.jsonl")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--retriever", default="jinaai/jina-embeddings-v2-base-en", help="model id/path, 'bm25', or 'hash' (smoke test)")
    p.add_argument("--reranker", default="llm", choices=["llm", "top1"])
    p.add_argument("--reranker-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--max-queries", type=int, default=0, help="debug: limit queries")
    return p


def load_common(a):
    catalog = load_catalog(a.catalog)
    tests = load_tests(a.tests)
    if a.max_queries:
        tests = tests[: a.max_queries]
    H = set(json.load(open(a.heldout)))
    seeds = [int(s) for s in a.seeds.split(",")]
    rer = Top1Reranker() if a.reranker == "top1" else LLMIndexReranker(a.reranker_model)
    return catalog, tests, H, seeds, rer


def exp_text_by_tool(entries: list[Entry]) -> dict[str, list[str]]:
    d = defaultdict(list)
    for e in entries:
        if e.source != "desc":
            d[e.tool_id].append(e.text)
    return d


class Index:
    """Entries plus a retriever over their texts."""

    def __init__(self, entries: list[Entry], kind: str, **kw):
        self.entries = entries
        self.r = make_retriever(kind, [e.text for e in entries], **kw) if entries else None

    def scores(self, q: str):
        return self.r.score(q)


def run_arm(name: str, seed: int, tests: list[Query], H: set[str], cand_fn: Callable[[Query], list[str]],
            reranker, render, shuffle_positions: int = 0) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    for q in tests:
        cands = cand_fn(q)
        if shuffle_positions:
            votes = []
            for _ in range(shuffle_positions):
                c = cands[:]
                rng.shuffle(c)
                votes.append(correct(reranker.choose(q.text, c, render), q.gold))
            acc = sum(votes) / len(votes)
        else:
            acc = correct(reranker.choose(q.text, cands, render), q.gold)
        rows.append({"arm": name, "seed": seed, "qid": q.qid, "stratum": stratum_of(q, H), "acc": acc,
                     "r5": hit(cands, q.gold), "cands": cands, "gold": q.gold})
    return rows


def paired_report(rows: list[dict], arm_a: str, arm_b: str, stratum: str = "heldout") -> dict:
    """Arm A minus arm B on one stratum, pooled over seeds; query- and tool-clustered intervals."""
    A = {(r["seed"], r["qid"]): r for r in rows if r["arm"] == arm_a and r["stratum"] == stratum}
    B = {(r["seed"], r["qid"]): r for r in rows if r["arm"] == arm_b and r["stratum"] == stratum}
    keys = sorted(set(A) & set(B))
    if not keys:
        return {}
    a = [A[k]["acc"] for k in keys]
    b = [B[k]["acc"] for k in keys]
    qc = [k[1] for k in keys]
    tc = [A[k]["gold"][0] for k in keys]  # first gold tool as cluster, as in the paper
    m, lo, hi = clustered_bootstrap_diff(a, b, qc)
    _, tlo, thi = clustered_bootstrap_diff(a, b, tc)
    n01, n10, p = mcnemar_exact([int(x >= 0.5) for x in a], [int(x >= 0.5) for x in b])
    return {"compare": f"{arm_a} - {arm_b}", "stratum": stratum, "n_pairs": len(keys), "diff_pts": round(100 * m, 1),
            "query_ci": [round(100 * lo, 1), round(100 * hi, 1)], "tool_ci": [round(100 * tlo, 1), round(100 * thi, 1)],
            "mcnemar": [n01, n10, p]}


def save(out: str, rows: list[dict], summaries: list[dict], tests: list[dict]) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(out) / "per_query.jsonl", rows)
    json.dump({"summary": summaries, "paired": tests}, open(Path(out) / "summary.json", "w"), indent=2)
    for s in summaries:
        print(json.dumps(s))
    for t in tests:
        print(json.dumps(t))
