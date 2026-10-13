"""Shared pieces for the STUN-Bench pending experiments (P0-P6).

Everything here is standalone. Plug your existing harness in through three
adapters: the data loader (JSONL files described in README), the embedding
model (any sentence-transformers id or a local checkpoint path), and the
reranker (default: the frozen-LLM index scorer used in the paper).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@dataclass
class Entry:
    """One retrievable unit. source is 'desc' or 'exp' (or 'gen' for generated)."""

    tool_id: str
    text: str
    source: str
    meta: dict = field(default_factory=dict)


@dataclass
class Query:
    qid: str
    text: str
    gold: list[str]
    group: str = ""


def load_catalog(path) -> dict[str, dict]:
    return {r["tool_id"]: r for r in read_jsonl(path)}


def load_tests(path) -> list[Query]:
    return [Query(r["qid"], r["query"], list(r["gold"]), r.get("group", "")) for r in read_jsonl(path)]


def load_client_experience(client_dir) -> list[list[dict]]:
    """One JSONL per client: rows {tool_id, text, kind}. Sorted by file name."""
    files = sorted(Path(client_dir).glob("client*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no client*.jsonl in {client_dir}")
    return [read_jsonl(p) for p in files]


def strata(queries: Sequence[Query], heldout: set[str]) -> dict[str, list[Query]]:
    """held-out: all gold tools in H; labeled-clean: none in H; mixed: some."""
    out = {"heldout": [], "labeled_clean": [], "mixed": []}
    for q in queries:
        k = sum(g in heldout for g in q.gold)
        if k == len(q.gold):
            out["heldout"].append(q)
        elif k == 0:
            out["labeled_clean"].append(q)
        else:
            out["mixed"].append(q)
    out["labeled"] = out["labeled_clean"] + out["mixed"]
    return out


# ----------------------------------------------------------------------------
# Index construction
# ----------------------------------------------------------------------------


def description_entries(catalog: dict[str, dict]) -> list[Entry]:
    return [Entry(t, c["description"], "desc") for t, c in catalog.items()]


def concat_experience(clients: list[list[dict]], allowed_tools: set[str] | None = None) -> list[Entry]:
    """P0 pool: every client item becomes its own entry. No clustering, no
    majority, no conflict handling (the FICAL-style concatenation)."""
    out = []
    for ci, rows in enumerate(clients):
        for r in rows:
            if allowed_tools is not None and r["tool_id"] not in allowed_tools:
                continue
            out.append(Entry(r["tool_id"], r["text"], "exp", {"client": ci, "kind": r.get("kind", "")}))
    return out


# ----------------------------------------------------------------------------
# Retrievers
# ----------------------------------------------------------------------------

_tok = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _tok.findall(s.lower())


class BM25:
    """Okapi BM25 with optional frozen corpus statistics (P2b).

    frozen_from: a list of texts whose IDF and average length are used for
    scoring, instead of the indexed corpus' own statistics.
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75, frozen_from: Sequence[str] | None = None):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        stat_docs = [Counter(tokenize(t)) for t in frozen_from] if frozen_from is not None else self.docs
        n = len(stat_docs)
        df = Counter()
        for d in stat_docs:
            df.update(d.keys())
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.idf_default = math.log(1 + (n + 0.5) / 0.5)  # unseen term under frozen stats
        self.avgdl = float(np.mean([sum(d.values()) for d in stat_docs])) if stat_docs else 1.0
        self.inv = defaultdict(list)
        for i, d in enumerate(self.docs):
            for w, tf in d.items():
                self.inv[w].append((i, tf))

    def score(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for w in set(tokenize(query)):
            idf = self.idf.get(w, self.idf_default)
            for i, tf in self.inv.get(w, ()):
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


class Dense:
    """Cosine retrieval with any sentence-transformers model or checkpoint path."""

    def __init__(self, model_name: str, texts: Sequence[str], batch_size: int = 64, query_prefix: str = "", doc_prefix: str = "", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.qp, self.dp = query_prefix, doc_prefix
        self.emb = self._enc([doc_prefix + t for t in texts], batch_size)

    def _enc(self, texts, bs):
        return np.asarray(self.model.encode(list(texts), batch_size=bs, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def score(self, query: str) -> np.ndarray:
        q = self._enc([self.qp + query], 1)[0]
        return self.emb @ q


class HashDense:
    """Dependency-free stand-in for smoke tests only (bag-of-words hashing)."""

    def __init__(self, texts: Sequence[str], dim: int = 2048):
        self.dim = dim
        self.emb = np.stack([self._v(t) for t in texts]) if texts else np.zeros((0, dim))

    def _v(self, t):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in tokenize(t):
            v[hash(w) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def score(self, query):
        return self.emb @ self._v(query)


def make_retriever(kind: str, texts: Sequence[str], **kw):
    """kind: 'bm25', 'bm25-frozen' (needs frozen_from=...), 'hash' or a model id/path."""
    if kind == "bm25":
        return BM25(texts)
    if kind == "bm25-frozen":
        return BM25(texts, frozen_from=kw["frozen_from"])
    if kind == "hash":
        return HashDense(texts)
    return Dense(kind, texts, **{k: v for k, v in kw.items() if k in ("batch_size", "query_prefix", "doc_prefix", "device")})


# ----------------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------------


def walk_down(scores: np.ndarray, entries: Sequence[Entry], k: int = 5, cap: int = 200) -> list[str]:
    """Walk the entry ranking until k distinct tools are collected (cap entries).
    Mirrors the paper's rule; raises if the list cannot be filled."""
    order = np.argsort(-scores, kind="stable")[:cap]
    seen: list[str] = []
    for i in order:
        t = entries[i].tool_id
        if t not in seen:
            seen.append(t)
            if len(seen) == k:
                return seen
    raise RuntimeError(f"could not fill {k} distinct tools within {cap} entries")


def tool_scores(scores: np.ndarray, entries: Sequence[Entry], agg: str = "max", sigma: float | None = None) -> dict[str, float]:
    """Aggregate entry scores per tool (P2a).

    max      : best entry (equivalent to walk-down ordering)
    mean     : mean entry score
    maxcorr  : max minus sigma * E[max of n standard normals]; removes the
               advantage a tool gets only from having more entries
    """
    per = defaultdict(list)
    for s, e in zip(scores, entries):
        per[e.tool_id].append(float(s))
    if agg == "max":
        return {t: max(v) for t, v in per.items()}
    if agg == "mean":
        return {t: float(np.mean(v)) for t, v in per.items()}
    if agg == "maxcorr":
        sig = sigma if sigma is not None else float(np.std(scores)) or 1.0
        return {t: max(v) - sig * expected_max_normal(len(v)) for t, v in per.items()}
    raise ValueError(agg)


_EMAX_CACHE: dict[int, float] = {}


def expected_max_normal(n: int, draws: int = 20000, seed: int = 0) -> float:
    if n <= 1:
        return 0.0
    if n not in _EMAX_CACHE:
        rng = np.random.default_rng(seed)
        _EMAX_CACHE[n] = float(rng.standard_normal((draws, n)).max(1).mean())
    return _EMAX_CACHE[n]


def top_k_tools(tscores: dict[str, float], k: int = 5) -> list[str]:
    return [t for t, _ in sorted(tscores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def per_source_z(scores: np.ndarray, entries: Sequence[Entry]) -> np.ndarray:
    """Standardize description and experience scores separately for this query,
    then merge (P2a variant)."""
    out = scores.astype(float).copy()
    for src in {e.source for e in entries}:
        idx = np.array([e.source == src for e in entries])
        v = out[idx]
        sd = v.std() or 1.0
        out[idx] = (v - v.mean()) / sd
    return out


def two_index(desc_scores, desc_entries, exp_scores, exp_entries, n_desc: int = 3, k: int = 5, cap: int = 200) -> list[str]:
    """Two-index router: n_desc slots from descriptions, the rest from experience;
    fall back to the other index if one cannot fill its share."""
    d = walk_down(desc_scores, desc_entries, k=k, cap=cap)
    out = d[:n_desc]
    if len(exp_entries):
        for t in walk_down(exp_scores, exp_entries, k=min(k, len({e.tool_id for e in exp_entries})), cap=cap * 10):
            if len(out) == k:
                break
            if t not in out:
                out.append(t)
    for t in d:
        if len(out) == k:
            break
        if t not in out:
            out.append(t)
    return out[:k]


# ----------------------------------------------------------------------------
# Reranking
# ----------------------------------------------------------------------------


class Top1Reranker:
    """Picks the first candidate. Smoke tests and retrieval-only top-1."""

    def choose(self, query: str, cands: list[str], render: Callable[[str], str]) -> str:
        return cands[0]


class LLMIndexReranker:
    """Frozen LLM that scores candidate indices by next-token log-probability,
    as in the paper. Replace PROMPT with your exact production prompt so that
    numbers stay comparable."""

    PROMPT = (
        "Choose the single best tool for the user request.\n\nRequest: {query}\n\nCandidates:\n{cands}\n\n"
        "Answer with the candidate number only.\nAnswer:"
    )

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self.ids = [self.tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 10)]

    def choose(self, query: str, cands: list[str], render: Callable[[str], str]) -> str:
        body = "\n".join(f"{i + 1}. {render(t)}" for i, t in enumerate(cands))
        text = self.PROMPT.format(query=query, cands=body)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0, -1]
        lp = logits[self.ids[: len(cands)]]
        return cands[int(lp.argmax())]


def default_render(catalog: dict[str, dict], exp_by_tool: dict[str, list[str]] | None = None, max_exp: int = 3) -> Callable[[str], str]:
    """Description first, then up to max_exp experience lines, as the paper's prompt does."""

    def render(t: str) -> str:
        s = f"{catalog[t].get('name', t)}: {catalog[t]['description']}"
        if exp_by_tool and exp_by_tool.get(t):
            s += " | when to use: " + " / ".join(exp_by_tool[t][:max_exp])
        return s

    return render


# ----------------------------------------------------------------------------
# Metrics and statistics
# ----------------------------------------------------------------------------


def hit(cands: list[str], gold: list[str]) -> int:
    return int(any(g in cands for g in gold))


def correct(pred: str, gold: list[str]) -> int:
    return int(pred in gold)


def clustered_bootstrap_diff(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Paired difference mean(a-b) with a 95% interval, resampling clusters
    (query ids, or the gold tool for tool-clustered intervals)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    sums = np.array([d[groups[k]].sum() for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, float]:
    from math import comb

    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return n01, n10, min(1.0, 2 * p)


def summarize(name: str, rows: list[dict]) -> dict:
    """rows: one dict per (seed, qid) with keys acc, r5, stratum."""
    out = {"arm": name}
    for st in ("heldout", "labeled", "labeled_clean", "mixed", "all"):
        sel = [r for r in rows if st == "all" or r["stratum"] == st or (st == "labeled" and r["stratum"] in ("labeled_clean", "mixed"))]
        if not sel:
            continue
        by_seed = defaultdict(list)
        for r in sel:
            by_seed[r["seed"]].append(r)
        accs = [np.mean([r["acc"] for r in v]) for v in by_seed.values()]
        r5s = [np.mean([r["r5"] for r in v]) for v in by_seed.values()]
        out[f"{st}_acc"] = round(float(np.mean(accs)), 3)
        out[f"{st}_acc_sd"] = round(float(np.std(accs)), 3)
        out[f"{st}_r5"] = round(float(np.mean(r5s)), 3)
        out[f"{st}_n"] = len(sel) // max(1, len(by_seed))
    return out


def stratum_of(q: Query, heldout: set[str]) -> str:
    k = sum(g in heldout for g in q.gold)
    return "heldout" if k == len(q.gold) else ("labeled_clean" if k == 0 else "mixed")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
