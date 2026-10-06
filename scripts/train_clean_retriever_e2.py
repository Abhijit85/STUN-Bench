#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import random
import re
import sys
import time
import types
from pathlib import Path

# Avoid importing a broken optional torchaudio build through transformers.audio_utils.
if "torchaudio" not in sys.modules:
    _torchaudio_stub = types.ModuleType("torchaudio")
    _torchaudio_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    _torchaudio_stub.__version__ = "0.0"
    sys.modules["torchaudio"] = _torchaudio_stub

import numpy as np
import torch

# This cluster venv has a torchvision build whose fake nms registration fails at
# import time. E2 does not use vision modules; keep BERT imports available.
_orig_register_fake = torch.library.register_fake
def _safe_register_fake(op_name, func=None, /, **kwargs):
    def _decorator(fn):
        try:
            return _orig_register_fake(op_name, fn, **kwargs)
        except RuntimeError as exc:
            if str(op_name) == "torchvision::nms" and "operator torchvision::nms does not exist" in str(exc):
                return fn
            raise
    if func is None:
        return _decorator
    return _decorator(func)
torch.library.register_fake = _safe_register_fake

import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

BASE = os.environ.get("E2_BGE_MODEL_PATH", "<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
QP = "Represent this sentence for searching relevant passages: "


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"ASSERTION FAILED: {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tool-docs", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-sha256", default=None)
    ap.add_argument("--test-queries", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--skip-a3", action="store_true", help="Skip A3 check entirely; use only for debugging.")
    ap.add_argument("--drop-a3", action="store_true", help="Drop base-BGE train queries with cos>=0.95 to any eval query, log them, and continue.")
    return ap.parse_args()


class BGEEncoder(torch.nn.Module):
    def __init__(self, model_name: str, max_len: int, device: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.max_len = max_len
        self.device = torch.device(device)
        self.to(self.device)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch).last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=1)

    @torch.no_grad()
    def encode_all(self, texts: list[str], batch_size: int = 512) -> torch.Tensor:
        self.eval()
        chunks = []
        for start in range(0, len(texts), batch_size):
            chunks.append(self.encode_batch(texts[start:start + batch_size]).detach().cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, self.model.config.hidden_size)


def recall_at_5(model: BGEEncoder, pairs: list[tuple[str, str]], docs: dict[str, str]) -> float:
    queries = [QP + q for q, _ in pairs]
    tools = sorted(docs)
    passages = [docs[t] for t in tools]
    q_emb = model.encode_all(queries, 512)
    d_emb = model.encode_all(passages, 512)
    hits = 0
    tool_index = {tool: i for i, tool in enumerate(tools)}
    for start in range(0, q_emb.shape[0], 256):
        sims = q_emb[start:start + 256] @ d_emb.T
        top = torch.topk(sims, k=min(5, d_emb.shape[0]), dim=1).indices.tolist()
        for offset, indices in enumerate(top):
            gold_tool = pairs[start + offset][1]
            hits += int(tool_index[gold_tool] in indices)
    return hits / len(pairs) if pairs else 0.0


def main() -> int:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    if a.heldout_sha256 and sha256_file(a.heldout) != a.heldout_sha256:
        fail("held-out list hash does not match --heldout-sha256")
    H = set(json.load(open(a.heldout)))
    docs = json.load(open(a.tool_docs))
    tests = json.load(open(a.test_queries))
    test_norm = {norm(q) for q in tests}
    items = [json.loads(l) for l in open(a.pairs)]

    pairs: list[tuple[str, str]] = []
    n_bad_h = 0
    n_bad_q = 0
    missing: set[str] = set()
    for it in items:
        if any(t in H for t in it["gold_tools"]):
            n_bad_h += 1
            continue
        if norm(it["query"]) in test_norm:
            n_bad_q += 1
            continue
        for t in it["gold_tools"]:
            if t not in docs:
                missing.add(t)
            else:
                pairs.append((it["query"], t))
    if n_bad_h:
        fail(f"A1: {n_bad_h} items label a held-out tool")
    if n_bad_q:
        fail(f"A2: {n_bad_q} items exactly match a test query")
    if missing:
        fail(f"A4: {len(missing)} gold tools lack descriptions, e.g. {sorted(missing)[:5]}")
    print(f"items={len(items)} pairs={len(pairs)} tools={len({t for _, t in pairs})}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEEncoder(BASE, a.max_len, device)

    dropped_a3_queries: list[str] = []
    if not a.skip_a3:
        uq = sorted({q for q, _ in pairs})
        E_tr = model.encode_all([QP + q for q in uq], 512)
        E_te = model.encode_all([QP + q for q in tests], 512)
        bad_queries: set[str] = set()
        for i in range(0, E_tr.shape[0], 8192):
            max_scores = (E_tr[i:i + 8192] @ E_te.T).max(dim=1).values
            for q, score in zip(uq[i:i + 8192], max_scores.tolist()):
                if score >= 0.95:
                    bad_queries.add(q)
        if bad_queries and not a.drop_a3:
            fail(f"A3: {len(bad_queries)} training queries have cos>=0.95 to a test query under {BASE}")
        if bad_queries:
            dropped_a3_queries = sorted(bad_queries)
            before_pairs = len(pairs)
            pairs = [(q, t) for q, t in pairs if q not in bad_queries]
            with open(os.path.join(a.out, "dropped_a3_queries.json"), "w") as handle:
                json.dump({"threshold": 0.95, "model": BASE, "dropped_query_count": len(dropped_a3_queries), "dropped_pair_count": before_pairs - len(pairs), "queries": dropped_a3_queries}, handle, indent=2)
            print(f"A3 drop: removed {len(dropped_a3_queries)} unique queries / {before_pairs - len(pairs)} pairs at cos>=0.95", flush=True)
        else:
            print("A3 passed: 0 training queries at cos>=0.95 to any test query", flush=True)

    random.shuffle(pairs)
    n_dev = max(200, int(a.dev_frac * len(pairs)))
    dev, train = pairs[:n_dev], pairs[n_dev:]
    before = {"dev_recall@5": recall_at_5(model, dev, docs)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loader = DataLoader(train, batch_size=a.batch, shuffle=True, drop_last=True)
    t0 = time.time()
    model.train()
    step = 0
    for epoch in range(a.epochs):
        for batch_pairs in loader:
            qs, ts = batch_pairs
            q_emb = model.encode_batch([QP + q for q in qs])
            d_emb = model.encode_batch([docs[t] for t in ts])
            logits = (q_emb @ d_emb.T) * 20.0
            labels = torch.arange(logits.shape[0], device=model.device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(json.dumps({"stage": "train_step", "step": step, "epoch": epoch, "loss": float(loss.detach().cpu())}), flush=True)
    after = {"dev_recall@5": recall_at_5(model, dev, docs)}

    model.model.save_pretrained(a.out)
    model.tokenizer.save_pretrained(a.out)
    ck = hashlib.sha256()
    for root, _, files in sorted(os.walk(a.out)):
        for f in sorted(files):
            if f.endswith((".safetensors", ".bin")):
                ck.update(open(os.path.join(root, f), "rb").read())
    manifest = dict(
        base=BASE,
        seed=a.seed,
        epochs=a.epochs,
        batch=a.batch,
        lr=a.lr,
        max_len=a.max_len,
        n_items=len(items),
        n_train_pairs=len(train),
        n_dev_pairs=len(dev),
        heldout_sha256=sha256_file(a.heldout),
        pairs_sha256=sha256_file(a.pairs),
        tool_docs_sha256=sha256_file(a.tool_docs),
        test_queries_sha256=sha256_file(a.test_queries),
        assertions=dict(A1="pass", A2="pass", A3="skipped" if a.skip_a3 else ("dropped" if dropped_a3_queries else "pass"), A4="pass"),
        a3_drop=dict(enabled=bool(a.drop_a3), dropped_query_count=len(dropped_a3_queries), dropped_queries_sha256=hashlib.sha256(json.dumps(dropped_a3_queries, sort_keys=True).encode("utf-8")).hexdigest()),
        dev_before=before,
        dev_after=after,
        train_seconds=round(time.time() - t0, 1),
        checkpoint_sha256=ck.hexdigest(),
        query_prefix=QP,
        implementation="direct_transformers_inbatch_negatives",
    )
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2, default=str)
    print(json.dumps({k: manifest[k] for k in ("seed", "n_train_pairs", "dev_before", "dev_after", "checkpoint_sha256")}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
