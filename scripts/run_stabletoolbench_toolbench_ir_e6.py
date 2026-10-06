#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import types
import importlib.machinery
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CANONICAL_ROOT = Path(os.environ.get('FEDRAG_CANONICAL_ROOT', '<REPO_ROOT>')).resolve()

@dataclass
class RoutedCandidate:
    candidate_id: str
    label: str
    parent_tool: str
    when_to_use: list[str]
    do_not_use_when: list[str]
    text: str
    embedding: list[float]
    provenance: str


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def load_package_file(path: Path) -> tuple[Any, str]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    package = payload['package']
    artifacts = []
    for item in package.get('artifacts', []):
        obj = types.SimpleNamespace(
            signature=item['signature'],
            text=item['text'],
            structured_payload=item.get('structured_payload'),
            metadata=item.get('metadata') or {},
        )
        artifacts.append(obj)
    return types.SimpleNamespace(source_id=package.get('source_id', 'loaded'), artifacts=artifacts, metadata=package.get('metadata') or {}), str(payload.get('package_sha256') or '')


def render_precautions(precautions: list[str], conflict_log: list[str]) -> list[str]:
    return list(dict.fromkeys([*precautions, *conflict_log]))


def _select_distinct_tools(ranked_indices: list[int], candidates: list[RoutedCandidate], limit: int) -> list[str]:
    selected = []
    seen = set()
    for idx in ranked_indices:
        tool = candidates[idx].parent_tool
        if tool in seen:
            continue
        seen.add(tool)
        selected.append(tool)
        if len(selected) >= limit:
            break
    return selected


def _expand_ranked_candidates(ranked_indices: list[int], candidates: list[RoutedCandidate], selected_tools: list[str]) -> list[RoutedCandidate]:
    expanded = []
    counts = {tool: 0 for tool in selected_tools}
    for idx in ranked_indices:
        cand = candidates[idx]
        if cand.parent_tool not in counts:
            continue
        if counts[cand.parent_tool] >= 2:
            continue
        expanded.append(cand)
        counts[cand.parent_tool] += 1
    return expanded


def build_ranked_candidates(candidates: list[RoutedCandidate], similarities: np.ndarray, retrieval_pool_size: int, top_k: int, retrieval_mode: str) -> tuple[list[RoutedCandidate], list[str]]:
    ranked_indices = np.argsort(-similarities).tolist()
    ranked_pool = ranked_indices[:max(top_k, retrieval_pool_size)]
    pool_tools = [candidates[idx].parent_tool for idx in ranked_pool]
    if retrieval_mode != 'distinct_tool_topk':
        return [candidates[idx] for idx in ranked_pool[:top_k]], pool_tools
    selected_tools = _select_distinct_tools(ranked_pool, candidates, top_k)
    return _expand_ranked_candidates(ranked_pool, candidates, selected_tools), pool_tools

DEFAULT_OUTPUT = CANONICAL_ROOT / 'artifacts' / 'results' / 'stabletoolbench_toolbench_ir_e6_r1'
DEFAULT_MODEL = CANONICAL_ROOT / 'external_models' / 'ToolBench' / 'ToolBench_IR_bert_based_uncased'
E6_DIRS = {
    42: CANONICAL_ROOT / 'artifacts' / 'results' / 'stabletoolbench_heldout_retriever_compare_e6_r4',
    123: CANONICAL_ROOT / 'artifacts' / 'results' / 'stabletoolbench_heldout_retriever_compare_e6_seed123_r4',
    456: CANONICAL_ROOT / 'artifacts' / 'results' / 'stabletoolbench_heldout_retriever_compare_e6_seed456_r4',
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stub_vision_audio() -> None:
    class _Interp:
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1
    for name in [
        'torchvision', 'torchvision.io', 'torchvision.transforms', 'torchvision.transforms.functional',
        'torchvision.transforms.v2', 'torchvision.transforms.v2.functional', 'torchaudio', 'torchaudio._extension',
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = m
    sys.modules['torchvision'].io = sys.modules['torchvision.io']
    sys.modules['torchvision'].transforms = sys.modules['torchvision.transforms']
    sys.modules['torchvision.io'].ImageReadMode = object
    sys.modules['torchvision.io'].decode_image = lambda *a, **k: None
    sys.modules['torchvision.transforms'].InterpolationMode = _Interp
    for fn in ['pil_to_tensor', 'to_pil_image', 'resize', 'center_crop', 'convert_image_dtype', 'normalize']:
        setattr(sys.modules['torchvision.transforms.functional'], fn, lambda *a, **k: None)


class ToolBenchIREncoder:
    def __init__(self, model_path: Path, device: str, batch_size: int) -> None:
        stub_vision_audio()
        from transformers import AutoModel, AutoTokenizer
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        outs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu().numpy().astype('float32'))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 768), dtype='float32')


def candidate_from_artifact(artifact: Any, text_override: str | None = None) -> RoutedCandidate:
    payload = artifact.structured_payload or {}
    when = []
    if isinstance(payload.get('tool_description'), str) and payload['tool_description'].strip():
        when.append(payload['tool_description'].strip())
    if isinstance(payload.get('scenario_context'), str) and payload['scenario_context'].strip():
        when.append(payload['scenario_context'].strip())
    if isinstance(payload.get('annex_summary'), str) and payload['annex_summary'].strip():
        when.append(payload['annex_summary'].strip())
    if not when:
        when = [(text_override or artifact.text)[:200]]
    precautions = [str(x).strip() for x in payload.get('precautions', []) if str(x).strip()] if isinstance(payload.get('precautions'), list) else []
    conflict_log = [str(x).strip() for x in payload.get('conflict_log', []) if str(x).strip()] if isinstance(payload.get('conflict_log'), list) else []
    return RoutedCandidate(
        candidate_id=artifact.signature,
        label=str((artifact.metadata or {}).get('tool') or artifact.signature),
        parent_tool=str((artifact.metadata or {}).get('tool') or artifact.signature),
        when_to_use=when,
        do_not_use_when=render_precautions(precautions, conflict_log),
        text=text_override or artifact.text,
        embedding=[],
        provenance=str((artifact.metadata or {}).get('artifact_origin') or (artifact.metadata or {}).get('source_group') or (artifact.metadata or {}).get('category') or 'stabletoolbench'),
    )


def native_doc_text(artifact: Any) -> str:
    payload = artifact.structured_payload or {}
    meta = artifact.metadata or {}
    tool = str(meta.get('tool') or meta.get('domain') or artifact.signature)
    category = str(meta.get('category') or 'unknown')
    desc = str(payload.get('tool_description') or artifact.text)
    scenario = str(payload.get('scenario_context') or '')
    return '\n'.join([
        f'category_name: {category}',
        f'tool_name: {tool}',
        f'api_name: {tool}',
        f'api_description: {desc}',
        f'description: {scenario}',
    ])


def make_candidates(package: Any, variant: str) -> list[RoutedCandidate]:
    out = []
    for artifact in package.artifacts:
        override = native_doc_text(artifact) if variant == 'native' else None
        out.append(candidate_from_artifact(artifact, override))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'n': len(bucket),
            'recall_at_5': sum(1 for r in bucket if r['gold_in_top_k']) / len(bucket) if bucket else 0.0,
            'retrieval_top1': sum(1 for r in bucket if (r['candidate_tools'] or [''])[0] in set(r['gold_tools'])) / len(bucket) if bucket else 0.0,
            'shortfall_count': sum(1 for r in bucket if len(set(r['candidate_tools'])) < 5),
            'mean_distinct_tools': statistics.mean([len(set(r['candidate_tools'])) for r in bucket]) if bucket else 0.0,
        }
    return {
        'all': one(rows),
        'heldout': one([r for r in rows if r['subset'] == 'heldout']),
        'labeled': one([r for r in rows if r['subset'] == 'labeled']),
    }


def run_arm(name: str, candidates: list[RoutedCandidate], encoder: ToolBenchIREncoder, queries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    texts = [c.text for c in candidates]
    index = encoder.encode(texts)
    query_embeddings = encoder.encode([q['query_text'] for q in queries])
    rows = []
    for q, emb in zip(queries, query_embeddings):
        scores = index @ emb if len(index) else np.asarray([], dtype='float32')
        ranked, pool_tools = build_ranked_candidates(candidates, scores, args.retrieval_pool_size, args.top_k, 'distinct_tool_topk')
        tools = [c.parent_tool for c in ranked]
        row = {
            'query_id': q['query_id'],
            'query_text': q['query_text'],
            'group': q.get('group'),
            'gold_tools': q['gold_tools'],
            'subset': q['subset'],
            'candidate_ids': [c.candidate_id for c in ranked],
            'candidate_tools': tools,
            'retrieval_pool_tools': pool_tools,
            'gold_in_top_k': any(t in set(q['gold_tools']) for t in tools),
            'shortfall': len(set(tools)) < args.top_k,
        }
        rows.append(row)
    metrics = summarize(rows)
    heldout_shortfalls = metrics['heldout']['shortfall_count']
    if heldout_shortfalls:
        raise RuntimeError(f'{name} has held-out shortfalls: {heldout_shortfalls}')
    return {'arm': name, 'retriever': 'toolbench_ir', 'rows': rows, 'metrics': metrics}


def aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({arm for seed_payload in per_seed.values() for arm in seed_payload})
    out = {}
    for arm in arms:
        out[arm] = {}
        for subset in ['all', 'heldout', 'labeled']:
            vals = [per_seed[str(seed)][arm]['metrics'][subset]['recall_at_5'] for seed in sorted(map(int, per_seed.keys()))]
            out[arm][subset] = {
                'mean_recall_at_5': statistics.mean(vals),
                'sd_recall_at_5': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'seeds': sorted(map(int, per_seed.keys())),
                'n': per_seed[str(sorted(map(int, per_seed.keys()))[0])][arm]['metrics'][subset]['n'],
            }
    if 'docs_canonical' in out and 'synapse_shared' in out:
        out['shared_minus_docs_heldout_recall'] = out['synapse_shared']['heldout']['mean_recall_at_5'] - out['docs_canonical']['heldout']['mean_recall_at_5']
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--seeds', default='42,123,456')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--retrieval-pool-size', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',') if s]
    commit = os.popen('git rev-parse HEAD').read().strip()
    dirty = os.popen('git status --porcelain').read().strip().splitlines()
    if dirty:
        raise RuntimeError(f'worktree dirty: {dirty[:5]}')
    encoder = ToolBenchIREncoder(args.model_path, args.device, args.batch_size)
    per_seed: dict[str, dict[str, Any]] = {}
    model_meta = {
        'checkpoint_id': 'ToolBench/ToolBench_IR_bert_based_uncased',
        'resolved_revision': 'cf4a904',
        'model_path': str(args.model_path),
        'config_sha256': sha256_file(args.model_path / 'config.json'),
        'pytorch_model_sha256': sha256_file(args.model_path / 'pytorch_model.bin'),
    }
    for seed in seeds:
        root = E6_DIRS[seed]
        seed_dir = root / f'seed_{seed}'
        docs_rows = json.loads((seed_dir / 'jina' / 'docs_only.json').read_text())['rows']
        queries = [{
            'query_id': r['query_id'], 'query_text': r['query_text'], 'group': r.get('group'),
            'gold_tools': r.get('gold_tools') or r.get('gold_parent_tools'), 'subset': r['subset'],
        } for r in docs_rows]
        flat_pkg, flat_hash = load_package_file(seed_dir / 'packages' / 'flat_pool.json')
        syn_pkg, syn_hash = load_package_file(seed_dir / 'packages' / 'synapse_shared.json')
        arms = {
            'docs_canonical': make_candidates(flat_pkg, 'canonical'),
            'docs_native': make_candidates(flat_pkg, 'native'),
            'synapse_shared': make_candidates(syn_pkg, 'canonical'),
        }
        per_seed[str(seed)] = {}
        for arm, candidates in arms.items():
            started = time.time()
            result = run_arm(arm, candidates, encoder, queries, args)
            result.update({
                'paper_eligible': True,
                'repo_commit': commit,
                'seed': seed,
                'candidate_rule': 'distinct5_walkdown',
                'retrieval_pool_size': args.retrieval_pool_size,
                'top_k': args.top_k,
                'source_e6_dir': str(root),
                'flat_pool_sha256': flat_hash,
                'synapse_shared_sha256': syn_hash,
                'model': model_meta,
                'elapsed_seconds': round(time.time() - started, 3),
            })
            out = args.output_dir / f'seed_{seed}' / 'toolbench_ir' / f'{arm}.json'
            save_json(out, result)
            per_seed[str(seed)][arm] = result
            print(json.dumps({'stage':'arm_done','seed':seed,'arm':arm,'heldout_r5':result['metrics']['heldout']['recall_at_5'],'labeled_r5':result['metrics']['labeled']['recall_at_5']}), flush=True)
    summary = {
        'paper_eligible': True,
        'config': {
            'repo_commit': commit,
            'seeds': seeds,
            'retriever': 'toolbench_ir',
            'candidate_rule': 'distinct5_walkdown',
            'retrieval_pool_size': args.retrieval_pool_size,
            'top_k': args.top_k,
            'model': model_meta,
            'source_e6_dirs': {str(k): str(v) for k, v in E6_DIRS.items() if k in seeds},
        },
        'per_seed': {seed: {arm: payload['metrics'] for arm, payload in arms.items()} for seed, arms in per_seed.items()},
        'aggregate': aggregate(per_seed),
    }
    save_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary['aggregate'], indent=2), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
