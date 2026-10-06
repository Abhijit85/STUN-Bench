#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.prepare_stabletoolbench_cross_client_e8 import TX_GROUP, force_single_owner_draws, item_record
from scripts.run_reranker_prompt_sweep import run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    assign_clients,
    batched_query_embeddings,
    build_client_package,
    build_doc_package,
    build_ranked_candidates,
    build_tool_registry,
    combine_packages,
    filter_eval_queries,
    filter_experience_items,
    limit_client_items,
    load_local_backend,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    maybe_cuda_synchronize,
    merge_heartbeat,
    package_to_candidates,
    read_edge_conflict_log,
    resolve_local_embedder,
    save_json,
    stable_hash,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import HELDOUT_GROUPS, filter_heldout_pool, heldout_tool_set
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage

DEFAULT_INPUT_DIR = REPO_ROOT / 'artifacts' / 'results' / 'stabletoolbench_cross_client_e8_inputs_r2'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'artifacts' / 'results' / 'stabletoolbench_cross_client_e8_full_r1'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='E8 full cross-client run for one seed.')
    p.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument('--stb-root', type=Path, default=DEFAULT_STB_ROOT)
    p.add_argument('--toolbench-instruction-dir', type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arms', default='local_only,docs_only,synapse,concat,two_index')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--retrieval-pool-size', type=int, default=200)
    p.add_argument('--retrieval-mode', default='distinct_tool_topk')
    p.add_argument('--reranker-variant', default='V3')
    p.add_argument('--embed-model', default='jina-embeddings-v2-base-en')
    p.add_argument('--merge-policy', default='conflict_log')
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--allow-dirty', action='store_true')
    return p.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


def experience_only_package(package: KnowledgePackage) -> KnowledgePackage:
    artifacts = []
    for artifact in package.artifacts:
        payload = artifact.structured_payload or {}
        if artifact.metadata.get('artifact_origin') == 'tool_doc' or payload.get('type') == 'tool_doc':
            continue
        artifacts.append(KnowledgeArtifact(signature=artifact.signature, text=artifact.text, structured_payload=payload, metadata=dict(artifact.metadata)))
    return KnowledgePackage(source_id=f'{package.source_id}_experience_only', artifacts=artifacts, metadata={**package.metadata, 'artifact_filter': 'exclude_tool_doc'})


def stratum_for(item: Any, heldout: set[str], tx_tools: set[str]) -> str:
    gold = set(item.gold_tools)
    if gold and gold <= heldout:
        return 'heldout'
    if gold & tx_tools:
        return 'tx_one_owner'
    return 'other_labeled'


def add_strata(result: dict[str, Any], query_by_id: dict[str, Any], heldout: set[str], tx_tools: set[str]) -> dict[str, Any]:
    for row in result.get('rows', []):
        item = query_by_id[str(row.get('query_id'))]
        row['subset'] = stratum_for(item, heldout, tx_tools)
        row['stratum'] = row['subset']
        row.setdefault('row_id', str(row.get('query_id')))
    result['stratum_metrics'] = {}
    for subset in ('heldout', 'tx_one_owner', 'other_labeled'):
        rows = [row for row in result.get('rows', []) if row.get('subset') == subset]
        result['stratum_metrics'][subset] = {
            'n': len(rows),
            'accuracy': sum(bool(row.get('correct', row.get('routed_correctly'))) for row in rows) / len(rows) if rows else 0.0,
            'recall_at_5': sum(bool(row.get('gold_in_top_k', row.get('gold_in_top_5'))) for row in rows) / len(rows) if rows else 0.0,
        }
    return result


def evaluate_package(name: str, package: KnowledgePackage, test_items: list[Any], jina_client: JinaAIClient, embed_model: str, backend: Any, args: argparse.Namespace, progress: ProgressLogger, *, seed: int, client_id: str | None = None) -> dict[str, Any]:
    progress.log('arm_prepare_begin', arm=name, seed=seed, client_id=client_id, query_count=len(test_items), artifact_count=len(package.artifacts))
    candidates = package_to_candidates(package, jina_client, embed_model)
    if candidates:
        matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    rows = []
    for idx, (item, embedding) in enumerate(zip(test_items, query_embeddings), start=1):
        started = time.perf_counter()
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        scores = matrix @ q if matrix.size else np.asarray([], dtype=np.float32)
        ranked, retrieval_pool_tools = build_ranked_candidates(candidates, scores, args.retrieval_pool_size, args.top_k, args.retrieval_mode)
        if len(ranked) < args.top_k:
            raise RuntimeError(f'{name} seed {seed} client {client_id} query {item.query_id} returned {len(ranked)} candidates')
        top = ranked[0]
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        out = run_prompt(backend, args.reranker_variant, 'toolbench', item.query, ranked, [], top)
        maybe_cuda_synchronize(backend)
        rows.append({
            'row_id': str(item.query_id) if client_id is None else f'{item.query_id}::{client_id}',
            'query_id': item.query_id,
            'query_text': item.query,
            'group': item.group,
            'client_id': client_id,
            'gold_parent_tools': item.gold_tools,
            'gold_tools': item.gold_tools,
            'predicted_tool': out.predicted_tool,
            'correct': out.predicted_tool in item.gold_tools,
            'routed_correctly': out.predicted_tool in item.gold_tools,
            'gold_in_top_k': any(candidate.parent_tool in item.gold_tools for candidate in ranked),
            'top_candidate_ids': [candidate.parent_tool for candidate in ranked],
            'retrieval_pool_tools': retrieval_pool_tools,
            'parse_ok': out.parse_ok,
            'fallback_used': out.fallback_used,
            'prompt_hash': out.prompt_hash,
            'rerank_s': time.perf_counter() - rerank_started,
            'total_s': time.perf_counter() - started,
        })
        if idx == 1 or idx % 50 == 0 or idx == len(test_items):
            progress.log('arm_progress', arm=name, seed=seed, client_id=client_id, completed_queries=idx, total_queries=len(test_items), running_accuracy=sum(row['correct'] for row in rows) / len(rows), running_recall_at_5=sum(row['gold_in_top_k'] for row in rows) / len(rows))
    result = summarize_rows(rows)
    result.update({'arm': name, 'seed': seed, 'client_id': client_id, 'rows': rows})
    return result


def evaluate_two_index(package_docs: KnowledgePackage, package_exp: KnowledgePackage, package_rerank: KnowledgePackage, test_items: list[Any], jina_client: JinaAIClient, embed_model: str, backend: Any, args: argparse.Namespace, progress: ProgressLogger, *, seed: int) -> dict[str, Any]:
    progress.log('arm_prepare_begin', arm='two_index', seed=seed, query_count=len(test_items), docs_artifacts=len(package_docs.artifacts), experience_artifacts=len(package_exp.artifacts))
    docs = package_to_candidates(package_docs, jina_client, embed_model)
    exp = package_to_candidates(package_exp, jina_client, embed_model)
    rerank_by_tool = {}
    for cand in package_to_candidates(package_rerank, jina_client, embed_model):
        rerank_by_tool.setdefault(cand.parent_tool, cand)
    def matrix(cands: list[Any]) -> np.ndarray:
        m = np.asarray([c.embedding for c in cands], dtype=np.float32) if cands else np.zeros((0, 0), dtype=np.float32)
        if m.size:
            norms = np.linalg.norm(m, axis=1); norms[norms == 0.0] = 1.0; m = m / norms[:, None]
        return m
    dm, xm = matrix(docs), matrix(exp)
    qemb = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    rows=[]
    for idx, (item, emb) in enumerate(zip(test_items, qemb), start=1):
        q=np.asarray(emb,dtype=np.float32); qn=float(np.linalg.norm(q)); q=q/qn if qn>0 else q
        ds=dm@q if dm.size else np.asarray([],dtype=np.float32)
        xs=xm@q if xm.size else np.asarray([],dtype=np.float32)
        dtop,_=build_ranked_candidates(docs, ds, args.retrieval_pool_size, args.retrieval_pool_size, 'distinct_tool_topk')
        xtop,_=build_ranked_candidates(exp, xs, args.retrieval_pool_size, args.retrieval_pool_size, 'distinct_tool_topk')
        chosen=[]; sources=[]
        for source, ranked, budget in (('docs', dtop, 3), ('experience', xtop, args.top_k)):
            added=0
            for cand in ranked:
                if cand.parent_tool in {c.parent_tool for c in chosen}:
                    continue
                chosen.append(cand); sources.append({'tool': cand.parent_tool, 'source': source}); added += 1
                if len(chosen) == args.top_k or added == budget:
                    break
            if len(chosen) == args.top_k:
                break
        if len(chosen) < args.top_k:
            for cand in dtop:
                if cand.parent_tool in {c.parent_tool for c in chosen}:
                    continue
                chosen.append(cand); sources.append({'tool': cand.parent_tool, 'source': 'docs_backfill'})
                if len(chosen) == args.top_k:
                    break
        if len(chosen) < args.top_k:
            raise RuntimeError(f'two_index seed {seed} query {item.query_id} returned {len(chosen)} candidates')
        rerank=[rerank_by_tool[c.parent_tool] for c in chosen if c.parent_tool in rerank_by_tool]
        top=rerank[0]
        out=run_prompt(backend, args.reranker_variant, 'toolbench', item.query, rerank, [], top)
        rows.append({'row_id': str(item.query_id), 'query_id': item.query_id, 'query_text': item.query, 'group': item.group, 'gold_parent_tools': item.gold_tools, 'gold_tools': item.gold_tools, 'predicted_tool': out.predicted_tool, 'correct': out.predicted_tool in item.gold_tools, 'routed_correctly': out.predicted_tool in item.gold_tools, 'gold_in_top_k': any(c.parent_tool in item.gold_tools for c in chosen), 'top_candidate_ids': [c.parent_tool for c in chosen], 'candidate_sources': sources, 'parse_ok': out.parse_ok, 'fallback_used': out.fallback_used, 'prompt_hash': out.prompt_hash})
        if idx == 1 or idx % 50 == 0 or idx == len(test_items):
            progress.log('arm_progress', arm='two_index', seed=seed, completed_queries=idx, total_queries=len(test_items), running_accuracy=sum(r['correct'] for r in rows)/len(rows), running_recall_at_5=sum(r['gold_in_top_k'] for r in rows)/len(rows))
    result=summarize_rows(rows); result.update({'arm':'two_index', 'seed':seed, 'fusion_rule':'docs3_experience2_backfill', 'rows':rows})
    return result


def aggregate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out={}
    for arm,res in results.items():
        out[arm]={'accuracy': res.get('accuracy',0.0), 'recall_at_5': res.get('recall_at_5',0.0), 'stratum_metrics': res.get('stratum_metrics',{})}
    return out


def main() -> int:
    load_dotenv(REPO_ROOT / '.env')
    args=parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty=assert_clean_tree(allow_dirty=args.allow_dirty)
    progress=ProgressLogger(args.output_dir)
    progress.log('launch', repo_commit=commit, dirty_entry_count=len(dirty), seed=args.seed, input_dir=str(args.input_dir))
    arms=set(parse_csv(args.arms))
    input_summary=json.loads((args.input_dir/'summary.json').read_text())
    seed_rec=input_summary['per_seed'][str(args.seed)]
    owners: dict[str,str]=seed_rec['owners']; tx_tools=set(owners)

    queries=load_stabletoolbench_queries(args.stb_root, GROUPS)
    train=load_toolbench_training_items(args.toolbench_instruction_dir)
    registry=build_tool_registry(queries+train)
    registry,_=apply_junk_filter(registry)
    queries,_=filter_eval_queries(queries, registry)
    train,_=filter_experience_items(train, registry)
    heldout_tools,_=heldout_tool_set(queries, HELDOUT_GROUPS)
    heldout=set(heldout_tools)
    train,_=filter_heldout_pool(train, heldout)
    query_by_id={str(q.query_id): q for q in queries}

    cfg=input_summary['config']
    client_count=int(cfg.get('client_count', len(seed_rec['client_item_query_ids'])))
    max_items=int(cfg.get('max_items_per_client', 5000))
    partition_mode=cfg.get('partition_mode', 'category')
    target_items=int(cfg.get('target_items_per_tool', 8))
    base=assign_clients(train, client_count, partition_mode, args.seed)
    capped=limit_client_items(base, max_items, args.seed)
    client_items, force_info=force_single_owner_draws(capped, train, tx_tools, owners, seed=args.seed, max_items_per_client=max_items, target_items_per_tool=target_items)
    for cid, items in client_items.items():
        digest=stable_hash([item_record(item) for item in items])
        if digest != seed_rec['client_item_sha256'][cid]:
            raise RuntimeError(f'client draw hash mismatch for {cid}: {digest} != {seed_rec["client_item_sha256"][cid]}')
    progress.log('inputs_ready', seed=args.seed, query_count=len(queries), train_count=len(train), tx_tool_count=len(tx_tools), heldout_tool_count=len(heldout), non_owner_mentions=force_info['non_owner_mention_count_total'])

    embedder=resolve_local_embedder(); jina=JinaAIClient(api_keys=[])
    with temporary_env({'JINA_LOCAL_EMBED_MODEL': embedder['model_path'], 'JINA_LOCAL_EMBED_DEVICE': embedder['device'], 'JINA_LOCAL_EMBED_LOCAL_ONLY': embedder['local_only'], 'JINA_API_KEY': None}):
        backend=load_local_backend(args.model_path)
        client_packages={}; client_hashes={}
        for cid, items in sorted(client_items.items()):
            pkg,h=build_client_package(cid, items, registry, jina, args.embed_model)
            client_packages[cid]=pkg; client_hashes[cid]=h
            progress.log('client_package_done', seed=args.seed, client_id=cid, item_count=len(items), artifact_count=len(pkg.artifacts), package_sha256=h)
        docs_pkg, docs_hash=build_doc_package(registry)
        progress.log('docs_package_done', seed=args.seed, artifact_count=len(docs_pkg.artifacts), package_sha256=docs_hash)
        results={}
        common={'paper_eligible': True, 'repo_commit': commit, 'seed': args.seed, 'metric_definition': {'correct':'predicted_tool in gold_tools', 'recall_at_5':'any gold tool in five candidates'}, 'client_sha256': client_hashes, 'tool_doc_sha256': docs_hash, 'heldout_tools_sha256': input_summary['config']['heldout_tools_sha256'], 'tx_tools_sha256': stable_hash(sorted(tx_tools)), 'retrieval_mode': args.retrieval_mode, 'retrieval_pool_size': args.retrieval_pool_size, 'top_k': args.top_k, 'reranker_variant': args.reranker_variant, 'embedder': embedder}
        if 'docs_only' in arms:
            res=evaluate_package('docs_only', docs_pkg, queries, jina, args.embed_model, backend, args, progress, seed=args.seed)
            add_strata(res, query_by_id, heldout, tx_tools); res.update(common); results['docs_only']=res; save_json(args.output_dir/'docs_only.json', res)
        if 'local_only' in arms:
            all_rows=[]; client_results=[]
            for cid,pkg in sorted(client_packages.items()):
                combined,h=combine_packages(f'{cid}_with_docs', [docs_pkg, pkg])
                q=[]
                for item in queries:
                    st=stratum_for(item, heldout, tx_tools)
                    if st == 'tx_one_owner' and any(owners.get(tool)==cid for tool in item.gold_tools if tool in owners):
                        continue
                    q.append(item)
                res=evaluate_package('local_only', combined, q, jina, args.embed_model, backend, args, progress, seed=args.seed, client_id=cid)
                add_strata(res, query_by_id, heldout, tx_tools); res.update({'combined_sha256': h}); client_results.append(res); all_rows.extend(res['rows']); save_json(args.output_dir/f'local_only_{cid}.json', res)
            merged=summarize_rows(all_rows); merged.update({'arm':'local_only', 'seed':args.seed, 'rows':all_rows, 'client_results': client_results}); add_strata(merged, query_by_id, heldout, tx_tools); merged.update(common); results['local_only']=merged; save_json(args.output_dir/'local_only.json', merged)
        synapse_pkg=None; synapse_hash=None; edge_log=[]
        if 'synapse' in arms or 'two_index' in arms:
            progress.log('arm_merge_begin', seed=args.seed, arm='synapse', merge_policy=args.merge_policy, client_artifact_count=sum(len(p.artifacts) for p in client_packages.values()))
            with temporary_env({'SYNAPSE_EDGE_MERGE_POLICY': args.merge_policy}):
                agg=EdgeAggregator(EdgeConfig(edge_id=f'e8_cross_client_seed_{args.seed}'))
                with merge_heartbeat(progress, seed=args.seed, arm='synapse', merge_policy=args.merge_policy):
                    merged=agg.merge_packages(list(client_packages.values()))
                edge_log=read_edge_conflict_log(agg)
            if merged is None:
                raise RuntimeError('synapse merge produced no package')
            synapse_pkg, synapse_hash=combine_packages('synapse_with_docs', [docs_pkg, merged])
            progress.log('arm_merge_done', seed=args.seed, arm='synapse', artifact_count=len(synapse_pkg.artifacts), package_sha256=synapse_hash)
        if 'synapse' in arms:
            res=evaluate_package('synapse', synapse_pkg, queries, jina, args.embed_model, backend, args, progress, seed=args.seed)
            add_strata(res, query_by_id, heldout, tx_tools); res.update(common); res.update({'compendium': {'global_sha256': synapse_hash, 'artifact_count': len(synapse_pkg.artifacts)}, 'edge_conflict_log': edge_log}); results['synapse']=res; save_json(args.output_dir/'synapse.json', res)
        if 'concat' in arms:
            concat_pkg, concat_hash=combine_packages('concat_with_docs', [docs_pkg, *[client_packages[k] for k in sorted(client_packages)]])
            res=evaluate_package('concat', concat_pkg, queries, jina, args.embed_model, backend, args, progress, seed=args.seed)
            add_strata(res, query_by_id, heldout, tx_tools); res.update(common); res.update({'compendium': {'global_sha256': concat_hash, 'artifact_count': len(concat_pkg.artifacts)}, 'merge_mode': 'server_concatenate_no_dedup_no_field_merge'}); results['concat']=res; save_json(args.output_dir/'concat.json', res)
        if 'two_index' in arms:
            exp_pkg=experience_only_package(synapse_pkg)
            res=evaluate_two_index(docs_pkg, exp_pkg, synapse_pkg, queries, jina, args.embed_model, backend, args, progress, seed=args.seed)
            add_strata(res, query_by_id, heldout, tx_tools); res.update(common); res.update({'source_compendium_sha256': synapse_hash, 'experience_artifact_count': len(exp_pkg.artifacts)}); results['two_index']=res; save_json(args.output_dir/'two_index.json', res)
        summary={'paper_eligible': True, 'repo_commit': commit, 'input_repo_commit': input_summary['repo_commit'], 'seed': args.seed, 'config': {'arms': sorted(arms), 'input_dir': str(args.input_dir), 'retrieval_mode': args.retrieval_mode, 'retrieval_pool_size': args.retrieval_pool_size, 'merge_policy': args.merge_policy}, 'aggregate': aggregate(results)}
        save_json(args.output_dir/'summary.json', summary)
    progress.log('complete', output_dir=str(args.output_dir))
    print(json.dumps(summary['aggregate'], indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
