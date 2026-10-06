#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, random, statistics, hashlib
from pathlib import Path
from typing import Any

ROOT=Path('.')
OUT=Path('artifacts/verification/review_tier0_stats_r1')
SEEDS=(42,123,456)

def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())

def rows(path: Path) -> list[dict[str, Any]]:
    return read(path).get('rows') or []

def outcome(row: dict[str, Any], name: str) -> bool:
    if name == 'correct':
        return bool(row.get('correct', row.get('routed_correctly')))
    if name == 'recall':
        return bool(row.get('gold_in_top_k', row.get('gold_in_top_5')))
    if name == 'top1':
        return bool(row.get('top1_correct', row.get('retrieval_top1_correct')))
    raise ValueError(name)

def qid(row: dict[str, Any]) -> str:
    return str(row.get('row_id') or row.get('query_id'))

def tool(row: dict[str, Any]) -> str:
    vals = row.get('gold_parent_tools') or row.get('gold_tools') or row.get('gold_tool') or []
    if isinstance(vals, list) and vals:
        return str(vals[0])
    if isinstance(vals, str):
        return vals
    return str(row.get('group') or 'unknown')

def load_map(path: Path, *, subset: str|None, outcome_name: str) -> dict[str, tuple[bool, dict[str, Any]]]:
    out={}
    for r in rows(path):
        if subset is not None and str(r.get('subset')) != subset:
            continue
        key=qid(r)
        if key in out:
            raise ValueError(f"duplicate paired key {key!r} in {path} subset={subset}")
        out[key] = (outcome(r, outcome_name), r)
    return out

def pair_file_rows(a_path: Path, b_path: Path, *, seed: int, subset: str|None, outcome_name: str, a_label: str, b_label: str) -> list[dict[str, Any]]:
    if not a_path.exists() or not b_path.exists():
        return []
    a=load_map(a_path, subset=subset, outcome_name=outcome_name)
    b=load_map(b_path, subset=subset, outcome_name=outcome_name)
    out=[]
    for k in sorted(set(a)&set(b)):
        av, ar = a[k]; bv, br = b[k]
        out.append({'query_id': k, 'seed': str(seed), 'tool_id': tool(ar), 'correct_a': int(av), 'correct_b': int(bv)})
    return out

def rate(vals: list[int]) -> float:
    return sum(vals)/len(vals) if vals else 0.0

def diff_pts(rs: list[dict[str, Any]]) -> float:
    return 100*(rate([r['correct_b'] for r in rs])-rate([r['correct_a'] for r in rs]))

def cluster_boot(rs: list[dict[str, Any]], key: str, B: int, alpha: float, seed: int) -> tuple[float,float]:
    rng=random.Random(seed)
    groups={}
    for r in rs:
        groups.setdefault(str(r[key]), []).append(r)
    ids=list(groups)
    if not ids: return (0.0,0.0)
    vals=[]
    for _ in range(B):
        sample=[]
        for gid in rng.choices(ids, k=len(ids)):
            sample.extend(groups[gid])
        vals.append(diff_pts(sample))
    vals.sort()
    return vals[int((alpha/2)*B)], vals[min(B-1, int((1-alpha/2)*B))]

def iid_boot(rs: list[dict[str, Any]], B: int, alpha: float, seed: int) -> tuple[float,float]:
    rng=random.Random(seed)
    if not rs: return (0.0,0.0)
    vals=[]
    n=len(rs)
    for _ in range(B):
        vals.append(diff_pts([rs[i] for i in (rng.randrange(n) for _ in range(n))]))
    vals.sort()
    return vals[int((alpha/2)*B)], vals[min(B-1, int((1-alpha/2)*B))]

def mcnemar_counts(rs: list[dict[str, Any]]) -> dict[str, Any]:
    a_only=sum(r['correct_a']==1 and r['correct_b']==0 for r in rs)
    b_only=sum(r['correct_a']==0 and r['correct_b']==1 for r in rs)
    n=a_only+b_only
    if n==0:
        p=1.0
    elif n <= 1024:
        p=min(1.0, 2*sum(math.comb(n,i) for i in range(min(a_only,b_only)+1))/(2**n))
    else:
        stat=(abs(a_only-b_only)-1)**2/n
        p=math.erfc(math.sqrt(stat/2))
    return {'a_only': a_only, 'b_only': b_only, 'discordant': n, 'p_exact_two_sided': p}

def summarize_pair(name: str, rs: list[dict[str, Any]], *, margin_pts: float=2.0, bootstrap: bool=True, B: int=2000) -> dict[str, Any]:
    d=diff_pts(rs) if rs else 0.0
    q95=cluster_boot(rs, 'query_id', B, .05, 20261001) if (rs and bootstrap) else (0,0)
    q90=cluster_boot(rs, 'query_id', B, .10, 20261002) if (rs and bootstrap) else (0,0)
    t95=cluster_boot(rs, 'tool_id', B, .05, 20261003) if (rs and bootstrap) else (0,0)
    u95=iid_boot(rs, B, .05, 20261004) if (rs and bootstrap) else (0,0)
    mc=mcnemar_counts(rs)
    return {
        'name': name, 'n': len(rs),
        'a_rate': rate([r['correct_a'] for r in rs]), 'b_rate': rate([r['correct_b'] for r in rs]),
        'b_minus_a_pts': d,
        'query_cluster_ci95': q95, 'query_cluster_half_width95': (q95[1]-q95[0])/2 if bootstrap else None,
        'query_cluster_ci90': q90,
        'tool_cluster_ci95': t95,
        'iid_ci95': u95, 'iid_half_width95': (u95[1]-u95[0])/2 if bootstrap else None,
        'tost_margin_pts': margin_pts,
        'tost_pass_query_cluster_90ci': (q90[0] > -margin_pts and q90[1] < margin_pts) if bootstrap else None,
        'mcnemar': mc,
    }

def write_csv(path: Path, rs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['query_id','seed','tool_id','correct_a','correct_b'])
        w.writeheader(); w.writerows(rs)

def e6_root(seed:int)->Path:
    return Path('artifacts/results/stabletoolbench_heldout_retriever_compare_e6_r4') if seed==42 else Path(f'artifacts/results/stabletoolbench_heldout_retriever_compare_e6_seed{seed}_r4')

def e6_walkdown_root(retriever: str) -> Path:
    if retriever == 'bm25':
        return Path('artifacts/results/stabletoolbench_heldout_retriever_walkdown_e6_bm25_r1')
    return Path('artifacts/results/stabletoolbench_heldout_retriever_walkdown_e6_jina_bge_r1')

def add_comp(comps, name, pairs, subset=None, outcome_name='correct', a_label='a', b_label='b'):
    rs=[]; missing=[]
    for seed,a,b in pairs:
        prs=pair_file_rows(Path(a), Path(b), seed=seed, subset=subset, outcome_name=outcome_name, a_label=a_label, b_label=b_label)
        if not prs: missing.append([seed,str(a),str(b)])
        rs.extend(prs)
    item=summarize_pair(name, rs)
    item.update({'a_label': a_label, 'b_label': b_label, 'subset': subset, 'outcome': outcome_name, 'missing': missing})
    comps.append(item)
    write_csv(OUT/'paired_rows'/f'{name}.csv', rs)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comps=[]
    # R1 concat-vs-synapse all and subset.
    add_comp(comps, 'r1_full729_concat_vs_synapse', [(s, f'artifacts/verification/stabletoolbench_benchmark_cached_replay_r2/seed_{s}/synapse.json', f'artifacts/results/stabletoolbench_concat_control_distinct5_r2/full/conflict_00/seed_{s}.json') for s in SEEDS], outcome_name='correct', a_label='synapse', b_label='concat')
    for ratev in (0,60):
        add_comp(comps, f'r1_subset258_rate{ratev:02d}_concat_vs_synapse', [(s, f'artifacts/results/stabletoolbench_typed_conflictlog_distinct5_table4_r1/conflict_{ratev:02d}/seed_{s}.json', f'artifacts/results/stabletoolbench_concat_control_distinct5_r2/subset_g1g2_instruction/conflict_{ratev:02d}/seed_{s}.json') for s in SEEDS], outcome_name='correct', a_label='synapse', b_label='concat')
    # R1 rendering, typed merge only.
    model_roots={
      'llama31_8b': {42:'artifacts/results/stabletoolbench_frozen_renderswap_seed42_r1',123:'artifacts/results/stabletoolbench_frozen_renderswap_seed123_r1',456:'artifacts/results/stabletoolbench_frozen_renderswap_seed456_r1'},
      'qwen25_7b': {42:'artifacts/results/stabletoolbench_frozen_renderswap_qwen25_7b_seeds42_123_e2x_r1',123:'artifacts/results/stabletoolbench_frozen_renderswap_qwen25_7b_seeds42_123_e2x_r1',456:'artifacts/results/stabletoolbench_frozen_renderswap_qwen25_7b_seed456_r1'},
      'qwen25_3b': {42:'artifacts/results/stabletoolbench_frozen_renderswap_qwen25_3b_seeds42_123_e2x_r5',123:'artifacts/results/stabletoolbench_frozen_renderswap_qwen25_3b_seeds42_123_e2x_r5',456:'artifacts/results/stabletoolbench_frozen_renderswap_qwen25_3b_seed456_e2x_r1'},
    }
    for model, roots in model_roots.items():
        for ratev in (0,60):
            add_comp(comps, f'r1_render_{model}_typedmerge_rate{ratev:02d}_json_vs_typed', [(s, f'{root}/typed_conflictlog/conflict_{ratev:02d}/typed/seed_{s}.json', f'{root}/typed_conflictlog/conflict_{ratev:02d}/flat/seed_{s}.json') for s,root in roots.items()], outcome_name='correct', a_label='typed_render', b_label='json_render')
    # R1 conflict policies pairwise across available seeds/rates.
    roots={42:'artifacts/results/stabletoolbench_2x2_seed42_r3b',456:'artifacts/results/stabletoolbench_2x2_seed456_r2',7:'artifacts/results/stabletoolbench_2x2_extra_seeds_r3',1234:'artifacts/results/stabletoolbench_2x2_extra_seeds_r3'}
    policy_pairs=[('typed_conflictlog','typed_majority'),('typed_conflictlog','typed_round_delayed'),('typed_majority','typed_round_delayed')]
    for ratev in (0,20,40,60):
        for aarm,barm in policy_pairs:
            pairs=[]
            for s,root in roots.items():
                a=Path(root)/aarm/f'conflict_{ratev:02d}'/f'seed_{s}.json'; b=Path(root)/barm/f'conflict_{ratev:02d}'/f'seed_{s}.json'
                if a.exists() and b.exists(): pairs.append((s,a,b))
            add_comp(comps, f'r1_conflict_rate{ratev:02d}_{barm}_vs_{aarm}', pairs, outcome_name='correct', a_label=aarm, b_label=barm)
    # R2 Table 2-like McNemar counts: STB split, ToolBench G1, sparse independent, ToolBench-IR on/off pool.
    r2=[]
    def add_r2(name,pairs,subset='heldout',outcome_name='recall'):
        rs=[]
        for seed,a,b in pairs:
            rs.extend(pair_file_rows(Path(a),Path(b),seed=seed,subset=subset,outcome_name=outcome_name,a_label='docs',b_label='shared'))
        s=summarize_pair(name,rs,bootstrap=False); s['mcnemar_counts']=s.pop('mcnemar'); r2.append(s); write_csv(OUT/'mcnemar_rows'/f'{name}.csv',rs)
    for retr in ('jina','bm25','bge'):
        root=e6_walkdown_root(retr)
        add_r2(f'r2_stabletoolbench_{retr}_shared_vs_docs_heldout', [(s,root/f'seed_{s}/{retr}/docs_only.json', root/f'seed_{s}/{retr}/synapse_shared.json') for s in SEEDS])
    for retr in ('jina','bm25','bge'):
        add_r2(f'r2_toolbench_g1_{retr}_shared_vs_docs_heldout', [(s,f'artifacts/results/toolbench_retrieval_heldout_e7_r2/G1/seed_{s}/{retr}/docs_only.json', f'artifacts/results/toolbench_retrieval_heldout_e7_r2/G1/seed_{s}/{retr}/shared.json') for s in SEEDS])
    for retr in ('jina','bm25','bge'):
        add_r2(f'r2_independent_sparse_{retr}_shared_vs_docs_heldout', [(s,f'artifacts/results/toolret_sparse_heldout_e7_r1/seed_{s}/{retr}/docs_only.json', f'artifacts/results/toolret_sparse_heldout_e7_r1/seed_{s}/{retr}/shared.json') for s in SEEDS])
    add_r2('r2_toolbench_ir_onpool_shared_vs_docs_heldout', [(s,f'artifacts/results/stabletoolbench_toolbench_ir_e6_r2/seed_{s}/toolbench_ir/docs_canonical.json', f'artifacts/results/stabletoolbench_toolbench_ir_e6_r2/seed_{s}/toolbench_ir/synapse_shared.json') for s in SEEDS])
    add_r2('r2_toolbench_ir_offpool_shared_vs_docs_heldout', [(s,f'artifacts/results/toolret_sparse_toolbench_ir_e7_r1/seed_{s}/toolbench_ir/docs_only.json', f'artifacts/results/toolret_sparse_toolbench_ir_e7_r1/seed_{s}/toolbench_ir/shared.json') for s in SEEDS])
    add_r2('r2_toolbench_ir_offpool_native_shared_vs_docs_heldout', [(s,f'artifacts/results/toolret_sparse_toolbench_ir_native_e7_r1/seed_{s}/toolbench_ir/docs_native.json', f'artifacts/results/toolret_sparse_toolbench_ir_native_e7_r1/seed_{s}/toolbench_ir/shared.json') for s in SEEDS])
    # R3 requested comparisons.
    r3=[]
    def add_r3(name,pairs,subset=None,outcome_name='correct'):
        rs=[]
        for seed,a,b in pairs:
            rs.extend(pair_file_rows(Path(a),Path(b),seed=seed,subset=subset,outcome_name=outcome_name,a_label='a',b_label='b'))
        s=summarize_pair(name,rs,bootstrap=False); s['mcnemar_counts']=s.pop('mcnemar'); r3.append(s); write_csv(OUT/'mcnemar_rows'/f'{name}.csv',rs)
    add_r3('r3_docs_vs_two_index_budget_jina_heldout_accuracy', [(s,e6_root(s)/f'seed_{s}/jina/docs_only.json', f'artifacts/results/stabletoolbench_two_index_router_e8_budgeted_distinct5_r2/seed_{s}/jina/docs_plus_experience_backfill.json') for s in SEEDS], subset='heldout', outcome_name='correct')
    add_r3('r3_synapse_vs_centralized_full729_accuracy', [(s,f'artifacts/verification/stabletoolbench_benchmark_cached_replay_r2/seed_{s}/synapse.json', f'artifacts/verification/stabletoolbench_benchmark_cached_replay_r2/seed_{s}/centralized.json') for s in SEEDS], outcome_name='correct')
    for mode in ('real','bounded_oracle'):
        for render in ('shared_candidates_compendium_render','shared_candidates_docs_render'):
            # P1 has no docs-vs-shared pair; skip file-level if rows not present.
            pass
    # Table 10 likely D3 synthetic expansion: compare synthetic_docs vs synthetic_shared on heldout recall.
    d3=Path('artifacts/results/stabletoolbench_symmetric_expansion_d3_r2')
    for retr in ('jina','bm25','bge'):
        add_r3(f'r3_table10_d3_{retr}_synthetic_plus_experience_vs_docs_heldout_recall', [(s,d3/f'seed_{s}/{retr}/synthetic_docs.json', d3/f'seed_{s}/{retr}/synthetic_plus_experience.json') for s in SEEDS], subset='heldout', outcome_name='recall')
    out={'r1_clustered': comps, 'r2_table2_mcnemar': r2, 'r3_extra_mcnemar': r3}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    # TSV summaries.
    for key,items in [('r1_clustered.tsv',comps),('r2_table2_mcnemar.tsv',r2),('r3_extra_mcnemar.tsv',r3)]:
        with (OUT/key).open('w') as f:
            f.write('name\tn\ta_rate\tb_rate\tdiff_pts\tq95_low\tq95_high\tq95_halfwidth\tiid95_low\tiid95_high\tiid95_halfwidth\ttost_pass\ta_only\tb_only\tp\n')
            for x in items:
                mc=x.get('mcnemar') or x.get('mcnemar_counts')
                half = '' if x['query_cluster_half_width95'] is None else f"{x['query_cluster_half_width95']:.3f}"
                ihalf = '' if x.get('iid_half_width95') is None else f"{x.get('iid_half_width95'):.3f}"
                ici = x.get('iid_ci95') or (0,0)
                f.write(f"{x['name']}\t{x['n']}\t{x['a_rate']:.6f}\t{x['b_rate']:.6f}\t{x['b_minus_a_pts']:.3f}\t{x['query_cluster_ci95'][0]:.3f}\t{x['query_cluster_ci95'][1]:.3f}\t{half}\t{ici[0]:.3f}\t{ici[1]:.3f}\t{ihalf}\t{x['tost_pass_query_cluster_90ci']}\t{mc['a_only']}\t{mc['b_only']}\t{mc['p_exact_two_sided']:.6g}\n")
    print('WROTE', OUT/'summary.json')
    print('R1 cells', len(comps), 'R2 cells', len(r2), 'R3 cells', len(r3))

if __name__=='__main__': main()
