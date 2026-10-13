# Anonymous STUN-Bench Release

This repository is a one-commit anonymous artifact release for the submitted
paper. It contains the evaluation harness, provenance records, sanitized result
summaries, hashed per-query logs, release filter manifest, split definitions,
and generated expansion queries needed to verify the reported tables.

It intentionally excludes raw ToolBench/StableToolBench data, raw query or
scenario text in logs, model weights not trained by us, paper drafts, git
history, API keys, local paths, and personal identifiers.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

To rebuild the StableToolBench release filter from public upstream data, place
ToolBench and StableToolBench at repo-local paths and run:

```bash
python scripts/export_stabletoolbench_release_filter_manifest.py \
  --stb-root external_datasets/StableToolBench \
  --toolbench-instruction-dir external_datasets/toolbench_hf/instruction \
  --output-dir rebuild/filters
```

Expected checks:

- kept items after all filters: `90704`
- filter manifest SHA-256: `750ffafecee965d59ff79074a7384be2841ae6129efdfc52c24864f61f4dcf4f`
- held-out tool-list SHA-256: `8e84717a6e4fbe985efe03e41c258cd04d4f61917cc1efb15df9f5471e200260`

## Models referenced

No third-party model weights are redistributed. Use these model IDs/weights:

- `jinaai/jina-embeddings-v2-base-en`
- `BAAI/bge-base-en-v1.5`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `ToolBench/ToolBench_IR_bert_based_uncased`

Fine-tuned BGE checkpoints are not bundled unless present under
`release/checkpoints/`; their hashes and recipes are in the result summaries.

## Table to script map

The machine-readable table-to-script and run map is in
`release/results/table_manifest_sanitized.json`. Key families:

- Filter/split audit: `scripts/export_stabletoolbench_release_filter_manifest.py`
- Label-held-out split: `scripts/run_stabletoolbench_heldout.py`, `scripts/replay_stabletoolbench_packages.py`
- Retriever comparisons/Table 2: `scripts/run_stabletoolbench_heldout_retriever_compare.py`, `scripts/run_stabletoolbench_toolbench_ir_e6.py`
- Category hold-out E7: `scripts/prepare_stabletoolbench_category_holdout_e7.py`, `scripts/run_stabletoolbench_category_holdout_e7.py`, `scripts/run_stabletoolbench_category_holdout_e7_two_index.py`
- Cross-client E8: `scripts/prepare_stabletoolbench_cross_client_e8.py`, `scripts/run_stabletoolbench_cross_client_e8.py`
- Frozen-candidate/rendering controls: `scripts/run_stabletoolbench_frozen_render_swap.py`
- Oracle/gold insertion: `scripts/run_stabletoolbench_oracle.py`, `scripts/run_stabletoolbench_heldout_oracle_replay.py`
- Two-index/fusion: `scripts/run_stabletoolbench_two_index_router.py`
- Expansion controls: `scripts/run_stabletoolbench_symmetric_expansion_d3.py`, `scripts/run_toolret_sparse_symmetric_expansion_e1.py`
- Clean retriever E2: `scripts/export_clean_retriever_e2_inputs.py`, `scripts/train_clean_retriever_e2.py`, `scripts/evaluate_clean_retriever_e2_toolret_sparse.py`
- Statistics: `scripts/run_review_tier0_stats.py`, `scripts/clustered_paired_uncertainty.py`, `scripts/analyze_paired_tests_d5.py`
- Table generation: `scripts/make_tables.py`

## Provenance

`PROVENANCE.tsv` records `run_dir`, `recorded_commit`, `script_path`, and the
Git blob hash for the script version used. If a script changed after a run, the
exact historical script is stored under `provenance/<commit>/...` and the blob
hash points there. Full git history is withheld for anonymity during review.

## Data statement

Raw ToolBench/StableToolBench queries, catalogs, instruction-pool items, and
model weights are not redistributed. Sanitized logs hash query text and retain
only IDs, candidates, predictions, correctness, subsets, seeds, and run IDs.
Generated expansion queries under `release/generated/` are model outputs used to
reproduce the expansion tables without rerunning generation.

## Citation

Anonymous submission. Citation information will be added after review.
