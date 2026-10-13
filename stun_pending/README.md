# STUN-Bench pending experiments (P0 to P7)

Code for the experiments marked `[PENDING]` in the v5 draft. The scripts are standalone: they do not import your existing harness, so you export your data once into the formats below and plug in your reranker prompt. Every script has been run end to end on synthetic data (`python -m stun.toy_data`). None has been run on the real pool, so the first step for each is the reproduction check in "Before you trust any number".

## Data you export once

| File | Rows |
| --- | --- |
| `catalog.jsonl` | `tool_id`, `name`, `description`, `category` for the 3,180 tools |
| `tests.jsonl` | `qid`, `query`, `gold` (list of tool ids), `group` (`G1_tool`, `G1_category`, `G1_instruction`, `G2_category`, `G2_instruction`, `G3_instruction`) |
| `heldout.json` | the 172 tool ids of H |
| `clients/seed{42,123,456}/client{0..4}.jsonl` | the experience items each client holds on the STUN-Bench pool: `tool_id`, `text`, `kind` (`scenario` or `precaution`). Export the items before any merge |
| `leak_filtered_pool.jsonl` (P3 only) | `item_id`, `query`, `labels` (list), `meta_text` (tool, API and category names joined) |

## Before you trust any number

1. Replace `LLMIndexReranker.PROMPT` and `default_render` in `stun/common.py` with your production prompt and candidate rendering (description first, then "when to use" and "do not use when" text). The numbers are only comparable to the paper if these match.
2. Run P0 under Jina-v2 and check the `docs` arm: it should reproduce 0.534 held-out accuracy and 0.689 held-out recall@5. If it does not, the export or the adapter differs from the harness, and nothing else should be run until it does.
3. Keep seeds 42/123/456 and the walk-down rule (five distinct tools, cap 200), which `walk_down` implements.

## Run order

P0 first, because every shared-index number in the paper changes with it. Then P1 and P2 (Vivek's imbalance and normalization points), P6, P3, P4 and P5. P7 is a reading task.

## P0. Concatenated pool (blocks the paper)

- Why: every shared-index result in v5 uses SYNAPSE's per-tool merged pool. The benchmark paper must not depend on the method paper's merge, and the two papers cannot share results while both are under review.
- Design: three arms on STUN-Bench: the description index, the shared index with every client item as its own entry (no clustering, majority or conflict handling), and the two-index router (3 description + 2 experience slots).
- Run once per retriever reported in the paper:
  ```
  python -m stun.p0_concat --catalog data/catalog.jsonl --tests data/tests.jsonl --heldout data/heldout.json \
    --clients-root data/clients --retriever jinaai/jina-embeddings-v2-base-en --out results/p0_jina
  ```
  Repeat with `BAAI/bge-base-en-v1.5`, `bm25`, `Qwen/Qwen3-Embedding-8B`, the clean BGE checkpoint path, `mangopy/ToolRet-trained-e5-base-v2` and the ToolBench-IR checkpoint. For retrieval-only rows (the retriever table, the gap table, depth 200), `--reranker top1` gives recall without the LLM.
- Cost: about 729 queries × 3 arms × 3 seeds = 6,561 reranker calls per retriever.
- Output: `summary.json` (accuracy and recall per stratum, mean and SD over seeds) and paired tests (shared minus docs, two-index minus docs) with query- and tool-clustered intervals. `python -m stun.make_table results/p0_*/summary.json` prints LaTeX rows.
- What changes in the paper: replace every `$^{p}$` row and the 0.278 headline. If the held-out loss under Jina stays above about 20 points, the abstract stands with new numbers. If it shrinks a lot, the abstract and RQ2 heading must be rewritten before anything else.

## P1. Matched-content control (Vivek: indexing imbalance)

- Question: is crowding caused by which tools get extra entries, or by those entries being written like queries?
- Design (2 × 2, plus the real concatenated index for reference). Every covered tool gets exactly 5 extra entries, each cut to the same number of words.
  - Style: query style vs description style with the same content. For used tools, the query-style entry is the real experience item and the description-style one is an LLM rewrite of it. Unused tools get both styles generated from their description.
  - Coverage: only used tools get entries (like the real index), or every tool does.
- Steps:
  ```
  python -m stun.p1_generate --catalog data/catalog.jsonl --heldout data/heldout.json \
    --clients-dir data/clients/seed42 --out data/p1/seed42/gen.jsonl      # repeat per seed
  python -m stun.p1_matched ... --gen-root data/p1 --words 12 --out results/p1_jina
  ```
  Set `--words` near the median word count of the real experience items, and read the printed entry counts to check that each cell kept enough tools.
- Cost: generation is about one call per used-tool item (up to 5 per tool) plus two per unused tool; ranking is 6 arms × 729 × 3 seeds.
- Reading the result:
  - Style effect: used/query minus used/desc.
  - Coverage effect: used/query minus all/query.
  - If the "all" cells come close to the description index, crowding is mainly which tools get entries. If used/query crowds much more than used/desc, the query-like wording matters.
  - Report both effects in RQ2's "Construction controls" paragraph, whichever way they come out.
- Check before running: every tool in a cell must have exactly `--n` entries. The script drops tools with fewer, so count the held-out tools covered in the "all" cells.

## P2. Score normalization and frozen BM25 statistics (Vivek: score normalization)

- Variants on the same concatenated shared index:
  - `max`: the best entry per tool, same as the paper's walk-down.
  - `mean`: the mean entry score per tool.
  - `maxcorr`: the max minus σ·E[max of n normals]. This removes the expected advantage a tool gets only from having n entries.
  - `srcz`: z-score description and experience scores separately per query, then walk down.
  - `bm25frozen`: BM25 with IDF and average length taken from the description index only.
  ```
  python -m stun.p2_normalize ... --retriever jinaai/jina-embeddings-v2-base-en --out results/p2_jina
  python -m stun.p2_normalize ... --retriever bm25 --out results/p2_bm25
  ```
- Reading: if crowding survives `maxcorr` and `srcz`, it is not an artifact of entry count or score scale. If one of them removes most of it, report that as a cheap fix and compare it with the two-index router in RQ3.
- Caution: `srcz` puts both sources on equal footing per query and can over-promote descriptions on labeled queries. Report labeled accuracy beside held-out.

## P3. Alternative held-out sets (Vivek: justify the held-out tools)

- `g2`: H plus every gold tool of the G2-category queries.
- `matched`: a set the size of H drawn from G1-instruction gold tools (tools ToolBench calls seen), matched to H on category and on pool frequency.
  ```
  python -m stun.p3_holdout ... --pool data/leak_filtered_pool.jsonl --variant g2 --out data/holdout_g2
  ```
  Then sample client experience from `data/holdout_g2/pool.jsonl` with your existing sampler (5 clients, 5,000 items, category-coherent), and run P0 with `--heldout data/holdout_g2/heldout.json`.
- Check `info.json`: removal counts per rule, query strata sizes, and for `matched` the median frequencies of H and the new set.
- Also report the 80 mixed queries separately in every P0 summary. The scripts already split `labeled_clean` and `mixed`.

## P4. Slot-budget sweep (Vivek: fixed 3+2)

- Budgets 5+0, 4+1, 3+2, 2+3 and 1+4.
- Per seed, held-out tools are split 50/50 into development and test tools, so the split is tool-disjoint. Labeled queries are split at random. The budget with the best development accuracy over both kinds of queries is chosen. Every budget is reported on the test queries, with the chosen one marked.
  ```
  python -m stun.p4_allocation ... --retriever jinaai/jina-embeddings-v2-base-en --out results/p4_jina
  ```
  Run it for Jina, BM25 and the clean BGE model.
- Ties go to more description slots.
- What changes: the "slot budget" paragraph in RQ3 reports which budget the development set picks and how sensitive test accuracy is to the choice. If 3+2 is not chosen, say so; 3+2 was fixed before any run.

## P5. Why clean BGE's router gains held-out accuracy while recall falls

- Run with `--retriever <clean BGE checkpoint>`. Three analyses:
  1. Candidate source × rendering cross: description index vs two-index router, crossed with description-only vs description+experience rendering.
  2. Accuracy on held-out queries whose gold tool is in both candidate lists.
  3. Candidate positions shuffled 5 times.
- Reading:
  - If the gain needs experience rendering, the reader uses the experience text.
  - If it holds when the gold tool is in both lists, the two-index list has easier distractors.
  - If shuffling removes it, it came from position.

## P6. Second clean tool-trained retriever, end to end

- Use your ToolBench-IR rerun trained without the held-out tools' items (the "removed" arm of Table tbir). Export it as a sentence-transformers checkpoint and run P0 with `--retriever <path>`.
- Report it beside the clean BGE model in Table 2's bottom block. RQ2's "much less" claim then rests on two models instead of one.

## P7. Do published systems put experience and descriptions in one index? (reading task)

For each experience-sharing method cited in the introduction and related work (FICAL, the shared-memory papers, FedTextGrad, Fed-ICL, Federation of Thoughts, plus FedAgentKE), record from the paper and, where released, the code:

| Method | Where experience is stored | Retrieved together with tool descriptions? | Evidence (section, figure or file:line) |
| --- | --- | --- | --- |

If most systems retrieve experience separately, say so in the paper and frame crowding as the risk of the simplest design. If several share one index, cite them where crowding is introduced.

## Lower priority

- Raw stored queries vs written scenarios as the experience format: P1's style axis covers most of this.
- A reranker larger than 8B: rerun P0 with `--reranker-model` set to the larger model on one seed.
- Significance tests for the independent-pool cells: rerun P0 retrieval-only (`--reranker top1`) on the independent pool and read the paired intervals.
