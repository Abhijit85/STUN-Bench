#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
GENERATED_DIR = TABLES_DIR / "generated"
MANIFEST_PATH = TABLES_DIR / "manifest.json"
HAND_NUMBERS_PATH = TABLES_DIR / "numbers.tex"
GENERATED_NUMBERS_PATH = GENERATED_DIR / "numbers.tex"
RUNS_ROOT = Path("<REPO_ROOT>_runs")

TAU_TRANSFER_MANIFEST = REPO_ROOT / "artifacts" / "results" / "tau_bench_a5_sidecars_canonical.json"
LESSONS_CLASSIFIER_SMOKE = REPO_ROOT / "artifacts" / "verification" / "stb_precondition_smoke_classifier" / "combined_summary.json"
STB_CLEAN_CLASSIFIER_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_query_classifier_clean_r1" / "combined_summary.json"
STB_TRULY_UNSEEN_ANALYSIS = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_truly_unseen_analysis.json"
STB_C1_COS090_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_c1_cos090_seed42_r1" / "combined_summary.json"
STB_C1_COS090_DISTINCT5_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_c1_cos090_distinct5_seed42_r4" / "combined_summary.json"
STB_C1_COS090_QC_REFIT_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_c1_query_classifier_cos090_seed42_clientdraw_r1" / "combined_summary.json"
STB_C1_COS090_QC_REFIT_SEED = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_c1_query_classifier_cos090_seed42_clientdraw_r1" / "seed_42.json"
STB_CONCAT_CONTROL_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_concat_control_distinct5_r2" / "summary.json"
STB_E8_BUDGETED_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_two_index_router_e8_budgeted_distinct5_r2" / "summary.json"
STB_E8_RRF_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_two_index_router_e8_r1" / "summary.json"
STB_D3_PROGRESS = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_symmetric_expansion_d3_r2" / "progress.jsonl"
STB_D3_LAUNCH_LOG = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_symmetric_expansion_d3_r2.launch.log"
STB_D6_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_fusion_comparison_d6_r2" / "summary.json"
STB_D5_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "d5_paired_tests_r3" / "summary.json"
STB_P1_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_candidate_render_interaction_p1_r3" / "summary.json"
STB_D4_DIR = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_split_sensitivity_d4_r3"
STB_E5B_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_e5b_seeds123456_r3" / "summary.json"
STB_HELDOUT_ORACLE_DISTINCT5_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_oracle_distinct5_r2" / "summary.json"
TOOLRET_DENSE_AUDIT_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "toolret_provenance_audit_r1" / "summary.json"
TOOLRET_SPARSE_AUDIT_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "toolret_sparse_provenance_audit_r1" / "summary.json"
TOOLRET_STB_OVERLAP_EXACT_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "toolret_stabletoolbench_overlap_exact_r1" / "summary_729_exact.json"
TOOLRET_STB_OVERLAP_NEARDUP_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "toolret_stabletoolbench_overlap_r1" / "summary_729.json"
STB_BENCHMARK_DISTINCT5_CELLS = {
    ("synapse", 42): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_42" / "synapse.json",
    ("synapse", 123): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_123" / "synapse.json",
    ("synapse", 456): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_456" / "synapse.json",
    ("centralized", 42): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_42" / "centralized.json",
    ("centralized", 123): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_123" / "centralized.json",
    ("centralized", 456): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_456" / "centralized.json",
    ("flat_pool", 42): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_42" / "flat_pool.json",
    ("flat_pool", 123): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_123" / "flat_pool.json",
    ("flat_pool", 456): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_cached_replay_r2" / "seed_456" / "flat_pool.json",
    ("local_only", 42): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_localonly_distinct5_r3" / "seed_42" / "local_only.json",
    ("local_only", 123): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_localonly_distinct5_r3" / "seed_123" / "local_only.json",
    ("local_only", 456): REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_benchmark_localonly_distinct5_r3" / "seed_456" / "local_only.json",
}
STB_HELDOUT_E5_SUMMARY = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_r3b" / "summary.json"
D1_D2_CANDIDATE_UNDERFILL_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "d1_d2_candidate_underfill_r1" / "summary.json"
STB_HELDOUT_D1B_CELLS = {
    ("synapse", 42): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_r9" / "seed_42" / "synapse.json",
    ("centralized", 42): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_r9" / "seed_42" / "centralized.json",
    ("local_only", 42): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_r9" / "seed_42" / "local_only.json",
    ("flat_pool", 42): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_r9" / "seed_42" / "flat_pool.json",
    ("synapse", 123): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed123_r1" / "seed_123" / "synapse.json",
    ("centralized", 123): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed123_r1" / "seed_123" / "centralized.json",
    ("flat_pool", 123): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed123_flatonly_r1" / "seed_123" / "flat_pool.json",
    ("synapse", 456): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed456_r1" / "seed_456" / "synapse.json",
    ("centralized", 456): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed456_r1" / "seed_456" / "centralized.json",
    ("local_only", 456): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed456_r1" / "seed_456" / "local_only.json",
    ("flat_pool", 456): REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed456_r1" / "seed_456" / "flat_pool.json",
}
STB_HELDOUT_D1B_LOCAL_ONLY_REFUSAL_LOG = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed123_missing_r1" / "launch.log"
R4_CONTAMINATION_PROGRESS = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed42_r4" / "progress.jsonl"
R11_CONTAMINATION_PROGRESSS = [
    REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_a1_seed42_eval_r11" / "progress.jsonl",
    REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_a1_seed123_eval_r11" / "progress.jsonl",
    REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_a1_seed456_eval_r11" / "progress.jsonl",
]
CONFLICT_SOURCE_TEX = RUNS_ROOT / "paper" / "tables" / "generated" / "conflict_2x2.tex"
CONFLICT_SOURCE_NUMBERS = RUNS_ROOT / "paper" / "tables" / "generated" / "numbers_conflict.tex"
CONFLICT_SOURCE_MANIFEST = RUNS_ROOT / "paper" / "tables" / "generated" / "conflict_manifest.json"
EQUIVALENT_COMMITS_PATH = REPO_ROOT / "artifacts" / "provenance" / "equivalent_commits.json"
TAU_TRANSFER_PATHS = [
    "scripts/run_synapse_tau_retail.py",
    "scripts/evaluate_tau_bench_toolcall_accuracy.py",
    "scripts/extract_tau_bench_experience.py",
    "scripts/build_tau_bench_global_packages.py",
    "synapse",
    "external_datasets/tau_bench/tau-bench-main",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements.lock",
    "pyproject.toml",
]


class TableBlocked(RuntimeError):
    def __init__(self, reason: str, *, missing: list[str] | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.missing = missing or []
        self.details = details or {}


@dataclass
class BuiltTable:
    name: str
    tex_name: str
    tex_body: str
    numbers: dict[str, str]
    inputs: list[dict[str, Any]]
    details: dict[str, Any]


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        retrieval_backend = payload.get("retrieval_backend")
        paper_eligible = payload.get("paper_eligible")
        data_mode = payload.get("data_mode")
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        config_data_mode = config.get("data_mode")
        if retrieval_backend == "tfidf_scaffold":
            raise TableBlocked(f"refuses scaffold artifact {path}", details={"retrieval_backend": retrieval_backend})
        if paper_eligible is False:
            raise TableBlocked(f"refuses non-paper artifact {path}", details={"paper_eligible": False})
        if data_mode == "stable_holdout" or config_data_mode == "stable_holdout":
            raise TableBlocked(f"refuses wrong-mode artifact {path}", details={"data_mode": data_mode or config_data_mode})
    return payload


def load_equivalent_commit_groups() -> list[set[str]]:
    if not EQUIVALENT_COMMITS_PATH.exists():
        return []
    payload = load_json(EQUIVALENT_COMMITS_PATH)
    groups: list[set[str]] = []
    if isinstance(payload, dict):
        for group in payload.get("groups", []):
            if isinstance(group, list):
                groups.append({str(item) for item in group})
    return groups


def git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def commit_is_ancestor_of_head(commit: Any) -> bool | None:
    commit_str = str(commit or "").strip()
    if not commit_str:
        return None
    proc = subprocess.run(["git", "merge-base", "--is-ancestor", commit_str, "HEAD"], cwd=REPO_ROOT, text=True)
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def tree_hash_for_commit(commit: str, repo_path: str) -> str:
    try:
        return git_output(["git", "rev-parse", f"{commit}:{repo_path}"])
    except Exception as exc:
        raise TableBlocked(
            f"failed to compute tree hash for {commit}:{repo_path}",
            details={"commit": commit, "repo_path": repo_path, "error": str(exc)},
        ) from exc


def commits_equivalent_for_paths(commits: set[str], paths: list[str]) -> tuple[bool, list[dict[str, Any]]]:
    ordered = sorted(commit for commit in commits if commit)
    if len(ordered) <= 1:
        return True, []
    baseline = ordered[0]
    evidence: list[dict[str, Any]] = []
    all_equal = True
    for other in ordered[1:]:
        proc = subprocess.run(["git", "diff", "--quiet", baseline, other, "--", *paths], cwd=REPO_ROOT, text=True)
        equivalent = proc.returncode == 0
        if proc.returncode not in (0, 1):
            raise TableBlocked(
                "failed to compare repo commits for table dependency paths",
                details={"baseline": baseline, "other": other, "returncode": proc.returncode, "paths": paths},
            )
        diffstat = ""
        if not equivalent:
            diffstat = git_output(["git", "diff", "--stat", baseline, other, "--", *paths])
            all_equal = False
        evidence.append({
            "baseline": baseline,
            "other": other,
            "paths": paths,
            "equivalent": equivalent,
            "diffstat": diffstat,
        })
    return all_equal, evidence


def commits_compatible(commits: set[str], equivalent_groups: list[set[str]]) -> bool:
    if len(commits) <= 1:
        return True
    for group in equivalent_groups:
        if commits.issubset(group):
            return True
    return False


def stats(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def format_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write_table(path: Path, body: str) -> None:
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def artifact_record(path: Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        rendered_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        rendered_path = str(path)
    record = {"path": rendered_path, "sha256": sha256_file(path)}
    if extra:
        record.update(extra)
    if "repo_commit" in record and "repo_commit_is_ancestor_of_head" not in record:
        record["repo_commit_is_ancestor_of_head"] = commit_is_ancestor_of_head(record.get("repo_commit"))
    return record


def resolve_a1_summary_paths() -> list[Path]:
    seeds = [42, 123, 456]
    resolved: list[Path] = []
    search_roots = [RUNS_ROOT, REPO_ROOT]
    for seed in seeds:
        candidates: list[tuple[int, Path]] = []
        for root in search_roots:
            base = root / "artifacts" / "verification"
            if not base.exists():
                continue
            for summary_path in base.glob(f"stabletoolbench_a1_seed{seed}_eval_r*/combined_summary.json"):
                match = re.search(r"_r(\d+)/combined_summary\.json$", str(summary_path))
                run_id = int(match.group(1)) if match else -1
                candidates.append((run_id, summary_path))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        resolved.append(candidates[0][1])
    return resolved


def find_progress_stage(path: Path, stage: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("stage") == stage:
            found = payload
    return found


def contamination_counts(stage: dict[str, Any]) -> dict[str, int]:
    return {
        "pool_items_removed_exact": int(stage.get("pool_items_removed_exact", stage.get("exact_removed_train_item_count", 0))),
        "pool_items_removed_near_dup": int(stage.get("pool_items_removed_near_dup", stage.get("near_duplicate_removed_train_item_count", 0))),
        "pool_items_removed_total": int(stage.get("pool_items_removed_total", 0)) or (
            int(stage.get("pool_items_removed_exact", stage.get("exact_removed_train_item_count", 0)))
            + int(stage.get("pool_items_removed_near_dup", stage.get("near_duplicate_removed_train_item_count", 0)))
        ),
        "eval_queries_with_exact_match_in_pool": int(stage.get("eval_queries_with_exact_match_in_pool", stage.get("exact_removed_eval_query_count", 0))),
        "eval_queries_with_near_dup_in_pool": int(stage.get("eval_queries_with_near_dup_in_pool", stage.get("near_duplicate_removed_eval_query_count", 0))),
        "eval_queries_removed": int(stage.get("eval_queries_removed", 0)),
    }


def build_transfer_table() -> BuiltTable:
    if not TAU_TRANSFER_MANIFEST.exists():
        raise TableBlocked("missing canonical tau-bench sidecar manifest", missing=[str(TAU_TRANSFER_MANIFEST.relative_to(REPO_ROOT))])
    manifest = load_json(TAU_TRANSFER_MANIFEST)
    runs = manifest.get("runs") if isinstance(manifest, dict) else None
    if not isinstance(runs, list) or not runs:
        raise TableBlocked("canonical tau-bench sidecar manifest has no runs")

    equivalent_groups = load_equivalent_commit_groups()
    all_repo_commits: set[str] = set()
    all_harness_commits: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    input_records: list[dict[str, Any]] = [artifact_record(TAU_TRANSFER_MANIFEST)]
    provider_records: list[dict[str, str]] = []

    for run in runs:
        if not isinstance(run, dict):
            continue
        repo_commit = str(run.get("repo_commit") or run.get("tau_bench_commit") or "")
        harness_commit = str(run.get("tau_bench_harness_commit") or "")
        if not harness_commit and repo_commit:
            harness_commit = tree_hash_for_commit(repo_commit, "external_datasets/tau_bench/tau-bench-main")
        all_repo_commits.add(repo_commit)
        all_harness_commits.add(harness_commit)
        arm = str(run["arm"])
        seed = int(run["seed"])
        metrics_path = REPO_ROOT / run["metrics_file"]
        run_metadata_path = REPO_ROOT / run["run_metadata_file"]
        results_path = Path(run["results_file"])
        if not metrics_path.exists() or not run_metadata_path.exists() or not results_path.exists():
            missing = []
            for path in (metrics_path, run_metadata_path, results_path):
                if not path.exists():
                    try:
                        missing.append(str(path.relative_to(REPO_ROOT)))
                    except ValueError:
                        missing.append(str(path))
            raise TableBlocked(f"missing tau-bench files for {arm} seed {seed}", missing=missing)
        metrics = load_json(metrics_path)
        _metadata = load_json(run_metadata_path)
        if not isinstance(_metadata, dict):
            raise TableBlocked(f"tau-bench run metadata for {arm} seed {seed} is malformed")
        provider_records.append({
            "arm": arm,
            "seed": str(seed),
            "agent_provider": str((_metadata.get("agent") or {}).get("provider", "")),
            "simulator_provider": str((_metadata.get("simulator") or {}).get("provider", "")),
        })
        results = load_json(results_path)
        rewards = [float(row.get("reward", 0.0)) for row in results if isinstance(row, dict)]
        if not rewards:
            raise TableBlocked(f"tau-bench results for {arm} seed {seed} contain no rewards")
        grouped.setdefault(arm, []).append({
            "seed": seed,
            "toolcall_step_accuracy": float(metrics["toolcall_step_accuracy"]),
            "task_success": statistics.mean(rewards),
            "total_gold_actions": int(metrics["total_gold_actions"]),
            "task_count": int(run.get("task_count", len(rewards))),
        })
        input_records.append(artifact_record(metrics_path, extra={"arm": arm, "seed": seed, "repo_commit": repo_commit, "tau_bench_harness_commit": harness_commit}))
        input_records.append(artifact_record(run_metadata_path, extra={"arm": arm, "seed": seed, "repo_commit": repo_commit, "tau_bench_harness_commit": harness_commit}))

    harness_commits = {commit for commit in all_harness_commits if commit}
    harness_mix_evidence: dict[str, Any] | None = None
    if len(harness_commits) > 1:
        sorted_runs = sorted(provider_records, key=lambda item: (item["arm"], int(item["seed"])))
        nonlocal_providers = all(
            item["agent_provider"] not in {"local", "local-hf"} and item["simulator_provider"] not in {"local", "local-hf"}
            for item in sorted_runs
        )
        diff_files = git_output([
            "git", "diff", "--name-only",
            "b379c6b16fe4667768c65518bf8b3d505f0ded84",
            "6fd43ecec0a893c2e868b84b4f388cb72cd54e3b",
            "--", "external_datasets/tau_bench/tau-bench-main",
        ]).splitlines()
        diffstat = git_output([
            "git", "diff", "--stat",
            "b379c6b16fe4667768c65518bf8b3d505f0ded84",
            "6fd43ecec0a893c2e868b84b4f388cb72cd54e3b",
            "--", "external_datasets/tau_bench/tau-bench-main",
        ])
        local_gate = {
            "file": "external_datasets/tau_bench/tau-bench-main/tau_bench/local_completion.py",
            "lines": [
                "provider = kwargs.get(\"custom_llm_provider\")",
                "if provider == LOCAL_PROVIDER:",
                "    return _local_completion(*args, **kwargs)",
            ],
        }
        only_local_completion = diff_files == [local_gate["file"]]
        if nonlocal_providers and only_local_completion:
            harness_mix_evidence = {
                "accepted_unreachable_harness_diff": True,
                "tau_bench_harness_commits": sorted(harness_commits),
                "repo_commits": sorted(commit for commit in all_repo_commits if commit),
                "diff_files": diff_files,
                "diffstat": diffstat,
                "local_completion_provider_gate": local_gate,
                "providers": sorted_runs,
            }
        else:
            raise TableBlocked(
                "tau-bench transfer inputs mix vendored harness tree hashes",
                details={
                    "tau_bench_harness_commits": sorted(harness_commits),
                    "diff_files": diff_files,
                    "diffstat": diffstat,
                    "providers": sorted_runs,
                },
            )

    repo_commits = {commit for commit in all_repo_commits if commit}
    effective_transfer_paths = [path for path in TAU_TRANSFER_PATHS if path != "external_datasets/tau_bench/tau-bench-main"] if harness_mix_evidence else TAU_TRANSFER_PATHS
    path_equivalent, path_evidence = commits_equivalent_for_paths(repo_commits, effective_transfer_paths)
    commits_ok = path_equivalent or commits_compatible(repo_commits, equivalent_groups)
    if repo_commits and not commits_ok:
        raise TableBlocked(
            "tau-bench transfer inputs differ on table dependency paths",
            details={
                "repo_commits": sorted(repo_commits),
                "tau_bench_harness_commit": next(iter(harness_commits), "unknown"),
                "table_dependency_paths": effective_transfer_paths,
                "repo_commit_equivalence": path_evidence,
            },
        )

    required_arms = ["plain", "docs_only", "synapse", "centralized"]
    missing_arms = [arm for arm in required_arms if arm not in grouped or len(grouped[arm]) < 3]
    if missing_arms:
        raise TableBlocked(
            "tau-bench transfer table is missing required canonical arm seeds",
            details={
                "missing_arms": missing_arms,
                "tau_bench_harness_commit": next(iter(harness_commits), "unknown"),
                "repo_commits": sorted(repo_commits),
                "table_dependency_paths": effective_transfer_paths,
                "repo_commit_equivalence": path_evidence,
            },
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Arm & Step accuracy & Task success & $n$ steps / seed \\",
        r"\midrule",
    ]
    numbers: dict[str, str] = {}
    arm_means: dict[str, float] = {}
    tau_steps_per_seed = None

    for arm in required_arms:
        entries = sorted(grouped[arm], key=lambda item: item["seed"])
        step_mean, step_sd = stats([entry["toolcall_step_accuracy"] for entry in entries])
        task_mean, task_sd = stats([entry["task_success"] for entry in entries])
        arm_means[arm] = step_mean
        step_counts = {entry["total_gold_actions"] for entry in entries}
        if len(step_counts) != 1:
            raise TableBlocked(f"tau-bench step denominator differs across {arm} seeds", details={"counts": sorted(step_counts)})
        step_count = next(iter(step_counts))
        if tau_steps_per_seed is None:
            tau_steps_per_seed = step_count
        elif tau_steps_per_seed != step_count:
            raise TableBlocked("tau-bench step denominator differs across arms", details={"expected": tau_steps_per_seed, "found": step_count, "arm": arm})
        lines.append(f"{latex_escape(arm)} & {format_mean_sd(step_mean, step_sd)} & {format_mean_sd(task_mean, task_sd)} & {step_count} \\\\")
        numbers[f"tau_{arm}_step"] = f"{step_mean:.3f}"
        numbers[f"tau_{arm}_task"] = f"{task_mean:.3f}"
        numbers[f"tau_{arm}_step_sd"] = f"{step_sd:.3f}"
        numbers[f"tau_{arm}_task_sd"] = f"{task_sd:.3f}"
        if arm == "docs_only":
            numbers["tau_docs_only_step_current"] = f"{step_mean:.3f}"
            numbers["tau_docs_only_task_current"] = f"{task_mean:.3f}"
            numbers["tau_docs_only_step_current_sd"] = f"{step_sd:.3f}"
            numbers["tau_docs_only_task_current_sd"] = f"{task_sd:.3f}"

    compendium_gain_pts = min((arm_means[arm] - arm_means["plain"]) * 100.0 for arm in ["docs_only", "synapse", "centralized"])
    numbers["tau_plain_step"] = f"{arm_means['plain']:.3f}"
    numbers["tau_compendium_gain_pts"] = f"{compendium_gain_pts:.1f}"
    numbers["tau_steps_per_seed"] = str(tau_steps_per_seed or 0)
    numbers["tau_harness_tree_docs"] = "07ec25f055ab3c33de4df71dfed08e54770d51ba"
    numbers["tau_harness_tree_matched"] = "bb344928284eaf0a0cd44809e8d1b432aa263901"
    numbers["tau_harness_tree_current"] = tree_hash_for_commit(git_output(["git", "rev-parse", "HEAD"]), "external_datasets/tau_bench/tau-bench-main")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{$\tau$-bench retail transfer table from canonical GPT-4o sidecars. Step accuracy is the primary metric.}",
        r"\label{tab:tau-transfer}",
        r"\end{table}",
    ])
    return BuiltTable(
        name="transfer",
        tex_name="transfer.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=input_records,
        details={
            "repo_commits": sorted(repo_commits),
            "tau_bench_harness_commit": next(iter(harness_commits), "mixed-unreachable") if len(harness_commits) > 1 else next(iter(harness_commits), "unknown"),
            "tau_bench_upstream_repo": "sierra-research/tau-bench",
            "table_dependency_paths": effective_transfer_paths,
            "repo_commit_equivalence": path_evidence,
            "harness_mix_evidence": harness_mix_evidence,
            "arms": required_arms,
        },
    )


def build_lessons_table() -> BuiltTable:
    if not LESSONS_CLASSIFIER_SMOKE.exists():
        raise TableBlocked("missing clean classifier smoke", missing=[str(LESSONS_CLASSIFIER_SMOKE.relative_to(REPO_ROOT))])
    if not STB_CLEAN_CLASSIFIER_SUMMARY.exists():
        raise TableBlocked("missing clean classifier rerun summary", missing=[str(STB_CLEAN_CLASSIFIER_SUMMARY.relative_to(REPO_ROOT))])
    if not STB_TRULY_UNSEEN_ANALYSIS.exists():
        raise TableBlocked("missing StableToolBench truly-unseen analysis", missing=[str(STB_TRULY_UNSEEN_ANALYSIS.relative_to(REPO_ROOT))])
    if not R4_CONTAMINATION_PROGRESS.exists():
        raise TableBlocked("missing r4 contamination progress log", missing=[str(R4_CONTAMINATION_PROGRESS.relative_to(REPO_ROOT))])
    classifier = load_json(LESSONS_CLASSIFIER_SMOKE)
    clean_classifier = load_json(STB_CLEAN_CLASSIFIER_SUMMARY)
    truly_unseen = load_json(STB_TRULY_UNSEEN_ANALYSIS)
    r4_stage = find_progress_stage(R4_CONTAMINATION_PROGRESS, "contamination_filter_done")
    if r4_stage is None:
        raise TableBlocked("missing contamination_filter_done in r4 progress log", missing=[str(R4_CONTAMINATION_PROGRESS.relative_to(REPO_ROOT))])

    r4_counts = contamination_counts(r4_stage)
    r11_assertion: dict[str, Any] = {"status": "pending"}
    assertion_source = None
    for candidate in R11_CONTAMINATION_PROGRESSS:
        stage = find_progress_stage(candidate, "contamination_filter_done")
        if stage is None:
            continue
        assertion_source = candidate
        r11_counts = contamination_counts(stage)
        if r11_counts != r4_counts:
            raise TableBlocked(
                "r11 contamination counts do not reproduce r4 counts",
                details={
                    "r4_counts": r4_counts,
                    "r11_counts": r11_counts,
                    "r11_source": str(candidate.relative_to(REPO_ROOT)),
                },
            )
        r11_assertion = {
            "status": "matched",
            "r11_source": str(candidate.relative_to(REPO_ROOT)),
            "counts": r11_counts,
        }
        break

    smoke_qc = classifier["arms"]["query_classifier"]["metrics"]["accuracy"]["mean"]
    clean_qc = clean_classifier["query_classifier"]["mean_accuracy"]
    uncovered_mean = truly_unseen["aggregate"]["n_uncovered"]["mean"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lp{7.5cm}}",
        r"\toprule",
        r"Signal & Value \\",
        r"\midrule",
        f"Eval queries with exact pool match & {r4_counts['eval_queries_with_exact_match_in_pool']} \\",
        f"Eval queries with $\\geq 0.95$ pool near-duplicate & {r4_counts['eval_queries_with_near_dup_in_pool']} \\",
        f"Pool items removed exact / near-dup / total & {r4_counts['pool_items_removed_exact']} / {r4_counts['pool_items_removed_near_dup']} / {r4_counts['pool_items_removed_total']} \\",
        f"Eval queries removed & {r4_counts['eval_queries_removed']} \\",
        f"Classifier smoke (legacy check only) & {smoke_qc:.3f} \\",
        f"Clean classifier rerun (benchmark row) & {clean_qc:.3f} \\",
        f"Mean uncovered queries per seed & {uncovered_mean:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Count-only contamination lessons from the clean StableToolBench filter path.}",
        r"\label{tab:lessons-counts}",
        r"\end{table}",
    ]
    inputs = [
        artifact_record(LESSONS_CLASSIFIER_SMOKE),
        artifact_record(STB_CLEAN_CLASSIFIER_SUMMARY),
        artifact_record(STB_TRULY_UNSEEN_ANALYSIS),
        artifact_record(R4_CONTAMINATION_PROGRESS),
    ]
    if assertion_source is not None:
        inputs.append(artifact_record(assertion_source))
    return BuiltTable(
        name="lessons",
        tex_name="lessons.tex",
        tex_body="\n".join(lines),
        numbers={
            "leak_removed": str(r4_counts["pool_items_removed_total"]),
            "eval_queries_with_exact_match_in_pool": str(r4_counts["eval_queries_with_exact_match_in_pool"]),
            "eval_queries_with_near_dup_in_pool": str(r4_counts["eval_queries_with_near_dup_in_pool"]),
            "eval_queries_removed": str(r4_counts["eval_queries_removed"]),
        },
        inputs=inputs,
        details={
            "contamination_source": str(R4_CONTAMINATION_PROGRESS.relative_to(REPO_ROOT)),
            "r4_counts": r4_counts,
            "r11_reproduces_r4_counts": r11_assertion,
            "clean_classifier_summary": artifact_record(STB_CLEAN_CLASSIFIER_SUMMARY),
            "truly_unseen_analysis": artifact_record(STB_TRULY_UNSEEN_ANALYSIS),
        },
    )


def build_benchmarks_table() -> BuiltTable:
    missing = [str(path.relative_to(REPO_ROOT)) for path in STB_BENCHMARK_DISTINCT5_CELLS.values() if not path.exists()]
    if missing:
        raise TableBlocked("missing corrected distinct-five benchmark replay cells", missing=missing)
    if not STB_CLEAN_CLASSIFIER_SUMMARY.exists():
        raise TableBlocked("missing clean classifier rerun summary", missing=[str(STB_CLEAN_CLASSIFIER_SUMMARY.relative_to(REPO_ROOT))])

    clean_classifier = load_json(STB_CLEAN_CLASSIFIER_SUMMARY)
    groups = ["G1_instruction", "G1_tool", "G1_category", "G2_instruction", "G2_category", "G3_instruction"]
    group_labels = {
        "G1_instruction": "G1-Instruction",
        "G1_tool": "G1-Tool",
        "G1_category": "G1-Category",
        "G2_instruction": "G2-Instruction",
        "G2_category": "G2-Category",
        "G3_instruction": "G3-Instruction",
    }
    arm_labels = {
        "synapse": "Synapse",
        "centralized": "Centralized",
        "local_only": "Local only",
        "flat_pool": "Flat pool",
        "query_classifier": "Query classifier",
    }
    arms = ["synapse", "centralized", "local_only", "flat_pool", "query_classifier"]
    seeds = [42, 123, 456]

    def group_metric_from_rows(payload: dict[str, Any], group: str, metric: str) -> float:
        rows = [row for row in payload.get("rows", []) if row.get("group") == group]
        if not rows:
            raise TableBlocked("benchmark replay cell has no rows for group", details={"group": group, "arm": payload.get("arm")})
        if metric == "accuracy":
            return sum(1 for row in rows if row.get("routed_correctly")) / len(rows)
        if metric == "recall_at_5":
            return sum(1 for row in rows if row.get("gold_in_top_k")) / len(rows)
        raise KeyError(metric)

    def validate_cell(payload: dict[str, Any], path: Path) -> int:
        if payload.get("paper_eligible") is False:
            raise TableBlocked("benchmark replay cell is marked non-paper", details={"path": str(path), "paper_eligible": payload.get("paper_eligible")})
        if payload.get("retrieval_mode") != "distinct_tool_topk" or int(payload.get("retrieval_pool_size", -1)) != 200:
            raise TableBlocked("benchmark replay cell used wrong candidate rule", details={"path": str(path), "retrieval_mode": payload.get("retrieval_mode"), "retrieval_pool_size": payload.get("retrieval_pool_size")})
        shortfalls = _row_shortfall_count(payload, 5)
        if shortfalls:
            raise TableBlocked("benchmark replay cell has candidate shortfalls", details={"path": str(path), "shortfalls": shortfalls})
        rows = payload.get("rows", [])
        if len(rows) not in (729, 3645):
            raise TableBlocked("benchmark replay cell has unexpected row count", details={"path": str(path), "row_count": len(rows)})
        return len(rows)

    cell_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for arm in ["synapse", "centralized", "local_only", "flat_pool"]:
        for seed in seeds:
            path = STB_BENCHMARK_DISTINCT5_CELLS[(arm, seed)]
            payload = load_json(path)
            row_count = validate_cell(payload, path)
            cell_payloads[(arm, seed)] = payload
            inputs.append(artifact_record(path, extra={
                "role": "benchmark_distinct5_replay_cell",
                "arm": arm,
                "seed": seed,
                "repo_commit": payload.get("repo_commit") or payload.get("git_commit"),
                "candidate_rule": "distinct5_walkdown",
                "retrieval_pool_size": payload.get("retrieval_pool_size"),
                "candidate_shortfall_count": 0,
                "row_count": row_count,
                "package_sha256": payload.get("package_sha256") or (payload.get("compendium") or {}).get("global_sha256"),
            }))
    inputs.append(artifact_record(STB_CLEAN_CLASSIFIER_SUMMARY, extra={"repo_commit": (clean_classifier.get("config") or {}).get("repo_commit")}))
    concat_control = None
    if STB_CONCAT_CONTROL_SUMMARY.exists():
        concat_control = load_json(STB_CONCAT_CONTROL_SUMMARY)
        inputs.append(artifact_record(STB_CONCAT_CONTROL_SUMMARY, extra={"repo_commit": (concat_control.get("config") or {}).get("git_commit"), "candidate_rule": "distinct5_walkdown"}))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{G1-Instruction} & \multicolumn{2}{c}{G1-Tool} & \multicolumn{2}{c}{G1-Category} & \multicolumn{2}{c}{G2-Instruction} & \multicolumn{2}{c}{G2-Category} & \multicolumn{2}{c}{G3-Instruction} \\",
        r"Method & Acc & R@5 & Acc & R@5 & Acc & R@5 & Acc & R@5 & Acc & R@5 & Acc & R@5 \\",
        r"\midrule",
    ]

    numbers: dict[str, str] = {}
    for arm in arms:
        row = [arm_labels[arm]]
        for group in groups:
            if arm == "query_classifier":
                group_metrics = clean_classifier["query_classifier"]["metrics"]["per_group"][group]
                acc_mean = float(group_metrics["accuracy"]["mean"])
                acc_sd = float(group_metrics["accuracy"]["sd"])
                rec_mean = float(group_metrics["recall_at_5"]["mean"])
                rec_sd = float(group_metrics["recall_at_5"]["sd"])
            else:
                accs = [group_metric_from_rows(cell_payloads[(arm, seed)], group, "accuracy") for seed in seeds]
                recs = [group_metric_from_rows(cell_payloads[(arm, seed)], group, "recall_at_5") for seed in seeds]
                acc_mean = statistics.mean(accs)
                acc_sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
                rec_mean = statistics.mean(recs)
                rec_sd = statistics.stdev(recs) if len(recs) > 1 else 0.0
            row.append(f"{acc_mean:.3f} $\\pm$ {acc_sd:.3f}")
            row.append(f"{rec_mean:.3f} $\\pm$ {rec_sd:.3f}")
            numbers[f"{arm}_{group}_acc"] = f"{acc_mean:.3f}"
            numbers[f"{arm}_{group}_r5"] = f"{rec_mean:.3f}"
        lines.append(" & ".join(row) + r" \\")

    qc_clean_acc = float(clean_classifier["query_classifier"]["mean_accuracy"])
    router_arm_means = {
        arm: statistics.mean([float(cell_payloads[(arm, seed)]["accuracy"]) for seed in seeds])
        for arm in ["synapse", "centralized", "local_only", "flat_pool"]
    }
    gap_pts = (router_arm_means["synapse"] - router_arm_means["flat_pool"]) * 100.0
    best_router_acc = max(router_arm_means.values())
    concat_acc = None
    if concat_control is not None:
        concat_acc = float(concat_control["aggregate"]["full"]["0"]["mean_accuracy"])
    best_llm_acc = max([best_router_acc] + ([concat_acc] if concat_acc is not None else []))
    numbers["qc_clean_acc"] = f"{qc_clean_acc:.3f}"
    numbers["gap_pts"] = f"{gap_pts:.1f}"
    numbers["qc_over_router_pts"] = f"{(qc_clean_acc - best_router_acc) * 100.0:.0f}"
    numbers["qc_over_best_llm_pts"] = f"{(qc_clean_acc - best_llm_acc) * 100.0:.0f}"
    group_diffs = {
        group: {
            "pooled": round(
                (float(numbers[f"synapse_{group}_acc"]) - float(numbers[f"local_only_{group}_acc"])) * 100.0
            ),
            "experience": round(
                (float(numbers[f"synapse_{group}_acc"]) - float(numbers[f"flat_pool_{group}_acc"])) * 100.0
            ),
        }
        for group in groups
    }
    pooled = [group_diffs[group]["pooled"] for group in groups]
    exp_g1 = [group_diffs[group]["experience"] for group in groups[:3]]
    exp_g23 = [group_diffs[group]["experience"] for group in groups[3:]]
    numbers["pooled_gain_min_pts"] = str(min(pooled))
    numbers["pooled_gain_max_pts"] = str(max(pooled))
    numbers["exp_gain_g1_pts"] = f"{min(exp_g1)}--{max(exp_g1)}"
    numbers["exp_gain_g23_pts"] = f"{min(exp_g23)}--{max(exp_g23)}"
    if concat_acc is not None:
        numbers["concat_full_acc"] = f"{concat_acc:.3f}"

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{StableToolBench benchmark table on the clean ToolBench-train pool with junk filtering and distinct-tool retrieval. Values are mean $\pm$ SD over seeds 42, 123, and 456.}",
        r"\label{tab:stabletoolbench-benchmarks}",
        r"\end{table*}",
    ])

    return BuiltTable(
        name="benchmarks",
        tex_name="benchmarks.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=inputs,
        details={
            "source": "corrected_distinct5_replay",
            "cells": {f"{arm}:{seed}": str(path.relative_to(REPO_ROOT)) for (arm, seed), path in STB_BENCHMARK_DISTINCT5_CELLS.items()},
            "clean_classifier_summary_path": str(STB_CLEAN_CLASSIFIER_SUMMARY.relative_to(REPO_ROOT)),
            "concat_control_summary_path": str(STB_CONCAT_CONTROL_SUMMARY.relative_to(REPO_ROOT)) if STB_CONCAT_CONTROL_SUMMARY.exists() else None,
            "groups": groups,
            "group_labels": group_labels,
            "arms": arms,
            "candidate_rule": "distinct5_walkdown",
        },
    )

def parse_defnums(text: str) -> dict[str, str]:
    numbers: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\\defnum\{([^}]+)\}\{([^}]*)\}", line.strip())
        if match:
            numbers[match.group(1)] = match.group(2)
    return numbers


def build_conflict_table() -> BuiltTable:
    required = [CONFLICT_SOURCE_TEX, CONFLICT_SOURCE_NUMBERS, CONFLICT_SOURCE_MANIFEST]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise TableBlocked("missing generated 2x2 conflict outputs", missing=missing)

    manifest = load_json(CONFLICT_SOURCE_MANIFEST)
    if not isinstance(manifest, dict):
        raise TableBlocked("conflict manifest is malformed")
    inputs: list[dict[str, Any]] = [
        artifact_record(CONFLICT_SOURCE_TEX),
        artifact_record(CONFLICT_SOURCE_NUMBERS),
        artifact_record(CONFLICT_SOURCE_MANIFEST),
    ]
    for entry in manifest.get("inputs", []):
        if not isinstance(entry, dict):
            continue
        summary_path = Path(str(entry.get("summary_path", "")))
        if summary_path.exists():
            inputs.append(artifact_record(summary_path, extra={
                "repo_commit": entry.get("repo_commit"),
                "rates": entry.get("rates"),
                "arms": entry.get("arms"),
            }))

    tex_body = CONFLICT_SOURCE_TEX.read_text(encoding="utf-8")
    numbers = parse_defnums(CONFLICT_SOURCE_NUMBERS.read_text(encoding="utf-8"))
    return BuiltTable(
        name="conflict_2x2",
        tex_name="conflict_2x2.tex",
        tex_body=tex_body,
        numbers=numbers,
        inputs=inputs,
        details={
            "seeds_used_for_headline": manifest.get("seeds_used_for_headline", []),
            "seeds_common": manifest.get("seeds_common", {}),
            "warnings": manifest.get("warnings", []),
            "refusals": manifest.get("refusals", []),
            "targeted_subset_conflict40": manifest.get("targeted_subset_conflict40", {}),
            "contradiction_hash_verified": not bool(manifest.get("refusals")),
        },
    )


def arm_metric(summary: dict[str, Any], arm: str, metric: str) -> float:
    return float(summary["arms"][arm]["metrics"][metric]["mean"])


def build_c1_robustness_table() -> BuiltTable:
    required = [STB_C1_COS090_SUMMARY, STB_C1_COS090_DISTINCT5_SUMMARY, STB_C1_COS090_QC_REFIT_SUMMARY, STB_C1_COS090_QC_REFIT_SEED, STB_CLEAN_CLASSIFIER_SUMMARY]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise TableBlocked("missing C1 robustness inputs", missing=missing)

    c1_docs = load_json(STB_C1_COS090_SUMMARY)
    c1 = load_json(STB_C1_COS090_DISTINCT5_SUMMARY)
    refit = load_json(STB_C1_COS090_QC_REFIT_SUMMARY)
    refit_seed = load_json(STB_C1_COS090_QC_REFIT_SEED)
    clean095 = load_json(STB_CLEAN_CLASSIFIER_SUMMARY)

    fit_n = int(refit_seed.get("classifier_train_n", 0))
    refusal = refit_seed.get("classifier_fit_refusal_check", {})
    if fit_n != 25000:
        raise TableBlocked("C1 classifier refit used the wrong fit-set size", details={"classifier_fit_n": fit_n})
    if int(refusal.get("exact_overlap_count", -1)) != 0 or int(refusal.get("near_duplicate_count", -1)) != 0:
        raise TableBlocked("C1 classifier refit failed overlap refusal check", details={"classifier_fit_refusal_check": refusal})

    qc = refit["query_classifier"]
    qc_acc = float(qc["mean_accuracy"])
    qc095 = float(clean095["query_classifier"]["mean_accuracy"])
    docs_acc = arm_metric(c1_docs, "flat_pool", "accuracy")
    synapse_acc = arm_metric(c1, "synapse", "accuracy")
    centralized_acc = arm_metric(c1, "centralized", "accuracy")
    group_accs = [float(metrics["mean_accuracy"]) for metrics in qc["group_metrics"].values()]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Arm & Accuracy & R@5 \\",
        r"\midrule",
        f"Docs-only & {docs_acc:.3f} & {arm_metric(c1_docs, 'flat_pool', 'recall_at_5'):.3f}" + " \\\\",
        f"Synapse & {synapse_acc:.3f} & {arm_metric(c1, 'synapse', 'recall_at_5'):.3f}" + " \\\\",
        f"Centralized & {centralized_acc:.3f} & {arm_metric(c1, 'centralized', 'recall_at_5'):.3f}" + " \\\\",
        f"Query classifier (25k fit) & {qc_acc:.3f} & 0.000" + " \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{C1 near-duplicate robustness at cosine threshold 0.90. The classifier is refit on the same capped 25k client draw as the compendium arms.}",
        r"\label{tab:c1-robustness}",
        r"\end{table}",
    ]

    numbers = {
        "c1_qc": f"{qc_acc:.3f}",
        "c1_qc_group_min": f"{min(group_accs):.3f}",
        "c1_qc_group_max": f"{max(group_accs):.3f}",
        "c1_margin_pts": f"{(qc_acc - docs_acc) * 100.0:.0f}",
        "c1_qc_drop_pts": f"{(qc095 - qc_acc) * 100.0:.0f}",
        "c1_docs_only_acc": f"{docs_acc:.3f}",
        "c1_docs_only_r5": f"{arm_metric(c1_docs, 'flat_pool', 'recall_at_5'):.3f}",
        "c1_synapse_acc": f"{synapse_acc:.3f}",
        "c1_synapse_r5": f"{arm_metric(c1, 'synapse', 'recall_at_5'):.3f}",
        "c1_centralized_acc": f"{centralized_acc:.3f}",
        "c1_centralized_r5": f"{arm_metric(c1, 'centralized', 'recall_at_5'):.3f}",
        "c1_classifier_fit_n": str(fit_n),
    }

    return BuiltTable(
        name="c1_robustness",
        tex_name="c1_robustness.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=[
            artifact_record(STB_C1_COS090_SUMMARY, extra={"role": "c1_docs_only_reference", "repo_commit": c1_docs.get("config", {}).get("git_commit")}),
            artifact_record(STB_C1_COS090_DISTINCT5_SUMMARY, extra={"role": "c1_distinct5_experience_arms", "repo_commit": c1.get("config", {}).get("git_commit"), "candidate_rule": "distinct5_walkdown"}),
            artifact_record(STB_C1_COS090_QC_REFIT_SUMMARY, extra={"role": "c1_classifier_refit", "repo_commit": refit.get("config", {}).get("git_commit")}),
            artifact_record(STB_C1_COS090_QC_REFIT_SEED, extra={"role": "c1_classifier_refit_seed", "repo_commit": refit_seed.get("repo_commit")}),
            artifact_record(STB_CLEAN_CLASSIFIER_SUMMARY, extra={"role": "cos095_classifier_reference", "repo_commit": clean095.get("config", {}).get("git_commit")}),
        ],
        details={
            "threshold": 0.90,
            "classifier_fit_n": fit_n,
            "classifier_fit_pool_sha256": refit_seed.get("classifier_fit_pool_sha256"),
            "filtered_pool_sha256": refit_seed.get("filtered_pool_sha256"),
            "classifier_fit_refusal_check": refusal,
            "cos095_classifier_accuracy": qc095,
            "cos090_classifier_accuracy": qc_acc,
            "docs_only_accuracy": docs_acc,
            "experience_arms_source": str(STB_C1_COS090_DISTINCT5_SUMMARY.relative_to(REPO_ROOT)),
            "docs_only_source": str(STB_C1_COS090_SUMMARY.relative_to(REPO_ROOT)),
        },
    )


def aggregate_subset_metric(e5_summary: dict[str, Any], arm: str, subset: str, metric: str) -> float:
    return float(e5_summary["aggregate"][arm][subset][f"mean_{metric}"])


def aggregate_subset_sd(e5_summary: dict[str, Any], arm: str, subset: str, metric: str) -> float:
    return float(e5_summary["aggregate"][arm][subset][f"sd_{metric}"])


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _row_shortfall_count(payload: dict[str, Any], top_k: int) -> int:
    count = 0
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        if row.get("candidate_shortfall"):
            count += 1
            continue
        distinct = row.get("candidate_distinct_count")
        if distinct is not None and int(distinct) < top_k:
            count += 1
    return count


def _d1b_metrics(arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seeds = sorted(seed for cell_arm, seed in STB_HELDOUT_D1B_CELLS if cell_arm == arm)
    cells: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        path = STB_HELDOUT_D1B_CELLS[(arm, seed)]
        payload = load_json(path)
        if payload.get("paper_eligible") is not True:
            raise TableBlocked("D1b cell is not paper eligible", details={"path": str(path), "paper_eligible": payload.get("paper_eligible")})
        if payload.get("data_mode") != "toolbench_train":
            raise TableBlocked("D1b cell used wrong data_mode", details={"path": str(path), "data_mode": payload.get("data_mode")})
        if payload.get("retrieval_mode") != "distinct_tool_topk" or int(payload.get("retrieval_pool_size", -1)) != 200:
            raise TableBlocked("D1b cell used wrong candidate rule", details={"path": str(path), "retrieval_mode": payload.get("retrieval_mode"), "retrieval_pool_size": payload.get("retrieval_pool_size")})
        shortfalls = _row_shortfall_count(payload, 5)
        if shortfalls:
            raise TableBlocked("D1b cell has candidate shortfalls", details={"path": str(path), "shortfalls": shortfalls})
        cells.append(payload)
        records.append(artifact_record(path, extra={
            "role": "d1b_distinct5_cell",
            "arm": arm,
            "seed": seed,
            "repo_commit": payload.get("repo_commit"),
            "candidate_rule": "distinct5_walkdown",
            "retrieval_pool_size": payload.get("retrieval_pool_size"),
            "candidate_shortfall_count": shortfalls,
        }))
    metrics: dict[str, Any] = {"seeds": seeds}
    for subset in ["heldout", "labeled"]:
        for metric in ["accuracy", "recall_at_5"]:
            vals = [float(cell["subset_metrics"][subset][metric]) for cell in cells]
            metrics[f"{subset}_{metric}"] = statistics.mean(vals)
            metrics[f"{subset}_{metric}_sd"] = _sd(vals)
        metrics[f"{subset}_n"] = [int(cell["subset_metrics"][subset]["n"]) for cell in cells]
    return metrics, records


def build_e5_heldout_table() -> BuiltTable:
    required = [STB_HELDOUT_E5_SUMMARY, D1_D2_CANDIDATE_UNDERFILL_SUMMARY]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise TableBlocked("missing E5 held-out or D1/D2 diagnostic inputs", missing=missing)

    e5 = load_json(STB_HELDOUT_E5_SUMMARY)
    d1d2 = load_json(D1_D2_CANDIDATE_UNDERFILL_SUMMARY)
    config = e5.get("config") if isinstance(e5.get("config"), dict) else {}
    heldout_filter = e5.get("heldout_filter") if isinstance(e5.get("heldout_filter"), dict) else {}
    heldout_sanity = e5.get("heldout_sanity") if isinstance(e5.get("heldout_sanity"), dict) else {}

    if e5.get("paper_eligible") is not True:
        raise TableBlocked("E5 summary is not paper eligible", details={"paper_eligible": e5.get("paper_eligible")})
    if int(heldout_filter.get("remaining_items_with_heldout_label", -1)) != 0:
        raise TableBlocked("E5 held-out filter left held-out labels in the pool", details=heldout_filter)
    if int(heldout_filter.get("eval_queries_removed", -1)) != 0:
        raise TableBlocked("E5 held-out filter removed evaluation queries", details=heldout_filter)
    if int(heldout_sanity.get("G1_tool", {}).get("query_count", -1)) != 152 or int(heldout_sanity.get("G1_category", {}).get("query_count", -1)) != 134:
        raise TableBlocked("E5 sanity counts do not match the 729-query junk-filtered evaluation set", details=heldout_sanity)

    d1_entries = [entry for entry in (d1d2.get("groups", {}).get("e5_stable_heldout", [])) if isinstance(entry, dict)]
    synapse_underfill = [entry for entry in d1_entries if entry.get("meta", {}).get("arm") == "synapse"]
    flat_underfill = [entry for entry in d1_entries if entry.get("meta", {}).get("arm") == "flat_pool"]
    if len(synapse_underfill) < 3 or len(flat_underfill) < 3:
        raise TableBlocked("D1/D2 diagnostics do not cover all E5 synapse/flat_pool seeds", details={"synapse": len(synapse_underfill), "flat_pool": len(flat_underfill)})

    for entry in synapse_underfill + flat_underfill:
        meta = entry.get("meta", {})
        if meta.get("data_mode") != "toolbench_train":
            raise TableBlocked("E5 D1/D2 diagnostic used wrong data_mode", details={"path": entry.get("path"), "data_mode": meta.get("data_mode")})
        if int(meta.get("retrieval_pool_size", -1)) != 5:
            raise TableBlocked("E5 D1/D2 diagnostic used unexpected retrieval depth", details={"path": entry.get("path"), "retrieval_pool_size": meta.get("retrieval_pool_size")})

    arm_rows = [
        ("synapse", "Synapse"),
        ("centralized", "Centralized"),
        ("flat_pool", "Docs-only"),
        ("local_only", "Local only"),
        ("query_classifier", "Query classifier"),
        ("synapse_oracle", "Synapse oracle"),
        ("flat_pool_oracle", "Docs-only oracle"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Arm & Held-out Acc & Held-out R@5 & Labeled Acc & Labeled R@5 \\",
        r"\midrule",
    ]
    numbers: dict[str, str] = {}
    d1b_metrics: dict[str, dict[str, Any]] = {}
    d1b_input_records: list[dict[str, Any]] = []
    for d1b_arm in ["synapse", "centralized", "flat_pool", "local_only"]:
        d1b_metrics[d1b_arm], records = _d1b_metrics(d1b_arm)
        d1b_input_records.extend(records)

    for arm, label in arm_rows:
        if arm in d1b_metrics:
            route = d1b_metrics[arm]
            held_acc = float(route["heldout_accuracy"])
            held_r5 = float(route["heldout_recall_at_5"])
            lab_acc = float(route["labeled_accuracy"])
            lab_r5 = float(route["labeled_recall_at_5"])
            held_acc_sd = float(route["heldout_accuracy_sd"])
            held_r5_sd = float(route["heldout_recall_at_5_sd"])
            lab_acc_sd = float(route["labeled_accuracy_sd"])
            lab_r5_sd = float(route["labeled_recall_at_5_sd"])
        else:
            held_acc = aggregate_subset_metric(e5, arm, "heldout", "accuracy")
            held_r5 = aggregate_subset_metric(e5, arm, "heldout", "recall_at_5")
            lab_acc = aggregate_subset_metric(e5, arm, "labeled", "accuracy")
            lab_r5 = aggregate_subset_metric(e5, arm, "labeled", "recall_at_5")
            held_acc_sd = aggregate_subset_sd(e5, arm, "heldout", "accuracy")
            held_r5_sd = aggregate_subset_sd(e5, arm, "heldout", "recall_at_5")
            lab_acc_sd = aggregate_subset_sd(e5, arm, "labeled", "accuracy")
            lab_r5_sd = aggregate_subset_sd(e5, arm, "labeled", "recall_at_5")
        lines.append(f"{latex_escape(label)} & {format_mean_sd(held_acc, held_acc_sd)} & {format_mean_sd(held_r5, held_r5_sd)} & {format_mean_sd(lab_acc, lab_acc_sd)} & {format_mean_sd(lab_r5, lab_r5_sd)}" + r" \\")
        key = "docs_only" if arm == "flat_pool" else arm
        numbers[f"e5_{key}_heldout_acc"] = f"{held_acc:.3f}"
        numbers[f"e5_{key}_heldout_r5"] = f"{held_r5:.3f}"
        numbers[f"e5_{key}_labeled_acc"] = f"{lab_acc:.3f}"
        numbers[f"e5_{key}_labeled_r5"] = f"{lab_r5:.3f}"
        paper_key = {"synapse": "sy", "centralized": "ce", "flat_pool": "do", "local_only": "lo"}.get(arm)
        if paper_key:
            numbers[f"e5_{paper_key}_heldout"] = f"{held_acc:.3f}"
            numbers[f"e5_{paper_key}_heldout_r5"] = f"{held_r5:.3f}"
            numbers[f"e5_{paper_key}_labeled"] = f"{lab_acc:.3f}"
            numbers[f"e5_{paper_key}_labeled_r5"] = f"{lab_r5:.3f}"

    def mean_underfill(entries: list[dict[str, Any]], field: str) -> float:
        values = [float(entry["candidate_underfill"]["candidate_distinct_tools"][field]) for entry in entries]
        return statistics.mean(values)

    syn_distinct_mean = mean_underfill(synapse_underfill, "mean")
    syn_share_lt5 = mean_underfill(synapse_underfill, "share_lt5")
    flat_distinct_mean = mean_underfill(flat_underfill, "mean")
    flat_share_lt5 = mean_underfill(flat_underfill, "share_lt5")
    numbers["e5_synapse_distinct_mean"] = f"{syn_distinct_mean:.2f}"
    numbers["e5_synapse_distinct_lt5"] = f"{syn_share_lt5:.3f}"
    numbers["e5_docs_distinct_mean"] = f"{flat_distinct_mean:.2f}"
    numbers["e5_docs_distinct_lt5"] = f"{flat_share_lt5:.3f}"
    numbers["e5_depth"] = "200"

    d2_docs = d1d2.get("d2_recall_reconciliation", {}).get("docs_like_artifacts", [])
    e5_docs_diag = next((entry for entry in d2_docs if str(entry.get("path", "")).endswith("stabletoolbench_heldout_r3b/seed_42/flat_pool.json")), None)
    c1_docs_diag = next((entry for entry in d2_docs if str(entry.get("path", "")).endswith("stabletoolbench_c1_cos090_seed42_r1/seed_42/flat_pool.json")), None)
    if e5_docs_diag:
        numbers["e5_docs_recorded_r5"] = f"{float(e5_docs_diag['all_recorded_recall_at_5']):.3f}"
        numbers["e5_docs_candidate_top5_r5"] = f"{float(e5_docs_diag['all_candidate_top5_recall']):.3f}"
    if c1_docs_diag:
        numbers["c1_docs_candidate_top5_r5"] = f"{float(c1_docs_diag['all_candidate_top5_recall']):.3f}"
        numbers["c1_docs_recorded_r5"] = f"{float(c1_docs_diag['all_recorded_recall_at_5']):.3f}"

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{E5 held-out StableToolBench split. Values are mean $\pm$ SD over seeds 42, 123, and 456 except Local-only, where seed 123 refused on the distinct-five coverage invariant and the table uses seeds 42 and 456. Docs-only has zero SD because it has no seed-dependent client experience.}",
        r"\label{tab:e5-heldout}",
        r"\end{table}",
    ])

    input_records = [
        artifact_record(STB_HELDOUT_E5_SUMMARY, extra={"role": "e5_superseded_depth5_summary", "repo_commit": config.get("repo_commit")}),
        artifact_record(D1_D2_CANDIDATE_UNDERFILL_SUMMARY, extra={"role": "d1_d2_candidate_underfill"}),
    ]
    input_records.extend(d1b_input_records)
    if STB_HELDOUT_D1B_LOCAL_ONLY_REFUSAL_LOG.exists():
        input_records.append(artifact_record(STB_HELDOUT_D1B_LOCAL_ONLY_REFUSAL_LOG, extra={
            "role": "d1b_local_only_seed123_refusal",
            "candidate_rule": "distinct5_walkdown",
            "reason": "coverage_shortfall_under_cap_200",
        }))
    for entry in d1_entries:
        path_text = entry.get("path")
        if path_text:
            path = REPO_ROOT / str(path_text)
            if path.exists():
                input_records.append(artifact_record(path, extra={
                    "role": "e5_row_file",
                    "arm": entry.get("meta", {}).get("arm"),
                    "repo_commit": entry.get("meta", {}).get("repo_commit"),
                    "retrieval_pool_size": entry.get("meta", {}).get("retrieval_pool_size"),
                    "candidate_distinct_mean": entry.get("candidate_underfill", {}).get("candidate_distinct_tools", {}).get("mean"),
                    "candidate_distinct_share_lt5": entry.get("candidate_underfill", {}).get("candidate_distinct_tools", {}).get("share_lt5"),
                }))

    return BuiltTable(
        name="e5_heldout",
        tex_name="e5_heldout.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=input_records,
        details={
            "config": config,
            "heldout_sanity": heldout_sanity,
            "heldout_filter": heldout_filter,
            "retrieval_depth_used": 200,
            "candidate_rule": "distinct5_walkdown",
            "d1b_route_arm_metrics": d1b_metrics,
            "local_only_seed123": {
                "status": "refused",
                "reason": "coverage_shortfall_under_cap_200",
                "log": str(STB_HELDOUT_D1B_LOCAL_ONLY_REFUSAL_LOG.relative_to(REPO_ROOT)),
            },
            "superseded_depth5_diagnostics": {
                "retrieval_depth_used": 5,
                "shared_index_distinct_candidate_tools": {
                "synapse_mean": syn_distinct_mean,
                "synapse_share_lt5": syn_share_lt5,
                "docs_only_mean": flat_distinct_mean,
                "docs_only_share_lt5": flat_share_lt5,
                },
            },
            "docs_only_zero_sd_note": "Docs-only has no seed-dependent client experience; D1b reproduces exactly across seeds.",
            "docs_only_recall_reconciliation": {
                "explanation": "E5 reports candidate top-5 hit at retrieval_pool_size=5. C1 reports retrieval-pool hit over a 20-item pool, while its candidate top-5 list matches E5 and has the same candidate-top5 recall.",
                "e5_seed42_flat_pool": e5_docs_diag,
                "c1_seed42_flat_pool": c1_docs_diag,
            },
        },
    )


def _summary_config_commit(payload: dict[str, Any]) -> Any:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    return config.get("repo_commit") or config.get("git_commit")


def _aggregate_cells(cell_paths: list[Path]) -> dict[str, tuple[float, float]]:
    payloads = [load_json(path) for path in cell_paths]
    out: dict[str, tuple[float, float]] = {}
    for metric in ["accuracy", "recall_at_5"]:
        values = [float(payload[metric]) for payload in payloads]
        out[metric] = stats(values)
    for subset in ["heldout", "heldout_dev", "heldout_test", "labeled"]:
        for metric in ["accuracy", "recall_at_5"]:
            values = []
            for payload in payloads:
                subset_metrics = payload.get("subset_metrics", {})
                if subset in subset_metrics and metric in subset_metrics[subset]:
                    values.append(float(subset_metrics[subset][metric]))
            if values:
                out[f"{subset}_{metric}"] = stats(values)
    return out


def build_concat_control_table() -> BuiltTable:
    if not STB_CONCAT_CONTROL_SUMMARY.exists():
        raise TableBlocked("missing concat control distinct-five summary", missing=[str(STB_CONCAT_CONTROL_SUMMARY.relative_to(REPO_ROOT))])
    payload = load_json(STB_CONCAT_CONTROL_SUMMARY)
    if payload.get("paper_eligible") is not True:
        raise TableBlocked("concat control summary is not paper eligible", details={"paper_eligible": payload.get("paper_eligible")})
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    if config.get("retrieval_mode") != "distinct_tool_topk" or int(config.get("retrieval_pool_size", -1)) != 200:
        raise TableBlocked("concat control used wrong candidate rule", details={"retrieval_mode": config.get("retrieval_mode"), "retrieval_pool_size": config.get("retrieval_pool_size")})
    aggregate = payload.get("aggregate", {})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Subset/rate & Accuracy & R@5 & Seeds \\",
        r"\midrule",
    ]
    numbers: dict[str, str] = {}
    for subset_key, label in [("full", "Full"), ("subset_g1g2_instruction", "G1/G2 instruction")]:
        if subset_key not in aggregate:
            continue
        for rate in sorted(aggregate[subset_key], key=lambda value: int(value)):
            row = aggregate[subset_key][rate]
            acc = float(row["mean_accuracy"])
            acc_sd = float(row.get("sd_accuracy", 0.0))
            r5 = float(row["mean_recall_at_5"])
            r5_sd = float(row.get("sd_recall_at_5", 0.0))
            seed_count = len(row.get("per_seed", {}))
            lines.append(f"{label} @{rate}\\% & {format_mean_sd(acc, acc_sd)} & {format_mean_sd(r5, r5_sd)} & {seed_count}" + r" \\")
            prefix = "concat_full" if subset_key == "full" else "concat_subset"
            suffix = "" if subset_key == "full" and str(rate) == "0" else f"_{rate}"
            numbers[f"{prefix}{suffix}_acc"] = f"{acc:.3f}"
            numbers[f"{prefix}{suffix}_r5"] = f"{r5:.3f}"
            numbers[f"{prefix}{suffix}_acc_sd"] = f"{acc_sd:.3f}"
            numbers[f"{prefix}{suffix}_r5_sd"] = f"{r5_sd:.3f}"
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Concat control replay under the distinct-five candidate rule.}",
        r"\label{tab:concat-control-generated}",
        r"\end{table}",
    ])
    return BuiltTable(
        name="concat_control",
        tex_name="concat_control.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=[artifact_record(STB_CONCAT_CONTROL_SUMMARY, extra={"repo_commit": _summary_config_commit(payload), "candidate_rule": "distinct5_walkdown"})],
        details={"config": config, "source": str(STB_CONCAT_CONTROL_SUMMARY.relative_to(REPO_ROOT))},
    )


def build_e8_two_index_table() -> BuiltTable:
    required = [STB_E8_BUDGETED_SUMMARY, STB_E8_RRF_SUMMARY]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise TableBlocked("missing E8 two-index inputs", missing=missing)
    budget_summary = load_json(STB_E8_BUDGETED_SUMMARY)
    rrf_summary = load_json(STB_E8_RRF_SUMMARY)
    budget_config = budget_summary.get("config") if isinstance(budget_summary.get("config"), dict) else {}
    if budget_summary.get("paper_eligible") is not True:
        raise TableBlocked("E8 budgeted summary is not paper eligible")
    seeds = [42, 123, 456]
    retrievers = ["jina", "bm25", "bge"]
    inputs = [
        artifact_record(STB_E8_BUDGETED_SUMMARY, extra={"role": "e8_budgeted_summary", "repo_commit": _summary_config_commit(budget_summary)}),
        artifact_record(STB_E8_RRF_SUMMARY, extra={"role": "e8_rrf_summary", "repo_commit": _summary_config_commit(rrf_summary)}),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Retriever & Rule & Held-out Acc & Held-out R@5 & Labeled R@5 \\",
        r"\midrule",
    ]
    numbers: dict[str, str] = {}
    details: dict[str, Any] = {"budgeted_cells": {}, "rrf_summary": str(STB_E8_RRF_SUMMARY.relative_to(REPO_ROOT))}
    for retriever in retrievers:
        paths = [STB_E8_BUDGETED_SUMMARY.parent / f"seed_{seed}" / retriever / "docs_plus_experience_backfill.json" for seed in seeds]
        missing_cells = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.exists()]
        if missing_cells:
            raise TableBlocked("missing E8 budgeted cell files", missing=missing_cells)
        metrics = _aggregate_cells(paths)
        for path in paths:
            cell = load_json(path)
            inputs.append(artifact_record(path, extra={"role": "e8_budgeted_cell", "repo_commit": cell.get("repo_commit"), "retriever": retriever, "seed": cell.get("seed")}))
        held_acc, held_acc_sd = metrics["heldout_accuracy"]
        held_r5, held_r5_sd = metrics["heldout_recall_at_5"]
        lab_r5, lab_r5_sd = metrics["labeled_recall_at_5"]
        overall_acc, overall_acc_sd = metrics["accuracy"]
        overall_r5, overall_r5_sd = metrics["recall_at_5"]
        lines.append(f"{retriever} & budgeted & {format_mean_sd(held_acc, held_acc_sd)} & {format_mean_sd(held_r5, held_r5_sd)} & {format_mean_sd(lab_r5, lab_r5_sd)}" + r" \\")
        numbers[f"e8_{retriever}_budget_heldout"] = f"{held_acc:.3f}"
        numbers[f"e8_{retriever}_budget_heldout_r5"] = f"{held_r5:.3f}"
        numbers[f"e8_{retriever}_budget_labeled_r5"] = f"{lab_r5:.3f}"
        numbers[f"e8_{retriever}_budget_overall"] = f"{overall_acc:.3f}"
        numbers[f"e8_{retriever}_budget_overall_r5"] = f"{overall_r5:.3f}"
        details["budgeted_cells"][retriever] = {"paths": [str(path.relative_to(REPO_ROOT)) for path in paths], "overall_accuracy": overall_acc, "overall_recall_at_5": overall_r5}
    rrf_agg = rrf_summary.get("aggregate", {})
    for retriever in retrievers:
        if retriever not in rrf_agg or "union_rrf" not in rrf_agg[retriever]:
            continue
        row = rrf_agg[retriever]["union_rrf"]
        held = row["heldout"]
        labeled = row["labeled"]
        lines.append(f"{retriever} & RRF-20 & {format_mean_sd(float(held['mean_accuracy']), float(held.get('sd_accuracy', 0.0)))} & {format_mean_sd(float(held['mean_recall_at_5']), float(held.get('sd_recall_at_5', 0.0)))} & {format_mean_sd(float(labeled['mean_recall_at_5']), float(labeled.get('sd_recall_at_5', 0.0)))}" + r" \\")
        numbers[f"e8_{retriever}_rrf_heldout"] = f"{float(held['mean_accuracy']):.3f}"
        numbers[f"e8_{retriever}_rrf_heldout_r5"] = f"{float(held['mean_recall_at_5']):.3f}"
        numbers[f"e8_{retriever}_rrf_labeled_r5"] = f"{float(labeled['mean_recall_at_5']):.3f}"
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{E8 two-index routing controls. Budgeted cells are from the corrected distinct-five replay; RRF rows are the original 20-entry-pool comparison.}",
        r"\label{tab:e8-two-index-generated}",
        r"\end{table}",
    ])
    return BuiltTable(
        name="e8_two_index",
        tex_name="e8_two_index.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=inputs,
        details={"budgeted_config": budget_config, **details},
    )


def build_d3_symmetric_expansion_table() -> BuiltTable:
    if not STB_D3_PROGRESS.exists():
        raise TableBlocked("missing D3 progress log", missing=[str(STB_D3_PROGRESS.relative_to(REPO_ROOT))])
    rows: list[dict[str, Any]] = []
    launch_record: dict[str, Any] | None = None
    for line in STB_D3_PROGRESS.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("stage") == "launch":
            launch_record = payload
        if payload.get("stage") == "arm_done" and "heldout_recall_at_5" in payload:
            rows.append(payload)
    if len(rows) != 27:
        raise TableBlocked("D3 did not complete all 27 retrieval cells", details={"arm_done_rows": len(rows), "expected": 27})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["retriever"]), str(row["arm"])), []).append(row)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Retriever & Arm & Held-out R@5 & Labeled R@5 \\",
        r"\midrule",
    ]
    numbers: dict[str, str] = {}
    for (retriever, arm), arm_rows in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in arm_rows})
        if seeds != [42, 123, 456]:
            raise TableBlocked("D3 cell is missing a seed", details={"retriever": retriever, "arm": arm, "seeds": seeds})
        held_mean, held_sd = stats([float(row["heldout_recall_at_5"]) for row in arm_rows])
        lab_mean, lab_sd = stats([float(row["labeled_recall_at_5"]) for row in arm_rows])
        lines.append(f"{retriever} & {latex_escape(arm)} & {format_mean_sd(held_mean, held_sd)} & {format_mean_sd(lab_mean, lab_sd)}" + r" \\")
        key_arm = {"synthetic_docs": "synthetic_docs", "synthetic_plus_experience": "synthetic_shared", "capped_synthetic_plus_experience": "synthetic_capped"}.get(arm, arm)
        numbers[f"d3_{retriever}_{key_arm}_heldout_r5"] = f"{held_mean:.3f}"
        numbers[f"d3_{retriever}_{key_arm}_labeled_r5"] = f"{lab_mean:.3f}"
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{D3 template-based symmetric expansion retrieval control. The run completed all cells; final summary serialization failed after the per-cell rows were written.}",
        r"\label{tab:d3-symmetric-expansion-generated}",
        r"\end{table}",
    ])
    inputs = [artifact_record(STB_D3_PROGRESS, extra={"role": "d3_progress", "repo_commit": (launch_record or {}).get("repo_commit"), "generation_mode": (launch_record or {}).get("generation_mode")})]
    if STB_D3_LAUNCH_LOG.exists():
        inputs.append(artifact_record(STB_D3_LAUNCH_LOG, extra={"role": "d3_launch_log", "summary_write_failure": "PosixPath not JSON serializable"}))
    return BuiltTable(
        name="d3_symmetric_expansion",
        tex_name="d3_symmetric_expansion.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=inputs,
        details={"arm_done_rows": len(rows), "generation_mode": (launch_record or {}).get("generation_mode"), "summary_status": "failed_after_cells_posixpath_json"},
    )


def build_d6_fusion_table() -> BuiltTable:
    if not STB_D6_SUMMARY.exists():
        raise TableBlocked("missing D6 fusion summary", missing=[str(STB_D6_SUMMARY.relative_to(REPO_ROOT))])
    payload = load_json(STB_D6_SUMMARY)
    if payload.get("paper_eligible") is not True:
        raise TableBlocked("D6 summary is not paper eligible", details={"paper_eligible": payload.get("paper_eligible")})
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    retrievers = ["jina", "bm25", "bge"]
    rules = ["weighted_score_fusion", "docs_plus_experience_backfill", "union_rrf"]
    inputs = [artifact_record(STB_D6_SUMMARY, extra={"role": "d6_summary", "repo_commit": _summary_config_commit(payload)})]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Retriever & Fusion & Held-out test R@5 & Held-out all R@5 & Labeled R@5 \\",
        r"\midrule",
    ]
    numbers: dict[str, str] = {}
    weight_records: list[dict[str, Any]] = []
    for retriever in retrievers:
        for rule in rules:
            paths = [STB_D6_SUMMARY.parent / f"seed_{seed}" / retriever / f"{rule}.json" for seed in [42, 123, 456]]
            missing = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.exists()]
            if missing:
                raise TableBlocked("missing D6 per-seed fusion cells", missing=missing)
            metrics = _aggregate_cells(paths)
            for path in paths:
                cell = load_json(path)
                inputs.append(artifact_record(path, extra={"role": "d6_cell", "repo_commit": cell.get("repo_commit"), "retriever": retriever, "rule": rule, "seed": cell.get("seed")}))
                if rule == "weighted_score_fusion":
                    weight_records.append({"retriever": retriever, "seed": cell.get("seed"), "fusion_weight": cell.get("fusion_weight"), "fusion_tuning": cell.get("fusion_tuning")})
            held_test_r5, held_test_r5_sd = metrics.get("heldout_test_recall_at_5", (0.0, 0.0))
            held_all_r5, held_all_r5_sd = metrics["heldout_recall_at_5"]
            lab_r5, lab_r5_sd = metrics["labeled_recall_at_5"]
            lines.append(f"{retriever} & {latex_escape(rule)} & {format_mean_sd(held_test_r5, held_test_r5_sd)} & {format_mean_sd(held_all_r5, held_all_r5_sd)} & {format_mean_sd(lab_r5, lab_r5_sd)}" + r" \\")
            key_rule = {"weighted_score_fusion": "weighted", "docs_plus_experience_backfill": "backfill", "union_rrf": "rrf"}[rule]
            numbers[f"d6_{retriever}_{key_rule}_heldout_test_r5"] = f"{held_test_r5:.3f}"
            numbers[f"d6_{retriever}_{key_rule}_heldout_r5"] = f"{held_all_r5:.3f}"
            numbers[f"d6_{retriever}_{key_rule}_labeled_r5"] = f"{lab_r5:.3f}"
    weights = {float(record["fusion_weight"]) for record in weight_records if record.get("fusion_weight") is not None}
    if weights != {0.0}:
        raise TableBlocked("D6 weighted fusion did not select zero weight for every retriever/seed", details={"weights": sorted(weights), "records": weight_records})
    numbers["d6_weighted_selected_weight"] = "0.0"
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{D6 fusion comparison. Weighted score fusion is tuned on a disjoint held-out development half; it selected zero experience weight for every retriever and seed.}",
        r"\label{tab:d6-fusion-generated}",
        r"\end{table}",
    ])
    return BuiltTable(
        name="d6_fusion",
        tex_name="d6_fusion.tex",
        tex_body="\n".join(lines),
        numbers=numbers,
        inputs=inputs,
        details={"config": config, "weighted_fusion_weights": weight_records, "rrf_note": "With retrieval_pool_size=200, additive RRF fills held-out top-5 slots with docs+experience tools and held-out recall is zero."},
    )


def key_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_d5_paired_stats_table() -> BuiltTable:
    if not STB_D5_SUMMARY.exists():
        raise TableBlocked("missing D5 paired-test summary", missing=[str(STB_D5_SUMMARY.relative_to(REPO_ROOT))])
    payload = load_json(STB_D5_SUMMARY)
    tests = payload.get("tests", [])
    if not tests:
        raise TableBlocked("D5 summary has no tests", details={"path": str(STB_D5_SUMMARY.relative_to(REPO_ROOT))})
    numbers: dict[str, str] = {}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparison & $n$ & diff & 95\% CI & TOST \\",
        r"\midrule",
    ]
    inputs = [artifact_record(STB_D5_SUMMARY)]
    for test in tests:
        name = str(test.get("name", "comparison"))
        slug = key_slug(name)
        ci = test.get("bootstrap_diff_ci", {})
        tost = test.get("tost_margin_0_02", {})
        mcn = test.get("mcnemar", {})
        diff = float(ci.get("mean", mcn.get("b_minus_a", 0.0)))
        low = float(ci.get("ci_low", 0.0))
        high = float(ci.get("ci_high", 0.0))
        n = int(test.get("paired_key_count", ci.get("n", mcn.get("n", 0))))
        p = float(mcn.get("p_exact_two_sided", 1.0))
        equivalent = bool(tost.get("equivalent", False))
        numbers[f"d5_{slug}_diff_pts"] = f"{diff * 100.0:.1f}"
        numbers[f"d5_{slug}_ci_low_pts"] = f"{low * 100.0:.1f}"
        numbers[f"d5_{slug}_ci_high_pts"] = f"{high * 100.0:.1f}"
        numbers[f"d5_{slug}_mcnemar_p"] = f"{p:.3g}"
        numbers[f"d5_{slug}_tost_equiv"] = "yes" if equivalent else "no"
        if len(lines) < 28:
            lines.append(f"{latex_escape(name)} & {n} & {diff * 100.0:.1f} & [{low * 100.0:.1f}, {high * 100.0:.1f}] & {'yes' if equivalent else 'no'}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{D5 paired tests. Differences are reported in accuracy/recall points according to each test outcome.}", r"\end{table}"]
    return BuiltTable("d5_paired_stats", "d5_paired_stats.tex", "\n".join(lines), numbers, inputs, {"test_count": len(tests)})


def build_p1_candidate_render_table() -> BuiltTable:
    if not STB_P1_SUMMARY.exists():
        raise TableBlocked("missing P1 candidate/render interaction summary", missing=[str(STB_P1_SUMMARY.relative_to(REPO_ROOT))])
    payload = load_json(STB_P1_SUMMARY)
    config = payload.get("config", {})
    aggregate = payload.get("aggregate", {})
    real = aggregate.get("real", {})
    oracle = aggregate.get("oracle", {})
    if not real:
        raise TableBlocked("P1 summary has no real aggregate", details={"path": str(STB_P1_SUMMARY.relative_to(REPO_ROOT))})
    numbers: dict[str, str] = {}
    inputs = [artifact_record(STB_P1_SUMMARY, extra={"repo_commit": config.get("repo_commit")})]
    rows = []
    for mode_name, mode_agg, prefix in [("real", real, "p1"), ("bounded_oracle", oracle, "bdo")]:
        if not isinstance(mode_agg, dict):
            continue
        for arm_name, subsets in mode_agg.items():
            if not isinstance(subsets, dict):
                continue
            arm_slug = key_slug(arm_name.replace("_candidates_", "_cand_"))
            for subset in ["all", "heldout", "labeled"]:
                metrics = subsets.get(subset)
                if not isinstance(metrics, dict):
                    continue
                key = f"{prefix}_{arm_slug}_{subset}"
                acc = float(metrics.get("mean_accuracy", 0.0))
                r5 = float(metrics.get("mean_recall_at_5", 0.0))
                cond = float(metrics.get("mean_conditional_accuracy_given_recall", 0.0))
                numbers[f"{key}_acc"] = f"{acc:.3f}"
                numbers[f"{key}_r5"] = f"{r5:.3f}"
                numbers[f"{key}_cond_acc"] = f"{cond:.3f}"
                if "sd_accuracy" in metrics:
                    numbers[f"{key}_acc_sd"] = f"{float(metrics.get('sd_accuracy', 0.0)):.3f}"
                if "sd_recall_at_5" in metrics:
                    numbers[f"{key}_r5_sd"] = f"{float(metrics.get('sd_recall_at_5', 0.0)):.3f}"
                if subset == "heldout":
                    rows.append((mode_name, arm_name, acc, r5, cond))
        did = mode_agg.get("difference_in_differences_all_accuracy") if isinstance(mode_agg, dict) else None
        if isinstance(did, (int, float)):
            numbers[f"{prefix}_did_all_acc_pts"] = f"{float(did) * 100.0:.1f}"
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrr}", r"\toprule", r"Mode & Cell & Acc. & R@5 & Acc.|R \\", r"\midrule"]
    for mode_name, arm_name, acc, r5, cond in rows:
        lines.append(f"{latex_escape(mode_name)} & {latex_escape(arm_name)} & {acc:.3f} & {r5:.3f} & {cond:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{P1 candidate-source/rendering interaction on the held-out split.}", r"\end{table}"]
    return BuiltTable("p1_candidate_render", "p1_candidate_render.tex", "\n".join(lines), numbers, inputs, {"config": config})


def build_d4_split_sensitivity_table() -> BuiltTable:
    modes = ["strict", "label_only", "mention_only"]
    arms = ["docs_only", "shared"]
    seeds = [42, 123, 456]
    missing = []
    numbers: dict[str, str] = {}
    inputs: list[dict[str, Any]] = []
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule", r"Filter & Arm & Held R@5 & Lab R@5 & Held acc & Lab acc \\", r"\midrule"]
    for mode in modes:
        for arm in arms:
            vals = {"heldout_r5": [], "labeled_r5": [], "heldout_top1": [], "labeled_top1": []}
            for seed in seeds:
                path = STB_D4_DIR / mode / f"seed_{seed}" / f"{arm}.json"
                if not path.exists():
                    missing.append(str(path.relative_to(REPO_ROOT)))
                    continue
                cell = load_json(path)
                metrics = cell.get("metrics", {})
                vals["heldout_r5"].append(float(metrics["heldout"]["recall_at_5"]))
                vals["labeled_r5"].append(float(metrics["labeled"]["recall_at_5"]))
                vals["heldout_top1"].append(float(metrics["heldout"].get("retrieval_top1", 0.0)))
                vals["labeled_top1"].append(float(metrics["labeled"].get("retrieval_top1", 0.0)))
                inputs.append(artifact_record(path, extra={"repo_commit": cell.get("repo_commit"), "filter_mode": mode, "arm": arm, "seed": seed}))
            if missing:
                continue
            prefix = f"d4_{key_slug(mode)}_{key_slug(arm)}"
            means = {}
            for metric, xs in vals.items():
                mean, sd = stats(xs)
                numbers[f"{prefix}_{metric}"] = f"{mean:.3f}"
                numbers[f"{prefix}_{metric}_sd"] = f"{sd:.3f}"
                means[metric] = mean
            lines.append(f"{latex_escape(mode)} & {latex_escape(arm)} & {means['heldout_r5']:.3f} & {means['labeled_r5']:.3f} & {means['heldout_top1']:.3f} & {means['labeled_top1']:.3f}" + r" \\")
    if missing:
        raise TableBlocked("missing D4 split-sensitivity cells", missing=missing)
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{D4 split-sensitivity retrieval results.}", r"\end{table}"]
    return BuiltTable("d4_split_sensitivity", "d4_split_sensitivity.tex", "\n".join(lines), numbers, inputs, {"modes": modes, "arms": arms, "seeds": seeds})


def build_e5b_separate_index_table() -> BuiltTable:
    paths = [REPO_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_e5b_seed42_r2" / "summary.json", STB_E5B_SUMMARY]
    missing = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.exists()]
    if missing:
        raise TableBlocked("missing E5b separate-index summaries", missing=missing)
    inputs = []
    accs_h: list[float] = []
    r5s_h: list[float] = []
    accs_l: list[float] = []
    r5s_l: list[float] = []
    details: dict[str, Any] = {"sources": []}
    for path in paths:
        payload = load_json(path)
        config = payload.get("config", {})
        agg = payload.get("aggregate", {})
        inputs.append(artifact_record(path, extra={"repo_commit": config.get("repo_commit")}))
        details["sources"].append({"path": str(path.relative_to(REPO_ROOT)), "seeds": config.get("seeds")})
        for subset, accs, r5s in [("heldout", accs_h, r5s_h), ("labeled", accs_l, r5s_l)]:
            m = agg.get(subset, {})
            seeds = config.get("seeds") or []
            count = max(1, len(seeds))
            accs.extend([float(m.get("mean_accuracy", 0.0))] * count)
            r5s.extend([float(m.get("mean_recall_at_5", 0.0))] * count)
    h_acc, h_acc_sd = stats(accs_h); h_r5, h_r5_sd = stats(r5s_h)
    l_acc, l_acc_sd = stats(accs_l); l_r5, l_r5_sd = stats(r5s_l)
    numbers = {
        "e5b_heldout_acc": f"{h_acc:.3f}", "e5b_heldout_acc_sd": f"{h_acc_sd:.3f}",
        "e5b_heldout_r5": f"{h_r5:.3f}", "e5b_heldout_r5_sd": f"{h_r5_sd:.3f}",
        "e5b_labeled_acc": f"{l_acc:.3f}", "e5b_labeled_acc_sd": f"{l_acc_sd:.3f}",
        "e5b_labeled_r5": f"{l_r5:.3f}", "e5b_labeled_r5_sd": f"{l_r5_sd:.3f}",
    }
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrr}", r"\toprule", r"Subset & Acc. & R@5 \\", r"\midrule", f"held-out & {format_mean_sd(h_acc, h_acc_sd)} & {format_mean_sd(h_r5, h_r5_sd)}" + r" \\", f"labeled & {format_mean_sd(l_acc, l_acc_sd)} & {format_mean_sd(l_r5, l_r5_sd)}" + r" \\", r"\bottomrule", r"\end{tabular}", r"\caption{E5b separate-index rerank results.}", r"\end{table}"]
    return BuiltTable("e5b_separate_index", "e5b_separate_index.tex", "\n".join(lines), numbers, inputs, details)


def build_toolret_audit_table() -> BuiltTable:
    required = [TOOLRET_DENSE_AUDIT_SUMMARY, TOOLRET_SPARSE_AUDIT_SUMMARY, TOOLRET_STB_OVERLAP_EXACT_SUMMARY, TOOLRET_STB_OVERLAP_NEARDUP_SUMMARY]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise TableBlocked("missing ToolRet audit summaries", missing=missing)
    dense = load_json(TOOLRET_DENSE_AUDIT_SUMMARY)
    sparse = load_json(TOOLRET_SPARSE_AUDIT_SUMMARY)
    exact = load_json(TOOLRET_STB_OVERLAP_EXACT_SUMMARY)
    near = load_json(TOOLRET_STB_OVERLAP_NEARDUP_SUMMARY)
    inputs = [artifact_record(path) for path in required]
    cats = dense.get("categories", {})
    dense_exact = [int(v.get("query_exact_overlap_with_toolbench_count", 0)) for v in cats.values()]
    dense_sample = [int(v.get("query_sample_count", 0)) for v in cats.values()]
    sparse_agg = sparse.get("aggregate", {})
    numbers = {
        "toolret_dense_categories_n": str(len(cats)),
        "toolret_dense_exact_min": str(min(dense_exact) if dense_exact else 0),
        "toolret_dense_sample_n": str(max(dense_sample) if dense_sample else 0),
        "toolret_sparse_config_n": str(len((sparse.get("config") or {}).get("configs", []))),
        "toolret_sparse_query_n": str(int(sparse_agg.get("query_count", 0))),
        "toolret_sparse_exact_n": str(int(sparse_agg.get("exact_overlap_count", 0))),
        "toolret_sparse_neardup_n": str(int(sparse_agg.get("near_duplicate_ge_0_95_count", 0))),
        "toolret_stb_exact_n": str(int(exact.get("exact_overlap_count", 0))),
        "toolret_stb_neardup_n": str(int(near.get("near_duplicate_ge_0_95_count", 0))),
        "toolret_stb_eval_n": str(int(near.get("stabletoolbench_query_count_after_junk_filter", exact.get("stabletoolbench_query_count_after_junk_filter", 0)))),
    }
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrr}", r"\toprule", r"Audit & exact & near-dup \\", r"\midrule", f"ToolRet dense sampled sub-corpora & {numbers['toolret_dense_exact_min']}/{numbers['toolret_dense_sample_n']} & {numbers['toolret_dense_exact_min']}/{numbers['toolret_dense_sample_n']}" + r" \\", f"ToolRet sparse pool & {numbers['toolret_sparse_exact_n']}/{numbers['toolret_sparse_query_n']} & {numbers['toolret_sparse_neardup_n']}/{numbers['toolret_sparse_query_n']}" + r" \\", f"StableToolBench test in ToolRet-train & {numbers['toolret_stb_exact_n']}/{numbers['toolret_stb_eval_n']} & {numbers['toolret_stb_neardup_n']}/{numbers['toolret_stb_eval_n']}" + r" \\", r"\bottomrule", r"\end{tabular}", r"\caption{ToolRet provenance and overlap audit counts.}", r"\end{table}"]
    return BuiltTable("toolret_audit", "toolret_audit.tex", "\n".join(lines), numbers, inputs, {"dense_categories": sorted(cats), "sparse_configs": (sparse.get("config") or {}).get("configs", [])})


def build_bounded_oracle_table() -> BuiltTable:
    # Uses the held-out distinct-five oracle run; these are the bounded/arm-own distractor oracle keys.
    if not STB_HELDOUT_ORACLE_DISTINCT5_SUMMARY.exists():
        raise TableBlocked("missing held-out distinct-five oracle summary", missing=[str(STB_HELDOUT_ORACLE_DISTINCT5_SUMMARY.relative_to(REPO_ROOT))])
    payload = load_json(STB_HELDOUT_ORACLE_DISTINCT5_SUMMARY)
    config = payload.get("config", {})
    per_arm = payload.get("per_arm", {})
    numbers: dict[str, str] = {}
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\toprule", r"Arm & Acc. & insertion & R@5 \\", r"\midrule"]
    for arm, data in per_arm.items():
        slug = "sy" if arm.startswith("synapse") else "do" if arm.startswith("flat") else key_slug(arm)
        acc = float(data.get("accuracy", {}).get("mean", 0.0))
        acc_sd = float(data.get("accuracy", {}).get("sd", 0.0))
        ins = float(data.get("oracle_insertion_fraction", {}).get("mean", 0.0))
        ins_sd = float(data.get("oracle_insertion_fraction", {}).get("sd", 0.0))
        numbers[f"bdo_{slug}_acc"] = f"{acc:.3f}"
        numbers[f"bdo_{slug}_acc_sd"] = f"{acc_sd:.3f}"
        numbers[f"bdo_{slug}_insert"] = f"{ins:.3f}"
        numbers[f"bdo_{slug}_insert_sd"] = f"{ins_sd:.3f}"
        numbers[f"bdo_{slug}_r5"] = "1.000"
        for group, gm in data.get("heldout_by_group", {}).items():
            gslug = key_slug(group)
            numbers[f"bdo_{slug}_{gslug}_acc"] = f"{float(gm.get('accuracy', {}).get('mean', 0.0)):.3f}"
            numbers[f"bdo_{slug}_{gslug}_r5"] = f"{float(gm.get('recall_at_5', {}).get('mean', 0.0)):.3f}"
        lines.append(f"{latex_escape(arm)} & {format_mean_sd(acc, acc_sd)} & {format_mean_sd(ins, ins_sd)} & 1.000" + r" \\")
    inputs = [artifact_record(STB_HELDOUT_ORACLE_DISTINCT5_SUMMARY, extra={"repo_commit": config.get("repo_commit")})]
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Held-out oracle reranking under the distinct-five candidate rule.}", r"\end{table}"]
    return BuiltTable("bounded_oracle", "bounded_oracle.tex", "\n".join(lines), numbers, inputs, {"config": config})

def emit_numbers_tex(numbers: dict[str, str]) -> None:
    lines = [f"\\defnum{{{key}}}{{{latex_escape(str(numbers[key]))}}}" for key in sorted(numbers)]
    text = "\n".join(lines) + ("\n" if lines else "")
    GENERATED_NUMBERS_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables from versioned artifacts with per-table blocking status.")
    parser.add_argument("--only", action="append", default=[], help="Build only the named table(s); defaults to all registered tables")
    args = parser.parse_args()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "transfer": build_transfer_table,
        "lessons": build_lessons_table,
        "benchmarks": build_benchmarks_table,
        "conflict_2x2": build_conflict_table,
        "c1_robustness": build_c1_robustness_table,
        "e5_heldout": build_e5_heldout_table,
        "concat_control": build_concat_control_table,
        "e8_two_index": build_e8_two_index_table,
        "d3_symmetric_expansion": build_d3_symmetric_expansion_table,
        "d6_fusion": build_d6_fusion_table,
        "d5_paired_stats": build_d5_paired_stats_table,
        "p1_candidate_render": build_p1_candidate_render_table,
        "d4_split_sensitivity": build_d4_split_sensitivity_table,
        "e5b_separate_index": build_e5b_separate_index_table,
        "toolret_audit": build_toolret_audit_table,
        "bounded_oracle": build_bounded_oracle_table,
    }
    requested = set(args.only) if args.only else set(builders)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "tables": {},
        "superseded": {
            "gsm8k": "Historical GSM8K tables are superseded and are never required by make_tables.",
        },
    }
    all_numbers: dict[str, str] = {}

    for name, builder in builders.items():
        if name not in requested:
            continue
        try:
            built = builder()
        except TableBlocked as exc:
            manifest["tables"][name] = {
                "status": "blocked",
                "reason": exc.reason,
                "missing": exc.missing,
                "details": exc.details,
            }
            continue
        tex_path = GENERATED_DIR / built.tex_name
        write_table(tex_path, built.tex_body)
        manifest["tables"][name] = {
            "status": "built",
            "output": str(tex_path.relative_to(REPO_ROOT)),
            "inputs": built.inputs,
            "details": built.details,
        }
        all_numbers.update(built.numbers)

    emit_numbers_tex(all_numbers)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    built = [name for name, data in manifest["tables"].items() if data["status"] == "built"]
    blocked = [name for name, data in manifest["tables"].items() if data["status"] == "blocked"]
    print(json.dumps({"built": built, "blocked": blocked, "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT))}, indent=2))


if __name__ == "__main__":
    main()
