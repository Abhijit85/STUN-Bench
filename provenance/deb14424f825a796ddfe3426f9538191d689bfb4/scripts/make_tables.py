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
NUMBERS_PATHS = [TABLES_DIR / "numbers.tex", GENERATED_DIR / "numbers.tex"]
RUNS_ROOT = Path("<REPO_ROOT>_runs")

TAU_TRANSFER_MANIFEST = REPO_ROOT / "artifacts" / "results" / "tau_bench_a5_sidecars_canonical.json"
LESSONS_CLASSIFIER_SMOKE = REPO_ROOT / "artifacts" / "verification" / "stb_precondition_smoke_classifier" / "combined_summary.json"
STB_CLEAN_CLASSIFIER_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_query_classifier_clean_r1" / "combined_summary.json"
STB_TRULY_UNSEEN_ANALYSIS = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_truly_unseen_analysis.json"
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
    summary_paths = resolve_a1_summary_paths()
    if len(summary_paths) < 3:
        expected = [
            "<REPO_ROOT>_runs/artifacts/verification/stabletoolbench_a1_seed42_eval_r*/combined_summary.json",
            "<REPO_ROOT>_runs/artifacts/verification/stabletoolbench_a1_seed123_eval_r*/combined_summary.json",
            "<REPO_ROOT>_runs/artifacts/verification/stabletoolbench_a1_seed456_eval_r*/combined_summary.json",
        ]
        raise TableBlocked("A1 benchmark summaries are not complete yet", missing=expected)

    summaries = [load_json(path) for path in summary_paths]
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
    inputs = [artifact_record(path) for path in summary_paths]
    inputs.append(artifact_record(STB_CLEAN_CLASSIFIER_SUMMARY))

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
                accs = [summary["arms"][arm]["metrics"]["per_group"][group]["accuracy"]["mean"] for summary in summaries]
                recs = [summary["arms"][arm]["metrics"]["per_group"][group]["recall_at_5"]["mean"] for summary in summaries]
                acc_mean = statistics.mean(accs) if accs else 0.0
                acc_sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
                rec_mean = statistics.mean(recs) if recs else 0.0
                rec_sd = statistics.stdev(recs) if len(recs) > 1 else 0.0
            row.append(f"{acc_mean:.3f} $\\pm$ {acc_sd:.3f}")
            row.append(f"{rec_mean:.3f} $\\pm$ {rec_sd:.3f}")
            numbers[f"{arm}_{group}_acc"] = f"{acc_mean:.3f}"
            numbers[f"{arm}_{group}_r5"] = f"{rec_mean:.3f}"
        lines.append(" & ".join(row) + r" \\")

    qc_clean_acc = float(clean_classifier["query_classifier"]["mean_accuracy"])
    synapse_means = [summary["arms"]["synapse"]["metrics"]["accuracy"]["mean"] for summary in summaries]
    flat_pool_means = [summary["arms"]["flat_pool"]["metrics"]["accuracy"]["mean"] for summary in summaries]
    gap_pts = (statistics.mean(synapse_means) - statistics.mean(flat_pool_means)) * 100.0 if synapse_means and flat_pool_means else 0.0
    numbers["qc_clean_acc"] = f"{qc_clean_acc:.3f}"
    numbers["gap_pts"] = f"{gap_pts:.1f}"

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
            "summary_paths": [str(path) for path in summary_paths],
            "clean_classifier_summary_path": str(STB_CLEAN_CLASSIFIER_SUMMARY),
            "groups": groups,
            "group_labels": group_labels,
            "arms": arms,
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

def emit_numbers_tex(numbers: dict[str, str]) -> None:
    lines = [f"\\defnum{{{key}}}{{{latex_escape(str(numbers[key]))}}}" for key in sorted(numbers)]
    text = "\n".join(lines) + ("\n" if lines else "")
    for path in NUMBERS_PATHS:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables from versioned artifacts with per-table blocking status.")
    parser.add_argument("--only", action="append", default=[], help="Build only the named table(s): transfer, lessons, benchmarks, conflict_2x2")
    args = parser.parse_args()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    requested = set(args.only) if args.only else {"transfer", "lessons", "benchmarks", "conflict_2x2"}
    builders = {
        "transfer": build_transfer_table,
        "lessons": build_lessons_table,
        "benchmarks": build_benchmarks_table,
        "conflict_2x2": build_conflict_table,
    }

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
