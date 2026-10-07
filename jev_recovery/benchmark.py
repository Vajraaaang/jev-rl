"""Paired, frozen-policy experiments and independent artifact verification.

Simulator health, never the advisor, is the recovery ground truth. Confidence
intervals resample scenario families and controller training seeds, rather than
treating deterministic policy repeats as independent observations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from typing import Any

from .agent import ChecklistPolicy, GreedyRepairPolicy, RandomPolicy, RecoveryAgent
from .jev import CAUSES, Diagnosis, DiagnosticError, DiagnosticProvider
from .learning import LinearQPolicy, train_policy
from .sandbox import IncidentEnv
from .scenarios import Incident, generate_suite

SCHEMA_VERSION = "recovery-benchmark-v2"
POLICIES = ("checklist", "random", "greedy", "rl_without_jev", "rl_with_jev",
            "rl_with_corrupted_advisor")
METRICS = ("recovered", "actions", "downtime_ticks", "interventions",
           "unnecessary_interventions", "tool_cost_units", "api_cost_usd",
           "input_tokens", "api_calls", "cache_hits", "reward", "safety_violations")
BREAKDOWNS = (("by_suite", "suite"), ("by_training_seed", "seed"),
              ("by_recoverability", "recoverable"), ("by_fault_combination", "fault_combination"),
              ("by_log_noise", "noise"), ("by_dependency_regime", "dependency_regime"))


def _absolute(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_absolute():
        raise ValueError("Artifact paths must be absolute")
    return result.resolve()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _RecordingAdvisor:
    """Capture the exact fixed sensor responses without giving it fault labels."""
    def __init__(self, provider: DiagnosticProvider, corrupted: bool = False):
        self.provider, self.corrupted = provider, corrupted
        self.calls: list[dict] = []

    def diagnose(self, logs: str) -> Diagnosis:
        try:
            result = self.provider.diagnose(logs)
            if self.corrupted:
                result = replace(result, probabilities={k: 1 - v for k, v in
                                  result.probabilities.items()},
                                 source="corrupted-" + result.source)
            self.calls.append({"logs": logs, "diagnosis": asdict(result)})
            return result
        except DiagnosticError as exc:
            self.calls.append({"logs": logs, "error": str(exc)})
            raise


class _ReplayAdvisor:
    """Replay acquired sensor responses exactly, including observed failures."""
    def __init__(self, calls: list[dict]):
        self.calls, self.index = calls, 0

    def diagnose(self, logs: str) -> Diagnosis:
        if self.index >= len(self.calls):
            raise ValueError("Replay requested an unrecorded diagnostic")
        call = self.calls[self.index]
        self.index += 1
        if call["logs"] != logs:
            raise ValueError("Replay diagnostic request differs from recorded logs")
        if "error" in call:
            raise DiagnosticError(call["error"])
        return Diagnosis(**call["diagnosis"])


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"episodes": 0, "recovery_rate": None}
    result = {"episodes": len(rows)}
    result["mean_agent_safety_blocks"] = statistics.fmean(len(row["agent_result"].get("safety_blocks", [])) for row in rows)
    result["total_diagnostic_failures"] = sum(row["agent_result"].get("diagnosis_failures", 0) for row in rows)
    for metric in METRICS:
        values = [float(row["metrics"].get(metric, 0)) for row in rows]
        result["recovery_rate" if metric == "recovered" else "mean_" + metric] = statistics.fmean(values)
        if metric in ("api_cost_usd", "input_tokens", "api_calls", "cache_hits"):
            result["total_" + metric] = sum(values)
    return result


def _aggregates(rows: list[dict]) -> dict:
    report = {}
    for name in POLICIES:
        selected = [row for row in rows if row["policy"] == name]
        report[name] = {"overall": _summarize(selected)}
        for dimension, key in BREAKDOWNS:
            labels = sorted({str(row[key]) for row in selected})
            report[name][dimension] = {label: _summarize([row for row in selected
                                       if str(row[key]) == label]) for label in labels}
    return report


def _paired_intervals(rows: list[dict], draws: int = 1200) -> dict:
    """Crossed bootstrap preserves pairing and clusters related suite variants.

    Training seeds and independent incident families are sampled with replacement
    separately. A family keeps all wording/combo transformations and policy
    results together. Reported uncertainty is conditional on this synthetic
    generator, not a confidence statement about production incidents.
    """
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["seed"], row["cluster_id"])][(row["scenario_id"], row["policy"])] = row
    seeds = sorted({row["seed"] for row in rows})
    clusters = sorted({row["cluster_id"] for row in rows})
    comparisons = (("rl_with_jev", "rl_without_jev"), ("rl_with_jev", "checklist"),
                   ("rl_with_jev", "rl_with_corrupted_advisor"))
    result = {"method": "crossed paired bootstrap of training seeds and incident families",
              "draws": draws, "training_seed_clusters": len(seeds),
              "incident_family_clusters": len(clusters), "confidence": 0.95,
              "interpretation": "Synthetic population only; few training seeds make seed uncertainty unstable.",
              "comparisons": {}}
    rng = random.Random(94731)
    for left, right in comparisons:
        comparison = {}
        for metric in ("recovered", "actions", "downtime_ticks", "tool_cost_units"):
            cells = {}
            for seed in seeds:
                for cluster in clusters:
                    records = paired[(seed, cluster)]
                    differences = [float(row["metrics"][metric]) - float(records[(scenario, right)]["metrics"][metric])
                                   for (scenario, policy), row in records.items() if policy == left]
                    if not differences:
                        raise ValueError("Unpaired scenario in evaluation")
                    cells[(seed, cluster)] = differences
            estimate = statistics.fmean(value for values in cells.values() for value in values)
            sampled = []
            for _ in range(draws):
                sample_seeds = rng.choices(seeds, k=len(seeds))
                sample_clusters = rng.choices(clusters, k=len(clusters))
                sampled.append(statistics.fmean(value for seed in sample_seeds for cluster in sample_clusters
                                               for value in cells[(seed, cluster)]))
            sampled.sort()
            comparison[metric] = {"difference": estimate,
                                  "ci95": [sampled[int(draws * .025)], sampled[min(draws - 1, int(draws * .975))]],
                                  "direction": "higher is better" if metric == "recovered" else "lower is better"}
        result["comparisons"][left + " minus " + right] = comparison
    return result


def _diagnostic_evaluation(incidents: list[Incident], provider: DiagnosticProvider) -> tuple[dict, list[dict]]:
    """Independently labeled initial incidents; truth is used only after prediction."""
    fixtures, tp, fp, fn, exact, top, brier, available = [], 0, 0, 0, 0, 0, 0.0, 0
    for incident in incidents:
        logs = IncidentEnv(incident).logs()
        fixture = {"scenario_id": incident.id, "logs": logs,
                   "labels": {cause: int(cause in incident.faults) for cause in CAUSES}}
        try:
            diagnosis = provider.diagnose(logs)
            fixture["diagnosis"] = asdict(diagnosis)
            labels = fixture["labels"]
            predictions = {cause: int(diagnosis.probabilities[cause] >= .5) for cause in CAUSES}
            brier += sum((diagnosis.probabilities[cause] - labels[cause]) ** 2 for cause in CAUSES) / len(CAUSES)
            tp += sum(predictions[cause] and labels[cause] for cause in CAUSES)
            fp += sum(predictions[cause] and not labels[cause] for cause in CAUSES)
            fn += sum(not predictions[cause] and labels[cause] for cause in CAUSES)
            exact += predictions == labels
            if incident.faults:
                top += diagnosis.ranked_causes[0] in incident.faults
            available += 1
        except DiagnosticError as exc:
            fixture["error"] = str(exc)
        fixtures.append(fixture)
    faulty_available = sum(any(item["labels"].values()) and "diagnosis" in item for item in fixtures)
    return {"fixtures": len(fixtures), "available": available, "failures": len(fixtures) - available,
            "multilabel_brier": brier / available if available else None,
            "multilabel_exact_match": exact / available if available else None,
            "micro_precision": tp / (tp + fp) if tp + fp else None,
            "micro_recall": tp / (tp + fn) if tp + fn else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "top_cause_in_true_set": top / faulty_available if faulty_available else None,
            "threshold": .5, "labels": "Injected initial faults, evaluated separately from recovery",
            "top_cause_denominator": faulty_available,
            "sampling": "One initial log excerpt per held-out joint-test incident; no controller-selected log bias"}, fixtures


def _provenance() -> dict:
    root = Path(__file__).resolve().parent.parent
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"python": sys.version, "platform": platform.platform(), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "schema_version": SCHEMA_VERSION,
            "source_sha256": {str(path.relative_to(root)): _sha(path)
                              for path in sorted((root / "jev_recovery").glob("*.py"))}}


def _policy(name: str, seed: int, checkpoints: dict, output: Path, incident: Incident):
    if name == "checklist":
        return ChecklistPolicy()
    if name == "random":
        identity = hashlib.sha256(f"{seed}:{incident.id}:{incident.seed}".encode()).digest()
        return RandomPolicy(seed=int.from_bytes(identity[:8], "big"))
    if name == "greedy":
        return GreedyRepairPolicy()
    arm = "rl_without_jev" if name == "rl_without_jev" else "rl_with_jev"
    return LinearQPolicy.load(output / checkpoints[str(seed)][arm]["path"])


def _dependency_regime(incident: Incident) -> str:
    if "dependency" not in incident.faults:
        return "not_injected"
    return "permanent" if incident.dependency_ready_at is None else "delayed" if incident.dependency_ready_at > 12 else "normal"


def _markdown(report: dict) -> str:
    lines = ["# Incident recovery benchmark", "", report["advisor"]["label"], "",
             "Simulator probes alone determine sustained recovery. The advisor never grades its own success.", "",
             "| Policy | Recovery | Actions | Downtime ticks | Tool cost units |", "|---|---:|---:|---:|---:|"]
    for name, metrics in report["policies"].items():
        row = metrics["overall"]
        lines.append(f"| {name} | {row['recovery_rate']:.1%} | {row['mean_actions']:.2f} | {row['mean_downtime_ticks']:.2f} | {row['mean_tool_cost_units']:.2f} |")
    lines += ["", "## Paired uncertainty", "", report["uncertainty"]["method"] + ". " + report["uncertainty"]["interpretation"], ""]
    for name, metrics in report["uncertainty"]["comparisons"].items():
        value = metrics["recovered"]
        lines.append(f"- {name}: recovery difference {value['difference']:.3f}, 95% interval [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}].")
    lines += ["", "## Limits", ""] + ["- " + limit for limit in report["limitations"]]
    lines += ["", "Diagnostic accuracy, per-suite, per-seed, impossible-incident, cost and provenance details are in report.json.", ""]
    return "\n".join(lines)


def run_benchmark(output: Path, episodes: int = 4000, count: int = 60,
                  seeds: tuple[int, ...] = (11, 22, 33), advisor_mode: str = "offline",
                  cache_path: str | Path | None = None) -> dict:
    output = _absolute(output)
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 6:
        raise ValueError("count must be at least six to cover scenario strata")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Training seeds must be distinct integers")
    source_provenance = _provenance()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve existing experiment evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    if cache_path is not None:
        cache_path = _absolute(cache_path)
    elif advisor_mode in ("offline", "live"):
        cache_path = output / "advisor-cache.json"
    provider = DiagnosticProvider(advisor_mode, cache_path)
    train = list(generate_suite("train", max(60, count), 1729))
    validation = list(generate_suite("validation", count, 2718))
    joint = list(generate_suite("test", count, 3141))
    wording = [replace(item, id="wording-" + item.id, wording="test") for item in train[:count]]
    combinations = [replace(item, id="combination-" + item.id, wording="train") for item in joint]
    stress = [replace(item, id="stress-" + item.id, max_ticks=(16, 24, 48)[index % 3],
                      restart_duration=(1, 3, 5)[index % 3], relapse_after=(1, 2, 3)[index % 3],
                      dependency_ready_at=(None, 16, 32)[index % 3], noise=True)
              for index, item in enumerate(generate_suite("test", count, 57721))]
    suites = {"joint_holdout": joint, "wording_only": wording,
              "combination_only": combinations, "stress": stress}
    cluster_map = {item.id: item.id for item in joint}
    cluster_map.update({item.id: original.id for item, original in zip(wording, train)})
    cluster_map.update({item.id: original.id for item, original in zip(combinations, joint)})
    cluster_map.update({item.id: item.id for item in stress})
    manifest = {"schema_version": SCHEMA_VERSION,
                "train": [asdict(item) for item in train], "validation": [asdict(item) for item in validation],
                "test_suites": {name: [asdict(item) for item in incidents] for name, incidents in suites.items()},
                "clusters": cluster_map,
                "split_design": "Joint split is disjoint wording and reserved combinations; wording-only reuses train dynamics with held-out text; combination-only uses test dynamics with train text. Independent stress incidents vary deadlines, operation duration, rapid relapse, permanent/late dependencies and noisy logs. Fixed checkpoint candidates are selected solely using saved validation incidents; test incidents never enter selection. This is an exploratory synthetic benchmark, not an external blinded lockbox.",
                "config": {"episodes": episodes, "count": count, "seeds": list(seeds), "advisor_mode": advisor_mode,
                           "cache_path": str(cache_path) if cache_path else None,
                           "validation_interval": max(1, episodes // 4),
                           "selection_metric": "mean_validation_reward"}}
    interval = manifest["config"]["validation_interval"]
    manifest["selection_plan"] = {
        "candidate_episodes": sorted(set(range(interval, episodes + 1, interval)) | {episodes}),
        "validation_manifest_sha256": hashlib.sha256(_json({"incidents": manifest["validation"]}).encode()).hexdigest(),
        "selection_metric": "mean_validation_reward", "test_set_used": False,
        "tie_break": "earliest checkpoint", "created_before_training": True}
    _write(output / "manifest.json", manifest)
    checkpoints = {}
    training = {}
    for seed in seeds:
        checkpoints[str(seed)], training[str(seed)] = {}, {}
        for name, use_jev in (("rl_without_jev", False), ("rl_with_jev", True)):
            policy, summary = train_policy(train, episodes=episodes, seed=seed, use_jev=use_jev,
                                           advisor=provider if use_jev else None,
                                           validation_incidents=validation,
                                           validation_interval=manifest["config"]["validation_interval"])
            path = output / "checkpoints" / f"{name}-{seed}.json"
            policy.save(path)
            checkpoints[str(seed)][name] = {"path": str(path.relative_to(output)), "sha256": _sha(path)}
            training[str(seed)][name] = summary
    _write(output / "training.json", training)
    rows = []
    for seed in seeds:
        for suite, incidents in suites.items():
            for incident in incidents:
                for name in POLICIES:
                    advisor = _RecordingAdvisor(provider, corrupted=name == "rl_with_corrupted_advisor")
                    env = IncidentEnv(incident, advisor=advisor)
                    policy = _policy(name, seed, checkpoints, output, incident)
                    agent = RecoveryAgent(policy)
                    agent_result = agent.run(env)
                    rows.append({"scenario_id": incident.id, "cluster_id": cluster_map[incident.id],
                                 "suite": suite, "seed": seed, "policy": name,
                                 "recoverable": incident.recoverable,
                                 "fault_combination": "+".join(incident.faults) or "healthy",
                                 "noise": incident.noise,
                                 "dependency_regime": _dependency_regime(incident),
                                 "metrics": env.metrics(), "trace": env.trace,
                                 "agent_result": agent_result, "diagnostic_calls": advisor.calls})
    (output / "episodes.jsonl").write_text("".join(_json(row) + "\n" for row in rows))
    diagnostics, fixtures = _diagnostic_evaluation(joint, provider)
    _write(output / "diagnostics.json", fixtures)
    if cache_path is not None and cache_path.exists():
        (output / "advisor-cache-snapshot.json").write_bytes(cache_path.read_bytes())
    final_sources = _provenance()["source_sha256"]
    if final_sources != source_provenance["source_sha256"]:
        raise RuntimeError("Source files changed during the experiment; preserve this partial run and rerun with frozen sources")
    report = {"schema_version": SCHEMA_VERSION, "config": manifest["config"],
              "advisor": {"mode": advisor_mode, "model": provider.model,
                          "label": {"offline": "Offline heuristic experiment; hosted Jev effectiveness is untested.",
                                    "cached": "Cached hosted Jev responses only; misses become diagnostic failures.",
                                    "live": "Hosted Jev acquisition with fixed requested weights."}[advisor_mode],
                          "acquisition_totals": dict(provider.statistics)},
              "training": training, "checkpoints": checkpoints,
              "selection_plan": manifest["selection_plan"],
              "policies": _aggregates(rows), "uncertainty": _paired_intervals(rows),
              "diagnostics": diagnostics, "provenance": source_provenance,
              "source_stability": {"checked_at_start_and_end": True, "unchanged": True,
                                   "final_source_sha256": final_sources},
              "limitations": ["Small synthetic sandbox; recovery results do not establish production safety or transfer.",
                               "Only local controllers learn. Diagnostic sensor weights and rubric remain fixed.",
                               "Independent cause scores are multilabel scores, not calibrated probabilities.",
                               "Three seeds by default provide limited information about training variability.",
                               "Impossible incidents remain in overall recovery denominators; subgroup metrics report them separately.",
                               "Costs are simulated tool units and provider acquisition estimates, not total infrastructure or output-token billing.",
                               "Fixed validation-only checkpoint selection; no hyperparameter search or test-set checkpoint selection.",
                               "Exploratory synthetic benchmark: previous test results have been inspected, so this is not an external blinded lockbox.",
                               "Advisor corruption is a probability-complement stress test, not a realistic model of all diagnostic failures."]}
    _write(output / "report.json", report)
    (output / "report.md").write_text(_markdown(report))
    inventory = {str(path.relative_to(output)): _sha(path) for path in sorted(output.rglob("*")) if path.is_file()}
    _write(output / "integrity.json", {"schema_version": SCHEMA_VERSION, "files": inventory})
    return report


def _load_run(output: Path) -> tuple[dict, dict, list[dict]]:
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines() if line.strip()]
    return manifest, report, rows


def audit_run(output: Path) -> dict:
    """Detect accidental corruption and invalid experimental identities.

    Hash inventories are integrity checks, not signatures against an attacker who
    can rewrite both artifacts and inventory. Semantic checks still run when the
    hashes have been regenerated.
    """
    output = _absolute(output)
    errors = []
    try:
        inventory = json.loads((output / "integrity.json").read_text())
        for relative, expected in inventory["files"].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output):
                errors.append("Inventory path escapes artifact directory")
                continue
            if not path.is_file() or _sha(path) != expected:
                errors.append("SHA256 mismatch: " + relative)
        manifest, report, rows = _load_run(output)
        if manifest["schema_version"] != SCHEMA_VERSION or report["schema_version"] != SCHEMA_VERSION:
            errors.append("Unsupported schema version")
        groups = {"train": manifest["train"], "validation": manifest["validation"],
                  **manifest["test_suites"]}
        identifiers = {name: {item["id"] for item in items} for name, items in groups.items()}
        for name, items in groups.items():
            if len(identifiers[name]) != len(items):
                errors.append("Duplicate scenario IDs: " + name)
            for item in items:
                Incident.from_dict(item)
        names = list(groups)
        for index, name in enumerate(names):
            for other in names[index + 1:]:
                if identifiers[name] & identifiers[other]:
                    errors.append(f"Overlapping scenario IDs: {name}/{other}")
        if {item["wording"] for item in groups["train"]} & {item["wording"] for item in groups["joint_holdout"]}:
            errors.append("Joint holdout wording leaks into training")
        train_faults = {tuple(item["faults"]) for item in groups["train"]}
        reserved = {tuple(item["faults"]) for item in groups["joint_holdout"] if len(item["faults"]) > 1}
        if reserved & train_faults:
            errors.append("Joint holdout fault combinations leak into training")
        if reserved & {tuple(item["faults"]) for item in groups["validation"]}:
            errors.append("Joint holdout fault combinations leak into validation selection")
        expected = {(seed, suite, item["id"], policy) for seed in manifest["config"]["seeds"]
                    for suite, items in manifest["test_suites"].items() for item in items for policy in POLICIES}
        actual = [(row["seed"], row["suite"], row["scenario_id"], row["policy"]) for row in rows]
        if set(actual) != expected or len(actual) != len(expected):
            errors.append("Evaluation identities are missing, duplicated or unpaired")
        incidents = {item["id"]: Incident.from_dict(item) for items in manifest["test_suites"].values() for item in items}
        for row in rows:
            incident = incidents.get(row["scenario_id"])
            if incident is None:
                continue
            if row["recoverable"] != incident.recoverable or row["cluster_id"] != manifest["clusters"][incident.id]:
                errors.append("Incorrect recoverability or cluster identity")
            if (row["fault_combination"] != ("+".join(incident.faults) or "healthy") or row["noise"] != incident.noise
                    or row["dependency_regime"] != _dependency_regime(incident)):
                errors.append("Incorrect evaluator subgroup metadata")
            for trace in row["trace"]:
                for key in ("observation_before", "observation_after"):
                    forbidden = {"faults", "scenario_id", "seed", "wording"} & set(trace.get(key, {}))
                    if forbidden:
                        errors.append("Hidden scenario labels in public observations")
            for metric in METRICS:
                value = row["metrics"].get(metric)
                if not isinstance(value, (int, float, bool)) or not math.isfinite(float(value)):
                    errors.append("Invalid episode metric: " + metric)
        if _aggregates(rows) != report["policies"]:
            errors.append("Reported aggregate metrics differ from episode records")
        if manifest["config"] != report["config"]:
            errors.append("Report configuration differs from immutable manifest")
        if "selection_plan" in manifest:
            plan = manifest["selection_plan"]
            if report.get("selection_plan") != plan:
                errors.append("Reported checkpoint selection plan differs from manifest")
            digest = hashlib.sha256(_json({"incidents": manifest["validation"]}).encode()).hexdigest()
            if plan["validation_manifest_sha256"] != digest or plan["test_set_used"]:
                errors.append("Validation checkpoint selection manifest mismatch or test-set selection")
            interval = manifest["config"]["validation_interval"]
            budget = manifest["config"]["episodes"]
            if plan["candidate_episodes"] != sorted(set(range(interval, budget + 1, interval)) | {budget}):
                errors.append("Checkpoint candidates differ from predeclared equal-budget plan")
        if "source_stability" in report:
            stability = report["source_stability"]
            if not stability["unchanged"] or stability["final_source_sha256"] != report["provenance"]["source_sha256"]:
                errors.append("Source files changed during the experiment")
        fixtures = json.loads((output / "diagnostics.json").read_text())
        fixture_calls = [{key: value for key, value in fixture.items() if key in ("logs", "diagnosis", "error")}
                         for fixture in fixtures]
        fixture_advisor = _ReplayAdvisor(fixture_calls)
        scores, regenerated_fixtures = _diagnostic_evaluation(
            [Incident.from_dict(item) for item in manifest["test_suites"]["joint_holdout"]], fixture_advisor)
        if scores != report["diagnostics"] or regenerated_fixtures != fixtures:
            errors.append("Diagnostic labels, logs or scores differ from independently labeled fixtures")
        if _paired_intervals(rows) != report["uncertainty"]:
            errors.append("Paired uncertainty differs from clustered episode records")
        for seed, arms in report["checkpoints"].items():
            for arm, checkpoint in arms.items():
                policy = None
                path = (output / checkpoint["path"]).resolve()
                if not path.is_relative_to(output) or _sha(path) != checkpoint["sha256"]:
                    errors.append("Frozen checkpoint checksum mismatch")
                else:
                    policy = LinearQPolicy.load(path)
                if report["training"][seed][arm]["episodes"] != manifest["config"]["episodes"]:
                    errors.append("Unequal controller training episode budget")
                if "selection_plan" in manifest:
                    summary, plan = report["training"][seed][arm], manifest["selection_plan"]
                    provenance, scores = summary["selection_provenance"], summary["validation_scores"]
                    if (summary["validation_manifest_sha256"] != plan["validation_manifest_sha256"]
                            or provenance["validation_ids"] != [item["id"] for item in manifest["validation"]]
                            or provenance["split"] != "validation"):
                        errors.append("Checkpoint selection used a different validation manifest")
                    if (provenance["candidate_episodes"] != plan["candidate_episodes"]
                            or [score["after_episode"] for score in scores] != plan["candidate_episodes"]
                            or not provenance["full_training_budget_completed"]
                            or not provenance["greedy_frozen_evaluation"]
                            or summary["selection_metric"] != plan["selection_metric"]):
                        errors.append("Checkpoint selection differs from predeclared candidate budget")
                    best = max(scores, key=lambda score: (score["mean_reward"], -score["after_episode"]))
                    if (summary["selected_after_episode"] != best["after_episode"]
                            or summary["weights_sha256"] != best["weights_sha256"]
                            or policy is not None and (policy.training_episodes != best["after_episode"]
                                                       or policy.checksum() != best["weights_sha256"])):
                        errors.append("Frozen checkpoint differs from the validation-only reward selection")
            summaries = report["training"][seed]
            if summaries["rl_with_jev"]["incident_schedule_sha256"] != summaries["rl_without_jev"]["incident_schedule_sha256"]:
                errors.append("Controller training incident schedules are not paired")
        if report["advisor"]["mode"] != "live" and report["advisor"]["acquisition_totals"].get("hosted_calls", 0):
            errors.append("Non-live benchmark recorded hosted API calls")
        return {"ok": not errors, "errors": sorted(set(errors)), "episodes_checked": len(rows),
                "files_checked": len(inventory["files"]),
                "trust_boundary": "Unsigned checksums detect corruption; source hashes establish identity, not authenticity."}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append("Invalid artifact structure: " + str(exc))
        return {"ok": False, "errors": sorted(set(errors))}


def replay_run(output: Path) -> dict:
    """Load frozen policies and rerun every trajectory without hosted calls."""
    output = _absolute(output)
    audit = audit_run(output)
    if not audit["ok"]:
        return {"ok": False, "errors": audit["errors"], "episodes_replayed": 0}
    manifest, report, rows = _load_run(output)
    incidents = {item["id"]: Incident.from_dict(item) for items in manifest["test_suites"].values() for item in items}
    errors = []
    for row in rows:
        incident = incidents[row["scenario_id"]]
        advisor = _ReplayAdvisor(row["diagnostic_calls"])
        env = IncidentEnv(incident, advisor=advisor)
        try:
            policy = _policy(row["policy"], row["seed"], report["checkpoints"], output, incident)
            agent_result = RecoveryAgent(policy).run(env)
            if (_json(env.metrics()) != _json(row["metrics"]) or _json(env.trace) != _json(row["trace"])
                    or _json(agent_result) != _json(row["agent_result"]) or advisor.index != len(advisor.calls)):
                errors.append(f"Trajectory mismatch: {row['seed']}/{row['policy']}/{incident.id}")
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            errors.append(f"Replay failure for {incident.id}: {exc}")
    return {"ok": not errors, "errors": errors, "episodes_replayed": len(rows),
            "diagnostic_replay": "Recorded fixed sensor responses; no network acquisition",
            "verification": "All frozen policy decisions, observations, rewards, metrics and traces reproduced"}
