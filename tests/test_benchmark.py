"""Research protocol, corruption, isolation, and full frozen replay checks."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from jev_recovery.benchmark import (POLICIES, _RecordingAdvisor,
                                    _provenance, audit_run, replay_run, run_benchmark)
from jev_recovery.jev import DiagnosticProvider


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.directory.name) / "reference"
        with patch("jev_recovery.jev.request.urlopen") as network:
            cls.report = run_benchmark(cls.output, episodes=12, count=6, seeds=(11, 22))
            network.assert_not_called()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def copy_run(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run"
        shutil.copytree(self.output, path)
        return path

    def rehash(self, path, name):
        inventory_path = path / "integrity.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        inventory_path.write_text(json.dumps(inventory))

    def test_every_policy_receives_exactly_the_same_scenarios_and_seeds(self):
        rows = [json.loads(line) for line in (self.output / "episodes.jsonl").read_text().splitlines()]
        identities = {policy: {(row["seed"], row["suite"], row["scenario_id"])
                              for row in rows if row["policy"] == policy} for policy in POLICIES}
        self.assertEqual(len(rows), 2 * 4 * 6 * len(POLICIES))
        for policy in POLICIES:
            self.assertEqual(identities[policy], identities["checklist"])
        for arms in self.report["training"].values():
            self.assertEqual(arms["rl_without_jev"]["episodes"], 12)
            self.assertEqual(arms["rl_with_jev"]["episodes"], 12)

    def test_checkpoint_selection_uses_only_precommitted_disjoint_validation(self):
        manifest = json.loads((self.output / "manifest.json").read_text())
        training_ids = {item["id"] for item in manifest["train"]}
        validation_ids = {item["id"] for item in manifest["validation"]}
        test_ids = {item["id"] for items in manifest["test_suites"].values() for item in items}
        self.assertFalse(validation_ids & training_ids)
        self.assertFalse(validation_ids & test_ids)
        plan = manifest["selection_plan"]
        self.assertEqual(plan["candidate_episodes"], [3, 6, 9, 12])
        self.assertFalse(plan["test_set_used"])
        for arms in self.report["training"].values():
            for summary in arms.values():
                self.assertEqual(summary["validation_manifest_sha256"], plan["validation_manifest_sha256"])
                self.assertEqual(set(summary["selection_provenance"]["validation_ids"]), validation_ids)
                self.assertIn(summary["selected_after_episode"], plan["candidate_episodes"])
                self.assertLessEqual(summary["selected_after_episode"], summary["episodes"])
                self.assertTrue(summary["selection_provenance"]["full_training_budget_completed"])

    def test_source_identity_is_checked_before_and_after_training(self):
        stability = self.report["source_stability"]
        self.assertTrue(stability["unchanged"])
        self.assertEqual(stability["final_source_sha256"], self.report["provenance"]["source_sha256"])
        before = _provenance()
        after = dict(before, source_sha256={"changed.py": "changed"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift"
            with patch("jev_recovery.benchmark._provenance", side_effect=[before, after]), self.assertRaisesRegex(RuntimeError, "Source files changed"):
                run_benchmark(path, episodes=1, count=6, seeds=(11,))
            self.assertTrue((path / "episodes.jsonl").exists())
            self.assertFalse((path / "report.json").exists())

    def test_semantic_audit_detects_forged_validation_checkpoint_selection(self):
        path = self.copy_run()
        report_path = path / "report.json"
        report = json.loads(report_path.read_text())
        report["training"]["11"]["rl_with_jev"]["selected_after_episode"] = 999
        report_path.write_text(json.dumps(report))
        self.rehash(path, "report.json")
        result = audit_run(path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("validation-only" in error for error in result["errors"]))

    def test_audit_and_independent_replay_of_every_frozen_episode(self):
        self.assertTrue(audit_run(self.output)["ok"])
        with patch("jev_recovery.jev.request.urlopen") as network:
            result = replay_run(self.output)
            network.assert_not_called()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["episodes_replayed"], 288)

    def test_integrity_detects_accidental_trajectory_corruption(self):
        path = self.copy_run()
        episodes = path / "episodes.jsonl"
        episodes.write_text(episodes.read_text().replace('"actions":', '"actions_corrupted":', 1))
        result = audit_run(path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("SHA256" in error for error in result["errors"]))

    def test_semantic_audit_detects_training_label_leak_even_with_new_hash(self):
        path = self.copy_run()
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["train"][0]["wording"] = "test"
        manifest_path.write_text(json.dumps(manifest))
        self.rehash(path, "manifest.json")
        result = audit_run(path)
        self.assertFalse(result["ok"])
        self.assertIn("Joint holdout wording leaks into training", result["errors"])

    def test_semantic_audit_detects_unpaired_episodes(self):
        path = self.copy_run()
        episodes_path = path / "episodes.jsonl"
        lines = episodes_path.read_text().splitlines()
        episodes_path.write_text("\n".join(lines[1:]) + "\n")
        self.rehash(path, "episodes.jsonl")
        self.assertFalse(audit_run(path)["ok"])

    def test_replay_detects_changed_trace_even_if_inventory_is_rewritten(self):
        path = self.copy_run()
        episodes_path = path / "episodes.jsonl"
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        rows[0]["trace"][0]["tick"] = 12345
        episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rehash(path, "episodes.jsonl")
        result = replay_run(path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("Trajectory mismatch" in error for error in result["errors"]))

    def test_seed_repetitions_do_not_become_independent_scenario_samples(self):
        uncertainty = self.report["uncertainty"]
        self.assertEqual(uncertainty["training_seed_clusters"], 2)
        self.assertEqual(uncertainty["incident_family_clusters"], 18)
        self.assertIn("paired", uncertainty["method"])
        self.assertIn("unstable", uncertainty["interpretation"])

    def test_stress_suite_includes_noisy_impossible_and_recoverable_incidents(self):
        manifest = json.loads((self.output / "manifest.json").read_text())
        stress = manifest["test_suites"]["stress"]
        self.assertTrue(all(incident["noise"] for incident in stress))
        self.assertEqual({incident["restart_duration"] for incident in stress}, {1, 3, 5})
        results = self.report["policies"]["checklist"]["by_recoverability"]
        self.assertIn("False", results)
        self.assertIn("True", results)
        self.assertEqual(results["False"]["recovery_rate"], 0)

    def test_diagnostic_labels_are_separate_from_advisor_requests(self):
        fixtures = json.loads((self.output / "diagnostics.json").read_text())
        self.assertEqual(len(fixtures), 6)
        self.assertEqual(self.report["diagnostics"]["fixtures"], 6)
        for fixture in fixtures:
            self.assertIsInstance(fixture["logs"], str)
            self.assertEqual(set(fixture["labels"]), {"credentials", "connections", "dependency"})
        self.assertEqual(self.report["advisor"]["acquisition_totals"]["hosted_calls"], 0)
        self.assertIn("untested", self.report["advisor"]["label"])
        self.assertTrue((self.output / "advisor-cache.json").exists())
        self.assertTrue((self.output / "advisor-cache-snapshot.json").exists())

    def test_semantic_audit_detects_forged_diagnostic_ground_truth(self):
        path = self.copy_run()
        fixtures_path = path / "diagnostics.json"
        fixtures = json.loads(fixtures_path.read_text())
        fixtures[0]["labels"]["credentials"] = 1 - fixtures[0]["labels"]["credentials"]
        fixtures_path.write_text(json.dumps(fixtures))
        self.rehash(path, "diagnostics.json")
        result = audit_run(path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("Diagnostic labels" in error for error in result["errors"]))

    def test_corrupted_sensor_uses_same_evidence_and_complements_scores(self):
        provider = DiagnosticProvider()
        recorder = _RecordingAdvisor(provider, corrupted=True)
        result = recorder.diagnose("database password rejected")
        self.assertAlmostEqual(result.probabilities["credentials"], .15)
        self.assertAlmostEqual(result.probabilities["connections"], .92)
        self.assertEqual(recorder.calls[0]["logs"], "database password rejected")
        self.assertEqual(result.source, "corrupted-offline-heuristic")

    def test_existing_evidence_is_never_overwritten(self):
        before = (self.output / "report.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "empty output"):
            run_benchmark(self.output, episodes=1, count=6, seeds=(11,))
        self.assertEqual((self.output / "report.json").read_bytes(), before)

    def test_invalid_benchmark_budgets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            for arguments in ({"episodes": 0}, {"episodes": True}, {"count": 2}, {"seeds": (1, 1)}):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    run_benchmark(path, **arguments)
        with self.assertRaisesRegex(ValueError, "absolute"):
            run_benchmark(Path("relative"))

    def test_frozen_checkpoint_digest_prevents_mutated_weights(self):
        path = self.copy_run()
        checkpoint = self.report["checkpoints"]["11"]["rl_with_jev"]["path"]
        checkpoint_path = path / checkpoint
        checkpoint_path.write_text(checkpoint_path.read_text() + " ")
        self.assertFalse(audit_run(path)["ok"])


if __name__ == "__main__":
    unittest.main()
