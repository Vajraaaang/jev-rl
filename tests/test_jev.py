import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from jev_recovery.jev import DiagnosticError, DiagnosticProvider


class Response:
    def __init__(self, data):
        self.data = data
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, size=-1):
        body = json.dumps(self.data).encode()
        return body if size < 0 else body[:size]


def api_response():
    return {"answers": {"credentials": {"noul": 0.9}, "connections": {"noul": 0.2},
                        "dependency": {"noul": 0.1}},
            "model": "jev-1.13.0", "usage": {"input_tokens": 100, "output_tokens": 12}}


class DiagnosticTests(unittest.TestCase):
    def test_provider_total_counts_acquisition_and_replay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.json'
            provider = DiagnosticProvider('offline', path)
            provider.diagnose('password rejected by database')
            provider.diagnose('password rejected by database')
            self.assertEqual(provider.statistics['offline_computations'], 1)
            self.assertEqual(provider.statistics['cache_hits'], 1)
            self.assertEqual(provider.statistics['hosted_calls'], 0)
            self.assertEqual(provider.statistics['estimated_api_cost_usd'], 0)

    def test_offline_provenance_and_heldout_wording(self):
        provider = DiagnosticProvider()
        for cause, text in {"credentials": "password rejected by database",
                            "connections": "too many clients; leased sessions not returned",
                            "dependency": "remote endpoint unreachable; breaker refuses requests"}.items():
            result = provider.diagnose(text)
            self.assertEqual(result.source, "offline-heuristic")
            self.assertEqual(result.probabilities[cause], 0.85)
            self.assertEqual(result.input_tokens, 0)

    def test_live_request_has_logs_and_no_hidden_truth(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                "jev_recovery.jev._safe_urlopen", return_value=Response(api_response())) as mock:
            result = DiagnosticProvider(mode="live").diagnose("authentication rejected for service account")
            call = mock.call_args.args[0]
            body = json.loads(call.data)
            self.assertEqual(set(body), {"state", "model", "questions"})
            self.assertEqual(body["state"], "authentication rejected for service account")
            self.assertEqual(body["questions"]["credentials"]["type"], "noul")
            self.assertEqual(call.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(result.source, "hosted-jev")
            self.assertAlmostEqual(result.estimated_cost_usd, 0.0000042)

    def test_cached_mode_only_replays_hosted_results_and_charges_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.json"
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                    "jev_recovery.jev._safe_urlopen", return_value=Response(api_response())):
                original = DiagnosticProvider("live", path).diagnose("auth rejected")
            with patch("jev_recovery.jev._safe_urlopen") as network:
                replay = DiagnosticProvider("cached", path).diagnose("auth rejected")
                self.assertTrue(replay.cache_hit)
                self.assertEqual(replay.probabilities, original.probabilities)
                self.assertEqual(replay.input_tokens, 100)
                self.assertEqual(replay.estimated_cost_usd, 0)
                with self.assertRaises(DiagnosticError):
                    DiagnosticProvider("cached", path).diagnose("different logs")
                with self.assertRaises(DiagnosticError):
                    DiagnosticProvider("cached", path, model="jev-1.12.0").diagnose("auth rejected")
                network.assert_not_called()
            records = json.loads(path.read_text())
            record = next(iter(records.values()))
            self.assertEqual(record["requested_model"], "jev-1.13.0")
            self.assertEqual(record["resolved_model"], "jev-1.13.0")

    def test_offline_cache_cannot_impersonate_hosted_jev(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.json"
            DiagnosticProvider("offline", path).diagnose("auth rejected")
            with self.assertRaises(DiagnosticError):
                DiagnosticProvider("cached", path).diagnose("auth rejected")

    def test_auth_failure_has_no_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DiagnosticError, "TYPESAFE_API_KEY"):
                DiagnosticProvider("live").diagnose("logs")
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                "jev_recovery.jev._safe_urlopen", side_effect=HTTPError("url", 401, "Unauthorized", {}, None)):
            with self.assertRaisesRegex(DiagnosticError, "HTTP 401"):
                DiagnosticProvider("live").diagnose("logs")

    def test_invalid_probabilities_and_usage_rejected(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"]:
            response = api_response()
            response["answers"]["credentials"]["noul"] = value
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                    "jev_recovery.jev._safe_urlopen", return_value=Response(response)):
                with self.assertRaises(DiagnosticError):
                    DiagnosticProvider("live").diagnose("logs")
        for value in [-1, 1.5, True]:
            response = api_response()
            response["usage"]["input_tokens"] = value
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                    "jev_recovery.jev._safe_urlopen", return_value=Response(response)):
                with self.assertRaises(DiagnosticError):
                    DiagnosticProvider("live").diagnose("logs")

    def test_cache_path_must_be_absolute(self):
        with self.assertRaises(ValueError):
            DiagnosticProvider(cache_path="relative.json")

    def test_moving_model_aliases_rejected(self):
        for model in ["jev-latest", "jev-preview", "", "jev-1.13"]:
            with self.assertRaises(ValueError):
                DiagnosticProvider(model=model)

    def test_live_model_mismatch_or_missing_model_rejected(self):
        for model in ["jev-1.12.0", None]:
            response = api_response()
            if model is None:
                del response["model"]
            else:
                response["model"] = model
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                    "jev_recovery.jev._safe_urlopen", return_value=Response(response)):
                with self.assertRaises(DiagnosticError):
                    DiagnosticProvider("live").diagnose("logs")

    def test_cache_resolved_model_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.json"
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                    "jev_recovery.jev._safe_urlopen", return_value=Response(api_response())):
                DiagnosticProvider("live", path).diagnose("logs")
            records = json.loads(path.read_text())
            next(iter(records.values()))["resolved_model"] = "jev-1.12.0"
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(DiagnosticError, "cache record"):
                DiagnosticProvider("cached", path).diagnose("logs")

    def test_missing_answer_key_rejected(self):
        response = api_response()
        del response["answers"]["credentials"]
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), patch(
                "jev_recovery.jev._safe_urlopen", return_value=Response(response)):
            with self.assertRaises(DiagnosticError):
                DiagnosticProvider("live").diagnose("logs")

    def test_ranked_causes_are_sorted_and_stable(self):
        result = DiagnosticProvider().diagnose("remote endpoint unreachable")
        self.assertEqual(result.ranked_causes, ["dependency", "credentials", "connections"])


if __name__ == "__main__":
    unittest.main()
