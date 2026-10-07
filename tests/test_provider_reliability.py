import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from jev_recovery.jev import DiagnosticError, DiagnosticProvider, redact_logs
from test_jev import Response, api_response


class ProviderReliabilityTests(unittest.TestCase):
    def test_redacts_credentials_and_preserves_diagnostic_words(self):
        text = 'authentication failed password=hunter2 token="abc123" Authorization: Bearer secretvalue postgres://alice:pass@db/app'
        result = redact_logs(text)
        for secret in ('hunter2', 'abc123', 'secretvalue', 'alice:pass'):
            self.assertNotIn(secret, result)
        self.assertIn('authentication failed', result)

    def test_json_secrets_are_redacted(self):
        text = '{"password": "TOPSECRET", "api_key": "KEYSECRET", "Authorization": "Bearer BEARSECRET"}'
        result = redact_logs(text)
        for secret in ('TOPSECRET', 'KEYSECRET', 'BEARSECRET'):
            self.assertNotIn(secret, result)

    def test_tiny_cost_budget_rejects_request_before_network(self):
        with patch('jev_recovery.jev._safe_urlopen') as network:
            with self.assertRaisesRegex(DiagnosticError, 'budget'):
                DiagnosticProvider('live', max_api_cost_usd=.000001).diagnose('authentication rejected')
            network.assert_not_called()

    def test_redirects_are_never_followed(self):
        from jev_recovery.jev import _NoRedirect
        from urllib.request import Request
        with self.assertRaises(DiagnosticError):
            _NoRedirect().redirect_request(Request('https://api.typesafe.ai/v1/systemone',
                headers={'Authorization': 'Bearer private'}), None, 302, 'redirect', {}, 'https://example.invalid')

    def test_multiple_provider_instances_merge_cache_writes(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.json'
            first, second = DiagnosticProvider('offline', path), DiagnosticProvider('offline', path)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(first.diagnose, 'authentication rejected'),
                           executor.submit(second.diagnose, 'pool exhausted')]
                for future in futures:
                    future.result()
            self.assertEqual(len(json.loads(path.read_text())), 2)

    def test_corrupted_hosted_sensor_retains_acquisition_costs(self):
        from jev_recovery.benchmark import _RecordingAdvisor
        from jev_recovery.sandbox import IncidentEnv
        from jev_recovery.scenarios import Incident
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test'}), patch(
            'jev_recovery.jev._safe_urlopen', return_value=Response(api_response())
        ):
            advisor = _RecordingAdvisor(DiagnosticProvider('live'), corrupted=True)
            env = IncidentEnv(Incident('billing', ('credentials',)), advisor)
            env.step('logs')
            env.step('jev')
            metrics = env.metrics()
            self.assertEqual(metrics['api_calls'], 1)
            self.assertEqual(metrics['input_tokens'], 100)
            self.assertAlmostEqual(metrics['api_cost_usd'], .0000042)
            self.assertEqual(env.observation()['evidence']['diagnosis']['source'], 'corrupted-hosted-jev')

    def test_live_and_disk_receive_only_redacted_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.json'
            with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'private-api-key'}), patch(
                'jev_recovery.jev._safe_urlopen', return_value=Response(api_response())) as call:
                DiagnosticProvider('live', path).diagnose('password=private-db-secret authentication rejected')
            self.assertNotIn('private-db-secret', call.call_args.args[0].data.decode())
            self.assertNotIn('private-db-secret', path.read_text())
            self.assertNotIn('private-api-key', path.read_text())

    def test_integrity_checksum_rejects_changed_probability(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.json'
            provider = DiagnosticProvider('offline', path)
            provider.diagnose('authentication failed')
            records = json.loads(path.read_text())
            next(iter(records.values()))['diagnosis']['probabilities']['credentials'] = 0.01
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(DiagnosticError, 'cache record'):
                provider.diagnose('authentication failed')

    def test_size_limit_rejected_before_network(self):
        with patch('jev_recovery.jev._safe_urlopen') as call:
            with self.assertRaises(DiagnosticError):
                DiagnosticProvider('live', max_log_chars=64).diagnose('x' * 65)
            call.assert_not_called()

    def test_bounded_retry_counts_attempts_and_success(self):
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test'}), patch(
            'jev_recovery.jev._safe_urlopen', side_effect=[HTTPError('url', 503, 'down', {}, None), Response(api_response())]
        ), patch('jev_recovery.jev.time.sleep'):
            provider = DiagnosticProvider('live', max_attempts=2)
            result = provider.diagnose('authentication rejected')
            self.assertEqual(result.source, 'hosted-jev')
            self.assertEqual(provider.statistics['network_attempts'], 2)
            self.assertEqual(provider.statistics['retries'], 1)
            self.assertEqual(provider.statistics['hosted_calls'], 1)

    def test_call_budget_allows_cache_replay_but_blocks_new_request(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test'}), patch(
                'jev_recovery.jev._safe_urlopen', return_value=Response(api_response())
            ) as network:
                provider = DiagnosticProvider('live', Path(temp) / 'cache.json', max_live_calls=1)
                provider.diagnose('authentication rejected')
                self.assertTrue(provider.diagnose('authentication rejected').cache_hit)
                with self.assertRaisesRegex(DiagnosticError, 'budget'):
                    provider.diagnose('pool exhausted')
                self.assertEqual(network.call_count, 1)


if __name__ == '__main__':
    unittest.main()
