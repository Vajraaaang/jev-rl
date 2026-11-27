"""HTTP integration invariants for the lab: isolation, health, bounded local tools."""
import concurrent.futures
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jev_recovery.service import create_server, MAX_BODY
from test_jev import Response, api_response


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name).resolve()
        cls.server = create_server(port=0, output=cls.output)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temp.cleanup()

    def call(self, route, data=None, headers=None, base=None):
        request = Request((base or self.base) + route, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def incident(self, faults=None):
        status, result = self.call('/api/incidents', {'faults': ['credentials', 'connections'] if faults is None else faults,
                                                     'wording': 'test', 'seed': 11})
        self.assertEqual(status, 201, result)
        return result

    def test_assets_are_self_contained_and_path_traversal_is_denied(self):
        for route in ('/', '/app.js', '/style.css'):
            with urlopen(self.base + route, timeout=5) as response:
                self.assertEqual(response.status, 200)
                body = response.read().decode()
                self.assertNotIn('https://', body)
                self.assertIn("default-src 'self'", response.headers['Content-Security-Policy'])
        self.assertEqual(self.call('/../README.md')[0], 404)
        self.assertEqual(self.call('/api/report?path=/etc/passwd')[0], 404)

    def test_live_dashboard_requires_key_before_binding(self):
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'TYPESAFE_API_KEY'):
                create_server(port=0, output=self.output, advisor_mode='live')

    def test_live_dashboard_consults_hosted_jev_and_reuses_cache_across_sessions(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TYPESAFE_API_KEY': 'fixture-dashboard-key'}), patch(
            'jev_recovery.jev._safe_urlopen', return_value=Response(api_response())
        ) as upstream:
            cache = Path(directory).resolve() / 'diagnostics.json'
            with create_server(port=0, output=self.output, advisor_mode='live', cache_path=cache) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    url = f'http://127.0.0.1:{server.server_port}'
                    status, meta = self.call('/api/meta', base=url)
                    self.assertEqual(status, 200)
                    self.assertTrue(meta['hosted_api_enabled'])
                    self.assertEqual(meta['advisor_mode'], 'live')
                    for expected_calls in (1, 0):
                        status, incident = self.call('/api/incidents', {'faults': ['credentials']}, base=url)
                        self.assertEqual(status, 201)
                        route = f'/api/incidents/{incident["id"]}/step'
                        self.assertEqual(self.call(route, {'action': 'jev'}, base=url)[0], 400)
                        _, inspected = self.call(route, {'action': 'logs'}, base=url)
                        status, result = self.call(route, {'action': 'jev'}, base=url)
                        self.assertEqual(status, 200)
                        diagnosis = result['observation']['evidence']['diagnosis']
                        self.assertEqual(diagnosis['source'], 'hosted-jev')
                        self.assertEqual(diagnosis['cache_hit'], not expected_calls)
                        self.assertEqual(result['metrics']['api_calls'], expected_calls)
                        self.assertEqual(json.loads(upstream.call_args.args[0].data)['state'], inspected['observation']['evidence']['logs'])
                        self.assertNotIn('fixture-dashboard-key', json.dumps([meta, incident, result]))
                    upstream.assert_called_once()
                    self.assertEqual(upstream.call_args.args[0].get_header('Authorization'), 'Bearer fixture-dashboard-key')
                    self.assertNotIn('fixture-dashboard-key', cache.read_text())
                finally:
                    server.shutdown()
                    thread.join(timeout=5)

    def test_independent_sessions_do_not_share_actions_or_evidence(self):
        first, second = self.incident(), self.incident([])
        route = f'/api/incidents/{first["id"]}'
        status, changed = self.call(route + '/step', {'action': 'logs'})
        self.assertEqual(status, 200)
        self.assertTrue(changed['observation']['evidence']['logs'])
        self.assertEqual(changed['metrics']['actions'], 1)
        _, unchanged = self.call(f'/api/incidents/{second["id"]}')
        self.assertEqual(unchanged['metrics']['actions'], 0)
        self.assertIsNone(unchanged['observation']['evidence']['logs'])
        for key in ('faults', 'seed', 'wording', 'scenario_id'):
            self.assertNotIn(key, changed['observation'])

    def test_checklist_autonomy_uses_independent_health_verification(self):
        session = self.incident()
        status, result = self.call(f'/api/incidents/{session["id"]}/run', {'policy': 'checklist'})
        self.assertEqual(status, 200, result)
        self.assertTrue(result['done'])
        self.assertTrue(result['metrics']['recovered'])
        self.assertEqual(result['advisor_source'], 'offline-heuristic')
        self.assertEqual(result['metrics']['api_calls'], 0)
        self.assertGreater(len(result['decisions']), 0)
        self.assertEqual(self.call(f'/api/incidents/{session["id"]}/step', {'action': 'wait'})[0], 409)

    def test_tools_policies_and_incident_parameters_are_strictly_bounded(self):
        session = self.incident()
        base = f'/api/incidents/{session["id"]}'
        for action in ('shell', 'rm -rf /', None, ['logs']):
            self.assertEqual(self.call(base + '/step', {'action': action})[0], 400)
        self.assertEqual(self.call(base + '/run', {'policy': '/tmp/policy.json'})[0], 400)
        self.assertEqual(self.call(base + '/run', {'policy': 'rl_with_jev'})[0], 409)
        for data in ([], {'seed': True}, {'seed': -1}, {'faults': 'credentials'}, {'faults': ['unknown']},
                     {'faults': ['credentials', 'credentials']}, {'wording': 'unknown'}, {'path': '/tmp/test'}):
            self.assertEqual(self.call('/api/incidents', data)[0], 400, data)
        self.assertEqual(self.call('/api/incidents/' + '0' * 32)[0], 404)
        self.assertEqual(self.call(base + '/step', {'action': 'wait'}, {'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.call(base + '/step', {'action': 'wait'}, {'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.call(base + '/step', {'action': 'jev'})[0], 400)
        request = Request(self.base + '/api/incidents', data=b' ' * (MAX_BODY + 1), headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 400)
        raised.exception.close()
        with self.assertRaises(ValueError):
            create_server(host='0.0.0.0', port=0)
        with self.assertRaises(ValueError):
            create_server(output='relative-directory', port=0)

    def test_parallel_mutations_are_serialized(self):
        session = self.incident()
        route = f'/api/incidents/{session["id"]}/step'
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.call(route, {'action': 'wait'}), range(6)))
        self.assertTrue(all(status == 200 for status, _ in results))
        _, result = self.call(f'/api/incidents/{session["id"]}')
        self.assertEqual(result['metrics']['actions'], 6)
        self.assertEqual(len(result['trace']), 6)

    def test_report_is_fixed_to_the_configured_output(self):
        payload = {'schema_version': 2, 'advisor': {'mode': 'offline'}, 'policies': {}}
        (self.output / 'report.json').write_text(json.dumps(payload))
        try:
            status, result = self.call('/api/report')
            self.assertEqual(status, 200)
            self.assertEqual(result, payload)
        finally:
            (self.output / 'report.json').unlink()
