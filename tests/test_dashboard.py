"""HTTP integration invariants for the lab: isolation, health, bounded local tools."""
import concurrent.futures
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jev_recovery.service import create_server, MAX_BODY


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

    def call(self, route, data=None, headers=None):
        request = Request(self.base + route, data=json.dumps(data).encode() if data is not None else None,
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
